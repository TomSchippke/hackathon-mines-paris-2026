"""Pipeline agentique v2 : clusters fins -> clusters de remédiation -> concentration, précurseurs, diagnostic.

Étapes : chargement (schéma fixe) ; extraction de la cause de chaque incident (LLM, citation
vérifiée) ; par partition, sur les causes connues : embeddings, UMAP, balayage HDBSCAN (DBCV),
étiquetage des clusters fins, reclassement des outliers, agent de fusion ; concentration ;
précurseurs (niveau fusionné) + escalades (niveau fin) ; contrôle de plausibilité ; preuves ;
diagnostic final ; criticité ; écriture des fichiers (§3.4 du plan).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import clustering, llm_client
from .agent import reassign_outliers_batch
from .concentration import compute_concentration
from .config import (
    CAUSE_COL,
    COST_COL,
    DATE_COL,
    DEFAULT_INPUT,
    DEFAULT_OUT_DIR,
    DOWNTIME_COL,
    EQUIPMENT_COL,
    FALLBACK_INPUT,
    ID_COL,
    MCS_SWEEP,
    OUTLIER_BATCH_SIZE,
    OUTLIER_LABEL,
    PARTITION_COL,
    PRECURSOR_N_PERMUTATIONS,
    SEVERITY_COL,
    SEVERITY_ORDER,
    TEXT_COL,
    UNKNOWN_CAUSE_LABEL,
)
from .criticality import compute_cluster_criticality
from .diagnosis import build_diagnosis_context, diagnose_merged_cluster
from .extraction import EXTRACTION_COLUMNS, extract_causes
from .labeling import build_fine_cards, label_fine_clusters
from .merging import FineState, run_merge_agent
from .precursor_agent import check_plausibility
from .precursors import EVIDENCE_COLUMNS, PAIR_COLUMNS, extract_evidence, mine_escalations, mine_precursors

REQUIRED_COLUMNS = [
    ID_COL, TEXT_COL, DATE_COL, EQUIPMENT_COL, "site", PARTITION_COL,
    SEVERITY_COL, COST_COL, DOWNTIME_COL,
]
MERGED_COL = "merged_cluster_id"
FINE_COL = "fine_cluster_id"
SWEEP_COLUMNS = ["partition", "min_cluster_size", "n_clusters", "outlier_rate", "dbcv"]
JSON_LIST_COLS = ["fine_cluster_ids", "prevented_outcomes", "recommendations", "early_warning_rules",
                  "common_cause_quotes"]
DIAGNOSIS_KEYS = ["problem_title", "problem_description", "common_cause", "common_cause_quotes",
                  "common_cause_source", "system_impact", "prevented_outcomes", "recommendations",
                  "early_warning_rules", "nominal", "nominal_proposed", "nominal_reason"]

OUTLIER_DIAGNOSIS = {
    "problem_title": "Incidents isolés et anomalies atypiques",
    "problem_description": "Ensemble d'incidents singuliers ne correspondant à aucun cluster récurrent établi.",
    "common_cause": "Origines variées et non récurrentes (facteurs externes exceptionnels, signaux faibles).",
    "system_impact": "Impacts ponctuels sans tendance systémique identifiée.",
    "prevented_outcomes": [],
    "recommendations": [
        {
            "action": "Maintien d'une veille opérationnelle et traitement au cas par cas.",
            "scope": "fleet",
            "targets": [],
            "rationale": "Aucune récurrence exploitable.",
        }
    ],
    "early_warning_rules": [],
}

UNKNOWN_CAUSE_DIAGNOSIS = {
    "problem_title": "Cause non précisée dans le rapport",
    "problem_description": "Incidents dont le rapport ne dit pas pourquoi ils sont arrivés (ou sans extrait "
                           "vérifiable) : ils ne sont rattachés à aucune cause.",
    "common_cause": "Non précisée.",
    "system_impact": "Impact chiffré, cause à documenter.",
    "prevented_outcomes": [],
    "recommendations": [
        {
            "action": "Compléter les rapports de ces incidents avec la cause constatée.",
            "scope": "fleet",
            "targets": [],
            "rationale": "Sans cause écrite, aucune action ciblée n'est possible.",
        }
    ],
    "early_warning_rules": [],
}


# --------------------------------------------------------------------------- chargement


def _resolve_input(input_csv: Path | None) -> Path:
    if input_csv is not None:
        return Path(input_csv)
    for cand in (DEFAULT_INPUT, FALLBACK_INPUT):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Fichier d'entrée introuvable ({DEFAULT_INPUT} ou {FALLBACK_INPUT}).")


def _load_incidents(input_csv: Path) -> pd.DataFrame:
    """Charge le CSV (schéma fixe), vérifie les colonnes, retire les textes vides, force les numériques."""
    df = pd.read_csv(input_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes obligatoires absentes de {input_csv} : {missing}. "
            f"Colonnes présentes : {list(df.columns)}"
        )
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL].astype(str).str.strip() != "")].copy()
    df[COST_COL] = pd.to_numeric(df[COST_COL], errors="coerce").fillna(0.0)
    df[DOWNTIME_COL] = pd.to_numeric(df[DOWNTIME_COL], errors="coerce").fillna(0.0)
    df[PARTITION_COL] = df[PARTITION_COL].fillna("inconnu").astype(str)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- par partition


def _process_partition(
    sub: pd.DataFrame, partition: str, llm: Callable[..., Any]
) -> dict[str, Any]:
    """Clustering de la partition sur les causes connues ; les causes non précisées forment un groupe
    à part (UNKNOWN_CAUSE_LABEL), jamais clusterisé."""
    known = sub["cause_known"].to_numpy(dtype=bool)
    res = _cluster_partition(sub[known].reset_index(drop=True), partition, llm)
    n_unknown = int((~known).sum())
    for key, fill in (("fine", UNKNOWN_CAUSE_LABEL), ("merged", UNKNOWN_CAUSE_LABEL), ("source", "cause_unknown")):
        full = np.full(len(sub), fill, dtype=object)
        full[known] = res[key]
        res[key] = full
    res["stats"].update(n=len(sub), n_unknown=n_unknown)
    if n_unknown:
        print(f"          {n_unknown} incidents sans cause précisée (groupe à part)")
    return res


def _cluster_partition(
    sub: pd.DataFrame, partition: str, llm: Callable[..., Any]
) -> dict[str, Any]:
    """Clustering fin, étiquetage, reclassement des outliers et fusion pour une partition.

    `sub` : lignes de la partition (index réinitialisé), toutes à cause connue ; le clustering porte
    sur la cause extraite (CAUSE_COL). Renvoie les étiquettes par ligne et les structures utiles à
    l'assemblage (cartes, titres, résultat de fusion, statistiques).
    """
    n = len(sub)
    source = np.array(["dense_cluster"] * n, dtype=object)
    stats: dict[str, Any] = {"partition": partition, "n": n, "mcs": None, "dbcv": None,
                             "sweep": None, "n_outliers": 0, "n_reassigned": 0}
    if n < 2 * min(MCS_SWEEP):
        print(f"       -> '{partition}' : {n} incidents, trop peu pour un clustering (tous isolés).")
        fine = np.array([OUTLIER_LABEL] * n, dtype=object)
        source[:] = "unclassified_outlier"
        stats.update(n_outliers=n, n_fine=0, n_merged=0)
        return {"fine": fine, "merged": fine.copy(), "source": source, "cards": {}, "titles": {},
                "fine_to_merged": {}, "merged_members": {}, "log": [], "stats": stats}

    texts = sub[CAUSE_COL].astype(str).tolist()
    print(f"       -> '{partition}' : embeddings + UMAP ({n} causes)...")
    emb = np.asarray(clustering.embed_partition(texts))
    X = clustering.reduce_umap(emb)
    sizes = [s for s in MCS_SWEEP if s <= n // 2]
    sweep = clustering.sweep_hdbscan(X, sizes)
    raw = np.asarray(sweep.labels)
    fine = np.array([OUTLIER_LABEL if k == -1 else f"{partition}_F{k}" for k in raw], dtype=object)
    n_fine0 = len(set(fine.tolist()) - {OUTLIER_LABEL})
    outlier_idx = np.where(raw == -1)[0]
    stats.update(mcs=int(sweep.min_cluster_size), dbcv=float(sweep.dbcv), sweep=sweep.table,
                 n_outliers=int(len(outlier_idx)))
    print(f"          balayage : min_cluster_size={sweep.min_cluster_size}, DBCV={sweep.dbcv:.3f}, "
          f"{n_fine0} clusters fins, {len(outlier_idx)} isolés")

    # Cartes + étiquettes des clusters fins
    cards = build_fine_cards(sub, fine, emb, partition, text_col=CAUSE_COL)
    titles = label_fine_clusters(cards, llm=llm)

    # Reclassement LLM des outliers vers les clusters fins de la même partition
    if len(outlier_idx) and cards:
        taxonomy = [
            {"cluster_id": fid, "problem_title": titles[fid]["fine_title"],
             "common_cause": titles[fid]["failure_mode"]}
            for fid in cards
        ]
        items = [{"incident_id": sub.at[i, ID_COL], "text": sub.at[i, CAUSE_COL]} for i in outlier_idx]
        assigned: dict[str, str] = {}
        for b in range(0, len(items), OUTLIER_BATCH_SIZE):
            for r in reassign_outliers_batch(items[b : b + OUTLIER_BATCH_SIZE], taxonomy, llm=llm):
                assigned[str(r["incident_id"])] = r["assigned_cluster_id"]
        for i in outlier_idx:
            target = assigned.get(str(sub.at[i, ID_COL]), OUTLIER_LABEL)
            if target in cards:
                fine[i] = target
                source[i] = "llm_reassigned"
            else:
                source[i] = "unclassified_outlier"
        stats["n_reassigned"] = int((source == "llm_reassigned").sum())
        print(f"          reclassement : {stats['n_reassigned']}/{len(outlier_idx)} isolés rattachés")
        cards = build_fine_cards(sub, fine, emb, partition, text_col=CAUSE_COL)
    elif len(outlier_idx):
        source[outlier_idx] = "unclassified_outlier"

    # Agent de fusion
    def relabel(labels: np.ndarray, new_ids: list[str]):
        new_cards = build_fine_cards(sub, labels, emb, partition, text_col=CAUSE_COL)
        wanted = {k: new_cards[k] for k in new_ids if k in new_cards}
        return wanted, label_fine_clusters(wanted, llm=llm)

    state = FineState(partition=partition, X_umap=X, embeddings=emb, fine_labels=fine,
                      cards=cards, titles=titles)
    mr = run_merge_agent(state, llm=llm, relabel=relabel)
    fine_final = np.array(list(mr.fine_labels), dtype=object)
    cards = build_fine_cards(sub, fine_final, emb, partition, text_col=CAUSE_COL)  # cartes finales (sous-clusters inclus)
    merged = np.array([mr.fine_to_merged.get(f, OUTLIER_LABEL) for f in fine_final], dtype=object)
    for e in mr.log:
        e["partition"] = partition
    n_merges = sum(1 for e in mr.log if e["action"] == "merge" and not e["vetoed"])
    n_veto = sum(1 for e in mr.log if e["action"] == "merge" and e["vetoed"])
    n_split = sum(1 for e in mr.log if e["action"] == "flag_heterogeneous" and e.get("accepted"))
    stats.update(n_fine=len(set(fine_final.tolist()) - {OUTLIER_LABEL}), n_merged=len(mr.merged_members))
    print(f"          fusion : {stats['n_fine']} fins -> {stats['n_merged']} fusionnés "
          f"({n_merges} fusions, {n_veto} vetos, {n_split} scissions)")
    return {"fine": fine_final, "merged": merged, "source": source, "cards": cards,
            "titles": mr.titles, "fine_to_merged": mr.fine_to_merged,
            "merged_members": mr.merged_members, "log": mr.log, "stats": stats}


# --------------------------------------------------------------------------- aides d'assemblage


def _merge_entry(merged_id: str, log: list[dict]) -> dict | None:
    """Entrée de log synthétique d'un cluster fusionné (fusions acceptées, rondes cumulées)."""
    ents = [e for e in log if e["action"] == "merge" and not e["vetoed"] and e["result_id"] == merged_id]
    if not ents:
        return None
    ents = sorted(ents, key=lambda e: e["round"])
    outcomes = [o for e in ents for o in e["prevented_outcomes"]]
    last = ents[-1]
    return {"shared_remediation": last["shared_remediation"], "prevented_outcomes": outcomes,
            "justification": last["justification"]}


def _plausibility_cards(
    merged_members: dict[str, list[str]], fine_cards: dict[str, dict],
    titles: dict[str, dict], log: list[dict],
) -> dict[str, dict]:
    """Cartes pour check_plausibility : clés = ids fusionnés (précurseurs) et ids fins (escalades)."""
    cards: dict[str, dict] = {}
    for fid, c in fine_cards.items():
        t = titles.get(fid, {})
        cards[fid] = {"title": t.get("fine_title", fid), "failure_mode": t.get("failure_mode", ""),
                      "representative_texts": list(c.get("representative_texts") or [])[:2]}
    for mid, members in merged_members.items():
        entry = _merge_entry(mid, log)
        title = (entry or {}).get("shared_remediation") or " / ".join(
            titles.get(f, {}).get("fine_title", f) for f in members)
        reps = [(fine_cards.get(f, {}).get("representative_texts") or [""])[0] for f in members[:3]]
        cards[mid] = {
            "title": title,
            "failure_mode": " / ".join(titles.get(f, {}).get("failure_mode", "") for f in members),
            "representative_texts": [r for r in reps if r],
        }
    return cards


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=_json_default)


# --------------------------------------------------------------------------- pipeline


def run_agentic_segmentation(
    input_csv: Path | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    llm: Callable[..., Any] | None = None,
    extractions_csv: Path | None = None,
) -> dict[str, Path]:
    """Exécute le pipeline complet et renvoie {nom logique: chemin} des fichiers écrits.

    `extractions_csv` : extractions déjà faites (par exemple un incidents_segmented.csv précédent),
    reprises pour les incidents au texte identique (voir extraction.extract_causes).
    """
    llm = llm or llm_client.call_llm_json  # résolu à l'appel : les défauts des modules sont liés à l'import
    out_dir = Path(out_dir)
    N = 8

    input_csv = _resolve_input(input_csv)
    print(f"\n[1/{N}] Chargement des incidents depuis {input_csv}...")
    df = _load_incidents(input_csv)
    print(f"       -> {len(df)} incidents exploitables chargés.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Cause de chaque incident (le clustering porte dessus)
    seed = pd.read_csv(extractions_csv) if extractions_csv is not None else None
    print(f"\n[2/{N}] Extraction de la cause de chaque incident"
          + (f" (reprise de {extractions_csv})" if seed is not None else "") + "...")
    df = df.drop(columns=[c for c in EXTRACTION_COLUMNS if c in df.columns])
    df = pd.concat([df, extract_causes(df, llm=llm, seed=seed)], axis=1)
    n_reused = int((df["extraction_source"] == "import").sum())
    print(f"       -> {int(df['cause_known'].sum())}/{len(df)} causes vérifiées "
          f"({n_reused} reprises, {len(df) - n_reused} extraites).")

    # --- Partitions : clustering fin, étiquetage, reclassement, fusion
    print(f"\n[3/{N}] Clustering fin + fusion par remédiation (partition par {PARTITION_COL})...")
    df[FINE_COL] = OUTLIER_LABEL
    df[MERGED_COL] = OUTLIER_LABEL
    df["assignment_source"] = "unclassified_outlier"
    fine_cards: dict[str, dict] = {}
    fine_titles: dict[str, dict] = {}
    merged_members: dict[str, list[str]] = {}
    merged_partition: dict[str, str] = {}
    log: list[dict] = []
    stats_list: list[dict] = []
    sweep_tables = []
    plaus_cards: dict[str, dict] = {}
    for partition in sorted(df[PARTITION_COL].unique()):
        idx = df.index[df[PARTITION_COL] == partition]
        sub = df.loc[idx].reset_index(drop=True)
        res = _process_partition(sub, partition, llm)
        df.loc[idx, FINE_COL] = res["fine"]
        df.loc[idx, MERGED_COL] = res["merged"]
        df.loc[idx, "assignment_source"] = res["source"]
        fine_cards.update(res["cards"])
        fine_titles.update(res["titles"])
        merged_members.update(res["merged_members"])
        merged_partition.update({m: partition for m in res["merged_members"]})
        log.extend(res["log"])
        stats_list.append(res["stats"])
        if res["stats"]["sweep"] is not None:
            sweep_tables.append(res["stats"]["sweep"].assign(partition=partition))
        plaus_cards.update(_plausibility_cards(res["merged_members"], res["cards"], res["titles"], res["log"]))

    # --- Concentration (où agir)
    print(f"\n[4/{N}] Profil de concentration (site, équipement, période) des clusters fusionnés...")
    conc = compute_concentration(df, MERGED_COL, PARTITION_COL)
    n_conc_sig = int(conc["significant"].sum()) if len(conc) else 0
    print(f"       -> {n_conc_sig} concentrations significatives sur {len(conc)} testées.")

    # --- Précurseurs + escalades + plausibilité + preuves
    print(f"\n[5/{N}] Fouille de précurseurs (fusionné) et d'escalades (fin), {PRECURSOR_N_PERMUTATIONS} permutations...")
    prec = mine_precursors(df, MERGED_COL, n_permutations=PRECURSOR_N_PERMUTATIONS)
    esc = mine_escalations(df, FINE_COL, MERGED_COL, n_permutations=PRECURSOR_N_PERMUTATIONS)
    frames = [f for f in (prec, esc) if len(f)]
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)
    n_sig = int(pairs["significant"].astype(bool).sum()) if len(pairs) else 0
    print(f"       -> {n_sig} règles statistiquement significatives ; contrôle de plausibilité LLM...")
    pairs = check_plausibility(pairs, plaus_cards, llm=llm)
    n_plaus = int((pairs["significant"].astype(bool) & pairs["plausible"].astype(bool)
                   & (pairs["kind"] != "recurrence")).sum()) if len(pairs) else 0
    print(f"       -> {n_plaus} règles significatives et plausibles.")
    # Preuves au niveau fusionné, sauf escalades (niveau fin)
    is_esc = pairs["kind"] == "escalation"
    ev_parts = [extract_evidence(df, pairs[~is_esc], label_col=MERGED_COL),
                extract_evidence(df, pairs[is_esc], label_col=FINE_COL)]
    ev_parts = [e for e in ev_parts if len(e)]
    evidence = pd.concat(ev_parts, ignore_index=True) if ev_parts else pd.DataFrame(columns=EVIDENCE_COLUMNS)

    # --- Criticité (cluster d'isolés inclus) puis diagnostic final
    print(f"\n[6/{N}] Criticité et diagnostic LLM des clusters fusionnés...")
    crit = compute_cluster_criticality(df, cluster_col=MERGED_COL, cost_col=COST_COL, downtime_col=DOWNTIME_COL)
    impact_cols = ["frequency", "mean_cost_eur", "total_cost_eur", "mean_downtime_hours",
                   "total_downtime_hours", "criticality_tier"]
    diagnoses: dict[str, dict] = {}
    todo = [m for m in crit[MERGED_COL] if m not in (OUTLIER_LABEL, UNKNOWN_CAUSE_LABEL)]
    for k, mid in enumerate(todo, 1):
        print(f"       [{k}/{len(todo)}] {mid}")
        impact = crit.loc[crit[MERGED_COL] == mid, impact_cols].iloc[0].to_dict()
        members = df[df[MERGED_COL] == mid]
        ctx = build_diagnosis_context(
            mid, merged_members.get(mid, []), fine_titles, fine_cards, _merge_entry(mid, log),
            conc, pairs, evidence, {k2: (v.item() if hasattr(v, "item") else v) for k2, v in impact.items()},
            member_texts=members[TEXT_COL].astype(str).tolist(),
            mild_share=float((members[SEVERITY_COL] == SEVERITY_ORDER[0]).mean()),
        )
        diagnoses[mid] = diagnose_merged_cluster(ctx, llm=llm)
    for label, diag in ((OUTLIER_LABEL, OUTLIER_DIAGNOSIS), (UNKNOWN_CAUSE_LABEL, UNKNOWN_CAUSE_DIAGNOSIS)):
        if (df[MERGED_COL] == label).any():
            diagnoses[label] = dict(diag)
    nominal = [m for m, d in diagnoses.items() if d.get("nominal")]
    rejected = [m for m, d in diagnoses.items() if d.get("nominal_proposed") and not d.get("nominal")]
    cited = sum(d.get("common_cause_source") == "diagnostic" for d in diagnoses.values())
    print(f"       -> {len(nominal)} clusters nominaux ({len(rejected)} proposés puis refusés : sévérité) ; "
          f"cause commune citée par le diagnostic pour {cited}/{len(todo)}, reprise de l'extraction sinon.")

    # --- Tables de sortie
    print(f"\n[7/{N}] Assemblage et sauvegarde dans {out_dir}...")
    merged_df = crit.copy()
    merged_df["partition"] = merged_df[MERGED_COL].map(merged_partition).fillna("")
    merged_df["fine_cluster_ids"] = merged_df[MERGED_COL].map(lambda m: merged_members.get(m, []))
    for key in DIAGNOSIS_KEYS:
        default = [] if key in JSON_LIST_COLS else (False if key in ("nominal", "nominal_proposed") else "")
        merged_df[key] = merged_df[MERGED_COL].map(lambda m, key=key, default=default: diagnoses.get(m, {}).get(key, default))
    merged_cols = [MERGED_COL, "partition", "fine_cluster_ids", "criticality_rank", "criticality_tier",
                   "frequency", "mean_cost_eur", "total_cost_eur", "mean_downtime_hours",
                   "total_downtime_hours", "problem_title", "problem_description", "common_cause",
                   "common_cause_quotes", "common_cause_source", "system_impact", "prevented_outcomes",
                   "recommendations", "early_warning_rules", "nominal", "nominal_proposed", "nominal_reason"]
    merged_df = merged_df[merged_cols].sort_values("criticality_rank").reset_index(drop=True)
    merged_csv = merged_df.copy()
    for c in JSON_LIST_COLS:
        merged_csv[c] = merged_csv[c].map(_dumps)

    title_map = merged_df.set_index(MERGED_COL)["problem_title"].to_dict()
    cause_map = merged_df.set_index(MERGED_COL)["common_cause"].to_dict()
    tier_map = merged_df.set_index(MERGED_COL)["criticality_tier"].to_dict()
    seg = df.copy()
    seg["cluster_id"] = seg[MERGED_COL]
    seg["problem_title"] = seg[MERGED_COL].map(title_map)
    seg["common_cause"] = seg[MERGED_COL].map(cause_map)
    seg["cluster_criticality_tier"] = seg[MERGED_COL].map(tier_map)
    lead = [ID_COL, FINE_COL, MERGED_COL, "cluster_id", "problem_title", "common_cause",
            "cluster_criticality_tier", "assignment_source"]
    seg = seg[lead + [c for c in seg.columns if c not in lead]]

    sizes = df[FINE_COL].value_counts()
    fine_rows = []
    f2m = {f: m for m, fs in merged_members.items() for f in fs}
    for fid in sorted(fine_cards):
        t = fine_titles.get(fid, {})
        fine_rows.append({
            FINE_COL: fid, "partition": fine_cards[fid].get("partition", ""),
            MERGED_COL: f2m.get(fid, ""), "size": int(sizes.get(fid, 0)),
            "fine_title": t.get("fine_title", ""), "failure_mode": t.get("failure_mode", ""),
            "keywords": ";".join(map(str, fine_cards[fid].get("keywords") or [])),
        })
    fine_df = pd.DataFrame(fine_rows, columns=[FINE_COL, "partition", MERGED_COL, "size",
                                               "fine_title", "failure_mode", "keywords"])
    sweep_df = (pd.concat(sweep_tables, ignore_index=True) if sweep_tables
                else pd.DataFrame(columns=SWEEP_COLUMNS))[SWEEP_COLUMNS]

    paths = {
        "incidents_segmented": out_dir / "incidents_segmented.csv",
        "clusters_fine": out_dir / "clusters_fine.csv",
        "clusters_merged": out_dir / "clusters_merged.csv",
        "merge_log": out_dir / "merge_log.json",
        "concentration": out_dir / "concentration.csv",
        "precursor_pairs": out_dir / "precursor_pairs.csv",
        "precursor_evidence": out_dir / "precursor_evidence.csv",
        "fine_sweep": out_dir / "fine_sweep.csv",
    }
    seg.to_csv(paths["incidents_segmented"], index=False)
    fine_df.to_csv(paths["clusters_fine"], index=False)
    merged_csv.to_csv(paths["clusters_merged"], index=False)
    paths["merge_log"].write_text(
        json.dumps(log, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    conc.to_csv(paths["concentration"], index=False)
    pairs[PAIR_COLUMNS].to_csv(paths["precursor_pairs"], index=False)
    evidence[EVIDENCE_COLUMNS].to_csv(paths["precursor_evidence"], index=False)
    sweep_df.to_csv(paths["fine_sweep"], index=False)
    for p in paths.values():
        print(f"       ✓ {p}")

    # --- Synthèse console
    print(f"\n[8/{N}] Synthèse")
    print("=" * 90)
    for s in stats_list:
        mcs = "n/a" if s["mcs"] is None else f"{s['mcs']}"
        dbcv = "n/a" if s["dbcv"] is None else f"{s['dbcv']:.3f}"
        print(f"Partition {s['partition']} : mcs={mcs} | DBCV={dbcv} | {s['n_fine']} fins -> {s['n_merged']} fusionnés | "
              f"isolés {s['n_outliers']} (reclassés {s['n_reassigned']}) | cause non précisée {s.get('n_unknown', 0)}")
    print(f"Concentrations significatives : {n_conc_sig} | règles précurseurs significatives et plausibles : {n_plaus}")
    print("=" * 90)
    print("TOP DES CLUSTERS PAR CRITICITÉ (FRÉQUENCE × COÛTS) :")
    print("=" * 90)
    for _, row in merged_df.head(8).iterrows():
        print(f"[{str(row['criticality_tier']).upper()}] Rang {row['criticality_rank']} | {row[MERGED_COL]} : {row['problem_title']}")
        print(f"    Impact : {row['frequency']} incidents | Coût total : {row['total_cost_eur']:,.2f} € "
              f"(Moyenne : {row['mean_cost_eur']:,.2f} €) | Arrêt total : {row['total_downtime_hours']:.1f} h")
        outs = row["prevented_outcomes"]
        print(f"    Défaillances évitées : {'; '.join(outs) if outs else 'n/a'}")
        recs = row["recommendations"]
        print(f"    Première recommandation : {recs[0]['action'] if recs else 'n/a'}")
        print("-" * 90)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline agentique de segmentation des incidents")
    parser.add_argument("--input", type=Path, default=None, help="Chemin du CSV d'incidents")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Dossier de sortie")
    parser.add_argument("--extractions", type=Path, default=None,
                        help="CSV d'extractions déjà faites (incident_id, text, cause_racine...) à reprendre")
    args = parser.parse_args()
    run_agentic_segmentation(input_csv=args.input, out_dir=args.out_dir, extractions_csv=args.extractions)


if __name__ == "__main__":
    main()
