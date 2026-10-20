"""Assemblage du rapport opérationnel : chargement des sorties, contexte (dicts simples), rendu Jinja2.

Entrées : `outputs/segmentation/` (incidents segmentés, clusters fins et fusionnés, concentrations,
escalades, précurseurs), `outputs/equipment/equipment_summary.csv` et, s'il existe,
`outputs/diagnostics/umap_stability.csv`. Sortie : un fichier HTML autonome (aucune ressource externe,
aucun JavaScript) et une page annexe avec la synthèse non retenue (`rapport_synthese_alt.html`).

Le cluster fusionné (« voie de remédiation ») porte l'action : le plan liste les voies non nominales
(hors isolés) par coût total, avec leurs actions numérotées A1..An dans l'ordre du diagnostic
(escalade, ciblée, flotte) et, pour chacune, ses preuves (citations, concentrations, séries).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

from agentic_pipeline.config import (
    COST_COL, DATE_COL, DOWNTIME_COL, EQUIPMENT_COL, ID_COL, OUTLIER_LABEL, PARTITION_COL,
    SEVERITY_COL, SEVERITY_ORDER, TEXT_COL,
)
from agentic_pipeline.diagnosis import quote_found, simplify

from . import charts, formatting
from .summary import llm_summary, template_summary

DEFAULT_SEG_DIR = Path("outputs/segmentation")
DEFAULT_EQ_DIR = Path("outputs/equipment")
DEFAULT_DIAG_DIR = Path("outputs/diagnostics")
DEFAULT_OUT = Path("outputs/rapport_operationnel.html")
ALT_NAME = "rapport_synthese_alt.html"
TEMPLATE_DIR = Path(__file__).parent / "templates"

N_OPEN_VOIES = 5  # voies dépliées par défaut
MAX_QUOTE_INCIDENTS = 6  # incidents affichés par citation
MAX_REPRESENTATIVES = 5
N_TOP_EQUIPMENT = 5
TIER_ORDER = ["Critique", "Élevée", "Moyenne", "Faible"]
SCOPE_LABELS = {"fleet": "flotte", "site": "site", "equipment": "équipement"}
PARTITION_LABELS = {"solar": "solaire", "wind": "éolien"}
# Colonnes listes JSON de clusters_merged.csv (comme pipeline.JSON_LIST_COLS, sans importer UMAP/HDBSCAN)
JSON_LIST_COLS = ["fine_cluster_ids", "prevented_outcomes", "recommendations", "early_warning_rules",
                  "common_cause_quotes"]
BADGES = {
    "escalade": "escalade",
    "concentration": "concentration",
    "verifier": "à vérifier",
    "inferee": "cause inférée – à confirmer",
    "nonverifie": "non vérifié par le critique",
    "defaut": "action par défaut",
}


# --------------------------------------------------------------------------- chargement


def _slug(s: Any) -> str:
    return simplify(s).replace(" ", "-") or "na"


def _s(v: Any) -> str:
    """Valeur texte d'une cellule (NaN -> chaîne vide)."""
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def _truthy(v: Any) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "1.0")


def _json_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    try:
        out = json.loads(v) if _s(v) else []
    except ValueError:
        return []
    return out if isinstance(out, list) else []


def _read(path: Path, required: bool = True, **kwargs) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Fichier attendu absent : {path} (lancer d'abord l'étape segmentation).")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_outputs(seg_dir: Path, eq_dir: Path, diag_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Charge les sorties du pipeline ; les colonnes listes JSON de clusters_merged sont décodées."""
    seg_dir, eq_dir = Path(seg_dir), Path(eq_dir)
    diag_dir = Path(diag_dir) if diag_dir is not None else seg_dir.parent / "diagnostics"
    ids = {ID_COL: str}
    merged = _read(seg_dir / "clusters_merged.csv")
    for c in JSON_LIST_COLS:
        if c in merged:
            merged[c] = merged[c].map(_json_list)
    return {
        "incidents": _read(seg_dir / "incidents_segmented.csv", dtype=ids),
        "merged": merged,
        "fine": _read(seg_dir / "clusters_fine.csv"),
        "concentration": _read(seg_dir / "concentration.csv", required=False, dtype={"value": str}),
        "escalations": _read(seg_dir / "escalations.csv", required=False,
                             dtype={"incident_ids": str, "costs": str, "equipment_id": str}),
        "pairs": _read(seg_dir / "precursor_pairs.csv", required=False),
        "equipment": _read(eq_dir / "equipment_summary.csv", required=False),
        "umap": _read(diag_dir / "umap_stability.csv", required=False),
    }


# --------------------------------------------------------------------------- incidents


class Incidents:
    """Fiches d'incident pour le rendu : lien relatif vers le PDF s'il existe, sinon texte CSV."""

    def __init__(self, df: pd.DataFrame, out_dir: Path, root: Path):
        self.df = df.set_index(df[ID_COL].astype(str), drop=False)
        self.out_dir, self.root = out_dir, root
        self._cache: dict[str, dict] = {}

    def get(self, uid: Any) -> dict:
        uid = str(uid).strip()
        if uid in self._cache:
            return self._cache[uid]
        if uid not in self.df.index:
            ref = {"id": uid, "href": None, "text": "", "date": "", "severity": "", "sev_slug": "na", "cost": 0.0}
        else:
            r = self.df.loc[uid]
            doc = _s(r.get("document_path"))
            href = None
            # chemin relatif à la racine du dépôt -> relatif à l'emplacement du rapport ; PDF absent du
            # disque : pas de lien mort, le texte est affiché à la place
            if doc and (self.root / doc).is_file():
                href = Path(os.path.relpath((self.root / doc).resolve(), self.out_dir.resolve())).as_posix()
            sev = _s(r.get(SEVERITY_COL))
            ref = {"id": uid, "href": href, "text": _s(r.get(TEXT_COL)), "date": _s(r.get(DATE_COL)),
                   "severity": sev, "sev_slug": _slug(sev), "cost": float(r.get(COST_COL) or 0.0),
                   "equipment": _s(r.get(EQUIPMENT_COL)), "site": _s(r.get("site"))}
        self._cache[uid] = ref
        return ref


def _quote_refs(quote: str, members: pd.DataFrame, incidents: Incidents) -> dict:
    """Citation attribuée aux incidents du cluster dont le texte la contient."""
    hits = [uid for uid, t in zip(members[ID_COL].astype(str), members[TEXT_COL].astype(str))
            if quote_found(quote, t)]
    return {"text": quote, "incidents": [incidents.get(u) for u in hits[:MAX_QUOTE_INCIDENTS]],
            "n": len(hits), "more": max(0, len(hits) - MAX_QUOTE_INCIDENTS)}


# --------------------------------------------------------------------------- voies


def _escalation_target(esc: pd.DataFrame, voie_id: str, equipment: str, incidents: Incidents) -> dict | None:
    if esc is None or esc.empty:
        return None
    rows = esc[(esc["merged_cluster_id"] == voie_id) & (esc["equipment_id"].astype(str) == equipment)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    uids = _s(r["incident_ids"]).split(";")
    sevs = _s(r["severity_sequence"]).split(";")
    dates = _s(r["dates"]).split(";")
    costs = [float(c) for c in _s(r["costs"]).split(";")]
    points = []
    for uid, sev, d, c in zip(uids, sevs, dates, costs):
        ref = dict(incidents.get(uid))
        ref.update(date=d, severity=sev, sev_slug=_slug(sev), cost=c)
        points.append(ref)
    sequence = " → ".join(sevs)
    return {
        "sequence": sequence, "n": int(r["n_incidents"]), "total_cost": float(r["total_cost_eur"]),
        "first": formatting.date_fr(r["first_date"]), "last": formatting.date_fr(r["last_date"]),
        "points": points,
        "svg": charts.timeline_svg(points, f"Série en escalade : {equipment}, {len(points)} incidents ({sequence})"),
    }


def _concentration_target(conc: pd.DataFrame, voie_id: str, scope: str, value: str) -> dict | None:
    if conc is None or conc.empty:
        return None
    dim = {"equipment": EQUIPMENT_COL, "site": "site"}.get(scope)
    rows = conc[(conc["merged_cluster_id"] == voie_id) & (conc["dimension"] == dim)
                & (conc["value"].astype(str) == value) & conc["significant"].map(_truthy)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return {"observed": int(r["observed"]), "expected": formatting.num(r["expected"], 1),
            "lift": formatting.num(r["lift"], 1), "q": formatting.num(r["q_value"], 3),
            "level": _s(r["evidence_level"])}


def _action(rec: dict, aid: str, voie: dict, data: dict, incidents: Incidents) -> dict:
    reasons = rec.get("target_reason") or {}
    to_verify = set(rec.get("targets_to_verify") or [])
    scope = rec.get("scope") or "fleet"
    targets = []
    for t in rec.get("targets") or []:
        reason = reasons.get(t, "")
        targets.append({
            "name": t, "to_verify": t in to_verify,
            "esc": (_escalation_target(data["escalations"], voie["id"], t, incidents)
                    if reason in ("escalation", "both") else None),
            "conc": (_concentration_target(data["concentration"], voie["id"], scope, t)
                     if reason in ("concentration", "both") else None),
        })
    vals = set(reasons.values())
    badges = []
    if vals & {"escalation", "both"}:
        badges.append("escalade")
    if vals & {"concentration", "both"}:
        badges.append("concentration")
    if to_verify:
        badges.append("verifier")
    if rec.get("confirm_cause_first") or voie["cause_source"] == "inferee":
        badges.append("inferee")
    if rec.get("critic_verdict") == "non_verifie":
        badges.append("nonverifie")
    if rec.get("critic_verdict") == "fallback":
        badges.append("defaut")
    text = _s(rec.get("action"))
    return {
        "id": aid, "text": text[:1].upper() + text[1:], "component": _s(rec.get("component")),
        "scope": scope, "scope_label": SCOPE_LABELS.get(scope, scope), "targets": targets,
        "rationale": _s(rec.get("rationale")),
        "badges": [{"slug": b, "label": BADGES[b]} for b in badges],
    }


def _voie_base(row: dict, fleet_cost: float) -> dict:
    tier = _s(row.get("criticality_tier"))
    return {
        "id": str(row["merged_cluster_id"]), "anchor": f"voie-{row['merged_cluster_id']}",
        "rank": int(row["criticality_rank"]), "title": _s(row.get("problem_title")) or str(row["merged_cluster_id"]),
        "tier": tier, "tier_slug": _slug(tier), "partition": PARTITION_LABELS.get(_s(row.get("partition")), _s(row.get("partition"))),
        "frequency": int(row["frequency"]), "total_cost": float(row["total_cost_eur"]),
        "mean_cost": float(row["mean_cost_eur"]), "downtime": float(row["total_downtime_hours"]),
        "cost_share": float(row["total_cost_eur"]) / fleet_cost if fleet_cost else 0.0,
        "cause_source": _s(row.get("common_cause_source")),
    }


def _plan_voie(row: dict, data: dict, incidents: Incidents, fleet_cost: float, first_action: int) -> dict:
    v = _voie_base(row, fleet_cost)
    inc = data["incidents"]
    members = inc[inc["merged_cluster_id"] == v["id"]]
    fine = data["fine"]
    fine_rows = fine[fine["fine_cluster_id"].isin(row["fine_cluster_ids"])] if len(fine) else fine
    fine_list = [{"id": _s(f["fine_cluster_id"]), "title": _s(f["fine_title"]), "failure_mode": _s(f["failure_mode"]),
                  "size": int(f["size"])} for f in fine_rows.sort_values("size", ascending=False).to_dict("records")]
    # Incidents représentatifs : le plus coûteux de chaque mode de défaillance (clusters fins)
    reps = (members.sort_values(COST_COL, ascending=False).drop_duplicates("fine_cluster_id")
            .head(MAX_REPRESENTATIVES)[ID_COL].tolist())
    v.update(
        description=_s(row.get("problem_description")), cause=_s(row.get("common_cause")),
        quotes=[_quote_refs(q, members, incidents) for q in row["common_cause_quotes"] if _s(q)],
        impact=_s(row.get("system_impact")),
        outcomes=[_s(o) for o in row["prevented_outcomes"] if _s(o)],
        fine=fine_list, representatives=[incidents.get(u) for u in reps],
        warning_rules=[r for r in row["early_warning_rules"] if isinstance(r, dict)],
    )
    v["actions"] = [_action(r, f"A{first_action + k}", v, data, incidents)
                    for k, r in enumerate(row["recommendations"]) if isinstance(r, dict)]
    return v


def _nominal_voie(row: dict, data: dict, incidents: Incidents, fleet_cost: float) -> dict:
    v = _voie_base(row, fleet_cost)
    inc = data["incidents"]
    members = inc[inc["merged_cluster_id"] == v["id"]]
    quote = _s(row.get("nominal_external_quote"))
    v.update(condition=_s(row.get("nominal_external_condition")), reason=_s(row.get("nominal_reason")),
             quote=_quote_refs(quote, members, incidents) if quote else None,
             mild_share=float((members[SEVERITY_COL] == SEVERITY_ORDER[0]).mean()) if len(members) else 0.0)
    return v


# --------------------------------------------------------------------------- contexte


def _fleet(data: dict, plan_ids: set[str], fleet_cost: float) -> dict:
    merged = data["merged"].sort_values("criticality_rank")
    rows, cum = [], 0.0
    for r in merged.to_dict("records"):
        cum += float(r["total_cost_eur"])
        tier = _s(r["criticality_tier"])
        mid = str(r["merged_cluster_id"])
        rows.append({"title": _s(r["problem_title"]) or mid, "cost": float(r["total_cost_eur"]),
                     "cum_share": cum / fleet_cost if fleet_cost else 0.0, "tier": tier, "tier_slug": _slug(tier),
                     "href": f"#voie-{mid}" if mid in plan_ids else "#sans-action"})
    inc = data["incidents"]
    groups = sorted(inc[PARTITION_COL].astype(str).unique())
    by_site = (inc.groupby(["site", PARTITION_COL]).agg(cost=(COST_COL, "sum"), n=(ID_COL, "size"))
               .reset_index().sort_values("cost", ascending=False))
    site_rows = [{"site": r["site"], "group": str(r[PARTITION_COL]), "cost": float(r["cost"]), "n": int(r["n"])}
                 for r in by_site.to_dict("records")]
    eq = data["equipment"]
    top_eq = []
    if len(eq) and "bad_actor_score" in eq:
        for r in eq.sort_values("bad_actor_score", ascending=False).head(N_TOP_EQUIPMENT).to_dict("records"):
            top_eq.append({"id": _s(r["equipment_id"]), "site": _s(r.get("site")), "n": int(r["total_incidents"]),
                           "cost": float(r["total_cost_eur"]), "downtime": float(r["total_downtime_hours"]),
                           "mtbf": r.get("mtbf_days"), "score": r.get("bad_actor_score"),
                           "problem": _s(r.get("most_frequent_problem"))})
    tiers = [{"label": t, "slug": _slug(t)} for t in TIER_ORDER if t in set(merged["criticality_tier"].astype(str))]
    return {"pareto_svg": charts.pareto_svg(rows), "site_svg": charts.site_svg(site_rows, groups),
            "groups": [{"label": PARTITION_LABELS.get(g, g), "cat": i + 1} for i, g in enumerate(groups)],
            "tiers": tiers, "top_equipment": top_eq}


def _method(data: dict, n_voies: int) -> dict:
    """Chiffres de la section « Méthode et limites », tous calculés depuis les sorties."""
    fine, merged, pairs, conc, umap = data["fine"], data["merged"], data["pairs"], data["concentration"], data["umap"]
    m: dict[str, Any] = {
        "n_fine": int((fine["size"] > 1).sum()) if len(fine) else 0,
        "n_merged": int((merged["merged_cluster_id"] != OUTLIER_LABEL).sum()),
        "n_voies": n_voies,
        "n_escalations": len(data["escalations"]),
    }
    if len(pairs):
        best = pairs[pairs["is_best_window"].map(_truthy)]
        pre, rec = best[best["kind"] == "precursor"], best[best["kind"] == "recurrence"]
        m["precursors"] = {"tested": len(pre), "significant": int(pre["significant"].map(_truthy).sum()),
                           "plausible": int((pre["significant"].map(_truthy) & pre["plausible"].map(_truthy)).sum()),
                           "rec_tested": len(rec), "rec_significant": int(rec["significant"].map(_truthy).sum())}
    if len(conc):
        sig = conc[conc["significant"].map(_truthy)]
        m["concentration"] = {"tested": len(conc), "significant": len(sig),
                              "strong": int((sig["evidence_level"] == "forte").sum()),
                              "to_verify": int((sig["evidence_level"] == "à vérifier").sum())}
    if len(umap) and "pairwise_ari_mean" in umap:
        m["umap"] = [{"partition": PARTITION_LABELS.get(p, p), "mean": formatting.num(g["pairwise_ari_mean"].iloc[0], 2),
                      "min": formatting.num(g["pairwise_ari_min"].iloc[0], 2), "n_seeds": g["seed"].nunique()}
                     for p, g in umap.groupby("partition")]
    m["nominal_accepted"] = int(merged["nominal"].map(_truthy).sum())
    m["nominal_rejected"] = int((merged["nominal_proposed"].map(_truthy) & ~merged["nominal"].map(_truthy)).sum())
    m["cause_inferred"] = int((merged["common_cause_source"].astype(str) == "inferee").sum())
    load_dotenv(find_dotenv(usecwd=True))
    m["model"] = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    return m


def build_context(data: dict, out_path: Path, root: Path) -> dict:
    """Contexte du gabarit (dicts et listes simples, plus les SVG déjà rendus)."""
    inc = data["incidents"].copy()
    inc[COST_COL] = pd.to_numeric(inc[COST_COL], errors="coerce").fillna(0.0)
    inc[DOWNTIME_COL] = pd.to_numeric(inc[DOWNTIME_COL], errors="coerce").fillna(0.0)
    data = {**data, "incidents": inc}
    incidents = Incidents(inc, out_path.parent, root)
    fleet_cost = float(inc[COST_COL].sum())
    merged = data["merged"].sort_values("criticality_rank")
    is_nominal = merged["nominal"].map(_truthy)
    is_outlier = merged["merged_cluster_id"] == OUTLIER_LABEL

    plan, n_actions = [], 0
    for row in merged[~is_nominal & ~is_outlier].to_dict("records"):
        v = _plan_voie(row, data, incidents, fleet_cost, n_actions + 1)
        v["rank"] = len(plan) + 1  # numérotation continue : les groupes sans action sont hors du plan
        v["open"] = len(plan) < N_OPEN_VOIES
        n_actions += len(v["actions"])
        plan.append(v)

    nominal = [_nominal_voie(r, data, incidents, fleet_cost) for r in merged[is_nominal & ~is_outlier].to_dict("records")]
    iso = inc[inc["merged_cluster_id"] == OUTLIER_LABEL].sort_values(COST_COL, ascending=False)
    isolated = {"n": len(iso), "cost": float(iso[COST_COL].sum()),
                "incidents": [incidents.get(u) for u in iso[ID_COL]]}
    no_action = {"nominal": nominal, "isolated": isolated,
                 "total_cost": sum(v["total_cost"] for v in nominal) + isolated["cost"]}

    # Actions d'escalade, classées par coût de la série (pour la synthèse)
    esc_actions = []
    for v in plan:
        for a in v["actions"]:
            series = [t for t in a["targets"] if t["esc"]]
            if series:
                t = max(series, key=lambda t: t["esc"]["total_cost"])
                esc_actions.append({"action_id": a["id"], "equipment": t["name"], "voie_title": v["title"],
                                    "sequence": t["esc"]["sequence"], "series_cost": t["esc"]["total_cost"]})
    esc_actions.sort(key=lambda e: -e["series_cost"])

    dates = pd.to_datetime(inc[DATE_COL], errors="coerce")
    plan_cost = sum(v["total_cost"] for v in plan)
    header = {
        "n_incidents": len(inc), "date_min": formatting.date_fr(dates.min()), "date_max": formatting.date_fr(dates.max()),
        "sites": sorted(inc["site"].astype(str).unique()),
        "asset_types": [PARTITION_LABELS.get(p, p) for p in sorted(inc[PARTITION_COL].astype(str).unique())],
        "n_equipment": inc[EQUIPMENT_COL].nunique(), "total_cost": fleet_cost,
        "total_downtime": float(inc[DOWNTIME_COL].sum()),
        "n_pdf": int(inc["document_path"].map(_s).astype(bool).sum()) if "document_path" in inc else 0,
        "generated": datetime.now().strftime("%d/%m/%Y à %H:%M"),
    }
    return {
        "header": header, "plan": plan, "n_actions": n_actions, "plan_cost": plan_cost,
        "plan_share": plan_cost / fleet_cost if fleet_cost else 0.0,
        "no_action": no_action, "escalation_actions": esc_actions,
        "fleet": _fleet(data, {v["id"] for v in plan}, fleet_cost),
        "method": _method(data, len(plan)),
    }


# --------------------------------------------------------------------------- rendu


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(eur=formatting.eur, pct=formatting.pct, num=formatting.num, hours=formatting.hours,
                       date_fr=formatting.date_fr, eur_short=formatting.eur_short,
                       share=formatting.share)
    return env


def render(ctx: dict, alt: bool = False) -> str:
    return _environment().get_template("rapport.html.j2").render(**ctx, alt_only=alt)


def build_report(
    seg_dir: Path | str = DEFAULT_SEG_DIR,
    eq_dir: Path | str = DEFAULT_EQ_DIR,
    out_path: Path | str = DEFAULT_OUT,
    summary: str = "template",
    llm: Callable[..., Any] | None = None,
    root: Path | str | None = None,
    diag_dir: Path | str | None = None,
) -> Path:
    """Écrit le rapport (`out_path`) et la synthèse non retenue (`rapport_synthese_alt.html` à côté).

    `summary` : « template » (déterministe) ou « llm » (un appel, validé, repli sur le gabarit).
    En mode « template », la synthèse LLM de la page annexe est lue dans le cache (aucun appel
    réseau), sauf si `llm` est fourni. `root` : racine à laquelle `document_path` est relatif
    (défaut : répertoire courant).
    """
    if summary not in ("template", "llm"):
        raise ValueError(f"summary doit valoir « template » ou « llm », pas {summary!r}.")
    out_path = Path(out_path)
    root = Path(root) if root is not None else Path.cwd()
    data = load_outputs(Path(seg_dir), Path(eq_dir), diag_dir)
    ctx = build_context(data, out_path, root)
    tpl = template_summary(ctx)
    if summary == "llm":
        main_sum, alt_sum = llm_summary(ctx, llm=llm), tpl
    else:
        main_sum, alt_sum = tpl, llm_summary(ctx, llm=llm, cache_only=llm is None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render({**ctx, "summary": main_sum}), encoding="utf-8")
    alt_path = out_path.parent / ALT_NAME
    alt_path.write_text(render({**ctx, "summary": alt_sum, "main_name": out_path.name}, alt=True), encoding="utf-8")
    print(f"   [OK] Rapport : {out_path} ({out_path.stat().st_size / 1024:.0f} Ko), synthèse "
          f"{main_sum['kind']} ; synthèse alternative ({alt_sum['kind']}) : {alt_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapport opérationnel HTML statique")
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR, help="Sorties de la segmentation")
    parser.add_argument("--eq-dir", type=Path, default=DEFAULT_EQ_DIR, help="Sorties de l'analyse par équipement")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Fichier HTML produit")
    parser.add_argument("--summary", choices=["template", "llm"], default="template",
                        help="synthèse déterministe (défaut) ou rédigée par le LLM (un appel)")
    args = parser.parse_args()
    build_report(args.seg_dir, args.eq_dir, args.out, summary=args.summary)


if __name__ == "__main__":
    main()
