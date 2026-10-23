"""Assemblage du rapport opérationnel : chargement des sorties, contexte (dicts simples), rendu Jinja2.

Entrées : `outputs/segmentation/` (incidents segmentés, clusters fins et fusionnés, concentrations,
escalades, précurseurs) et, s'il existe, `outputs/diagnostics/umap_stability.csv`. Sortie : un fichier
HTML autonome (aucune ressource externe, aucun JavaScript) à onglets :

- Synthèse : chiffres clés, coût mensuel par type d'actif, segmentation des problèmes et Pareto ;
- Actions ciblées (une ligne par équipement ou site, classées par le coût des incidents de cette cible),
  Programmes flotte (classés par le coût du groupe), Preuves par groupe, Sans action requise ;
- Méthode (jury) : construction, qualité des actions mesurée par le code, résultats négatifs, limites.

Onglets et filtres (type d'actif, site) sont en CSS seul (`:target`, `:has`). Les actions sont
numérotées A1..An dans l'ordre de lecture : actions ciblées, puis programmes flotte.
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

DEFAULT_SEG_DIR = Path("outputs/segmentation")
DEFAULT_OUT = Path("outputs/rapport_operationnel.html")
TEMPLATE_DIR = Path(__file__).parent / "templates"

MAX_QUOTE_INCIDENTS = 6  # incidents affichés par citation
MAX_REPRESENTATIVES = 5
TIER_ORDER = ["Critique", "Élevée", "Moyenne", "Faible"]
SCOPE_LABELS = {"fleet": "flotte", "site": "site", "equipment": "équipement"}
PARTITION_LABELS = {"solar": "solaire", "wind": "éolien"}
FLEET_NOUNS = {"solar": "installations photovoltaïques", "wind": "éoliennes"}
URGENCY_ESCALATION = "agir avant la prochaine occurrence"
# Colonnes listes JSON de clusters_merged.csv (comme pipeline.JSON_LIST_COLS, sans importer UMAP/HDBSCAN)
JSON_LIST_COLS = ["fine_cluster_ids", "recommendations", "common_cause_quotes"]
BADGES = {
    "escalade": "escalade",
    "concentration": "concentration",
    "verifier": "à vérifier",
    "inferee": "cause inférée – à confirmer",
    "defaut": "action par défaut",
    "veille": "ajoutée par le code",
}
# Série en escalade qu'aucune action du groupe ne vise : action de veille ajoutée par le code
WATCH_ACTION = {"verb": "Inspecter", "component": "l'équipement", "detail": "série en escalade sans action proposée",
                "scope": "equipment", "critic_verdict": "veille"}


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


def load_outputs(seg_dir: Path, diag_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Charge les sorties du pipeline ; les colonnes listes JSON de clusters_merged sont décodées."""
    seg_dir = Path(seg_dir)
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


def _action(rec: dict, voie: dict, data: dict, incidents: Incidents) -> dict:
    """Action d'une voie : texte (verbe, composant, détail), cibles avec leurs preuves, badges.
    L'identifiant A1..An est attribué plus tard, dans l'ordre de lecture du rapport."""
    reasons = rec.get("target_reason") or {}
    to_verify = set(rec.get("targets_to_verify") or [])
    scope = rec.get("scope") or "fleet"
    targets = []
    for t in rec.get("targets") or []:
        reason = reasons.get(t, "")
        targets.append({
            "name": t, "to_verify": t in to_verify, "escalation": reason in ("escalation", "both"),
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
    if rec.get("critic_verdict") == "fallback":
        badges.append("defaut")
    if rec.get("critic_verdict") == "veille":
        badges.append("veille")
    verb, component = _s(rec.get("verb")), _s(rec.get("component"))
    if not verb:  # action par défaut : phrase complète, sans champs
        text = _s(rec.get("action"))
        verb, component = text[:1].upper() + text[1:], ""
    return {
        "id": "", "verb": verb[:1].upper() + verb[1:], "component": component, "detail": _s(rec.get("detail")),
        "scope": scope, "scope_label": SCOPE_LABELS.get(scope, scope), "targets": targets,
        "escalation": any(t["escalation"] for t in targets), "verdict": _s(rec.get("critic_verdict")),
        "badges": [{"slug": b, "label": BADGES[b]} for b in badges],
        "voie_id": voie["id"], "voie_title": voie["title"], "voie_anchor": voie["anchor"], "asset": voie["asset"],
    }


def _voie_base(row: dict, fleet_cost: float) -> dict:
    tier = _s(row.get("criticality_tier"))
    partition = _s(row.get("partition"))
    return {
        "id": str(row["merged_cluster_id"]), "anchor": f"voie-{row['merged_cluster_id']}",
        "rank": int(row["criticality_rank"]), "title": _s(row.get("problem_title")) or str(row["merged_cluster_id"]),
        "tier": tier, "tier_slug": _slug(tier), "partition": PARTITION_LABELS.get(partition, partition),
        "asset": partition, "fleet_noun": FLEET_NOUNS.get(partition, "équipements"),
        "frequency": int(row["frequency"]), "total_cost": float(row["total_cost_eur"]),
        "mean_cost": float(row["mean_cost_eur"]), "downtime": float(row["total_downtime_hours"]),
        "cost_share": float(row["total_cost_eur"]) / fleet_cost if fleet_cost else 0.0,
        "cause_source": _s(row.get("common_cause_source")),
    }


def _voie(row: dict, data: dict, incidents: Incidents, fleet_cost: float) -> dict:
    """Fiche de preuves d'un groupe du plan, avec ses actions (non numérotées)."""
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
        fine=fine_list, representatives=[incidents.get(u) for u in reps],
    )
    recs = [r for r in row["recommendations"] if isinstance(r, dict)]
    targeted = {t for r in recs for t in r.get("targets") or []}
    esc = data["escalations"]
    series = esc.loc[esc["merged_cluster_id"] == v["id"], "equipment_id"].astype(str) if len(esc) else []
    recs += [{**WATCH_ACTION, "targets": [e], "target_reason": {e: "escalation"}}
             for e in dict.fromkeys(series) if e not in targeted]
    v["actions"] = [_action(r, v, data, incidents) for r in recs]
    # Preuves par cible (une fois par cible, même si plusieurs actions la visent)
    evidence: dict[str, dict] = {}
    for a in v["actions"]:
        for t in a["targets"]:
            e = evidence.setdefault(t["name"], {"name": t["name"], "esc": None, "conc": None})
            e["esc"] = e["esc"] or t["esc"]
            e["conc"] = e["conc"] or t["conc"]
    v["evidence"] = [e for e in evidence.values() if e["esc"] or e["conc"]]
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


# --------------------------------------------------------------------------- tableaux d'actions


def _target_rows(plan: list[dict], inc: pd.DataFrame) -> list[dict]:
    """Une ligne par cible (équipement ou site) avec ses actions, toutes voies confondues. Coût de la
    ligne : coût des incidents de la cible dans chacune de ses voies (chaque voie comptée une fois)."""
    rows: dict[tuple[str, str], dict] = {}
    for v in plan:
        members = inc[inc["merged_cluster_id"] == v["id"]]
        for a in v["actions"]:
            for t in a["targets"]:
                key = (a["scope"], t["name"])
                col = EQUIPMENT_COL if a["scope"] == "equipment" else "site"
                own = members[members[col].astype(str) == t["name"]]
                row = rows.setdefault(key, {
                    "name": t["name"], "kind": SCOPE_LABELS.get(a["scope"], a["scope"]),
                    "site": "" if a["scope"] == "equipment" else t["name"], "asset": v["asset"],
                    "voies": {}, "actions": [], "escalation": False, "to_verify": False})
                if a["scope"] == "equipment" and not row["site"]:
                    sites = inc.loc[inc[EQUIPMENT_COL].astype(str) == t["name"], "site"]
                    row["site"] = _s(sites.iloc[0]) if len(sites) else ""
                row["site_slug"] = _slug(row["site"])
                row["voies"][v["id"]] = (float(own[COST_COL].sum()), len(own))
                row["escalation"] |= t["escalation"]
                row["to_verify"] |= t["to_verify"]
                row["actions"].append({"action": a, "target": t, "cost": float(own[COST_COL].sum()),
                                       "n": len(own), "same": [],
                                       "groups": [{"title": v["title"], "anchor": v["anchor"],
                                                   "n": len(own), "cost": float(own[COST_COL].sum())}]})
    out = []
    for row in rows.values():
        row["is_site"] = row["kind"] == SCOPE_LABELS["site"]
        row["cost"] = sum(c for c, _ in row["voies"].values())
        row["n"] = sum(n for _, n in row["voies"].values())
        row["actions"].sort(key=lambda x: (not x["target"]["escalation"], -x["cost"]))
        # Inspections de veille d'une même machine (une par groupe) : une seule ligne, groupes listés
        watch = [x for x in row["actions"] if x["action"]["verdict"] == "veille"]
        for x in watch[1:]:
            watch[0]["groups"] += x["groups"]
            watch[0]["same"].append(x["action"])
            watch[0]["cost"] += x["cost"]
            watch[0]["n"] += x["n"]
        row["actions"] = [x for x in row["actions"] if x not in watch[1:]]
        out.append(row)
    out.sort(key=lambda r: -r["cost"])
    return out


def _number_actions(target_rows: list[dict], fleet: list[dict]) -> int:
    """Identifiants A1..An dans l'ordre de lecture : actions ciblées, puis programmes flotte."""
    n = 0
    for occ in [x for r in target_rows for x in r["actions"]] + [{"action": a} for a in fleet]:
        a = occ["action"]
        if not a["id"]:
            n += 1
            a["id"] = f"A{n}"
            occ["anchor"] = True  # ancre HTML sur la première occurrence (une action peut viser plusieurs cibles)
        for same in occ.get("same", []):  # inspections fusionnées : même identifiant pour chaque groupe
            same["id"] = a["id"]
    return n


# --------------------------------------------------------------------------- synthèse et vue flotte


def _summary(data: dict, fleet_cost: float) -> dict:
    """Onglet Synthèse : chiffres clés, coût mensuel par type d'actif, segmentation des problèmes
    (tous les groupes, sans les actions) et son Pareto."""
    inc = data["incidents"]
    groups = sorted(inc[PARTITION_COL].astype(str).unique())
    months = pd.to_datetime(inc[DATE_COL], errors="coerce").dt.to_period("M")
    monthly = inc.assign(month=months).groupby(["month", PARTITION_COL])[COST_COL].sum().unstack(fill_value=0.0)
    if len(monthly):
        monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"), fill_value=0.0)
    series = [{"label": PARTITION_LABELS.get(g, g), "cat": i + 1,
               "points": [(m, float(monthly.at[m, g])) for m in monthly.index] if g in monthly else []}
              for i, g in enumerate(groups)]
    by_asset = inc.groupby(PARTITION_COL)[COST_COL].sum()

    merged = data["merged"].sort_values("criticality_rank")
    rows, cum = [], 0.0
    for r in merged.to_dict("records"):
        cost = float(r["total_cost_eur"])
        cum += cost
        tier = _s(r["criticality_tier"])
        mid = str(r["merged_cluster_id"])
        partition = _s(r.get("partition"))
        status = "isolés" if mid == OUTLIER_LABEL else ("nominal" if _truthy(r.get("nominal")) else "")
        rows.append({"title": _s(r["problem_title"]) or mid, "description": _s(r.get("problem_description")),
                     "asset": PARTITION_LABELS.get(partition, partition), "n": int(r["frequency"]), "cost": cost,
                     "share": cost / fleet_cost if fleet_cost else 0.0,
                     "cum_share": cum / fleet_cost if fleet_cost else 0.0, "tier": tier, "tier_slug": _slug(tier),
                     "status": status, "href": "#sans-action" if mid == OUTLIER_LABEL else f"#voie-{mid}"})
    tiers = [{"label": t, "slug": _slug(t)} for t in TIER_ORDER if t in set(merged["criticality_tier"].astype(str))]
    return {"monthly_svg": charts.monthly_svg(series), "pareto_svg": charts.pareto_svg(rows), "groups": rows,
            "tiers": tiers, "n_groups": int((merged["merged_cluster_id"] != OUTLIER_LABEL).sum()),
            "assets": [{"label": s["label"], "cat": s["cat"], "cost": float(by_asset.get(g, 0.0)),
                        "share": float(by_asset.get(g, 0.0)) / fleet_cost if fleet_cost else 0.0}
                       for g, s in zip(groups, series)]}


def _site_chart(data: dict) -> dict:
    inc = data["incidents"]
    groups = sorted(inc[PARTITION_COL].astype(str).unique())
    by_site = (inc.groupby(["site", PARTITION_COL]).agg(cost=(COST_COL, "sum"), n=(ID_COL, "size"))
               .reset_index().sort_values("cost", ascending=False))
    site_rows = [{"site": r["site"], "group": str(r[PARTITION_COL]), "cost": float(r["cost"]), "n": int(r["n"])}
                 for r in by_site.to_dict("records")]
    return {"site_svg": charts.site_svg(site_rows, groups),
            "groups": [{"label": PARTITION_LABELS.get(g, g), "cat": i + 1} for i, g in enumerate(groups)]}


def _filters(inc: pd.DataFrame) -> dict:
    """Valeurs des filtres type d'actif et site (onglets d'actions), dans l'ordre alphabétique."""
    assets = sorted(inc[PARTITION_COL].astype(str).unique())
    sites = inc.groupby("site")[PARTITION_COL].first().astype(str)
    return {"assets": [{"key": a, "label": PARTITION_LABELS.get(a, a)} for a in assets],
            "sites": [{"slug": _slug(s), "name": s, "asset": a} for s, a in sites.items()]}


def _quality(plan: list[dict]) -> dict:
    """Qualité des actions et des causes des groupes du plan, mesurée par le code (aucun jugement LLM)."""
    verdicts = [a["verdict"] for v in plan for a in v["actions"]]
    n = len(verdicts) - verdicts.count("veille")
    src = [v["cause_source"] for v in plan]
    return {"n_actions": n, "valide": verdicts.count("valide"), "corrigee": verdicts.count("corrigee"),
            "watch": verdicts.count("veille"),
            "fallback": verdicts.count("fallback"),
            "fallback_share": verdicts.count("fallback") / n if n else 0.0,
            "n_groups": len(plan), "cause_cited": src.count("citee"), "cause_inferred": src.count("inferee")}


def _method(data: dict, n_voies: int) -> dict:
    """Chiffres de la page méthode, tous calculés depuis les sorties."""
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
    load_dotenv(find_dotenv(usecwd=True))
    m["model"] = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    return m


# --------------------------------------------------------------------------- contexte


def build_context(data: dict, out_path: Path, root: Path) -> dict:
    """Contexte des deux gabarits (dicts et listes simples, plus les SVG déjà rendus)."""
    inc = data["incidents"].copy()
    inc[COST_COL] = pd.to_numeric(inc[COST_COL], errors="coerce").fillna(0.0)
    inc[DOWNTIME_COL] = pd.to_numeric(inc[DOWNTIME_COL], errors="coerce").fillna(0.0)
    data = {**data, "incidents": inc}
    incidents = Incidents(inc, out_path.parent, root)
    fleet_cost = float(inc[COST_COL].sum())
    merged = data["merged"].sort_values("criticality_rank")
    is_nominal = merged["nominal"].map(_truthy)
    is_outlier = merged["merged_cluster_id"] == OUTLIER_LABEL

    plan = [_voie(r, data, incidents, fleet_cost) for r in merged[~is_nominal & ~is_outlier].to_dict("records")]
    targets = _target_rows(plan, inc)
    fleet = [a for v in plan for a in v["actions"] if a["scope"] == "fleet"]  # plan déjà trié par coût du groupe
    voie_by_id = {v["id"]: v for v in plan}
    for a in fleet:
        v = voie_by_id[a["voie_id"]]
        a.update(cost=v["total_cost"], cost_share=v["cost_share"], tier=v["tier"], tier_slug=v["tier_slug"],
                 fleet_noun=v["fleet_noun"])
    n_actions = _number_actions(targets, fleet)
    for v in plan:
        v["action_ids"] = [a["id"] for a in v["actions"] if a["id"]]

    nominal = [_nominal_voie(r, data, incidents, fleet_cost) for r in merged[is_nominal & ~is_outlier].to_dict("records")]
    iso = inc[inc["merged_cluster_id"] == OUTLIER_LABEL].sort_values(COST_COL, ascending=False)
    isolated = {"n": len(iso), "cost": float(iso[COST_COL].sum()),
                "incidents": [incidents.get(u) for u in iso[ID_COL]]}
    no_action = {"nominal": nominal, "isolated": isolated,
                 "total_cost": sum(v["total_cost"] for v in nominal) + isolated["cost"]}

    dates = pd.to_datetime(inc[DATE_COL], errors="coerce")
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
        "header": header, "targets": targets, "fleet": fleet, "plan": plan, "n_actions": n_actions,
        "targets_cost": sum(r["cost"] for r in targets), "no_action": no_action,
        "urgency_escalation": URGENCY_ESCALATION, "summary": _summary(data, fleet_cost),
        "site_chart": _site_chart(data), "filters": _filters(inc),
        "tabs": [{"id": "synthese", "label": "Synthèse"}, {"id": "ciblees", "label": "Actions ciblées"},
                 {"id": "flotte", "label": "Programmes flotte"}, {"id": "preuves", "label": "Preuves"},
                 {"id": "sans-action", "label": "Sans action"}, {"id": "methode", "label": "Méthode"}],
        "quality": _quality(plan), "method": _method(data, len(plan)),
    }


# --------------------------------------------------------------------------- rendu


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(eur=formatting.eur, pct=formatting.pct, num=formatting.num, hours=formatting.hours,
                       date_fr=formatting.date_fr, eur_short=formatting.eur_short,
                       share=formatting.share)
    return env


def render(ctx: dict) -> str:
    return _environment().get_template("rapport.html.j2").render(**ctx)


def build_report(
    seg_dir: Path | str = DEFAULT_SEG_DIR,
    out_path: Path | str = DEFAULT_OUT,
    root: Path | str | None = None,
    diag_dir: Path | str | None = None,
) -> Path:
    """Écrit le rapport à onglets (`out_path`). `root` : racine à laquelle `document_path` est
    relatif (défaut : répertoire courant)."""
    out_path = Path(out_path)
    root = Path(root) if root is not None else Path.cwd()
    data = load_outputs(Path(seg_dir), diag_dir)
    ctx = build_context(data, out_path, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(ctx), encoding="utf-8")
    print(f"   [OK] Rapport : {out_path} ({out_path.stat().st_size / 1024:.0f} Ko)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapport opérationnel HTML statique")
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR, help="Sorties de la segmentation")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Fichier HTML produit")
    args = parser.parse_args()
    build_report(args.seg_dir, args.out)


if __name__ == "__main__":
    main()
