"""Synthèse des coûts pour la première page du rapport : tendance annuelle et concentration du coût.

Deux questions, calculées uniquement à partir des incidents et des clusters fusionnés :
- le coût augmente-t-il ? (par année, variation entre les deux dernières années) ;
- sur quoi se concentre-t-il ? (gravité, incidents les plus chers, fiches les plus coûteuses par partition).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import (
    COST_COL,
    DATE_COL,
    DOWNTIME_COL,
    OUTLIER_LABEL,
    OVERVIEW_TOP_CLUSTERS,
    OVERVIEW_TOP_INCIDENT_SHARE,
    PARTITION_COL,
    SEVERITY_COL,
    SEVERITY_ORDER,
)


def _pct(part: float, total: float) -> float:
    return round(100 * part / total, 1) if total else 0.0


def _change(new: float, old: float) -> float | None:
    return round(100 * (new - old) / old, 1) if old else None


def _years(df: pd.DataFrame) -> list[dict]:
    """Totaux par année civile. Une année incomplète (début ou fin des données) est aussi ramenée à
    365 jours (`*_annualized`) pour rester comparable aux autres."""
    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    d = df.assign(_date=dates).dropna(subset=["_date"])
    if d.empty:
        return []
    start, end = d["_date"].min().normalize(), d["_date"].max().normalize()
    rows = []
    for year, g in d.groupby(d["_date"].dt.year):
        first, last = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31)
        days_in_year = (last - first).days + 1
        covered = (min(end, last) - max(start, first)).days + 1
        scale = days_in_year / covered
        cost, downtime = float(g[COST_COL].sum()), float(g[DOWNTIME_COL].sum())
        rows.append({
            "year": int(year), "incidents": int(len(g)), "cost": round(cost, 2), "downtime": round(downtime, 1),
            "mean_cost": round(cost / len(g), 2), "coverage_days": int(covered), "partial": covered < days_in_year,
            "incidents_annualized": round(len(g) * scale, 1), "cost_annualized": round(cost * scale, 2),
            "downtime_annualized": round(downtime * scale, 1),
            "by_partition": {str(p): {"incidents": int(len(gp)), "cost": round(float(gp[COST_COL].sum()), 2)}
                             for p, gp in g.groupby(PARTITION_COL)},
        })
    return rows


def _trend(years: list[dict]) -> dict | None:
    """Variation entre les deux dernières années (valeurs ramenées à 365 jours si l'une est incomplète)."""
    if len(years) < 2:
        return None
    old, new = years[-2], years[-1]
    return {
        "from_year": old["year"], "to_year": new["year"], "annualized": old["partial"] or new["partial"],
        "cost_from": old["cost_annualized"], "cost_to": new["cost_annualized"],
        "cost_change_pct": _change(new["cost_annualized"], old["cost_annualized"]),
        "incidents_change_pct": _change(new["incidents_annualized"], old["incidents_annualized"]),
        "mean_cost_change_pct": _change(new["mean_cost"], old["mean_cost"]),
        "downtime_change_pct": _change(new["downtime_annualized"], old["downtime_annualized"]),
    }


def _severity(df: pd.DataFrame, total_cost: float) -> list[dict]:
    """Part des incidents et part du coût par niveau de gravité, du plus grave au moins grave."""
    rows = []
    for level in reversed(SEVERITY_ORDER):
        g = df[df[SEVERITY_COL] == level]
        rows.append({"severity": level, "incidents": int(len(g)), "incident_share_pct": _pct(len(g), len(df)),
                     "cost": round(float(g[COST_COL].sum()), 2), "cost_share_pct": _pct(g[COST_COL].sum(), total_cost)})
    return rows


def _top_incidents(df: pd.DataFrame, total_cost: float) -> dict:
    n = max(1, round(len(df) * OVERVIEW_TOP_INCIDENT_SHARE))
    return {"incident_share_pct": round(100 * OVERVIEW_TOP_INCIDENT_SHARE, 1), "incidents": int(n),
            "cost_share_pct": _pct(df[COST_COL].nlargest(n).sum(), total_cost)}


def _partitions(df: pd.DataFrame, merged: pd.DataFrame, total_cost: float) -> list[dict]:
    """Par partition : son poids (incidents, coût) et les fiches qui en font l'essentiel du coût.
    Les isolés et les fiches de fonctionnement normal ne sont pas des causes à traiter : exclus du classement,
    mais comptés dans le coût total de la partition (dénominateur)."""
    nominal = merged["nominal"].astype(str).str.lower() == "true" if "nominal" in merged else False
    actionable = merged[(merged["merged_cluster_id"] != OUTLIER_LABEL) & ~nominal]
    rows = []
    for part, g in df.groupby(PARTITION_COL):
        part_cost = float(g[COST_COL].sum())
        top = actionable[actionable["partition"] == part].nlargest(OVERVIEW_TOP_CLUSTERS, "total_cost_eur")
        clusters = [{"merged_cluster_id": r["merged_cluster_id"], "problem_title": str(r["problem_title"]),
                     "incidents": int(r["frequency"]), "cost": round(float(r["total_cost_eur"]), 2),
                     "cost_share_pct": _pct(r["total_cost_eur"], part_cost)} for _, r in top.iterrows()]
        rows.append({"partition": str(part), "incidents": int(len(g)), "incident_share_pct": _pct(len(g), len(df)),
                     "cost": round(part_cost, 2), "cost_share_pct": _pct(part_cost, total_cost),
                     "n_clusters": int((actionable["partition"] == part).sum()), "top_clusters": clusters,
                     "top_cost_share_pct": _pct(sum(c["cost"] for c in clusters), part_cost)})
    return sorted(rows, key=lambda r: -r["cost"])


def compute_cost_overview(df: pd.DataFrame, merged: pd.DataFrame) -> dict[str, Any]:
    """`df` : incidents (une ligne par incident) ; `merged` : clusters fusionnés avec `partition`,
    `problem_title`, `frequency`, `total_cost_eur` et `nominal`."""
    total_cost = float(df[COST_COL].sum())
    years = _years(df)
    return {
        "incidents": int(len(df)), "cost": round(total_cost, 2), "downtime": round(float(df[DOWNTIME_COL].sum()), 1),
        "years": years, "trend": _trend(years),
        "severity": _severity(df, total_cost), "top_incidents": _top_incidents(df, total_cost),
        "partitions": _partitions(df, merged, total_cost),
    }
