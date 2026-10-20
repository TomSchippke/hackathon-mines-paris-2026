"""Calcul des indicateurs agrégés par équipement.

Pas d'audit par technicien : sans comparaison des récidives ajustée au site, des indicateurs
par technicien compareraient les techniciens de façon injuste (machines et conditions différentes).
"""

from __future__ import annotations

import pandas as pd

from .config import (
    ASSET_COL,
    COST_COL,
    DOWNTIME_COL,
    EQUIPMENT_COL,
    PROBLEM_COL,
    SEVERITY_COL,
    SITE_COL,
    TOP_SEVERITY,
)


def compute_equipment_metrics(timeline_df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les métriques de fiabilité (comptes, coût, indisponibilité, MTBF) par équipement."""
    records = []
    for eq_id, group in timeline_df.groupby(EQUIPMENT_COL):
        total_incidents = len(group)
        first_date = group["datetime"].min()
        last_date = group["datetime"].max()
        span_days = max(1, (last_date - first_date).days)

        site = group[SITE_COL].iloc[0] if SITE_COL in group.columns else "N/A"
        asset_type = group[ASSET_COL].iloc[0] if ASSET_COL in group.columns else "N/A"

        total_downtime = group[DOWNTIME_COL].sum() if DOWNTIME_COL in group.columns else 0.0
        mean_downtime = group[DOWNTIME_COL].mean() if DOWNTIME_COL in group.columns else 0.0
        total_cost = group[COST_COL].sum() if COST_COL in group.columns else 0.0
        mean_cost = group[COST_COL].mean() if COST_COL in group.columns else 0.0

        # MTBF : moyenne des délais entre incidents consécutifs
        valid_delays = group["days_since_last_incident"].dropna()
        mtbf_days = float(valid_delays.mean()) if len(valid_delays) > 0 else float(span_days)
        median_delay = float(valid_delays.median()) if len(valid_delays) > 0 else float(span_days)
        min_delay = float(valid_delays.min()) if len(valid_delays) > 0 else 0.0

        # Incidents au niveau de sévérité le plus élevé
        severe_count = 0
        if SEVERITY_COL in group.columns:
            severe_count = int(
                (group[SEVERITY_COL].astype(str).str.lower().str.strip() == TOP_SEVERITY.lower()).sum()
            )

        top_problem = "N/A"
        if PROBLEM_COL in group.columns:
            prob_series = group[PROBLEM_COL].dropna()
            if not prob_series.empty:
                top_problem = prob_series.mode().iloc[0]

        rec = {
            EQUIPMENT_COL: eq_id,
            "site": site,
            "asset_type": asset_type,
            "total_incidents": total_incidents,
            "first_date": first_date.strftime("%Y-%m-%d"),
            "last_date": last_date.strftime("%Y-%m-%d"),
            "active_span_days": span_days,
            "total_downtime_hours": round(total_downtime, 2),
            "mean_downtime_hours": round(mean_downtime, 2),
            "total_cost_eur": round(total_cost, 2),
            "mean_cost_eur": round(mean_cost, 2),
            "mtbf_days": round(mtbf_days, 1),
            "median_days_between_incidents": round(median_delay, 1),
            "min_days_between_incidents": round(min_delay, 1),
            "severe_incidents": severe_count,
            "most_frequent_problem": top_problem,
        }
        records.append(rec)

    res_df = pd.DataFrame(records)

    # Bad Actor Score : coût (4/7) et indisponibilité (3/7), poids relatifs 40/30 renormalisés,
    # min-max normalisés puis ramenés sur 0-100.
    if res_df.empty:
        return res_df
    c_norm = (res_df["total_cost_eur"] - res_df["total_cost_eur"].min()) / max(
        1e-5, (res_df["total_cost_eur"].max() - res_df["total_cost_eur"].min())
    )
    d_norm = (res_df["total_downtime_hours"] - res_df["total_downtime_hours"].min()) / max(
        1e-5, (res_df["total_downtime_hours"].max() - res_df["total_downtime_hours"].min())
    )
    res_df["bad_actor_score"] = ((c_norm * 0.4 + d_norm * 0.3) / 0.7 * 100).round(1)
    return res_df.sort_values(by="bad_actor_score", ascending=False).reset_index(drop=True)
