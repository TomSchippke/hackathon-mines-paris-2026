"""Calcul des indicateurs agrégés par équipement et par technicien."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from .config import (
    ASSET_COL,
    CLUSTER_COL,
    COST_COL,
    DATE_COL,
    DOWNTIME_COL,
    EQUIPMENT_COL,
    ID_COL,
    PROBLEM_COL,
    RECURRENCE_WINDOWS,
    SEVERITY_COL,
    SITE_COL,
    TECHNICIAN_COL,
    TOP_SEVERITY,
)


def compute_equipment_metrics(
    timeline_df: pd.DataFrame,
    windows: Sequence[int] = RECURRENCE_WINDOWS,
) -> pd.DataFrame:
    """Agrège les métriques de fiabilité, MTBF et récidive par équipement."""
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

        rec_counts = {}
        rec_rates = {}
        opportunity_count = max(1, total_incidents - 1)
        for w in windows:
            col_name = f"is_recurrence_{w}d"
            cnt = int(group[col_name].sum()) if col_name in group.columns else 0
            rec_counts[f"recurrences_{w}d"] = cnt
            rec_rates[f"recurrence_rate_{w}d"] = round(cnt / opportunity_count * 100, 1)

        # Incidents au niveau de sévérité le plus élevé
        severe_count = 0
        if SEVERITY_COL in group.columns:
            severe_count = int(
                (group[SEVERITY_COL].astype(str).str.lower().str.strip() == TOP_SEVERITY.lower()).sum()
            )

        unique_techs = 0
        top_tech = "N/A"
        if TECHNICIAN_COL in group.columns:
            tech_series = group[TECHNICIAN_COL].dropna()
            unique_techs = tech_series.nunique()
            if not tech_series.empty:
                top_tech = tech_series.mode().iloc[0]

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
            "unique_technicians": unique_techs,
            "most_frequent_technician": top_tech,
            "most_frequent_problem": top_problem,
            **rec_counts,
            **rec_rates,
        }
        records.append(rec)

    res_df = pd.DataFrame(records)

    # Calcul d'un Bad Actor Score (normalisation Min-Max de coût, downtime et récidive 30j)
    if not res_df.empty:
        c_norm = (res_df["total_cost_eur"] - res_df["total_cost_eur"].min()) / max(
            1e-5, (res_df["total_cost_eur"].max() - res_df["total_cost_eur"].min())
        )
        d_norm = (res_df["total_downtime_hours"] - res_df["total_downtime_hours"].min()) / max(
            1e-5, (res_df["total_downtime_hours"].max() - res_df["total_downtime_hours"].min())
        )
        r_col = "recurrence_rate_30d"
        r_norm = (res_df[r_col] - res_df[r_col].min()) / max(
            1e-5, (res_df[r_col].max() - res_df[r_col].min())
        )
        res_df["bad_actor_score"] = (
            (c_norm * 0.4 + d_norm * 0.3 + r_norm * 0.3) * 100
        ).round(1)

    return res_df.sort_values(by="bad_actor_score", ascending=False).reset_index(drop=True)


def compute_technician_metrics(
    timeline_df: pd.DataFrame,
    windows: Sequence[int] = RECURRENCE_WINDOWS,
) -> pd.DataFrame:
    """Évalue la qualité et l'efficacité des interventions par technicien."""
    if TECHNICIAN_COL not in timeline_df.columns:
        return pd.DataFrame()

    records = []
    for tech_id, group in timeline_df.groupby(TECHNICIAN_COL):
        total_interventions = len(group)
        unique_eqs = group[EQUIPMENT_COL].nunique()
        sites = list(group[SITE_COL].dropna().unique()) if SITE_COL in group.columns else []

        mean_downtime = group[DOWNTIME_COL].mean() if DOWNTIME_COL in group.columns else 0.0
        mean_cost = group[COST_COL].mean() if COST_COL in group.columns else 0.0

        # Délai moyen de survie post-intervention (jours avant la prochaine panne sur cet équipement)
        post_delays = group["days_until_next_incident"].dropna()
        mean_survival = float(post_delays.mean()) if len(post_delays) > 0 else np.nan

        # Récidives provoquées / suivies après intervention de ce technicien
        rec_data = {}
        for w in windows:
            col_followed = f"followed_by_recurrence_{w}d"
            cnt = int(group[col_followed].sum()) if col_followed in group.columns else 0
            rec_data[f"post_recurrences_{w}d"] = cnt
            valid_opp = len(post_delays)
            rate = round(cnt / valid_opp * 100, 1) if valid_opp > 0 else 0.0
            rec_data[f"post_recurrence_rate_{w}d"] = rate

        rec = {
            TECHNICIAN_COL: tech_id,
            "total_interventions": total_interventions,
            "unique_equipments": unique_eqs,
            "sites": ", ".join(sites),
            "mean_intervention_downtime": round(mean_downtime, 2),
            "mean_intervention_cost": round(mean_cost, 2),
            "mean_post_survival_days": round(mean_survival, 1) if not np.isnan(mean_survival) else None,
            **rec_data,
        }
        records.append(rec)

    res_df = pd.DataFrame(records)
    # Trier par taux de récidive 30j ascendant (meilleurs techniciens en premier)
    sort_col = "post_recurrence_rate_30d"
    if sort_col in res_df.columns:
        res_df = res_df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    return res_df
