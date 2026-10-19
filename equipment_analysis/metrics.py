"""Calcul des indicateurs agrégés par équipement et par technicien."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from agentic_pipeline.concentration import _benjamini_hochberg

from .config import (
    ASSET_COL,
    COST_COL,
    DEFAULT_RECURRENCE_WINDOW,
    DOWNTIME_COL,
    EQUIPMENT_COL,
    PROBLEM_COL,
    RANDOM_STATE,
    RECURRENCE_FDR,
    RECURRENCE_N_SIMULATIONS,
    RECURRENCE_WINDOWS,
    SEVERITY_COL,
    SITE_COL,
    TECHNICIAN_COL,
    TOP_SEVERITY,
)


def _chance_recurrences(
    dates: pd.Series,
    windows: Sequence[int],
    n_sim: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float]]:
    """Récidives attendues au hasard et p-value unilatérale, par fenêtre, pour un équipement.

    Modèle nul : mêmes première et dernière dates, incidents intermédiaires tirés uniformément
    entre les deux. Un équipement qui tombe souvent en panne a mécaniquement beaucoup d'intervalles
    courts : seul l'excès par rapport à ce modèle signale une réparation qui ne tient pas.
    """
    days = np.sort((dates - dates.min()).dt.total_seconds().to_numpy() / 86400.0)
    if len(days) < 3:
        return {w: (float(np.sum(np.diff(days) <= w)), 1.0) for w in windows}
    span = days[-1]
    inner = np.sort(rng.uniform(0.0, span, size=(n_sim, len(days) - 2)), axis=1)
    sims = np.hstack([np.zeros((n_sim, 1)), inner, np.full((n_sim, 1), span)])
    gaps = np.diff(sims, axis=1)
    out = {}
    for w in windows:
        observed = int(np.sum(np.diff(days) <= w))
        null = np.sum(gaps <= w, axis=1)
        out[w] = (float(null.mean()), float((1 + np.sum(null >= observed)) / (1 + n_sim)))
    return out


def compute_equipment_metrics(
    timeline_df: pd.DataFrame,
    windows: Sequence[int] = RECURRENCE_WINDOWS,
    n_sim: int = RECURRENCE_N_SIMULATIONS,
) -> pd.DataFrame:
    """Agrège les métriques de fiabilité, MTBF et récidive (observée et attendue au hasard) par équipement."""
    records = []
    rng = np.random.default_rng(RANDOM_STATE)

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
        chance = _chance_recurrences(group["datetime"], windows, n_sim, rng)
        for w in windows:
            expected, p_value = chance[w]
            rec[f"expected_recurrences_{w}d"] = round(expected, 1)
            rec[f"excess_recurrences_{w}d"] = round(rec_counts[f"recurrences_{w}d"] - expected, 1)
            rec[f"recurrence_p_{w}d"] = p_value
        records.append(rec)

    res_df = pd.DataFrame(records)
    for w in windows:
        if not res_df.empty:
            res_df[f"recurrence_q_{w}d"] = _benjamini_hochberg(res_df[f"recurrence_p_{w}d"].to_numpy())
            res_df[f"recurrence_above_chance_{w}d"] = (
                (res_df[f"recurrence_q_{w}d"] <= RECURRENCE_FDR) & (res_df[f"excess_recurrences_{w}d"] > 0)
            )

    # Bad Actor Score : coût (40 %), indisponibilité (30 %) et récidives au-delà du hasard (30 %).
    # Seul l'excès significatif compte : sans aucun équipement au-dessus du hasard, le score ne
    # repose que sur le coût et l'indisponibilité (poids renormalisés).
    if not res_df.empty:
        c_norm = (res_df["total_cost_eur"] - res_df["total_cost_eur"].min()) / max(
            1e-5, (res_df["total_cost_eur"].max() - res_df["total_cost_eur"].min())
        )
        d_norm = (res_df["total_downtime_hours"] - res_df["total_downtime_hours"].min()) / max(
            1e-5, (res_df["total_downtime_hours"].max() - res_df["total_downtime_hours"].min())
        )
        w = DEFAULT_RECURRENCE_WINDOW
        if f"recurrence_above_chance_{w}d" in res_df:
            excess = res_df[f"excess_recurrences_{w}d"].where(res_df[f"recurrence_above_chance_{w}d"], 0.0)
        else:
            excess = pd.Series(0.0, index=res_df.index)
        r_weight = 0.3 if excess.max() > 0 else 0.0
        r_norm = excess / excess.max() if r_weight else excess
        res_df["bad_actor_score"] = (
            (c_norm * 0.4 + d_norm * 0.3 + r_norm * r_weight) / (0.7 + r_weight) * 100
        ).round(1)

    return res_df.sort_values(by="bad_actor_score", ascending=False).reset_index(drop=True)


def _expected_vs_site(opps: pd.DataFrame, tech_id: str, flag_col: str) -> tuple[float, float]:
    """Récidives attendues pour un technicien au taux des autres techniciens de ses sites, et p-value."""
    if flag_col not in opps.columns or SITE_COL not in opps.columns:
        return np.nan, np.nan
    mine = opps[opps[TECHNICIAN_COL] == tech_id]
    others = opps[(opps[TECHNICIAN_COL] != tech_id) & opps[SITE_COL].isin(mine[SITE_COL].unique())]
    other_rate = others.groupby(SITE_COL)[flag_col].mean()
    if mine.empty or not mine[SITE_COL].isin(other_rate.index).all():
        return np.nan, np.nan
    expected = float(mine[SITE_COL].map(other_rate).sum())
    n, k = len(mine), int(mine[flag_col].sum())
    p_value = binomtest(k, n, min(1.0, max(0.0, expected / n))).pvalue
    return expected, float(p_value)


def compute_technician_metrics(
    timeline_df: pd.DataFrame,
    windows: Sequence[int] = RECURRENCE_WINDOWS,
) -> pd.DataFrame:
    """Évalue la qualité et l'efficacité des interventions par technicien.

    Le taux de récidive d'un technicien n'a de sens que comparé aux autres techniciens du même
    site (mêmes machines, mêmes conditions) : pour chaque intervention, le taux attendu est celui
    des autres techniciens de ce site. Test binomial bilatéral, puis Benjamini-Hochberg sur les
    techniciens. Sans autre technicien sur le site, aucune comparaison n'est possible (NaN).
    """
    if TECHNICIAN_COL not in timeline_df.columns:
        return pd.DataFrame()
    opps = timeline_df[timeline_df["days_until_next_incident"].notna()]

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
            expected, p_value = _expected_vs_site(opps, tech_id, f"followed_by_recurrence_{w}d")
            rec_data[f"site_recurrence_rate_{w}d"] = (
                round(expected / valid_opp * 100, 1) if valid_opp and not np.isnan(expected) else np.nan
            )
            rec_data[f"post_recurrence_p_vs_site_{w}d"] = p_value

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
    for w in windows:
        p_col = f"post_recurrence_p_vs_site_{w}d"
        if res_df.empty or p_col not in res_df:
            continue
        tested = res_df[p_col].notna()
        q = pd.Series(np.nan, index=res_df.index)
        q[tested] = _benjamini_hochberg(res_df.loc[tested, p_col].to_numpy())
        res_df[f"post_recurrence_q_vs_site_{w}d"] = q
        res_df[f"differs_from_site_{w}d"] = q <= RECURRENCE_FDR
    # Trier par taux de récidive 30j ascendant (meilleurs techniciens en premier)
    sort_col = "post_recurrence_rate_30d"
    if sort_col in res_df.columns:
        res_df = res_df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    return res_df
