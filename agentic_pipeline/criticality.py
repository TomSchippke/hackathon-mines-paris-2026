"""Calcul des métriques opérationnelles et de la criticité financière (Pareto)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cluster_criticality(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    cost_col: str = "estimated_cost_eur",
    downtime_col: str = "downtime_hours",
) -> pd.DataFrame:
    """Calcule pour chaque cluster les métriques d'impact et son niveau de criticité Pareto."""
    # Agrégation des métriques par cluster
    agg_funcs = {
        cost_col: ["count", "mean", "sum"],
    }
    if downtime_col in df.columns:
        agg_funcs[downtime_col] = ["mean", "sum"]

    grouped = df.groupby(cluster_col).agg(agg_funcs)

    # Aplatissement des multi-index
    summary = pd.DataFrame(index=grouped.index)
    summary["frequency"] = grouped[(cost_col, "count")].astype(int)
    summary["mean_cost_eur"] = grouped[(cost_col, "mean")].round(2)
    summary["total_cost_eur"] = grouped[(cost_col, "sum")].round(2)

    if downtime_col in df.columns:
        summary["mean_downtime_hours"] = grouped[(downtime_col, "mean")].round(2)
        summary["total_downtime_hours"] = grouped[(downtime_col, "sum")].round(2)
    else:
        summary["mean_downtime_hours"] = 0.0
        summary["total_downtime_hours"] = 0.0

    summary = summary.reset_index()

    # Tri par coût total décroissant (impact opérationnel maximal en premier)
    summary = summary.sort_values(by="total_cost_eur", ascending=False).reset_index(drop=True)
    summary["criticality_rank"] = np.arange(1, len(summary) + 1)

    # Attribution des tiers de criticité selon l'analyse Pareto (règle 80/20 adaptée)
    total_fleet_cost = summary["total_cost_eur"].sum()
    if total_fleet_cost > 0:
        cum_cost_ratio = summary["total_cost_eur"].cumsum() / total_fleet_cost
    else:
        cum_cost_ratio = pd.Series(0.0, index=summary.index)

    def assign_tier(ratio: float, rank: int, total_clusters: int) -> str:
        # Si un cluster unique concentre la majorité ou rank 1 dans petit groupe
        if ratio <= 0.70 or rank == 1:
            return "Critique"
        elif ratio <= 0.90:
            return "Élevée"
        elif ratio <= 0.98:
            return "Moyenne"
        else:
            return "Faible"

    summary["criticality_tier"] = [
        assign_tier(r, rank, len(summary))
        for r, rank in zip(cum_cost_ratio, summary["criticality_rank"])
    ]

    return summary
