"""Profil de concentration des clusters fusionnés (site, équipement, période)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from agentic_pipeline.config import (
    CONCENTRATION_COLS,
    CONCENTRATION_FDR,
    CONCENTRATION_MIN_LIFT,
    CONCENTRATION_MIN_OBS,
    CONCENTRATION_PERIOD,
    CONCENTRATION_STRONG_Q,
    DATE_COL,
    OUTLIER_LABEL,
    PARTITION_COL,
)

OUTPUT_COLUMNS = [
    "merged_cluster_id",
    "dimension",
    "value",
    "observed",
    "expected",
    "lift",
    "p_value",
    "q_value",
    "significant",
    "evidence_level",
]
STRONG, TO_VERIFY = "forte", "à vérifier"


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Corrige les p-values par Benjamini-Hochberg (q-values monotones, bornées à 1)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    # Monotonie : minimum cumulé depuis la fin
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(scaled, 0.0, 1.0)
    return q


def _dimension_series(df: pd.DataFrame, dimension: str, date_col: str) -> pd.Series:
    """Renvoie la série de valeurs (chaînes) d'une dimension ; la période est dérivée de la date."""
    if dimension == "period":
        dates = pd.to_datetime(df[date_col], errors="coerce")
        return dates.dt.to_period(CONCENTRATION_PERIOD).astype("string")
    return df[dimension].astype("string")


def compute_concentration(
    df: pd.DataFrame,
    cluster_col: str = "merged_cluster_id",
    partition_col: str = PARTITION_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """Mesure la sur-représentation de sites, équipements et périodes dans chaque cluster.

    Pour chaque cluster (hors outliers) et chaque dimension, l'effectif attendu est la taille
    du cluster multipliée par la part de la valeur dans la partition du cluster. Un test
    binomial unilatéral donne la p-value, corrigée par Benjamini-Hochberg au sein du cluster.
    Une concentration significative a un niveau de preuve « forte » si q ≤ CONCENTRATION_STRONG_Q,
    « à vérifier » sinon.
    """
    columns = [*CONCENTRATION_COLS, "period"]
    data = df[df[cluster_col].notna() & (df[cluster_col] != OUTLIER_LABEL)]

    # Partition de chaque cluster : valeur majoritaire de ses incidents
    cluster_partition = data.groupby(cluster_col)[partition_col].agg(lambda s: s.mode().iloc[0])
    partition_of_row = df[partition_col]

    rows: list[dict] = []
    for dimension in columns:
        values = _dimension_series(df, dimension, date_col)
        # Parts de base : calculées sur toutes les lignes de chaque partition
        base = pd.DataFrame({"part": partition_of_row, "val": values}).dropna()
        base_counts = base.groupby(["part", "val"]).size()
        base_totals = base.groupby("part").size()

        sub = pd.DataFrame(
            {"cluster": df[cluster_col], "val": values}, index=df.index
        ).loc[data.index].dropna()
        for (cluster, value), observed in sub.groupby(["cluster", "val"]).size().items():
            partition = cluster_partition[cluster]
            n_cluster = int((sub["cluster"] == cluster).sum())
            share = base_counts.get((partition, value), 0) / base_totals[partition]
            expected = n_cluster * share
            p_value = binomtest(int(observed), n_cluster, share, alternative="greater").pvalue
            rows.append(
                {
                    "merged_cluster_id": cluster,
                    "dimension": dimension,
                    "value": value,
                    "observed": int(observed),
                    "expected": float(expected),
                    "lift": float(observed / expected) if expected > 0 else np.inf,
                    "p_value": float(p_value),
                }
            )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(rows)
    # Correction BH au sein de chaque cluster (toutes dimensions confondues)
    out["q_value"] = out.groupby("merged_cluster_id")["p_value"].transform(
        lambda s: pd.Series(_benjamini_hochberg(s.to_numpy()), index=s.index)
    )
    out["significant"] = (
        (out["q_value"] <= CONCENTRATION_FDR)
        & (out["lift"] >= CONCENTRATION_MIN_LIFT)
        & (out["observed"] >= CONCENTRATION_MIN_OBS)
    )
    # Niveau de preuve des concentrations significatives ; vide sinon
    out["evidence_level"] = np.where(
        ~out["significant"], "", np.where(out["q_value"] <= CONCENTRATION_STRONG_Q, STRONG, TO_VERIFY)
    )
    out = out.sort_values(["merged_cluster_id", "q_value"], kind="mergesort")
    return out[OUTPUT_COLUMNS].reset_index(drop=True)
