"""Reconstitution chronologique descriptive des incidents par équipement."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.config import OUTLIER_LABEL

from .config import (
    CLUSTER_COL,
    COST_COL,
    DATE_COL,
    DOWNTIME_COL,
    EQUIPMENT_COL,
    ID_COL,
    PROBLEM_COL,
    SEVERITY_COL,
)


def compute_equipment_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la timeline ordonnée des incidents pour chaque équipement.

    Retourne une copie enrichie du DataFrame avec des colonnes descriptives relatives aux délais
    inter-pannes et aux incidents précédent/suivant. Aucun indicateur de récidive : sur ces
    données, l'écart médian entre deux incidents d'un équipement (21 jours) rend un taux de
    récidive à fenêtre fixe indiscernable du taux de base.
    """
    df = df.copy()

    # S'assurer de la présence des colonnes minimales
    for col in (EQUIPMENT_COL, DATE_COL):
        if col not in df.columns:
            raise KeyError(f"Colonne requise manquante : {col}")

    # Conversion de date
    df["datetime"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    sort_cols = [EQUIPMENT_COL, "datetime"]
    if ID_COL in df.columns:
        sort_cols.append(ID_COL)
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Groupby par équipement
    grouped = df.groupby(EQUIPMENT_COL)

    # 1. Ordre de l'incident et métriques cumulées
    df["incident_order"] = grouped.cumcount() + 1
    if DOWNTIME_COL in df.columns:
        df["cumulative_downtime_hours"] = grouped[DOWNTIME_COL].cumsum().round(2)
    if COST_COL in df.columns:
        df["cumulative_cost_eur"] = grouped[COST_COL].cumsum().round(2)

    # 2. Délais inter-pannes
    prev_date = grouped["datetime"].shift(1)
    next_date = grouped["datetime"].shift(-1)

    df["days_since_last_incident"] = (df["datetime"] - prev_date).dt.total_seconds() / 86400.0
    df["days_until_next_incident"] = (next_date - df["datetime"]).dt.total_seconds() / 86400.0

    # 3. Informations sur l'incident précédent (antécédent immédiat)
    if ID_COL in df.columns:
        df["prev_incident_id"] = grouped[ID_COL].shift(1)
        df["next_incident_id"] = grouped[ID_COL].shift(-1)
    if SEVERITY_COL in df.columns:
        df["prev_severity"] = grouped[SEVERITY_COL].shift(1)
    if PROBLEM_COL in df.columns:
        df["prev_problem_title"] = grouped[PROBLEM_COL].shift(1)
    if CLUSTER_COL in df.columns:
        df["prev_cluster_id"] = grouped[CLUSTER_COL].shift(1)

    # 4. Même cluster/cause que l'incident précédent (les isolés ne forment pas un cluster)
    if CLUSTER_COL in df.columns:
        df["same_cluster_as_prev"] = (
            df["prev_cluster_id"].notna()
            & (df[CLUSTER_COL] == df["prev_cluster_id"])
            & (df[CLUSTER_COL] != OUTLIER_LABEL)
        )

    return df
