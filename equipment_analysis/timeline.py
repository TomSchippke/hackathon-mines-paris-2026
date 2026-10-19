"""Reconstitution chronologique et calcul des récidives par équipement."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .config import (
    CLUSTER_COL,
    COST_COL,
    DATE_COL,
    DOWNTIME_COL,
    EQUIPMENT_COL,
    ID_COL,
    PROBLEM_COL,
    RECURRENCE_WINDOWS,
    SEVERITY_COL,
    TECHNICIAN_COL,
)


def compute_equipment_timeline(
    df: pd.DataFrame,
    windows: Sequence[int] = RECURRENCE_WINDOWS,
) -> pd.DataFrame:
    """Calcule la timeline ordonnée des incidents pour chaque équipement et marque les récidives.
    
    Retourne une copie enrichie du DataFrame avec des colonnes relatives aux délais
    inter-pannes et aux interventions passées/futures.
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
    if TECHNICIAN_COL in df.columns:
        df["prev_technician_id"] = grouped[TECHNICIAN_COL].shift(1)
        df["next_technician_id"] = grouped[TECHNICIAN_COL].shift(-1)
    if SEVERITY_COL in df.columns:
        df["prev_severity"] = grouped[SEVERITY_COL].shift(1)
    if PROBLEM_COL in df.columns:
        df["prev_problem_title"] = grouped[PROBLEM_COL].shift(1)
    if CLUSTER_COL in df.columns:
        df["prev_cluster_id"] = grouped[CLUSTER_COL].shift(1)

    # 4. Flags de récidive pour chaque fenêtre (ex: 7j, 15j, 30j, 60j)
    for w in windows:
        # L'incident actuel est-il survenu dans les N jours après le précédent ?
        df[f"is_recurrence_{w}d"] = (
            df["days_since_last_incident"].notna() & (df["days_since_last_incident"] <= w)
        )
        # L'intervention actuelle a-t-elle été suivie d'une récidive sous N jours ?
        df[f"followed_by_recurrence_{w}d"] = (
            df["days_until_next_incident"].notna() & (df["days_until_next_incident"] <= w)
        )

    # Récidive avec le même technicien
    if TECHNICIAN_COL in df.columns:
        df["same_technician_as_prev"] = (
            df["prev_technician_id"].notna()
            & (df[TECHNICIAN_COL] == df["prev_technician_id"])
        )

    # Récidive sur le même cluster/cause si disponible
    if CLUSTER_COL in df.columns:
        df["same_cluster_as_prev"] = (
            df["prev_cluster_id"].notna()
            & (df[CLUSTER_COL] == df["prev_cluster_id"])
        )

    return df
