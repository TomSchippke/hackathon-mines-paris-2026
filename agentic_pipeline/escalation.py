"""Veille d'escalade opérationnelle : sévérité qui monte sur une même machine.

Liste descriptive, sans test de significativité : avec au plus quelques incidents par série,
un test par série ne survit jamais à la correction pour tests multiples (par exemple, 6
incidents sur 3 niveaux de sévérité donnent au plus 90 ordres distinguables, donc p >= ~0,011).

Une série = tous les incidents d'un même ``merged_cluster_id`` sur un même ``equipment_id``,
sur toute la période (pas de fenêtre). Elle est retenue si elle compte au moins
``ESCALATION_MIN_INCIDENTS`` incidents et si, triée par date (ex æquo : sévérité croissante, puis incident_id), le rang
de sévérité ne décroît jamais et le dernier rang dépasse strictement le premier.
Classement par coût total de la série (décroissant).

Complète le profil de concentration (qui dit *où*) avec *ça empire dans le temps sur la même
machine*. Aucun LLM, aucun réseau ; résultat déterministe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import (
    COST_COL, DATE_COL, EQUIPMENT_COL, ESCALATION_MIN_INCIDENTS, ID_COL, OUTLIER_LABEL,
    PARTITION_COL, SEVERITY_COL, SEVERITY_ORDER,
)

ESCALATION_COLUMNS = ["equipment_id", "merged_cluster_id", "partition", "n_incidents",
                      "severity_sequence", "incident_ids", "dates", "costs",
                      "total_cost_eur", "first_date", "last_date"]


def _severity_ranks(series: pd.Series) -> pd.Series:
    """Rang de sévérité ; sévérité inconnue -> NaN (avertissement unique, ligne exclue)."""
    # contrairement à precursors._severity_ranks (rang max), on exclut : une veille n'invente pas d'escalade
    order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    ranks = series.map(lambda s: order.get(s, order.get(str(s).strip().lower())))
    unknown = ranks.isna()
    if unknown.any():
        warnings.warn(
            f"{int(unknown.sum())} incident(s) avec une sévérité inconnue "
            f"({sorted(series[unknown].astype(str).unique())[:5]}) : exclus de la veille d'escalade.",
            stacklevel=3,
        )
    return ranks


def find_escalations(df: pd.DataFrame, label_col: str = "merged_cluster_id") -> pd.DataFrame:
    """Séries (équipement x cluster fusionné) dont la sévérité ne fait que monter.

    Retourne un DataFrame de colonnes ``ESCALATION_COLUMNS``, trié par coût total décroissant ;
    vide si aucune série ne qualifie.
    """
    work = pd.DataFrame({
        "uid": df[ID_COL].astype(str).to_numpy(),
        "label": df[label_col].to_numpy(),
        "equipment": df[EQUIPMENT_COL].astype(str).to_numpy(),
        "date": pd.to_datetime(df[DATE_COL], errors="coerce").to_numpy(),
        "sev": _severity_ranks(df[SEVERITY_COL]).to_numpy(),
        "cost": pd.to_numeric(df[COST_COL], errors="coerce").fillna(0.0).to_numpy(),
        "partition": (df[PARTITION_COL].astype(str).to_numpy()
                      if PARTITION_COL in df.columns else np.full(len(df), "", dtype=object)),
    })
    work = work[work["label"].notna() & (work["label"] != OUTLIER_LABEL)
                & work["date"].notna() & work["sev"].notna()].copy()
    work["label"] = work["label"].astype(str)
    work["sev"] = work["sev"].astype(int)
    # ex æquo de date : sévérité croissante (jamais un ordre arbitraire par uid, "10" < "9"), puis uid
    work = work.sort_values(["equipment", "label", "date", "sev", "uid"], kind="stable")

    rows = []
    for (equip, label), g in work.groupby(["equipment", "label"], sort=True):
        if len(g) < ESCALATION_MIN_INCIDENTS:
            continue
        sev = g["sev"].to_numpy()
        if np.any(np.diff(sev) < 0) or sev[-1] <= sev[0]:
            continue
        dates = g["date"].dt.strftime("%Y-%m-%d")
        rows.append({
            "equipment_id": equip,
            "merged_cluster_id": label,
            "partition": g["partition"].iloc[0],
            "n_incidents": int(len(g)),
            "severity_sequence": ";".join(SEVERITY_ORDER[r] for r in sev),
            "incident_ids": ";".join(g["uid"]),
            "dates": ";".join(dates),
            "costs": ";".join(f"{c:.2f}" for c in g["cost"]),
            "total_cost_eur": round(float(g["cost"].sum()), 2),
            "first_date": dates.iloc[0],
            "last_date": dates.iloc[-1],
        })
    if not rows:
        return pd.DataFrame(columns=ESCALATION_COLUMNS)
    out = pd.DataFrame(rows, columns=ESCALATION_COLUMNS)
    return (out.sort_values(["total_cost_eur", "equipment_id", "merged_cluster_id"],
                            ascending=[False, True, True], kind="stable")
            .reset_index(drop=True))
