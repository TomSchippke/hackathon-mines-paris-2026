"""Tests de la chronologie par équipement et du score Bad Actor (coût + indisponibilité), sans LLM."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.config import OUTLIER_LABEL
from equipment_analysis.metrics import compute_equipment_metrics
from equipment_analysis.timeline import compute_equipment_timeline

START = pd.Timestamp("2023-01-01")


def _incidents(equipment: str, days: list[int], cost: float = 1000.0, downtime: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame({
        "incident_id": [f"{equipment}-{d}" for d in days],
        "equipment_id": equipment,
        "date": [START + pd.Timedelta(days=d) for d in days],
        "site": "S1",
        "asset_type": "pump",
        "severity": "mineure",
        "estimated_cost_eur": cost,
        "downtime_hours": downtime,
        "cluster_id": [d % 2 for d in days],
    })


def test_timeline_columns_and_gaps():
    tl = compute_equipment_timeline(_incidents("E1", [0, 10, 40]))
    assert list(tl["incident_order"]) == [1, 2, 3]
    assert list(tl["cumulative_cost_eur"]) == [1000.0, 2000.0, 3000.0]
    assert tl["days_since_last_incident"].tolist()[1:] == [10.0, 30.0]
    assert tl["days_until_next_incident"].tolist()[:2] == [10.0, 30.0]
    assert tl.at[2, "prev_incident_id"] == "E1-10"
    assert tl.at[1, "same_cluster_as_prev"] and not tl.at[0, "same_cluster_as_prev"]
    assert not any(c.startswith(("is_recurrence", "followed_by")) or "technician" in c for c in tl.columns)


def test_isolated_incidents_are_not_a_shared_cluster():
    df = _incidents("E1", [0, 10, 40])
    df["cluster_id"] = [OUTLIER_LABEL, OUTLIER_LABEL, "C1"]
    tl = compute_equipment_timeline(df)
    assert not tl["same_cluster_as_prev"].any()


def test_equipment_metrics_basic():
    eq = compute_equipment_metrics(compute_equipment_timeline(_incidents("E1", [0, 10, 40]))).set_index("equipment_id")
    assert eq.at["E1", "total_incidents"] == 3
    assert eq.at["E1", "active_span_days"] == 40
    assert eq.at["E1", "mtbf_days"] == 20.0
    assert eq.at["E1", "median_days_between_incidents"] == 20.0
    assert eq.at["E1", "min_days_between_incidents"] == 10.0
    assert eq.at["E1", "total_cost_eur"] == 3000.0
    assert not any("recurrence" in c or "technician" in c for c in eq.columns)


def test_bad_actor_score_uses_only_cost_and_downtime():
    # Même coût et même arrêt total, chronologies très différentes : scores égaux
    df = pd.concat([
        _incidents("E-BURST", [0, 1, 2, 3]),
        _incidents("E-SPREAD", [0, 200, 400, 600]),
        _incidents("E-CHEAP", [0, 100], cost=100.0, downtime=0.5),
    ], ignore_index=True)
    eq = compute_equipment_metrics(compute_equipment_timeline(df)).set_index("equipment_id")
    assert eq.at["E-BURST", "bad_actor_score"] == eq.at["E-SPREAD", "bad_actor_score"] == 100.0
    assert eq.at["E-CHEAP", "bad_actor_score"] == 0.0


def test_bad_actor_score_weights_and_order():
    # Coût maximal seul : 4/7 ; arrêt maximal seul : 3/7
    df = pd.concat([
        _incidents("E-COST", [0, 5], cost=5000.0, downtime=1.0),
        _incidents("E-DOWN", [0, 5], cost=100.0, downtime=9.0),
        _incidents("E-ZERO", [0, 5], cost=100.0, downtime=1.0),
    ], ignore_index=True)
    eq = compute_equipment_metrics(compute_equipment_timeline(df))
    scores = eq.set_index("equipment_id")["bad_actor_score"]
    assert scores["E-COST"] == round(400 / 7, 1) and scores["E-DOWN"] == round(300 / 7, 1)
    assert scores["E-ZERO"] == 0.0
    assert list(eq["bad_actor_score"]) == sorted(eq["bad_actor_score"], reverse=True)
