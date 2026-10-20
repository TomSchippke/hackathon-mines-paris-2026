"""Synthèse des coûts : tendance annuelle (années incomplètes ramenées à 365 jours) et concentration."""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_pipeline.cost_overview import compute_cost_overview


def _incidents() -> pd.DataFrame:
    rows = []
    # 2024 complète : 4 incidents à 100 €. 2025 : seulement jusqu'au 1er juillet (182 jours), 2 incidents à 300 €.
    for d in ["2024-01-01", "2024-04-01", "2024-08-01", "2024-12-31"]:
        rows.append(("A_1", "solar", "mineure", d, 100.0))
    for d in ["2025-03-01", "2025-07-01"]:
        rows.append(("B_1", "wind", "sévère", d, 300.0))
    df = pd.DataFrame(rows, columns=["merged_cluster_id", "asset_type", "severity", "date", "estimated_cost_eur"])
    return df.assign(downtime_hours=1.0)


def _merged() -> pd.DataFrame:
    return pd.DataFrame({
        "merged_cluster_id": ["A_1", "B_1", "isole_atypique"],
        "partition": ["solar", "wind", ""],
        "problem_title": ["Cause A", "Cause B", "Isolés"],
        "frequency": [4, 2, 0],
        "total_cost_eur": [400.0, 600.0, 0.0],
        "nominal": [False, "True", False],  # lu depuis le CSV : booléen ou texte
    })


def test_trend_annualizes_partial_year():
    ov = compute_cost_overview(_incidents(), _merged())
    y2024, y2025 = ov["years"]
    assert not y2024["partial"] and y2025["partial"] and y2025["coverage_days"] == 182
    assert y2025["cost_annualized"] == pytest.approx(600 * 365 / 182, rel=1e-3)
    t = ov["trend"]
    assert (t["from_year"], t["to_year"], t["annualized"]) == (2024, 2025, True)
    assert t["cost_change_pct"] == pytest.approx(100 * (600 * 365 / 182 - 400) / 400, abs=0.1)
    assert t["mean_cost_change_pct"] == 200.0  # 100 € -> 300 € par incident, sans annualisation


def test_concentration_shares():
    ov = compute_cost_overview(_incidents(), _merged())
    sev = {s["severity"]: s for s in ov["severity"]}
    assert [s["severity"] for s in ov["severity"]] == ["sévère", "modérée", "mineure"]
    assert (sev["sévère"]["incident_share_pct"], sev["sévère"]["cost_share_pct"]) == (33.3, 60.0)
    assert ov["top_incidents"]["incidents"] == 1 and ov["top_incidents"]["cost_share_pct"] == 30.0
    parts = {p["partition"]: p for p in ov["partitions"]}
    # B_1 est nominal : exclu du classement, mais compté dans le coût de l'éolien
    assert parts["wind"]["top_clusters"] == [] and parts["wind"]["cost"] == 600.0
    assert [c["merged_cluster_id"] for c in parts["solar"]["top_clusters"]] == ["A_1"]
    assert parts["solar"]["top_cost_share_pct"] == 100.0


def test_single_year_has_no_trend():
    df = _incidents()
    ov = compute_cost_overview(df[df["date"].str.startswith("2024")], _merged())
    assert ov["trend"] is None and len(ov["years"]) == 1
