"""Tests de `agentic_pipeline.escalation.find_escalations` (données synthétiques)."""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_pipeline.config import OUTLIER_LABEL
from agentic_pipeline.escalation import ESCALATION_COLUMNS, find_escalations


def _rows(equip, label, sevs, costs=None, start="2024-01-01", prefix=None, asset="solar"):
    prefix = prefix or equip
    out = []
    for i, s in enumerate(sevs):
        out.append({
            "incident_id": f"{prefix}-{label}-{i}",
            "equipment_id": equip,
            "merged_cluster_id": label,
            "asset_type": asset,
            "severity": s,
            "date": (pd.Timestamp(start) + pd.Timedelta(days=10 * i)).strftime("%Y-%m-%d"),
            "estimated_cost_eur": costs[i] if costs else 100.0,
        })
    return out


def _df(*groups):
    return pd.DataFrame([r for g in groups for r in g])


SEQ = ["mineure", "mineure", "modérée", "modérée", "sévère", "sévère"]


def test_planted_series_found():
    df = _df(_rows("E1", "C1", SEQ, costs=[1, 2, 3, 4, 5, 6.5]))
    res = find_escalations(df)
    assert list(res.columns) == ESCALATION_COLUMNS
    assert len(res) == 1
    r = res.iloc[0]
    assert r["severity_sequence"] == ";".join(SEQ)
    assert r["n_incidents"] == 6
    assert r["total_cost_eur"] == 21.5
    assert r["costs"] == "1.00;2.00;3.00;4.00;5.00;6.50"
    assert r["incident_ids"].split(";")[0] == "E1-C1-0"
    assert r["first_date"] == "2024-01-01" and r["last_date"] == "2024-02-20"
    assert r["partition"] == "solar"


def test_excluded_cases():
    df = _df(
        _rows("D", "C1", ["mineure", "sévère", "modérée"]),
        _rows("F", "C1", ["modérée"] * 4),
        _rows("S", "C1", ["mineure", "sévère"]),
    )
    assert find_escalations(df).empty


def test_no_contamination():
    df = _df(
        _rows("E1", "C1", ["mineure", "mineure"]),
        _rows("E1", "C2", ["modérée", "sévère"], prefix="x"),
        _rows("E1", OUTLIER_LABEL, ["sévère"], prefix="o"),
        _rows("E2", "C1", ["sévère"], prefix="y"),
    )
    assert find_escalations(df).empty
    df2 = _df(_rows("E1", "C1", ["mineure", "modérée", "sévère"]),
              _rows("E1", OUTLIER_LABEL, ["mineure"] * 3, prefix="o"))
    res = find_escalations(df2)
    assert len(res) == 1 and res.iloc[0]["n_incidents"] == 3


def test_ranking_by_total_cost():
    s = ["mineure", "modérée", "sévère"]
    df = _df(_rows("A", "C1", s, [1, 1, 1]), _rows("B", "C1", s, [10, 10, 10]),
             _rows("C", "C1", s, [5, 5, 5]))
    assert list(find_escalations(df)["equipment_id"]) == ["B", "C", "A"]


def test_empty_input():
    res = find_escalations(_df(_rows("A", "C1", ["mineure"])).iloc[0:0])
    assert res.empty and list(res.columns) == ESCALATION_COLUMNS


def test_unknown_severity_warns_and_excluded():
    rows = _rows("A", "C1", ["mineure", "modérée", "sévère"])
    rows[1]["severity"] = "bizarre"
    with pytest.warns(UserWarning, match="sévérité inconnue"):
        res = find_escalations(_df(rows))
    assert res.empty  # il ne reste que 2 incidents


def test_same_day_tie_ordered_by_increasing_severity():
    # ids choisis pour que l'ordre par uid (chaîne) inverse la sévérité : "10" < "9"
    rows = [
        {"incident_id": "9", "equipment_id": "A", "merged_cluster_id": "C1", "asset_type": "solar",
         "severity": "mineure", "date": "2024-01-01", "estimated_cost_eur": 1.0},
        {"incident_id": "10", "equipment_id": "A", "merged_cluster_id": "C1", "asset_type": "solar",
         "severity": "modérée", "date": "2024-01-01", "estimated_cost_eur": 1.0},
        {"incident_id": "11", "equipment_id": "A", "merged_cluster_id": "C1", "asset_type": "solar",
         "severity": "sévère", "date": "2024-02-01", "estimated_cost_eur": 1.0},
    ]
    res = find_escalations(pd.DataFrame(rows))
    assert len(res) == 1
    assert res.iloc[0]["severity_sequence"] == "mineure;modérée;sévère"
    assert res.iloc[0]["incident_ids"] == "9;10;11"


def test_nan_cost_counts_zero_and_bad_date_excluded():
    rows = _rows("A", "C1", ["mineure", "modérée", "sévère", "sévère"], costs=[1.0, float("nan"), 2.0, 3.0])
    rows[3]["date"] = "pas une date"
    res = find_escalations(_df(rows))
    assert len(res) == 1
    r = res.iloc[0]
    assert r["n_incidents"] == 3
    assert r["total_cost_eur"] == 3.0
    assert r["costs"] == "1.00;0.00;2.00"
