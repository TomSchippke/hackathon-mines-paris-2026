"""Tests du profil de concentration (données synthétiques, sans LLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_pipeline.concentration import compute_concentration
from agentic_pipeline.config import OUTLIER_LABEL


def _background(n: int, partition: str, n_equip: int = 20, seed: int = 0) -> pd.DataFrame:
    """Incidents de fond répartis uniformément sur équipements, sites et dates."""
    rng = np.random.default_rng(seed)
    equip = rng.integers(0, n_equip, n)
    return pd.DataFrame(
        {
            "asset_type": partition,
            "equipment_id": [f"{partition}_E{e}" for e in equip],
            "site": [f"S{e % 4}" for e in equip],
            "date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
            "merged_cluster_id": "bg",
        }
    )


def test_concentrated_cluster_is_significant():
    bg = _background(1000, "pump")
    n = 50
    rng = np.random.default_rng(1)
    equip = np.where(rng.random(n) < 0.8, "pump_E0", "pump_E5")
    cl = pd.DataFrame(
        {
            "asset_type": "pump",
            "equipment_id": equip,
            "site": "S0",
            "date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
            "merged_cluster_id": "pump_M1",
        }
    )
    res = compute_concentration(pd.concat([bg, cl], ignore_index=True))
    assert list(res.columns) == [
        "merged_cluster_id", "dimension", "value", "observed", "expected",
        "lift", "p_value", "q_value", "significant", "evidence_level",
    ]
    row = res[(res.merged_cluster_id == "pump_M1") & (res.dimension == "equipment_id")
              & (res.value == "pump_E0")].iloc[0]
    assert row.significant and row.lift > 5
    assert row.evidence_level == "forte"
    assert (res.loc[~res.significant, "evidence_level"] == "").all()
    assert res.merged_cluster_id.is_monotonic_increasing


def test_uniform_cluster_has_nothing_significant():
    df = _background(1500, "pump", seed=3)
    df["merged_cluster_id"] = np.where(np.arange(len(df)) % 3 == 0, "pump_M1", "pump_M2")
    res = compute_concentration(df)
    assert len(res) > 0
    assert not res.significant.any()


def test_outliers_excluded_and_partition_share_used():
    # Deux partitions : E0 vaut 50 % de "pump" mais 5 % de "valve"
    pump = _background(200, "pump", n_equip=2, seed=4)
    pump["equipment_id"] = ["E0"] * 100 + ["E1"] * 100
    valve = _background(200, "valve", n_equip=2, seed=5)
    valve["equipment_id"] = ["E0"] * 10 + ["E1"] * 190
    pump["merged_cluster_id"] = "pump_M1"
    valve["merged_cluster_id"] = "valve_M1"
    out = _background(30, "pump", seed=6)
    out["merged_cluster_id"] = OUTLIER_LABEL
    res = compute_concentration(pd.concat([pump, valve, out], ignore_index=True))

    assert OUTLIER_LABEL not in set(res.merged_cluster_id)
    r_pump = res[(res.merged_cluster_id == "pump_M1") & (res.dimension == "equipment_id")
                 & (res.value == "E0")].iloc[0]
    r_valve = res[(res.merged_cluster_id == "valve_M1") & (res.dimension == "equipment_id")
                  & (res.value == "E0")].iloc[0]
    # Base de pump : (100 + éventuels outliers sur E0) / 230 ; base de valve : 10/200 = 5 %
    assert abs(r_valve.expected - 200 * 0.05) < 1e-9
    assert abs(r_pump.expected - 200 * (100 + (out.equipment_id == "E0").sum()) / 230) < 1e-9
    assert set(res[res.dimension == "period"].value.iloc[:1]).pop().count("Q") == 1


def test_weak_concentration_is_flagged_to_verify():
    # 9 incidents sur 16 à E0, qui pèse environ 25 % de la partition : significatif mais q > 0,05
    bg = _background(400, "pump", n_equip=4, seed=7)
    rng = np.random.default_rng(8)
    cl = pd.DataFrame({
        "asset_type": "pump",
        "equipment_id": ["pump_E0"] * 9 + [f"pump_E{1 + i % 3}" for i in range(7)],
        "site": "S0",
        "date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, 16), unit="D"),
        "merged_cluster_id": "pump_M1",
    })
    res = compute_concentration(pd.concat([bg, cl], ignore_index=True))
    row = res[(res.merged_cluster_id == "pump_M1") & (res.dimension == "equipment_id")
              & (res.value == "pump_E0")].iloc[0]
    assert row.significant and 0.05 < row.q_value <= 0.10
    assert row.evidence_level == "à vérifier"
