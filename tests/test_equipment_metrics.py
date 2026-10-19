"""Tests des récidives comparées au hasard (équipements) et au site (techniciens), sans LLM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equipment_analysis.metrics import compute_equipment_metrics, compute_technician_metrics
from equipment_analysis.timeline import compute_equipment_timeline

START = pd.Timestamp("2023-01-01")


def _incidents(equipment: str, days: list[int], technician: str, site: str = "S1") -> pd.DataFrame:
    return pd.DataFrame({
        "equipment_id": equipment,
        "date": [START + pd.Timedelta(days=d) for d in days],
        "technician_id": technician,
        "site": site,
        "asset_type": "pump",
        "severity": "mineure",
        "estimated_cost_eur": 1000.0,
        "downtime_hours": 2.0,
    })


def test_paired_failures_are_above_chance_regular_ones_are_not():
    # E-PAIR : chaque panne est suivie d'une rechute 2 jours plus tard ; E-REG : une panne tous les 90 jours
    paired = sorted(d for k in range(10) for d in (k * 200, k * 200 + 2))
    regular = [k * 90 for k in range(20)]
    df = pd.concat([_incidents("E-PAIR", paired, "T1"), _incidents("E-REG", regular, "T1")], ignore_index=True)
    eq = compute_equipment_metrics(compute_equipment_timeline(df), windows=[7, 30], n_sim=500).set_index("equipment_id")

    assert eq.at["E-PAIR", "recurrences_7d"] == 10
    assert eq.at["E-PAIR", "expected_recurrences_7d"] < 3
    assert eq.at["E-PAIR", "recurrence_above_chance_7d"] and eq.at["E-PAIR", "recurrence_above_chance_30d"]
    assert not eq.at["E-REG", "recurrence_above_chance_7d"]
    assert eq.at["E-REG", "excess_recurrences_30d"] < 0
    # La récidive au-delà du hasard pèse dans le score : à coût et arrêt égaux, E-PAIR passe devant
    assert eq.at["E-PAIR", "bad_actor_score"] > eq.at["E-REG", "bad_actor_score"]


def test_no_recurrence_signal_leaves_cost_and_downtime_only():
    rng = np.random.default_rng(0)
    frames = [_incidents(f"E{i}", sorted(rng.integers(0, 1000, 30).tolist()), "T1") for i in range(4)]
    df = pd.concat(frames, ignore_index=True)
    df.loc[df.equipment_id == "E0", "estimated_cost_eur"] = 5000.0
    eq = compute_equipment_metrics(compute_equipment_timeline(df), windows=[30], n_sim=300)
    assert not eq["recurrence_above_chance_30d"].any()
    # Poids renormalisés : coût 0,4 / 0,7 et arrêt 0,3 / 0,7 (arrêt identique ici, donc nul)
    assert eq.iloc[0]["equipment_id"] == "E0" and eq.iloc[0]["bad_actor_score"] == round(40 / 0.7, 1)


def test_technician_compared_to_others_on_same_site():
    rng = np.random.default_rng(1)
    frames = []
    for i in range(6):
        # T-BAD intervient en premier, la machine retombe en panne 3 jours plus tard (intervention de T-OK)
        starts = sorted(rng.choice(np.arange(0, 1000, 40), size=10, replace=False).tolist())
        bad = _incidents(f"E{i}", starts, "T-BAD")
        ok = _incidents(f"E{i}", [d + 3 for d in starts], "T-OK")
        frames += [bad, ok]
    frames.append(_incidents("E-OTHER", [0, 200, 400], "T-ALONE", site="S2"))
    timeline = compute_equipment_timeline(pd.concat(frames, ignore_index=True))
    tech = compute_technician_metrics(timeline, windows=[7]).set_index("technician_id")

    assert tech.at["T-BAD", "post_recurrence_rate_7d"] == 100.0
    assert tech.at["T-BAD", "site_recurrence_rate_7d"] == 0.0
    assert tech.at["T-BAD", "differs_from_site_7d"] and tech.at["T-OK", "differs_from_site_7d"]
    # Seul technicien de son site : pas de comparaison possible
    assert np.isnan(tech.at["T-ALONE", "site_recurrence_rate_7d"])
    assert not tech.at["T-ALONE", "differs_from_site_7d"]
