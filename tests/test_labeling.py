"""Tests hors ligne de labeling.py et du reclassement des outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_pipeline.agent import reassign_outliers_batch
from agentic_pipeline.config import OUTLIER_LABEL
from agentic_pipeline.labeling import build_fine_cards, label_fine_clusters


def _cards():
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame({
        "text": ["fuite huile transformateur"] * 5 + ["vibration anormale rotor pale"] * 5 + ["divers", "autre"],
        "severity": ["mineure", "sévère", "mineure", "mineure", "modérée"] * 2 + ["mineure", "sévère"],
        "estimated_cost_eur": np.arange(n) * 100.0,
        "site": ["A"] * n,
    })
    labels = np.array(["P_F0"] * 5 + ["P_F1"] * 5 + [OUTLIER_LABEL] * 2)
    emb = rng.normal(size=(n, 4))
    return build_fine_cards(df, labels, emb, "P")


def test_build_fine_cards():
    cards = _cards()
    assert set(cards) == {"P_F0", "P_F1"}
    c = cards["P_F0"]
    assert c["partition"] == "P" and c["size"] == 5
    assert c["fine_cluster_id"] == "P_F0"
    assert c["severity_distribution"] == {"mineure": 3, "sévère": 1, "modérée": 1}
    assert len(c["representative_texts"]) == 4
    assert all("fuite" in t for t in c["representative_texts"])


def test_label_valid(no_llm):
    cards = _cards()
    fake = lambda **kw: {"fine_title": " T ", "failure_mode": "mode"}
    out = label_fine_clusters(cards, llm=fake)
    assert out["P_F0"] == {"fine_title": "T", "failure_mode": "mode"}


def test_label_missing_keys_and_non_dict(no_llm):
    cards = _cards()
    out = label_fine_clusters(cards, llm=lambda **kw: {})
    for fid, card in cards.items():
        assert out[fid]["fine_title"] == " ".join(card["keywords"][:3])
        assert out[fid]["failure_mode"] == ""
    out = label_fine_clusters(cards, llm=lambda **kw: ["x"])
    assert set(out) == set(cards)
    out = label_fine_clusters(cards, llm=lambda **kw: {"fine_title": 3, "failure_mode": None})
    assert out["P_F1"]["failure_mode"] == ""


def test_label_exception(no_llm):
    cards = _cards()

    def boom(**kw):
        raise RuntimeError("x")

    out = label_fine_clusters(cards, llm=boom)
    assert set(out) == set(cards) and out["P_F0"]["failure_mode"] == ""


def test_label_default_llm_is_guarded(no_llm):
    # call_llm_json par défaut est lié à l'import : une erreur ne doit pas planter
    out = label_fine_clusters(_cards(), llm=no_llm)
    assert set(out) == {"P_F0", "P_F1"}


TAX = [
    {"cluster_id": "P_F0", "problem_title": "Fuite", "common_cause": "joint"},
    {"cluster_id": "P_F1", "problem_title": "Vibration", "common_cause": "rotor"},
]
OUT = [{"incident_id": 1, "text": "a"}, {"incident_id": 2, "text": "b"}, {"incident_id": 3, "text": "c"}]


def test_reassign_valid_unknown_missing(no_llm):
    fake = lambda **kw: {"classifications": [
        {"incident_id": 1, "assigned_cluster_id": "P_F1", "reason": "ok"},
        {"incident_id": 2, "assigned_cluster_id": "ZZ_F9"},
        "garbage",
    ]}
    res = reassign_outliers_batch(OUT, TAX, llm=fake)
    assert [r["assigned_cluster_id"] for r in res] == ["P_F1", OUTLIER_LABEL, OUTLIER_LABEL]
    assert res[0]["reason"] == "ok"


def test_reassign_malformed(no_llm):
    for bad in ({"classifications": "nope"}, {"classifications": None}, {}, ["x"]):
        res = reassign_outliers_batch(OUT, TAX, llm=lambda bad=bad, **kw: bad)
        assert [r["assigned_cluster_id"] for r in res] == [OUTLIER_LABEL] * 3


def test_reassign_llm_exception(no_llm):
    def boom(**kw):
        raise RuntimeError("x")

    res = reassign_outliers_batch(OUT, TAX, llm=boom)
    assert [r["assigned_cluster_id"] for r in res] == [OUTLIER_LABEL] * 3


def test_reassign_reason_none(no_llm):
    fake = lambda **kw: {"classifications": [{"incident_id": 1, "assigned_cluster_id": "P_F0", "reason": None}]}
    assert reassign_outliers_batch(OUT, TAX, llm=fake)[0]["reason"] == "Analyse causale LLM."


def test_default_label_empty_keywords_and_caps(no_llm):
    card = {"fine_cluster_id": "P_F7", "partition": "P", "size": 1, "keywords": [],
            "representative_texts": [], "severity_distribution": {}}
    out = label_fine_clusters({"P_F7": card}, llm=lambda **kw: {})
    assert out["P_F7"]["fine_title"] == "P_F7"
    out = label_fine_clusters({"P_F7": card}, llm=lambda **kw: {"fine_title": "t" * 900, "failure_mode": "m" * 900})
    assert len(out["P_F7"]["fine_title"]) == 200 and len(out["P_F7"]["failure_mode"]) == 500
