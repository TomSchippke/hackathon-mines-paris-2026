"""Tests hors-ligne de agentic_pipeline.precursor_agent (LLM factice)."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.precursor_agent import check_plausibility

COLS = ["kind", "partition", "antecedent", "consequent", "window_days",
        "n_hits", "n_equipments", "n_consequent", "expected_hits", "lift", "p_value",
        "q_value", "is_best_window", "significant", "plausible", "mechanism", "monitoring_rule"]


def _row(kind="precursor", significant=True, a="P_M1", b="P_M2"):
    return dict(kind=kind, partition="P", antecedent=a, consequent=b,
                window_days=30, n_hits=8, n_equipments=4, n_consequent=12, expected_hits=2.0,
                lift=4.0, p_value=0.001, q_value=0.01, is_best_window=True,
                significant=significant, plausible=False, mechanism="", monitoring_rule="")


def _pairs(rows):
    return pd.DataFrame(rows, columns=COLS)


CARDS = {"P_M1": {"title": "Vibrations", "failure_mode": "usure", "representative_texts": ["t1", "t2", "t3"]}}


class Fake:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def __call__(self, system_prompt, user_prompt, cache_tag="general", **kw):
        self.calls.append((user_prompt, cache_tag))
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def test_valid_response(no_llm):
    f = Fake({"plausible": True, "mechanism": "usure roulement", "monitoring_rule": "si >=2 A en 30 j -> inspecter"})
    out = check_plausibility(_pairs([_row()]), CARDS, llm=f)
    assert len(f.calls) == 1 and f.calls[0][1] == "precursor_check"
    assert "P_M2" in f.calls[0][0] and "Vibrations" in f.calls[0][0]
    assert bool(out.loc[0, "plausible"]) is True
    assert out.loc[0, "mechanism"] == "usure roulement"
    assert out.loc[0, "monitoring_rule"].startswith("si")
    assert list(out.columns) == COLS


def test_string_booleans(no_llm):
    for s, exp in [("true", True), ("False", False)]:
        f = Fake({"plausible": s, "mechanism": "m", "monitoring_rule": "r"})
        out = check_plausibility(_pairs([_row()]), CARDS, llm=f)
        assert bool(out.loc[0, "plausible"]) is exp


def test_malformed(no_llm):
    for resp in ["texte", ["x"], None, {"plausible": True}, {"mechanism": "m", "monitoring_rule": "r"},
                 {"plausible": "peut-etre", "mechanism": "m", "monitoring_rule": "r"},
                 {"plausible": True, "mechanism": 3}]:
        out = check_plausibility(_pairs([_row()]), CARDS, llm=Fake(resp))
        assert bool(out.loc[0, "plausible"]) is False
        assert out.loc[0, "mechanism"] == "" and out.loc[0, "monitoring_rule"] == ""


def test_exception_continues(no_llm):
    n = {"i": 0}

    def llm(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("boom")
        return {"plausible": True, "mechanism": "m", "monitoring_rule": "r"}

    out = check_plausibility(_pairs([_row(), _row(b="P_M3")]), CARDS, llm=llm)
    assert bool(out.loc[0, "plausible"]) is False and out.loc[0, "mechanism"] == ""
    assert bool(out.loc[1, "plausible"]) is True


def test_only_significant_precursor(no_llm):
    f = Fake({"plausible": True, "mechanism": "m", "monitoring_rule": "r"})
    rows = [_row(significant=False), _row(kind="recurrence", a="P_M1", b="P_M1"),
            _row(kind="precursor")]
    out = check_plausibility(_pairs(rows), CARDS, llm=f)
    assert len(f.calls) == 1
    assert out["plausible"].tolist() == [False, False, True]
    assert out.loc[0, "mechanism"] == "" and out.loc[1, "monitoring_rule"] == ""


def test_input_not_mutated_and_empty(no_llm):
    p = _pairs([_row()])
    before = p.copy(deep=True)
    out = check_plausibility(p, CARDS, llm=Fake({"plausible": True, "mechanism": "m", "monitoring_rule": "r"}))
    pd.testing.assert_frame_equal(p, before)
    assert out is not p
    assert len(check_plausibility(_pairs([]), CARDS, llm=Fake({}))) == 0


def test_numeric_booleans(no_llm):
    for v, exp in [(1, True), (0, False), (1.0, True)]:
        out = check_plausibility(_pairs([_row()]), CARDS, llm=Fake({"plausible": v, "mechanism": "m", "monitoring_rule": "r"}))
        assert bool(out.loc[0, "plausible"]) is exp


def test_false_without_rule_keeps_mechanism(no_llm):
    f = Fake({"plausible": False, "mechanism": "cause commune externe: tempête"})
    out = check_plausibility(_pairs([_row()]), CARDS, llm=f)
    assert bool(out.loc[0, "plausible"]) is False
    assert out.loc[0, "mechanism"] == "cause commune externe: tempête"
    assert out.loc[0, "monitoring_rule"] == ""
    f = Fake({"plausible": False, "mechanism": "x", "monitoring_rule": "r"})
    assert check_plausibility(_pairs([_row()]), CARDS, llm=f).loc[0, "monitoring_rule"] == ""
    f = Fake({"plausible": True, "mechanism": "m", "monitoring_rule": None})
    o = check_plausibility(_pairs([_row()]), CARDS, llm=f)
    assert bool(o.loc[0, "plausible"]) is True and o.loc[0, "monitoring_rule"] == ""
    f = Fake({"plausible": True, "mechanism": "a" * 900, "monitoring_rule": "b" * 900})
    o = check_plausibility(_pairs([_row()]), CARDS, llm=f)
    assert len(o.loc[0, "mechanism"]) == 500 and len(o.loc[0, "monitoring_rule"]) == 500


def test_fine_card_title_fallback(no_llm):
    f = Fake({"plausible": True, "mechanism": "m", "monitoring_rule": "r"})
    cards = {"P_M1": {"fine_title": "Titre fin", "failure_mode": "x", "representative_texts": []}}
    check_plausibility(_pairs([_row()]), cards, llm=f)
    assert "Titre fin" in f.calls[0][0]
