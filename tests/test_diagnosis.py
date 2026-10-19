"""Tests hors-ligne du diagnostic final (LLM simulé)."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.diagnosis import _build_user_prompt, build_diagnosis_context, diagnose_merged_cluster

M = "Pompe_M1"


def _ctx(**over):
    conc = pd.DataFrame([
        dict(merged_cluster_id=M, dimension="equipment_id", value="EQ-1", observed=5, expected=1, lift=5, p_value=.01, q_value=.02, significant=True),
        dict(merged_cluster_id=M, dimension="site", value="Lyon", observed=5, expected=1, lift=5, p_value=.01, q_value=.02, significant=False),
        dict(merged_cluster_id=M, dimension="equipment_id", value="EQ-2", observed=4, expected=1, lift=4, p_value=.03, q_value=.08, significant=True, evidence_level="à vérifier"),
        dict(merged_cluster_id="Autre_M2", dimension="equipment_id", value="EQ-9", observed=5, expected=1, lift=5, p_value=.01, q_value=.02, significant=True),
    ])
    base = dict(kind="precursor", scope_cluster="", window_days=30, n_hits=6, n_equipments=4, lift=3.0, q_value=0.01,
                significant=True, plausible=True, mechanism="m", monitoring_rule="r")
    pairs = pd.DataFrame([
        dict(base, antecedent="Pompe_M3", consequent=M),
        dict(base, antecedent="Pompe_M4", consequent=M, plausible=False),
        dict(base, antecedent="Pompe_M5", consequent=M, significant=False),
        dict(base, antecedent="Pompe_M6", consequent="Pompe_M7"),
        dict(base, kind="escalation", scope_cluster=M, antecedent="Pompe_F1", consequent="Pompe_F2", window_days=7),
        dict(base, kind="escalation", scope_cluster="Autre_M2", antecedent="X_F1", consequent="X_F2"),
    ])
    ev = pd.DataFrame({"rule_id": ["Pompe_M3->Pompe_M1@30d"] * 3})
    kw = dict(
        merged_id=M, member_fine_ids=["Pompe_F1", "Pompe_F2"],
        fine_titles={"Pompe_F1": {"fine_title": "Fuite joint", "failure_mode": "fuite"}},
        fine_cards={f: {"keywords": ["a"], "size": 10, "representative_texts": ["x" * 5000] * 5}
                    for f in ("Pompe_F1", "Pompe_F2")},
        merge_log_entry={"shared_remediation": "Remplacer joints"}, concentration=conc,
        precursor_pairs=pairs, evidence=ev, impact={"frequency": 20},
    )
    kw.update(over)
    return build_diagnosis_context(**kw)


def test_context_filters(no_llm):
    ctx = _ctx()
    assert [c["value"] for c in ctx["concentration"]] == ["EQ-1", "EQ-2"]
    ids = {r["rule_id"] for r in ctx["precursor_rules"]}
    assert ids == {"Pompe_M3->Pompe_M1@30d", "Pompe_F1->Pompe_F2@7d"}
    by = {r["rule_id"]: r for r in ctx["precursor_rules"]}
    assert by["Pompe_M3->Pompe_M1@30d"]["n_evidence_incidents"] == 3


def test_invented_targets_and_rules_dropped(no_llm):
    ctx = _ctx()

    def llm(**kw):
        if kw["cache_tag"] == "nominal_check":
            return {}
        assert kw["cache_tag"] == "diagnosis_v3"
        return {
            "problem_title": "T", "problem_description": "D", "common_cause": "C", "system_impact": "I",
            "prevented_outcomes": ["panne", 3],
            "recommendations": [
                {"action": "a1", "scope": "equipment", "targets": ["EQ-1", "EQ-FAKE"], "rationale": "r"},
                {"action": "a2", "scope": "equipment", "targets": ["EQ-FAKE"], "rationale": "r"},
                {"action": "a3", "scope": "galaxy", "targets": ["EQ-1"], "rationale": "r"},
                {"action": "a4", "scope": "fleet", "targets": ["EQ-1"], "rationale": "r"},
                {"action": "a5", "scope": "equipment", "targets": ["EQ-1", "EQ-2"], "rationale": "r"},
            ],
            "early_warning_rules": [
                {"rule_id": "Pompe_M3->Pompe_M1@30d", "rule": "x", "action": "y"},
                {"rule_id": "FAKE->FAKE@7d", "rule": "x", "action": "y"},
            ],
        }

    out = diagnose_merged_cluster(ctx, llm=llm)
    r = out["recommendations"]
    assert r[0]["scope"] == "equipment" and r[0]["targets"] == ["EQ-1"]
    assert r[1]["scope"] == "fleet" and r[1]["targets"] == []
    assert r[2]["scope"] == "fleet" and r[2]["targets"] == []
    assert r[3]["scope"] == "fleet" and r[3]["targets"] == []
    assert r[0]["targets_to_verify"] == []
    assert r[4]["targets"] == ["EQ-1", "EQ-2"] and r[4]["targets_to_verify"] == ["EQ-2"]
    assert [x["rule_id"] for x in out["early_warning_rules"]] == ["Pompe_M3->Pompe_M1@30d"]
    assert out["prevented_outcomes"] == ["panne"]


def test_malformed_and_exception_defaults(no_llm):
    ctx = _ctx()
    for bad in ("nope", {"recommendations": "x", "early_warning_rules": 3, "problem_title": 5}, None):
        out = diagnose_merged_cluster(ctx, llm=lambda **kw: bad)
        assert out["problem_title"] == "Remplacer joints"
        assert out["recommendations"] == [] and out["early_warning_rules"] == []

    def boom(**kw):
        raise RuntimeError("x")

    out = diagnose_merged_cluster(_ctx(merge_log_entry=None), llm=boom)
    assert out["problem_title"] == "Fuite joint"
    assert out["prevented_outcomes"] == []


def test_prompt_size_bounded(no_llm):
    seen = {}

    def llm(**kw):
        seen["p"] = kw["user_prompt"]
        return {}

    ctx = _ctx(member_fine_ids=[f"Pompe_F{i}" for i in range(20)],
               fine_cards={f"Pompe_F{i}": {"keywords": [], "size": 1, "representative_texts": ["y" * 5000] * 4}
                           for i in range(20)})
    diagnose_merged_cluster(ctx, llm=llm)
    assert seen["p"].count("Cause :") <= 8
    assert len(seen["p"]) < 12000


def test_target_must_match_scope_dimension(no_llm):
    ctx = _ctx()
    ctx["concentration"].append(dict(dimension="period", value="2024Q1", observed=4, expected=1, lift=4, q_value=.01))
    ctx["concentration"].append(dict(dimension="site", value="Lyon", observed=4, expected=1, lift=4, q_value=.01))

    def llm(**kw):
        return {"recommendations": [
            {"action": "a", "scope": "site", "targets": ["EQ-1"], "rationale": "r"},
            {"action": "b", "scope": "equipment", "targets": ["Lyon", "2024Q1"], "rationale": "r"},
            {"action": "c", "scope": "site", "targets": ["Lyon", "2024Q1"], "rationale": "r"},
        ]}

    r = diagnose_merged_cluster(ctx, llm=llm)["recommendations"]
    assert (r[0]["scope"], r[0]["targets"]) == ("fleet", [])
    assert (r[1]["scope"], r[1]["targets"]) == ("fleet", [])
    assert (r[2]["scope"], r[2]["targets"]) == ("site", ["Lyon"])


def test_unhashable_llm_values_do_not_discard_response(no_llm):
    ctx = _ctx()
    long = "x" * 5000

    def llm(**kw):
        return {
            "problem_title": long, "problem_description": long,
            "recommendations": [
                {"action": "ok", "scope": ["equipment"], "targets": [{"a": 1}, ["EQ-1"], "EQ-1"], "rationale": "r"},
                {"action": "ok2", "scope": "equipment", "targets": ["EQ-1"],
                                          "targets_to_verify": [], "rationale": "r"},
            ],
            "early_warning_rules": [
                {"rule_id": ["Pompe_M3->Pompe_M1@30d"], "rule": "x", "action": "y"},
                {"rule_id": {"a": 1}, "rule": "x", "action": "y"},
                {"rule_id": "Pompe_M3->Pompe_M1@30d", "rule": "x", "action": "y"},
            ],
        }

    out = diagnose_merged_cluster(ctx, llm=llm)
    assert len(out["problem_title"]) == 200 and len(out["problem_description"]) == 2000
    assert out["recommendations"][1] == {"action": "ok2", "scope": "equipment", "targets": ["EQ-1"],
                                          "targets_to_verify": [], "rationale": "r"}
    assert out["recommendations"][0]["scope"] == "fleet"
    assert [x["rule_id"] for x in out["early_warning_rules"]] == ["Pompe_M3->Pompe_M1@30d"]


def _cited_ctx(mild_share=0.9):
    cards = {"Pompe_F1": {"size": 10, "keywords": [], "representative_texts": ["Joint usé par la chaleur"],
                          "representative_quotes": ["le joint était usé par la chaleur du moteur"],
                          "representative_consequences": ["arrêt de sécurité de la pompe"]}}
    return _ctx(fine_cards=cards, member_fine_ids=["Pompe_F1"],
                member_texts=["Arrêt de la pompe. Le joint était usé par la chaleur du moteur voisin."],
                mild_share=mild_share)


def test_common_cause_needs_a_quote_from_the_reports(no_llm):
    ctx = _cited_ctx()
    assert "Citation : « le joint était usé par la chaleur du moteur »" in _build_user_prompt(ctx)

    out = diagnose_merged_cluster(ctx, llm=lambda **kw: {
        "common_cause": "Chaleur du moteur voisin", "common_cause_quotes": ["Le joint était usé par la chaleur"]})
    assert out["common_cause"] == "Chaleur du moteur voisin" and out["common_cause_source"] == "diagnostic"

    # Citation absente des rapports : la cause du LLM est remplacée par la cause extraite
    out = diagnose_merged_cluster(ctx, llm=lambda **kw: {
        "common_cause": "Défaut de fabrication", "common_cause_quotes": ["un défaut de fabrication du joint"]})
    assert out["common_cause"] == "Joint usé par la chaleur" and out["common_cause_source"] == "extraction"
    assert out["common_cause_quotes"] == ["le joint était usé par la chaleur du moteur"]


def test_nominal_needs_protection_only_no_defect_and_mild_severity(no_llm):
    rec = {"action": "Revoir les seuils", "scope": "fleet", "targets": [], "rationale": "r"}

    def llm_for(protection_only, defect_found):
        def llm(**kw):
            if kw["cache_tag"] == "nominal_check":
                assert "Conséquence : arrêt de sécurité de la pompe" in kw["user_prompt"]
                return {"protection_only": protection_only, "defect_found": defect_found,
                        "reason": "protection déclenchée comme prévu"}
            return {"recommendations": [rec]}
        return llm

    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=llm_for(True, False))
    assert out["nominal"] and out["recommendations"] == []
    assert out["nominal_reason"] == "protection déclenchée comme prévu"
    # Défaut constaté : jamais nominal
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=llm_for(True, True))
    assert not out["nominal_proposed"] and not out["nominal"] and len(out["recommendations"]) == 1
    # Proposé mais trop de cas graves : refusé par le code
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.5), llm=llm_for(True, False))
    assert out["nominal_proposed"] and not out["nominal"] and len(out["recommendations"]) == 1
