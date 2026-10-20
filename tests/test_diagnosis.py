"""Tests hors-ligne du diagnostic final (LLM simulé)."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.diagnosis import (
    FALLBACK_ACTION, MAX_RETRY_TEXTS, SYSTEM_CAUSE_RETRY_PROMPT, SYSTEM_DIAGNOSIS_PROMPT_V2,
    SYSTEM_NOMINAL_PROMPT, SYSTEM_RECOMMENDATION_CRITIC_PROMPT, build_diagnosis_context,
    diagnose_merged_cluster, quote_found,
)

M = "Pompe_M1"
REPORT = "Arrêt de sécurité de la pompe. Le joint était usé par la chaleur du moteur voisin."
REC_KEYS = {"action", "component", "scope", "targets", "target_reason", "targets_to_verify", "rationale",
            "confirm_cause_first", "critic_verdict"}


def _escalations():
    return pd.DataFrame([
        dict(equipment_id="EQ-1", merged_cluster_id=M, partition="P", n_incidents=3,
             severity_sequence="mineure;modérée;sévère", incident_ids="a;b;c",
             dates="2024-01-01;2024-02-01;2024-03-01", costs="1;2;3", total_cost_eur=6.0,
             first_date="2024-01-01", last_date="2024-03-01"),
        dict(equipment_id="EQ-ESC", merged_cluster_id=M, partition="P", n_incidents=2,
             severity_sequence="mineure;sévère", incident_ids="d;e", dates="2024-01-05;2024-04-01",
             costs="1;9", total_cost_eur=10.0, first_date="2024-01-05", last_date="2024-04-01"),
        dict(equipment_id="EQ-OTHER", merged_cluster_id="Autre_M2", partition="P", n_incidents=2,
             severity_sequence="mineure;sévère", incident_ids="f;g", dates="2024-01-05;2024-04-01",
             costs="1;9", total_cost_eur=10.0, first_date="2024-01-05", last_date="2024-04-01"),
    ])


def _ctx(**over):
    conc = pd.DataFrame([
        dict(merged_cluster_id=M, dimension="equipment_id", value="EQ-1", observed=5, expected=1, lift=5, p_value=.01, q_value=.02, significant=True),
        dict(merged_cluster_id=M, dimension="site", value="Lyon", observed=5, expected=1, lift=5, p_value=.01, q_value=.02, significant=False),
        dict(merged_cluster_id=M, dimension="equipment_id", value="EQ-2", observed=4, expected=1, lift=4, p_value=.03, q_value=.08, significant=True, evidence_level="à vérifier"),
        dict(merged_cluster_id="Autre_M2", dimension="equipment_id", value="EQ-9", observed=5, expected=1, lift=5, p_value=.01, q_value=.02, significant=True),
    ])
    base = dict(kind="precursor", window_days=30, n_hits=6, n_equipments=4, lift=3.0, q_value=0.01,
                significant=True, plausible=True, mechanism="m", monitoring_rule="r")
    pairs = pd.DataFrame([
        dict(base, antecedent="Pompe_M3", consequent=M),
        dict(base, antecedent="Pompe_M4", consequent=M, plausible=False),
        dict(base, antecedent="Pompe_M5", consequent=M, significant=False),
        dict(base, antecedent="Pompe_M6", consequent="Pompe_M7"),
        dict(base, kind="recurrence", antecedent=M, consequent=M),
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


def _keep_all(**kw):
    return {"recommendations": [{"index": i, "verdict": "keep", "component": "joint", "action": "x"}
                                for i in range(1, 30)]}


def make_llm(main=None, critic=_keep_all, nominal=None, retry=None, seen=None):
    """LLM simulé qui répond selon le prompt système ; `seen` collecte les appels."""
    handlers = {SYSTEM_DIAGNOSIS_PROMPT_V2: main, SYSTEM_RECOMMENDATION_CRITIC_PROMPT: critic,
                SYSTEM_NOMINAL_PROMPT: nominal, SYSTEM_CAUSE_RETRY_PROMPT: retry}

    def llm(**kw):
        if seen is not None:
            seen.append(kw)
        h = handlers[kw["system_prompt"]]
        if h is None:
            return {}
        return h(**kw) if callable(h) else h
    return llm


def _cited(recs):
    """Réponse principale avec cause citée (pas de relance)."""
    return {"common_cause": "Chaleur", "common_cause_quotes": ["Le joint était usé par la chaleur"],
            "recommendations": recs}


def _cited_ctx(mild_share=0.9, **over):
    cards = {"Pompe_F1": {"size": 10, "keywords": [], "representative_texts": [REPORT]}}
    kw = dict(fine_cards=cards, member_fine_ids=["Pompe_F1"], member_texts=[REPORT], mild_share=mild_share,
              escalations=_escalations())
    kw.update(over)
    return _ctx(**kw)


def test_context_filters(no_llm):
    ctx = _ctx(escalations=_escalations())
    assert [c["value"] for c in ctx["concentration"]] == ["EQ-1", "EQ-2"]
    assert {r["rule_id"] for r in ctx["precursor_rules"]} == {"Pompe_M3->Pompe_M1@30d"}
    assert ctx["precursor_rules"][0]["n_evidence_incidents"] == 3
    assert [e["equipment_id"] for e in ctx["escalations"]] == ["EQ-1", "EQ-ESC"]
    assert ctx["escalations"][0] == {"equipment_id": "EQ-1", "n_incidents": 3,
                                     "severity_sequence": "mineure;modérée;sévère",
                                     "dates": "2024-01-01;2024-02-01;2024-03-01", "total_cost_eur": 6.0}
    assert _ctx()["escalations"] == []


def test_escalations_listed_in_prompt(no_llm):
    seen = []
    diagnose_merged_cluster(_ctx(escalations=_escalations()), llm=make_llm(seen=seen))
    p = next(k["user_prompt"] for k in seen if k["system_prompt"] == SYSTEM_DIAGNOSIS_PROMPT_V2)
    assert "EQ-ESC" in p and "mineure;sévère" in p and "EQ-OTHER" not in p


def test_invented_targets_and_rules_dropped(no_llm):
    ctx = _ctx()
    main = {
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
    out = diagnose_merged_cluster(ctx, llm=make_llm(main=main))
    r = {x["action"]: x for x in out["recommendations"]}
    assert [x["action"] for x in out["recommendations"]] == ["a1", "a5", "a2", "a3", "a4"]
    assert r["a1"]["scope"] == "equipment" and r["a1"]["targets"] == ["EQ-1"]
    assert r["a1"]["target_reason"] == {"EQ-1": "concentration"} and r["a1"]["targets_to_verify"] == []
    for a in ("a2", "a3", "a4"):
        assert r[a]["scope"] == "fleet" and r[a]["targets"] == [] and r[a]["target_reason"] == {}
    assert r["a5"]["targets"] == ["EQ-1", "EQ-2"] and r["a5"]["targets_to_verify"] == ["EQ-2"]
    assert all(set(x) == REC_KEYS for x in out["recommendations"])
    assert [x["rule_id"] for x in out["early_warning_rules"]] == ["Pompe_M3->Pompe_M1@30d"]
    assert out["prevented_outcomes"] == ["panne"]


def test_escalation_targets_and_ordering(no_llm):
    main = _cited([
        {"action": "flotte", "scope": "fleet", "targets": [], "rationale": "r"},
        {"action": "conc", "scope": "equipment", "targets": ["EQ-2"], "rationale": "r"},
        {"action": "esc", "scope": "equipment", "targets": ["EQ-ESC", "EQ-OTHER"], "rationale": "r"},
        {"action": "both", "scope": "equipment", "targets": ["EQ-1"], "rationale": "r"},
        {"action": "fake", "scope": "equipment", "targets": ["EQ-FAKE"], "rationale": "r"},
    ])
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.1), llm=make_llm(main=main))
    recs = out["recommendations"]
    assert [x["action"] for x in recs] == ["esc", "both", "conc", "flotte", "fake"]
    r = {x["action"]: x for x in recs}
    assert r["esc"]["targets"] == ["EQ-ESC"] and r["esc"]["target_reason"] == {"EQ-ESC": "escalation"}
    assert r["esc"]["targets_to_verify"] == []
    assert r["both"]["target_reason"] == {"EQ-1": "both"}
    assert r["conc"]["target_reason"] == {"EQ-2": "concentration"} and r["conc"]["targets_to_verify"] == ["EQ-2"]
    assert r["fake"]["scope"] == "fleet" and r["fake"]["targets"] == []


def test_critic_rewrite_keeps_scope_and_targets(no_llm):
    main = _cited([
        {"action": "Inspection régulière", "scope": "equipment", "targets": ["EQ-1"], "rationale": "r"},
        {"action": "Remplacer le joint", "scope": "fleet", "targets": [], "rationale": "r"},
        {"action": "Former le personnel", "scope": "fleet", "targets": [], "rationale": "r"},
        {"action": "Contrôle", "scope": "fleet", "targets": [], "rationale": "r"},
    ])
    critic = {"recommendations": [
        {"index": 1, "verdict": "rewrite", "component": "joint", "action": "Remplacer le joint usé",
         "scope": "fleet", "targets": ["EQ-9"], "target_reason": {}, "targets_to_verify": ["EQ-1"]},
        {"index": 1, "verdict": "drop"},  # doublon ignoré
        {"index": 2, "verdict": "keep", "component": "joint", "action": "autre texte"},
        {"index": 3, "verdict": "drop", "component": "", "action": ""},
        {"index": 4, "verdict": "keep", "component": "", "action": "Contrôle"},  # pas de composant
        {"index": 99, "verdict": "keep", "component": "x", "action": "inventé"},
    ]}
    recs = diagnose_merged_cluster(_cited_ctx(mild_share=0.1), llm=make_llm(main=main, critic=critic))["recommendations"]
    assert [x["action"] for x in recs] == ["Remplacer le joint usé", "Remplacer le joint"]
    assert recs[0]["scope"] == "equipment" and recs[0]["targets"] == ["EQ-1"]
    assert recs[0]["target_reason"] == {"EQ-1": "both"} and recs[0]["targets_to_verify"] == []
    assert (recs[0]["component"], recs[0]["critic_verdict"]) == ("joint", "rewrite")
    assert (recs[1]["component"], recs[1]["critic_verdict"]) == ("joint", "keep")


def test_critic_drops_all_gives_fallback(no_llm):
    main = _cited([{"action": "Surveillance renforcée", "scope": "equipment", "targets": ["EQ-1"], "rationale": "r"}])
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.1),
                                  llm=make_llm(main=main, critic={"recommendations": [{"index": 1, "verdict": "drop"}]}))
    assert out["recommendations"] == [{
        "action": FALLBACK_ACTION, "component": "", "scope": "fleet", "targets": [], "target_reason": {},
        "targets_to_verify": [], "rationale": "", "confirm_cause_first": False, "critic_verdict": "fallback"}]


def test_critic_failure_keeps_originals(no_llm):
    main = _cited([{"action": "a", "scope": "equipment", "targets": ["EQ-1"], "rationale": "r"}])

    def boom(**kw):
        raise RuntimeError("api")

    for critic in (boom, {"recommendations": "x"}, "nope"):
        recs = diagnose_merged_cluster(_cited_ctx(mild_share=0.1), llm=make_llm(main=main, critic=critic))["recommendations"]
        assert [(x["action"], x["critic_verdict"], x["component"]) for x in recs] == [("a", "non_verifie", "")]


def test_malformed_and_exception_defaults(no_llm):
    ctx = _ctx()
    for bad in ("nope", {"recommendations": "x", "early_warning_rules": 3, "problem_title": 5}, None):
        out = diagnose_merged_cluster(ctx, llm=lambda **kw: bad)
        assert out["problem_title"] == "Remplacer joints"
        assert [r["action"] for r in out["recommendations"]] == [FALLBACK_ACTION]
        assert out["early_warning_rules"] == []

    def boom(**kw):
        raise RuntimeError("x")

    out = diagnose_merged_cluster(_ctx(merge_log_entry=None), llm=boom)
    assert out["problem_title"] == "Fuite joint"
    assert out["prevented_outcomes"] == [] and out["common_cause_source"] == "inferee"
    assert out["nominal_external_condition"] == "" and out["nominal_external_quote"] == ""


def test_prompt_size_bounded(no_llm):
    seen = []
    ctx = _ctx(member_fine_ids=[f"Pompe_F{i}" for i in range(20)],
               fine_cards={f"Pompe_F{i}": {"keywords": [], "size": 1, "representative_texts": ["y" * 5000] * 4}
                           for i in range(20)})
    diagnose_merged_cluster(ctx, llm=make_llm(seen=seen))
    p = next(k["user_prompt"] for k in seen if k["system_prompt"] == SYSTEM_DIAGNOSIS_PROMPT_V2)
    assert p.count("Exemple :") <= 8
    assert len(p) < 12000
    assert seen[0]["cache_tag"] == "diagnosis_v4"


def test_target_must_match_scope_dimension(no_llm):
    ctx = _ctx()
    ctx["concentration"].append(dict(dimension="period", value="2024Q1", observed=4, expected=1, lift=4, q_value=.01))
    ctx["concentration"].append(dict(dimension="site", value="Lyon", observed=4, expected=1, lift=4, q_value=.01))
    main = {"recommendations": [
        {"action": "a", "scope": "site", "targets": ["EQ-1"], "rationale": "r"},
        {"action": "b", "scope": "equipment", "targets": ["Lyon", "2024Q1"], "rationale": "r"},
        {"action": "c", "scope": "site", "targets": ["Lyon", "2024Q1"], "rationale": "r"},
    ]}
    r = {x["action"]: x for x in diagnose_merged_cluster(ctx, llm=make_llm(main=main))["recommendations"]}
    assert (r["a"]["scope"], r["a"]["targets"]) == ("fleet", [])
    assert (r["b"]["scope"], r["b"]["targets"]) == ("fleet", [])
    assert (r["c"]["scope"], r["c"]["targets"], r["c"]["target_reason"]) == ("site", ["Lyon"], {"Lyon": "concentration"})


def test_unhashable_llm_values_do_not_discard_response(no_llm):
    long = "x" * 5000
    main = {
        "problem_title": long, "problem_description": long,
        "recommendations": [
            {"action": "ok", "scope": ["equipment"], "targets": [{"a": 1}, ["EQ-1"], "EQ-1"], "rationale": "r"},
            {"action": "ok2", "scope": "equipment", "targets": ["EQ-1"], "targets_to_verify": [], "rationale": "r"},
        ],
        "early_warning_rules": [
            {"rule_id": ["Pompe_M3->Pompe_M1@30d"], "rule": "x", "action": "y"},
            {"rule_id": {"a": 1}, "rule": "x", "action": "y"},
            {"rule_id": "Pompe_M3->Pompe_M1@30d", "rule": "x", "action": "y"},
        ],
    }
    critic = {"recommendations": [{"index": 1, "verdict": "keep", "component": "c", "action": "ok"},
                                  {"index": 2, "verdict": "keep", "component": "c", "action": "ok2"}]}
    out = diagnose_merged_cluster(_ctx(member_texts=[REPORT]),
                                  llm=make_llm(main=main, critic=critic,
                                               retry={"quotes": ["Le joint était usé par la chaleur"]}))
    assert len(out["problem_title"]) == 200 and len(out["problem_description"]) == 2000
    assert out["recommendations"][0] == {
        "action": "ok2", "component": "c", "scope": "equipment", "targets": ["EQ-1"],
        "target_reason": {"EQ-1": "concentration"}, "targets_to_verify": [], "rationale": "r",
        "confirm_cause_first": False, "critic_verdict": "keep"}
    assert out["recommendations"][1]["scope"] == "fleet"
    assert [x["rule_id"] for x in out["early_warning_rules"]] == ["Pompe_M3->Pompe_M1@30d"]


def test_quote_found_tolerates_accents_and_punctuation_only():
    assert quote_found("le joint etait use par la chaleur du moteur", REPORT)
    assert not quote_found("le joint était cassé par la foudre", REPORT)
    assert not quote_found("joint usé", REPORT)  # trop court pour être vérifiable


def test_common_cause_cited_without_retry(no_llm):
    seen = []
    out = diagnose_merged_cluster(_cited_ctx(), llm=make_llm(main={
        "common_cause": "Chaleur du moteur voisin", "common_cause_quotes": ["Le joint était usé par la chaleur"]},
        seen=seen))
    assert out["common_cause"] == "Chaleur du moteur voisin" and out["common_cause_source"] == "citee"
    assert out["common_cause_quotes"] == ["Le joint était usé par la chaleur"]
    assert not any(k["cache_tag"] == "cause_retry" for k in seen)


def test_cause_retry_succeeds(no_llm):
    seen = []
    retry = {"common_cause": "Chaleur du moteur voisin", "quotes": ["usé par la chaleur du moteur voisin"]}
    out = diagnose_merged_cluster(_cited_ctx(), llm=make_llm(main={
        "common_cause": "Défaut de fabrication", "common_cause_quotes": ["un défaut de fabrication du joint"]},
        retry=retry, seen=seen))
    assert out["common_cause_source"] == "citee" and out["common_cause"] == "Chaleur du moteur voisin"
    assert out["common_cause_quotes"] == ["usé par la chaleur du moteur voisin"]
    k = next(k for k in seen if k["system_prompt"] == SYSTEM_CAUSE_RETRY_PROMPT)
    assert k["cache_tag"] == "cause_retry" and "Défaut de fabrication" in k["user_prompt"]


def test_cause_retry_fails_marks_inferred_and_targeted_recs(no_llm):
    main = {"common_cause": "Défaut de fabrication", "common_cause_quotes": ["un défaut de fabrication du joint"],
            "recommendations": [
                {"action": "flotte", "scope": "fleet", "targets": [], "rationale": "r"},
                {"action": "cible", "scope": "equipment", "targets": ["EQ-ESC"], "rationale": "r"}]}
    for retry, cause in (({"common_cause": "", "quotes": ["phrase inventée absente des rapports"]}, "Défaut de fabrication"),
                         ({"common_cause": "Usure thermique", "quotes": []}, "Usure thermique")):
        out = diagnose_merged_cluster(_cited_ctx(mild_share=0.1), llm=make_llm(main=main, retry=retry))
        assert out["common_cause_source"] == "inferee" and out["common_cause_quotes"] == []
        assert out["common_cause"] == cause
        assert [(r["action"], r["confirm_cause_first"]) for r in out["recommendations"]] == [
            ("cible", True), ("flotte", False)]


def test_cause_retry_uses_at_most_30_texts(no_llm):
    seen = []
    texts = [f"rapport numéro {i} " + "z" * 2000 for i in range(250)]
    diagnose_merged_cluster(_ctx(member_texts=texts), llm=make_llm(seen=seen))
    p = next(k["user_prompt"] for k in seen if k["system_prompt"] == SYSTEM_CAUSE_RETRY_PROMPT)
    assert p.count("rapport numéro") == MAX_RETRY_TEXTS
    assert "rapport numéro 29 " in p and "rapport numéro 30 " not in p
    assert len(p) < MAX_RETRY_TEXTS * 700


def _nominal_llm(protection_only=True, defect_found=False, condition="vent fort",
                 quote="Arrêt de sécurité de la pompe", seen=None):
    rec = {"action": "Revoir les seuils", "scope": "fleet", "targets": [], "rationale": "r"}

    def nominal(**kw):
        assert REPORT in kw["user_prompt"]
        return {"protection_only": protection_only, "defect_found": defect_found, "external_condition": condition,
                "external_condition_quote": quote, "reason": "protection déclenchée comme prévu"}
    return make_llm(main=_cited([rec]), nominal=nominal, seen=seen)


def test_nominal_needs_quoted_external_condition_and_mild_severity(no_llm):
    seen = []
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=_nominal_llm(seen=seen))
    assert out["nominal"] and out["recommendations"] == []
    assert out["nominal_reason"] == "protection déclenchée comme prévu"
    assert out["nominal_external_condition"] == "vent fort"
    assert out["nominal_external_quote"] == "Arrêt de sécurité de la pompe"
    assert not any(k["cache_tag"] == "recommendation_critic" for k in seen)  # pas de critique si nominal
    # Défaut constaté : jamais nominal
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=_nominal_llm(defect_found=True))
    assert not out["nominal_proposed"] and not out["nominal"] and len(out["recommendations"]) == 1
    # Citation de la condition extérieure absente des rapports : refusé, citation non conservée
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=_nominal_llm(quote="rafales de vent violentes signalées"))
    assert not out["nominal_proposed"] and not out["nominal"] and out["nominal_external_quote"] == ""
    assert out["nominal_external_condition"] == "vent fort"
    # Condition vide : refusé même avec une citation retrouvée
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=_nominal_llm(condition=""))
    assert not out["nominal_proposed"]
    # Proposé mais trop de cas graves : refusé par le code
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.5), llm=_nominal_llm())
    assert out["nominal_proposed"] and not out["nominal"] and len(out["recommendations"]) == 1


def test_weak_concentration_plus_escalation_not_to_verify(no_llm):
    esc = pd.concat([_escalations(), _escalations().iloc[[1]].assign(equipment_id="EQ-2")], ignore_index=True)
    main = _cited([{"action": "x", "scope": "equipment", "targets": ["EQ-2"], "rationale": "r"}])
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.1, escalations=esc), llm=make_llm(main=main))
    rec = out["recommendations"][0]
    assert rec["target_reason"] == {"EQ-2": "both"} and rec["targets_to_verify"] == []


def _one_rec_with_critic(critic):
    main = _cited([{"action": "a", "scope": "equipment", "targets": ["EQ-1"], "rationale": "r"}])
    return diagnose_merged_cluster(_cited_ctx(mild_share=0.1), llm=make_llm(main=main, critic=critic))["recommendations"]


def test_critic_without_valid_index_is_malformed(no_llm):
    for verdicts in ([], [{"index": 0, "verdict": "keep", "component": "c", "action": "a"}]):
        recs = _one_rec_with_critic({"recommendations": verdicts})
        assert [(x["action"], x["critic_verdict"], x["component"]) for x in recs] == [("a", "non_verifie", "")]


def test_critic_accepts_float_and_string_indices(no_llm):
    recs = _one_rec_with_critic({"recommendations": [{"index": 1.0, "verdict": "keep", "component": "c", "action": "a"}]})
    assert (recs[0]["component"], recs[0]["critic_verdict"]) == ("c", "keep")
    recs = _one_rec_with_critic({"recommendations": [{"index": "1", "verdict": "keep", "component": "c", "action": "a"}]})
    assert (recs[0]["component"], recs[0]["critic_verdict"]) == ("c", "keep")
