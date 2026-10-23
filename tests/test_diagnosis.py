"""Tests hors-ligne du diagnostic final (LLM simulé)."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.diagnosis import (
    FALLBACK_ACTION, MAX_RETRY_TEXTS, SYSTEM_ACTION_RETRY_PROMPT, SYSTEM_CAUSE_RETRY_PROMPT,
    SYSTEM_DIAGNOSIS_PROMPT_V2, SYSTEM_NOMINAL_PROMPT, build_diagnosis_context, diagnose_merged_cluster,
    quote_found,
)

M = "Pompe_M1"
REPORT = "Arrêt de sécurité de la pompe. Le joint était usé par la chaleur du moteur voisin."
REC_KEYS = {"action", "verb", "component", "detail", "scope", "targets", "target_reason", "targets_to_verify",
            "rationale", "confirm_cause_first", "critic_verdict"}


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
    kw = dict(
        merged_id=M, member_fine_ids=["Pompe_F1", "Pompe_F2"],
        fine_titles={"Pompe_F1": {"fine_title": "Fuite joint", "failure_mode": "fuite"}},
        fine_cards={f: {"keywords": ["a"], "size": 10, "representative_texts": ["x" * 5000] * 5}
                    for f in ("Pompe_F1", "Pompe_F2")},
        merge_log_entry={"shared_remediation": "Remplacer joints"}, concentration=conc,
        impact={"frequency": 20}, member_texts=[REPORT],
    )
    kw.update(over)
    return build_diagnosis_context(**kw)


def rec(component="joint", scope="fleet", targets=(), verbe="Remplacer", detail=""):
    return {"verbe": verbe, "composant": component, "detail": detail, "scope": scope,
            "targets": list(targets), "rationale": "r"}


def make_llm(main=None, retry_action=None, nominal=None, retry=None, seen=None):
    """LLM simulé qui répond selon le prompt système ; `seen` collecte les appels."""
    handlers = {SYSTEM_DIAGNOSIS_PROMPT_V2: main, SYSTEM_ACTION_RETRY_PROMPT: retry_action,
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
    kw = dict(fine_cards=cards, member_fine_ids=["Pompe_F1"], mild_share=mild_share, escalations=_escalations())
    kw.update(over)
    return _ctx(**kw)


def _run(recs, retry_action=None, seen=None, **ctx):
    ctx.setdefault("mild_share", 0.1)
    return diagnose_merged_cluster(_cited_ctx(**ctx), llm=make_llm(main=_cited(recs), retry_action=retry_action,
                                                                    seen=seen))


def test_context_filters(no_llm):
    ctx = _ctx(escalations=_escalations())
    assert [c["value"] for c in ctx["concentration"]] == ["EQ-1", "EQ-2"]
    assert [e["equipment_id"] for e in ctx["escalations"]] == ["EQ-1", "EQ-ESC"]
    assert ctx["escalations"][0] == {"equipment_id": "EQ-1", "n_incidents": 3,
                                     "severity_sequence": "mineure;modérée;sévère",
                                     "dates": "2024-01-01;2024-02-01;2024-03-01", "total_cost_eur": 6.0}
    assert _ctx()["escalations"] == []
    assert "precursor_rules" not in ctx


def test_escalations_listed_in_prompt(no_llm):
    seen = []
    diagnose_merged_cluster(_ctx(escalations=_escalations()), llm=make_llm(seen=seen))
    p = next(k["user_prompt"] for k in seen if k["system_prompt"] == SYSTEM_DIAGNOSIS_PROMPT_V2)
    assert "EQ-ESC" in p and "mineure;sévère" in p and "EQ-OTHER" not in p
    assert "précurseur" not in p.lower()


def test_invented_targets_dropped(no_llm):
    main = {
        "problem_title": "T", "problem_description": "D", "common_cause": "C",
        "recommendations": [
            rec("joint", "equipment", ["EQ-1", "EQ-FAKE"]),
            rec("pompe", "equipment", ["EQ-FAKE"]),
            rec("moteur", "galaxy", ["EQ-1"]),
            rec("joint usé", "fleet", ["EQ-1"]),
            rec("joint", "equipment", ["EQ-1", "EQ-2"], verbe="Nettoyer"),
        ],
    }
    out = diagnose_merged_cluster(_ctx(), llm=make_llm(main=main))
    r = {x["action"]: x for x in out["recommendations"]}
    assert [x["action"] for x in out["recommendations"]] == [
        "Remplacer joint", "Nettoyer joint", "Remplacer pompe", "Remplacer moteur", "Remplacer joint usé"]
    assert r["Remplacer joint"]["scope"] == "equipment" and r["Remplacer joint"]["targets"] == ["EQ-1"]
    assert r["Remplacer joint"]["target_reason"] == {"EQ-1": "concentration"}
    assert r["Remplacer joint"]["targets_to_verify"] == []
    for a in ("Remplacer pompe", "Remplacer moteur", "Remplacer joint usé"):
        assert r[a]["scope"] == "fleet" and r[a]["targets"] == [] and r[a]["target_reason"] == {}
    assert r["Nettoyer joint"]["targets"] == ["EQ-1", "EQ-2"] and r["Nettoyer joint"]["targets_to_verify"] == ["EQ-2"]
    assert all(set(x) == REC_KEYS for x in out["recommendations"])
    assert {"system_impact", "prevented_outcomes", "early_warning_rules"}.isdisjoint(out)


def test_escalation_targets_and_ordering(no_llm):
    recs = _run([
        rec("pompe"),
        rec("joint", "equipment", ["EQ-2"]),
        rec("moteur", "equipment", ["EQ-ESC", "EQ-OTHER"]),
        rec("joint", "equipment", ["EQ-1"], verbe="Nettoyer"),
        rec("moteur voisin", "equipment", ["EQ-FAKE"]),
    ])["recommendations"]
    assert [x["component"] for x in recs] == ["moteur", "joint", "joint", "pompe", "moteur voisin"]
    esc, both, conc = recs[0], recs[1], recs[2]
    assert esc["targets"] == ["EQ-ESC"] and esc["target_reason"] == {"EQ-ESC": "escalation"}
    assert esc["targets_to_verify"] == []
    assert both["target_reason"] == {"EQ-1": "both"}
    assert conc["target_reason"] == {"EQ-2": "concentration"} and conc["targets_to_verify"] == ["EQ-2"]
    assert recs[4]["scope"] == "fleet" and recs[4]["targets"] == []


def test_action_text_built_by_code(no_llm):
    long = " ".join(f"mot{i}" for i in range(30))
    r = _run([rec("joint", detail="vérifier l'usure thermique"), rec("pompe", verbe="inspecter", detail=long)])
    a, b = r["recommendations"]
    assert a["action"] == "Remplacer joint : vérifier l'usure thermique" and a["critic_verdict"] == "valide"
    assert b["action"].startswith("Inspecter pompe : mot0") and b["detail"].endswith("mot11…")


def test_code_refuses_generic_or_ungrounded(no_llm):
    seen = []
    out = _run([
        rec("joint", verbe="Surveiller"),                 # verbe générique
        rec("anomalies", verbe="Inspecter"),               # composant vague
        rec("roulement", verbe="Remplacer"),               # absent des rapports
        rec("joint", verbe="joint"),                       # pas un infinitif
        rec("joints", verbe="Remplacer"),                  # pluriel retrouvé : accepté
    ], seen=seen)
    assert [r["component"] for r in out["recommendations"]] == ["joints"]
    retry = next(k for k in seen if k["system_prompt"] == SYSTEM_ACTION_RETRY_PROMPT)
    p = retry["user_prompt"]
    assert retry["cache_tag"] == "action_retry"
    assert "ne dit pas quoi faire" in p and "ne désigne pas" in p and "ne figure dans aucun rapport" in p
    assert "à l'infinitif" in p and REPORT in p
    assert "surveiller" not in SYSTEM_ACTION_RETRY_PROMPT.lower()  # liste de refus jamais montrée au LLM


def test_retry_fixes_keep_scope_and_targets(no_llm):
    retry = {"recommendations": [
        {"index": 1, "verbe": "Remplacer", "composant": "joint", "detail": "côté moteur",
         "scope": "fleet", "targets": ["EQ-9"]},
        {"index": 1, "verbe": "Inspecter", "composant": "pompe"},  # doublon ignoré
        {"index": "2", "verbe": "Surveiller", "composant": "pompe"},  # toujours refusée
        {"index": 99, "verbe": "Remplacer", "composant": "moteur"},
    ]}
    recs = _run([rec("roulement", "equipment", ["EQ-1"]), rec("palier")], retry_action=retry)["recommendations"]
    assert len(recs) == 1
    r = recs[0]
    assert (r["action"], r["critic_verdict"]) == ("Remplacer joint : côté moteur", "corrigee")
    assert r["scope"] == "equipment" and r["targets"] == ["EQ-1"] and r["target_reason"] == {"EQ-1": "both"}


def test_all_refused_gives_fallback(no_llm):
    def boom(**kw):
        raise RuntimeError("api")

    for retry in (None, boom, {"recommendations": "x"}, "nope",
                  {"recommendations": [{"index": 1, "verbe": "Surveiller", "composant": "joint"}]}):
        out = _run([rec("joint", "equipment", ["EQ-1"], verbe="Surveiller")], retry_action=retry)
        assert out["recommendations"] == [{
            "action": FALLBACK_ACTION, "verb": "", "component": "", "detail": "", "scope": "fleet", "targets": [],
            "target_reason": {}, "targets_to_verify": [], "rationale": "", "confirm_cause_first": False,
            "critic_verdict": "fallback"}]


def test_no_retry_call_when_all_valid(no_llm):
    seen = []
    _run([rec("joint")], seen=seen)
    assert not any(k["system_prompt"] == SYSTEM_ACTION_RETRY_PROMPT for k in seen)


def test_duplicates_merged(no_llm):
    recs = _run([rec("joint", "equipment", ["EQ-1"]), rec("Joint", "equipment", ["EQ-2"]),
                 rec("joint")])["recommendations"]
    assert [(r["scope"], r["targets"]) for r in recs] == [("equipment", ["EQ-1", "EQ-2"]), ("fleet", [])]
    assert recs[0]["target_reason"] == {"EQ-1": "both", "EQ-2": "concentration"}
    assert recs[0]["targets_to_verify"] == ["EQ-2"]


def test_malformed_and_exception_defaults(no_llm):
    ctx = _ctx()
    for bad in ("nope", {"recommendations": "x", "problem_title": 5}, None):
        out = diagnose_merged_cluster(ctx, llm=lambda **kw: bad)
        assert out["problem_title"] == "Remplacer joints"
        assert [r["action"] for r in out["recommendations"]] == [FALLBACK_ACTION]

    def boom(**kw):
        raise RuntimeError("x")

    out = diagnose_merged_cluster(_ctx(merge_log_entry=None), llm=boom)
    assert out["problem_title"] == "Fuite joint"
    assert out["common_cause_source"] == "inferee"
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
    assert seen[0]["cache_tag"] == "diagnosis_v5"


def test_target_must_match_scope_dimension(no_llm):
    ctx = _ctx()
    ctx["concentration"].append(dict(dimension="period", value="2024Q1", observed=4, expected=1, lift=4, q_value=.01))
    ctx["concentration"].append(dict(dimension="site", value="Lyon", observed=4, expected=1, lift=4, q_value=.01))
    main = {"recommendations": [rec("joint", "site", ["EQ-1"]), rec("pompe", "equipment", ["Lyon", "2024Q1"]),
                                rec("moteur", "site", ["Lyon", "2024Q1"])]}
    r = {x["component"]: x for x in diagnose_merged_cluster(ctx, llm=make_llm(main=main))["recommendations"]}
    assert (r["joint"]["scope"], r["joint"]["targets"]) == ("fleet", [])
    assert (r["pompe"]["scope"], r["pompe"]["targets"]) == ("fleet", [])
    assert (r["moteur"]["scope"], r["moteur"]["targets"], r["moteur"]["target_reason"]) == ("site", ["Lyon"], {"Lyon": "concentration"})


def test_unhashable_llm_values_do_not_discard_response(no_llm):
    long = "x" * 5000
    main = {
        "problem_title": long, "problem_description": long,
        "recommendations": [
            {"verbe": "Remplacer", "composant": "joint", "scope": ["equipment"], "targets": [{"a": 1}, ["EQ-1"], "EQ-1"]},
            {"verbe": "Inspecter", "composant": "pompe", "scope": "equipment", "targets": ["EQ-1"], "rationale": "r"},
            {"verbe": ["x"], "composant": {"a": 1}},
        ],
    }
    out = diagnose_merged_cluster(_ctx(), llm=make_llm(main=main, retry={"quotes": ["Le joint était usé par la chaleur"]}))
    assert len(out["problem_title"]) == 200 and len(out["problem_description"]) == 2000
    assert out["recommendations"][0] == {
        "action": "Inspecter pompe", "verb": "Inspecter", "component": "pompe", "detail": "", "scope": "equipment",
        "targets": ["EQ-1"], "target_reason": {"EQ-1": "concentration"}, "targets_to_verify": [], "rationale": "r",
        "confirm_cause_first": False, "critic_verdict": "valide"}
    assert out["recommendations"][1]["scope"] == "fleet" and len(out["recommendations"]) == 2


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
            "recommendations": [rec("pompe"), rec("joint", "equipment", ["EQ-ESC"])]}
    for retry, cause in (({"common_cause": "", "quotes": ["phrase inventée absente des rapports"]}, "Défaut de fabrication"),
                         ({"common_cause": "Usure thermique", "quotes": []}, "Usure thermique")):
        out = diagnose_merged_cluster(_cited_ctx(mild_share=0.1), llm=make_llm(main=main, retry=retry))
        assert out["common_cause_source"] == "inferee" and out["common_cause_quotes"] == []
        assert out["common_cause"] == cause
        assert [(r["component"], r["confirm_cause_first"]) for r in out["recommendations"]] == [
            ("joint", True), ("pompe", False)]


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

    def nominal(**kw):
        assert REPORT in kw["user_prompt"]
        return {"protection_only": protection_only, "defect_found": defect_found, "external_condition": condition,
                "external_condition_quote": quote, "reason": "protection déclenchée comme prévu"}
    return make_llm(main=_cited([rec()]), nominal=nominal, seen=seen)


def test_nominal_needs_quoted_external_condition_and_mild_severity(no_llm):
    seen = []
    out = diagnose_merged_cluster(_cited_ctx(mild_share=0.9), llm=_nominal_llm(seen=seen))
    assert out["nominal"] and out["recommendations"] == []
    assert out["nominal_reason"] == "protection déclenchée comme prévu"
    assert out["nominal_external_condition"] == "vent fort"
    assert out["nominal_external_quote"] == "Arrêt de sécurité de la pompe"
    assert not any(k["cache_tag"] == "action_retry" for k in seen)  # pas de contrôle si nominal
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
    rec_ = _run([rec("joint", "equipment", ["EQ-2"])], escalations=esc)["recommendations"][0]
    assert rec_["target_reason"] == {"EQ-2": "both"} and rec_["targets_to_verify"] == []
