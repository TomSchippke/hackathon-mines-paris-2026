"""Tests du rapport opérationnel HTML : petites sorties factices écrites dans tmp_path, aucun appel réseau."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from agentic_pipeline.config import OUTLIER_LABEL
from report.build import ALT_NAME, build_report

SECTIONS = ["Synthèse", "Vue flotte", "Plan d'action", "Sans action requise", "Méthode et limites"]
TEXT = {1: "Le roulement principal présente une usure avancée du chemin de roulement.",
        2: "Le roulement principal a surchauffé, usure avancée du chemin de roulement constatée.",
        3: "Usure avancée du chemin de roulement sur le roulement principal, remplacement.",
        4: "Le câble d'alimentation présente une gaine fissurée sous l'effet des UV.",
        5: "Gaine fissurée du câble d'alimentation, isolement dégradé.",
        6: "La protection s'est déclenchée suite à une rafale de vent, reprise normale ensuite.",
        7: "Après une rafale de vent la protection s'est déclenchée sans dommage.",
        8: "Incident singulier sans rapport avec les autres groupes."}


def _rec(action, scope="fleet", targets=(), reason=None, verify=(), verdict="keep", confirm=False):
    return {"action": action, "component": "composant", "scope": scope, "targets": list(targets),
            "target_reason": reason or {}, "targets_to_verify": list(verify), "rationale": "",
            "confirm_cause_first": confirm, "critic_verdict": verdict}


def _merged_row(mid, rank, tier, freq, cost, title, fine_ids, recs, nominal=False, quotes=(), source="citee", **kw):
    row = {"merged_cluster_id": mid, "partition": "wind", "fine_cluster_ids": json.dumps(fine_ids),
           "criticality_rank": rank, "criticality_tier": tier, "frequency": freq, "mean_cost_eur": cost / freq,
           "total_cost_eur": cost, "mean_downtime_hours": 1.0, "total_downtime_hours": 2.0 * freq,
           "problem_title": title, "problem_description": f"Description {title}.", "common_cause": f"Cause {title}.",
           "common_cause_quotes": json.dumps(list(quotes), ensure_ascii=False), "common_cause_source": source,
           "system_impact": "", "prevented_outcomes": json.dumps(["Panne évitée"]),
           "recommendations": json.dumps(recs, ensure_ascii=False), "early_warning_rules": "[]",
           "nominal": nominal, "nominal_proposed": nominal, "nominal_reason": "", "nominal_external_condition": "",
           "nominal_external_quote": ""}
    row.update(kw)
    return row


@pytest.fixture
def fixture_dirs(tmp_path: Path) -> dict:
    """Arborescence minimale : outputs/segmentation, outputs/equipment ; PDF pour les incidents impairs."""
    seg, eq = tmp_path / "outputs" / "segmentation", tmp_path / "outputs" / "equipment"
    seg.mkdir(parents=True)
    eq.mkdir(parents=True)
    labels = {1: ("wind_F0", "wind_M0"), 2: ("wind_F0", "wind_M0"), 3: ("wind_F0", "wind_M0"),
              4: ("wind_F1", "wind_M1"), 5: ("wind_F1", "wind_M1"),
              6: ("wind_F2", "wind_M2"), 7: ("wind_F2", "wind_M2"), 8: ("wind_O0", OUTLIER_LABEL)}
    sev = {1: "mineure", 2: "modérée", 3: "sévère", 4: "modérée", 5: "sévère", 6: "mineure", 7: "mineure", 8: "sévère"}
    rows = []
    for i in range(1, 9):
        rows.append({"incident_id": i, "fine_cluster_id": labels[i][0], "merged_cluster_id": labels[i][1],
                     "site": "Parc A" if i < 5 else "Parc B", "asset_type": "wind", "equipment_id": "WTG01" if i < 4 else "WTG02",
                     "date": f"2024-0{i}-15", "severity": sev[i], "downtime_hours": 2.0,
                     "estimated_cost_eur": 1000.0 * i, "text": TEXT[i],
                     "document_path": f"data/pdf_scan/incident_{i}.pdf" if i % 2 else ""})
    pd.DataFrame(rows).to_csv(seg / "incidents_segmented.csv", index=False)
    merged = [
        _merged_row("wind_M0", 1, "Critique", 3, 6000.0, "Usure du roulement principal", ["wind_F0"],
                    [_rec("Remplacer le roulement principal de WTG01", "equipment", ["WTG01"], {"WTG01": "both"}),
                     _rec("Graisser les roulements de la flotte")],
                    quotes=["usure avancée du chemin de roulement"]),
        _merged_row("wind_M1", 2, "Élevée", 2, 9000.0, "Gaine de câble fissurée", ["wind_F1"],
                    [_rec("Remplacer les câbles du Parc A", "site", ["Parc A"], {"Parc A": "concentration"},
                          verify=["Parc A"], verdict="non_verifie", confirm=True)], source="inferee"),
        _merged_row("wind_M2", 3, "Moyenne", 2, 13000.0, "Protection sur rafale", ["wind_F2"], [], nominal=True,
                    nominal_reason="Protection seule.", nominal_external_condition="rafale de vent",
                    nominal_external_quote="rafale de vent la protection"),
        _merged_row(OUTLIER_LABEL, 4, "Faible", 1, 8000.0, "Incidents isolés et anomalies atypiques", [],
                    [_rec("Maintien d'une veille opérationnelle.", verdict="fallback")]),
    ]
    pd.DataFrame(merged).to_csv(seg / "clusters_merged.csv", index=False)
    pd.DataFrame([
        {"fine_cluster_id": "wind_F0", "partition": "wind", "merged_cluster_id": "wind_M0", "size": 3,
         "fine_title": "Usure roulement", "failure_mode": "Usure du chemin de roulement", "keywords": "roulement"},
        {"fine_cluster_id": "wind_F1", "partition": "wind", "merged_cluster_id": "wind_M1", "size": 2,
         "fine_title": "Gaine fissurée", "failure_mode": "Fissuration UV", "keywords": "câble"},
        {"fine_cluster_id": "wind_F2", "partition": "wind", "merged_cluster_id": "wind_M2", "size": 2,
         "fine_title": "Rafale", "failure_mode": "Déclenchement protection", "keywords": "rafale"},
        {"fine_cluster_id": "wind_O0", "partition": "wind", "merged_cluster_id": OUTLIER_LABEL, "size": 1,
         "fine_title": "Singulier", "failure_mode": "", "keywords": ""},
    ]).to_csv(seg / "clusters_fine.csv", index=False)
    pd.DataFrame([
        {"merged_cluster_id": "wind_M0", "dimension": "equipment_id", "value": "WTG01", "observed": 3, "expected": 1.2,
         "lift": 2.5, "p_value": 0.01, "q_value": 0.02, "significant": True, "evidence_level": "forte"},
        {"merged_cluster_id": "wind_M1", "dimension": "site", "value": "Parc A", "observed": 3, "expected": 1.5,
         "lift": 2.0, "p_value": 0.04, "q_value": 0.08, "significant": True, "evidence_level": "à vérifier"},
        {"merged_cluster_id": "wind_M1", "dimension": "site", "value": "Parc B", "observed": 1, "expected": 1.0,
         "lift": 1.0, "p_value": 0.5, "q_value": 0.6, "significant": False, "evidence_level": ""},
    ]).to_csv(seg / "concentration.csv", index=False)
    pd.DataFrame([{"equipment_id": "WTG01", "merged_cluster_id": "wind_M0", "partition": "wind", "n_incidents": 3,
                   "severity_sequence": "mineure;modérée;sévère", "incident_ids": "1;2;3",
                   "dates": "2024-01-15;2024-02-15;2024-03-15", "costs": "1000.00;2000.00;3000.00",
                   "total_cost_eur": 6000.0, "first_date": "2024-01-15", "last_date": "2024-03-15"}]
                 ).to_csv(seg / "escalations.csv", index=False)
    pd.DataFrame([{"kind": "precursor", "partition": "wind", "antecedent": "wind_M1", "consequent": "wind_M0",
                   "window_days": 30, "n_hits": 5, "n_equipments": 3, "n_consequent": 3, "expected_hits": 4.0,
                   "lift": 1.2, "p_value": 0.3, "q_value": 0.6, "is_best_window": True, "significant": False,
                   "plausible": False, "mechanism": "", "monitoring_rule": ""}]).to_csv(seg / "precursor_pairs.csv", index=False)
    pd.DataFrame([{"equipment_id": "WTG01", "site": "Parc A", "asset_type": "wind", "total_incidents": 3,
                   "total_cost_eur": 6000.0, "total_downtime_hours": 6.0, "mtbf_days": 30.0, "bad_actor_score": 90.0,
                   "most_frequent_problem": "Usure du roulement principal"}]).to_csv(eq / "equipment_summary.csv", index=False)
    # Seul le PDF de l'incident 1 est présent sur le disque ; ceux des autres incidents impairs manquent
    (tmp_path / "data" / "pdf_scan").mkdir(parents=True)
    (tmp_path / "data" / "pdf_scan" / "incident_1.pdf").write_bytes(b"%PDF-1.4\n")
    return {"root": tmp_path, "seg": seg, "eq": eq, "out": tmp_path / "outputs" / "rapport_operationnel.html"}


def _build(d: dict, **kw) -> str:
    path = build_report(d["seg"], d["eq"], d["out"], root=d["root"], **kw)
    return path.read_text(encoding="utf-8")


def _section(html: str, anchor: str) -> str:
    m = re.search(rf'<section id="{anchor}">(.*?)</section>', html, re.S)
    assert m, anchor
    return m.group(1)


def test_sections_and_no_external_resources(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    assert 'id="entete"' in html
    for title in SECTIONS:
        assert f"<h2>{title}</h2>" in html.replace("&#39;", "'")
    assert "http" not in html and "<script" not in html
    assert (fixture_dirs["out"].parent / ALT_NAME).exists()


def test_template_summary_refs_are_anchors(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    synth = _section(html, "synthese")
    refs = set(re.findall(r'href="#(A\d+)"', synth))
    assert refs, "la synthèse cite au moins une action"
    for ref in refs:
        assert f'id="{ref}"' in html
    for href in re.findall(r'href="#([^"]+)"', synth):
        assert f'id="{href}"' in html
    # l'escalade de WTG01 est citée et ordonnée en premier dans sa voie
    assert "WTG01" in synth and "agir avant la prochaine occurrence" in synth


def test_incident_links_relative_and_inline_text(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    # incident 1 : PDF, lien relatif depuis outputs/
    assert 'href="../data/pdf_scan/incident_1.pdf"' in html
    assert not re.search(r'href="/[^"]*\.pdf"', html)
    # incident 2 : sans PDF, texte du CSV en ligne ; incident 3 : PDF déclaré mais absent du disque, pas de lien mort
    assert "incident_2.pdf" not in html
    assert "incident_3.pdf" not in html
    assert "<summary" in html and "a surchauffé, usure avancée" in html


def test_badges_and_grounding(fixture_dirs, no_llm):
    plan = _section(_build(fixture_dirs), "plan")
    for label in ("escalade", "concentration", "à vérifier", "cause inférée – à confirmer", "non vérifié par le critique"):
        assert label in plan
    assert "lift 2,0" in plan and "q&nbsp;=&nbsp;0,080" in plan.replace(" ", "&nbsp;")
    assert "<svg" in plan  # frise de la série en escalade
    assert "Usure du chemin de roulement" in plan  # mode de défaillance couvert


def test_nominal_and_isolated_not_in_plan(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    plan, none = _section(html, "plan"), _section(html, "sans-action")
    assert "Protection sur rafale" not in plan and "Protection sur rafale" in none
    assert "rafale de vent" in none
    assert "Incidents isolés et anomalies atypiques" not in plan
    assert "Maintien d" not in plan  # action de repli des isolés absente du plan
    assert "#8" in none
    assert re.findall(r'id="(A\d+)"', plan) == ["A1", "A2", "A3"]


def test_llm_summary_filters_invented_ids(fixture_dirs, no_llm):
    calls = []

    def fake_llm(system_prompt, user_prompt, cache_tag):
        calls.append(cache_tag)
        assert "A1" in user_prompt and "A3" in user_prompt
        return {"sentences": [
            {"text": "Remplacer le roulement de WTG01.", "action_ids": ["A1"]},
            {"text": "Graisser la flotte (actions : [A2]).", "action_ids": ["A2"]},
            {"text": "Remplacer les câbles du Parc A.", "action_ids": ["A3", "A1"]},
            {"text": "Action inventée.", "action_ids": ["A99"]},
            {"text": "Phrase sans action.", "action_ids": []},
        ]}

    synth = _section(_build(fixture_dirs, summary="llm", llm=fake_llm), "synthese")
    assert calls == ["report_summary"]
    assert "Remplacer le roulement de WTG01." in synth
    assert "Graisser la flotte." in synth and "[A2]" not in synth  # identifiants recopiés retirés du texte
    assert "Action inventée" not in synth and "A99" not in synth and "Phrase sans action" not in synth
    alt = (fixture_dirs["out"].parent / ALT_NAME).read_text(encoding="utf-8")
    assert "déterministe" in alt and "rapport_operationnel.html#" in alt


def test_llm_summary_falls_back_below_three_sentences(fixture_dirs, no_llm):
    def fake_llm(system_prompt, user_prompt, cache_tag):
        return {"sentences": [{"text": "Une seule phrase valide.", "action_ids": ["A1"]},
                              {"text": "Invention.", "action_ids": ["A42"]}]}

    synth = _section(_build(fixture_dirs, summary="llm", llm=fake_llm), "synthese")
    assert "Une seule phrase valide" not in synth
    assert "repli sur le gabarit" in synth and "agir avant la prochaine occurrence" in synth
