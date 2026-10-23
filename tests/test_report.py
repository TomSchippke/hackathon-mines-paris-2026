"""Tests du rapport opérationnel HTML : petites sorties factices écrites dans tmp_path, aucun appel réseau."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from agentic_pipeline.config import OUTLIER_LABEL
from report.build import METHOD_NAME, build_report

SECTIONS = ["Actions ciblées", "Programmes flotte", "Preuves par groupe", "Sans action requise"]
METHOD_SECTIONS = ["Construction", "Vue flotte", "Qualité des actions", "Résultats négatifs", "Limites"]
TEXT = {1: "Le roulement principal présente une usure avancée du chemin de roulement.",
        2: "Le roulement principal a surchauffé, usure avancée du chemin de roulement constatée.",
        3: "Usure avancée du chemin de roulement sur le roulement principal, remplacement.",
        4: "Le câble d'alimentation présente une gaine fissurée sous l'effet des UV.",
        5: "Gaine fissurée du câble d'alimentation, isolement dégradé.",
        6: "La protection s'est déclenchée suite à une rafale de vent, reprise normale ensuite.",
        7: "Après une rafale de vent la protection s'est déclenchée sans dommage.",
        8: "Incident singulier sans rapport avec les autres groupes."}


def _rec(verb, component, detail="", scope="fleet", targets=(), reason=None, verify=(), verdict="valide",
         confirm=False):
    return {"action": f"{verb} {component}", "verb": verb, "component": component, "detail": detail,
            "scope": scope, "targets": list(targets), "target_reason": reason or {},
            "targets_to_verify": list(verify), "rationale": "", "confirm_cause_first": confirm,
            "critic_verdict": verdict}


def _merged_row(mid, rank, tier, freq, cost, title, fine_ids, recs, nominal=False, quotes=(), source="citee", **kw):
    row = {"merged_cluster_id": mid, "partition": "wind", "fine_cluster_ids": json.dumps(fine_ids),
           "criticality_rank": rank, "criticality_tier": tier, "frequency": freq, "mean_cost_eur": cost / freq,
           "total_cost_eur": cost, "mean_downtime_hours": 1.0, "total_downtime_hours": 2.0 * freq,
           "problem_title": title, "problem_description": f"Description {title}.", "common_cause": f"Cause {title}.",
           "common_cause_quotes": json.dumps(list(quotes), ensure_ascii=False), "common_cause_source": source,
           "recommendations": json.dumps(recs, ensure_ascii=False),
           "nominal": nominal, "nominal_proposed": nominal, "nominal_reason": "", "nominal_external_condition": "",
           "nominal_external_quote": ""}
    row.update(kw)
    return row


@pytest.fixture
def fixture_dirs(tmp_path: Path) -> dict:
    """Arborescence minimale : outputs/segmentation ; PDF pour les incidents impairs."""
    seg = tmp_path / "outputs" / "segmentation"
    seg.mkdir(parents=True)
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
                    [_rec("Remplacer", "roulement principal", "chemin de roulement usé", "equipment", ["WTG01"],
                          {"WTG01": "both"}),
                     _rec("Graisser", "roulements")],
                    quotes=["usure avancée du chemin de roulement"]),
        _merged_row("wind_M1", 2, "Élevée", 2, 9000.0, "Gaine de câble fissurée", ["wind_F1"],
                    [_rec("Remplacer", "câble d'alimentation", "", "site", ["Parc A"], {"Parc A": "concentration"},
                          verify=["Parc A"], verdict="corrigee", confirm=True),
                     {**_rec("", ""), "action": "Aucune action spécifique identifiée : inspecter pour établir la cause.",
                      "verdict": "", "critic_verdict": "fallback"}], source="inferee"),
        _merged_row("wind_M2", 3, "Moyenne", 2, 13000.0, "Protection sur rafale", ["wind_F2"], [], nominal=True,
                    nominal_reason="Protection seule.", nominal_external_condition="rafale de vent",
                    nominal_external_quote="rafale de vent la protection"),
        _merged_row(OUTLIER_LABEL, 4, "Faible", 1, 8000.0, "Incidents isolés et anomalies atypiques", [],
                    [{**_rec("", ""), "action": "Maintien d'une veille opérationnelle.", "critic_verdict": "fallback"}]),
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
    # Seul le PDF de l'incident 1 est présent sur le disque ; ceux des autres incidents impairs manquent
    (tmp_path / "data" / "pdf_scan").mkdir(parents=True)
    (tmp_path / "data" / "pdf_scan" / "incident_1.pdf").write_bytes(b"%PDF-1.4\n")
    return {"root": tmp_path, "seg": seg, "out": tmp_path / "outputs" / "rapport_operationnel.html"}


def _build(d: dict, **kw) -> str:
    path = build_report(d["seg"], d["out"], root=d["root"], **kw)
    return path.read_text(encoding="utf-8")


def _section(html: str, anchor: str) -> str:
    m = re.search(rf'<section id="{anchor}">(.*?)</section>', html, re.S)
    assert m, anchor
    return m.group(1)


def _method(d: dict) -> str:
    return (d["out"].parent / METHOD_NAME).read_text(encoding="utf-8")


def test_sections_and_no_external_resources(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    assert 'id="entete"' in html
    for title in SECTIONS:
        assert f"<h2>{title}</h2>" in html.replace("&#39;", "'")
    method = _method(fixture_dirs)
    for title in METHOD_SECTIONS:
        assert f"<h2>{title}</h2>" in method
    for page in (html, method):
        assert "http" not in page and "<script" not in page
    assert not (fixture_dirs["out"].parent / "rapport_synthese_alt.html").exists()
    assert "Synthèse" not in html and "bad actor" not in html


def test_targets_ranked_by_own_cost_with_escalation(fixture_dirs, no_llm):
    targets = _section(_build(fixture_dirs), "ciblees")
    # WTG01 : incidents 1-3 du groupe wind_M0 (6 000 €) ; Parc A : incident 4 du groupe wind_M1 (4 000 €)
    assert targets.index("WTG01") < targets.index("Parc A")
    assert "6 000 €" in targets and "4 000 €" in targets
    assert "9 000" not in targets  # coût du groupe wind_M1, jamais affiché pour la cible
    assert "agir avant la prochaine occurrence" in targets and 'class="escalating"' in targets
    assert "<b>Remplacer</b> roulement principal" in targets and "chemin de roulement usé" in targets
    for label in ("escalade", "concentration", "à vérifier", "cause inférée – à confirmer"):
        assert label in targets
    assert "équipement · Parc A" in targets


def test_fleet_programmes(fixture_dirs, no_llm):
    fleet = _section(_build(fixture_dirs), "flotte")
    assert "<b>Graisser</b> roulements" in fleet and "Sur l'ensemble des éoliennes du parc" in fleet
    assert "action par défaut" in fleet and "inspecter pour établir la cause" in fleet
    assert "Remplacer</b> roulement" not in fleet  # action ciblée absente des programmes


def test_action_ids_reading_order_and_backlinks(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    ids = re.findall(r'id="(A\d+)"', html)
    assert ids == ["A1", "A2", "A3", "A4"]  # ciblées (WTG01, Parc A), puis programmes flotte
    proofs = _section(html, "preuves")
    for ref in set(re.findall(r'href="#(A\d+)"', proofs)):
        assert f'id="{ref}"' in html
    for anchor in re.findall(r'href="#(voie-[^"]+)"', html):
        assert f'id="{anchor}"' in html


def test_action_with_two_targets_anchored_once(fixture_dirs, no_llm):
    path = fixture_dirs["seg"] / "clusters_merged.csv"
    merged = pd.read_csv(path)
    recs = json.loads(merged.loc[0, "recommendations"])
    recs[0]["targets"], recs[0]["target_reason"] = ["WTG01", "WTG02"], {"WTG01": "both", "WTG02": "escalation"}
    merged.loc[0, "recommendations"] = json.dumps(recs, ensure_ascii=False)
    merged.to_csv(path, index=False)
    html = _build(fixture_dirs)
    assert html.count('id="A1"') == 1
    assert _section(html, "ciblees").count('<span class="aid">A1</span>') == 2


def test_incident_links_relative_and_inline_text(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    # incident 1 : PDF, lien relatif depuis outputs/
    assert 'href="../data/pdf_scan/incident_1.pdf"' in html
    assert not re.search(r'href="/[^"]*\.pdf"', html)
    # incident 2 : sans PDF, texte du CSV en ligne ; incident 3 : PDF déclaré mais absent du disque, pas de lien mort
    assert "incident_2.pdf" not in html
    assert "incident_3.pdf" not in html
    assert "<summary" in html and "a surchauffé, usure avancée" in html


def test_evidence_sheets(fixture_dirs, no_llm):
    proofs = _section(_build(fixture_dirs), "preuves")
    assert "lift 2,0" in proofs and "q&nbsp;=&nbsp;0,080" in proofs.replace(" ", "&nbsp;")
    assert "<svg" in proofs  # frise de la série en escalade
    assert "Usure du chemin de roulement" in proofs  # mode de défaillance couvert
    assert "usure avancée du chemin de roulement" in proofs  # citation
    for removed in ("Défaillances évitées", "Panne évitée", "Règles d'alerte", "Impact"):
        assert removed not in proofs


def test_nominal_and_isolated_not_in_tables(fixture_dirs, no_llm):
    html = _build(fixture_dirs)
    tables = _section(html, "ciblees") + _section(html, "flotte") + _section(html, "preuves")
    none = _section(html, "sans-action")
    assert "Protection sur rafale" not in tables and "Protection sur rafale" in none
    assert "rafale de vent" in none
    assert "Incidents isolés et anomalies atypiques" not in tables
    assert "Maintien d" not in tables
    assert "#8" in none


def test_method_page_quality_and_charts(fixture_dirs, no_llm):
    _build(fixture_dirs)
    method = _method(fixture_dirs)
    quality = _section(method, "qualite")
    # 4 actions au plan : 2 valides, 1 corrigée, 1 par défaut
    assert re.search(r"<b class=\"num\">2</b><span>acceptées du premier coup", quality)
    assert re.search(r"<b class=\"num\">1</b><span>acceptées après relance", quality)
    assert "actions par défaut (25" in quality
    assert "1/2</b><span>causes citées" in quality
    assert "<svg" in _section(method, "vue-flotte") and "rapport_operationnel.html#voie-wind_M0" in method
    assert "Sur ce run" not in method


def test_every_escalation_series_gets_a_row(fixture_dirs, no_llm):
    path = fixture_dirs["seg"] / "clusters_merged.csv"
    merged = pd.read_csv(path)
    recs = json.loads(merged.loc[0, "recommendations"])
    merged.loc[0, "recommendations"] = json.dumps(recs[1:], ensure_ascii=False)  # plus d'action sur WTG01
    merged.to_csv(path, index=False)
    html = _build(fixture_dirs)
    targets = _section(html, "ciblees")
    assert "WTG01" in targets and "<b>Inspecter</b> l&#39;équipement" in targets
    assert "ajoutée par le code" in targets and "agir avant la prochaine occurrence" in targets
    assert "<svg" in _section(html, "preuves")  # la frise reste dans la fiche
    quality = _section(_method(fixture_dirs), "qualite")
    assert "1 série en escalade" in quality and "1</b><span>acceptées du premier coup" in quality
