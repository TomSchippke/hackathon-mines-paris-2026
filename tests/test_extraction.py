"""Tests de l'extraction des causes (LLM simulé) : citation vérifiée, cause inconnue, reprise."""

from __future__ import annotations

import pandas as pd

from agentic_pipeline.config import CAUSE_UNKNOWN_TEXT
from agentic_pipeline.extraction import extract_causes, quote_found

REPORT = ("Arrêt de l'onduleur après une surchauffe. L'inspection a montré que le ventilateur de "
          "refroidissement était bloqué par des débris accumulés depuis plusieurs mois.")


def _df(texts):
    return pd.DataFrame({"incident_id": range(1, len(texts) + 1), "text": texts, "site": "S-SECRET",
                         "severity": "sévère"})


def test_quote_found_tolerates_accents_and_punctuation_only():
    assert quote_found("le ventilateur de refroidissement etait bloque par des debris", REPORT)
    assert not quote_found("le ventilateur de refroidissement était cassé par la foudre", REPORT)
    assert not quote_found("ventilateur bloqué", REPORT)  # trop court pour être vérifiable


def test_cause_kept_only_with_verified_quote(no_llm):
    seen = []

    def llm(system_prompt, user_prompt, cache_tag):
        seen.append(user_prompt)
        if "foudre" in user_prompt:  # citation inventée
            return {"cause_racine": "Coup de foudre", "cause_type": "explicite",
                    "indice": "la foudre est tombée sur le poste de livraison", "consequence": "arrêt"}
        if "rien" in user_prompt:
            return {"cause_racine": "Cause non déterminée", "cause_type": "explicite", "indice": "rien"}
        return {"cause_racine": "Ventilateur bloqué par des débris", "cause_type": "explicite",
                "indice": "le ventilateur de refroidissement était bloqué par des débris",
                "consequence": "surchauffe et arrêt de l'onduleur", "composant_brut": "ventilateur"}

    out = extract_causes(_df([REPORT, "Arrêt après un orage, foudre possible.", "rien à signaler"]), llm=llm)
    assert out["cause_known"].tolist() == [True, False, False]
    assert out.at[0, "cause_racine"] == "Ventilateur bloqué par des débris"
    assert (out.loc[1:, "cause_racine"] == CAUSE_UNKNOWN_TEXT).all() and (out.loc[1:, "indice"] == "").all()
    assert out.at[1, "consequence"] == "arrêt"  # la conséquence reste utile même sans cause
    # Le LLM ne voit que la description : ni site, ni sévérité
    assert all("S-SECRET" not in p and "sévère" not in p for p in seen)


def test_llm_failure_gives_unknown_cause(no_llm):
    def boom(**kw):
        raise RuntimeError("quota")

    out = extract_causes(_df([REPORT]), llm=boom)
    assert not out.at[0, "cause_known"] and out.at[0, "cause_racine"] == CAUSE_UNKNOWN_TEXT


def test_seed_reused_only_for_identical_text(no_llm):
    calls = []

    def llm(**kw):
        calls.append(kw["user_prompt"])
        return {}

    seed = pd.DataFrame({
        "incident_id": [1, 2], "text": [REPORT, "ancien texte"],
        "cause_racine": ["Débris dans le ventilateur", "x"], "cause_type": ["explicite", "explicite"],
        "indice": ["bloqué par des débris accumulés depuis plusieurs mois", "ancien texte cité ici"],
        "consequence": ["surchauffe", ""], "composant_brut": [None, None],
    })
    out = extract_causes(_df([REPORT, "nouveau texte"]), llm=llm, seed=seed)
    assert out["extraction_source"].tolist() == ["import", "llm"]
    assert out.at[0, "cause_known"] and out.at[0, "composant_brut"] == ""
    assert len(calls) == 1 and "nouveau texte" in calls[0]


def test_seed_without_extraction_columns_is_ignored(no_llm):
    seed = pd.DataFrame({"incident_id": [1], "text": [REPORT]})  # ancienne sortie, sans cause
    out = extract_causes(_df([REPORT]), llm=lambda **kw: {}, seed=seed)
    assert out.at[0, "extraction_source"] == "llm"
