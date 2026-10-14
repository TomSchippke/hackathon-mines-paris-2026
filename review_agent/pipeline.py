"""Étape 3 : agent vérificateur.

Relit le travail de l'agent d'analyse (outputs/2_analyse/) et juge s'il est fiable :
  1. contrôles automatiques par le code (clustering conforme, thèmes complets, causes extraites,
     profils attribués une fois chacun, plan de remédiation pour chaque site) ;
  2. un agent LLM lit le rapport et le journal, compare les affirmations du rapport aux chiffres
     recalculés, consigne chaque problème avec sa preuve, et rend un verdict.
Le vérificateur est en lecture seule : il ne modifie rien de l'analyse.

Sorties dans outputs/3_verification/ :
    verification.md        contrôles, constats, verdict (à lire)
    verification.json      la même chose, structurée
    journal.md             déroulé de la session du vérificateur

Usage :
    python -m review_agent
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from analysis_agent.report import md_table
from analysis_agent.facts import profile_indicators, rank_text
from llm.agent import dispatch, journal_markdown, run_agent
from paths import ANALYSIS_DIR, REVIEW_DIR

from .config import MAX_TURNS, SYSTEM_PROMPT
from .tools import TOOLS, ReviewWorkspace


def render(ws: ReviewWorkspace) -> str:
    verdict = ws.verdict or {"verdict": "non rendu", "synthese": "Le vérificateur n'a pas conclu (voir journal.md)."}
    lines = ["# Vérification de l'analyse\n", f"**Verdict : {verdict['verdict']}**\n", verdict["synthese"],
             "\n## Contrôles automatiques\n",
             md_table(pd.DataFrame(ws.checks).assign(ok=lambda d: d["ok"].map({True: "✅", False: "❌"})))]
    problems = [f for f in ws.findings if f["gravite"] != "conforme"]
    checked = [f for f in ws.findings if f["gravite"] == "conforme"]
    lines.append("\n## Problèmes relevés\n")
    if problems:
        order = {"bloquant": 0, "majeur": 1, "mineur": 2}
        lines.append(md_table(pd.DataFrame(problems).sort_values("gravite", key=lambda g: g.map(order))
                              [["gravite", "source", "cible_site", "cible_champ", "constat", "preuve", "consigne"]]))
    else:
        lines.append("Aucun problème relevé.")
    lines.append("\n## Points vérifiés et conformes\n")
    lines.append(md_table(pd.DataFrame(checked)[["cible_site", "cible_champ", "constat", "preuve"]]) if checked else "Aucun.")
    return "\n".join(lines) + "\n"


def run(max_turns: int = MAX_TURNS) -> str:
    """Lance la vérification. Renvoie le verdict."""
    ws = ReviewWorkspace(ANALYSIS_DIR)
    failed = [c for c in ws.checks if not c["ok"]]
    print(f"Contrôles automatiques : {len(ws.checks) - len(failed)}/{len(ws.checks)} réussis")
    for c in failed:
        print(f"  ❌ {c['controle']} : {c['detail']}")

    if ws.report_md is None:  # rien à relire : inutile de solliciter le LLM
        ws.verdict = {"verdict": "a_refaire", "synthese": "VERDICT AUTOMATIQUE : l'agent d'analyse n'a pas produit de "
                                                          "rapport (voir outputs/2_analyse/journal.md)."}
        journal = [{"tour": 0, "type": "automatique", "texte": "pas de rapport : verdict a_refaire sans agent"}]
    else:
        # Le rapport, les contrôles et les chiffres de chaque site sont donnés d'emblée : moins d'étapes
        # pour l'agent, qui se concentre sur la comparaison et le jugement.
        ws.report_read = True
        facts = {site: ws.site_facts(site) for site in sorted(ws.incidents["site"].unique())}
        indicators = {a: rank_text(profile_indicators(ws.incidents[ws.incidents["asset_type"] == a]))
                      for a in sorted(ws.incidents["asset_type"].unique())}
        task = ("CONTRÔLES AUTOMATIQUES (faits par le code) :\n" + json.dumps(ws.checks, ensure_ascii=False)
                + "\n\nRAPPORT À VÉRIFIER :\n" + ws.report_md
                + "\n\nCHIFFRES RECALCULÉS PAR SITE (à comparer au rapport) :\n"
                + json.dumps(facts, ensure_ascii=False, default=str)
                + "\n\nINDICATEURS DES PROFILS (critères du sujet mesurés sur chaque site, rang parmi les 3 "
                  "sites du même type, 1 = le plus élevé) :\n" + json.dumps(indicators, ensure_ascii=False)
                + f"\n\nLe journal de l'agent d'analyse contient {len(ws.journal)} événements (read_journal). "
                  "Vérifie, consigne chaque problème avec record_finding, puis appelle conclude.")
        tools = [t for t in TOOLS if t["function"]["name"] != "read_report"]
        journal, _ = run_agent(SYSTEM_PROMPT, task, tools, lambda name, args: dispatch(ws.tools, name, args),
                               is_done=lambda: ws.verdict is not None, max_turns=max_turns, role="review",
                               nudge=lambda: (f"Consigne encore un point pour : {ws.uncovered()}, puis appelle conclude."
                                              if ws.uncovered() else "Appelle conclude."))
        if ws.verdict is None:
            ws.verdict = ws.rule_verdict()
            journal.append({"tour": 0, "type": "automatique", "texte": ws.verdict["synthese"]})

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "journal.md").write_text(journal_markdown(journal, "Journal du vérificateur"), encoding="utf-8")
    (REVIEW_DIR / "verification.md").write_text(render(ws), encoding="utf-8")
    (REVIEW_DIR / "verification.json").write_text(
        json.dumps({"verdict": ws.verdict, "controles": ws.checks, "constats": ws.findings}, ensure_ascii=False,
                   indent=2), encoding="utf-8")

    by_level = pd.Series([f["gravite"] for f in ws.findings]).value_counts().to_dict()
    print(f"Constats : {by_level or 'aucun'}")
    print(f"Verdict : {ws.verdict['verdict']}  ->  {REVIEW_DIR / 'verification.md'}")
    return ws.verdict["verdict"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    run(parser.parse_args().max_turns)
