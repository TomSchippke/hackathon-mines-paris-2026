"""Étape 2 : agent d'analyse, en phases.

Le CODE fixe l'ordre des phases ; dans chaque phase, l'AGENT LLM (Azure OpenAI, modèle du .env)
prend les décisions avec 2 ou 3 outils bornés (voir tools.py) :

    0. code     extraction de la cause et de la conséquence de chaque incident (LLM, en cache)
    1. agent    pour chaque type (solar, wind) et chaque dimension (cause, consequence) :
                  a. clustering : essaie des réglages (run_clustering), en choisit un (select_clustering)
                  b. thèmes     : lit les clusters (topic_details), les nomme / regroupe (define_themes)
    2. agent    rapport : liens cause -> conséquence, statistiques par site, profils, plan d'actions
                (liens_cause_consequence, site_evidence, write_report)

Si l'agent n'atteint pas l'objectif d'une phase, le code la termine de façon simple (meilleur
clustering conforme selon les scores ; thème nommé par les mots-clés) et le signale dans le journal
et dans le rapport. La suite peut donc toujours tourner.

Sorties dans outputs/2_analyse/ :
    rapport_analyse.md       le rapport final (texte de l'agent + tableaux calculés)
    journal.md / journal.json   déroulé complet, phase par phase (relu par l'agent vérificateur)
    decisions.json           clusterings choisis, contenu du rapport, scores des essais
    clustering_essais.csv    scores de chaque essai de clustering
    themes.csv               thèmes de causes et de conséquences : nom, clusters, taille, justification
    incidents_analyse.csv    chaque incident : cause, conséquence, leurs clusters et thèmes, action

Usage :
    python -m analysis_agent
"""

from __future__ import annotations

import argparse
import copy
import json

import pandas as pd

from llm.agent import dispatch, journal_markdown, run_agent
from paths import ANALYSIS_DIR, ANALYSIS_JOURNAL, ANALYSIS_REPORT, INCIDENTS_FULL

from .config import (ASSET_LABELS, ASSET_TYPES, DIMENSION_LABELS, DIMENSIONS, PHASE_CLUSTERING, PHASE_REPORT,
                     PHASE_THEMES, ROLE, SITE_PROFILES, TURNS)
from .report import export, render
from .tools import TOOLS, AnalysisWorkspace

SINGULAR = {"cause": "cause", "consequence": "conséquence"}


def phase_tools(ws: AnalysisWorkspace, names: list[str], fixed: dict):
    """Outils d'une phase : seulement `names`, avec les arguments `fixed` (type, dimension) imposés
    par le code et retirés du schéma vu par l'agent (il ne peut pas se tromper de cible)."""
    schemas = []
    for t in TOOLS:
        if t["function"]["name"] in names:
            t = copy.deepcopy(t)
            params = t["function"]["parameters"]
            for arg in fixed:
                params["properties"].pop(arg, None)
                params["required"] = [r for r in params["required"] if r != arg]
            schemas.append(t)
    return schemas, lambda name, args: dispatch(ws.tools, name, {**args, **fixed})


def run_phase(ws, journal: list, title: str, task: str, names: list[str], fixed: dict, done, turns: int) -> bool:
    journal.append({"tour": 0, "type": "phase", "texte": title})
    tools, execute = phase_tools(ws, names, fixed)
    events, ok = run_agent(ROLE, task, tools, execute, is_done=done, max_turns=turns)
    journal.extend(events)
    print(f"  {'✅' if ok else '⚠️ '} {title}")
    return ok


def fallback_clustering(ws: AnalysisWorkspace, asset_type: str, dimension: str) -> str:
    """L'agent n'a pas choisi : le code prend l'essai conforme au meilleur DBCV (en faisant au besoin
    quelques essais de plus, dans le budget)."""
    k = ws.key(asset_type, dimension)
    for size in (10, 15, 20, 30, 8, 40):
        if any(r["key"] == k and r["summary"]["conforme_aux_seuils"] for r in ws.runs.values()):
            break
        dispatch(ws.tools, "run_clustering", {"asset_type": asset_type, "dimension": dimension, "min_cluster_size": size})
    runs = [r["summary"] for r in ws.runs.values() if r["key"] == k]
    conform = [s for s in runs if s["conforme_aux_seuils"]]
    best = max(conform or runs, key=lambda s: (s["dbcv"], s["stabilite"]))
    ws.selected[k] = {"run_id": best["run_id"], "justification":
                      "CHOIX AUTOMATIQUE par le code (l'agent n'a pas choisi) : meilleur DBCV parmi les essais "
                      + ("conformes." if conform else "faits ; AUCUN essai n'est conforme aux seuils.")}
    return best["run_id"]


def fallback_themes(ws: AnalysisWorkspace, asset_type: str, dimension: str) -> list[int]:
    """Clusters que l'agent n'a pas rangés : un thème chacun, nommé par ses mots-clés."""
    run = ws.chosen(asset_type, dimension)
    missing = ws.unthemed(asset_type, dimension)
    for t in missing:
        dispatch(ws.tools, "define_themes", {"asset_type": asset_type, "dimension": dimension, "themes": [{
            "name": ", ".join(run["keywords"][t][:3])[:60], "topics": [t],
            "description": "Thème automatique (non nommé par l'agent) : mots-clés du cluster.",
            "justification": "THÈME AUTOMATIQUE par le code : l'agent n'a pas rangé ce cluster."}]})
    return missing


def run(max_turns: int | None = None) -> bool:
    """Lance le workflow. Renvoie True si le rapport a été produit. `max_turns` remplace, s'il est
    donné, le nombre de tours maximum de chaque phase."""
    if not INCIDENTS_FULL.exists():
        raise SystemExit(f"{INCIDENTS_FULL} introuvable : lancer d'abord l'étape pdf.")
    df = pd.read_csv(INCIDENTS_FULL)
    df = df[df["text"].notna()].reset_index(drop=True)
    ws = AnalysisWorkspace(df)
    turns = {phase: max_turns or n for phase, n in TURNS.items()}
    journal: list[dict] = []
    print(f"Workflow d'analyse : {len(df)} incidents")

    # 0. Extraction (code) : pas de décision à prendre, donc pas d'agent.
    for asset_type in ASSET_TYPES:
        result = dispatch(ws.tools, "extract_causes", {"asset_type": asset_type})
        journal.append({"tour": 0, "type": "outil", "outil": "extract_causes (code)",
                        "arguments": {"asset_type": asset_type}, "resultat": result})

    # 1. Clustering puis thèmes, pour chaque type et chaque dimension (agent, une phase à la fois).
    for asset_type in ASSET_TYPES:
        for dimension in DIMENSIONS:
            k = ws.key(asset_type, dimension)
            fixed = {"asset_type": asset_type, "dimension": dimension}
            label, asset = DIMENSION_LABELS[dimension], ASSET_LABELS[asset_type] + "s"
            if not run_phase(ws, journal, f"Clustering {k}",
                             PHASE_CLUSTERING.format(label=label, asset=asset),
                             ["run_clustering", "select_clustering"], fixed, lambda: k in ws.selected,
                             turns["clustering"]):
                run_id = fallback_clustering(ws, asset_type, dimension)
                journal.append({"tour": 0, "type": "automatique", "texte": f"{k} : clustering {run_id} choisi par le code"})

            clusters = [{"cluster": c["cluster"], "incidents": c["incidents"], "mots_cles": c["mots_cles"][:4]}
                        for c in ws.chosen(asset_type, dimension)["summary"]["clusters"]]
            if not run_phase(ws, journal, f"Thèmes {k}",
                             PHASE_THEMES.format(label=label, asset=asset, singular=SINGULAR[dimension],
                                                 clusters=json.dumps(clusters, ensure_ascii=False)),
                             ["topic_details", "define_themes", "remove_theme"], fixed,
                             lambda: not ws.unthemed(asset_type, dimension), turns["themes"]):
                missing = fallback_themes(ws, asset_type, dimension)
                journal.append({"tour": 0, "type": "automatique",
                                "texte": f"{k} : {len(missing)} clusters rangés automatiquement {missing}"})

    # 2. Rapport (agent).
    profiles = json.dumps(SITE_PROFILES, ensure_ascii=False, indent=1)
    sites = {a: ws.sites[a] for a in ASSET_TYPES}
    finished = run_phase(ws, journal, "Rapport",
                         PHASE_REPORT.format(profiles=profiles) + f"\nSites : {json.dumps(sites, ensure_ascii=False)}",
                         ["liens_cause_consequence", "site_evidence", "write_report"], {},
                         lambda: ws.report is not None, turns["report"])

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_JOURNAL.write_text(json.dumps(journal, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (ANALYSIS_DIR / "journal.md").write_text(journal_markdown(journal, "Journal de l'agent d'analyse"), encoding="utf-8")
    export(ws, ANALYSIS_DIR)
    if ANALYSIS_REPORT.exists():
        ANALYSIS_REPORT.unlink()  # pas d'ancien rapport qui ferait croire que celui-ci a réussi
    if ws.report is not None:
        ANALYSIS_REPORT.write_text(render(ws), encoding="utf-8")

    calls = [e for e in journal if e["type"] == "outil"]
    refused = [e for e in calls if not e["resultat"].get("ok")]
    auto = [e for e in journal if e["type"] == "automatique"]
    print(f"{len(calls)} appels d'outils, dont {len(refused)} refusés ; {len(ws.runs)} essais de clustering ; "
          f"{len(ws.themes)} thèmes ; {len(auto)} reprises automatiques par le code")
    print(f"Sauvegardé dans {ANALYSIS_DIR}/")
    print(f"Rapport : {ANALYSIS_REPORT}" if finished else
          "⚠️  L'agent n'a pas rédigé le rapport : voir journal.md (phase « Rapport »).")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-turns", type=int, default=None, help="tours maximum par phase (défaut : config.TURNS)")
    run(parser.parse_args().max_turns)

