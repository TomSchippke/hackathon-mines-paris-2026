"""Étape 2 : agent d'analyse, en phases.

Le CODE fixe l'ordre des phases ; dans chaque phase, l'AGENT LLM (Azure OpenAI, modèle du .env)
prend les décisions avec 2 ou 3 outils bornés (voir tools.py) :

    0. code     extraction de la cause et de la conséquence de chaque incident (LLM, en cache)
    1. agent    pour chaque type (solar, wind) et chaque dimension (cause, consequence) :
                  a. clustering : essaie des réglages (run_clustering), en choisit un (select_clustering)
                  b. thèmes     : lit les clusters (topic_details), les nomme / regroupe (define_themes)
    2. agent    profils : lit les points distinctifs de chaque site (site_evidence) et associe chaque
                site à un profil A, B, C (assign_profiles)
    3. code+LLM rédaction : les faits sont calculés en Python (facts.py) ; le LLM ne fait que les
                reformuler et un validateur vérifie chaque chiffre et chaque phénomène cité (writing.py)

Si l'agent n'atteint pas l'objectif d'une phase, le code la termine de façon simple (meilleur
clustering conforme selon les scores ; thème nommé par sa catégorie ou son facteur dominant ; profils
selon les règles d'attentes) et le signale dans le journal et dans le rapport.

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

from . import facts as F
from .config import (MIN_CHOSEN_DBCV, ASSET_LABELS, ASSET_TYPES, DIMENSION_LABELS, DIMENSIONS, PHASE_CLUSTERING, PHASE_PROFILES,
                     PHASE_THEMES, ROLE, SITE_PROFILES, TURNS)
from .report import export, render
from .tools import TOOLS, AnalysisWorkspace
from .validation_inferences import inference_sample
from .judge import judge_report
from .writing import site_texts, synthesis

SINGULAR = {"cause": "cause", "consequence": "conséquence"}


def phase_tools(ws: AnalysisWorkspace, names: list[str], fixed: dict):
    """Outils d'une phase : seulement `names`, avec les arguments `fixed` (type, dimension) imposés
    par le code et retirés du schéma vu par l'agent (il ne peut pas se tromper de cible)."""
    schemas, accepted = [], {}
    for t in TOOLS:
        if t["function"]["name"] in names:
            t = copy.deepcopy(t)
            params = t["function"]["parameters"]
            accepted[t["function"]["name"]] = [a for a in fixed if a in params["properties"]]
            for arg in fixed:
                params["properties"].pop(arg, None)
                params["required"] = [r for r in params["required"] if r != arg]
            schemas.append(t)
    # Un outil ne reçoit que les arguments imposés qu'il accepte (remove_theme n'a pas de dimension).
    return schemas, lambda name, args: dispatch(ws.tools, name, {**args, **{a: fixed[a] for a in accepted.get(name, [])}})


def run_phase(ws, journal: list, title: str, task: str, names: list[str], fixed: dict, done, turns: int,
              nudge=None) -> bool:
    journal.append({"tour": 0, "type": "phase", "texte": title})
    tools, execute = phase_tools(ws, names, fixed)
    events, ok = run_agent(ROLE, task, tools, execute, is_done=done, max_turns=turns, nudge=nudge)
    journal.extend(events)
    print(f"  {'OK' if ok else 'Not OK'} => {title}")
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
    good = [s for s in conform if s["dbcv"] >= MIN_CHOSEN_DBCV]
    # Le plus détaillé parmi les essais conformes de séparation au moins moyenne ; sinon le meilleur DBCV.
    best = max(good, key=lambda s: s["nombre_clusters"]) if good else max(conform or runs, key=lambda s: s["dbcv"])
    ws.selected[k] = {"run_id": best["run_id"], "justification":
                      "CHOIX AUTOMATIQUE par le code (l'agent n'a pas choisi) : "
                      + ("le plus détaillé des essais conformes de séparation au moins moyenne." if good else
                         "meilleur DBCV parmi les essais " + ("conformes." if conform else "faits ; AUCUN n'est conforme."))}
    return best["run_id"]


def fallback_themes(ws: AnalysisWorkspace, asset_type: str, dimension: str) -> list[int]:
    """Clusters que l'agent n'a pas rangés : un thème chacun, nommé par la catégorie de conséquence
    (ou le facteur de cause) dominante de ses incidents, extraite par le LLM dans une liste fermée."""
    run = ws.chosen(asset_type, dimension)
    column = "categorie_consequence" if dimension == "consequence" else "facteur_principal"
    missing = ws.unthemed(asset_type, dimension)
    for t in missing:
        values = ws.causes.loc[run["positions"][run["labels"] == t], column].value_counts()
        name = f"{values.index[0]} ({', '.join(run['keywords'][t][:2])})" if len(values) else ", ".join(run["keywords"][t][:3])
        dispatch(ws.tools, "define_themes", {"asset_type": asset_type, "dimension": dimension, "themes": [{
            "name": name[:60], "topics": [t],
            "description": f"Thème automatique : {column.replace('_', ' ')} dominant(e) du cluster et mots-clés.",
            "justification": "THÈME AUTOMATIQUE par le code : l'agent n'a pas rangé ce cluster."}]})
    return missing


def fallback_merge(ws: AnalysisWorkspace, asset_type: str, dimension: str) -> list:
    """Doublons de thèmes que l'agent n'a pas fusionnés : fusion sous le nom du plus gros thème."""
    merged = []
    while pairs := ws.duplicate_themes(asset_type, dimension):
        a, b, _ = pairs[0]
        name = max((ws.themes[a], ws.themes[b]), key=lambda t: t["incidents"])["name"]
        result = dispatch(ws.tools, "merge_themes", {"asset_type": asset_type, "dimension": dimension,
                                                     "theme_ids": [a, b], "name": name})
        if not result.get("ok"):
            break
        merged.append((a, b))
    return merged


def fallback_profiles(ws: AnalysisWorkspace) -> None:
    """L'agent n'a pas attribué les profils : pour chaque type, l'attribution qui respecte le plus
    d'attentes des profils (config.PROFILE_EXPECTATIONS : « élevé » = site en tête, « faible » = en queue)."""
    table = ws.incidents_table()
    assignments = []
    for asset_type in ASSET_TYPES:
        indicators = F.profile_indicators(table[table["asset_type"] == asset_type])
        best = F.best_assignments(indicators, asset_type)[0]
        assignments += [{"site": site, "profil": profile} for site, profile in best.items()]
    dispatch(ws.tools, "assign_profiles", {"assignments": assignments, "raisonnement":
             "ATTRIBUTION AUTOMATIQUE par le code : celle qui respecte le plus d'attentes des profils."})


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

            column = "categorie_consequence" if dimension == "consequence" else "facteur_principal"
            # Les clusters sont décrits dans la consigne : l'agent n'a qu'à les nommer (pas de lecture en boucle).
            details = ws.topic_details(asset_type, dimension, ws.topic_ids(asset_type, dimension))["clusters"]
            phrases = DIMENSIONS[dimension]
            clusters = "\n".join(
                f"- cluster {d['cluster']} ({d['incidents']} incidents) ; {column.replace('_', ' ')} : "
                f"{next(iter(d['facteurs_pct'] if dimension == 'cause' else d['categories_consequence_pct']), '?')} ; "
                f"mots-clés : {', '.join(d['mots_cles'][:4])} ; exemples : {' / '.join(d[phrases + '_exemples'][:3])}"
                for d in details)
            pairs = ws.similar_cluster_pairs(asset_type, dimension)
            suggestions = ("\n".join(f"- clusters {a} et {b} : similarité {s}, même {column.replace('_', ' ')} dominant(e)"
                                     for a, b, s in pairs) or "- aucune")

            def remaining(a=asset_type, d=dimension) -> str:
                todo = []
                if ws.unthemed(a, d):
                    todo.append(f"clusters sans thème : {ws.unthemed(a, d)} (define_theme)")
                if ws.duplicate_themes(a, d):
                    todo.append(f"thèmes en doublon à fusionner : {ws.duplicate_themes(a, d)} (merge_themes)")
                return "Il reste à faire : " + " ; ".join(todo) + "."

            def done(a=asset_type, d=dimension) -> bool:
                return not ws.unthemed(a, d) and not ws.duplicate_themes(a, d)

            if not run_phase(ws, journal, f"Thèmes {k}",
                             PHASE_THEMES.format(label=label, asset=asset, singular=SINGULAR[dimension],
                                                 column=column.replace("_", " "), clusters=clusters,
                                                 suggestions=suggestions),
                             ["define_theme", "compare_clusters", "merge_themes"], fixed, done,
                             turns["themes"], nudge=remaining):
                missing = fallback_themes(ws, asset_type, dimension)
                merged = fallback_merge(ws, asset_type, dimension)
                journal.append({"tour": 0, "type": "automatique",
                                "texte": f"{k} : {len(missing)} clusters rangés automatiquement {missing} ; "
                                         f"{len(merged)} doublons fusionnés automatiquement {merged}"})

    # 2. Profils (agent).
    profiles = json.dumps(SITE_PROFILES, ensure_ascii=False, indent=1)
    sites = {a: ws.sites[a] for a in ASSET_TYPES}
    if not run_phase(ws, journal, "Profils",
                     PHASE_PROFILES.format(profiles=profiles) + f"\nSites : {json.dumps(sites, ensure_ascii=False)}",
                     ["site_evidence", "assign_profiles"], {}, lambda: ws.profiles is not None, turns["profiles"]):
        fallback_profiles(ws)
        journal.append({"tour": 0, "type": "automatique", "texte": "profils attribués par le code (règles d'attentes)"})

    # 3. Rédaction contrôlée (faits Python, reformulation LLM, validateur), puis relecture par le juge.
    journal.append({"tour": 0, "type": "phase", "texte": "Rédaction (faits calculés, reformulés et validés)"})
    table = ws.incidents_table()
    site_reports = []
    for asset_type in ASSET_TYPES:
        sub = table[table["asset_type"] == asset_type]
        indicators = F.profile_indicators(sub)
        for site in ws.sites[asset_type]:
            site_reports.append(site_texts(site, asset_type, ws.profiles[site]["profil"], sub, indicators, journal))
    fleet = F.fleet_priorities(table)
    synthese, synthese_facts = synthesis(table, site_reports, fleet, journal)
    ws.report = {"sites": site_reports, "synthese": synthese, "faits_synthese": synthese_facts}
    journal.append({"tour": 0, "type": "phase", "texte": "Relecture par le juge (grille, correction validée par le code)"})
    judge_report(ws.report, journal)
    finished = True

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_JOURNAL.write_text(json.dumps(journal, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (ANALYSIS_DIR / "journal.md").write_text(journal_markdown(journal, "Journal de l'agent d'analyse"), encoding="utf-8")
    export(ws, ANALYSIS_DIR)
    if ANALYSIS_REPORT.exists():
        ANALYSIS_REPORT.unlink()  # pas d'ancien rapport qui ferait croire que celui-ci a réussi
    sample = inference_sample(ws.incidents_table())  # créé une fois, jamais écrasé (verdicts humains)
    if ws.report is not None:
        ANALYSIS_REPORT.write_text(render(ws), encoding="utf-8")

    calls = [e for e in journal if e["type"] == "outil"]
    refused = [e for e in calls if not e["resultat"].get("ok")]
    auto = [e for e in journal if e["type"] == "automatique"]
    print(f"{len(calls)} appels d'outils, dont {len(refused)} refusés ; {len(ws.runs)} essais de clustering ; "
          f"{len(ws.themes)} thèmes ; {len(auto)} reprises automatiques par le code")
    print(f"Sauvegardé dans {ANALYSIS_DIR}/ ; validation manuelle des inférences : {sample}")
    print(f"Rapport : {ANALYSIS_REPORT}" if finished else
          "Pas de rapport : voir journal.md.")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-turns", type=int, default=None, help="tours maximum par phase (défaut : config.TURNS)")
    run(parser.parse_args().max_turns)

