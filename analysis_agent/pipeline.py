"""Étape 2 : agent d'analyse, en phases.

Le CODE fixe l'ordre des phases ; dans chaque phase, l'AGENT LLM (Azure OpenAI, modèle du .env)
prend les décisions avec 2 ou 3 outils bornés (voir tools.py). Ce que l'agent sait du problème vient
uniquement du cadrage (étape 1b, framing/ : contexte écrit par l'utilisateur + données) :

    0. code     extraction, par incident : cause racine, causes contributives, conséquence (LLM, en cache)
    0b. code    normalisation des composants bruts de chaque segment (LLM, une passe par segment)
    1. pour chaque segment et chaque dimension (cause, consequence) :
                  a. agent  clustering : essaie des réglages (run_clustering), en choisit un (select_clustering)
                  b. agent  thèmes : lit les clusters (topic_details), les nomme / regroupe (define_theme...)
                  c. code+LLM affectation : chaque phrase reçoit un thème ou « aucun » (assign_phrases) ; tant
                     qu'il reste plus de NONE_MAX_PCT % de « aucun » (au plus MAX_RECLUSTER_ROUNDS tours), les
                     phrases « aucun » sont reclustérisées, nommées par l'agent, puis réaffectées
    2. agent    priorités : heatmap thèmes de causes x thèmes de conséquences (heatmap_hotspots)
                et choisit les chaînes cause -> conséquence prioritaires (select_priority_chains)
    3. agent    arrêts nominaux : choisit, parmi les thèmes de conséquences, ceux qui sont un fonctionnement
                normal de protection (nominal_evidence, flag_nominal), vérifié par le code
    4. code+LLM rédaction : les faits sont calculés en Python (facts.py) ; le LLM ne fait que les
                reformuler et un validateur vérifie chaque chiffre et chaque phénomène cité (writing.py) :
                caractérisation et causes dominantes de chaque site, une mesure par couple (site, cause) du
                plan, fiche de chaque cause, synthèse ; puis relecture par le juge

Si l'agent n'atteint pas l'objectif d'une phase, le code la termine de façon simple (meilleur
clustering conforme ; thème nommé d'après ses mots-clés ; chaînes les plus coûteuses ; aucun arrêt
nominal) et le signale dans le journal et dans le rapport.

Sorties dans outputs/2_analyse/ :
    rapport_analyse.md       le rapport final (texte de l'agent + tableaux calculés)
    journal.md / journal.json   déroulé complet, phase par phase
    decisions.json           clusterings choisis, contenu du rapport, nominaux, causes isolées, composants, essais
    clustering_essais.csv    scores de chaque essai de clustering (reprises incluses, marquées)
    themes.csv               thèmes de causes et de conséquences : nom, phrases, cohésion, justification
    phrases.csv              chaque phrase de cause ou de conséquence, avec son thème (ou vide = isolée)
    heatmap_<type>.csv/.png  heatmap thèmes de causes x thèmes de conséquences (coût, incidents, arrêt)
    fiches_causes.csv        fiche et criticité Pareto de chaque thème de cause
    mesures.csv              plan d'actions : une mesure par couple (site, cause)
    incidents_analyse.csv    chaque incident : cause, conséquence, thèmes, composant, arrêt nominal

Usage :
    python -m analysis_agent
"""

from __future__ import annotations

import argparse
import copy
import json
import re

import pandas as pd

from llm.agent import dispatch, journal_markdown, run_agent
from paths import ANALYSIS_DIR, ANALYSIS_JOURNAL, ANALYSIS_REPORT, FRAMING, INCIDENTS_FULL

from . import facts as F
from .assignment import assign_phrases
from .components import NON_RANGEES, normalize_components
from .config import (CONSEQUENCE_WORDS, DIMENSION_LABELS, DIMENSIONS, MAX_RECLUSTER_ROUNDS, MIN_CHOSEN_DBCV,
                     NONE_MAX_PCT, PHASE_CLUSTERING, PHASE_NOMINAL, PHASE_PRIORITIES, PHASE_THEMES,
                     PRIORITY_CHAINS_RANGE, ROLE, TURNS)
from .judge import judge_report
from .report import export, render
from .tools import AnalysisWorkspace, tool_schemas
from .validation_inferences import inference_sample
from .writing import cause_sheets, pair_measure, site_texts, synthesis

SINGULAR = {"cause": "cause", "consequence": "conséquence"}


def phase_tools(ws: AnalysisWorkspace, names: list[str], fixed: dict):
    """Outils d'une phase : seulement `names`, avec les arguments `fixed` (type, dimension) imposés
    par le code et retirés du schéma vu par l'agent (il ne peut pas se tromper de cible)."""
    schemas, accepted = [], {}
    for t in tool_schemas():
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
    events, ok = run_agent(ROLE.format(domaine=ws.cadrage.domain), task, tools, execute, is_done=done, max_turns=turns,
                           nudge=nudge)
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
    """Clusters que l'agent n'a pas rangés : un thème chacun, nommé d'après les mots-clés du cluster
    (sans mot de conséquence, refusé par define_themes pour une cause). Le premier nom accepté est gardé."""
    run = ws.chosen(asset_type, dimension)
    missing = ws.unthemed(asset_type, dimension)
    for t in missing:
        keywords = [k for k in run["keywords"][t] if not (dimension == "cause" and re.search(CONSEQUENCE_WORDS, k, re.I))]
        # Mots-clés uni- et bigrammes : on ne garde que ceux qui apportent un mot nouveau (« électriques poste »
        # puis « électriques » ferait doublon).
        distinct, seen = [], set()
        for k in keywords:
            if set(k.lower().split()) - seen:
                distinct.append(k)
                seen |= set(k.lower().split())
        candidates = [", ".join(distinct[:3]).capitalize(), f"Groupe de {DIMENSION_LABELS[dimension]} {t}"]
        for name in candidates:
            result = dispatch(ws.tools, "define_themes", {"asset_type": asset_type, "dimension": dimension, "themes": [{
                "name": name[:60], "topics": [t],
                "description": "Thème automatique : mots-clés dominants du cluster.",
                "justification": "THÈME AUTOMATIQUE par le code : l'agent n'a pas rangé ce cluster."}]})
            if result.get("ok") and result["resultats"][0]["ok"]:
                break
    return missing


def fallback_merge(ws: AnalysisWorkspace, asset_type: str, dimension: str) -> list:
    """Doublons de thèmes que l'agent n'a pas fusionnés : fusion sous le nom du plus gros thème."""
    merged = []
    while pairs := ws.duplicate_themes(asset_type, dimension):
        a, b, _ = pairs[0]
        name = max((ws.themes[a], ws.themes[b]), key=lambda t: t["phrases"])["name"]
        result = dispatch(ws.tools, "merge_themes", {"asset_type": asset_type, "dimension": dimension,
                                                     "theme_ids": [a, b], "name": name})
        if not result.get("ok"):
            break
        merged.append((a, b))
    return merged


def fallback_priorities(ws: AnalysisWorkspace, asset_type: str) -> None:
    """L'agent n'a pas choisi : les chaînes les plus coûteuses de la heatmap."""
    top = ws.heatmap_hotspots(asset_type)["cases"][:PRIORITY_CHAINS_RANGE[0]]
    dispatch(ws.tools, "select_priority_chains", {"asset_type": asset_type, "chains": [c["id"] for c in top],
             "justification": "CHOIX AUTOMATIQUE par le code : les chaînes au plus fort coût total."})


def themes_phase(ws: AnalysisWorkspace, journal: list, asset_type: str, dimension: str, turns: dict,
                 suffixe: str = "") -> None:
    """Phase de thèmes sur l'essai ACTIF (clustering choisi, ou reprise) : l'agent nomme les clusters ; le code
    complète ce qu'il n'a pas rangé ni fusionné."""
    k = ws.key(asset_type, dimension)
    fixed = {"asset_type": asset_type, "dimension": dimension}
    label, asset = DIMENSION_LABELS[dimension], f"du segment « {ws.cadrage.label(asset_type)} »"
    # Les clusters sont décrits dans la consigne : l'agent n'a qu'à les nommer (pas de lecture en boucle).
    details = ws.topic_details(asset_type, dimension, ws.topic_ids(asset_type, dimension))["clusters"]
    clusters = "\n".join(
        f"- cluster {d['cluster']} ({d['incidents']} incidents) ; mots-clés : {', '.join(d['mots_cles'][:4])} ; "
        f"exemples : {' / '.join(d['phrases_typiques'][:3])}" for d in details)
    suggestions = "\n".join(f"- clusters {a} et {b} : similarité {s}"
                            for a, b, s in ws.similar_cluster_pairs(asset_type, dimension)) or "- aucune"

    def remaining() -> str:
        todo = []
        if ws.unthemed(asset_type, dimension):
            todo.append(f"clusters sans thème : {ws.unthemed(asset_type, dimension)} (define_theme)")
        if ws.duplicate_themes(asset_type, dimension):
            todo.append(f"thèmes en doublon à fusionner : {ws.duplicate_themes(asset_type, dimension)} (merge_themes)")
        return "Il reste à faire : " + " ; ".join(todo) + "."

    def done() -> bool:
        return not ws.unthemed(asset_type, dimension) and not ws.duplicate_themes(asset_type, dimension)

    if not run_phase(ws, journal, f"Thèmes {k}{suffixe}",
                     PHASE_THEMES.format(label=label, asset=asset, singular=SINGULAR[dimension],
                                         clusters=clusters, suggestions=suggestions),
                     ["define_theme", "compare_clusters", "merge_themes"], fixed, done, turns["themes"], nudge=remaining):
        missing = fallback_themes(ws, asset_type, dimension)
        merged = fallback_merge(ws, asset_type, dimension)
        journal.append({"tour": 0, "type": "automatique",
                        "texte": f"{k}{suffixe} : {len(missing)} clusters rangés automatiquement {missing} ; "
                                 f"{len(merged)} doublons fusionnés automatiquement {merged}"})


def affect(ws: AnalysisWorkspace, texts: list[str], cards: list[dict]) -> list[str | None]:
    """Thème de chaque phrase (None = aucun). Sans thème à proposer, aucune phrase n'est rangée."""
    if not texts:
        return []
    if not cards:
        return [None] * len(texts)
    return assign_phrases(texts, cards, ws.cadrage.domain)


def assign_themes(ws: AnalysisWorkspace, journal: list, asset_type: str, dimension: str, turns: dict) -> None:
    """Affectation des phrases aux thèmes (code + LLM), puis reprises tant qu'il reste trop de phrases « aucun » :
    reclustering des seules phrases « aucun », thèmes nommés par l'agent sur cet essai, réaffectation des seules
    phrases restées « aucun ». Termine toujours avec l'essai principal actif."""
    k = ws.key(asset_type, dimension)
    main = ws.selected[k]["run_id"]
    ws.set_active_run(asset_type, dimension, main)
    rows, _ = ws.phrase_vectors(asset_type, dimension)
    ws.apply_assignment(asset_type, dimension, affect(ws, ws.phrases.loc[rows, "texte"].tolist(), ws.theme_cards(asset_type, dimension)))
    journal.append({"tour": 0, "type": "phase", "texte": f"Affectation {k} : {ws.none_share(asset_type, dimension)} % "
                                                        "de phrases « aucun » après l'affectation"})
    round_ = 0
    while ws.none_share(asset_type, dimension) > NONE_MAX_PCT and round_ < MAX_RECLUSTER_ROUNDS:
        round_ += 1
        before = ws.none_share(asset_type, dimension)
        run_id = ws.recluster_none(asset_type, dimension)
        if run_id is None:
            journal.append({"tour": round_, "type": "automatique", "texte": f"{k} : reprise {round_} sans cluster ; "
                            f"{before} % de « aucun » gardés"})
            break
        ws.set_active_run(asset_type, dimension, run_id)
        themes_phase(ws, journal, asset_type, dimension, turns, suffixe=f" (reprise {round_}, {run_id})")
        new_cards = ws.theme_cards(asset_type, dimension)
        ws.set_active_run(asset_type, dimension, main)
        cards = ws.theme_cards(asset_type, dimension) + new_cards
        none = ws.none_phrases(asset_type, dimension)
        ids = affect(ws, ws.phrases.loc[none, "texte"].tolist(), cards)
        ws.apply_assignment(asset_type, dimension, ids, phrase_index=none)
        journal.append({"tour": round_, "type": "phase", "texte": f"{k} : reprise {round_} ({run_id}) : « aucun » "
                        f"{before} % -> {ws.none_share(asset_type, dimension)} %"})
    ws.set_active_run(asset_type, dimension, main)


def fallback_nominal(ws: AnalysisWorkspace, journal: list, asset_type: str) -> None:
    """L'agent n'a pas choisi : aucun thème nominal (sans arrêt nominal, la perte est comptée comme une panne)."""
    ws.nominal[asset_type] = {"theme_ids": [], "justification":
                              "AUCUN THÈME NOMINAL (choix automatique par le code : l'agent n'a pas choisi)"}
    journal.append({"tour": 0, "type": "automatique", "texte": f"{asset_type} : aucun arrêt nominal retenu par le code"})


def run(max_turns: int | None = None) -> bool:
    """Lance le workflow. Renvoie True si le rapport a été produit. `max_turns` remplace, s'il est
    donné, le nombre de tours maximum de chaque phase."""
    if not INCIDENTS_FULL.exists():
        raise SystemExit(f"{INCIDENTS_FULL} introuvable : lancer d'abord l'étape pdf.")
    if not FRAMING.exists():
        from framing.pipeline import run as run_framing
        run_framing()
    df = pd.read_csv(INCIDENTS_FULL)
    df = df[df["text"].notna()].reset_index(drop=True)
    ws = AnalysisWorkspace(df)
    turns = {phase: max_turns or n for phase, n in TURNS.items()}
    journal: list[dict] = []
    print(f"Workflow d'analyse : {len(df)} incidents")

    # 0. Extraction (code) : pas de décision à prendre, donc pas d'agent. Construit aussi la table des phrases.
    for asset_type in ws.segments:
        result = dispatch(ws.tools, "extract_causes", {"asset_type": asset_type})
        journal.append({"tour": 0, "type": "outil", "outil": "extract_causes (code)",
                        "arguments": {"asset_type": asset_type}, "resultat": result})

    # 0b. Composants (code + LLM) : valeurs brutes de chaque segment regroupées, toutes rangées (sinon consignées).
    for asset_type in ws.segments:
        raw = ws.causes.reindex(ws.positions(asset_type))["composant_brut"].dropna().astype(str).tolist()
        mapping = normalize_components(raw, ws.cadrage.domain)
        unplaced = mapping.pop(NON_RANGEES, None)
        ws.components[asset_type] = mapping
        journal.append({"tour": 0, "type": "automatique", "texte": f"{asset_type} : {len(set(mapping.values()))} composants "
                        f"normalisés pour {len(mapping)} valeurs brutes" + (f" ; non rangées par le LLM : {unplaced}"
                                                                           if unplaced else "")})

    # 1. Clustering, thèmes, puis affectation des phrases, pour chaque type et chaque dimension.
    for asset_type in ws.segments:
        for dimension in DIMENSIONS:
            k = ws.key(asset_type, dimension)
            fixed = {"asset_type": asset_type, "dimension": dimension}
            label, asset = DIMENSION_LABELS[dimension], f"du segment « {ws.cadrage.label(asset_type)} »"
            if not run_phase(ws, journal, f"Clustering {k}",
                             PHASE_CLUSTERING.format(label=label, asset=asset),
                             ["run_clustering", "select_clustering"], fixed, lambda: k in ws.selected,
                             turns["clustering"]):
                run_id = fallback_clustering(ws, asset_type, dimension)
                journal.append({"tour": 0, "type": "automatique", "texte": f"{k} : clustering {run_id} choisi par le code"})
            themes_phase(ws, journal, asset_type, dimension, turns)
            assign_themes(ws, journal, asset_type, dimension, turns)

    # 2. Chaînes cause -> conséquence prioritaires, lues sur la heatmap (agent).
    for asset_type in ws.segments:
        if not run_phase(ws, journal, f"Priorités {asset_type}",
                         PHASE_PRIORITIES.format(asset=f"du segment « {ws.cadrage.label(asset_type)} »",
                                                 sites=", ".join(ws.sites[asset_type])),
                         ["heatmap_hotspots", "select_priority_chains"], {"asset_type": asset_type},
                         lambda a=asset_type: a in ws.priority_chains, turns["priorities"]):
            fallback_priorities(ws, asset_type)
            journal.append({"tour": 0, "type": "automatique", "texte": f"{asset_type} : chaînes prioritaires choisies par le code"})

    # 3. Arrêts nominaux (agent, vérifié par le code dans flag_nominal).
    for asset_type in ws.segments:
        run_phase(ws, journal, f"Arrêts nominaux {asset_type}",
                  PHASE_NOMINAL.format(asset=f"du segment « {ws.cadrage.label(asset_type)} »"),
                  ["nominal_evidence", "flag_nominal"], {"asset_type": asset_type},
                  lambda a=asset_type: a in ws.nominal, turns["nominal"])
        if asset_type not in ws.nominal:
            fallback_nominal(ws, journal, asset_type)

    # 4. Rédaction contrôlée (faits Python, reformulation LLM, validateur), puis relecture par le juge.
    journal.append({"tour": 0, "type": "phase", "texte": "Rédaction (faits calculés, reformulés et validés)"})
    table = ws.incidents_table()
    names = {tid: th["name"] for tid, th in ws.themes.items() if th["dimension"] == "cause"}
    site_reports = []
    for asset_type in ws.segments:
        sub = table[table["asset_type"] == asset_type]
        indicators = F.site_indicators(sub, names)
        for site in ws.sites[asset_type]:
            site_reports.append(site_texts(site, asset_type, sub, indicators, journal))
    pairs, coverage = F.measure_pairs(table, names)
    measures = [{**pair, **pair_measure(pair, table, journal)} for pair in pairs.to_dict(orient="records")]
    for s in site_reports:
        s["mesures"] = [m["id"] for m in measures if m["site"] == s["site"]]
    isolated = {a: F.isolated_causes(ws.phrases, a) for a in ws.segments}
    sheets = cause_sheets(table, measures, journal)
    synthese, synthese_facts = synthesis(table, measures, coverage, isolated, journal, ws.priority_chains)
    ws.report = {"sites": site_reports, "mesures": measures, "couverture": coverage, "fiches": sheets,
                 "synthese": synthese, "faits_synthese": synthese_facts}
    journal.append({"tour": 0, "type": "phase", "texte": "Relecture par le juge (grille, correction validée par le code)"})
    judge_report(ws.report, journal, table)
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
          f"{len(ws.themes)} thèmes ; {len(measures)} mesures ; {len(auto)} reprises automatiques par le code")
    print(f"Sauvegardé dans {ANALYSIS_DIR}/ ; validation manuelle des inférences : {sample}")
    print(f"Rapport : {ANALYSIS_REPORT}" if finished else
          "Pas de rapport : voir journal.md.")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-turns", type=int, default=None, help="tours maximum par phase (défaut : config.TURNS)")
    run(parser.parse_args().max_turns)
