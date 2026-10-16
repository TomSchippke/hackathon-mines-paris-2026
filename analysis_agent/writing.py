"""Rédaction contrôlée : le LLM reformule des faits calculés en Python, un validateur vérifie.

Pour chaque texte du rapport (caractérisation ou causes dominantes d'un site, mesure d'un couple
site x cause, synthèse) :
  1. le LLM reçoit UNIQUEMENT les faits calculés (facts.py) et répond en JSON imposé ;
  2. le validateur compare sa réponse aux faits :
       - chaque nombre cité doit exister dans les faits (arrondi toléré) ;
       - chaque phénomène cité (un mot d'un nom de thème de cause, ou l'âge ou l'environnement du site)
         doit figurer dans les faits ;
       - le mot « rang » est interdit (les rangs inventés étaient l'erreur la plus fréquente) ;
       - une mesure doit être préventive, non vague, liée à la cause, et nommer son site ;
  3. en cas d'erreur, le LLM reçoit la liste des erreurs et réessaie (WRITE_RETRIES fois) ;
  4. si l'erreur persiste, le texte fautif est remplacé par une phrase générée en Python (une mesure
     devient « À définir » avec ses chiffres), et c'est consigné dans le journal.
"""

from __future__ import annotations

import json
import re
from typing import Callable

import pandas as pd

from framing.cadrage import current
from llm.client import chat

from . import facts as F
from .config import (ACTION_TYPES, CURATIVE_PATTERN, GENERIC_ACTIONS, TO_DEFINE, WRITE_RETRIES, WRITER_ROLE)

# Phénomènes surveillés par le validateur, en plus des noms de thèmes de cause : ce qui décrit le site
# sans être mesuré (âge, environnement). Interdit si absent des faits.
GENERIC_CONCEPTS = {
    "âge du site": r"\b\d+\s*ans\b|âgé|récent|ancien|jeunesse|vieux",
    "environnement du site": r"\bzones?\b|côti|continental|altitude|plaine|agricole|océan|littoral|maritime",
}
NUMBER = re.compile(r"(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?\s*(k€|M€|k ?€|M ?€)?")
WORD = r"[a-zàâçéèêëîïôûùüœ]{5,}"


# ------------------------------------------------------------------------------------------------
# Validateur
# ------------------------------------------------------------------------------------------------

def cause_names(incidents: pd.DataFrame) -> list[str]:
    """Noms des thèmes de cause connue présents dans la table (causes racines)."""
    return sorted(F.theme_names(incidents).values())


def concept_patterns(theme_names: list[str]) -> dict[str, str]:
    """Phénomènes surveillés : chaque mot d'au moins 5 lettres d'un nom de thème de cause (reconnu au début
    d'un mot, pour les pluriels), plus GENERIC_CONCEPTS. Un texte ne peut citer un phénomène que s'il figure
    dans les faits."""
    words = {w for name in theme_names for w in re.findall(WORD, name.lower())}
    return {**{w: rf"\b{re.escape(w)}" for w in sorted(words)}, **GENERIC_CONCEPTS}


def numbers(text: str) -> list[float]:
    found = []
    for whole, decimals, unit in NUMBER.findall(text):
        value = float(re.sub(r"\D", "", whole) + (f".{decimals}" if decimals else ""))
        if unit:
            value *= 1_000 if unit.startswith("k") else 1_000_000
        found.append(value)
    return found


def concepts(text: str, patterns: dict[str, str]) -> set[str]:
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)}


def text_errors(text: str, facts_text: str, patterns: dict[str, str]) -> list[str]:
    """Écarts entre un texte rédigé et les faits qu'on lui a donnés."""
    errors = []
    allowed = numbers(facts_text)
    for x in numbers(text):
        if x <= 10 and x == int(x):  # petits entiers (« 3 causes », « 2 sites ») : toujours acceptés
            continue
        if not any(abs(x - a) <= max(0.6, 0.02 * abs(a)) for a in allowed):
            errors.append(f"le nombre {x:g} ne figure pas dans les faits")
    extra = concepts(text, patterns) - concepts(facts_text, patterns)
    if extra:
        errors.append(f"cite {', '.join(sorted(extra))}, absent des faits")
    if re.search(r"\brangs?\b", text, re.IGNORECASE):
        errors.append("le mot « rang » est interdit : décris l'écart avec les chiffres")
    return errors


def action_errors(action: str) -> list[str]:
    if action.startswith(TO_DEFINE):  # absence d'action assumée et signalée : pas une erreur de rédaction
        return []
    if len(action.split()) < 5:
        return ["action trop courte : décris une action concrète (quoi, sur quel équipement)"]
    if any(re.search(p, action, re.IGNORECASE) for p in GENERIC_ACTIONS):
        return ["action trop vague : agis sur le mécanisme précis de la cause (pas « renforcer la maintenance »)"]
    return []


def measure_errors(action: dict, cause: str, site: str, other_sites: list[str]) -> list[str]:
    """Une mesure du plan doit être PRÉVENTIVE, sur CE site, avec un mécanisme lié à la cause."""
    errors = []
    mesure, mecanisme = action.get("mesure", ""), action.get("mecanisme", "")
    if mesure.startswith(TO_DEFINE):
        return []
    if re.search(CURATIVE_PATTERN, mesure, re.IGNORECASE):
        errors.append("mesure curative (réparation ponctuelle) : propose une mesure PRÉVENTIVE ou organisationnelle "
                      "à l'échelle du site (programme d'inspection, fréquence de nettoyage, protection, procédure, monitoring)")
    errors += action_errors(mesure)
    if len(mecanisme.split()) < 6:
        errors.append("mécanisme trop court : explique en une phrase pourquoi la mesure réduit cette cause")
    cause_words = re.findall(WORD, cause.lower())
    if cause_words and not any(w in mecanisme.lower() for w in cause_words):
        errors.append(f"le mécanisme doit expliquer l'effet sur la cause « {cause} »")
    if not re.search(rf"\b{re.escape(site)}\b|\bce site\b", mesure, re.IGNORECASE):
        errors.append(f"la mesure doit nommer le site ({site}) ou écrire « ce site »")
    named = [o for o in other_sites if re.search(rf"\b{re.escape(o)}\b", mesure + " " + mecanisme)]
    if named:
        errors.append(f"la mesure cite un autre site : {', '.join(named)}")
    return errors


# ------------------------------------------------------------------------------------------------
# Rédaction avec reprises
# ------------------------------------------------------------------------------------------------

def write(name: str, task: str, facts_text: str, schema: dict, check: Callable[[dict], dict[str, list[str]]],
          fallback: Callable[[str, dict], object], journal: list[dict]) -> dict:
    """Fait rédiger un texte par le LLM, le valide, réessaie, puis remplace ce qui reste faux.

    `check(réponse)` renvoie {champ: [erreurs]} ; `fallback(champ, réponse)` donne la valeur Python
    qui remplace un champ encore faux après les reprises.
    """
    response_format = {"type": "json_schema", "json_schema": {"name": "redaction", "strict": True, "schema": schema}}
    messages = [{"role": "system", "content": WRITER_ROLE}, {"role": "user", "content": f"{task}\n\nFAITS :\n{facts_text}"}]
    out, errors = {}, {}
    for attempt in range(1, WRITE_RETRIES + 2):
        answer = chat(messages, role="writing", response_format=response_format)
        out = json.loads(answer["content"])
        errors = {k: v for k, v in check(out).items() if v}
        journal.append({"tour": attempt, "type": "redaction",
                        "texte": f"{name} — essai {attempt} : " + ("validé" if not errors else
                                                                   "erreurs " + json.dumps(errors, ensure_ascii=False))})
        if not errors:
            return out
        messages += [{"role": "assistant", "content": answer["content"]},
                     {"role": "user", "content": "Le validateur a trouvé ces erreurs : "
                      + json.dumps(errors, ensure_ascii=False) + ". Corrige-les en ne gardant que les faits fournis, "
                      "et réponds à nouveau en JSON."}]
    for field in errors:
        out = fallback(field, out)
        journal.append({"tour": 0, "type": "automatique",
                        "texte": f"{name} : « {field} » encore faux après {WRITE_RETRIES} reprises, remplacé par le "
                                 "texte généré en Python"})
    return out


# ------------------------------------------------------------------------------------------------
# Sites : caractérisation et causes dominantes
# ------------------------------------------------------------------------------------------------

def site_texts(site: str, asset_type: str, incidents: pd.DataFrame, indicators: dict, journal: list[dict]) -> dict:
    """Ce qui distingue le site des autres sites de son segment, et ses causes dominantes (connues, les plus
    fréquentes). `incidents` : les incidents du segment ; `indicators` : F.site_indicators de ce segment."""
    g = incidents[incidents["site"] == site]
    points = F.distinctive_points(indicators, site)
    dominant = F.dominant_causes(g)
    patterns = concept_patterns(cause_names(incidents))
    points_facts = "\n".join(f"- {F.point_text(p)}" for p in points) or "- aucun critère ne distingue nettement ce site"
    dominant_facts = "\n".join(f"- {d['cause']} : {d['incidents']} incidents, {d['part_incidents_pct']} % des incidents "
                               f"du site" for d in dominant)
    facts_text = (f"Site : {site} ({current().label(asset_type)}).\n"
                  f"Points distinctifs du site (par rapport aux autres sites du segment) :\n{points_facts}\n"
                  f"Causes connues les plus fréquentes :\n{dominant_facts}")
    task = ("Rédige en JSON :\n"
            "- caracterisation : 2 phrases qui disent ce qui distingue ce site des autres sites du même segment, en "
            "citant les points distinctifs et leurs chiffres (ou, s'il n'y en a aucun, qu'aucun critère ne le "
            "distingue nettement). N'évoque aucun phénomène absent des points distinctifs.\n"
            "- causes_dominantes : 1 phrase qui reformule les causes connues les plus fréquentes, avec leurs chiffres.")
    schema = {"type": "object", "additionalProperties": False, "required": ["caracterisation", "causes_dominantes"],
              "properties": {"caracterisation": {"type": "string"}, "causes_dominantes": {"type": "string"}}}

    def check(out: dict) -> dict[str, list[str]]:
        return {"caracterisation": text_errors(out["caracterisation"], points_facts, patterns),
                "causes_dominantes": text_errors(out["causes_dominantes"], dominant_facts, patterns)}

    def fallback(field: str, out: dict) -> dict:
        out = dict(out)
        if field == "caracterisation":
            out[field] = ("Ce qui distingue ce site : " + " ; ".join(F.point_text(p) for p in points) + "." if points else
                          "Aucun critère ne distingue nettement ce site des autres sites du segment.")
        elif field == "causes_dominantes":
            out[field] = "Causes connues les plus fréquentes : " + ", ".join(
                f"{d['cause']} ({d['incidents']} incidents, {d['part_incidents_pct']} % des incidents du site)"
                for d in dominant) + "."
        return out

    out = write(f"Rédaction {site}", task, facts_text, schema, check, fallback, journal)
    return {"site": site, "asset_type": asset_type, "caracterisation": out["caracterisation"],
            "points_distinctifs": [F.point_text(p) for p in points], "causes_dominantes": out["causes_dominantes"],
            "causes_dominantes_chiffres": dominant,
            "origine_causes_dominantes": F.cause_origin(g, [d["cause"] for d in dominant]),
            "mesures": [], "faits": {"caracterisation": points_facts, "causes_dominantes": dominant_facts}}


# ------------------------------------------------------------------------------------------------
# Plan : une mesure par couple (site, cause)
# ------------------------------------------------------------------------------------------------

def pair_measure(pair: dict, incidents: pd.DataFrame, journal: list[dict]) -> dict:
    """Mesure d'un couple (site, cause) du plan : une action terrain sur CE site contre CETTE cause, avec son
    mécanisme. `pair` : une ligne de F.measure_pairs ; `incidents` : la table complète des incidents.
    Sans mesure valide après les reprises : « À définir », avec les chiffres (jamais de mesure générique)."""
    site, cause, asset_type = pair["site"], pair["cause"], pair["asset_type"]
    seg = incidents[incidents["asset_type"] == asset_type]
    g = seg[(seg["site"] == site) & (seg["theme_cause_id"] == pair["theme_id"])]
    g = g[~F.nominal_stops(g)]
    others = sorted(set(incidents["site"]) - {site})
    patterns = concept_patterns(cause_names(seg))
    causes_ex = " / ".join(g["cause_racine"].dropna().drop_duplicates().head(3)) or "aucune"
    actions_ex = " / ".join(F.typical_actions(g, cause, 3)) or "aucune"
    facts_text = (f"Site : {site} ({current().label(asset_type)}). Cause : {cause}.\n"
                  f"{pair['incidents']} incidents, coût par an {F.euros(pair['cout_par_an'])} (coût total "
                  f"{F.euros(pair['cout_total'])}), {pair['part_cout_pct']} % du coût des causes connues.\n"
                  f"Portée de la cause : {pair['portee']}.\n"
                  f"Composant principal : {pair['composant_principal']}. Conséquence principale : "
                  f"{pair['consequence_principale'] or 'non précisée'}.\n"
                  f"Exemples de causes extraites : {causes_ex}.\n"
                  f"Actions correctives les plus fréquentes (extraction) : {actions_ex}.")
    task = ("Rédige en JSON UNE mesure pour ce site et cette cause :\n"
            f"- type : une catégorie de la liste ;\n"
            f"- mesure : UNE action terrain concrète sur ce site ({site}) : quoi, sur quel équipement, à quelle "
            "fréquence ou sur quel déclencheur. Elle doit nommer le site ou écrire « ce site ». Préventive : pas une "
            "réparation ponctuelle (« remplacer le module » est refusé), pas une formule générique (« renforcer la "
            "maintenance » est refusé) ;\n"
            "- mecanisme : une phrase qui explique pourquoi cette action réduit cette cause précise.")
    schema = {"type": "object", "additionalProperties": False, "required": ["type", "mesure", "mecanisme"],
              "properties": {"type": {"type": "string", "enum": ACTION_TYPES},
                             "mesure": {"type": "string"}, "mecanisme": {"type": "string"}}}

    def check(out: dict) -> dict[str, list[str]]:
        text = out["mesure"] + " " + out["mecanisme"]
        return {"mesure": measure_errors(out, cause, site, others) + text_errors(text, facts_text, patterns)}

    def fallback(field: str, out: dict) -> dict:
        return {"type": "procédure", "mesure": f"{TO_DEFINE} : {cause} sur {site} ({pair['incidents']} incidents, "
                f"{F.euros(pair['cout_par_an'])} par an, composant {pair['composant_principal']})",
                "mecanisme": "", "a_definir": True}

    out = write(f"Mesure {site} / {cause}", task, facts_text, schema, check, fallback, journal)
    a_definir = bool(out.get("a_definir")) or out["mesure"].startswith(TO_DEFINE)
    return {"type": out["type"], "mesure": out["mesure"], "mecanisme": out["mecanisme"], "a_definir": a_definir,
            "faits": facts_text}


# ------------------------------------------------------------------------------------------------
# Fiches des causes (sans mesure : les mesures sont au niveau du couple site x cause)
# ------------------------------------------------------------------------------------------------

def cause_sheets(incidents: pd.DataFrame, measures: list[dict], journal: list[dict]) -> list[dict]:
    """Fiche de chaque thème de cause connue : description et conséquences typiques (heatmap), criticité
    Pareto, sites ayant une mesure pour cette cause. Mêmes règles que le reste : faits calculés, validés."""
    sheets = []
    for asset_type in sorted(incidents["asset_type"].unique()):
        sub = incidents[incidents["asset_type"] == asset_type]
        chains = F.chain_stats(sub)
        patterns = concept_patterns(cause_names(sub))
        for row in F.cause_criticality(sub).itertuples():
            g = sub[sub["theme_cause"] == row.cause]
            cons = chains[chains["cause"] == row.cause].head(3)
            sites = (100 * g["site"].value_counts(normalize=True)).round(1)
            cons_facts = "\n".join(f"- {c.consequence} : {c.incidents} incidents, coût total {F.euros(c.cout_total)}"
                                   for c in cons.itertuples()) or "- aucune conséquence classée"
            with_measure = sorted({m["site"] for m in measures
                                   if m["asset_type"] == asset_type and m["theme_id"] == row.theme_id})
            facts_text = (f"Cause : {row.cause} ({current().label(asset_type)}).\n"
                          f"{row.incidents} incidents, coût total {F.euros(row.cout_total)}, coût moyen "
                          f"{F.euros(row.cout_moyen)}, {row.part_cout_pct} % du coût des causes connues.\n"
                          f"Sites : " + ", ".join(f"{s} {v} %" for s, v in sites.items()) + ".\n"
                          f"Maintenance suspectée (hypothèse) : {row.maintenance_suspectee_pct} % des incidents.\n"
                          f"Exemples de causes extraites : " + " / ".join(g["cause_racine"].dropna().head(3)) + "\n"
                          f"Conséquences les plus coûteuses :\n{cons_facts}")
            task = ("Rédige en JSON la fiche de cette cause pour une équipe de maintenance :\n"
                    "- description : 1 à 2 phrases sur ce qui se passe et pourquoi, avec les mots des faits.\n"
                    "- consequences : 1 phrase qui reformule les conséquences les plus coûteuses, avec leurs chiffres.")
            schema = {"type": "object", "additionalProperties": False, "required": ["description", "consequences"],
                      "properties": {"description": {"type": "string"}, "consequences": {"type": "string"}}}

            def check(out: dict, facts=facts_text, cf=cons_facts, pats=patterns) -> dict:
                return {"description": text_errors(out["description"], facts, pats),
                        "consequences": text_errors(out["consequences"], cf, pats)}

            def fallback(field: str, out: dict, cause=row.cause, n=row.incidents, cons=cons) -> dict:
                out = dict(out)
                if field == "description":
                    out[field] = f"{cause} : {n} incidents."
                elif field == "consequences":
                    out[field] = "Conséquences les plus coûteuses : " + (", ".join(
                        f"{c.consequence} ({c.incidents} incidents, {F.euros(c.cout_total)})" for c in cons.itertuples())
                        or "aucune conséquence classée") + "."
                return out

            out = write(f"Fiche {asset_type} / {row.cause}", task, facts_text, schema, check, fallback, journal)
            sheets.append({"asset_type": asset_type, "theme_id": row.theme_id, "cause": row.cause, "rang": row.rang,
                           "criticite": row.criticite, "incidents": row.incidents, "cout_total": row.cout_total,
                           "cout_moyen": row.cout_moyen, "arret_total_h": row.arret_total_h,
                           "part_cout_pct": row.part_cout_pct, "cumul_cout_pct": row.cumul_cout_pct,
                           "maintenance_suspectee_pct": row.maintenance_suspectee_pct,
                           "consequence_principale": row.consequence_principale,
                           "composant_principal": row.composant_principal,
                           "description": out["description"], "consequences": out["consequences"],
                           "sites_avec_mesure": with_measure, "faits": facts_text})
    return sheets


# ------------------------------------------------------------------------------------------------
# Synthèse
# ------------------------------------------------------------------------------------------------

def synthesis(incidents: pd.DataFrame, measures: list[dict], coverage: dict, isolated: dict[str, dict], journal: list[dict],
              chains: dict[str, dict] | None = None) -> tuple[str, str]:
    """Synthèse pour un responsable : coût, causes non précisées, arrêts nominaux, couverture du plan, 3 mesures
    prioritaires, causes isolées et chaînes prioritaires. Renvoie (texte, faits)."""
    by_site = incidents.groupby("site")["estimated_cost_eur"].sum().sort_values(ascending=False)
    structural = F.structural_losses(incidents)
    patterns = concept_patterns(cause_names(incidents))
    plafond = " ; plafond de mesures atteint, le reste des causes n'est pas couvert" if coverage["plafond_atteint"] else ""
    facts_text = "\n".join([
        f"- {len(incidents)} incidents, coût total {F.euros(incidents['estimated_cost_eur'].sum())}.",
        *(f"- {current().label(a)} : {F.unknown_share(incidents[incidents['asset_type'] == a])} % des incidents sans cause "
          "précisée dans le rapport." for a in sorted(incidents["asset_type"].unique())),
        f"- Site le plus coûteux : {by_site.index[0]} ({F.euros(by_site.iloc[0])}).",
        f"- Arrêts de sécurité nominaux (fonctionnement normal des protections, pas des pannes) : "
        f"{int(structural['arrets'].sum())} incidents, {F.euros(structural['cout_par_an'].sum())} par an de perte structurelle.",
        f"- {coverage['mesures']} mesures couvrent {coverage['pct']} % du coût des causes connues hors arrêts nominaux"
        f"{plafond}.",
        *(f"- Mesure {m['id']} ({m['site']}, {m['cause']}, {F.euros(m['cout_par_an'])} par an) : {m['mesure']}"
          for m in measures[:3]),
        *(f"- Causes isolées ({current().label(a)}) : {d['pct']} % des phrases de cause ne sont rattachées à aucun thème."
          for a, d in isolated.items()),
        *(f"- Chaîne prioritaire ({current().label(a)}) : {c['cause']} -> {c['consequence']} ({c['incidents']} incidents, "
          f"{F.euros(c['cout_total'])})" for a, d in (chains or {}).items() for c in d["chaines"][:2]),
    ])
    task = ("Rédige en JSON une synthèse de 4 à 7 phrases pour un responsable : coût total, part des causes non "
            "précisées, site le plus coûteux, arrêts de sécurité nominaux (perte structurelle, pas des pannes), la "
            "couverture du plan (nombre de mesures et part du coût couverte), les 3 mesures prioritaires avec leur coût "
            "par an, puis les chaînes cause -> conséquence prioritaires.")
    schema = {"type": "object", "additionalProperties": False, "required": ["synthese"],
              "properties": {"synthese": {"type": "string"}}}

    def check(out: dict) -> dict[str, list[str]]:
        errors = text_errors(out["synthese"], facts_text, patterns)
        if len([x for x in numbers(out["synthese"]) if x > 10]) < 3:
            errors.append("cite au moins 3 chiffres des faits")
        return {"synthese": errors}

    def fallback(field: str, out: dict) -> dict:
        return {"synthese": "Chiffres clés :\n" + facts_text}

    return write("Synthèse", task, facts_text, schema, check, fallback, journal)["synthese"], facts_text
