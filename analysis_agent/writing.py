"""Rédaction contrôlée : le LLM reformule des faits calculés en Python, un validateur vérifie.

Pour chaque texte du rapport (justification d'un profil, causes dominantes, actions, synthèse) :
  1. le LLM reçoit UNIQUEMENT les faits calculés (facts.py) et répond en JSON imposé ;
  2. le validateur compare sa réponse aux faits :
       - chaque nombre cité doit exister dans les faits (arrondi toléré) ;
       - chaque phénomène cité (corrosion, foudre, chaleur...) doit figurer dans les faits ;
       - le mot « rang » est interdit (les rangs inventés étaient l'erreur la plus fréquente) ;
       - une action ne doit pas être vague (« renforcer la maintenance ») ;
  3. en cas d'erreur, le LLM reçoit la liste des erreurs et réessaie (WRITE_RETRIES fois) ;
  4. si l'erreur persiste, le texte fautif est remplacé par une phrase générée en Python, et c'est
     consigné dans le journal.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from llm.client import chat

from . import facts as F
from .config import (ACTION_TYPES, CURATIVE_PATTERN, GENERIC_ACTIONS, PREVENTIVE_CATALOGUE, SITE_PROFILES,
                     WRITE_RETRIES, WRITER_ROLE)

# Phénomènes surveillés par le validateur : un texte ne peut en citer un que s'il figure dans les faits.
CONCEPTS = {
    "chaleur": r"chaleur|canicule",
    "tempête / vent fort": r"tempête|vent fort|rafale",
    "foudre / orage": r"foudre|orage",
    "grêle": r"grêle",
    "embruns / corrosion": r"embrun|corrosion|salin",
    "poussière / encrassement": r"poussière|encrass|salissure",
    "nuisibles": r"nuisible|rongeu|rongement|animaux",
    "sol / fondations": r"\bsols?\b|fondation|tassement",
    "défaut de fabrication": r"fabrica|série défectueuse",
    "usure / vieillissement": r"usure|vieilli",
    # Ce que décrit un profil sans être mesuré : âge, environnement. Interdit si absent des faits
    # (une justification doit s'appuyer sur les mesures, pas recopier la description du profil).
    "âge du site": r"\b\d+\s*ans\b|âgé|récent|ancien|jeunesse|vieux",
    "environnement du site": r"\bzones?\b|côti|continental|altitude|plaine|agricole|océan|littoral|maritime",
}
NUMBER = re.compile(r"(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?\s*(k€|M€|k ?€|M ?€)?")


# ------------------------------------------------------------------------------------------------
# Validateur
# ------------------------------------------------------------------------------------------------

def numbers(text: str) -> list[float]:
    found = []
    for whole, decimals, unit in NUMBER.findall(text):
        value = float(re.sub(r"\D", "", whole) + (f".{decimals}" if decimals else ""))
        if unit:
            value *= 1_000 if unit.startswith("k") else 1_000_000
        found.append(value)
    return found


def concepts(text: str) -> set[str]:
    return {name for name, pattern in CONCEPTS.items() if re.search(pattern, text, re.IGNORECASE)}


def text_errors(text: str, facts_text: str) -> list[str]:
    """Écarts entre un texte rédigé et les faits qu'on lui a donnés."""
    errors = []
    allowed = numbers(facts_text)
    for x in numbers(text):
        if x <= 10 and x == int(x):  # petits entiers (« 3 causes », « 2 sites ») : toujours acceptés
            continue
        if not any(abs(x - a) <= max(0.6, 0.02 * abs(a)) for a in allowed):
            errors.append(f"le nombre {x:g} ne figure pas dans les faits")
    extra = concepts(text) - concepts(facts_text)
    if extra:
        errors.append(f"cite {', '.join(sorted(extra))}, absent des faits")
    if re.search(r"\brangs?\b", text, re.IGNORECASE):
        errors.append("le mot « rang » est interdit : décris l'écart avec les chiffres")
    return errors


TO_DEFINE = "Action à définir par l'équipe maintenance"


def action_errors(action: str) -> list[str]:
    if action.startswith(TO_DEFINE):  # absence d'action assumée et signalée : pas une erreur de rédaction
        return []
    if len(action.split()) < 5:
        return ["action trop courte : décris une action concrète (quoi, sur quel équipement)"]
    if any(re.search(p, action, re.IGNORECASE) for p in GENERIC_ACTIONS):
        return ["action trop vague : agis sur le mécanisme précis de la cause (pas « renforcer la maintenance »)"]
    return []


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


def measure_errors(action: dict, cause: str, factor: str) -> list[str]:
    """Une mesure du plan doit être PRÉVENTIVE, à l'échelle du site, avec un mécanisme lié à la cause."""
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
    cause_words = {w for w in re.findall(r"[a-zàâçéèêëîïôûùüœ]{5,}", (cause + " " + factor).lower())}
    linked = bool(concepts(mecanisme) & concepts(cause + " " + factor)) or any(w in mecanisme.lower() for w in cause_words)
    if not linked:
        errors.append(f"le mécanisme doit expliquer l'effet sur la cause « {cause} »")
    return errors


def catalogue_measure(cause: str, factor: str) -> dict:
    """Mesure préventive type du facteur (config.PREVENTIVE_CATALOGUE), utilisée si le LLM échoue."""
    if factor in PREVENTIVE_CATALOGUE:
        kind, mesure, mecanisme = PREVENTIVE_CATALOGUE[factor]
        return {"cause": cause, "type": kind, "mesure": mesure + ".", "mecanisme": mecanisme + "."}
    return {"cause": cause, "type": "procédure", "mesure": f"{TO_DEFINE} : analyser les incidents « {cause} » pour définir "
            "une mesure préventive.", "mecanisme": "aucune mesure préventive type n'existe pour ce facteur."}


def site_texts(site: str, asset_type: str, profile: str, incidents, indicators: dict, journal: list[dict]) -> dict:
    """Justification du profil, causes dominantes et une mesure préventive par cause prioritaire."""
    g = incidents[incidents["site"] == site]
    points = F.distinctive_points(indicators, site)
    dominant = F.dominant_causes(g)
    priority = F.priority_causes(g)
    factors = {p["cause"]: F.dominant_factor(incidents, p["cause"]) for p in priority}

    by_elimination = not F.expectations_met(indicators, site, asset_type, profile)
    profile_facts = "\n".join(f"- {F.point_text(p)}" for p in points) or "- aucun critère ne distingue nettement ce site"
    if by_elimination:
        profile_facts += ("\n- Profil attribué par élimination : aucun critère mesuré ne correspond nettement à ce "
                          "profil ; les autres sites correspondent mieux aux autres profils.")
    dominant_facts = "\n".join(f"- {d['cause']} : {d['incidents']} incidents, {d['part_incidents_pct']} % des incidents "
                               f"du site" for d in dominant)
    priority_facts = "\n".join(f"- {p['cause']} (facteur : {factors[p['cause']] or 'non précisé'}) : {p['incidents']} "
                               f"incidents, coût total {F.euros(p['cout_total'])}" for p in priority)
    facts_text = (f"Site : {site}. Profil attribué : {profile} ({SITE_PROFILES[asset_type][profile]}).\n"
                  f"Points distinctifs du site :\n{profile_facts}\n"
                  f"Causes connues les plus fréquentes :\n{dominant_facts}\n"
                  f"Causes connues les plus coûteuses, hors arrêts de sécurité nominaux (à traiter dans le plan) :\n"
                  f"{priority_facts}")
    task = ("Rédige en JSON :\n"
            "- justification_profil : 2 phrases qui expliquent pourquoi ce site correspond au profil, en citant les "
            "points distinctifs et leurs chiffres. Ne décris pas le profil lui-même et n'évoque aucun phénomène absent "
            "des points distinctifs.\n"
            "- causes_dominantes : 1 phrase qui reformule les causes connues les plus fréquentes, avec leurs chiffres.\n"
            "- actions : pour CHAQUE cause la plus coûteuse, une MESURE PRÉVENTIVE ou ORGANISATIONNELLE à l'échelle du "
            "site (programme d'inspection, fréquence de nettoyage, équipement de protection, procédure, monitoring), "
            "PAS une réparation ponctuelle (« remplacer le module » est refusé). Donne son type, la mesure (quoi, sur "
            "quels équipements, à quelle fréquence) et son mécanisme (pourquoi elle réduit cette cause).")
    names = [p["cause"] for p in priority]
    schema = {"type": "object", "additionalProperties": False,
              "required": ["justification_profil", "causes_dominantes", "actions"],
              "properties": {"justification_profil": {"type": "string"}, "causes_dominantes": {"type": "string"},
                             "actions": {"type": "array", "items": {
                                 "type": "object", "additionalProperties": False,
                                 "required": ["cause", "type", "mesure", "mecanisme"],
                                 "properties": {"cause": {"type": "string", "enum": names or ["aucune"]},
                                                "type": {"type": "string", "enum": ACTION_TYPES},
                                                "mesure": {"type": "string"}, "mecanisme": {"type": "string"}}}}}}

    def check(out: dict) -> dict[str, list[str]]:
        errors = {"justification_profil": text_errors(out["justification_profil"], profile_facts)
                  + (["dis explicitement que le profil est attribué par élimination"]
                     if by_elimination and "élimination" not in out["justification_profil"].lower() else []),
                  "causes_dominantes": text_errors(out["causes_dominantes"], dominant_facts)}
        given = {a["cause"]: a for a in out["actions"]}
        for cause in names:
            errors[f"action:{cause}"] = (["mesure manquante pour cette cause"] if cause not in given else
                                         measure_errors(given[cause], cause, factors[cause]))
        return errors

    def fallback(field: str, out: dict) -> dict:
        out = dict(out)
        if field == "justification_profil":
            out[field] = ((f"Profil {profile} attribué par élimination (aucun critère mesuré ne lui correspond nettement)"
                           if by_elimination else f"Profil {profile} retenu")
                          + (" ; points distinctifs : " + " ; ".join(F.point_text(p) for p in points) if points else "") + ".")
        elif field == "causes_dominantes":
            out[field] = "Causes connues les plus fréquentes : " + ", ".join(
                f"{d['cause']} ({d['incidents']} incidents, {d['part_incidents_pct']} % des incidents du site)" for d in dominant) + "."
        elif field.startswith("action:"):
            cause = field.split(":", 1)[1]
            out["actions"] = [a for a in out["actions"] if a["cause"] != cause] + [catalogue_measure(cause, factors[cause])]
        return out

    out = write(f"Rédaction {site}", task, facts_text, schema, check, fallback, journal)
    actions = {a["cause"]: a for a in out["actions"]}
    plan = [{"cause": p["cause"], "facteur": factors[p["cause"]], "type": actions.get(p["cause"], {}).get("type", ""),
             "action": actions.get(p["cause"], {}).get("mesure", ""), "mecanisme": actions.get(p["cause"], {}).get("mecanisme", ""),
             "priorite": p["priorite"], "incidents": p["incidents"], "cout_total": p["cout_total"],
             "part_cout_pct": p["part_cout_pct"]} for p in priority]
    return {"site": site, "profil": profile, "points_distinctifs": [F.point_text(p) for p in points],
            "justification_profil": out["justification_profil"], "causes_dominantes": out["causes_dominantes"],
            "causes_dominantes_chiffres": dominant, "origine_causes_dominantes": F.cause_origin(g, [d["cause"] for d in dominant]),
            "plan": plan, "par_elimination": by_elimination,
            "faits": {"justification_profil": profile_facts, "causes_dominantes": dominant_facts, "plan": priority_facts}}


def synthesis(incidents, site_reports: list[dict], fleet, journal: list[dict]) -> tuple[str, str]:
    """Synthèse pour un responsable : chiffres clés et actions prioritaires du parc."""
    by_site = incidents.groupby("site")["estimated_cost_eur"].sum().sort_values(ascending=False)
    actions = {(r["site"], p["cause"]): p["action"] for r in site_reports for p in r["plan"]}
    structural = F.structural_losses(incidents)
    top = fleet.head(3)
    facts_text = "\n".join([
        f"- {len(incidents)} incidents, coût total {F.euros(incidents['estimated_cost_eur'].sum())}.",
        *(f"- {a} : {F.unknown_share(incidents[incidents['asset_type'] == a])} % des incidents sans cause précisée "
          "dans le rapport." for a in sorted(incidents["asset_type"].unique())),
        f"- Site le plus coûteux : {by_site.index[0]} ({F.euros(by_site.iloc[0])}).",
        f"- Arrêts de sécurité nominaux (fonctionnement normal des protections, pas des pannes) : "
        f"{int(structural['arrets'].sum())} incidents, {F.euros(structural['cout_par_an'].sum())} par an de perte structurelle.",
        *(f"- Priorité {i} du parc : {r.cause} sur {r.site} ({r.incidents} incidents, {F.euros(r.cout_par_an)} par an) ; "
          f"mesure : {actions.get((r.site, r.cause), 'à définir')}" for i, r in enumerate(top.itertuples(), 1)),
    ])
    task = ("Rédige en JSON une synthèse de 4 à 6 phrases pour un responsable : coût total, part des causes non "
            "précisées, site le plus coûteux, arrêts de sécurité nominaux (perte structurelle, pas des pannes), puis les "
            "3 mesures prioritaires du parc avec leur coût par an.")
    schema = {"type": "object", "additionalProperties": False, "required": ["synthese"],
              "properties": {"synthese": {"type": "string"}}}

    def check(out: dict) -> dict[str, list[str]]:
        errors = text_errors(out["synthese"], facts_text)
        if len([x for x in numbers(out["synthese"]) if x > 10]) < 3:
            errors.append("cite au moins 3 chiffres des faits")
        return {"synthese": errors}

    def fallback(field: str, out: dict) -> dict:
        return {"synthese": "Chiffres clés :\n" + facts_text}

    return write("Synthèse", task, facts_text, schema, check, fallback, journal)["synthese"], facts_text
