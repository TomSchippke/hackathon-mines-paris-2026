"""Rédaction contrôlée : le LLM reformule des faits calculés en Python, un validateur vérifie.

Pour chaque texte du rapport (profil ou caractérisation d'un site, causes dominantes, actions, synthèse) :
  1. le LLM reçoit UNIQUEMENT les faits calculés (facts.py) et répond en JSON imposé ;
  2. le validateur compare sa réponse aux faits :
       - chaque nombre cité doit exister dans les faits (arrondi toléré) ;
       - chaque phénomène cité (un facteur du cadrage, reconnu par ses mots-clés) doit figurer dans les faits ;
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

from framing.cadrage import current
from llm.client import chat

from . import facts as F
from .config import ACTION_TYPES, CURATIVE_PATTERN, GENERIC_ACTIONS, WRITE_RETRIES, WRITER_ROLE

# Phénomènes surveillés par le validateur : un texte ne peut en citer un que s'il figure dans les faits.
# Les facteurs du cadrage (reconnus par leurs mots-clés), plus ce que décrit un profil ou un contexte
# sans être mesuré : âge, environnement. Interdit si absent des faits (une justification doit
# s'appuyer sur les mesures, pas recopier la description du profil ou du contexte).
GENERIC_CONCEPTS = {
    "usure / vieillissement": r"usure|vieilli",
    "âge du site": r"\b\d+\s*ans\b|âgé|récent|ancien|jeunesse|vieux",
    "environnement du site": r"\bzones?\b|côti|continental|altitude|plaine|agricole|océan|littoral|maritime",
}


def concept_patterns() -> dict[str, str]:
    return {**{name: rf"\b(?:{pattern})" for name, pattern in current().concepts().items()}, **GENERIC_CONCEPTS}
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
    return {name for name, pattern in concept_patterns().items() if re.search(pattern, text, re.IGNORECASE)}


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
    """Mesure préventive type du facteur (proposée au cadrage), utilisée si le LLM échoue."""
    if current().measure(factor):
        kind, mesure, mecanisme = current().measure(factor)
        return {"cause": cause, "type": kind, "mesure": mesure + ".", "mecanisme": mecanisme + "."}
    return {"cause": cause, "type": "procédure", "mesure": f"{TO_DEFINE} : analyser les incidents « {cause} » pour définir "
            "une mesure préventive.", "mecanisme": "aucune mesure préventive type n'existe pour ce facteur."}


def site_texts(site: str, asset_type: str, profile: str | None, incidents, indicators: dict, journal: list[dict]) -> dict:
    """Justification du profil (ou, sans profils au cadrage, ce qui distingue le site), causes
    dominantes et une mesure préventive par cause prioritaire."""
    g = incidents[incidents["site"] == site]
    points = F.distinctive_points(indicators, site)
    dominant = F.dominant_causes(g)
    priority = F.priority_causes(g)
    factors = {p["cause"]: F.dominant_factor(incidents, p["cause"]) for p in priority}

    by_elimination = profile is not None and not F.expectations_met(indicators, site, asset_type, profile)
    profile_facts = "\n".join(f"- {F.point_text(p)}" for p in points) or "- aucun critère ne distingue nettement ce site"
    if by_elimination:
        profile_facts += ("\n- Profil attribué par élimination : aucun critère mesuré ne correspond nettement à ce "
                          "profil ; les autres sites correspondent mieux aux autres profils.")
    dominant_facts = "\n".join(f"- {d['cause']} : {d['incidents']} incidents, {d['part_incidents_pct']} % des incidents "
                               f"du site" for d in dominant)
    priority_facts = "\n".join(f"- {p['cause']} (facteur : {factors[p['cause']] or 'non précisé'}) : {p['incidents']} "
                               f"incidents, coût total {F.euros(p['cout_total'])}" for p in priority)
    profile_line = (f" Profil attribué : {profile} ({current().profiles(asset_type)[profile]})." if profile else "")
    facts_text = (f"Site : {site} ({current().label(asset_type)}).{profile_line}\n"
                  f"Points distinctifs du site :\n{profile_facts}\n"
                  f"Causes connues les plus fréquentes :\n{dominant_facts}\n"
                  f"Causes connues les plus coûteuses, hors arrêts de sécurité nominaux (à traiter dans le plan) :\n"
                  f"{priority_facts}")
    task = ("Rédige en JSON :\n"
            + ("- justification_profil : 2 phrases qui expliquent pourquoi ce site correspond au profil, en citant les "
               "points distinctifs et leurs chiffres. Ne décris pas le profil lui-même et n'évoque aucun phénomène absent "
               "des points distinctifs.\n" if profile else
               "- justification_profil : 2 phrases qui disent ce qui distingue ce site des autres sites du même segment, "
               "en citant les points distinctifs et leurs chiffres (ou, s'il n'y en a aucun, qu'aucun critère ne le "
               "distingue nettement). N'évoque aucun phénomène absent des points distinctifs.\n")
            +
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
            lead = (f"Profil {profile} attribué par élimination (aucun critère mesuré ne lui correspond nettement)"
                    if by_elimination else f"Profil {profile} retenu" if profile else
                    "Ce qui distingue ce site" if points else
                    "Aucun critère ne distingue nettement ce site des autres sites du segment")
            out[field] = lead + (" ; points distinctifs : " + " ; ".join(F.point_text(p) for p in points) if points else "") + "."
        elif field == "causes_dominantes":
            out[field] = "Causes connues les plus fréquentes : " + ", ".join(
                f"{d['cause']} ({d['incidents']} incidents, {d['part_incidents_pct']} % des incidents du site)" for d in dominant) + "."
        elif field.startswith("action:"):
            cause = field.split(":", 1)[1]
            out["actions"] = [a for a in out["actions"] if a["cause"] != cause] + [catalogue_measure(cause, factors[cause])]
        return out

    out = write(f"Rédaction {site}", task, facts_text, schema, check, fallback, journal)
    actions = {a["cause"]: a for a in out["actions"]}
    plan = [{"cause": p["cause"], "facteur": factors[p["cause"]],
             "consequence_principale": F.main_consequence(g, p["cause"]), "type": actions.get(p["cause"], {}).get("type", ""),
             "action": actions.get(p["cause"], {}).get("mesure", ""), "mecanisme": actions.get(p["cause"], {}).get("mecanisme", ""),
             "priorite": p["priorite"], "incidents": p["incidents"], "cout_total": p["cout_total"],
             "part_cout_pct": p["part_cout_pct"]} for p in priority]
    return {"site": site, "profil": profile, "points_distinctifs": [F.point_text(p) for p in points],
            "justification_profil": out["justification_profil"], "causes_dominantes": out["causes_dominantes"],
            "causes_dominantes_chiffres": dominant, "origine_causes_dominantes": F.cause_origin(g, [d["cause"] for d in dominant]),
            "plan": plan, "par_elimination": by_elimination,
            "faits": {"justification_profil": profile_facts, "causes_dominantes": dominant_facts, "plan": priority_facts}}


def synthesis(incidents, site_reports: list[dict], fleet, journal: list[dict],
              chains: dict[str, dict] | None = None) -> tuple[str, str]:
    """Synthèse pour un responsable : chiffres clés, chaînes cause -> conséquence prioritaires (heatmap)
    et actions prioritaires de l'ensemble des sites."""
    by_site = incidents.groupby("site")["estimated_cost_eur"].sum().sort_values(ascending=False)
    actions = {(r["site"], p["cause"]): p["action"] for r in site_reports for p in r["plan"]}
    structural = F.structural_losses(incidents)
    top = fleet.head(3)
    facts_text = "\n".join([
        f"- {len(incidents)} incidents, coût total {F.euros(incidents['estimated_cost_eur'].sum())}.",
        *(f"- {current().label(a)} : {F.unknown_share(incidents[incidents['asset_type'] == a])} % des incidents sans cause précisée "
          "dans le rapport." for a in sorted(incidents["asset_type"].unique())),
        f"- Site le plus coûteux : {by_site.index[0]} ({F.euros(by_site.iloc[0])}).",
        f"- Arrêts de sécurité nominaux (fonctionnement normal des protections, pas des pannes) : "
        f"{int(structural['arrets'].sum())} incidents, {F.euros(structural['cout_par_an'].sum())} par an de perte structurelle.",
        *(f"- Priorité {i} de l'ensemble des sites : {r.cause} sur {r.site} ({r.incidents} incidents, {F.euros(r.cout_par_an)} par an) ; "
          f"mesure : {actions.get((r.site, r.cause), 'à définir')}" for i, r in enumerate(top.itertuples(), 1)),
        *(f"- Chaîne prioritaire ({current().label(a)}) : {c['cause']} -> {c['consequence']} ({c['incidents']} incidents, "
          f"{F.euros(c['cout_total'])})" for a, d in (chains or {}).items() for c in d["chaines"][:2]),
    ])
    task = ("Rédige en JSON une synthèse de 4 à 7 phrases pour un responsable : coût total, part des causes non "
            "précisées, site le plus coûteux, arrêts de sécurité nominaux (perte structurelle, pas des pannes), les "
            "chaînes cause -> conséquence prioritaires, puis les 3 mesures prioritaires de l'ensemble des sites avec leur coût par an.")
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


def cause_sheets(incidents, journal: list[dict]) -> list[dict]:
    """Fiche diagnostic de chaque thème de cause connue (idée de la branche octave), avec sa criticité
    Pareto : description, conséquences typiques (heatmap) et mesure préventive avec son mécanisme.
    Mêmes règles que le reste du rapport : faits calculés, reformulés par le LLM, validés."""
    sheets = []
    for asset_type in sorted(incidents["asset_type"].unique()):
        sub = incidents[incidents["asset_type"] == asset_type]
        chains = F.chain_stats(sub)
        for row in F.cause_criticality(sub).itertuples():
            g = sub[sub["theme_cause"] == row.cause]
            cons = chains[chains["cause"] == row.cause].head(3)
            sites = (100 * g["site"].value_counts(normalize=True)).round(1)
            cons_facts = "\n".join(f"- {c.consequence} : {c.incidents} incidents, coût total {F.euros(c.cout_total)}"
                                   for c in cons.itertuples()) or "- aucune conséquence classée"
            facts_text = (f"Cause : {row.cause} ({current().label(asset_type)}). Facteur dominant : {row.facteur or 'non précisé'}.\n"
                          f"{row.incidents} incidents, coût total {F.euros(row.cout_total)}, coût moyen "
                          f"{F.euros(row.cout_moyen)}, {row.part_cout_pct} % du coût des causes connues.\n"
                          f"Sites : " + ", ".join(f"{s} {v} %" for s, v in sites.items()) + ".\n"
                          f"Maintenance suspectée (hypothèse) : {row.maintenance_suspectee_pct} % des incidents.\n"
                          f"Exemples de causes extraites : " + " / ".join(g["cause_racine"].dropna().head(3)) + "\n"
                          f"Conséquences les plus coûteuses :\n{cons_facts}")
            task = ("Rédige en JSON la fiche de cette cause pour une équipe de maintenance :\n"
                    "- description : 1 à 2 phrases sur ce qui se passe et pourquoi, avec les mots des faits.\n"
                    "- consequences : 1 phrase qui reformule les conséquences les plus coûteuses, avec leurs chiffres.\n"
                    "- type, mesure, mecanisme : une MESURE PRÉVENTIVE ou ORGANISATIONNELLE (quoi, sur quels équipements, à "
                    "quelle fréquence), pas une réparation ponctuelle, et son mécanisme (pourquoi elle réduit cette cause).")
            schema = {"type": "object", "additionalProperties": False,
                      "required": ["description", "consequences", "type", "mesure", "mecanisme"],
                      "properties": {"description": {"type": "string"}, "consequences": {"type": "string"},
                                     "type": {"type": "string", "enum": ACTION_TYPES},
                                     "mesure": {"type": "string"}, "mecanisme": {"type": "string"}}}

            def check(out: dict, cause=row.cause, factor=row.facteur, facts=facts_text, cf=cons_facts) -> dict:
                return {"description": text_errors(out["description"], facts),
                        "consequences": text_errors(out["consequences"], cf),
                        "mesure": measure_errors(out, cause, factor)}

            def fallback(field: str, out: dict, cause=row.cause, factor=row.facteur, cons=cons, n=row.incidents) -> dict:
                out = dict(out)
                if field == "description":
                    out[field] = f"{cause} : {n} incidents" + (f", facteur dominant {factor}." if factor else ".")
                elif field == "consequences":
                    out[field] = "Conséquences les plus coûteuses : " + (", ".join(
                        f"{c.consequence} ({c.incidents} incidents, {F.euros(c.cout_total)})" for c in cons.itertuples())
                        or "aucune conséquence classée") + "."
                else:
                    m = catalogue_measure(cause, factor)
                    out.update(type=m["type"], mesure=m["mesure"], mecanisme=m["mecanisme"])
                return out

            out = write(f"Fiche {asset_type} / {row.cause}", task, facts_text, schema, check, fallback, journal)
            sheets.append({"asset_type": asset_type, "theme_id": row.theme_id, "cause": row.cause, "rang": row.rang,
                           "criticite": row.criticite, "facteur": row.facteur, "incidents": row.incidents,
                           "cout_total": row.cout_total, "cout_moyen": row.cout_moyen, "arret_total_h": row.arret_total_h,
                           "part_cout_pct": row.part_cout_pct, "cumul_cout_pct": row.cumul_cout_pct,
                           "maintenance_suspectee_pct": row.maintenance_suspectee_pct,
                           "consequence_principale": row.consequence_principale,
                           "description": out["description"], "consequences": out["consequences"], "type": out["type"],
                           "mesure": out["mesure"], "mecanisme": out["mecanisme"], "faits": facts_text})
    return sheets
