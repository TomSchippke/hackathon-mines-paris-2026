"""Juge : relit chaque texte du rapport avec une grille précise et propose une correction.

Un texte à la fois (justification d'un profil, causes dominantes, chaque mesure du plan, mesure de
chaque fiche de cause, synthèse) :
  1. le juge (LLM, usage « review ») reçoit le texte, les faits sur lesquels il repose et une grille
     de questions ; il renvoie les problèmes trouvés et un texte corrigé ;
  2. le CODE tranche : la correction n'est acceptée que si elle passe le même validateur que la
     rédaction (chiffres et phénomènes présents dans les faits, mesure préventive...). Sinon le texte
     d'origine est gardé. Un juge trop zélé ne peut donc pas dégrader le rapport.
Chaque décision est consignée dans le journal.
"""

from __future__ import annotations

import json

from llm.client import chat

from .config import JUDGE_ROLE, PREVENTIVE_CATALOGUE
from .writing import measure_errors, numbers, text_errors

GRID_JUSTIFICATION = """1. Chaque affirmation (chiffre, phénomène) figure-t-elle dans les faits ?
2. Y a-t-il une extrapolation non mesurée (déduire la qualité de la maintenance, l'âge ou la localisation, \
interpréter un coût) ?
3. Si les faits indiquent « attribué par élimination », le texte le dit-il clairement ?
4. Coquilles, répétitions, formulation maladroite ?"""

GRID_CAUSES = """1. Les causes citées sont-elles les plus fréquentes des faits, dans le bon ordre ?
2. Les chiffres sont-ils exacts ?
3. Coquilles, formulation maladroite ?"""

GRID_MEASURE = """1. La mesure nomme-t-elle un équipement précis ?
2. Vise-t-elle un défaut précis de la cause (pas « détecter l'usure » en général) ?
3. Indique-t-elle une fréquence ou un déclencheur (après chaque tempête, avant l'été...) ?
4. Réduit-elle vraiment la cause, ou se contente-t-elle de la surveiller ?
5. Le mécanisme est-il concret, et non circulaire (pas « une inspection régulière permet de détecter ») ?
La correction doit rester une mesure PRÉVENTIVE ou organisationnelle à l'échelle du site, jamais une \
réparation ponctuelle."""

GRID_SYNTHESIS = """1. Chaque mesure prioritaire citée a-t-elle son site, sa cause et son coût par an ?
2. Les priorités sont-elles formulées sans redondance ?
3. Les chiffres sont-ils exacts, sans coquille (par exemple « près de » devant un montant exact) ?
4. Les arrêts de sécurité nominaux sont-ils présentés comme une perte structurelle, pas comme une panne ?"""


def ask_judge(kind: str, text: str, facts: str, grid: str, fields: dict) -> dict:
    """Un passage du juge. `fields` : champs de la correction attendue (nom -> description)."""
    schema = {"type": "object", "additionalProperties": False, "required": ["problemes", *fields],
              "properties": {"problemes": {"type": "array", "items": {"type": "string"}},
                             **{k: {"type": "string", "description": v} for k, v in fields.items()}}}
    answer = chat([{"role": "system", "content": JUDGE_ROLE},
                   {"role": "user", "content": f"TEXTE ({kind}) :\n{text}\n\nFAITS :\n{facts}\n\nGRILLE :\n{grid}"}],
                  role="review", response_format={"type": "json_schema",
                                                  "json_schema": {"name": "jugement", "strict": True, "schema": schema}})
    return json.loads(answer["content"])


def decide(name: str, verdict: dict, errors: list[str], changed: bool, journal: list[dict], stats: dict) -> bool:
    """Accepte la correction du juge si elle change le texte et passe le validateur ; consigne la décision."""
    problems = verdict["problemes"]
    if not problems or not changed:
        stats["inchanges"] += 1
        journal.append({"tour": 0, "type": "redaction", "texte": f"Juge — {name} : aucun problème"})
        return False
    if errors:
        stats["refuses"] += 1
        journal.append({"tour": 0, "type": "redaction", "texte": f"Juge — {name} : problèmes {problems} ; correction "
                        f"REFUSÉE par le validateur ({errors}), texte d'origine gardé"})
        return False
    stats["corriges"] += 1
    journal.append({"tour": 0, "type": "redaction", "texte": f"Juge — {name} : problèmes {problems} ; correction acceptée"})
    return True


def judge_report(report: dict, journal: list[dict]) -> dict:
    """Relit et corrige le rapport en place. Renvoie le décompte des décisions."""
    stats = {"corriges": 0, "refuses": 0, "inchanges": 0}
    text_field = {"correction": "le texte corrigé (le texte d'origine s'il n'y a rien à corriger)"}
    for s in report["sites"]:
        facts = s["faits"]
        for field, grid in (("justification_profil", GRID_JUSTIFICATION), ("causes_dominantes", GRID_CAUSES)):
            v = ask_judge(field, s[field], facts[field], grid, text_field)
            errors = text_errors(v["correction"], facts[field])
            if field == "justification_profil" and s["par_elimination"] and "élimination" not in v["correction"].lower():
                errors.append("la mention « par élimination » a disparu")
            if decide(f"{s['site']} / {field}", v, errors, v["correction"].strip() != s[field].strip(), journal, stats):
                s[field] = v["correction"]
        for p in s["plan"]:
            hint = PREVENTIVE_CATALOGUE.get(p["facteur"])
            plan_facts = (f"Cause : {p['cause']} (facteur : {p['facteur'] or 'non précisé'}) ; {p['incidents']} incidents.\n"
                          + (f"Exemple de mesure préventive pour ce facteur : {hint[1]} ({hint[2]})." if hint else ""))
            text = f"Mesure : {p['action']}\nMécanisme : {p['mecanisme']}"
            v = ask_judge("mesure du plan", text, plan_facts, GRID_MEASURE,
                          {"mesure": "la mesure corrigée", "mecanisme": "le mécanisme corrigé"})
            errors = measure_errors(v, p["cause"], p["facteur"])
            changed = (v["mesure"].strip(), v["mecanisme"].strip()) != (p["action"].strip(), p["mecanisme"].strip())
            if decide(f"{s['site']} / mesure « {p['cause']} »", v, errors, changed, journal, stats):
                p["action"], p["mecanisme"] = v["mesure"], v["mecanisme"]
    for f in report.get("fiches", []):
        hint = PREVENTIVE_CATALOGUE.get(f["facteur"])
        sheet_facts = f["faits"] + (f"\nExemple de mesure préventive pour ce facteur : {hint[1]} ({hint[2]})." if hint else "")
        v = ask_judge("mesure d'une fiche de cause", f"Mesure : {f['mesure']}\nMécanisme : {f['mecanisme']}", sheet_facts,
                      GRID_MEASURE, {"mesure": "la mesure corrigée", "mecanisme": "le mécanisme corrigé"})
        errors = measure_errors(v, f["cause"], f["facteur"])
        changed = (v["mesure"].strip(), v["mecanisme"].strip()) != (f["mesure"].strip(), f["mecanisme"].strip())
        if decide(f"fiche {f['asset_type']} / mesure « {f['cause']} »", v, errors, changed, journal, stats):
            f["mesure"], f["mecanisme"] = v["mesure"], v["mecanisme"]
    v = ask_judge("synthèse", report["synthese"], report["faits_synthese"], GRID_SYNTHESIS, text_field)
    errors = text_errors(v["correction"], report["faits_synthese"])
    if len([x for x in numbers(v["correction"]) if x > 10]) < 3:
        errors.append("moins de 3 chiffres des faits")
    if decide("synthèse", v, errors, v["correction"].strip() != report["synthese"].strip(), journal, stats):
        report["synthese"] = v["correction"]
    report["juge"] = stats
    print(f"  Juge : {stats['corriges']} textes corrigés, {stats['refuses']} corrections refusées par le validateur, "
          f"{stats['inchanges']} inchangés")
    return stats
