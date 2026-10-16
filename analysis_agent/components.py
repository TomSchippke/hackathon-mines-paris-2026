"""Normalisation des composants bruts d'un segment (D8).

Le composant est saisi en texte libre à l'extraction (composant_brut). Un seul appel LLM regroupe les
valeurs distinctes d'un segment en composants normalisés ; le code vérifie que chaque valeur est rangée.
"""

from __future__ import annotations

import json
import math

from analysis_agent.config import MAX_COMPONENT_GROUPS
from llm.client import chat

# Clé spéciale du dictionnaire renvoyé : valeur = JSON (liste des valeurs brutes non rangées par le LLM).
NON_RANGEES = "__non_rangees__"

SYSTEM_PROMPT = """Tu normalises des noms de composants d'équipements industriels, saisis en texte libre
dans des rapports d'incident. Tu regroupes les noms bruts qui désignent le même type de composant.
Règles :
- au plus {max_groups} groupes ; chaque groupe a un nom court en français, en minuscules ;
- chaque valeur brute fournie figure dans EXACTEMENT un groupe ;
- n'utilise que les valeurs brutes fournies, recopiées à l'identique ; n'en invente aucune."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "composants",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "groupes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nom": {"type": "string"},
                            "valeurs": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["nom", "valeurs"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["groupes"],
            "additionalProperties": False,
        },
    },
}


def _texte(value) -> str:
    """Valeur brute en texte sans espaces de bord ; None, NaN et chaînes vides donnent ''."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _groupes(content: str | None) -> list[tuple[str, list]]:
    """Lit la réponse : liste de (nom, valeurs). Réponse invalide : aucun groupe (ne lève jamais)."""
    try:
        groupes = json.loads(content or "")["groupes"]
    except (ValueError, TypeError, KeyError):
        return []
    if not isinstance(groupes, list):
        return []
    result = []
    for groupe in groupes:
        if isinstance(groupe, dict) and isinstance(groupe.get("valeurs"), list):
            result.append((_texte(groupe.get("nom")).lower(), groupe["valeurs"]))
    return result


def normalize_components(raw_values: list[str], context: str) -> dict[str, str]:
    """Regroupe les composants bruts distincts d'un segment en au plus MAX_COMPONENT_GROUPS
    composants normalisés. JSON strict : {"groupes": [{"nom", "valeurs": [str]}]}.

    Retour : {valeur brute (sans espaces de bord): nom du groupe (minuscules)}.
    - Une valeur placée dans plusieurs groupes prend le premier.
    - Une valeur renvoyée par le LLM mais absente de l'entrée est ignorée ; un groupe sans nom est ignoré.
    - Une valeur brute oubliée par le LLM (ou réponse invalide) est rangée sous son propre texte en
      minuscules, et listée dans la clé spéciale NON_RANGEES ("__non_rangees__") : sa valeur est une
      chaîne JSON (liste des valeurs brutes concernées, ensure_ascii=False).
    - Le nombre maximal de groupes est seulement demandé dans le prompt : le code ne le vérifie pas et
      garde tous les groupes renvoyés.
    Valeurs vides ou nulles exclues. Aucune valeur : {} sans appel.
    """
    values = sorted({_texte(v) for v in raw_values} - {""})
    if not values:
        return {}
    known = set(values)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_groups=MAX_COMPONENT_GROUPS)},
        {"role": "user", "content": f"{context}\n\nValeurs brutes :\n" + "\n".join(f"- {v}" for v in values)},
    ]
    answer = chat(messages, role="extraction", response_format=RESPONSE_FORMAT)

    result: dict[str, str] = {}
    for nom, valeurs in _groupes(answer.get("content")):
        if not nom:
            continue
        for valeur in valeurs:
            valeur = _texte(valeur)
            if valeur in known:
                result.setdefault(valeur, nom)

    unplaced = [v for v in values if v not in result]
    for valeur in unplaced:
        result[valeur] = valeur.lower()
    if unplaced:
        result[NON_RANGEES] = json.dumps(unplaced, ensure_ascii=False)
    return result
