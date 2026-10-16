"""Affectation de chaque phrase d'un segment à un thème découvert (D3).

Le clustering propose des groupes, les thèmes les nomment ; ce module demande ensuite au LLM de ranger
CHAQUE phrase dans un thème, ou de répondre « aucun ». Une réponse invalide ne fait jamais planter le
pipeline : la phrase concernée reçoit None.
"""

from __future__ import annotations

import json

from analysis_agent.config import ASSIGN_BATCH, NONE_LABEL
from llm.client import chat

# Nombre maximal d'exemples de phrases montrés par thème dans le prompt.
MAX_EXAMPLES = 3

SYSTEM_PROMPT = """Tu classes des phrases d'un rapport d'incident de maintenance dans des thèmes.
Pour chaque phrase, choisis le thème qui décrit LE MÊME PHÉNOMÈNE (même cause ou même conséquence,
pas seulement le même domaine ou le même équipement).
Si aucun thème ne décrit ce phénomène, réponds « {none} » : ne force jamais une phrase dans le thème
le plus proche.
Réponds en JSON, une affectation par phrase, avec l'index de la phrase et l'identifiant du thème."""


def _schema(theme_ids: list[str]) -> dict:
    """Schéma JSON strict de la réponse : un thème (ou NONE_LABEL) par index de phrase."""
    return {
        "type": "object",
        "properties": {
            "affectations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "theme": {"type": "string", "enum": theme_ids + [NONE_LABEL]},
                    },
                    "required": ["index", "theme"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["affectations"],
        "additionalProperties": False,
    }


def _user_message(batch: list[str], themes: list[dict], context: str) -> str:
    """Contexte, thèmes numérotés (id, nom, description, exemples), puis phrases numérotées à partir de 0."""
    lines = [context, "", "Thèmes :"]
    for number, theme in enumerate(themes, start=1):
        examples = " ; ".join(f"« {e} »" for e in theme.get("examples", [])[:MAX_EXAMPLES])
        lines.append(f"{number}. [{theme['id']}] {theme['name']} : {theme['description']}")
        if examples:
            lines.append(f"   Exemples : {examples}")
    lines += ["", "Phrases à classer (index local au lot) :"]
    lines += [f"{index}. {phrase}" for index, phrase in enumerate(batch)]
    return "\n".join(lines)


def _parse_affectations(content: str | None, batch_size: int, theme_ids: set[str]) -> list[str | None]:
    """Lit la réponse d'un lot. Index absent, hors bornes ou en double (on garde le premier), thème
    inconnu, JSON invalide : None pour les phrases concernées. Ne lève jamais d'exception."""
    result: list[str | None] = [None] * batch_size
    try:
        affectations = json.loads(content or "")["affectations"]
    except (ValueError, TypeError, KeyError):
        return result
    if not isinstance(affectations, list):
        return result
    seen: set[int] = set()
    for item in affectations:
        if not isinstance(item, dict):
            continue
        index, theme = item.get("index"), item.get("theme")
        # bool est un sous-type d'int en Python : on l'écarte explicitement
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < batch_size:
            continue
        if index in seen:
            continue
        seen.add(index)
        if isinstance(theme, str) and theme in theme_ids:
            result[index] = theme
    return result


def assign_phrases(phrases: list[str], themes: list[dict], context: str) -> list[str | None]:
    """themes: [{"id", "name", "description", "examples": [str]}]. Retourne, pour chaque phrase,
    l'id du thème choisi ou None (NONE_LABEL). Appels par lots de ASSIGN_BATCH, JSON strict :
    {"affectations": [{"index": int, "theme": enum(ids + [NONE_LABEL])}]}. Une phrase absente ou
    un index invalide dans la réponse -> None. Aucune phrase ou aucun thème : pas d'appel."""
    if not phrases:
        return []
    if not themes:
        return [None] * len(phrases)
    theme_ids = [theme["id"] for theme in themes]
    response_format = {"type": "json_schema",
                       "json_schema": {"name": "affectation", "strict": True, "schema": _schema(theme_ids)}}
    system = {"role": "system", "content": SYSTEM_PROMPT.format(none=NONE_LABEL)}
    result: list[str | None] = []
    for start in range(0, len(phrases), ASSIGN_BATCH):
        batch = phrases[start:start + ASSIGN_BATCH]
        messages = [system, {"role": "user", "content": _user_message(batch, themes, context)}]
        answer = chat(messages, role="extraction", response_format=response_format)
        result += _parse_affectations(answer.get("content"), len(batch), set(theme_ids))
    return result
