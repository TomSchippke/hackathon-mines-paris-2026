"""Boucle d'agent générique, utilisée par l'agent d'analyse et par l'agent vérificateur.

À chaque tour :
  1. le LLM reçoit la conversation et la liste des outils, et choisit un ou plusieurs outils ;
  2. le code exécute chaque outil (fonction Python) et renvoie le résultat au LLM ;
  3. on recommence, jusqu'à ce que l'outil final ait réussi ou que le nombre de tours soit atteint.

Tout est consigné dans un journal (liste d'événements) : raisonnement du LLM, outils appelés,
arguments, résultats. Le journal est sauvegardé en JSON (relu par l'agent vérificateur) et en
Markdown (lisible par un humain).

Les réponses du LLM sont en cache (llm.client.chat) : relancer avec les mêmes données rejoue
exactement la même session, sans aucun appel.

Économie de tokens : à chaque tour, toute la conversation est renvoyée au modèle. Pour qu'elle ne
grossisse pas sans fin (et ne dépasse pas le quota de tokens par minute d'Azure), seuls les
KEEP_FULL_RESULTS derniers résultats d'outils sont envoyés en entier ; les plus anciens sont
abrégés (l'agent peut rappeler l'outil s'il en a besoin). Le journal, lui, garde tout.
"""

from __future__ import annotations

import json
from typing import Callable

from .client import chat

NUDGE = "Continue en appelant les outils. Termine en appelant l'outil final."
KEEP_FULL_RESULTS = 6        # derniers résultats d'outils envoyés en entier au modèle
MAX_SAME_REFUSAL = 3         # un même appel refusé 3 fois : la session s'arrête (l'agent tourne en rond)
MAX_SAME_CALL = 3            # un même appel (même réussi) répété 3 fois : idem
OLD_RESULT_CHARS = 400       # taille des résultats plus anciens, abrégés


def compact(messages: list[dict]) -> list[dict]:
    """Copie de la conversation où les anciens résultats d'outils sont abrégés."""
    tool_positions = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool_positions[:-KEEP_FULL_RESULTS])
    return [{**m, "content": m["content"][:OLD_RESULT_CHARS] + " [...] (résultat ancien abrégé : rappeler l'outil "
             "si besoin)"} if i in old and len(m["content"]) > OLD_RESULT_CHARS else m
            for i, m in enumerate(messages)]


class ToolError(Exception):
    """Appel d'outil refusé (argument hors bornes, étape préalable manquante...). Le message est
    renvoyé à l'agent, qui peut corriger son appel."""


def run_agent(system_prompt: str, task: str, tools: list[dict], execute: Callable[[str, dict], dict],
              is_done: Callable[[], bool], max_turns: int, role: str = "analysis",
              nudge: Callable[[], str] | None = None) -> tuple[list[dict], bool]:
    """Fait travailler un agent. Renvoie (journal, terminé correctement ?).

    `execute(nom, arguments)` exécute un outil et renvoie un dict sérialisable en JSON.
    `is_done()` dit si l'objectif est atteint. `role` : usage du modèle (voir llm.client.deployment).
    `nudge()` : relance envoyée quand l'agent répond par du texte au lieu d'agir (par exemple la
    liste de ce qui reste à faire) ; par défaut, NUDGE.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    journal: list[dict] = []
    refusals: dict[str, int] = {}  # appel exact (outil + arguments) -> nombre de refus
    calls: dict[str, int] = {}     # appel exact -> nombre d'appels (réussis ou non)
    for turn in range(1, max_turns + 1):
        answer = chat(compact(messages), role=role, tools=tools)
        assistant = {"role": "assistant", "content": answer["content"]}
        if answer["tool_calls"]:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in answer["tool_calls"]]
        messages.append(assistant)
        if answer["content"]:
            journal.append({"tour": turn, "type": "raisonnement", "texte": answer["content"]})
        if not answer["tool_calls"]:
            messages.append({"role": "user", "content": nudge() if nudge else NUDGE})
            continue

        for call in answer["tool_calls"]:
            try:
                arguments = json.loads(call["arguments"] or "{}")
                result = execute(call["name"], arguments)
            except json.JSONDecodeError:
                arguments, result = call["arguments"], {"ok": False, "refus": "arguments JSON invalides"}
            signature = call["name"] + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            calls[signature] = calls.get(signature, 0) + 1
            if not result.get("ok"):
                refusals[signature] = refusals.get(signature, 0) + 1
                if refusals[signature] > 1:
                    result["attention"] = (f"cet appel exact a déjà été refusé {refusals[signature]} fois pour la même "
                                           "raison : ne le répète pas, change d'arguments ou d'outil.")
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
            journal.append({"tour": turn, "type": "outil", "outil": call["name"], "arguments": arguments,
                            "resultat": result})
        if is_done():
            journal.append({"tour": turn, "type": "fin", "texte": f"terminé en {turn} tours"})
            return journal, True
        if refusals and max(refusals.values()) >= MAX_SAME_REFUSAL:
            journal.append({"tour": turn, "type": "fin", "texte": f"arrêté au tour {turn} : l'agent répète un appel "
                                                                  f"refusé {MAX_SAME_REFUSAL} fois (il tourne en rond)"})
            return journal, False
        if max(calls.values(), default=0) >= MAX_SAME_CALL:
            journal.append({"tour": turn, "type": "fin", "texte": f"arrêté au tour {turn} : l'agent répète le même appel "
                                                                  f"{MAX_SAME_CALL} fois (il tourne en rond)"})
            return journal, False
    journal.append({"tour": max_turns, "type": "fin", "texte": f"arrêté après {max_turns} tours sans terminer"})
    return journal, False


def dispatch(tools: dict[str, Callable], name: str, arguments: dict) -> dict:
    """Appelle l'outil `name`. Un refus (ToolError) ou une erreur est renvoyé à l'agent, jamais levé."""
    if name not in tools:
        return {"ok": False, "refus": f"outil inconnu : {name}"}
    try:
        return {"ok": True, **tools[name](**arguments)}
    except ToolError as e:
        return {"ok": False, "refus": str(e)}
    except TypeError as e:  # argument manquant ou inattendu
        return {"ok": False, "refus": f"arguments invalides : {e}"}
    except Exception as e:  # erreur de calcul : signalée à l'agent, n'arrête pas la session
        return {"ok": False, "erreur": f"{type(e).__name__}: {e}"}


def journal_markdown(journal: list[dict], title: str, max_chars: int = 800) -> str:
    """Version lisible du journal."""
    lines = [f"# {title}\n"]
    for event in journal:
        if event["type"] == "raisonnement":
            lines.append(f"\n**Tour {event['tour']} — raisonnement :** {event['texte']}\n")
        elif event["type"] == "outil":
            result = json.dumps(event["resultat"], ensure_ascii=False, default=str)
            if len(result) > max_chars:
                result = result[:max_chars] + " [...]"
            status = "✅" if event["resultat"].get("ok") else "⛔"
            args = json.dumps(event["arguments"], ensure_ascii=False)
            lines.append(f"- Tour {event['tour']} {status} `{event['outil']}({args})`\n  -> {result}")
        elif event["type"] == "phase":
            lines.append(f"\n## Phase : {event['texte']}\n")
        elif event["type"] == "automatique":
            lines.append(f"\n> ⚙️ **Reprise automatique par le code :** {event['texte']}\n")
        else:
            lines.append(f"\n**{event['texte']}**\n")
    return "\n".join(lines) + "\n"
