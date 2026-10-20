"""Synthèse du rapport : gabarit déterministe, ou une rédaction LLM validée par le code.

Une synthèse est un dict {"kind": "template"|"llm", "lines": [...], "note": str}. Chaque ligne :
{"text": str, "href": ancre de la ligne ou "", "refs": [identifiants d'action A1..An]}. Les
identifiants sont rendus en liens vers les actions du plan.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agentic_pipeline import llm_client
from agentic_pipeline.config import CACHE_DIR

from .formatting import NBSP, eur, pct

CACHE_TAG = "report_summary"
MIN_LLM_SENTENCES = 3
MAX_LLM_SENTENCES = 5
MAX_SENTENCE_CHARS = 400
# Identifiants recopiés dans le texte (« (actions : [A1, A2]) ») : retirés, ils sont rendus en liens
_INLINE_IDS = re.compile(r"\s*\((?:actions?|action_ids)?\s*:?\s*\[?\s*A\d+(?:\s*,\s*A\d+)*\s*\]?\s*\)", re.I)

SYSTEM_SUMMARY_PROMPT = """Tu rédiges la synthèse d'un rapport opérationnel de maintenance, lue en \
quelques minutes par un responsable d'exploitation. On te donne la liste des actions du plan, \
numérotées (A1, A2...), avec la voie de remédiation qu'elles traitent, leur portée, leurs cibles et le \
coût des incidents de cette voie.

Règles :
- 3 à 5 phrases courtes, en français, chacune appuyée sur une ou plusieurs actions de la liste ;
- chaque phrase indique dans "action_ids" les identifiants des actions sur lesquelles elle repose ;
- n'utilise que les chiffres, équipements, sites et actions de la liste ; n'invente rien ;
- commence par ce qui pèse le plus en coût ; signale les équipements dont la sévérité monte.

Réponds en JSON : {"sentences": [{"text": "...", "action_ids": ["A1"]}]}"""


# --------------------------------------------------------------------------- gabarit


def template_summary(ctx: dict) -> dict:
    """Synthèse déterministe : 3 voies les plus coûteuses, 1 à 2 escalades, groupes sans action."""
    lines: list[dict] = []
    for v in ctx["plan"][:3]:
        n = len(v["actions"])
        lines.append({
            "text": f"{v['title']}{NBSP}: {eur(v['total_cost'])} ({pct(v['cost_share'])} du coût flotte), "
                    f"{n} action{'s' if n > 1 else ''}",
            "href": f"#{v['anchor']}",
            "refs": [a["id"] for a in v["actions"]],
        })
    no_action = ctx["no_action"]
    n_groups = len(no_action["nominal"]) + (1 if no_action["isolated"]["n"] else 0)
    max_esc = 2 if n_groups == 0 else 1  # synthèse limitée à 5 lignes
    for e in ctx["escalation_actions"][:max_esc]:
        lines.append({
            "text": f"{e['equipment']} ({e['voie_title']}){NBSP}: sévérité {e['sequence']}, "
                    f"agir avant la prochaine occurrence",
            "href": f"#{e['action_id']}",
            "refs": [e["action_id"]],
        })
    if n_groups:
        parts = []
        if no_action["nominal"]:
            k = len(no_action["nominal"])
            parts.append(f"{k} fonctionnement{'s nominaux' if k > 1 else ' nominal'}")
        if no_action["isolated"]["n"]:
            parts.append(f"{no_action['isolated']['n']} incidents isolés")
        lines.append({
            "text": f"{n_groups} groupe{'s' if n_groups > 1 else ''} sans action requise ({', '.join(parts)}, "
                    f"{eur(no_action['total_cost'])})",
            "href": "#sans-action",
            "refs": [],
        })
    return {"kind": "template", "lines": lines, "note": ""}


# --------------------------------------------------------------------------- LLM


def _actions_prompt(ctx: dict) -> str:
    lines = ["Actions du plan :"]
    for v in ctx["plan"]:
        for a in v["actions"]:
            targets = ", ".join(t["name"] for t in a["targets"]) or "toute la flotte"
            esc = " ; sévérité en hausse sur " + ", ".join(
                f"{t['name']} ({t['esc']['sequence']})" for t in a["targets"] if t.get("esc")) \
                if any(t.get("esc") for t in a["targets"]) else ""
            lines.append(f"- {a['id']} | voie : {v['title']} (coût des incidents : "
                         f"{round(v['total_cost'])} EUR, {round(100 * v['cost_share'])} % du coût flotte, "
                         f"{v['frequency']} incidents) | action : {a['text']} | portée : {a['scope_label']} | "
                         f"cibles : {targets}{esc}")
    return "\n".join(lines)


def _validate_llm(result: Any, valid_ids: set[str]) -> list[dict]:
    """Garde les phrases non vides dont les action_ids sont non vides et tous connus."""
    sentences = result.get("sentences") if isinstance(result, dict) else None
    out = []
    for s in sentences if isinstance(sentences, list) else []:
        if not isinstance(s, dict) or not isinstance(s.get("text"), str) or not s["text"].strip():
            continue
        ids = s.get("action_ids")
        ids = [i.strip() for i in ids if isinstance(i, str)] if isinstance(ids, list) else []
        if not ids or any(i not in valid_ids for i in ids):
            continue
        text = _INLINE_IDS.sub("", s["text"]).strip()
        out.append({"text": text[:MAX_SENTENCE_CHARS], "href": "",
                    "refs": list(dict.fromkeys(ids))})
    return out[:MAX_LLM_SENTENCES]


def _cached_call(system_prompt: str, user_prompt: str) -> dict | None:
    """Lecture seule du cache de `call_llm_json` (aucun appel réseau) ; None si absent."""
    path = CACHE_DIR / CACHE_TAG / f"{llm_client._hash_request(system_prompt, user_prompt)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def llm_summary(ctx: dict, llm: Callable[..., Any] | None = None, cache_only: bool = False) -> dict:
    """Synthèse rédigée par le LLM (un appel), validée ; repli sur le gabarit si moins de
    MIN_LLM_SENTENCES phrases valides. `cache_only` : ne lit que le cache (aucun appel réseau)."""
    user_prompt = _actions_prompt(ctx)
    valid_ids = {a["id"] for v in ctx["plan"] for a in v["actions"]}
    if cache_only:
        result = _cached_call(SYSTEM_SUMMARY_PROMPT, user_prompt)
        if result is None:
            out = template_summary(ctx)
            out.update(kind="absent", note="Synthèse LLM non générée : lancer avec --summary llm.")
            return out
    else:
        llm = llm or llm_client.call_llm_json
        try:
            result = llm(system_prompt=SYSTEM_SUMMARY_PROMPT, user_prompt=user_prompt, cache_tag=CACHE_TAG)
        except Exception as e:  # réseau, quota, JSON invalide : repli sur le gabarit
            print(f"   [synthèse LLM] échec de l'appel ({e}) : repli sur le gabarit.")
            result = None
    lines = _validate_llm(result, valid_ids)
    n_raw = len(result.get("sentences") or []) if isinstance(result, dict) and isinstance(result.get("sentences"), list) else 0
    if len(lines) < MIN_LLM_SENTENCES:
        out = template_summary(ctx)
        out["note"] = (f"Synthèse LLM rejetée ({len(lines)} phrase(s) valide(s) sur {n_raw}, minimum "
                       f"{MIN_LLM_SENTENCES}) : repli sur le gabarit.")
        return out
    note = f"{len(lines)} phrase(s) gardée(s) sur {n_raw} proposée(s) : chaque phrase cite des actions existantes."
    return {"kind": "llm", "lines": lines, "note": note}
