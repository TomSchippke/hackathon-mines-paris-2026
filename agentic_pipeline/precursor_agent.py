"""Contrôle de plausibilité (LLM) des paires précurseurs significatives."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .llm_client import call_llm_json

CHECKED_KINDS = {"precursor", "escalation"}
MAX_LEN = 500

SYSTEM_PRECURSOR_PROMPT = """Tu es un ingénieur fiabilité expert en maintenance d'équipements et de machines industriels.
On te présente une association statistique : des incidents de type A (antécédent) sont suivis, sur le MÊME équipement, par des incidents de type B (conséquent) dans une fenêtre de temps donnée.

Ta tâche : juger si cette association est un vrai signal précurseur.
- Propose un mécanisme causal ou de dégradation physiquement plausible reliant A à B sur le même équipement.
- Propose une règle de surveillance concrète, de la forme : "si ≥ k incidents A en W jours sur un équipement → action".
- Tu as EXPLICITEMENT le droit de répondre plausible=false si aucun mécanisme physique crédible n'existe (simple corrélation, coïncidence).
- Si une cause commune externe aux deux incidents peut provoquer à la fois A et B, ce n'est PAS un lien précurseur : réponds plausible=false et commence le mécanisme par "cause commune externe: ".
- Si plausible=false, monitoring_rule doit être une chaîne vide.

Réponds uniquement en JSON : {"plausible": true|false, "mechanism": "...", "monitoring_rule": "..."}"""


def _card_block(label: str, cid: str, card: dict | None) -> str:
    card = card or {}
    title = card.get("title") or card.get("fine_title") or card.get("problem_title") or cid
    lines = [f"{label} (id: {cid})", f"- Titre : {title}"]
    lines.append(f"- Mode de défaillance : {card.get('failure_mode') or 'non renseigné'}")
    texts = list(card.get("representative_texts") or [])[:2]
    for i, t in enumerate(texts, 1):
        lines.append(f"- Exemple {i} : {str(t)[:400]}")
    return "\n".join(lines)


def _build_prompt(row: pd.Series, cards: dict[str, dict]) -> str:
    a, b = str(row["antecedent"]), str(row["consequent"])
    scope = row.get("scope_cluster", "")
    scope_txt = f"\nContexte : escalade au sein du cluster {scope}." if isinstance(scope, str) and scope else ""
    return (
        f"{_card_block('Antécédent A', a, cards.get(a))}\n\n"
        f"{_card_block('Conséquent B', b, cards.get(b))}\n\n"
        f"Statistiques :{scope_txt}\n"
        f"- Fenêtre : {int(row['window_days'])} jours\n"
        f"- Lift : {float(row['lift']):.2f}\n"
        f"- Nombre de B précédés d'un A (n_hits) : {int(row['n_hits'])}\n"
        f"- Équipements distincts : {int(row['n_equipments'])}\n"
        f"- Nombre total de B (n_consequent) : {int(row['n_consequent'])}\n\n"
        "A précède-t-il B de façon physiquement plausible sur un même équipement ?"
    )


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "vrai", "oui", "yes"):
            return True
        if s in ("false", "faux", "non", "no"):
            return False
    return None


def _parse(resp: Any) -> tuple[bool, str, str]:
    if not isinstance(resp, dict):
        return False, "", ""
    plausible = _to_bool(resp.get("plausible"))
    mech, rule = resp.get("mechanism"), resp.get("monitoring_rule")
    if plausible is None or not isinstance(mech, str):
        return False, "", ""
    rule = rule.strip()[:MAX_LEN] if isinstance(rule, str) else ""
    if not plausible:
        rule = ""
    return plausible, mech.strip()[:MAX_LEN], rule


def check_plausibility(
    pairs: pd.DataFrame,
    cards: dict[str, dict],
    llm: Callable = call_llm_json,
) -> pd.DataFrame:
    """Remplit plausible / mechanism / monitoring_rule (copie, entrée non modifiée)."""
    out = pairs.copy()
    plausible = pd.Series(False, index=out.index, dtype=bool)
    mechanism = pd.Series("", index=out.index, dtype=object)
    rule = pd.Series("", index=out.index, dtype=object)

    if len(out):
        mask = (out["significant"].astype(bool)) & out["kind"].isin(CHECKED_KINDS)
        for idx in out.index[mask.to_numpy()]:
            try:
                resp = llm(
                    system_prompt=SYSTEM_PRECURSOR_PROMPT,
                    user_prompt=_build_prompt(out.loc[idx], cards),
                    cache_tag="precursor_check",
                )
                p, m, r = _parse(resp)
            except Exception:
                p, m, r = False, "", ""
            plausible[idx], mechanism[idx], rule[idx] = p, m, r

    out["plausible"] = plausible
    out["mechanism"] = mechanism
    out["monitoring_rule"] = rule
    return out
