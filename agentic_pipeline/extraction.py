"""Cause de chaque incident, extraite par le LLM et vérifiée par le code.

Un rapport mêle ce qui a déclenché l'incident et ce qu'il a provoqué ; regroupé tel quel, il donne
des clusters séparés par la gravité des dégâts plutôt que par la cause. Ici le LLM lit chaque
description et renvoie la cause racine, un extrait qui la justifie, la conséquence et le composant,
avec les mots du rapport et sans liste fermée. Le clustering porte ensuite sur la cause seule.

Le LLM ne voit que la description : ni le site, ni les identifiants, ni la sévérité ou le coût
(sinon la gravité entrerait dans la cause). L'extrait cité est cherché dans la description : s'il
n'y est pas, ou si le rapport ne dit pas la cause, l'incident est classé « cause non précisée ».

Des extractions déjà faites peuvent être réutilisées (`seed`) : elles ne sont prises que pour un
incident au texte identique, et leur extrait est vérifié de la même façon.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pandas as pd

from .config import (
    CAUSE_COL,
    CAUSE_QUOTE_COL,
    CAUSE_UNKNOWN_TEXT,
    CONSEQUENCE_COL,
    EXTRACTION_WORKERS,
    ID_COL,
    QUOTE_MIN_MATCH,
    TEXT_COL,
)
from .llm_client import call_llm_json

CAUSE_TYPES = ("explicite", "inférée", "inconnue")
EXTRACTION_COLUMNS = [CAUSE_COL, "cause_type", CAUSE_QUOTE_COL, CONSEQUENCE_COL, "composant_brut",
                      "cause_known", "extraction_source"]
UNKNOWN_PATTERN = re.compile(r"non (précis|determin|détermin|identifi|connu|renseign|indiqu)|inconnu|"
                             r"pas (précis|indiqu|d'information)|aucune (cause|information|indication)",
                             re.IGNORECASE)

SYSTEM_EXTRACTION_PROMPT = f"""Tu es ingénieur fiabilité. On te donne la description d'un rapport \
d'incident industriel. Réponds en JSON avec les clés suivantes.
- "cause_racine" : POURQUOI c'est arrivé, en une phrase courte, avec les mots du rapport. Remonte au \
phénomène qui a déclenché l'enchaînement plutôt qu'au symptôme final. Ne décris pas les dégâts ici.
- "cause_type" : « explicite » si la description écrit la cause ; « inférée » seulement si elle ne \
l'écrit pas mais contient un indice concret qui la désigne ; « inconnue » sinon. Pour une cause \
inconnue, écris exactement « {CAUSE_UNKNOWN_TEXT} » et laisse "indice" vide. En cas de doute, \
choisis « inconnue » : une cause inventée est pire qu'une cause inconnue.
- "indice" : l'extrait de la description qui justifie la cause, RECOPIÉ MOT POUR MOT (10 à 20 mots). \
Il sera vérifié : un extrait absent du rapport annule la cause.
- "consequence" : ce que l'incident a provoqué (dégât, arrêt, déclenchement d'une protection, \
risque), en une phrase courte, sans durée ni montant. Ne répète pas la cause.
- "composant_brut" : le composant touché, avec les mots du rapport (vide si non dit)."""


def simplify(text: Any) -> str:
    """Minuscules, sans accents ni ponctuation : tolère les petites variations de recopie."""
    text = unicodedata.normalize("NFKD", str(text or "").lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def quote_found(quote: Any, text: Any) -> bool:
    """L'extrait cité figure-t-il dans le texte (à QUOTE_MIN_MATCH près) ? Trois mots au moins."""
    q, t = simplify(quote), simplify(text)
    if len(q.split()) < 3 or not t:
        return False
    if q in t:
        return True
    m = difflib.SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size >= QUOTE_MIN_MATCH * len(q)


def _s(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def normalize_extraction(raw: Any, text: str, source: str) -> dict:
    """Contrôles déterministes : champs attendus, extrait vérifié, une seule valeur de cause inconnue."""
    raw = raw if isinstance(raw, dict) else {}
    out = {
        CAUSE_COL: _s(raw.get(CAUSE_COL))[:300],
        "cause_type": raw.get("cause_type") if raw.get("cause_type") in CAUSE_TYPES else "inconnue",
        CAUSE_QUOTE_COL: _s(raw.get(CAUSE_QUOTE_COL))[:400],
        CONSEQUENCE_COL: _s(raw.get(CONSEQUENCE_COL))[:300],
        "composant_brut": _s(raw.get("composant_brut"))[:200],
        "extraction_source": source,
    }
    known = (out["cause_type"] != "inconnue" and bool(out[CAUSE_COL])
             and not UNKNOWN_PATTERN.search(out[CAUSE_COL]) and quote_found(out[CAUSE_QUOTE_COL], text))
    if not known:
        out.update({CAUSE_COL: CAUSE_UNKNOWN_TEXT, "cause_type": "inconnue", CAUSE_QUOTE_COL: ""})
    out["cause_known"] = known
    return out


def _from_llm(text: str, llm: Callable[..., Any]) -> dict:
    try:
        raw = llm(system_prompt=SYSTEM_EXTRACTION_PROMPT, user_prompt=f"Description : {text}",
                  cache_tag="cause_extraction")
    except Exception:
        raw = None
    return normalize_extraction(raw, text, "llm")


def extract_causes(
    df: pd.DataFrame,
    llm: Callable[..., Any] = call_llm_json,
    seed: pd.DataFrame | None = None,
    workers: int = EXTRACTION_WORKERS,
) -> pd.DataFrame:
    """Une ligne d'extraction par incident (même index que `df`, colonnes EXTRACTION_COLUMNS).

    `seed` : extractions déjà faites (colonnes ID_COL, TEXT_COL et les champs extraits), reprises
    pour les incidents au texte identique ; les autres sont envoyés au LLM (réponses en cache). Un
    fichier sans colonne CAUSE_COL (sortie d'une version antérieure) est ignoré.
    """
    texts = df[TEXT_COL].astype(str)
    reused: dict[int, dict] = {}
    if seed is not None and CAUSE_COL not in seed.columns:
        print(f"       (extractions reprises ignorées : pas de colonne {CAUSE_COL})")
        seed = None
    if seed is not None and len(seed):
        by_id = seed.drop_duplicates(ID_COL).set_index(seed.drop_duplicates(ID_COL)[ID_COL].astype(str))
        for i, (inc, text) in enumerate(zip(df[ID_COL].astype(str), texts)):
            if inc in by_id.index and str(by_id.at[inc, TEXT_COL]) == text:
                row = by_id.loc[inc]
                raw = {k: (row.get(k) if pd.notna(row.get(k)) else "") for k in
                       (CAUSE_COL, "cause_type", CAUSE_QUOTE_COL, CONSEQUENCE_COL, "composant_brut")}
                reused[i] = normalize_extraction(raw, text, "import")
    todo = [i for i in range(len(df)) if i not in reused]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fresh = dict(zip(todo, pool.map(lambda i: _from_llm(texts.iloc[i], llm), todo)))
    rows = [reused.get(i) or fresh[i] for i in range(len(df))]
    return pd.DataFrame(rows, index=df.index, columns=EXTRACTION_COLUMNS)
