"""Fiches (cards) et étiquetage LLM léger des clusters fins (sans recommandations)."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd

from .clustering import (
    extract_ctfidf_keywords,
    get_cluster_metadata_profile,
    select_representative_documents,
)
from .config import (
    COST_COL,
    NUM_REPRESENTATIVE_DOCS,
    OUTLIER_LABEL,
    SEVERITY_COL,
    TEXT_COL,
)
from .llm_client import call_llm_json


class FineCard(TypedDict):
    """Fiche d'un cluster fin (un dict simple, sérialisable en JSON)."""

    fine_cluster_id: str
    partition: str
    size: int
    keywords: list[str]
    representative_texts: list[str]
    metadata_profile: dict[str, Any]
    severity_distribution: dict[str, int]
    mean_cost: float


SYSTEM_FINE_LABEL_PROMPT = """Tu es un expert en fiabilité industrielle.
On te présente un petit groupe d'incidents très similaires (cluster fin) d'un même type d'actif.
Donne-lui une étiquette courte et factuelle, SANS recommandation ni action corrective.

Réponds STRICTEMENT avec un objet JSON (textes en français) :
- "fine_title" : intitulé court (max ~8 mots) du mode de défaillance observé.
- "failure_mode" : une phrase décrivant le mode de défaillance (ce qui se produit, composant concerné)."""


def build_fine_cards(
    df: pd.DataFrame,
    fine_labels: np.ndarray,
    embeddings: np.ndarray,
    partition: str,
    text_col: str = TEXT_COL,
    cost_col: str = COST_COL,
    severity_col: str = SEVERITY_COL,
) -> dict[str, FineCard]:
    """Construit une fiche par cluster fin d'une partition (le bruit OUTLIER_LABEL est ignoré).

    `df` : lignes de la partition (index réinitialisé), alignées avec `fine_labels` et
    `embeddings`. Les labels texte sont factorisés en entiers avant d'appeler les helpers
    de clustering (écrits pour des labels entiers, -1 = bruit).
    """
    df = df.reset_index(drop=True)
    labels = np.asarray(fine_labels).astype(str)
    if not (len(df) == len(labels) == len(embeddings)):
        raise ValueError("df, fine_labels et embeddings doivent avoir la même longueur.")

    ids = [u for u in dict.fromkeys(labels.tolist()) if u != OUTLIER_LABEL]
    code = {u: i for i, u in enumerate(ids)}
    int_labels = np.array([code.get(l, -1) for l in labels], dtype=int)
    texts = df[text_col].astype(str).tolist()

    keywords = extract_ctfidf_keywords(texts, int_labels)
    reps = select_representative_documents(
        texts, np.asarray(embeddings), int_labels, n_docs=NUM_REPRESENTATIVE_DOCS
    )
    work = df.assign(_fine_int=int_labels)

    cards: dict[str, FineCard] = {}
    for fid in ids:
        k = code[fid]
        sub = work[work["_fine_int"] == k]
        if severity_col in sub.columns:
            sev = {str(a): int(b) for a, b in sub[severity_col].value_counts().items()}
        else:
            sev = {}
        if cost_col in sub.columns:
            mc = pd.to_numeric(sub[cost_col], errors="coerce").mean()
            mean_cost = float(mc) if pd.notna(mc) else 0.0
        else:
            mean_cost = 0.0
        cards[fid] = FineCard(
            fine_cluster_id=fid,
            partition=partition,
            size=int(len(sub)),
            keywords=list(keywords.get(k, [])),
            representative_texts=list(reps.get(k, [])),
            metadata_profile=get_cluster_metadata_profile(work, k, label_col="_fine_int"),
            severity_distribution=sev,
            mean_cost=mean_cost,
        )
    return cards


def _default_label(card: FineCard) -> dict[str, str]:
    title = " ".join(card.get("keywords", [])[:3]).strip() or str(card.get("fine_cluster_id", ""))
    return {"fine_title": title, "failure_mode": ""}


def _card_prompt(card: FineCard) -> str:
    texts = "\n".join(f"- {t.strip()}" for t in card["representative_texts"])
    return (
        f"Type d'actif : {card['partition']}\n"
        f"Taille : {card['size']} incidents\n"
        f"Mots-clés : {', '.join(card['keywords'])}\n"
        f"Sévérités : {card['severity_distribution']}\n"
        f"Incidents représentatifs :\n{texts}\n"
    )


def label_fine_clusters(
    cards: dict[str, FineCard],
    llm: Callable[..., dict[str, Any]] = call_llm_json,
) -> dict[str, dict[str, str]]:
    """Un appel LLM léger par fiche -> {fine_id: {"fine_title", "failure_mode"}}.

    Jamais d'exception : sortie mal formée ou erreur du LLM -> valeurs par défaut
    (titre = 3 premiers mots-clés, failure_mode = "").
    """
    out: dict[str, dict[str, str]] = {}
    for fid, card in cards.items():
        label = _default_label(card)
        try:
            res = llm(
                system_prompt=SYSTEM_FINE_LABEL_PROMPT,
                user_prompt=_card_prompt(card),
                cache_tag="fine_label",
            )
            if isinstance(res, dict):
                title = res.get("fine_title")
                mode = res.get("failure_mode")
                if isinstance(title, str) and title.strip():
                    label["fine_title"] = title.strip()[:200]
                if isinstance(mode, str):
                    label["failure_mode"] = mode.strip()[:500]
        except Exception:
            pass
        out[fid] = label
    return out
