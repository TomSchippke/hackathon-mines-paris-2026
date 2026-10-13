"""Module de clustering non supervisé (UMAP + HDBSCAN + c-TF-IDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from topic_modeling.embeddings import embed_texts

from .config import (
    CACHE_DIR,
    EMBEDDINGS_CACHE_DIR,
    FRENCH_STOPWORDS,
    MIN_CLUSTER_SIZE,
    NGRAM_RANGE,
    NUM_REPRESENTATIVE_DOCS,
    UMAP_PARAMS,
)


def extract_ctfidf_keywords(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 8,
) -> dict[int, list[str]]:
    """Extrait les mots-clés distinctifs de chaque cluster via c-TF-IDF."""
    df = pd.DataFrame({"text": texts, "label": labels})
    clusters = sorted(df["label"].unique())

    # Concaténation de tous les textes par cluster
    docs_per_cluster = df.groupby("label")["text"].apply(lambda s: " ".join(s)).reset_index()

    vectorizer = CountVectorizer(
        stop_words=FRENCH_STOPWORDS,
        ngram_range=NGRAM_RANGE,
        token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b",
    )
    try:
        X = vectorizer.fit_transform(docs_per_cluster["text"])
        words = vectorizer.get_feature_names_out()

        # c-TF-IDF formulation
        tf = X.toarray()
        tf_norm = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
        idf = np.log((len(docs_per_cluster) + 1) / (1 + (X > 0).sum(axis=0))) + 1
        ctfidf = tf_norm * idf

        cluster_keywords = {}
        for idx, row in docs_per_cluster.iterrows():
            cluster_id = row["label"]
            scores = ctfidf[idx]
            top_word_indices = scores.argsort()[::-1][:top_n]
            cluster_keywords[cluster_id] = [words[i] for i in top_word_indices if scores[i] > 0]
        return cluster_keywords
    except Exception:
        # Secours si le vocabulaire est trop restreint
        return {c: [] for c in clusters}


def select_representative_documents(
    texts: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_docs: int = NUM_REPRESENTATIVE_DOCS,
) -> dict[int, list[str]]:
    """Sélectionne les incidents les plus proches du centroïde géométrique de chaque cluster."""
    representatives: dict[int, list[str]] = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            # Pour le bruit, on prend un échantillon aléatoire ou les premiers
            idx = np.where(labels == -1)[0][:n_docs]
            representatives[label] = [texts[i] for i in idx]
            continue

        idx = np.where(labels == label)[0]
        if len(idx) <= n_docs:
            representatives[label] = [texts[i] for i in idx]
            continue

        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_emb)[0]
        top_local_idx = sims.argsort()[::-1][:n_docs]
        representatives[label] = [texts[idx[i]] for i in top_local_idx]

    return representatives


def get_cluster_metadata_profile(df: pd.DataFrame, cluster_label: int, label_col: str = "raw_cluster") -> dict[str, Any]:
    """Extrait la répartition des métadonnées optionnelles pour enrichir le contexte de l'agent."""
    sub = df[df[label_col] == cluster_label]
    profile: dict[str, Any] = {"count": len(sub)}

    for col in ["severity", "weather_condition", "site", "asset_type"]:
        if col in sub.columns and sub[col].notna().any():
            top_vals = sub[col].value_counts().head(3).to_dict()
            profile[col] = top_vals

    return profile


def cluster_texts(
    df: pd.DataFrame,
    text_col: str = "text",
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    cache_root: Path = EMBEDDINGS_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[str]], dict[int, list[str]]]:
    """Exécute UMAP + HDBSCAN sur les embeddings et renvoie (labels, embeddings, keywords, representatives)."""
    texts = df[text_col].fillna("").tolist()
    embeddings = embed_texts(texts, cache_root)

    # Réduction UMAP
    reducer = UMAP(**UMAP_PARAMS)
    umap_embeddings = reducer.fit_transform(embeddings)

    # Clustering HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(umap_embeddings)

    # Extraction des mots-clés et documents représentatifs
    keywords = extract_ctfidf_keywords(texts, labels)
    representatives = select_representative_documents(texts, embeddings, labels)

    return labels, embeddings, keywords, representatives
