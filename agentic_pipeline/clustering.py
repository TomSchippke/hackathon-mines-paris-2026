"""Module de clustering non supervisé (UMAP + HDBSCAN + c-TF-IDF)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from topic_modeling.embeddings import embed_texts

from .config import (
    EMBEDDINGS_CACHE_DIR,
    FRENCH_STOPWORDS,
    MCS_SWEEP,
    MIN_SAMPLES,
    NGRAM_RANGE,
    NUM_REPRESENTATIVE_DOCS,
    OUTLIER_LABEL,
    RANDOM_STATE,
    SWEEP_MAX_OUTLIER_RATE,
    SWEEP_MIN_CLUSTERS,
    UMAP_PARAMS,
)


@dataclass
class SweepResult:
    """Résultat du balayage HDBSCAN : meilleur réglage retenu et table de tous les essais."""

    labels: np.ndarray
    min_cluster_size: int
    dbcv: float
    table: pd.DataFrame  # colonnes : min_cluster_size, n_clusters, outlier_rate, dbcv


def embed_partition(texts: list[str]) -> np.ndarray:
    """Embeddings (avec cache disque) des textes d'une partition."""
    return embed_texts(list(texts), EMBEDDINGS_CACHE_DIR)


def reduce_umap(emb: np.ndarray) -> np.ndarray:
    """Réduction UMAP des embeddings (paramètres et graine fixés dans la config)."""
    params = dict(UMAP_PARAMS, random_state=RANDOM_STATE)
    n = len(emb)
    # Garde-fou pour les petites partitions : init spectrale impossible (eigsh exige k < N)
    params["n_neighbors"] = max(2, min(params["n_neighbors"], n - 1))
    if n < 10:
        params["init"] = "random"
        params["n_components"] = max(1, min(params["n_components"], n - 2))
    return UMAP(**params).fit_transform(emb)


def dbcv_score(X_umap: np.ndarray, labels: np.ndarray) -> float:
    """DBCV (hors bruit). Renvoie -1.0 si moins de 2 clusters, en cas d'erreur ou de nan."""
    try:
        labels = np.asarray(labels, dtype=object)
        # Bruit : -1 ou OUTLIER_LABEL (masque élément par élément, labels int ou str)
        noise = np.array([lab == -1 or lab == OUTLIER_LABEL for lab in labels], dtype=bool)
        kept = labels[~noise]
        codes, uniques = pd.factorize(kept)
        if len(uniques) < 2:
            return -1.0
        score = float(validity_index(np.asarray(X_umap)[~noise].astype(np.float64), codes))
        return -1.0 if not np.isfinite(score) else score
    except Exception:
        return -1.0


def _select_best(table: pd.DataFrame) -> int:
    """Index de la ligne retenue : max DBCV parmi les réglages admissibles, à égalité le plus fin."""
    ok = table[(table["outlier_rate"] <= SWEEP_MAX_OUTLIER_RATE) & (table["n_clusters"] >= SWEEP_MIN_CLUSTERS)]
    pool = ok if len(ok) else table
    best = pool.sort_values(["dbcv", "min_cluster_size"], ascending=[False, True])
    return int(best.index[0])


def sweep_hdbscan(X_umap: np.ndarray, sizes: list[int] = MCS_SWEEP) -> SweepResult:
    """Balaye min_cluster_size, évalue chaque clustering par DBCV et retient le meilleur (§3.2)."""
    rows, all_labels = [], []
    for size in sizes:
        clusterer = HDBSCAN(
            min_cluster_size=size,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X_umap)
        all_labels.append(labels)
        rows.append(
            {
                "min_cluster_size": size,
                "n_clusters": len(set(labels) - {-1}),
                "outlier_rate": float(np.mean(labels == -1)),
                "dbcv": dbcv_score(X_umap, labels),
            }
        )
    table = pd.DataFrame(rows)
    best = _select_best(table)
    return SweepResult(
        labels=all_labels[best],
        min_cluster_size=int(table.loc[best, "min_cluster_size"]),
        dbcv=float(table.loc[best, "dbcv"]),
        table=table,
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
        idf = np.log((len(docs_per_cluster) + 1) / (1 + np.asarray((X > 0).sum(axis=0)).ravel())) + 1
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


def representative_indices(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_docs: int = NUM_REPRESENTATIVE_DOCS,
) -> dict[int, list[int]]:
    """Positions des incidents les plus proches du centroïde de chaque cluster (bruit : les premiers)."""
    representatives: dict[int, list[int]] = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if label == -1 or len(idx) <= n_docs:
            representatives[label] = idx[:n_docs].tolist()
            continue
        cluster_emb = embeddings[idx]
        sims = cosine_similarity(cluster_emb.mean(axis=0, keepdims=True), cluster_emb)[0]
        representatives[label] = [int(idx[i]) for i in sims.argsort()[::-1][:n_docs]]
    return representatives
