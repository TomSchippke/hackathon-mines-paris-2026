"""Diagnostic de stabilité UMAP : le clustering fin dépend-il de la graine ?

Réponse à l'objection « un DBCV de 0,9 en espace UMAP est un artefact de UMAP » : on relance UMAP
avec plusieurs graines et on mesure l'accord (Adjusted Rand Index) des clusters fins HDBSCAN.
Aucun appel à l'API : les embeddings doivent déjà être dans le cache disque.

Usage : .venv/bin/python -m scripts.umap_stability [--seeds 42 0 1 2 3] [--input CSV]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from agentic_pipeline import clustering
from agentic_pipeline.config import (
    DEFAULT_INPUT,
    EMBEDDINGS_CACHE_DIR,
    FALLBACK_INPUT,
    MCS_SWEEP,
    MIN_SAMPLES,
    PARTITION_COL,
    RANDOM_STATE,
    TEXT_COL,
)
from agentic_pipeline.pipeline import _load_incidents
from topic_modeling.embeddings import EmbeddingCache

OUT_CSV = Path("outputs/diagnostics/umap_stability.csv")
DEFAULT_SEEDS = [42, 0, 1, 2, 3]


def _hdbscan(X: np.ndarray, mcs: int) -> np.ndarray:
    return HDBSCAN(min_cluster_size=mcs, min_samples=MIN_SAMPLES, metric="euclidean",
                   cluster_selection_method="eom").fit_predict(X)


def _ari(a: np.ndarray, b: np.ndarray, non_noise: bool = False) -> float:
    if non_noise:
        keep = (a != -1) & (b != -1)
        if keep.sum() < 2:
            return float("nan")
        a, b = a[keep], b[keep]
    return float(adjusted_rand_score(a, b))


def _embeddings_from_cache(texts: list[str]) -> np.ndarray:
    """Embeddings lus dans le cache disque uniquement ; erreur claire si un texte manque."""
    load_dotenv(find_dotenv(usecwd=True), override=True)
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embeddings")
    cache = EmbeddingCache(EMBEDDINGS_CACHE_DIR, deployment)
    missing = cache.missing(texts)
    if missing:
        sys.exit(f"Erreur : {len(missing)} texte(s) absent(s) du cache d'embeddings ({cache.path}). "
                 "Ce script n'appelle jamais l'API : lancez d'abord le pipeline pour remplir le cache.")
    return cache.matrix(texts)


def analyse_partition(partition: str, texts: list[str], seeds: list[int]) -> list[dict]:
    n = len(texts)
    emb = _embeddings_from_cache(texts)
    sizes = [s for s in MCS_SWEEP if s <= n // 2]
    ref_seed = RANDOM_STATE if RANDOM_STATE in seeds else seeds[0]
    print(f"\n=== Partition '{partition}' ({n} incidents) ===")

    X = {s: clustering.reduce_umap(emb, s) for s in seeds}
    sweeps = {s: clustering.sweep_hdbscan(X[s], sizes) for s in seeds}
    fixed_mcs = int(sweeps[ref_seed].min_cluster_size)
    print(f"min_cluster_size du pipeline (graine {ref_seed}) : {fixed_mcs}")
    labels = {s: _hdbscan(X[s], fixed_mcs) for s in seeds}
    labels_sel = {s: np.asarray(sweeps[s].labels) for s in seeds}

    rows = []
    for s in seeds:
        lab = labels[s]
        sw = sweeps[s]
        sel = sw.table[sw.table["min_cluster_size"] == sw.min_cluster_size].iloc[0]
        rows.append({
            "partition": partition, "seed": s, "ref_seed": ref_seed, "n": n,
            "mcs_fixed": fixed_mcs,
            "n_clusters_fixed": len(set(lab) - {-1}),
            "outlier_rate_fixed": float(np.mean(lab == -1)),
            "dbcv_fixed": clustering.dbcv_score(X[s], lab),
            "mcs": int(sw.min_cluster_size),
            "n_clusters": int(sel["n_clusters"]),
            "outlier_rate": float(sel["outlier_rate"]),
            "dbcv": float(sw.dbcv),
            "ari_vs_ref": _ari(lab, labels[ref_seed]),
            "ari_vs_ref_non_noise": _ari(lab, labels[ref_seed], non_noise=True),
            "ari_vs_ref_sweep": _ari(labels_sel[s], labels_sel[ref_seed]),
        })

    pairs = list(itertools.combinations(seeds, 2))
    pw = [_ari(labels[a], labels[b]) for a, b in pairs]
    pw_nn = [_ari(labels[a], labels[b], non_noise=True) for a, b in pairs]
    others = [r for r in rows if r["seed"] != ref_seed]
    print(f"ARI par paires (mcs={fixed_mcs}, {len(pairs)} paires) : moyenne {np.nanmean(pw):.3f}, "
          f"min {np.nanmin(pw):.3f}")
    print(f"ARI par paires hors bruit des deux côtés : moyenne {np.nanmean(pw_nn):.3f}, "
          f"min {np.nanmin(pw_nn):.3f}")
    print(f"ARI vs graine {ref_seed} : moyenne {np.mean([r['ari_vs_ref'] for r in others]):.3f}, "
          f"min {np.min([r['ari_vs_ref'] for r in others]):.3f}")
    print("Sélection complète par graine (mcs / clusters / isolés / DBCV) : "
          + " ; ".join(f"{r['seed']}: {r['mcs']}/{r['n_clusters']}/{r['outlier_rate']:.0%}/{r['dbcv']:.2f}"
                       for r in rows))
    for r in rows:
        r.update(pairwise_ari_mean=float(np.nanmean(pw)), pairwise_ari_min=float(np.nanmin(pw)),
                 pairwise_ari_nn_mean=float(np.nanmean(pw_nn)), pairwise_ari_nn_min=float(np.nanmin(pw_nn)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="graines UMAP")
    ap.add_argument("--input", type=Path, default=None, help="CSV des incidents")
    args = ap.parse_args()

    input_csv = args.input or (DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT)
    if not input_csv.exists():
        sys.exit(f"Erreur : fichier d'entrée introuvable ({input_csv}).")
    df = _load_incidents(input_csv)
    print(f"{len(df)} incidents chargés depuis {input_csv} ; graines : {args.seeds}")

    rows: list[dict] = []
    for partition in sorted(df[PARTITION_COL].unique()):
        texts = df.loc[df[PARTITION_COL] == partition, TEXT_COL].astype(str).tolist()
        if len(texts) < 2 * min(MCS_SWEEP):
            print(f"\nPartition '{partition}' : {len(texts)} incidents, trop peu, ignorée.")
            continue
        rows += analyse_partition(partition, texts, args.seeds)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nRésultats écrits dans {OUT_CSV}")


if __name__ == "__main__":
    main()
