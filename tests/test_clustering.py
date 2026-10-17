"""Tests hors-ligne du clustering (aucun appel réseau)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from agentic_pipeline.clustering import SweepResult, _select_best, dbcv_score, sweep_hdbscan


def _blobs(n=300, dims=2):
    X, _ = make_blobs(n_samples=n, centers=3, n_features=dims, cluster_std=0.6, random_state=0)
    return X


def test_sweep_finds_three_blobs():
    res = sweep_hdbscan(_blobs())
    assert isinstance(res, SweepResult)
    assert len(set(res.labels) - {-1}) == 3
    assert res.dbcv > 0.3
    assert list(res.table.columns) == ["min_cluster_size", "n_clusters", "outlier_rate", "dbcv"]


def test_selection_rule():
    table = pd.DataFrame(
        {
            "min_cluster_size": [5, 7, 10, 15],
            "n_clusters": [4, 3, 3, 1],
            "outlier_rate": [0.50, 0.10, 0.10, 0.0],  # 5 : trop de bruit ; 15 : un seul cluster
            "dbcv": [0.9, 0.5, 0.5, 0.99],
        }
    )
    assert table.loc[_select_best(table), "min_cluster_size"] == 7  # égalité -> plus fin


def test_selection_fallback_when_none_qualifies():
    table = pd.DataFrame(
        {"min_cluster_size": [5, 10], "n_clusters": [1, 1], "outlier_rate": [0.5, 0.6], "dbcv": [0.1, 0.2]}
    )
    assert table.loc[_select_best(table), "min_cluster_size"] == 10


def test_dbcv_single_cluster_returns_minus_one():
    X = _blobs()
    assert dbcv_score(X, np.zeros(len(X), dtype=int)) == -1.0
    labels = np.zeros(len(X), dtype=int)
    labels[:50] = -1  # bruit + un cluster
    assert dbcv_score(X, labels) == -1.0


def test_dbcv_timing_900_points_5d(capsys):
    X = _blobs(900, 5)
    labels = np.repeat([0, 1, 2], 300)
    t = time.perf_counter()
    score = dbcv_score(X, labels)
    dt = time.perf_counter() - t
    print(f"dbcv_score 900x5 : {dt:.2f}s (score={score:.3f})")
    assert -1.0 <= score <= 1.0


def test_dbcv_string_labels_match_int_labels():
    X, y = make_blobs(300, centers=3, n_features=2, cluster_std=0.6, random_state=0)
    y = y.copy()
    y[:10] = -1
    ints = dbcv_score(X, y)
    strs = np.array([f"solar_F{k}" if k != -1 else "isole_atypique" for k in y], dtype=object)
    assert ints > 0.3
    assert dbcv_score(X, strs) == ints
    # OUTLIER_LABEL se comporte comme -1
    mixed = np.array([k if k != -1 else "isole_atypique" for k in y], dtype=object)
    assert dbcv_score(X, mixed) == ints


import pytest


@pytest.mark.parametrize("n", [3, 5])
def test_reduce_umap_tiny_inputs(n):
    from agentic_pipeline.clustering import reduce_umap

    emb = np.random.RandomState(0).rand(n, 32)
    Z = reduce_umap(emb)
    assert Z.shape[0] == n
    res = sweep_hdbscan(Z, sizes=[5])
    assert (res.labels == -1).all()
    assert res.dbcv == -1.0
