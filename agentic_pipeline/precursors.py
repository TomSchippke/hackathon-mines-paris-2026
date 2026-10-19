"""Fouille de précurseurs : paires (A -> B) sur un même équipement, fenêtre glissante,
modèle nul par permutation intra-équipement, correction de Benjamini-Hochberg.

Implémente la section 3.5 de ``docs/implementation_plan_clustering_v2.md``.
Statistiques uniquement : la plausibilité (LLM) est ajoutée en aval.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from .concentration import _benjamini_hochberg
from .config import (
    DATE_COL,
    EQUIPMENT_COL,
    ID_COL,
    OUTLIER_LABEL,
    PARTITION_COL,
    PRECURSOR_FDR,
    PRECURSOR_MIN_EQUIPMENTS,
    PRECURSOR_MIN_HITS,
    PRECURSOR_N_PERMUTATIONS,
    PRECURSOR_WINDOWS,
    RANDOM_STATE,
    SEVERITY_COL,
    SEVERITY_ORDER,
)

PAIR_COLUMNS = [
    "kind", "partition", "scope_cluster", "antecedent", "consequent", "window_days",
    "n_hits", "n_equipments", "n_consequent", "expected_hits", "lift", "p_value",
    "q_value", "is_best_window", "significant", "plausible", "mechanism", "monitoring_rule",
]
EVIDENCE_COLUMNS = [
    "rule_id", "consequent_incident_id", "equipment_id", "consequent_date",
    "antecedent_incident_ids", "antecedent_dates", "n_antecedents", "window_days",
]
# Colonnes des lignes testées avant correction BH (_test_partition)
_ROW_COLUMNS = ["partition", "antecedent", "consequent", "window_days", "n_hits", "n_equipments",
                "n_consequent", "expected_hits", "lift", "p_value"]
_GLOBAL = "global"
_SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def _severity_ranks(series: pd.Series) -> np.ndarray:
    """Rang de sévérité ; sévérité inconnue -> rang maximal (avertissement unique)."""
    order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    ranks = series.map(lambda s: order.get(s, order.get(str(s).strip().lower())))
    unknown = ranks.isna()
    if unknown.any():
        warnings.warn(
            f"{int(unknown.sum())} incident(s) avec une sévérité inconnue "
            f"({sorted(series[unknown].astype(str).unique())[:5]}) : rang maximal appliqué.",
            stacklevel=3,
        )
    return ranks.fillna(len(SEVERITY_ORDER)).astype(int).to_numpy()


def _prepare(
    df: pd.DataFrame, label_col: str, equipment_col: str, date_col: str,
    severity_col: str, id_col: str, partition_col: str | None,
) -> pd.DataFrame:
    """Table de travail normalisée, sans outliers, triée par (partition, équipement, date)."""
    work = pd.DataFrame({
        "uid": df[id_col].to_numpy() if id_col in df.columns else np.arange(len(df)),
        "label": df[label_col].to_numpy(),
        "equipment": df[equipment_col].astype(str).to_numpy(),
        "date": pd.to_datetime(df[date_col]).to_numpy(),
        "sev": _severity_ranks(df[severity_col]),
        "partition": (df[partition_col].astype(str).to_numpy()
                      if partition_col and partition_col in df.columns
                      else np.full(len(df), _GLOBAL, dtype=object)),
        "orig_index": df.index.to_numpy(),
    })
    work = work[work["label"].notna() & (work["label"] != OUTLIER_LABEL)]
    work = work.dropna(subset=["date"])
    work["label"] = work["label"].astype(str)
    return work.sort_values(["partition", "equipment", "date"], kind="stable").reset_index(drop=True)


def _prior_pairs(equip: np.ndarray, days: np.ndarray, max_window: float):
    """Indices (i, j, dt) avec j antérieur strict à i, même équipement, dt <= max_window.

    ``equip`` est trié par groupe contiguë, ``days`` croissant dans chaque groupe.
    """
    ii: list[np.ndarray] = []
    jj: list[np.ndarray] = []
    starts = np.flatnonzero(np.r_[True, equip[1:] != equip[:-1]])
    ends = np.r_[starts[1:], len(equip)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        d = days[s:e]
        dt = d[:, None] - d[None, :]
        mask = (dt > 0) & (dt <= max_window)
        a, b = np.nonzero(mask)
        ii.append(a + s)
        jj.append(b + s)
    if not ii:
        z = np.zeros(0, dtype=int)
        return z, z, np.zeros(0)
    i = np.concatenate(ii)
    j = np.concatenate(jj)
    dt = days[i] - days[j]
    order = np.argsort(-dt, kind="stable")  # dt décroissant : le plus petit écrit en dernier
    return i[order], j[order], dt[order]


# ---------------------------------------------------------------------------
# Comptage des hits (vectorisé)
# ---------------------------------------------------------------------------

def _hit_counts(lab, sev, pi, pj, pdt, windows, n_labels):
    """Matrices ``[w][B, A]`` du nombre d'incidents B ayant un antécédent A dans la fenêtre."""
    n = len(lab)
    ok = sev[pj] <= sev[pi]
    i, j, dt = pi[ok], pj[ok], pdt[ok]
    mind = np.full((n, n_labels), np.inf)
    mind[i, lab[j]] = dt  # dt décroissant -> la plus petite distance l'emporte
    rows, cols = np.nonzero(np.isfinite(mind))
    vals = mind[rows, cols]
    out = []
    for w in windows:
        m = vals <= w
        flat = lab[rows[m]] * n_labels + cols[m]
        out.append(np.bincount(flat, minlength=n_labels * n_labels).reshape(n_labels, n_labels))
    return out


def _observed_equipments(lab, sev, equip_codes, pi, pj, pdt, w, n_labels):
    """Nombre d'équipements distincts parmi les incidents B en hit, par paire (B, A)."""
    ok = (sev[pj] <= sev[pi]) & (pdt <= w)
    i, j = pi[ok], pj[ok]
    key = np.unique(np.stack([i, lab[j]], axis=1), axis=0) if len(i) else np.zeros((0, 2), int)
    res = np.zeros((n_labels, n_labels), dtype=int)
    if len(key):
        eq = equip_codes[key[:, 0]]
        trip = np.unique(np.stack([lab[key[:, 0]], key[:, 1], eq], axis=1), axis=0)
        np.add.at(res, (trip[:, 0], trip[:, 1]), 1)
    return res


# ---------------------------------------------------------------------------
# Test par partition (sans correction)
# ---------------------------------------------------------------------------

def _test_partition(
    part: pd.DataFrame, windows: Sequence[int], n_permutations: int,
    rng: np.random.Generator, min_hits: int, min_equipments: int,
) -> pd.DataFrame:
    """Lignes (A, B, W) d'une partition, avec p-values brutes (NaN si support insuffisant)."""
    labels = sorted(part["label"].unique())
    L = len(labels)
    lab = pd.Categorical(part["label"], categories=labels).codes.astype(int)
    sev = part["sev"].to_numpy()
    equip_codes = pd.factorize(part["equipment"])[0]
    days = part["date"].to_numpy().astype("datetime64[s]").astype(np.int64) / _SECONDS_PER_DAY
    windows = [int(w) for w in windows]
    pi, pj, pdt = _prior_pairs(part["equipment"].to_numpy(), days, max(windows))

    obs = _hit_counts(lab, sev, pi, pj, pdt, windows, L)
    n_eq = [_observed_equipments(lab, sev, equip_codes, pi, pj, pdt, w, L) for w in windows]
    tested = [(o >= min_hits) & (e >= min_equipments) for o, e in zip(obs, n_eq)]

    sums = [np.zeros((L, L)) for _ in windows]
    ge = [np.zeros((L, L)) for _ in windows]
    if any(t.any() for t in tested) and n_permutations > 0:
        for _ in range(n_permutations):
            # permutation intra-équipement : tri par (équipement, clé aléatoire)
            perm = np.argsort(equip_codes + rng.random(len(lab)) * 0.999, kind="stable")
            # les groupes sont contigus et dans le même ordre -> perm reste dans le groupe
            null = _hit_counts(lab[perm], sev[perm], pi, pj, pdt, windows, L)
            for k in range(len(windows)):
                sums[k] += null[k]
                ge[k] += null[k] >= obs[k]

    n_cons = np.bincount(lab, minlength=L)
    records = []
    for k, w in enumerate(windows):
        exp = sums[k] / max(n_permutations, 1)
        for b in range(L):
            for a in range(L):
                t = bool(tested[k][b, a])
                records.append((
                    part["partition"].iloc[0], labels[a], labels[b], w,
                    int(obs[k][b, a]), int(n_eq[k][b, a]), int(n_cons[b]),
                    float(exp[b, a]) if t else np.nan,
                    float(obs[k][b, a] / max(exp[b, a], 1e-9)) if t else np.nan,
                    (1 + ge[k][b, a]) / (1 + n_permutations) if t else np.nan,
                ))
    return pd.DataFrame(records, columns=_ROW_COLUMNS)


def _compute_rows(
    work: pd.DataFrame, windows, n_permutations, random_state, min_hits, min_equipments,
) -> pd.DataFrame:
    """Teste toutes les partitions de ``work`` ; aucune correction multiple ici."""
    rng = np.random.default_rng(random_state)
    parts = [
        _test_partition(g.reset_index(drop=True), windows, n_permutations, rng,
                        min_hits, min_equipments)
        for _, g in work.groupby("partition", sort=True)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_ROW_COLUMNS)


# ---------------------------------------------------------------------------
# Correction BH et mise en forme
# ---------------------------------------------------------------------------

def _finalize(rows: pd.DataFrame, kind: str, fdr: float) -> pd.DataFrame:
    """Une correction BH par ``kind`` (recurrence vs precursor/escalation) sur les lignes testées."""
    rows = rows.copy()
    rows["kind"] = np.where(rows["antecedent"] == rows["consequent"], "recurrence", kind)
    rows["q_value"] = np.nan
    tested = rows["p_value"].notna().to_numpy()
    # un pool BH par kind : les récurrences ne diluent pas les tests de précurseurs
    for k in rows["kind"].unique():
        m = tested & (rows["kind"] == k).to_numpy()
        if m.any():
            rows.loc[m, "q_value"] = _benjamini_hochberg(rows.loc[m, "p_value"].to_numpy())
    rows["is_best_window"] = False
    t = rows[tested].sort_values(["q_value", "window_days"], kind="stable")
    best = t.drop_duplicates(["partition", "scope_cluster", "antecedent", "consequent"]).index
    rows.loc[best, "is_best_window"] = True
    rows["significant"] = (
        rows["is_best_window"] & (rows["q_value"] <= fdr) & (rows["lift"] > 1)
    )
    rows["plausible"] = False
    rows["mechanism"] = ""
    rows["monitoring_rule"] = ""
    return rows[PAIR_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def mine_precursors(
    df: pd.DataFrame,
    label_col: str = "merged_cluster_id",
    windows: Sequence[int] = PRECURSOR_WINDOWS,
    n_permutations: int = PRECURSOR_N_PERMUTATIONS,
    random_state: int = RANDOM_STATE,
    min_hits: int = PRECURSOR_MIN_HITS,
    min_equipments: int = PRECURSOR_MIN_EQUIPMENTS,
    fdr: float = PRECURSOR_FDR,
    equipment_col: str = EQUIPMENT_COL,
    date_col: str = DATE_COL,
    severity_col: str = SEVERITY_COL,
    id_col: str = ID_COL,
    partition_col: str | None = PARTITION_COL,
    kind: str = "precursor",
    scope_cluster: str = "",
) -> pd.DataFrame:
    """Fouille les paires (antécédent -> conséquent) sur un même équipement.

    Une seule correction BH est appliquée sur toutes les lignes testées de l'appel.
    """
    work = _prepare(df, label_col, equipment_col, date_col, severity_col, id_col, partition_col)
    rows = _compute_rows(work, windows, n_permutations, random_state, min_hits, min_equipments)
    rows["scope_cluster"] = scope_cluster
    return _finalize(rows, kind, fdr)


def mine_escalations(
    df: pd.DataFrame,
    fine_col: str = "fine_cluster_id",
    merged_col: str = "merged_cluster_id",
    windows: Sequence[int] = PRECURSOR_WINDOWS,
    n_permutations: int = PRECURSOR_N_PERMUTATIONS,
    random_state: int = RANDOM_STATE,
    min_hits: int = PRECURSOR_MIN_HITS,
    min_equipments: int = PRECURSOR_MIN_EQUIPMENTS,
    fdr: float = PRECURSOR_FDR,
    equipment_col: str = EQUIPMENT_COL,
    date_col: str = DATE_COL,
    severity_col: str = SEVERITY_COL,
    id_col: str = ID_COL,
    partition_col: str | None = PARTITION_COL,
) -> pd.DataFrame:
    """Escalade fine au sein de chaque cluster fusionné ayant au moins 2 clusters fins.

    Un seul appel, donc une seule correction BH sur l'ensemble des lignes testées.
    """
    valid = df[df[merged_col].notna() & (df[merged_col] != OUTLIER_LABEL)
               & df[fine_col].notna() & (df[fine_col] != OUTLIER_LABEL)]
    chunks = []
    for merged_id, sub in valid.groupby(merged_col, sort=True):
        if sub[fine_col].nunique() < 2:
            continue
        work = _prepare(sub, fine_col, equipment_col, date_col, severity_col, id_col, partition_col)
        rows = _compute_rows(work, windows, n_permutations, random_state, min_hits, min_equipments)
        rows["scope_cluster"] = str(merged_id)
        chunks.append(rows)
    rows = (pd.concat(chunks, ignore_index=True) if chunks
            else pd.DataFrame(columns=[*_ROW_COLUMNS, "scope_cluster"]))
    return _finalize(rows, "escalation", fdr)


def extract_evidence(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    label_col: str = "merged_cluster_id",
    equipment_col: str = EQUIPMENT_COL,
    date_col: str = DATE_COL,
    severity_col: str = SEVERITY_COL,
    id_col: str = ID_COL,
    partition_col: str | None = PARTITION_COL,
    merged_col: str = "merged_cluster_id",
) -> pd.DataFrame:
    """Incidents conséquents en hit pour chaque règle ``significant & plausible``.

    Les lignes d'escalade sont restreintes aux incidents de leur ``scope_cluster``
    (colonne ``merged_col``, si présente). Les récurrences sont ignorées.
    """
    if "significant" not in pairs.columns or pairs.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    keep = pairs["significant"].astype(bool)
    if "plausible" in pairs.columns:
        keep &= pairs["plausible"].astype(bool)
    keep &= pairs["kind"] != "recurrence"
    sel = pairs[keep]
    if sel.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    work = _prepare(df, label_col, equipment_col, date_col, severity_col, id_col, partition_col)
    if merged_col in df.columns:
        work["scope"] = df.loc[work["orig_index"], merged_col].astype(str).to_numpy()
    else:
        work["scope"] = ""
    out = []
    for r in sel.itertuples(index=False):
        sub = work[work["partition"] == str(r.partition)]
        if r.kind == "escalation" and str(r.scope_cluster) and merged_col in df.columns:
            sub = sub[sub["scope"] == str(r.scope_cluster)]
        rule_id = f"{r.antecedent}->{r.consequent}@{int(r.window_days)}d"
        for _, g in sub.groupby("equipment", sort=True):
            ants = g[g["label"] == r.antecedent]
            if ants.empty:
                continue
            for b in g[g["label"] == r.consequent].itertuples():
                delta = (b.date - ants["date"]).dt.total_seconds() / _SECONDS_PER_DAY
                m = (delta > 0) & (delta <= r.window_days) & (ants["sev"] <= b.sev)
                if not m.any():
                    continue
                a = ants[m]
                out.append({
                    "rule_id": rule_id,
                    "consequent_incident_id": b.uid,
                    "equipment_id": b.equipment,
                    "consequent_date": pd.Timestamp(b.date).strftime("%Y-%m-%d"),
                    "antecedent_incident_ids": ";".join(map(str, a["uid"])),
                    "antecedent_dates": ";".join(a["date"].dt.strftime("%Y-%m-%d")),
                    "n_antecedents": int(m.sum()),
                    "window_days": int(r.window_days),
                })
    return pd.DataFrame(out, columns=EVIDENCE_COLUMNS)
