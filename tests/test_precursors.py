"""Tests de `agentic_pipeline.precursors.mine_precursors` (boîte noire, spec §3.5).

Les données sont synthétiques : 40 équipements, ~3 ans, processus de Poisson,
6 labels, sévérités aléatoires. Un motif X -> Y est injecté (ou non) selon le cas.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from agentic_pipeline.precursors import mine_precursors

SEV = ["mineure", "modérée", "sévère"]
SEV_RANK = {s: i for i, s in enumerate(SEV)}
LABELS = [f"L{k}" for k in range(6)]
X, Y = "L0", "L1"
N_EQUIP = 40
N_DAYS = 3 * 365
RATE_PER_DAY = 45 / N_DAYS  # ~45 incidents / équipement sur 3 ans
START = pd.Timestamp("2021-01-01")
N_PERM = 200
# X et Y plus rares que les autres labels : moins de hits fortuits, signal planté net.
LABEL_P = [0.10, 0.10, 0.20, 0.20, 0.20, 0.20]
WINDOWS = [7, 30, 60]


def _background(rng: np.random.Generator, equip_ids: list[str]) -> list[dict]:
    """Incidents de fond : jours Poisson, label et sévérité indépendants du temps."""
    rows = []
    for eq in equip_ids:
        n = rng.poisson(RATE_PER_DAY * N_DAYS)
        days = rng.integers(0, N_DAYS, size=n)
        labs = rng.choice(LABELS, size=n, p=LABEL_P)
        sevs = rng.choice(SEV, size=n, p=[0.5, 0.3, 0.2])
        rows += [
            {"equipment_id": eq, "day": int(d), "label": str(l), "severity": str(s)}
            for d, l, s in zip(days, labs, sevs)
        ]
    return rows


def _plant(rng, rows, equip_ids, *, frac=0.4, n_ante=2, gap=(3, 20), sev_mode="le"):
    """Avant `frac` des incidents Y des `equip_ids`, insère `n_ante` incidents X
    `gap` jours plus tôt. sev_mode='le' : sev(X) <= sev(Y) ; 'gt' : sev(X) > sev(Y)."""
    extra = []
    for r in rows:
        if r["equipment_id"] not in equip_ids or r["label"] != Y:
            continue
        if rng.random() >= frac:
            continue
        ry = SEV_RANK[r["severity"]]
        if sev_mode == "gt" and ry == len(SEV) - 1:
            continue  # rien de plus sévère que le maximum
        for _ in range(n_ante):
            g = int(rng.integers(gap[0], gap[1] + 1))
            if r["day"] - g < 0:
                continue
            rk = int(rng.integers(0, ry + 1)) if sev_mode == "le" else int(rng.integers(ry + 1, len(SEV)))
            extra.append({"equipment_id": r["equipment_id"], "day": r["day"] - g,
                          "label": X, "severity": SEV[rk]})
    return rows + extra


def _to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # Une seule ligne par (équipement, jour) n'est pas exigée ; on trie pour un ordre stable.
    df = df.sort_values(["equipment_id", "day", "label"]).reset_index(drop=True)
    df["date"] = (START + pd.to_timedelta(df["day"], unit="D")).dt.strftime("%Y-%m-%d")
    df["incident_id"] = np.arange(len(df))
    return df[["incident_id", "equipment_id", "date", "severity", "label"]]


def make_data(seed: int, *, plant_equips: int = 0, lemon: bool = False, **plant_kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    equips = [f"E{i:02d}" for i in range(N_EQUIP)]
    rows = _background(rng, equips)
    if plant_equips or lemon:
        targets = set(equips[:plant_equips]) if plant_equips else {equips[0]}
        if lemon:
            # Machine « citron » : beaucoup d'occurrences Y sur un seul équipement.
            extra = [{"equipment_id": equips[0], "day": int(d), "label": Y, "severity": "sévère"}
                     for d in rng.integers(60, N_DAYS, size=40)]
            rows = rows + extra
        rows = _plant(rng, rows, targets, **plant_kw)
    return _to_df(rows)


def naive_hits(df: pd.DataFrame, a: str, b: str, window: int) -> tuple[int, int]:
    """Référence O(n²) de hit(b) du §3.5 : (n_hits, n_equipments)."""
    d = df.assign(t=pd.to_datetime(df["date"]), r=df["severity"].map(SEV_RANK))
    n, eqs = 0, set()
    for eq, g in d.groupby("equipment_id"):
        ants = g[g["label"] == a]
        for _, rb in g[g["label"] == b].iterrows():
            ok = False
            for _, ra in ants.iterrows():
                if ra["incident_id"] == rb["incident_id"]:
                    continue
                delta = (rb["t"] - ra["t"]).days
                if 1 <= delta <= window and ra["r"] <= rb["r"]:
                    ok = True
                    break
            if ok:
                n += 1
                eqs.add(eq)
    return n, len(eqs)


def run(df, **kw):
    kw.setdefault("n_permutations", N_PERM)
    return mine_precursors(df, label_col="label", windows=WINDOWS, random_state=0, **kw)


def _pair(res, a, b, kind=None):
    m = (res["antecedent"] == a) & (res["consequent"] == b)
    if kind:
        m &= res["kind"] == kind
    return res[m]


# Le spec dit 40 % des Y précédés de 2 X. Avec 200 permutations, p >= 1/201 et la
# correction BH sur ~80-100 lignes testées rend q <= 0.05 inatteignable (p_min * m / k).
# Pour le cas « motif planté » on utilise donc 1000 permutations (valeur de D7), un
# support un peu plus strict (moins de lignes testées) et 70 % de Y précédés (signal
# stable sur 5 graines de données ; avec 40 % le lift à 30 j reste autour de 1.5-2).
PLANT_FRAC = 0.7
PLANT_KW = dict(n_permutations=1000, min_hits=15, min_equipments=5)


@pytest.fixture(scope="module")
def planted_df():
    return make_data(1, plant_equips=12, frac=PLANT_FRAC)


@pytest.fixture(scope="module")
def planted_res(planted_df):
    return run(planted_df, **PLANT_KW)


def test_schema(planted_res):
    cols = {"kind", "antecedent", "consequent", "window_days", "n_hits", "n_equipments",
            "lift", "p_value", "q_value", "is_best_window", "significant"}
    assert cols <= set(planted_res.columns)
    assert set(planted_res["window_days"]) <= set(WINDOWS)


def test_generator_sanity(planted_df):
    """Le motif injecté produit nettement plus de hits X->Y que des paires témoins."""
    n, ne = naive_hits(planted_df, X, Y, 30)
    ctrl = [naive_hits(planted_df, a, Y, 30)[0] for a in ("L2", "L3", "L4", "L5")]
    assert ne >= 12
    assert n > 1.5 * max(ctrl)


def test_planted_pattern_significant(planted_res):
    rows = _pair(planted_res, X, Y, "precursor")
    assert len(rows) == len(WINDOWS)
    best = rows[rows["is_best_window"]]
    assert len(best) == 1
    best = best.iloc[0]
    assert bool(best["significant"])
    assert best["lift"] > 2
    # Écarts injectés 3-20 j : 30 j capte tout, 7 j une partie, 60 j ajoute du bruit.
    # Mais la p-value sature (plancher 1/(1+N)) pour 7 j ET 30 j, les q-values sont alors
    # à égalité et le §3.5 départage vers la plus petite fenêtre (7). Asserter exactement 30
    # serait donc fragile : on accepte 7 ou 30 (jamais 60, dilué par le bruit).
    assert int(best["window_days"]) in (7, 30)
    r30 = rows[rows["window_days"] == 30].iloc[0]
    assert r30["p_value"] <= 0.01 and r30["lift"] > 1.5
    assert r30["q_value"] <= 0.05


def test_n_hits_matches_naive(planted_df, planted_res):
    r30 = _pair(planted_res, X, Y, "precursor")
    r30 = r30[r30["window_days"] == 30].iloc[0]
    n, ne = naive_hits(planted_df, X, Y, 30)
    assert int(r30["n_hits"]) == n
    assert int(r30["n_equipments"]) == ne


def test_null_no_significant_precursor():
    df = make_data(7)
    res = run(df)
    prec = res[res["kind"] == "precursor"]
    assert len(prec) > 0
    assert int(prec["significant"].sum()) == 0


def test_lemon_machine_not_significant():
    # Motif planté sur un seul équipement : beaucoup de hits mais < min_equipments.
    df = make_data(3, lemon=True, frac=1.0)
    n, ne = naive_hits(df, X, Y, 30)
    assert n >= 10  # beaucoup de hits au total, dont la quasi-totalité sur E00
    res = run(df)
    rows = _pair(res, X, Y, "precursor")
    # Le motif dominant vient d'un seul équipement : les hits sur les autres sont du bruit.
    assert not rows["significant"].any()


def test_severity_rule():
    # X toujours plus sévère que Y -> aucun hit planté ne compte.
    df = make_data(4, plant_equips=12, sev_mode="gt")
    res = run(df)
    rows = _pair(res, X, Y, "precursor")
    assert not rows["significant"].any()
    for w in WINDOWS:
        r = rows[rows["window_days"] == w].iloc[0]
        n, _ = naive_hits(df, X, Y, w)
        assert int(r["n_hits"]) == n


def test_recurrence_kind(planted_res):
    yy = _pair(planted_res, Y, Y)
    assert len(yy) > 0
    assert set(yy["kind"]) == {"recurrence"}
    prec = planted_res[planted_res["kind"] == "precursor"]
    assert not (prec["antecedent"] == prec["consequent"]).any()


def test_runtime_budget(planted_df):
    t0 = time.time()
    run(planted_df)
    assert time.time() - t0 < 10
