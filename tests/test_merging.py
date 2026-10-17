"""Tests hors-ligne de l'agent de fusion (LLM factice, données synthétiques)."""

from __future__ import annotations

import re

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from agentic_pipeline.config import MERGE_MAX_ROUNDS, OUTLIER_LABEL
from agentic_pipeline.merging import FineState, MergeResult, run_merge_agent

LOG_KEYS = {
    "round", "action", "members", "result_id", "shared_remediation", "prevented_outcomes",
    "justification", "dbcv_before", "dbcv_after", "vetoed", "critic_verdict",
}


def make_state(blobs, n_out=0, seed=0):
    """blobs : liste de (id, centre_2d, n, groupe_embedding) ; même groupe = embeddings voisins."""
    rng = np.random.RandomState(seed)
    group_vec: dict[str, np.ndarray] = {}
    Xs, Es, labs = [], [], []
    for fid, center, n, grp in blobs:
        x, _ = make_blobs(n_samples=n, centers=[center], cluster_std=0.4, n_features=2, random_state=seed + len(labs))
        if grp not in group_vec:
            group_vec[grp] = rng.randn(8)
        e = group_vec[grp] + 0.05 * rng.randn(n, 8)
        Xs.append(x)
        Es.append(e)
        labs += [fid] * n
    if n_out:
        Xs.append(rng.uniform(-30, 30, (n_out, 2)))
        Es.append(rng.randn(n_out, 8))
        labs += [OUTLIER_LABEL] * n_out
    ids = [b[0] for b in blobs]
    cards = {i: {"keywords": [i], "representative_texts": [f"texte {i}"], "size": 1} for i in ids}
    titles = {i: {"fine_title": f"titre {i}", "failure_mode": f"mode {i}"} for i in ids}
    return FineState("P", np.vstack(Xs), np.vstack(Es), np.array(labs, dtype=object), cards, titles)


@pytest.fixture(autouse=True)
def no_hard_ceiling(monkeypatch):
    # Les blobs synthétiques sont très séparés : sans ce réglage, toute fusion heurte le veto ferme
    monkeypatch.setattr("agentic_pipeline.merging.MERGE_DBCV_HARD_DROP", float("inf"))


def ids_in(prompt: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"### Cluster (\S+)", prompt)))


class FakeLLM:
    def __init__(self, merge_fn=None, critic=True):
        self.merge_fn = merge_fn or (lambda ids, user: {"merges": [], "flags": []})
        self.critic = critic
        self.calls = []

    def __call__(self, system, user, cache_tag="general", **kw):
        self.calls.append(cache_tag)
        if cache_tag == "merge":
            return self.merge_fn(ids_in(user), user)
        if cache_tag == "merge_critic":
            return {
                "component": "joint" if self.critic else "",
                "action": "remplacer" if self.critic else "",
                "addresses_all_members": self.critic,
                "single_specific_remediation": self.critic,
                "reason": "test",
            }
        raise AssertionError(cache_tag)


def merge_of(members):
    return {
        "members": members,
        "shared_remediation": "remplacer le joint",
        "prevented_outcomes": [{"fine_cluster_id": m, "outcome": f"panne {m}"} for m in members],
        "justification": "même action",
    }


def check_invariants(state, res: MergeResult):
    final = {x for x in res.fine_labels.tolist() if x != OUTLIER_LABEL}
    assert set(res.fine_to_merged) == final
    assert set(sum(res.merged_members.values(), [])) == final
    for e in res.log:
        assert LOG_KEYS <= set(e)


def three_blobs():
    return make_state(
        [("P_F0", (0, 0), 40, "a"), ("P_F1", (1.2, 0), 40, "a"), ("P_F2", (15, 15), 40, "b"), ("P_F3", (-15, 15), 40, "c")],
        n_out=5,
    )


def test_a_duplicates_merged(no_llm):
    st = three_blobs()
    llm = FakeLLM(lambda ids, u: {"merges": [merge_of(["P_F0", "P_F1"])] if "P_F0" in ids else [], "flags": []})
    res = run_merge_agent(st, llm=llm)
    assert res.fine_to_merged["P_F0"] == res.fine_to_merged["P_F1"]
    assert res.fine_to_merged["P_F2"] != res.fine_to_merged["P_F0"]
    assert res.fine_to_merged["P_F2"] != res.fine_to_merged["P_F3"]
    # numérotation par taille décroissante : le groupe de 80 points est M0
    assert res.fine_to_merged["P_F0"] == "P_M0"
    assert res.merged_members["P_M0"] == ["P_F0", "P_F1"]
    acc = [e for e in res.log if e["action"] == "merge" and not e["vetoed"]]
    assert acc and acc[0]["result_id"] == "P_M0"
    check_invariants(st, res)


def test_b_veto_when_dbcv_drops_and_critic_false(no_llm):
    st = three_blobs()
    llm = FakeLLM(lambda ids, u: {"merges": [merge_of(["P_F0", "P_F2"])] if "P_F2" in ids else [], "flags": []}, critic=False)
    res = run_merge_agent(st, llm=llm)
    assert res.fine_to_merged["P_F0"] != res.fine_to_merged["P_F2"]
    vetoed = [e for e in res.log if e["vetoed"]]
    assert vetoed and vetoed[0]["result_id"] is None
    assert vetoed[0]["dbcv_after"] < vetoed[0]["dbcv_before"]
    assert vetoed[0]["critic_verdict"]["single_specific_remediation"] is False
    assert "merge_critic" in llm.calls
    check_invariants(st, res)


def test_critic_true_accepts_large_drop(no_llm):
    st = three_blobs()
    llm = FakeLLM(lambda ids, u: {"merges": [merge_of(["P_F0", "P_F2"])] if "P_F2" in ids else [], "flags": []}, critic=True)
    res = run_merge_agent(st, llm=llm, max_rounds=1)
    assert res.fine_to_merged["P_F0"] == res.fine_to_merged["P_F2"]


def test_critic_without_named_component_vetoes(no_llm):
    # Le critique strict exige un composant et une action nommés, pas seulement un booléen
    st = three_blobs()
    llm = FakeLLM(lambda ids, u: {"merges": [merge_of(["P_F0", "P_F2"])] if "P_F2" in ids else [], "flags": []})
    real = llm.__call__

    def vague(system, user, cache_tag="general", **kw):
        if cache_tag == "merge_critic":
            return {"single_specific_remediation": True, "addresses_all_members": True, "component": "", "action": ""}
        return real(system, user, cache_tag=cache_tag)

    res = run_merge_agent(st, llm=vague, max_rounds=1)
    assert res.fine_to_merged["P_F0"] != res.fine_to_merged["P_F2"]
    assert any(e.get("vetoed") for e in res.log)


def test_hard_ceiling_vetoes_without_critic(no_llm, monkeypatch):
    monkeypatch.setattr("agentic_pipeline.merging.MERGE_DBCV_HARD_DROP", 0.35)
    st = three_blobs()
    llm = FakeLLM(lambda ids, u: {"merges": [merge_of(["P_F0", "P_F2"])] if "P_F2" in ids else [], "flags": []}, critic=True)
    res = run_merge_agent(st, llm=llm, max_rounds=1)
    assert res.fine_to_merged["P_F0"] != res.fine_to_merged["P_F2"]
    assert "merge_critic" not in llm.calls
    assert [e for e in res.log if e.get("vetoed")][0]["critic_verdict"]["hard_veto"] is True


def test_rejected_merge_not_reproposed(no_llm):
    st = three_blobs()
    llm = FakeLLM(lambda ids, u: {"merges": [merge_of(["P_F0", "P_F2"])] if "P_F2" in ids else [], "flags": []}, critic=False)
    # Un second tour n'a lieu que si le premier accepte quelque chose : on fusionne aussi F0/F1 la première fois
    first = {"done": False}

    def merge_fn(ids, u):
        if not first["done"] and "P_F1" in ids:
            first["done"] = True
            return {"merges": [merge_of(["P_F0", "P_F1"]), merge_of(["P_F3", "P_F2"])], "flags": []}
        return {"merges": [merge_of([i for i in ids if i != "P_F3"][:2] + ["P_F3"])], "flags": []}

    llm.merge_fn = merge_fn
    res = run_merge_agent(st, llm=llm, max_rounds=3)
    vetoed = [frozenset(e["members"]) for e in res.log if e.get("vetoed")]
    # Aucun ensemble refusé n'est contenu dans une proposition ultérieure évaluée
    for i, v in enumerate(vetoed):
        assert not any(v <= w for w in vetoed[i + 1:])


def test_c_flag_splits_bimodal_cluster(no_llm):
    st = make_state([("P_F0", (0, 0), 40, "a"), ("P_F1", (12, 0), 40, "a"), ("P_F2", (30, 30), 40, "b")])
    # F0 et F1 sont réunis par erreur sous une seule étiquette fine
    st.fine_labels[st.fine_labels == "P_F1"] = "P_F0"
    del st.cards["P_F1"], st.titles["P_F1"]
    llm = FakeLLM(lambda ids, u: {"merges": [], "flags": [{"fine_cluster_id": "P_F0", "reason": "bimodal"}]} if "P_F0" in ids else {})
    res = run_merge_agent(st, llm=llm)
    assert {"P_F0_s0", "P_F0_s1"} <= set(res.fine_to_merged)
    assert "P_F0" not in res.fine_to_merged
    assert sorted(np.bincount(np.unique(res.fine_labels[:80], return_inverse=True)[1])) == [40, 40]
    assert "(sous-groupe" in res.titles["P_F0_s0"]["fine_title"]
    flag = [e for e in res.log if e["action"] == "flag_heterogeneous"][0]
    assert flag["result_id"] == ["P_F0_s0", "P_F0_s1"]
    check_invariants(st, res)


def test_c_relabel_callback_used(no_llm):
    st = make_state([("P_F0", (0, 0), 40, "a"), ("P_F1", (12, 0), 40, "a"), ("P_F2", (30, 30), 40, "b")])
    st.fine_labels[st.fine_labels == "P_F1"] = "P_F0"
    llm = FakeLLM(lambda ids, u: {"merges": [], "flags": [{"fine_cluster_id": "P_F0", "reason": "x"}]} if "P_F0" in ids else {})
    seen = {}

    def relabel(labels, new_ids):
        seen["ids"] = new_ids
        return (
            {i: {"keywords": ["k"], "representative_texts": [], "size": 1} for i in new_ids},
            {i: {"fine_title": f"neuf {i}", "failure_mode": "m"} for i in new_ids},
        )

    res = run_merge_agent(st, llm=llm, relabel=relabel)
    assert seen["ids"] == ["P_F0_s0", "P_F0_s1"]
    assert res.titles["P_F0_s1"]["fine_title"] == "neuf P_F0_s1"


def test_d_stops_after_max_rounds(no_llm):
    blobs = [(f"P_F{i}", (i * 1.5, 0), 30, "a") for i in range(8)]
    st = make_state(blobs)

    def always_merge(ids, u):
        return {"merges": [merge_of(ids[:2])] if len(ids) >= 2 else [], "flags": []}

    res = run_merge_agent(st, llm=FakeLLM(always_merge), max_rounds=MERGE_MAX_ROUNDS)
    assert max(e["round"] for e in res.log) <= 3
    res1 = run_merge_agent(st, llm=FakeLLM(always_merge), max_rounds=1)
    assert max(e["round"] for e in res1.log) == 1
    check_invariants(st, res)


def test_stops_when_nothing_accepted(no_llm):
    st = three_blobs()
    llm = FakeLLM()
    res = run_merge_agent(st, llm=llm)
    assert res.log == []
    assert len(set(res.fine_to_merged.values())) == 4
    n_calls_one_round = len(llm.calls)
    assert n_calls_one_round >= 1
    check_invariants(st, res)


def test_transitive_merge_across_rounds(no_llm):
    st = make_state([(f"P_F{i}", (i * 1.5, 0), 30, "a") for i in range(3)])
    state = {"round": 0}

    def fn(ids, u):
        if len(ids) < 2:
            return {}
        state["round"] += 1
        return {"merges": [merge_of(ids[:2])], "flags": []}

    res = run_merge_agent(st, llm=FakeLLM(fn), max_rounds=3)
    assert len(set(res.fine_to_merged.values())) == 1
    assert res.merged_members["P_M0"] == ["P_F0", "P_F1", "P_F2"]
    assert any(len(e["members"]) == 3 for e in res.log if e["action"] == "merge")


def test_e_unknown_ids_and_bad_responses_ignored(no_llm):
    st = three_blobs()
    bad = [
        {"merges": [merge_of(["P_F0", "ZZZ"])], "flags": [{"fine_cluster_id": "ZZZ", "reason": "?"}]},
        {"merges": "pas une liste", "flags": None},
        ["liste"],
        {"merges": [{"members": "P_F0"}, {"members": ["P_F0", "P_F0"]}, 3]},
    ]
    for resp in bad:
        res = run_merge_agent(st, llm=FakeLLM(lambda ids, u, r=resp: r))
        assert len(set(res.fine_to_merged.values())) == 4
        assert res.log == []
        check_invariants(st, res)


def test_cluster_in_two_merges_keeps_first(no_llm):
    st = make_state([(f"P_F{i}", (i * 1.5, 0), 30, "a") for i in range(3)])
    fn = lambda ids, u: {"merges": [merge_of(["P_F0", "P_F1"]), merge_of(["P_F1", "P_F2"])], "flags": []} if len(ids) == 3 else {}
    res = run_merge_agent(st, llm=FakeLLM(fn), max_rounds=1)
    assert res.fine_to_merged["P_F0"] == res.fine_to_merged["P_F1"] != res.fine_to_merged["P_F2"]


def test_llm_exception_does_not_crash(no_llm):
    def boom(*a, **k):
        raise RuntimeError("réseau")

    st = three_blobs()
    res = run_merge_agent(st, llm=boom)
    assert len(res.fine_to_merged) == 4
    check_invariants(st, res)


def test_prompt_size_capped_and_empty_title(no_llm):
    from agentic_pipeline.merging import PROMPT_MAX_CHARS, _Node, _group_prompt, _node_block

    cards, titles, nodes = {}, {}, {}
    for n in range(8):
        mem = [f"P_F{n}_{m}" for m in range(5)]
        for f in mem:
            cards[f] = {"keywords": ["a"] * 8, "representative_texts": ["x" * 700, "y" * 700]}
            titles[f] = {"fine_title": "t" * 80, "failure_mode": "m" * 200}
        nodes[f"N{n}"] = _Node(f"N{n}", mem)
    cur = np.array([f"N{n}" for n in range(8)], dtype=object)
    out = _group_prompt("P", list(nodes), nodes, cur, titles, cards)
    assert len(out) < PROMPT_MAX_CHARS
    assert all(f"### Cluster N{n}" in out for n in range(8))
    blk = _node_block(_Node("Z", ["P_Z"]), 1, {"P_Z": {"fine_title": ""}}, {})
    assert "Membre P_Z : P_Z" in blk


def test_split_refused_when_subgroups_get_same_title(no_llm):
    # Le labelliseur donne le même intitulé aux deux sous-groupes : pas un mélange de modes
    st = make_state([("P_F0", (0, 0), 40, "a"), ("P_F1", (12, 0), 40, "a"), ("P_F2", (30, 30), 40, "b")])
    st.fine_labels[st.fine_labels == "P_F1"] = "P_F0"
    del st.cards["P_F1"], st.titles["P_F1"]
    llm = FakeLLM(lambda ids, u: {"merges": [], "flags": [{"fine_cluster_id": "P_F0", "reason": "x"}]} if "P_F0" in ids else {})

    def relabel(labels, new_ids):
        return {}, {n: {"fine_title": "Arrêt automatique vent excessif", "failure_mode": ""} for n in new_ids}

    res = run_merge_agent(st, llm=llm, relabel=relabel)
    assert "P_F0" in res.fine_to_merged and "P_F0_s0" not in res.fine_to_merged
    flag = [e for e in res.log if e["action"] == "flag_heterogeneous"][0]
    assert flag["accepted"] is False and "indiscernables" in flag["justification"]
