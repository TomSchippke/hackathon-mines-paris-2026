"""Test de fumée du pipeline complet : embeddings et LLM factices, aucun appel réseau."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_pipeline import clustering, pipeline
from agentic_pipeline.precursors import mine_escalations, mine_precursors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "incidents_full.csv"

EXPECTED_COLUMNS = {
    "incidents_segmented": ["incident_id", "fine_cluster_id", "merged_cluster_id", "cluster_id",
                            "problem_title", "common_cause", "cluster_criticality_tier",
                            "assignment_source"],
    "clusters_fine": ["fine_cluster_id", "partition", "merged_cluster_id", "size", "fine_title",
                      "failure_mode", "keywords"],
    "clusters_merged": ["merged_cluster_id", "partition", "fine_cluster_ids", "criticality_rank",
                        "criticality_tier", "frequency", "mean_cost_eur", "total_cost_eur",
                        "mean_downtime_hours", "total_downtime_hours", "problem_title",
                        "problem_description", "common_cause", "common_cause_quotes",
                        "common_cause_source", "system_impact", "prevented_outcomes",
                        "recommendations", "early_warning_rules", "nominal", "nominal_reason"],
    "concentration": ["merged_cluster_id", "dimension", "value", "observed", "expected", "lift",
                      "p_value", "q_value", "significant"],
    "precursor_pairs": ["kind", "partition", "scope_cluster", "antecedent", "consequent",
                        "window_days", "n_hits", "n_equipments", "n_consequent", "expected_hits",
                        "lift", "p_value", "q_value", "is_best_window", "significant", "plausible",
                        "mechanism", "monitoring_rule"],
    "precursor_evidence": ["rule_id", "consequent_incident_id", "equipment_id", "consequent_date",
                           "antecedent_incident_ids", "antecedent_dates", "n_antecedents",
                           "window_days"],
    "fine_sweep": ["partition", "min_cluster_size", "n_clusters", "outlier_rate", "dbcv"],
}


def _fake_embed(texts, cache_dir=None, **kwargs):
    """Vecteurs 64-d regroupés autour de 6 centres (affectation déterministe par texte)."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(6, 64))
    out = np.empty((len(texts), 64))
    for i, t in enumerate(texts):
        h = zlib.crc32(str(t).encode())
        r = np.random.default_rng(h)
        out[i] = centers[h % 6] + 0.15 * r.normal(size=64)
    return out


class FakeLLM:
    def __init__(self):
        self.calls: dict[str, int] = {}

    def __call__(self, system_prompt, user_prompt, cache_tag="general", **kw):
        self.calls[cache_tag] = self.calls.get(cache_tag, 0) + 1
        h = zlib.crc32(user_prompt.encode()) % 100000
        if cache_tag == "fine_label":
            return {"fine_title": f"Défaut type {h}", "failure_mode": f"Mode de défaillance {h}"}
        if cache_tag == "outlier_reassignment":
            ids = re.findall(r"^- ID: (\S+) \|", user_prompt, flags=re.M)
            incs = re.findall(r"^Incident ID (\S+) :", user_prompt, flags=re.M)
            return {"classifications": [
                {"incident_id": inc, "assigned_cluster_id": ids[j % len(ids)] if j % 2 == 0 else "inconnu",
                 "reason": "test"} for j, inc in enumerate(incs)]}
        if cache_tag == "merge":
            ids = re.findall(r"^### Cluster (\S+) \(taille", user_prompt, flags=re.M)
            if len(ids) < 2:
                return {"merges": [], "flags": []}
            return {"merges": [{"members": ids[:2], "shared_remediation": "Remplacer le composant X",
                                "prevented_outcomes": [{"fine_cluster_id": i, "outcome": f"panne {i}"} for i in ids[:2]],
                                "justification": "même action"}],
                    "flags": [{"fine_cluster_id": "id_inconnu", "reason": "ignoré"}]}
        if cache_tag == "merge_critic":
            return {"component": "joint", "action": "remplacer", "addresses_all_members": True,
                    "single_specific_remediation": True, "reason": "ok"}
        if cache_tag == "precursor_check":
            return {"plausible": True, "mechanism": "dégradation progressive", "monitoring_rule": "si 2 A en 30 j -> inspecter"}
        if cache_tag == "nominal_check":
            return {"protection_only": True, "defect_found": False, "reason": "test"}
        if cache_tag == "diagnosis_v3":
            return {"problem_title": f"Diagnostic {h}", "problem_description": "desc", "common_cause": "cause",
                    "system_impact": "impact", "prevented_outcomes": ["arrêt machine"],
                    "common_cause_quotes": ["citation inventée absente des rapports"],
                    "recommendations": [{"action": "Inspecter", "scope": "equipment", "targets": ["EQUIP-INVENTE"],
                                         "rationale": "r"}],
                    "early_warning_rules": [{"rule_id": "inventé", "rule": "x", "action": "y"}]}
        raise AssertionError(f"cache_tag inattendu : {cache_tag}")


@pytest.fixture
def sample_csv(tmp_path):
    if not DATA.exists():
        pytest.skip("data/incidents_full.csv absent")
    df = pd.read_csv(DATA)
    sample = pd.concat([df[df["asset_type"] == a].sample(n=100, random_state=1)
                        for a in sorted(df["asset_type"].unique())])
    p = tmp_path / "sample.csv"
    sample.to_csv(p, index=False)
    return p, len(sample)


def test_missing_columns_error(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"incident_id": [1], "text": ["x"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Colonnes obligatoires absentes"):
        pipeline.run_agentic_segmentation(p, tmp_path / "o", llm=lambda *a, **k: {})


def test_pipeline_smoke(sample_csv, tmp_path, monkeypatch, no_llm):
    csv, n = sample_csv
    monkeypatch.setattr("agentic_pipeline.clustering.embed_texts", _fake_embed)
    monkeypatch.setattr(pipeline, "PRECURSOR_N_PERMUTATIONS", 100)
    # Seuils assouplis pour exercer plausibilité + preuves même sur 200 lignes
    monkeypatch.setattr(pipeline, "mine_precursors", lambda *a, **k: mine_precursors(
        *a, min_hits=2, min_equipments=1, fdr=1.0, **k))
    monkeypatch.setattr(pipeline, "mine_escalations", lambda *a, **k: mine_escalations(
        *a, min_hits=2, min_equipments=1, fdr=1.0, **k))
    # Force des isolés (bruit HDBSCAN) pour exercer reclassement et cluster isole_atypique
    real_sweep = clustering.sweep_hdbscan

    def sweep_with_noise(X, sizes=None):
        res = real_sweep(X, sizes) if sizes is not None else real_sweep(X)
        labels = np.array(res.labels)
        labels[::7] = -1
        return clustering.SweepResult(labels, res.min_cluster_size, res.dbcv, res.table)

    monkeypatch.setattr(clustering, "sweep_hdbscan", sweep_with_noise)
    llm = FakeLLM()
    out = tmp_path / "out"
    paths = pipeline.run_agentic_segmentation(csv, out, llm=llm)

    assert set(paths) >= set(EXPECTED_COLUMNS) | {"merge_log"}
    for name, cols in EXPECTED_COLUMNS.items():
        assert paths[name].exists(), name
        got = pd.read_csv(paths[name]).columns
        assert set(cols) <= set(got), (name, set(cols) - set(got))
    assert not (out / "clusters_summary.csv").exists()
    log = json.loads(paths["merge_log"].read_text(encoding="utf-8"))
    assert isinstance(log, list)
    overview = json.loads(paths["cost_overview"].read_text(encoding="utf-8"))
    assert overview["incidents"] == n and overview["years"] and len(overview["severity"]) == 3
    assert {p["partition"] for p in overview["partitions"]} == {"solar", "wind"}

    seg = pd.read_csv(paths["incidents_segmented"])
    assert len(seg) == n
    assert seg["fine_cluster_id"].notna().all() and seg["merged_cluster_id"].notna().all()
    assert (seg["cluster_id"] == seg["merged_cluster_id"]).all()
    assert set(seg["assignment_source"]) <= {"dense_cluster", "llm_reassigned", "unclassified_outlier"}
    merged = pd.read_csv(paths["clusters_merged"])
    assert set(seg["merged_cluster_id"]) == set(merged["merged_cluster_id"])
    for c in ("prevented_outcomes", "recommendations", "early_warning_rules", "fine_cluster_ids"):
        assert all(isinstance(json.loads(v), list) for v in merged[c])
    # cibles inventées filtrées par la validation du diagnostic ; citation inventée : cause non justifiée
    diagnosed = merged[merged["merged_cluster_id"] != "isole_atypique"]
    assert (diagnosed["common_cause_source"] == "non_citee").all()
    assert all(t not in str(r) for r in merged["recommendations"] for t in ["EQUIP-INVENTE"])
    print("sources:", seg["assignment_source"].value_counts().to_dict(), "calls:", llm.calls,
          "evidence rows:", len(pd.read_csv(paths["precursor_evidence"])))
    assert llm.calls.get("outlier_reassignment", 0) > 0
    assert "isole_atypique" in set(seg["merged_cluster_id"])
    assert llm.calls.get("fine_label", 0) > 0 and llm.calls.get("diagnosis_v3", 0) > 0
