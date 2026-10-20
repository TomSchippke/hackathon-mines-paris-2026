> **Note (document historique).** Ce plan est la spécification v2 d'origine. Il n'est plus la référence : le
> code actuel est décrit dans [`agentic_pipeline/README.md`](../agentic_pipeline/README.md). Depuis ce plan :
> - l'application Streamlit (`webapp/`, `.streamlit/`) est supprimée ; un rapport HTML statique est à venir ;
> - `agent.py` (reclassement LLM des isolés, décision D12) est supprimé : chaque isolé HDBSCAN devient un cluster
>   fin singleton, étiqueté puis soumis à l'agent de fusion ; un singleton jamais fusionné devient `isole_atypique` ;
> - l'« escalade » au niveau des clusters fins (décision D4, type `escalation`, `scope_cluster`) est remplacée par une
>   veille d'escalade descriptive (`escalation.py`, `escalations.csv`) : séries équipement × cluster fusionné ;
> - l'audit par technicien du module `equipment_analysis` est supprimé (récidives à fenêtre fixe aussi) ;
> - le diagnostic ajoute un critique de recommandations, des cibles d'escalade, une cause commune citée ou inférée,
>   et un fonctionnement nominal exigeant une condition extérieure citée ;
> - la criticité classe par coût total (paliers Pareto 70 / 90 / 98 %).

# Implementation plan — Clustering v2 (remediation clusters, concentration, precursors)

Shared spec for the orchestrator and the sub-agents. **Read sections 1–3 before any task.**
Each task in section 4 is self-contained: files owned, interface, acceptance criteria, verification.

---

## 1. Decisions (frozen, do not re-litigate)

| # | Decision |
|---|---|
| D1 | A merged cluster = **one distinct remediation pathway**. DBCV is a guardrail. Every merged cluster lists the distinct failure outcomes its fix prevents. Recommendations name specific sites / equipments. |
| D2 | Clustering features = **text embeddings only**. Site / equipment / period are used *after* clustering, in a concentration profile (lift vs expected + test). |
| D3 | Precursors: **fleet-level statistical mining vs a null model** → LLM plausibility check on significant pairs → per-incident evidence in the report. |
| D4 | **Two levels**: `fine_cluster_id` (failure mode, HDBSCAN) and `merged_cluster_id` (remediation, merge agent). Precursor mining on merged level + fine-level escalation inside each merged cluster. |
| D5 | Merge agent: actions = `merge` and `flag_heterogeneous` only (flag → deterministic HDBSCAN re-cluster of that cluster). Works on local candidate groups. DBCV drop ≤ 15 % → accept; 15–35 % → strict critic must name one component + one action covering every member; > 35 % → hard veto (amended after real runs: the same-model critic was too lenient). A rejected set of fine clusters is never re-proposed. ≤ 3 rounds. `merge_log.json` is auditable. |
| D6 | **Domain-agnostic, not schema-agnostic.** The `incidents_full.csv` columns are a fixed contract. Domain literals (severity order, language, stopwords) live only in `agentic_pipeline/config.py`. Remove column auto-detection. |
| D7 | Precursor stats: sev(A) ≤ sev(B); A→A = `recurrence`; windows {7, 30, 60} d, best window per pair; Benjamini–Hochberg per kind; null = within-equipment label permutation (10 000, see §3.5); support ≥ 5 hits on ≥ 3 distinct equipments. |
| D8 | `topics` step removed from the default `main.py` run (module kept, not deleted). `agentic_pipeline` is the single clustering. |
| D9 | Validation = synthetic injection `pytest` for precursor mining only. |
| D10 | Webapp report = one card per remediation pathway (fix, prevented outcomes, where to act, early-warning rules, evidence, fine drill-down). Final diagnosis runs **last** on merged clusters with concentration + precursor context. |
| D11 | Partition by `asset_type` is kept (`PARTITION_COL`). Sweep, merges and fine escalation stay inside a partition. |
| D12 | Outlier reassignment targets **fine** clusters, before merging. |

## 2. Ground rules for every sub-agent

- Python: `.venv/bin/python` (no `python` on PATH, no pip in the venv: `uv pip install --python .venv/bin/python <pkg>` — but do not add dependencies). Tests: `.venv/bin/python -m pytest`.
- `tests/conftest.py` puts the repo root on `sys.path` and provides a `no_llm` fixture. `pipeline.py` is expected to be broken (imports removed APIs) until T9 — do not import it in Wave 1/2 tests. `DEFAULT_SPLIT_COL` is still in config until T9 removes it.
- **Only edit the files your task owns.** Do not touch untracked dirs (`analysis_agent/`, `cause_extraction/`, `framing/`, `llm/`, `review_agent/`), `pdf_pipeline/`, `topic_modeling/` (except importing `embed_texts`), `data/`, `outputs/` (except writing new outputs during an authorized run).
- Match surrounding style: French docstrings/comments, `from __future__ import annotations`, type hints, module-level constants in `config.py`.
- **No network / LLM calls in tests.** Monkeypatch `agentic_pipeline.llm_client.call_llm_json` (or the name imported into your module). Real LLM runs only in task T9-run, done by the orchestrator.
- Every LLM call goes through `call_llm_json(system_prompt, user_prompt, cache_tag=...)` (disk-cached, JSON mode). Validate LLM output defensively: unknown IDs are dropped, missing keys get defaults — never crash on a bad response.
- Pure-stats functions must be deterministic given `random_state`.
- Do not commit. Report back: files changed, commands run with their output (tail), any deviation from the contract and why.

## 3. Contracts

### 3.1 `agentic_pipeline/config.py` (written by orchestrator in Wave 0)

New constants (existing ones kept unless stated):

```python
LANGUAGE = "fr"
SEVERITY_ORDER = ["mineure", "modérée", "sévère"]          # rang 0 < 1 < 2
PARTITION_COL = "asset_type"                               # remplace DEFAULT_SPLIT_COL
EQUIPMENT_COL = "equipment_id"
SEVERITY_COL = "severity"
CONCENTRATION_COLS = ["site", "equipment_id"]
CONCENTRATION_PERIOD = "Q"                                 # rafale temporelle par trimestre
CONCENTRATION_MIN_LIFT = 1.5
CONCENTRATION_MIN_OBS = 3
CONCENTRATION_FDR = 0.10

MCS_SWEEP = [5, 7, 10, 15, 20, 25]                         # min_cluster_size balayés par partition
MIN_SAMPLES = None                                         # défaut hdbscan (= min_cluster_size)

MERGE_NEIGHBOURS = 4                                       # k plus proches centroïdes par cluster fin
MERGE_MAX_GROUP = 8                                        # clusters max par prompt
MERGE_MAX_ROUNDS = 3
MERGE_DBCV_MAX_DROP = 0.15                                 # chute relative tolérée sans critique

PRECURSOR_WINDOWS = [7, 30, 60]
PRECURSOR_N_PERMUTATIONS = 1000
PRECURSOR_FDR = 0.05
PRECURSOR_MIN_HITS = 5
PRECURSOR_MIN_EQUIPMENTS = 3
RANDOM_STATE = 42

OUTLIER_LABEL = "isole_atypique"
```

Removed: `DEFAULT_SPLIT_COL` (→ `PARTITION_COL`). Column-name constants stay (`TEXT_COL`, `COST_COL`, …).

### 3.2 Module layout (`agentic_pipeline/`)

| Module | Owner task | Public API |
|---|---|---|
| `clustering.py` | T2 | `embed_partition(texts) -> np.ndarray` · `reduce_umap(emb) -> np.ndarray` · `dbcv_score(X_umap, labels) -> float` · `sweep_hdbscan(X_umap, sizes=MCS_SWEEP) -> SweepResult` · existing `extract_ctfidf_keywords`, `select_representative_documents` kept |
| `labeling.py` (new) | T3 | `label_fine_clusters(cards: dict[str, FineCard]) -> dict[str, dict]` · `build_fine_cards(...)` |
| `agent.py` | T3 | `reassign_outliers_batch` retargeted to fine taxonomy (same signature); old `diagnose_cluster` removed by T8 |
| `merging.py` (new) | T4 | `run_merge_agent(fine: FineState) -> MergeResult` |
| `concentration.py` (new) | T5 | `compute_concentration(df, cluster_col, partition_col) -> pd.DataFrame` |
| `precursors.py` (new) | T6 | `mine_precursors(df, label_col, ...) -> pd.DataFrame` · `extract_evidence(df, pairs, label_col) -> pd.DataFrame` |
| `precursor_agent.py` (new) | T7 | `check_plausibility(pairs, cards) -> pd.DataFrame` |
| `diagnosis.py` (new) | T8 | `diagnose_merged_cluster(context: dict) -> dict` |
| `pipeline.py` | T9 | `run_agentic_segmentation(input_csv=None, out_dir=DEFAULT_OUT_DIR) -> dict[str, Path]` |
| `criticality.py` | — | unchanged |

`SweepResult` (dataclass in `clustering.py`): `labels: np.ndarray`, `min_cluster_size: int`, `dbcv: float`, `table: pd.DataFrame` (columns `min_cluster_size, n_clusters, outlier_rate, dbcv`).
Selection rule: max DBCV among settings with `outlier_rate <= 0.35` and `n_clusters >= 2`; tie → smaller `min_cluster_size` (finer). If none qualifies, max DBCV overall.

`dbcv_score`: `hdbscan.validity.validity_index(X.astype(np.float64), labels)` ignoring noise (-1); returns `nan`-safe float (`-1.0` if < 2 clusters or on error).

### 3.3 Identifiers

- Fine: `f"{partition}_F{k}"` (k = HDBSCAN label); sub-clusters from a heterogeneity re-cluster: `f"{parent}_s{j}"`.
- Merged: `f"{partition}_M{n}"`, n sequential by descending size.
- Outliers that stay unassigned: `OUTLIER_LABEL` for both levels. They are **excluded** from merging, precursor mining and concentration, but kept in criticality and outputs.

### 3.4 Output files (`outputs/segmentation/`)

| File | Grain | Key columns |
|---|---|---|
| `incidents_segmented.csv` | incident | all input cols + `fine_cluster_id`, `merged_cluster_id`, **`cluster_id` (= merged, alias kept for `equipment_analysis` & webapp)**, `problem_title`, `common_cause`, `cluster_criticality_tier`, `assignment_source` ∈ {`dense_cluster`, `llm_reassigned`, `unclassified_outlier`} |
| `clusters_fine.csv` | fine cluster | `fine_cluster_id, partition, merged_cluster_id, size, fine_title, failure_mode, keywords` (`;`-joined) |
| `clusters_merged.csv` | merged cluster | `merged_cluster_id, partition, fine_cluster_ids, criticality_rank, criticality_tier, frequency, mean/total_cost_eur, mean/total_downtime_hours, problem_title, problem_description, common_cause, system_impact, prevented_outcomes (JSON list), recommendations (JSON list), early_warning_rules (JSON list)` |
| `merge_log.json` | list | `{round, action: merge|flag_heterogeneous, members, result_id, shared_remediation, prevented_outcomes:[{fine_cluster_id, outcome}], justification, dbcv_before, dbcv_after, vetoed, critic_verdict}` |
| `concentration.csv` | cluster × dimension × value | `merged_cluster_id, dimension (site|equipment_id|period), value, observed, expected, lift, p_value, q_value, significant` |
| `precursor_pairs.csv` | pair × window | `kind (precursor|recurrence|escalation), partition, scope_cluster (merged id for escalation else ""), antecedent, consequent, window_days, n_hits, n_equipments, n_consequent, expected_hits, lift, p_value, q_value, is_best_window, significant, plausible, mechanism, monitoring_rule` |
| `precursor_evidence.csv` | consequent incident × rule | `rule_id, consequent_incident_id, equipment_id, consequent_date, antecedent_incident_ids (;-joined), antecedent_dates, n_antecedents, window_days` |
| `fine_sweep.csv` | partition × mcs | sweep table per partition |
| `clusters_summary.csv` | — | **removed** (webapp reads `clusters_merged.csv`) |

### 3.5 Precursor algorithm (normative — T6 implements, T10 tests against it)

Input: incidents with `equipment_id`, parsed `datetime`, `label` (merged or fine), `sev_rank` (from `SEVERITY_ORDER`), outliers removed.

For a pair (A, B), A ≠ B, window W:
- `hit(b)` = ∃ incident a on the same equipment with label A, `t_b − W ≤ t_a < t_b` (strictly before; same-day excluded), and `sev_rank(a) ≤ sev_rank(b)`.
- `n_hits` = number of B incidents with `hit(b)`; `n_equipments` = distinct equipments among them.
- A = B → `kind="recurrence"` (same `hit` definition, a ≠ b), computed and output but **not** sent to plausibility nor shown as early-warning rule.
- Support filter **before** testing: `n_hits ≥ PRECURSOR_MIN_HITS` and `n_equipments ≥ PRECURSOR_MIN_EQUIPMENTS`. Pairs failing it are output with `p_value = NaN`, `significant=False`.
- Null model: for each permutation, **within each equipment**, shuffle the (label, sev_rank) tuples across that equipment's timestamps. Recompute `n_hits` for all tested pairs × windows in the same pass. `expected_hits` = mean null; `lift = n_hits / max(expected_hits, 1e-9)`; `p = (1 + #{null ≥ obs}) / (1 + N)`.
- BH correction over the tested (pair, window) rows of one call, **one pool per `kind`** (precursor/escalation vs recurrence). 10 000 permutations by default: with 1 000 the p-value floor (1e-3) makes q ≤ 0.05 unreachable after BH over ~200 tests (found by T10). `is_best_window` = min q per pair (tie → smaller W). `significant = is_best_window & q ≤ PRECURSOR_FDR & lift > 1`.
- Escalation: within each merged cluster with ≥ 2 fine clusters, run the same on `fine_cluster_id` restricted to that cluster's incidents; `kind="escalation"`, `scope_cluster` set. Same BH pool as the merged-level call? **No** — one BH per call; escalation is one call over all merged clusters.
- Performance: 1800 incidents, ~25 labels/partition, 1000 perms, 3 windows → must run in < 60 s. Vectorize per equipment (precompute, for each incident, the indices of prior incidents within max(W) on the same equipment; permutations only re-index labels).

## 4. Tasks

Model column: **H** = Haiku, **S** = Sonnet. Each task is implemented by one agent and verified by a **separate** agent (V-column model) that did not write the code.

### Wave 0 — orchestrator (no sub-agent)
- Write `agentic_pipeline/config.py` constants (§3.1); add `pytest>=8.0` to `requirements.txt`; install it into `.venv`; create `tests/__init__.py`, `tests/conftest.py` (fixture `no_llm` that monkeypatches `call_llm_json` to raise).
- Snapshot current `outputs/segmentation/` to `outputs/segmentation_v1/` for comparison.

### Wave 1 — independent, run in parallel

**T2 · Fine clustering with DBCV sweep** — S / V: S
- Owns: `agentic_pipeline/clustering.py`.
- Implement §3.2 API. Keep embedding via `topic_modeling.embeddings.embed_texts`. Remove `cluster_texts` (replaced by the composable functions). Keep `get_cluster_metadata_profile` (used by labeling).
- Accept: on synthetic blobs (`sklearn.datasets.make_blobs`, 3 centers, 300 pts) `sweep_hdbscan` finds 3 clusters and DBCV > 0.3; selection rule of §3.2 holds; `dbcv_score` returns -1.0 for a single cluster without raising.
- Verify: `tests/test_clustering.py` (written by implementer, no network); `.venv/bin/python -m pytest tests/test_clustering.py -q`.

**T5 · Concentration profile** — S / V: S
- Owns: `agentic_pipeline/concentration.py`, `tests/test_concentration.py`.
- For each merged cluster (excluding outliers) and each dimension in `CONCENTRATION_COLS` + `period` (date → `CONCENTRATION_PERIOD`): expected = cluster size × share of the value **within the cluster's partition**; one-sided binomial test (`scipy.stats.binomtest`, alternative="greater"); BH within each cluster; `significant = q ≤ CONCENTRATION_FDR & lift ≥ CONCENTRATION_MIN_LIFT & observed ≥ CONCENTRATION_MIN_OBS`. Output schema §3.4.
- Accept: synthetic test where 80 % of cluster X sits on equipment E (base share 5 %) → E significant; uniform cluster → nothing significant.
- Verify: pytest file above. Check `scipy` is installed (it is a transitive dep of hdbscan); do not add new deps otherwise.

**T6 · Precursor mining (stats only)** — S / V: S
- Owns: `agentic_pipeline/precursors.py`.
- Implement §3.5 exactly, plus `extract_evidence(df, pairs, label_col)` producing `precursor_evidence.csv` rows for rows with `significant & plausible` (if `plausible` column missing, use `significant`). `rule_id = f"{antecedent}->{consequent}@{window_days}d"`.
- Accept: T10's tests pass; runtime on real data (merged labels can be faked with current `outputs/segmentation_v1/incidents_segmented.csv` `cluster_id`) < 60 s — print timing.
- Verify: `.venv/bin/python -m pytest tests/test_precursors.py -q` and a timing script in the scratchpad.

**T10 · Synthetic injection test for precursors** — S (different agent from T6) / V: orchestrator
- Owns: `tests/test_precursors.py`. Written **from §3.5 only**, without reading T6's implementation (black-box), against `mine_precursors(df, label_col="label", windows=..., n_permutations=..., random_state=...)`.
- Synthetic generator inside the test file (no real data): 40 equipments, ~3 years, Poisson incident process, 6 labels, random severities.
- Cases:
  1. **Planted pattern**: on 12 equipments, before 40 % of `Y` incidents insert 2 `X` incidents 3–20 days earlier with severity ≤ Y → (X, Y) significant, best window = 30, lift > 2.
  2. **Null**: fully random data → no `kind="precursor"` row significant (allow ≤ 1 false positive? **No: assert 0** with fixed seed; if flaky, fix seed, not threshold).
  3. **Lemon machine**: pattern planted on 1 equipment only (many hits) → not significant (min equipments).
  4. **Severity rule**: planted X always more severe than Y → not counted as hits.
  5. **Recurrence**: Y→Y tagged `kind="recurrence"`, never `precursor`.
  6. Use `n_permutations=200` in tests for speed; whole file < 30 s.
- Verify: orchestrator runs it against T6; failures are triaged (spec ambiguity → orchestrator decides and updates §3.5; implementation bug → back to T6).

**T11 · equipment_analysis fixes** — H / V: S
- Owns: `equipment_analysis/config.py`, `equipment_analysis/metrics.py`.
- Replace the literal `"critique"` with the highest level of `agentic_pipeline.config.SEVERITY_ORDER` (rename column to `severe_incidents`, keep meaning "top severity"). `CLUSTER_COL` stays `cluster_id` (alias = merged). `PROBLEM_COL` unchanged.
- Accept: on `data/incidents_full.csv`, `severe_incidents` sum equals `(severity == "sévère").sum()` (210).
- Verify: one-liner run of `run_equipment_analysis(input_csv="data/incidents_full.csv", out_dir=<scratchpad>)` and the check above; grep shows no `"critique"` left in `equipment_analysis/`. Check `webapp/app.py` for references to `critical_incidents` and **report** them (do not edit — T13 owns the webapp).

**T12 · main.py** — H / V: S
- Owns: `main.py`.
- Default steps = `["pdf", "segmentation", "equipment"]` (`topics` still selectable). Segmentation call matches the new `run_agentic_segmentation(input_csv, out_dir)` signature; remove `--min-cluster-size` wiring into segmentation (the sweep decides) but keep it for `topics`. Update module docstring.
- Accept: `.venv/bin/python main.py --help` works; `python -c "import main"` OK (it will fail until T9 lands if the signature differs — use keyword args matching §3.2 and verify after Wave 3).

### Wave 2 — depends on T2 (merged into tree and verified)

**T3 · Fine labeling + outlier retarget** — S / V: S
- Owns: `agentic_pipeline/labeling.py`, `agentic_pipeline/agent.py` (outlier part only).
- `build_fine_cards`: per fine cluster → keywords (c-TF-IDF), 4 representative texts, metadata profile, size, severity distribution, mean cost.
- `label_fine_clusters`: **cheap** LLM call per cluster (cache_tag `fine_label`) → `{"fine_title", "failure_mode"}` (one sentence each). No remediation here.
- Outlier reassignment: taxonomy = fine clusters of the **same partition** (`fine_title` + `failure_mode`), batches of `OUTLIER_BATCH_SIZE`; invalid IDs → `OUTLIER_LABEL`.
- Accept/Verify: pytest with monkeypatched LLM returning canned JSON, incl. malformed responses (missing keys, unknown cluster id, non-list) → no crash, defaults applied. `tests/test_labeling.py`.

**T4 · Merge agent** — S / V: S
- Owns: `agentic_pipeline/merging.py`, `tests/test_merging.py`.
- Input `FineState` (dataclass): per partition `X_umap`, `embeddings`, current fine labels (array of fine ids), fine cards + labels, texts.
- Per round:
  1. Candidate groups: for each fine cluster, `MERGE_NEIGHBOURS` nearest by centroid cosine (original embeddings) + pairs with near-identical titles (normalized token Jaccard ≥ 0.6). Union into connected groups; split groups > `MERGE_MAX_GROUP` by greedy nearest chunks.
  2. One LLM call per group (cache_tag `merge`) → `{"merges":[{"members":[...],"shared_remediation":str,"prevented_outcomes":[{"fine_cluster_id","outcome"}],"justification":str}], "flags":[{"fine_cluster_id","reason"}]}`. Prompt states the criterion: *merge only if one concrete remediation action addresses all members; list each member's outcome prevented*.
  3. For each merge: `dbcv_before/after` via `dbcv_score` on the partition with current vs merged labels. If relative drop > `MERGE_DBCV_MAX_DROP`: critic call (cache_tag `merge_critic`) → `{"single_specific_remediation": bool, "reason"}`; reject if false (`vetoed=True`).
  4. Flags: re-run `sweep_hdbscan` on that cluster's `X_umap` rows with `sizes=[s for s in MCS_SWEEP if s < cluster_size/2]`; accept split if ≥ 2 clusters and DBCV > 0; new fine ids `{parent}_s{j}`; new sub-clusters need labels → call T3's `label_fine_clusters` (inject as a callable param to keep tests offline).
  - Stop when a round produces no accepted merge or split, or after `MERGE_MAX_ROUNDS`. Merges are transitive across rounds (merged groups become nodes, centroid = mean of member points).
- Output `MergeResult`: `fine_to_merged: dict[str,str]`, `fine_labels` (possibly with splits), `merged_members: dict[str, list[str]]`, `log: list[dict]` (§3.4 schema).
- Accept: offline tests with a fake LLM: (a) two duplicate clusters get merged; (b) a merge with large DBCV drop and critic=false is vetoed and logged; (c) flag on a cluster made of 2 well-separated blobs splits it; (d) loop stops at ≤ 3 rounds; (e) unknown IDs in LLM output ignored.

**T7 · Precursor plausibility + rules** — S / V: S
- Owns: `agentic_pipeline/precursor_agent.py`, `tests/test_precursor_agent.py`.
- For each `significant` row with `kind ∈ {precursor, escalation}`: one LLM call (cache_tag `precursor_check`) with both clusters' cards (title, failure mode, 2 representative texts) + stats (window, lift, n_hits, n_equipments) → `{"plausible": bool, "mechanism": str, "monitoring_rule": str}`. The prompt must ask for a physically plausible mechanism and allow "plausible=false" (common external cause such as weather → mark false with mechanism "cause commune externe: …").
- Fills `plausible, mechanism, monitoring_rule` columns; non-significant rows: `plausible=False`, empty strings.
- Verify: offline test with fake LLM (valid, malformed, exception → defaults `plausible=False`).

**T8 · Final diagnosis of merged clusters** — S / V: S
- Owns: `agentic_pipeline/diagnosis.py`, `tests/test_diagnosis.py`; removes `diagnose_cluster` + `SYSTEM_DIAGNOSIS_PROMPT` from `agent.py` (coordinate: T3 also edits `agent.py` → **T8 runs after T3 is merged**, or T8 only adds `diagnosis.py` and the orchestrator deletes the old function in Wave 3).
- Context dict per merged cluster: member fine cards (+ titles/failure modes), merge-log entry (shared remediation, prevented outcomes), significant concentration rows, plausible precursor rules **into** this cluster (incl. escalation rules scoped to it) with evidence counts, impact metrics.
- Output JSON (cache_tag `diagnosis_v2`): `problem_title, problem_description, common_cause, system_impact, prevented_outcomes: [str], recommendations: [{"action","scope": "fleet"|"site"|"equipment","targets":[...],"rationale"}], early_warning_rules: [{"rule_id","rule","action"}]`.
- **Post-validation (hard rule)**: every `targets` entry must appear in the provided concentration values (or be empty for `fleet`); invented targets are dropped; if a recommendation loses all targets and scope ≠ fleet → scope becomes `fleet`. `early_warning_rules.rule_id` must be one of the provided rule ids, else dropped.
- Verify: offline test where fake LLM invents an equipment id → dropped; invented rule id → dropped.

### Wave 3 — integration

**T9 · Pipeline orchestration** — S / V: S, then orchestrator real run
- Owns: `agentic_pipeline/pipeline.py`, `agentic_pipeline/agent.py` (delete old diagnosis if T8 didn't).
- Flow: load (fixed schema, drop empty text, coerce numerics, `sev_rank`) → per partition: embed → UMAP → sweep → fine labels → fine cards + labeling → outlier reassignment → merge agent → assemble incident table → concentration (merged) → precursors (merged + escalation) → plausibility → evidence → diagnosis (merged) → criticality (`compute_cluster_criticality` on `merged_cluster_id`, outliers included) → write all §3.4 files → console top-8 print (adapted).
- Delete `validate_and_detect_columns`; assert required columns with a clear error listing missing ones.
- CLI: `--input`, `--out-dir`. (`--min-cluster-size` and `--no-split` removed.)
- Verify (offline): `tests/test_pipeline_smoke.py` — monkeypatch `embed_texts` (random but clustered vectors) and `call_llm_json` (canned generic answers), run on a 200-row sample of `data/incidents_full.csv` into a tmp dir; assert every §3.4 file exists with its columns, every incident has both ids, `cluster_id == merged_cluster_id`.
- **T9-run (orchestrator, real LLM)**: `.venv/bin/python main.py --steps segmentation equipment`. Checks: no duplicate-title merged clusters within a partition; `merge_log.json` non-empty; `precursor_pairs.csv` present; spot-read 3 merged-cluster diagnoses for equipment-specific targets; equipment step runs on new output.

### Wave 4 — presentation (parallel)

**T13 · Webapp action-plan cards** — S / V: S (+ orchestrator screenshot via `/run` if available)
- Owns: `webapp/app.py`.
- New first tab **"Plan d'action"**: one expandable card per merged cluster ordered by `criticality_rank`, with sections Fix / Prévient / Où agir (significant concentration rows, lift) / Signaux précurseurs (plausible rules, lift, window, n equipments) / Preuves (evidence rows, link to equipment tab via selected `equipment_id`) / Modes de défaillance (fine clusters). Filters: partition, tier.
- Existing tabs: switch from `clusters_summary.csv` to `clusters_merged.csv`. (T11: the webapp has no `critical_incidents` reference; `equipment_summary.csv` now exposes `severe_incidents` — may be shown in the equipment tab.)
- Verify: `.venv/bin/python -m py_compile webapp/app.py`; launch `streamlit run webapp/app.py --server.headless true` for 20 s and check no traceback in logs; verifier reviews that every JSON-list column is parsed with a fallback.

**T14 · READMEs** — H / V: orchestrator
- Owns: `README.md` (pipeline table), new `agentic_pipeline/README.md` (method: sweep, merge agent, concentration, precursors with null model and limits, outputs table from §3.4).

## 5. Execution & verification protocol (orchestrator)

1. Wave 0 myself. Then launch each wave's implementers in parallel (one agent per task, `isolation: worktree` not needed — file ownership is disjoint; T3/T8 `agent.py` overlap handled by ordering).
2. After each implementer returns: launch its verifier with the task section + the diff (`git diff -- <owned files>`). Verifier checks: acceptance criteria, contract conformity (§3), no edits outside owned files (`git status`), tests run and pass, no network in tests. Verifier returns PASS/FAIL with evidence. FAIL → send findings back to the **same** implementer (SendMessage), max 2 loops, then orchestrator takes over.
3. Orchestrator re-runs every test file itself at the end of each wave: `.venv/bin/python -m pytest tests -q`.
4. Wave gate: next wave starts only when all tasks of the current wave are PASS.
5. Final: full `pytest`, T9-run on real data, compare with `outputs/segmentation_v1/` (cluster count, duplicate titles, outlier rate), report to user. No commit without user approval.

## 6. Risks / watch-outs

- **DBCV on UMAP space with `validity_index`** can be slow (O(n²) per cluster) — fine at ~900 pts/partition; if > 10 s per call, T2 may switch to computing on a cached mutual-reachability graph — report, don't silently change metric.
- **Precursor base rate**: median inter-incident gap is 21 days → many raw hits; the permutation null is what makes results meaningful. Never report hits without lift/q.
- **Weather confounding**: storms cause both "arrêt vent fort" and "rupture" on many machines at once → the null (within-equipment permutation) does not remove fleet-wide simultaneous shocks. T7's prompt flags common external causes as `plausible=False`; mention this limit in the README.
- **LLM cost**: ~60 fine labels + ~15–30 merge calls/round + critic calls + ~20–40 plausibility + ~20 diagnoses; all cached by prompt hash, so prompt edits invalidate caches — keep prompts stable once T9-run starts.
