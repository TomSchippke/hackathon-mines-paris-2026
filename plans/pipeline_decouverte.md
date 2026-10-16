# Implementation plan — discovery-based pipeline

Workflow for **Haiku sub-agents**, coordinated by an orchestrator (the main session). Each work
package (WP) below is self-contained: an agent reads this file, its WP section, and only the files
it lists. Code, comments, prompts and report text stay **in French**, in the style of the
surrounding code (short functions, docstrings in French, comment density similar to the existing).

Baseline: the current working tree of `branch-agnostic-pipeline` (framing step `framing/` already
added). **Commit it before starting**, so each WP diff is reviewable.

---

## 1. Target behaviour (decisions taken)

| # | Decision |
|---|---|
| D1 | One cause taxonomy, **discovered** from the documents. No predefined factor list: `facteur_principal`, `facteurs_secondaires` disappear. Themes are the only cause label. |
| D2 | Per incident: one **root cause** + 0–n **contributing causes**, free text. Counts / Pareto / plans use the root cause; site comparison uses root OR contributing. |
| D3 | Hybrid discovery: clustering (UMAP + HDBSCAN on phrase embeddings) proposes groups → agent names/merges themes → **LLM assigns every phrase** to a theme or `aucun`. |
| D3b | Clustering on meaning only. Site, weather, severity, cost, date, component are analysis dimensions afterwards. |
| D4 | One taxonomy **per segment** (`asset_type`). |
| D5 | Consequences: same pipeline. The **nominal stop** flag is chosen by the agent on consequence themes after discovery, checked by code. |
| D6 | **No site profiles.** Each site is characterised by what distinguishes it from the other sites of its segment. Framing keeps only: `domaine`, `segments` (labels), `severites` (order). |
| D7 | Remove `lie_vieillissement`, `lie_maintenance`. Keep `maintenance_suspectee` (+ `justification_maintenance`) as an explicit hypothesis, never a cause. |
| D8 | `composant`: free text at extraction (`composant_brut`), then **normalised** per segment by one LLM pass; code checks every raw value is mapped. |
| D9 | Evidence rule for **every** cause (root and contributing): `type` + `indice` quote verified by code; unverified contributing cause is dropped and counted. |
| D10 | A measure belongs to a **(site, cause theme) pair**: concrete field action, backed by the pair's figures. Cause sheets carry no measure. No generic fallback: "À définir" with the figures. |
| D11 | Pairs ranked **across all sites** by cost per year, no per-site quota. |
| D12 | Pareto cut: keep pairs until they cover **80 %** of the cost of known, non-nominal causes; **≤ 12** measures; **≥ 5** incidents per pair. |
| D13 | Fully automatic taxonomy (no human override file for themes). |
| D14 | `aucun` loop: stop when < **5 %** of a segment's phrases are `aucun`, or after **2** reclustering rounds. Remainder = "causes isolées, non regroupées", shown with examples. |

Assumed defaults for points not explicitly settled (orchestrator may change them):
- Unknown causes (`cause_type = inconnue`) stay excluded from discovery and counted apart (as now).
- Theme merge guard: cohesion ratio + centroid similarity only (no dominant-factor rule).
- Accuracy check (manual sample of theme assignments): **out of scope** of this plan, see §6.

---

## 2. Data contracts (all WPs must follow these exactly)

### 2.1 Constants — `analysis_agent/config.py` (written by WP0, read by all)

```python
# Découverte et affectation (D3, D14)
ASSIGN_BATCH = 20                  # phrases par appel LLM d'affectation
NONE_LABEL = "aucun"               # réponse « aucun thème ne convient »
NONE_MAX_PCT = 5.0                 # part maximale de phrases « aucun » par segment (D14)
MAX_RECLUSTER_ROUNDS = 2
RECLUSTER_MIN_CLUSTER_SIZE = 5     # taille minimale d'un cluster lors d'une reprise
# Composants (D8)
MAX_COMPONENT_GROUPS = 15
# Arrêts nominaux (D5) : part minimale des incidents d'un thème dans la sévérité la plus faible
NOMINAL_MIN_MILDEST_PCT = 80.0
# Plan d'actions (D10-D12)
PLAN_COVERAGE_PCT = 80.0
PLAN_MAX_MEASURES = 12
PLAN_MIN_INCIDENTS = 5
SCOPE_SITE, SCOPE_SEGMENT = "spécifique au site", "commun au segment"
TO_DEFINE = "À définir"
```

Remove from `config.py`: `NEUTRAL_FACTORS`, `PHASE_PROFILES`, `TURNS["profiles"]`, `PRIORITY_THRESHOLDS`,
`FLEET_PRIORITIES`. Add `TURNS["nominal"] = 8` and the prompt `PHASE_NOMINAL` (see WP3).

### 2.2 Extraction output — one row per incident (`AnalysisWorkspace.causes`)

| column | type | notes |
|---|---|---|
| `cause_racine` | str | `CAUSE_INCONNUE` if unknown |
| `cause_type` | `explicite` / `inférée` / `inconnue` | |
| `indice`, `indice_verifie`, `cause_annulee` | str, bool, str | unchanged rule |
| `confiance` | `faible` / `moyenne` / `haute` | |
| `causes_contributives` | str (JSON list) | `[{"texte", "type", "indice"}]`, **verified only** |
| `contributives_rejetees` | int | contributing causes dropped (quote not found) |
| `consequence` | str | free text |
| `composant_brut` | str | free text, words of the report |
| `maintenance_suspectee`, `justification_maintenance` | enum, str | unchanged |
| `action_corrective` | str | unchanged |
| `erreur_analyse` | str | `""` if OK |

Removed: `composant`, `facteur_principal`, `facteurs_secondaires`, `categorie_consequence`,
`lie_vieillissement`, `lie_maintenance`.

### 2.3 Phrase table — `AnalysisWorkspace.phrases` (WP3)

One row per phrase to cluster/assign: `phrase_id` (str `"<pos>:<role>:<k>"`), `position` (int,
index in `ws.df`), `asset_type`, `dimension` (`cause` / `consequence`), `role` (`racine` /
`contributive` / `consequence`), `texte`. Cause phrases = root causes not unknown/uncertain + all
verified contributing causes. Consequence phrases = non-empty `consequence`.

Assignment result: column `theme_id` on the phrase table (str theme id, or `None` = isolated).

### 2.4 Incident table — `AnalysisWorkspace.incidents_table()` (WP3, read by WP4/5/6)

Existing columns kept: `theme_cause`, `theme_cause_id`, `theme_consequence`, `theme_consequence_id`
(from the **root** / consequence phrase assignment; `F.UNKNOWN`, `F.UNCERTAIN`, `F.UNCLASSIFIED`
as now; `F.UNCLASSIFIED` now means "isolated cause"). `cluster_*` columns are dropped.
New columns:
- `themes_contributifs_ids` — `"|"`-joined theme ids of verified contributing causes (`""` if none);
- `composant` — normalised component (WP2), `"non précisé"` if empty;
- `nominal` — bool, True if consequence theme is flagged nominal AND severity is the mildest.

### 2.5 `decisions.json["rapport"]` (WP5 writes, WP6/WP7 read)

```json
{
  "sites": [{"site", "asset_type", "caracterisation", "points_distinctifs": [str],
             "causes_dominantes", "causes_dominantes_chiffres": [...], "origine_causes_dominantes": {...},
             "mesures": [measure ids], "faits": {...}}],
  "mesures": [{"id": "M01", "site", "asset_type", "theme_id", "cause", "portee": SCOPE_SITE|SCOPE_SEGMENT,
               "incidents", "cout_total", "cout_par_an", "part_cout_pct", "cumul_pct",
               "composant_principal", "consequence_principale", "type", "mesure", "mecanisme",
               "a_definir": bool, "faits": str}],
  "couverture": {"pct": float, "plafond_atteint": bool, "mesures": int},
  "fiches": [{... as now WITHOUT type/mesure/mecanisme ..., "sites_avec_mesure": [site]}],
  "synthese": str, "faits_synthese": str, "juge": {...}
}
```

Top-level `decisions.json` additions: `"nominaux": {segment: {"theme_ids", "justification"}}`,
`"isoles": {segment: {"pct", "phrases": int, "exemples": [str]}}`,
`"composants": {segment: {raw: group}}`, `"colonnes_contexte"` (kept). Removed: `"profils"`.

---

## 3. Agent count and waves

**8 Haiku agent runs, at most 4 in parallel**, + orchestrator integration.

Why this number: Haiku is reliable on a task that touches **≤ 3 files with an exact interface**,
and unreliable on cross-cutting refactors. The split follows **file ownership** (no two agents in
the same wave edit the same file) and the **data dependencies** of §2. Fewer agents would force
Haiku into 6–8-file changes; more would put two agents in `tools.py` or `facts.py` and create
merge conflicts. Contracts in §2 are fixed up front so wave-1 agents can work in parallel against
interfaces that do not exist yet.

```
Wave 0  (1 agent)   WP0 foundations: config.py constants, framing shrink
Wave 1  (4 agents)  WP1 extraction | WP2 assignment+components modules | WP3 discovery workspace | WP4 facts
Wave 2  (1 agent)   WP5 writing + judge + pipeline orchestration + report
Wave 3  (2 agents)  WP6 review agent | WP7 webapp/export + e2e harness + READMEs
Wave 4  (orchestrator) run e2e harness, lint, fix integration bugs, final review
```

Rules for every agent:
- Edit **only** the files listed as "owns". Read others freely.
- Do not run anything that calls Azure (no `.env`). Verify with `uvx pyflakes <files>` and the
  WP's acceptance snippet.
- If the contract in §2 seems wrong, **do not improvise**: stop and report the issue in the final
  message.
- Final message: files changed, functions added/removed, acceptance results, open issues.

---

## 4. Work packages

### WP0 — Foundations (wave 0)

**Owns:** `analysis_agent/config.py`, `framing/config.py`, `framing/pipeline.py`, `framing/cadrage.py`,
`framing/README.md`.

1. `analysis_agent/config.py`: add the constants of §2.1; remove the listed ones; keep everything
   else. Keep `ACTION_TYPES` re-export.
2. Framing shrink (D6): the LLM schema and `validate()` keep only `domaine`, `segments`,
   `severites`. Remove facteurs, composants, categories_consequence, profils, keywords, measures,
   `FACTORS_RANGE`, `COMPONENTS_RANGE`, `CONSEQUENCES_RANGE`, `AGE_CRITERION`,
   `MAINTENANCE_CRITERION`, `RESERVED`, `DIRECTIONS`. Keep `UNDETERMINED` only if still imported
   elsewhere (grep), else remove.
3. `Cadrage` keeps: `domain`, `segments`, `label()`, `severities`, `mildest`, `most_severe`.
   Remove the other properties/methods.
4. `framing/pipeline.markdown()` and `framing/README.md`: describe only the three elements.

**Acceptance:** `uvx pyflakes framing analysis_agent/config.py`;
`python -c "import framing.pipeline, analysis_agent.config"` from repo root.
(Other modules will break until wave 1–2; that is expected.)

### WP1 — Extraction (wave 1)

**Owns:** `cause_extraction/config.py`, `cause_extraction/pipeline.py`, `cause_extraction/README.md`.

1. New JSON schema (strict) and prompt producing the columns of §2.2. Contributing causes:
   `{"texte", "type" (explicite|inférée), "indice"}`, 0–4 items, each with its own verbatim quote.
   Prompt: root cause = the phenomenon that started the chain; contributing causes = conditions
   that made it possible or worse; never a consequence; no closed lists.
2. `normalize_cause()`: unchanged rule for the root cause. New
   `normalize_contributing(items, text) -> tuple[list[dict], int]`: keeps items whose
   `quote_found(indice, text)` is True, returns (kept, n_dropped).
3. `analyse()` serialises `causes_contributives` with `json.dumps(..., ensure_ascii=False)`.
4. `run()` (manual exploration): replace `facteurs_par_site.csv` / `composants_par_site.csv` by
   `contributives_par_site.csv` (mean number of contributing causes, % dropped) — keep it small.
5. Remove every use of `framing.cadrage` lists (only `current().label()` / `domain` remain).

**Acceptance:** pyflakes; unit check in a scratch script: `normalize_contributing` keeps a quote
present in the text, drops an invented one; `response_format()["json_schema"]["schema"]["required"]`
contains exactly the §2.2 fields produced by the LLM.

### WP2 — Assignment and component modules (wave 1)

**Owns (new files):** `analysis_agent/assignment.py`, `analysis_agent/components.py`.
Pure functions, no workspace dependency, LLM via `llm.client.chat(role="extraction")`.

`assignment.py`:
```python
def assign_phrases(phrases: list[str], themes: list[dict], context: str) -> list[str | None]:
    """themes: [{"id", "name", "description", "examples": [str]}]. Retourne, pour chaque phrase,
    l'id du thème choisi ou None (NONE_LABEL). Appels par lots de ASSIGN_BATCH, JSON strict :
    {"affectations": [{"index": int, "theme": enum(ids + [NONE_LABEL])}]}. Une phrase absente ou
    un index invalide dans la réponse -> None."""
```
Prompt (French): choose the theme describing **the same phenomenon**; `aucun` if none fits — never
force. Include each theme's name, description and ≤ 3 examples.

`components.py`:
```python
def normalize_components(raw_values: list[str], context: str) -> dict[str, str]:
    """Regroupe les composants bruts distincts d'un segment en au plus MAX_COMPONENT_GROUPS
    composants normalisés. JSON strict : {"groupes": [{"nom", "valeurs": [str]}]}. Le code vérifie
    que chaque valeur brute est rangée une fois ; une valeur oubliée -> son propre nom en minuscules
    (consigné dans le retour via la clé spéciale "__non_rangees__" : liste). Valeurs vides exclues."""
```
Handle > 300 distinct values by sending them sorted, in one call (values are short).

**Acceptance:** pyflakes; scratch test with a mocked `chat` (monkeypatch
`analysis_agent.assignment.chat`) returning a fixed JSON: bad index → None; unknown theme → None;
component missing from groups → reported in `__non_rangees__`.

### WP3 — Discovery workspace (wave 1)

**Owns:** `analysis_agent/tools.py`.

1. Remove: `site_evidence`, `assign_profiles`, `NEUTRAL_FACTORS` use, every
   `facteur_principal` / `categorie_consequence` use, the dominant-factor merge rule
   (`split_by`, `dominant` based on factors, `similar_cluster_pairs` / `duplicate_themes` factor
   condition → similarity ≥ `MERGE_SIMILARITY` only), `self.profiles`.
2. Build `self.phrases` (§2.3) after extraction (`build_phrases()`); `phrase_vectors` keys on
   (`asset_type`, `dimension`) over that table (root + contributing for causes).
3. Themes unchanged in structure (`asset_type`, `dimension`, `name`, `description`, `topics`,
   `incidents` → rename to `phrases`, `cohesion`).
4. New methods (not agent tools, called by the pipeline):
   - `theme_cards(asset_type, dimension) -> list[dict]` → input of `assign_phrases`
     (id, name, description, 3 typical phrases closest to the centroid);
   - `apply_assignment(asset_type, dimension, theme_ids: list[str | None])` → writes
     `self.phrases.theme_id` for that key;
   - `none_share(asset_type, dimension) -> float` (% of phrases with `theme_id` None);
   - `recluster_none(asset_type, dimension) -> str | None`: clusters only the None phrases with
     `RECLUSTER_MIN_CLUSTER_SIZE` (single HDBSCAN fit, no budget), stores it as a run
     `"<key>_reprise<k>"` whose topics must be themed by the agent (same themes phase) — returns
     the run id, or None if no cluster found.
5. New agent tool `flag_nominal(asset_type, theme_ids: list[str], justification: str)`:
   consequence themes only; refused if a theme has < `NOMINAL_MIN_MILDEST_PCT` % of its incidents at
   `current().mildest` severity (message gives the measured %); empty list allowed (nothing
   nominal). Stores `self.nominal[asset_type]`. Plus `nominal_evidence(asset_type)` tool: per
   consequence theme, incidents, % mildest severity, mean cost, 3 examples.
6. `incidents_table()` produces §2.4 (root phrase → `theme_cause*`, contributing →
   `themes_contributifs_ids`, consequence → `theme_consequence*`, `composant` from
   `self.components[asset_type]` mapping of `composant_brut`, `nominal`).
7. `tool_schemas()`: drop removed tools, add `flag_nominal`, `nominal_evidence`.

**Acceptance:** pyflakes; `python -c "import analysis_agent.tools"`; scratch test building a
workspace on a 60-row synthetic frame with a pre-filled `ws.causes` (no LLM): `build_phrases()`
row counts, `incidents_table()` columns ⊇ §2.4.

### WP4 — Facts (wave 1)

**Owns:** `analysis_agent/facts.py`.

1. Remove: profile functions (`expectations_met`, `profile_compatibility`, `best_assignments`,
   `assignment_score`, `profile_gaps`, `explicit_only_profiles`), `priority_causes`,
   `dominant_factor`, `fleet_priorities`, `PRIORITY_THRESHOLDS` use.
2. `nominal_stops(incidents)` → `incidents["nominal"]` (computed in the table, §2.4).
3. Rename `profile_indicators` → `site_indicators(incidents)`: one criterion per cause theme of
   the segment = % of site incidents where the theme is root **or** contributing; then the
   secondary indicators (maintenance suspectée, most severe share, mean cost). Keep
   `rank_text`, `position`, `distinctive_points`, `point_text`.
4. New `measure_pairs(incidents) -> tuple[pd.DataFrame, dict]` (D11–D12): known, non-nominal root
   causes grouped by (site, theme_cause_id); keep pairs with ≥ `PLAN_MIN_INCIDENTS`; sort by
   `cout_par_an` desc; keep while cumulative share of known non-nominal cost < `PLAN_COVERAGE_PCT`
   (include the pair that crosses it), cap `PLAN_MAX_MEASURES`. Columns: `id` (M01…), `site`,
   `asset_type`, `theme_id`, `cause`, `incidents`, `cout_total`, `cout_par_an`, `part_cout_pct`,
   `cumul_pct`, `portee` (`SCOPE_SITE` if `position(site_indicators(segment), cause, site) ==
   "élevé"` else `SCOPE_SEGMENT`), `composant_principal` (mode of `composant`),
   `consequence_principale` (`main_consequence`). Second value: `{"pct", "plafond_atteint", "mesures"}`.
5. `cause_criticality` drops `facteur`; adds `composant_principal`.
6. New `isolated_causes(phrases, asset_type) -> dict` (pct, count, 5 examples) — takes the phrase
   table.

**Acceptance:** pyflakes; scratch test on a synthetic incident table with the §2.4 columns:
coverage stops at the first pair crossing 80 %, cap 12 respected, pairs < 5 incidents excluded,
nominal incidents excluded.

### WP5 — Writing, judge, pipeline, report (wave 2)

**Owns:** `analysis_agent/writing.py`, `analysis_agent/judge.py`, `analysis_agent/pipeline.py`,
`analysis_agent/report.py`, `analysis_agent/README.md`.

1. `writing.py`
   - Concepts guard: build patterns from **theme names** of the segment (words ≥ 5 letters) +
     `GENERIC_CONCEPTS` (age, environment); remove framing concepts and `catalogue_measure`.
   - `site_texts(site, asset_type, incidents, indicators, journal)` → `caracterisation` +
     `causes_dominantes` only (no profile, no actions).
   - New `pair_measure(pair: dict, incidents, journal) -> dict`: facts = site, cause, incidents,
     cost/yr, share, scope, main component, main consequence, 3 extracted cause phrases, 3 most
     frequent `action_corrective`. Task: one **field action on this site** for this cause (what,
     on which equipment, frequency/trigger) + mechanism. Validator: `measure_errors` + must name
     the site or "ce site", no other site name. After retries: `{"type": "procédure", "mesure":
     f"{TO_DEFINE} : {cause} sur {site} ({n} incidents, {cost}/an, composant {comp})",
     "mecanisme": "", "a_definir": True}`.
   - `cause_sheets`: description + consequences only, add `sites_avec_mesure`.
   - `synthesis`: facts = total cost, unknown share per segment, nominal stops, coverage
     (`couverture`), top 3 measures with site/cause/cost per year, isolated-causes share.
2. `judge.py`: judge `caracterisation`, `causes_dominantes`, each pair measure (grid adds:
   "names this site's equipment, not a generic action?"), synthesis. Remove sheet-measure judging.
3. `pipeline.py` phases:
   0 extraction (code) → 0b `build_phrases` + component normalisation per segment (code, WP2) →
   1 per segment × dimension: clustering phase, themes phase (agent, as now) → **1b assignment
   loop** (code): `assign_phrases` → `apply_assignment` → while `none_share > NONE_MAX_PCT` and
   round < `MAX_RECLUSTER_ROUNDS`: `recluster_none` → themes phase on the new run (agent, fallback
   as now) → reassign the None phrases only → 2 priorities (agent, as now) → **3 nominal** (agent:
   `nominal_evidence`, `flag_nominal`; fallback: empty list, logged) → 4 writing (`site_texts`,
   `measure_pairs` + `pair_measure`, `cause_sheets`, `synthesis`) → judge.
   Remove the profiles phase and `fallback_profiles`.
4. `report.py`: remove profile sections/robustness test; §7 per site: characterisation +
   distinctive points + figures + list of its measures; new **plan section**: measures table
   sorted by cost/yr (site, cause, scope, measure, mechanism, cost/yr, cumulative %), coverage
   sentence ("N mesures couvrent X % du coût des causes connues"), "À définir" rows flagged;
   new **isolated causes** section with examples; nominal section uses `ws.nominal` + its
   justification. `export()` writes §2.5 and the new top-level keys.

**Acceptance:** pyflakes on the package; `python -c "import analysis_agent.pipeline"`.

### WP6 — Review agent (wave 3)

**Owns:** `review_agent/config.py`, `review_agent/pipeline.py`, `review_agent/tools.py`,
`review_agent/README.md`.

- Remove profile checks and profile fields; `FIELDS` → `caracterisation`, `causes_dominantes`,
  `mesures`, `synthese`, `autre`.
- New automatic checks: plan coverage reached or cap stated; every measure names its site and is
  not `a_definir` (else `majeur` auto-finding with the figures); isolated-causes share ≤
  `NONE_MAX_PCT` per segment (else `mineur`); nominal themes: % at mildest severity shown;
  contributing causes dropped (information).
- `site_facts` uses `site_indicators`; `theme_facts` reads phrase-level examples if available.
- System prompt: check each measure is a field action on that site, targets a cause that is
  frequent/costly there, and is not a nominal stop.

**Acceptance:** pyflakes; `python -c "import review_agent.pipeline"`.

### WP7 — Webapp, export, e2e harness, docs (wave 3)

**Owns:** `webapp/app.py`, `webapp/figures.py`, `webapp/export_html.py`, `README.md`,
`outputs/README.md`, `tests/e2e_mock.py` (new).

- Sites tab: characterisation instead of profile; site measures from `rapport["mesures"]`.
  New "Plan d'actions" view: measures ranked by cost/yr with scope badge and cumulative coverage
  (bar + cumulative line, see `fig_pareto` for style). Remove `fig_site_criteria` profile wording.
- `tests/e2e_mock.py`: port the reference harness `plans/e2e_mock_reference.py` (current
  pipeline, mocked LLM and embeddings, synthetic data) to the new contracts: mocked extraction returns root + contributing causes with
  real quotes and one invented quote; mocked assignment maps phrases to themes by keyword and
  returns `aucun` for one synthetic "unexpected" cause; mocked components. Runs in a temp dir:
  `python tests/e2e_mock.py`. Asserts: report exists; ≤ 12 measures; coverage ≥ 80 % or cap;
  isolated section present; no profile wording; `contributives_rejetees` > 0.
- READMEs: describe the new flow (§1) briefly.

**Acceptance:** pyflakes; `python tests/e2e_mock.py` passes **after** wave 2 is merged (WP7 runs in
wave 3, so it is).

### Wave 4 — Orchestrator integration

1. Run `uvx pyflakes` on all packages and `python tests/e2e_mock.py`.
2. Grep for leftovers: `facteur_principal|categorie_consequence|lie_vieill|lie_maint|profil|PROFILE`.
3. Fix integration bugs directly (cross-file issues are expected at the WP3/WP4/WP5 boundary).
4. Review the diff against §1 decisions; report what is untested (live Azure run).

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Haiku drifts from the contract | §2 is exact; agents must stop and report instead of improvising; orchestrator checks columns in wave 4 |
| Assignment cost (≈ 1 call / 20 phrases / round) | batching, cache; extraction model |
| Contributing causes inflate site comparisons | quote rule (D9), dropped count reported |
| Nominal flag drops real failures | code check on mildest-severity share, fallback = nothing nominal, review check |
| Reclustering finds nothing | loop stops, remainder shown as isolated causes |
| First real run redoes all LLM calls | expected; warn in README |

## 6. Out of scope (to decide later)

- Manual accuracy check of theme assignment (extend `inferences_a_valider.csv`).
- `pdf_pipeline/` format-specific parsing.
- Topic-modeling exploration module (`topic_modeling/`), left as is.
