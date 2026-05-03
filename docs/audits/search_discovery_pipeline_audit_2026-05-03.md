# Search / Discovery Pipeline Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: candidate discovery, GA search, GPU/HPC search entrypoints, CLI/UI/batch search paths, search reproducibility, candidate ranking/filtering/export.

## Summary

The search stack is powerful, but it currently behaves more like several related search systems than one canonical discovery pipeline.

Relevant entrypoints include:

- `forex-cli search`
- `forex-cli discover`
- `forex-cli batch-discover`
- UI/app discovery service
- `forex-models::GeneticStrategyExpert`
- `forex-search::discovery_gpu`
- `forex-search::hpc_gpu_discovery`
- `forex-search::genetic::search_engine`

The main risk is semantic drift: the same symbol/config/data can go through different search windows, feature preparation, evaluator settings, randomness, filters, ranking, and export logic depending on the entrypoint.

## Findings

### 1. Search is not fully deterministic by default

`genetic/search_engine.rs` supports `FOREX_BOT_SEARCH_SEED`, but when no seed is set it seeds from `rand::rng()`. Other search-related paths also use `rand::rng()`, including `discovery.rs`, `discovery_gpu.rs`, `hpc_gpu_discovery.rs`, `cubecl_ga.rs`, `quality.rs`, and `forex-models/src/genetic.rs`.

`forex-models/src/genetic.rs` even comments that the RNG is seeded, but the visible label-search path uses `let mut rng = rand::rng();`.

**Risk:** same config + same data may produce different portfolios across runs. This makes debugging, audit, and CPU/GPU parity difficult.

**Severity:** High.

**Fix direction:** add one `search_seed` to typed config and propagate it to every search/discovery path. Export the effective seed in every profile/artifact. If no seed is set, generate one once at run start, log/export it, and reuse it everywhere.

---

### 2. `forex-cli search` is not equivalent to `discover`

`cmd_search` calls `evolve_search` directly. That path uses default evaluation config when no explicit `EvaluationConfig` is passed. `cmd_discover` instead builds a `DiscoveryConfig`, runs `run_discovery_cycle`, then saves portfolio/profile/quality/trade artifacts.

**Risk:** the operator can run `search` and think it represents the full discovery pipeline, but it bypasses discovery filtering, quality screen, portfolio construction, export profile, and possibly settings-derived cost/symbol config.

**Severity:** Medium-High.

**Fix direction:** rename `search` to `raw-search` or make it call the same canonical discovery pipeline in a dry-run mode. The CLI help should clearly distinguish raw GA search from portfolio discovery.

---

### 3. UI discovery, CLI discovery, and batch discovery use different data slicing semantics

UI discovery cuts to 80% in-sample before running discovery. CLI `discover` runs `run_discovery_cycle` on the prepared full feature frame. Batch discovery also routes through orchestration and does not appear to use the same explicit UI 80/20 split.

**Risk:** the same requested discovery can produce different candidates depending on whether it is launched from UI, CLI, or batch mode.

**Severity:** High.

**Fix direction:** centralize split policy inside `forex-search`, not in app/CLI wrappers. All entrypoints should call one function that returns explicit split ranges and pass/fail validation status.

---

### 4. Search-time feature selection and stage-1 funnel are env-driven

`run_discovery_cycle_with_progress` reads env vars such as `FOREX_BOT_PREFILTER_TOP_K`, `FOREX_BOT_PREFILTER_INSAMPLE`, and `FOREX_BOT_FUNNEL_STAGE1_PCT`. These influence the feature set and the first search window.

**Risk:** two runs with identical config files can choose different features/candidates if the environment differs.

**Severity:** High.

**Fix direction:** move these into typed config or export them in `DiscoveryRunProfile` as effective runtime overrides.

---

### 5. Search archive dedup still appears strategy-id based in master

`genetic/search_engine.rs` currently deduplicates archive entries using `strategy_id` when present, otherwise a formatted indices/weights/threshold string. The canonical `gene_signature_hash` approach exists in newer work but is not fully wired into master archive insertion.

**Risk:** duplicate or near-duplicate genes can survive under different strategy IDs, inflating archive diversity and candidate counts.

**Severity:** Medium-High.

**Fix direction:** deduplicate archive and portfolio candidates by canonical normalized gene signature, not by strategy ID.

---

### 6. Candidate ranking still depends on `gene.fitness` as an income multiplier base

`finalize_candidates_with_progress` ranks candidates using an income-focused score based on `gene.fitness` multiplied by consistency/win-rate/safety/profit-factor terms. But `gene.fitness` is not guaranteed to mean net profit across all search paths.

**Risk:** strategies may be ranked by composite fitness while the UI/export implies income/profit quality.

**Severity:** High.

**Fix direction:** create typed fields: `search_score`, `net_profit`, `quality_score`, `oos_score`, `portfolio_score`. Candidate ranking should state exactly which score it uses.

---

### 7. Search filtering uses signal count as trade-count proxy before full simulation

The discovery filter stage calls `signals_for_gene`, counts non-zero signals, and uses that as the min-trades gate. This is not equivalent to executed trades under SL/TP, max hold, min hold, max trades/day, gaps, kill zones, and position state.

**Risk:** candidates can be filtered in or out based on signal density rather than actual trade count.

**Severity:** High.

**Fix direction:** use a cheap canonical trade simulator for min-trades filtering, or explicitly label this as `signal_count_prefilter` and avoid using it as a hard final gate.

---

### 8. `signals_for_gene` is not the same as full evaluator signal synthesis

`signals_for_gene` combines weighted feature columns and thresholds. The population evaluator includes SMC flags/gating. Several downstream discovery/quality/gauntlet paths use `signals_for_gene`.

**Risk:** a gene discovered with SMC-aware evaluation can later be filtered, trade-logged, or portfolio-tested with non-SMC signals.

**Severity:** High.

**Fix direction:** expose one canonical signal synthesis function and use it everywhere.

---

### 9. GPU/HPC search paths are approximate presearch, not canonical discovery acceptance

`discovery_gpu.rs` explicitly documents that it uses returns-based tensor fitness with flat cost and does not model SL/TP, spread, or commission. `hpc_gpu_discovery.rs` is an island/HPC style path and should be treated the same unless it calls the canonical parity evaluator.

**Risk:** GPU/HPC search results can be mistaken for final validated strategies.

**Severity:** High.

**Fix direction:** label tensor/HPC GPU search output as `approximate_presearch`. Final portfolio export must require canonical CPU/cubecl parity backtest and OOS validation.

---

### 10. Search result artifacts do not yet prove one canonical validation chain

Discovery exports portfolio/profile/quality/trades, but the profile currently records config and counts more clearly than executed validation gates and exact split ranges.

**Risk:** exported portfolio may not prove which path created it, which evaluator was used, whether OOS/WFV/CPCV ran, or which env/runtime knobs affected the run.

**Severity:** High.

**Fix direction:** add `SearchRunContract` / `PortfolioContract` with: entrypoint, git commit, feature schema hash, seed, effective env overrides, split ranges, evaluator backend per stage, candidate ranking formula, validation gates/results, and export timestamp.

## Recommended implementation order

1. Create a canonical `SearchRunContract` and require all entrypoints to build it.
2. Move split policy from UI/CLI wrappers into `forex-search`.
3. Add one run seed and export it.
4. Convert raw `search` CLI to either `raw-search` or canonical discovery dry-run.
5. Replace archive strategy-id dedup with canonical gene signature dedup.
6. Replace signal-count trade proxy with canonical cheap trade simulation.
7. Use one signal synthesis path everywhere.
8. Mark tensor/HPC GPU discovery as approximate presearch unless followed by canonical validation.
9. Export split ranges, evaluator backend, runtime overrides, and validation pass/fail status.
10. Add reproducibility tests: same seed + same data + same contract must produce same candidate order and same exported portfolio.

## Bottom line

The search system is close to useful, but the next milestone should be making search reproducible and canonical. Every route into search must produce a portfolio that can answer: which data was searched, which features were selected, which seed was used, which evaluator accepted it, which validation gates ran, and why this candidate ranked above another.
