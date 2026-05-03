# Search / Backtest / Forward Validation / CPU-GPU Audit

**Timestamp:** 2026-05-03 14:46 Europe/Berlin  
**Repository:** `kosred/forex-ai`  
**Target branch for this report:** `master`  
**Review focus:** data ingestion, feature alignment, strategy search, backtest, forward/walk-forward validation, CPCV, CPU/GPU parity, and HPC GPU search semantics.

This report intentionally does **not** review model training/inference internals outside the search/backtest/validation path.

---

## Executive summary

The main risk is not simply whether the bot uses GPU. The critical question is whether the GPU paths run the **same trading logic** as the CPU reference path.

Current reading shows three distinct families:

1. **CPU / cubecl evaluator GA path**  
   This is the path closest to the real SL/TP/spread/commission backtest semantics.

2. **cubecl full CUDA backtest path**  
   This is intended to accelerate the same GA evaluation, but currently has a CPU/GPU timing mismatch in the full CUDA backtest entry signal.

3. **tensor GPU discovery / HPC island discovery paths**  
   These run strategy search on GPU, but use returns/action-style fitness with flat cost, not the same full SL/TP/spread/commission semantics as the CPU GA search. These should be treated as a separate search family, not as a parity replacement for the main backtest evaluator.

The strongest current risks are:

- Higher-timeframe resampling/alignment may leak future HTF candle information into lower-timeframe bars.
- The full CUDA backtest path uses current-bar signal for entry while the CPU path uses prior-bar signal.
- `fitness` is used with different meanings across search/filtering contexts and should not be treated as net profit.
- Walk-forward/CPCV code exists, but it does not appear to be a mandatory hard gate in the main discovery acceptance path based on current reading.
- UI discovery and batch discovery appear to slice/validate data differently.

---

## Current status by branch

### In `master`

Known existing components:

- `crates/forex-data/src/core/resample.rs` contains OHLCV resampling.
- `crates/forex-data/src/core/features.rs` contains `align_features_by_ns`.
- `crates/forex-search/src/eval.rs` contains the CPU reference backtest/evaluation core and GPU-enabled evaluator dispatch.
- `crates/forex-search/src/cubecl_eval.rs` contains the cubecl CUDA signal/backtest evaluator.
- `crates/forex-search/src/discovery.rs` contains discovery/search orchestration, filtering, ranking, quality screen, and portfolio construction logic.
- `crates/forex-search/src/validation.rs` contains embargoed walk-forward backtest and `CombinatorialPurgedCV`.
- `crates/forex-search/src/gauntlet.rs` contains a separate strategy gauntlet validation layer.
- `crates/forex-search/src/discovery_gpu.rs` and `crates/forex-search/src/hpc_gpu_discovery.rs` contain tensor/HPC GPU discovery paths.

### Outside `master` / currently in PR branch

Branch reviewed: `ariadne/evo-search-gpu-fixes` / PR #5.

Already applied in the PR branch, but not in `master`:

- `crates/forex-search/src/genetic/evolution_math.rs`
  - Adds `reset_gene_metrics`.
  - Normalizes random/mutated genes.
  - Resets derived metrics after crossover/mutation.
  - This is a real source fix for stale derived metrics.

- `crates/forex-search/src/genetic/diversity.rs`
  - Adds bucket-based diversity archive selection.
  - Useful, but not enough by itself for canonical deduplication.

- `crates/forex-search/src/genetic/regime_labels.rs`
  - Adds rolling regime-window labeling.
  - Useful as a post-search labeling/ranking tool, but should be inserted after archive capping/diversity, not over huge raw candidate sets.

- `crates/forex-search/src/genetic/mod.rs`
  - Exports the new modules and `reset_gene_metrics`.

Queued but not applied to source in PR branch:

- `search_engine.rs` canonical archive dedup using `gene_signature_hash`.
- `cubecl_eval.rs` full CUDA backtest entry timing change from `signals[i]` to `signals[i - 1]`.

These queued items remain important and should not be considered fixed in `master`.

---

## High-risk findings

### 1. Higher-timeframe resampling can leak future candle information

**Files:**

- `crates/forex-data/src/core/resample.rs`
- `crates/forex-data/src/core/features.rs`
- `crates/forex-data/src/lib.rs`

**Observed behavior:**

`resample_ohlcv` creates higher-timeframe candles and stores their timestamp as `current_bucket_start`. The OHLC values for that bucket include the whole bucket. Example: an H1 candle for 10:00-10:59 is timestamped as 10:00 but contains high/low/close information only known after the hour completes.

`align_features_by_ns` then aligns higher-timeframe features into base timeframe rows using `feature_ns <= base_ts` with forward-fill behavior.

**Risk:**

A lower-timeframe row may see completed H1/H4/D1 features before that higher-timeframe candle has actually closed. This is future leakage and can make strategies look much better than they are.

**Severity:** Critical.

**Recommended fix direction:**

- Treat resampled higher-timeframe timestamps as **bar close timestamps**, not bucket start timestamps; or
- Keep bucket start timestamps but shift HTF features by one full HTF bar before alignment; or
- During alignment, use `usable_ts = bucket_start + timeframe_duration` for HTF data.

A one-row shift at the base timeframe is not enough for H1/H4/D1 leakage.

---

### 2. CPU vs full CUDA backtest signal timing mismatch

**Files:**

- `crates/forex-search/src/eval.rs`
- `crates/forex-search/src/cubecl_eval.rs`

**Observed behavior:**

The CPU evaluator uses prior-bar signal timing for entries: signal at bar `i - 1`, fill at bar `i`.

The full CUDA backtest path currently opens positions using current-bar signal `signals_flat[signal_base + i]`.

**Risk:**

The full CUDA backtest can enter using same-bar information, introducing look-ahead bias compared with the CPU reference evaluator.

**Severity:** Critical.

**Current fix status:**

A patch exists in the PR branch queue to change CUDA entry to `signals_flat[signal_base + i - 1]`, but this is not applied to source in `master`.

---

### 3. Tensor GPU / HPC discovery is not equivalent to CPU GA backtest

**Files:**

- `crates/forex-search/src/discovery_gpu.rs`
- `crates/forex-search/src/hpc_gpu_discovery.rs`
- `crates/forex-search/src/lib.rs` CPU fallback GPU discovery module

**Observed behavior:**

The tensor GPU discovery and HPC island discovery paths use returns/action-style fitness, generally along the lines of:

- actions from tensor signals,
- next open/close returns,
- flat transaction cost,
- tensor-level Sortino/consistency/drawdown style scoring.

They do not run the same full SL/TP/spread/commission/trade-state simulation as the CPU GA evaluator.

**Risk:**

Running search on GPUs is useful, but these paths do not guarantee the same strategy ranking as the real backtest evaluator. A genome found by HPC discovery may look good under tensor-return fitness and fail under SL/TP/spread/commission backtest.

**Severity:** High.

**Recommended fix direction:**

Keep these as separate search families unless/until they are made parity-compatible. For a reliable GPU-first bot, the GPU search path should call an evaluator that matches the CPU reference backtest semantics.

---

### 4. `fitness` does not always mean net profit

**Files:**

- `crates/forex-search/src/genetic/strategy_gene.rs`
- `crates/forex-search/src/genetic/evolution_math.rs`
- `crates/forex-search/src/discovery.rs`

**Observed behavior:**

`fitness` is used as a general score field. In the GA/backtest path it can represent a composite score derived from metrics. In filtering/ranking contexts it may be treated like profit or income score.

**Risk:**

Filters such as minimum profit can accidentally compare against a composite score instead of true net profit. Ranking and UI summaries can also show “best” strategies based on a score that is not profit.

**Severity:** High.

**Recommended fix direction:**

Separate fields/contracts:

- `net_profit`
- `ranking_score`
- `fitness_score`
- `quality_score`

Filtering by profit should use true net profit from metrics, not `gene.fitness`.

---

### 5. Search evaluation and final quality screen may use different cost models

**Files:**

- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/eval.rs`

**Observed behavior:**

The GA search path receives `EvaluationConfig` containing symbol/currency/spread/commission context. Later quality-screen logic uses `discovery_backtest_settings(gene)` which starts from `BacktestSettings::default()` and mainly overrides SL/TP and kill zones.

**Risk:**

A strategy can be evaluated under one cost model during search and under another cost model during final screening.

**Severity:** High.

**Recommended fix direction:**

Build one canonical backtest/cost settings object per discovery run and pass it consistently through:

- GA evaluation,
- quality screen,
- gauntlet,
- walk-forward validation,
- exported reports.

---

### 6. Signal count is used as proxy for trade count

**File:** `crates/forex-search/src/discovery.rs`

**Observed behavior:**

Candidate filtering counts non-zero signals as a proxy for trade count.

**Risk:**

Non-zero signal count is not the same as actual executed trade count. Actual trades depend on current position state, TP/SL, max hold, min hold, max trades/day, gaps, kill zones, and exits.

**Severity:** High.

**Recommended fix direction:**

Use simulated trade count from the same backtest settings used for evaluation, or explicitly rename this as `signal_count_proxy` and do not use it as a hard trade-count gate.

---

### 7. Walk-forward/CPCV exists but does not appear to be a hard discovery gate

**Files:**

- `crates/forex-search/src/validation.rs`
- `crates/forex-search/src/discovery.rs`

**Observed behavior:**

`validation.rs` contains:

- `embargoed_walkforward_backtest`
- `WalkforwardBacktestInput`
- `CombinatorialPurgedCV`

Current reading did not show this as a mandatory hard gate in the main `discovery.rs` candidate acceptance flow.

**Risk:**

The bot can expose config fields for walk-forward/CPCV while the actual candidate selection may not be forced through those validations.

**Severity:** High.

**Recommended fix direction:**

Make validation status explicit in the discovery result:

- `walkforward_tested: bool`
- `cpcv_tested: bool`
- `walkforward_passed: bool`
- `cpcv_passed: bool`

Do not export/accept final portfolio candidates unless the configured validation gates have actually run.

---

### 8. Existing walk-forward evaluates fixed signals, not full retrain-per-split WFO

**File:** `crates/forex-search/src/validation.rs`

**Observed behavior:**

`embargoed_walkforward_backtest` receives precomputed `signals` and evaluates test slices after train/embargo sections.

**Risk:**

This is useful forward validation of an already discovered strategy, but it is not full walk-forward optimization where each split retrains/reselects on the training part and tests on the next future part.

**Severity:** Medium-High.

**Recommended fix direction:**

Document the distinction:

- `embargoed_walkforward_backtest` = fixed-strategy forward validation.
- `walkforward_discovery_retrain` = future needed full WFO search/retrain per split.

---

### 9. UI discovery and batch discovery slice data differently

**Files:**

- `crates/forex-app/src/app_services/discovery.rs`
- `crates/forex-search/src/orchestration.rs`

**Observed behavior:**

The UI/app discovery service cuts the dataset to an 80% in-sample region before running discovery.

The batch orchestrator calls `run_discovery_cycle` directly on the full prepared feature frame and base OHLCV.

**Risk:**

“Discovery” means different things depending on entrypoint. Results from UI and batch/CLI may not be comparable.

**Severity:** Medium-High.

**Recommended fix direction:**

Create one shared discovery data-slicing policy and use it in UI, CLI, batch, and tests.

---

### 10. Stage-1 search window may be narrower than expected

**File:** `crates/forex-search/src/discovery.rs`

**Observed behavior:**

Discovery performs a stage-1 fast evaluation on a percentage of the available data, defaulting to a recent fraction. If the app service has already cut the data to 80% in-sample, stage-1 may only search over the last part of that in-sample subset.

**Risk:**

The initial GA may optimize over a narrower slice than intended, increasing overfitting to recent in-sample conditions.

**Severity:** Medium.

**Recommended fix direction:**

Report the exact rows/time window used for:

- full dataset,
- in-sample slice,
- stage-1 slice,
- final validation slice.

---

### 11. Gauntlet exists but must share the same cost model

**File:** `crates/forex-search/src/gauntlet.rs`

**Observed behavior:**

`StrategyGauntlet` evaluates a gene with `fast_evaluate_strategy_core` and applies thresholds for win rate, profit factor, max drawdown, max daily drawdown, and net profit.

**Risk:**

It uses its own `BacktestSettings` from config/defaults. If these are not the same settings used during search and quality screen, it becomes another inconsistent validation layer.

**Severity:** Medium.

**Recommended fix direction:**

Pass the same canonical `BacktestSettings` into gauntlet that search and quality screen use.

---

## Existing repairs observed in PR branch

These are not in `master` yet.

### Repaired in branch: stale derived metrics after gene mutation/crossover

**File:** `crates/forex-search/src/genetic/evolution_math.rs`

PR branch adds `reset_gene_metrics`, normalizes generated genes, and resets stale derived fields after crossover/mutation.

**Status:** Fixed in PR branch, not merged to `master`.

### Added in branch: diversity archive module

**File:** `crates/forex-search/src/genetic/diversity.rs`

Adds bucket-based archive diversification.

**Status:** Added in PR branch, not merged to `master`.

**Important limitation:** It should not replace canonical dedup in `search_engine.rs`.

### Added in branch: regime labeling module

**File:** `crates/forex-search/src/genetic/regime_labels.rs`

Adds rolling window regime labeling for strategies.

**Status:** Added in PR branch, not merged to `master`.

**Important limitation:** This is potentially expensive and should run after candidate capping/diversification.

---

## Pending fixes not yet applied to source

### Pending: archive dedup by canonical gene signature

**Target file:** `crates/forex-search/src/genetic/search_engine.rs`

Current archive dedup appears based on `strategy_id` or formatted fields. It should use canonical `gene_signature_hash(gene)` after normalization.

**Status:** Patch queued in PR branch, not applied to source and not in `master`.

### Pending: CUDA full backtest causal entry timing

**Target file:** `crates/forex-search/src/cubecl_eval.rs`

Change full CUDA backtest entry from current-bar signal to prior-bar signal.

Expected direction:

```rust
let s = signals_flat[signal_base + i - 1];
```

instead of:

```rust
let s = signals_flat[signal_base + i];
```

**Status:** Patch queued in PR branch, not applied to source and not in `master`.

---

## GPU-first architecture notes

The user goal is valid: most expensive search/backtest/training-adjacent work should run on GPU, especially during large-scale strategy search.

However, GPU-first is only safe if the GPU code is **semantic-parity GPU**, not just “some faster metric on GPU”.

Recommended classification:

### A. Parity GPU path

GPU code must match CPU reference:

- same signal timing,
- same position state machine,
- same TP/SL rules,
- same spread/commission/pip value,
- same max hold/min hold,
- same max trades/day,
- same gap handling,
- same monthly/consistency metrics,
- same output metric layout.

The cubecl evaluator should become this path.

### B. Approximate GPU presearch path

Tensor/HPC discovery can remain as a fast approximate presearch path, but its output should be treated as candidates requiring full parity backtest validation.

### C. Validation path

Final acceptance should require CPU/GPU parity backtest or a proven GPU parity evaluator, then walk-forward/CPCV/gauntlet.

---

## Recommended next action order

1. Fix HTF resampling/alignment leakage.
2. Fix full CUDA backtest signal timing.
3. Define canonical cost model and pass it everywhere.
4. Separate `fitness_score` from `net_profit`.
5. Enforce validation flags in discovery output.
6. Wire walk-forward/CPCV as optional-but-real hard gates.
7. Treat tensor/HPC GPU discovery as approximate presearch unless parity evaluator is used.
8. Add CI coverage for `forex-search` and GPU-relevant compile paths where feasible.
9. Add CPU-vs-GPU parity tests on tiny deterministic OHLCV/signals.
10. Add data contract tests for resampled HTF timestamps and MTF alignment.

---

## Open verification checklist

- [ ] Confirm whether `validation.rs` functions are called from all relevant discovery entrypoints.
- [ ] Confirm whether `CombinatorialPurgedCV` affects final portfolio acceptance.
- [ ] Confirm exact metric layout in CPU evaluator vs cubecl evaluator.
- [ ] Add deterministic CPU/GPU backtest parity fixture.
- [ ] Add HTF leakage regression test using synthetic M1 → H1 resampling.
- [ ] Confirm whether all exported portfolios record which validation gates were actually run.
- [ ] Confirm CI runs `cargo check/test -p forex-search` and relevant feature combinations.

---

## Bottom line

The bot already has serious pieces for search, GPU acceleration, walk-forward validation, CPCV, and gauntlet-style filtering.

But today the safest interpretation is:

- `master` has multiple search/backtest/validation paths with different semantics.
- PR branch fixes some gene-evolution correctness issues but does not yet merge into `master`.
- Two critical leakage/parity issues remain open: HTF resampling timestamp leakage and CUDA same-bar entry.
- HPC/tensor GPU discovery should not be treated as equivalent to the full SL/TP backtest path until parity is proven or explicitly implemented.

For prop-challenge-grade reliability, the next milestone should be a single documented contract for data timing, cost model, signal timing, metric layout, and validation gates, with CPU/GPU parity tests around that contract.
