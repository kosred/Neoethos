# Search / Backtest / Forward Validation / CPU-GPU Audit

**Created:** 2026-05-03 14:46 Europe/Berlin  
**Last updated:** 2026-05-03 15:06 Europe/Berlin  
**Repository:** `kosred/forex-ai`  
**Target branch:** `master`  
**Scope:** data ingestion, feature alignment, search/discovery, backtest, forward/walk-forward validation, CPCV, CPU/GPU parity, and HPC GPU search semantics.

This audit intentionally does **not** review model training/inference internals outside the search/backtest/validation path.

---

## Executive summary

The main issue is not whether the bot uses GPU. The critical issue is whether the GPU paths run the **same trading semantics** as the CPU reference path.

Current reading shows three different search/evaluation families:

1. **CPU / cubecl evaluator GA path** — closest to real SL/TP/spread/commission stateful backtest semantics.
2. **cubecl full CUDA backtest path** — intended to accelerate the same GA evaluation, but still has a known CPU/GPU signal-timing mismatch.
3. **tensor GPU discovery / HPC island discovery paths** — fast GPU search using simplified return/action-style fitness and flat costs. Useful as approximate presearch, but not equivalent to the full SL/TP backtest path unless parity is implemented and tested.

Highest-priority risks:

- Higher-timeframe resampling/alignment can leak future HTF candle information.
- Timestamp unit handling appears inconsistent: data/resampling uses nanoseconds while search/eval/quality expects milliseconds.
- Full CUDA backtest uses current-bar signal while CPU uses prior-bar signal.
- `fitness` is used with different meanings and should not be treated as net profit.
- Search, quality screen, gauntlet, walk-forward, and trade logs can use different signal/cost/session contracts.
- The bot does not yet expose one canonical trader-style day/week/month ledger.

---

## Branch status

### In `master`

Known relevant components:

- `crates/forex-data/src/core/resample.rs` — OHLCV resampling.
- `crates/forex-data/src/core/features.rs` — feature alignment.
- `crates/forex-search/src/eval.rs` — CPU reference backtest/evaluation and GPU dispatch.
- `crates/forex-search/src/cubecl_eval.rs` — cubecl CUDA signal/backtest evaluator.
- `crates/forex-search/src/genetic/search_engine.rs` — GA search and archive.
- `crates/forex-search/src/discovery.rs` — discovery/search orchestration, filtering, ranking, quality screen, portfolio logic.
- `crates/forex-search/src/validation.rs` — embargoed walk-forward and CPCV utilities.
- `crates/forex-search/src/gauntlet.rs` — strategy gauntlet validation layer.
- `crates/forex-search/src/quality.rs` — trade-based quality analysis, monthly consistency, Monte Carlo daily-block bootstrap.
- `crates/forex-search/src/discovery_gpu.rs` and `crates/forex-search/src/hpc_gpu_discovery.rs` — tensor/HPC GPU discovery paths.

### In PR branch, not merged to `master`

Branch reviewed: `ariadne/evo-search-gpu-fixes` / PR #5.

Already applied in branch but not in `master`:

- `genetic/evolution_math.rs`: adds `reset_gene_metrics`, normalizes generated/mutated genes, resets derived metrics after crossover/mutation.
- `genetic/diversity.rs`: adds bucket-based archive diversity selection.
- `genetic/regime_labels.rs`: adds rolling regime-window strategy labeling.
- `genetic/mod.rs`: exports the new modules and `reset_gene_metrics`.

Queued but not applied to source:

- `search_engine.rs`: canonical archive dedup with `gene_signature_hash`.
- `cubecl_eval.rs`: CUDA full backtest causal entry change from `signals[i]` to `signals[i - 1]`.

---

## Findings

### 1. HTF resampling can leak future candle information

**Files:** `forex-data/src/core/resample.rs`, `forex-data/src/core/features.rs`, `forex-data/src/lib.rs`

`resample_ohlcv` creates higher-timeframe candles with timestamp equal to the bucket start. The OHLC values contain the whole bucket. For example, an H1 candle for 10:00-10:59 can be timestamped as 10:00 while its high/low/close are only known after the hour closes.

`align_features_by_ns` aligns features using `feature_ns <= base_ts` with forward-fill. That means lower-timeframe rows may see completed H1/H4/D1 values before the HTF candle is actually closed.

**Severity:** Critical.

**Fix direction:** timestamps for resampled HTF candles must represent bar close time, or alignment must use `bucket_start + timeframe_duration`, or HTF features must be shifted by one full HTF bar. A one-row shift on M1 is not enough for H1/H4/D1.

---

### 2. Timestamp unit contract is inconsistent

**Files:** `resample.rs`, `search_engine.rs`, `eval.rs`, `validation.rs`, `quality.rs`

The resampling code uses nanosecond units (`1_000_000_000` per second). Search/evaluation/quality code expects milliseconds in multiple places: `timestamp_millis_opt`, `86_400_000`, `3_600_000`, `gap_threshold_ms`, and daily Monte Carlo bucketing via `entry_time / 86_400_000`.

**Risk:** If nanosecond timestamps reach search/eval, day/month bucketing, duration hours, gap exits, max-trades-per-day, weekend/kill-zone logic, daily DD, monthly consistency, and Monte Carlo daily blocks can all silently break.

**Severity:** Critical.

**Fix direction:** create one timestamp contract. Prefer normalizing all timestamps to milliseconds before entering `forex-search`, or introduce a typed timestamp wrapper. Add ms/us/ns tests.

---

### 3. CPU vs full CUDA backtest signal timing mismatch

**Files:** `eval.rs`, `cubecl_eval.rs`

CPU evaluator enters using the prior bar signal: `signals[i - 1]`, fill at bar `i`. Full CUDA currently opens with current-bar signal `signals_flat[signal_base + i]`.

**Risk:** CUDA full backtest can introduce same-bar look-ahead and rank strategies differently from CPU.

**Severity:** Critical.

**Fix status:** patch exists in PR queue but is not applied to source or `master`.

---

### 4. Tensor/HPC GPU discovery is not semantic parity with CPU GA backtest

**Files:** `discovery_gpu.rs`, `hpc_gpu_discovery.rs`, `lib.rs`

Tensor/HPC GPU discovery appears to use action/return-style fitness, next open/close returns, simplified flat cost, and tensor-level scoring. It does not run the same stateful SL/TP/spread/commission/position-machine as CPU GA evaluation.

**Risk:** A strategy found by HPC GPU search can look good under approximate tensor fitness and fail under the real backtest.

**Severity:** High.

**Fix direction:** treat tensor/HPC GPU discovery as approximate presearch unless it calls a parity evaluator. Final acceptance must pass the full parity backtest.

---

### 5. `fitness` does not always mean net profit

**Files:** `strategy_gene.rs`, `evolution_math.rs`, `discovery.rs`

`fitness` is a general score. In the GA path it can represent a composite score derived from metrics. Some filters/ranking logic can treat it as profit.

**Risk:** min-profit filtering, anomaly guard, UI summary, or ranking can compare against composite score instead of true net profit.

**Severity:** High.

**Fix direction:** separate `net_profit`, `ranking_score`, `fitness_score`, and `quality_score`. Profit filters must use true net profit, not `gene.fitness`.

---

### 6. Search and final quality screen can use different cost/session contracts

**Files:** `discovery.rs`, `search_engine.rs`, `eval.rs`, `gauntlet.rs`

GA search receives `EvaluationConfig`, but later quality/gauntlet paths can build/use separate `BacktestSettings`. Defaults may leave fields like `kill_zones_enabled`, `min_hold_bars`, `max_trades_per_day`, or `gap_threshold_ms` unset.

**Risk:** strategies can pass search under one cost/session model and fail later under another.

**Severity:** High.

**Fix direction:** create one canonical `SearchBacktestContract` from config/risk/symbol metadata and pass it to CPU GA, CUDA evaluator, quality screen, gauntlet, WFV/CPCV, and trade logs.

---

### 7. Search evaluation ignores important live/prop settings

**Files:** `strategy_gene.rs`, `search_engine.rs`, `forex-core/src/config.rs`

`RiskConfig` includes important constraints: kill zones, max trades/day, spread/slippage/commission, challenge/prop-style risk settings. `EvaluationConfig` carries only a subset. `evaluate_genes` builds `BacktestSettings` and leaves several fields at default.

**Risk:** GA can optimize under looser conditions than live/prop challenge trading.

**Severity:** High.

**Fix direction:** the canonical backtest contract should include risk/session/prop constraints, not only spread and commission.

---

### 8. Signal count is used as proxy for trade count

**File:** `discovery.rs`

Candidate filtering counts non-zero signals as trade-count proxy.

**Risk:** non-zero signal count is not executed trade count. Actual trades depend on position state, TP/SL, max hold, min hold, max trades/day, gaps, kill zones, and exits.

**Severity:** High.

**Fix direction:** use simulated trade count from the same backtest settings used for evaluation, or rename it explicitly as `signal_count_proxy` and do not use it as a hard trade gate.

---

### 9. Walk-forward/CPCV exists but does not appear to be a hard discovery gate

**Files:** `validation.rs`, `discovery.rs`

`validation.rs` contains `embargoed_walkforward_backtest`, `WalkforwardBacktestInput`, and `CombinatorialPurgedCV`. Current reading did not show these as mandatory gates in main discovery candidate acceptance.

**Risk:** config can expose WFV/CPCV while final candidate selection may not be forced through them.

**Severity:** High.

**Fix direction:** discovery results should explicitly carry `walkforward_tested`, `walkforward_passed`, `cpcv_tested`, `cpcv_passed`. No final export should be considered robust unless configured gates actually ran.

---

### 10. Existing walk-forward evaluates fixed signals, not full retrain-per-split WFO

**File:** `validation.rs`

`embargoed_walkforward_backtest` evaluates precomputed `signals` across test slices. That is useful fixed-strategy forward validation, but not full walk-forward optimization where each split retrains/reselects on train and tests future.

**Severity:** Medium-High.

**Fix direction:** document the distinction and add a separate `walkforward_discovery_retrain` if full WFO is required.

---

### 11. Walk-forward diagnostics pass `days` as timestamps

**Files:** `validation.rs`, `eval.rs`

`walkforward_risk_diagnostics` passes `days` into `simulate_trades_core`, but `simulate_trades_core` expects real timestamps.

**Risk:** duration, gap detection, kill-zone/weekend rules, internal day bucketing, and max-trades-per-day can be wrong inside WFV diagnostics.

**Severity:** High.

**Fix direction:** `WalkforwardBacktestInput` should carry both real `timestamps` and day buckets. Pass timestamps to simulation, use day buckets only for aggregation.

---

### 12. UI discovery and batch discovery slice data differently

**Files:** `forex-app/src/app_services/discovery.rs`, `orchestration.rs`

UI/app discovery cuts the dataset to an 80% in-sample region before discovery. Batch orchestrator calls `run_discovery_cycle` on the full feature frame and OHLCV.

**Risk:** “discovery” has different semantics depending on entrypoint.

**Severity:** Medium-High.

**Fix direction:** create one shared discovery slicing policy used by UI, CLI, batch, and tests.

---

### 13. Stage-1 search window may be narrower than expected

**File:** `discovery.rs`

Discovery does stage-1 fast evaluation on a recent fraction. If app service already cut data to 80% in-sample, stage-1 may only search on the last part of that in-sample subset.

**Risk:** stronger overfitting to recent in-sample conditions.

**Severity:** Medium.

**Fix direction:** every run should log/export exact windows: full dataset, in-sample, stage-1, validation/OOS.

---

### 14. Canonical day/week/month ledger is missing

**Files:** `eval.rs`, `cubecl_eval.rs`, `quality.rs`, `validation.rs`

Current code has pieces of period tracking: daily DD and monthly PnL in eval, monthly consistency in quality, daily blocks for Monte Carlo. But there is no single canonical period ledger.

A trader needs period knowledge: daily PnL/DD, weekly rhythm, monthly consistency, payout-cycle behavior, risk concentration, recovery after losing periods, and rule breaches.

**Severity:** High.

**Fix direction:** replace the narrow “flush final month” idea with a full day/week/month period ledger emitted by reference backtest and mirrored by GPU parity evaluator.

Suggested shape:

```rust
struct PeriodLedger {
    days: Vec<PeriodStats>,
    weeks: Vec<PeriodStats>,
    months: Vec<PeriodStats>,
}

struct PeriodStats {
    key: i64,
    start_ts: i64,
    end_ts: i64,
    equity_open: f64,
    equity_close: f64,
    pnl: f64,
    pnl_pct: f64,
    max_drawdown: f64,
    max_intraday_drawdown: f64,
    trade_count: usize,
    wins: usize,
    losses: usize,
    profit_factor: f64,
    expectancy: f64,
    rule_breaches: Vec<String>,
}
```

---

### 15. Final open period is not safely closed

**Files:** `eval.rs`, `cubecl_eval.rs`

Monthly PnL appears written when month changes. The current/open final month can be omitted before Sharpe/consistency. Similar risk exists for any future day/week/month ledger if open periods are not finalized.

**Severity:** High.

**Fix direction:** finalize all open day/week/month periods at backtest end. Add tests for single-day, single-week, and single-month backtests.

---

### 16. Fast metrics and trade simulation appear to differ on kill-zone/session rules

**File:** `eval.rs`

`simulate_trades_core` contains session-aware Friday force-exit and Monday/Friday entry blocking. The visible `fast_evaluate_strategy_core` path does not show equivalent logic.

**Risk:** search metrics and detailed trade logs can disagree.

**Severity:** High.

**Fix direction:** share one position state machine. Metrics and trade logs should be two output modes of the same simulation logic.

---

### 17. `signals_for_gene` used outside evaluator does not apply SMC gating

**Files:** `search_engine.rs`, `discovery.rs`, `gauntlet.rs`

Population evaluator synthesizes signals with indicator weights plus SMC gating/flags. `signals_for_gene(features, gene)` computes signals from weighted features and thresholds only.

That function appears used by validation/gauntlet/discovery trade-log style paths.

**Risk:** a gene searched with SMC gating can later be trade-logged or quality-tested with signals that ignore its SMC flags.

**Severity:** High.

**Fix direction:** expose one canonical signal synthesis function used everywhere: CPU evaluator, CUDA signal kernel, trade logs, quality screen, gauntlet, WFV/CPCV.

---

### 18. Metric layout is implicit and duplicated

**Files:** `eval.rs`, `cubecl_eval.rs`, `validation.rs`, `discovery.rs`

Evaluation returns `[f64; 11]` and downstream modules depend on numeric indexes. CUDA has a smaller internal metric width and reconstructs/derives the full array.

**Risk:** index mismatch can silently corrupt selection, validation, UI display, or export.

**Severity:** Medium-High.

**Fix direction:** replace raw arrays with typed `BacktestMetrics`, or at least define named constants for all indexes.

---

### 19. Gauntlet must share the same contract

**File:** `gauntlet.rs`

Gauntlet uses its own settings/defaults and appears to use `signals_for_gene`, not necessarily the same SMC-gated signal path as search.

**Risk:** gauntlet becomes another inconsistent validation layer.

**Severity:** Medium-High.

**Fix direction:** gauntlet should use the same canonical search/backtest contract and canonical signal synthesis as search and validation.

---

## GPU-first architecture note

The target is valid: expensive search/backtest/training-adjacent work should run mostly on GPU.

But GPU-first is only safe if GPU means **semantic-parity GPU**, not just “a faster metric on GPU”. A parity GPU path must match CPU on:

- timestamp unit,
- HTF feature availability timing,
- signal timing,
- canonical signal synthesis including SMC gating,
- position state machine,
- TP/SL/trailing rules,
- spread/commission/pip value,
- max hold/min hold,
- max trades/day,
- gap handling,
- kill-zone/session rules,
- day/week/month ledger,
- metric layout.

Tensor/HPC discovery can remain a fast approximate presearch path, but final acceptance should require parity backtest.

---

## Recommended next action order

1. Fix HTF resampling/alignment leakage.
2. Normalize timestamp units across data/search/backtest/validation.
3. Fix CUDA full backtest prior-bar signal timing.
4. Introduce one canonical search/backtest contract from config/risk/symbol metadata.
5. Add canonical signal synthesis shared by CPU, CUDA, trade logs, gauntlet, and validation.
6. Fix walk-forward diagnostics to pass true timestamps into `simulate_trades_core`.
7. Build and return day/week/month period ledger; close all open periods at test end.
8. Unify fast metrics and trade simulation via one position state machine.
9. Separate `fitness_score` from `net_profit`.
10. Replace raw metric indexes with typed metrics or named constants.
11. Enforce validation flags in discovery output.
12. Wire WFV/CPCV as optional-but-real hard gates.
13. Treat tensor/HPC GPU discovery as approximate presearch unless parity evaluator is used.
14. Add CI coverage for `forex-search` and GPU-relevant feature combinations.
15. Add deterministic CPU-vs-GPU parity tests.

---

## Open verification checklist

- [ ] Confirm whether `validation.rs` functions are called from all relevant discovery entrypoints.
- [ ] Confirm whether `CombinatorialPurgedCV` affects final portfolio acceptance.
- [ ] Confirm exact metric layout in CPU evaluator vs cubecl evaluator.
- [ ] Add deterministic CPU/GPU backtest parity fixture.
- [ ] Add HTF leakage regression test using synthetic M1 → H1 resampling.
- [ ] Confirm whether exported portfolios record which validation gates actually ran.
- [ ] Confirm CI runs `cargo check/test -p forex-search` and relevant feature combinations.
- [ ] Add timestamp unit tests for ms/us/ns input vectors.
- [ ] Add single-day/week/month tests to ensure final periods are included.
- [ ] Add daily/weekly/monthly ledger tests.
- [ ] Add kill-zone parity tests between fast metrics and full trade simulation.
- [ ] Add signal synthesis parity tests: CPU evaluator vs `signals_for_gene` vs CUDA signal kernel.
- [ ] Add SMC-gated signal parity tests.

---

## Bottom line

The bot already has serious pieces for search, GPU acceleration, WFV/CPCV utilities, quality analysis, and gauntlet filtering.

But `master` currently has multiple search/backtest/validation paths with different semantics. The next milestone should be one documented contract for data timing, timestamp unit, cost model, signal timing, SMC signal synthesis, position state machine, period ledger, metric layout, and validation gates — with CPU/GPU parity tests around that contract.
