# Search / Backtest / Forward Validation / CPU-GPU Audit

**Created:** 2026-05-03 14:46 Europe/Berlin  
**Last updated:** 2026-05-03 16:02 Europe/Berlin  
**Repository:** `kosred/forex-ai`  
**Target branch:** `master`  
**Scope:** data ingestion, feature alignment, search/discovery, backtest, forward/walk-forward validation, CPCV, CPU/GPU parity, and HPC GPU search semantics.

This audit intentionally does **not** review model training/inference internals outside the search/backtest/validation path.

---

## Executive summary

The main issue is not whether the bot uses GPU. The critical issue is whether the GPU paths run the **same trading semantics** as the CPU reference path.

Current reading shows three different search/evaluation families:

1. **CPU / cubecl evaluator GA path** — closest to real SL/TP/spread/commission stateful backtest semantics.
2. **cubecl full CUDA backtest path** — intended to accelerate the same GA evaluation, but still has known CPU/GPU semantic gaps.
3. **tensor GPU discovery / HPC island discovery paths** — fast GPU search using simplified return/action-style fitness and flat costs. Useful as approximate presearch, but not equivalent to the full SL/TP backtest path unless parity is implemented and tested.

Highest-priority risks:

- Higher-timeframe resampling/alignment can leak future HTF candle information.
- Timestamp unit handling appears inconsistent: data/resampling uses nanoseconds while search/eval/quality expects milliseconds.
- Full CUDA backtest uses current-bar signal while CPU uses prior-bar signal.
- Full CUDA backtest does not expose the same kill-zone/session/live execution contract as the CPU/trade simulation paths.
- `fitness` is used with different meanings and should not be treated as net profit.
- Search, quality screen, gauntlet, walk-forward, live order execution, and trade logs can use different signal/cost/session/execution contracts.
- The bot does not yet expose one canonical trader-style day/week/month ledger.
- Several feature/stop-target helpers are not yet clearly bound to the same timeframe and walk-forward data contract.
- UI discovery withholds 20% OOS from search, but current reading does not show a mandatory OOS replay before portfolio export.

---

## Branch status

### In `master`

Known relevant components:

- `crates/forex-data/src/core/resample.rs` — OHLCV resampling.
- `crates/forex-data/src/core/features.rs` — feature alignment.
- `crates/forex-data/src/core/quant_features.rs` — timeframe-dependent quant features.
- `crates/forex-search/src/stop_target.rs` — volatility/regime based SL/TP inference.
- `crates/forex-search/src/eval.rs` — CPU reference backtest/evaluation and GPU dispatch.
- `crates/forex-search/src/cubecl_eval.rs` — cubecl CUDA signal/backtest evaluator.
- `crates/forex-search/src/genetic/search_engine.rs` — GA search and archive.
- `crates/forex-search/src/discovery.rs` — discovery/search orchestration, filtering, ranking, quality screen, portfolio logic.
- `crates/forex-search/src/validation.rs` — embargoed walk-forward and CPCV utilities.
- `crates/forex-search/src/gauntlet.rs` — strategy gauntlet validation layer.
- `crates/forex-search/src/quality.rs` — trade-based quality analysis, monthly consistency, Monte Carlo daily-block bootstrap.
- `crates/forex-search/src/discovery_gpu.rs` and `crates/forex-search/src/hpc_gpu_discovery.rs` — tensor/HPC GPU discovery paths.
- `crates/forex-core/src/domain/order_execution.rs` — live/order execution helper with partial TP, entry patience, and edge/cost checks.
- `crates/forex-core/src/domain/risk.rs` — live risk manager with prop-firm gates, dynamic sizing, recovery mode, and circuit breaker logic.

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

**Files:** `resample.rs`, `search_engine.rs`, `eval.rs`, `validation.rs`, `quality.rs`, `cubecl_eval.rs`

The resampling code uses nanosecond units (`1_000_000_000` per second). Search/evaluation/quality code expects milliseconds in multiple places: `timestamp_millis_opt`, `86_400_000`, `3_600_000`, `gap_threshold_ms`, and daily Monte Carlo bucketing via `entry_time / 86_400_000`.

`cubecl_eval.rs` also names and passes `timestamp_deltas_ms`, but the delta is computed directly from raw timestamps. If those timestamps are nanoseconds, the GPU path receives nanosecond deltas as if they were milliseconds.

**Risk:** day/month bucketing, duration hours, gap exits, max-trades-per-day, weekend/kill-zone logic, daily DD, monthly consistency, Monte Carlo daily blocks, and CUDA gap exits can silently break.

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

### 4. Full CUDA backtest lacks the full session/kill-zone contract

**Files:** `cubecl_eval.rs`, `eval.rs`, `forex-core/src/config.rs`

The CUDA backtest kernel accepts max hold, min hold, max trades/day, gap threshold, trailing settings, spread, commission, and pip value. It does not expose `kill_zones_enabled`, Friday force-exit, Monday/Friday entry blocks, broker timezone day boundaries, or trading session windows.

`simulate_trades_core` contains more session-aware behavior, while CUDA full backtest is a narrower position simulator.

**Risk:** full CUDA backtest can become fast but not equivalent to the CPU/live/session contract. Search results may pass GPU backtest and fail when realistic session constraints are applied.

**Severity:** Critical.

**Fix direction:** move all session/kill-zone/day-boundary logic into a shared backtest contract and implement it in both CPU and GPU evaluators. GPU parity tests must include Friday close, Monday entry block, max trades/day, gap exits, and broker day boundaries.

---

### 5. Tensor/HPC GPU discovery is not semantic parity with CPU GA backtest

**Files:** `discovery_gpu.rs`, `hpc_gpu_discovery.rs`, `lib.rs`

Tensor/HPC GPU discovery appears to use action/return-style fitness, next open/close returns, simplified flat cost, and tensor-level scoring. It does not run the same stateful SL/TP/spread/commission/position-machine as CPU GA evaluation.

**Risk:** a strategy found by HPC GPU search can look good under approximate tensor fitness and fail under the real backtest.

**Severity:** High.

**Fix direction:** treat tensor/HPC GPU discovery as approximate presearch unless it calls a parity evaluator. Final acceptance must pass the full parity backtest.

---

### 6. `fitness` does not always mean net profit

**Files:** `strategy_gene.rs`, `evolution_math.rs`, `discovery.rs`

`fitness` is a general score. In the GA path it can represent a composite score derived from metrics. Some filters/ranking logic can treat it as profit.

**Risk:** min-profit filtering, anomaly guard, UI summary, or ranking can compare against composite score instead of true net profit.

**Severity:** High.

**Fix direction:** separate `net_profit`, `ranking_score`, `fitness_score`, and `quality_score`. Profit filters must use true net profit, not `gene.fitness`.

---

### 7. Search and final quality screen can use different cost/session contracts

**Files:** `discovery.rs`, `search_engine.rs`, `eval.rs`, `gauntlet.rs`

GA search receives `EvaluationConfig`, but later quality/gauntlet paths can build/use separate `BacktestSettings`. Defaults may leave fields like `kill_zones_enabled`, `min_hold_bars`, `max_trades_per_day`, or `gap_threshold_ms` unset.

**Risk:** strategies can pass search under one cost/session model and fail later under another.

**Severity:** High.

**Fix direction:** create one canonical `SearchBacktestContract` from config/risk/symbol metadata and pass it to CPU GA, CUDA evaluator, quality screen, gauntlet, WFV/CPCV, and trade logs.

---

### 8. Search evaluation ignores important live/prop settings

**Files:** `strategy_gene.rs`, `search_engine.rs`, `forex-core/src/config.rs`

`RiskConfig` includes important constraints: kill zones, max trades/day, spread/slippage/commission, challenge/prop-style risk settings. `EvaluationConfig` carries only a subset. `evaluate_genes` builds `BacktestSettings` and leaves several fields at default.

**Risk:** GA can optimize under looser conditions than live/prop challenge trading.

**Severity:** High.

**Fix direction:** the canonical backtest contract should include risk/session/prop constraints, not only spread and commission.

---

### 9. Backtest cost model does not include slippage as a first-class setting

**Files:** `eval.rs`, `cubecl_eval.rs`, `forex-core/src/config.rs`, `forex-core/src/domain/order_execution.rs`

`RiskConfig` has `slippage_pips` and `slippage_guard_multiplier`. Live/order edge checks accept `slippage_pips`. But the search backtest settings visible in `eval.rs`/CUDA carry spread and commission, not slippage.

**Risk:** search can select strategies whose edge disappears once realistic slippage is applied. This matters especially for M1/M5 and high trade count strategies.

**Severity:** High.

**Fix direction:** include slippage in `BacktestSettings` / `SearchBacktestContract` and apply it consistently in CPU, CUDA, WFV/CPCV, gauntlet, quality, and live edge checks.

---

### 10. Live order execution has features not represented in search/backtest

**Files:** `forex-core/src/domain/order_execution.rs`, `eval.rs`, `cubecl_eval.rs`

Live/order execution supports partial take profit, multiple R-level TP legs, entry patience/pullback behavior, and min edge cost multiple checks. The search/backtest path appears to simulate one entry, one SL, one TP, optional trailing, and fixed spread/commission.

**Risk:** the strategy that is backtested is not necessarily the strategy that is live-executed. Partial TP and entry patience can materially change expectancy, drawdown, win rate, trade duration, and prop-firm compliance.

**Severity:** High.

**Fix direction:** either backtest the same execution policy used live, or disable live-only execution features for strategies that were not validated with those features. Execution policy must be part of the canonical contract and exported with the portfolio.

---

### 11. Signal count is used as proxy for trade count

**File:** `discovery.rs`

Candidate filtering counts non-zero signals as trade-count proxy.

**Risk:** non-zero signal count is not executed trade count. Actual trades depend on position state, TP/SL, max hold, min hold, max trades/day, gaps, kill zones, and exits.

**Severity:** High.

**Fix direction:** use simulated trade count from the same backtest settings used for evaluation, or rename it explicitly as `signal_count_proxy` and do not use it as a hard trade gate.

---

### 12. Walk-forward/CPCV exists but does not appear to be a hard discovery gate

**Files:** `validation.rs`, `discovery.rs`

`validation.rs` contains `embargoed_walkforward_backtest`, `WalkforwardBacktestInput`, and `CombinatorialPurgedCV`. Current reading did not show these as mandatory gates in main discovery candidate acceptance.

**Risk:** config can expose WFV/CPCV while final candidate selection may not be forced through them.

**Severity:** High.

**Fix direction:** discovery results should explicitly carry `walkforward_tested`, `walkforward_passed`, `cpcv_tested`, `cpcv_passed`. No final export should be considered robust unless configured gates actually ran.

---

### 13. Existing walk-forward evaluates fixed signals, not full retrain-per-split WFO

**File:** `validation.rs`

`embargoed_walkforward_backtest` evaluates precomputed `signals` across test slices. That is useful fixed-strategy forward validation, but not full walk-forward optimization where each split retrains/reselects on train and tests future.

**Severity:** Medium-High.

**Fix direction:** document the distinction and add a separate `walkforward_discovery_retrain` if full WFO is required.

---

### 14. Walk-forward diagnostics pass `days` as timestamps

**Files:** `validation.rs`, `eval.rs`

`walkforward_risk_diagnostics` passes `days` into `simulate_trades_core`, but `simulate_trades_core` expects real timestamps.

**Risk:** duration, gap detection, kill-zone/weekend rules, internal day bucketing, and max-trades-per-day can be wrong inside WFV diagnostics.

**Severity:** High.

**Fix direction:** `WalkforwardBacktestInput` should carry both real `timestamps` and day buckets. Pass timestamps to simulation, use day buckets only for aggregation.

---

### 15. UI discovery and batch discovery slice data differently

**Files:** `forex-app/src/app_services/discovery.rs`, `orchestration.rs`

UI/app discovery cuts the dataset to an 80% in-sample region before discovery. Batch orchestrator calls `run_discovery_cycle` on the full feature frame and OHLCV.

**Risk:** “discovery” has different semantics depending on entrypoint.

**Severity:** Medium-High.

**Fix direction:** create one shared discovery slicing policy used by UI, CLI, batch, and tests.

---

### 16. Stage-1 search window may be narrower than expected

**File:** `discovery.rs`

Discovery does stage-1 fast evaluation on a recent fraction. If app service already cut data to 80% in-sample, stage-1 may only search on the last part of that in-sample subset.

**Risk:** stronger overfitting to recent in-sample conditions.

**Severity:** Medium.

**Fix direction:** every run should log/export exact windows: full dataset, in-sample, stage-1, validation/OOS.

---

### 17. Canonical day/week/month ledger is missing

**Files:** `eval.rs`, `cubecl_eval.rs`, `quality.rs`, `validation.rs`

Current code has pieces of period tracking: daily DD and monthly PnL in eval, monthly consistency in quality, daily blocks for Monte Carlo. But there is no single canonical period ledger.

A trader needs period knowledge: daily PnL/DD, weekly rhythm, monthly consistency, payout-cycle behavior, risk concentration, recovery after losing periods, and rule breaches.

**Severity:** High.

**Fix direction:** replace the narrow “flush final month” idea with a full day/week/month period ledger emitted by reference backtest and mirrored by GPU parity evaluator.

---

### 18. Final open period is not safely closed

**Files:** `eval.rs`, `cubecl_eval.rs`

Monthly PnL appears written when month changes. The current/open final month can be omitted before Sharpe/consistency. Similar risk exists for any future day/week/month ledger if open periods are not finalized.

**Severity:** High.

**Fix direction:** finalize all open day/week/month periods at backtest end. Add tests for single-day, single-week, and single-month backtests.

---

### 19. Fast metrics and trade simulation appear to differ on kill-zone/session rules

**File:** `eval.rs`

`simulate_trades_core` contains session-aware Friday force-exit and Monday/Friday entry blocking. The visible `fast_evaluate_strategy_core` path does not show equivalent logic.

**Risk:** search metrics and detailed trade logs can disagree.

**Severity:** High.

**Fix direction:** share one position state machine. Metrics and trade logs should be two output modes of the same simulation logic.

---

### 20. `signals_for_gene` used outside evaluator does not apply SMC gating

**Files:** `search_engine.rs`, `discovery.rs`, `gauntlet.rs`

Population evaluator synthesizes signals with indicator weights plus SMC gating/flags. `signals_for_gene(features, gene)` computes signals from weighted features and thresholds only.

That function appears used by validation/gauntlet/discovery trade-log style paths.

**Risk:** a gene searched with SMC gating can later be trade-logged or quality-tested with signals that ignore its SMC flags.

**Severity:** High.

**Fix direction:** expose one canonical signal synthesis function used everywhere: CPU evaluator, CUDA signal kernel, trade logs, quality screen, gauntlet, WFV/CPCV.

---

### 21. Metric layout is implicit and duplicated

**Files:** `eval.rs`, `cubecl_eval.rs`, `validation.rs`, `discovery.rs`

Evaluation returns `[f64; 11]` and downstream modules depend on numeric indexes. CUDA has a smaller internal metric width and reconstructs/derives the full array.

**Risk:** index mismatch can silently corrupt selection, validation, UI display, or export.

**Severity:** Medium-High.

**Fix direction:** replace raw arrays with typed `BacktestMetrics`, or at least define named constants for all indexes.

---

### 22. Gauntlet must share the same contract

**File:** `gauntlet.rs`

Gauntlet uses its own settings/defaults and appears to use `signals_for_gene`, not necessarily the same SMC-gated signal path as search.

**Risk:** gauntlet becomes another inconsistent validation layer.

**Severity:** Medium-High.

**Fix direction:** gauntlet should use the same canonical search/backtest contract and canonical signal synthesis as search and validation.

---

### 23. Stop/target inference can use future data when defaults are inferred from the full OHLCV slice

**Files:** `search_engine.rs`, `stop_target.rs`

`resolve_stop_target_arrays` calls `infer_stop_target_pips` using the full OHLCV arrays when a gene has invalid/missing SL/TP. The stop-target helper contains volatility/regime logic based on trailing slices and `.last()` style calculations over the supplied data.

**Risk:** if SL/TP defaults are inferred from the whole search/evaluation slice, the stop/target choice can be influenced by future volatility/regime information. This is especially dangerous if the same inferred defaults are then used for all bars in the backtest or during validation.

**Severity:** High.

**Fix direction:** SL/TP inference must be causal. Either genes must carry fixed SL/TP values created before validation, or stop/target inference must be computed per bar using only data available up to that bar, or per train split using only train data and then frozen for test/OOS.

---

### 24. Initial balance is not part of one canonical backtest contract

**Files:** `validation.rs`, `quality.rs`, `forex-core/src/config.rs`, `discovery.rs`

`validation.rs` hardcodes `WALKFORWARD_INITIAL_BALANCE = 100_000.0`. `RiskConfig` has its own `initial_balance` default. `quality.rs` receives `initial_balance` from the caller. Search/eval metrics also need a consistent base equity to calculate percentages and drawdowns.

**Risk:** daily returns, max daily DD, total DD, Calmar, monthly returns, and prop-compliance checks can be calculated against different account sizes depending on module.

**Severity:** High.

**Fix direction:** `initial_balance` must be part of the canonical `SearchBacktestContract` and passed into search, validation, quality, ledger, CPU evaluator, and GPU evaluator.

---

### 25. Some “daily/weekly” quant features use fixed bar counts instead of timeframe-aware calendar windows

**File:** `forex-data/src/core/quant_features.rs`

`quant_features.rs` contains features described as previous day/week distances using fixed periods such as 24 and 120 bars, with comments like “proxy for previous day on H1”. But the same feature function is called for whatever timeframe is being processed.

**Risk:** on M1, 24 bars is 24 minutes, not a previous day. On M5, 24 bars is two hours. On H1 it approximates a day. This can make feature meanings inconsistent across base and higher timeframes and can mislead the GA.

**Severity:** Medium-High.

**Fix direction:** timeframe-dependent features must receive timeframe metadata and convert calendar periods into bar counts, or be renamed honestly as fixed-bar lookback features.

---

### 26. Higher timeframe list can duplicate the base timeframe

**Files:** `forex-core/src/config.rs`, `forex-data/src/lib.rs`

System config defaults include many higher timeframes, including `M1`. `prepare_multitimeframe_features_with_options` always pushes base features first, then aligns every requested higher timeframe if present. If `higher_tfs` includes the base timeframe, the base feature set can be duplicated under a timeframe prefix.

**Risk:** duplicated columns can overweight the base timeframe, create redundant genes, and make archive diversity look better than it really is.

**Severity:** Medium.

**Fix direction:** remove `base_tf` from `higher_tfs` before alignment, or explicitly allow duplication only when intended and log it.

---

### 27. Duplicate timestamp handling drops rows instead of reconciling market data

**File:** `forex-data/src/lib.rs`

`normalize_ohlcv` sorts rows by timestamp and then deduplicates by timestamp. This keeps one row and drops the rest.

**Risk:** if duplicate timestamps come from broker corrections, overlapping imports, or partial candles, data can be silently lost. Depending on ordering, the kept row may not be the final/correct candle.

**Severity:** Medium.

**Fix direction:** duplicate timestamp handling should be explicit: reject duplicates, keep last with log, or aggregate deterministically according to OHLCV rules. Silent dedup should not be allowed for backtest-grade data.

---

### 28. Backtest appears to use fixed pip-value / implicit fixed position sizing, not live dynamic risk sizing

**Files:** `eval.rs`, `cubecl_eval.rs`, `forex-core/src/domain/risk.rs`, `forex-core/src/config.rs`

The backtest PnL path uses pip movement multiplied by `pip_value_per_lot` / pip-value style settings. The live risk module has a much richer position sizing model with equity, confidence, uncertainty, volatility targeting, Kelly-style scaling, recovery mode, and drawdown-aware risk reduction.

**Risk:** the search can rank strategies under an implicit fixed-size model while live trading resizes positions dynamically. That changes return, drawdown, daily loss risk, recovery behavior, and prop-firm compliance. A strategy that is good under fixed sizing is not necessarily good under dynamic risk sizing, and vice versa.

**Severity:** High.

**Fix direction:** add position sizing mode to the canonical backtest contract: fixed lot, fixed risk percent, volatility-targeted risk, or live-risk-manager parity. The same sizing policy must be used in search, validation, quality reports, and live deployment.

---

### 29. Live risk gates are not represented as a first-class search/backtest gate

**Files:** `forex-core/src/domain/risk.rs`, `eval.rs`, `validation.rs`, `discovery.rs`

The live `RiskManager` can block trades based on total/daily/intraday drawdown, recovery mode, circuit breaker, monthly/challenge target reached, confidence threshold, session window, night block, news kill window, revenge-trade detection, max trades per day, and strategy rank/sharpe restrictions under drawdown.

**Risk:** search/backtest can accept a strategy that live runtime would frequently block. This means the backtested equity curve may not match the deployable equity curve.

**Severity:** High.

**Fix direction:** model live risk gates in the backtest engine, or export a separate `live_gate_adjusted_backtest`. At minimum, report the number of trades that would have been blocked by live risk rules.

---

### 30. Classic VectorTA feature generation uses zero timestamps

**File:** `forex-data/src/core/hpc_ta.rs`

`compute_classic_ta_columns` builds a `Candles` object for VectorTA with `timestamps = vec![0i64; n]` instead of the real OHLCV timestamps.

**Risk:** most classic price-only indicators may still work, but any indicator that depends on timestamps, sessions, day boundaries, anchored periods, or calendar logic will be wrong or degenerate. This also makes feature parity weaker between modules that use real timestamps and modules that do not.

**Severity:** Medium-High.

**Fix direction:** pass real timestamps into the VectorTA candle object. Add a test that a time-dependent indicator changes when timestamps change.

---

### 31. FeatureFrame construction does not clearly sanitize non-finite feature values

**Files:** `forex-data/src/lib.rs`, `forex-data/src/core/features.rs`, `cubecl_eval.rs`, `eval.rs`

`compute_hpc_feature_frame` copies feature columns into an `Array2<f32>` without a visible central non-finite cleanup stage. Feature alignment can also return `NaN` for unavailable higher-timeframe rows.

**Risk:** NaN/Inf values can enter CPU and GPU signal synthesis. CPU and GPU can handle NaNs differently, and threshold comparisons with NaN usually produce no signal. That can silently bias early rows, HTF alignment gaps, or entire indicators.

**Severity:** Medium-High.

**Fix direction:** introduce a feature sanitation policy at the `FeatureFrame` boundary: reject non-finite values, fill with a documented neutral value, add validity masks, or drop warmup rows. The same policy must run before CPU and GPU evaluation.

---

### 32. Final open position handling must be explicit and parity-tested

**Files:** `eval.rs`, `cubecl_eval.rs`, `quality.rs`

The visible full CUDA kernel computes final net profit after the main loop, and no explicit final close of an open position is visible in the kernel excerpt. CPU behavior must be checked and matched exactly.

**Risk:** if one path leaves final positions unrealized while another force-closes or mark-to-markets them, net profit, trade count, win rate, PF, expectancy, daily/monthly ledger, and prop compliance can diverge.

**Severity:** Medium-High.

**Fix direction:** define one end-of-test policy: force close at final close, mark-to-market open positions separately, or reject windows with open positions. Apply the same policy in CPU metrics, CUDA metrics, trade logs, quality reports, WFV/CPCV, and live replay.

---

## Additional findings appended 2026-05-03 16:02 Europe/Berlin

### 33. UI withholds OOS rows but does not appear to replay/export mandatory OOS validation

**Files:** `forex-app/src/app_services/discovery.rs`, `forex-search/src/discovery.rs`

The UI discovery service cuts the data to 80% in-sample before running `run_discovery_cycle_with_progress`. It then saves the resulting portfolio, profile, quality report, and trade logs. Current reading does not show an automatic replay on the withheld final 20% before saving/exporting the portfolio.

**Risk:** the code may give a false sense of OOS protection. Withholding 20% is only useful if strategies are later replayed on that withheld slice and the OOS result gates export. Otherwise the final 20% is merely unused, not validating anything.

**Severity:** High.

**Fix direction:** after discovery on the 80% in-sample segment, replay the selected candidates on the withheld 20% OOS segment using the same canonical backtest contract. Export OOS metrics and block portfolio save unless configured OOS gates pass.

---

### 34. Discovery profile records validation configuration but not validation execution results

**Files:** `forex-search/src/discovery.rs`, `forex-app/src/app_services/discovery.rs`

`DiscoveryRunProfile` records fields such as `walkforward_splits`, `embargo_minutes`, `enable_cpcv`, `cpcv_n_splits`, and filter settings. It also records observed candidate and portfolio counts. Current reading does not show executed WFV/CPCV/OOS result objects in the profile.

**Risk:** exported metadata can look validation-aware because it contains WFV/CPCV config values, while the profile may not prove that validation actually ran or passed.

**Severity:** Medium-High.

**Fix direction:** add explicit execution/result fields: `oos_tested`, `oos_passed`, `walkforward_tested`, `walkforward_passed`, `cpcv_tested`, `cpcv_passed`, plus summary metrics and failure reasons.

---

### 35. Regime robustness gate uses a hardcoded absolute account loss assumption

**File:** `forex-search/src/discovery.rs`

`validate_regime_robustness` rejects if regime-specific PnL falls below `-3000.0`, with a comment assuming a 100k account and 3% loss. This is disconnected from the configured initial balance and prop-firm risk contract.

**Risk:** on a 10k account, -3000 is a 30% loss and far too loose. On a 200k account, -3000 is 1.5% and may be too strict. This can distort regime validation and candidate selection.

**Severity:** High.

**Fix direction:** compute regime loss limits from the canonical `initial_balance` and configured max regime loss percentage. Export regime-specific PnL/DD as part of the period/regime ledger.

---

### 36. Portfolio export lacks a complete feature/backtest schema contract

**Files:** `forex-search/src/discovery.rs`, `forex-app/src/app_services/discovery.rs`

`GeneExport` exports strategy id, selected indicator names, feature indices, weights, thresholds, fitness, Sharpe, win rate, TP, and SL. Current reading did not show a full feature schema hash, timeframe contract, timestamp unit, HTF alignment policy, feature sanitation policy, cost model, execution policy, or validation-gate summary in the portfolio artifact.

**Risk:** a portfolio can be loaded later against a different feature order, feature set, timeframe list, data timing convention, cost model, or live execution policy. Since genes depend on feature indices, this can silently change strategy behavior.

**Severity:** High.

**Fix direction:** export a `PortfolioContract` alongside every portfolio: feature names and hash, base/higher timeframes, timestamp unit, HTF availability policy, feature sanitation policy, cost/slippage/commission model, sizing mode, execution policy, validation gates/results, git commit, and config hash. Refuse to load/deploy if the live contract does not match.

---

### 37. UI completion summary still ranks “best” by fitness rather than validation/quality contract

**File:** `forex-app/src/app_services/discovery.rs`

`completed_snapshot` selects the best displayed strategy from the portfolio by `gene.fitness`. It separately reports best quality strategy if quality metrics exist, but the main best strategy highlight still uses fitness.

**Risk:** the UI can promote a strategy based on composite fitness rather than validated OOS/quality/risk-adjusted results. This reinforces the earlier `fitness != net_profit` problem at the user-facing layer.

**Severity:** Medium.

**Fix direction:** choose UI “best strategy” from a typed ranking contract: OOS pass first, WFV/CPCV pass, quality score, live-gate-adjusted expectancy, and drawdown safety. Display raw fitness only as an internal search score.

---

## GPU-first architecture note

The target is valid: expensive search/backtest/training-adjacent work should run mostly on GPU.

But GPU-first is only safe if GPU means **semantic-parity GPU**, not just “a faster metric on GPU”. A parity GPU path must match CPU/live/export contract on:

- timestamp unit,
- HTF feature availability timing,
- timeframe-aware feature windows,
- feature sanitation / non-finite policy,
- feature schema hash and load-time contract checks,
- causal stop/target inference,
- initial balance,
- fixed vs dynamic position sizing,
- live risk gates,
- slippage/spread/commission,
- live execution policy such as partial TP and entry patience,
- signal timing,
- canonical signal synthesis including SMC gating,
- position state machine,
- final open-position policy,
- TP/SL/trailing rules,
- pip value / lot sizing contract,
- max hold/min hold,
- max trades/day,
- gap handling,
- kill-zone/session rules,
- broker timezone/day boundary,
- day/week/month ledger,
- OOS/WFV/CPCV validation gates,
- metric layout.

Tensor/HPC discovery can remain a fast approximate presearch path, but final acceptance should require parity backtest.

---

## Recommended next action order

1. Fix HTF resampling/alignment leakage.
2. Normalize timestamp units across data/search/backtest/validation/GPU.
3. Fix CUDA full backtest prior-bar signal timing.
4. Add kill-zone/session/broker-day parity to CUDA or disable CUDA full backtest as final acceptance path until implemented.
5. Introduce one canonical search/backtest/execution contract from config/risk/symbol metadata, including initial balance, slippage, sizing mode, and live risk gates.
6. Add canonical signal synthesis shared by CPU, CUDA, trade logs, gauntlet, and validation.
7. Fix or freeze causal SL/TP inference.
8. Add mandatory OOS replay on withheld UI data before saving/exporting portfolios.
9. Decide whether live-only partial TP / entry patience must be simulated or disabled for validated strategies.
10. Fix walk-forward diagnostics to pass true timestamps into `simulate_trades_core`.
11. Build and return day/week/month period ledger; close all open periods at test end.
12. Unify fast metrics and trade simulation via one position state machine.
13. Add a feature sanitation contract for NaN/Inf/warmup rows.
14. Pass real timestamps into VectorTA classic indicator computation.
15. Export a portfolio contract with feature schema hash, config hash, validation results, and git commit.
16. Separate `fitness_score` from `net_profit` and UI best-strategy ranking.
17. Replace raw metric indexes with typed metrics or named constants.
18. Make timeframe-dependent features timeframe-aware.
19. Make duplicate timestamp handling explicit and audited.
20. Enforce validation flags in discovery output.
21. Wire WFV/CPCV as optional-but-real hard gates.
22. Treat tensor/HPC GPU discovery as approximate presearch unless parity evaluator is used.
23. Add CI coverage for `forex-search` and GPU-relevant feature combinations.
24. Add deterministic CPU-vs-GPU-live parity tests.

---

## Open verification checklist

- [ ] Confirm whether `validation.rs` functions are called from all relevant discovery entrypoints.
- [ ] Confirm whether `CombinatorialPurgedCV` affects final portfolio acceptance.
- [ ] Confirm whether UI withheld 20% OOS is replayed before portfolio export.
- [ ] Confirm exact metric layout in CPU evaluator vs cubecl evaluator.
- [ ] Add deterministic CPU/GPU backtest parity fixture.
- [ ] Add HTF leakage regression test using synthetic M1 → H1 resampling.
- [ ] Confirm whether exported portfolios record which validation gates actually ran.
- [ ] Confirm CI runs `cargo check/test -p forex-search` and relevant feature combinations.
- [ ] Add timestamp unit tests for ms/us/ns input vectors.
- [ ] Add CUDA timestamp delta/gap-exit tests.
- [ ] Add CUDA kill-zone/session parity tests.
- [ ] Add slippage-cost parity tests.
- [ ] Add dynamic sizing and live-risk-gate replay tests.
- [ ] Add OOS replay/export gate tests.
- [ ] Add portfolio schema-hash/load-contract tests.
- [ ] Add regime robustness threshold tests using different initial balances.
- [ ] Add partial-TP/entry-patience backtest vs live execution tests, or assert those features are disabled for validated portfolios.
- [ ] Add final open-position policy tests.
- [ ] Add single-day/week/month tests to ensure final periods are included.
- [ ] Add daily/weekly/monthly ledger tests.
- [ ] Add kill-zone parity tests between fast metrics and full trade simulation.
- [ ] Add signal synthesis parity tests: CPU evaluator vs `signals_for_gene` vs CUDA signal kernel.
- [ ] Add SMC-gated signal parity tests.
- [ ] Add causal SL/TP inference tests.
- [ ] Add timeframe-aware quant feature tests.
- [ ] Add duplicate-feature guard tests when `higher_tfs` includes `base_tf`.
- [ ] Add duplicate timestamp ingest tests.
- [ ] Add feature sanitation tests for NaN/Inf/warmup values.
- [ ] Add VectorTA timestamp propagation test.

---

## Bottom line

The bot already has serious pieces for search, GPU acceleration, WFV/CPCV utilities, quality analysis, gauntlet filtering, live order execution, and live risk management.

But `master` currently has multiple search/backtest/validation/live-execution/export paths with different semantics. The next milestone should be one documented contract for data timing, timestamp unit, timeframe-aware features, feature sanitation, feature schema, causal stop/target inference, cost/slippage model, initial balance, position sizing, live risk gates, execution policy, signal timing, SMC signal synthesis, position state machine, final open-position policy, period ledger, metric layout, validation gates, and portfolio export/load safety — with CPU/GPU/live-execution/OOS parity tests around that contract.
