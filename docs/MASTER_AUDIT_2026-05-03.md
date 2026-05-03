# Forex-AI Master Audit

**Created:** 2026-05-03 Europe/Berlin  
**Repository:** kosred/forex-ai  
**Branch at time of audit:** master  
**Consolidates:** dead_code, python_pyo3_legacy, rust_env_flags, search_backtest_forward_cpu_gpu, search_discovery_pipeline, search_portfolio_artifact, ctrader_api_test_report, gpu_rollout_report, search_cpu_gpu_optimization_notes  
**Status key:** ✅ FIXED · 🔄 IN PROGRESS · ⬜ OPEN

---

## CRITICAL — Show-stoppers for prop-firm trading

| ID | File(s) | Issue | Status |
|---|---|---|---|
| C1 | `forex-data/src/core/resample.rs`, `features.rs` | **HTF resampling lookahead.** Resampled H1/H4/D1 candles are timestamped at bucket-start while carrying OHLC from the full closed bucket. `align_features_by_ns` then makes lower-TF rows see completed HTF values before the HTF bar actually closes. | ⬜ OPEN |
| C2 | `resample.rs`, `search_engine.rs`, `eval.rs`, `validation.rs`, `quality.rs`, `cubecl_eval.rs` | **Timestamp unit mismatch.** Data/resampling uses nanoseconds (`1_000_000_000`). Search/eval/quality/CUDA uses milliseconds (`86_400_000`, `3_600_000`, `timestamp_millis_opt`). Day/month bucketing, gap exits, kill-zone logic, and Monte Carlo blocks all silently break when ns timestamps are passed in. | ⬜ OPEN |
| C3 | `cubecl_eval.rs` | **CUDA entry timing lookahead.** Full CUDA backtest opens positions using `signals_flat[signal_base + i]` (current bar). CPU evaluator uses `signals[i-1]` (prior bar). CUDA strategies are therefore ranked under a 1-bar lookahead advantage. | ⬜ OPEN |
| C4 | `cubecl_eval.rs`, `eval.rs` | **CUDA session/kill-zone contract missing.** CUDA kernel models max-hold, spread, commission, and pip-value but does not implement kill zones, Friday force-exit, Monday/Friday entry blocks, broker-timezone day boundaries, or session windows. CPU `simulate_trades_core` does. | ⬜ OPEN |
| C5 | `discovery.rs`, `forex-app/src/app_services/discovery.rs`, `forex-cli` | **Feature-index / feature-name mismatch after prefilter.** `run_discovery_cycle_with_progress` may call `prefilter_features`, replacing the feature matrix and name list inside discovery. Discovered gene indices are relative to the filtered matrix. All callers (`save_portfolio_json`) supply the original `features.names` from before prefiltering. Exported indicator names can point to the wrong columns. | ⬜ OPEN |
| C6 | `.github/workflows/ci.yml` | **CI does not cover `forex-search` or `forex-models`.** `cargo check/test` runs only for `forex-app`, `forex-core`, `forex-news`. Search/backtest/model regressions land without CI catching them. | 🔄 IN PROGRESS |
| C7 | `discovery.rs`, `genetic/search_engine.rs`, `cubecl_eval.rs`, etc. | **Search/discovery semantics controlled by env flags.** Examples: `FOREX_BOT_PREFILTER_TOP_K`, `FOREX_BOT_PREFILTER_INSAMPLE`, `FOREX_BOT_FUNNEL_STAGE1_PCT`, `FOREX_BOT_SEARCH_SEED`, SMC gate/schedule knobs, archive mode/thresholds, survivor policy, novelty weight, backtest initial equity. Two runs with identical config + data produce different portfolios when env differs. | ⬜ OPEN |
| C8 | `forex-core/src/domain/risk.rs` | **`RiskManager` is dead code — not wired.** The advertised prop-firm risk guard (`risk.rs` ~700 LoC) has zero callers anywhere in the workspace. `prop_firm_pre_trade_check` in `trading.rs` uses a weaker inline replacement. | ⬜ OPEN |

---

## HIGH — Significant correctness and safety issues

### Prop-firm execution safety

| ID | File:line | Issue | Status |
|---|---|---|---|
| H1 | `trading.rs:1715-1725` | **`ctrader_account_equity` returns only `balance`, drops unrealized PnL.** Open losing positions do not count toward the 4.5% daily-loss limit until they close. Account can blow before a single close event. | ✅ FIXED |
| H2 | `trading.rs:1727-1735` | **Hard-coded pip position lookup** (`JPY→2, XAU→2, else 4`). Wrong for crypto, most indices, other metals, and 5-digit FX brokers. Must read `digits`/`pipPosition` from cached `ProtoOASymbol` metadata. | ✅ FIXED (FX-only lookup, full metadata cache ⬜ OPEN) |
| H3 | `trading.rs:3317` | **Hard-coded `$10 per pip per standard lot`.** `estimated_loss = pip_distance × (volume/100.0) × 10.0`. Wrong for cross pairs and non-USD accounts. Must use `pip_value_per_lot` from symbol metadata × quote→account FX rate. | ✅ FIXED (estimate improved; full cross-pair rate ⬜ OPEN) |
| H4 | `trading.rs:3308-3328` | **`risk_per_trade` violation only warns, order proceeds.** | ✅ FIXED (returns `Err`) |
| H5 | `trading.rs:1152,1191` | **`day_start_equity` never resets at calendar day boundary.** Daily-loss limit silently becomes a session-since-startup limit. | ✅ FIXED (`handle_day_boundary` added) |
| H6 | `trading.rs:1149-1151` | **`initial_equity` set once on first connect, no prop-firm phase rollover** (challenge → verification → funded). | ✅ FIXED (`handle_phase_rollover` added) |
| H7 | `ctrader_execution.rs:417-457` | **`ensure_authenticated` retries with the same expired token.** No call to `refresh_token_bundle`. Mid-session token expiry kills trading. | ⬜ OPEN |
| H8 | `ctrader_execution.rs:804-859` | **`validate_execution_outcome` never checks execution status.** `PartialFill` treated identically to `Filled`. | ✅ FIXED (PartialFill returned as error) |
| H9 | `ctrader_execution.rs` | **`read_matching_response` blocks indefinitely on mismatched payloads.** No socket read timeout. | ✅ FIXED (30s `tokio::time::timeout`) |

### Search determinism and ranking

| ID | File:line | Issue | Status |
|---|---|---|---|
| H10 | `genetic/evolution_math.rs:453,543,650` | **Unseeded RNG in GA helpers.** `new_random_gene`, `crossover`, `mutate` each instantiate a fresh `rand::rng()`. Config seed is ignored for GA offspring. | ✅ FIXED (`&mut StdRng` threaded through) |
| H11 | `cubecl_eval.rs:425-441` | **`mean_std` on GPU does not filter non-finite samples.** CPU version does. One NaN `month_pnl` gives different Sharpe between CPU and GPU → strategies rank differently. | ✅ FIXED (NaN filter + Bessel correction) |
| H12 | `cubecl_eval.rs:390-396` | **GPU profit-factor not capped.** CPU caps at 10.0. Tiny `gross_loss` produces 1e6 PF that destabilises sort. | ✅ FIXED (capped at 10.0) |
| H13 | `genetic/strategy_gene.rs:178-180` | **`estimate_pip_value_per_lot` wrong for cross pairs.** Uses symbol's own price instead of quote→account conversion rate. Same class as H3 in backtest cost model. | ✅ FIXED (quote_to_account_rate parameter added) |
| H14 | `quality.rs:177-181` | **Calmar = 0 for flawless equity curve.** `if max_dd < 1e-9 → 0.0` instead of saturating. | ✅ FIXED (returns 1000.0 when max_dd < 1e-9 and net > 0) |
| H15 | `eval.rs:387` | **1-bar intra-bar lookahead.** `signals[i]` drives entry at bar `i` using bar-`i` close/high/low. Must be `signals[i-1]`. | ✅ FIXED |
| H16 | `discovery.rs` | **Regime robustness gate uses hardcoded `-3000.0` absolute loss.** Assumes 100k account, 3% rule. On a 10k account this is 30% — dangerously loose. Must compute from `initial_balance × max_regime_loss_pct`. | ⬜ OPEN |
| H17 | `validation.rs` | **`WALKFORWARD_INITIAL_BALANCE = 100_000.0` hardcoded.** All WFV daily/monthly metrics and prop-compliance checks run against this fixed base, not the actual configured balance. | ⬜ OPEN |
| H18 | `validation.rs` / `eval.rs` | **Walk-forward diagnostics pass `days` as timestamps into `simulate_trades_core`.** Duration, gap detection, kill-zone/weekend rules, and max-trades/day calculations are wrong inside WFV. | ⬜ OPEN |
| H19 | `eval.rs`, `cubecl_eval.rs` | **Final open month / final open period not safely closed before metric computation.** Monthly PnL is written on month change; the last open month can be omitted from Sharpe and consistency. | ⬜ OPEN |
| H20 | `orchestration.rs:100` | **Batch discovery aborts the whole batch on a single `run_discovery_cycle` error** via `?`, while all other errors in the same loop increment `summary.skipped_*` and continue. | ✅ FIXED |
| H21 | `quality.rs:307-365` | **Monthly consistency: months with 1 trade have the same weight as months with 50.** No minimum-trades guard. | ✅ FIXED (`min_trades_per_month = 4`) |
| H22 | `gauntlet.rs:75-89` | **`warn_only=true` returns true without logging what failed.** | ✅ FIXED (emits `tracing::warn!` describing failed metric) |

### Search pipeline consistency

| ID | File(s) | Issue | Status |
|---|---|---|---|
| H23 | `search_engine.rs`, `discovery.rs`, etc. | **Archive dedup by strategy_id, not canonical gene signature.** Duplicate/near-duplicate genes survive under different IDs, inflating archive diversity. | ✅ FIXED (dedup by `gene_signature_hash`) |
| H24 | `search_engine.rs`, `discovery.rs`, `gauntlet.rs` | **`signals_for_gene` used outside evaluator does not apply SMC gating.** Population evaluator synthesizes signals with SMC flags; `signals_for_gene` computes weighted features only. Validation/gauntlet/quality paths use the simplified version. | ✅ FIXED (changed to `signals_for_gene_full`) |
| H25 | `discovery.rs`, `eval.rs`, `gauntlet.rs` | **Signal count used as proxy for trade count.** Non-zero signal count ≠ executed trades (depends on SL/TP, max-hold, position state, gaps, kill zones). | ⬜ OPEN |
| H26 | `forex-app/src/app_services/discovery.rs`, `orchestration.rs` | **Batch discovery uses `prepare_multitimeframe_features(&ds_ready, tf, &[], None)` — empty higher-TF list.** CLI/UI can pass configured higher TFs. Same symbol/config produces different feature universe per entrypoint. | ⬜ OPEN |
| H27 | `forex-app/src/app_services/discovery.rs` | **UI withholds 20% OOS from search but does not replay/export mandatory OOS validation before portfolio save.** The 20% is unused, not validating. | ⬜ OPEN |
| H28 | `discovery.rs` | **`DiscoveryResult` does not carry `effective_feature_names` or `feature_mapping`.** Callers save portfolio with original pre-prefilter names (see C5). | ⬜ OPEN |
| H29 | `discovery.rs`, `search_engine.rs` | **`fitness` used with different meanings across paths.** GA composite score, candidate ranking formula, and UI/export imply income/profit. Must separate `search_score`, `net_profit`, `quality_score`, `oos_score`, `portfolio_score`. | ⬜ OPEN |
| H30 | Various | **No `SearchRunContract` / `PortfolioContract`.** Exported portfolio does not prove which data was searched, which features were selected, which seed was used, which evaluator accepted it, which validation gates ran, why a candidate ranked above another. | ⬜ OPEN |
| H31 | `discovery.rs` | **Discovery not fully deterministic.** No single run-seed propagated to all search/discovery paths (`discovery.rs`, `discovery_gpu.rs`, `hpc_gpu_discovery.rs`, `cubecl_ga.rs`, `quality.rs`, `forex-models/genetic.rs`). | ⬜ OPEN |
| H32 | `forex-app/src/app_services/discovery.rs`, `orchestration.rs` | **UI/batch discovery slice data differently.** UI cuts to 80% in-sample. Batch orchestrator passes the full feature frame. Same config produces different candidates. | ⬜ OPEN |
| H33 | `discovery.rs`, `search_engine.rs` | **Search/quality/gauntlet can use different cost/session contracts.** `EvaluationConfig` vs `BacktestSettings` defaults leave kill zones, min-hold, max-trades/day, gap threshold unset in some paths. | ⬜ OPEN |
| H34 | `discovery_gpu.rs`, `hpc_gpu_discovery.rs` | **Tensor/HPC GPU discovery uses simplified return-based fitness, not SL/TP/spread/commission semantics.** Documented as approximate presearch, but can be mistaken for final validated strategies. | ⬜ OPEN |
| H35 | `eval.rs`, `cubecl_eval.rs` | **Backtest uses fixed pip-value/lot-size, not live dynamic risk sizing.** Live `RiskManager` has equity-based, volatility-targeted, Kelly-scaled sizing. A strategy good under fixed sizing may not be good under dynamic sizing, and vice versa. | ⬜ OPEN |
| H36 | `eval.rs`, `cubecl_eval.rs` | **Live risk gates not modelled in backtest.** Circuit breaker, recovery mode, daily/intraday DD blocks, revenge-trade detection, confidence threshold, etc. can block trades live that were counted in search. | ⬜ OPEN |
| H37 | `stop_target.rs`, `search_engine.rs` | **Stop/target inference can use future data.** `resolve_stop_target_arrays` calls `infer_stop_target_pips` on the full OHLCV slice. SL/TP defaults can be influenced by future volatility/regime. | ⬜ OPEN |
| H38 | `eval.rs`, `cubecl_eval.rs` | **Backtest slippage not first-class.** `RiskConfig` has `slippage_pips` / `slippage_guard_multiplier`. CPU/CUDA backtest settings carry spread and commission but not slippage. Edge disappears under realistic slippage for M1/M5. | ⬜ OPEN |
| H39 | `validation.rs` | **WFV/CPCV not a hard discovery gate.** `embargoed_walkforward_backtest` and `CombinatorialPurgedCV` exist but are not shown as mandatory gates in candidate acceptance. `DiscoveryRunProfile` records WFV config but not executed validation results. | ⬜ OPEN |

### Model training

| ID | File:line | Issue | Status |
|---|---|---|---|
| H40 | `training_orchestrator.rs:3043-3235` | **Post-HPO full-data refit records `val_rows=0` in metadata**, hiding the original split. | ⬜ OPEN |
| H41 | `runtime/hpo.rs` | **`embargo_rows` can be 0 if `embargo_minutes==0`.** No floor. Train→val label leakage. | ⬜ OPEN |
| H42 | `deep_models.rs:1341-1388` | **Burn early stopping monitors training loss, not validation loss.** No `val_frame` passed to early-stop monitor. | ⬜ OPEN |
| H43 | `tree_models/{lightgbm,xgboost,catboost}.rs` | **Tree model fits: no `eval_set` passed.** Boosters run full `num_iterations`, no early stopping. | ⬜ OPEN |
| H44 | `anomaly/forest_impl.rs` | **Anomaly threshold computed from training scores, not validation.** Optimistic threshold. | ⬜ OPEN |

---

## MEDIUM-HIGH — Correctness drift, ranking quality

| ID | File(s) | Issue | Status |
|---|---|---|---|
| MH1 | `forex-core/src/config.rs:225-341` | **Risk config fields lack serde bound validation.** `risk_per_trade: 50` (meaning 50%) silently passes. Need custom `Deserialize` validators clamping to safe ranges with `tracing::error!`. | ⬜ OPEN |
| MH2 | `forex-core/src/config.rs:104` | **`session_timezone` hard-coded "UTC".** cTrader prop-firm reset typically EET. Need `broker_timezone` field validated against account metadata at bootstrap. | ✅ FIXED (`broker_timezone` added) |
| MH3 | `ctrader_data.rs` | **Symbol metadata (`pip_size`, `contract_size`, `min_stop_distance`) fetched on every order.** Must be cached with a TTL (e.g., 1-hour, refresh at first trade of day). | ⬜ OPEN |
| MH4 | `eval.rs` | **`fast_evaluate_strategy_core` lacks Friday force-exit and Monday/Friday entry blocking** present in `simulate_trades_core`. Fast metrics and detailed trade logs disagree on session-aware behavior. | ⬜ OPEN |
| MH5 | `eval.rs`, `cubecl_eval.rs`, etc. | **Raw metric layout: `[f64; 11]` with implicit numeric indexes.** CUDA has a narrower internal metric width and reconstructs the full array. Index mismatch silently corrupts selection, validation, UI, or export. | ⬜ OPEN |
| MH6 | `gauntlet.rs` | **Gauntlet uses its own settings and `signals_for_gene` (simplified).** Not the canonical search/backtest contract or SMC-gated signal path. | ⬜ OPEN |
| MH7 | `forex-data/src/core/quant_features.rs` | **"Daily/weekly" quant features use fixed bar counts (`24`, `120`) regardless of timeframe.** On M1, 24 bars = 24 minutes. On H1, 24 bars ≈ 1 day. Features mislead the GA on non-H1 timeframes. | ⬜ OPEN |
| MH8 | `forex-core/src/config.rs`, `forex-data/src/lib.rs` | **`higher_tfs` can include `base_tf`.** `prepare_multitimeframe_features_with_options` pushes base features first, then processes every requested HTF. Duplicates overweight the base TF and inflate archive diversity. | ⬜ OPEN |
| MH9 | `forex-data/src/lib.rs` | **Duplicate timestamp rows silently dropped (`dedup_by`).** If duplicates come from broker corrections or partial candles, the wrong row may be kept. Must be explicit: reject, keep-last-with-log, or aggregate per OHLCV rules. | ⬜ OPEN |
| MH10 | `forex-data/src/core/hpc_ta.rs` | **`compute_classic_ta_columns` builds VectorTA `Candles` with `timestamps = vec![0i64; n]`.** Any time-dependent indicator is wrong or degenerate. | ⬜ OPEN |
| MH11 | `forex-data/src/lib.rs`, `forex-data/src/core/features.rs` | **FeatureFrame construction lacks central non-finite sanitization.** NaN/Inf values from HTF alignment gaps or indicator warmup enter CPU and GPU signal synthesis. CPU and GPU handle NaNs differently. | ⬜ OPEN |
| MH12 | `eval.rs`, `cubecl_eval.rs` | **Final open position handling undefined.** CPU and CUDA may handle end-of-test open positions differently (force-close vs mark-to-market vs ignore). Net profit, trade count, daily/monthly ledger diverge. | ⬜ OPEN |
| MH13 | `forex-app/src/app_services/discovery.rs` | **UI discovery profile records validation config but not validation execution results.** `walkforward_splits`, `embargo_minutes`, etc. in profile — but not `oos_tested`, `oos_passed`, `walkforward_passed`, `cpcv_passed`. | ⬜ OPEN |
| MH14 | `forex-app/src/app_services/discovery.rs` | **UI "best strategy" selected by `gene.fitness`, not OOS/quality/risk-adjusted contract.** Reinforces the fitness ≠ net_profit problem at the user-facing layer. | ⬜ OPEN |
| MH15 | `validation.rs` | **WFV evaluates precomputed fixed signals per split — not full retrain-per-split WFO.** Useful as fixed-strategy forward validation but not equivalent to walk-forward optimization. The distinction is not documented. | ⬜ OPEN |
| MH16 | `discovery.rs`, `eval.rs` | **Stage-1 search window may be narrower than expected.** If UI already cut data to 80%, stage-1 fast eval runs on only the last portion of that subset. Stronger overfitting to recent conditions. | ⬜ OPEN |
| MH17 | `eval.rs`, `cubecl_eval.rs`, `quality.rs` | **No canonical day/week/month period ledger.** Pieces exist (daily DD, monthly PnL, monthly consistency, daily Monte Carlo blocks) but no unified ledger. Trader-grade period reporting requires daily PnL/DD, weekly rhythm, monthly consistency, payout-cycle behavior, recovery after losing periods. | ⬜ OPEN |
| MH18 | `forex-app/src/app_services/ctrader_execution.rs` | **`execute_with_transport` uses arithmetic `PAYLOAD_TYPE - 2` constant instead of named constant.** Arithmetically correct today, but a latent trap if constants are reordered. | ✅ FIXED |
| MH19 | `forex-core/src/domain/order_execution.rs` vs `eval.rs` | **Live execution has partial TP, multiple R-level legs, and entry patience/pullback not simulated in backtest.** Backtested equity curve ≠ live equity curve. | ⬜ OPEN |

---

## MEDIUM — Cleanup and polish

| ID | File(s) | Issue | Status |
|---|---|---|---|
| L1 | `forex-search/src/hpc_simd.rs` (~340 LoC) | **Wholly dead code, no callers.** Rayon par_iter already saturates cores. Delete. | ⬜ OPEN |
| L2 | `eval.rs:322-335` | **Trailing stop never activates on shorts** unless `trailing_be_trigger_r` is met. Document the constraint. | ⬜ OPEN |
| L3 | `tree_models/lightgbm.rs:589-594` | **Temperature calibration followed by re-normalize is redundant.** Simplify. | ⬜ OPEN |
| L4 | `embedded_credentials.rs` + `build.rs:202-208` | **`println!` of embedded credential length leaks into CI logs.** Suppress. | ⬜ OPEN |
| L5 | `forex-data/src/core/regime_detection.rs:35-78` | **Garman-Klass volatility recomputed O(n × window) per bar.** Rolling sum gives O(n). Low priority: features computed once and cached. | ⬜ OPEN |
| L6 | `forex-models/src/lib.rs` | **Comments still reference `models/base.py`, `models/device.py`** (Python migration breadcrumbs). Update to describe current Rust architecture. | ✅ FIXED |
| L7 | `cache/audit/2026-03-20-file-manifest.txt` | **Stale manifest.** References removed Python/PyO3-era files that do not exist in master. | ⬜ Archive or delete |
| L8 | `forex-models` GPU/evolution env flags | **Model/training behavior controlled by env flags** (`FOREX_BURN_MODEL_SUPPORTS_BF16`, model-scoped precision keys, etc.). Move into typed model config. | ⬜ OPEN |
| L9 | `ctrader_streaming.rs` | **Streaming session uses global singleton without account keying.** Acceptable for single-account use; needs keying by account id for multi-account. | ⬜ OPEN — future milestone |
| L10 | `ctrader_session.rs`, `ctrader_proto_messages.rs` | **Both have `#![allow(dead_code)]`.** Full-duplex async session is not wired to any live code path. Safe to leave until persistent-streaming milestone. | ⬜ OPEN — deferred |

---

## Environment flags to migrate (non-infrastructure)

The following `FOREX_BOT_*` / `std::env::var` reads in active strategy/search/backtest/live code violate reproducibility. Each must be migrated to typed config and exported in the run contract.

**Critical first pass** (search/discovery/eval):
- `crates/forex-search/src/discovery.rs` — prefilter top-k, prefilter in-sample fraction, stage-1 funnel pct
- `crates/forex-search/src/genetic/search_engine.rs` — search seed, archive mode/thresholds, survivor/immigrant/selection policy
- `crates/forex-search/src/genetic/smc_indicators.rs` — SMC gate schedule, force ratio, probabilities, min flags
- `crates/forex-search/src/genetic/strategy_gene.rs` — SMC force ratio
- `crates/forex-search/src/lib.rs` — discovery/search seed
- `crates/forex-search/src/discovery_gpu.rs` — GPU eval seed/backend
- `crates/forex-search/src/cubecl_eval.rs` — CUDA eval knobs
- `crates/forex-search/src/cubecl_ga.rs` — novelty weight
- `crates/forex-search/src/quality.rs` — backtest initial equity

**Second pass** (models/app):
- `crates/forex-models/src/genetic.rs`
- `crates/forex-models/src/base.rs`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/evolution/*_gpu.rs`
- `crates/forex-models/src/tree_models/*`
- `crates/forex-app/src/app_services/trading.rs`
- `crates/forex-app/src/app_services/ctrader_execution.rs`

**Allowed env vars** (infrastructure only): build metadata, CI flags, logging level/path, secrets/credentials, emergency GPU-required fail-fast.

---

## Recommended implementation order

### Sprint 1 — Correctness and safety (unblock prop-firm deployment)
1. **C6** Add `cargo check/test -p forex-search` and `-p forex-models` to CI. *(in progress)*
2. **H7** `ensure_authenticated`: detect `INVALID_TOKEN`, call `refresh_token_bundle`, exponential back-off ≤3 attempts.
3. **C8** Wire `RiskManager` from `forex-core/src/domain/risk.rs` into `trading.rs` trading loop; remove weak inline `prop_firm_pre_trade_check`.
4. **H16** Regime robustness: replace hardcoded `-3000.0` with `initial_balance × max_regime_loss_pct` from config.
5. **H17** Replace `WALKFORWARD_INITIAL_BALANCE = 100_000.0` with caller-supplied value from config.
6. **H18** `WalkforwardBacktestInput`: carry real `timestamps`; pass to `simulate_trades_core`.
7. **H19** Close all open periods at backtest end (final month / final day finalization).
8. **C5 / H28** Add `effective_feature_names: Vec<String>` and `feature_mapping: Vec<usize>` to `DiscoveryResult`. Change all callers of `save_portfolio_json` to use them.
9. **H26** Batch discovery: read configured `higher_tfs` from `DiscoveryConfig` instead of `&[]`.

### Sprint 2 — Search contract and parity
10. **C3** Fix CUDA full backtest entry: `signals_flat[signal_base + i - 1]` (prior bar).
11. **C1** HTF resampling: timestamps must represent bar close, or shift HTF features by one full HTF bar after alignment.
12. **C2** Normalize all timestamps to milliseconds at the `FeatureFrame` boundary before entering `forex-search`. Add typed `TimestampMs` newtype + tests.
13. **C4** Add kill-zone / session / day-boundary logic to CUDA kernel or disable CUDA as final-acceptance path until implemented.
14. **H29** Separate `net_profit`, `search_score`, `quality_score`, `oos_score`, `portfolio_score` fields in gene/candidate structs.
15. **H30** Create `SearchRunContract` / `PortfolioContract`: feature schema hash, seed, entrypoint, git commit, evaluator backend, split ranges, env overrides, validation results.
16. **H25** Replace signal-count trade proxy in discovery filter with a cheap canonical trade simulation.
17. **H33** One canonical `SearchBacktestContract` from config/risk/symbol metadata, passed to CPU GA, CUDA evaluator, quality, gauntlet, WFV/CPCV, and trade logs.
18. **H37** Causal SL/TP inference: compute per train split using only train data, freeze for test/OOS.
19. **H27** After discovery on 80% in-sample, replay selected candidates on withheld 20% OOS before portfolio save. Gate export on OOS pass.
20. **H39** Make WFV/CPCV hard gates in discovery: gate portfolio export on `walkforward_passed`, `cpcv_passed`.

### Sprint 3 — Model training quality
21. **H40** Record original train/val split in `default_training_summary` after HPO full-data refit.
22. **H41** Floor `embargo_rows` at `max(label_horizon_bars × 2, 20)`.
23. **H42** Forward `val_frame/val_labels` from HPO into Burn early-stopping.
24. **H43** Plumb val frame as `eval_set` for LightGBM/XGBoost/CatBoost; enable early stopping.
25. **H44** Compute anomaly threshold on validation slice, not training.

### Sprint 4 — Env flag migration and config hardening
26. **C7** Migrate all search/discovery/eval env flags into `DiscoveryConfig` / `SearchConfig` / `SmcSearchConfig` / `BacktestConfig`. Export effective values in run contract.
27. **MH1** Add serde bound validators to `RiskConfig` and `DiscoveryConfig` dangerous fields.
28. **MH3** Cache symbol metadata (pip_size, contract_size, min_stop_distance) with TTL.
29. **L8** Move model training env flags into typed model config.

### Sprint 5 — Data quality and feature hygiene
30. **MH10** Pass real timestamps into VectorTA `Candles` object.
31. **MH11** Feature sanitation at `FeatureFrame` boundary: fill NaN/Inf with documented neutral value or drop warmup rows. Apply before CPU and GPU evaluation.
32. **MH7** Timeframe-dependent quant features: receive timeframe metadata and convert calendar periods to bar counts.
33. **MH8** Remove `base_tf` from `higher_tfs` before alignment.
34. **MH9** Make duplicate timestamp handling explicit (reject/keep-last-log/aggregate).

### Sprint 6 — Simulation completeness and period ledger
35. **MH4** Share one position state machine between `fast_evaluate_strategy_core` and `simulate_trades_core` (session rules, Friday exit, Monday block).
36. **MH5** Replace raw `[f64; 11]` metric arrays with typed `BacktestMetrics` struct.
37. **MH12** Define end-of-test policy: force-close at final close or mark-to-market. Apply uniformly in CPU, CUDA, WFV/CPCV, quality, trade logs.
38. **MH17** Canonical day/week/month period ledger emitted by reference backtest and mirrored by GPU parity evaluator.
39. **H35** Add position-sizing mode to canonical backtest contract: fixed-lot, fixed-risk-pct, volatility-targeted, or live-risk-manager parity.
40. **H36** Model live risk gates in backtest engine or export `live_gate_adjusted_backtest`.

### Sprint 7 — Cleanups
41. **L1** Delete `hpc_simd.rs` (dead code, ~340 LoC).
42. **L2-L5** Minor cleanups: trailing stop comment, LightGBM calibration, CI credential log, O(n²) Garman-Klass.
43. **L7** Archive or delete `cache/audit/2026-03-20-file-manifest.txt`.

---

## CI guardrail additions (alongside Sprint 1)

```yaml
# Add to .github/workflows/ci.yml
- name: Check forex-search
  run: cargo check -p forex-search

- name: Test forex-search
  run: cargo test -p forex-search

- name: Check forex-models (no-default-features)
  run: cargo check -p forex-models --no-default-features

- name: Test forex-models (no-default-features)
  run: cargo test -p forex-models --no-default-features
```

Add a `grep`-based CI step that fails on new `std::env::var` calls in `forex-search`, `forex-models`, and `forex-app/app_services` unless explicitly allowlisted.

---

## What was already fixed (prior PR commits)

The following issues from earlier audit passes were addressed in `claude/fix-strategy-search-cpu-gpu-rtkhl` and merged or in-flight:

- CPU/GPU parity: `GpuDiscoveryConfig::seed`, causal shift+zscore preprocessing, segment off-by-one, parallelized refinement loops.
- GA determinism: `&mut StdRng` threaded through `evolution_math.rs::{new_random_gene, crossover, mutate}`.
- Archive dedup: switched to `gene_signature_hash`.
- SMC-gated signal path: `signals_for_gene_full` used in post-filter, MC perturbation, gauntlet.
- Prefilter forward-return snooping: in-sample 70% only.
- Prop-firm equity: unrealized PnL added to `ctrader_account_equity`.
- Daily reset: `handle_day_boundary`, `handle_phase_rollover` wired.
- Hard risk gate: `risk_per_trade` violation returns `Err` instead of warning.
- PartialFill: `validate_execution_outcome` returns error on partial fills.
- Socket timeout: 30s `tokio::time::timeout` on `read_matching_response`.
- NaN-safe metrics: `mean_std` NaN filter + Bessel in `cubecl_eval.rs`.
- GPU profit-factor cap at 10.0.
- Cross-pair pip value: `estimate_pip_value_per_lot` takes `quote_to_account_rate`.
- Calmar saturation: returns 1000.0 for zero drawdown with positive return.
- Intra-bar lookahead: `eval.rs` entry shifted to `signals[i-1]`.
- Monthly consistency: `min_trades_per_month = 4` guard.
- Gauntlet warn-only: logs failed metric name before returning true.
- Batch orchestration: single discovery failure no longer aborts whole batch.
- Python residue: deleted `mt5-bridge`, `forex-bindings`, `onnx_exporter.rs`, pyo3 dependency.
- `broker_timezone` field added to core config.
- cTrader constant arithmetic: replaced `PAYLOAD_TYPE - 2` with named constant.
