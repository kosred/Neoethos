# Forex-AI Consolidated Execution Plan

**Created:** 2026-05-04 Europe/Berlin
**Repository:** kosred/forex-ai
**Branch:** `claude/fix-strategy-search-cpu-gpu-rtkhl`
**Source audits (2026-05-03 + 2026-05-04 only — older audits intentionally excluded):**

```
docs/audits/architecture_unification_duplicate_code_cleanup_audit_2026-05-03.md
docs/audits/artifact_intent_clarification_training_vs_search_resume_2026-05-03.md
docs/audits/core_config_domain_modularization_audit_2026-05-04.md
docs/audits/cpu_gpu_semantic_parity_requirement_2026-05-03.md
docs/audits/custom_cuda_kernel_preservation_audit_2026-05-03.md
docs/audits/dead_code_and_stale_artifacts_audit_2026-05-03.md
docs/audits/deep_duplicate_logic_and_unified_scheduler_audit_2026-05-03.md
docs/audits/deep_search_engine_state_audit_pass2_2026-05-03.md
docs/audits/evaluation_contract_deep_audit_pass3_2026-05-03.md
docs/audits/evolution_neat_crfmnes_gpu_first_audit_2026-05-03.md
docs/audits/feature_timestamp_mtf_causality_deep_audit_pass5_2026-05-03.md
docs/audits/forex_data_functional_audit_2026-05-04.md
docs/audits/forex_search_functional_audit_2026-05-04.md
docs/audits/generic_scheduler_small_files_refactor_note_2026-05-04.md
docs/audits/gpu_cuda_hpc_parity_deep_audit_pass4_2026-05-03.md
docs/audits/gpu_first_kernel_everywhere_report_2026-05-03.md
docs/audits/hardware_autodetect_config_ui_architecture_2026-05-03.md
docs/audits/model_runtime_backend_fragmentation_audit_2026-05-03.md
docs/audits/modularization_maintainability_refactor_principle_2026-05-04.md
docs/audits/python_pyo3_legacy_audit_2026-05-03.md
docs/audits/quality_challenge_validation_refactor_audit_2026-05-04.md
docs/audits/rust_env_flags_config_debt_audit_2026-05-03.md
docs/audits/search_backtest_forward_cpu_gpu_audit_2026-05-03_14-46.md
docs/audits/search_checkpoint_resume_requirement_2026-05-03.md
docs/audits/search_discovery_pipeline_audit_2026-05-03.md
docs/audits/search_gpu_discovery_scheduler_audit_2026-05-03.md
docs/audits/search_orchestration_refactor_audit_2026-05-03.md
docs/audits/search_portfolio_artifact_contract_audit_2026-05-03.md
docs/audits/search_to_live_bridge_audit_2026-05-03.md
docs/audits/training_model_artifact_contract_audit_2026-05-03.md
docs/audits/unified_module_logic_architecture_2026-05-04.md
docs/audits/universal_hardware_parity_requirement_2026-05-03.md
```

This plan **only contains open items**. Every issue already fixed in the
current branch (PR #3) has been removed; the closed register is at the bottom.

---

## CRITICAL — show-stoppers for prop-firm trading

These must land before any live deployment. They cause silent rank
inversions, label leakage, or risk-rule violations.

| ID | File(s) | Issue | Fix |
|---|---|---|---|
| **C1** | `forex-data/src/core/resample.rs`, `features.rs` | **HTF resampling lookahead.** Resampled H1/H4/D1 candles are timestamped at bucket-start but carry OHLC from the full closed bucket. `align_features_by_ns` then makes lower-TF rows see completed HTF values before the HTF bar actually closes. | Introduce `CandleTimestampPolicy::{OpenTime, CloseTime}` and a separate `available_at_ms`. HTF features become available only after the HTF candle closes. |
| **C2** | `resample.rs`, `search_engine.rs`, `eval.rs`, `validation.rs`, `quality.rs`, `cubecl_eval.rs` | **Timestamp unit mismatch.** Data/resampling uses ns (`1_000_000_000`); search/eval/quality/CUDA uses ms (`86_400_000`, `3_600_000`, `timestamp_millis_opt`). Day/month bucketing, gap exits, kill zones, and Monte Carlo blocks silently break under ns timestamps. | Normalize to `TimestampMs` newtype at the `FeatureFrame` boundary. Carry `TimestampUnit` in `Ohlcv` metadata. |
| **C4** | `cubecl_eval.rs`, `eval.rs` | **CUDA session contract missing.** CUDA kernel models max-hold, spread, commission, and pip-value but not kill zones, Friday force-exit, Monday/Friday entry blocks, broker-tz day boundaries, or session windows. CPU `simulate_trades_core` does. | Either wire session/kill-zone logic into the CUDA kernel or downgrade CUDA to approximate-presearch only and require canonical CPU validation before portfolio acceptance. |
| **C5** | `discovery.rs`, `forex-app/src/app_services/discovery.rs`, `forex-cli` | **Feature-index ↔ feature-name mismatch after prefilter.** `prefilter_features` replaces the matrix and name list inside discovery. Discovered gene indices are relative to the filtered matrix; callers (`save_portfolio_json`) supply the original `features.names`. Exported indicator names can point to the wrong columns. | Add `effective_feature_names: Vec<String>` and `feature_mapping: Vec<usize>` to `DiscoveryResult`. Change every `save_portfolio_json` caller to use them. |
| **C6** | `.github/workflows/ci.yml` | **CI does not cover `forex-search` or `forex-models`.** Search/backtest/model regressions land without CI catching them. (In progress.) | Add `cargo check`/`cargo test` jobs for `-p forex-search` and `-p forex-models --no-default-features`. |
| **C7** | `discovery.rs`, `genetic/search_engine.rs`, `cubecl_eval.rs`, etc. | **Search/discovery semantics controlled by env flags.** `FOREX_BOT_PREFILTER_TOP_K`, `FOREX_BOT_PREFILTER_INSAMPLE`, `FOREX_BOT_FUNNEL_STAGE1_PCT`, `FOREX_BOT_SEARCH_SEED`, archive mode/thresholds, survivor policy, novelty weight, backtest initial equity — two runs with identical config + data produce different portfolios when env differs. | Migrate to typed config (`DiscoveryConfig`, `SearchConfig`, `SmcSearchConfig`, `BacktestConfig`). Export effective values in the run contract. Allowlist only build/CI/secret env vars. |
| **C8** | `forex-core/src/domain/risk.rs` | **`RiskManager` is dead code — not wired.** ~700 LoC prop-firm risk guard has zero callers. `prop_firm_pre_trade_check` in `trading.rs` uses a weaker inline replacement. | Wire `RiskManager` into `trading.rs` trading loop; delete the inline replacement. |

---

## HIGH — significant correctness and safety

### Prop-firm execution safety

| ID | File:line | Issue | Fix |
|---|---|---|---|
| **H2'** | `trading.rs:1727-1735` | Hard-coded pip-position lookup is FX-only. Crypto/indices/non-gold metals still wrong. | Cache full `ProtoOASymbol` metadata (`digits`/`pipPosition`) and use it for every symbol. |
| **H3'** | `trading.rs:3317` | Estimated loss now better, but still not full cross-pair quote→account FX rate. | Multiply `pip_value_per_lot` × `quote_currency → account_currency` rate from cached metadata. |
| **MH3** | `ctrader_data.rs` | Symbol metadata (`pip_size`, `contract_size`, `min_stop_distance`) fetched on every order. | `Cache<symbol_id, SymbolInfo>` with 1-hour TTL refreshed at first trade of day. Single fix that unblocks H2', H3'. |

### Search determinism & ranking

| ID | File:line | Issue | Fix |
|---|---|---|---|
| **H16** | `discovery.rs` | Regime-robustness gate uses hardcoded `-3000.0` absolute loss. On a 10k account this is 30% — dangerously loose. | Replace with `initial_balance × max_regime_loss_pct` from config. |
| **H17** | `validation.rs` | `WALKFORWARD_INITIAL_BALANCE = 100_000.0` hardcoded. All WFV daily/monthly metrics and prop checks ignore the configured balance. | Pass `initial_balance` from config through `WalkforwardBacktestInput`. |
| **H18** | `validation.rs`, `eval.rs` | Walk-forward diagnostics pass `days` as timestamps to `simulate_trades_core`. Duration/gap/kill-zone/max-trades-per-day computations are wrong inside WFV. | Carry real `timestamps` in `WalkforwardBacktestInput`; pass to `simulate_trades_core`. |
| **H19** | `eval.rs`, `cubecl_eval.rs` | Final open month/day not safely closed before metric computation. Last open month can be omitted from Sharpe and consistency. | Finalize all open periods at end of backtest in CPU and CUDA paths. |
| **H25** | `discovery.rs`, `eval.rs`, `gauntlet.rs` | Signal count used as proxy for trade count. Non-zero signal count ≠ executed trades (depends on SL/TP, max-hold, position state, gaps, kill zones). | Replace with cheap canonical trade simulation in the discovery filter. |
| **H26** | `forex-app/src/app_services/discovery.rs`, `orchestration.rs` | Batch discovery passes `&[]` for higher TFs. Same symbol/config produces different feature universe per entrypoint. | Read configured `higher_tfs` from `DiscoveryConfig`. |
| **H27** | `forex-app/src/app_services/discovery.rs` | UI withholds 20% OOS from search but does not replay/export OOS validation before portfolio save. The 20% is unused. | After discovery, replay selected candidates on the withheld 20% OOS. Gate export on OOS pass. |
| **H28** | `discovery.rs` | `DiscoveryResult` does not carry `effective_feature_names` or `feature_mapping`. Callers save portfolio with original pre-prefilter names. | Same fix as C5 — single change resolves both. |
| **H29** | `discovery.rs`, `search_engine.rs` | `fitness` field used with different meanings across paths (GA composite, candidate ranking, UI/export imply income). | Separate `search_score`, `net_profit`, `quality_score`, `oos_score`, `portfolio_score` in gene/candidate structs. |
| **H30** | Various | No `SearchRunContract` / `PortfolioContract`. Exported portfolio does not prove which data was searched, which features selected, which seed used, which evaluator accepted it, which gates ran. | Define typed contracts + provenance hash + git commit + env overrides + validation results. Mandatory in every artifact. |
| **H31** | `discovery.rs` and friends | Not fully deterministic. No single run-seed propagated to all paths (`discovery_gpu.rs`, `hpc_gpu_discovery.rs`, `cubecl_ga.rs`, `quality.rs`, `forex-models/genetic.rs`). | Single `RunSeed` threaded everywhere. Test: same seed + same config + same data ⇒ identical portfolio. |
| **H32** | `forex-app/src/app_services/discovery.rs`, `orchestration.rs` | UI/batch discovery slice data differently. UI cuts to 80% in-sample; batch passes the full frame. Same config produces different candidates. | One canonical `SearchInputSliceContract` consumed by UI and batch. |
| **H33** | `discovery.rs`, `search_engine.rs` | Search/quality/gauntlet can use different cost/session contracts. Defaults leave kill zones, min-hold, max-trades/day, gap-threshold unset in some paths. | One canonical `SearchBacktestContract` from config/risk/symbol metadata, passed to CPU GA, CUDA evaluator, quality, gauntlet, WFV/CPCV, trade logs. |
| **H34** | `discovery_gpu.rs`, `hpc_gpu_discovery.rs` | Tensor/HPC GPU discovery uses simplified return-based fitness (no SL/TP/spread/commission). | Tag artifacts as `ApproxSearchCandidateBatch`. Require canonical CPU validation before becoming `ValidatedStrategyGene`. |
| **H35** | `eval.rs`, `cubecl_eval.rs` | Backtest uses fixed pip-value/lot-size, not live dynamic risk sizing (RiskManager has equity-based, vol-targeted, Kelly-scaled sizing). | Add position-sizing mode to canonical backtest contract: `FixedLot`, `FixedRiskPct`, `VolTargeted`, `LiveRiskManagerParity`. |
| **H36** | `eval.rs`, `cubecl_eval.rs` | Live risk gates not modelled in backtest (circuit breaker, recovery mode, daily/intraday DD blocks, revenge-trade detection, confidence threshold). | Either model in backtest or export `live_gate_adjusted_backtest` so the operator sees the live-conditioned curve. |
| **H37** | `stop_target.rs`, `search_engine.rs` | Stop/target inference can use future data. `resolve_stop_target_arrays` calls `infer_stop_target_pips` on the full OHLCV slice. | Compute SL/TP defaults per train split using only train data; freeze for test/OOS. |
| **H38** | `eval.rs`, `cubecl_eval.rs` | Slippage not first-class. `RiskConfig` has `slippage_pips`/`slippage_guard_multiplier`; CPU/CUDA backtest ignores them. | Add `slippage_pips` to `BacktestSettings`; apply at fill time. Edge often disappears under realistic M1/M5 slippage. |
| **H39** | `validation.rs` | WFV/CPCV not a hard discovery gate. `embargoed_walkforward_backtest` / `CombinatorialPurgedCV` exist but are not mandatory in candidate acceptance. | Make `walkforward_passed` and `cpcv_passed` mandatory in `DiscoveryRunProfile`; gate portfolio export on them. |

### Model training quality

| ID | File:line | Issue | Fix |
|---|---|---|---|
| **H40** | `training_orchestrator.rs:3043-3235` | Post-HPO full-data refit records `val_rows=0` in `default_training_summary`, hiding the original split. | Either record the original train/val split alongside refit metadata, or skip the full-data refit and keep the HPO-trained model. |
| **H41** | `runtime/hpo.rs` | `embargo_rows` can be 0 if `embargo_minutes==0`. Train→val label leakage possible. | Floor at `max(label_horizon_bars × 2, 20)`. |
| **H44** | `anomaly/forest_impl.rs` | Anomaly threshold computed from training scores (optimistic). | Compute on the same val slice HPO uses, via `fit_with_validation`. |

---

## MEDIUM — correctness drift, ranking quality

| ID | File(s) | Issue | Fix |
|---|---|---|---|
| **MH1** | `forex-core/src/config.rs:225-341` | Risk config fields lack serde bound validation. `risk_per_trade: 50` (meaning 50%) silently passes. | Custom `Deserialize` validators clamping to safe ranges with `tracing::error!` on out-of-range values. |
| **MH4** | `eval.rs` | `fast_evaluate_strategy_core` lacks Friday force-exit and Monday/Friday entry blocking that `simulate_trades_core` has. Fast metrics and detailed trade logs disagree. | Share one position state machine between the two. |
| **MH5** | `eval.rs`, `cubecl_eval.rs`, etc. | Raw metric layout `[f64; 11]` with implicit indices. CUDA has narrower internal width and reconstructs the full array. Index mismatch silently corrupts selection/validation/UI/export. | Replace with typed `BacktestMetrics` struct. |
| **MH6** | `gauntlet.rs` | Gauntlet uses its own settings and the simplified `signals_for_gene`. Not the canonical search/backtest contract. | Consume `SearchBacktestContract` (H33). |
| **MH7** | `forex-data/src/core/quant_features.rs` | "Daily/weekly" quant features use fixed bar counts (`24`, `120`) regardless of timeframe. On M1, `24` bars = 24 minutes; on H1, ≈ 1 day. Misleads the GA on non-H1 timeframes. | Carry `TimeframeContext { bar_duration_ms, bars_per_day, bars_per_week, session_calendar }`. Convert calendar periods to bar counts at runtime. |
| **MH8** | `forex-core/src/config.rs`, `forex-data/src/lib.rs` | `higher_tfs` can include `base_tf`. `prepare_multitimeframe_features_with_options` then duplicates the base TF, overweighting it. | Strip `base_tf` from `higher_tfs` before alignment. Reject in config validation too. |
| **MH9** | `forex-data/src/lib.rs` | Duplicate timestamp rows silently dropped (`dedup_by`). If duplicates come from broker corrections or partial candles, the wrong row may be kept. | Make duplicate handling explicit: reject, keep-last-with-log, or aggregate per OHLCV rules. |
| **MH10** | `forex-data/src/core/hpc_ta.rs` | `compute_classic_ta_columns` builds VectorTA `Candles` with `timestamps = vec![0i64; n]`. Any time-dependent indicator is wrong or degenerate. | Pass real timestamps with explicit `TimestampUnit`. Mark indicator registry entries that need calendar awareness. |
| **MH11** | `forex-data/src/lib.rs`, `core/features.rs` | FeatureFrame construction lacks central non-finite sanitization. NaN/Inf from HTF alignment gaps or indicator warmup enter CPU and GPU signal synthesis with different semantics. | Sanitize NaN/Inf at the `FeatureFrame` boundary (drop warmup rows or fill with documented neutral). Apply identically to CPU and GPU. |
| **MH12** | `eval.rs`, `cubecl_eval.rs` | Final open position handling undefined. CPU and CUDA may handle end-of-test open positions differently (force-close vs mark-to-market vs ignore). Net profit / trade count / monthly ledger diverge. | Single end-of-test policy applied uniformly across CPU, CUDA, WFV/CPCV, quality, trade logs. |
| **MH13** | `forex-app/src/app_services/discovery.rs` | UI discovery profile records validation config but not validation execution results. | Add `oos_tested`, `oos_passed`, `walkforward_passed`, `cpcv_passed` to `DiscoveryRunProfile`. |
| **MH14** | `forex-app/src/app_services/discovery.rs` | UI "best strategy" selected by `gene.fitness`, not OOS/quality/risk-adjusted contract. | Use the `portfolio_score` from H29, gated by H39 results. |
| **MH15** | `validation.rs` | WFV evaluates precomputed fixed signals per split — not full retrain-per-split WFO. Distinction is not documented. | Add a clear `WalkForwardKind::{FixedSignal, RetrainPerSplit}` enum and document. |
| **MH16** | `discovery.rs`, `eval.rs` | Stage-1 search window may be narrower than expected. If UI cut data to 80%, stage-1 fast eval runs only on the last portion of that subset. | Compute stage-1 window from full dataset percent, not pre-sliced subset. |
| **MH17** | `eval.rs`, `cubecl_eval.rs`, `quality.rs` | No canonical day/week/month period ledger. Pieces exist (daily DD, monthly PnL, monthly consistency, MC blocks) but no unified ledger. | Single `PeriodLedger { daily, weekly, monthly }` emitted by reference backtest and mirrored by CUDA evaluator. |
| **MH19** | `forex-core/src/domain/order_execution.rs` vs `eval.rs` | Live execution has partial TP, multi-R legs, and entry patience/pullback that backtest does not simulate. Backtested equity curve ≠ live. | Either model in backtest or scope-document the divergence per artifact. |

---

## LOW — cleanup, docs, dead code

| ID | File(s) | Issue | Fix |
|---|---|---|---|
| **L1** | `forex-search/src/hpc_simd.rs` | ~340 LoC of dead code, no callers. Rayon par_iter already saturates cores. | Delete. |
| **L2** | `eval.rs:322-335` | Trailing stop never activates on shorts unless `trailing_be_trigger_r` is met. | Document the constraint in a one-line comment. |
| **L3** | `tree_models/lightgbm.rs:589-594` | Temperature calibration followed by re-normalize is redundant. | Simplify. |
| **L4** | `embedded_credentials.rs`, `build.rs:202-208` | `println!` of embedded credential length leaks into CI logs. | Suppress in release builds. |
| **L5** | `forex-data/src/core/regime_detection.rs:35-78` | Garman-Klass volatility recomputed O(n × window) per bar. Rolling sum gives O(n). Low priority because features are computed once and cached. | Convert to rolling sum. |
| **L7** | `cache/audit/2026-03-20-file-manifest.txt` | Stale manifest references removed Python/PyO3-era files. | Archive or delete. |
| **L8** | `forex-models` GPU/evolution env flags | Model/training behavior controlled by env (`FOREX_BURN_MODEL_SUPPORTS_BF16`, model-scoped precision keys). | Move into typed model config. |
| **L9** | `ctrader_streaming.rs` | Streaming session uses global singleton without account keying. | Defer until multi-account milestone; acceptable for single-account. |
| **L10** | `ctrader_session.rs`, `ctrader_proto_messages.rs` | Both have `#![allow(dead_code)]`. Full-duplex async session not wired. | Defer until persistent-streaming milestone. |

---

## NEW (2026-05-04 audits) — architecture & modularization

These are higher-effort than the C/H/MH items above but represent the
target end-state. They don't block prop-firm deployment if the C/H items
are landed; they make the codebase maintainable for the next 12 months.

| ID | Concern | Direction |
|---|---|---|
| **A1** | Files too large, multiple lifecycle stages mixed in one type. | Split: `forex-search/src/{search,genetic,signal,evaluation,kernels,presearch,validation,risk}/`. Smallest possible focused modules. |
| **A2** | `Gene` mixes candidate, evaluated, validated, and portfolio responsibilities. | Split into `CandidateGene`, `EvaluatedGene`, `ValidatedStrategyGene`, `PortfolioStrategy`. |
| **A3** | Hyperstack-N3-specific scheduler in `hpc.rs` and `hpc_gpu_discovery.rs`. | Replace with generic `runtime/{hardware_profile,hardware_probe,topology,device_assignment,scheduler,work_unit}.rs`. Hyperstack becomes one detected profile. |
| **A4** | `quality.rs`, `challenge.rs`, regime gates spread over multiple crates. | Unify under `validation/{trade_ledger,metrics,quality_policy,monte_carlo,challenge_policy,validation_report}.rs`. |
| **A5** | Feature generation exists in multiple modules and is recreated by GPU paths. | One owner: `forex-data/features/*`. CPU and GPU pipelines consume the same `FeatureFrame` contract. |
| **A6** | SMC features partly in `forex-data/core/smc.rs` and partly in `forex-search/genetic/smc_indicators.rs`. | One owner: `forex-data/features/smc.rs`. Search consumes it via typed `SmcSearchPolicy`. |
| **A7** | No stable VectorTA registry — `ALL_INDICATORS` is a static list. | Generate from VectorTA dispatch metadata. Validate at build-time that every listed indicator exists. |
| **A8** | Custom CUDA kernels (in `cubecl_eval.rs`, `cubecl_ga.rs`, `crfmnes_gpu.rs`, `neat_gpu.rs`, `linear_gpu.rs`) are scattered with their own device/precision/env code. | Keep kernels. Move device/precision/env handling into the generic scheduler from A3. |
| **A9** | Model training reads `FOREX_BURN_MODEL_SUPPORTS_BF16` etc. directly. | Same migration as C7 but for `forex-models`. |
| **A10** | UI/install-time hardware autodetection idea recorded. | Build `runtime/hardware_probe.rs` → write a generated default config → UI exposes it as editable settings. Replaces most env vars for the installable distribution. |

---

## Search checkpoint / resume (2026-05-03 audit)

| ID | Concern | Direction |
|---|---|---|
| **S1** | Strategy search starts from zero every run. Long-running discovery wastes compute and re-discovers the same early-stage candidates. | Add `SearchCheckpoint` covering RNG state, archive, signature memory, generation counter, GA population, hall-of-fame, validation cache. Persist atomically. |
| **S2** | `SeenSignatureMemory` stores only signatures, not gene bodies / metrics / regimes / OOS results. | Extend to a `SearchMemoryStore { signatures, gene_bodies, eval_metrics, validation_history, regime_tags }`. |
| **S3** | Training profiles persist; discovery profiles do not yet support resume. | Same atomic-write pattern as model artifacts. |

---

## Search-to-live bridge (2026-05-03 audit)

| ID | Concern | Direction |
|---|---|---|
| **B1** | Discovery portfolio JSON is not loadable as a runtime artifact. No `load_portfolio_json` / importer. | Build a strict `PortfolioArtifact` loader with schema validation and a round-trip test. |
| **B2** | `trading.rs` has no path from a discovered portfolio into live signal generation. | Add `signal_engine` consuming `PortfolioArtifact` and feeding the existing risk/exec gate. |
| **B3** | Model artifacts and search artifacts have different provenance schemas. | Align both under one `RuntimeArtifactMetadata` envelope (already exists for models — extend to search). |

---

## Recommended sprint order

### Sprint 1 — unblock prop-firm trading (P0)

1. **C8** Wire `RiskManager` into `trading.rs`; remove inline `prop_firm_pre_trade_check`.
2. **MH3 + H2' + H3'** Single symbol-metadata cache (TTL 1h) → fixes H2' pip lookup and H3' loss estimate.
3. **C6** Add `cargo check`/`cargo test` for `-p forex-search` and `-p forex-models --no-default-features` to CI.
4. **H16** Replace hardcoded `-3000.0` regime-loss gate with `initial_balance × max_regime_loss_pct`.
5. **H17 + H18** WFV uses configured `initial_balance` and real `timestamps`.
6. **H19 + MH12** Close all open periods at backtest end; single end-of-test policy in CPU/CUDA/WFV.
7. **C5 + H28** Add `effective_feature_names` and `feature_mapping` to `DiscoveryResult`; update every `save_portfolio_json` caller.
8. **H26** Batch discovery reads `higher_tfs` from `DiscoveryConfig`.

### Sprint 2 — search contract & CPU/GPU parity

9. **C1** HTF resampling: `CandleTimestampPolicy` + `available_at_ms`. Fixes the bucket-start lookahead.
10. **C2** `TimestampMs` newtype at `FeatureFrame` boundary; `TimestampUnit` in `Ohlcv`.
11. **C4** Either wire session/kill-zone/day-boundary logic into the CUDA kernel or downgrade CUDA to approximate-presearch with mandatory CPU validation. Tag artifacts (H34).
12. **H29 + H30** Split `fitness` into `search_score`, `net_profit`, `quality_score`, `oos_score`, `portfolio_score`. Define `SearchRunContract` / `PortfolioContract`.
13. **H25** Replace signal-count proxy with cheap canonical trade simulation.
14. **H33** One canonical `SearchBacktestContract` consumed by CPU GA, CUDA evaluator, quality, gauntlet, WFV/CPCV, trade logs.
15. **H37** Causal SL/TP inference per train split.
16. **H27** OOS replay before portfolio save.
17. **H39 + MH13** WFV/CPCV mandatory gates; record execution results in profile.
18. **H31** Single `RunSeed` threaded through every search path.
19. **H32** One canonical `SearchInputSliceContract` for UI and batch.

### Sprint 3 — model training quality

20. **H40** Record original train/val split in HPO refit metadata (or skip the refit).
21. **H41** Floor `embargo_rows` at `max(label_horizon_bars × 2, 20)`.
22. **H44** Anomaly threshold computed on the val slice via `fit_with_validation`.

### Sprint 4 — env-flag migration & config hardening (C7)

23. **C7 first pass** (search/discovery/eval): `discovery.rs`, `genetic/search_engine.rs`, `genetic/smc_indicators.rs`, `strategy_gene.rs`, `discovery_gpu.rs`, `cubecl_eval.rs`, `cubecl_ga.rs`, `quality.rs`.
24. **C7 second pass** (models/app): `forex-models/{genetic,base,training_orchestrator,evolution/*_gpu,tree_models/*}`, `forex-app/{trading,ctrader_execution}`.
25. **MH1** Serde bound validators for `RiskConfig` and `DiscoveryConfig`.
26. **L8** Move model training env flags into typed model config.

### Sprint 5 — data quality & feature hygiene

27. **MH10** Real timestamps in VectorTA `Candles`.
28. **MH11** Central NaN/Inf sanitisation at `FeatureFrame` boundary.
29. **MH7** Timeframe-aware quant features (`TimeframeContext`).
30. **MH8** Strip `base_tf` from `higher_tfs`.
31. **MH9** Explicit duplicate-timestamp policy.

### Sprint 6 — simulation completeness & period ledger

32. **MH4** Share one position state machine between fast/slow evaluators.
33. **MH5** Typed `BacktestMetrics` replacing `[f64; 11]`.
34. **MH17** Canonical `PeriodLedger` (daily/weekly/monthly).
35. **H35** Position-sizing mode in canonical backtest contract.
36. **H36** Live-risk-gate-adjusted backtest export.
37. **H38** Slippage in CPU/CUDA backtest.
38. **MH19** Document or model partial TP / multi-R legs.

### Sprint 7 — search resume / portfolio loader

39. **S1 + S2 + S3** `SearchCheckpoint`, extended `SearchMemoryStore`, atomic discovery profile persistence.
40. **B1 + B2 + B3** `PortfolioArtifact` loader + signal engine + unified `RuntimeArtifactMetadata`.

### Sprint 8 — modularization & generic scheduler

41. **A3 + A8** Generic `runtime/scheduler` replaces `hpc.rs` + `hpc_gpu_discovery.rs`. Custom kernels stay; their device/precision/env handling moves to the scheduler.
42. **A1 + A2** Split large search files; `Gene` lifecycle stages.
43. **A4** Unified `validation/` module.
44. **A5 + A6** One owner each for features and SMC.
45. **A7** Generated VectorTA registry.
46. **A10** Install-time hardware probe + UI-editable config (replaces remaining env vars in the installable distribution).

### Sprint 9 — cleanups

47. **L1** Delete `hpc_simd.rs`.
48. **L2-L5** Trailing-stop comment, LightGBM calibration, build.rs credential log, O(n²) Garman-Klass.
49. **L7** Archive or delete the stale 2026-03-20 manifest.

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

Add a `grep`-based CI step that fails on new `std::env::var` calls in
`forex-search`, `forex-models`, and `forex-app/app_services` unless
explicitly allowlisted (build/CI/secret/emergency-GPU-required only).

---

## Closed register — already fixed in `claude/fix-strategy-search-cpu-gpu-rtkhl`

These items were closed by commits in this branch and are intentionally
**not** part of the open work above.

| ID | Issue | Closing commit/scope |
|---|---|---|
| C3 | CUDA backtest entered on current-bar signal | Prior-bar signal + matching CPU |
| H1 | `ctrader_account_equity` dropped unrealized PnL | Sum `position.profit` over reconcile |
| H2 (FX) | Hard-coded pip-position lookup | FX subset cached; full metadata cache pending as H2' |
| H3 (estimate) | `estimated_loss` hard-coded `× 10.0` | Estimate improved; full FX rate pending as H3' |
| H4 | `risk_per_trade` violation only warned | Hard error |
| H5 | `day_start_equity` never reset | `handle_day_boundary` wired |
| H6 | No prop-firm phase rollover | `handle_phase_rollover` wired |
| H7 | OAuth token expiry kills trading | Sentinel + `force_refresh_ctrader_token_bundle` retry once |
| H8 | `PartialFill` treated as `Filled` | Returns error |
| H9 | `read_matching_response` blocks indefinitely | 30s `tokio::time::timeout` |
| H10 | Unseeded RNG in GA helpers | `&mut StdRng` threaded |
| H11 | GPU `mean_std` no NaN filter | NaN filter + Bessel correction |
| H12 | GPU profit-factor uncapped | Cap at 10.0 |
| H13 | Cross-pair pip value wrong | `quote_to_account_rate` parameter |
| H14 | Calmar = 0 for flawless equity | Saturates to 1000.0 |
| H15 | Intra-bar lookahead `signals[i]` | Shifted to `signals[i-1]` |
| H20 | Batch aborted on single discovery failure | `match` + `discovery_failures` counter |
| H21 | Monthly consistency single-trade weighting | `min_trades_per_month = 4` |
| H22 | `gauntlet.warn_only=true` silent pass | `tracing::warn!` describes failed metric |
| H23 | Archive dedup by strategy_id | Dedup by `gene_signature_hash` |
| H24 | `signals_for_gene` skipped SMC gating in validation/gauntlet | Switched to `signals_for_gene_full` |
| H42 | Burn early-stop monitored training loss | `fit_with_validation` + external val frame |
| H43 | Tree models had no `eval_set` | LightGBM `train_with_valid`; XGBoost manual early-stop loop; CatBoost val logged (CLI plumbing pending) |
| MH2 | `session_timezone` hard-coded UTC | `broker_timezone` field added |
| MH18 | `PAYLOAD_TYPE - 2` arithmetic | Named constant |
| L6 | Comments referenced removed Python files | Updated to Rust architecture |
| Python residue | `mt5-bridge`, `forex-bindings`, `onnx_exporter.rs`, pyo3 | Deleted |
| Prefilter snooping | Forward returns from full data | In-sample-only (70%) |
| GPU fallback warnings | Silent CPU fallback | Loud warnings + `FOREX_BOT_REQUIRE_GPU` panic guard |
| Python README/docs | Mentioned Python | Updated to pure Rust |

---

## How to use this plan

1. Pick the next item from Sprint 1 (or whichever sprint you're on).
2. Read the linked audit(s) for the full reasoning.
3. Implement, test, commit on `claude/fix-strategy-search-cpu-gpu-rtkhl`.
4. Update this file's status as items close.
5. Push to PR #3 when each sprint is ready for review.

The tier1/tier2/tier3 split from the earlier mutable-crafting plan is
superseded by this document. Anything not listed here is either closed
(see the register above) or out of scope (UI rewrite, multi-account
streaming, etc.).
