# Forex-AI Improvement Plan

**Created:** 2026-05-04 Europe/Berlin
**Last refresh:** 2026-05-04 (post-discussion: deduplication-first refactor)
**Repository:** kosred/forex-ai
**Branch:** `claude/fix-strategy-search-cpu-gpu-rtkhl`
**Source:** 32 audit files dated 2026-05-03 / 2026-05-04 in `docs/audits/`

This plan only contains **open work**. Items already closed by commits in
this branch are listed at the bottom for traceability and removed from the
sprint queue.

---

## Goal

Make the bot:

1. **Correct** — fix lookahead, timestamp-unit, RNG-seed, and risk-rule
   issues that silently corrupt search ranking and live execution.
2. **Smaller** — delete duplicated logic by introducing one owner per
   concept. Realistic net target: **−3.000 to −5.000 LoC (~10–18%)**.
3. **Faster** — fewer files compile, one canonical hot path per concept,
   GPU and CPU share the same state machine.
4. **Stable** — every fix lives in one place. Bug fixes propagate to all
   callers automatically.
5. **Maintainable** — small focused modules, typed contracts, explicit
   provenance, no scattered env reads.

The improvement is achieved by a **deduplication-first** refactor: each
concept gets one owner module with backend implementations behind a trait;
the duplicate implementations are **deleted in the same PR** that
introduces the owner. No deprecation period.

---

## Duplication map (the source of the LoC reduction)

These are the seven concepts that account for almost all of the
deletable code. Each row is one Sprint-8 PR.

| Concept | Currently lives in | Becomes one owner at | Estimated LoC removed |
|---|---|---|---|
| **Backtest state machine** | `eval.rs::fast_evaluate_strategy_core`, `eval.rs::simulate_trades_core`, `cubecl_eval.rs::backtest_population_kernel`, `gauntlet.rs`, `validation.rs::walkforward_risk_diagnostics`, `quality.rs` (re-simulates trades), `discovery.rs` signal-count proxy | `forex-search/evaluation/canonical_backtest.rs` + `BacktestBackend` trait → CPU / CubeCL-CUDA / HPC impls | **−1.500 to −2.500** |
| **Feature pipeline & SMC** | `forex-data/core/smc.rs`, `forex-search/genetic/smc_indicators.rs`, plus CPU+GPU re-implementations of alignment/scaling | `forex-data/features/` single owner; `forex-search` consumes via typed `FeatureFrame` + `SmcSearchPolicy` | **−800 to −1.500** |
| **GPU device / precision / env handling** | `cubecl_eval.rs`, `cubecl_ga.rs`, `linear_gpu.rs`, `crfmnes_gpu.rs`, `neat_gpu.rs` — each calls `cuda_device_id()`, reads precision env, does its own fallback | `forex-core/runtime/scheduler.rs` + `DeviceAssignment` typed contract; kernels stay, only their device handling moves | **−500 to −800** |
| **Risk logic** | `forex-core/domain/risk.rs::RiskManager` (700 LoC dead), inline `prop_firm_pre_trade_check` in `trading.rs`, `forex-search/challenge.rs`, risk gates scattered in `eval.rs` and `validation.rs` | `forex-core/domain/risk.rs::RiskManager` becomes the single owner; other paths call it | **−400 to −700** |
| **Hardware planning** | `forex-core/system.rs::HardwareExecutionPlan`, `forex-search/hpc.rs` (Hyperstack-specific), `forex-search/hpc_gpu_discovery.rs`, plus inline GPU detection in 5+ files | `forex-core/runtime/{hardware_profile, hardware_probe, topology}.rs`; Hyperstack becomes one detected profile | **−300 to −500** |
| **Timestamp handling** | `forex-data/core/timestamps.rs` exists but is ignored by `eval.rs`, `cubecl_eval.rs`, `quality.rs`, `smc.rs`, `validation.rs`, `search_engine.rs` (each assumes ms or recomputes from raw `i64`) | `TimestampMs` newtype enforced at `FeatureFrame` boundary; all consumers take it | **−150 to −300** |
| **Genetic core** | `forex-search/genetic/*` (search GA), `forex-models/genetic.rs` (model GA), `forex-models/evolution/*` (NEAT, CRFMNES) — each has its own selection / mutation / crossover | `forex-core/genetic/` shared core (selection policies, RNG plumbing); search/models/evolution call it | **−500 to −1.000** |
| **Env-flag reads** (after C7 migration) | ~40 `std::env::var` reads in `forex-search`, `forex-models`, `forex-app/app_services` controlling search/eval/training behavior | Typed config (`DiscoveryConfig`, `SearchConfig`, `BacktestConfig`, model configs); env limited to build/CI/secrets | **−200 to −400** |

| Sum | | | **−4.350 to −7.700 LoC** |

| New code added | | | **+1.500 to +2.500 LoC** |
| (typed contracts, traits, scheduler skeleton, lifecycle stages, validation policy) | | | |

| **Net** | | | **−2.850 to −5.200 LoC** |

These numbers assume the deduplication is done strictly: in each PR
the new owner is introduced **and** the old duplicates are deleted in
the same commit set. Skipping the delete step keeps the project
larger; a CI grep gate (see Sprint 4) prevents the duplicates from
silently re-emerging.

---

## CRITICAL — show-stoppers for prop-firm trading

| ID | File(s) | Issue | Fix |
|---|---|---|---|
| **C1** | `forex-data/src/core/resample.rs`, `features.rs` | HTF resampling lookahead. Resampled H1/H4/D1 candles are timestamped at bucket-start but carry OHLC from the closed bucket. `align_features_by_ns` then leaks future HTF values into lower-TF rows. | `CandleTimestampPolicy::{OpenTime, CloseTime}` + separate `available_at_ms`. HTF features become available only after the HTF candle closes. |
| **C2** | `resample.rs`, `search_engine.rs`, `eval.rs`, `validation.rs`, `quality.rs`, `cubecl_eval.rs` | Timestamp unit mismatch. Data uses ns, search/eval/quality/CUDA assume ms. Day/month bucketing, gap exits, kill zones, and Monte Carlo blocks silently break under ns. | `TimestampMs` newtype at the `FeatureFrame` boundary; `TimestampUnit` carried in `Ohlcv` metadata. Single-owner timestamp module from the duplication map. |
| **C4** | `cubecl_eval.rs`, `eval.rs` | CUDA session contract missing. CUDA models max-hold/spread/commission/pip-value but not kill zones, Friday force-exit, Monday/Friday entry blocks, broker-tz day boundaries. CPU `simulate_trades_core` does. | Wire session/kill-zone logic into the CUDA kernel **or** downgrade CUDA to approximate-presearch only and require canonical CPU validation. |
| **C5** | `discovery.rs`, `forex-app/src/app_services/discovery.rs`, `forex-cli` | Feature-index ↔ feature-name mismatch after prefilter. `prefilter_features` replaces the matrix and name list inside discovery. Discovered gene indices are relative to the filtered matrix; callers (`save_portfolio_json`) supply original names. Exported indicator names can point to wrong columns. | Add `effective_feature_names: Vec<String>` and `feature_mapping: Vec<usize>` to `DiscoveryResult`. Change every `save_portfolio_json` caller. |
| **C6** | `.github/workflows/ci.yml` | CI does not cover `forex-search` or `forex-models`. Search/backtest/model regressions land without CI catching them. (In progress.) | Add `cargo check` and `cargo test` for `-p forex-search` and `-p forex-models --no-default-features`. |
| **C7** | `discovery.rs`, `genetic/search_engine.rs`, `cubecl_eval.rs`, etc. | Search/discovery semantics controlled by env flags. Two runs with identical config + data produce different portfolios when env differs. | Migrate to typed config (`DiscoveryConfig`, `SearchConfig`, `SmcSearchConfig`, `BacktestConfig`). Export effective values in the run contract. Allowlist only build/CI/secret env vars. |
| **C8** | `forex-core/src/domain/risk.rs` | `RiskManager` (~700 LoC) has zero callers anywhere. `prop_firm_pre_trade_check` in `trading.rs` uses a weaker inline replacement. | Wire `RiskManager` into `trading.rs` trading loop; delete the inline replacement. Owner from the duplication map. |

---

## HIGH — significant correctness and safety

### Prop-firm execution safety

| ID | File:line | Issue | Fix |
|---|---|---|---|
| **H2'** | `trading.rs:1727-1735` | Hard-coded pip-position lookup is FX-only. Crypto/indices/non-gold metals still wrong. | Use the symbol-metadata cache (MH3) to read `digits`/`pipPosition` from `ProtoOASymbol`. |
| **H3'** | `trading.rs:3317` | Loss estimate improved but still not full cross-pair quote→account FX rate. | Multiply `pip_value_per_lot` × `quote_currency → account_currency` rate from cached metadata. |
| **MH3** | `ctrader_data.rs` | Symbol metadata fetched on every order. | `Cache<symbol_id, SymbolInfo>` with 1h TTL refreshed at first trade of day. Single fix unblocks H2', H3'. |

### Search determinism & ranking

| ID | File:line | Issue | Fix |
|---|---|---|---|
| **H16** | `discovery.rs` | Regime-robustness gate uses hardcoded `-3000.0` absolute loss. On 10k account = 30%. | Replace with `initial_balance × max_regime_loss_pct` from config. |
| **H17** | `validation.rs` | `WALKFORWARD_INITIAL_BALANCE = 100_000.0` hardcoded. WFV daily/monthly metrics ignore the configured balance. | Pass `initial_balance` from config through `WalkforwardBacktestInput`. |
| **H18** | `validation.rs`, `eval.rs` | WFV diagnostics pass `days` as timestamps to `simulate_trades_core`. Duration/gap/kill-zone/max-trades-per-day all wrong inside WFV. | Carry real `timestamps` in `WalkforwardBacktestInput`; pass to `simulate_trades_core`. |
| **H19** | `eval.rs`, `cubecl_eval.rs` | Final open month/day not safely closed before metric computation. Last open month omitted from Sharpe and consistency. | Finalize all open periods at end of backtest in CPU and CUDA paths. |
| **H25** | `discovery.rs`, `eval.rs`, `gauntlet.rs` | Signal count used as proxy for trade count. Non-zero signal count ≠ executed trades. | Replace with cheap canonical trade simulation in the discovery filter — same state machine as the duplication-map owner. |
| **H26** | `forex-app/src/app_services/discovery.rs`, `orchestration.rs` | Batch discovery passes `&[]` for higher TFs. Same symbol/config produces different feature universe per entrypoint. | Read configured `higher_tfs` from `DiscoveryConfig`. |
| **H27** | `forex-app/src/app_services/discovery.rs` | UI withholds 20% OOS but does not replay/validate before portfolio save. The 20% is unused. | After discovery, replay selected candidates on the withheld 20% OOS. Gate export on OOS pass. |
| **H28** | `discovery.rs` | `DiscoveryResult` does not carry `effective_feature_names` or `feature_mapping`. | Same fix as C5. |
| **H29** | `discovery.rs`, `search_engine.rs` | `fitness` field used with different meanings across paths (GA composite, ranking, UI/export imply income). | Separate `search_score`, `net_profit`, `quality_score`, `oos_score`, `portfolio_score`. |
| **H30** | Various | No `SearchRunContract` / `PortfolioContract`. Exported portfolio cannot prove which data/features/seed/evaluator/gates produced it. | Define typed contracts + provenance hash + git commit + env overrides + validation results. Mandatory in every artifact. |
| **H31** | `discovery.rs` and friends | Not fully deterministic. No single run-seed propagated to all paths (`discovery_gpu.rs`, `hpc_gpu_discovery.rs`, `cubecl_ga.rs`, `quality.rs`, `forex-models/genetic.rs`). | Single `RunSeed` threaded everywhere. Test: same seed + same config + same data ⇒ identical portfolio. |
| **H32** | `forex-app/src/app_services/discovery.rs`, `orchestration.rs` | UI/batch discovery slice data differently. UI cuts to 80%, batch passes the full frame. | One canonical `SearchInputSliceContract`. |
| **H33** | `discovery.rs`, `search_engine.rs` | Search/quality/gauntlet use different cost/session contracts. Defaults leave kill zones, min-hold, max-trades/day, gap-threshold unset in some paths. | One canonical `SearchBacktestContract` consumed by every backtest path — exactly what the duplication-map owner provides. |
| **H34** | `discovery_gpu.rs`, `hpc_gpu_discovery.rs` | Tensor/HPC GPU discovery uses simplified return-based fitness (no SL/TP/spread/commission). Documented but easily mistaken for canonical. | Tag artifacts as `ApproxSearchCandidateBatch`. Require canonical CPU validation before becoming `ValidatedStrategyGene`. |
| **H35** | `eval.rs`, `cubecl_eval.rs` | Backtest uses fixed pip-value/lot-size, not live dynamic risk sizing. | Add position-sizing mode to canonical backtest contract: `FixedLot`, `FixedRiskPct`, `VolTargeted`, `LiveRiskManagerParity`. |
| **H36** | `eval.rs`, `cubecl_eval.rs` | Live risk gates not modelled in backtest (circuit breaker, recovery mode, daily/intraday DD blocks, revenge-trade detection, confidence threshold). | Either model in backtest or export `live_gate_adjusted_backtest`. |
| **H37** | `stop_target.rs`, `search_engine.rs` | SL/TP inference can use future data. `resolve_stop_target_arrays` calls `infer_stop_target_pips` on the full OHLCV slice. | Compute SL/TP defaults per train split using only train data; freeze for test/OOS. |
| **H38** | `eval.rs`, `cubecl_eval.rs` | Slippage not first-class. `RiskConfig` has it; backtest ignores it. | Add `slippage_pips` to `BacktestSettings`; apply at fill time. |
| **H39** | `validation.rs` | WFV/CPCV not a hard discovery gate. | Make `walkforward_passed` and `cpcv_passed` mandatory in `DiscoveryRunProfile`; gate portfolio export on them. |

### Model training quality

| ID | File:line | Issue | Fix |
|---|---|---|---|
| **H40** | `training_orchestrator.rs:3043-3235` | Post-HPO full-data refit records `val_rows=0`, hiding the original split. | Either record the original train/val split alongside refit metadata, or skip the full-data refit. |
| **H41** | `runtime/hpo.rs` | `embargo_rows` can be 0 if `embargo_minutes==0`. Train→val label leakage possible. | Floor at `max(label_horizon_bars × 2, 20)`. |
| **H44** | `anomaly/forest_impl.rs` | Anomaly threshold computed from training scores (optimistic). | Compute on the val slice via `fit_with_validation`. |

---

## MEDIUM — correctness drift, ranking quality

| ID | File(s) | Issue | Fix |
|---|---|---|---|
| **MH1** | `forex-core/src/config.rs:225-341` | Risk config fields lack serde bound validation. `risk_per_trade: 50` (50%) silently passes. | Custom `Deserialize` validators clamping to safe ranges, `tracing::error!` on out-of-range. |
| **MH4** | `eval.rs` | `fast_evaluate_strategy_core` lacks Friday force-exit and Monday/Friday entry blocking. Fast metrics and detailed trade logs disagree. | Resolved by the backtest-state-machine owner from the duplication map. |
| **MH5** | `eval.rs`, `cubecl_eval.rs`, etc. | Raw metric layout `[f64; 11]` with implicit indices. Index mismatch silently corrupts selection/validation/UI/export. | Typed `BacktestMetrics` struct. |
| **MH6** | `gauntlet.rs` | Gauntlet uses its own settings and the simplified `signals_for_gene`. | Consume the canonical `SearchBacktestContract` (H33). |
| **MH7** | `forex-data/src/core/quant_features.rs` | "Daily/weekly" quant features use fixed bar counts (`24`, `120`) regardless of timeframe. | `TimeframeContext { bar_duration_ms, bars_per_day, bars_per_week }`. |
| **MH8** | `forex-core/src/config.rs`, `forex-data/src/lib.rs` | `higher_tfs` can include `base_tf`, duplicating it. | Strip `base_tf` before alignment; reject in config validation. |
| **MH9** | `forex-data/src/lib.rs` | Duplicate timestamp rows silently dropped (`dedup_by`). Wrong row may be kept. | Make duplicate handling explicit: reject, keep-last-with-log, or aggregate. |
| **MH10** | `forex-data/src/core/hpc_ta.rs` | `compute_classic_ta_columns` builds VectorTA `Candles` with `timestamps = vec![0i64; n]`. Time-dependent indicators are wrong. | Pass real timestamps with explicit `TimestampUnit`. |
| **MH11** | `forex-data/src/lib.rs`, `core/features.rs` | `FeatureFrame` lacks central NaN/Inf sanitation. CPU and GPU handle NaNs differently. | Sanitize at the `FeatureFrame` boundary. |
| **MH12** | `eval.rs`, `cubecl_eval.rs` | Final open position handling undefined. CPU and CUDA may differ. | Single end-of-test policy applied uniformly. Resolved by backtest owner. |
| **MH13** | `forex-app/src/app_services/discovery.rs` | UI discovery profile records validation config but not validation results. | Add `oos_tested`, `oos_passed`, `walkforward_passed`, `cpcv_passed`. |
| **MH14** | `forex-app/src/app_services/discovery.rs` | UI "best strategy" selected by `gene.fitness`, not OOS/quality contract. | Use `portfolio_score` (H29), gated by H39 results. |
| **MH15** | `validation.rs` | WFV evaluates precomputed signals — not retrain-per-split WFO. Distinction undocumented. | `WalkForwardKind::{FixedSignal, RetrainPerSplit}` enum + docs. |
| **MH16** | `discovery.rs`, `eval.rs` | Stage-1 search window narrower than expected when UI already cut to 80%. | Compute stage-1 window from full dataset percent. |
| **MH17** | `eval.rs`, `cubecl_eval.rs`, `quality.rs` | No canonical day/week/month period ledger. | Single `PeriodLedger { daily, weekly, monthly }` emitted by reference backtest, mirrored by CUDA. |
| **MH19** | `forex-core/src/domain/order_execution.rs` vs `eval.rs` | Live execution has partial TP, multi-R legs, entry patience/pullback that backtest does not simulate. | Either model in backtest or document divergence per artifact. |

---

## LOW — cleanup, docs, dead code

| ID | File(s) | Issue | Fix |
|---|---|---|---|
| **L1** | `forex-search/src/hpc_simd.rs` | ~340 LoC dead code. | Delete. |
| **L2** | `eval.rs:322-335` | Trailing stop never activates on shorts unless `trailing_be_trigger_r` is met. | One-line comment documenting the constraint. |
| **L3** | `tree_models/lightgbm.rs:589-594` | Temperature calibration followed by re-normalize is redundant. | Simplify. |
| **L4** | `embedded_credentials.rs`, `build.rs:202-208` | `println!` of embedded credential length leaks into CI logs. | Suppress in release builds. |
| **L5** | `forex-data/src/core/regime_detection.rs:35-78` | Garman-Klass volatility O(n × window). | Convert to rolling sum O(n). |
| **L7** | `cache/audit/2026-03-20-file-manifest.txt` | Stale manifest references removed Python files. | Archive or delete. |
| **L8** | `forex-models` GPU/evolution env flags | Model behavior controlled by env (`FOREX_BURN_MODEL_SUPPORTS_BF16`). | Move into typed model config (folds into C7 second pass). |
| **L9** | `ctrader_streaming.rs` | Streaming session uses global singleton without account keying. | Defer until multi-account milestone. |
| **L10** | `ctrader_session.rs`, `ctrader_proto_messages.rs` | `#![allow(dead_code)]` — full-duplex async session not wired. | Defer until persistent-streaming milestone. |

---

## Search resume & live bridge

| ID | Concern | Fix |
|---|---|---|
| **S1** | Strategy search starts from zero every run. Long-running discovery wastes compute and re-discovers the same early-stage candidates. | `SearchCheckpoint { rng_state, archive, signature_memory, generation, ga_population, hall_of_fame, validation_cache }`, persisted atomically. |
| **S2** | `SeenSignatureMemory` stores only signatures, not gene bodies / metrics / regimes / OOS results. | `SearchMemoryStore { signatures, gene_bodies, eval_metrics, validation_history, regime_tags }`. |
| **S3** | Discovery profiles do not yet support resume. | Atomic-write pattern (same as model artifacts). |
| **B1** | Discovery portfolio JSON is not loadable as a runtime artifact. | Strict `PortfolioArtifact` loader with schema validation and round-trip test. |
| **B2** | `trading.rs` has no path from a discovered portfolio to live signals. | `signal_engine` consuming `PortfolioArtifact`, feeding the existing risk/exec gate. |
| **B3** | Model and search artifacts have different provenance schemas. | One `RuntimeArtifactMetadata` envelope (extend the existing model one). |

---

## Sprint order

The order is chosen so that **correctness lands first** (live deployment
unblocked), the **deduplication owners arrive next** (so subsequent
work uses them), and the **architectural tidy-up is last** (lowest
urgency, highest coordination cost).

### Sprint 1 — Unblock prop-firm trading (P0 correctness)

1. **C8** Wire `RiskManager` into `trading.rs`; delete inline `prop_firm_pre_trade_check`. Also resolves the duplication-map "Risk logic" owner.
2. **MH3 + H2' + H3'** Symbol-metadata cache (TTL 1h) → fixes pip lookup and loss estimate.
3. **C6** Add `cargo check` / `cargo test` jobs for `-p forex-search` and `-p forex-models --no-default-features`.
4. **H16** Replace hardcoded `-3000.0` regime-loss gate with `initial_balance × max_regime_loss_pct`.
5. **H17 + H18** WFV uses configured `initial_balance` and real `timestamps`.
6. **H19 + MH12** Close all open periods at backtest end.
7. **C5 + H28** `effective_feature_names` and `feature_mapping` in `DiscoveryResult`; update every caller.
8. **H26** Batch discovery reads `higher_tfs` from `DiscoveryConfig`.

### Sprint 2 — Backtest state machine owner (largest LoC win)

This is the duplication-map row "Backtest state machine". Single PR
chain. Each PR introduces one piece of the new owner and **deletes**
the corresponding duplicate.

9. **H33** Define `SearchBacktestContract` and `BacktestBackend` trait in `forex-search/evaluation/`.
10. **C3 (already ✅)** + canonical CPU impl wired for `simulate_trades_core` callers.
11. CUDA backend (`backtest_population_kernel`) refactored to consume the same contract; behavior parity test added.
12. Migrate `gauntlet.rs`, `walkforward_risk_diagnostics`, `quality.rs` re-simulation, `discovery.rs` signal-count proxy to the new owner. **Delete** their bespoke trade-loop code.
13. **H25** Discovery filter switches to canonical trade simulation.
14. **MH4 + MH6** Resolved automatically once the bespoke implementations are deleted.
15. **MH5** Replace `[f64; 11]` with typed `BacktestMetrics` (carried by the contract).
16. **MH17** `PeriodLedger { daily, weekly, monthly }` emitted by the canonical evaluator.

Expected LoC: **−1.500 to −2.500**.

### Sprint 3 — Search contract & CPU/GPU parity

17. **C1** HTF resampling: `CandleTimestampPolicy` + `available_at_ms`.
18. **C2** `TimestampMs` newtype enforced at `FeatureFrame` boundary; `TimestampUnit` in `Ohlcv`. (Duplication-map "Timestamp handling" — delete the 6 hand-rolled implementations.)
19. **C4** Wire session/kill-zone/day-boundary into the CUDA kernel **or** downgrade CUDA to approximate-presearch (H34). Whichever is chosen, document the artifact tag clearly.
20. **H29 + H30** Split `fitness` into `search_score`, `net_profit`, `quality_score`, `oos_score`, `portfolio_score`. Define `SearchRunContract` + `PortfolioContract`.
21. **H37** Causal SL/TP inference per train split.
22. **H27** OOS replay before portfolio save.
23. **H39 + MH13** WFV/CPCV mandatory gates; record execution results in profile.
24. **H31** Single `RunSeed` threaded through every search path.
25. **H32** One canonical `SearchInputSliceContract` for UI and batch.

### Sprint 4 — Env-flag migration & config hardening (C7)

26. **C7 first pass** (search/discovery/eval): `discovery.rs`, `genetic/search_engine.rs`, `genetic/smc_indicators.rs`, `strategy_gene.rs`, `discovery_gpu.rs`, `cubecl_eval.rs`, `cubecl_ga.rs`, `quality.rs`.
27. **C7 second pass** (models/app + L8): `forex-models/{genetic, base, training_orchestrator, evolution/*_gpu, tree_models/*}`, `forex-app/{trading, ctrader_execution}`.
28. **MH1** Serde bound validators for `RiskConfig` and `DiscoveryConfig`.
29. **CI grep gate**: fail on new `std::env::var` calls in `forex-search`, `forex-models`, `forex-app/app_services` unless explicitly allowlisted (build/CI/secrets/emergency-GPU-required).

### Sprint 5 — Model training quality

30. **H40** Record original train/val split in HPO refit metadata (or skip the refit).
31. **H41** Floor `embargo_rows` at `max(label_horizon_bars × 2, 20)`.
32. **H44** Anomaly threshold on val slice via `fit_with_validation`.

### Sprint 6 — Data quality & feature hygiene

33. **MH10** Real timestamps in VectorTA `Candles`.
34. **MH11** Central NaN/Inf sanitisation at `FeatureFrame` boundary.
35. **MH7** Timeframe-aware quant features.
36. **MH8** Strip `base_tf` from `higher_tfs`.
37. **MH9** Explicit duplicate-timestamp policy.

### Sprint 7 — Simulation completeness

38. **H35** Position-sizing mode in canonical backtest contract.
39. **H36** Live-risk-gate-adjusted backtest export.
40. **H38** Slippage in CPU/CUDA backtest.
41. **MH19** Document or model partial TP / multi-R legs.
42. **MH15** `WalkForwardKind::{FixedSignal, RetrainPerSplit}` enum + docs.
43. **MH14 + MH16** UI selects best by `portfolio_score`; stage-1 window from full dataset.

### Sprint 8 — Search resume / portfolio loader

44. **S1 + S2 + S3** `SearchCheckpoint`, extended `SearchMemoryStore`, atomic discovery profile persistence.
45. **B1 + B2 + B3** `PortfolioArtifact` loader + signal engine + unified `RuntimeArtifactMetadata`.

### Sprint 9 — Modularization & generic scheduler

This is the rest of the duplication map. **Each row = one PR set
that introduces the owner and deletes the duplicates.**

46. **Feature/SMC owner** (duplication-map row 2): `forex-data/features/` becomes the single owner. Search consumes `FeatureFrame` + `SmcSearchPolicy`. Delete `forex-search/genetic/smc_indicators.rs` re-implementations.
47. **GPU device/precision/env owner** (duplication-map row 3): `forex-core/runtime/scheduler.rs` + `DeviceAssignment`. Custom kernels stay; their device handling moves. Audit `custom_cuda_kernel_preservation_audit_2026-05-03.md` is explicit: keep the kernels.
48. **Hardware planning owner** (duplication-map row 5): `forex-core/runtime/{hardware_profile, hardware_probe, topology}.rs`. Hyperstack becomes one detected profile, not a hardcoded path. Replaces `hpc.rs` and `hpc_gpu_discovery.rs`.
49. **Genetic core owner** (duplication-map row 7): `forex-core/genetic/` shared selection / mutation / crossover; search/models/evolution call it.
50. **Validation owner** (A4): `validation/{trade_ledger, metrics, quality_policy, monte_carlo, challenge_policy, validation_report}.rs`. Merges `quality.rs` and `challenge.rs`.
51. **Lifecycle stages of `Gene`** (A2): split into `CandidateGene → EvaluatedGene → ValidatedStrategyGene → PortfolioStrategy`.
52. **VectorTA registry** (A7): generated from dispatch metadata; build-time validation that every listed indicator exists.
53. **Install-time hardware probe + UI-editable config** (A10): replaces remaining env vars in the installable distribution.

### Sprint 10 — Cleanups

54. **L1** Delete `hpc_simd.rs`.
55. **L2-L5** Trailing-stop comment, LightGBM calibration, build.rs credential log, O(n²) Garman-Klass.
56. **L7** Archive or delete the stale 2026-03-20 manifest.

---

## CI guardrails to add alongside Sprint 1

```yaml
# .github/workflows/ci.yml
- name: Check forex-search
  run: cargo check -p forex-search

- name: Test forex-search
  run: cargo test -p forex-search

- name: Check forex-models (no-default-features)
  run: cargo check -p forex-models --no-default-features

- name: Test forex-models (no-default-features)
  run: cargo test -p forex-models --no-default-features
```

Plus, alongside Sprint 4:

```bash
# Fail on new env reads outside allowlist
git grep -nE 'std::env::var\("(?!FOREX_BOT_REQUIRE_GPU|RUST_LOG|CI|GITHUB_)' \
  -- 'crates/forex-search/**/*.rs' 'crates/forex-models/**/*.rs' \
     'crates/forex-app/src/app_services/**/*.rs' \
  && exit 1 || exit 0
```

---

## Closed register — already fixed in this branch

These items were closed by commits in `claude/fix-strategy-search-cpu-gpu-rtkhl`
and are intentionally **not** part of the open work above.

| ID | Issue | Closing scope |
|---|---|---|
| C3 | CUDA backtest entered on current-bar signal | Prior-bar signal matched to CPU |
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
| H24 | `signals_for_gene` skipped SMC gating | Switched to `signals_for_gene_full` |
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

1. Pick the next item from the current sprint.
2. Read the linked audit(s) for full reasoning.
3. Implement, test, commit on `claude/fix-strategy-search-cpu-gpu-rtkhl`.
4. **For Sprint 2 and Sprint 9 items: in the same PR, delete the
   duplicate implementations the new owner replaces.** Don't leave
   "deprecated" code behind — that's how the project grew to its
   current size.
5. Update the closed register and remove the item from open work.
6. Push to PR #3 when each sprint is ready for review.

The expected outcome at the end of Sprint 9 is a project that is:
**−2.850 to −5.200 LoC smaller**, **faster to compile and test**,
**deterministic given the same seed**, **safe to deploy in a
prop-firm**, and **easier to maintain** because each concept lives
in exactly one place.
