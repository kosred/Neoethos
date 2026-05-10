# Operator-facing `*_profile.json` field reference

Every discovery cycle writes a sibling `*_profile.json` next to the
portfolio export. This document is the field-by-field reference for
that JSON. It exists because the audit calls out that operators must
be able to read a persisted profile and reconstruct the run's
configuration, validation outcome, and promotion-readiness without
opening source code.

The schema lives in
[`crates/forex-search/src/discovery.rs`](../../crates/forex-search/src/discovery.rs)
on `DiscoveryRunProfile` and is built by `build_discovery_profile`.

---

## Run identity

| Field | Type | Source | Notes |
|---|---|---|---|
| `timeframe_label` | string | `DiscoveryConfig::timeframe_label` | The base timeframe of the run, e.g. `"M1"`. |
| `population` | usize | config | Genetic search population. |
| `generations` | usize | config | Genetic search generation cap. |
| `max_indicators` | usize | config | Max indicators per gene. |
| `candidate_count_target` / `portfolio_size_target` | usize | config | Search sizing knobs. |
| `max_rows` | usize | derived | Effective row cap (`max_rows_by_timeframe` ∩ global). |
| `max_runtime_hours` | f64 | config | Wall-clock budget; `0.0` means unlimited. |

## Search filters

| Field | Type | Notes |
|---|---|---|
| `corr_threshold` | f64 | Portfolio correlation cutoff. |
| `min_trades_per_day` | f64 | Filter floor. |
| `walkforward_splits` | usize | Walk-forward fold count. |
| `embargo_minutes` | usize | Embargo between train/test windows. |
| `enable_cpcv` / `cpcv_*` | mixed | Combinatorial purged CV configuration. |
| `filters` | object | Full `FilteringConfig` snapshot (max DD, min profit, etc.). |

## Resolved runtime overrides

| Field | Type | Notes |
|---|---|---|
| `prefilter_top_k` | usize | After Phase 17, this is the resolved typed value (`0` disables the prefilter). |
| `prefilter_insample_frac` | f64 | After Phase 17 — clamped to `(0, 1]`. |
| `funnel_stage1_pct` | f64 | After Phase 17 — clamped to `[0.01, 1.0]`. |
| `determinism_policy` | object | Phase 51. `{ "mode": "deterministic", "seed": <u64> }` or `{ "mode": "best_effort" }` or `{ "mode": "non_deterministic_allowed" }`. |

## Run observations

| Field | Type | Notes |
|---|---|---|
| `candidates_observed` / `portfolio_observed` | usize | Sizes of the candidate pool and selected portfolio. |
| `quality_metrics_observed` | usize | Number of strategies passed through `StrategyQualityAnalyzer`. |
| `logged_trade_sets` | usize | Strategies that opted into trade logging via `FilteringConfig::log_trades`. |

## Validation gates

| Field | Type | Notes |
|---|---|---|
| `walkforward_passed` | bool | Aggregate from per-strategy walk-forward summaries. |
| `cpcv_passed` | bool | Combinatorial purged CV pass under the configured `cpcv_min_phi`. |
| `cpcv_fold_count` / `cpcv_profitable_fold_ratio` | usize / f64 | Diagnostic detail for the CPCV check. |
| `validation_temporal_contract_hash` | optional string | Aggregate temporal-contract hash that produced the canonical artifacts. `null` until at least one validation artifact ships. |

## Per-kind artifact counts

| Field | Type | Notes |
|---|---|---|
| `canonical_backtest_artifacts_observed` | usize | One per portfolio strategy when canonical backtest ran. |
| `walkforward_validation_artifacts_observed` | usize | Same shape, walk-forward counterpart. |
| `forward_test_validation_artifacts_observed` | usize | Phase 24 — forward-test on the held-out tail. |
| `prop_firm_validation_artifacts_observed` | usize | Phase 29 — prop-firm risk validation on the same tail. |

## Promotion-readiness summary (Phases 48-51)

| Field | Type | Notes |
|---|---|---|
| `validation_evidence_hashes` | object | One field per kind (`canonical_backtest`, `walkforward`, `forward_test`, `prop_firm`, `live_execution_simulation`). Each is `string` when the kind shipped at least one artifact, `null` otherwise. |
| `validation_evidence_complete` | bool | `true` only when every kind including `live_execution_simulation` has a hash. **Currently always `false`** because the simulator is deferred. |
| `validation_evidence_missing_kinds` | string array | Sorted list of kind names missing from this run. Always contains `"live_execution_simulation"` until the simulator lands. |

Operators can read these three fields together to decide whether a
portfolio is promotion-ready without instantiating a
`LivePromotionGate`. The same data feeds the promotion-readiness
runbook ([`promotion_readiness.md`](promotion_readiness.md)).

## Cross-references

- Per-artifact contracts: [`artifact_safety.md`](artifact_safety.md)
- Promotion gate flow + rejection reasons:
  [`promotion_readiness.md`](promotion_readiness.md)

## Forward compatibility

The schema is additive. Phases 49 / 51 added fields without removing
any; future phases that wire the live-execution simulator will
populate `validation_evidence_hashes.live_execution_simulation` and
flip `validation_evidence_complete` to `true` once the simulator
artifact ships. Operators should expect new fields to appear and
should not reject profiles with unrecognized keys.
