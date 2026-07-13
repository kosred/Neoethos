# Audit Remediation Wave 2 Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development`. Author tests with the fixes, but execute no build, test, lint, formatter, or package-manager command until all Waves 1-4 are authored and reviewed.

**Goal:** Correct timestamp, causality, import validation, timeframe semantics, deterministic sampling, allocation, and preprocessing math without changing unrelated strategy behavior.

**Architecture:** Canonical timestamps are epoch milliseconds. Higher-timeframe candles carry open time and availability-at-close separately. Calendar features derive from UTC timestamps and explicit timeframe duration. Dataset-dependent search state belongs to one run. Preprocessing is an immutable fit/apply artifact owned by the outer split.

---

## Chunk 1: Timestamp and availability contracts

### Task 1: Millisecond resampling and stale age (D01)

**Files:**
- Modify: `crates/neoethos-core/src/contracts/temporal.rs`
- Modify: `crates/neoethos-data/src/core/resample.rs`
- Modify: `crates/neoethos-data/src/lib.rs` (`prepare_multitimeframe_features*`, stale rebuild checks)
- Modify: `crates/neoethos-cli/src/main.rs` (`cmd_resample` supplies the source timeframe)
- Create: `crates/neoethos-data/src/timestamp_and_mtf_contract.rs`
- Modify: `crates/neoethos-data/src/lib.rs` to register the test module under `#[cfg(test)]`

- [ ] Re-read timestamp normalization, every resample/alignment/staleness caller, temporal contracts, and nearby tests; confirm all stored/in-memory OHLCV timestamps are epoch milliseconds before editing.
- [ ] Change the public boundary to `resample_ohlcv(src, source_tf, target_tf)`. Replace nanosecond constants and ambiguous duration names with checked `timeframe_duration_ms(tf) -> Result<i64>` and `_ms` variables. Require `target_duration_ms > source_duration_ms`, reject unsupported/overflowing arithmetic, and update both `ensure_timeframes_with_resample` call sites plus CLI `cmd_resample`; use saturating subtraction only for age comparisons.
- [ ] Keep bucket keys/open timestamps in milliseconds and preserve OHLCV order/volume semantics.
- [ ] Add named tests `timestamp_and_mtf_contract::epoch_ms_m1_to_m5_aggregates_exactly`, `timestamp_and_mtf_contract::epoch_ms_m1_to_h1_aggregates_exactly`, and `timestamp_and_mtf_contract::stale_age_uses_milliseconds_at_boundary`.

### Task 2: Closed higher-timeframe availability (D02)

**Files:**
- Modify: `crates/neoethos-core/src/contracts/temporal.rs` (`TemporalIndex`/availability helper)
- Modify: `crates/neoethos-data/src/core/resample.rs`
- Modify: `crates/neoethos-data/src/core/features.rs` (`align_features_asof`)
- Modify: `crates/neoethos-data/src/lib.rs` (`prepare_multitimeframe_features*`)
- Modify: `crates/neoethos-data/src/timestamp_and_mtf_contract.rs`

- [ ] Re-read all higher-TF producers/consumers and alignment tests; confirm the current open-stamp immediate alignment and incomplete-final-bucket exposure before editing.
- [ ] Represent a resampled candle with its original `open_time_ms`; derive checked `available_at_ms = open_time_ms + timeframe_duration_ms`. Do not overwrite the open timestamp used by OHLCV consumers.
- [ ] Pass an explicit availability vector to as-of alignment and expose a value only when `available_at_ms <= base_time_ms`. Reject length/order mismatches rather than guessing.
- [ ] Determine completeness from the explicit source duration: a target bucket `[open, close)` is complete only when the final source candle's own close (`source_open_ms + source_duration_ms`) reaches `close`. Emit only complete buckets, omit the final incomplete bucket, and set availability exactly to target `close`.
- [ ] Add named tests `timestamp_and_mtf_contract::higher_tf_value_appears_only_at_close`, `timestamp_and_mtf_contract::future_intracandle_append_cannot_change_available_rows`, `timestamp_and_mtf_contract::incomplete_final_bucket_is_excluded`, and `timestamp_and_mtf_contract::open_time_and_available_time_remain_distinct`.

## Chunk 2: Input and calendar semantics

### Task 3: Strict CSV and JSON import validation (D03)

**Files:**
- Modify: `crates/neoethos-data/src/core/universal_importer.rs`
- Modify: `crates/neoethos-data/src/lib.rs` (`normalize_ohlcv` boundary validation)
- Create: `crates/neoethos-data/src/import_and_calendar_contract.rs`
- Modify: `crates/neoethos-data/src/lib.rs` to register the test module
- Create fixtures: `crates/neoethos-data/test_fixtures/import_invalid.csv` and `import_invalid.json`

- [ ] Re-read CSV/JSON detection, required/optional field parsers, row counters, normalization, and importer callers/tests; confirm every zero-coercion and malformed-row path before editing.
- [ ] Define a distinct `RowValidationReport { accepted_rows, rejected_rows, rejection_counts, bounded_examples }` and `ParsedOhlcv { ohlcv, row_validation }`; CSV/JSON/JSONL parsers return `ParsedOhlcv`, while Parquet/Vortex create an equivalent validation report after typed conversion. Required timestamp/OHLC parse failures reject the row. Never substitute zero.
- [ ] Extend existing `ImportFileResult` with backward-compatible defaulted `accepted_rows`, `rejected_rows`, and `rejection_counts`; keep `rows` as the accepted canonical row count. `import_one_file` propagates parser reports into it, and the existing aggregate `ImportReport` sums the per-file accepted/rejected counts without renaming or replacing that type.
- [ ] Require epoch milliseconds within `2000-01-01T00:00:00Z..=now+24h`, finite OHLC strictly greater than zero, `low <= min(open,close) <= max(open,close) <= high`, and optional volume either absent or finite/nonnegative.
- [ ] Reject blank/ragged/missing-column CSV and malformed/shape-incompatible JSON explicitly; fail the import if no valid rows remain while reporting counts and first bounded examples.
- [ ] Add named tests `import_and_calendar_contract::csv_blank_ragged_zero_and_invalid_timestamp_rows_are_counted`, `import_and_calendar_contract::json_missing_malformed_and_nonfinite_rows_are_counted`, `import_and_calendar_contract::ohlc_order_and_volume_contract_is_enforced`, and `import_and_calendar_contract::all_invalid_input_fails_with_report`.

### Task 4: UTC/timeframe-aware quantitative features (D04)

**Files:**
- Modify: `crates/neoethos-data/src/core/quant_features.rs`
- Modify: `crates/neoethos-data/src/core/session_features.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-data/src/lib.rs` feature-builder call sites
- Modify: `crates/neoethos-data/src/import_and_calendar_contract.rs`

- [ ] Re-read every quant/session feature constructor and caller plus timestamp/timeframe metadata flow; confirm all fixed 24/120-bar day/week, ORB, pivot, and annualization assumptions before editing.
- [ ] Pass `timeframe_duration_ms` and epoch-ms timestamps into the feature builders. Use UTC calendar day (`00:00..24:00`) and ISO week boundaries as the authoritative deterministic baseline; exchange-session overrides remain separate explicit inputs.
- [ ] Compute previous-day/week pivots only from completed UTC periods. Define ORB from an explicit UTC session start and configured elapsed duration, never a fixed bar count.
- [ ] Annualize bar returns with `365.2425 * 24h / timeframe_duration`, while trading-day metrics that consume market sessions use the existing explicit 252-day contract; name each basis in the API.
- [ ] Add named tests `import_and_calendar_contract::calendar_features_match_across_m1_m5_h1_d1`, `import_and_calendar_contract::pivots_use_only_completed_utc_periods`, `import_and_calendar_contract::orb_duration_is_timeframe_invariant`, and `import_and_calendar_contract::annualization_basis_is_explicit`.

## Chunk 3: Corrected deterministic math

### Task 5: Validated log-price Hurst estimator (D05)

**Files:**
- Modify: `crates/neoethos-search/src/stop_target.rs`
- Create: `crates/neoethos-search/src/corrected_math_contract.rs`
- Modify: `crates/neoethos-search/src/lib.rs` to register the test module

- [ ] Re-read the estimator, consumers, thresholds, and tests; confirm it currently differences returns rather than log prices.
- [ ] Estimate the log-log slope from standard deviation of `ln(price[t+lag]) - ln(price[t])` over deterministic valid lags; require positive finite prices, sufficient observations/lag points, finite nonzero variance, and a valid regression denominator.
- [ ] Return an explicit unavailable result for invalid/constant inputs and clamp only mathematically valid estimates to the documented range.
- [ ] Generate fixtures with fixed seeds and tolerances; add `corrected_math_contract::hurst_random_walk_is_near_half`, `hurst_persistent_process_exceeds_random_walk`, `hurst_antipersistent_process_is_below_random_walk`, and `hurst_rejects_invalid_constant_or_short_prices`.

### Task 6: Per-run adaptive threshold ladder (D06)

**Files:**
- Modify: `crates/neoethos-search/src/genetic/evolution_math.rs`
- Modify: `crates/neoethos-search/src/genetic/mod.rs` (remove global-ladder exports; export explicit-ladder helpers)
- Modify: `crates/neoethos-search/src/genetic/search_engine.rs` (`evolve_search*`, gene initialization/mutation helpers)
- Modify: `crates/neoethos-search/src/discovery.rs` (derive and pass ladder for one run)
- Modify: `crates/neoethos-search/src/corrected_math_contract.rs`

- [ ] Re-read `ADAPTIVE_THRESHOLD_LADDER`, installation/current accessors, every random gene/mutation caller, discovery derivation, and tests; confirm first-process-run leakage.
- [ ] Remove the dataset-derived `OnceLock`, `install_adaptive_threshold_ladder`, and `current_threshold_ladder` exports. Define `ThresholdLadder([f32; 6])` as an immutable value in the search-run configuration; discovery derives either the dataset ladder or the static fallback and passes it through `evolve_search*` into initialization/mutation.
- [ ] Preserve public compatibility for direct `new_random_gene`, `generate_random_genes`, and `mutate` callers as thin static-fallback wrappers. Add explicit `_with_ladder` variants used by every production search-engine initialization/mutation path; no production path reads process-global dataset state.
- [ ] Do not cache by symbol or process. Two concurrent/sequential runs own separate values; existing environment/runtime snapshots are unaffected.
- [ ] Add `corrected_math_contract::adaptive_ladder_is_owned_by_one_run`, `reversed_symbol_order_produces_identical_per_symbol_ladders`, and `concurrent_runs_cannot_observe_each_others_ladder`.

### Task 7: Seeded block bootstrap with replacement (D07)

**Files:**
- Modify: `crates/neoethos-search/src/quality.rs` (Monte Carlo block sampling)
- Modify: `crates/neoethos-search/src/genetic/runtime_overrides.rs` (canonical seed source)
- Modify: `crates/neoethos-search/src/export_state.rs` and `crates/neoethos-search/src/live_portfolio.rs` (persist seed/provenance)
- Modify: `crates/neoethos-search/src/corrected_math_contract.rs`

- [ ] Re-read daily-block construction, RNG sources, report/artifact serialization, and callers/tests; confirm shuffle-without-replacement and unseeded behavior.
- [ ] Sort daily blocks chronologically, preserve trade order within each day, sample day-block indices independently with replacement, concatenate until the original trade horizon, and trim only the final sampled block to exact length.
- [ ] Derive a ChaCha seed from the configured run seed plus stable candidate identity; persist both run seed and derivation version in quality/live artifacts. A missing seed follows the existing explicit non-deterministic policy and records the generated seed.
- [ ] Add `corrected_math_contract::block_bootstrap_same_seed_is_identical`, `different_seeds_change_sample`, `sampling_can_repeat_and_omit_days`, and `within_day_order_and_original_horizon_are_preserved`.

### Task 8: Capped allocation with explicit residual cash (D08)

**Files:**
- Modify: `crates/neoethos-search/src/portfolio.rs`
- Modify: `crates/neoethos-search/src/lib.rs` (export the corrected isolated allocation result)
- Modify: `crates/neoethos-search/src/corrected_math_contract.rs`

- [ ] Re-read allocation, normalization, workspace callers, exports, and tests; confirm post-clamp renormalization can exceed the cap and record the source proof that `PortfolioOptimizer::get_optimal_allocation` has no non-test workspace consumer.
- [ ] Keep this as a corrected isolated public API; do not wire it into discovery or live artifacts without a proven consumer. Return `PortfolioAllocation { allocations: HashMap<String, AllocationResult>, residual_cash }`. Project/water-fill nonnegative scores without lifting a capped element. Reject non-finite/negative scores and invalid caps.
- [ ] Feasible invariant: weights sum to 1 and residual is 0. Infeasible `n * max_weight < 1`: every eligible weight is at most the cap, weights sum to `n * max_weight`, and residual equals `1 - sum(weights)`.
- [ ] Update only the optimizer's tests and `neoethos-search` re-export. The explicit wrapper carries residual cash to future callers; no unrelated PnL/risk/artifact behavior is invented.
- [ ] Add `corrected_math_contract::capped_allocation_never_exceeds_cap`, `feasible_allocation_sums_to_one`, `infeasible_allocation_reports_residual_cash`, `allocation_is_nonnegative_and_permutation_equivariant`, and `invalid_inputs_are_rejected`.

## Chunk 4: Causal normalization

### Task 9: Immutable fit/apply preprocessing artifact (D09)

**Files:**
- Modify: `crates/neoethos-data/src/core/normalization.rs`
- Modify: `crates/neoethos-data/src/lib.rs` (`append_feature_block`, `normalize_block_columns`, preparation entry points)
- Modify: `crates/neoethos-search/src/discovery.rs` (outer search/train split owns fit)
- Modify: `crates/neoethos-search/src/live_portfolio.rs` (`LivePortfolioArtifact` preprocessing metadata)
- Modify: `crates/neoethos-models/src/runtime/training_artifact.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs` (fold train fit; validation/OOS apply)
- Modify: `crates/neoethos-trader/src/data_replay.rs` and `crates/neoethos-app/src/app_services/live_parity.rs` (immutable live/replay apply)
- Create: `crates/neoethos-data/src/causal_preprocessing_contract.rs`
- Create: `crates/neoethos-models/src/causal_preprocessing_contract.rs`
- Modify both crate `lib.rs` files to register the test modules

- [ ] Re-read every normalization call, feature-name projection, outer split/fold builder, artifact loader, replay/live consumer, and tests; confirm every current full-series fit before editing.
- [ ] Define versioned `PreprocessingStats { feature_names, medians, scales, clip }`; `fit_stats(train)` consumes only the supplied training rows and `apply_stats(stats, frame)` never recomputes. Reject duplicate/reordered/missing names, length drift, non-finite stats, and nonpositive scales.
- [ ] In discovery, Wave 3's outer search/train prefix fits once; validation and untouched OOS apply that frozen artifact. In model CPCV, each fold fits on its own training rows and applies to that fold's validation rows; the final promoted artifact fits on the authorized full training prefix only.
- [ ] Persist the exact ordered feature schema and stats in both live portfolio and model training artifacts. Replay and live parity load/apply them; legacy `normalize_features=true` artifacts without stats fail with a migration/retrain error rather than fitting on live data.
- [ ] Add data tests `causal_preprocessing_contract::future_append_cannot_change_historical_normalized_rows`, `fit_uses_only_supplied_training_prefix`, and `schema_reorder_or_invalid_stats_fail_closed`.
- [ ] Add model tests `causal_preprocessing_contract::each_fold_fits_only_its_training_rows`, `promoted_artifact_round_trips_frozen_stats`, and `legacy_normalized_artifact_without_stats_requires_retrain`.

## Deferred verification gate for Wave 2

- [ ] Execute nothing below until all production code, tests, fixtures, manifests, lockfiles, and documentation in Waves 1-4 are authored and reviewed.
- [ ] At focused-verification step 5 run `cargo test -p neoethos-data timestamp_and_mtf_contract -- --nocapture`, `cargo test -p neoethos-data import_and_calendar_contract -- --nocapture`, `cargo test -p neoethos-search corrected_math_contract -- --nocapture`, `cargo test -p neoethos-data causal_preprocessing_contract -- --nocapture`, and `cargo test -p neoethos-models causal_preprocessing_contract -- --nocapture` as separate commands.
- [ ] For each command require exit code 0, nonzero execution of every named contract test, no warnings/failures, and complete stdout/stderr inspection from first to last byte.
- [ ] Continue in the approved full command order; any diagnostic reopens authoring and requires affected-layer plus final-suite repetition without suppression.
