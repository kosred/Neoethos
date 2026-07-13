# Audit Remediation Wave 3 Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development`. Author regression tests with every correction, but execute no verification until production code, tests, fixtures, manifests, lockfiles, and documentation for all four waves are authored and reviewed.

**Goal:** Restore truthful backtest, validation, model artifact, replay, checkpoint, and Codex behavior.

**Architecture:** One chronological split contract owns search/train, validation, and untouched OOS boundaries. Evaluation resets state at every discontinuity. Model folds carry row identity and complete coverage. Replay/live share one decision payload. Operational persistence records explicit stage outcomes. Authentication and streaming failures remain structured and secret-safe.

---

## Chunk 1: Backtest and validation truthfulness

### Task 1: Terminal liquidation and elapsed-horizon metrics (B01, B06)

**Files:**
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-search/src/quality.rs`
- Create: `crates/neoethos-search/src/validation_and_terminal_position_contract.rs`
- Modify: `crates/neoethos-search/src/lib.rs` to register the test module

- [ ] Re-read all entry/exit branches, trade-ledger construction, cost/carry fields, metric callers, and tests; confirm the missing final position and active-day Sharpe defects before editing.
- [ ] Extract one exit routine used by stops/signals/end-of-island/end-of-series. At the final finite close, force-liquidate an open position through identical spread, commission, conversion fee, carry, slippage, trade count, MFE/MAE, and ledger fields.
- [ ] Derive annualization from first-to-last finite timestamp elapsed calendar span; return unavailable/zero per the existing metric contract for fewer than two valid timestamps, never infer duration from active trade days.
- [ ] Add named tests `validation_and_terminal_position_contract::final_open_long_and_short_are_liquidated_once`, `terminal_liquidation_reconciles_pnl_costs_and_ledger`, `already_closed_position_is_not_double_counted`, and `sparse_year_sharpe_uses_full_elapsed_horizon`.

### Task 2: Outer chronological split and frozen candidates (B02)

**Files:**
- Modify: `crates/neoethos-search/src/discovery.rs` (`run_discovery`, candidate freeze/evaluation stages)
- Modify: `crates/neoethos-search/src/validation.rs` (`Walkforward*`, `ForwardTest*`, split helpers)
- Modify: `crates/neoethos-search/src/checkpoint.rs` (`SearchCheckpointScope`, `PortfolioSelectionArtifactFile`)
- Modify: `crates/neoethos-search/src/export_state.rs`
- Modify: `crates/neoethos-core/src/config.rs` (`ModelSettings` outer-split fields)
- Modify: `config.yaml`
- Modify: `crates/neoethos-search/src/validation_and_terminal_position_contract.rs`

- [ ] Re-read search, survivor selection, walk-forward/forward-test producers, checkpoint resume hashes, artifact scopes, and tests; confirm full-series candidate selection before editing.
- [ ] Add bounded settings `outer_search_pct=0.70`, `outer_validation_pct=0.15`, `outer_purge_pct=0.02`, and `outer_embargo_pct=0.01`; OOS receives the remaining usable rows. Require search+validation percentages in `(0,1)` with sum `<1`, and purge/embargo in `[0,0.10]`.
- [ ] Define `OuterSplit { search_train, validation, oos, purge, embargo }` over ordered row ranges. For `n` rows compute `purge=ceil(n*resolved.outer_purge_pct)`, `embargo=ceil(n*resolved.outer_embargo_pct)`, `usable=n-purge-embargo`, then allocate floor(`usable*resolved.outer_search_pct`) search rows, floor(`usable*resolved.outer_validation_pct`) validation rows, and the remainder to OOS in the exact order `search | purge | validation | embargo | oos`. The 0.70/0.15 values are defaults only. Require at least 256/80/80 search/validation/OOS rows after gaps or return a structured insufficient-history error.
- [ ] Freeze candidate signatures after search/train. Validation chooses only among the frozen set using the lexicographic tuple: finite validation Sharpe descending, finite profit factor descending, finite max drawdown ascending, canonical gene-signature hash ascending. Untouched OOS reports the already chosen candidate and cannot select or mutate it.
- [ ] Add search/train, validation, OOS, purge, and embargo hashes/ranges to `SearchCheckpointScope`, `PortfolioSelectionArtifactFile`, `WalkforwardValidationScope`, and `ForwardTestValidationScope`; legacy checkpoints without them are non-resumable with an explicit rerun message.
- [ ] Add `validation_and_terminal_position_contract::validation_mutation_cannot_change_search_candidates`, `oos_mutation_cannot_change_selected_candidate`, `purge_and_embargo_rows_are_unused`, and `segment_hashes_round_trip_and_prevent_resume_drift`.

### Task 3: CPCV contiguous islands reset state (B02)

**Files:**
- Modify: `crates/neoethos-search/src/validation.rs` (`CombinatorialPurgedCV` output)
- Modify: `crates/neoethos-search/src/discovery.rs` (CPCV evaluator)
- Modify: `crates/neoethos-search/src/validation_and_terminal_position_contract.rs`

- [ ] Re-read CPCV index construction and every consumer; confirm disjoint test indices are currently concatenated into one artificial series.
- [ ] Return ordered contiguous test ranges, not one flattened vector. For each island initialize cash/position/carry state independently and force-liquidate at that island's final close before aggregation.
- [ ] Aggregate island metrics only after evaluation; never carry positions, time deltas, or costs across gaps.
- [ ] Add `validation_and_terminal_position_contract::cpcv_gap_creates_two_independent_islands`, `position_is_liquidated_and_reset_at_each_island`, and `gap_rows_cannot_affect_island_metrics`.

### Task 4: Reject non-finite promotion evidence (B07)

**Files:**
- Modify: `crates/neoethos-core/src/domain/promotion_gate.rs`
- Create: `crates/neoethos-core/src/promotion_nonfinite_contract.rs`
- Modify: `crates/neoethos-core/src/lib.rs` to register the test module

- [ ] Re-read all promotion metric constructors/aggregators and tests; confirm NaN drawdown can collapse to a permissive value.
- [ ] Validate every strategy metric and aggregate before comparison; reject NaN, positive infinity, and negative infinity with the field/strategy identity preserved.
- [ ] Add `promotion_nonfinite_contract::nan_posinf_and_neginf_fail_every_metric_field` and `nonfinite_member_cannot_be_hidden_by_portfolio_aggregation`.

## Chunk 2: Model selection and artifacts

### Task 5: Genetic train/validation/OOS separation (B03)

**Files:**
- Modify: `crates/neoethos-models/src/genetic.rs`
- Create: `crates/neoethos-models/src/selection_and_hpo_contract.rs`
- Modify: `crates/neoethos-models/src/lib.rs` to register the test module

- [ ] Re-read genetic split/evolution/parent selection/final reporting and tests; confirm validation-tail reuse each generation.
- [ ] Consume Task 2's already-resolved `OuterSplit` ranges; do not recompute or hardcode percentages in the genetic module. Evolve and select parents only on its search/train rows. Use validation only to choose among frozen generation/model snapshots with the same finite Sharpe/PF/drawdown/signature lexicographic tuple; evaluate purged OOS once after selection.
- [ ] Add `selection_and_hpo_contract::genetic_oos_mutation_cannot_change_population_or_selection`, `validation_is_not_used_for_parent_fitness`, and `purged_oos_is_evaluated_once_after_freeze`.

### Task 6: Preserve row identity through regime/budget filtering (B08)

**Files:**
- Create: `crates/neoethos-models/src/training_rows.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs` (`derive_regime_buckets`, sampled-frame construction)
- Modify: `crates/neoethos-models/src/lib.rs` to declare the crate-private `training_rows` module
- Modify: `crates/neoethos-models/src/selection_and_hpo_contract.rs`

- [ ] Re-read sample slicing, feature budgets, `derive_regime_buckets`, raw OHLCV access, and tests; confirm relative filtered indices are applied to original rows.
- [ ] Carry `TrainingRowId { original_index, timestamp_ms }` beside every sampled/filtered feature row. Join regime labels by exact row ID and reject duplicates, missing identities, timestamp mismatch, or out-of-range indices.
- [ ] Add `selection_and_hpo_contract::sparse_shifted_rows_keep_original_regime_identity` and `missing_duplicate_or_mismatched_row_identity_fails`.

### Task 7: Complete model CPCV coverage and aggregation (B09)

**Files:**
- Create: `crates/neoethos-models/src/cpcv_selection.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs` (`optimize_model_config_cpcv` delegates to module)
- Modify: `crates/neoethos-models/src/runtime/hpo.rs` (`OptimizationTrialRecord` coverage fields)
- Modify: `crates/neoethos-models/src/lib.rs` to declare the crate-private `cpcv_selection` module
- Modify: `crates/neoethos-models/src/selection_and_hpo_contract.rs`

- [ ] Re-read CPCV split planning, per-fold train/predict/metric errors, chosen-trial persistence, and tests; confirm partial folds and first-fold representation.
- [ ] Record planned/completed fold counts and each failure. Reject the candidate on the first or later failed fold while retaining the original error; only candidates with all planned folds participate in selection.
- [ ] Aggregate score-like metrics by arithmetic mean and loss/drawdown/error metrics by worst case; persist all-fold provenance, never one representative fold.
- [ ] Add `selection_and_hpo_contract::later_fold_failure_rejects_candidate`, `partial_coverage_cannot_win`, and `all_fold_mean_scores_and_worst_losses_are_persisted`.

### Task 8: Probability simplex and no-HPO report (B10)

**Files:**
- Modify: `crates/neoethos-models/src/runtime/hpo.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs` (small-data/HPO branch)
- Modify: `crates/neoethos-models/src/selection_and_hpo_contract.rs`

- [ ] Re-read every probability producer/metric caller and small-data report writer; confirm inconsistent normalization and invalid empty selected report.
- [ ] Validate each row has the expected class count, finite nonnegative entries, and positive finite sum; normalize exactly once at the metric boundary and reject invalid rows.
- [ ] Represent skipped HPO as `None`, or persist one fully valid base trial if the artifact schema requires a report; never select from an empty trial list.
- [ ] Add `selection_and_hpo_contract::nan_inf_negative_zero_sum_and_wrong_width_probabilities_fail`, `valid_rows_normalize_once`, and `small_dataset_persists_none_or_valid_base_trial`.

### Task 9: DQN artifact integrity and FP32 policy (B11)

**Files:**
- Modify: `crates/neoethos-models/src/rl/dqn_impl.rs`
- Modify: `crates/neoethos-models/src/rl/dqn_impl_tests.rs`
- Create: `crates/neoethos-models/src/artifact_and_inventory_contract.rs`
- Modify: `crates/neoethos-models/src/lib.rs` to register the test module

- [ ] Re-read default/no-default save/load branches, snapshot claims, precision selection, and failing precision tests; confirm the integrity/policy drift.
- [ ] Apply the same schema/version/hash/network-shape snapshot validation in feature-enabled and no-feature builds. Declare FP32 as the single supported persisted/inference precision and make errors/tests match it.
- [ ] Add `artifact_and_inventory_contract::default_and_no_default_dqn_validate_same_snapshot_claims`, `tampered_snapshot_fails_in_every_feature_branch`, and `dqn_precision_contract_is_fp32`.

### Task 10: Swarm history and degradation reason (B12)

**Files:**
- Modify: `crates/neoethos-models/src/ensemble_inference/bootstrap.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/swarm_adapter.rs`
- Modify: `crates/neoethos-models/src/forecasting/swarm_impl.rs`
- Modify: `crates/neoethos-models/src/forecasting/swarm_impl_tests.rs`
- Modify: `crates/neoethos-models/src/artifact_and_inventory_contract.rs`

- [ ] Re-read bootstrap adapter construction, required history window, last-row vote, and degradation propagation; confirm one-row input and reason overwrite.
- [ ] Pass at least the model-declared history window to swarm inference, then select the last-row vote. Preserve the first specific degradation cause through bootstrap/reporting instead of replacing it with a generic fallback.
- [ ] Add `artifact_and_inventory_contract::loaded_swarm_with_history_emits_non_neutral_last_vote`, `insufficient_history_fails_or_degrades_specifically`, and `specific_swarm_degradation_reason_round_trips`.

### Task 11: Exit-agent inventory truthfulness (B13)

**Files:**
- Modify: `crates/neoethos-models/src/training_orchestrator.rs` (`create_dispatch_plan`, model config mapping)
- Modify: `crates/neoethos-models/src/runtime/capabilities.rs`
- Modify: `crates/neoethos-models/src/artifact_and_inventory_contract.rs`

- [ ] Re-read every implicit/explicit `exit_agent` request, registry/bootstrap consumer, and tests; confirm no production voter consumes its automatic artifact.
- [ ] Remove all automatic insertion/default training of `exit_agent`. Retain a truthful explicit operator request and capability/dispatch path, but do not register it as an ensemble voter.
- [ ] Add `artifact_and_inventory_contract::default_inventory_has_no_exit_agent`, `explicit_exit_agent_request_trains_truthfully`, and `exit_agent_is_not_registered_as_voter`.

## Chunk 3: Trader and operational parity

### Task 12: Replay/live signal, sizing, bracket, and cost parity (B04)

**Files:**
- Modify: `crates/neoethos-trader/src/contracts.rs`
- Modify: `crates/neoethos-trader/src/signal.rs`
- Modify: `crates/neoethos-trader/src/decision.rs`
- Modify: `crates/neoethos-trader/src/gene_signal.rs`
- Modify: `crates/neoethos-trader/src/blend_signal.rs`
- Modify: `crates/neoethos-trader/src/data_replay.rs`
- Modify: `crates/neoethos-trader/src/engine.rs`
- Modify: `crates/neoethos-app/src/app_services/live_trading.rs`
- Create: `crates/neoethos-trader/src/replay_and_position_contract.rs`
- Modify: `crates/neoethos-trader/src/lib.rs` to register the test module

- [ ] Re-read gene combination, replay defaults, decision sizing, live adapter, pip metadata, cost profile, and tests; confirm every dropped/divergent field before editing.
- [ ] Define one `CombinedSignal` carrying direction, finite confidence, native optional SL/TP distances, sizing inputs/result, pip-size conversion provenance, and spread/commission/slippage/carry metadata.
- [ ] Count SL and TP contributors independently only when finite/positive. Replay and live consume the same combined structure and decision math; no adapter may reconstruct default confidence/brackets/costs.
- [ ] Add `replay_and_position_contract::replay_and_live_match_direction_confidence_native_brackets`, `sizing_and_pip_conversion_match`, `cost_metadata_matches`, and `invalid_bracket_does_not_discard_other_valid_bracket`.

### Task 13: Conservative position mutation (B05)

**Files:**
- Modify: `crates/neoethos-trader/src/position.rs`
- Modify: `crates/neoethos-trader/src/replay_and_position_contract.rs`

- [ ] Re-read open/stop/partial-close mutation order and tests; confirm optimistic gap fills and pre-validation mutation.
- [ ] For a long stop gapped below, fill at bar open minus adverse slippage; for a short stop gapped above, fill at bar open plus adverse slippage. Non-gap stops retain the existing trigger/fill contract.
- [ ] Validate before mutation: finite positive fill, finite positive close volume, close volume no greater than remaining, non-flat side, nonempty unique ID, and finite price/size. On rejection state is byte-equivalent to before.
- [ ] Add cases for negative/zero/NaN/infinite fill; zero/negative/NaN/infinite/oversized close volume; flat opens; empty/duplicate IDs; both gap directions; and explicit no-mutation assertions.

### Task 14: Truthful auto-loop checkpoints (B14)

**Files:**
- Create: `crates/neoethos-cli/src/auto_loop_checkpoint.rs`
- Modify: `crates/neoethos-cli/src/main.rs` (`cmd_auto_loop` delegates state transitions)
- Modify: `crates/neoethos-cli/Cargo.toml` only if the Wave 1 atomic persistence crate link is missing

- [ ] Re-read the full auto-loop discovery/training result handling, resume parsing, direct writes, and tests; confirm unconditional completion.
- [ ] Define top-level `AutoLoopCheckpointV2 { schema_version: 2, started_at, updated_at, skip_training, work_units: BTreeMap<WorkUnitKey, WorkUnitCheckpoint> }`. Each `WorkUnitCheckpoint` carries symbol/timeframe plus discovery/training `StageOutcome`; outcomes are `Pending`, `Succeeded { at }`, `RetryableFailure { at, error }`, or `NotRequired { reason }`.
- [ ] Legacy `AutoLoopCheckpoint { completed, ... }` has no truthful stage/skip evidence because the old loop appended pairs after failures. Migrate every legacy pair to both stages `Pending` with a `legacy_unverified` audit note, so resume retries it; never infer success. New v2 runs set training `NotRequired` only from the current persisted `skip_training=true` flag.
- [ ] Mark a stage succeeded only after its call returns success. Persist each transition through the Wave 1 locked atomic primitive; resume skips only succeeded required stages and retries failures/pending stages.
- [ ] Add module tests named `auto_loop_checkpoint_contract::legacy_checkpoint_migrates_conservatively`, `failed_discovery_is_retryable_and_training_not_started`, `failed_training_retries_without_repeating_successful_discovery`, and `successful_resume_skips_only_completed_stages`.

### Task 15: Codex auth freshness, secrecy, SSE, and callback deadline (B15)

**Files:**
- Modify: `crates/neoethos-codex/src/auth_store.rs`
- Modify: `crates/neoethos-codex/src/client.rs`
- Modify: `crates/neoethos-codex/src/oauth.rs`
- Modify: `crates/neoethos-codex/src/callback.rs`
- Modify: `crates/neoethos-codex/Cargo.toml` to depend on `neoethos-core` for the Wave 1 atomic primitive
- Create: `crates/neoethos-codex/src/auth_stream_callback_contract.rs`
- Modify: `crates/neoethos-codex/src/lib.rs` to register the test module

- [ ] Re-read auth schema/load/save/debug, all request/401 branches, SSE parser terminal states, callback accept/read loop, and tests; confirm each independent defect before editing.
- [ ] Preserve expiry and unknown modern auth fields with `flatten`; persist refresh results through the Wave 1 atomic primitive. Implement redacted `Debug` for every token-bearing type and recursively redact known/raw secret keys in extras.
- [ ] On the first 401 acquire one async refresh mutex, re-check whether another caller already refreshed, refresh/persist once, and retry the request exactly once. A second 401 is terminal; concurrent requests cannot stampede.
- [ ] Treat explicit SSE error events, malformed JSON, stream truncation before terminal completion, and missing terminal output as structured errors while retaining bounded non-secret context.
- [ ] Apply one overall deadline covering callback accept, header read, and body read; require the exact loopback path and method before consuming the authorization code.
- [ ] Add named tests `auth_stream_callback_contract::unknown_fields_and_expiry_round_trip`, `debug_and_nested_extras_never_expose_secrets`, `concurrent_401s_refresh_once_and_retry_once`, `second_401_is_terminal`, `failed_malformed_and_truncated_sse_are_errors`, `wrong_method_or_path_is_rejected`, and `slow_headers_or_body_hit_one_overall_deadline`.

## Deferred verification gate for Wave 3

- [ ] Execute nothing below until all Waves 1-4 authoring and reviews are complete.
- [ ] At focused step 5 run the exact design commands separately for `validation_and_terminal_position_contract`, `promotion_nonfinite_contract`, `selection_and_hpo_contract`, `artifact_and_inventory_contract` (default and no-default), `replay_and_position_contract`, `auto_loop_checkpoint_contract`, and `auth_stream_callback_contract`.
- [ ] Require exit code 0, nonzero execution of every named test, no warnings/failures, and complete stdout/stderr inspection for each command.
- [ ] Continue with the approved full order. Any diagnostic reopens authoring; fix without suppression and repeat the affected layer plus final full suite.
