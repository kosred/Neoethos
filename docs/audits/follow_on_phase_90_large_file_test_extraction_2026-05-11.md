# Follow-on Phase 90 - large-file test extraction

Date: 2026-05-11

## Source gap

The Phase 89 follow-up scan found first-party Rust files above 1500 lines.
Some files are production-heavy and need architectural splits, but others had
large inline test modules that could be separated without changing runtime
behavior.

## Changes

- Moved `forex-app` trading tests out of
  `app_services/trading.rs` into `app_services/trading_tests.rs`.
- Moved the `forex-models` ensemble test module out of `ensemble.rs` into
  `ensemble_tests.rs`.
- Kept test-only helper methods that are part of the parent module's test
  support surface in place when moving them would cross production type
  boundaries.
- Updated a stale `forex-app` discovery test fixture to construct
  `DiscoveryResult` with the current validation/artifact contract fields.

## Remaining large-file targets

This phase reduced inline test weight but did not claim the production-heavy
files are solved. The next meaningful splits still need behavior-preserving
module boundaries for `swarm_impl.rs`, `dqn_impl.rs`,
`training_orchestrator.rs`, `discovery.rs`, `trading.rs`, `burn_models.rs`,
and the remaining model/search monoliths above 1500 lines.

## Verification

- `cargo test -p forex-app --bin forex-app app_services::trading -- --nocapture`
- `cargo test -p forex-app --bin forex-app app_services::discovery::tests::success_snapshot_carries_candidate_and_portfolio_counters -- --nocapture`
- `cargo test -p forex-models --lib ensemble:: -- --nocapture`
- `cargo fmt --check`
- `git diff --check`
