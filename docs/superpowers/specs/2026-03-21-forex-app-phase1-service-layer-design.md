# Forex App Phase 1 Service Layer Design

## Goal

Convert the Rust desktop app from a UI shell with mock discovery/training actions into a real operator shell that talks to backend services through a clean application-service layer.

The immediate design target is not a full trading terminal. The target is a stable Phase 1 foundation:

- real `Discovery` execution from the UI
- real `Training` execution from the UI
- explicit job states
- safe cancellation from the UI
- live progress and operator-facing reports
- canonical logging integration through `logs/forex-ai.log`

## Scope

This design covers:

- `crates/forex-app`
- the `forex-data`, `forex-search`, and `forex-models` integration seams that the app invokes
- the UI reporting contract for `Discovery` and `Training`
- canonical log access from the UI

This design does not yet cover:

- multi-broker adapters beyond the existing MT5 seam
- manual order entry / execution center
- rich charting workspace
- news/calendar aggregation
- persisted workspaces, layouts, or operator preferences
- risk dashboard implementation

## Problem Statement

The active app entrypoint still has two correctness gaps:

1. the `Discovery` tab simulates progress with a mock loop instead of calling the real backend
2. the `Training` tab logs an action but does not start a training backend flow

There is also a structural problem:

- `crates/forex-app/src/main.rs` owns UI, runtime bootstrap, MT5 actions, discovery placeholders, training placeholders, and state management in one place

This makes future work harder:

- progress reporting is ad hoc
- cancellation has no real contract
- operator reporting is inconsistent
- UI code is tightly coupled to backend execution details

## Design Summary

Phase 1 introduces an application-service layer inside `crates/forex-app`.

The UI will stop invoking backend logic directly and will instead talk to service objects that own:

- request validation
- job lifecycle
- progress updates
- report snapshots
- cancellation checkpoints
- canonical log updates

The UI becomes a thin operator shell over typed job and report state.

## Architectural Approach

Recommended approach:

- keep `forex-app` as the UI crate
- add focused application-service modules inside the crate
- keep actual backend execution in existing runtime crates:
  - `forex-data`
  - `forex-search`
  - `forex-models`
  - `mt5-bridge`
- expose app-owned service contracts that translate backend behavior into operator-safe UI state

This gives Phase 1 a clean boundary without prematurely creating a new top-level workspace crate.

## Runtime Model

The app will use typed jobs rather than ad hoc spawned tasks.

Core job concepts:

- `JobId`
- `JobKind`
  - `Discovery`
  - `Training`
- `JobState`
  - `Queued`
  - `Running`
  - `Succeeded`
  - `Degraded`
  - `Failed`
  - `Cancelled`
- `JobProgress`
  - `percent`
  - `stage`
  - `message`
- `JobReport`
  - warnings
  - errors
  - counters
  - partial results
  - final result summary
  - canonical log path

The app does not need distributed scheduling in Phase 1. A simple per-job task model is enough as long as the contracts are explicit.

## Cancellation Contract

The user explicitly requires stopping a running task from the UI.

Phase 1 will implement safe cancellation, not forced termination.

Cancellation behavior:

- each job gets a cancel flag or equivalent cancellation token
- long-running discovery/training work must check cancellation at safe checkpoints
- if cancellation is requested:
  - the job stops cleanly
  - the final state becomes `Cancelled`
  - the canonical log records `Cancelled`
  - the UI report reflects the cancellation reason

This avoids abrupt task termination that could leave inconsistent outputs or half-written artifacts.

## Operator Reporting Contract

Status alone is not enough.

Every active job must expose three operator-facing layers:

1. `Status`
   - current state only
2. `Progress`
   - stage and percent
3. `Report`
   - warnings
   - errors
   - counters
   - partial results
   - final summary

### Discovery Report Requirements

The `Discovery` report must support:

- how many strategies were evaluated
- how many were rejected
- how many passed validation
- how many entered the portfolio
- best candidates so far
- complete metrics for successful strategies
- explicit degraded or failed states when dataset, timeframe, or validation constraints block progress

Phase 1 does not need every future metric, but it must support structured extension.

### Training Report Requirements

The `Training` report must support:

- which model is active
- which models completed
- which models failed
- model-level warnings
- model-level errors
- final training summary

If the current Rust training backend cannot deliver fine-grained progress for every model yet, the service layer must still expose coarse stages honestly rather than inventing fake percentages.

## UI Contract

Phase 1 focuses on the `Discovery` and `Training` tabs.

Each tab should show:

- current status badge
- progress bar and stage text
- live warnings/errors panel
- report summary area
- result area
- `Start`
- `Stop`
- `Open Log`

The UI should not force the operator to inspect the raw log to understand current behavior. The log remains a forensic tool, not the main operator surface.

## Canonical Log Integration

The active logging contract remains:

- one canonical file at `logs/forex-ai.log`
- subsystem-scoped sections
- `current` and `previous` only

Phase 1 extends the app usage of that contract:

- `APP` for UI lifecycle and top-level application actions
- `DISCOVERY` for discovery jobs
- `TRAINING` for training jobs

The `Open Log` UI action opens:

- `logs/forex-ai.log`

No new ad hoc UI-specific log files are allowed on the supported path.

## File Structure Recommendation

The current `main.rs` should be decomposed aggressively into smaller focused files.

Recommended structure:

- `crates/forex-app/src/main.rs`
  - runtime bootstrap only
  - args, settings load, shared logging init, app construction

- `crates/forex-app/src/app_state.rs`
  - UI state container
  - active tab
  - selected symbol
  - current job/report snapshots
  - no backend execution logic

- `crates/forex-app/src/app_services/mod.rs`
  - public service facade

- `crates/forex-app/src/app_services/jobs.rs`
  - `JobId`
  - `JobKind`
  - `JobState`
  - `JobProgress`
  - `JobReport`
  - cancellation token/flag support

- `crates/forex-app/src/app_services/discovery.rs`
  - discovery service
  - backend call orchestration
  - progress and report updates

- `crates/forex-app/src/app_services/training.rs`
  - training service
  - backend call orchestration
  - progress and report updates

- `crates/forex-app/src/ui/mod.rs`
  - UI module exports

- `crates/forex-app/src/ui/components.rs`
  - shared widgets and helpers

- `crates/forex-app/src/ui/trading.rs`
  - MT5/local trading tab rendering only

- `crates/forex-app/src/ui/discovery.rs`
  - discovery tab rendering only

- `crates/forex-app/src/ui/training.rs`
  - training tab rendering only

Additional file splits are encouraged if a file starts to grow large. The design goal is small focused files, not a new monolith broken into nominal modules.

## Service Execution Strategy

Phase 1 should use asynchronous task execution already available in the app through Tokio.

Service rules:

- the UI requests a job start
- the service validates the request
- the service spawns the backend task
- the service publishes progress/report snapshots back to app state
- the UI reads snapshots and renders them

The UI must not own backend task internals.

## Error Handling Rules

- backend failures must map to explicit `Failed` or `Degraded`, never silent success
- cancellation must map to `Cancelled`, not generic failure
- validation errors must surface before the job starts
- unsupported backend behavior must be reported honestly
- no mock progress on the supported path

If a backend cannot provide a metric yet, the UI should say that the detail is unavailable rather than inventing it.

## Enterprise Constraints

Phase 1 should preserve the current enterprise direction:

- no warnings on the supported verification path
- no silent fallbacks where correctness matters
- no fake success paths
- clear operator visibility
- canonical logging only
- thin UI over explicit services

## Verification Strategy

Required verification lanes for the implementation tranche:

1. `cargo test -p forex-app`
   - new service and UI helper tests
2. `cargo clippy -p forex-app --all-targets -- -D warnings`
3. `cargo test --workspace`
4. `cargo clippy --workspace --all-targets -- -D warnings`
5. `cargo run -p forex-app -- --headless --config config.yaml`
6. `cargo run -p forex-app -- --headless --local --config config.yaml`
7. inspect `logs/forex-ai.log`
   - `APP`, `DISCOVERY`, and `TRAINING` update correctly
   - cancellation surfaces as `Cancelled`

## Acceptance Criteria

Phase 1 is complete when:

- the `Discovery` tab no longer uses the mock progress loop
- the `Training` tab no longer performs log-only placeholder actions
- the app uses a real application-service layer
- the UI can start and stop discovery jobs
- the UI can start and stop training jobs
- the UI shows real status, progress, warnings/errors, and report snapshots
- `Open Log` opens the canonical log file
- the app crate and workspace remain warning-clean on the supported lane

## Out Of Scope For This Tranche

These remain explicit future phases:

- multi-broker adapter expansion
- rich chart workspace
- manual trading order ticket
- news/calendar aggregation
- risk dashboard
- persisted workspaces and layouts

This keeps Phase 1 focused on turning the app from a mock shell into a real operator shell for discovery and training.
