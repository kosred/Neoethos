# Canonical Sectioned Logging Design

## Goal

Define one enterprise-safe logging contract for the bot so operators can inspect one stable file in one stable location, without timestamped log sprawl and without losing context from unrelated subsystems.

The design target is a single canonical file:

- `logs/forex-ai.log`

The file must be readable by humans during debugging and safe for automated updating by multiple runtime entrypoints.

## Scope

This design covers:

- the shared logging contract in `crates/forex-core`
- first-adopter runtime integration in:
  - `crates/forex-app`
  - `crates/forex-cli`
  - `crates/mt5-bridge`

This design does not yet cover full adoption by every runtime crate. Later tranches can migrate more subsystems onto the same contract after the core behavior is verified.

## Problem Statement

The current runtime has two problems:

1. ad-hoc logging initialization and output paths can diverge between crates
2. operational debugging becomes noisy when logs are split across multiple files or per-run timestamped outputs

The user requirement is explicit:

- one stable file
- no log-file explosion
- no forced deletion of unrelated subsystem evidence
- each subsystem should keep only the latest two runs so operators can compare current vs previous behavior

## Design Summary

The bot will use one canonical log file containing stable sections, similar to chapters in a book.

Each section corresponds to one subsystem and stores:

- `current`
- `previous`

When a new run for a section arrives:

- `current` becomes `previous`
- the incoming run becomes `current`
- anything older is discarded

No other section is modified.

## Canonical File Layout

The canonical file path is:

- `logs/forex-ai.log`

The canonical file is plain text with structured section boundaries. The top-level order is fixed:

- `SYSTEM`
- `APP`
- `CLI`
- `DISCOVERY`
- `TRAINING`
- `MT5`
- `BINDINGS`

The file format is sectioned text, not arbitrary append-only logs. Each section contains metadata and two retained entries.

Example shape:

```text
===== SECTION SYSTEM =====
updated_at=2026-03-21T14:30:00Z

--- CURRENT ---
run_id=...
operation=...
status=...
message=...
body:
...

--- PREVIOUS ---
run_id=...
operation=...
status=...
message=...
body:
...

===== SECTION APP =====
...
```

The exact formatting can be refined during implementation, but the critical contract is:

- deterministic section boundaries
- deterministic `current` / `previous` slots
- one canonical file only

## Subsystem Model

Each log write targets exactly one subsystem section.

Initial section set:

- `SYSTEM`
  - logging setup, file-repair, recovery notices, shared runtime environment findings
- `APP`
  - GUI and headless app lifecycle events
- `CLI`
  - top-level CLI command lifecycle
- `DISCOVERY`
  - strategy discovery runs and errors
- `TRAINING`
  - training runs and errors
- `MT5`
  - bridge initialization, terminal info, broker failures
- `BINDINGS`
  - binding-specific failures that are not better attributed elsewhere

If a later subsystem needs its own section, it can be added, but the default should be to reuse one of the above instead of creating unbounded categories.

## Record Contract

Each retained run entry should carry stable fields:

- `run_id`
- `parent_run_id` when applicable
- `started_at`
- `finished_at`
- `subsystem`
- `operation`
- `status`
- `symbol`
- `timeframe`
- `error_code`
- `message`
- `body`

Field semantics:

- `status` is explicit and finite: `STARTED`, `SUCCESS`, `DEGRADED`, `FAILED`
- `message` is one-line operator summary
- `body` is multiline detail for quick debugging

No silent downgrade from `FAILED` to `SUCCESS`.

## Update Algorithm

The canonical file must be updated atomically and only for the relevant section.

High-level algorithm:

1. open canonical file path
2. acquire process-safe file lock
3. parse existing sectioned file if present
4. if file missing, initialize all required sections
5. replace only the target section:
   - `previous <- current`
   - `current <- new entry`
6. write the full canonical file to a temporary file
7. atomically replace the old file with the new file
8. release lock

This avoids:

- cross-process truncation
- interleaved writes
- accidental deletion of unrelated subsystem state

## Malformed File Recovery

Malformed file state must not be hidden.

If the canonical file cannot be parsed:

- the runtime must rebuild a valid canonical file
- the `SYSTEM` section `current` entry must explicitly record:
  - that recovery happened
  - why parsing failed
  - which subsystem triggered recovery

This preserves operator visibility instead of silently resetting evidence.

## Logging Initialization Contract

`crates/forex-core` remains the owner of logging initialization.

Requirements:

- no crate-local `tracing_subscriber::fmt::init()` outside the shared core logging setup on supported runtime paths
- console logging remains available
- the existing `WorkerGuard` flush guarantee remains retained
- runtime entrypoints emit structured section updates through the shared sectioned-log writer

This means `crates/forex-app/src/main.rs` must stop bypassing the shared contract.

## First-Adopter Integration

First integration scope:

- `crates/forex-core`
  - add sectioned-log model and atomic writer
- `crates/forex-app`
  - initialize shared logging instead of local `tracing_subscriber` init
  - log app/headless lifecycle into `APP`
- `crates/forex-cli`
  - log command lifecycle into `CLI`
  - route training/discovery command summaries into the matching subsystem sections
- `crates/mt5-bridge`
  - write MT5 initialization/result records into `MT5`

This keeps the first tranche small enough to verify without touching every crate simultaneously.

## File Structure Recommendation

To keep responsibilities clear, the core implementation should be split instead of forcing everything into `logging.rs`.

Recommended ownership:

- `crates/forex-core/src/logging.rs`
  - subscriber setup
  - console/file tracing setup
  - bridge to the sectioned log writer
- `crates/forex-core/src/sectioned_log.rs`
  - canonical file format
  - record/section types
  - parser/renderer
  - atomic section replacement
  - file locking
- `crates/forex-core/src/lib.rs`
  - export the new module

This keeps the logging backend readable and testable.

## Error Handling Rules

- failure to update the canonical sectioned log must never be silent
- the runtime may continue if console logging is still available
- the failure must still be emitted clearly to console/stderr
- subsystem writes must not erase unrelated sections
- no timestamped fallback file creation on the supported path

## Verification Strategy

Required verification for the implementation tranche:

1. unit tests in `forex-core` for:
   - create file from empty state
   - update one section without mutating others
   - rotate `current -> previous`
   - discard older-than-previous history
   - recover malformed file into valid canonical form

2. runtime verification:
   - `cargo run -p forex-cli -- load --symbol EURUSD --timeframe M1 --root data`
   - `cargo run -p forex-cli -- discover ...`
   - `cargo run -p forex-app -- --headless --config config.yaml`

3. full gates:
   - `cargo test --workspace`
   - `cargo clippy --workspace --all-targets -- -D warnings`

## Non-Goals

This tranche does not introduce:

- a database-backed audit ledger
- indefinite history retention
- per-run separate log files
- UI log viewers
- cross-machine log aggregation

Those can be layered later if needed, but they are not required to solve the current operational problem.
