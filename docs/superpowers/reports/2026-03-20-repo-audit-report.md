# Repo Audit Report

## Executive Summary

## Repository Census

Tracked surface:
- `tracked files`: 1081
- `Rust source files`: 106
- `Python files`: 6
- `TOML files`: 14
- `Markdown files`: 15
- `examples/` files: 3
- `scripts/` files: 5
- `tests/` files: 0 tracked files
- `crates/` files: 119 tracked files

## Subsystem Classification Matrix

| subsystem | status | audit treatment |
|---|---|---|
| `crates/forex-app` | runtime-critical | full static + runtime + line-by-line audit |
| `crates/forex-cli` | runtime-critical | full static + runtime + line-by-line audit |
| `crates/mt5-bridge` | runtime-critical | full static + contract + environment audit |
| `crates/forex-core` | runtime-critical | full static + line-by-line audit |
| `crates/forex-data` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-search` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-models` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-bindings` | runtime-critical | full static + Python-contract audit |
| `crates/forex-news` | audit-required | static + line-by-line audit, runtime only if reachable from active paths |
| remaining Python files | audit-required | classify as runtime code, bridge, bootstrap, or dead seam |
| top-level config/script/service assets | audit-required | audit as integration assets |
| `tests/` | evidence-only | audit for coverage gaps and stale assumptions |
| `examples/` | evidence-only | audit for coverage gaps and stale assumptions |
| `vendor/` | excluded | inventory only unless a local patch affects behavior |
| `cache/`, `logs/`, `target/`, generated artifacts | excluded | evidence only, not first-party source |

## Active-File Classification Matrix

| file class | status | notes |
|---|---|---|
| runtime-critical crates | `runtime-critical` | direct supported runtime paths |
| shim and binding files | `audit-required bridge` | can affect runtime contracts |
| helper scripts and top-level config | `static-only` | integration assets, not runtime modules |
| examples and tests | `evidence-only` | supporting verification assets |
| vendor and generated output | `excluded` | do not treat as first-party code quality surface |

## Repository Census Details

### Excluded Classes

- `vendor/`: excluded from code-quality findings unless a local patch is implicated
- `cache/`: excluded, evidence-only artifacts only
- `logs/`: excluded, evidence-only artifacts only
- `target/`: excluded build output
- generated local outputs: excluded unless they are evidence for a finding

### Workspace Members

- `crates/forex-search`
- `crates/forex-cli`
- `crates/forex-data`
- `crates/forex-models`
- `crates/forex-core`
- `crates/forex-bindings`
- `crates/forex-app`
- `crates/mt5-bridge`
- `crates/forex-news`

### Binaries And Entrypoints

- `crates/forex-cli/src/main.rs`
- `crates/forex-app/src/main.rs`
- `examples/parallel_training.rs`
- `examples/tree_models_rust_example.rs`
- `examples/true_parallel_training.rs`

### Python-Dependent Crates And Bindings

- `crates/forex-bindings` exports the first-party Python shim at `crates/forex-bindings/forex_bindings/__init__.py`
- `crates/mt5-bridge` still owns the Python-backed MT5 bridge contract
- vendor TA-Lib helper scripts remain tracked but excluded from first-party code-quality findings

### UI Crate

- `crates/forex-app` is the Rust desktop UI crate and the only active UI surface in the tracked workspace

### MT5 Bridge Crate

- `crates/mt5-bridge` is the MT5 integration seam and remains runtime-critical because it owns broker initialization and terminal-info contracts

### Remaining Python And Shim Presence

- `crates/forex-bindings/forex_bindings/__init__.py` is the only tracked first-party Python shim file
- vendor Python helpers tracked under `vendor/talib-sys/dependencies/tmp/ta-lib/swig/src/tools/test_python/`
- vendor Python interface file tracked under `vendor/talib-sys/dependencies/tmp/ta-lib/swig/src/interface/python.py`

### Test And Examples Surface

- `tests/`: 0 tracked files at the current HEAD census
- `examples/`: 3 tracked Rust examples

## Static Verification Findings

### cargo check --workspace

- Exit code: 0
- Result state: PASS WITH FINDINGS
- Warning clusters: 7
- Total warnings observed: 58

#### Warnings by subsystem

- crates/forex-core/src/domain/risk.rs: unused imports in risk module
- crates/forex-core/src/domain/portfolio.rs: unused imports and dead correlation_threshold field
- crates/forex-news/src/openai.rs, crates/forex-news/src/perplexity.rs, crates/forex-news/src/lib.rs: unused imports
- crates/forex-data/src/core/features.rs: unused imports and unused bail import
- crates/forex-search/src/validation.rs: unused imports
- crates/forex-models/src/*: many unused imports, unused variables, and dead model fields across tree/evolution/RL/statistical/anomaly modules
- crates/forex-bindings/src/*: unused imports and deprecated downcast calls
- crates/forex-app/src/main.rs: unused imports and deprecated downcast call


### cargo test --workspace

- Exit code: 1
- Result state: FAIL
- Failing test: `crates/forex-models/src/hardware.rs:377` in `hardware::tests::test_gpu_distribution`
- Root cause: `distribute_gpu_assignment()` uses zero-based modulo arithmetic, but the test contract is explicitly written as one-based model numbering (`Model 1 -> GPU 0`, `Model 2 -> GPU 1`, etc.), so implementation and test disagree on indexing semantics.
- Evidence: `cargo test --workspace` failed with `left: 1 right: 0` at `crates\forex-models\src\hardware.rs:377`
- Recommended fix direction: normalize the index before modulo or change the contract to explicit 0-based indexing and update the test accordingly
### cargo clippy --workspace --all-targets -- -D warnings

- Exit code: 1
- Result state: FAIL
- High-signal clippy errors in `crates/forex-core`:
  - `crates/forex-core/src/domain/risk.rs`: unused imports, `manual_range_contains`, `collapsible_if`, `too_many_arguments`, `manual_clamp`, and `derivable_impls`
  - `crates/forex-core/src/domain/portfolio.rs`: unused imports, dead `correlation_threshold` field, and `needless_range_loop`
  - `crates/forex-core/src/domain/drift_monitor.rs`: `manual_clamp`
  - `crates/forex-core/src/system.rs`: `new_without_default` and `needless_borrows_for_generic_args`
- High-signal clippy errors in `crates/forex-data`:
  - `crates/forex-data/src/core/features.rs`: unused imports and `derivable_impls`
  - `crates/forex-data/src/core/indicators.rs`: `needless_range_loop`
  - `crates/forex-data/src/core/resample.rs`: `collapsible_if`
  - `crates/forex-data/src/lib.rs`: `unnecessary_map_or`
- Evidence: `cargo clippy --workspace --all-targets -- -D warnings` stopped on `forex-core` after surfacing 16 errors there and additional errors in `forex-data`
- Recommended fix direction: clean the shared core/data crates first, then rerun clippy before touching higher-level layers

### cargo build -p forex-cli

- Exit code: 0
- Result state: PASS WITH FINDINGS
- Build completed successfully in `dev` profile
- Repeated warning clusters observed in `crates/forex-core`, `crates/forex-data`, `crates/forex-search`, and `crates/forex-models`
- No new build-only blocker surfaced beyond the existing warning inventory

### cargo build -p forex-app

- Exit code: 0
- Result state: PASS WITH FINDINGS
- Build completed successfully in `dev` profile
- Repeated upstream warning clusters remained visible
- App-local warnings still present in `crates/forex-app/src/main.rs` for unused `Arc` and `Mutex` imports

### Python Contract Probes

- `python -c "import sys; print(sys.version)"`: PASS, Python `3.13.9`
- `python -c "import forex_bindings"`: PASS
- `python -c "import MetaTrader5"`: PASS

### Informational Lanes

- `cargo check -p forex-models`: PASS WITH FINDINGS, repeated warning cluster in `crates/forex-models`
- `cargo check -p forex-search`: PASS WITH FINDINGS, repeated warning cluster in `crates/forex-search` plus shared warnings from `crates/forex-data`
- `baseline-linux`: N/A in the current Windows-only session; retain as review-only lane until a Linux host is available
## Runtime Findings

Pending Chunk 2 runtime execution.

## File-By-File Findings

Pending line-by-line audit.

## Contract And Operational Findings

Pending contract audit.

## Warning Inventory

Pending baseline and runtime sweeps.

## Recommended Fix Tranches

Pending stabilization backlog population.

## Findings Ledger Schema

Each JSON line in `cache/audit/2026-03-20-findings.jsonl` must include:
- `category`
- `severity`
- `lane`
- `command`
- `file`
- `line`
- `summary`
- `evidence`
- `root_cause`
- `recommended_fix`

Allowed `category` values:
- `build breakage`
- `test failure`
- `lint/warning`
- `runtime breakage`
- `correctness bug`
- `contract mismatch`
- `dead or unreachable code`
- `observability gap`
- `performance risk`
- `architectural smell`
