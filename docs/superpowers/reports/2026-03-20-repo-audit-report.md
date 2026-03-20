# Repo Audit Report

## Executive Summary

## Repository Census

Tracked surface:
- `tracked files`: 1080
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

Pending Chunk 1 baseline execution.

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
