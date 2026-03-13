# Rust-First No-Fallback Runtime Design

## Goal

Converge the bot onto one Rust-first runtime with no Python fallback codepaths. The runtime must behave the same on Windows and Linux, with bootstrap scripts installing the required toolchains and native dependencies per machine instead of keeping two logic stacks alive.

## Design Decisions

### 1. One runtime, no fallback

- Rust bindings and Rust-native crates become mandatory for offline training, discovery, feature engineering, and live/runtime data handling.
- Python fallback selection is removed from registries, trainers, and runtime switches.
- Python remains only where the implementation is still temporary and on the active migration path. Those modules are treated as debt, not as supported fallback backends.

### 2. Two in-memory data shapes only

The runtime keeps only two primary data contracts:

- Columnar/table contract: Arrow-compatible columnar data, implemented in Rust through Polars where table semantics are still useful.
- Dense numeric contract: `ndarray` plus memmap-backed arrays for training and inference hot paths.

This avoids pandas object churn, duplicated copies, and GIL-heavy Python loops.

### 3. Best-fit replacement for pandas by subsystem

There is no single universal replacement for pandas in this repo.

- Use `ndarray` and memmap for model matrices, pooled datasets, feature matrices, and inference tensors.
- Use Arrow/Polars for parquet IO, lazy scans, joins, grouped transforms, and columnar exchange between crates.
- Use `tch`/LibTorch for deep-model training once a deep family leaves Python.

This matches the current Rust workspace better than a "Polars everywhere" rewrite and is safer than betting the critical deep-model path on a young pure-Rust training stack.

## Research Summary

The recommended stack is based on the official documentation for:

- Polars migration guidance from pandas: `https://docs.pola.rs/user-guide/migration/pandas/`
- Apache Arrow columnar format and C data interface:
  - `https://arrow.apache.org/docs/format/Columnar.html`
  - `https://arrow.apache.org/docs/format/CDataInterface.html`
- NumPy memmap for bounded-memory array access:
  - `https://numpy.org/doc/stable/reference/generated/numpy.memmap.html`
- PyTorch C++ frontend / LibTorch for non-Python deep-model execution:
  - `https://docs.pytorch.org/cppdocs/`

## Current Repository Fit

The repository already aligns with this direction:

- `crates/forex-data` already uses `polars` and `ndarray`
- `crates/forex-models` already uses `ndarray`, `polars`, and optional `tch`
- `crates/forex-bindings` already bridges Rust crates into Python with PyO3

So the migration is primarily about removing fallback seams and normalizing data contracts, not introducing an entirely new stack.

## Architecture

### Runtime ownership

- Rust owns:
  - history loading
  - MT5/live frame normalization
  - resampling
  - feature engineering
  - label generation
  - strategy search and backtests
  - pooled dataset assembly
  - tree-model training/inference
  - deep-model orchestration as each family is ported
- Python is reduced incrementally and ultimately removed from runtime-critical paths.

### Live/runtime data flow

- MT5/live payloads are normalized into lightweight frame-native structures immediately.
- Live execution, signal generation, and drift monitoring operate on generic frame/matrix contracts rather than pandas-only APIs such as `.iloc`, `.loc`, `.resample`, or `.dropna`.

### Deep models

- Deep models do not move to pure-Rust experimental frameworks first.
- When ported, they move to Rust via `tch`/LibTorch.
- Recommended port order:
  1. `mlp`
  2. `transformer`
  3. `nbeats`
  4. `tide`

## Installer Strategy

The runtime should ship with OS-specific bootstrap scripts rather than fallback codepaths.

### Windows

- Install Rust toolchain
- Install required Visual C++ build tools
- Install LibTorch
- Build Rust crates/bindings with the supported feature set
- Install MT5-specific runtime prerequisites

### Linux

- Install Rust toolchain
- Install system build dependencies
- Install LibTorch
- Build Rust crates/bindings with the supported feature set
- Install broker/runtime prerequisites

The bootstrap layer may still use Python or shell scripting during the transition, but the bot logic must not depend on a Python fallback backend after bootstrap completes.

## Deletion Policy

A Python module is safe to delete only when all of the following are true:

1. The runtime registry no longer imports or selects it.
2. The main real path has been verified against the Rust replacement.
3. Save/load persistence contracts match.
4. No supported config path can still activate the Python implementation.
5. Bootstrap scripts on Windows and Linux can build or install the required Rust runtime automatically.

This means "a Rust equivalent exists" is not enough.

## Migration Phases

### Phase 1: Remove pandas from hot paths and runtime gates

- Eliminate pandas-only assumptions in live execution and feature/signal code.
- Remove pandas reconstruction inside Rust/PyO3 bridges where Python models already accept arrays.
- Make Rust tree runtime mandatory for strict runtime mode.

### Phase 2: Remove Python selection for already-ported families

- Lock tree models to Rust-only selection.
- Delete safe Python compatibility modules once the default supported build includes the necessary bindings.

### Phase 3: Port deep families to `tch`/LibTorch

- Port `mlp` first, then `transformer`, `nbeats`, and `tide`.
- Remove Python deep runtime selection for each family after train/predict/save/load parity is verified.

### Phase 4: Final Python runtime cleanup

- Replace Python orchestration entrypoints with Rust-first CLIs/services.
- Remove Python runtime dependencies that are no longer needed after the last active model family is ported.

## Immediate Constraints

- Broad deletion today would be unsafe because several Python modules are still selected by the runtime.
- The highest-value next work is not deleting files blindly; it is removing the remaining pandas-oriented bridge logic and making the Rust-first contract universal.
