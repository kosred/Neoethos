# Rust Offline Core Milestone A+B Design

## Goal

Complete the first deletion-safe Rust migration milestone for `forex-ai` by moving the offline data core plus the feature/label/discovery-input core to Rust-owned behavior. Python remains only as a temporary adapter around the Rust outputs. MT5/live trading is explicitly out of scope for this milestone.

This milestone covers:

- offline data loading and normalization
- resampling and time alignment utilities
- prepared feature dataset construction
- base-signal and discovery-input generation
- label generation
- strict Rust-mode failure behavior for these paths

This milestone does not cover:

- global pooled-dataset assembly and training orchestration
- final CLI/runtime migration
- MT5/live trading

The broader future roadmap is tracked separately in [2026-03-13-rust-migration-roadmap.md](C:/Users/konst/development/forex-ai/docs/superpowers/specs/2026-03-13-rust-migration-roadmap.md).

## Why This Milestone

The repo already has substantial Rust coverage, but the first offline handoff is still incomplete:

- Rust data and feature loaders exist, but Python still owns parts of alignment, shaping, and fallback behavior.
- Discovery inputs can still silently degrade to zero or neutral outputs.
- Python still contains multiple copies of timestamp and frame-shaping logic.
- Some modules look “ported” but are not yet safe to delete because callers still depend on Python behavior.

Milestone A+B is the smallest boundary that removes the worst pandas/GIL/alignment risk without entangling the much larger training-orchestration rewrite.

## Scope

### In scope

#### A. Offline data core

- symbol/timeframe discovery for offline training inputs
- OHLCV load and normalize
- resample or materialize required timeframes
- timestamp normalization to canonical `int64` nanoseconds
- sorted/deduplicated frame order
- frame-like to Rust-native conversion boundary

#### B. Feature, label, and discovery-input core

- Rust-owned prepared feature tensors or matrices
- Rust-owned base-signal/discovery-input generation
- TA-Lib/SMC signal generation used by discovery input paths
- Rust-owned label generation, including triple-barrier labels
- strict Rust-mode behavior when a required Rust feature is unavailable

### Out of scope

- pooled global dataset alignment across symbols
- shard merge for global training
- model training orchestration
- trainer/runtime shell rewrite
- MT5/live execution
- deleting Python news ingestion or broker modules

## Rust Ownership Boundary

This milestone must establish one canonical offline dataset contract.

### Canonical Rust output

Rust is the source of truth for a prepared single-symbol offline dataset with these fields:

- `features`: `float32` 2D matrix, row-major, shape `(rows, features)`
- `labels`: `int8` 1D vector, shape `(rows,)`
- `index_ns`: sorted, deduplicated `int64` nanosecond timestamps, shape `(rows,)`
- `feature_names`: ordered feature-name list matching matrix columns
- `market_metadata`: optional aligned metadata needed only if a remaining Python caller still requires it temporarily

If `market_metadata` is present, it must already be aligned to `index_ns` by Rust. Python adapters may pass it through, but they must not reorder, reshape, or extend it into a second source of truth. If no temporary consumer needs it, the canonical Rust output should omit it.

### Ownership by crate

- `crates/forex-data`
  - offline symbol/timeframe loading
  - resampling/materialized timeframe generation
  - canonical timestamp order
  - feature-column preparation

- `crates/forex-search`
  - discovery-input signal generation
  - base-signal computation kernels
  - label-generation kernels that depend on OHLCV and thresholds

- `crates/forex-bindings`
  - the temporary Python seam
  - exposes canonical Rust dataset output to Python
  - must not reimplement business logic that belongs in `forex-data` or `forex-search`

### Allowed Python consumers during this milestone

The following Python modules may temporarily consume the Rust dataset, but they are adapters only:

- `src/forex_bot/data/loader.py`
- `src/forex_bot/features/pipeline.py`
- `src/forex_bot/strategy/discovery_tensor.py`

These modules may translate or route Rust outputs, but they must not recompute authoritative offline logic once the Rust path exists.

`src/forex_bot/strategy/evo_prop.py` is in scope as a caller that must be redirected off Python-owned discovery-input logic during this milestone. It is not an allowed long-term owner of the canonical dataset contract.

## Current Python Surfaces In Scope

### Primary migration targets

- `src/forex_bot/data/loader.py`
- `src/forex_bot/features/pipeline.py`
- `src/forex_bot/features/talib_mixer.py`
- discovery-input portions of `src/forex_bot/strategy/evo_prop.py`
- discovery-input portions of `src/forex_bot/strategy/discovery_tensor.py`

### Rust surfaces that must align with them

- `crates/forex-data/src/lib.rs`
- `crates/forex-search/src/lib.rs`
- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/eval.rs`
- `crates/forex-bindings/src/lib.rs`

## Keep/Delete Status For Milestone A+B

### Expected deletion candidates by milestone end

These are allowed deletion targets if parity and caller cleanup are complete:

- `src/forex_bot/features/talib_mixer.py`
  - only if all active discovery-input generation no longer depends on Python-side signal assembly
- Python-only helper functions in `src/forex_bot/data/loader.py`
  - specifically legacy resample/alignment helpers that duplicate Rust behavior
- Python-only helper functions in `src/forex_bot/features/pipeline.py`
  - specifically label/base-signal helpers superseded by Rust outputs

### Must explicitly remain after milestone end

- `src/forex_bot/features/pipeline.py`
  - as a thin adapter until training orchestration is migrated
- `src/forex_bot/data/loader.py`
  - as a thin adapter and compatibility surface for remaining callers
- `src/forex_bot/execution/training_service.py`
  - deferred to the next major milestone
- `src/forex_bot/training/trainer.py`
  - deferred to the training-orchestration milestone
- `src/forex_bot/execution/*`
  - live execution stays outside this milestone
- `src/forex_bot/data/news/*`
  - ingestion/storage stays outside this milestone

No full Python file is deleted unless its remaining callers are removed or redirected.

## News And Session Alignment Decision

Full Rust ownership of news/session joins is deferred.

### Required for this milestone

- the strict Rust offline path must behave correctly without Python news joins
- Python adapters must not silently inject a partially aligned news path into the Rust dataset contract
- if news features are unavailable in the Rust-first path, the behavior must be explicit and stable

### Not required for this milestone

- porting all Python news ingestion, storage, rescoring, or join logic to Rust

This keeps A+B bounded. News/session joins can be scheduled as a follow-up once the core offline dataset boundary is stable.

## Error Handling

The milestone must replace silent degradation with explicit behavior.

### Required error-handling rules

- missing Rust bindings on a Rust-first path must hard-fail or return a blocked empty dataset only where the runtime contract already expects blocking behavior
- shape mismatches from Rust outputs must be treated as explicit errors
- strict Rust mode must not silently fall back to Python dataframes or Python label generation
- unsupported discovery-input kernels must fail explicitly instead of returning valid-looking zero signals

### Specific anti-patterns to remove

- all-zero discovery outputs caused by unwarmed cache or missing Rust execution
- all-neutral labels returned as a quiet fallback
- Python-side timestamp alignment becoming the real source of truth
- Python-side feature extension changing the canonical dataset contract after Rust output is produced

## Testing Strategy

This milestone needs a bounded verification set, not the full repo test surface.

### Python verification gate

These tests are the milestone gate for A+B:

- `PYTHONPATH=src pytest tests/test_loader_io_backend.py -v`
- `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py -v`
- `PYTHONPATH=src pytest tests/test_talib_mixer_compute_signals.py -v`
- `PYTHONPATH=src pytest tests/test_discovery_tensor_rust_eval.py -v`
- `PYTHONPATH=src pytest tests/test_run_prop_discovery_numpy.py -v`

### Rust verification gate

These Rust checks are the milestone gate for A+B:

- `cargo test -p forex-data`
- `cargo test -p forex-search`
- `cargo test -p forex-bindings`

### Additional required assertions

- strict Rust-mode tests must prove no Python fallback is used
- targeted tests must confirm sorted/deduplicated index behavior where the Rust dataset is consumed
- deletion candidates must be checked against active imports/usages before removal

## Success Criteria

Milestone A+B is complete when:

- offline single-symbol prepared datasets come from a canonical Rust-owned contract
- feature and label generation do not silently degrade in strict Rust mode
- discovery inputs no longer depend on Python-side signal recomputation
- Python modules in scope are adapter layers only
- at least one in-scope Python implementation surface is deletion-ready or partially deleted without breaking the verification gate

## Non-Goals

- migrating pooled global training data assembly
- migrating trainer or worker orchestration
- rewriting the full runtime shell
- deleting Python everywhere in one pass
- moving MT5/live execution into Rust in this milestone
