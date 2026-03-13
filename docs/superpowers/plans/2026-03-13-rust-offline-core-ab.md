# Rust Offline Core A+B Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the first Rust-owned offline milestone by making data loading, feature preparation, label generation, and discovery-input assembly use a canonical Rust dataset contract with no silent Python fallback.

**Architecture:** `crates/forex-data` owns offline frame loading and normalization, `crates/forex-search` owns discovery-input and label kernels, and `crates/forex-bindings` exposes the canonical prepared offline dataset contract to Python. Python modules in `data.loader`, `features.pipeline`, and discovery callers remain temporary adapters only and must stop recomputing authoritative offline logic.

**Tech Stack:** Python 3.13, pytest, Rust workspace crates (`forex-data`, `forex-search`, `forex-bindings`), NumPy, PyO3

---

## File Structure

**Rust source of truth:**
- `crates/forex-data/src/lib.rs`
- `crates/forex-search/src/lib.rs`
- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/eval.rs`
- `crates/forex-bindings/src/lib.rs`

**Temporary Python adapters in scope:**
- `src/forex_bot/data/loader.py`
- `src/forex_bot/features/pipeline.py`
- `src/forex_bot/features/talib_mixer.py`
- `src/forex_bot/strategy/discovery_tensor.py`
- discovery-input call sites in `src/forex_bot/strategy/evo_prop.py`

**Primary tests:**
- `tests/test_loader_io_backend.py`
- `tests/test_pipeline_rust_numpy.py`
- `tests/test_talib_mixer_compute_signals.py`
- `tests/test_discovery_tensor_rust_eval.py`
- `tests/test_run_prop_discovery_numpy.py`

## Chunk 1: Canonical Offline Dataset Contract

### Task 1: Add bindings-level prepared offline dataset contract

**Files:**
- Modify: `crates/forex-bindings/src/lib.rs`
- Modify: `crates/forex-data/src/lib.rs`
- Test: `tests/test_pipeline_rust_numpy.py`
- Test: `tests/test_loader_io_backend.py`

- [ ] **Step 1: Write the failing Python-side contract test**

Add a targeted test proving the Rust-prepared dataset exposes:
- `features`
- `feature_names`
- `labels`
- `index_ns`
- optional already-aligned `market_metadata`

The contract test must also prove:
- `features` and `labels` use the milestone A+B dtypes/shapes expected by Python consumers
- `feature_names` matches the feature-column count exactly
- `index_ns` is sorted and deduplicated
- `market_metadata` is either absent by default or already aligned to the canonical row index when present

Add a loader-side adapter test in `tests/test_loader_io_backend.py` proving the bindings-backed consumer path receives the same canonical sorted/deduplicated offline dataset contract.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py tests/test_loader_io_backend.py -k canonical_dataset -v`
Expected: FAIL because the current binding/payload contract does not expose the milestone A+B shape and loader-consumer guarantees explicitly.

- [ ] **Step 3: Implement the minimal Rust contract**

Create or tighten one canonical prepared-dataset payload in `crates/forex-bindings/src/lib.rs`, backed by `crates/forex-data/src/lib.rs`, and ensure timestamps are sorted/deduplicated before the payload reaches Python.

- [ ] **Step 4: Run the targeted verification**

Run: `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py tests/test_loader_io_backend.py -k canonical_dataset -v`
Expected: PASS

- [ ] **Step 5: Run the Rust verification for this contract**

Run: `cargo test -p forex-data`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

## Chunk 2: Eliminate Python-Owned Feature/Label Authority

### Task 2: Make Rust label output authoritative in `features.pipeline`

**Files:**
- Modify: `src/forex_bot/features/pipeline.py`
- Modify: `crates/forex-bindings/src/lib.rs`
- Modify: `crates/forex-search/src/eval.rs`
- Test: `tests/test_pipeline_rust_numpy.py`

- [ ] **Step 1: Write the failing tests**

Add tests that prove:
- strict Rust mode does not silently return neutral labels
- shape mismatches hard-fail instead of degrading

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py -k "triple_barrier or strict" -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement the minimal code**

Make `forex_bindings.load_symbol_features(...)` plus Rust label outputs from `crates/forex-search/src/eval.rs` the only source of truth for labels on the Rust path, and reduce `pipeline.py` to validation plus adapter logic only.

- [ ] **Step 4: Run targeted verification**

Run: `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py -k "triple_barrier or strict" -v`
Expected: PASS

- [ ] **Step 5: Run Rust verification**

Run: `cargo test -p forex-search`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

### Task 3: Make Rust base-signal output authoritative in `features.pipeline`

**Files:**
- Modify: `src/forex_bot/features/pipeline.py`
- Modify: `crates/forex-bindings/src/lib.rs`
- Modify: `crates/forex-search/src/discovery.rs`
- Test: `tests/test_pipeline_rust_numpy.py`

- [ ] **Step 1: Write the failing tests**

Add tests that prove:
- Python does not append or reshape `base_signal` after Rust already defined the dataset
- strict Rust mode fails clearly if the Rust base-signal output is malformed
- Python remains an adapter and not a second source of truth for discovery-input shaping

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py -k "base_signal" -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement the minimal code**

Make the Rust payload exposed by `forex_bindings.load_symbol_features(...)` the only source of truth for `base_signal` on the Rust path, and remove Python-side post-hoc `base_signal` shaping from `pipeline.py`.

- [ ] **Step 4: Run targeted verification**

Run: `PYTHONPATH=src pytest tests/test_pipeline_rust_numpy.py -k "base_signal" -v`
Expected: PASS

- [ ] **Step 5: Run Rust verification**

Run: `cargo test -p forex-search`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

### Task 4: Keep discovery-input signals Rust-owned even without precomputed Python caches

**Files:**
- Modify: `src/forex_bot/features/talib_mixer.py`
- Modify: `crates/forex-search/src/discovery.rs`
- Modify: `crates/forex-bindings/src/lib.rs`
- Test: `tests/test_talib_mixer_compute_signals.py`
- Test: `tests/test_discovery_tensor_rust_eval.py`

- [ ] **Step 1: Preserve and extend the failing tests**

Add/keep tests proving:
- on-demand Rust signal computation works without a warmed cache
- transposed/batched Rust outputs are normalized safely
- discovery callers no longer depend on Python-side indicator recomputation

- [ ] **Step 2: Run the targeted tests to verify the current gap**

Run: `PYTHONPATH=src pytest tests/test_talib_mixer_compute_signals.py tests/test_discovery_tensor_rust_eval.py -v`
Expected: FAIL if remaining Python-owned discovery-input behavior exists.

- [ ] **Step 3: Implement the minimal code**

Move the remaining authoritative discovery-input signal generation into Rust. Keep only these temporary adapter behaviors in `talib_mixer.py`:
- caller input normalization
- shape validation for Rust-returned outputs
- temporary compatibility routing to the Rust binding

Do not leave Python-side indicator recomputation as the source of truth.

- [ ] **Step 4: Run targeted verification**

Run: `PYTHONPATH=src pytest tests/test_talib_mixer_compute_signals.py tests/test_discovery_tensor_rust_eval.py -v`
Expected: PASS

- [ ] **Step 5: Run Rust verification**

Run: `cargo test -p forex-search`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

## Chunk 3: Tighten Offline Data Adapters

### Task 5: Make Rust frame order and dedup authoritative in `data.loader`

**Files:**
- Modify: `src/forex_bot/data/loader.py`
- Modify: `crates/forex-data/src/lib.rs`
- Modify: `crates/forex-bindings/src/lib.rs`
- Test: `tests/test_loader_io_backend.py`
- Test: `tests/test_run_prop_discovery_numpy.py`

- [ ] **Step 1: Write the failing tests**

Add tests proving:
- the Rust path is the source of truth for sorted/deduped offline frame order

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_loader_io_backend.py tests/test_run_prop_discovery_numpy.py -k "sort or dedup or order" -v`
Expected: FAIL on the new adapter-boundary assertions.

- [ ] **Step 3: Implement the minimal Rust/data-loader changes**

Make the Rust frame-loading/binding path exposed through `forex_bindings.load_symbol_frames(...)` the source of truth for sorted/deduplicated offline frame order, and reduce `loader.py` to validation/routing only for that behavior.

- [ ] **Step 4: Run targeted verification**

Run: `PYTHONPATH=src pytest tests/test_loader_io_backend.py tests/test_run_prop_discovery_numpy.py -k "sort or dedup or order" -v`
Expected: PASS

- [ ] **Step 5: Run Rust verification**

Run: `cargo test -p forex-data`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

### Task 6: Remove strict-Rust Python resample fallback from `data.loader`

**Files:**
- Modify: `src/forex_bot/data/loader.py`
- Modify: `crates/forex-data/src/lib.rs`
- Modify: `crates/forex-bindings/src/lib.rs`
- Test: `tests/test_loader_io_backend.py`
- Test: `tests/test_run_prop_discovery_numpy.py`

- [ ] **Step 1: Write the failing tests**

Add tests proving:
- strict Rust mode does not route to Python dataframe resampling
- adapter code fails explicitly when the Rust resample/load path is unavailable or malformed

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_loader_io_backend.py tests/test_run_prop_discovery_numpy.py -k "resample or strict" -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement the minimal Rust/data-loader changes**

Make `forex_bindings.load_symbol_frames(...)` plus `crates/forex-data` resample behavior the only authoritative offline resample path in strict Rust mode, and delete the Python fallback branch for that mode.

- [ ] **Step 4: Run targeted verification**

Run: `PYTHONPATH=src pytest tests/test_loader_io_backend.py tests/test_run_prop_discovery_numpy.py -k "resample or strict" -v`
Expected: PASS

- [ ] **Step 5: Run Rust verification**

Run: `cargo test -p forex-data`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

## Chunk 4: Deletion Gate

### Task 7: Remove deletion-ready Python implementation surfaces

**Files:**
- Modify: `src/forex_bot/features/talib_mixer.py`
- Modify: `src/forex_bot/features/pipeline.py`
- Modify: `src/forex_bot/data/loader.py`
- Delete or trim only the functions proven redundant by prior tasks

- [ ] **Step 1: Write the failing deletion-gate tests**

Add or tighten deletion-gate tests named with a dedicated selector such as `cleanup_gate` or `deletion_gate` that prove callers no longer rely on the Python-owned helper being removed.

The deletion surfaces in this milestone must stay narrow and explicit:
- `pipeline.py`: Python-owned label fallback branches and Python post-hoc `base_signal` shaping that duplicate the Rust dataset contract
- `loader.py`: strict-Rust Python resample fallback branches and Python sort/dedup helpers superseded by the Rust bindings path
- `talib_mixer.py`: Python-side indicator recomputation helpers that duplicate Rust discovery-input generation, while retaining input normalization, shape validation, and Rust routing

- [ ] **Step 2: Run the targeted tests to verify they fail without the new redirection**

Run: `PYTHONPATH=src pytest tests/test_loader_io_backend.py tests/test_pipeline_rust_numpy.py tests/test_talib_mixer_compute_signals.py tests/test_discovery_tensor_rust_eval.py tests/test_run_prop_discovery_numpy.py -k "cleanup_gate or deletion_gate" -v`
Expected: FAIL on the deletion-gate assertion before removal/redirection is complete.

- [ ] **Step 3: Audit active imports and usages before deletion**

Use `rg` or equivalent to verify every deletion candidate has no remaining active imports/usages/callers outside the temporary adapter surfaces kept by the spec.

- [ ] **Step 4: Delete or trim the minimal Python code**

Only remove the explicit helper surfaces proven redundant by prior tasks. Do not delete `pipeline.py` or `loader.py` wholesale in this milestone. Do not remove the `talib_mixer.py` adapter until active discovery-input callers are fully redirected.

- [ ] **Step 5: Run the full milestone Python gate**

Run: `PYTHONPATH=src pytest tests/test_loader_io_backend.py tests/test_pipeline_rust_numpy.py tests/test_talib_mixer_compute_signals.py tests/test_discovery_tensor_rust_eval.py tests/test_run_prop_discovery_numpy.py -v`
Expected: PASS

- [ ] **Step 6: Run the full milestone Rust gate**

Run: `cargo test -p forex-data`
Expected: PASS

Run: `cargo test -p forex-search`
Expected: PASS

Run: `cargo test -p forex-bindings`
Expected: PASS

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-13-rust-offline-core-ab.md`. The next execution path should use `subagent-driven-development`, one task at a time, with TDD and review gates.
