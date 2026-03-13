# Rust Offline Migration Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline training and discovery path more Rust-native by eliminating one high-risk Python zero-fallback path first, then tightening the PyO3 training seam.

**Architecture:** Keep Python as a temporary orchestration shell while shifting correctness-critical and memory-heavy offline logic to Rust. Phase 1 starts with `talib_mixer` on-demand Rust signal evaluation so discovery cannot silently degrade to zero signals when bulk cache warmup is skipped, then follows with PyO3 tree-model execution improvements and larger training-service migration work.

**Tech Stack:** Python 3.13, pytest, NumPy, PyO3, Rust workspace crates (`forex-bindings`, `forex-search`, `forex-models`)

---

## Chunk 1: TALib Mixer On-Demand Rust Signals

### Task 1: Restore non-zero on-demand Rust signal computation

**Files:**
- Modify: `tests/test_talib_mixer_compute_signals.py`
- Modify: `src/forex_bot/features/talib_mixer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_signals_uses_rust_binding_without_precomputed_cache(monkeypatch):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_talib_mixer_compute_signals.py -k on_demand_rust -v`
Expected: FAIL because `compute_signals()` currently returns an all-zero vector unless `_rust_signal_cache` was populated beforehand.

- [ ] **Step 3: Write minimal implementation**

```python
if rust_key not in self._rust_signal_cache:
    self._try_rust_signal(df, gene)
```

Add a focused helper that calls `talib_bulk_signals_ohlcv` for one eligible gene and stores the result without clearing unrelated cached entries.

- [ ] **Step 4: Run targeted tests to verify it passes**

Run: `pytest tests/test_talib_mixer_compute_signals.py -v`
Expected: PASS for the new on-demand Rust case and existing cache/transpose/numpy-frame cases.

### Task 2: Keep strict zero fallback only for truly unavailable Rust paths

**Files:**
- Modify: `tests/test_talib_mixer_compute_signals.py`
- Modify: `src/forex_bot/features/talib_mixer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_signals_keeps_zero_fallback_when_rust_unavailable(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails if behavior regresses**

Run: `pytest tests/test_talib_mixer_compute_signals.py -k zero_fallback -v`
Expected: FAIL only if the new on-demand path incorrectly tries to use missing bindings.

- [ ] **Step 3: Write minimal implementation**

Ensure the on-demand helper exits early when bindings are unavailable or the gene cannot yet be represented by the Rust binding contract.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/test_talib_mixer_compute_signals.py -v`
Expected: PASS with both on-demand Rust and true-zero fallback behavior covered.

## Chunk 2: PyO3 Tree-Model Seam

### Task 3: Release the GIL around Rust tree model fit/predict

**Files:**
- Modify: `crates/forex-bindings/src/lib.rs`
- Test: `crates/forex-models/tests/tree_models_integration.rs`

- [ ] **Step 1: Write the failing test**

Add a regression/integration test that exercises tree-model fit/predict from multiple threads or documents the expected detached execution boundary.

- [ ] **Step 2: Run test/build to verify the current seam is insufficient**

Run: `cargo test -p forex-models tree_models_integration -- --nocapture`
Expected: Existing behavior lacks detached execution coverage.

- [ ] **Step 3: Write minimal implementation**

Wrap fit/predict heavy work in `py.detach(...)` and keep Python object extraction outside detached sections.

- [ ] **Step 4: Run Rust verification**

Run: `cargo test -p forex-models`
Expected: PASS

## Chunk 3: Training-Service Migration Prep

### Task 4: Define Rust-native pooled dataset boundary

**Files:**
- Modify: `src/forex_bot/execution/training_service.py`
- Modify: `crates/forex-bindings/src/lib.rs`
- Modify: `tests/test_training_service_global_numpy_dataset.py`

- [ ] **Step 1: Write failing boundary test for pooled dataset assembly**
- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_training_service_global_numpy_dataset.py -v`

- [ ] **Step 3: Introduce a narrow Rust binding for pooled alignment/sort/merge**
- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_training_service_global_numpy_dataset.py -v`

