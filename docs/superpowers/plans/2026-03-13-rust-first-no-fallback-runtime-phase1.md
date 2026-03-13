# Rust-First No-Fallback Runtime Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining pandas-oriented runtime seams from the hot path, make Rust tree execution mandatory in strict runtime mode, and prepare the repo for safe deletion of Python modules that are no longer selected.

**Architecture:** Keep the runtime on two primary data shapes only: Arrow/Polars for columnar IO and `ndarray`/memmap for dense model matrices. Phase 1 does not attempt the full deep-model rewrite; it focuses on removing pandas bridges and fallback selection so later deletion work is mechanically safe.

**Tech Stack:** Rust workspace crates (`forex-data`, `forex-models`, `forex-bindings`), PyO3, Polars, `ndarray`, NumPy memmap, Python 3.13, pytest, cargo

---

## File Structure

- `src/forex_bot/models/registry.py`
  Runtime model selection and backend routing. This is where Python fallback selection must be removed.
- `src/forex_bot/training/trainer.py`
  Main offline training orchestration and pandas-free model filtering.
- `src/forex_bot/training/parallel_worker.py`
  Worker-side model selection and memmap dataset loading.
- `src/forex_bot/features/engine.py`
  Live signal generation over generic frames and arrays.
- `src/forex_bot/execution/order_execution.py`
  Live execution helpers and scalar extraction from frame-native results.
- `src/forex_bot/execution/mt5_state_manager.py`
  MT5/live state and feature persistence contracts.
- `src/forex_bot/execution/trading_loop.py`
  Live loop orchestration and drift monitoring hooks.
- `src/forex_bot/execution/drift_monitor.py`
  Feature drift monitoring; must accept frame-native inputs.
- `crates/forex-models/src/neural_networks.rs`
  PyO3 deep-model wrappers; currently rebuild pandas objects.
- `crates/forex-models/src/genetic.rs`
  PyO3 bridge still reconstructing pandas.
- `crates/forex-models/src/onnx_exporter.rs`
  Export/inference bridge still reconstructing pandas.
- `scripts/prepare_ubuntu_24_04_py313.sh`
  Existing Linux bootstrap script to extend instead of creating a parallel install path.
- `scripts/verify_bindings.py`
  Existing runtime verification helper.

## Chunk 1: Rust-Only Runtime Selection

### Task 1: Remove Python tree fallback selection from the runtime registry

**Files:**
- Create: `tests/test_runtime_registry_rust_only.py`
- Modify: `src/forex_bot/models/registry.py`

- [ ] **Step 1: Write the failing test**

```python
def test_strict_runtime_rejects_python_tree_backend(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "rust_strict")
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_runtime_registry_rust_only.py -v`
Expected: FAIL because the registry still allows Python fallback behavior in at least one tree-model path.

- [ ] **Step 3: Write minimal implementation**

Update `registry.py` so strict runtime mode never resolves tree families to legacy Python backends and errors clearly when the required Rust class is unavailable.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_runtime_registry_rust_only.py -v`
Expected: PASS

## Chunk 2: Frame-Native Live Runtime

### Task 2: Make live signal and execution helpers frame-native instead of pandas-oriented

**Files:**
- Create: `tests/test_live_frame_native_paths.py`
- Modify: `src/forex_bot/features/engine.py`
- Modify: `src/forex_bot/execution/order_execution.py`
- Modify: `src/forex_bot/execution/mt5_state_manager.py`
- Modify: `src/forex_bot/execution/trading_loop.py`
- Modify: `src/forex_bot/execution/drift_monitor.py`

- [ ] **Step 1: Write the failing test**

```python
def test_live_paths_accept_rust_frame_without_pandas():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_live_frame_native_paths.py -v`
Expected: FAIL because at least one live helper still assumes `.iloc`, `.loc`, `.dropna`, or other pandas-only behavior.

- [ ] **Step 3: Write minimal implementation**

Replace pandas-only checks with generic frame helpers and array extraction utilities. Keep behavior unchanged for live trading decisions.

- [ ] **Step 4: Run targeted test**

Run: `PYTHONPATH=src pytest tests/test_live_frame_native_paths.py -v`
Expected: PASS

- [ ] **Step 5: Run real regression command**

Run:
```powershell
PYTHONPATH=src python -m forex_bot.main --train --quick-e2e --quick-rows 256 --quick-budget-seconds 10 --symbol EURUSD --runtime-profile rust_fast
```
Expected: Completes with `Training Complete.` and no new frame-compatibility regressions.

## Chunk 3: Remove Pandas Reconstruction From Rust Bridges

### Task 3: Stop rebuilding pandas objects in deep-model PyO3 wrappers

**Files:**
- Create: `tests/test_rust_bridge_no_pandas.py`
- Modify: `crates/forex-models/src/neural_networks.rs`

- [ ] **Step 1: Write the failing test**

```python
def test_mlp_bridge_accepts_numpy_without_pandas_roundtrip():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_rust_bridge_no_pandas.py -k mlp -v`
Expected: FAIL because the Rust wrapper still reconstructs pandas `DataFrame`/`Series`.

- [ ] **Step 3: Write minimal implementation**

Update the MLP wrapper to pass NumPy arrays directly into the Python model contract instead of rebuilding pandas objects.

- [ ] **Step 4: Run targeted test**

Run: `PYTHONPATH=src pytest tests/test_rust_bridge_no_pandas.py -k mlp -v`
Expected: PASS

### Task 4: Apply the same no-pandas bridge contract to other PyO3 wrappers

**Files:**
- Modify: `crates/forex-models/src/genetic.rs`
- Modify: `crates/forex-models/src/onnx_exporter.rs`

- [ ] **Step 1: Write the failing test**

Extend `tests/test_rust_bridge_no_pandas.py` with genetic/exporter coverage.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_rust_bridge_no_pandas.py -v`
Expected: FAIL for the wrappers still constructing pandas objects internally.

- [ ] **Step 3: Write minimal implementation**

Use array-based bridging for fit/predict/export paths instead of pandas reconstruction.

- [ ] **Step 4: Run Rust/Python verification**

Run:
```powershell
cargo test -p forex-models
PYTHONPATH=src pytest tests/test_rust_bridge_no_pandas.py -v
```
Expected: PASS

## Chunk 4: Bootstrap and Safe Deletion Gate

### Task 5: Make runtime installation explicit and verifiable on Windows/Linux

**Files:**
- Create: `scripts/bootstrap_runtime.ps1`
- Create: `scripts/bootstrap_runtime.sh`
- Modify: `scripts/prepare_ubuntu_24_04_py313.sh`
- Modify: `scripts/verify_bindings.py`

- [ ] **Step 1: Write the failing verification**

Define binding/runtime checks in `scripts/verify_bindings.py` for the supported Rust feature set.

- [ ] **Step 2: Run verification to confirm the current setup is incomplete**

Run: `PYTHONPATH=src python scripts/verify_bindings.py`
Expected: FAIL or report missing required runtime pieces on at least one supported configuration.

- [ ] **Step 3: Write minimal implementation**

Add bootstrap scripts that install the required toolchain/dependencies and build the mandatory Rust runtime on Windows and Linux.

- [ ] **Step 4: Run verification**

Run: `PYTHONPATH=src python scripts/verify_bindings.py`
Expected: PASS on the local machine after bootstrap/build.

### Task 6: Delete only the Python files that are truly dead after verification

**Files:**
- Create: `docs/superpowers/specs/2026-03-13-rust-first-deletion-manifest.md`
- Delete: exact Python modules only after verification proves they are no longer reachable

- [ ] **Step 1: Write the deletion manifest**

Document each candidate file, its Rust replacement, and the verification proving it is no longer selected.

- [ ] **Step 2: Verify deadness before deletion**

Run:
```powershell
rg -n "exact_module_name" src
PYTHONPATH=src python -m forex_bot.main --train --quick-e2e --quick-rows 256 --quick-budget-seconds 10 --symbol EURUSD --runtime-profile rust_fast
```
Expected: No supported runtime path references the file being deleted.

- [ ] **Step 3: Delete the safe module**

Use `apply_patch` deletions only after the manifest entry is complete.

- [ ] **Step 4: Re-run regression verification**

Run:
```powershell
PYTHONPATH=src python -m forex_bot.main --train --quick-e2e --quick-rows 256 --quick-budget-seconds 10 --symbol EURUSD --runtime-profile rust_fast
```
Expected: PASS after each deletion tranche
