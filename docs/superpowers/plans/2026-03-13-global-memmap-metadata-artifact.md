# Global Memmap Metadata Artifact Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist frame-native OHLC metadata alongside the pooled memmap dataset so pandas-free training does not recompute features just to recover metadata for metadata-dependent models.

**Architecture:** Keep the fix on the existing Rust-first offline path. Global pooling already streams `X.npy` and `y.npy`; this tranche adds a small frame-native metadata artifact written from the pooled metadata parts and loaded by the trainer before any fallback regeneration path is considered.

**Tech Stack:** Python, numpy, joblib, frame-native `_NumpyFrame` metadata containers, pytest, real `python -m forex_bot.main` verification

---

## Chunk 1: Global Memmap Metadata Artifact

### Task 1: Add the regression and minimal persistence helper

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/src/forex_bot/execution/training_service.py`
- Test: `C:/Users/konst/development/forex-ai/tests/test_global_memmap_metadata_artifact.py`

- [ ] **Step 1: Write the failing test**

```python
def test_persist_pooled_metadata_artifact_writes_metadata_pickle(tmp_path):
    ...
    assert (tmp_path / "metadata.pkl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_global_memmap_metadata_artifact.py -v`
Expected: FAIL because the helper does not exist or does not persist the artifact

- [ ] **Step 3: Write minimal implementation**

```python
def _persist_pooled_metadata_artifact(memmap_dir: Path, pooled_meta: list[Any]) -> Any | None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_global_memmap_metadata_artifact.py -v`
Expected: PASS

### Task 2: Integrate the helper into global pooled memmap training

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/src/forex_bot/execution/training_service.py`
- Modify: `C:/Users/konst/development/forex-ai/src/forex_bot/training/trainer.py`
- Test: `C:/Users/konst/development/forex-ai/tests/test_global_memmap_metadata_artifact.py`

- [ ] **Step 1: Extend the regression to cover the training-path preference**

```python
def test_train_all_pandas_free_prefers_metadata_artifact_over_regeneration(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_global_memmap_metadata_artifact.py -v`
Expected: FAIL because trainer still reaches regeneration logic

- [ ] **Step 3: Write minimal implementation**

```python
if pooled_meta and memmap_dir is not None:
    meta_train = _persist_pooled_metadata_artifact(memmap_dir, pooled_meta)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_global_memmap_metadata_artifact.py -v`
Expected: PASS

### Task 3: Verify on the real training path

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/src/forex_bot/execution/training_service.py`
- Modify: `C:/Users/konst/development/forex-ai/src/forex_bot/training/trainer.py`
- Verify: `C:/Users/konst/development/forex-ai/logs/parallel_workers/`

- [ ] **Step 1: Run the real command**

Run:

```powershell
PYTHONPATH=src python -m forex_bot.main --train --quick-e2e --quick-rows 256 --quick-budget-seconds 10 --symbol EURUSD --runtime-profile rust_fast
```

Expected: command completes with `Training Complete.`

- [ ] **Step 2: Inspect runtime logs**

Run:

```powershell
Get-ChildItem cache/global_pool | Sort-Object Name -Descending | Select-Object -First 1
Get-ChildItem <latest_pool_dir>
```

Expected: `metadata.pkl` exists in the latest pool dir and the main log does not print `Pandas-free memmap: regenerated metadata`

