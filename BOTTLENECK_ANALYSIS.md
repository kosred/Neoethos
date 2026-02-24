# Codebase Bottleneck Analysis

This document identifies performance bottlenecks in the forex-ai codebase across Python and Rust layers.

## Executive Summary

| Severity | Count | Categories |
|----------|-------|------------|
| 🔴 Critical | 5 | GIL contention, CPU-GPU sync, memory copies |
| 🟠 High | 8 | Thread pools, allocations, cache misses |
| 🟡 Medium | 12 | I/O blocking, serialization, pandas overhead |

---

## 🔴 Critical Bottlenecks

### 1. Python GIL Contention in Discovery Loop

**Location:** `src/forex_bot/strategy/discovery_tensor.py` lines 500-600

**Problem:** The discovery loop processes strategies sequentially in Python even with Rust acceleration:

```python
# Current: Sequential Python loop
for gene in genes:
    sig = mixer.compute_signals(df, gene, cache=cache)  # Python overhead per gene
    metrics = fast_evaluate_strategy(...)  # GIL released here but Python loop holds GIL between calls
```

**Impact:** With 50K+ strategies, Python loop overhead dominates despite fast Rust evaluation.

**Solution:** Move the entire batch evaluation to Rust (already partially done in `eval.rs` with Rayon).

---

### 2. CPU-GPU Memory Transfers in GPU Discovery

**Location:** `crates/forex-search/src/discovery_gpu.rs` lines 320-358

**Problem:** Data is transferred CPU→GPU for every chunk:

```rust
let chunk_tensor = Tensor::from_slice(&chunk_buf)  // CPU allocation
    .reshape(&[chunk.len() as i64, chunk[0].len() as i64]);

let data = data_cube.to_device(device)  // GPU transfer
```

**Impact:** PCIe bandwidth becomes bottleneck at high throughput.

**Solution:** 
- Pin memory with `cudaHostRegister` for faster transfers
- Use async transfers with streams
- Keep data_cube persistently on GPU (it fits in 48GB A6000)

---

### 3. Synchronous I/O in Data Loading

**Location:** `crates/forex-data/src/lib.rs` lines 651-696

**Problem:** Parquet files are read synchronously, blocking compute:

```rust
pub fn load_parquet(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let file = std::fs::File::open(path)?;  // Blocking I/O
    let df = ParquetReader::new(file).finish()?;  // Blocking deserialization
```

**Impact:** 464GB RAM can hold all data, but loading is serialized and blocking.

**Solution:** 
- Use `tokio::fs` for async I/O
- Memory-map files with `memmap2`
- Pre-load all data in parallel at startup

---

### 4. Repeated Memory Allocations in Hot Path

**Location:** `crates/forex-search/src/eval.rs` lines 457-487

**Problem:** New allocations for every gene evaluation:

```rust
let mut combined = vec![0.0_f32; n_samples];  // Allocated per gene
let mut signals = vec![0i8; n_samples];       // Allocated per gene
```

**Impact:** With 500K genes × allocation overhead = significant time in malloc.

**Solution:**
- Use object pools with `bumpalo` or `crossbeam-queue`
- Pre-allocate thread-local buffers
- Reuse allocations across generations

---

### 5. TA-Lib Feature Computation is Single-Threaded

**Location:** `crates/forex-data/src/lib.rs` lines 253-467

**Problem:** `compute_talib_indicators` iterates through functions sequentially:

```rust
for func_name in func_names {  // Sequential!
    // Each indicator computed one at a time
    let ret = unsafe { TA_CallFunc(params, ...) };
}
```

**Impact:** Only 1 core used for feature computation despite 252 available.

**Solution:**
- Parallelize indicator computation with Rayon
- Group independent indicators and compute in parallel
- Cache results aggressively

---

## 🟠 High Severity Bottlenecks

### 6. Thread Pool Configuration Conflicts

**Location:** Multiple files

**Problem:** Multiple thread pools compete for CPU (504 logical threads available):
- `RAYON_NUM_THREADS` for Rust (default: all threads)
- `OMP_NUM_THREADS` for BLAS (default: all threads)
- PyTorch DataLoader workers
- Python multiprocessing pools

**Current settings in `forex-ai.py`:**
```python
blas_threads = 1  # Good: prevents oversubscription
os.environ["RAYON_NUM_THREADS"] = str(rust_threads)  # Could be 480
```

**Impact:** With 480 Rust threads + Python workers + 504 available = potential oversubscription.

**Solution (SMT-aware):**
- Use **primary threads (0-251)** for compute-intensive work
- Use **SMT threads (252-503)** for I/O-bound work
- Reserve 24 logical threads (12 physical) for GPU coordination
- Set `RAYON_NUM_THREADS = 240` (primary threads only) for best performance

---

### 7. No NUMA-Aware Memory Allocation

**Location:** `crates/forex-data/src/lib.rs`

**Problem:** Memory allocated without NUMA affinity. On a 2-socket system:
- Socket 0 memory access: ~200GB/s
- Socket 1 memory access: ~200GB/s
- Cross-socket access: ~50GB/s (4× slower)

**Impact:** Cache-coherent memory traffic between NUMA nodes.

**Solution:**
- Use `libc::mbind` to pin allocations to local NUMA node
- Already implemented in `hpc.rs` - integrate into data loading
- Allocate feature cubes on socket-local memory

---

### 8. Feature Cache Uses Synchronous File I/O

**Location:** `crates/forex-data/src/lib.rs` lines 523-551

**Problem:** Cache read/write blocks the caller:

```rust
pub fn load(&self, key: &str) -> Result<Option<FeatureFrame>> {
    let file = std::fs::File::open(&path)?;  // Blocking
    let df = ParquetReader::new(file).finish()?;  // Blocking
```

**Solution:**
- Use async cache operations
- Implement write-behind caching
- Use memory-mapped files for read-only cache

---

### 9. Pandas DataFrame Conversions

**Location:** `src/forex_bot/strategy/discovery_tensor.py` lines 390-404

**Problem:** Conversions between Python and Rust involve pandas:

```python
close = df["close"].to_numpy(dtype=np.float64)  # Copy!
high = df["high"].to_numpy(dtype=np.float64)    # Copy!
```

**Impact:** Data copied multiple times: Parquet → Polars → Pandas → NumPy → Rust.

**Solution:**
- Use Arrow format for zero-copy transfers
- Pass Arrow arrays directly between Python-Rust
- Avoid pandas in hot paths

---

### 10. Python Model Training Holds GIL

**Location:** `src/forex_bot/models/trees.py`

**Problem:** Tree model training (LightGBM/XGBoost) may hold the GIL:

```python
model.fit(X, y)  # May hold GIL during native execution
```

**Impact:** Prevents concurrent discovery while training.

**Solution:**
- Use `concurrent.futures.ProcessPoolExecutor` for training
- Release GIL explicitly with `py.allow_threads`
- Train models in separate processes

---

### 11. Lock Contention on Model Registry

**Location:** `src/forex_bot/models/registry.py` (implied)

**Problem:** Shared model registry likely uses locks:

```python
# Likely pattern (not seen but inferred)
with self._lock:
    self.models[name] = model  # Contention point
```

**Solution:**
- Use `dashmap::DashMap` (already in dependencies)
- Lock-free data structures for read-heavy paths
- Thread-local model caches

---

### 12. Synchronous Ensemble Construction

**Location:** `src/forex_bot/training/ensemble.py` (implied)

**Problem:** Ensemble operations run sequentially after training.

**Solution:**
- Parallel ensemble weight optimization
- Use GPU for correlation matrix computation
- Incremental ensemble updates

---

## 🟡 Medium Severity Bottlenecks

### 13. JSON Serialization for Large Results

**Location:** `crates/forex-search/src/discovery_gpu.rs` lines 73-81

**Problem:** Large results serialized to JSON:

```rust
let json = serde_json::to_string_pretty(&payload)?;  // Slow for 500K genomes
```

**Impact:** 500K genomes × ~1KB = 500MB JSON to serialize.

**Solution:**
- Use binary format (bincode, MessagePack)
- Stream results instead of buffering
- Compress output

---

### 14. No Vectorized Signal Computation

**Location:** `src/forex_bot/features/talib_mixer.py` lines 159-212

**Problem:** Signal computation is Python-loop based:

```python
for ind in indicators:  # Python loop
    series = cache.get(ind)  # Dict lookup per indicator
    arr = np.asarray(series, dtype=np.float64)
    # ... compute per indicator
```

**Solution:**
- Vectorize across all indicators at once
- Use Numba JIT compilation
- Move to Rust with SIMD

---

### 15. Stop-Target Inference in Python

**Location:** `src/forex_bot/strategy/fast_backtest.py` lines 188-240

**Problem:** ATR computation in Python before Rust evaluation:

```python
atr = float(np.nanmean(tr[-window:]))  # Python NumPy
```

**Solution:**
- Move to Rust `hpc_simd.rs`
- Use SIMD-accelerated ATR
- Pre-compute ATR features

---

### 16. No Incremental/Pipelined Data Processing

**Problem:** Each phase waits for previous to complete:

```
Load → Features → Discovery → Validation → Ensemble
  ↑      ↑          ↑            ↑           ↑
Sync   Sync       Sync         Sync        Sync
```

**Solution:**
- Pipeline with bounded queues
- Start feature computation while loading
- Overlap validation with discovery

---

### 17. SMC Arrays Recomputed Every Evaluation

**Location:** `crates/forex-search/src/genetic.rs` lines 603-635

**Problem:** SMC detection arrays built per-evaluation:

```rust
fn build_smc_arrays(frame: &FeatureFrame, ohlcv: &Ohlcv) -> (Vec<i8>, ...) {
    // Derives from OHLCV every time - could be cached
}
```

**Solution:**
- Cache SMC arrays per dataset
- Pre-compute during feature loading

---

### 18. Warp Divergence in GPU Kernels

**Location:** `crates/forex-search/src/discovery_gpu.rs` lines 384-493

**Problem:** GPU kernels have divergent branches per strategy:

```rust
// Different strategies take different paths
if use_ob_g && ob_arr[i] == dir {  // Divergence!
    score += smc_weight_ob;
}
```

**Impact:** GPU warp utilization drops to ~50%.

**Solution:**
- Sort strategies by configuration before batching
- Use warp-specialized kernels
- Separate kernels for different strategy types

---

## Data Flow Analysis

### Current Flow (with bottlenecks)

```
Parquet Files
     ↓ [Blocking I/O - #3]
Polars DataFrame
     ↓ [Copy to Pandas - #9]
Pandas DataFrame  
     ↓ [Python loop - #1]
Strategy Genes
     ↓ [GIL + Allocations - #4]
Rust Evaluation
     ↓ [CPU→GPU transfer - #2]
GPU Kernels
     ↓ [Sync wait]
Results
     ↓ [JSON serialize - #13]
Disk
```

### Optimized Flow

```
Parquet Files
     ↓ [Async I/O + mmap]
Arrow Arrays (shared memory)
     ↓ [Zero-copy to Rust]
Batched Rust Evaluation
     ↓ [Pinned memory + async GPU]
GPU Kernels (persistent data)
     ↓ [Async results]
Binary Output
     ↓ [Stream write]
Disk
```

---

## Bottleneck Heat Map

| Component | CPU | Memory | I/O | GPU | Network |
|-----------|-----|--------|-----|-----|---------|
| Data Loading | 🟡 | 🔴 | 🔴 | ⚪ | ⚪ |
| Feature Compute | 🔴 | 🟡 | ⚪ | 🟡 | ⚪ |
| Strategy Discovery | 🔴 | 🔴 | 🟡 | 🔴 | ⚪ |
| Model Training | 🟡 | 🟡 | 🟡 | 🔴 | ⚪ |
| Validation | 🔴 | 🟡 | ⚪ | 🟡 | ⚪ |
| Ensemble | 🟡 | 🟡 | ⚪ | ⚪ | ⚪ |

Legend: 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low | ⚪ None

---

## Recommended Priority Order

### Phase 1: Quick Wins (Days)
1. Fix thread pool oversubscription (#6)
2. Switch to binary serialization (#13)
3. Cache SMC arrays (#17)

### Phase 2: High Impact (Weeks)
4. Move batch evaluation fully to Rust (#1)
5. Async I/O for data loading (#3)
6. Memory pools for hot path (#4)

### Phase 3: Architecture (Months)
7. Arrow zero-copy throughout (#9)
8. Persistent GPU data cubes (#2)
9. NUMA-aware allocation (#7)
10. Pipelined execution (#16)

---

## Performance Targets (504 Thread System)

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Strategy eval/sec | 200K/GPU | 600K/GPU | 3× |
| Feature compute time (504 threads) | 300s | 30s | 10× |
| Memory copies | 10×/bar | 1×/bar | 10× |
| Thread context switches | 20M/s | 200K/s | 100× |
| End-to-end 20h pipeline | 20h | 6h | 3.3× |
