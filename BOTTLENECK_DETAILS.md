# Detailed Bottleneck Analysis by File

## `forex-ai.py` - Main Entry Point

### Lines 62-110: Environment Setup
**Issue:** Sets many environment variables but doesn't validate conflicts.

```python
# PROBLEM: These could conflict if subprocesses inherit
os.environ["RAYON_NUM_THREADS"] = str(rust_threads)  # 240
os.environ["OMP_NUM_THREADS"] = str(blas_threads)     # 1
```

**Impact:** If a subprocess spawns another subprocess, thread count explodes.

**Fix:** Use `os.environ.pop()` to clean up before subprocess calls, or use context managers.

---

### Lines 178-195: Global Thread Settings
**Issue:** Sets global BLAS threads to 1 (good) but doesn't prevent oversubscription from Python side.

```python
# This is good but incomplete
blas_threads = 1
os.environ["OMP_NUM_THREADS"] = str(blas_threads)
```

**Impact:** With 240 Rust threads + 252 Python processes + 1 BLAS thread = potential 493 threads.

**Fix:** Use a central thread budget allocator:
```python
class ThreadBudget:
    total = 252
    gpu_coordination = 12  # Reserve for GPU
    rust_compute = 240     # MAX
    python_workers = 0     # Use async instead
```

---

## `crates/forex-search/src/eval.rs` - Core Evaluation

### Lines 457-487: Per-Gene Allocation Hot Path
**Critical bottleneck for large populations.**

```rust
// Lines 462-463: Allocated for EVERY gene
let mut combined = vec![0.0_f32; n_samples];
let mut signals = vec![0i8; n_samples];
```

**With 500K genes × 100K samples:**
- `combined`: 500K × 100K × 4 bytes = 200GB allocated
- `signals`: 500K × 100K × 1 byte = 50GB allocated
- Total: 250GB of allocations per generation!

**Fix:** Thread-local object pool:
```rust
thread_local! {
    static COMBINED_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(1024*1024));
    static SIGNALS_BUF: RefCell<Vec<i8>> = RefCell::new(Vec::with_capacity(1024*1024));
}
```

---

### Lines 457-487: Sequential Signal Generation
**Inside the Rayon parallel iterator, signal generation is sequential:**

```rust
// Lines 480-487: Sequential loop inside parallel iterator
for i in 0..n_samples {
    let v = combined[i];
    if v >= lt { signals[i] = 1; }
    else if v <= st { signals[i] = -1; }
}
```

**Fix:** Use SIMD comparison:
```rust
// AVX2: Compare 8 floats at once
let v = _mm256_loadu_ps(&combined[i]);
let gt_mask = _mm256_cmp_ps(v, lt_vec, _CMP_GE_OQ);
// ...
```

---

### Lines 516-546: SMC Gate Branching
**Branch-heavy code in evaluation loop:**

```rust
if use_ob_g && ob_arr[i] == dir {  // Branch 1
    score += smc_weight_ob;
}
if use_fvg_g && fvg_arr[i] == dir {  // Branch 2
    score += smc_weight_fvg;
}
// ... 6 more branches
```

**Impact:** Branch misprediction ~10-20 cycles each.

**Fix:** Pre-compute masks:
```rust
let ob_mask = if use_ob_g { ob_arr } else { &[1i8; n_samples] };
// Vectorized: score += ob_mask[i] * smc_weight_ob;
```

---

## `crates/forex-search/src/discovery_gpu.rs` - GPU Discovery

### Lines 320-358: Synchronous CPU→GPU Transfers
**Blocks CPU while waiting for GPU:**

```rust
// Line 336-337: Allocates on CPU, then copies
let chunk_tensor = Tensor::from_slice(&chunk_buf)
    .reshape(&[chunk.len() as i64, chunk[0].len() as i64]);

// Line 343: Synchronous transfer
let fit = evaluate_population_gpu(data_cube, ohlc_cube, &part, config, device)?;
```

**Impact:** CPU idle while GPU works, then GPU idle while CPU prepares next chunk.

**Fix:** Double buffering with CUDA streams:
```rust
// Stream 0: Compute current
// Stream 1: Copy next chunk (async)
// Stream 2: Copy results back (async)
```

---

### Lines 396-398: Redundant Device Transfers
**Data cube transferred every evaluation:**

```rust
let data = data_cube.to_device(device).to_kind(Kind::Float);  // Every time!
let ohlc = ohlc_cube.to_device(device).to_kind(Kind::Float);  // Every time!
```

**With 8 GPUs:** Same data transferred 8 times!

**Fix:** Keep data_cube on GPU persistently (fits in 48GB):
```rust
struct GpuCache {
    data_cube: Tensor,  // Stays on GPU
    ohlc_cube: Tensor,  // Stays on GPU
}
```

---

## `crates/forex-data/src/lib.rs` - Data Loading

### Lines 253-467: Single-Threaded TA-Lib
**Sequential indicator computation:**

```rust
for func_name in func_names {  // 100+ indicators, sequential!
    // ... setup
    let ret = unsafe { TA_CallFunc(params, ...) };  // CPU-bound
}
```

**With 100 indicators:** 100× sequential = slow.

**Fix:** Group independent indicators:
```rust
// Group by input dependencies
let groups = vec![
    vec!["RSI", "MOM", "ROC"],      // Independent group 1
    vec!["MACD", "MACDEXT"],         // Independent group 2
    vec!["BBANDS", "STDDEV"],        // Independent group 3
];

groups.par_iter().for_each(|group| {
    for indicator in group {
        compute_indicator(indicator);  // Parallel groups
    }
});
```

---

### Lines 730-810: Feature Frame Construction
**Multiple passes over data:**

```rust
// Pass 1: indicators
for (name, values) in indicators {
    columns.push(vals.iter().map(|v| *v as f32).collect());  // Alloc + copy
}

// Pass 2: Array2 construction
for (col_idx, vals) in columns.iter().enumerate() {
    for i in 0..len {
        out[(i, col_idx)] = vals[i];  // Another copy
    }
}
```

**Impact:** 3× memory traffic.

**Fix:** Direct construction:
```rust
let mut out = Array2::zeros((n_rows, n_cols));
for (col_idx, (name, values)) in indicators.iter().enumerate() {
    for (i, v) in values.iter().enumerate() {
        out[(i, col_idx)] = *v as f32;  // Single pass
    }
}
```

---

## `src/forex_bot/strategy/discovery_tensor.py`

### Lines 500-600: Python Loop Over Genes
**GIL-held Python loop:**

```python
scored = []
for gene in genes:  # 50K+ iterations in Python!
    sig = mixer.compute_signals(df, gene, cache=cache)  # More Python
    metrics = fast_evaluate_strategy(...)  # GIL released
    gene.fitness = score  # GIL re-acquired
    scored.append((score, gene))  # Python list ops
```

**Impact:** 50K × Python overhead = seconds of GIL contention.

**Fix:** Move loop to Rust (which already exists in `eval.rs`):
```python
# Use the Rust batch evaluation directly
result = forex_bindings.batch_evaluate(
    features, genes,  # All at once
)
```

---

### Lines 124-157: Indicator Cache Lookup
**Dictionary lookup per indicator per gene:**

```python
for ind in needed:  # For each indicator
    for gene in population:  # For each gene
        if ind in gene.params:  # Dict lookup
            params = gene.params.get(ind)  # Another lookup
```

**Complexity:** O(n_indicators × n_genes × n_population)

**Fix:** Pre-compute union of all indicators:
```python
all_indicators = set().union(*(g.indicators for g in population))
cache = {ind: compute_indicator(ind) for ind in all_indicators}
```

---

## `src/forex_bot/models/trees.py`

### Lines 178-200: Feature Augmentation Copies
**Multiple DataFrame copies:**

```python
out = df.copy()  # Copy 1
out["ret1"] = ret1  # Copy 2 (implicit)
out["ret1_lag1"] = ret1.shift(1).fillna(0.0)  # Copy 3
# ... 8 more columns
```

**Impact:** 10× data copies for feature augmentation.

**Fix:** In-place operations:
```python
df.assign(
    ret1=ret1,
    ret1_lag1=ret1.shift(1).fillna(0.0),
    # ... in single operation
)
```

---

## `crates/forex-search/src/genetic.rs`

### Lines 602-635: SMC Array Recomputation
**Derived arrays computed every evaluation:**

```rust
fn build_smc_arrays(frame: &FeatureFrame, ohlcv: &Ohlcv) -> ... {
    // Derives from OHLCV every time - expensive!
    derive_smc_arrays(ohlcv)  // Lines 530-601
}
```

**Called from:** `evaluate_genes` → once per batch

**Impact:** Redundant computation.

**Fix:** Cache in `SymbolDataset`:
```rust
struct SymbolDataset {
    symbol: String,
    frames: HashMap<String, Ohlcv>,
    smc_arrays: OnceCell<SmcArrays>,  // Lazy cache
}
```

---

### Lines 88-196: Signature Hash Memory
**HashSet grows unbounded:**

```rust
struct SeenSignatureMemory {
    all: HashSet<u64>,  // Grows forever!
    pending: Vec<u64>,
}
```

**With 500K genes/generation × 200 generations:** 100M entries = 800MB+

**Fix:** Bloom filter for probabilistic deduplication:
```rust
struct SeenSignatureMemory {
    bloom: BloomFilter,     // Fixed 100MB, 1% FP rate
    recent: LruCache<u64>,  // Last 1M only
}
```

---

## `src/forex_bot/training/trainer.py`

### Lines 244-285: Micro-Benchmark Blocking
**Synchronous benchmark during initialization:**

```python
bench_result = self.benchmarker.run_micro_benchmark(
    X, pd.Series([0] * min(len(X), 10)), device
)
```

**Impact:** Blocks training start for benchmark.

**Fix:** Async benchmark or cache results.

---

### Lines 156-178: Reflection in Hot Path
**Signature inspection on every call:**

```python
def _fit_accepts_metadata(model: ExpertModel) -> bool:
    sig = inspect.signature(model.fit)  # Expensive!
    return "metadata" in sig.parameters
```

**Called:** For every model fit.

**Fix:** Cache on model class:
```python
@functools.lru_cache(maxsize=128)
def _fit_accepts_metadata_cached(model_class) -> bool:
    ...
```

---

## Summary Table: Most Critical Lines

| File | Line(s) | Issue | Impact | Fix Priority |
|------|---------|-------|--------|--------------|
| `eval.rs` | 462-463 | Per-gene allocations | 🔴 Critical | 1 |
| `discovery_gpu.rs` | 396-398 | Redundant GPU transfers | 🔴 Critical | 1 |
| `lib.rs` (data) | 253-467 | Sequential TA-Lib | 🔴 Critical | 2 |
| `discovery_tensor.py` | 500-600 | Python loop | 🔴 Critical | 2 |
| `eval.rs` | 480-487 | Sequential signals | 🟠 High | 3 |
| `genetic.rs` | 602-635 | SMC recomputation | 🟠 High | 4 |
| `discovery_gpu.rs` | 320-358 | Sync transfers | 🟠 High | 3 |
| `eval.rs` | 516-546 | Branch divergence | 🟠 High | 5 |
