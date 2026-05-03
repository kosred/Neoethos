# Search GPU Discovery / Scheduler Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Scope: GPU search, CubeCL evaluation, CUDA reproduction, tensor discovery, and HPC island discovery.

## Important scope note

The following files were inspected:

- `crates/forex-search/src/cubecl_eval.rs` partially; the connector truncated the later part of the file.
- `crates/forex-search/src/cubecl_ga.rs` fully.
- `crates/forex-search/src/discovery_gpu.rs` partially; the central GPU discovery, multi-GPU split, and reproduction bridge were visible.
- `crates/forex-search/src/hpc_gpu_discovery.rs` partially; the island model, evaluation loop, migration, and approximate tensor fitness were visible.

No production code was changed by this audit.

## Core finding

The search layer currently has two different GPU philosophies:

1. CubeCL kernels that try to implement signal synthesis, reproduction, and backtest-like evaluation.
2. Tensor/tch discovery paths that perform fast returns-based approximate search.

These paths are useful, but they must not be treated as the same kind of result.

The required distinction is:

```text
approximate GPU presearch != canonical strategy validation
```

Approximate paths may find candidates quickly. Final accepted strategies must pass through a canonical evaluator with CPU/GPU parity guarantees.

## `cubecl_eval.rs`

### Important kernels

The file contains at least two major custom CubeCL/CUDA kernels:

```rust
synthesize_signals_kernel
backtest_population_kernel
```

These are valuable and must not be deleted blindly.

`synthesize_signals_kernel` builds signals for many genes and many samples using:

- indicator weights
- long/short thresholds
- SMC flags
- SMC row data
- SMC weights
- gate threshold

`backtest_population_kernel` evaluates a population of signals and computes:

- net profit
- peak equity
- max drawdown
- win rate
- profit factor
- expectancy
- max daily drawdown
- trade counts
- monthly PnL

### Critical semantic risk

The GPU backtest opens a new position using:

```rust
let s = signals_flat[signal_base + i];
```

This appears to use the current bar signal at index `i`.

If the CPU evaluator uses prior-bar causality, for example `signals[i - 1]`, then the GPU path has a possible same-bar lookahead / CPU-GPU semantic mismatch.

This is a correctness issue, not a performance issue. A fast evaluator with different signal timing can produce false strategy rankings.

### Timestamp risk

The visible code computes timestamp deltas as raw differences:

```rust
timestamps[i] - timestamps[i - 1]
```

and treats the result as milliseconds. If the input timestamps are nanoseconds, seconds, or mixed units, `gap_threshold_ms` behavior becomes wrong.

The canonical GPU evaluator must receive normalized milliseconds or an explicit timestamp unit contract.

### Current runtime/device issues

The file currently uses env-driven runtime choices such as:

- `FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE`
- `FOREX_BOT_SEARCH_EVAL_CUDA_KERNEL`
- `FOREX_BOT_SEARCH_BACKTEST_CUDA_KERNEL`
- `FOREX_BOT_SEARCH_EVAL_PRECISION`
- `FOREX_BOT_TRAIN_PRECISION`
- `FOREX_TRAIN_PRECISION`

These choices should move into typed scheduler/runtime config. Kernel files should receive a `DeviceAssignment` and precision policy from the scheduler.

## `cubecl_ga.rs`

### Important kernel

The file contains:

```rust
blend_mutate_kernel
```

This is a custom CUDA/CubeCL reproduction kernel. It blends parent genomes or uses the mean vector, adds noise, clamps values to `[-1, 1]`, and returns child genomes.

This file was fully inspected and should be preserved.

### Positive point

`try_generate_children_cuda` receives RNG from its caller. This is better than using global random inside the kernel wrapper because deterministic search can be achieved if the caller/scheduler supplies seeded RNG streams.

### Limitations

The GPU kernel only performs the final numerical blend/mutate step. Parent selection, crossover decision, and noise generation are still CPU-side.

This is acceptable initially, but multi-GPU execution should be represented as scheduled work units such as:

```rust
WorkUnitKind::SearchReproductionBatch
```

rather than a local CUDA helper.

## `discovery_gpu.rs`

### Role

This file implements tensor/tch GPU discovery. It uses a fast returns-based fitness rather than full SL/TP/spread/commission simulation.

The visible design note explicitly says this path is not equivalent to the CPU GA driven by `evolve_search` and does not model SL/TP, spread, or commission.

Therefore this path should be labeled as:

```rust
GpuTensorApproxPresearch
```

not as final canonical validation.

### Positive points

The file has useful architecture:

- `GpuDiscoveryConfig`
- explicit population/generation/chunk/device config
- optional deterministic seed
- multi-GPU chunk splitting
- data/ohlc cubes kept resident per GPU
- CUDA reproduction bridge through `try_generate_children_cuda`

### Problems

`resolve_execution_mode` can silently degrade CUDA search to CPU unless an env flag is set.

This must become typed policy:

```rust
GpuRequired -> fail closed if no GPU
GpuPreferred -> explicit degraded CPU fallback
CpuOnly -> CPU reference
```

The result type records `used_gpu`, `runtime_backend`, and `degraded_reason`, but it needs stronger semantics:

- canonical vs approximate
- feature schema hash
- dataset fingerprint
- device assignment
- precision contract
- validation status

The result currently carries `feature_names: frames[0].names.clone()` and synthetic timeframe labels like `tf_0`, `tf_1`. This is too weak for live/training handoff.

## `hpc_gpu_discovery.rs`

### Role

This file implements an island-model GPU search for HPC mode. It is oriented toward an 8xA6000 / NVLink style topology.

Useful ideas:

- one GPU per island
- parallel island evaluation
- elite migration
- NVLink-aware migration
- CPU affinity per GPU
- larger chunks for A6000
- island-local reproduction

### Major limitation

Although the header mentions multi-fidelity screening and thorough CPU validation, the visible code returns top elites directly as `GpuDiscoveryResult` without a visible canonical validation step before return.

Therefore this path should be treated as:

```rust
HpcIslandApproxPresearch
```

not as final accepted strategy generation.

### Determinism issue

The visible code uses `rand::rng()` in island initialization, selection, evolution, and segment generation.

For reproducible search, resume support, and parity testing, island RNG should be explicitly seeded:

```text
global_seed + island_id + generation + shard_id
```

### Topology issue

This file is not a general scheduler. It encodes assumptions about HPC mode, A6000-like chunking, and NVLink pairs.

For 16 RTX 4060 cards, the correct abstraction is not hardcoded NVLink island logic. It should be a topology-aware scheduler profile that can handle:

- no NVLink
- PCIe-only groups
- different VRAM sizes
- different device speeds
- per-device work queues

## Required labels / runtime modes

Introduce explicit modes so artifacts cannot confuse approximate and canonical results:

```rust
pub enum SearchEvaluationMode {
    GpuTensorApproxPresearch,
    HpcIslandApproxPresearch,
    CubeClCudaCanonicalBacktest,
    CpuCanonicalBacktest,
}
```

or as part of a broader:

```rust
pub enum RuntimeMode {
    GpuCanonical,
    GpuApproximatePresearch,
    CpuReference,
    CpuFallbackAllowed,
    CpuFallbackDegraded,
    Unsupported,
}
```

## Required work units

The scheduler should represent search as work units:

```rust
WorkUnitKind::SignalSynthesisBatch
WorkUnitKind::BacktestGeneBatch
WorkUnitKind::SearchReproductionBatch
WorkUnitKind::GpuTensorPresearchBatch
WorkUnitKind::HpcIslandBatch
WorkUnitKind::ValidationWindowShard
WorkUnitKind::SymbolSearchJob
```

Useful sharding strategies:

```rust
ShardingStrategy::ByGeneRange
ShardingStrategy::ByCandidateRange
ShardingStrategy::BySymbol
ShardingStrategy::ByTimeframe
ShardingStrategy::ByValidationWindow
ShardingStrategy::ByRowChunk
ShardingStrategy::ByIsland
```

## Required tests before scheduler refactor

Before making this the GPU-first search engine, add parity tests.

### Signal synthesis parity

- CPU signal generation vs `synthesize_signals_kernel`
- no SMC flags
- full SMC flags
- partial SMC gate
- threshold edge cases
- bf16/fp32 tolerance

### Backtest parity

- CPU evaluator vs `backtest_population_kernel`
- prior-bar vs current-bar signal timing
- SL before TP intrabar policy
- TP before SL intrabar policy if supported
- spread and commission
- max hold bars
- min hold bars
- max trades per day
- timestamp gap close
- daily drawdown
- monthly PnL aggregation
- final open-position policy

### Reproduction parity

- CPU child generation vs `blend_mutate_kernel`
- crossover child
- mean-only child
- clamp to `[-1, 1]`
- bf16/fp32 tolerance
- deterministic seeded RNG stream

### Approximate presearch safety

- approximate candidates must not be saved as final accepted strategies
- final acceptance must require canonical validation
- artifacts must record approximate vs canonical mode

## Migration plan

1. Preserve existing custom CubeCL/CUDA kernels.
2. Add typed runtime/search evaluation modes.
3. Add CPU/GPU parity tests for signal synthesis and backtest first.
4. Fix any signal timing mismatch before increasing GPU scale.
5. Move env/device/precision selection into scheduler config.
6. Wrap `synthesize_signals_kernel`, `backtest_population_kernel`, and `blend_mutate_kernel` behind scheduler-managed kernel interfaces.
7. Label tensor/tch discovery paths as approximate presearch.
8. Require canonical validation before saving final accepted strategies.
9. Generalize HPC island logic into topology-aware scheduler profiles.

## Bottom line

The search GPU layer has valuable kernels and useful high-throughput approximate discovery paths. The risk is semantic confusion.

A fast approximate search path is useful only if final candidates pass through a canonical evaluator. The immediate priority is CPU/GPU semantic parity for the CubeCL backtest path, especially signal timing and timestamp units. After that, the scheduler can safely scale search across all available GPUs.
