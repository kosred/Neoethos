# Evolution / NEAT / CRFMNES GPU-First Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Scope: careful file-by-file analysis of the neuro-evolution stack with focus on GPU-first execution and multi-GPU scheduling.

## Important scope note

`crates/forex-models/src/evolution/crfmnes_impl.rs` is a large file. The GitHub connector repeatedly truncated it around the middle of the file. Therefore this audit does not claim complete line-by-line coverage of all 1600+ lines of `crfmnes_impl.rs`.

The following were inspected enough to support the findings below:

- upper/middle part of `crfmnes_impl.rs`
- full `crates/forex-models/src/evolution/crfmnes_gpu.rs`
- large visible part of `crates/forex-models/src/evolution/neat_impl.rs`, including the evolution loop and GPU bridge
- full `crates/forex-models/src/evolution/neat_gpu.rs`

No production code was changed by this audit.

## Core GPU-first requirement

The evolution stack should follow the global project requirement:

```text
GPU-first everywhere.
CPU only as reference, orchestration, I/O, artifact writing, checkpointing, lightweight validation, and explicit fallback.
```

For evolution/search workloads, this means the primary GPU units should be:

- population evaluation batches
- candidate-loss batches
- islands
- symbols
- timeframes
- validation windows
- hyperparameter trials
- fold/window shards
- row chunks when datasets are large enough

The main performance goal is not just to run one kernel on `cuda:0`. The goal is to create enough independent work units to keep all available GPUs busy.

## CRFMNES / NeuroEvo analysis

Files:

- `crates/forex-models/src/evolution/crfmnes_impl.rs`
- `crates/forex-models/src/evolution/crfmnes_gpu.rs`

### Current architecture

`crfmnes_impl.rs` appears to act as the CPU-side optimizer/orchestrator and artifact/runtime layer.

It contains:

- `SimpleEvolutionState`
- optional `CrfmnesEvolutionState`
- `NeuroEvoOptimizer`
- `NeuroEvoArtifact`
- `NeuroEvoExpert`
- CPU parameter decoding
- CPU MLP forward pass
- CPU selection loss calculation
- runtime metadata / artifact validation
- device policy and degraded runtime reporting

The inspected code shows two CPU-centered optimizer paths:

1. `CrfmnesEvolutionState`, enabled by the `neuro-evolution` feature, using the Rust CRFMNES/nalgebra backend.
2. `SimpleEvolutionState`, a fallback simple evolution strategy.

Both keep optimizer/search state on CPU.

### GPU file role

`crfmnes_gpu.rs` contains real custom CubeCL/CUDA code and must be preserved.

Important kernel:

```rust
candidate_loss_kernel
```

This kernel evaluates many candidate parameter vectors. For each candidate it runs a small MLP:

```text
features -> hidden tanh layer -> 3 logits -> softmax loss
```

It outputs:

- selection loss
- train loss
- validation loss

### What is valuable and must be preserved

Preserve:

- CPU optimizer/reference logic in `crfmnes_impl.rs`
- artifact/runtime validation logic, until replaced by a shared contract
- `candidate_loss_kernel`
- candidate flattening logic
- train/validation weighted selection semantics: `0.65 * train + 0.35 * val`
- L2 penalty semantics

### Current GPU limitations

The current CRFMNES GPU path is a single-GPU candidate-loss helper, not a full GPU-first evolution engine.

Limitations:

- CUDA device selection happens inside `crfmnes_gpu.rs`.
- It reads env vars such as `FOREX_BOT_NEURO_EVO_CUDA_DEVICE`.
- Kernel units are selected inside the file via `FOREX_BOT_NEURO_EVO_KERNEL_UNITS`.
- GPU enable/disable is controlled by `FOREX_BOT_NEURO_EVO_CUDA_KERNEL`.
- Candidate inputs are converted from `Vec<f64>` to `Vec<f32>` for CUDA evaluation; this needs explicit precision provenance.
- The kernel uses one GPU work item per candidate and loops over all rows inside that work item.
- Small populations can underutilize one GPU, and will massively underutilize a 16-GPU system.
- Seeding in the inspected CRFMNES paths uses `rand::rng()` / `rand::random()` patterns, which are not ideal for reproducible search.
- The evaluation budget is env-driven via `FOREX_NEURO_EVO_MAX_EVALS`.

### GPU-first direction for CRFMNES

Do not create a separate `crfmnes_multi_gpu.rs`.

Instead, wrap the existing CUDA loss evaluator as:

```rust
CrfmnesCandidateLossCudaKernel
```

and execute it through a shared scheduler using work units such as:

```rust
WorkUnitKind::EvolutionCandidateBatch
WorkUnitKind::EvolutionIslandBatch
WorkUnitKind::HyperparameterTrial
WorkUnitKind::ValidationWindowShard
WorkUnitKind::SymbolModelTrainingJob
```

The scheduler should decide whether to:

- run many small independent jobs concurrently across GPUs,
- split a large candidate population across GPUs,
- split very large datasets by row chunks,
- run CPU reference validation for selected candidates.

For a 16-GPU system, the first practical speedup should come from candidate/island/symbol/fold/trial parallelism, not from forcing one small population to span all GPUs.

## NEAT analysis

Files:

- `crates/forex-models/src/evolution/neat_impl.rs`
- `crates/forex-models/src/evolution/neat_gpu.rs`

### Current architecture

`neat_impl.rs` contains the NEAT evolutionary orchestration.

It includes:

- seed population creation
- genome scoring
- species assignment
- adjusted fitness
- parent selection
- crossover
- mutation
- immigrant injection
- best genome tracking
- runtime metadata
- artifact validation
- CPU/Rayon evaluation fallback
- optional CUDA fitness evaluation bridge

The visible `evolve_population` loop shows that GPU is used only for population fitness scoring. Speciation, breeding, mutation, and generation management remain CPU-side.

This is reasonable as an initial design because evaluation is the heavy part.

### GPU file role

`neat_gpu.rs` contains real custom CubeCL/CUDA code and must be preserved.

Important kernel:

```rust
neat_population_metrics_kernel
```

This kernel evaluates a batch of NEAT genomes on GPU.

The file also contains a crucial graph-to-GPU bridge:

```rust
flatten_population
```

This flattens each NEAT genome into GPU-friendly buffers:

- node counts
- node offsets
- edge offsets
- edge sources
- edge weights
- activation codes
- biases
- input indices
- output indices
- bias indices
- evaluation order indices
- complexity penalties
- scratch buffer size

This is valuable custom work and must not be deleted.

### What is valuable and must be preserved

Preserve:

- `evolve_population` orchestration
- `assign_species`
- `adjusted_species_scores`
- `select_parent`
- `breed_child`
- `allocate_species_slots`
- `build_seed_population`
- `evaluate_probabilities` as CPU reference
- `NeatDatasetEvaluator` as CPU reference
- `flatten_population`
- `neat_population_metrics_kernel`
- activation mapping between Rust NEAT activations and kernel activation codes
- complexity penalty semantics
- train/validation weighted scoring semantics

### Current GPU limitations

The current NEAT GPU path is useful but still single-device and optional.

Limitations:

- CUDA device selection happens inside `neat_gpu.rs`.
- It reads env vars such as `FOREX_BOT_NEAT_CUDA_DEVICE`.
- Kernel units are selected via `FOREX_BOT_NEAT_KERNEL_UNITS`.
- GPU enable/disable is controlled by `FOREX_BOT_NEAT_CUDA_KERNEL`.
- The GPU kernel evaluates one genome per GPU work item and loops over all rows inside that item.
- The current bridge falls back to CPU/Rayon if CUDA returns wrong count or errors.
- Runtime reporting is string-based and should become typed runtime provenance.
- There is no scheduler-level sharding across multiple GPUs.

### Positive point: deterministic seed

Compared to the inspected CRFMNES path, NEAT is more deterministic-friendly because it persists and uses a `seed: u64` and initializes RNG via `Xoroshiro128PlusPlus::seed_from_u64(self.seed)`.

This is a strong pattern and should be preserved.

### GPU-first direction for NEAT

Do not create a separate `neat_multi_gpu.rs`.

Wrap the existing CUDA evaluator as:

```rust
NeatPopulationMetricsCudaKernel
```

and execute it through the shared scheduler.

Useful work units:

```rust
WorkUnitKind::EvolutionPopulationBatch
WorkUnitKind::EvolutionIslandBatch
WorkUnitKind::SymbolModelTrainingJob
WorkUnitKind::ValidationWindowShard
WorkUnitKind::HyperparameterTrial
```

Useful sharding strategies:

```rust
ShardingStrategy::ByCandidateRange
ShardingStrategy::BySymbol
ShardingStrategy::ByTimeframe
ShardingStrategy::ByValidationWindow
ShardingStrategy::ByHyperparameterTrial
ShardingStrategy::ByRowChunk
```

For many GPUs, the scheduler can distribute:

```text
GPU 0: genomes 0..N for symbol A
GPU 1: genomes 0..N for symbol B
GPU 2: island 0
GPU 3: island 1
GPU 4: validation window 0
GPU 5: validation window 1
...
```

or, for very large populations:

```text
GPU 0: genomes 0..255
GPU 1: genomes 256..511
GPU 2: genomes 512..767
...
```

## Main shared problem across CRFMNES and NEAT

Both systems have the same architectural issue:

```text
CPU orchestrator + optional single-GPU evaluator + env-driven device policy
```

This should become:

```text
CPU orchestrator/reference + scheduler-managed GPU kernels + typed runtime provenance
```

## Required shared scheduler/runtime changes

Add or consolidate shared types such as:

```rust
pub struct DeviceAssignment {
    pub backend: AcceleratorBackend,
    pub device_ids: Vec<usize>,
    pub precision: TrainingPrecision,
    pub memory_budget_gb: Option<f64>,
    pub kernel_units: Option<u32>,
}

pub enum RuntimeMode {
    GpuCanonical,
    GpuApproximatePresearch,
    CpuReference,
    CpuFallbackAllowed,
    CpuFallbackDegraded,
    Unsupported,
}

pub enum WorkUnitKind {
    EvolutionCandidateBatch,
    EvolutionPopulationBatch,
    EvolutionIslandBatch,
    SymbolModelTrainingJob,
    ValidationWindowShard,
    HyperparameterTrial,
}
```

Each GPU kernel should receive a `DeviceAssignment` instead of reading env vars internally.

## Required tests before refactor/removal

Before removing or merging duplicate paths, add parity tests:

### CRFMNES

- CPU `loss_for_params` vs CUDA `candidate_loss_kernel` on a tiny deterministic dataset.
- Train-only scoring vs train+validation weighted scoring.
- f64 CPU candidate vs f32 CUDA candidate tolerance.
- candidate count mismatch failure.
- label range failure.

### NEAT

- CPU `NeatDatasetEvaluator` vs CUDA `neat_population_metrics_kernel` on a tiny deterministic population.
- activation-code parity for Identity, Sigmoid, Tanh, ReLU, LeakyReLU.
- complexity penalty parity.
- train/validation weighted metrics parity.
- flattened graph topology consistency.
- CPU fallback path only when explicitly allowed.

## Deletion policy

No CRFMNES/NEAT GPU file should be deleted until:

1. the custom kernel is wrapped behind scheduler-managed interface,
2. all callers pass through scheduler-managed work units,
3. CPU/GPU parity tests exist,
4. runtime artifacts record device assignment and runtime mode,
5. old and new metrics match within explicit tolerance.

## Bottom line

The NEAT and CRFMNES evolution stack already contains valuable GPU kernels. The problem is not the absence of GPU work. The problem is that GPU execution is currently optional, single-device, env-driven, and local to each model.

The correct direction is GPU-first evolution scheduling:

- keep CPU code as reference/orchestration,
- preserve custom CUDA kernels,
- remove env/device selection from kernel files,
- introduce scheduler-managed `DeviceAssignment`,
- generate enough candidate/population/island/symbol/window/trial work units to saturate all available GPUs.

NEAT is especially promising because population graph evaluation maps naturally to GPU batch execution. CRFMNES is also valuable, but needs larger candidate/island/trial batching to avoid GPU underutilization.
