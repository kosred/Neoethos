# Deep Duplicate Logic / Unified Scheduler Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: duplicate CPU/GPU/multi-GPU/HPC logic, model-training parallelism, search distribution, train/validation split duplication, artifact boilerplate, and unified scheduler design.

## Summary

The repo already contains useful building blocks for hardware planning and GPU acceleration, but the implementation is fragmented.

The problem is not that GPU code is missing. The problem is that GPU, multi-GPU, CPU, and HPC concepts are implemented as separate paths instead of one unified work scheduler.

The desired architecture should be:

```text
One canonical pipeline
One work scheduler
Many backend kernels
Any number of devices
```

Not:

```text
cpu file
single gpu file
multi gpu file
hpc file
special cuda file
special model gpu file
```

## High-level finding

The codebase has a `HardwareExecutionPlan` / `WorkloadExecutionPlan` foundation in `forex-core/src/system.rs`, including:

- accelerator backend
- device string
- device IDs
- precision
- CPU thread budget
- batch size
- memory budget
- workload kind

This is the correct direction.

However, many downstream modules do not consume this plan as their source of truth. Instead, they independently read env vars, pick CUDA device 0, or implement separate CUDA/HPC logic.

## Duplicate / fragmented logic families

### 1. CPU linear model versus CUDA linear model

Files:

- `crates/forex-models/src/statistical/linear_impl.rs`
- `crates/forex-models/src/statistical/linear_gpu.rs`

The CPU file owns the linear softmax artifact, CPU training loop, validation loss, early stopping, scaler, metadata, prediction, save/load, and optional CUDA fallback.

The GPU file owns its own CUDA softmax gradient, apply, loss, prediction kernels, device selection, kernel units, flattening, validation, and runtime backend label.

This is useful functionality, but it is not unified. CUDA is attached as a special case.

**Cleanup direction:** keep kernels separate, but move scheduling/device selection to a shared `DeviceScheduler`. Linear softmax should submit work units; it should not independently choose CUDA device via env.

---

### 2. CRFMNES / neuro-evolution CPU versus CUDA loss evaluation

Files:

- `crates/forex-models/src/evolution/crfmnes_impl.rs`
- `crates/forex-models/src/evolution/crfmnes_gpu.rs`

The CPU/evolution file owns optimizer state, fallback ES, CRFMNES backend, artifacts, scaler, metadata, prediction, train/val split, etc.

The GPU file owns only candidate loss CUDA evaluation, with its own CUDA device selection and kernel units.

**Problem:** this is not truly multi-device. It chooses one CUDA device and runs candidate loss there.

**Cleanup direction:** candidate evaluation is an obvious `WorkUnit`: candidate batch + feature matrix + labels. A scheduler should shard candidates across any number of available devices.

---

### 3. NEAT likely repeats the same pattern

Files found:

- `crates/forex-models/src/evolution/neat_impl.rs`
- `crates/forex-models/src/evolution/neat_gpu.rs`

This mirrors CRFMNES: CPU/evolution logic plus separate GPU evaluator.

**Cleanup direction:** NEAT and CRFMNES should share one `EvolutionEvaluationKernel` abstraction, with CPU/GPU backends.

---

### 4. Search has multiple GPU/HPC paths

Files:

- `crates/forex-search/src/cubecl_eval.rs`
- `crates/forex-search/src/cubecl_ga.rs`
- `crates/forex-search/src/discovery_gpu.rs`
- `crates/forex-search/src/hpc_gpu_discovery.rs`
- `crates/forex-search/src/hpc.rs`

These cover different pieces:

- CubeCL evaluator/backtest
- CubeCL reproduction/offspring generation
- returns-based GPU discovery
- HPC island discovery
- Hyperstack N3-specific topology helpers

**Problem:** some are canonical-ish evaluators, some are approximate presearch, and some are topology optimizations. They should not all be separate discovery paths.

**Cleanup direction:** split concepts cleanly:

- `SearchScheduler`
- `SignalKernel`
- `BacktestKernel`
- `ReproductionKernel`
- `ApproxPresearchKernel`
- `TopologyProfile`

Then each kernel can have CPU/CUDA/WGPU/ROCM implementations.

---

### 5. Hyperstack N3 HPC logic is too hardware-specific to be a general path

File:

- `crates/forex-search/src/hpc.rs`

It contains hardcoded assumptions for:

- 8x RTX A6000
- 48GB VRAM
- 252 physical cores / 504 logical threads
- NUMA socket mappings
- NVLink pairs

This is useful as a topology profile, but it should not be a separate general scheduler.

**Cleanup direction:** convert to:

```rust
HardwareTopologyProfile::HyperstackN3A6000x8
```

or dynamically detected topology hints that feed the scheduler.

---

### 6. Parallel model trainer is CPU-thread parallel, not device-aware

File:

- `crates/forex-models/src/parallel_trainer.rs`

It uses Rayon to train multiple models in parallel. This is useful, but it only manages CPU threads.

It does not know about:

- `device_ids`
- VRAM budgets
- model/device affinity
- one-model-per-GPU scheduling
- multi-GPU sharding
- avoiding multiple GPU-heavy models on the same card

**Risk:** parallel training can oversubscribe one GPU while others are idle, or launch too many GPU-heavy jobs at once.

**Cleanup direction:** make `parallel_trainer` submit `TrainingWorkUnit`s to `DeviceScheduler`.

---

### 7. Train/validation split logic is duplicated

Search found `split_train_val_indices` in multiple files:

- `statistical/linear_impl.rs`
- `statistical/bayesian_impl.rs`
- `evolution/crfmnes_impl.rs`
- `evolution/neat_impl.rs`

There is also `time_series_train_val_split` in `base.rs`.

**Risk:** different model families train/validate on subtly different splits, embargo rules, and small-dataset behavior.

**Cleanup direction:** define one `SplitContract` and one `TimeSeriesSplitter`.

---

### 8. Artifact validation boilerplate is repeated per model family

Linear, Bayesian, NEAT, CRFMNES, genetic, and other models each validate feature columns, metadata, label mapping, train/val rows, scaler dimensions, and fitted state in their own way.

**Risk:** artifacts become stricter in one model and weaker in another.

**Cleanup direction:** shared `ModelArtifactContract` and reusable validators.

---

### 9. Device selection is scattered

Examples:

- statistical CUDA reads `FOREX_BOT_STATISTICAL_CUDA_DEVICE`
- neuro-evo CUDA reads `FOREX_BOT_NEURO_EVO_CUDA_DEVICE`
- search CUDA reads `FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE`
- hardware planner already has `device_ids`

**Problem:** device choice is duplicated and env-driven inside modules.

**Cleanup direction:** only config/hardware planning reads env. Kernels receive a `DeviceAssignment` from scheduler.

---

## Desired unified hardware model

### Device inventory

```rust
pub struct DeviceInventory {
    pub devices: Vec<ComputeDevice>,
}

pub struct ComputeDevice {
    pub logical_id: usize,
    pub backend: AcceleratorBackend,
    pub vendor: DeviceVendor,
    pub memory_gb: f64,
    pub supported_precisions: Vec<TrainingPrecision>,
    pub topology: DeviceTopologyInfo,
}
```

### Work unit

```rust
pub enum WorkUnitKind {
    SearchIndicatorShard,
    SearchGeneBatch,
    SignalSynthesisBatch,
    BacktestGeneBatch,
    EvolutionCandidateBatch,
    ModelTrainingBatch,
    ModelInferenceBatch,
    FeatureComputationBatch,
}

pub struct WorkUnit {
    pub id: String,
    pub kind: WorkUnitKind,
    pub estimated_memory_gb: f64,
    pub estimated_flops: f64,
    pub deterministic_seed: Option<u64>,
    pub payload_ref: WorkPayloadRef,
}
```

### Shard plan

```rust
pub struct ShardPlan {
    pub assignments: Vec<DeviceWorkAssignment>,
}

pub struct DeviceWorkAssignment {
    pub device_id: usize,
    pub backend: AcceleratorBackend,
    pub work_units: Vec<WorkUnit>,
    pub cpu_threads: Vec<usize>,
    pub memory_budget_gb: f64,
}
```

### Backend kernel

```rust
pub trait BackendKernel {
    fn supports(&self, device: &ComputeDevice, work: &WorkUnit) -> BackendSupport;
    fn execute(&self, assignment: &DeviceWorkAssignment) -> anyhow::Result<WorkResult>;
}
```

## How search should distribute work

The search system should produce work units like:

- indicator subset shards
- gene population shards
- signal synthesis batches
- backtest/evaluation batches
- mutation/reproduction batches
- validation shards

Example:

```text
GPU 0: indicators 0..100, gene batches A-D
GPU 1: indicators 101..300, gene batches E-H
GPU 2: indicators 301..500, validation shards I-L
CPU: artifact write, checkpoint, lightweight validation, fallback work
```

The key point: this should be generated by a scheduler from available hardware and workload sizes, not by hardcoded per-file GPU logic.

## Important caution

For some workloads, splitting by indicator ranges is useful. For others, splitting by genes/candidates is safer.

Examples:

- signal synthesis can shard by genes or indicators depending on memory layout
- backtest evaluation usually shards cleanly by gene/candidate
- feature computation can shard by timeframe/symbol/indicator groups
- model training usually shards by model, batch, fold, or hyperparameter trial

Therefore the scheduler should support multiple sharding strategies:

```rust
pub enum ShardingStrategy {
    ByGeneRange,
    ByIndicatorRange,
    ByCandidateRange,
    BySymbol,
    ByTimeframe,
    ByModel,
    ByHyperparameterTrial,
    ByValidationSplit,
}
```

## What should be unified first

### First: device selection

Stop every module from picking CUDA device itself.

Create:

```rust
pub struct DeviceAssignment {
    pub backend: AcceleratorBackend,
    pub device_ids: Vec<usize>,
    pub precision: TrainingPrecision,
}
```

Then pass this into GPU kernels.

### Second: split contract

Replace duplicate `split_train_val_indices` with shared:

```rust
pub struct SplitContract {
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
    pub embargo_rows: usize,
    pub method: SplitMethod,
}
```

### Third: score semantics

Add `GeneScoreSemantics` and stop mixing backtest/classification/approx fitness.

### Fourth: scheduler wrappers

Do not rewrite kernels first. Wrap existing CPU/GPU kernels behind `BackendKernel` and `DeviceScheduler`.

### Fifth: delete/rename duplicate paths

Only after parity tests exist.

## Recommended new module layout

```text
crates/forex-core/src/hardware/
    inventory.rs
    topology.rs
    scheduler.rs
    work_unit.rs
    backend.rs

crates/forex-search/src/kernels/
    signal.rs
    backtest.rs
    reproduction.rs
    presearch.rs

crates/forex-models/src/kernels/
    linear_softmax.rs
    evolution_loss.rs
    inference.rs
```

## Migration plan

1. Keep existing kernels.
2. Add `DeviceAssignment` and pass it into current GPU functions.
3. Add `DeviceScheduler` that accepts `WorkUnit`s and returns `ShardPlan`.
4. Convert search evaluation to submit `BacktestGeneBatch` work units.
5. Convert search reproduction to submit `ReproductionBatch` work units.
6. Convert evolution candidate evaluation to submit `EvolutionCandidateBatch` work units.
7. Convert linear softmax training/prediction to submit `ModelTrainingBatch` / `ModelInferenceBatch` work units.
8. Replace per-module env CUDA device reads with scheduler assignment.
9. Rename approximate paths clearly.
10. Add parity tests and only then delete duplicate direct paths.

## Bottom line

The user’s diagnosis is correct. The repo has many places where the same conceptual work is implemented again for CPU, GPU, HPC, or a specific model family. The right fix is not another GPU file or another multi-GPU file. The right fix is one scheduler and one canonical pipeline that can distribute work over one or many devices, regardless of Nvidia/AMD/Intel backend support.
