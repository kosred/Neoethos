# GPU-First Kernel-Everywhere Architecture Report

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Scope: analysis so far and architecture requirement for GPU-first execution across the bot.

## Core requirement

The target architecture is:

```text
GPU-first everywhere.
CPU only as reference, orchestration, I/O, checkpointing, artifact writing, lightweight validation, and explicit fallback.
```

This applies to feature building, indicators, scaling, training, inference, signal synthesis, backtest, forward test, walk-forward validation, search, genetic evolution, hyperparameter search, and final candidate validation.

The CPU path remains important as a correctness reference and emergency fallback, but it should not be the default path for heavy loops.

## Main finding

The repo already contains useful GPU code and real custom kernels. The problem is that GPU execution is fragmented and often single-device. Many modules still treat GPU as a local optional path instead of as the primary compute layer managed by one scheduler.

The architecture should move from:

```text
CPU path + optional single-GPU path + separate HPC path
```

to:

```text
one canonical contract + one scheduler + many backend kernels + many devices
```

## Files analyzed so far

### `crates/forex-models/src/statistical/linear_impl.rs`

This is the CPU reference implementation for linear softmax, logistic, and elasticnet models.

Preserve:

- CPU softmax training loop
- `logits_from_features`
- `cross_entropy_loss`
- runtime metadata checks
- ElasticNet and Logistic model wrappers

Future cleanup targets:

- duplicated train/validation split
- duplicated artifact staging
- duplicated runtime metadata validation
- embedded GPU fallback bridge

GPU-first direction:

Keep it as CPU reference and model contract holder. It should not choose the device. GPU execution should be scheduler-managed.

### `crates/forex-models/src/statistical/linear_gpu.rs`

This file contains real custom CubeCL/CUDA kernels and must be preserved.

Kernels found:

- `softmax_gradient_kernel`
- `softmax_apply_kernel`
- `softmax_loss_kernel`
- `softmax_predict_kernel`

Current limitation:

- single-device only
- device selected internally via env vars
- kernel units selected internally via env vars
- validation loss kernel is not fully parallel over rows
- gradient kernel loops over all rows per parameter

GPU-first direction:

Wrap this as `LinearSoftmaxCudaKernel` or equivalent. It should receive a `DeviceAssignment` from a scheduler instead of reading CUDA device settings internally.

### `crates/forex-models/src/statistical/common.rs`

This is the statistical helper layer.

Preserve:

- `remap_three_class_labels`
- feature column checks
- CPU `FeatureScaler` as reference
- CPU `softmax_rows` as reference
- JSON artifact helpers until shared artifact writer exists

Current limitation:

- feature scaling is CPU-only
- feature matrix conversion centers CPU memory
- runtime backend fallback reports CPU when GPU is requested
- device policy parsing is statistical-only

GPU-first direction:

Move compute-heavy scaling, softmax, feature conversion, and batch preparation to GPU-aware paths. Keep CPU versions as reference/fallback.

### `crates/forex-models/src/statistical/bayesian_impl.rs`

This is CPU Bayesian one-vs-rest logistic regression with posterior variance approximation.

Preserve:

- `sigmoid`
- `fit_binary_posterior`
- `predictive_logit`
- `BayesianClassPosterior`
- `BayesianLogitExpert`

Current limitation:

- CPU-only training and prediction
- duplicated split, artifact, and metadata logic
- uses CPU scaler and CPU ndarray path

GPU-first direction:

Add a GPU backend for the three one-vs-rest classifiers. The most practical first step is to schedule classes, symbols, folds, models, or hyperparameter trials across GPUs rather than forcing one small model to span all cards.

## Custom GPU files that must not be deleted blindly

These files contain valuable custom GPU logic:

- `crates/forex-search/src/cubecl_eval.rs`
- `crates/forex-search/src/cubecl_ga.rs`
- `crates/forex-models/src/statistical/linear_gpu.rs`
- `crates/forex-models/src/evolution/crfmnes_gpu.rs`
- `crates/forex-models/src/evolution/neat_gpu.rs`

They should become backend kernel modules behind a shared scheduler.

## Required shared concepts

The repo needs shared runtime/scheduler types such as:

```rust
pub struct DeviceAssignment {
    pub backend: AcceleratorBackend,
    pub device_ids: Vec<usize>,
    pub precision: TrainingPrecision,
    pub memory_budget_gb: Option<f64>,
    pub kernel_units: Option<u32>,
}

pub enum WorkUnitKind {
    FeatureBuildShard,
    IndicatorShard,
    FeatureScaleBatch,
    SignalSynthesisBatch,
    BacktestGeneBatch,
    ForwardTestBatch,
    WalkForwardValidationShard,
    SearchGeneBatch,
    SearchReproductionBatch,
    EvolutionCandidateBatch,
    ModelTrainingJob,
    ModelInferenceBatch,
    HyperparameterTrial,
}

pub enum ShardingStrategy {
    ByGeneRange,
    ByIndicatorRange,
    ByCandidateRange,
    BySymbol,
    ByTimeframe,
    ByModel,
    ByHyperparameterTrial,
    ByValidationWindow,
    ByFeatureColumnRange,
    ByRowChunk,
}
```

Backend-specific files may still initialize CUDA, CubeCL, Burn, Candle, WGPU, or ROCm internally, but they must receive device assignment from the scheduler.

## Scheduling principle for many GPUs

A single GPU is just one device assignment. Multi-GPU is many assignments. There should not be separate business logic for single GPU and multi-GPU.

For systems with many GPUs, the highest-value parallelism is likely:

- gene/candidate batch parallelism
- symbol/timeframe parallelism
- model-family parallelism
- hyperparameter trial parallelism
- validation-window parallelism
- feature/indicator shard parallelism

Not every small model should be distributed across all GPUs. The scheduler should decide whether to shard one large job or run many independent jobs concurrently.

## Required runtime policy

Current behavior often degrades to CPU when GPU is requested. GPU-first architecture needs explicit runtime modes:

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

If GPU is required and a backend cannot satisfy the contract, this must be visible in runtime metadata. Approximate GPU presearch must not be confused with final canonical validation.

## Migration plan

1. Preserve existing custom GPU kernels.
2. Add shared scheduler/runtime types first.
3. Wrap existing kernels behind scheduler-managed interfaces.
4. Move env/device selection out of model and kernel files.
5. Add GPU-aware feature/scaler/indicator paths.
6. Make search emit work units for candidates, indicators, symbols, windows, and validation shards.
7. Keep CPU as reference and explicit fallback.
8. Add CPU/GPU parity tests before removing duplicate helpers.

## Bottom line

The project should become GPU-first across the whole bot. Existing custom CUDA/CubeCL code is valuable and should be preserved. The main work is to stop treating GPU as an optional local fallback and instead build one scheduler that distributes all heavy compute work across available devices while recording exact runtime provenance in artifacts.
