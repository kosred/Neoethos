# Forex Models Functional Audit

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: functional/refactor audit of `crates/forex-models`.

No production code was changed by this audit.

## Files inspected

- `crates/forex-models/Cargo.toml`
- `crates/forex-models/src/lib.rs`
- `crates/forex-models/src/base.rs`
- `crates/forex-models/src/runtime/mod.rs`
- `crates/forex-models/src/runtime/artifacts.rs`
- `crates/forex-models/src/runtime/profile.rs`
- `crates/forex-models/src/runtime/capabilities.rs`
- `crates/forex-models/src/runtime/prediction.rs`
- `crates/forex-models/src/runtime/dispatch.rs`
- `crates/forex-models/src/runtime/exports.rs`
- `crates/forex-models/src/runtime/hpo.rs`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/parallel_trainer.rs`
- `crates/forex-models/src/statistical/mod.rs`
- `crates/forex-models/src/statistical/common.rs`
- `crates/forex-models/src/statistical/linear_impl.rs`
- `crates/forex-models/src/statistical/linear_gpu.rs`
- `crates/forex-models/src/statistical/bayesian_impl.rs`
- `crates/forex-models/src/evolution/mod.rs`
- `crates/forex-models/src/evolution/crfmnes_impl.rs`
- `crates/forex-models/src/evolution/crfmnes_gpu.rs`
- `crates/forex-models/src/evolution/neat_impl.rs`
- `crates/forex-models/src/evolution/neat_gpu.rs`
- `crates/forex-models/src/tree_models/mod.rs`
- `crates/forex-models/src/tree_models/common.rs`
- `crates/forex-models/src/tree_models/config.rs`
- `crates/forex-models/src/tree_models/lightgbm.rs`
- `crates/forex-models/src/tree_models/xgboost.rs`
- `crates/forex-models/src/deep_models.rs`
- `crates/forex-models/src/burn_models.rs`
- `crates/forex-models/src/hardware.rs`
- `crates/forex-models/src/registry.rs`
- `crates/forex-models/src/evaluation_helpers.rs`
- `crates/forex-models/src/ensemble.rs`
- `crates/forex-models/src/genetic.rs`
- `crates/forex-models/src/exit_agent.rs`
- `crates/forex-models/src/rl/mod.rs`
- `crates/forex-models/src/rl/dqn_impl.rs`
- `crates/forex-models/src/anomaly/mod.rs`
- `crates/forex-models/src/anomaly/forest_impl.rs`
- `crates/forex-models/src/forecasting/mod.rs`
- `crates/forex-models/src/forecasting/swarm_impl.rs`
- `crates/forex-models/src/streaming/mod.rs`
- `crates/forex-models/src/streaming/adaptive_impl.rs`

## Scope limitation

Several large implementation files were partially truncated by the GitHub connector. The visible sections were enough to identify the important functional contracts and refactor risks, but this is a functional/refactor audit rather than a literal every-line proof for every truncated tail.

Large/truncated files included:

- `base.rs`
- `training_orchestrator.rs`
- `deep_models.rs`
- `burn_models.rs`
- `ensemble.rs`
- `genetic.rs`
- `exit_agent.rs`
- `rl/dqn_impl.rs`
- `forecasting/swarm_impl.rs`
- `streaming/adaptive_impl.rs`

## Core conclusion

`forex-models` has many strong pieces:

- typed runtime metadata,
- runtime prediction contracts,
- training summaries,
- HPO reports,
- ONNX export reports,
- tree fallback artifacts,
- Burn runtime provenance,
- custom GPU kernels for statistical and evolutionary models,
- strict feature-column validation in many model families.

The main problem is fragmentation.

The crate currently has several large files that own too many unrelated concepts, and device/runtime decisions are duplicated across multiple layers.

Target architecture:

```text
one runtime/provenance contract
+ one central hardware scheduler
+ small model-family modules
+ backend kernels behind explicit adapters
+ no env-driven production semantics
+ deterministic training/search/RL policies
```

## Runtime/artifacts layer

The runtime modules are generally good:

- `runtime/artifacts.rs`
- `runtime/profile.rs`
- `runtime/prediction.rs`
- `runtime/dispatch.rs`
- `runtime/exports.rs`
- `runtime/hpo.rs`

They are small, typed, and contain useful validation.

Important positive pieces:

- `RuntimeArtifactMetadata`
- `TrainingSummaryMetadata`
- `TrainingRuntimeProfile`
- runtime prediction records
- dispatch plan records
- HPO reports
- ONNX export reports
- atomic artifact writes

Main missing provenance fields:

```rust
feature_schema_hash
dataset_fingerprint
timestamp_unit
feature_availability_policy_hash
training_config_hash
runtime_config_hash
seed
hardware_profile_id
device_assignment
canonical_or_approx_mode
```

These fields are required so model artifacts can be trusted and matched safely to search/live/runtime inputs.

## Runtime capabilities

`runtime/capabilities.rs` is useful and should be preserved.

Problem: it still reads env vars for device/precision behavior:

```text
FOREX_BOT_<MODEL>_DEVICE
FOREX_BOT_META_DEVICE
FOREX_BOT_<MODEL>_TRAIN_PRECISION
FOREX_BOT_TRAIN_PRECISION
FOREX_TRAIN_PRECISION
```

These should come from a resolved runtime config and scheduler, not from model runtime helpers.

## `base.rs`

`base.rs` is useful but too large.

It includes:

- `ExpertModel`
- dataframe conversion helpers
- feature column extraction
- strict numeric validation
- sample weights
- train/validation splitting
- downsampling
- runtime prediction helpers
- runtime artifact metadata helpers
- drift detection
- early stopping

Problems:

- too many responsibilities in one file,
- env-driven early stopping/drift settings,
- timestamp validation can be weak when timestamp columns are absent.

Suggested split:

```text
base/traits.rs
base/feature_matrix.rs
base/labels.rs
base/splits.rs
base/sampling.rs
base/prediction.rs
base/metadata.rs
base/drift.rs
base/early_stop.rs
```

## Training orchestration

`training_orchestrator.rs` is the main training center.

Positive:

- uses `Settings`,
- builds `HardwareExecutionPlan`,
- writes runtime/HPO/ONNX reports,
- applies L1 feature selection on a train-prefix only,
- builds model dispatch plan.

Problems:

- it is a god file,
- it mixes data loading, feature building, label creation, feature selection, model configuration, hardware planning, HPO, ONNX export, artifact writing, and progress reporting,
- it still reads `FOREX_BOT_DATA_ROOT` directly,
- downstream model modules still make their own device decisions.

Suggested split:

```text
training/orchestrator.rs
training/data.rs
training/labels.rs
training/feature_selection.rs
training/model_config.rs
training/hpo.rs
training/onnx.rs
training/artifacts.rs
training/dispatch.rs
training/scheduler_bridge.rs
```

## `parallel_trainer.rs`

Positive:

- bounded Rayon thread pool,
- progress events,
- aggregated training summary,
- useful tests.

Problem:

It is CPU/Rayon-centric and reads thread env vars.

For GPU-first execution, this module must not decide parallelism by itself. It should receive scheduler assignments:

```rust
TrainingWorkUnit
DeviceAssignment
CpuThreadBudget
GpuMemoryBudget
```

Otherwise multi-GPU training can oversubscribe VRAM or run too many GPU-heavy models at once.

## Hardware/device policy duplication

Multiple files decide hardware/device behavior:

- `forex-core/src/system.rs`
- `forex-models/src/hardware.rs`
- `forex-models/src/tree_models/config.rs`
- `forex-models/src/runtime/capabilities.rs`
- `forex-models/src/registry.rs`
- `forex-models/src/burn_models.rs`
- individual model wrappers

This must be centralized.

Target:

```text
forex-core runtime scheduler is the source of truth
models receive DeviceAssignment / WorkloadExecutionPlan
models do not probe GPUs or parse env vars directly
```

## Tree models

Files inspected include `tree_models/common.rs`, `tree_models/config.rs`, `lightgbm.rs`, and `xgboost.rs`.

Positive:

- native backend support,
- local surrogate fallback,
- feature-column validation,
- metadata sidecars,
- runtime artifacts,
- probability normalization/calibration,
- gpu-only failure modes,
- degraded runtime details.

Problems:

- device/GPU policy is duplicated in `tree_models/config.rs`,
- GPU detection duplicates core hardware probing,
- wrapper logic is repeated across tree backends,
- local fallback artifacts must be clearly labeled as degraded runtime, not equivalent native model output.

Suggested structure:

```text
tree/common.rs
tree/config.rs
tree/backend_adapter.rs
tree/runtime_artifact.rs
tree/fallback.rs
tree/lightgbm.rs
tree/xgboost.rs
tree/catboost.rs
tree/sklears.rs
```

## Statistical models

Previously inspected statistical files show:

- `linear_impl.rs` is a useful CPU reference implementation.
- `linear_gpu.rs` contains valuable custom CubeCL/CUDA kernels.
- `bayesian_impl.rs` is CPU-only but useful.
- `statistical/common.rs` owns useful shared artifact/scaler helpers.

Do not delete custom GPU kernels.

Refactor direction:

```rust
LinearSoftmaxCpuReference
LinearSoftmaxCudaKernel
BayesianLogitCpuModel
StatisticalArtifactContract
```

GPU kernels should be called through scheduler-managed backend adapters.

## Evolution models

Previously inspected evolution files show:

- CRFMNES CPU implementation is useful as reference/orchestrator.
- `crfmnes_gpu.rs` has a custom CUDA candidate loss kernel.
- NEAT CPU implementation is useful as reference/orchestrator.
- `neat_gpu.rs` has a custom CUDA population metrics kernel.

Do not delete these kernels.

Refactor direction:

```rust
CrfmnesCandidateLossCudaKernel
NeatPopulationMetricsCudaKernel
EvolutionWorkUnit
EvolutionArtifactContract
```

Problems:

- some simple train/validation splits are duplicated,
- some runtime limits still come from env,
- GPU kernels are not yet scheduler-managed.

## Deep/Burn models

`deep_models.rs` and `burn_models.rs` are large and should be split.

Positive:

- Burn model family is mostly pure Rust,
- WGPU/CPU backend abstraction exists,
- runtime device selection is recorded,
- dtype validation exists,
- training reports exist,
- feature-column and metadata checks exist.

Problems:

- `deep_models.rs` is a large all-in-one wrapper for many model kinds,
- `burn_models.rs` contains backend selection, tensor helpers, training utilities, and many model architectures in one file,
- default Burn backend can be CPU unless WGPU feature is enabled,
- runtime selection should come from central scheduler.

Suggested split:

```text
burn/backend.rs
burn/device.rs
burn/training.rs
burn/tensor_utils.rs
burn/precision.rs
burn/models/mlp.rs
burn/models/nbeats.rs
burn/models/tide.rs
burn/models/tabnet.rs
burn/models/kan.rs
burn/models/transformer.rs
burn/models/patchtst.rs
burn/models/timesnet.rs

deep/kind.rs
deep/artifact.rs
deep/runtime.rs
deep/train.rs
deep/save_load.rs
```

## Ensemble/meta layer

`ensemble.rs` contains:

- meta blender,
- probability calibration,
- conformal gate,
- meta decision stack,
- artifact validation,
- staged writes,
- runtime predictions.

Positive:

- strong validation before save/load,
- feature-column checks,
- calibration/conformal concepts are useful.

Problem:

The file is too large and owns several distinct concepts.

Suggested split:

```text
ensemble/meta_blender.rs
ensemble/calibration.rs
ensemble/conformal.rs
ensemble/meta_stack.rs
ensemble/artifacts.rs
```

## Genetic model bridge

`forex-models/src/genetic.rs` bridges model training with `forex-search`.

Important issue:

It has two backend modes:

```rust
DiscoveryBacked
LabelSearch
```

Both persist `Gene`, but the meaning of the metrics differs.

A discovery-backed gene is evaluated by market/backtest semantics.
A label-search gene is evaluated by classification/label metrics.

These must not be treated as equivalent artifacts.

Required fix:

```rust
GeneScoreSemantics::BacktestDiscovery
GeneScoreSemantics::LabelClassification
```

or separate artifact types entirely.

Problems:

- `SmcSearchConfig::from_env()` is used,
- label search uses unseeded `rand::rng()`,
- timestamp inference/slicing logic is duplicated,
- `Gene::normalize` still has unseeded RNG in search crate.

## Exit agent / RL

`exit_agent.rs` and `rl/dqn_impl.rs` contain useful RL/exit logic and strong artifact validation.

Positive:

- runtime metadata validation,
- replay/pending state validation,
- training reports,
- reward model logic,
- fallback Q artifacts in RL.

Problems:

- unseeded RNG for exploration/sampling,
- local device policy instead of central scheduler,
- reward semantics are not explicitly tied to canonical trade/execution contracts,
- large files should be split.

Required direction:

```rust
RlTrainingPolicy
RlRewardContract
RlEpisodeBuilder
RlArtifactContract
ExitAgentRuntimePolicy
```

Training must be reproducible when a seed is supplied.

## Anomaly model

`anomaly/forest_impl.rs` is relatively clean.

Positive:

- extended isolation forest when feature is enabled,
- diagonal profile fallback otherwise,
- strict artifact validation,
- feature-column matching,
- runtime metadata.

Required improvement:

Clearly preserve degraded backend mode in runtime decisions:

```text
extended_isolation_forest
vs
diagonal_profile
```

and add shared provenance fields.

## Forecasting / swarm

`forecasting/swarm_impl.rs` has external swarm mode and local fallback mode.

Positive:

- explicit `runtime_mode`,
- explicit `runtime_degraded_reason`,
- candidate reports,
- forecast snapshots,
- repair/rebuild logic.

Problem:

It appears to use its own artifact contract rather than the common `RuntimeArtifactMetadata` contract used by other model families.

Required improvement:

Align with common runtime metadata and provenance.

## Streaming/adaptive models

`streaming/adaptive_impl.rs` contains online PA and Hoeffding logic.

Positive:

- artifact validation,
- fallback modes,
- metadata sidecars,
- feature-column matching.

Problems:

- large file with multiple model families,
- fallback/committee logic should be split,
- runtime/provenance should be aligned with common artifact contract.

Suggested split:

```text
streaming/online_pa.rs
streaming/online_hoeffding.rs
streaming/fallback.rs
streaming/artifacts.rs
```

## `evaluation_helpers.rs`

This file contains `simple_backtest` for model-level sanity evaluation.

Important note:

This is not the canonical trading/backtest evaluator.

It maps probabilities to signals and uses:

```text
signal at row i -> close[i] to close[i+1]
```

This is acceptable as a model sanity score, but it must not be confused with `forex-search` canonical backtest semantics.

Suggested rename:

```rust
model_sanity_backtest
```

or move to:

```text
evaluation/model_sanity.rs
```

## ONNX in `lib.rs`

`lib.rs` contains ONNX inference implementation.

This should move to a dedicated module:

```text
runtime/onnx.rs
```

or:

```text
inference/onnx.rs
```

`lib.rs` should remain a module registry/re-export file.

## P0 findings

1. Runtime/device decisions are duplicated across several modules instead of coming from one scheduler.
2. Model artifacts lack full provenance: feature schema hash, dataset fingerprint, timestamp policy, config hash, runtime plan, seed.
3. `GeneticStrategyExpert` can persist `Gene` artifacts with different metric semantics depending on backend mode.
4. Deep/Burn/training/ensemble files are too large and hard to maintain.
5. Unseeded RNG exists in genetic label search, exit agent, and RL paths.
6. `evaluation_helpers::simple_backtest` must not be treated as canonical backtest.

## P1 findings

1. Tree wrapper artifact/fallback logic should be centralized.
2. Burn/deep model files should be split by backend, training, artifact, and model kind.
3. `parallel_trainer.rs` should be scheduler-driven instead of thread-env driven.
4. ONNX implementation should move out of `lib.rs`.
5. Swarm/adaptive/anomaly artifacts should align fully with common runtime metadata.
6. CPU fallbacks should always record degraded reason and never look equivalent to native/GPU execution.

## Required tests

Add tests for:

- runtime metadata includes feature schema hash and dataset fingerprint,
- model artifact refuses feature schema mismatch,
- tree native backend vs local fallback records correct degraded reason,
- scheduler-driven device assignment is respected by tree/deep/evolution/statistical models,
- Burn device selection is reproducible from runtime plan,
- genetic DiscoveryBacked vs LabelSearch artifacts cannot be mixed silently,
- exit agent and RL training are deterministic with fixed seed,
- `simple_backtest` is not used as canonical evaluator,
- ONNX export report validates expected runtime metadata,
- parallel trainer respects scheduler budgets.

## Proposed target structure

```text
forex-models/src/
  lib.rs
  base/
    traits.rs
    feature_matrix.rs
    labels.rs
    splits.rs
    sampling.rs
    prediction.rs
    metadata.rs
    drift.rs
  runtime/
    artifacts.rs
    profile.rs
    prediction.rs
    capabilities.rs
    dispatch.rs
    hpo.rs
    exports.rs
    onnx.rs
    provenance.rs
  training/
    orchestrator.rs
    data.rs
    labels.rs
    feature_selection.rs
    model_config.rs
    hpo.rs
    artifacts.rs
    dispatch.rs
    scheduler_bridge.rs
  tree/
    common.rs
    backend_adapter.rs
    fallback.rs
    runtime_artifact.rs
    lightgbm.rs
    xgboost.rs
    catboost.rs
    sklears.rs
  statistical/
    common.rs
    linear_cpu.rs
    linear_cuda.rs
    bayesian.rs
  evolution/
    crfmnes_cpu.rs
    crfmnes_cuda.rs
    neat_cpu.rs
    neat_cuda.rs
  burn/
    backend.rs
    device.rs
    precision.rs
    training.rs
    tensor_utils.rs
    models/
      mlp.rs
      nbeats.rs
      tide.rs
      tabnet.rs
      kan.rs
      transformer.rs
      patchtst.rs
      timesnet.rs
  deep/
    kind.rs
    artifact.rs
    runtime.rs
    train.rs
    save_load.rs
  ensemble/
    meta_blender.rs
    calibration.rs
    conformal.rs
    meta_stack.rs
    artifacts.rs
  rl/
    dqn.rs
    rewards.rs
    episodes.rs
    artifacts.rs
  exit/
    agent.rs
    rewards.rs
    artifacts.rs
  anomaly/
    isolation_forest.rs
  forecasting/
    swarm.rs
  streaming/
    online_pa.rs
    online_hoeffding.rs
    artifacts.rs
```

## Bottom line

`forex-models` has strong model coverage and many good validation mechanisms. The next step is not to delete capability. The next step is to make it smaller, scheduler-driven, deterministic, and provenance-complete.

The custom GPU kernels should be preserved. CPU implementations should remain as reference/fallback where appropriate. Runtime artifacts must clearly record whether a model ran native GPU, native CPU, surrogate fallback, approximate mode, or degraded mode.
