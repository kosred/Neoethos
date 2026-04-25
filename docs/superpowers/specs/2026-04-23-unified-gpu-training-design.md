# Unified GPU Training Design

Date: 2026-04-23

## Goal

Establish a truthful, production-oriented GPU training architecture for the model and search stack so CPU-only paths gain real accelerator support where feasible, while tree models keep their native upstream GPU runtimes and explicit precision limits.

## Scope

### In scope

- portable Rust GPU training foundation for deep/Burn-backed models
- truthful precision/runtime metadata for GPU-capable training
- migration path for CPU-only evolutionary, RL, and search components toward a shared GPU contract
- preserving model autonomy and current inference semantics

### Out of scope

- fake claims of universal BF16/FP8 support where the upstream library does not provide it
- replacing XGBoost, LightGBM, or CatBoost with custom tree trainers
- ONNX work

## Current Reality

The repository already has partial GPU awareness:

- `crates/forex-models/src/burn_models.rs` uses Burn with optional `burn-wgpu`
- `crates/forex-models/src/rl/dqn_impl.rs` has a CUDA-oriented lane
- `crates/forex-search` has a CUDA/tch GPU discovery path
- tree models expose native upstream GPU backends through their own libraries

The main gap is that the code still mixes:

- truthful runtime metadata
- CPU-first training loops
- backend-specific GPU paths
- precision metadata that is not always tied to real tensor execution

## Design Principles

### 1. Truth over aspiration

Every model family must report:

- requested device policy
- effective device policy
- execution backend
- actual training precision
- degraded reason when GPU or lower precision was requested but not achieved

### 2. One portable GPU lane where feasible

Deep and later portable GPU workloads should converge on the Burn/CubeCL/WGPU lane already present in the repo instead of inventing more unrelated runtimes.

### 3. Native tree backends stay native

Tree models remain on their official GPU runtimes:

- XGBoost: CUDA GPU
- LightGBM: GPU/CUDA depending on upstream backend
- CatBoost: GPU

Their precision contract remains whatever the upstream library actually supports. We do not pretend they are BF16 trainers when the docs do not support that claim.

### 4. Training and inference are separate concerns

For Burn-backed deep models, the training loop may use lower precision tensors on GPU, but persisted artifacts and runtime inference remain allowed to normalize back to FP32 when that keeps save/load and inference stable.

## Approved Architecture

### Deep models

Use the existing Burn runtime as the common portable GPU training layer.

The first implementation tranche makes the Burn deep models:

- select a real tensor dtype for training
- run BF16 training when the backend and device support it
- fall back truthfully to FP32 otherwise
- keep post-training runtime models and saved artifacts in FP32 for stable inference and artifact compatibility

### RL, search, and evolutionary workloads

These workloads will follow the same runtime contract:

- request GPU explicitly
- execute on a real GPU backend when available
- emit degraded reasons when falling back

They do not all have to use the exact same internal crate today, but they must conform to the same truthfulness contract and converge toward the same hardware planner.

### Tree models

Tree models use their upstream GPU runtimes and keep explicit precision limitations in metadata and UI surfaces.

## File Ownership

### Immediate implementation tranche

- `C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs`
  - real training dtype selection
  - module/tensor casting for BF16-capable training
  - truthful precision fallback reasons

- `C:/Users/konst/development/forex-ai/crates/forex-models/src/deep_models.rs`
  - persist the real training precision contract
  - keep runtime inference and save/load coherent with the Burn training report

- `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`
  - keep orchestration metadata aligned with the new training precision/runtime truth

### Later tranches

- `C:/Users/konst/development/forex-ai/crates/forex-models/src/rl/dqn_impl.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-models/src/evolution/neat_impl.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-models/src/evolution/crfmnes_impl.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-search/src/discovery_gpu.rs`
- `C:/Users/konst/development/forex-ai/crates/forex-search/src/hpc_gpu_discovery.rs`

## Error Handling

- GPU requested but backend unavailable: train on CPU only if the runtime emits a degraded reason
- BF16 requested but unsupported by the active Burn backend/device: fall back to FP32 with an explicit reason
- FP8/BF4 requested on Burn: do not claim support unless the runtime actually executes with those dtypes

## Verification Strategy

Because the workspace disk fills quickly with repeated compiles, verification stays light:

- `rustfmt --edition 2024 --check` on touched files
- `git diff --check`
- targeted code inspection for precision/runtime coherence

Heavy cargo compile/test passes are deferred until the larger batch is ready.
