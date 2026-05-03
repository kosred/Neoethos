# Custom CUDA Kernel Preservation / Scheduler Refactor Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: custom CUDA/CubeCL kernels, what must be preserved, what should move to scheduler/config, and safe migration boundaries.

## Summary

The repo does contain custom CUDA/CubeCL work that should not be deleted during cleanup.

The cleanup target is not the kernels themselves. The cleanup target is the duplicated scheduling/device/config logic around those kernels.

Correct rule:

```text
Keep custom kernels.
Move device selection, env flags, sharding, backend choice, and work distribution into one scheduler.
```

## Files with custom CUDA/CubeCL kernels

Search found five primary files that directly create CUDA devices/clients:

- `crates/forex-search/src/cubecl_eval.rs`
- `crates/forex-search/src/cubecl_ga.rs`
- `crates/forex-models/src/statistical/linear_gpu.rs`
- `crates/forex-models/src/evolution/crfmnes_gpu.rs`
- `crates/forex-models/src/evolution/neat_gpu.rs`

These are kernel files, not simple duplicate code.

## Kernel inventory

### 1. Search CubeCL evaluator

File:

- `crates/forex-search/src/cubecl_eval.rs`

Purpose:

- signal/evaluation/backtest GPU acceleration for strategy search.

Preserve:

- CubeCL kernel implementation
- flattened buffers
- metric output layout until typed metrics replace it
- backend-specific launch logic

Move out:

- device selection
- env flags
- fallback policy
- precision policy
- CPU/GPU semantic decisions

Required wrapper:

```rust
BacktestKernel::CubeClCuda
```

---

### 2. Search CubeCL reproduction / offspring generation

File:

- `crates/forex-search/src/cubecl_ga.rs`

Purpose:

- custom CUDA/CubeCL offspring generation via blend/mutate kernel.

Preserve:

- `blend_mutate_kernel`
- `CudaReproductionBatch`
- bf16/fp32 kernel launch support
- batch flattening/unflattening

Move out:

- `device_id` decision
- env kernel units
- precision policy source
- child-count shard splitting

Required wrapper:

```rust
ReproductionKernel::CubeClCuda
```

---

### 3. Linear softmax CUDA kernels

File:

- `crates/forex-models/src/statistical/linear_gpu.rs`

Purpose:

- custom softmax gradient/apply/loss/predict CUDA kernels.

Preserve:

- gradient kernel
- apply kernel
- loss kernel
- predict kernel
- flattening and shape checks

Move out:

- `statistical_cuda_kernel_enabled`
- per-model CUDA device env selection
- kernel unit env selection
- runtime backend string decision
- CPU fallback decision

Required wrapper:

```rust
ModelKernel::LinearSoftmaxCuda
```

---

### 4. CRFMNES / neuro-evolution candidate loss CUDA kernel

File:

- `crates/forex-models/src/evolution/crfmnes_gpu.rs`

Purpose:

- custom candidate-loss evaluation kernel for neuro-evolution.

Preserve:

- `candidate_loss_kernel`
- candidate/feature/label flattening
- train/validation loss evaluation

Move out:

- `neuro_evo_cuda_kernel_enabled`
- CUDA device env selection
- kernel unit env selection
- candidate batch sharding

Required wrapper:

```rust
EvolutionLossKernel::CrfmnesCuda
```

---

### 5. NEAT population metrics CUDA kernel

File:

- `crates/forex-models/src/evolution/neat_gpu.rs`

Purpose:

- custom NEAT graph/population flattening and CUDA population metrics kernel.

This is especially important and must not be thrown away. It handles graph-like NEAT structures by flattening population topology into buffers.

Preserve:

- `neat_population_metrics_kernel`
- activation mapping
- graph/topology flattening
- CSR-like edge/source/weight representation
- scratch buffer design
- population metrics output

Move out:

- `neat_cuda_kernel_enabled`
- CUDA device env selection
- kernel unit env selection
- train/validation shard choice

Required wrapper:

```rust
EvolutionLossKernel::NeatCuda
```

## What must not happen

Do not delete `*_gpu.rs` files blindly.

Do not merge custom kernel code into large CPU files.

Do not rewrite kernels until parity tests exist.

Do not remove approximate GPU paths until their role is replaced by scheduler-managed approximate kernels.

## What should happen

### Step 1: Extract common device assignment

Add a shared type:

```rust
pub struct DeviceAssignment {
    pub backend: AcceleratorBackend,
    pub device_ids: Vec<usize>,
    pub precision: TrainingPrecision,
    pub kernel_units: Option<u32>,
}
```

### Step 2: Change kernel entrypoints

Move from:

```rust
fn try_x_cuda(..., policy: &str)
```

or env-based device lookup to:

```rust
fn try_x_cuda(..., assignment: &DeviceAssignment)
```

The kernel may still use CUDA internally, but it must not decide which CUDA card to use by reading env itself.

### Step 3: Add shard-aware wrappers

For candidate/gene/population workloads:

```rust
fn evaluate_sharded(
    work: CandidateBatch,
    assignments: &[DeviceAssignment],
) -> Result<Vec<CandidateMetrics>>
```

The scheduler splits work across devices.

### Step 4: Keep kernel code backend-specific

It is fine for CUDA, WGPU, ROCm, CPU kernels to live in separate files. The important point is that they implement one shared trait and receive assignments.

Example:

```rust
trait CandidateLossKernel {
    fn supports(&self, assignment: &DeviceAssignment) -> BackendSupport;
    fn evaluate(&self, batch: &CandidateBatch, assignment: &DeviceAssignment) -> Result<Vec<LossMetrics>>;
}
```

### Step 5: Stop treating single-GPU and multi-GPU as separate systems

A single GPU is just a scheduler plan with one `DeviceAssignment`.

A multi-GPU system is a scheduler plan with many `DeviceAssignment`s.

No separate business logic should exist for `single_gpu` vs `multi_gpu`.

## Migration table

| Current file | Keep? | New role |
|---|---:|---|
| `cubecl_eval.rs` | Yes | `BacktestKernel::CubeClCuda` |
| `cubecl_ga.rs` | Yes | `ReproductionKernel::CubeClCuda` |
| `linear_gpu.rs` | Yes | `ModelKernel::LinearSoftmaxCuda` |
| `crfmnes_gpu.rs` | Yes | `EvolutionLossKernel::CrfmnesCuda` |
| `neat_gpu.rs` | Yes | `EvolutionLossKernel::NeatCuda` |
| `hpc.rs` | Partially | `TopologyProfile`, not separate scheduler |
| `discovery_gpu.rs` | Rename/Wrap | `ApproxPresearchKernel`, not final discovery |
| `hpc_gpu_discovery.rs` | Rename/Wrap | `ApproxPresearchKernel` + topology-aware scheduler |

## Deletion policy

A GPU/HPC file can only be deleted when:

1. its custom kernels are moved or wrapped,
2. all callers use scheduler-managed entrypoints,
3. parity tests exist,
4. old and new outputs match within contract tolerance,
5. artifact metadata records the same backend/precision/device assignment.

Until then, keep the files.

## Bottom line

The user is right to warn about custom CUDA code. The repo does contain valuable kernels. The cleanup should not remove them. The correct cleanup is to preserve kernels as backend implementations while unifying scheduling, device assignment, sharding, env/config handling, and artifact reporting.
