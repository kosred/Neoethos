# CUDA builds, GPU residency, and verification

CUDA is the active accelerator route. Vulkan/WGPU is retired; ROCm/HIP is not
a connected production backend. This guide describes build capabilities and
remaining runtime boundaries, not a claim of complete GPU readiness.

## Hardware coverage

GeForce RTX 30, RTX 40, and RTX 50 use compute capabilities 8.6, 8.9, and 12.0,
respectively. A native release can include all three exact targets:
`86;89;120`. This also covers listed workstation cards with those capabilities.
It does not promise compatibility with an unknown future architecture. See
[NVIDIA's current GPU table](https://developer.nvidia.com/cuda/gpus).

The amount of work admitted must depend on the selected device's free VRAM,
dataset shape and allocator limits, not its marketing name. Supporting an
architecture does not imply identical model sizes, speed or numerical parity
on every card. A matching driver is needed for execution, not for offline
compilation with an explicit supported architecture.

## Local compilation without a card

Use the repository's pinned Rust toolchain, a supported host C++ compiler, and
a compatible CUDA toolkit with `nvcc`, headers, link libraries and `cuobjdump`.
Official [Windows installation guidance](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
also describes toolkit components; a driver installation is not required
merely to prepare CUDA objects.

The currently locked `cudarc 0.19.9` build script recognizes CUDA through 13.3.
CUDA 13.4 can compile native NeoEthos units, but its successful version response
is rejected by that dependency's automatic version detector. A separate 13.3
toolkit avoids changing the dependency graph or mislabeling compiler versions.

Example process-local PowerShell environment; replace the toolkit path:

```powershell
$taskCuda = 'C:\toolchains\cuda-13.3'
$env:CUDA_PATH = $taskCuda
$env:CUDA_HOME = $taskCuda
$env:PATH = (Join-Path $taskCuda 'bin') + ';' + $env:PATH
$env:CUDACXX = Join-Path $taskCuda 'bin\nvcc.exe'
$env:CUDAOBJDUMP = Join-Path $taskCuda 'bin\cuobjdump.exe'
$env:CUOBJDUMP = $env:CUDAOBJDUMP
$env:NEOETHOS_CUDA_BUILD_MODE = 'cross_release_explicit'
$env:NEOETHOS_CUDA_ARCHS = '86;89;120'
```

The native build policy rejects legacy `CUDA_ARCHS`, free-form flags and
ambiguous compiler authorities. Leave `NVCC`, `NVCC_ARGS`, `CUDA_FILTER`,
`CUDA_KERNEL_DIR`, `DOCS_RS` and other legacy CUDA overrides unset.
The repository builder selects the conforming MSVC preprocessor explicitly.
Do not suppress CCCL diagnostics or enable fast math to make a build pass.

Compile the full default workspace first, then the independently gated CUDA
surfaces; these commands compile tests but do not execute them:

```text
cargo test --locked --workspace --all-targets --no-run --features neoethos-trader/ml-blend,tauri/custom-protocol --jobs 2
cargo test --locked -p neoethos-models --no-default-features --features neuro-evolution-gpu,statistical-gpu,burn-cuda-backend --lib --no-run --jobs 2
cargo test --locked -p neoethos-search -p neoethos-gpu-cuda --features neoethos-search/gpu-cuda --all-targets --no-run --jobs 2
```

Use `--offline` when dependencies are already cached. Reuse the generated
test executables for selected hardware-independent tests instead of rebuilding
for every filter. Native-linked tests may still need driver DLLs just to load;
a linked executable is not evidence of successful test execution.

## Feature boundaries

| Feature | Compiled surface; not automatic runtime acceptance |
|---|---|
| Search `gpu-b-adapter` | Rust adapter with explicit no-CUDA stub |
| Search `gpu-b-native` | Native CUDA archive and vector-ta cubins |
| Search `gpu-cuda` | Native CUDA, vector-ta and CubeCL |
| Models `neuro-evolution-gpu,statistical-gpu,burn-cuda-backend` | Evolutionary, statistical and Burn CUDA paths without tree/RL CUDA builds |
| CLI `gpu-nvidia-full` | Broad Search/Models CUDA aggregate, including Burn explicitly |

Generated CUDA MLP configurations use `capacity_mode=auto`: actual input rows,
full validation rows and live free VRAM determine the admitted hidden-width
range. HPO explores `capacity_fraction` within that range instead of reusing
the old fixed width catalogue. The effective width is saved with the model;
loading does not resize its weights for a different card. An explicit user
`hidden_dim` override selects fixed capacity unless auto was explicitly chosen.
CPU configuration is unchanged. This is an MLP-specific implementation, not
automatic scaling of every neural/tree/RL family. Its training-memory estimate
includes optimizer/validation work and headroom, but is not a hard allocator
reservation against competing processes.

The admitted range also respects the training plan's memory budget. Existing
sealed handoffs keep their original budget; mixed-device plans currently use
the smallest detected device. Moving such a plan to a larger card does not
automatically enlarge that authority. Replanning for the selected device is
required before claiming that the additional VRAM is available to the trial.

The broad aggregate has additional native dependencies. In particular, the
current LightGBM builder does not produce a CUDA learner on Windows. Do not
claim all-model CUDA readiness from a Windows aggregate build. Validate the
intended Linux deployment and each selected family separately.

The pinned native Discovery orchestration is currently Linux-gated. Its
research wiring is Generation-0-only for admitted legacy objectives; current
goal-bound Risky runs remain explicitly refused pending versioned native
support. The production prepared-discovery entry point also refuses an
unintegrated full native pipeline. A successful build must not remove these
boundaries.

## What GPU end-to-end means here

Keep the large feature store, candidate population, indicator intermediates,
backtest state and model tensors on the selected GPU across repeated work.
Transfer compact control inputs and final metrics/artifacts, not entire
matrices after every generation. Uploading a dataset once per evaluation call
is not the same as retaining it across an entire evolutionary run.

Parallelize independent candidates, scenarios, folds and model work within
measured memory/compute limits. Preserve chronological price/fill/ledger
ordering inside each individual backtest. Large independent batches can
occupy the GPU without violating that causality. More simultaneous streams
are not automatically faster when memory bandwidth or registers are saturated.

CPU orchestration, disk/network ingestion, UI and artifact persistence remain
host responsibilities. The performance objective is to remove repeated
host/device transfer and host computation from the heavy inner loop, not to
claim that all operating-system work runs on a graphics card.
[NVIDIA's optimization guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
prioritizes measured hotspots, data residency and sufficient parallel work.

## Acceptance on real hardware

Record the exact commit/source hashes, toolkit, driver, device, dataset,
timeframe, seeds, population, candidate/fold counts, model shape and risk mode.
Measure the same complete workload on CPU and CUDA. Account for preparation,
transfers, kernel work, validation and artifact writing; report warm-up/JIT
separately rather than silently excluding it.

Require actual kernel traces, host/device transfer bytes and time, launch gaps,
occupancy/resource limits and peak memory. An allocated CUDA context, a
nonzero VRAM reading, a source-contract test or a compile result is not GPU
execution evidence. Use Nsight Systems for timeline/transfer attribution and
Nsight Compute for kernel resource/throughput analysis.

Check trading math against independent known outcomes as well as CPU/GPU
comparison. Preserve precision-sensitive indicators and financial arithmetic;
do not replace them with lower precision merely to increase utilization.
Model-family precision and tolerances need their own recorded acceptance.

Only after this evidence can a speedup or larger supported workload be claimed.
Hours-to-minutes is an end-to-end benchmark target, not a guarantee inferred
from CUDA-core count.
