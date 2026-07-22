# GPU Remediation Design

**Date:** 2026-07-22
**Scope:** GPU build correctness, hardware discovery, scheduling/resource contracts, CPU/GPU numerical parity, fallback safety, and measured throughput.

## Problem statement

NeoEthos currently exposes GPU modes that do not form one reliable end-to-end contract. The Windows WGPU/Vulkan build is broken by incompatible `windows` crate versions. Hardware discovery does not enumerate normal WGPU adapters, so an AMD integrated GPU can remain invisible unless manually named. The scheduler estimates a population as if it were sharded across all GPUs, while the worker intentionally executes the full population on one device. Some planned CPU, batch, and memory limits are calculated but never reach their runtime consumers. GPU parity tests also skip silently on machines without a usable device.

The intended behavior is not “always use a GPU.” It is: discover usable hardware, select the fastest valid backend for the workload, apply one explicit resource plan to every layer, preserve deterministic CPU-equivalent trading semantics, and fall back to CPU with an observable reason when GPU execution is unavailable or unsafe.

## Design principles

1. **Correctness before acceleration.** GPU output must match the canonical CPU evaluator within explicit tolerances and must preserve fill, fee, stop-loss, take-profit, liquidation, CPCV, and walk-forward semantics.
2. **Capabilities, not vendor assumptions.** Runtime selection is based on enumerated adapters and supported backends, not only `nvidia-smi` or `rocminfo`.
3. **One resolved execution plan.** Backend, device, precision, CPU budget, batch size, and memory budget are resolved once and propagated to all consumers.
4. **No fictional sharding.** Memory estimates and job plans describe what the worker actually executes. Until multi-device CubeCL execution is safe, one combo uses one GPU and must fit its entire population on that GPU.
5. **Fail closed for correctness, fall back for availability.** Parity or invariant violations reject GPU results. Device absence, allocation failure, or unsupported backend may trigger an explicit CPU fallback where the caller permits it.
6. **Benchmarks inform policy; they do not replace limits.** Hardware utilization is increased only after correctness and memory guardrails are active.

## Remediation phases

### 1. Reproducible WGPU build

Unify the Windows dependency family used by `wgpu-hal` and `gpu-allocator`. Add a Windows GPU compile gate so a clean dependency resolution cannot reintroduce mismatched WinRT types. Prefer the smallest compatible version pin; vendor only if the upstream constraints cannot produce a coherent graph.

### 2. Adapter discovery and backend resolution

Enumerate WGPU adapters at runtime and normalize them into the existing accelerator inventory. Preserve CUDA and ROCm probes, but make external commands bounded and make timeout behavior explicit. Resolve `auto` using actual capabilities and a small benchmark/capability check; an environment override remains authoritative.

### 3. Honest scheduling and resource propagation

Change the scheduler contract so single-device workers size the whole population against one selected card. Remove sharded-memory claims until the worker really partitions and merges work. Propagate the CPU budget into child startup before thread pools are initialized. Carry planned batch and memory limits into the search/model runtime rather than leaving them as metadata.

### 4. Numerical and fallback gates

Keep deterministic CPU/GPU fixtures for baseline, SL/TP, fees, Monte Carlo, CPCV, and walk-forward evaluation. GPU-required CI must fail when no device is available; ordinary CPU CI may skip device execution but must still compile every supported GPU feature. Allocation/device-loss tests must prove explicit fallback or explicit failure, never partial results.

### 5. Throughput tuning

After correctness gates pass, benchmark representative matrix sizes and population shapes. Tune stream count, batch size, worker concurrency, and CPU thread budgets per resolved hardware plan. Report throughput together with peak memory and parity status so a faster but semantically different path cannot be promoted.

## Initial acceptance criteria

- `cargo check -p neoethos-search --features gpu-vulkan` succeeds on Windows from the locked graph.
- The resolved WGPU dependency graph contains compatible `windows` types for `wgpu-hal` and `gpu-allocator`.
- An available WGPU adapter can be discovered without a manually configured device name.
- A single-device scheduler never divides the estimated population memory by the number of installed cards.
- Child workers honor the assigned CPU budget before initializing global pools.
- GPU-required parity tests fail rather than skip when a GPU is required.
- CPU fallback emits the rejected backend/device and reason.
- Performance measurements include workload shape, backend/device, elapsed time, throughput, and peak/estimated memory.

## Non-goals for the first patch

- Implementing unsafe concurrent CubeCL clients across multiple devices.
- Claiming universal speedup for tiny workloads where transfer/compile overhead dominates.
- Enabling reduced precision before parity tolerances and model-specific validation are defined.
- Treating integrated-GPU shared-memory figures reported by Windows as dedicated VRAM.
