# GPU-native discovery pipeline — Stage 1

**Status:** Approved architecture record  
**Stage:** Foundations and benchmark readiness  
**Baseline commit:** `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`  
**Scope owner:** `neoethos-search` discovery/backtesting pipeline

This document is the authoritative specification for Stage 1 of the NeoEthos GPU-native discovery redesign. It incorporates the approved design and all later corrections. Earlier planning notes and addenda are superseded by this integrated version.

## 1. Objective

NeoEthos must support a strict GPU execution mode in which no candidate-dependent strategy computation is silently moved to the CPU. The final system is judged by end-to-end wall time, correctness, device residency and explicit execution evidence — not merely by whether the host submitted one or more GPU kernels.

Stage 1 does **not** claim that the final GPU architecture has been selected. It creates the contracts, correctness machinery, instrumentation, executable prototypes and remote benchmark kit required to make that decision from measurements on a rented NVIDIA RTX A6000.

The design targets the following eventual invariant:

```text
GPU-required work unit:
    CPU strategy-compute executions      = 0
    CPU candidate backtests              = 0
    CPU validation simulations           = 0
    silent CPU fallbacks                  = 0
    intermediate full-result D2H copies  = 0
```

CPU work remains permitted for disk I/O, configuration, launch orchestration, UI/progress reporting, artifact serialization and compact final-result handling.

## 2. Current evidence and working hypotheses

The current implementation uses a one-worker-per-gene population backtest with a serial, branch-heavy equity walk across bars. On dense timeframes, the hybrid scheduler has measured the CPU lane as faster and has therefore routed substantial work to the CPU. A previous rented GPU session consequently spent money while the intended discovery workload ran primarily on the CPU.

The causes of the observed GPU underperformance are **working hypotheses**, not proven conclusions. Nsight profiling must quantify the contribution of:

- the serial per-gene bar walk;
- insufficient candidate-axis occupancy at normal population sizes;
- signal-matrix materialization;
- host-to-device uploads and device-to-host readbacks;
- small or fragmented batches;
- allocation and memory-pool behaviour;
- process-wide or device-wide launch serialization;
- divergent strategy paths and register pressure;
- CPU-only post-search and validation stages.

No architecture decision or speedup claim may be based on the hypotheses alone.

## 3. Stage boundaries

### Stage 1 — this specification

Stage 1 delivers card-independent and compile-verifiable foundations plus benchmark readiness:

- explicit backend and fallback policy;
- fail-fast capability checks;
- audited CPU strategy execution;
- versioned trading and discovery semantics;
- a field-specific, twelve-level parity harness;
- device-safe FFI layouts and device-resident engine contracts;
- NVTX and benchmark instrumentation;
- executable Prototype A, B and C baselines;
- native CUDA scaffolding;
- a fail-fast rented-A6000 run kit.

### Stage 2 — after A6000 measurements

Stage 2 migrates **all remaining candidate-dependent CPU computation** identified by the capability inventory. The inventory is authoritative; the following list is a minimum, not an exhaustive boundary:

- signal and minimum-trade filtering;
- quality screening and risk diagnostics;
- prop-firm window simulation;
- candidate-pool correlation and portfolio selection;
- PBO, ranking and host-side validation reductions;
- canonical and forward-tail replay;
- robustness permutation and plateau evaluation;
- host-side fold gathering or result sorting;
- GA generation, selection, crossover, mutation, deduplication and archive management;
- any future candidate-dependent CPU evaluator discovered by the audit.

### Stage 3 — separate workstream

GPU-native ML training and inference are explicitly outside Stage 1 and Stage 2. They require a separate architecture and benchmark plan.

The mesh remains functional throughout but is not the priority of this redesign.

## 4. Hard stop gates

The following actions are forbidden before real discrete-NVIDIA measurements:

- selecting Prototype A, B or C as the final architecture;
- making A6000-specific tuning decisions;
- publishing speedup claims;
- definitively deleting an existing engine;
- treating iGPU measurements as an A6000 proxy;
- changing strategy semantics to obtain faster benchmarks;
- hiding unsupported capabilities with CPU fallback.

CubeCL remains the portable correctness/reference backend and the non-NVIDIA path.

## 5. Definition of GPU-native

A work unit is computationally GPU-native only when all candidate-dependent numerical work requested by that work unit runs on an accelerator backend and the host does not recompute or complete it on the CPU.

Allowed host responsibilities:

- disk and network I/O;
- parsing and validating configuration;
- creating device sessions and submitting work;
- lightweight progress reporting;
- writing artifacts and logs;
- explicit compact readback of final survivors or debug traces.

Disallowed in strict GPU mode:

- CPU signal synthesis;
- CPU candidate backtesting;
- CPU Monte Carlo evaluation;
- CPU walk-forward/CPCV/PBO computation;
- CPU risk or prop-firm simulation;
- CPU candidate correlation or ranking;
- CPU robustness and replay backtests;
- hidden fallback after GPU failure.

## 6. Backend model

Backend selection is represented by independent policy axes rather than one ambiguous string.

```rust
pub enum DevicePreference {
    Cpu,
    Auto,
    Gpu,
}

pub enum FallbackPolicy {
    AllowCpu,
    ForbidCpu,
}

pub enum AcceleratorHint {
    Any,
    Cuda,
    Wgpu,
    Vulkan,
    Rocm,
}

pub struct EvaluationBackend {
    pub device: DevicePreference,
    pub fallback: FallbackPolicy,
    pub accelerator_hint: AcceleratorHint,
}
```

Canonical configuration mapping:

| Configuration | Device | Fallback |
|---|---:|---:|
| `cpu` | CPU | allowed |
| `auto` | automatic | allowed |
| `gpu` | GPU preferred | allowed |
| `gpu_required` | GPU required | forbidden |

For discovery, `models.prop_search_device` overrides the global `system.enable_gpu_preference`. `NEOETHOS_REQUIRE_GPU` may only escalate execution to `Gpu + ForbidCpu`; it may never downgrade a configured strict mode.

Environment boolean parsing is explicit:

```text
true:  1, true, yes, on
false: 0, false, no, off, empty, unset
```

Invalid combinations such as `Cpu + ForbidCpu` fail during configuration or preflight.

## 7. GPU failure policy

GPU failure handling is typed and mode-aware.

```rust
pub enum GpuAction {
    RetryOnGpu,
    FallbackToCpu,
    FailLoud,
}
```

Rules:

- `ParityViolation` always fails loud.
- `WrongShape` is an internal/correctness failure and always fails loud.
- `NoAdapter`, `UnsupportedBackend` and `DeviceLost` may fall back only when CPU fallback is explicitly allowed.
- `AllocationPressure` first triggers bounded GPU-only rebatching or workspace reduction.
- Strict GPU mode fails only after deterministic GPU retries are exhausted; it never falls back to CPU.

Rebatching must preserve exactly:

- candidate IDs;
- scenario IDs;
- candidate and scenario ordering;
- deterministic RNG counters;
- final output ordering and semantics.

Retry limits, attempted batch sizes and the final failure cause are reported explicitly.

## 8. CPU strategy-compute audit

All CPU strategy evaluators are routed through a central audited wrapper rather than relying on scattered guards.

```rust
cpu_strategy::run(
    backend,
    audit_context,
    category,
    call_site,
    || { /* CPU strategy computation */ },
)
```

The audit is scoped per work unit, never process-global.

```rust
pub struct CpuStrategyAuditContext {
    pub work_unit_id: WorkUnitId,
    pub attempted_by_category: Counters,
    pub executed_by_category: Counters,
}
```

Minimum categories include population evaluation, signal synthesis, candidate backtest, validation simulation, risk diagnostics, correlation/ranking and robustness/replay.

In `ForbidCpu`, an attempted CPU call is recorded and then rejected before execution. Clean strict-mode runs require zero executed CPU strategy calls.

CPU reference/parity runs execute in a separate validation mode and are never mixed into production GPU-required timing runs.

## 9. Full-pipeline capability preflight

Before the GA begins, strict GPU mode performs a typed capability preflight across every requested stage.

If any stage lacks a GPU implementation, the run fails immediately with a complete unsupported-stage list. It must never spend hours in the GA before failing in a late CPU-only gate.

During Stage 1, full discovery in strict mode is expected to report unsupported Stage 2 stages. Engine-only and already-GPU validation benchmarks may still assert the provisional zero-CPU invariant.

The capability manifest is generated from the actual pipeline inventory and includes backend, engine, strategy-feature and scenario support.

## 10. Versioned semantics

Two independent canonical descriptors prevent execution semantics from being confused with search/selection policy.

### 10.1 Trading semantics

```rust
pub const TRADING_SEMANTICS_VERSION: u32 = ...;
```

`TRADING_SEMANTICS_HASH` is derived from a versioned canonical `TradingSemanticsDescriptor`, not documentation text or source bytes. It covers at least:

- signal threshold and direction rules;
- SMC gate behaviour;
- entry timing;
- same-bar SL/TP precedence;
- fixed and adaptive stops;
- trailing and break-even rules;
- spread, commission, swap and conversion costs;
- confidence and position sizing;
- equity and drawdown state;
- daily/monthly/prop-firm calendar boundaries;
- maximum-hold and forced-exit behaviour.

### 10.2 Discovery semantics

```rust
pub const DISCOVERY_SEMANTICS_VERSION: u32 = ...;
```

`DISCOVERY_SEMANTICS_HASH` is derived from a versioned canonical `DiscoverySemanticsDescriptor`. It covers at least:

- fitness fields and formulas;
- integer/quantized ranking policy;
- tie-breaking and candidate identity;
- filtering rules;
- validation verdict rules;
- PBO/WF/CPCV selection policy;
- survivor and portfolio selection policy.

Both version/hash pairs are written to benchmark reports and validation artifacts.

The known SMC mismatch is resolved before parity expansion. A legacy-before versus canonical-after fixture records any intentional survivor change. From that point forward, every backend must reproduce the new canonical semantics; legacy survivor preservation is not promised.

## 11. Canonical ranking and identity

Floating-point tolerance cannot coexist with exact survivor ordering unless ranking uses a shared canonical key.

CPU and GPU therefore construct the same integer/quantized `RankKey` for ordering. The quantization policy is part of `DISCOVERY_SEMANTICS_HASH`.

Final tie-breaking order:

1. canonical primary rank fields;
2. canonical secondary rank fields;
3. gene signature hash as a fast discriminator;
4. canonical serialized gene bytes;
5. stable candidate/trial ID.

Canonical gene serialization must:

- sort indicator-weight pairs in a defined order;
- normalize negative zero;
- reject non-finite values;
- use explicit field widths and endianness;
- include all semantic fields;
- avoid relying solely on the existing quantized hash.

## 12. Twelve-level parity hierarchy

Parity is evaluated in causal order. A final-PnL match alone is insufficient.

1. score before threshold;
2. signal direction;
3. confidence;
4. candidate entry events;
5. candidate exit bar and reason;
6. accepted-trade sequence;
7. position size and costs;
8. equity after each accepted trade;
9. daily/monthly/prop-firm state;
10. final metrics;
11. validation verdict;
12. final survivor ordering.

Exact comparison is required for discrete state: signals, bars, reasons, IDs, accepted-trade order, verdicts and survivor order.

Scores, confidence, equity, PnL and derived floating metrics use declared field-specific absolute, relative and ULP policies. Tolerances are versioned and reported; they are not one global magic epsilon.

`compare_traces()` reports the first diverging level.

Levels 1–9 are exercised by tiny direct engine fixtures. Levels 10–12 are exercised by a deterministic Stage 1 integration fixture. Passing levels 10–12 during Stage 1 does not prove that the complete production pipeline is GPU-native.

The GPU trace path is a separate compile-time kernel specialization with separate trace buffers, not a runtime branch in the production kernel.

Integrated-GPU tests call the CubeCL/WGPU backend directly through a test-only override and do not rely on the production scheduler that intentionally skips iGPUs.

## 13. Contracts and FFI layouts

`neoethos-gpu-contracts` contains two separate representation layers.

### Host DTOs

Ergonomic Rust/Serde structures may use `Vec`, `String` and typed enums.

### Device and FFI POD

Device-facing structures use `#[repr(C)]`, fixed-width primitives and explicit offsets/counts. They contain no Rust `Vec`, `String`, native-layout enums or unbounded `bool`.

Rust and C++ compile-time assertions verify:

- total size;
- alignment;
- field offsets;
- enum/tag values;
- shared `ABI_VERSION`.

The contract covers datasets, gene CSR arrays, scenarios, outcomes, trades, metrics, prop-firm state and index maps.

## 14. Scenario descriptors and deterministic RNG

Global scenario batching uses compact device descriptors rather than cloned genes.

A descriptor includes:

- base candidate ID;
- scenario ID and type;
- seed/counter;
- window/index-map descriptor;
- cost overrides;
- parameter perturbation descriptor;
- segmented-reduction key.

A counter-based Philox-style RNG contract derives parameters on device from stable identifiers. Reordering or rebatching cannot change generated scenarios.

## 15. Device-resident BacktestEngine

The engine API must not force a host readback between pipeline stages.

Conceptual contract:

```rust
trait BacktestEngine {
    fn upload_session(...) -> Result<DatasetHandle>;
    fn upload_genes(...) -> Result<GeneBufferHandle>;
    fn upload_scenarios(...) -> Result<ScenarioBufferHandle>;

    fn evaluate(
        &self,
        dataset: &DatasetHandle,
        genes: &GeneBufferHandle,
        scenarios: &ScenarioBufferHandle,
    ) -> Result<DeviceMetricsHandle>;

    fn filter(
        &self,
        metrics: &DeviceMetricsHandle,
        policy: &DeviceFilterPolicy,
    ) -> Result<DeviceSelectionHandle>;

    fn readback_compact(
        &self,
        selection: &DeviceSelectionHandle,
    ) -> Result<HostSurvivorSummary>;
}
```

Opaque handles are bound to:

- session ID;
- backend ID;
- physical/logical device ID;
- workspace generation/version;
- buffer kind;
- parent buffer/session relationships where required.

Runtime validation rejects stale, cross-session, cross-device and cross-backend handles.

Synchronization semantics are explicit. Operations either consume/return event or fence handles, or clearly define blocking behaviour. The trait may not depend on hidden global synchronization.

Device-transfer instrumentation records:

- dataset uploads per session;
- gene/scenario uploads;
- full and compact D2H copies;
- reuploads between chained stages;
- synchronization events;
- transferred bytes.

Prototype A acceptance requires one dataset upload per session, no intermediate full-metric D2H readback, no reupload between chained stages and only explicit compact readback.

## 16. GpuDiscoverySession ownership

A session owns the accelerator context and reusable allocations for one symbol/timeframe work unit.

```text
GpuDiscoverySession
  dataset and feature buffers
  SMC and calendar arrays
  gene buffers
  scenario descriptors
  validation index maps
  reusable workspaces
  device metrics and selections
  stream/event resources
  transfer counters
```

The final scheduler uses bounded streams, reusable workspaces, explicit dependencies and memory-aware backpressure. A process-wide launch mutex may exist only as a documented migration guard and is forbidden in benchmarked prototype paths.

## 17. Benchmark methodology

The benchmark runner supports two fixture classes:

- deterministic tiny fixtures for correctness;
- hashed, representative real-data snapshots for H1, M30, M15, M5 and M1.

It performs separate passes for:

1. clean wall-time measurements;
2. diagnostic counters;
3. Nsight Systems;
4. Nsight Compute.

Diagnostics and traces must not contaminate clean timing runs.

Sweeps include:

- population and GPU batch size;
- number of bars;
- number of features;
- scenario count and density;
- fixed-row workloads;
- fixed-calendar-duration workloads.

Each report contains:

- Git SHA;
- legacy/canonical baseline identity;
- dataset and configuration hashes;
- trading and discovery semantics versions/hashes;
- seed;
- backend and prototype;
- CPU, RAM, GPU and device class;
- driver, runtime and CUDA toolkit versions;
- clocks, power and thermal state where available;
- warm-up count and measured repetitions;
- median, P95 and variance;
- candidates/s, candidate-bars/s and trades/s;
- peak VRAM;
- event density and hold-length distribution;
- transfer counts and bytes;
- parity status and capability coverage.

Unsupported metrics are emitted as `null`, never fabricated.

## 18. Prototype comparison

### Prototype A — fused exact bar walk

A persistent, exact GPU baseline that fuses signal synthesis, SMC gating, position state and metrics while keeping the dataset and intermediates device-resident.

Acceptance:

- levels 1–9 pass directly through the engine;
- levels 10–12 pass through the Stage 1 integration harness;
- one dataset upload per session;
- no dense signal readback;
- no CPU candidate post-processing;
- compact output only.

This is a correctness baseline, not a preselected winner.

### Prototype B — warp/subwarp cooperative walk

A real executable prototype in which a warp or supported subgroup cooperates on candidate work where useful. It must not claim benefit until measured.

WGPU/iGPU execution is capability-dependent. The implementation checks subgroup operations and width; unsupported devices return typed `UnsupportedCapability`. The real correctness/performance test runs on the A6000 when required.

### Prototype C — sparse event, first hit and device stitch

A real executable minimal engine for a declared strategy subset. It performs event compaction, fixed/adaptive barrier first-hit work and exact device-side stitching. Unsupported trailing or path-dependent semantics return typed status rather than panic or silently use CPU.

### Fair-comparison rules

All A/B/C measurements use identical datasets, genes, scenarios and strategy subsets.

Reports separate:

- performance on the common capability intersection;
- coverage and unsupported percentage over the complete workload.

The algorithm comparison uses the same backend. CubeCL versus native CUDA is measured as a separate axis so backend differences are not confused with architectural differences.

## 19. Native CUDA scaffold

`neoethos-gpu-cuda` provides CUDA C++/CCCL code behind a stable C ABI over the shared POD layouts.

Card-independent checks:

- CUDA C++ compilation where the toolkit exists;
- Rust/C++ link and symbol checks;
- ABI version validation;
- size/alignment/offset static assertions.

Real-GPU-gated checks:

- Rust → C ABI call;
- CUDA allocation and upload;
- one real kernel or CCCL primitive;
- readback and CPU-reference parity;
- Compute Sanitizer.

The scaffold is not declared production-ready or faster. Experimental Rust-to-PTX infrastructure is not a production dependency in Stage 1.

## 20. Rented A6000 run kit

`scripts/gpu-bench/` performs fail-fast validation before expensive work:

- GPU visibility and identity;
- driver/toolkit/runtime compatibility;
- container GPU access;
- CUDA smoke test;
- CUPTI and Nsight permissions;
- strict backend/preflight behaviour;
- zero-CPU engine audit;
- dataset and config hashes;
- sufficient disk/RAM/VRAM;
- output persistence.

The kit builds pinned legacy and candidate worktrees/images. It keeps two references:

- the historical legacy baseline at `2be1408...`;
- the canonical Stage 1 baseline after the semantics fix.

It runs baseline and prototypes in separate clean, diagnostic and profiler passes.

The final report presents a Pareto surface by timeframe, population, scenario density, coverage, VRAM and correctness. It does not automatically select one engine from a single aggregate score. A recorded human decision gate follows the measurements.

## 21. Implementation plan

### Phase 0 — backend axes, preflight and CPU audit

- **0.0:** publish this authoritative English and Greek design record.
- **0.1:** implement device preference, fallback policy, accelerator hint, configuration mapping and precedence.
- **0.2:** implement typed failure actions, WrongShape reclassification and deterministic GPU rebatching.
- **0.3:** centralize CPU strategy evaluation behind a per-work-unit audited wrapper.
- **0.4:** implement full-pipeline GPU capability inventory and fail-fast preflight.

### Phase 1 — canonical semantics and parity

- **1.0:** resolve SMC semantics, add trading/discovery descriptors, versions and hashes, record legacy/canonical delta.
- **1.1:** add field-specific parity policies, canonical RankKey and deterministic identity/order.
- **1.2:** add separate CPU/GPU trace specializations.
- **1.3:** add levels 1–9 direct fixtures and levels 10–12 integration fixtures, including direct iGPU backend invocation.

### Phase 2 — contracts and device residency

- **2.1:** add `neoethos-gpu-contracts` host DTO and device POD layers with ABI assertions.
- **2.2:** add scenario descriptors, counter-based RNG contract and segmented keys.
- **2.3:** add session-bound device handles, explicit events/fences, transfer instrumentation and CubeCL session implementation.

### Phase 3 — instrumentation and benchmarks

- **3.1:** add optional NVTX ranges for all discovery stages.
- **3.2:** add the multi-pass benchmark runner, snapshots, sweeps and machine-readable reports.

### Phase 4 — executable prototypes

- **4.1:** make Prototype A satisfy persistence, transfer and parity acceptance.
- **4.2:** implement executable minimal B and C engines with typed capability reporting and fair-comparison fixtures.

### Phase 5 — CUDA scaffold and remote kit

- **5.1:** add the CUDA/CCCL FFI scaffold and split compile-time versus real-GPU verification.
- **5.2:** add the pinned, fail-fast A6000 benchmark/profiling kit and Pareto report.

## 22. Verification by phase

### Phase 0

```bash
cargo test -p neoethos-search
cargo check --features gpu-vulkan
```

Tests cover configuration mapping/precedence, real boolean parsing, failure-action matrix, WrongShape fail-loud behaviour, rebatch determinism, per-work-unit audit counters and typed preflight output.

### Phase 1

Tests cover canonical SMC semantics, both semantic hashes, field-specific parity, canonical ranking, collision fixtures, trace separation and direct-backend iGPU parity.

### Phase 2

Tests cover POD layout assertions, handle ownership/staleness, explicit synchronization, transfer counters and device-to-device chaining without forced D2H.

### Phase 3

The CLI benchmark produces valid JSON for tiny and snapshot fixtures. iGPU-only unsupported metrics are `null`. Timing, diagnostics and profiler passes are separate.

### Phase 4

Prototype A passes direct and integration parity as defined. Minimal B/C engines execute on supported backends and return typed unsupported status elsewhere. Common-intersection fixtures are identical across engines.

### Phase 5

Card-independent CUDA compile/link/ABI checks pass where the toolkit is available. The runtime smoke test and Compute Sanitizer are explicitly GPU-gated. The run kit passes dry-run linting before rental.

## 23. Delivery protocol

Every implementation commit is delivered separately with:

- commit SHA;
- changed files and affected APIs;
- concise change report;
- exact test/verification results;
- known limitations and typed unsupported capabilities;
- an explicit note for any deviation from this approved scope.

A new architecture-planning round is opened only if code, tests or real measurements expose a blocker that changes an approved contract.

## 24. Decision authority

Correctness fixtures and measured end-to-end results are authoritative. Kernel microbenchmarks, theoretical occupancy and implementation preference do not override full-pipeline evidence.

No fixed candidate target is embedded in the architecture. Search breadth remains a function of measured throughput, available VRAM, time budget and required validation depth.
