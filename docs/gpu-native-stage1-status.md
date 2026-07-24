# GPU-native discovery Stage 1 — implementation status

This record reports implemented and verified behaviour. It is not a performance claim and does not select Prototype A, B or C.

## Verified foundations

- Typed backend policy preserves `cpu`, `auto` and GPU-preferred semantics and adds strict `gpu_required`.
- Strict population evaluation forbids CPU strategy execution and uses bounded GPU-only rebatching for allocation pressure.
- Full-discovery capability preflight fails before the GA when a requested candidate-dependent stage is not GPU-native.
- Per-work-unit CPU audit records attempted and executed calls by category.
- `WrongShape` and parity violations are correctness failures, never hidden by CPU recomputation.
- Trading and discovery semantics have separate versioned canonical descriptors and hashes.
- The F-315 final SMC gate is carried through search results, checkpoints and subsequent validation/replay calls.
- Canonical integer ranking and deterministic gene serialization define exact survivor ordering.
- The causal twelve-level parity comparator reports the first divergent level with field-specific abs/rel/ULP policies.
- A separate compile-time GPU trace specialization emits score-before-threshold, raw/final signal, confidence, SMC score and SMC gate for parity levels 1–3; it is not a runtime branch in the production kernel.
- `neoethos-gpu-contracts` provides host DTOs, C-compatible device POD layouts, ABI assertions and the Philox reference contract.
- Session/backend/device/generation-bound handles, explicit synchronization semantics and transfer invariants are defined by `BacktestEngine`.
- Optional NVTX ranges and attributed JSON benchmark reports separate clean timing, diagnostics, Nsight Systems and Nsight Compute passes.

## Executable benchmark/readiness paths

### Prototype A

- The existing fused CubeCL population evaluator is the exact correctness baseline.
- `gpu_required` reaches the real population evaluator rather than acting as a decorative configuration value.
- `bench --execute-tiny` runs a deterministic GPU population workload with CPU oracle work outside timing, per-repetition zero-CPU audit and compact attributed JSON output.
- Transfer telemetry reports resident-cache hits/misses, uploads, dense/compact readbacks, chained reuploads and synchronization events. Telemetry is disabled during clean timing/profiler passes.

### Prototype B

- A deterministic subgroup-strided host reference defines the fixed/adaptive-at-entry first-hit subset and exact same-bar precedence.
- Unsupported subgroup widths return typed `UnsupportedCapability`/errors.
- The native CUDA scaffold contains a one-warp-per-event first-hit kernel using lane-strided scans and `__shfl_down_sync` earliest-hit reduction. Its C ABI, layouts, input validation and honest no-CUDA stub are host-CI verifiable; actual `nvcc` compilation and runtime parity remain gated to the rented NVIDIA session.

### Prototype C

- A deterministic compact-event host reference preserves candidate/scenario identity and order.
- A CubeCL compact-event first-hit/stitch kernel is available for CUDA/WGPU builds for the fixed/adaptive-at-entry subset.
- The CubeCL host launcher and kernel compile in the Vulkan feature build; direct WGPU and CUDA parity probes are part of the final CI/preflight paths.
- Break-even, trailing and prop-firm/path-dependent strategies route conservatively to the exact persistent walk.

### Real-data snapshots

- A versioned JSON population snapshot schema validates OHLC, feature-major arrays, CSR genes, calendars, SMC data and explicit backtest settings.
- `prepare_snapshot.py` converts canonical real-data CSV into deterministic H1/M30/M15/M5/M1 benchmark snapshots and prints SHA-256 attribution.
- `bench --execute-snapshot` runs Prototype A on a real-data snapshot with the same zero-CPU timing/audit/parity/report rules as the tiny fixture.
- The rented-GPU matrix hashes each timeframe snapshot independently. Unsupported full-population B/C jobs and the historical legacy adapter remain explicitly blocked rather than fabricated.

## Rented A6000 gate

The run kit performs GPU/VRAM/RAM/disk checks, verifies CUDA, CUPTI, Nsight Systems, Nsight Compute and Compute Sanitizer, pins legacy/candidate worktrees, and runs the real Rust → C ABI → CUDA smoke/parity tests before paid measurements. The preflight also executes direct CUDA correctness probes for the warp-cooperative B subset, compact-event C subset and causal signal trace.

The following remain unverified until a real NVIDIA GPU is rented:

- native CUDA compilation and runtime parity of Prototype B;
- actual CUDA smoke execution and Compute Sanitizer;
- CUDA execution of Prototype C and the trace specialization;
- A/B/C wall time, occupancy and VRAM measurements;
- the relative cost of compute, allocation, synchronization, H2D and D2H;
- any final architecture selection or speedup claim.

## Explicit limitations and later-stage work

- Stage 1 does not make full discovery GPU-native. Quality screening, prop-firm windows, correlation, PBO, canonical/forward replay, robustness and other inventoried candidate-dependent CPU stages remain Stage 2.
- The full device-resident GA (generation, selection, crossover, mutation, deduplication and archive management) remains Stage 2.
- The generic `BacktestEngine` residency/handle contract is implemented, but the existing full-semantics Prototype-A evaluator has not yet been completely refactored into an end-to-end handle-chained trait implementation. Current acceptance is proven through the real fused evaluator plus transfer telemetry, not a fabricated adapter.
- Prototype B/C currently cover the declared static/adaptive-at-entry first-hit intersection, not every strategy feature.
- The separate GPU causal trace currently covers levels 1–3 and SMC gating. Full per-entry/per-exit/equity/calendar trace buffers for levels 4–9 remain additional correctness instrumentation.
- Historical legacy measurement needs a pinned adapter because the historical commit predates the attributed benchmark command.
- Passing integration parity does not prove full-pipeline GPU-native execution.

## Stop gates preserved

No engine is selected, no existing backend is removed, no A6000-specific tuning is applied and no speedup is claimed before real measurements and a recorded human decision gate.
