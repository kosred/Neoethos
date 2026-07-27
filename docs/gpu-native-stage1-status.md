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
- A second test-only CubeCL specialization emits device-associated entry/exit/trade/size-cost/equity/calendar/FTMO traces for levels 4–9. Fixed and adaptive-at-entry stops, same-bar stop precedence, trailing break-even, costs, risk sizing, day/month rollover and all three FTMO flags have deterministic CPU-oracle and direct Vulkan parity coverage.
- Levels 10–12 remain in the deterministic integration harness: Prototype A supplies level-10 metrics, while the host-owned validation and canonical-ranking stages must supply explicit candidate IDs, verdicts and survivor IDs. The harness shape-checks those artifacts and never fabricates levels 11–12 from metric positions.
- `neoethos-gpu-contracts` provides host DTOs, C-compatible device POD layouts, ABI assertions and the Philox reference contract.
- Session/backend/device/generation-bound handles, explicit synchronization semantics and transfer invariants are defined by `BacktestEngine`.
- Optional NVTX ranges and attributed JSON benchmark reports separate clean timing, diagnostics, Nsight Systems and Nsight Compute passes.

## Executable benchmark/readiness paths

### Prototype A

- The existing fused CubeCL population evaluator is the exact correctness baseline.
- A real long-lived `GpuDiscoverySession<PrototypeAResources<R>>` implements the generic `BacktestEngine` contract. It owns and reuses one backend client plus resident dataset, gene, scenario, workspace, metrics and selection resources.
- Dataset, gene, scenario, metrics, selection and event handles are bound to session/backend/device/generation/buffer kind and validated at every operation.
- Evaluation consumes resident handles; chained selection stays on device, and `readback_compact` is the intentional host boundary. The direct Vulkan acceptance test proves one logical dataset upload, no dense intermediate D2H and no chained re-upload.
- `gpu_required` reaches the real population evaluator rather than acting as a decorative configuration value.
- `bench --execute-tiny` runs a deterministic GPU population workload with CPU oracle work outside timing, per-repetition zero-CPU audit and compact attributed JSON output.
- Transfer telemetry reports resident-cache hits/misses, uploads, dense/compact readbacks, chained reuploads and synchronization events. Telemetry is disabled during clean timing/profiler passes.

### Prototype B

- A deterministic subgroup-strided host reference defines the fixed/adaptive-at-entry first-hit subset and exact same-bar precedence.
- Unsupported subgroup widths return typed `UnsupportedCapability`/errors.
- The native CUDA scaffold contains a one-warp-per-event first-hit kernel using lane-strided scans and `__shfl_down_sync` earliest-hit reduction. Its C ABI, layouts, input validation and honest no-CUDA stub are host-CI verifiable; actual `nvcc` compilation and runtime parity remain gated to the rented NVIDIA session.
- A persistent population session now exists behind the same C ABI. One session owns one non-default stream, one logical dataset upload and every device workspace; `evaluate` runs the complete canonical chain on that stream — signal synthesis, causal entry emission in candidate-major/bar-ascending order, warp-cooperative first hit with exact gap/stop/target/max-hold precedence, and the f64 cost, sizing and metric reduction. Emission uses a block-wide scan so ordering is deterministic, and event-capacity overflow is a typed failure rather than a truncation. The adaptive-at-entry base distance crosses the ABI as f64 so the canonical stop distance is never narrowed.
- `PrototypeBBacktestEngine` enforces the same handle, event and transfer contracts as Prototype A, refuses populations outside the common B/C intersection before any kernel is submitted, and destroys the native session on every drop path.
- **Executed on a real NVIDIA GPU (RTX 3060 Ti, CUDA 12.2, driver 535.288.01, 2026-07-27).** `nvcc` compiles the device unit, the Rust → C ABI → CUDA smoke test runs, and the population parity test matches the canonical oracle at level 10 with `NEOETHOS_REQUIRE_GPU=1` set, so it could not have passed by skipping. Compute Sanitizer memcheck over the same path reports zero errors. Total rental cost of that gate: $0.05.
- Two build failures were found there that no CUDA-free check could reach, and both are fixed: cc-rs handed gcc flags to `nvcc` and emitted `--device-c` without a device-link step, and the `mold` linker pinned by `.cargo/config.toml` silently dropped nvcc's fatbin-registration constructors, producing a binary that built and linked cleanly and then failed every launch with "invalid device function".

### Prototype C

- A deterministic compact-event host reference preserves candidate/scenario identity and order.
- A CubeCL compact-event first-hit/stitch kernel is available for CUDA/WGPU builds for the fixed/adaptive-at-entry subset.
- The CubeCL host launcher and kernel compile in the Vulkan feature build; direct WGPU and CUDA parity probes are part of the final CI/preflight paths.
- Hosted runners without a Vulkan adapter explicitly skip only the known CubeCL no-adapter condition; every other panic or parity mismatch remains a CI failure.
- Break-even, trailing and prop-firm/path-dependent strategies route conservatively to the exact persistent walk.
- A full resident population pipeline now exists: signal synthesis, causal entry emission, compact first-hit search, deterministic non-overlapping trade stitching and the exact cost/sizing/metric reduction all run on device. Exactly one control scalar — the emitted event total — crosses to the host between kernels; it sizes the sparse passes and enforces the declared capacity before any event is written, so an over-capacity population is a typed refusal rather than an out-of-range device write. `readback_compact` is the only metrics D2H boundary.
- Device storage is shader-portable SoA `i32`/`f32` in pip space. The C-ABI structs used by native-CUDA Prototype B never enter this runtime and no native pointer crosses into CubeCL. Canonical `u64` identity lives in resident host tables and is re-attached at the compact boundary after range and finiteness validation.
- **Executed on a real AMD Vulkan adapter, not merely compiled.** Fixed-stop, adaptive-at-entry and gap-exit/daily-cap populations all match the canonical oracle at parity level 10 under the existing field-specific f32 tolerance, with guards that reject a trivially-zero comparison. Lifecycle coverage proves one logical dataset upload, zero dense readbacks, zero chained re-uploads, a deterministic repeated evaluation, a refused second dataset upload, a refused over-capacity population and a refused trailing population.

### Real-data snapshots

- A versioned JSON population snapshot schema validates OHLC, feature-major arrays, CSR genes, calendars, SMC data and explicit backtest settings.
- `neoethos-cli bench-prepare` converts canonical real-data CSV into deterministic H1/M30/M15/M5/M1 benchmark snapshots, revalidates them through the fixture the benchmark consumes, and prints SHA-256 attribution. The paid-run path is Rust-only: `preflight.sh` neither requires nor invokes `python3`, and the legacy Python helpers remain in the tree as isolated tooling.
- `bench --execute-snapshot` runs Prototype A, B or C on a real-data snapshot with the same oracle-before-timing, parity and report rules as the tiny fixture.
- The rented-GPU matrix hashes each timeframe snapshot independently and now marks Prototype A, B and C candidate jobs executable, recording the cargo feature each one requires. The historical legacy adapter remains explicitly blocked rather than fabricated.

## Rented A6000 gate

The run kit performs GPU/VRAM/RAM/disk checks, verifies CUDA, CUPTI, Nsight Systems, Nsight Compute and Compute Sanitizer, pins legacy/candidate worktrees, and runs the real Rust → C ABI → CUDA smoke/parity tests before paid measurements. The preflight also executes direct CUDA correctness probes for the warp-cooperative B subset, compact-event C subset and causal signal trace.

Verified on a rented RTX 3060 Ti on 2026-07-27, for $0.05:

- native CUDA compilation of the whole device unit;
- Rust → C ABI → CUDA smoke execution;
- Prototype B persistent population session parity against the canonical oracle at level 10, with skips turned into failures;
- Compute Sanitizer memcheck over the B population path — zero errors.

The following still require the *target* card and remain unverified:

- CUDA execution of Prototype C and the trace specialization (its population pipeline is proven on Vulkan, not yet on CUDA);
- A/B/C wall time, occupancy and VRAM measurements on an A6000-class device;
- the relative cost of compute, allocation, synchronization, H2D and D2H;
- any final architecture selection or speedup claim.

A 3060 Ti proves correctness, not performance. Nothing measured on it may inform the A/B/C choice.

## Card-independent completion status

The two implementation gaps defined in `docs/gpu-native-stage1-opus-handoff.md` are complete on the candidate branch:

- Gap A: the full-semantics Prototype-A evaluator is wired through the real handle-chained, device-resident `BacktestEngine`; no host-side map is presented as device residency.
- Gap B: diagnostic causal traces cover levels 4–9 and report their first divergence through `ParityTrace`/`compare_traces`; direct tests bypass scheduler policy and emit candidate/scenario identity from the device.
- Known CubeCL default/class-selector adapter-absence panics are caught inside client initialization while the registry lock is live, the tentative init key is removed, and callers receive typed unsupported status. Unknown panics and parity mismatches still fail.

The B/C population adapters defined in the Stage-1 B/C plan are also complete on this branch:

- Prototype B: persistent native-CUDA population session plus its `BacktestEngine` adapter, complete and **verified on real CUDA hardware** for correctness and memory safety.
- Prototype C: resident CubeCL population pipeline plus its `BacktestEngine` adapter, card-independent complete **and executed against the canonical oracle on a real Vulkan adapter**.
- One shared benchmark protocol runs both through `bench --execute-tiny` / `--execute-snapshot` with `--prototype b|c`, measuring only the supported partition and reporting the remainder as a coverage gap. Dispatch never falls back to another engine or to the CPU.

This closes the card-independent Stage-1 implementation work. It does not close the rented-A6000 evidence gate below and is not twelve-level engine parity: Prototype A proves level 10, while levels 11–12 are independently owned host-stage artifacts. Prototype C's real-hardware evidence is Vulkan-only and its metric path is f32, so it is not a substitute for the CUDA measurements.

## Explicit limitations and later-stage work

- Stage 1 does not make full discovery GPU-native. Quality screening, prop-firm windows, correlation, PBO, canonical/forward replay, robustness and other inventoried candidate-dependent CPU stages remain Stage 2.
- The full device-resident GA (generation, selection, crossover, mutation, deduplication and archive management) remains Stage 2.
- Prototype B/C currently cover the declared static/adaptive-at-entry first-hit intersection, not every strategy feature.
- Historical legacy measurement needs a pinned adapter because the historical commit predates the attributed benchmark command.
- Passing integration parity does not prove full-pipeline GPU-native execution.

## Stop gates preserved

No engine is selected, no existing backend is removed, no A6000-specific tuning is applied and no speedup is claimed before real measurements and a recorded human decision gate. Every timing figure produced so far is a local functional check, not a comparison: the Vulkan numbers come from an integrated adapter and say nothing about the A6000 or about A-versus-B-versus-C.
