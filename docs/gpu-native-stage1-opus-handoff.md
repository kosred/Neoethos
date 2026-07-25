# GPU-native discovery Stage 1 — Opus handoff

This handoff is operational. Do not reopen the architecture-planning cycle unless implementation or real-GPU evidence contradicts an approved contract.

## Repository state

- Repository: `kosred/Neoethos`
- Branch: `agent/gpu-native-stage1`
- Draft PR: `#13`
- Base: `master` at `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`
- No merge, engine selection, speedup claim or A6000-specific tuning is approved yet.

Read first:

1. `docs/gpu-native-redesign.md`
2. `docs/gpu-native-stage1-status.md`
3. PR #13 body

## Already implemented — do not redo

- typed backend/fallback/accelerator policy and real `gpu_required` production routing;
- bounded deterministic GPU-only rebatching and fail-loud correctness failures;
- per-work-unit CPU strategy audit and full-pipeline capability preflight;
- separate trading/discovery semantics descriptors and hashes;
- F-315 effective SMC gate propagation through checkpoints and replay/validation;
- canonical RankKey and deterministic canonical gene bytes;
- twelve-level field-specific causal parity comparator;
- signal/SMC GPU trace specialization for levels 1–3;
- GPU contracts crate, C-compatible POD ABI, Philox contract and handle validation;
- Prototype-A fused evaluator execution, transfer telemetry and tiny/snapshot benchmark commands;
- Prototype-B host reference plus native CUDA warp kernel;
- Prototype-C host reference plus CubeCL compact-event first-hit kernel;
- real-data snapshot exporter/schema, multi-pass report schema and A6000 run kit;
- read-only branch CI and explicit no-adapter handling for hosted runners.

## Remaining card-independent Stage 1 gaps

Complete these as two focused commits. Run the full read-only CI after each commit and include a short change report.

### Gap A — real Prototype-A `BacktestEngine` implementation

Implement the existing full-semantics fused evaluator behind the generic `BacktestEngine` contract without fabricating device residency.

Acceptance requirements:

- one long-lived `GpuDiscoverySession` owns/reuses the backend client and workspaces;
- opaque handles are bound to session/backend/device/generation/buffer kind and validated on every operation;
- `upload_dataset` performs one logical dataset upload per session;
- gene/scenario uploads preserve canonical IDs, ordering and deterministic RNG counters under rebatching;
- `evaluate` consumes device handles and produces a device metrics handle;
- chained operations do not perform a full intermediate D2H or re-upload;
- explicit synchronization/event semantics are observable; no hidden global synchronization;
- `readback_compact` is the only intentional host result boundary;
- transfer instrumentation proves one dataset upload, zero full intermediate D2H and zero chained reuploads;
- unsupported device filtering must return a typed status rather than silently filtering candidates on CPU;
- do not create an in-memory host map and label it device-resident.

A temporary adapter that calls the fused evaluator but round-trips dense candidate data through the host does **not** satisfy this gap.

### Gap B — causal trace levels 4–9

Extend the separate diagnostic specialization so the tiny deterministic fixture can compare:

4. entry events;
5. exit bar and reason, including same-bar precedence;
6. accepted-trade sequence;
7. position size and costs;
8. equity after each trade;
9. calendar and prop-firm state.

Acceptance requirements:

- compile-time/test-only specialization, never a runtime branch in the production kernel;
- exact ordering/identity for discrete events and declared abs/rel/ULP policies for floats;
- covers fixed and adaptive-at-entry stops, trailing, break-even, costs, sizing, day/month rollover and FTMO state;
- returns the first causal divergence through the existing `ParityTrace`/`compare_traces` API;
- direct backend test bypasses the production integrated-GPU scheduler skip;
- no-adapter is an explicit skip/typed unsupported condition; parity mismatches remain failures;
- levels 10–12 continue through the deterministic integration harness.

Do not claim twelve-level engine parity while levels 4–9 are represented only by final metrics.

## Verification order

After each remaining gap:

1. formatting check for changed Rust files;
2. `cargo test -p neoethos-gpu-contracts`;
3. `cargo test -p neoethos-gpu-cuda`;
4. `cargo test -p neoethos-search`;
5. `cargo check -p neoethos-cli`;
6. `cargo check -p neoethos-search --features gpu-vulkan`;
7. `cargo check -p neoethos-cli --features gpu-vulkan`;
8. direct WGPU Prototype-C and trace probes when a Vulkan adapter exists;
9. confirm that hosted no-adapter skips match only the known CubeCL adapter-absence signature.

Do not mark the PR ready for review until this matrix is green or an exact blocker is documented.

## Real-A6000 gate after the two gaps

Run on a rented RTX A6000 with the candidate branch pinned:

1. `scripts/gpu-bench/preflight.sh`
2. Rust → C ABI → CUDA smoke
3. direct CUDA parity for Prototype B, Prototype C and signal/trade traces
4. Compute Sanitizer
5. clean wall-time pass
6. diagnostic counters pass
7. Nsight Systems pass
8. Nsight Compute pass
9. tiny plus hashed H1/M30/M15/M5/M1 snapshots
10. common-capability A/B/C comparison and full-workload coverage report

The historical legacy adapter must be reported as blocked until implemented; do not substitute the canonical candidate baseline and call it legacy.

## Decision gate

After the A6000 results, present Pareto evidence by timeframe, population, scenario density, capability coverage, VRAM and correctness. A human records the final architecture choice. Only then should Stage 2 begin.

## Stage 2 remains out of scope

Do not migrate quality screening, prop-firm windows, correlation, PBO, robustness, host validation/ranking or the device-resident GA as part of this Stage 1 completion unless the approved scope is explicitly changed.
