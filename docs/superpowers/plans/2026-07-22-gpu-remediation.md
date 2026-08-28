# GPU Remediation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make NeoEthos GPU execution compile reproducibly, discover real adapters, enforce honest resource limits, preserve CPU/GPU trading parity, and tune throughput from measured hardware capabilities.

**Architecture:** Keep the CPU evaluator as the canonical semantic reference. Resolve a single hardware execution plan from bounded probes plus WGPU adapter enumeration, then propagate that plan to scheduler, child process, search, and model consumers. Until CubeCL multi-device isolation is proven, schedule one full combo on one GPU and size memory accordingly.

**Tech Stack:** Rust workspace, CubeCL 0.10, wgpu 29, Windows WinRT bindings, Cargo feature gates, GitHub Actions, deterministic Rust tests.

---

### Task 1: Repair and gate the Windows WGPU dependency graph

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `.github/workflows/ci.yml`
- Test: Cargo feature compile probe

**Step 1: Record the failing probe**

Run: `cargo check -p neoethos-search --features gpu-vulkan -j1`

Expected: FAIL in `wgpu-hal` with mismatched `windows 0.61.x` and `windows 0.62.x` D3D12 types.

**Step 2: Apply the smallest coherent Windows version constraint**

Pin the workspace Windows binding to the release accepted by both `wgpu-hal 29.0.3` and `gpu-allocator 0.28.0`. Refresh only the affected lockfile packages. If this does not produce one compatible D3D12 type family, vendor `gpu-allocator` and widen its Windows constraint instead of modifying Cargo's registry cache.

**Step 3: Verify dependency coherence**

Run: `cargo tree -p neoethos-search --features gpu-vulkan -i windows@0.62.0`

Expected: the WGPU/D3D12 path resolves through the same `windows 0.62.0` package.

**Step 4: Verify the feature build**

Run: `cargo check -p neoethos-search --features gpu-vulkan -j1`

Expected: PASS.

**Step 5: Add a non-optional Windows compile gate**

Add a CI check that compiles `neoethos-search` with `gpu-vulkan` on a Windows runner even when no physical GPU is present.

**Step 6: Commit**

```text
fix: restore Windows WGPU build
```

### Task 2: Discover WGPU adapters and resolve a truthful backend

**Files:**
- Modify: `crates/neoethos-core/Cargo.toml`
- Modify: `crates/neoethos-core/src/system.rs`
- Test: `crates/neoethos-core/src/system.rs`

**Step 1: Add failing adapter-normalization tests**

Create pure tests for WGPU adapter metadata normalization, integrated/discrete classification, deduplication with CUDA/ROCm devices, and shared-memory handling. Keep real adapter enumeration behind a probe boundary so unit tests remain deterministic.

**Step 2: Run the focused tests and confirm RED**

Run: `cargo test -p neoethos-core system::tests::wgpu -- --nocapture`

Expected: FAIL because WGPU adapters are not part of accelerator discovery.

**Step 3: Implement bounded WGPU enumeration**

Enumerate adapters using wgpu's instance API, map backend/name/device type/vendor/device identifiers into `AcceleratorDevice`, and treat integrated/shared memory conservatively. Merge explicit overrides and vendor probes without duplicate physical devices.

**Step 4: Bound all external probes**

Route NVIDIA/ROCm name, memory, and capability commands through the same timeout-aware probe boundary. Make timeout/failure reasons observable.

**Step 5: Run focused and crate tests**

Run: `cargo test -p neoethos-core system::tests -- --nocapture`

Expected: PASS.

**Step 6: Commit**

```text
feat: discover WGPU adapters safely
```

### Task 3: Make scheduler memory estimates match single-device execution

**Files:**
- Modify: `crates/neoethos-core/src/scheduler.rs`
- Modify: `crates/neoethos-cli/src/main.rs`
- Test: `crates/neoethos-core/src/scheduler.rs`
- Test: CLI unit tests near discovery child construction

**Step 1: Add a failing multi-card regression test**

Given two installed cards and a worker contract that assigns one card, assert that admission requires the full population to fit on the selected card and does not divide memory by two.

**Step 2: Run focused scheduler tests and confirm RED**

Run: `cargo test -p neoethos-core scheduler::tests -- --nocapture`

Expected: FAIL because `plan_combo` currently models sharded genes.

**Step 3: Implement the single-device plan contract**

Separate installed-card inventory from per-job assigned devices. Set `genes_per_card` to the full admitted population for GPU jobs. Reject or shrink a combo based on one selected device's usable memory. Remove CLI logs or claims that imply active multi-card sharding.

**Step 4: Add a failing child CPU-budget propagation test**

Test the child command/config construction and prove that the assigned CPU budget becomes a runtime override before any global pool is initialized.

**Step 5: Implement explicit child budget propagation**

Prefer an explicit CLI option or a startup merge that applies `NEOETHOS_BOT_CPU_BUDGET` before settings-derived overrides are installed. Preserve configured values when no assignment is present.

**Step 6: Verify scheduler and CLI tests**

Run: `cargo test -p neoethos-core scheduler -- --nocapture`

Run: `cargo test -p neoethos-cli --no-default-features -- --nocapture`

Expected: PASS.

**Step 7: Commit**

```text
fix: enforce honest GPU worker budgets
```

### Task 4: Propagate planned batch, CPU, and memory limits to model consumers

**Files:**
- Modify: `crates/neoethos-models/src/training_orchestrator.rs`
- Modify: `crates/neoethos-models/src/hardware.rs`
- Modify: model-specific parameter readers as identified by the tests
- Test: `crates/neoethos-models/src/training_orchestrator.rs`

**Step 1: Add failing plan-propagation tests**

Assert that applying a `HardwareExecutionPlan` updates backend, device, precision, CPU thread count, training batch size, inference batch size, and memory budget in the parameter object consumed by trainers.

**Step 2: Run the focused tests and confirm RED**

Run: `cargo test -p neoethos-models training_orchestrator -- --nocapture`

Expected: FAIL because only backend/device/precision are currently injected.

**Step 3: Implement one canonical parameter mapping**

Add typed constants/helpers for hardware-plan keys and make every trainer read the same resolved values. Clamp inner native thread pools so outer concurrency multiplied by inner threads never exceeds the resolved CPU budget.

**Step 4: Verify focused tests**

Run: `cargo test -p neoethos-models training_orchestrator hardware -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```text
fix: apply resolved hardware limits to trainers
```

### Task 5: Strengthen GPU parity, fallback, and CI behavior

**Files:**
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-search/src/cubecl_eval.rs`
- Modify: `.github/workflows/ci.yml`
- Test: `crates/neoethos-search/src/eval.rs`

**Step 1: Extract and test the require-GPU decision**

Add deterministic tests proving that device absence skips only in ordinary CPU CI and is a hard failure when `NEOETHOS_REQUIRE_GPU=1`.

**Step 2: Add fallback classification tests**

Classify unsupported backend, no adapter, allocation pressure, device loss, and numerical parity failure. Assert that only availability failures may fall back and that the reason is returned/logged.

**Step 3: Implement minimal fallback policy**

Return structured failures from GPU client/evaluation setup. Do not accept partial GPU outputs. Preserve the CPU evaluator as the only fallback implementation.

**Step 4: Verify CPU tests and feature compiles**

Run: `cargo test -p neoethos-search --no-default-features`

Run: `cargo check -p neoethos-search --features gpu-vulkan -j1`

Expected: PASS.

**Step 5: Verify on GPU-required CI**

Run the existing CUDA and AMD GPU parity jobs with `NEOETHOS_REQUIRE_GPU=1` and retain artifacts for adapter, driver, workload shape, and parity metrics.

**Step 6: Commit**

```text
test: enforce GPU parity and fallback contracts
```

### Task 6: Tune throughput only after correctness gates pass

**Files:**
- Modify: `cubecl.toml`
- Modify: `crates/neoethos-models/src/hardware.rs`
- Add: `crates/neoethos-search/benches/gpu_eval.rs` or a feature-gated example if benches are unsuitable
- Modify: GPU runbook documentation

**Step 1: Add a reproducible benchmark harness**

Cover small/medium/large row counts and population shapes. Emit backend, adapter, driver, streams, batch size, CPU budget, elapsed time, evaluations per second, and estimated/observed memory.

**Step 2: Establish baseline measurements**

Run on representative integrated WGPU, discrete AMD, and NVIDIA hardware. Confirm parity before recording throughput.

**Step 3: Tune within memory guardrails**

Sweep stream count, batch size, and concurrency. Reject candidates that exceed the memory budget, regress parity, or slow representative workloads.

**Step 4: Document hardware-specific defaults and overrides**

Keep safe defaults portable; store measured overrides by capability class rather than marketing model name.

**Step 5: Run final workspace verification**

Run: `cargo fmt --all -- --check`

Run: `cargo check --workspace --locked -j1`

Run: `cargo test -p neoethos-core scheduler system -- --nocapture`

Run: `cargo check -p neoethos-search --features gpu-vulkan --locked -j1`

Expected: PASS, with physical-device parity/benchmark work explicitly reported when the local machine cannot execute every backend.

**Step 6: Commit**

```text
perf: tune GPU execution from measured workloads
```
