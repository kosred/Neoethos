# SEARCH Resident Generation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the diagnostic compatibility path from production generation evaluation, then connect complete SEARCH generations through the resident CUDA metrics/rank/selection/offspring pipeline without changing search results.

**Architecture:** Stage S1 introduces a one-shot host-visible metrics-only result solely as a measured migration boundary: one terminal metric readback, no outcome ledger or accepted-trade scalar. Stage S2 replaces that terminal boundary with stream-ordered device ranking and evolution across the complete run. Every stage is parity- and profiler-gated on the RTX 3090 before the next starts.

**Tech Stack:** Rust 2024, native CUDA C++17, Cargo nightly-2026-04-07, CUDA sm_86 with `--fmad=false`, Nsight Systems, Nsight Compute, Compute Sanitizer.

---

## Chunk 1: Stage S1 metrics-only production evaluation

### Task 1: Freeze VPS preimages and observe RED

**Files:**
- Inspect: `crates/neoethos-search/src/gpu_native/prototype_b_population_eval.rs`
- Inspect: `crates/neoethos-gpu-cuda/src/population.rs`
- Inspect: `crates/neoethos-gpu-cuda/native/prototype_b_population.cu`
- Inspect: `crates/neoethos-gpu-cuda/src/lib.rs`
- Inspect: `crates/neoethos-gpu-cuda/build.rs`
- Create test: `crates/neoethos-search/tests/production_metrics_only_generation_v1_contract.rs`

- [ ] Record SHA-256, byte count, line-ending mode, and relevant source excerpts for every file that any S1 task may modify. Record an explicit `ABSENT` receipt for every planned new file.
- [ ] Create reviewed forward and reverse patches before applying any production hunk. A compile, parity, sanitizer, receipt-validation, or profiler failure restores the exact preimage before another task starts.
- [ ] Write a source contract requiring the production adapter to call a one-shot metrics-only host-result API and forbidding `.evaluate(`, `.wait(`, `.read_metrics(`, diagnostic outcomes, and accepted-trade readback inside that adapter.
- [ ] Run the focused test directly and observe the expected RED against the current compatibility path.
- [ ] Archive the RED command, exit status, and complete output under `target/audit-logs/search-resident-generation-v1/s1-red/`.

### Task 1A: Preserve a real before baseline

**Files:**
- Create receipts under: `target/audit-logs/search-resident-generation-v1/s1-before/`

- [ ] Build and hash the exact preimage sm_86 binary with the pinned toolchain and strict CUDA flags.
- [ ] Run a deterministic compute-only population fixture with exact CPU/device metric parity. Label it `ComputeOnlyNotFinancialSearch` and capture cold/warm Nsight traces.
- [ ] Resolve a valid sealed broker-financial-truth input containing the required synchronized bid/ask, conversion legs, symbol contract, unrealized-PnL, and close-deal reconciliation authorities. Never bypass or mock the production gate.
- [ ] If valid financial truth is available, run and archive the end-to-end before benchmark with exact dataset/config/seed/population/generations/repetitions. If it is unavailable, record a typed blocker; S1 may be device-validated but cannot claim an end-to-end production speedup or advance to S2 production routing.

### Task 2: Add a one-shot metrics-only host result

**Files:**
- Modify: `crates/neoethos-gpu-cuda/native/prototype_b_population.cu`
- Modify: `crates/neoethos-gpu-cuda/src/population.rs`
- Test: `crates/neoethos-gpu-cuda/src/resident_population_session_v3_device_tests.rs`

- [ ] Write a CUDA device test requiring exact metric rows for multiple scenarios while reporting zero outcome bytes, zero outcome-seed launches, zero accepted-trade atomics/readbacks, exactly one terminal synchronization, and one bounded metric-row D2H.
- [ ] Run it against the current API and observe RED because no metrics-only multi-row terminal consumer exists.
- [ ] Add a native one-shot consumer that validates the resident receipt, synchronizes the recorded event once, copies exactly `scenario_count * sizeof(NeoPopulationMetricRow)`, accounts that transfer, and restores strict-idle state.
- [ ] Add a sealed move-only Rust result owner. It consumes `self`, is neither `Default` nor `Deserialize`, keeps ordered rows private, and binds session, pending-event, plan, scenario order/count, view/gene/scenario/settings/build identities, transfer accounting, and receipt identity. Do not expose raw device pointers or a caller-mintable completion receipt.
- [ ] Add RED then GREEN tests for stale event, cross-session receipt, wrong row count/order, caller-construction resistance, native error, validation error, and unconsumed-drop poisoning. `InFlight -> StrictIdle` occurs only after successful readback and Rust receipt validation; every failure poisons the session.
- [ ] Run focused Rust tests and the isolated sm_86 NVCC compile; require warning-free output.
- [ ] Run the device test, Compute Sanitizer memcheck, and exact parity fixture.

### Task 3: Route production evaluation through S1

**Files:**
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_population_eval.rs`
- Test: `crates/neoethos-search/tests/production_metrics_only_generation_v1_contract.rs`
- Test: existing Prototype-B parity and residency suites

- [ ] Replace compatibility `evaluate -> wait -> read_metrics` with `enqueue_metrics_only_v1 -> consume_host_metrics_v1`.
- [ ] Preserve telemetry identities and add counters that distinguish one terminal metric readback from forbidden diagnostic readback.
- [ ] Run the source contract and observe GREEN.
- [ ] Run warning-denied crate tests and all Prototype-B exact parity tests.
- [ ] Run the RTX 3090 parity fixture and Compute Sanitizer.
- [ ] Archive postimage hashes and a reverse patch.

### Task 4: Measure S1 before/after

**Files:**
- Create receipts under: `target/audit-logs/search-resident-generation-v1/s1-profile/`

- [ ] Build one exact sm_86 binary and record binary/native source hashes.
- [ ] Run cold and warmed Nsight Systems captures on identical input, population, scenarios, bars, seed, and repetitions.
- [ ] Record kernel geometry, kernel count, H2D/D2H counts and bytes, syncs, idle gaps, SM activity, and wall p50/p95.
- [ ] Run Nsight Compute on `population_reduce_kernel` and record waves per SM, occupancy, scheduler, SOL, and memory sections.
- [ ] Declare S1 GREEN only if parity is exact, all diagnostic allocations/seed/readbacks are zero, and the profiler receipt is internally consistent. A neutral or slower result is retained honestly and informs S2; it is not hidden.

## Chunk 2: Stage S2 fully resident generations

### Task 5: Inventory the missing build, ownership, scoring, and production bridges

**Files:**
- Existing test: `crates/neoethos-search/tests/gpu_resident_generation_pipeline_v1_contract.rs`
- Create later: `crates/neoethos-search/src/gpu_full_discovery/gpu_resident_generation_pipeline_v1.rs`
- Modify: `crates/neoethos-search/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/build.rs`
- Existing low-level module: `crates/neoethos-gpu-cuda/src/resident_generation_v1.rs`
- Existing native source: `crates/neoethos-gpu-cuda/native/resident_generation_v1.cu`
- Existing ABI: `crates/neoethos-gpu-cuda/native/resident_generation_v1_abi.cuh`
- Existing dormant scoring source: `crates/neoethos-gpu-cuda/native/resident_scoring_novelty_v1.cu`
- Existing scoring ABI: `crates/neoethos-gpu-cuda/native/resident_scoring_novelty_v1_abi.cuh`

- [ ] Run the existing resident-generation source contract and preserve the RED showing the production module is absent.
- [ ] Add RED source/link contracts proving the resident-generation Rust module is not exported and both resident-generation and scoring/novelty CUDA/ABI sources are absent from `build.rs` rerun/build inputs.
- [ ] Inventory every required exact authority, allocation receipt, device primitive, final-output owner, feature gate, exported symbol, and build manifest identity before creating or registering any module.
- [ ] Document the incompatible current seam: `ResidentPopulationMetricsV1<'session>` owns the population session/event, while resident generation currently accepts `ResidentScoredDecisionRowsEventImportV1` that assumes scoring/novelty decision keys already exist.
- [ ] Write RED contracts for a crate-private move-only population handoff, device scoring/novelty decision-key production, exact scenario/order/scoring identities, same-stream event dependency, and caller-construction resistance.
- [ ] Write a failing production integration test proving the current loop still performs host metric readback, ranking, novelty/archive decisions, and evolution.

### Task 6: Register and compile the dormant resident-generation native module

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/build.rs`
- Modify only for evidenced compile defects: `crates/neoethos-gpu-cuda/src/resident_generation_v1.rs`
- Native source: `crates/neoethos-gpu-cuda/native/resident_generation_v1.cu`
- Native ABI: `crates/neoethos-gpu-cuda/native/resident_generation_v1_abi.cuh`
- Native scoring source: `crates/neoethos-gpu-cuda/native/resident_scoring_novelty_v1.cu`
- Native scoring ABI: `crates/neoethos-gpu-cuda/native/resident_scoring_novelty_v1_abi.cuh`

- [ ] Register the resident-generation Rust module under the exact CUDA feature and add both native `.cu` files plus both ABI rerun inputs to the build manifest.
- [ ] Compile the smallest warning-denied Rust feature matrix and isolated strict sm_86 CUDA object before connecting production.
- [ ] Run symbol/ABI/source-closure tests and archive build resource usage.

### Task 7: Build the move-only metrics-to-decision ownership bridge

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/population.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_generation_v1.rs`
- Create: `crates/neoethos-search/src/gpu_full_discovery/gpu_resident_generation_pipeline_v1.rs`
- Modify: `crates/neoethos-search/src/lib.rs`

- [ ] Add a failing deterministic device fixture covering exact device scoring/novelty keys, stable total ordering, ties, deduplication, selection, crossover, mutation, and Philox draw addressing.
- [ ] Add a crate-private move-only handoff that consumes the population metrics owner without revealing a pointer, event, or caller-supplied semantic hash.
- [ ] Resolve ownership without a self-reference: either move the entire owned `PopulationSession` into the resident-generation run or make the run lifetime-parameterized over the borrowed session. A successful path may not use `mem::forget`, leak a `Box<dyn Any>`, manufacture `'static`, or store a pointer into its own movable owner.
- [ ] Produce scoring and novelty decision keys on device with exact bound identities. Fail closed for any scoring/novelty semantics that have no exact GPU authority.
- [ ] Preserve the population stream and event dependency through the imported decision rows; never synchronize merely to cross Rust ownership types.
- [ ] Implement the run-scoped owner for population, genes, metrics, scratch, stream, identities, and one allocation lifetime.
- [ ] Consume metrics-only completion through an event dependency; launch ranking/selection/offspring without host wait or metric D2H.
- [ ] Close the complete loop for every generation: evaluate the current device gene buffer, score/novelty on device, rank/select/evolve, swap current/next device gene buffers, and evaluate the next generation. After initial upload there may be no CPU gene packing and no gene/scenario H2D inside the generation loop.
- [ ] Seal a move-only final outcome with bounded final diagnostics and exact content identities.
- [ ] Run source contracts, warning-denied compilation, NVCC, exact CPU/device generation parity, memcheck, and racecheck.

### Task 8: Replace the host generation loop

**Files:**
- Modify: `crates/neoethos-search/src/genetic/search_engine.rs`
- Modify: `crates/neoethos-search/src/discovery.rs`
- Test: resident generation integration and historical search authority suites

- [ ] Write a failing integration test proving production still executes per-generation CPU ranking/evolution and metric readback.
- [ ] Route GPU-required discovery through the sealed resident-generation owner for the exact supported selection semantics.
- [ ] Fail closed for unsupported selection/evolution semantics; do not fall back to CPU when a CUDA-required run has started.
- [ ] Keep the existing CPU-only search route unchanged.
- [ ] Run identical-seed CPU-reference/device parity on every generation identity and final selected content.
- [ ] Run the exact historical authority suite. Do not weaken `BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1`.

### Task 9: Measure S2 and enforce GREEN or restore

**Files:**
- Create receipts under: `target/audit-logs/search-resident-generation-v1/s2-profile/`

- [ ] Capture cold and warm Nsight Systems traces for the same S1 workload.
- [ ] Require per-generation metric D2H, explicit host wait, and host decision counts to be zero.
- [ ] Require exact every-generation/final parity, zero invariant reuploads, zero unexpected transfer-accounting drift, warning-free build, memcheck/racecheck zero errors/leaks, useful GPU duty cycle at least 80%, idle-gap p95 at most 1 ms, and internally consistent allocation/event receipts. Any failure restores the last verified S1 postimage, which is the exact S2 preimage, before S3.
- [ ] Report end-to-end wall p50/p95, GPU duty cycle, idle-gap p95/max, blocks, waves/SM, achieved occupancy, transfer overlap, and every pass/fail threshold.
- [ ] Compare unchanged population first. Only after parity and profiling, test semantically independent in-flight scenario/fold/search batching.
- [ ] Do not change default population, generations, or seed based solely on CUDA-core count.

## Chunk 3: Stage S3 occupancy and workload geometry

### Task 10: Fill the card with semantically independent work

**Files:**
- Modify only after measured S2 evidence identifies the exact scheduler seam.
- Create receipts under: `target/audit-logs/search-resident-generation-v1/s3-profile/`

- [ ] Preserve the unchanged-population S2 profile as the baseline.
- [ ] Inventory independent work already required by the same search: folds, costs, robustness treatments, time windows, or independent searches. Do not fabricate scenarios and do not change their aggregation semantics.
- [ ] Write a RED scheduler/receipt test for the chosen batching seam, including deterministic ordering and a fixed VRAM/workspace budget.
- [ ] Implement the smallest bounded batching change, preserving seeds and final output identity.
- [ ] Run exact parity, warning-denied build, memcheck, racecheck, cold/warm Nsight Systems, and representative Nsight Compute.
- [ ] Require the shared profiler gates or restore S2. Report honestly if the serial per-scenario recurrence remains the limiting factor even with enough blocks.

## Chunk 4: Completion checkpoint

### Task 11: Verify, archive, and hand off before TRAINING

- [ ] Run complete warning-denied SEARCH workspace tests relevant to native CUDA and CPU canonical parity.
- [ ] Run Compute Sanitizer memcheck and racecheck on the final exact binary.
- [ ] Rehash every modified/created file and archive explicit preimage-or-ABSENT receipt, postimage, reviewed forward patch, tested reverse patch, build log, parity log, sanitizer logs, Nsight reports, and benchmark JSON.
- [ ] Summarize measured improvements and remaining SEARCH limitations without extrapolation.
- [ ] Start the separate TRAINING fit-resident-tensor plan only after every S3 gate is GREEN or the SEARCH stage is explicitly restored to the last verified stage and documented.
