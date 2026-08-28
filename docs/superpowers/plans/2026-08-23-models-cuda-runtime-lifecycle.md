# Models CUDA Runtime Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox syntax for tracking.

**Goal:** Make all 15 normal `gpu-cuda` model surfaces plus the 12 separately compiled Burn CUDA surfaces strict, exact-device, persistence-safe, and provable on an RTX 3090.

**Architecture:** A shared typed CUDA policy parser and NVIDIA-only visibility probe form the authority. Thin adapters route the resolved ordinal into CubeCL, Candle/rlkit, XGBoost, LightGBM, and CatBoost, while artifacts and capability reporting preserve the same truth.

**Tech Stack:** Rust nightly-2026-04-07, Cargo features `gpu-cuda` and `burn-cuda-backend`, CubeCL CUDA, Burn CUDA, Candle/rlkit CUDA, native XGBoost CUDA, native LightGBM CUDA, CatBoost CLI GPU.

---

## Status

**Complete for the `neoethos-models` CUDA scope on 2026-08-23.** Normal CUDA covers 15 named surfaces through seven mandatory lifecycle gates. The separately compiled Burn CUDA backend covers ten deep-model kinds, ExitAgent, and SAC through 12 mandatory lifecycle gates. CPU remains valid when the corresponding CUDA feature/card is absent; a present NVIDIA path executes CUDA or fails loudly.

## Chunk 1: Policy and routing

### Task 1: Strict CUDA policy authority

**Files:**
- Modify: crates/neoethos-models/src/common.rs
- Modify: crates/neoethos-models/src/tree_models/config.rs
- Test: crates/neoethos-models/tests/models_cuda_runtime_source_contract.rs

- [x] Write a source-only failing contract that rejects malformed gpu ordinals, forbids silent ordinal-zero fallback, requires NVIDIA-only visibility, and requires pure Auto resolution coverage for zero and nonzero NVIDIA counts.
- [x] Run the contract directly with rustc and confirm the expected failure.
- [x] Add a typed CUDA policy parser and exact ordinal result.
- [x] Split the NVIDIA visibility probe from the cross-vendor GPU count; preserve driver masks and remove ROCm from CUDA decisions.
- [x] Add unit tests for accepted aliases, rejected vendor policies, malformed ordinals, explicit CPU, Auto with zero NVIDIA devices, and Auto with a visible NVIDIA device.
- [x] Run the source contract again and confirm this section is green.

### Task 2: CubeCL and DQN routing

**Files:**
- Modify: crates/neoethos-models/src/evolution/neat_gpu.rs
- Modify: crates/neoethos-models/src/evolution/neat_impl.rs
- Modify: crates/neoethos-models/src/evolution/crfmnes_gpu.rs
- Modify: crates/neoethos-models/src/evolution/crfmnes_impl.rs
- Modify: crates/neoethos-models/src/statistical/linear_gpu.rs
- Modify: crates/neoethos-models/src/rl/dqn_impl.rs

- [x] Add RED contracts requiring Result-based exact ordinals and prohibiting Auto CUDA-init fallback to CPU on NVIDIA hosts.
- [x] Resolve Auto before the NEAT and CR-FM-NES kernel branch and propagate every CUDA error.
- [x] Route the exact resolved ordinal into statistical CubeCL kernels.
- [x] Make DQN Auto consult NVIDIA presence first: zero selects CPU; nonzero requires successful Device::new_cuda.
- [x] Apply the same DQN rule during artifact load/inference.
- [x] Keep explicit CPU behavior unchanged.

## Chunk 2: Native tree models and persistence

### Task 3: Exact tree ordinals

**Files:**
- Modify: crates/neoethos-models/src/tree_models/config.rs
- Modify: crates/neoethos-models/src/tree_models/xgboost.rs
- Modify: crates/neoethos-models/src/tree_models/lightgbm.rs
- Modify: crates/neoethos-models/src/tree_models/catboost.rs

- [x] Add RED unit/source contracts for invalid tree policies and cuda:N preservation.
- [x] Route XGBoost to device=cuda:N.
- [x] Route LightGBM to device_type=cuda plus gpu_device_id=N.
- [x] Route CatBoost to task-type GPU plus devices=N.
- [x] Persist requested and resolved exact device values and validate them on load.

### Task 4: XGBoost reload and CatBoost construction

**Files:**
- Modify: crates/neoethos-models/src/tree_models/xgboost.rs
- Modify: crates/neoethos-models/src/tree_models/catboost.rs
- Modify narrowly: crates/neoethos-models/src/training_orchestrator.rs at CatBoost construction

- [x] Add a RED XGBoost test that loads a CUDA artifact and proves the booster device is reapplied instead of merely warning about drift.
- [x] Fail load when NVIDIA is present but the required XGBoost CUDA runtime cannot initialize.
- [x] Reapply exact device after Booster::load before inference.
- [x] Add a CatBoost constructor that accepts the final parameter map and derives device preference, ordinal, GPU-only mode, and threads once.
- [x] Replace the post-construction params assignment in build_expert_model with that constructor.
- [x] Add a RED/GREEN constructor test using device=cuda:2 and gpu_only=true.

## Chunk 3: Capability census and lifecycle gates

### Task 5: Fifteen-surface capability truth

**Files:**
- Modify: crates/neoethos-models/src/registry.rs
- Test: crates/neoethos-models/tests/models_cuda_runtime_source_contract.rs

- [x] Add a RED exact-set assertion for the 15 CUDA-capable names.
- [x] Map NEAT and neuro_evo to neuro-evolution-gpu.
- [x] Map Logistic and ElasticNet to statistical-gpu.
- [x] Map the four meta wrappers to XGBoost.
- [x] Separate supports from prefers where an operator gate prevents automatic use.

### Task 6: Mandatory real-card lifecycle tests

**Files:**
- Extend: crates/neoethos-models/tests/tree_cuda_device.rs
- Extend: crates/neoethos-models/tests/rl_cuda_contract.rs
- Extend: crates/neoethos-models/tests/neuro_evolution_cuda_contract.rs
- Extend: crates/neoethos-models/tests/statistical_cuda_contract.rs
- Extend focused in-module tests only where private state is required.

- [x] Add exact tree gates for XGBoost, RF, DART, LightGBM, CatBoost, and CatBoost alternate.
- [x] Add an exact DQN gate using small deterministic episodes.
- [x] Add exact NEAT and CR-FM-NES gates with bounded populations/generations.
- [x] Add exact Logistic and ElasticNet gates.
- [x] Add exact meta-wrapper gates.
- [x] Each persistent model performs train, infer, save, load, infer and asserts prediction shape/finiteness plus CUDA artifact/backend identity.
- [x] Tests assert NVIDIA presence and fail rather than return/ignore/skip.

## Chunk 4: Verification handoff

### Task 7: Source-stable verification

- [x] Run rustfmt on only modified model files.
- [x] Run direct source contracts and record test counts and hashes.
- [x] Inspect the full diff for stale fallbacks, warnings-only device drift, ROCm-to-CUDA aliases, and test skips.
- [x] Send root exact Cargo filters for each family and expected nonzero test counts.

### Task 8: RTX 3090 gates coordinated by root

- [x] Root confirms no cargo/rustc/nvcc process before each command.
- [x] Build once with gpu-cuda defaults under warning denial.
- [x] Run tree, DQN, evolution, statistical, and meta lifecycle filters one at a time.
- [x] Capture complete logs and 100 ms nvidia-smi telemetry for every family.
- [x] Treat compile-only, zero-test, CPU backend, skipped test, or absent telemetry activity as RED.

## Burn CUDA extension and final evidence

- [x] Keep `burn-cuda-backend` a separate, CUDA-only compiled boundary; reject CPU selection and artifact/backend drift.
- [x] Cover all ten `DeepModelKind` variants, ExitAgent, and SAC with 12 non-skipping train → infer → save/load → infer gates on exact ordinal 0.
- [x] Add an outer Burn/CubeCL residency scope with Fusion drain, all-initialized-stream cleanup, pinned/device-pool cleanup, and unwind-safe fail-loud reporting.
- [x] Backport CubeK matmul and convolution TMA capability preflight so Ampere sm86 never calls Hopper-only Tensor Map APIs.
- [x] Move Burn validation and SAC stop-gradient/inference work to the official `valid()` / `inner()` backend boundary.
- [x] Assert SAC training teardown, first inference, and reloaded inference each leave zero new Fusion handles.
- [x] Report Burn CUDA support/preference for Deep, Exit, and SAC whenever either Burn GPU backend is compiled.

Verified RTX 3090 evidence:

- Normal `gpu-cuda`: seven exact lifecycle gates cover all 15 registry surfaces; all were GREEN with per-family telemetry and no skip/CPU fallback.
- Shared CubeCL low-level matrix: 5/5 functional; Compute Sanitizer `ERROR SUMMARY: 0`, `LEAK SUMMARY: 0`; production evolution/statistical reruns GREEN.
- Burn low-level gate: functional GREEN; Compute Sanitizer `ERROR SUMMARY: 0`, `LEAK SUMMARY: 0`.
- Burn full lifecycle matrix: 12/12 in 84.41 seconds; Compute Sanitizer `ERROR SUMMARY: 0`, `LEAK SUMMARY: 0`.
- SAC phase gate: 1/1 in 22.89 seconds with zero handles at training teardown, first inference, and reloaded inference. Isolated SAC sanitizer: 1/1 in 12.71 seconds, errors 0, leaks 0.
- Final standalone `burn-cuda-backend` capability compile: 16.87 seconds, warnings 0, errors 0; exact capability contract 1/1.
- Final source-only censuses: normal CUDA 9/9, Burn surface/capability 6/6, Burn lifecycle/TMA/allocator 12/12.

Durable evidence archives:

- `target/audit-logs/vps-3090/run-20260823T014900Z/models-burn-final-evidence.tar.gz` — SHA-256 `0a2a7e76b9709dcd4bca1a1f68a2ff25b3da4dd769d8933bec9a0d57f39d5b8b`.
- `target/audit-logs/vps-3090/run-20260823T014900Z/models-burn-capability-evidence.tar.gz` — SHA-256 `2b6b502de6d438c1ca335f9b9ca78ffacd61530061cef9d35e427ad4fd236be1`.
- `target/audit-logs/vps-3090/run-20260823T014900Z/models-cubecl-lifecycle-evidence.tar.gz` — SHA-256 `c0fbba03c4f5a7bace0a9bd469119eb2857d2c596f5df5dc7408f8b7e18c6a13`.

Completion boundary: this closes `neoethos-models` CUDA routing, persistence, capability truth, real-card lifecycle, and allocator/kernel sanitizer coverage. It does not by itself prove the separate data/search pipeline, a full discovery run, full production training, repository integration/merge, or non-CUDA Vulkan/ROCm parity.
