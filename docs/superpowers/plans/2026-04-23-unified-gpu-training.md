# Unified GPU Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current mixed GPU/runtime metadata into real GPU-capable training behavior, starting with the Burn deep-model lane and preserving truthful native GPU backends for the tree models.

**Architecture:** Keep the mixed architecture already approved by the user. Deep models gain a real portable GPU training precision layer in Burn. Tree models keep official upstream GPU runtimes with explicit precision limits. RL, search, and evolutionary code will be migrated afterward to the same runtime truth contract.

**Tech Stack:** Rust 2024, Burn 0.20.1, burn-wgpu, ndarray, existing `forex-models` runtime metadata, lightweight formatting and diff verification.

---

## Chunk 1: Burn Training Precision Becomes Real

### Task 1: Make Burn precision resolution executable instead of metadata-only

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs`

- [ ] **Step 1: Add failing tests for truthful precision resolution**

Add tests that require:
- CPU execution resolves to `fp32`
- BF16 request only resolves to `bf16` when the backend/device path supports it
- FP8/BF4 requests degrade truthfully instead of being reported as executed when no runtime implementation exists

- [ ] **Step 2: Run a lightweight targeted check only if needed**

Run at most:
`rustfmt --edition 2024 --check C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs`

Expected: formatting-only feedback, no full cargo build.

- [ ] **Step 3: Implement runtime precision selection**

Add:
- a precise Burn training dtype helper
- backend/device capability checks using Burn dtype support
- explicit fallback reasons for unsupported lower-precision requests

- [ ] **Step 4: Re-run lightweight verification**

Run:
`rustfmt --edition 2024 --check C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs`

Expected: PASS.

### Task 2: Execute BF16 tensors and BF16 parameters in the Burn training loop

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs`

- [ ] **Step 1: Add a module float-tensor casting helper**

Implement a `ModuleMapper`-based helper that can cast trainable float tensors inside Burn modules to the requested runtime dtype.

- [ ] **Step 2: Add dtype-aware tensor materialization**

Replace FP32-only batch tensor creation with dtype-aware helpers for:
- feature tensors
- class-weight tensors
- validation tensors

- [ ] **Step 3: Preserve stable inference/runtime behavior**

Normalize the trained model back to FP32 before returning it to the runtime layer so:
- immediate inference remains stable
- artifact save/load stays coherent with `FullPrecisionSettings`

## Chunk 2: Deep Model Runtime Coherence

### Task 3: Thread the real Burn precision contract through deep-model artifacts

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/deep_models.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/deep_models.rs`

- [ ] **Step 1: Tighten runtime validation**

Ensure persisted deep-model params and training reports accept only truly implemented precision values.

- [ ] **Step 2: Keep fit/save/load consistent**

Persist:
- requested device policy
- effective device policy
- execution backend
- actual training precision
- degraded precision reason if any

- [ ] **Step 3: Keep runtime inference independent from training dtype**

Do not let a BF16 training run break current inference, runtime metadata, or artifact loading.

## Chunk 3: Orchestrator Alignment

### Task 4: Keep orchestrator/runtime reporting aligned with the new deep-model contract

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`

- [ ] **Step 1: Inspect existing runtime profile/report writing for precision drift**

Check for places where orchestrator code assumes device/backend truth but ignores actual training precision or degraded reasons.

- [ ] **Step 2: Patch metadata plumbing if needed**

Adjust only the runtime metadata/report surfaces necessary to keep the new Burn precision truth visible and consistent.

## Chunk 4: Lightweight Verification

### Task 5: Run non-compile-heavy verification

**Files:**
- Modify: none

- [ ] **Step 1: Format check touched files**

Run:
`rustfmt --edition 2024 --check C:/Users/konst/development/forex-ai/crates/forex-models/src/burn_models.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/deep_models.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`

- [ ] **Step 2: Diff sanity check**

Run:
`git diff --check`

Expected: no whitespace or conflict issues.

- [ ] **Step 3: Defer heavy cargo compile/test**

Do not run `cargo check`, `cargo build`, or `cargo test` in this tranche unless the user explicitly requests it or a local contradiction makes it unavoidable.
