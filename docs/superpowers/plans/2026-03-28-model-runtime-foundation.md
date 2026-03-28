# Model Runtime Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a truthful runtime foundation for all declared model families so later model tranches can plug into a deterministic training, artifact, and prediction contract.

**Architecture:** Add a new `runtime` namespace inside `forex-models` for capabilities, artifacts, prediction outputs, and dispatch planning. Refactor the registry and training orchestrator to use that runtime layer so configured models resolve to explicit families and capability states before any training begins.

**Tech Stack:** Rust, `anyhow`, `serde`, `serde_json`, existing `forex-models` crate patterns, cargo test/clippy.

---

## Chunk 1: Runtime Skeleton

### Task 1: Add runtime module shell

**Files:**
- Create: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/mod.rs`
- Create: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/capabilities.rs`
- Create: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/artifacts.rs`
- Create: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/prediction.rs`
- Create: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/dispatch.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/lib.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/capabilities.rs`

- [ ] **Step 1: Write the failing test**

Add tests that expect:
- a `ModelFamily` enum
- a `CapabilityState` enum with `Planned`, `Implemented`, `Verified`
- a `ModelCapability` struct

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models runtime::capabilities -- --nocapture`

Expected: compile failure or missing symbol errors.

- [ ] **Step 3: Write minimal implementation**

Create the runtime files and export them from `lib.rs`. Define the enums and structs only, with no business logic yet.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models runtime::capabilities -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime C:/Users/konst/development/forex-ai/crates/forex-models/src/lib.rs
git commit -m "feat: add model runtime module shell"
```

### Task 2: Add shared artifact metadata contract

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/artifacts.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/artifacts.rs`

- [ ] **Step 1: Write the failing test**

Add a round-trip serialization test for:
- model name
- family
- capability state
- feature columns
- label mapping
- training summary metadata

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models runtime::artifacts -- --nocapture`

Expected: compile failure or missing serde fields.

- [ ] **Step 3: Write minimal implementation**

Add `RuntimeArtifactMetadata` and related structs with `Serialize`, `Deserialize`, and `PartialEq`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models runtime::artifacts -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/artifacts.rs
git commit -m "feat: add shared model artifact metadata"
```

### Task 3: Add shared prediction output contract

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/prediction.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/prediction.rs`

- [ ] **Step 1: Write the failing test**

Add tests that require a shared prediction output containing:
- class probabilities
- optional confidence
- optional abstain flag
- model metadata

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models runtime::prediction -- --nocapture`

Expected: compile failure.

- [ ] **Step 3: Write minimal implementation**

Define `RuntimePrediction`, `PredictionMetadata`, and any small helpers needed to validate shape assumptions.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models runtime::prediction -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/prediction.rs
git commit -m "feat: add shared prediction runtime contract"
```

## Chunk 2: Truthful Registry

### Task 4: Replace string-only registry metadata with capability records

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/registry.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/capabilities.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/registry.rs`

- [ ] **Step 1: Write the failing test**

Add tests that require:
- each known model name to map to a `ModelCapability`
- family classification for current names in `config.yaml`
- no missing registry entry for the currently configured model names

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models registry -- --nocapture`

Expected: FAIL due to missing capability records.

- [ ] **Step 3: Write minimal implementation**

Refactor the registry so the public surface returns rich capability data instead of name-only metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models registry -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/registry.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/capabilities.rs
git commit -m "feat: make model registry capability-aware"
```

### Task 5: Add config-driven registry coverage test

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/registry.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/registry.rs`

- [ ] **Step 1: Write the failing test**

Add a test that loads `Settings::default()` and asserts that all configured `ml_models` resolve to a known runtime capability.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models registry::tests:: -- --nocapture`

Expected: FAIL for missing names or mapping mismatch.

- [ ] **Step 3: Write minimal implementation**

Add the missing registry coverage helpers or family aliases needed for the configured names.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models registry::tests:: -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/registry.rs
git commit -m "test: validate configured model names against runtime registry"
```

## Chunk 3: Dispatch Plan

### Task 6: Add explicit dispatch planning

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/dispatch.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/dispatch.rs`

- [ ] **Step 1: Write the failing test**

Add tests that require:
- a configured model list to resolve to a deterministic dispatch plan
- each dispatch entry to include model name, family, and capability state

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models runtime::dispatch -- --nocapture`

Expected: FAIL due to missing dispatcher types.

- [ ] **Step 3: Write minimal implementation**

Create dispatch plan structs and helpers. Update the orchestrator to build a dispatch plan before any training starts.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models runtime::dispatch -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/dispatch.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs
git commit -m "feat: add model training dispatch plan"
```

### Task 7: Remove late generic placeholder dispatch in orchestrator

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/parallel_trainer.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`

- [ ] **Step 1: Write the failing test**

Add tests asserting that the orchestrator fails early with capability/dispatch errors instead of entering the generic closure and bailing later.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models training_orchestrator -- --nocapture`

Expected: FAIL due to current late `bail!` behavior.

- [ ] **Step 3: Write minimal implementation**

Refactor `get_enabled_models`, `map_model_type`, and the training path to use the dispatch plan. Keep trainer execution placeholder-free at the orchestration boundary.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models training_orchestrator -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/parallel_trainer.rs
git commit -m "refactor: make training orchestrator dispatch-first"
```

## Chunk 4: Shared Runtime Consumption

### Task 8: Add base trait helpers for runtime artifact and prediction integration

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/base.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/artifacts.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/prediction.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/base.rs`

- [ ] **Step 1: Write the failing test**

Add tests for helper functions that attach feature columns, label mapping, and runtime metadata to saved artifacts or produced predictions.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models base -- --nocapture`

Expected: FAIL due to missing runtime helpers.

- [ ] **Step 3: Write minimal implementation**

Add helper structs or trait methods needed so later families can adopt the shared runtime contracts without duplicating metadata logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models base -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/base.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/artifacts.rs C:/Users/konst/development/forex-ai/crates/forex-models/src/runtime/prediction.rs
git commit -m "feat: add shared runtime helpers for model metadata"
```

### Task 9: Add top-level integration test for current configured model inventory

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`
- Test: `C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs`

- [ ] **Step 1: Write the failing test**

Add a test that reads the configured model names and asserts:
- every name resolves to a capability
- every capability resolves to a family
- dispatch planning remains deterministic

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-models training_orchestrator::tests:: -- --nocapture`

Expected: FAIL until all wiring is complete.

- [ ] **Step 3: Write minimal implementation**

Fill any remaining runtime wiring gaps needed for deterministic end-to-end planning.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-models training_orchestrator::tests:: -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs
git commit -m "test: verify model runtime plan for configured inventory"
```

## Chunk 5: Final Verification And Handoff

### Task 10: Run verification and document the tranche boundary

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/docs/superpowers/specs/2026-03-28-model-runtime-foundation-design.md` (only if the implemented boundary differs from the approved design)
- Modify: `C:/Users/konst/development/forex-ai/docs/superpowers/plans/2026-03-28-model-runtime-foundation.md` (checklist updates only during execution)

- [ ] **Step 1: Run focused crate tests**

Run: `cargo test -p forex-models -- --nocapture`

Expected: PASS.

- [ ] **Step 2: Run focused clippy**

Run: `cargo clippy -p forex-models --all-targets -- -D warnings`

Expected: PASS.

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace -- --nocapture`

Expected: PASS.

- [ ] **Step 4: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add C:/Users/konst/development/forex-ai/crates/forex-models/src C:/Users/konst/development/forex-ai/docs/superpowers/specs/2026-03-28-model-runtime-foundation-design.md C:/Users/konst/development/forex-ai/docs/superpowers/plans/2026-03-28-model-runtime-foundation.md
git commit -m "feat: add model runtime foundation"
```

Plan complete and saved to `docs/superpowers/plans/2026-03-28-model-runtime-foundation.md`. Ready to execute?
