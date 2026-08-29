# Autoresearch Batch Promotion Bindings Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind every promoted gene to the exact streaming batch receipt, scope, effective configuration, and local feature vocabulary that selected it.

**Architecture:** Replace promotion evidence v3's flat portfolio with a v4 ordered per-batch structure. Validate effective result identity at the writer boundary and validate the entire self-contained binding again at the loader boundary before OOS is journalled spent.

**Tech Stack:** Rust, serde/serde_json, anyhow, neoethos canonical search receipt/scope types, SHA-256, Cargo tests.

---

## Chunk 1: Fail-closed schema and loader

### Task 1: Express v4 invariants as failing tests

**Files:**
- Modify: `crates/neoethos-autoresearch/src/runner.rs`
- Test: `crates/neoethos-autoresearch/src/runner/streaming.rs`

- [ ] Add fixtures for two distinct batch receipts with distinct feature-plan
  identities, exact evaluated windows, effective search-config hashes, local
  feature names, and cursor-tagged genes.
- [ ] Add one happy-path loader test.
- [ ] Add individual refusal tests for missing binding, swapped ordinal/cursor,
  stale receipt digest, changed feature plan, wrong evaluated window, wrong
  effective config, and flattened untagged genes.
- [ ] Run the focused tests and capture the expected RED failures before changing
  production code.

### Task 2: Implement the minimal v4 schema and validator

**Files:**
- Modify: `crates/neoethos-autoresearch/src/runner.rs`

- [ ] Add deny-unknown-fields v4 top-level, batch-binding, and tagged-gene types.
- [ ] Keep the proposal stamp separate from per-batch effective config hashes.
- [ ] Validate counts, ordinals, cursors, tags, local names, gene indices,
  receipt digest, exact scope/window, expected effective config, and frozen
  dataset receipt bindings/provenance.
- [ ] Make `load_promotion_portfolio` require the expected effective search
  config and run all validation before returning.
- [ ] Run the focused loader tests and capture GREEN.

## Chunk 2: Writer and OOS consumption

### Task 3: Test exact multi-batch persistence

**Files:**
- Test: `crates/neoethos-autoresearch/src/runner/streaming.rs`

- [ ] Add a RED test showing v3 flattening cannot represent two batch receipts.
- [ ] Add a RED test showing a result effective config that differs from the
  exact expected charged request config is refused before evidence is written.

### Task 4: Persist and consume exact bindings

**Files:**
- Modify: `crates/neoethos-autoresearch/src/runner/streaming.rs`
- Modify: `crates/neoethos-autoresearch/src/runner.rs`

- [ ] Derive one expected effective config hash from the exact config passed to
  discovery and compare every batch result to it before durable evidence writes.
- [ ] Persist ordered survivor batches with their own result receipt, digest,
  exact evaluated window, result config hash, local names, and tagged local
  genes.
- [ ] Revalidate every binding before trial-return or promotion evidence is
  adopted.
- [ ] Project and evaluate each batch's local genes by its own local names in
  OOS; remove the flattened path.
- [ ] Run focused tests and capture GREEN.

## Chunk 3: Verification

### Task 5: Verify source and contracts

**Files:**
- Modify only as required by compiler diagnostics within the owned autoresearch
  runner/tests boundary.

- [ ] Run rustfmt on the owned Rust files and inspect the diff.
- [ ] Run focused tests warning-denied.
- [ ] Run full `neoethos-autoresearch` tests warning-denied.
- [ ] Review complete logs in INFO, WARNING, ERROR order and report exact counts.
- [ ] Confirm no model dependency or session/journal v2/v3 schema change was
  introduced.
