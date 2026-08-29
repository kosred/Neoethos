# FRAMA Large-Window Deque Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the host FRAMA large-window deque so it matches the direct two-half formula exactly in O(N).

**Architecture:** Keep two monotonic max/min pairs whose membership is the exact older and newer half of the prior `window` bars. Advance both pairs by one boundary after each output and preserve the existing host recurrence and public semantics.

**Tech Stack:** Rust, VectorTA indicator unit tests, Cargo.

---

## Chunk 1: RED-first repair

### Task 1: Freeze the failing mathematical contract

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs:1839`
- Test: `vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs`

- [x] **Step 1: Write the failing test**

Add a test-only direct two-half FRAMA reference and a deterministic 4096-row window-200 case. Compare every valid result by `to_bits()` and include the row in the failure message.

- [x] **Step 2: Run the focused test to verify RED**

Run from `vendor/vector-ta-0.2.9-patched`:

`cargo test --lib frama_large_window_deque_matches_direct_halves -- --exact --nocapture`

Expected: FAIL, with the first mismatch at row 297.

### Task 2: Repair exact half-window ownership

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs:455-603`

- [x] **Step 1: Implement the minimal O(N) repair**

Initialize left/right monotonic pairs to the exact halves. Compute from those pairs, then advance the outgoing, crossing, and newest indices after each row. Derive the full range from the two half ranges. Leave the EMA expression and all constants unchanged.

- [x] **Step 2: Run the focused test to verify GREEN**

Run the same focused command. Expected: PASS.

- [x] **Step 3: Run focused FRAMA regression tests**

Run: `cargo test --lib frama -- --nocapture`

Expected: all FRAMA tests PASS. If local compilation is materially heavy, stop after the focused test and reserve the complete build/device suite for the parent release run.

- [x] **Step 4: Re-freeze hashes and inspect the exact diff**

Record the post-edit SHA-256 for `frama.rs`, confirm only the approved host file plus design/plan/test changed, and report CUDA FMA/libm residuals separately.
