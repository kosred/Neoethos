# Tree CUDA Runtime Proof Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make XGBoost and LightGBM GPU-only training verifiably execute CUDA and fail loudly instead of silently using CPU.

**Architecture:** Add one dedicated integration-test binary that installs the exact tree runtime once and exercises both production experts. Modernize only the XGBoost CUDA probe/device resolution needed to make production and the test use `hist` plus `device=cuda` consistently.

**Tech Stack:** Rust, XGBoost 3.x C API, LightGBM CUDA learner, Cargo integration tests, NVIDIA telemetry.

---

### Task 1: Mandatory device contract

**Files:**
- Create: `crates/neoethos-models/tests/tree_cuda_device.rs`
- Modify: `crates/neoethos-models/src/tree_models/xgboost.rs`

- [ ] Write a focused test that installs CUDA + GPU-only settings, trains and predicts with both experts, and verifies their normal runtime artifacts record CUDA.
- [ ] Run the test on the RTX 3090 and confirm the current XGBoost legacy probe/fallback produces the intended RED.
- [ ] Change the XGBoost probe to `tree_method=hist` plus `device=cuda`, and make GPU-only training reject an unavailable CUDA runtime before creating the real training matrix.
- [ ] Run the same test with `RUSTFLAGS=-Dwarnings`, require both experts to pass, and inspect the full log and telemetry.

### Task 2: Artifact and completion audit

**Files:**
- Verify: `target/audit-logs/vps-3090/run-20260822T221935Z/`

- [ ] Copy the exact test log and telemetry locally and record SHA-256 hashes.
- [ ] Confirm zero warning/error/skip/fallback diagnostics and positive GPU activity.
- [ ] Confirm the aggregate `gpu-cuda --all-targets` compile remains warning-clean.
- [ ] Destroy the paid RTX 3090 instance after all remote evidence is local and verified.
