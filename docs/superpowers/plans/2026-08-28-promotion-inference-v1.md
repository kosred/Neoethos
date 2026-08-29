# Promotion Inference V1 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task. Preserve the shared-source/Cargo serialization boundary throughout.

**Goal:** Add an explicit-policy, exact-inventory opener for a frozen P3 promotion candidate that returns a move-only in-memory verified ensemble without ambient settings or paths.

**Architecture:** A checked policy seals ordered model/runtime/voting facts. The opener consumes the P3 manifest and policy, validates exact installed inventory and typed runtime sidecars, loads experts through an injected/default registry, upgrades partial loader output to all-or-refuse, constructs soft voting from the explicit policy, and re-verifies the P3 tree after loading. A private carrier owns the loaded ensemble plus its manifest/handoff/policy evidence.

**Tech Stack:** Rust, serde/serde_json, existing `neoethos-core` runtime artifacts, existing `neoethos-models` ensemble registry/soft voting, frozen P3 promotion-candidate handoff/install APIs.

---

### Task 1: Freeze current-source API census

**Files:**
- Read: `crates/neoethos-models/src/promotion_candidate_training_v1.rs`
- Read: `crates/neoethos-models/src/promotion_candidate_training_v1/install.rs`
- Read: `crates/neoethos-models/src/ensemble_inference/mod.rs`
- Read: `crates/neoethos-models/src/ensemble_inference/bootstrap.rs`
- Read: `crates/neoethos-models/src/ensemble_inference/soft_voting.rs`
- Read: `crates/neoethos-models/src/runtime/profile.rs`
- Read: `crates/neoethos-core/src/system/runtime_artifact.rs`

1. Rehash frozen P3 files and confirm the manifest `73fd1cb2...` still verifies.
2. Record exact public getters and constructor signatures needed by P4B.
3. Confirm the default registry builder has no settings lookup and identify any loader-required filesystem layout.
4. Confirm exact serialization names for backend, precision, output kind, runtime profile, and provenance.
5. Stop if any frozen API differs materially from the approved design.

### Task 2: Write immutable behavior RED locally

**Files:**
- Create: `.codex-p4b/crates/neoethos-models/src/promotion_inference_v1_tests.rs`

1. Build a fixture through the real P3 handoff and candidate installer APIs.
2. Add fake `ExpertLoader`/`ExpertModel` implementations through a private injectable opener seam.
3. Add exact-success test proving ordered experts, policy identity, retained manifest/handoff, and no path-dependent prediction.
4. Add mutation/refusal tests for missing, unexpected, degraded/partial, model file, tree/evidence, handoff identity, policy identity, feature order, backend/planned backend, precision, output kind, weight, and post-load mutation.
5. Keep the file below 900 lines.
6. Hash and count the local test; do not upload or run Cargo.

### Task 3: Write independent source-contract RED locally

**Files:**
- Create: `.codex-p4b/crates/neoethos-models/tests/promotion_inference_v1_source_contract.rs`

1. Assert the real public opener/type names and required signatures.
2. Assert forbidden ambient symbols are absent: `Settings::load`, `build_ensemble_for_symbol`, `load_experts_for_symbol`, settings-derived voting helpers, current/latest paths, and gene fallback.
3. Assert typed runtime artifact parsing, exact missing/unexpected inventory checks, explicit soft-voting construction, and pre/post P3 verification markers.
4. Assert carrier fields/constructor remain private and that no `Clone`, `Copy`, `Serialize`, or `Deserialize` implementation/derive is attached.
5. Assert no promotion-authority or promotion-eligible minting is present.
6. Keep the file below 900 lines and compute its hash.

### Task 4: Capture authoritative RED after P4A release

**Files:**
- Upload: `crates/neoethos-models/src/promotion_inference_v1_tests.rs`
- Upload: `crates/neoethos-models/tests/promotion_inference_v1_source_contract.rs`
- Modify test-only module declaration as minimally required.

1. Obtain explicit P4A source/Cargo release.
2. Guard upload and rehash exact remote bytes.
3. Run only the focused behavior and source-contract tests under `RUSTFLAGS=-Dwarnings`.
4. Require failures to be exclusively the absent approved P4B production seam.
5. Freeze RED log, status, source manifest, hashes, and line counts before any production edit.

### Task 5: Implement checked policy and artifact helpers

**Files:**
- Create: `crates/neoethos-models/src/promotion_inference_v1.rs`
- Create: `crates/neoethos-models/src/promotion_inference_v1/artifact.rs`

1. Implement bounded checked policy construction and deterministic identity recomputation.
2. Implement canonical inventory validation and exact P3 relative-path comparison.
3. Implement bounded typed runtime-artifact read and provenance/runtime-safety validation.
4. Implement explicit conversion from policy to `SoftVotingEnsembleConfig` with no settings lookup.
5. Run only the smallest policy/artifact mutation subset; fix compiler/test diagnostics one at a time.

### Task 6: Implement exact opener and carrier

**Files:**
- Modify: `crates/neoethos-models/src/promotion_inference_v1.rs`
- Modify: `crates/neoethos-models/src/lib.rs`

1. Implement private registry-injectable opener and public default-registry wrapper.
2. Pre-scan required and unexpected model inventory.
3. Perform P3 pre-load verify and handoff reopen/binding.
4. Validate each runtime artifact, load requested experts, and refuse all partial/degraded/unexpected/order/output/feature/backend/precision drift.
5. Construct explicit soft voting and perform P3 post-load verify.
6. Return private-field, move-only, non-serializable carrier owning the loaded ensemble.
7. Add only the approved public reexports.

### Task 7: Focused GREEN and scope verification

1. Run the behavior test under `-Dwarnings`; require every RED mutation to turn GREEN.
2. Run the independent source contract under `-Dwarnings`.
3. Run `cargo fmt --check` on all touched Rust files.
4. Verify each scope is below 900 changed lines and no unrelated file changed.
5. Inspect complete INFO, WARNING, and ERROR output, not only tails.

### Task 8: Default/CUDA regression and freeze

1. Coordinate a serialized Cargo window.
2. Run neoethos-models focused/default all-targets under `-Dwarnings`.
3. Run the applicable CUDA/gpu-nvidia all-targets graph under `-Dwarnings`.
4. Run the smallest downstream app compile/test that consumes neoethos-models public exports.
5. Rehash sources, tests, logs, and statuses into an immutable P4B freeze manifest.
6. Explicitly release source/Cargo and report any scope not hardware-validated.

### Task 9: Completion review

1. Re-read the approved design and verify every non-goal remains excluded.
2. Request bounded independent code review before P4C consumes the carrier.
3. Do not merge, promote, or modify live/promotion code in this scope.
