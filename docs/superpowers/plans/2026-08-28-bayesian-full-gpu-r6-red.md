# Bayesian Logistic Full-GPU R6 RED Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an R6 RED-only acceptance bundle that rejects hidden CPU Bayesian GPU work, proves exact device-output dependency and five canonical kernel identities, enforces semantic embargo dataflow, contains every paid-process descendant, and cannot spend the final GPU attempt before all cheap gates pass.

**Architecture:** Keep all production source identical to authority `7824e19`. Put independent oracle, evidence, AST, process, fixture, and provenance logic in focused test-support modules; run every synthetic contract locally before the single missing-API compile RED; leave the real RTX parent ignored and make it consume those same validated helpers. Freeze the accepted RED-contract blobs separately from local evidence. A later, separately authorized production phase must preserve those exact contract blobs, freeze a distinct integrated implementation identity, and deploy only that integrated commit as a verified Git bundle into a fresh detached checkout.

**Tech Stack:** Rust integration tests, `syn`/`quote` scoped AST analysis, `nalgebra`/`ndarray` f64 reference math, `sha2`, `serde`/`serde_json`, `rusqlite`, Windows Job Objects, Unix process groups through `libc`, Git plumbing, Cargo locked/offline metadata, NVIDIA Nsight Systems SQLite/CUPTI/NVTX.

---

## File map

No tracked file matched by the repository-wide `:(glob)**/src/**` pathspec may change in this plan. This includes `crates/neoethos-models/src`, `crates/neoethos-core/src`, `crates/neoethos-execution-budget/src`, and every other production source root.

**Test-only dependency files**

- Modify: `crates/neoethos-models/Cargo.toml`
- Modify: `Cargo.lock`

**Focused support modules**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/mod.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/oracle.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/fixture.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/cuda_evidence.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/source_contract.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/summary_contract.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/process_tree.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/provenance.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/timing.rs`

**Integration-test targets**

- Create: `crates/neoethos-models/tests/bayesian_full_gpu_r6_support.rs`
- Create: `crates/neoethos-models/tests/bayesian_gpu_route_embargo_r6_contract.rs`
- Create: `crates/neoethos-models/tests/training_summary_embargo_r6_contract.rs`
- Create: `crates/neoethos-models/tests/paid_process_tree_r6_contract.rs`
- Create: `crates/neoethos-models/tests/bayesian_full_gpu_r6_contract.rs`
- Create: `crates/neoethos-models/tests/bayesian_full_gpu_r6_acceptance.rs`

**Tracked freeze evidence**

- Create later: `audit/bayesian-full-gpu-r6-red/README.md`
- Create later: `audit/bayesian-full-gpu-r6-red/commands.txt`
- Create later: `audit/bayesian-full-gpu-r6-red/local-*.txt`
- Create later: `audit/bayesian-full-gpu-r6-red/red-contract-source-blobs.sha256`
- Create later: `audit/bayesian-full-gpu-r6-red/red-contract-vendor-blobs.sha256`
- Create later: `audit/bayesian-full-gpu-r6-red/manifest.sha256`

## Global execution rules

Every build, metadata, or test command in this plan is run from the isolated R6 root with:

```powershell
$env:CARGO_INCREMENTAL='0'
$env:RUSTFLAGS='-Dwarnings'
$env:CARGO_TARGET_DIR='<one reviewed isolated R6 target path>'
```

Every Cargo invocation that resolves dependencies includes `--locked --offline`. Build and test commands additionally include `-j7`; `cargo metadata` does not accept a jobs argument. Stop on the first unexpected compiler family, warning, test name, count, or exit classification. Do not run a broad workspace build, a GPU binary, an ignored GPU/acceptance test, `nsys profile`, SSH/SCP, registry access, or a network command during local R6 construction. The only local ignored-test exception is Task 6's private non-GPU survivor helper, invoked transitively by the two exact containment tests and never selected as a standalone gate.

Use direct `rustfmt` from the pinned toolchain for changed Rust test files; do not add a Cargo formatting invocation that cannot carry `--locked --offline`.

## Chunk 1: Authority and independently GREEN primitives

### Task 1: Reconfirm authority and create the test-only module boundary

**Files:**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/mod.rs`
- Modify: `crates/neoethos-models/Cargo.toml`
- Modify: `Cargo.lock`

- [ ] **Step 1: Reconfirm the exact branch base before writing tests**

Run:

```powershell
git cat-file -e '7824e191c04b4eb78e547728ad7cdb78f915a2af^{commit}'
git rev-parse '7824e191c04b4eb78e547728ad7cdb78f915a2af^{tree}'
git merge-base HEAD 7824e191c04b4eb78e547728ad7cdb78f915a2af
git log --oneline 7824e191c04b4eb78e547728ad7cdb78f915a2af..HEAD
git diff --name-status 7824e191c04b4eb78e547728ad7cdb78f915a2af..HEAD --
git status --porcelain=v1 --untracked-files=all
git diff --name-only 7824e191c04b4eb78e547728ad7cdb78f915a2af -- ':(glob)**/src/**'
```

Expected: the authority object exists, its tree is `e5c9cc23b1e97c6955d379d584f6fe555fe701dc`, and the merge base is exactly `7824e191c04b4eb78e547728ad7cdb78f915a2af`. The bounded diff must contain exactly three additions and no modification/deletion: `docs/superpowers/specs/2026-08-28-bayesian-full-gpu-r6-red-design.md`, `docs/superpowers/plans/2026-08-28-bayesian-full-gpu-r6-red.md`, and `audit/bayesian-full-gpu-r6-red/design-manifest.sha256`. Status is clean and the protected production diff is empty. Current `HEAD` is intentionally not required to equal the authority after the documentation freeze.

- [ ] **Step 2: Declare only needed dev dependencies**

Add `quote = "1.0.47"`, `syn = { version = "2.0.119", features = ["full", "visit"] }`, and `rusqlite = "0.37.0"` only if the authoritative manifest does not already expose them directly to integration tests. First verify that those exact package/version/checksum entries already exist in the authoritative locked closure. Mechanically update only the existing `neoethos-models` package dependency-name list in `Cargo.lock` with `apply_patch`; do not run an unlocked Cargo lock refresh. If any package object or transitive edge is absent, stop and request a plan revision rather than invoking Cargo without `--locked --offline`. Reuse existing direct `libc`, `nalgebra`, `ndarray`, `serde`, `serde_json`, and `sha2` dependencies. Do not add a Windows crate merely for convenience; the process test may use narrow raw FFI.

- [ ] **Step 3: Create the support module export surface**

`mod.rs` declares the eight focused modules and re-exports only types/functions consumed by the six test targets. Keep private helpers private so a test cannot bypass a validator.

- [ ] **Step 4: Prove locked/offline resolution before compiling**

Run:

```powershell
cargo +nightly-2026-04-07 metadata --locked --offline --format-version=1 --no-deps
```

Expected: exit 0 with no network attempt and no untracked/ignored path dependency.

- [ ] **Step 5: Commit the test-support boundary**

Run:

```powershell
git add crates/neoethos-models/Cargo.toml crates/neoethos-models/tests/support/bayesian_r6/mod.rs Cargo.lock
git commit -m "test(models): scaffold Bayesian r6 RED support"
```

### Task 2: Anchor fixtures and retain the independent Bayesian oracle

**Files:**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/fixture.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/oracle.rs`
- Create: `crates/neoethos-models/tests/bayesian_full_gpu_r6_support.rs`

- [ ] **Step 1: Write the tiny-fixture constant tests first**

Create literal train/OOS matrices and labels in the parent test. Freeze four reviewed 64-hex SHA-256 strings over four domain-separated streams. Every stream starts with exact ASCII/NUL `neoethos.bayesian.fixture.v1\0`, then the one-byte phase (`01` train or `02` OOS), then the one-byte kind (`01` features or `02` labels):

```text
feature stream = schema || phase || 01 || u64 rows LE || u64 columns LE
                 || row-major f64::to_bits LE
label stream   = schema || phase || 02 || u64 label count LE
                 || i32 labels LE
```

The four expected strings are literal constants named for train features, train labels, OOS features, and OOS labels. No runtime call to the worker generator or shared hash helper may derive them.

The test names are:

```rust
#[test]
fn tiny_fixture_constants_anchor_parent_and_worker_encodings() { /* exact constants */ }

#[test]
fn fixture_hashes_bind_train_and_oos_features_and_labels() { /* four hashes + mutations */ }
```

Require a one-bit feature mutation, row/column reorder, phase change, and label mutation to change the relevant hash.

- [ ] **Step 2: Run only the two new tests and record the initial RED**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test bayesian_full_gpu_r6_support tiny_fixture_constants_anchor_parent_and_worker_encodings -- --exact --nocapture
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test bayesian_full_gpu_r6_support fixture_hashes_bind_train_and_oos_features_and_labels -- --exact --nocapture
```

Expected: compile/test RED because the independent fixture encoder is not implemented. No production error is acceptable here.

- [ ] **Step 3: Implement two independent fixture paths**

Implement a parent reference encoder without calling the worker fixture generator. Implement the worker generator separately. Both must reproduce the literal tiny constants before either may generate the `1_000_000 x {64,128}` train and `131_071 x {64,128}` OOS identities.

- [ ] **Step 4: Add the independent normal/extreme/ill-conditioned oracle test**

The public test is:

```rust
#[test]
fn public_cpu_posterior_and_probabilities_match_independent_oracle_for_all_cases() {
    assert_public_cpu_matches_oracle(&fixture_cases());
}
```

`fixture_cases()` returns exactly `normal`, `extreme-finite`, and `ill-conditioned`. The oracle owns z-score scaling, OVR MAP iterations, augmented Hessian, jittered Cholesky inverse, predictive correction, and softmax. It compares every scaler value, weight, bias, covariance entry, probability, shape, finiteness condition, and positive covariance diagonal against a genuine public CPU artifact/output.

- [ ] **Step 5: Run the three fixture/oracle tests GREEN**

Run the support target with exact test filters one at a time, then together. Expected: three passed, zero failed, zero ignored.

- [ ] **Step 6: Commit fixture and oracle support**

Run:

```powershell
git add crates/neoethos-models/tests/support/bayesian_r6/fixture.rs crates/neoethos-models/tests/support/bayesian_r6/oracle.rs crates/neoethos-models/tests/bayesian_full_gpu_r6_support.rs
git commit -m "test(models): anchor Bayesian r6 fixtures and oracle"
```

### Task 3: Build exact kernel and D2H evidence validators

**Files:**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/cuda_evidence.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/timing.rs`
- Create: `crates/neoethos-models/tests/support/bayesian_r6/provenance.rs`
- Modify: `crates/neoethos-models/tests/bayesian_full_gpu_r6_support.rs`

- [ ] **Step 1: Define exact stage identities before validation code**

Use a closed enum and exact symbols:

```rust
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum BayesianCudaStage {
    Preprocess,
    MapUpdate,
    Hessian,
    Cholesky,
    Inference,
}

const CANONICAL_STAGE_SYMBOLS: [(BayesianCudaStage, &str); 5] = [
    (BayesianCudaStage::Preprocess, "neoethos_bayesian_preprocess_f64"),
    (BayesianCudaStage::MapUpdate, "neoethos_bayesian_map_update_f64"),
    (BayesianCudaStage::Hessian, "neoethos_bayesian_hessian_f64"),
    (BayesianCudaStage::Cholesky, "neoethos_bayesian_cholesky_f64"),
    (BayesianCudaStage::Inference, "neoethos_bayesian_inference_f64"),
];
```

`KernelIdentity` equality is based on canonical symbol/stage, not activity index. Repeated activities aggregate under one identity.

- [ ] **Step 2: Write the exact positive and five negative kernel tests**

Add these test names:

```rust
    five_exact_canonical_bayesian_kernel_identities_with_bound_d2h_pass
    generic_decoy_plus_cpu_backend_reports_every_missing_stage
    stuffed_mega_kernel_does_not_match_any_canonical_identity
    duplicate_canonical_activity_rows_count_as_one_identity
    five_activity_indices_with_one_stuffed_name_count_as_zero_identities
    cpu_outputs_plus_five_decoys_and_native_metadata_are_rejected
```

The positive has all five exact symbols, ordered ranges, dimension-bound grid work/H2D, and distinct posterior/probability D2H correlations. The negatives use plausible timings and metadata so the intended semantic defect—not malformed setup—causes rejection.

- [ ] **Step 3: Add exact public-owner byte/correlation tamper coverage**

Define a `DeviceOutputBinding` carrying capture nonce, output kind, producing stage, NVTX range/correlation identity, dimensions, byte length, post-call public-owner SHA-256, and the static source-flow receipt that identifies the successful D2H destination and its no-write path into that public owner. Add:

```rust
#[test]
fn tampered_public_owner_byte_or_d2h_correlation_is_rejected() { /* both mutations */ }
```

The validator compares the canonical post-call public-owner bytes against independently encoded artifact/probability bytes, then requires the separate static no-transform ownership proof and correlated dimension-exact D2H activity. It does not invent a payload hash from Nsight rows. One changed public-owner byte, source-flow identity, or correlation must fail despite correct native metadata and canonical names.

- [ ] **Step 4: Preserve raw timing and dynamic identity tests**

Add:

```rust
timing_receipt_preserves_raw_samples
dynamic_git_identity_rejects_dirty_or_non_object_output
```

Require one excluded warm-up, exactly three non-zero raw samples in original order, a derived median, 40-hex live Git object identities, and empty porcelain status.

- [ ] **Step 5: Add exact paid argv and deployment-receipt tests**

Add:

```rust
paid_argv_requires_exact_offline_statistical_gpu_features
blob_and_bundle_receipts_reject_checkout_or_transfer_drift
```

The exact paid argv is represented as an array, not parsed from a display string. Mutate or remove each of `--locked`, `--offline`, `-j7`, `--no-default-features`, `--features`, `statistical-gpu`, target name, exact parent name, `--nocapture`, and `--test-threads=1`; every mutation must fail.

The deployment receipt negative changes bundle hash, commit, tree, one blob hash, checkout cleanliness, transfer mode, and executable hash independently.

- [ ] **Step 6: Run the complete support target**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test bayesian_full_gpu_r6_support -- --nocapture
```

Expected exact result: **14 passed; 0 failed; 0 ignored** with these names:

1. `public_cpu_posterior_and_probabilities_match_independent_oracle_for_all_cases`
2. `five_exact_canonical_bayesian_kernel_identities_with_bound_d2h_pass`
3. `generic_decoy_plus_cpu_backend_reports_every_missing_stage`
4. `stuffed_mega_kernel_does_not_match_any_canonical_identity`
5. `duplicate_canonical_activity_rows_count_as_one_identity`
6. `five_activity_indices_with_one_stuffed_name_count_as_zero_identities`
7. `cpu_outputs_plus_five_decoys_and_native_metadata_are_rejected`
8. `tampered_public_owner_byte_or_d2h_correlation_is_rejected`
9. `tiny_fixture_constants_anchor_parent_and_worker_encodings`
10. `fixture_hashes_bind_train_and_oos_features_and_labels`
11. `timing_receipt_preserves_raw_samples`
12. `dynamic_git_identity_rejects_dirty_or_non_object_output`
13. `paid_argv_requires_exact_offline_statistical_gpu_features`
14. `blob_and_bundle_receipts_reject_checkout_or_transfer_drift`

- [ ] **Step 7: Commit the independently GREEN validators**

Run:

```powershell
git add crates/neoethos-models/tests/support/bayesian_r6/cuda_evidence.rs crates/neoethos-models/tests/support/bayesian_r6/timing.rs crates/neoethos-models/tests/support/bayesian_r6/provenance.rs crates/neoethos-models/tests/bayesian_full_gpu_r6_support.rs
git commit -m "test(models): reject Bayesian r6 decoy evidence"
```

## Chunk 2: Cheap source and process behavior REDs

### Task 4: Enforce hidden-CPU and exact D2H flow on the real GPU route

**Files:**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/source_contract.rs`
- Create: `crates/neoethos-models/tests/bayesian_gpu_route_embargo_r6_contract.rs`

- [ ] **Step 1: Implement a statement-ordered lexical environment**

The environment pushes/pops function, closure, block, match-arm, and module scopes. It records each binding at its statement position and resolves only the nearest live binding. It follows path, method, UFCS, function-item, reference, parenthesized/grouped, closure, and local-alias expressions. Cycles and unsupported forms return a diagnostic.

- [ ] **Step 2: Build a route-sensitive public call graph**

Locate exactly one `ExpertModel for BayesianLogitExpert` implementation. Starting from `fit`, `predict_proba`, and `predict_runtime`, resolve the policy dispatch and select only the explicit `Gpu { ordinal: 0 }`/equivalent `gpu:0` route. Report a missing, ambiguous, or unrecognized GPU arm.

- [ ] **Step 3: Add five synthetic tests**

Use these names:

```rust
direct_ufcs_alias_and_closure_cpu_references_are_rejected
cpu_lease_taint_cannot_enter_gpu_numerical_scope
legal_cpu_arm_does_not_contaminate_isolated_gpu_arm
posterior_and_probability_d2h_ownership_flows_are_required
lexical_shadowing_resolves_the_binding_live_at_each_call
```

The positive synthetic GPU fixture has a unique fit host destination passed by mutable reference to a successful D2H call that dominates its no-write flow into every artifact field, and an inference destination with the same proof into the returned matrix. Record D2H call span, producing device buffer, destination binding, success guard, permitted moves/shape views, and every public-owner sink. A write through an alias, host numerical transform, unrelated buffer, failed/unchecked copy, or non-dominating copy breaks the proof. Each negative changes one path without changing surrounding names.

- [ ] **Step 4: Add one aggregate live-source test**

```rust
#[test]
fn real_public_bayesian_gpu_route_has_no_hidden_cpu_and_is_d2h_bound() {
    let failures = inspect_real_public_gpu_route();
    assert!(failures.is_empty(), "all GPU-route violations:\n{}", failures.join("\n"));
}
```

On authority `7824e19`, require the report to identify public `fit -> fit_cpu`, public `predict_proba -> predict_proba_cpu`, lease numerical scope, missing native GPU dispatch, missing posterior D2H-to-artifact flow, and missing probability D2H-to-return flow. Do not abort after the first finding.

- [ ] **Step 5: Run the route contract and classify the intentional RED**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test bayesian_gpu_route_embargo_r6_contract -- --nocapture
```

Expected exact result: **5 passed; 1 intentionally failed; 0 ignored**. The sole failing test is the live-source test and contains every expected current-production violation. Any synthetic failure, parser error, warning, or extra live test failure is unexpected.

- [ ] **Step 6: Commit the source RED**

Run:

```powershell
git add crates/neoethos-models/tests/support/bayesian_r6/source_contract.rs crates/neoethos-models/tests/bayesian_gpu_route_embargo_r6_contract.rs
git commit -m "test(models): embargo hidden CPU Bayesian GPU routes"
```

### Task 5: Strengthen training-summary constructor and validator semantics

**Files:**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/summary_contract.rs`
- Create: `crates/neoethos-models/tests/training_summary_embargo_r6_contract.rs`

- [ ] **Step 1: Reuse the scoped environment, not a file-global map**

Resolve constructor/type aliases and row aliases at each exact expression position. Record semantic role provenance as a set of `Dataset`, `Train`, `Embargo`, and `Validation`; fail closed on unknown provenance.

- [ ] **Step 2: Require four ordered singleton constructor flows**

Production constructors must resolve to:

```rust
[
    RowRoleSet::only(Dataset),
    RowRoleSet::only(Train),
    RowRoleSet::only(Embargo),
    RowRoleSet::only(Validation),
]
```

Pairwise distinctness is mandatory. Arity four alone is insufficient.

- [ ] **Step 3: Require a live terminating invariant guard**

Normalize only the exact equality `dataset == train + embargo + val`. Accept a top-level mismatch guard with `bail!`/`return Err` and equivalent `ensure!`. Require the guard to dominate successful return. Reject dead assignments, logging-only branches, debug assertions, constant-false ancestors, never-called closures, post-return expressions, missing roles, or duplicates.

- [ ] **Step 4: Add six passing synthetic/census tests**

Use these names:

```rust
alias_resolver_covers_use_type_local_and_qualified_constructors
scoped_binding_resolver_handles_inner_and_restored_outer_shadowing
constructor_dataflow_rejects_duplicate_and_shadowed_arguments
validator_accepts_live_bail_and_ensure_four_way_invariants
validator_rejects_dead_arithmetic_nonterminating_and_false_guards
fail_all_helpers_accumulate_every_named_producer_and_validator
```

- [ ] **Step 5: Add four real-source behavior REDs**

Use these names:

```rust
recursive_81_file_ast_census_requires_typed_four_way_summary
all_three_real_summary_producer_failures_are_reported_together
all_three_real_summary_validator_failures_are_reported_together
scope_aware_constructor_inventory_requires_four_distinct_row_flows
```

- [ ] **Step 6: Run and classify the exact embargo RED**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test training_summary_embargo_r6_contract -- --nocapture
```

Expected exact result on `7824e19`: **6 passed; 4 intentionally failed; 0 ignored**. The producer and validator failures each name Bayesian, linear, and deep sites together. Any lifetime/compiler error, early abort, synthetic failure, or different count is unexpected.

- [ ] **Step 7: Commit the semantic embargo RED**

Run:

```powershell
git add crates/neoethos-models/tests/support/bayesian_r6/summary_contract.rs crates/neoethos-models/tests/training_summary_embargo_r6_contract.rs
git commit -m "test(models): require live four-way embargo invariants"
```

### Task 6: Contain and reap complete process trees

**Files:**

- Create: `crates/neoethos-models/tests/support/bayesian_r6/process_tree.rs`
- Create: `crates/neoethos-models/tests/paid_process_tree_r6_contract.rs`

- [ ] **Step 1: Define one reusable containment interface**

Use one owner type for utility commands and acceptance workers:

```rust
struct ContainedChild { /* platform owner + child + pipes */ }

impl ContainedChild {
    fn spawn(command: &mut Command, context: &'static str) -> Result<Self, ProcessTreeError>;
    fn try_wait(&mut self) -> Result<Option<ExitStatus>, ProcessTreeError>;
    fn terminate_tree_and_wait(&mut self, deadline: Instant) -> Result<ExitStatus, ProcessTreeError>;
    fn finish_and_wait(&mut self, deadline: Instant) -> Result<ExitStatus, ProcessTreeError>;
}
```

`Drop` calls the same bounded terminate-and-wait path when ownership is still live. It never silently ignores an active descendant; if `Drop` cannot report an error, it records a shared fatal receipt checked by the owning test before success.

- [ ] **Step 2: Implement race-free Windows Job Object containment**

Create the process suspended, assign it to a kill-on-close Job Object, then resume. Do not enable breakaway. On termination, close stdin, terminate the job, wait the root, require active-process count zero, join readers, and close every handle.

- [ ] **Step 3: Implement Linux process-group containment**

Create a dedicated process group before `exec`. Kill the negative PGID, wait root, escalate within a short deadline, verify `ESRCH`/no group, and join readers. Keep unsupported targets compile-time fail-closed.

- [ ] **Step 4: Write the survivor helper, two containment tests, and the paid-state trace test**

The ignored helper spawns a grandchild that writes a heartbeat and ignores the parent protocol. Tests are:

```rust
timeout_kills_and_waits_for_the_entire_descendant_tree
drop_kills_and_waits_for_the_entire_descendant_tree
paid_state_trace_rejects_intervening_work_and_postclaim_cpu
```

Each containment test records the grandchild PID/creation identity, observes at least one heartbeat, triggers timeout or drop, then requires root reaped, zero job/group processes, grandchild absent, and no heartbeat advance after a bounded observation window. The trace test first accepts `Prepared -> ReceiptSynced -> Claimed -> OsSpawnAttempt`, then injects preparation/allocation between claim and spawn and CPU numerical work after claim; both traces must be rejected with all offending events reported.

- [ ] **Step 5: Run the process contract locally**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test paid_process_tree_r6_contract -- --nocapture --test-threads=1
```

Expected exact result: **3 passed; 0 failed; 1 ignored**. Inspect native process state after the test; no helper or grandchild may survive.

- [ ] **Step 6: Commit process containment**

Run:

```powershell
git add crates/neoethos-models/tests/support/bayesian_r6/process_tree.rs crates/neoethos-models/tests/paid_process_tree_r6_contract.rs
git commit -m "test(models): contain Bayesian r6 process trees"
```

## Chunk 3: Public lifecycle, sole compile RED, and paid parent

### Task 7: Add the public lifecycle contract and isolate the one missing API

**Files:**

- Create: `crates/neoethos-models/tests/bayesian_full_gpu_r6_contract.rs`
- Create: `crates/neoethos-models/tests/bayesian_full_gpu_r6_acceptance.rs`

- [ ] **Step 1: Add CPU control and one ignored real GPU lifecycle**

The contract target's CPU control uses a legal deterministic three-class fixture and real `ExpertModel::fit`/`predict_proba`; it checks convergence and non-empty normalized probabilities. Its one ignored GPU lifecycle uses only public fit, predict, save, and load; requires exact same-device repeatability, native CUDA metadata, artifact integrity, corruption rejection, and transactional receiver preservation. It contains no profiler child entry point.

- [ ] **Step 2: Run the contract without ignored tests**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_contract -- --nocapture
```

Expected exact result: **1 passed; 0 failed; 1 ignored**. The ignored lifecycle test must not appear as executed.

- [ ] **Step 3: Create one typed adapter containing the sole missing call**

Use the canonical name exactly once:

```rust
fn execute_cpu7_with_accepted_lease<R, Work>(
    executor: &BudgetedCpuExecutor,
    transfer: CpuLeaseTransfer,
    work: Work,
) -> Result<R, BudgetedCpuExecutorError>
where
    R: Send,
    Work: FnOnce(&CpuLease) -> R + Send,
{
    executor.execute_with_lease(transfer, work)
}
```

Do not add `execute_scoped_with_lease`, a trait shim, a conditional fallback, a test extension method, or a second invocation. The acceptance parent and CPU worker both route through this adapter.

- [ ] **Step 4: Make all acceptance code type-complete before the compile RED**

Use explicit closure argument and return types, complete platform cfg branches, and existing independently GREEN support types. The acceptance target contains exactly two ignored tests: `serialized_parent_owns_all_exact_shapes_and_verdicts` and `profiled_child_runs_exactly_one_shape`. Search that target and require exactly one source occurrence of `.execute_with_lease(` and zero occurrences of `execute_scoped_with_lease`.

- [ ] **Step 5: Compile the acceptance target and inspect the full log**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_acceptance --no-run
```

Expected: non-zero exit with exactly one `error[E0599]`, at the adapter call, naming absent `execute_with_lease` on `BudgetedCpuExecutor`. Expected warning count is zero. Any other diagnostic family or a second E0599 stops the plan.

- [ ] **Step 6: Commit the public contract and authentic compile RED**

Run:

```powershell
git add crates/neoethos-models/tests/bayesian_full_gpu_r6_contract.rs crates/neoethos-models/tests/bayesian_full_gpu_r6_acceptance.rs
git commit -m "test(models): pin Bayesian r6 public acceptance RED"
```

### Task 8: Complete the ignored paid parent without running it

**Files:**

- Modify: `crates/neoethos-models/tests/bayesian_full_gpu_r6_acceptance.rs`

- [ ] **Step 1: Replace every direct child with `ContainedChild`**

Both `bounded_output` and `WorkerProcess` use the shared process-tree owner. The `nsys` process is the job/group root, so its test-worker descendants cannot outlive timeout, disconnect, panic unwind, or `Drop`. `WorkerProcess` wraps `current_exe()` with `nsys` and passes `--ignored --exact profiled_child_runs_exactly_one_shape --nocapture --test-threads=1`; therefore the exact acceptance executable built and hashed for the outer parent is also the only Bayesian child executable.

- [ ] **Step 2: Persist independent fixture and output identities**

Before timing, compare worker train-feature, train-label, OOS-feature, and OOS-label hashes to the parent reference hashes calibrated by the tiny constants. Immediately after each public call, persist canonical public in-memory posterior bytes, artifact JSON, returned probability bytes, static source-flow receipts, and all hashes. Do not label these post-call bytes as an independently sampled D2H payload; their device provenance is the conjunction of source-flow, canonical output equality, and correlated profiler evidence.

- [ ] **Step 3: Query exact Nsight/CUPTI/NVTX rows**

Export SQLite, inventory schemas defensively, read exact name columns and correlation/range columns, preserve every raw row, and feed normalized records into the already-GREEN validator. Exact canonical identity, stage order, meaningful work, and two output bindings are mandatory.

- [ ] **Step 4: Build the `FreePreflight` receipt before claim**

Complete and persist:

- exact argv/environment validation;
- live integrated implementation commit/tree, frozen R6 contract commit/tree, and exact executable hash;
- Git-bundle transfer receipt and fresh detached-checkout proof;
- canonical source/vendor blob ledgers and no ignored/untracked dependency inputs;
- `Cargo.lock` and locked/offline metadata closure;
- process-tree behavioral receipt;
- tiny and large fixture identities;
- exact CPU7 control plus all CPU timing samples;
- CPU/oracle normal, extreme-finite, and ill-conditioned receipts; and
- Nsight/CUDA/device preflight without a Bayesian GPU workload.

Call `sync_all` on every file and its containing directory where supported before leaving `FreePreflight`.

- [ ] **Step 5: Prepare completely, then make claim and OS spawn adjacent**

Before the claim, create and validate paths, exact `nsys`/child argv and environment, profiler configuration, pipe/handle ownership, containment configuration, deadlines, and a single-use `PreparedProfileSpawn`. Persist and sync the preclaim receipt only after this preparation succeeds. The state transition is then structurally:

```rust
let prepared = WorkerProcess::prepare_profiled(first_shape, parent_deadline)?;
persist_and_sync_preclaim_receipt(&prepared, &preflight)?;
let claim = claim_single_paid_attempt(&evidence_base, &preflight)?;
let first_gpu = prepared.spawn_now(&claim)?;
```

`spawn_now` performs no allocation, path/pipe creation, serialization, source check, or other fallible preparation. Its first fallible operation is the containment-mediated OS process creation (suspended creation before Job assignment/resume on Windows; process-group creation at spawn on Linux). No CPU call, Git/Cargo command, new receipt directory, profiler export, sleep, loop, or retry occurs between claim and that call. Add an injected event-trace contract that rejects an allocation/preparation event between claim and OS spawn, and separately rejects any CPU numerical event after claim.

- [ ] **Step 6: Use one profiler tree per width and stop first**

For width 64, one worker performs lifecycle, warm-up, three timed fit-plus-OOS-predict samples, save/load, and oracle output. After it exits, export and validate all evidence before starting width 128. Repeat only if width 64 passed. At most one process tree exists. No CPU workload follows claim.

- [ ] **Step 7: Pin ceilings and permanent no-retry behavior**

Retain exact widths `[64, 128]`, `TRAIN_ROWS=1_000_000`, `OOS_ROWS=131_071`, one warm-up, three samples, five-minute readiness, eight-minute command deadline, 30-minute parent ceiling, one exclusive lock, one create-new permanent claim, and no retry mechanism.

- [ ] **Step 8: Pin the exact paid argv receipt**

The stored argv array must correspond exactly to:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_acceptance -- --ignored --exact serialized_parent_owns_all_exact_shapes_and_verdicts --nocapture --test-threads=1
```

Store `CARGO_INCREMENTAL=0`, `RUSTFLAGS=-Dwarnings`, the evidence directory, and the one-run sentinel separately. Never serialize a weaker display-only command.

- [ ] **Step 9: Re-run only the acceptance compile RED**

Expected classification remains the sole `E0599` for `execute_with_lease`. This task must not introduce a second compiler error despite adding all ignored runtime logic.

- [ ] **Step 10: Commit the ignored paid parent**

Run:

```powershell
git add crates/neoethos-models/tests/bayesian_full_gpu_r6_acceptance.rs
git commit -m "test(models): bound Bayesian r6 paid evidence flow"
```

## Chunk 4: Local classification and portable freeze

### Task 9: Run the one authorized local sequence and preserve raw outputs

**Files:**

- Create: `audit/bayesian-full-gpu-r6-red/README.md`
- Create: `audit/bayesian-full-gpu-r6-red/commands.txt`
- Create: `audit/bayesian-full-gpu-r6-red/local-01-support.txt`
- Create: `audit/bayesian-full-gpu-r6-red/local-02-route-red.txt`
- Create: `audit/bayesian-full-gpu-r6-red/local-03-summary-red.txt`
- Create: `audit/bayesian-full-gpu-r6-red/local-04-process-tree.txt`
- Create: `audit/bayesian-full-gpu-r6-red/local-05-public-contract.txt`
- Create: `audit/bayesian-full-gpu-r6-red/local-06-acceptance-compile-red.txt`

- [ ] **Step 1: Obtain authorization, bundle the RED contract, and use a fresh checkout**

Confirm no other broad Cargo process is active and sufficient disk space exists. From the clean Task 8 commit, record `<red-contract-source-commit>`, create a dedicated advertised tag, make a local canonical bundle, verify/list its head, hash it, require a nonexistent fresh destination, clone `--no-checkout`, and detach the exact commit:

```powershell
git tag bayesian-r6-red-contract-<full-red-contract-commit> <full-red-contract-commit>
git rev-parse 'bayesian-r6-red-contract-<full-red-contract-commit>^{commit}'
git bundle create '<persistent-evidence-path>\bayesian-r6-red-contract-<full-red-contract-commit>.bundle' 'refs/tags/bayesian-r6-red-contract-<full-red-contract-commit>'
git bundle verify '<persistent-evidence-path>\bayesian-r6-red-contract-<full-red-contract-commit>.bundle'
git bundle list-heads '<persistent-evidence-path>\bayesian-r6-red-contract-<full-red-contract-commit>.bundle'
Get-FileHash -Algorithm SHA256 -LiteralPath '<persistent-evidence-path>\bayesian-r6-red-contract-<full-red-contract-commit>.bundle'
if (Test-Path -LiteralPath '<fresh-R6-contract-checkout>') { throw 'fresh checkout destination already exists' }
git clone --no-checkout '<persistent-evidence-path>\bayesian-r6-red-contract-<full-red-contract-commit>.bundle' '<fresh-R6-contract-checkout>'
git -C '<fresh-R6-contract-checkout>' checkout --detach <full-red-contract-commit>
git -C '<fresh-R6-contract-checkout>' rev-parse HEAD
git -C '<fresh-R6-contract-checkout>' rev-parse 'HEAD^{tree}'
git -C '<fresh-R6-contract-checkout>' status --porcelain=v1 --untracked-files=all
```

Require the tag and advertised bundle head to resolve to the exact RED-contract commit, the fresh status to be empty, and the fresh tree to match the recorded contract tree. From the fresh checkout, run `cargo +nightly-2026-04-07 metadata --locked --offline --format-version=1 --no-deps`, then use one isolated fresh-checkout target for all six gates; do not clean between them. Persist logs outside the checkout until they are added to the original R6 evidence commit. This bundle proves only the portable RED contract and is never an RTX/paid implementation payload.

- [ ] **Step 2: Run Gate 1 support GREEN**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test bayesian_full_gpu_r6_support -- --nocapture
```

Expected: 14/14 passed.

- [ ] **Step 3: Run Gate 2 route behavior RED**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test bayesian_gpu_route_embargo_r6_contract -- --nocapture
```

Expected: 5 passed, exactly 1 intended live-source failure with complete hidden-CPU and D2H findings.

- [ ] **Step 4: Run Gate 3 summary behavior RED**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test training_summary_embargo_r6_contract -- --nocapture
```

Expected: 6 passed, exactly 4 intended real-source failures with all named producers/validators.

- [ ] **Step 5: Run Gate 4 process-tree GREEN**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --test paid_process_tree_r6_contract -- --nocapture --test-threads=1
```

Expected: 3 passed, 1 ignored helper, and no surviving descendant.

- [ ] **Step 6: Run Gate 5 public contract GREEN/ignored**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_contract -- --nocapture
```

Expected: 1 passed, 1 ignored.

- [ ] **Step 7: Run Gate 6 sole compile RED**

Run:

```powershell
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_acceptance --no-run
```

Expected: exactly one `E0599` for `execute_with_lease`, zero warnings, and no other error family.

- [ ] **Step 8: Preserve complete output and classifications**

Each `.txt` file contains exact environment, working directory, target path, command array, complete stdout and stderr as separate byte streams with their own sequence order, native exit status, test counts/names, and expected/actual classification. Separate pipe readers do not claim a total cross-stream order. Do not use `.log`; verify every evidence file is visible to ordinary `git status`.

- [ ] **Step 9: Stop on any mismatch**

Do not reinterpret an unexpected diagnostic as an intended RED. Preserve the raw file, report the first mismatch, and wait for authorization before editing or rebuilding.

### Task 10: Freeze canonical R6 RED-contract blobs and specify the later deployment mode

**Files:**

- Create: `audit/bayesian-full-gpu-r6-red/red-contract-source-blobs.sha256`
- Create: `audit/bayesian-full-gpu-r6-red/red-contract-vendor-blobs.sha256`
- Create: `audit/bayesian-full-gpu-r6-red/manifest.sha256`
- Modify: `audit/bayesian-full-gpu-r6-red/README.md`

- [ ] **Step 1: Record the already-committed RED-contract source identity**

Immediately after Task 8 and before Gate 1, require a clean checkout and record that full HEAD/tree as the **R6 RED-contract source commit/tree**. It has zero protected production diff and its acceptance target still produces the sole `E0599`; it is not a built implementation identity and cannot authorize a paid executable. Do not modify tests, lockfile, or support code after Gate 1 starts.

- [ ] **Step 2: Preserve local classifications in a distinct RED-evidence commit**

After the six gates, add only the raw `.txt` logs, README, and manifests in an **R6 RED-evidence commit**. Every receipt names the RED-contract source commit/tree. Never label this evidence commit or the RED-contract commit as the implementation used by a paid executable.

- [ ] **Step 3: Generate ledgers from committed RED-contract blobs**

Use `git ls-tree -r -z --full-tree <red-contract-source-commit>` and `git cat-file blob <object-id>`. Record mode, Git blob ID, byte length, SHA-256, and Git-relative path sorted by raw path bytes. Generate the vendor subset independently. Exclude each ledger/manifest from its own hash scope. The ledger explicitly records the six R6 integration targets and all support modules so a later integrated candidate can prove their mode/blob IDs unchanged.

- [ ] **Step 4: Prove every dependency/evidence input is tracked or absent**

Run `git ls-files --error-unmatch` for every manifest entry. Enumerate both ordinary and ignored untracked inputs in the exact dependency/evidence scopes:

```powershell
git ls-files --others --exclude-standard -- ':(glob)**/Cargo.toml' 'Cargo.lock' 'rust-toolchain*' '.cargo/**' ':(glob)**/build.rs' ':(glob)**/src/**' 'crates/neoethos-models/tests/**' 'vendor/**' 'audit/bayesian-full-gpu-r6-red/**'
git ls-files --others --ignored --exclude-standard -- ':(glob)**/Cargo.toml' 'Cargo.lock' 'rust-toolchain*' '.cargo/**' ':(glob)**/build.rs' ':(glob)**/src/**' 'crates/neoethos-models/tests/**' 'vendor/**' 'audit/bayesian-full-gpu-r6-red/**'
git diff --name-only 7824e191c04b4eb78e547728ad7cdb78f915a2af -- ':(glob)**/src/**'
```

Expected: both untracked enumerations and the protected production diff are empty. Any ignored vendor, path dependency, test input, raw log, or manifest is a freeze failure.

- [ ] **Step 5: Final immutable verification of the R6 RED freeze**

Run Git-only checks after the RED-evidence commit:

```powershell
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git merge-base HEAD 7824e191c04b4eb78e547728ad7cdb78f915a2af
git status --porcelain=v1 --untracked-files=all
git diff --name-only 7824e191c04b4eb78e547728ad7cdb78f915a2af -- ':(glob)**/src/**'
git fsck --no-dangling
```

Expected: clean RED-evidence commit, merge base exactly `7824e19`, empty protected diff, valid objects, and a Task 9 fresh-bundle receipt reproducing the sole `E0599` from the RED-contract source commit. Do not amend after reporting the immutable identities.

- [ ] **Step 6: Clean only the reviewed isolated target**

After verifying no compiler owns it and its resolved absolute path is the exact R6 target, run:

```powershell
cargo +nightly-2026-04-07 clean --locked --offline -p neoethos-models --target-dir '<exact-reviewed-R6-target>'
```

Report removed bytes and free-space delta. Do not touch another target or the rejected R5 clone.

#### Deferred integrated-candidate and RTX protocol — not executable in this R6 plan

Production implementation, candidate bundling, transfer, successful acceptance build, and paid execution require separate authorization after independent acceptance of the R6 RED freeze. That later phase must create a distinct **integrated implementation commit/tree**, prove every frozen R6 contract path has the exact RED-contract mode/blob ID, close all live-source REDs, and make the acceptance target compile. It must then create an advertised immutable ref and bundle that ref:

```powershell
git tag bayesian-r6-integrated-<full-integrated-commit> <full-integrated-commit>
git rev-parse 'bayesian-r6-integrated-<full-integrated-commit>^{commit}'
git bundle create '<persistent-evidence-path>\bayesian-r6-<full-integrated-commit>.bundle' 'refs/tags/bayesian-r6-integrated-<full-integrated-commit>'
git bundle verify '<persistent-evidence-path>\bayesian-r6-<full-integrated-commit>.bundle'
git bundle list-heads '<persistent-evidence-path>\bayesian-r6-<full-integrated-commit>.bundle'
Get-FileHash -Algorithm SHA256 -LiteralPath '<persistent-evidence-path>\bayesian-r6-<full-integrated-commit>.bundle'
scp '<persistent-evidence-path>\bayesian-r6-<full-integrated-commit>.bundle' 'neoethos-vast-<gpu>:/persistent/neoethos/bayesian-r6/'
```

The only transfer payload is that SHA-256-pinned bundle over the configured `neoethos-vast-<gpu>` SSH alias; no worktree copy, zip, rsync tree, uncommitted patch, or alternate archive may substitute. On the RTX host, before the permanent claim, verify the received SHA-256, require a nonexistent checkout destination, clone `--no-checkout` from the bundle into that new directory, detach at the exact full integrated commit, and verify exact HEAD/tree, clean porcelain, bundle advertised ref, source/vendor ledgers, and frozen R6 mode/blob IDs:

```bash
sha256sum -- /persistent/neoethos/bayesian-r6/bayesian-r6-<full-integrated-commit>.bundle
test ! -e /persistent/neoethos/bayesian-r6/checkout-<full-integrated-commit>
git clone --no-checkout /persistent/neoethos/bayesian-r6/bayesian-r6-<full-integrated-commit>.bundle /persistent/neoethos/bayesian-r6/checkout-<full-integrated-commit>
git -C /persistent/neoethos/bayesian-r6/checkout-<full-integrated-commit> checkout --detach <full-integrated-commit>
git -C /persistent/neoethos/bayesian-r6/checkout-<full-integrated-commit> rev-parse HEAD
git -C /persistent/neoethos/bayesian-r6/checkout-<full-integrated-commit> rev-parse 'HEAD^{tree}'
git -C /persistent/neoethos/bayesian-r6/checkout-<full-integrated-commit> status --porcelain=v1 --untracked-files=all
```

From that fresh checkout, all commands remain locked/offline. Require support **14/14**, route **6/6**, summary **10/10**, process **3 passed/1 ignored**, public contract **1 passed/1 ignored**, then run this successful compile-only command with both acceptance tests still ignored:

```bash
CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_acceptance --no-run --message-format=json
```

Select the single compiler-artifact whose target is `bayesian_full_gpu_r6_acceptance`, persist its exact path/size/SHA-256, and only then invoke the exact paid command from the design. The RED-contract or RED-evidence identities can never satisfy this integrated-candidate gate.

## Execution boundary

Completing this plan produces an independently reviewable **RED-only test bundle**, not a production Bayesian CUDA implementation and not paid-card authorization. Production work begins only in a new, separately authorized task after an independent reviewer accepts the R6 source/tests/evidence freeze.
