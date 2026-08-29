# Bayesian Logistic Full-GPU R6 RED Design

## Status

This document is a design authority for a future RED-only test bundle. It does not authorize production implementation, a Cargo build, GPU execution, network access, VPS access, registry access, or a paid run. The R6 tests described here do not exist at this design commit.

## Authority, isolation, and protected scope

- R6 starts from the current authoritative commit `7824e191c04b4eb78e547728ad7cdb78f915a2af`, tree `e5c9cc23b1e97c6955d379d584f6fe555fe701dc`, parent `8bb73169027f520cd53b624604349eacea5c000c`.
- The isolated branch is `codex/bayes-logit-full-gpu-r6-red` in a new sibling clone. The rejected R5 checkout at `target/codex-stage/bayesian-r5-red/repo45` remains byte-for-byte outside R6.
- R5 is not the R6 parent. R5 contains useful historical results, but its reviewer-rejected test implementation is not an authority to replay. Starting at `7824e19` retains the current Search correction and prevents rejected R5 assumptions from silently becoming R6 source.
- R6 may eventually change integration tests, test-only dependencies, documentation, and tracked audit evidence. It must not change any tracked production path matched by `**/src/**`, including `crates/neoethos-models/src`, `crates/neoethos-core/src`, and `crates/neoethos-execution-budget/src`.
- Every R6 commit must have an empty `git diff` against `7824e19` for the repository-wide `:(glob)**/src/**` pathspec. A production-source change anywhere in the repository is a scope failure, not an R6 implementation step.
- Every future Cargo command that resolves dependencies uses `--locked --offline`; only build, test, and clean commands additionally use their supported concurrency/target arguments (`-j7` for build/test; no `-j` for metadata). All builds/tests set `CARGO_INCREMENTAL=0` and `RUSTFLAGS=-Dwarnings`. No ignored RTX test runs during local R6 work.

## Why R5 was rejected

The R5 review found no P0, but it found multiple proof gaps:

1. the public Bayesian GPU route could still reach CPU numerical work;
2. posterior and probability bytes were not proven to originate from exact device-to-host outputs;
3. five activity rows could impersonate five semantic kernels through duplicate or stuffed names;
4. training-summary validation accepted arbitrary binary expressions and a file-global alias map mishandled scope and shadowing;
5. timeout and drop killed only direct children, not profiler/worker descendants;
6. the paid command did not pin the exact feature set;
7. the freeze hashed checkout bytes and ignored `.log` evidence instead of proving portable committed/deployed bytes;
8. parent fixture expectations reused the worker generator; and
9. the permanent paid claim was created before cheap preflight and CPU/oracle work.

R6 uses a layered proof: cheap synthetic and source contracts reject these defects before any paid process, then the ignored acceptance parent combines independent profiler, byte, artifact, timing, and provenance evidence.

## Canonical CPU7 API prerequisite

The one desired production API name is **`BudgetedCpuExecutor::execute_with_lease`**. No test source or command receipt may substitute another lease-bearing spelling. Consistent naming matters because the acceptance compile gate is required to produce one isolated diagnostic.

The wished-for signature and semantics are:

```rust
pub fn execute_with_lease<R, Work>(
    &self,
    transfer: CpuLeaseTransfer,
    work: Work,
) -> Result<R, BudgetedCpuExecutorError>
where
    R: Send,
    Work: FnOnce(&CpuLease) -> R + Send;
```

The method must consume a transfer owned by the executor's broker, accept it once, select the private Rayon pool whose exact width equals the accepted lease, invoke the callback inside both that pool and the accepted lease scope, expose that exact accepted lease to the callback, and retain it until all callback work finishes. It may delegate internally to existing executor machinery, but it may not acquire an unrelated lease, narrow the width, manufacture a probe lease, or return capacity while scoped work is live.

The R6 acceptance target contains exactly one method call to `execute_with_lease`, inside one typed adapter. All surrounding types and callback return types are explicit so the absent method yields exactly one `E0599` family and no inference cascade. No other R6 target references the missing method. The compile RED is therefore:

- expected: one `E0599` whose receiver is `BudgetedCpuExecutor` and whose missing item is `execute_with_lease`;
- forbidden: any syntax error, lifetime error, warning, unresolved import, second missing API, `execute_scoped_with_lease`, or unrelated diagnostic family.

The callback receives the accepted width-seven lease. Inside the same timed callback, the real public CPU `fit` and OOS `predict_proba` calls receive that lease directly. Before and after those calls, the fixture observes seven distinct lease-bound `neoethos-cpu-*` worker contexts, native TIDs, exact pool width, nested-acquisition rejection, and meaningful per-worker native CPU-time growth. A separate broadcast probe cannot satisfy the contract.

## Real public GPU route: hidden-CPU embargo

The source contract starts from the actual `ExpertModel` implementation for `BayesianLogitExpert`, not from a test adapter or private CUDA helper. It builds a conservative lexical call graph and symbolically selects the arm reached when `statistical_device_policy("bayes_logit")` resolves to `gpu:0`.

The GPU-selected graph must:

- reach one native Bayesian CUDA fit path and one native CUDA probability path from the public methods;
- reject a missing or ambiguous GPU arm;
- reject direct method calls, UFCS calls, path references, function pointers, local aliases, closures, and transitive helper references to `fit_cpu` or `predict_proba_cpu`;
- allow the trait-mandated `&CpuLease` only in the public signature, while rejecting reads, aliases, captures, calls to `scope`, or passing that lease into GPU numerical work;
- reject a host `softmax_rows` or equivalent host numerical finish after the inference download, because the required D2H payload is the final probability matrix, not logits for later CPU completion; and
- fail closed on an unsupported dispatch or alias form rather than assuming it is safe.

CPU-only arms may retain `fit_cpu`, `predict_proba_cpu`, and `CpuLease::scope`. The embargo is route-sensitive: it forbids their reachability from the real GPU policy arm without deleting the legal CPU implementation.

Synthetic tests cover direct calls, UFCS, aliased function items, closure capture, transitive helpers, lease aliasing, a legitimate isolated CPU arm, and shadowed locals. One live-source test accumulates all GPU-route violations before asserting so the RED report is complete.

## Device-output data-dependency proof

Native metadata and kernel activity are corroboration, not proof that the public results came from the device. R6 requires three independent links.

### 1. Static ownership flow

The route analyser requires a fit host destination passed by unique mutable reference to the exact successful CUDA D2H operation. That operation must dominate every subsequent use, and the destination's canonical f64 payload must flow into the persisted scaler and all three class posteriors (weights, bias, and full covariance). It separately requires an inference host destination passed to the exact successful D2H operation whose payload becomes the returned `Array2<f64>` without host numerical transformation.

Aliases are resolved with lexical scope and statement order. Permitted transformations after the copy are ownership moves, checked shape construction, immutable views, canonical byte encoding, and serialization. Recalculation, element-wise mutation, another write through an alias, host softmax, replacing a field, or joining data from an unrelated buffer breaks the proof. The analyser records the source span of the D2H call, destination binding, producing device buffer, and no-write path to each public owner.

### 2. Independent profiler correlation

The parent exports Nsight Systems SQLite and reads raw CUPTI kernel, runtime/driver, memcpy, and NVTX rows. Each posterior or probability receipt names a unique capture nonce and correlation/range identity. The matching D2H activity must:

- occur after the producing canonical kernel range;
- be inside the correct posterior or inference download range;
- copy at least the exact dimension-derived byte count;
- have positive duration; and
- be unique for the claimed payload.

### 3. Exact canonical public-owner bytes

Immediately after the public call, the worker persists the public in-memory posterior fields and returned probability matrix as canonical bytes before any JSON conversion. The parent independently parses `model.json` and encodes posterior values in the specified order, and independently encodes the returned probability matrix. Each independent encoding must equal its corresponding post-call public-owner hash. Those bytes are tied to the device by the separately implemented static unique-owner/no-write proof and the correlated dimension-exact D2H activity; Nsight does not expose payload bytes, so R6 does **not** claim a separately sampled “D2H snapshot hash.” The canonical public-owner bytes, parsed artifact, returned values, source-flow receipt, profiler rows, and final receipt are all persisted.

The posterior encoding is schema tag, feature width, scaler means, scaler standard deviations, class count, then for each class: weight count, row-major f64 weights, bias, covariance rows/columns, and row-major covariance values. The probability encoding is schema tag, row count, column count, and row-major f64 values. Integers and `f64::to_bits()` are little-endian.

A required negative supplies CPU-computed posterior/probabilities, five CUDA decoy activities, and plausible native-CUDA metadata. It must fail canonical-kernel identity, static D2H-to-public-owner flow, and D2H correlation. A second negative changes one canonical public-owner byte or correlation identity and must fail even when every metadata string and dimension is plausible.

## Five exact canonical kernel identities

Activity indices are not identities. R6 requires these exact exported symbols:

| Stage | Canonical symbol |
|---|---|
| Preprocessing | `neoethos_bayesian_preprocess_f64` |
| MAP update | `neoethos_bayesian_map_update_f64` |
| Hessian | `neoethos_bayesian_hessian_f64` |
| Cholesky | `neoethos_bayesian_cholesky_f64` |
| Inference | `neoethos_bayesian_inference_f64` |

The Nsight extractor uses an exact raw exported/short symbol where available. If only a demangled name is available, it removes only a documented CUDA signature wrapper and then requires exact equality; arbitrary substring matching is forbidden. Repeated launches of one symbol aggregate duration and grid work under one identity and never increase the distinct-stage count.

Validation requires a set cardinality of five, exactly one canonical identity per stage, stage-appropriate positive duration/grid work, dimension-bound aggregate work, H2D before preprocessing, fit-stage ordering through Cholesky, inference after a fitted posterior, and correlated posterior/probability D2H transfers. Required negatives are:

- one generic decoy plus CPU backend;
- one name-stuffed mega-kernel;
- five activity rows with one duplicated canonical name;
- five rows carrying the same stuffed name with different activity indices; and
- CPU outputs plus five distinct decoy kernels plus native metadata.

Every negative reports all missing or contradictory evidence rather than aborting at the first item.

## Independent Bayesian oracle and fixture authority

The R5 independent f64 oracle design remains valid and is retained in R6: deterministic z-score preprocessing, three one-vs-rest MAP fits, full augmented Hessian, jittered Cholesky inverse, predictive variance correction, and three-class softmax. It compares weights, biases, full covariance matrices, and non-empty OOS probabilities for normal, extreme-finite, and ill-conditioned/collinear fixtures.

R6 corrects fixture circularity. A tiny fixture has literal, reviewed train and OOS values and labels. The parent owns four fixed 64-hex SHA-256 constants over four domain-separated byte streams. Every stream begins with the exact ASCII/NUL schema bytes `neoethos.bayesian.fixture.v1\0`, followed by a one-byte phase (`0x01` train, `0x02` OOS) and one-byte kind (`0x01` features, `0x02` labels):

- a feature stream then contains little-endian `u64` rows, little-endian `u64` columns, and row-major little-endian `f64::to_bits()` values;
- a label stream then contains little-endian `u64` label count and little-endian two's-complement `i32` labels.

The four constants are generated once by an independent offline calculation and copied into the parent test as reviewed literals. Neither the worker fixture generator nor any shared hashing helper calculates the expected constants at runtime. The separately written parent large-shape reference generator must first reproduce all four tiny literals, then generates the large train-feature, train-label, OOS-feature, and OOS-label identities. A bit flip, row/column reorder, feature/label substitution, train/OOS substitution, or label drift changes the identity.

## Training-summary embargo: semantic AST proof

The recursive 81-file `neoethos-models/src` census remains fail-all and alias-resistant, but R6 replaces the global local-binding map with a statement-ordered lexical environment. Function, closure, module, match-arm, and block scopes push and pop bindings; lookup resolves the nearest live binding at the call/comparison position. Unsupported destructuring or macro-generated construction fails closed.

### Constructor proof

For every production `TrainingSummaryMetadata::{new,new_unchecked,raw_for_validation}` construction, R6 resolves each argument to a semantic role set. The required ordered role sets are exactly:

```text
[{dataset_rows}, {train_rows}, {embargo_rows}, {val_rows}]
```

Each set must be a singleton and the four values must be distinct dataflows. Four arguments alone do not pass. `new(dataset, train, train, val)`, a shadowed embargo alias that resolves to train rows, or a calculated substitute without explicit embargo provenance fails. Test-only deliberately invalid constructors may be classified separately, but cannot make the production census pass.

### Validator proof

A validator passes only when a reachable guard that dominates its success exit enforces the normalized invariant:

```text
dataset_rows == train_rows + embargo_rows + val_rows
```

Accepted forms are a top-level mismatch guard whose violated branch executes `bail!` or `return Err(...)`, and an equivalent top-level `ensure!(equality, ...)`. Addition is normalized associatively, but the exact four semantic roles are required. A binary expression in dead arithmetic, an empty `if`, logging without rejection, `debug_assert!`, a guard under `if false`, a never-called closure, a comparison after unconditional return, or a guard that omits/duplicates a role fails.

Synthetic negatives cover dead arithmetic, duplicate constructor argument, shadowed alias, non-terminating guard, and constant-false guard. Real producer and validator diagnostics still accumulate Bayesian, linear, and deep failures before one assertion.

## Process-tree containment

One test-owned `ContainedChild` implementation is reused by both bounded utility commands and the paid `WorkerProcess`; the acceptance runner cannot use a weaker path.

### Windows

The runner creates a Job Object, sets `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, creates the root process suspended, assigns it to the job before resume, and does not allow breakaway. Timeout or `Drop` closes stdin, calls `TerminateJobObject`, waits for the root handle, queries job accounting until active processes are zero, joins pipe readers, and closes handles. Failure to assign, terminate, wait, or prove zero descendants is fatal.

### Linux

The runner creates a dedicated process group before `exec`. Timeout or `Drop` closes stdin, sends termination to the negative process-group ID, waits for the root, escalates to `SIGKILL` within a short bound, and verifies the group no longer exists before joining readers. Failure is fatal.

The behavioral contract launches a private helper that spawns a grandchild heartbeat process and then becomes unresponsive. Separate timeout and drop tests require the root to be reaped, the job/group survivor census to reach zero, the recorded grandchild identity to disappear, and the heartbeat to stop advancing. Merely killing the direct child fails. The paid profiler child is a second ignored test in `bayesian_full_gpu_r6_acceptance`; `WorkerProcess` invokes `current_exe()` with that exact child test name, so the built and hashed acceptance executable is also the only Bayesian worker executable.

## Canonical committed and deployed identity

Working-tree bytes are not the portable authority. R6 uses Git blob bytes and keeps four identities distinct:

1. **R6 RED-contract source commit/tree:** the accepted test-only R6 bundle, test dependency closure, and documentation, with repository-wide production `**/src/**` diff zero against `7824e19` and the sole acceptance `E0599` still present;
2. **R6 RED-evidence commit/tree:** a later test-only commit containing tracked local `.txt` RED/GREEN logs and manifests that point back to the RED-contract source identity;
3. **future integrated implementation commit/tree:** a separately authorized post-R6 commit containing the exact accepted R6 contract blobs plus the production implementation that makes the live source contracts and acceptance compile gate GREEN; this is the only identity from which a paid executable may be built; and
4. **paid evidence commit/tree:** a later commit containing tracked paid `.txt` logs and manifests that refer back to the integrated implementation identity.

Neither R6 RED identity nor a paid evidence commit may be relabelled as the tested implementation. Runtime code obtains the integrated implementation identity dynamically; no hard-coded authority, RED-contract commit, RED-evidence commit, or other preimplementation commit can pass as the tested implementation.

The R6 freeze ledger is generated from `git ls-tree -r -z --full-tree <red-contract-source-commit>`. A future paid source ledger is independently regenerated from `git ls-tree -r -z --full-tree <integrated-implementation-commit>`, and must prove that every accepted R6 contract path retains the exact frozen mode and blob ID. For every regular blob each ledger records mode, Git object ID, byte length, SHA-256 of `git cat-file blob` bytes, and path in ordinal byte order. A ledger excludes itself to avoid recursion. Vendor paths are exact subset ledgers. Symlinks, submodules, ignored dependency inputs, untracked dependency inputs, duplicate paths, and missing blobs fail.

This blob ledger is the EOL authority. Checkout CRLF/LF conversion is not used for identity. Future raw logs use a tracked `.txt` extension, or are explicitly force-added and then proven by `git ls-files --error-unmatch`; ignored `.log` files cannot satisfy the manifest.

The only accepted RTX source transfer mode is a SHA-256-pinned Git bundle produced from the future exact integrated implementation commit. Creating or transferring that bundle is outside the R6 RED-only plan and requires separate production and RTX authorization. Before the permanent paid claim:

1. create and verify the bundle locally;
2. transfer that exact bundle through the approved SSH/SCP route;
3. verify its SHA-256 on the RTX host;
4. clone it into a new empty directory;
5. detach-checkout the exact integrated implementation commit;
6. require exact `HEAD`, tree, clean status, blob/vendor ledgers, and unchanged accepted R6 contract blob IDs;
7. run locked/offline dependency metadata and require all R6 support, route, summary, process, and public non-ignored gates GREEN (the R6 live-source REDs must now be closed);
8. build the exact statistical-GPU acceptance target successfully, proving the R6 `execute_with_lease` E0599 is closed, without running an ignored test; and
9. record the executable path, byte count, and SHA-256.

A copied worktree, zip file, rsync directory, uncommitted patch, or checkout needing an ignored vendor file is rejected.

## Exact paid command and receipt

The future paid command is exactly:

```powershell
$env:CARGO_INCREMENTAL='0'
$env:RUSTFLAGS='-Dwarnings'
cargo +nightly-2026-04-07 test --locked --offline -j7 -p neoethos-models --no-default-features --features statistical-gpu --test bayesian_full_gpu_r6_acceptance -- --ignored --exact serialized_parent_owns_all_exact_shapes_and_verdicts --nocapture --test-threads=1
```

The receipt stores the executable/argv as an array, working directory, exact environment allowlist, integrated implementation commit/tree, frozen R6 contract identity, and executable hash. A string that omits `--no-default-features`, changes the feature set, names another test target, runs more than one test thread, or adds another ignored test is rejected.

## Paid-flow state machine

The ignored parent has explicit `FreePreflight`, `Claimed`, `PaidGpu`, and `Finalized` states.

All of these finish in `FreePreflight`:

1. sentinel and exclusive-parent lock validation;
2. exact command and environment receipt validation;
3. Git-bundle transfer, fresh checkout, blob ledger, clean-tree, vendor, lockfile, dependency, and executable verification;
4. process-tree timeout/drop survivor contracts or their exact preflight receipt;
5. tiny fixture constants and all four large-shape fixture hashes;
6. exact CPU7 executor behavior, CPU timing samples for every feature width, and meaningful seven-worker CPU-time evidence;
7. normal, extreme-finite, and ill-conditioned CPU/oracle gates;
8. Nsight/CUDA tool and device preflight that launches no Bayesian workload; and
9. complete preparation of profiler paths, exact child argv/environment, inherited handles/pipes, containment configuration, and a single-use `PreparedProfileSpawn`; and
10. persistence and `sync_all` of the complete preclaim receipt.

After that receipt is durable, the next two fallible operations are:

```text
create_new(permanent paid-attempt claim)
PreparedProfileSpawn::spawn_now(first nsys-profiled Bayesian GPU process tree)
```

There is no CPU workload, source verification, allocation, path creation, pipe setup, retry loop, or unrelated command between them. `spawn_now` is single-use and its first fallible operation is the containment-mediated OS process creation; it performs no hidden preparation before that call. An injected event-trace validator rejects any event between claim and OS spawn and any CPU numerical event after claim. The claim is permanent even if spawn or the first GPU gate fails.

For each feature width, one profiled worker tree performs lifecycle, one excluded warm-up, three timed fit-plus-OOS-predict samples, artifact save/load, and oracle output. The parent kills/waits the complete tree, exports and validates the report, persists raw evidence, and stops before the next width on any failure. At most one profiler tree exists at a time. The entire parent retains the 30-minute wall ceiling, five-minute readiness ceiling, eight-minute command ceiling, one parent, two ordered widths, no retries, and stop-on-first-failure.

## Planned local gate map

All counts below are requirements for the future R6 implementation, not results from this design-only commit.

| Gate | Expected classification |
|---|---|
| `bayesian_full_gpu_r6_support` | 14 passed; 0 failed; 0 ignored |
| `bayesian_gpu_route_embargo_r6_contract` | 5 synthetic tests passed; 1 live-source test intentionally failed with aggregated GPU-route/dataflow diagnostics |
| `training_summary_embargo_r6_contract` | 6 synthetic/census tests passed; 4 real-source tests intentionally failed |
| `paid_process_tree_r6_contract` | 3 passed; 0 failed; 1 ignored private survivor helper |
| `bayesian_full_gpu_r6_contract` without ignored tests | 1 passed; 0 failed; 1 ignored public GPU lifecycle |
| `bayesian_full_gpu_r6_acceptance --no-run` | compile RED with exactly one `E0599` for `execute_with_lease`; no other warning/error family |
| RTX acceptance parent + profiled child | both ignored and unrun until a separately authorized integrated implementation and one paid attempt |

The valid R5 oracle, lifecycle, timing, and single-diagnostic results remain historical evidence. They are not counted as R6 execution evidence and do not waive fresh R6 gates.

## Design-phase exit criteria

The design phase may be committed only when:

- only the R6 design, implementation plan, and their manifest changed;
- the branch base is exact `7824e19`;
- protected production diff versus `7824e19` is empty;
- the rejected R5 status is unchanged;
- `execute_with_lease` is the sole desired API spelling;
- no Cargo, GPU, network, VPS, registry, or paid command ran; and
- the documentation manifest hashes canonical document bytes and excludes itself.
