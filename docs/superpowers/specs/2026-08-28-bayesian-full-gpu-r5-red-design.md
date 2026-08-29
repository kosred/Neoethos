# Bayesian Logistic Full-GPU R5 RED Design

## Authority and scope

- Source provenance starts at `45c64ccacd3c42d5bd07cccbf8985020b931ef47`; the R5 commits are replayed onto its reviewed authoritative descendant `b114c01b036820e42c90ef6c10fa7aa35a1838de` (via `fc4cf49` and `97a66b1`) in the isolated clone.
- R5 may change tests, test-only dependencies, documentation, evidence, and the tracked vendor closure required by the preserved workspace.
- R5 must not change `crates/neoethos-models/src`, `crates/neoethos-core/src`, or `crates/neoethos-execution-budget/src`.
- No R5 command may contact the network, a GPU, a VPS, or a registry. Cargo commands use `--locked --offline`.
- Runtime acceptance records the checked-out implementation commit and tree dynamically. The provenance and preimplementation authority above are never the claimed tested implementation.

## R4 rejection and chosen approach

R4 was rejected because its CPU7 executor call was only an identity probe, its math checks were easy accuracy checks rather than an independent posterior oracle, its Nsight validator accepted any CUDA activity, its evidence omitted raw inputs/timings/executable and dependency closure identity, its clean clone did not resolve without ignored vendor files, its timeout/workload count was unsafe for the remaining credit, and its AST producer test could be bypassed by aliases and stopped at the first producer failure.

Three approaches were considered:

1. **Compile-RED production API prerequisite plus independently GREEN validators (selected).** A compile fixture calls the wished-for lease-bearing executor API. The real-card target uses that same API, so it cannot silently substitute a probe. Math, CUPTI-ledger, provenance, and AST validators compile and run without the missing API.
2. **AST-only executor inspection.** This would keep the acceptance target compilable by using a test adapter, but it would not prove that timed public model calls received the accepted lease. Rejected.
3. **Implement the executor API in R5.** This would erase the intended RED and violate the production-source boundary. Rejected.

## Exact CPU7 prerequisite

The wished-for production boundary is a `BudgetedCpuExecutor` operation that:

1. consumes a `CpuLeaseTransfer` issued by that executor's broker;
2. accepts that transfer exactly once;
3. selects a private Rayon pool whose width equals the accepted lease width;
4. enters the accepted lease scope and invokes `FnOnce(&CpuLease)` from that private pool;
5. retains the accepted lease until the callback and every scoped worker task finish.

The compile-RED fixture requests an exact seven-permit lease and calls the wished-for operation. Inside its callback it requires a width-seven delivered lease, a width-seven current Rayon pool, seven distinct `neoethos-cpu-*` native worker identities, and rejection of fresh nested acquisition in every worker context. It snapshots native per-thread CPU time in all seven workers immediately before and after the real public fit plus OOS prediction and requires meaningful CPU-time growth in every worker. The timed calls receive the delivered lease directly. A broadcast-only identity probe or a serial fit surrounded by decoy worker activity cannot satisfy the fixture.

The pinned API exposes only `execute`/`execute_scoped`, which consume and hide the accepted lease. Therefore this fixture must fail only on the absent lease-bearing method until production supplies the boundary.

## Independent Bayesian oracle

A test-owned oracle implements the mathematical contract independently of production helpers:

- deterministic z-score preprocessing;
- chronological train/embargo/validation split;
- three one-vs-rest f64 MAP logistic fits;
- augmented full Hessian `Z^T S Z + alpha I`;
- jittered Cholesky inverse for the full Laplace covariance;
- predictive `z^T Sigma z` correction and three-class softmax.

The oracle compares saved public-model weights, biases, full covariance matrices, and OOS probabilities for normal, extreme finite, and ill-conditioned/collinear fixtures. Local CPU-to-oracle checks validate the oracle without hardware. The ignored RTX acceptance requires CPU-to-GPU, CPU-to-oracle, and GPU-to-oracle parity for the same parent-owned fixture hashes.

## Independent CUDA evidence

The parent runs the public GPU lifecycle under Nsight Systems and exports profiler-owned SQLite. The validator resolves CUPTI kernel names from the database and requires five distinct, minimum-duration `neoethos_bayesian_*` activities, one for each stage:

- preprocessing;
- MAP update;
- Hessian construction;
- Cholesky/factorization;
- inference.

It also requires non-trivial aggregate named-stage duration, dimension-bound named-stage grid work, and host-to-device/device-to-host transfer bytes consistent with the hashed fixture dimensions. Generic kernels, CUDA API traffic alone, missing semantic stages, zero-duration activity, or CPU/fallback backend metadata fail. Local negative tests prove both that a decoy kernel plus CPU result is rejected and that one name-stuffed mega-kernel cannot impersonate five distinct stages.

## Reproducibility and evidence

Before running hardware work, the parent requires a clean Git tree and persists:

- dynamic implementation commit and tree;
- hashes of relevant implementation sources;
- exact test executable hash;
- `Cargo.lock` hash;
- `cargo metadata --locked --offline` output and hash;
- a per-file tracked vendor closure manifest, its count/bytes, and aggregate hash;
- train-feature, train-label, OOS-feature, and OOS-label hashes;
- every raw warm-up and timed duration, not only medians;
- public artifacts, Nsight report/SQLite hashes, oracle outputs, and final receipts.

The R5 commit tracks every local path dependency required by the preserved workspace. A fresh clone must not need ignored or untracked vendor content.

## Budgeted real-card flow

One parent owns an exclusive evidence-directory lock and atomically writes a permanent paid-attempt claim before launching hardware work, so a retry requires explicit human review. Work is serialized and stops at the first failed gate. Each exact shape gets one excluded warm-up and three timed samples per role; one profiled GPU lifecycle supplies semantic kernel evidence. The parent enforces short per-command deadlines and a single 30-minute wall ceiling for the entire hardware test. It kills the active child on timeout and never starts the next shape after a failure. There are no repeated six-hour waits.

## Alias-resistant embargo census

The recursive 81-file AST census finds the typed summary schema plus all three exact production producers and validators. It resolves renamed, chained, and nested imports; type aliases; qualified paths; `Self`; parenthesized callees; local constructor bindings; and local row-count bindings instead of matching one spelling. It accumulates diagnostics for Bayesian, linear, and deep sites and asserts only after all three were inspected, so each RED report lists all producer or validator failures in one run.

## Expected local classification

- Lease-bearing CPU7 compile fixture: **RED**, only because the production executor method is absent.
- Public CPU lifecycle support: **GREEN**; the real GPU lifecycle is compiled but ignored outside the one paid parent.
- Embargo producer and validator censuses: **RED**, each listing all three stale sites.
- Independent oracle self-checks, CUPTI semantic/decoy validators, provenance validators, recursive parsing, and ignored-test listing: **GREEN**.
- Real-card parent: compile-gated by the missing executor API and ignored locally; it is never executed in R5.
