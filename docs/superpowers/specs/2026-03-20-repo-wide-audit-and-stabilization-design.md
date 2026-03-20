# Repo-Wide Audit And Stabilization Design

## Goal

Run a complete, evidence-first audit of the current codebase after the recent Python-to-Rust migration work, identify correctness issues and warning sources across all active files, and define the stabilization baseline required before UI/backend integration resumes.

## Why This Exists

The repository has changed significantly in the last few days:

- the active branch is now `master`
- the runtime is much more Rust-centric than the earlier hybrid state
- a new desktop UI exists in `crates/forex-app`
- MT5 integration currently flows through `crates/mt5-bridge`
- multiple historical assumptions about the Python runtime are no longer reliable

Because of that, any work that starts directly with UI fixes or broad refactoring risks building on stale assumptions. The first subproject must therefore be a repo-wide audit and stabilization pass.

## Scope

This design covers only the first subproject:

- full repository census
- real-path runtime audit
- build/test/lint/warning audit
- line-by-line code audit across active files
- backend contract audit across CLI, app, data, model, search, and MT5 seams
- stabilization findings and prioritized fix plan

## Out Of Scope

The following are explicitly out of scope for this first subproject:

- implementing the UI/backend connection
- redesigning the UI
- deleting large parts of the codebase before reachability is proven
- broad code reduction targets such as “cut the codebase in half”
- feature expansion unrelated to audit findings
- full live trading validation against a real broker session

The UI is not ignored; it is inventoried and audited as code, but UI integration work is deferred until the runtime and backend contracts are stable.

## Primary Priorities

Priority order for this subproject:

1. correctness
2. explicit runtime behavior
3. warning elimination on enforced paths
4. contract clarity between subsystems
5. observability and diagnosability
6. performance and structural cleanup

This means a real runtime bug outranks a warning, and a silent contract mismatch outranks cosmetic cleanup.

## Audit Strategy

The audit is evidence-first. It must not devolve into an impressionistic code review.

Each finding must be backed by one or more of:

- a reproducible command
- a compiler/test/linter warning or failure
- a concrete file/line static finding
- an official documentation reference when behavior or APIs are uncertain

The user has explicitly requested that uncertainty be resolved via online verification. Therefore, any uncertain external behavior, crate API, framework expectation, or platform contract must be checked against official sources before a fix is proposed.

## Audit Layers

### Layer A: Repository Census

Build a current map of:

- workspace crates
- binaries and entrypoints
- shared libraries/bindings
- remaining Python and shim files
- configs and manifests
- scripts and service wrappers
- tests and verification assets

This layer defines what “all files” means in the active repository, so later audit work can be exhaustive rather than ad hoc.

### Layer B: Static Verification Sweep

Run the repository-wide verification baseline for the active Rust-first system, including:

- `cargo check --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- targeted `cargo build` where binaries or feature gates require it

If Python tooling or generated bindings are still involved in the current build path, those checks are included as supporting evidence but do not redefine the runtime architecture.

### Layer C: Real-Path Runtime Sweep

Exercise the actual paths that define whether the system is alive:

- `forex-cli` data/discovery/training commands
- `forex-app --headless --local`
- `forex-app` GUI startup smoke validation
- MT5 bridge initialization and terminal-info contract checks without requiring full live trading

This layer is the primary guard against false confidence from clean compilation but broken runtime behavior.

### Layer D: Line-By-Line Code Audit

Read active files crate-by-crate and module-by-module to find:

- migration leftovers from the old Python architecture
- stale comments and mismatched documentation
- dead compatibility seams
- incomplete error handling
- silent fallbacks
- unsound assumptions across threads, async tasks, or FFI boundaries
- warning-prone patterns
- files whose responsibilities are no longer coherent

This is where “full audit of all files” happens, but after the dynamic evidence layers above have already identified the highest-risk areas.

### Layer E: Contract And Enterprise Hardening Audit

Evaluate subsystem boundaries and operational qualities:

- CLI contract
- app-to-engine contract
- MT5 bridge contract
- crate boundaries
- logging behavior
- metrics and progress reporting
- checkpointing and recovery points
- failure visibility
- reproducibility of startup and training flows

This layer translates “enterprise level” into concrete engineering checks rather than vague aesthetic goals.

## Finding Taxonomy

Every audit finding must be categorized as one of:

- build breakage
- test failure
- lint/warning
- runtime breakage
- correctness bug
- contract mismatch
- dead or unreachable code
- observability gap
- performance risk
- architectural smell

Every finding also gets:

- severity: critical, important, minor
- exact file path and line reference where applicable
- evidence
- root cause summary
- recommended fix direction

## Deliverables

This subproject must produce four outputs:

### 1. Audit Spec

This document, defining scope, method, and standards.

### 2. Audit Report

A living report of findings grouped by subsystem and severity.

### 3. Stabilization Plan

A prioritized execution plan that converts findings into fix tranches.

### 4. Verification Matrix

A set of commands and expected outcomes proving each critical subsystem is clean or showing where it is still failing.

## Success Criteria

This subproject is successful when:

- the current repository surface is fully enumerated
- the active runtime paths have been exercised and recorded
- all build/test/lint warnings and failures on enforced paths are known
- all active files have been reviewed or explicitly classified as out of scope for this tranche
- the repo has a concrete findings ledger
- the next implementation work can proceed from verified facts instead of assumptions

## Enterprise Baseline For This Project

For this codebase, “enterprise level” in the stabilization phase means:

- no build warnings on the enforced path
- no silent correctness failures
- clear subsystem contracts
- actionable errors and logs
- reproducible startup and training commands
- no mystery modules pretending to be active
- no hidden runtime forks left over from the migration

It does not mean “maximum abstraction” or “feature bloat”. The objective is defensible, inspectable engineering quality.

## Risks And Constraints

- The repo has changed recently enough that historical mental models may be wrong.
- Some MT5 behavior is platform- and installation-dependent, so official docs and runtime probes may both be needed.
- Some crates may compile cleanly while still having runtime contract gaps.
- The codebase may still contain intentionally temporary seams from the migration; these should be identified explicitly rather than erased blindly.

## Recommended Next Step

The next step is to write a concrete implementation plan for executing this audit in bounded tranches:

1. census and baseline verification
2. real-path runtime audit
3. full file-by-file audit
4. findings ledger and stabilization priorities

