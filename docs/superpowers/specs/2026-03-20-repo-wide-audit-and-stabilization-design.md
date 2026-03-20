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

For this subproject, “full repository audit” means:

- all workspace crates
- all top-level runtime/config/build assets
- all remaining Python files
- all scripts and service wrappers
- tests and examples as evidence-bearing supporting assets

It does not mean auditing generated logs, caches, build artifacts, or vendored third-party code as if they were first-party runtime modules.

## Out Of Scope

The following are explicitly out of scope for this first subproject:

- implementing the UI/backend connection
- redesigning the UI
- deleting large parts of the codebase before reachability is proven
- broad code reduction targets such as “cut the codebase in half”
- feature expansion unrelated to audit findings
- full live trading validation against a real broker session

The UI is not ignored; it is inventoried and audited as code, but UI integration work is deferred until the runtime and backend contracts are stable.

## Subsystem Classification Matrix

Every top-level subsystem must be classified before the audit begins:

| Subsystem | Status | Audit Treatment |
|----------|--------|-----------------|
| `crates/forex-app` | runtime-critical | full static + runtime + line-by-line audit |
| `crates/forex-cli` | runtime-critical | full static + runtime + line-by-line audit |
| `crates/mt5-bridge` | runtime-critical | full static + contract + environment audit |
| `crates/forex-core` | runtime-critical | full static + line-by-line audit |
| `crates/forex-data` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-search` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-models` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-bindings` | runtime-critical | full static + Python-contract audit |
| `crates/forex-news` | audit-required | static + line-by-line audit, runtime only if reachable from active paths |
| remaining Python files | audit-required | classify as runtime code, bridge, bootstrap, or dead seam |
| top-level configs/scripts/services | audit-required | audit as integration assets |
| `tests/` and `examples/` | evidence-only | audit for coverage gaps and stale assumptions |
| `vendor/` | excluded from code-quality findings | inventory only unless a local patch affects behavior |
| `cache/`, `logs/`, `target/`, generated artifacts | excluded | evidence only, not first-party source |

## Active File Definition

For this subproject, a file is considered “active” if it falls into one of these classes:

- `runtime-critical`: directly used by supported app/CLI/runtime paths
- `static-only`: not exercised in the default runtime path, but compiled, loaded, or required by supported builds
- `audit-required bridge`: shim, binding, or compatibility file that can affect runtime contracts
- `evidence-only`: tests/examples used to validate assumptions but not treated as runtime modules

Every file or directory class outside those buckets must be marked explicitly as `excluded`, not left ambiguous.

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

Run the repository-wide verification baseline for the supported profiles, not an unbounded “all optional features everywhere” sweep.

The baseline must be split into named lanes:

- `required baseline lane`: supported local developer/runtime profile on the current OS
- `required Python-contract lane`: verifies Python-dependent runtime contracts that are still active
- `optional informational lanes`: heavyweight or platform-specific integrations that can surface issues without being promoted to baseline blockers

The baseline must not treat unsupported optional integrations as release blockers unless the project explicitly declares them supported for this environment.

Required static checks for the supported baseline include:

- `cargo check --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- targeted `cargo build` where binaries or supported feature gates require it

Informational lanes may include feature-specific or platform-specific builds, but those findings must be labeled `informational` unless they affect a supported baseline profile.

If Python tooling or generated bindings are still involved in the current build path, they are part of the audit baseline and must be verified explicitly rather than treated as historical leftovers.

### Layer C: Real-Path Runtime Sweep

Exercise the actual paths that define whether the system is alive:

- `forex-cli` data/discovery/training commands
- `forex-app --headless --local`
- `forex-app` GUI startup smoke validation
- MT5 bridge initialization and terminal-info contract checks without requiring full live trading

Every runtime probe must declare:

- prerequisites
- timeout
- expected environment
- result state

Allowed result states:

- `PASS`: command/path worked as expected
- `FAIL`: reproducible code or contract defect
- `BLOCKED`: environment prerequisite missing, with explicit reason
- `N/A`: path not applicable to the current OS or profile

Examples:

- GUI startup may be `BLOCKED` in a non-interactive session
- MT5 bridge may be `BLOCKED` if the local MT5 installation or `MetaTrader5` Python module is missing
- a graceful, explicit “offline/not available” path is not a `FAIL` if the code handles it as designed

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

This layer does not authorize broad redesign. It asks concrete audit questions:

- does this boundary have an explicit contract?
- does failure become visible and diagnosable?
- is recovery/checkpoint behavior explicit?
- are supported startup and execution paths reproducible?

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
- Some active runtime paths still depend on Python environments and bindings, so Python validation remains part of the supported audit surface.
- Some crates may compile cleanly while still having runtime contract gaps.
- The codebase may still contain intentionally temporary seams from the migration; these should be identified explicitly rather than erased blindly.

## Recommended Next Step

The next step is to write a concrete implementation plan for executing this audit in bounded tranches:

1. census and baseline verification
2. real-path runtime audit
3. full file-by-file audit
4. findings ledger and stabilization priorities
