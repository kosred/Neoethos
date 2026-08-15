# Unified CPU Execution Budget Design

**Date:** 2026-08-15

**Status:** Architecture A approved by the operator and independent design review

**Scope:** NeoEthos build parallelism and aggregate CPU-bound execution limits across the root, `mcp`, and `mesh` Cargo workspaces and the process trees NeoEthos manages

## Objective

NeoEthos must use the machine it is actually allowed to use, while leaving capacity equivalent to one SMT core free for operating-system stability. This is an aggregate ceiling for CPU-bound NeoEthos work in one managed process tree, not a separate allowance that every pool and child may consume again. In portable terms, the default CPU-work ceiling is:

```text
effective_available_logical_threads = std::thread::available_parallelism()
reserved_logical_threads            = min(2, effective_available_logical_threads - 1)
automatic_worker_limit              = max(1, effective_available_logical_threads - 2)
effective_worker_limit              = min(
                                        automatic_worker_limit,
                                        persistent_requested_limit if present,
                                        legacy_persistent_limit if present,
                                        parent_assigned_limit if present
                                      )
```

Every requested limit is optional and must be positive. Each one may narrow the automatic ceiling, but none may restore the two reserved logical threads or exceed the capacity visible to the process.

Examples:

| Effective logical threads | Automatic workers | Explicit request | Effective workers |
|---:|---:|---:|---:|
| 1 | 1 | none | 1 |
| 2 | 1 | none | 1 |
| 4 | 2 | none | 2 |
| 12 | 10 | none | 10 |
| 12 | 10 | 4 | 4 |
| 12 | 10 | 9999 | 10 |
| 23 | 21 | none | 21 |

The 23-thread case is not hypothetical: the CUDA VPS exposed 96 host logical CPUs through `sysinfo`, but its cgroup quota made Rust's `available_parallelism()` return 23. The correct NeoEthos worker ceiling there is 21, not 94 and not 95.

## Current inconsistencies

The current source does not enforce one answer:

- `neoethos_core::system::resolve_cpu_budget` defaults to `total - 1`; an oversized explicit value clamps to `total`, consuming the intended reserve.
- `HardwareProbe::detect` uses `sysinfo.cpus().len()`, which reports host topology rather than the process/cgroup allowance and therefore overstates VPS capacity.
- `SystemConfig::default` and `neoethos-models::cpu_threads_hint` independently compute `cores - 1`.
- `neoethos-search` leaves Rayon at its default when no override exists, which uses all available logical CPUs.
- multiple `#[tokio::main]` entry points construct runtimes before the resolved settings and execution budget are available.
- the scheduler divides `HardwareProfile::cpu_cores`, so a raw or stale host count propagates into every child assignment.
- `.cargo/config.toml` fixes every rustc process at `-Z threads=8` but does not set Cargo's job limit. On small machines that can oversubscribe; on large machines it leaves capacity unused.

These are not merely naming differences. They can make the app, search, model trainers, child processes, logs, and VPS scheduler disagree about the same run. Giving the same value to several independent pools is also not an aggregate limit: search, model training, Tokio blocking work, and two child processes could each consume the entire value concurrently.

## Chosen architecture: one budget authority plus hierarchical leases

Create a zero-dependency leaf crate, `neoethos-execution-budget`, containing portable CPU-capacity detection, pure resolution logic, immutable process-budget installation, and an RAII permit broker built from `std::sync`. It must not depend on `neoethos-core`, Tokio, Rayon, tracing, GPU libraries, or ML libraries.

This boundary is necessary because the repository contains three deliberately separate Cargo workspaces. The root engine can reach the crate through `neoethos-core`; the isolated `mcp` and `mesh` workspaces can use the same small crate by path without importing the trading engine's GPU/ML dependency graph.

The crate exposes three layers:

1. Pure, deterministic calculation from `(available_logical_threads, requested_limit)` for exhaustive unit tests.
2. Process installation from `std::thread::available_parallelism()` before any global pool or async runtime is created.
3. A process-global `CpuPermitBroker` whose initial permit count is the installed final limit and whose non-cloneable `CpuLease` values can be split, transferred, and returned by RAII.

The resolved value is immutable for the life of a process. A second installation with the same inputs is idempotent. A conflicting second installation is a startup error rather than a silent first-writer-wins condition.

The capacity record must retain all of the following so diagnostics do not conflate them:

- host logical threads, when a host inventory API can report them;
- effective logical threads available to this process;
- reserved logical threads;
- automatic worker limit;
- optional persistent, legacy-persistent, and parent-assigned limits with separate provenance;
- final effective worker limit.

Only the final effective worker limit may seed the permit broker or drive CPU-bound execution.

### Aggregate accounting contract

Every repository-controlled CPU-bound production entry point must own a lease before it starts work:

- pure Rust parallel work acquires a lease before entering a Rayon pool;
- model/native-library adapters receive the lease width and configure their library's **total** CPU-worker count to no more than that width;
- CPU-heavy Tokio work acquires a lease before `spawn_blocking`; ordinary `tokio::spawn` remains for I/O and short orchestration only;
- a managed child process consumes a parent lease for its entire lifetime and receives that exact width through `--cpu-threads`;
- GPU feeders and CPU fallbacks consume leases, while device kernels do not.

Acquisition completes before work is submitted to Rayon or Tokio, so pool workers never wait for permits while holding work needed by the lease owner. Code holding a lease may split that lease for nested work, but must not acquire fresh permits; debug assertions and the API shape reject nested acquisition to prevent deadlock.

The leaf crate exposes immediate `try_acquire` and a blocking wait used only by a dedicated coordinator thread or a synchronous top-level CLI thread. Async handlers never wait on a `Mutex`/`Condvar` and never call blocking acquisition on a Tokio core worker. The app-side adapter submits an `AdmissionRequest` to one dedicated coordinator thread and awaits a Tokio oneshot; cancellation removes or marks the queued request without needing the permit, and child-exit/lease-drop wakes the coordinator. Child reservations have explicit priority over opportunistic local work, with FIFO ordering inside each priority. Thus permit scarcity delays the job, not the runtime responsible for heartbeat, stop, cancellation, and child cleanup.

Rayon pools are no longer independent authorities. Search and model code execute through a `BudgetedCpuExecutor`: it creates or selects a pool whose width equals the admitted lease and holds that lease until all scoped work completes. A cached pool may retain idle OS threads, but it cannot execute CPU work without a matching lease. The existing global-search, local-trainer, and native-model paths must all be routed through this executor or a native adapter with the same lease contract. Repository checks inventory direct production uses of Rayon, `spawn_blocking`, OpenMP/native thread setters, and `num_cpus` so an unmanaged path cannot silently reappear.

For each native library, the adapter documents from current upstream documentation whether its thread parameter includes the caller and any helper threads. The adapter translates lease width accordingly and has a concurrency probe in tests. If a library cannot be bounded at its API boundary, that backend is unsupported for concurrent production use until an OS-level containment test exists; it does not receive a guessed thread count.

### Transitive/global thread-pool policy

Repository grep is not enough because dependencies can initialize hidden global pools. The implementation inventory combines `cargo tree` feature graphs, current dependency source/documentation, environment/thread-setting APIs, and measured thread-name/concurrency probes for at least Vortex scan/decode work, ndarray-Rayon, DuckDB, Burn CPU, CubeCL CPU fallbacks, BLAS/OpenMP, and every enabled tree-model backend. Every enabled CPU backend is classified as `lease_native_width`, `exclusive_global_pool`, `single_thread_under_partial_lease`, `device_only`, or `unsupported_concurrent`; an unclassified backend fails startup when its path is requested.

### Vortex-only data boundary and mandatory Polars removal

The product decision is one supported data engine: Vortex. Vortex is the sole production persisted and out-of-core columnar format and the sole scan/projection/filter layer. `Ohlcv` and `FeatureFrame` remain application-owned typed views over decoded or projected Vortex data; they are not a second storage/query backend. Small bounded frames may be materialized as model-native ndarray/slice memory, but a disk-backed feature frame is Vortex, not a custom file format. Model adapters consume `FeatureFrame` accessors or model-native views, with typed label vectors. No model trait, trainer, ensemble adapter, test fixture, importer, or public API may expose a Polars `DataFrame`, `Series`, or `LazyFrame`, and no Polars compatibility wrapper is introduced.

This is a mandatory first implementation phase, before the CPU-budget integration can be accepted. The current reviewed tree has not completed the earlier intended removal: Polars is a direct dependency of `neoethos-data` and `neoethos-models`, occurs in 42 Rust source/test files, and `cargo tree -p neoethos-app -i polars` reaches the production app through both crates. Those model interfaces are migrated to `FeatureFrame`/typed targets.

Runtime discovery, loading, charting, feature computation, training, backtest, and live paths accept only validated `.vortex` datasets. The explicit user import boundary remains a supported production feature: a user may supply CSV, TSV, JSON, JSONL, Parquet, Arrow IPC/Feather, or an existing Vortex file and receive a canonical Vortex dataset. The import source vocabulary is split from runtime storage identity: `ImportSourceFormat` owns those source variants, while the canonical dataset layer has only Vortex. A non-Vortex source can never be registered, cached, scanned, or returned as a runnable dataset.

The app `/data/import` workflow and the CLI import command call one shared import service. It detects or validates the declared source format only inside that boundary, streams bounded batches, maps aliases through an explicit schema contract, preserves `i64` timestamps and `f64` OHLCV prices, and fails the whole import on a malformed, null, non-finite, inconsistent, or schema-ambiguous market row instead of silently dropping it or synthesizing a timestamp. It writes a temporary canonical Vortex file, reopens and verifies it, then atomically publishes it with source hash/schema/provenance metadata. Failure leaves no discoverable partial output. Vortex input is validated and atomically registered without decode/re-encode when safe. Runtime `loader.rs` never auto-converts: a direct CSV/Parquet/etc. runtime path fails with an instruction to import it first. The existing `validate_f32_precision`/`F32_DOWNCAST_TOLERANCE` import gate is deleted: import never downcasts broker prices through f32 or rejects valid f64 values for failing an f32 round trip.

Import is a budgeted production workload, not an exception. The app awaits coordinator admission before it creates the import `spawn_blocking` task; the CLI obtains a synchronous top-level lease before entering the shared service. The admitted lease is transferred into the job and remains live across parsing, source hashing, decompression, validation, Vortex encode/write, reopen verification, atomic publish, and complete failure cleanup. The service divides that one reservation among bounded pipeline stages rather than acquiring independent permits that could deadlock. Arrow CSV/JSON/IPC readers, Parquet decompression, compression codecs, hashing, and Vortex workers are included in the dependency-pool classification; any worker width that cannot be bounded or accounted for fails closed under the same unsupported-concurrent policy.

The importer is Polars-free. CSV/TSV and JSON/JSONL use bounded direct/Arrow readers; Parquet uses the official Arrow `ParquetRecordBatchReaderBuilder` with a bounded batch size and projection; Arrow IPC uses its file/stream readers. Record batches convert through Vortex/Arrow interoperability or the same typed OHLCV validator, never through a DataFrame compatibility layer. Broker downloads and historical Bid/Ask capture bypass import and write versioned canonical Vortex schemas directly.

The existing feature-major `.fstore` mmap is also migrated, not grandfathered. The replacement writes a Vortex `StructArray` schema containing row identity/timestamp and named feature columns in bounded chunks, and represents out-of-core `FeatureFrame`/window state with a Vortex path plus immutable schema/range metadata. Gene/model consumers request only the named columns and row ranges they need through Vortex projection/filter pushdown; a budgeted, bounded decoded-chunk cache prevents repeated full scans without becoming a second on-disk format. Temporary feature Vortex files live in a run-scoped directory, are atomically finalized before use, are removed by RAII on normal exit, and are swept by manifest/age on next startup after a crash. No `.fstore` writer, reader, extension, or compatibility fallback remains after parity and performance gates pass.

Removal acceptance is workspace-wide, not just an app feature check: no `polars` dependency declaration, package in `Cargo.lock`, result in any workspace `cargo tree`, Rust import, fully-qualified use, feature flag, or stale Polars-specific runtime setting may remain. A second guard allows non-Vortex format handling only inside the explicit import module/API/CLI adapter, rejects any such branch in runtime discovery/load/cache/model paths, rejects non-Vortex canonical outputs, and rejects the `.fstore` format/path. Reporting exports such as a validation summary CSV are not market-data engines and remain separately named. Until both proofs pass, the branch is an incomplete migration and must not claim either Vortex-only operation or a bounded production CPU path. We do not add `POLARS_MAX_THREADS` or a Polars scheduling adapter for code whose approved end state is deletion.

Vortex itself is still subject to the execution budget. Its current Rust interface represents tabular data as named `StructArray` fields and supports scan-time projection/filter pushdown; it is not treated as a mutable DataFrame API. Every Vortex scan, decode, compression, and write path is inventoried for dependency-owned CPU work and admitted through the same lease contract. If the locked Vortex version internally launches work whose width cannot be bounded, the affected concurrent path is marked unsupported or isolated until measurement and an upstream-supported containment mechanism prove the aggregate ceiling.

The guarantee is about admitted CPU-bound workers, not the literal number of process threads. Tokio I/O workers, sleeping cached workers, GPU-driver threads, and short control-plane activity may exist without consuming a CPU-work permit. Logs must use this exact distinction; the design does not claim a portable hard OS CPU quota.

## Configuration and precedence

`system.hardware.cpu_budget` remains the operator's persistent narrowing control. `--cpu-threads` remains the scheduler's per-child ephemeral narrowing control.

`models.backtest_runtime.rayon_threads` is retired as a second operator control. During one compatibility window the sealed settings loader still accepts it as a `legacy_persistent_limit`, validates it as positive, and emits one structured WARN naming the replacement. If both old and new keys exist, their minimum wins; the legacy value can never enlarge `system.hardware.cpu_budget`. The settings API and knob catalog expose only `system.hardware.cpu_budget`, and the next settings save writes only the canonical key. Search and model consumers stop reading the legacy field directly. A retired-key regression proves an old file loads with the same or narrower limit, warns, and round-trips without the old key.

Resolution order is:

```text
OS/cgroup available_parallelism
    -> subtract the fixed two-thread stability reserve
    -> narrow by system.hardware.cpu_budget, if present
    -> narrow by retired models.backtest_runtime.rayon_threads, if present
    -> narrow by parent --cpu-threads assignment, if present
    -> install one immutable process budget
```

The parent assignment is a separate startup input; it must not mutate either persistent setting. `Settings::apply_process_cpu_assignment`, which currently overwrites both fields, is removed and replaced by pure `ExecutionBudgetInputs::from_settings_and_parent`. Every cap composes by minimum. Zero in either YAML key fails in the sealed `Settings` load boundary; zero or malformed `--cpu-threads` fails argument parsing before any runtime is constructed.

Legacy `system.n_jobs` stays derived and non-settable. Its diagnostic text and computed value must use the unified resolver and say “effective logical threads minus two,” not “cores minus one.”

Required startup fields distinguish every source rather than collapsing them:

```text
host_logical_threads
effective_available_logical_threads
reserved_logical_threads
automatic_worker_limit
persistent_requested_worker_limit
legacy_persistent_worker_limit
parent_assigned_worker_limit
effective_worker_limit
active_limit_sources
```

## Startup sequence

Every executable must install the budget before creating a CPU pool or a multi-thread Tokio runtime.

For settings-aware executables (`neoethos-app`, `neoethos-cli`, and desktop), startup becomes:

1. Parse only the arguments needed to locate configuration.
2. Load and validate settings.
3. Parse any parent `--cpu-threads` cap without mutating settings.
4. Resolve and install the process execution budget.
5. Install hardware overrides first, followed by search/model/data runtime settings.
6. Construct Tokio/Rayon/native-library execution facilities.
7. Emit one structured startup record with the complete capacity calculation.

The `#[tokio::main]` entry points must be replaced with small synchronous entry points that build a Tokio multi-thread runtime using the resolved worker limit and then call an async body. The settings-free top-level `mcp` and `mesh` sidecars install the automatic or parent-assigned budget before building their runtimes.

### Production entry-point matrix

| Entry point | Settings source | Parent cap | Runtime action |
|---|---|---|---|
| `crates/neoethos-app/src/main.rs` | explicit `--config` | accepted when scheduler-spawned | synchronous preflight, then bounded Tokio runtime |
| `crates/neoethos-cli/src/main.rs` | canonical sealed settings | `--cpu-threads` | install budget before any search/model override or command |
| `desktop/src-tauri/src/lib.rs` | resolved desktop `config.yaml` | none today | synchronous seed/load, custom Tokio handle, then Tauri builder |
| `crates/neoethos-mcp/src/main.rs` | root control-plane config | optional parent cap | synchronous preflight, then bounded Tokio runtime |
| top-level `mcp/src/main.rs` | sidecar config | desktop `--cpu-threads` when managed | synchronous preflight, then bounded Tokio runtime |
| top-level `mesh/src/main.rs` | sidecar arguments/state | desktop `--cpu-threads` when managed | synchronous preflight, then bounded Tokio runtime |

### Desktop preflight and Tauri runtime

Desktop first-run seeding moves before `tauri::Builder::default()`. Path selection is split into a Tauri-independent synchronous `resolve_data_root()` and a seed step that writes the exact build-time bundled `resources/config.yaml` and broker metadata resource via `include_bytes!`; it no longer needs `App::path()` inside `.setup`. Seed and config-load failures keep the existing visible fail-closed dialog semantics.

After loading settings, desktop installs the budget, builds a `tokio::runtime::Builder::new_multi_thread()` with the resolved worker limit, retains the `Runtime` local for the entire blocking Tauri event loop, and calls `tauri::async_runtime::set(runtime.handle().clone())` exactly once **before** `tauri::Builder::default()`. Tauri's current contract says the underlying runtime must not be dropped and `set` panics if a runtime was already installed. Pure ordering tests prove no Tauri builder or plugin/setup hook can run first; a packaged first-run test proves the embedded seed and resolved editable file are byte-identical before settings migration.

Tokio thread count is a ceiling on runtime workers, not a claim that all threads are continuously busy. CPU-heavy work must still be routed through the budgeted Rayon/native-library lanes; blocking training must not be allowed to multiply unchecked behind Tokio tasks.

## Hardware profile and truthful reporting

`HardwareProfile` must stop using a raw `sysinfo` CPU count as schedulable capacity. Host inventory and live effective capacity are separate facts.

The existing serialized profile remains backward compatible and does **not** acquire an authoritative worker-limit field. Its legacy `cpu_cores` remains inventory-only so old files and APIs continue to deserialize, but scheduler/workload-planning constructors must receive the ephemeral installed `ExecutionBudget` explicitly. A profile captured on a 96-thread host and loaded inside a 23-thread cgroup therefore plans with 21 permits. Old and current profiles have the same exact behavior: they deserialize their inventory, and live planning ignores that inventory for worker capacity. A regression fixture asserts this result rather than relying on “conservative normalization.”

Logs and APIs use the term “logical threads” or “workers,” not physical “cores,” unless physical cores were actually measured. The headless keep-alive line must report the resolved worker limit and the effective logical-thread count instead of `num_cpus::get()`.

If `available_parallelism()` fails, execution falls back to one worker and emits WARN with the detection error and fallback. If a requested limit is reduced, startup emits INFO containing requested and effective values. A conflicting late installation is ERROR and aborts startup.

## Pool and scheduler integration

- Search removes `None => Rayon default` and enters parallel evaluation only through a lease-backed executor.
- The model parallel trainer does not own a second free-standing allowance. Its outer concurrency receives a parent lease and splits it; each per-model/native adapter receives only its child lease.
- XGBoost, LightGBM, CatBoost, statistical trainers, and other native libraries receive the per-workload lease width, never an independent `num_cpus::get()` result. Adapter-specific thread semantics are documented and tested rather than assumed uniform.
- CPU fallbacks and GPU feeders use the same broker. A GPU kernel launch itself does not hold CPU permits while the host thread is merely waiting, unless post-processing is active.
- The scheduler receives `&ExecutionBudget`/permit broker explicitly, never `HardwareProfile::cpu_cores`.

### Fixed child reservations

Dynamic recomputation of `parent_limit / active_children` is removed because already-running children cannot be resized. At scheduler construction, the planner chooses a maximum simultaneous child-slot count from RAM/GPU policy, clamps it to the parent permit count, and deterministically partitions the parent permits into fixed slot widths:

```text
slots     = min(configured_simultaneous_children, parent_limit)
base      = parent_limit / slots
remainder = parent_limit % slots
slot[i]   = base + (i < remainder ? 1 : 0)
sum(slot) = parent_limit
```

Before a child is spawned, the parent acquires the chosen slot's permits, passes that fixed width as `--cpu-threads`, and holds the lease in `RunningItem` until normal exit, error, cancellation, or crash cleanup. A running child is never resized. If no complete slot is available, admission waits. Unoccupied slots remain available to parent in-process work only through ordinary leases; child admission has priority so temporary local work cannot starve a queued slot. Tests cover every admission/completion order and prove the sum of live child reservations plus live parent leases never exceeds the parent limit.

Desktop-managed MCP and mesh sidecars use the same lifetime reservation path rather than raw `Command::spawn`. Each control-plane sidecar receives one fixed permit by default because its production duties are I/O/orchestration; any future CPU-heavy sidecar feature must request a larger declared slot. If a permit is unavailable, launch is deferred with structured INFO instead of creating an unbudgeted child. A child independently re-clamps its parent assignment against its own `available_parallelism() - 2`, protecting affinity- or cgroup-restricted descendants.

Independently launched NeoEthos binaries have no parent from which to obtain a cross-process lease and therefore install their own process-local ceiling. The startup record explicitly says `coordination_scope=process_local`; this design does not claim an OS-wide quota across unrelated launches. Every child that NeoEthos itself spawns is coordinated and uses `coordination_scope=managed_process_tree`.

### Mesh wire compatibility

Existing mesh JSON fields named `cpu_cores` remain accepted as a legacy alias for host inventory during one protocol window. New payloads add `host_logical_threads` and `effective_worker_limit` with distinct meanings; peers must never sum `cpu_cores` as schedulable capacity. Mixed-version fixtures prove old peers still deserialize while new scheduling uses only the explicit effective field. The legacy field is emitted until the protocol-version retirement date documented in the implementation plan.

A peer announcement without a valid positive `effective_worker_limit` has `capacity_state=unknown`, is observer-only/unschedulable, and emits one structured WARN with its protocol version and an upgrade requirement. There is no fallback to legacy `cpu_cores`, no invented one-worker assignment, and no remote work dispatch to that peer. Once an upgraded announcement supplies the field, normal admission may begin.

## GPU interaction

The CPU budget does not reduce GPU kernel occupancy, GPU batch width, CUDA streams, or the number of available cards. It limits CPU feeders, preprocessing, orchestration, and CPU fallbacks.

GPU-present paths must continue to execute real GPU kernels or fail loudly. No CPU budget mechanism may silently turn a requested CUDA/WGPU operation into a CPU implementation. When several cards or GPU workloads run concurrently, their CPU feeder shares come from the same parent worker limit.

CPU/GPU numerical parity and real-card performance remain separate validation gates. A successful CPU-budget build does not constitute GPU validation.

## Trading arithmetic and broker-truth invariance

This CPU scheduling change must not change any formula, unit, rounding rule, or input for spread, commission, swap, pip/tick value, lot sizing, risk, margin, conversion, or PnL. It must not add or preserve a constant, “typical” value, OHLC-derived proxy, hourly mean, or missing-field-to-zero fallback and then describe that result as broker-real.

The broker-truth remediation is a separate design/release gate. Its non-negotiable boundary for this work is:

- live truth comes from the exact cTrader account/symbol contract, broker Bid/Ask events, broker unrealized-PnL response, and `ProtoOAClosePositionDetail`/deal records;
- historical spread is the contemporaneous `ask - bid` reconstructed by replaying both broker quote streams in timestamp order, not candle `open - close`, not a fixed symbol field, and not a UTC-hour average. Bid and Ask updates need not share identical timestamps, so the broker-truth design must define and test last-known-quote/staleness semantics from captured broker behavior rather than perform a naive exact timestamp join;
- cTrader historical Bid and Ask ticks are requested separately in windows of at most seven days and paginated by the broker response; actual broker retention must be proven with a read-only account probe rather than inferred from the timestamp schema;
- if the selected historical range lacks synchronized broker Bid/Ask and conversion-leg coverage, that run fails the broker-real validation gate. Approximate sensitivity scenarios may exist only under an explicit `scenario=approximate` identity and can never be promoted or presented as historical broker PnL;
- missing cTrader fields remain unsupported/fail-closed until official protocol evidence, captured raw payloads, and a demo-account reconciliation establish the broker's behavior.

The CPU regression uses one immutable captured broker fixture containing exact metadata units, Bid/Ask events, conversion legs, commissions, swaps, and closed-deal truth. With deterministic seeds, workers `1` and `auto` must produce the same ordered trade ledger and the complete metric tuple bit-for-bit. If a mathematically justified algorithm requires a non-associative parallel reduction, the design must first define a deterministic reduction tree and prove the final ledger/metrics are identical; looser tolerances do not silently replace this acceptance criterion.

## Build parallelism

Repository-default compilation uses Cargo's portable negative job count:

```toml
[build]
jobs = -2
```

Cargo interprets this as the logical CPU count plus the negative value, with a minimum of one. This is a repository default for concurrent Cargo jobs, not a hard aggregate guarantee over rustc internals, build scripts, CMake, Ninja, linkers, or tools that ignore Cargo's jobserver. Native build scripts must inherit/use the Cargo jobserver where supported, and validation records any process that does not.

The fixed `-Z threads=8` rustc frontend setting is removed from both Windows and Linux target flags because multiplying eight frontend threads by several concurrent Cargo jobs defeats the intended reserve and hardcodes the build to neither the local machine nor the VPS. Stale explanatory references are removed from `.github/workflows/ci.yml` and `Cargo.toml` as part of the same change; repository/CI search must return no live `-Zthreads`, `threads=8`, or larger fixed Cargo-job override.

An explicit command-line `cargo -j` or external `CARGO_BUILD_JOBS` is outside the application's runtime configuration and can still override Cargo. Repository commands and CI must not set a larger fixed value. Validation records the effective host/process capacity alongside the command used.

## Tests and acceptance criteria

### Pure resolution tests

- Table-test 1, 2, 3, 4, 12, 23, and 96 effective logical threads.
- Verify the automatic result is always at least one and never above `available - reserve`.
- Verify requests of 1, below-auto, equal-auto, and above-auto only narrow or preserve the automatic result.
- Verify persistent, legacy, and parent caps compose by minimum and retain separate provenance.
- Verify `cpu_budget: 0` and legacy `rayon_threads: 0` fail sealed settings loading with the exact key in the diagnostic.
- Verify conflicting second installation fails and equal re-installation is idempotent in subprocess tests, or one serial process-global test binary, so `OnceLock` state cannot leak between cases.

### Integration tests

- Update `hardware_derived_not_settable` expectations from 11/12 to 10/12 and add the oversized-request regression (`9999 -> 10`).
- Test that HardwareProbe planning uses cgroup/effective capacity, not raw host inventory, through an injectable pure snapshot.
- Test import, search, model, native adapters, CPU fallbacks, GPU feeders, and scheduler consumers against one permit broker; concurrent probes assert the maximum admitted CPU-worker sum never exceeds the budget.
- Complete the Vortex-only migration first. Workspace guards prove there are no Polars declarations, lockfile packages, dependency-tree nodes, source imports/uses, compatibility wrappers, settings, non-Vortex runtime discovery/load/cache/model branches, non-Vortex canonical outputs, or `.fstore` artifacts/readers/writers. Non-Vortex parsing is allow-listed only in the shared import boundary.
- Import contract tests cover CSV, TSV, JSON, JSONL, Parquet, Arrow IPC/Feather, and Vortex input. Each fixed fixture must produce the same reopened canonical Vortex values, row order, timestamps, optional volume semantics, schema/version, and provenance; binary f64 sources compare price bits exactly, while decimal text sources compare their direct f64 parse with no intermediate f32 round trip. Include a valid high-precision price that the former f32 gate rejected. Deleting the source after import must not affect runtime loading. Directly passing each non-Vortex source to the runtime loader fails with the import-first diagnostic. Corrupt, malformed, null, non-finite, truncated, schema-ambiguous, missing-timestamp, and interrupted imports publish no dataset and drop no rows silently. A large-fixture probe proves bounded batch memory.
- Import admission tests prove the app receives a coordinator grant before `spawn_blocking`, the CLI holds a top-level lease, cancellation before and after admission is leak-free, and the lease outlives success publish or complete failure cleanup. Pipeline-stage tests prove parsing, hashing, reader/codec/decompression workers, validation, Vortex writing, and reopen verification share the admitted reservation without nested-acquisition deadlock.
- Model input/label parity tests compare the old fixed fixtures against the new `FeatureFrame`/typed-target path before the old path is deleted, including column names/order, timestamps/row identity, null/non-finite/error behavior, row count, row and column selection, label alignment, predictions, and full training metadata.
- Out-of-core parity tests write the same large-enough feature fixture through old `.fstore` and new Vortex paths during migration, then compare projected columns, row windows, feature values, ordering, and model/GA outputs. Crash tests prove incomplete Vortex scratch files are never opened and stale run manifests are cleaned without deleting active runs.
- Inventory dependency-owned pools from the locked Vortex-only production feature graph. Vortex, Arrow CSV/JSON/IPC, Parquet/codecs, hashing, ndarray, DuckDB, Burn, CubeCL CPU, BLAS/OpenMP, and tree adapters must each have a tested classification; requesting an unclassified threaded backend fails instead of running unbudgeted.
- Saturate a one-worker budget: hold its only lease, queue a second async admission and a child reservation, then exercise heartbeat, cancellation, stop, and child-exit cleanup. Tokio remains responsive, the cancelled request never starts, and lease return admits the next valid request without a core worker ever blocking on `Condvar`.
- Test fixed scheduler slot widths and every admission/completion/crash order; already-running children retain their original width and live reservations always sum to at most the parent budget.
- Test desktop MCP/mesh spawn arguments include their reserved `--cpu-threads` width and every exit path releases the lifetime lease.
- Test all settings-aware entry points install the hardware/process budget before search/model registries. Startup-capture integration tests assert every structured field and `coordination_scope` for app, CLI, desktop, root control plane, top-level MCP, and mesh—not only pure runtime-builder helpers.
- Test each Tokio/Tauri entry point's builder receives the resolved worker limit. The desktop test also proves preflight seeding/load and `async_runtime::set` precede `tauri::Builder`.
- Load an old serialized profile with `cpu_cores=96`, inject live `effective=23`, and assert every scheduler assignment derives from 21 rather than the persisted inventory.
- Mixed-version mesh fixtures accept legacy `cpu_cores` but never use it as the new schedulable field; a legacy peer without `effective_worker_limit` remains observer-only and receives no assignment until an upgraded announcement arrives.
- With the same fixed broker fixture and deterministic RNG seed, compare workers `1` versus `auto`: ordered fills/ledger, net and gross PnL, spread, commission, swap, conversion fees, drawdown, win rate, profit factor, expectancy, Sharpe/Sortino/Calmar, monthly series, trade count, and promotion verdict must be identical.

### Build and runtime validation

- Run formatting, focused unit tests, and `cargo check --workspace --all-targets`, preserving the complete output from the first INFO line through the final status.
- Run the separate `mcp` and `mesh` workspace tests and checks with the same full-log policy.
- Acceptance is not exit-code-only: there must be zero new warnings; every pre-existing INFO/WARN/ERROR line is classified by workspace and source, linked to an owner/follow-up or explicitly justified as expected, and unexpected dead-code/unused-path diagnostics fail the gate.
- Search root, `mcp`, and `mesh` for retired CPU controls, unmanaged pool construction, `num_cpus` capacity decisions, stale `-Zthreads`, and legacy mesh scheduling uses; every remaining occurrence is classified.
- On the local 6-core/12-thread host, verify the logged automatic worker limit is 10.
- On a quota-limited Linux/VPS process, verify the result follows `available_parallelism`, not the host CPU inventory.
- Record clean-build wall time and peak memory before/after Polars removal. Benchmark canonical Vortex OHLCV scan, projected out-of-core feature windows, repeated GA selected-column access, end-to-end training, and peak resident memory against the pre-removal fixed fixture. Results are reported rather than inferred from library reputation; any material regression is profiled and repaired within the Vortex path, not hidden by retaining Polars or `.fstore` in production.
- Run a concurrency probe on both hosts that overlaps a non-Vortex import through Vortex verification/publish, Vortex scan/decode, search, model training, Tokio blocking work, and managed children and records both the admitted sum and observed active dependency workers at or below the installed limit.
- Re-run CUDA tests on a real NVIDIA card before claiming GPU integration; inspect all build/runtime warnings and parity tuples, not only pass/fail summaries.

## Alternatives considered

### Local fixes in each crate

Changing every `-1` to `-2` would be quick but would retain multiple authorities, different fallbacks, startup-order races, and future drift. Rejected.

### Environment variables as the primary authority

Setting Rayon, Tokio, OpenMP, and model-specific variables at launch would remain invisible to typed configuration and would still allow contradictory values. Standard third-party environment variables may remain supported at their documented boundaries, but they are not NeoEthos' source of truth. Rejected.

### OS affinity or a cross-process global semaphore

Affinity can reserve particular CPUs and a shared semaphore can coordinate unrelated processes, but both add platform-specific lifecycle and crash-recovery complexity. The current requirement is satisfied for repository-controlled CPU work by hierarchical in-process leases plus lifetime reservations for every managed child. A hard quota shared by unrelated operator-launched processes is not claimed and remains deferred unless measurements show it is required.

## Completion boundary

This design is complete only when source, tests, logs, and all three workspaces agree on the same effective limit, aggregate lease accounting passes under overlap, and the fixed broker fixture is invariant between one and automatic workers. Local compilation proves neither real-card execution nor trading profitability. The CPU-budget work must be reported separately from CUDA parity, the broader broker-truth remediation, OOS validation, and final merge to `master`.
