# Unified CPU Execution Budget Design

**Date:** 2026-08-15

**Status:** Approved direction (Architecture A)

**Scope:** NeoEthos build parallelism and process-local CPU execution limits across the root, `mcp`, and `mesh` Cargo workspaces

## Objective

NeoEthos must use the machine it is actually allowed to use, while leaving capacity equivalent to one SMT core free for operating-system stability. In portable terms, the default CPU worker ceiling is:

```text
effective_available_logical_threads = std::thread::available_parallelism()
reserved_logical_threads            = min(2, effective_available_logical_threads - 1)
automatic_worker_limit              = max(1, effective_available_logical_threads - 2)
effective_worker_limit              = min(automatic_worker_limit, explicit_requested_limit)
```

The explicit requested limit is optional. It may narrow the automatic ceiling, but it must never restore the two reserved logical threads or exceed the capacity visible to the process.

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

These are not merely naming differences. They can make the app, search, model trainers, child processes, logs, and VPS scheduler disagree about the same run.

## Chosen architecture: one lightweight budget authority

Create a zero-dependency leaf crate, `neoethos-execution-budget`, containing only portable CPU-capacity detection, pure resolution logic, and process-budget installation. It must not depend on `neoethos-core`, Tokio, Rayon, tracing, GPU libraries, or ML libraries.

This boundary is necessary because the repository contains three deliberately separate Cargo workspaces. The root engine can reach the crate through `neoethos-core`; the isolated `mcp` and `mesh` workspaces can use the same small crate by path without importing the trading engine's GPU/ML dependency graph.

The crate exposes two layers:

1. Pure, deterministic calculation from `(available_logical_threads, requested_limit)` for exhaustive unit tests.
2. Process installation from `std::thread::available_parallelism()` before any global pool or async runtime is created.

The resolved value is immutable for the life of a process. A second installation with the same inputs is idempotent. A conflicting second installation is a startup error rather than a silent first-writer-wins condition.

The capacity record must retain all of the following so diagnostics do not conflate them:

- host logical threads, when a host inventory API can report them;
- effective logical threads available to this process;
- reserved logical threads;
- automatic worker limit;
- optional requested limit and its source;
- final effective worker limit.

Only the final effective worker limit may drive CPU-bound execution.

## Configuration and precedence

`system.hardware.cpu_budget` remains the operator's persistent narrowing control. `--cpu-threads` remains the scheduler's per-child ephemeral narrowing control.

Resolution order is:

```text
OS/cgroup available_parallelism
    -> subtract the fixed two-thread stability reserve
    -> narrow by system.hardware.cpu_budget, if present
    -> narrow by parent --cpu-threads assignment, if present
    -> install one immutable process budget
```

The parent assignment must not overwrite the persistent request with a larger number. Both requested values are caps; their minimum wins. Zero remains invalid and must fail at the parsing/config boundary.

Legacy `system.n_jobs` stays derived and non-settable. Its diagnostic text and computed value must use the unified resolver and say “effective logical threads minus two,” not “cores minus one.”

## Startup sequence

Every executable must install the budget before creating a CPU pool or a multi-thread Tokio runtime.

For settings-aware executables (`neoethos-app`, `neoethos-cli`, and any desktop entry point), startup becomes:

1. Parse only the arguments needed to locate configuration.
2. Load and validate settings.
3. Apply any parent `--cpu-threads` cap.
4. Resolve and install the process execution budget.
5. Install hardware overrides first, followed by search/model/data runtime settings.
6. Construct Tokio/Rayon/native-library execution facilities.
7. Emit one structured startup record with the complete capacity calculation.

The `#[tokio::main]` entry points must be replaced with small synchronous entry points that build a Tokio multi-thread runtime using the resolved worker limit and then call an async body. The settings-free `mcp` and `mesh` sidecars install the automatic budget before building their runtimes.

Tokio thread count is a ceiling on runtime workers, not a claim that all threads are continuously busy. CPU-heavy work must still be routed through the budgeted Rayon/native-library lanes; blocking training must not be allowed to multiply unchecked behind Tokio tasks.

## Hardware profile and truthful reporting

`HardwareProfile` must stop using a raw `sysinfo` CPU count as schedulable capacity. Host inventory and effective capacity are separate facts.

The existing serialized profile remains backward compatible, but current profiles add explicit effective-capacity and worker-limit fields. Scheduler and workload planning consume the worker limit, not the legacy `cpu_cores` inventory field. Old profiles are normalized conservatively and live execution re-probes current process capacity rather than trusting a profile written on another machine.

Logs and APIs use the term “logical threads” or “workers,” not physical “cores,” unless physical cores were actually measured. The headless keep-alive line must report the resolved worker limit and the effective logical-thread count instead of `num_cpus::get()`.

Required startup log fields:

```text
host_logical_threads
effective_available_logical_threads
reserved_logical_threads
automatic_worker_limit
requested_worker_limit
requested_limit_source
effective_worker_limit
```

If `available_parallelism()` fails, execution falls back to one worker and emits WARN with the detection error and fallback. If a requested limit is reduced, startup emits INFO containing requested and effective values. A conflicting late installation is ERROR and aborts startup.

## Pool and scheduler integration

- The search global Rayon pool is always created with the installed effective worker limit; `None => Rayon default` is removed from production startup.
- The model parallel trainer's outer pool uses the process limit. Existing per-model partitioning continues to divide that limit so outer concurrency multiplied by each native model's thread count cannot exceed the assigned share by design.
- XGBoost, LightGBM, CatBoost, statistical trainers, and other native libraries receive the per-workload share, never an independent `num_cpus::get()` result.
- The scheduler begins with the parent process worker limit and divides it among active children. The sum of `--cpu-threads` assignments for concurrently admitted children must be no greater than the parent limit.
- A child independently re-clamps its assignment against its own `available_parallelism() - 2`, protecting containerized or affinity-restricted children.
- Independently launched sidecars receive their own process-local ceiling. This design does not claim to impose a hard OS-wide quota across unrelated processes. Scheduler-spawned engine children are coordinated; an OS-level cross-process lease manager is outside this change.

## GPU interaction

The CPU budget does not reduce GPU kernel occupancy, GPU batch width, CUDA streams, or the number of available cards. It limits CPU feeders, preprocessing, orchestration, and CPU fallbacks.

GPU-present paths must continue to execute real GPU kernels or fail loudly. No CPU budget mechanism may silently turn a requested CUDA/WGPU operation into a CPU implementation. When several cards or GPU workloads run concurrently, their CPU feeder shares come from the same parent worker limit.

CPU/GPU numerical parity and real-card performance remain separate validation gates. A successful CPU-budget build does not constitute GPU validation.

## Build parallelism

Repository-default compilation uses Cargo's portable negative job count:

```toml
[build]
jobs = -2
```

Cargo interprets this as the logical CPU count plus the negative value, with a minimum of one. The fixed `-Z threads=8` rustc frontend setting is removed from both Windows and Linux target flags because multiplying eight frontend threads by several concurrent Cargo jobs defeats the aggregate budget and hardcodes the build to neither the local machine nor the VPS.

An explicit command-line `cargo -j` or external `CARGO_BUILD_JOBS` is outside the application's runtime configuration and can still override Cargo. Repository commands and CI must not set a larger fixed value. Validation records the effective host/process capacity alongside the command used.

## Tests and acceptance criteria

### Pure resolution tests

- Table-test 1, 2, 3, 4, 12, 23, and 96 effective logical threads.
- Verify the automatic result is always at least one and never above `available - reserve`.
- Verify requests of 1, below-auto, equal-auto, and above-auto only narrow or preserve the automatic result.
- Verify persistent and parent caps compose by minimum.
- Verify conflicting second installation fails and equal re-installation is idempotent.

### Integration tests

- Update `hardware_derived_not_settable` expectations from 11/12 to 10/12 and add the oversized-request regression (`9999 -> 10`).
- Test that HardwareProbe planning uses cgroup/effective capacity, not raw host inventory, through an injectable pure snapshot.
- Test search, model, and scheduler consumers against one installed test budget.
- Test scheduler assignments sum to at most the parent budget for every admission step.
- Test all settings-aware entry points install the hardware/process budget before the search/model registries.
- Test each Tokio entry point's builder receives the resolved worker limit through a pure builder/config helper.
- Validate old serialized hardware profiles and current profiles without silently treating raw host inventory as live capacity.

### Build and runtime validation

- Run formatting, focused unit tests, and `cargo check --workspace --all-targets` while capturing complete INFO/WARNING/ERROR output.
- Run the separate `mcp` and `mesh` workspace tests and checks.
- On the local 6-core/12-thread host, verify the logged automatic worker limit is 10.
- On a quota-limited Linux/VPS process, verify the result follows `available_parallelism`, not the host CPU inventory.
- Re-run CUDA tests on a real NVIDIA card before claiming GPU integration; inspect all build/runtime warnings and parity tuples, not only pass/fail summaries.

## Alternatives considered

### Local fixes in each crate

Changing every `-1` to `-2` would be quick but would retain multiple authorities, different fallbacks, startup-order races, and future drift. Rejected.

### Environment variables as the primary authority

Setting Rayon, Tokio, OpenMP, and model-specific variables at launch would remain invisible to typed configuration and would still allow contradictory values. Standard third-party environment variables may remain supported at their documented boundaries, but they are not NeoEthos' source of truth. Rejected.

### OS affinity or a cross-process global semaphore

Affinity can reserve particular CPUs and a shared semaphore can coordinate unrelated processes, but both add platform-specific lifecycle and crash-recovery complexity. The current requirement is satisfied portably by one process budget plus scheduler partitioning of child processes. Deferred unless measurements prove independent sidecars materially oversubscribe the host.

## Completion boundary

This design is complete only when source, tests, logs, and all three workspaces agree on the same effective limit. Local compilation proves neither real-card execution nor trading profitability. The CPU-budget work must be reported separately from CUDA parity, mathematical backtest correctness, OOS validation, and final merge to `master`.
