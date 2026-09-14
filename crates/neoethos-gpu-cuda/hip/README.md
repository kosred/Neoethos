# ADR: owned AMD HIP runtime before Search admission

Status: accepted for the additive ownership layer; complete HIP Search remains pending.
Date: 2026-09-13.

## Context

CPU, CUDA and HIP must share mathematical semantics and produce independently
verified results. HIPIFY supplies the kernel translation, not Rust ownership or
backend identity. Current Search contracts explicitly identify CUDA contexts,
NVCC and SASS; replacing those values with HIP pointers would make them untrue.

The user requires local code work before paid GPU acceptance. No rental is part
of this change.

## Decision

Use `neoethos-gpu-cuda`'s additive `hip-runtime` feature for a distinct real
Rust-to-AMD-HIP ownership API. It does not enable `cuda`, `cust` or `vector-ta`,
does not replace CUDA ABI V2, and does not make current Discovery admit HIP.

The native owner creates its own nondefault stream and records actual device,
UUID and runtime stream identity. A nonreused process-local lease identifier is
an application ownership token, **not a CUDA context ID**. Raw handles cannot be
registered by callers. Native error information crosses the Rust boundary.

The safe Rust owner is initially thread-confined. No device reset, external
stream destruction, raw ownership adoption or cross-thread handle sharing is
exposed. The modern HIP runtime owns device contexts; this adapter does not call
the deprecated primary-context Retain/Release or current-context APIs. Live
stream checks and the private ownership boundary are required; an ambiguous
failure quarantines resources instead of freeing them. External resets remain
forbidden while leases are live; stream ownership cannot prevent other unsafe
code from resetting the device.

## Options considered

- Alias HIP to existing CUDA identities: rejected; CUDA-specific build and
  context assertions would become false.
- Copy the entire CUDA crate: rejected; duplicates mathematical and state-machine
  implementations and makes parity harder to maintain.
- Add the small HIP ownership layer in the existing crate: selected; reuses the
  repository boundary without changing CUDA behavior or admitting an incomplete
  HIP Search route.

## Build and use

Native Linux x86-64 GNU, official ROCm 7.2.3 installed:

```sh
ROCM_PATH=/opt/rocm-7.2.3 cargo build -p neoethos-gpu-cuda --features hip-runtime
```

This command builds **host runtime API calls**, not device kernels. The builder
uses ROCm's Clang in explicit C++ mode, disables implicit Clang configuration,
preserves complete compiler logs, and links the real `amdhip64` library. No GPU
autodetection is needed for compilation. `DOCS_RS` remains typechecking only.
Deployment must make the matching ROCm shared libraries available to the loader
(for example through the ROCm installer or its documented library configuration).

Use `hip_runtime_v1::HipRunLeaseV1` from Rust. Construction can fail when no
ROCm device exists; there is no CPU substitute. Query/synchronize/close operations
are ownership controls, not numerical parity or kernel-execution evidence.
Prefer explicit close so the caller receives cleanup failure information.

To build the real API probe once and then execute that binary without rebuilding:

```sh
ROCM_PATH=/opt/rocm-7.2.3 cargo build -p neoethos-gpu-cuda --features hip-runtime --example hip_runtime_probe
./target/debug/examples/hip_runtime_probe --ordinal 0
```

If a manually installed ROCm is not registered with the Linux dynamic loader,
the binary exits before calling HIP (`libamdhip64.so.7` not found). For the
explicit installation above, set the matching path for that invocation only:

```sh
env LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib ./target/debug/examples/hip_runtime_probe --ordinal 0
```

This selects installed runtime libraries; it neither supplies a GPU nor changes
the system loader configuration.

No compatible device is an error exit, not a skipped-success result. Even a
successful lease probe explicitly reports that it executed no device kernels.

## Device producers and shared native kernels

The optional features below extend the same owner; enabling them is not a
claim that CPU/CUDA/HIP numerical parity has passed on hardware.

| Feature | Connected code | Does not establish |
| --- | --- | --- |
| `hip-runtime` | Real HIP lease, stream-ordered buffers, retained pinned uploads/readback and native consumer borrows | Device kernel execution |
| `hip-session-kernels` | The existing Session-v2 CUDA source translated and compiled for one exact AMD architecture; typed leased launch | Full Data feature coverage |
| `hip-native-kernels` | The original 16 native translation units, backend-specific Search ownership and typed SMC-v3 producer | Production HIP Discovery admission |
| `hip-device-fixtures` | Explicit native fixture code and device-only regression targets | A production Search input carrier or a skipped-success GPU result |

For example, prepare the existing Data Session diagnostic once, then run its
binary without invoking Cargo again:

```sh
ROCM_PATH=/opt/rocm-7.2.3 NEOETHOS_HIP_ARCH=gfx942 cargo build -p neoethos-data --features gpu-hip-session --example hip_session_smoke
env LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib ./target/debug/examples/hip_session_smoke --device 0
```

`gfx942` is an explicit example target, not a substitute for detecting the
acceptance machine's architecture. Compilation needs the installed ROCm SDK,
not a card. Runtime execution requires the matching real device. Native CUDA
and HIP kernel features cannot coexist in one binary: internal algorithm symbol
names are shared, but CUDA capability queries remain unavailable in a HIP build.

The production builder invokes AMD's official `hipify-perl` on the original
source/include closure. It preserves complete translation/compiler logs,
checks the emitted AMD device images and exact archive members, and binds the
artifact/source hashes in manifests. Failed or timed-out translation cannot
publish success. Translation has a separate finite deadline from small host
tools because the full population source exceeds 120 seconds locally.
Completed translations are reused only when the exact source, official Perl
translator, Perl interpreter, arguments, translated output and complete logs
still match their recorded hashes. A failed or incomplete attempt is not cached.
Linux compiler/translator children run in owned process groups with bounded
cleanup; a surviving descendant prevents artifact acceptance. This cache avoids
repeating translation, not the compiler or genuine device acceptance.
The installed ROCm 7.2.3 `hipify-clang` emits initialization errors despite
exit zero; it is not accepted as a successful alternative. This is the upstream
[HIPIFY initialization-driver issue](https://github.com/ROCm/HIPIFY/pull/2322).

Data Session and SMC share one immutable OHLCV owner rather than uploading the
six input lanes twice. SMC adds exactly `441 * rows + 100` logical output bytes;
both producers plus their shared inputs use `696 * rows + 100` logical bytes.
These are allocation extents, not measured allocator usage or available VRAM.
The original SMC birth-row representation requires `rows - 1 <= INT32_MAX`.

SMC's successful enqueue does not publish initialized outputs. The owner first
reads and synchronizes its four-byte device error, requires zero, then reads
and synchronizes its 96-byte calendar/SMC-slot hashes. A semantic refusal
invalidates previous outputs without poisoning a healthy runtime; ambiguous
runtime failures retain/quarantine resources. Those hashes do **not** seal the
feature columns. Data's `PreparedHipCanonicalFeatureStoreV1` now assembles an
exact, ordered selection from the real Session/SMC owners into bar-major f64,
packed-u4 validity and the existing canonical Merkle format. It retains the
verified canonical source segment and producer recipe. It is not a fitted
feature-screen selection or full-family Discovery admission.

Enabled normalization consumes the same original Data training split and
policy-v3 name classifier as CUDA/CPU. The common native f64 normalizer returns
only bounded control data and six fit words per selected column. Data validates
those actual words into the portable fitted state; raw producer outputs retain
distinct `pre-normalize:` names, and the wrapping node records the portable fit
hash. Disabled mode creates no fitted node. A native transport digest is not a
substitute for the name-aware portable fitted-state identity.

`ResidentHipCanonicalFeatureStoreV1::bind_population_v1` connects that physical
store to the existing strict `PopulationSession` evaluator. Data privately
supplies the exact close/high/low, generated calendar and SMC11 buffers; packed
features, validity and timestamps remain in their original GPU allocations.
The returned borrowed guard retains canonical source/recipe/fit metadata and
all nine native buffer pins through checked cleanup. Metric output carries
explicit HIP device identity, not CUDA context or financial/OOS authority.
Real lease revalidation brackets ordinary cohort execution and result publication.
Ambiguous native failures quarantine the owner; they do not permit CPU fallback.

`bind_population_for_search_v3` additionally connects that same parent to the
existing adaptive generation/rank/archive state machine. It preserves the Data
borrow until terminal completion restores the exact original population owner.
Rejected inputs are checked before detachment; unfinished or ambiguous native
work quarantines its lease. No bare native owner escapes the HIP wrapper.
The Search bind separately budgets concurrent evaluator capacity `C` and actual
month capacity `M`, using the existing `PopulationMetricsOnlyPlanV1`, in addition
to allocator headroom. Logical population `P` is unchanged: `0 < C <= P`, with
every candidate evaluated in chunks, including a partial final chunk. This
physical execution bridge is not financial/selection/holdout authorization.

The `hip_shared_producers_smoke` example exercises this whole bounded connection
on a temporary synthetic canonical generation, including raw/enabled fits,
three adaptive generations (`P=12`, `C=5`), archive plus full last-population
export, then ordinary cohort evaluation on the same restored parent. These are
small diagnostic values, never production search limits. It also checks known
flat-price commission arithmetic, supplied-cohort identity, and zero parent
re-upload. It is a real-device executable, not a strategy-quality test:

```sh
ROCM_PATH=/opt/rocm-7.2.3 NEOETHOS_HIP_ARCH=gfx942 cargo build -p neoethos-data --no-default-features --features gpu-hip-session,gpu-hip-smc --example hip_shared_producers_smoke
env LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib ./target/debug/examples/hip_shared_producers_smoke --device 0 --normalization enabled
```

No compatible device means failure before any fixture can report success.
The example's test binary exposes the same integrated run as the explicitly
ignored `tests::actual_hip_shared_producers_pack_ga_and_population` test, requiring
`NEOETHOS_REQUIRE_GPU=1` and enabled normalization. Invoke it with `--ignored
--exact` and that test name; optional `NEOETHOS_HIP_DEVICE` selects the ordinal.
The other three example tests are host controls, not evidence of GPU execution.

Native consumers pin initialized input buffers until checked cleanup. A producer
cannot overwrite a borrowed output; logical release defers physical free while
pins remain. Same-stream ordering is required throughout. Explicit close
reports cleanup failures rather than treating destructor execution as proof.

## Consequences and remaining connection work

The ownership API can now be compiled and exercised independently without
rebuilding the whole CUDA kernel set. Host injected-operation tests verify the
actual lifecycle policy, not HIP hardware correctness.

Before enabling HIP Discovery, connect the source-bound selected Data store and
population evaluator to the genuine Search configuration/selection/holdout and
application orchestration. The physical generation lifecycle is connected to
the common engine, but does not confer that higher-level authorization. Native
HIP Search identity and archive lifetimes use the real HIP owner, never substituted CUDA context
IDs. Five HIPIFY warnings refer to CUDA identity calls in CUDA-only branches;
the sixth is CUDA `__trap`, whose AMD branch uses `__builtin_trap`.
The HIP population fixture is deliberately not a production input authority.
Full indicator coverage, Search-to-model integration and per-trade CPU/CUDA/HIP
parity still require acceptance evidence. Session and SMC preserve the original
causal kernel algorithms; their port alone is not a parallel-speedup result.
Real-device execution follows code readiness.

## ROCm model build environment

The Burn/CubeCL model route is separate from the native Search kernel archive.
Its `cubecl-hip-sys` build script executes `hipconfig` through `PATH`, not through
the `ROCM_PATH` variable. Set both the installation paths and `PATH` explicitly
for a manually installed SDK:

```sh
env PATH=/opt/rocm-7.2.3/bin:$PATH \
    ROCM_PATH=/opt/rocm-7.2.3 HIP_PATH=/opt/rocm-7.2.3 \
    LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib CARGO_INCREMENTAL=0 \
    cargo test -p neoethos-models --no-default-features --features gpu-rocm --all-targets --no-run
```

Without `hipconfig`, the dependency can typecheck with its latest bundled
bindings but omit `amdhip64` and `hiprtc` linker instructions. That is **not** a
usable ROCm model build. Review the dependency's full build output and perform
the final link; never treat that fallback warning or `cargo check` as readiness.
Changing `HIP_PATH` also invalidates that dependency's cached configuration;
changing only `PATH` does not trigger its declared Cargo rerun checks.
The pinned April 7 nightly can also hit the upstream
[`AttrId` incremental-cache ICE](https://github.com/rust-lang/rust/issues/154878).
`CARGO_INCREMENTAL=0` in the command avoids that compiler cache path without
changing the toolchain or source semantics. Cargo still reuses compatible
dependency artifacts; workspace and patched crates may need recompilation.
Run the resulting test binary directly for subsequent host checks. Actual
device model training/reload tests still require a compatible GPU.

## Official sources

- [ROCm 7.2.3 stream API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.3/doxygen/html/group___stream_o.html): creation, stream device/ID queries and synchronization/destruction semantics.
- [ROCm 7.2.3 deprecated APIs](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.3/reference/deprecated_api_list.html): replace deprecated HIP context calls with device/stream APIs, not warning suppression.
- [ROCm 7.2.3 context implementation](https://raw.githubusercontent.com/ROCm/rocm-systems/rocm-7.2.3/projects/clr/hipamd/src/hip_context.cpp): HIP contexts are not CUDA reset-generation IDs.

The architecture skill guided the additive boundary and rejection of duplicate
engines; documentation guidance separates callable ownership from Search and
hardware readiness.
