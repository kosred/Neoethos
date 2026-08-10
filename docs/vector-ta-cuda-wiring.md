# vector-ta CUDA indicator lane — wiring, the arch trap, and the box recipe

Landed 2026-08-09 (task #22). This document covers three things:

1. **The one edit this change still needs**, in a file another workflow owns —
   apply it verbatim once that workflow lands.
2. **The build recipe for the rented card**, including the two environment
   variables without which the lane is either unloadable or unmeasurable.
3. **The decisions and their evidence** — why 0.2.9 and not 0.3.1, why
   `cuda-build-ptx` and not `cuda`, why ten indicators and not eighteen.

---

## 1. Pending edit — `crates/neoethos-search/src/discovery.rs`

`discovery.rs` is owned by the concurrent prototype-B workflow, so this was not
applied. It is a one-line addition next to the existing GA device summary, so
the run log reports BOTH device lanes instead of only one.

At `crates/neoethos-search/src/discovery.rs:5928` the current line is:

```rust
    crate::eval_telemetry::device_summary();
```

Replace it with:

```rust
    crate::eval_telemetry::device_summary();
    // Second device lane: the vector-ta indicator sweep in the feature build.
    // Printed even when empty — a run with no indicator frames still states
    // whether this binary has a CUDA indicator lane compiled in, so a 0%
    // reading is attributable rather than inferred (same contract as
    // `device_summary` above).
    neoethos_data::core::indicator_telemetry::indicator_device_summary();
```

`neoethos-search` already depends on `neoethos-data`, and
`indicator_telemetry` compiles in EVERY build (card-less included) precisely so
this call site needs no `cfg`.

Nothing else in the concurrently-owned set
(`prototype_b_population.cu`, `neoethos_gpu_cuda.h`, `layout_asserts.cpp`,
`neoethos-gpu-cuda/build.rs`, `neoethos-gpu-contracts/src/lib.rs`,
`prototype_b_population_eval.rs`, `eval_telemetry.rs`, `search_engine.rs`)
required a change.

---

## 2. Building and running on the card

### Set nothing. That is the point.

```sh
cargo build -p neoethos-cli --features gpu-nvidia --release
```

One build, one artifact, four cards.
`vendor/vector-ta-0.2.9-patched/build.rs` compiles every kernel to a single
`-fatbin` carrying `-gencode arch=compute_X,code=sm_X` for **80, 86, 89, 90**
plus `-gencode arch=compute_90,code=compute_90` for forward JIT:

| Card | Arch | How it is served |
|---|---|---|
| A100 | sm_80 | real SASS from its own gencode |
| RTX 3090 / A10 | sm_86 | real SASS from its own gencode |
| RTX 4090 / L40S | sm_89 | real SASS from its own gencode |
| H100 | sm_90 | real SASS from its own gencode |
| anything newer | sm_100+ | driver JITs the embedded `compute_90` PTX |

The set is intersected with `nvcc --list-gpu-arch`, so CUDA 11 (no sm_90) and
CUDA 13 (no sm_80) narrow gracefully with a `cargo:warning` naming exactly what
was dropped, instead of failing the build.

**NEVER set `CUDA_ARCH=`.** The singular form builds a
**SINGLE-architecture fatbin**: `CUDA_ARCH=sm_89` produces an artifact that
will not load on a 3090 or an A100. That is the defect class that cost this
project eight months. It is still accepted, because narrowing is sometimes
wanted deliberately, and vector-ta prints a `cargo:warning` when it happens. To
narrow for a faster iteration loop, use the **list** form:

```sh
CUDA_ARCHS=86 cargo build -p neoethos-cli --features gpu-nvidia --release
```

`crates/neoethos-data/build.rs` no longer resolves an arch at all. It used to,
falling through to `nvidia-smi` and panicking when there was no card — which
broke CUDA builds in containers and CI, and printed the BUILD host's arch in
the diagnostic meant to explain a mismatch on the RUN host. The arch strings
the runtime quotes now come from vector-ta's own
`module_loader::{COMPILED_ARCHS, COMPILED_PTX_ARCH}`: one source, describing
the kernels rather than the machine that compiled them.

### `CUDA_FAST_MATH` — leave it unset

vector-ta's `build.rs` now defaults fast math **OFF**, and
`fast_math_requested` returns false for the f64 lane *before* it reads the
variable at all. So `kernels/cuda/neoethos_f64_kernels.cu` is always compiled
with `-prec-div=true -prec-sqrt=true -fmad=false -ftz=false` and never with
`--use_fast_math`, whatever the environment says.

`crates/neoethos-data/build.rs` additionally **panics** on any
`CUDA_FAST_MATH` other than `0` for a `gpu-cuda` build, because the crate's
other 329 kernels would still be affected and a binary built that way cannot
make a parity claim while looking identical to one that can.

Consequence for the checklist: you **cannot** prove the f64 opt-out with
`CUDA_FAST_MATH=1 cargo build -p neoethos-data --features gpu-cuda` — that
aborts before nvcc runs. Prove it against vector-ta directly:

```sh
CUDA_FAST_MATH=1 cargo build -p vector-ta --features cuda-build-ptx
grep -c approx "$OUT_DIR/neoethos_f64_kernels.ptx"   # must be 0
```

### Running the parity test

```sh
NEOETHOS_REQUIRE_GPU=1 \
cargo test -p neoethos-data --features gpu-cuda -- --nocapture gpu_cpu_indicator_sweep_parity
```

`NEOETHOS_REQUIRE_GPU=1` turns "no usable CUDA lane" from a skip into a
failure. It now ALSO selects `IndicatorComputePolicy::RequireGpu` for any
production caller that does not name a policy
(`hpc_ta::resolved_indicator_compute_policy`), so a lane failure stops the run
instead of logging a warning and finishing with CPU numbers.

`--nocapture` is what prints the measured per-indicator deviation — that output
is the point of the run. The CPU side of the comparison is pinned to
`Kernel::Scalar`, the reference every kernel was written against, so **0.0 of
budget is a legitimate target**: if every indicator reports 0.0, tighten
`hpc_ta::tests::parity_tolerance` to exact equality. Anything above 0 names the
single cell where a rounding differs and must be explained before acceptance —
never widened away.

### Useful diagnostics

| Variable | Effect |
|---|---|
| `CUDA_PROBE_DEBUG=1` | vector-ta prints which stage of `cuda_available()` failed |
| `CUDA_MODULE_LOAD_DEBUG=1` | prints `using PTX path for <stem>` / cubin load failures per kernel |
| `NEOETHOS_REQUIRE_GPU=1` | makes an unavailable indicator lane fatal in the parity test |

Do **not** set `CUDA_MEM_CHECK=0`: it disables vector-ta's `mem_get_info`
pre-flight, which is what turns an oversized allocation into a loud
`OutOfMemory { required, free, headroom }` instead of a crash.

---

## 3. Decisions and evidence

### Stay on vector-ta 0.2.9; do not upgrade to 0.3.1

A mechanical diff of every `pub fn|struct|enum|trait` across `src/indicators`,
`src/cuda` and `src/utilities` between the two published crates yields an
**empty removed-set and an empty added-set**. The only source changes are
pyo3 0.25 → 0.29 renames (`allow_threads` → `detach`, `PyObject` →
`Py<PyAny>`) behind the optional `python` feature, which this workspace does
not enable.

Nothing changes GPU-side either: 0.3.1 ships the same 329 `.cu` kernels, the
same 329 prebuilt PTX files (**all still `.target sm_89`**), the same 258
wrappers, the same `cust = "0.3.2"`, and an identical `[features]` table. It
does **not** fix the compute_89 trap.

And it does not carry the fix we vendored 0.2.9 for: the
`assert!(warm <= cols, "warm prefix exceeds row width")` is still at
`src/utilities/helpers.rs:159` in the extracted 0.3.1 source. Upgrading means
re-vendoring and re-applying our clamp patch for zero benefit, while adding a
variable to the parity experiment.

Revisit when upstream ships the `helpers.rs` fix, or when a release adds an
**f64 device lane** — that, not a version number, is what would make this path
parity-capable.

### `cuda-build-ptx`, not `cuda`

The plain `cuda` feature stages vector-ta's prebuilt PTX. All 329 shipped files
carry a literal `.target sm_89`; not one targets a virtual `compute_*` arch, so
PTX forward-compatibility cannot save them. On an sm_86 device every module
load fails, per indicator, at first use.

`cuda-build-ptx` compiles the kernels with nvcc instead. That is the documented
escape — but as shipped it defaulted the nvcc `-arch` to `compute_89`, i.e. it
reproduced the exact trap it exists to avoid. See the vendor patch below.

### Ten indicators on the device, eight on the CPU — and why two were nearly wrong

`hpc_ta::MULTI_PERIOD_IDS` has eighteen entries.
`gpu_indicators::GPU_SWEEP_SPECS` has ten: `sma`, `ema`, `rsi`, `roc`, `mom`,
`atr`, `adx`, `willr`, `cci`, `mfi`. The other eight (`stoch`, `macd`,
`bollinger_bands`, `keltner`, `supertrend`, `tsi`, `obv`, `vwap`) have a
multi-output or non-period device contract and stay on the CPU — enumerated up
front and reported as `CpuIndicatorNotPortable`, never discovered by a failed
launch mid-run.

The ten share one parameter contract. All three of vector-ta's range resolvers
— `resolve_period_range` (cuda.rs:14478), `resolve_named_range` (cuda.rs:14586)
and `resolve_usize_range_param_device` (cuda.rs:14511) — read
`period_start` / `period_end` / `period_step`, which is why a single uniform
parameter set serves the whole table, `atr` included (its primary key is
`length_*` but it falls back to `period_*` at cuda.rs:14599-14609).

**Two of them would have computed a different indicator, not a less precise
one.** `cci` and `mfi` take `hlc3` (typical price) on the CPU:

* `cpu_batch.rs:3401` — `extract_slice_input("cci", req.data, "hlc3")`
* `cpu_batch.rs:2867` — `source.unwrap_or("hlc3")` for the MFI typical price

but the device accessors hand the kernel `close`:

* `cuda_device_prices_from_req` (cuda.rs:14869) → `ohlc.source().unwrap_or(ohlc.close())`
* `cuda_device_close_volume_from_req` (cuda.rs:14994) → `ohlcv.close()`

So the engine uploads an explicit `hlc3` series and serves `cci` through a
`Slice` ref and `mfi` through a `CloseVolume(hlc3, volume)` ref.
`gpu_indicators::tests::hlc3_sourced_indicators_do_not_use_the_close_ref` pins
this — it is a correctness property, not something the parity tolerance should
be asked to notice.

### One launch per period, not one batched contiguous sweep

vector-ta expands a sweep arithmetically (`expand_periods_i32`, cuda.rs:5435).
The periods this codebase wants — `[7, 21, 50, 100, 200]` — are not an
arithmetic progression, so the only batched form is contiguous `7..=200 step 1`:
194 rows of which 189 are discarded. At 843k bars that is a 654 MB device
allocation to keep 17 MB, and its size is a function of the requested period
RANGE rather than of the hardware — which is the wrong side of the NEVER-OOM
invariant. One launch per period keeps the output at exactly `n × 4` bytes
(3.4 MB at 843k bars) and computes nothing that is thrown away.

Known cost, to be measured on the card: `compute_cuda_device` constructs a
fresh wrapper per call and each `new()` does a full `Module::from_ptx` JIT
(there is no module cache in the crate) — roughly 50 JITs per frame. Acceptable
because the feature build runs once per frame per timeframe, not inside the GA
hot loop. If it ever moves into a hot loop, hold the concrete
`CudaSma`/`CudaRsi`/… handles instead of going through the dispatcher.

### This lane cannot be bit-parity, and says so

vector-ta's device layer is **f32-only** — `src/cuda/device_types.rs` contains
no `f64` at all — while every CPU indicator returns f64. Adoption is therefore
a measured-divergence decision. Three things make it loud rather than silent:

* the first GPU-routed frame logs at **WARN** naming the device, both arches,
  and the precision class;
* `IndicatorRunSummary::precision` carries `"f32 device vs f64 cpu reference"`
  into the run summary;
* `gpu_cpu_indicator_sweep_parity` measures it per indicator against real
  EURUSD M1 fixture data and prints the numbers.

---

## 4. Verbatim record of the edits outside `crates/neoethos-data/`

Re-apply these if a merge drops them.

### `vendor/vector-ta-0.2.9-patched/build.rs` — multi-arch fatbin

> **SUPERSEDED.** An earlier revision of this document recorded a
> single-arch-plus-placeholder-cubin design here and said "re-apply if a merge
> drops it". Do not. Re-applying it would put the arch pin back. What follows
> is the shape that is actually in the tree.

`fn target_archs(nvcc)` resolves the architecture SET once, in this order:

1. `CUDA_ARCHS` — the list form, e.g. `80,86,89,90` or a narrowing `86`.
2. `CUDA_ARCH` — the SINGULAR form, which yields `vec![a]`, i.e. a
   single-architecture fatbin. It emits a `cargo:warning` saying so. Do not use
   it (see §2).
3. `DEFAULT_TARGET_ARCHS` = `[80, 86, 89, 90]` — no card named, no probing of
   the build host.

That set is intersected with `nvcc --list-gpu-arch`, and the surviving set is
recorded as `VECTOR_TA_CUDA_ARCHS` / `VECTOR_TA_CUDA_PTX_ARCH`, which
`src/cuda/module_loader.rs` re-exports as `COMPILED_ARCHS` /
`COMPILED_PTX_ARCH` and quotes verbatim in its load-failure diagnostic.

`fn compile_kernel` then emits, per source:

* a standalone `<stem>.ptx` at the **lowest** target arch, used as the loader's
  fallback (PTX runs forward, so the lowest is the only one that loads
  everywhere in the set);
* one `<stem>.fatbin` with `-gencode=arch=compute_X,code=sm_X` for every arch
  in the set, plus `-gencode=arch=compute_MAX,code=compute_MAX` so a card
  newer than anything we compiled for JITs instead of failing.

Gone entirely: `TARGET_CUBIN_MAJOR/MINOR = 8/9`, `current_context_is_sm89()`,
the `kernels/ptx/compute_89` prebuilt default, the `_sm89.cubin` filename and
the `-arch sm_89` cubin step.

### `crates/neoethos-app/Cargo.toml`

```toml
gpu-nvidia = [
    "neoethos-data/gpu-cuda",
    "neoethos-models/gpu-cuda",
    "neoethos-search/gpu-cuda",
]
```

### `crates/neoethos-cli/Cargo.toml`

```toml
gpu-nvidia = [
    "neoethos-data/gpu-cuda",
    "neoethos-models/gpu-cuda",
    "neoethos-search/gpu-cuda",
]
```

`neoethos-data/gpu-cuda` had been **declared and named by nothing** —
`grep -rn "neoethos-data/gpu-cuda"` over the repo returned zero matches — so
vector-ta's 329 CUDA indicator kernels had never been in a shipped binary.
These two lines are what actually put the lane in the build.
