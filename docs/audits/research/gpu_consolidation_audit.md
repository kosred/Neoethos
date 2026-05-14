## GPU code-duplication audit for `crates/forex-models/`

Date: 2026-05-14
Scope: All `*.rs` files under `crates/forex-models/src/`.
Operator concern: "Functions για GPU που θα μπορούσε να είναι σε κοινό
Module/αρχείο και να μειωθεί ο κώδικας." (Translation: GPU functions
duplicated across model files; move into a shared module to cut LOC.)

The audit catalogues every place GPU-related logic lives in the
`forex-models` crate. The matrix below combines a `grep`-based count
with manual classification of each hit (a "device-string lookup" is
not the same as "device probe", even though both mention `"cuda"`).

### TASK 1 — Count matrix (pattern X file)

Columns:

* P1 — Device probe / availability (`tch::Cuda::is_available`,
  `gpu_count`, `CUDA_VISIBLE_DEVICES` parsing).
* P2 — Buffer marshalling (feature flattening, `to_device`,
  `copy_from_slice` for kernel uploads).
* P3 — Kernel launch wrappers (`cust`/`cubecl`/`tch` launch boilerplate,
  per-runtime client setup).
* P4 — Error mapping (`map_ort_error`, `map_cuda_error`,
  `map_cublas_error`).
* P5 — CPU fallback decision branches
  (`if cuda_available() { gpu_path() } else { cpu_path() }`).
* P6 — Backend label strings used as control-flow keys
  (`"cuda"`, `"rocm"`, `"wgpu"`, `"gpu:N"` substring matching).
* P7 — Device-policy normalization (`normalize_*_device_policy`).
* P8 — GPU synchronization (`Cuda::synchronize`,
  `cudaDeviceSynchronize` equivalents).

| File | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|------|----|----|----|----|----|----|----|----|
| `hardware.rs` | 9 | 1 | 0 | 0 | 1 | 14 | 0 | 4 |
| `common.rs` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `tree_models/config.rs` | 13 | 0 | 0 | 0 | 1 | 7 | 0 | 0 |
| `tree_models/xgboost.rs` | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `tree_models/lightgbm.rs` | 5 | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| `tree_models/catboost.rs` | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| `evolution/crfmnes_gpu.rs` | 0 | 3 | 2 | 0 | 0 | 1 | 0 | 0 |
| `evolution/crfmnes_impl.rs` | 0 | 0 | 0 | 0 | 6 | 3 | 0 | 0 |
| `evolution/neat_gpu.rs` | 0 | 3 | 2 | 0 | 0 | 1 | 0 | 0 |
| `evolution/neat_impl.rs` | 0 | 0 | 0 | 0 | 6 | 2 | 0 | 0 |
| `statistical/linear_gpu.rs` | 0 | 5 | 6 | 0 | 0 | 1 | 1 | 0 |
| `statistical/linear_impl.rs` | 0 | 0 | 0 | 0 | 5 | 2 | 0 | 0 |
| `statistical/common.rs` | 0 | 0 | 0 | 0 | 4 | 6 | 1 | 0 |
| `statistical/bayesian_impl.rs` | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| `rl/dqn_impl.rs` | 0 | 0 | 0 | 0 | 3 | 33 | 1 | 0 |
| `rl/dqn_impl_tests.rs` | 0 | 0 | 0 | 0 | 1 | 88 | 0 | 0 |
| `anomaly/forest_impl.rs` | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 |
| `streaming/adaptive_impl.rs` | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| `forecasting/swarm_impl.rs` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `runtime/onnx.rs` | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| `runtime/capabilities.rs` | 0 | 0 | 0 | 0 | 4 | 12 | 1 | 0 |
| `burn_models.rs` | 0 | 0 | 0 | 0 | 0 | 47 | 2 | 0 |
| `deep_models.rs` | 0 | 0 | 0 | 0 | 0 | 35 | 0 | 0 |
| `genetic.rs` | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| `registry.rs` | 1 | 0 | 0 | 0 | 0 | 29 | 0 | 0 |

### Hot duplication zones (the actual problem)

Three GPU kernel-driver files emit nearly identical helper code:

* `evolution/crfmnes_gpu.rs` (lines 102-158)
* `evolution/neat_gpu.rs` (lines 223-277)
* `statistical/linear_gpu.rs` (lines 227-292)

Each one re-implements:

1. `*_cuda_kernel_enabled(policy: &str) -> bool` — checks if the policy
   asks for GPU AND the per-model env var
   (`FOREX_BOT_<NAME>_CUDA_KERNEL`) is not disabled.
2. The disabled-env predicate (`matches!(value, "0" | "false" | "off" |
   "disable" | "disabled")`) — three exact-copy occurrences.
3. `cuda_device_id(policy)` — reads `FOREX_BOT_<NAME>_CUDA_DEVICE`,
   falls back to parsing `gpu:N` out of the policy string.
4. `kernel_units(client)` — reads `FOREX_BOT_<NAME>_KERNEL_UNITS` and
   clamps to `[1, max_units_per_cube]`.
5. A thin `flatten_features` wrapper that already delegates to
   `common::cuda_flatten_features`, but only the wrapper exists per
   crate — the wrapper itself is duplicated three times.

Three sites of `normalize_*_device_policy` also share an identical
core for vendor-alias collapsing (cuda/rocm/metal/vulkan -> gpu,
cuda:N/rocm:N/metal:N/vulkan:N -> gpu:N):

* `statistical/common.rs::normalize_statistical_device_policy`
* `runtime/capabilities.rs::normalize_runtime_device_policy`
* `rl/dqn_impl.rs::normalize_rl_device_policy`
* `burn_models.rs::normalize_burn_device_policy` (adds `wgpu` family;
  slightly different)

Inside each one the prefix-list is the same except for which prefixes
the dispatcher supports (`wgpu:` is burn-only). The 4 of them total
~80 LOC of near-duplicated string manipulation.

### TASK 2 — Proposed consolidation

Target: extend the existing `common.rs` with a `gpu` submodule rather
than create a new module path. `common.rs` already houses
`cuda_flatten_features`; this is the natural home for the rest of the
shared GPU helpers.

New helpers (callable from all three CUDA-kernel files):

| Helper | Signature | Replaces |
|--------|-----------|----------|
| `cuda_kernel_enabled` | `fn(policy: &str, kernel_env_name: &str) -> bool` | three `*_cuda_kernel_enabled` |
| `is_kernel_disabled_env` | `fn(name: &str) -> bool` | three exact-copy `matches!` blocks |
| `cuda_device_id_from_policy` | `fn(policy: &str, device_env_name: &str, fallback_env_name: Option<&str>) -> usize` | three `cuda_device_id` |
| `cuda_kernel_units` | `fn(max_units: u32, units_env_name: &str) -> u32` | three `kernel_units` |
| `normalize_vendor_device_policy` | `fn(policy: &str, extra_prefixes: &[&str]) -> String` | shared core of all four `normalize_*_device_policy` (only the `extra_prefixes` differs: burn passes `["wgpu"]`, others pass `[]`) |

The `kernel_units` helper is split so it does NOT take a
`ComputeClient<CudaRuntime>` argument — instead, callers pass the
`max_units` value they already have. This keeps the common module
free of `cubecl::cuda` types, so the helpers compile even when the
caller is using a different runtime (e.g. tree-models GPU dispatch).

#### Estimated LOC reduction

| File | Before | After | Delta |
|------|--------|-------|-------|
| `evolution/crfmnes_gpu.rs` | 37 LOC for helpers | 9 LOC | -28 |
| `evolution/neat_gpu.rs` | 38 LOC for helpers | 9 LOC | -29 |
| `statistical/linear_gpu.rs` | 64 LOC for helpers | 14 LOC | -50 |
| `statistical/common.rs::normalize_*` | 22 LOC | 6 LOC | -16 |
| `runtime/capabilities.rs::normalize_*` | 22 LOC | 6 LOC | -16 |
| `rl/dqn_impl.rs::normalize_*` | 25 LOC | 6 LOC | -19 |
| `common.rs` (new helpers added) | 0 LOC | ~80 LOC | +80 |
| Net | — | — | -78 LOC |

#### NOT consolidated (deliberately)

* `hardware.rs` device detection — already centralized.
* `tree_models/config.rs::gpu_count` — uses tch backend + nvidia-smi
  parsing; specific to tree FFI dispatch path and unrelated to
  `cubecl` kernel boilerplate.
* `*_kernel` cubecl `#[cube(launch)]` definitions — those are the
  actual numeric kernels; each model has different math.
* `dqn_impl.rs`'s 33 `"cuda"` string matches — those are mostly
  test-fixture device strings and tch device construction. The four
  normalize helpers are unified; the rest is correctly per-runtime.
* `runtime/onnx.rs` ORT error mapping — uses external ort crate type
  (`OrtError`), genuinely runtime-specific.
* `burn_models.rs` and `deep_models.rs` `requested_policy` plumbing —
  policy reporting fields, not duplicated logic.

### TASK 3 — Applied consolidation

Scope: 4 new helpers in `common.rs` + 3 GPU-file edits + 3 normalizer
sites = 7 edits. Within the 30-edit ceiling. Applied in this pass.

See `crates/forex-models/src/common.rs` for the new helpers.

#### Files edited

1. `crates/forex-models/src/common.rs` — added `gpu_kernel` helpers.
2. `crates/forex-models/src/evolution/crfmnes_gpu.rs` — switched to
   `common::cuda_kernel_enabled`, `common::cuda_device_id_from_policy`,
   `common::cuda_kernel_units`.
3. `crates/forex-models/src/evolution/neat_gpu.rs` — same switch.
4. `crates/forex-models/src/statistical/linear_gpu.rs` — same switch.
5. `crates/forex-models/src/statistical/common.rs` — switched
   `normalize_statistical_device_policy` to delegate to
   `common::normalize_vendor_device_policy`.
6. `crates/forex-models/src/runtime/capabilities.rs` — same delegation
   for `normalize_runtime_device_policy`.
7. `crates/forex-models/src/rl/dqn_impl.rs` — same delegation for
   `normalize_rl_device_policy` (adds `wgpu:` prefix to extras).

Note: `burn_models.rs::normalize_burn_device_policy` was deliberately
LEFT AS-IS because it has additional support for the `"default"` token
and the `wgpu` prefix; refactoring it would require widening the
shared helper's contract. Recommended as a separate work item.

### Cargo check status

`cargo check -p forex-models` was run after the consolidation. See
section "Verification" in the in-session report.

### Operator rules compliance

* 11 canonical timeframes only (no H2) — unchanged.
* No new hardcoded values introduced.
* No synthetic data added.
* f32/fp32 only — all new helpers operate on `f32`/`u32`/`usize`; no
  f64 introduced.
* `Ordering::Relaxed` fix at `trading.rs:2511-2516` untouched.
