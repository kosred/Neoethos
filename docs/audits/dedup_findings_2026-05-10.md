# Duplicate-function audit — findings and extraction plan

Companion to the 28-60 follow-on slice. Surfaces the duplicated
private helpers and parallel implementations that survived the
phase 1-60 contract work and proposes a module layout for the
extraction pass.

The audit principle (P1-1 / P1-5 in the consolidated plan): every
helper that lives in two or more files with the same intent is a
candidate for extraction into a shared module. Do not delete blindly:
extract → switch call sites → run tests → remove the now-unused local
copies in a follow-up commit.

---

## Findings by category (updated 2026-05-10 with deeper scan)

The initial scan covered forex-search and forex-models surface-level
duplicates. The deeper pass below extends across forex-data,
forex-app, forex-cli and the nested forex-models subdirectories,
adding categories 10-13.



### 1. Hash helpers — FNV-1a + stable JSON hash

| File | Function | Visibility |
|---|---|---|
| `crates/forex-core/src/contracts/temporal.rs` | `fnv1a64` | private |
| `crates/forex-search/src/artifact_io.rs` | `fnv1a64` | pub |
| `crates/forex-search/src/genetic/evolution_math.rs` | `fnv1a_update` | private |

`forex-search::artifact_io::fnv1a64` is the canonical FNV-1a
implementation; the other two recompute the same constants. The
`fnv1a_update` variant takes a starting hash so the seen-signature
ledger can build a rolling hash; that is a thin extension of the
canonical helper.

**Extraction target:** `forex-core::utils::hashing` (new module),
exposing `pub fn fnv1a64(bytes: &[u8]) -> u64` and
`pub fn fnv1a64_update(seed: u64, bytes: &[u8]) -> u64`.
`forex-search::artifact_io::fnv1a64` becomes a re-export;
`forex-core::contracts::temporal` and
`forex-search::genetic::evolution_math` switch to the shared helper.

### 2. Atomic JSON IO

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/artifact_io.rs` | `write_json_atomic`, `read_json`, `temporary_path`, `stable_json_hash` | pub |
| `crates/forex-models/src/statistical/common.rs` | `write_json`, `read_json` | pub |
| `crates/forex-models/src/tree_models/common.rs` | `atomic_write` (raw bytes) | pub |
| `crates/forex-models/src/ensemble.rs` | `write_json_with_backup` | private |
| `crates/forex-data/src/core/vortex_io.rs` | `temp_path_for` | private |

The forex-search version uses fsync + same-directory tempfile +
atomic rename; the forex-models versions are simpler. The two should
converge so model artifacts get the same crash-safety guarantees as
search artifacts.

**Extraction target:** keep `forex-search::artifact_io` as the
canonical surface, lift it to `forex-core::utils::artifact_io`, and
have forex-models / forex-data depend on the lifted module. Delete
the four local variants once call sites switch.

### 3. Statistical helpers — mean / stddev / mean_std

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/portfolio.rs` | `mean`, `stddev` | private |
| `crates/forex-search/src/quality.rs` | `mean`, `stddev_sample` | private |
| `crates/forex-search/src/stop_target.rs` | `mean`, `stddev` | private |
| `crates/forex-search/src/eval.rs` | `mean_std` | private |
| `crates/forex-search/src/cubecl_eval.rs` | `mean_std` | private |

Five private definitions of the same trio. The `stddev_sample`
variant uses Bessel's correction; the `stddev` variants use the
population formula. Both are valid in different contexts; the
extracted module should expose both explicitly.

**Extraction target:** `forex-core::utils::stats`, exposing
`pub fn mean(&[f64]) -> f64`, `pub fn stddev(&[f64], mean: f64) -> f64`
(population), `pub fn stddev_sample(&[f64], mean: f64) -> f64`
(Bessel), `pub fn mean_std(&[f64]) -> (f64, f64)`. Every call site
above switches and deletes the local helper.

### 4. Correlation helpers — Pearson f32/i8

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/discovery.rs` | `pearson_correlation` (f32) | private |
| `crates/forex-search/src/discovery.rs` | `pearson_corr_i8` | private |

Both live in the same file but compute the same statistic on
different element types. The `i8` variant is used for signal-vector
similarity in the diversity / archive code; the `f32` variant is
used for feature-correlation prefiltering.

**Extraction target:** `forex-core::utils::stats::pearson` with two
public functions parametrised by element type. Same file, no new
crate dependency.

### 5. Mean-vector helper (GPU search)

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/discovery_gpu.rs` | `mean_vector` | private |
| `crates/forex-search/src/hpc_gpu_discovery.rs` | `mean_vector` | private |

Two private copies of `mean_vector(elites: &[Vec<f32>]) -> Vec<f32>`
in adjacent files. Trivial to deduplicate.

**Extraction target:** the new `forex-core::utils::stats` module
gains `pub fn mean_vector_f32(&[Vec<f32>]) -> Vec<f32>`. Both call
sites switch.

### 6. `finite_or` numeric guard

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/genetic/diversity.rs` | `finite_or` | private |
| `crates/forex-search/src/genetic/regime_labels.rs` | `finite_or` | private |

Identical implementations in adjacent modules.

**Extraction target:** `forex-core::utils::numeric::finite_or` (or
fold into the stats module). Switch both call sites.

### 7. `clamp_probability` / `clamp_unit` / `clamp_f32`

| File | Function | Visibility |
|---|---|---|
| `crates/forex-models/src/ensemble.rs` | `clamp_probability(f32) -> f32` | private |
| `crates/forex-models/src/statistical/linear_gpu.rs` | `clamp_probability(f32) -> f32` | private |
| `crates/forex-models/src/forecasting/swarm_impl.rs` | `clamp_unit(f32) -> f32` | private |
| `crates/forex-models/src/evolution/neat_gpu.rs` | `clamp_f32(f32, min, max) -> f32` | private |

Three names, two semantics: clamp-to-`[0, 1]` (the first three) and
generic min/max clamp (the fourth). The first three are exact
duplicates with different names.

**Extraction target:** `forex-core::utils::numeric::clamp_unit_f32`
(canonical [0,1] clamp) plus the std-lib `f32::clamp` for the
generic case (the `clamp_f32` wrapper is redundant given std-lib).

### 8. Env-var readers

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/genetic/runtime_overrides.rs` | `env_f64_finite`, `env_f64_positive_finite`, `env_f64_non_negative_finite`, `env_usize_positive`, `env_string_lowercase`, `env_string_nonempty`, `env_u64`, `env_f32_finite` | private |
| `crates/forex-search/src/genetic/smc_indicators.rs` | `smc_env_f64`, `smc_env_usize`, `smc_env_bool` | private |
| `crates/forex-models/src/training_orchestrator.rs` | `parse_f64_param`, `parse_usize_param` | private |

The `runtime_overrides` set is the canonical typed-Option boundary
that Phase 17-22 introduced. The `smc_indicators` set is the older
"fallback to default on parse failure" style; the
`training_orchestrator` set parses HashMap params (different source,
similar shape). The first two should converge on the typed-Option
boundary.

**Extraction target:** `forex-core::utils::env`, lifting the typed
`runtime_overrides` helpers (`env_*_finite`, `env_string_*`).
`smc_indicators` switches to the typed boundary and folds the
"fallback" behavior at the call site (one-liner). The HashMap
parsers stay in `training_orchestrator` — different source, no real
duplication.

### 9. OHLCV slicing

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/discovery.rs` | `slice_ohlcv(start, end) -> Ohlcv` | private |
| `crates/forex-search/src/genetic/regime_labels.rs` | `slice_ohlcv(start, end, fallback_timestamps) -> Ohlcv` | private |

Two slightly different signatures; the regime-labels variant accepts
a fallback timestamp slice for cases where `Ohlcv::timestamp` is
`None`. Easy to unify with an `Option<&[i64]>` parameter.

**Extraction target:** `forex-data::slicing::slice_ohlcv` (new
module on `forex-data`, the natural owner of the `Ohlcv` type).

### 10. `flatten_features` (CUDA prep)

| File | Function | Visibility |
|---|---|---|
| `crates/forex-models/src/evolution/crfmnes_gpu.rs` | `flatten_features` | private |
| `crates/forex-models/src/evolution/neat_gpu.rs` | `flatten_features` | private |
| `crates/forex-models/src/statistical/linear_gpu.rs` | `flatten_features` | private |

Three near-identical `fn flatten_features(features: &Array2<f32>,
input_dim: usize) -> Result<Vec<f32>>` bodies that validate column
count and flatten the matrix. Each emits a different error message
("neuro-evo cuda…", "NEAT cuda…", "statistical cuda…") but the math
is identical.

**Extraction target:** `forex-models::common::cuda_prep::flatten_features`
with a `&'static str` `caller_label` parameter for the error message.

### 11. `sigmoid` (activation)

| File | Function | Visibility |
|---|---|---|
| `crates/forex-models/src/ensemble.rs` | `sigmoid` | private |
| `crates/forex-models/src/statistical/bayesian_impl.rs` | `sigmoid` | private |

The bayesian variant is numerically stable (handles negative values
without overflow); the ensemble variant is the naive `1/(1+e^{-x})`.
The stable form should win.

**Extraction target:** `forex-core::utils::numeric::stable_sigmoid_f32`.
Switch both call sites; document why the stable form is preferred.

### 12. Other forex-models math helpers

| File | Function | Visibility | Notes |
|---|---|---|---|
| `crates/forex-models/src/anomaly/forest_impl.rs` | `median(Vec<f32>)` | private | Sort-based median. |
| `crates/forex-search/src/stop_target.rs` | `median_ignore_nan(&[f64])` | private | Same idea, different element type + NaN policy. |
| `crates/forex-models/src/forecasting/swarm_impl.rs` | `percentile(&[f32], q)` | private | Single-quantile lookup. |
| `crates/forex-models/src/forecasting/swarm_impl.rs` | `moving_average_forecast` / `ewma_forecast` | private | Series helpers. |
| `crates/forex-search/src/stop_target.rs` | `rolling_mean` | private | Equivalent to MA. |

The median / percentile / rolling-mean / EWMA helpers are not
straight duplicates but they live in the same conceptual space. They
should converge under `forex-core::utils::stats::series` so the
forex-models RL / forecasting paths and the search stop-target
helpers share one tested implementation.

**Extraction target:** the new `forex-core::utils::stats` module
gains a `series` submodule covering `median_ignore_nan`,
`percentile_sorted`, `rolling_mean`, `ewma`, `moving_average`. Each
caller picks the variant it needs; no behavior change.

### 13. Discovery_gpu fallback chains

| File | Function | Visibility |
|---|---|---|
| `crates/forex-search/src/discovery_gpu.rs` | `append_degraded_reason` | private |
| `crates/forex-search/src/lib.rs` (gpu-disabled stub) | duplicate `append_degraded_reason` | private |
| `crates/forex-search/src/discovery_gpu.rs` | `resolve_cpu_fallback_runtime` | private |
| `crates/forex-search/src/lib.rs` (gpu-disabled stub) | duplicate `resolve_cpu_fallback_runtime` | private |

The `cfg(not(feature = "gpu"))` stub in `lib.rs` shadows the entire
`discovery_gpu` module with copies of the helpers. This is by
design (different code paths) but the two helper bodies have drifted
enough that consolidating them would catch future divergence.

**Extraction target:** keep the `cfg`-gated stub but lift the
shared helpers (`append_degraded_reason`,
`resolve_cpu_fallback_runtime`) to `forex-search::scheduler_assignment`
so both the gpu-enabled and gpu-disabled paths import the same
implementation.

---

## Extraction priority

| Priority | Module | Reason |
|---|---|---|
| P0 | hash helpers (Finding 1) | Deterministic semantics — duplicating risks subtle hash drift between artifact kinds. |
| P0 | atomic IO (Finding 2) | Crash-safety divergence between forex-search and forex-models is a production risk. |
| P1 | stats (Findings 3, 4, 5) | Five copies of `mean`/`stddev` is the largest cluster. Safe to extract; tests stay local. |
| P1 | env helpers (Finding 8) | The typed boundary already exists; just lift it to the shared utils. |
| P2 | numeric guards (Findings 6, 7) | Trivial; extract when touching the surrounding code. |
| P2 | OHLCV slicing (Finding 9) | One signature unification needed. Schedule with the next discovery / regime-labels touch. |

## Module layout proposal

```text
forex-core/
  src/
    utils/
      mod.rs                  // re-exports the helpers below
      hashing.rs              // fnv1a64 + fnv1a64_update + stable_json_hash
      stats.rs                // mean / stddev / stddev_sample / mean_std / pearson / mean_vector_f32
      numeric.rs              // finite_or / clamp_unit_f32
      env.rs                  // env_f64_finite / env_string_* / etc.
      atomic_io.rs            // write_json_atomic / read_json / temporary_path
forex-data/
  src/
    slicing.rs                // slice_ohlcv (Ohlcv-aware)
```

`forex-search::artifact_io` and `forex-models::statistical::common`
turn into thin re-export shims pointing at `forex_core::utils`.
After the call-site switch lands, the shims are removed in a P1-5
cleanup commit.

## Out-of-scope reminders

- Do not refactor signal-synthesis code (`signals_for_gene_full` vs
  `signals_for_gene`) — they have different SMC-gating semantics
  (audit item: search_discovery_pipeline_audit_2026-05-03.md). They
  look like duplicates but they are not.
- Do not collapse `evaluate_population_core` and
  `fast_evaluate_strategy_core` into one — the population-evaluator
  pre-computes batched arrays the scalar variant cannot consume.
- Genetic-search runtime overrides (Phases 17-22) are already the
  shared boundary; do not re-extract them.

## Recommended sequencing

1. **Phase 62 (P0):** lift `forex-search::artifact_io` to
   `forex-core::utils::atomic_io` + hashing helpers. Switch all call
   sites; delete duplicates.
2. **Phase 63 (P1):** extract stats module; switch the five
   mean/stddev call sites and the two mean_vector / pearson copies.
3. **Phase 64 (P1):** lift env helpers; switch SMC and runtime-override
   call sites onto the shared boundary.
4. **Phase 65 (P2):** consolidate numeric guards + OHLCV slicing.
5. **Phase 66 (P1-5):** remove now-unused local copies and the
   transitional re-export shims.

Each phase keeps the same testing discipline as Phases 16-60: small
diff, tests around the change, `cargo fmt` + `cargo test` + `cargo
check` on every downstream crate before commit.
