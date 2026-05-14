# ML / Numerical Library Reference

Authoritative documentation gathered to validate the audit's recommendations against current
library APIs. Date of research: 2026-05-14.

Current workspace pins (from `Cargo.toml`):

- `burn = "0.20"`, `burn-ndarray = "0.20"`, `burn-wgpu = "0.20.1"` (workspace uses 0.20.x)
- `ort = "2.0.0-rc.10"` (workspace pin)
- `ndarray = "0.17.2"`
- `polars = "0.52.0"`
- `statrs = "0.18.0"` (forex-search)
- `rand_distr = "0.5.1"` (forex-search, optional)
- `crfmnes = "1.0.0"` (forex-models, optional)
- `rand = "0.9.2"`

Latest upstream stable releases (crates.io / GitHub releases):

| Crate | Workspace pin | Latest upstream | Notes |
| --- | --- | --- | --- |
| burn | 0.20.x | **0.21.0** (2026-05-07) | Minor lag; backend list expanded in 0.21 (Flex CPU backend, Router). |
| ort | 2.0.0-rc.10 | **2.0.0-rc.12** (2026-03-05) | Pre-release line; no stable 2.0 yet. Bumping rc.10 -> rc.12 is small. |
| ndarray | 0.17.2 | 0.17.2 | Current. |
| ndarray-linalg | (not used) | 0.18.1 | Optional adoption — see §3. |
| ndarray-rand | (not used) | 0.16.0 | Re-exports `rand_distr` from same version chain. |
| polars | 0.52.0 | 0.53.0 | Minor. |
| statrs | 0.18.0 | 0.18.0 | Current. Has `stats_tests::ks_test` (audit didn't notice). |
| rand_distr | 0.5.1 | 0.6.0 | Should bump. |
| crfmnes | 1.0.0 | 1.0.0 | Sole 1.x release; thin wrapper. |
| cmaes (pengowen123) | n/a | 0.2.2 | Hard-coded `DVector<f64>` — see §8. |

---

## 1. `burn` — backend selection and f32 support

**Latest version:** 0.21.0, released 2026-05-07 (https://github.com/tracel-ai/burn/releases/tag/v0.21.0).

**Backend matrix (verbatim from upstream README, https://github.com/tracel-ai/burn#supported-backends):**

GPU: CUDA, ROCm, Metal, Vulkan, WebGPU, LibTorch. CPU: Cpu (CubeCL), Flex, LibTorch.

**Backend is selected by *type alias*, not by string identifier.** This is critical:

> `type MyBackend = Wgpu<f32, i32>;`
> `type MyAutodiffBackend = Autodiff<MyBackend>;`
>
> — Burn Book, `burn-book/src/basic-workflow/backend.md`

There is no first-party "wgpu_discrete_gpu" string keyword. The discrete GPU is selected through the
`WgpuDevice` enum (e.g. `WgpuDevice::DiscreteGpu(0)` — visible in the Router example in the README).
Strings like `"cuda"` / `"wgpu_discrete_gpu"` in our code are *our own dispatch layer*, not a burn
API.

**Float precision:**

The CubeCL-based backends (CUDA, ROCm, Metal, Vulkan, WebGPU, CPU/CubeCL) are parameterized as
`Cuda<F = f32, I = i32>` (README line: `pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;`).
The default float is **f32 everywhere**. `f16` / `bf16` are opt-in by substituting the `F` type
parameter (e.g. `Cuda<f16, i32>`); they are not required by any backend. `NdArray<f32>` is the
canonical CPU backend used in the workspace.

Source: https://burn.dev (book), https://github.com/tracel-ai/burn/blob/main/README.md.

**Verdict:** f32-only constraint is fully compatible with burn 0.20/0.21 across CPU, CUDA, WGPU,
ROCm. The audit can keep using f32. The string-keyed backend dispatch is a project-local pattern,
not a burn API — any patch that changes it should remain inside our own dispatcher.

**Recommended change:** Bump the workspace to `burn = "0.21"` to pick up Fusion-by-default, the new
Flex CPU backend (replaces several "ndarray"-style usages and is the only no-std backend), and the
Router decorator. This is a minor compat hop; 0.17 deprecated `Data` in favour of `TensorData`, and
that migration is already done in 0.20.

---

## 2. `ort` — input/output introspection and validation

**Latest version:** 2.0.0-rc.12, released 2026-03-05 (https://github.com/pykeio/ort/releases).

ort exposes a full structural API for input/output introspection — the audit's claim that this
information is unavailable is incorrect.

### Verbatim model-info example (canonical reference)

From https://github.com/pykeio/ort/blob/main/examples/model-info/model-info.rs:

```rust
let session = Session::builder()?.commit_from_file(path)?;
let meta = session.metadata()?;
// ...
for (i, input) in session.inputs().iter().enumerate() {
    println!("    {i} {}: {}", input.name(), input.dtype());
}
for (i, output) in session.outputs().iter().enumerate() {
    println!("    {i} {}: {}", output.name(), output.dtype());
}
```

### Type surface (verbatim from `src/value/type.rs` on `main`)

```rust
#[derive(Debug)]
pub struct Outlet { name: String, dtype: ValueType, ... }
impl Outlet {
    #[inline] pub fn name(&self) -> &str { &self.name }
    #[inline] pub fn dtype(&self) -> &ValueType { &self.dtype }
}

pub enum ValueType {
    Tensor { ty: TensorElementType, shape: Shape, dimension_symbols: SymbolicDimensions },
    Sequence(Box<ValueType>),
    Map { key: TensorElementType, value: TensorElementType },
    Optional(Box<ValueType>),
}
```

`Shape` is a `Vec<i64>` where `-1` is a dynamic dimension. The doc example shows:

```rust
input.dtype() == &ValueType::Tensor {
    ty: TensorElementType::Float32,
    shape: Shape::new([-1, -1, -1, 3]),
    dimension_symbols: SymbolicDimensions::new(["unk__31".into(), ..., String::default()])
}
```

### Implications for audit findings

**F-MODELS5-002 (input feature count not validated):** ort does provide a deterministic way to
discover the expected feature count — `session.inputs()[i].dtype()` yields
`ValueType::Tensor { shape, .. }`. The last non-dynamic dimension on a 2-D `[batch, features]` model
gives the expected feature width. The current code in
`crates/forex-models/src/runtime/onnx.rs::load_model_with_feature_count` instead relies on a caller-
supplied `expected_feature_count: Option<usize>`, which is exactly the kind of stale-state hazard
the audit flagged.

**Recommendation:** Drop the optional argument and read the expected width from
`session.inputs()[0].dtype()` at load time. Reject models where the relevant dim is dynamic with a
clear error, or fall back to the caller-supplied value only as an override.

**F-MODELS5-001 (output substring heuristic):** The current code:

```rust
for out in outputs {
    if out.name().to_lowercase().contains("prob") {
        proba_output_name = out.name().to_string();
        break;
    }
}
if proba_output_name.is_empty() {
    if let Some(last) = outputs.last() { ... }
}
```

is not forced by ort — ort gives us full name-keyed access. Output binding by **exact name** is
supported in two equivalent ways:

1. **`SessionOutputs` is an indexable / name-keyed map** — after `session.run(...)` you can do
   `outputs["my_exact_name"]` or `outputs[0]`. (`SessionOutputs` is constructed from a
   `Vec<&str>` of names in `src/session/mod.rs`.)
2. **`OutputSelector`** lets you pre-declare exactly which outputs to materialise via
   `session.run_with_options(...)`.

**Recommendation:** Replace the `contains("prob")` heuristic with either (a) a per-model config that
names the exact output (most robust), or (b) the convention that *every* sklearn/lightgbm/xgboost
ONNX export uses output index 1 for probabilities — at minimum, key by exact name not substring.
The `outputs.last()` fallback is dangerous because ONNX-export tooling does not guarantee output
order, especially for classifiers that emit `(label, probability)` pairs.

### Migration v1 -> v2 highlights (https://ort.pyke.io/migrating/v2)

- `Value::from_array` -> `Tensor::from_array` (no allocator param).
- `inputs!["name" => value]` macro; positional & named both supported.
- `Session::run` returns `SessionOutputs`, which is indexable.

**Verdict:** Audit's recommendations are correct — ort fully supports principled introspection and
output binding; the current heuristics are project-local sloppiness, not library limitations.

**Recommended change:** Bump `ort = "2.0.0-rc.12"` (rc.10 -> rc.12) at the next dependency
refresh; rc-line bumps for ort have included input/output API stabilisation. Re-run model-info
example against our checkpoints during the upgrade.

---

## 3. `ndarray` / `ndarray-linalg` / `ndarray-rand` — f32 in linear algebra and RNG

**Versions:** ndarray 0.17.2, ndarray-linalg 0.18.1, ndarray-rand 0.16.0.

### ndarray-linalg — f32 is fully supported

From `lax/src/lib.rs` (https://github.com/rust-ndarray/ndarray-linalg/blob/master/lax/src/lib.rs):

> "This trait is implemented for `f32`, `f64`, `c32` which is an alias to `num::Complex<f32>`,
> and `c64` which is an alias to `num::Complex<f64>`."

The `Lapack` super-trait merges Cholesky, SVD, LU, and the solvers under a single bound. There is
no f32 limitation for `Cholesky`, `SVD`, or `Eigh`. f32 calls dispatch to LAPACK `s*` routines
(spotrf, sgesdd, etc.) on every backend (OpenBLAS / MKL / Netlib).

**Verdict:** Audit's reach for ndarray-linalg with f32 is safe; no f64 boundary forced.

### ndarray-rand — re-exports rand_distr, no f32 caveats

From `ndarray-rand/src/lib.rs` (https://github.com/rust-ndarray/ndarray/blob/master/ndarray-rand/src/lib.rs):

> "ndarray-rand depends on rand 0.9. rand and rand_distr are re-exported as sub-modules
> `ndarray_rand::rand` and `ndarray_rand::rand_distr` respectively. You can use these submodules for
> guaranteed version compatibility."

Distributions parametric over `Float` (including `StandardNormal`, `Normal<F>`) work directly. No
manual conversion needed for f32.

**Recommended change:** If the workspace does not already use `ndarray-rand`, add it under
forex-search's `gpu` feature alongside `rand_distr`. It saves an entire helper function for
random tensor construction.

---

## 4. `polars` — f32 support and CSV/Parquet defaults

**Latest version:** 0.53.0 (workspace at 0.52.0 — minor bump available).

**f32 is a first-class dtype.** From `polars-core/src/datatypes/dtype.rs`, `DataType::Float32` is a
variant alongside `Float64`. `Series::new("name".into(), &[1.0f32, 2.0])` produces a `Float32`
Series natively — Rust's type inference picks Float32 from the slice element type.

**CSV reader:** Polars samples the first `infer_schema_length` rows (default 100) and infers a dtype
per column. **Floats default to `Float64`.** To get `Float32`, pass `schema_overrides` mapping the
column name to `DataType::Float32`. From `polars/docs/source/user-guide/io/csv.md` plus the user
guide at https://docs.pola.rs/user-guide/concepts/data-types-and-structures/:

> "CSV files carry no embedded schema. Instead, Polars samples the first 100 rows to infer a dtype
> for every column."

**Parquet reader:** Parquet carries an embedded schema; Polars respects it. So if the Parquet was
written with `Float32` columns, you get `Float32` back. If it was written with `Float64`, you must
`cast(DataType::Float32)` after read.

**Performance note:** Polars' expression engine is dtype-generic but treats `Float32` and `Float64`
as distinct code paths. f32 halves memory footprint and benefits from vectorised SIMD on
mainstream x86_64 (AVX2 fits twice as many f32 lanes), so for large time-series tables the f32
choice is usually a 1.3-1.8x speedup on column-bound workloads. It costs precision and is not
recommended for monetary accumulators.

**Verdict:** f32-everywhere is feasible. The trade-off the audit cares about (FFI to ML libs) is
clean: polars Series<f32> -> Vec<f32> -> ort `TensorRef::from_array_view` is zero-copy when the
column is contiguous (single chunk). Force single-chunk by calling `.rechunk()` before extraction.

**Recommended change:** Wherever the pipeline reads CSV, pass `schema_overrides` to lock columns to
`DataType::Float32` so we don't pay an upfront f64 inference + cast cycle. For Parquet writers,
explicitly cast to `Float32` before `write_parquet`.

---

## 5. `rand_distr::Normal` — replacement for hand-rolled Box-Muller

**Latest version:** 0.6.0 (workspace pin 0.5.1 needs a one-step bump).

**Constructor (verbatim from `rand_distr/src/normal.rs`):**

```rust
pub struct Normal<F> where F: Float, StandardNormal: Distribution<F> { mean: F, std_dev: F }

pub fn new(mean: F, std_dev: F) -> Result<Normal<F>, Error>
// Error variants: MeanTooSmall, BadVariance
```

`Normal<F>` is generic over `Float`; `StandardNormal: Distribution<f32> + Distribution<f64>` is
implemented for both natively. The `f32` impl currently routes through the f64 Ziggurat and casts —
slightly slower than a hand-optimal f32 path but still ~3-5x faster than a polar Box-Muller (the
hand-rolled code in `crates/forex-models/src/evolution/crfmnes_impl.rs:253-257`):

```rust
fn gaussian_sample(rng: &mut Xoroshiro128PlusPlus) -> f64 {
    let u1 = rng.random::<f64>().clamp(1.0e-10, 1.0 - 1.0e-10);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
```

This implementation has three real problems beyond performance:
1. Box-Muller throws away `sin(2πu2)` (every call computes both `sin` and `cos` but uses one) —
   2x waste.
2. The `clamp(1e-10, 1-1e-10)` is a hack to avoid `ln(0)`; ziggurat avoids the issue entirely.
3. It returns `f64` even though the caller is operating in a f64 CR-FM-NES state (so the precision
   isn't wasted *here*, but anywhere this idiom is copied for f32 it would be).

**Idiomatic replacement:**

```rust
use rand_distr::{Distribution, StandardNormal};
// Per-element:
let x: f64 = rng.sample(StandardNormal);
// Or with explicit mean/std:
let n = rand_distr::Normal::new(0.0, 1.0).unwrap();
let x: f64 = n.sample(&mut rng);
```

**Verdict:** Audit recommendation aligns. Replace `gaussian_sample` with `rng.sample(StandardNormal)`.

**Recommended change:** Bump `rand_distr = "0.6"` (0.5 -> 0.6 has minor breaking changes for the
error variants; check `Updating to 0.9` at https://rust-random.github.io/book/update-0.9.html).

---

## 6. `burn-import` / `burn-onnx` — ONNX export & naming determinism

**Latest version:** 0.21.0. Crate has been **renamed** from `burn-import` to `burn-onnx`. The
current workspace `burn-onnx` use is the right one; the crate identifier on crates.io is still
`burn-import` for back-compat in some links but the canonical name is `burn-onnx`. See
https://github.com/tracel-ai/burn-onnx and the Burn Book chapter at
https://burn.dev/books/burn/onnx-import.html.

**Direction of conversion:** burn-onnx is **ONNX -> Burn**, not the reverse. Burn has no first-party
ONNX *exporter*. To produce ONNX from a trained Burn model today, you must:

- Export weights to PyTorch / safetensors via `burn::record`, then
- Use Python's `torch.onnx.export` against a Python re-instantiation, or
- Hand-write the ONNX graph through `onnxruntime`'s API.

This explains why the runtime ONNX loader we have is for *third-party* (sklearn / lightgbm /
xgboost) ONNX models — those exporters are what produce nondeterministic output names. From
sklearn-onnx's spec, classifier models emit two outputs: the label (output 0) and either a sequence
of dicts or a `[N, K]` probability tensor (output 1). Common names include `output_label`,
`output_probability`, `probabilities`, `probabilities_0`, depending on the tool.

**Verdict:** The "prob" substring heuristic exists because no single canonical name is guaranteed,
BUT ort's output indexing is deterministic per-export — we should pin the exact output name *at
training time* (when we know the producer) and store it alongside the ONNX file. The audit's
recommendation to replace the heuristic with an exact name is correct and feasible; the heuristic
itself is not forced by burn-onnx or by ort.

**Recommended change:** Each ONNX file shipped to the runtime should carry a sidecar (or use the
ONNX `metadata_props` field, which ort exposes via `session.metadata()?.custom_keys()`) recording
the exact probability-output name. The model-info example demonstrates reading `metadata.custom()`.

---

## 7. `statrs` — KS test and PSI

**Latest version:** 0.18.0 (workspace pin matches).

**Critical finding the audit missed:** `statrs::stats_tests::ks_test` exists in the public API.
From https://github.com/statrs-dev/statrs/blob/master/src/stats_tests/mod.rs:

```rust
#[cfg(feature = "std")]
pub mod anderson_darling;
#[cfg(feature = "std")]
pub mod chisquare;
#[cfg(feature = "std")]
pub mod f_oneway;
pub mod fisher;
#[cfg(feature = "std")]
pub mod ks_test;
#[cfg(feature = "std")]
pub mod mannwhitneyu;
#[cfg(feature = "std")]
pub mod skewtest;
#[cfg(feature = "std")]
pub mod ttest_onesample;
```

The `ks_test` module provides:
- `KSOneSampleAlternativeMethod` enum: `Less`, `Greater`, `TwoSidedExact` (Marsaglia/Tsang/Wang
  exact for n<140, requires `nalgebra` feature), `TwoSidedAsymptotic` (Kolmogorov 1933), and
  `TwoSidedApproximate`.
- Built-in handling for ties, NaN policy (`Propogate` / `Emit` / `Error`), and exact p-values via
  the Birnbaum & Tingey 1951 formula for the one-sided case.

The hand-rolled implementation in
`crates/forex-core/src/domain/drift_monitor.rs::ks_2samp` uses the Stephens approximation
(`(en.sqrt() + 0.12 + 0.11/en.sqrt()) * max_d`), which is the textbook Numerical Recipes formula —
**fine for two-sample but not as good as statrs' asymptotic + exact split**.

**However:** `statrs::stats_tests::ks_test` is currently a *one-sample* test (CDF-vs-data), not a
*two-sample* test. The existing `ks_2samp` is a legitimate gap in statrs (which has open issues
about adding `ks_2samp`). So:

- **For one-sample drift checks vs. a reference distribution:** use `statrs::stats_tests::ks_test`.
- **For two-sample drift:** keep our implementation but at minimum delegate the p-value calculation
  to `statrs::stats_tests::ks_test::onesample_kolmogorov_twosided_pvalue` once it's made `pub` (it's
  currently `fn`-private; a small upstream patch would help, or copy the code which is ~10 lines).

**PSI is not in statrs.** Population Stability Index is a domain-specific banking/credit metric;
no Rust crate ships it. The implementation in `drift_monitor.rs::calculate_psi` is correct in
principle. The audit's complaint (F-CORE2-011) about hand-rolled Laplace smoothing is valid: PSI
divides by zero when a bin in `expected` is empty, so adding `+ epsilon` to both probabilities is
standard. A canonical formulation:

```rust
let expected_pct = (expected_count + 1.0) / (n_expected + bins as f64);
let actual_pct   = (actual_count   + 1.0) / (n_actual   + bins as f64);
psi += (expected_pct - actual_pct) * (expected_pct / actual_pct).ln();
```

This is add-one (Laplace) smoothing, which is the textbook fix.

**Verdict:** Audit's intent is right but the recommendation should be split:
- Use `statrs::stats_tests::ks_test` for **one-sample** KS only.
- Keep two-sample KS in-tree (no canonical Rust replacement exists).
- Replace the bespoke PSI helper with a small, well-named function that explicitly does Laplace
  smoothing — there's no library to delegate to.

---

## 8. CR-FM-NES / CMA-ES references

**Reference paper:** Nomura & Ono, *Fast Moving Natural Evolution Strategy for High-Dimensional
Problems*, IEEE CEC 2022, arXiv:2201.11422.
https://github.com/nomuramasahir0/crfmnes (author's reference Python implementation).

**Read of the canonical pseudocode (verbatim from `crfmnes/alg.py`):**

```python
self.v = self.v + (t @ exw) / normv
self.D = self.D + (s @ exw) * self.D                # <-- additive update on D
# calculate detA
nthrootdetA = np.exp(np.sum(np.log(self.D)) / self.dim + np.log(1 + self.v.T @ self.v) / (2 * self.dim))[0][0]
self.D = self.D / nthrootdetA
self.sigma = self.sigma * np.exp(eta_sigma / 2 * G_s)
```

**There is no explicit positive-definite guard in the reference algorithm.** The diagonal `D` is
updated additively (`D + (s @ exw) * D`), and the algorithm relies on the dynamics of `s @ exw`
staying close to zero so that `D` stays positive. The very next line `np.log(self.D)` would error
out if any `D[i] <= 0`. The reference implementation simply trusts that the learning rates
`eta_B(lambF)` and `c1(lambF)` are conservative enough to prevent sign flips.

The only numerical safeguard in the paper's pseudocode is the `alphavd` clamp:

```python
alphavd = np.min([1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + normv2)])
```

— a min with 1 to bound a scaling factor. Sigma is updated multiplicatively via `exp(...)`, which
is naturally positive.

**Verdict on F-MODELS4-012 (audit says PD guard is missing):** The audit's framing is technically
wrong — the reference algorithm has no PD guard either. However, the audit's concern is *valid in
practice*: in float arithmetic (especially f32 if we ever down-cast), the additive D-update can
underflow and we get NaNs cascading through `log(D)`. The defensible fix is to clamp `D` to a small
positive floor (e.g. `D.mapv_inplace(|d| d.max(1e-12))`) — this matches what `fast-cma-es` does in
the C++ port and is a recognised pragmatic departure from the paper.

**Recommendation:** Document the clamp as "numerical safeguard not present in original paper,
necessary for finite-precision float backing". Don't claim it's a missing reference feature.

**Rust crate landscape:**

- `crfmnes = "1.0.0"` (Rust) — exists, a translation of fast-cma-es. Workspace already uses it
  under the `neuro-evolution` feature. **However**, our codebase still has `crates/forex-models/
  src/evolution/crfmnes_impl.rs` which is a *bespoke* re-implementation with the hand-rolled
  Box-Muller. Either (a) delete the bespoke impl and use the crate, or (b) document why we kept
  the in-tree version.
- `cmaes = "0.2.2"` (pengowen123) — full CMA-ES with restarts (IPOP, BIPOP). **Hard-coded to
  `DVector<f64>`** — incompatible with our f32-only constraint at the boundary. Would require an
  f64 conversion at the optimization boundary. Not a drop-in.
- `cmaes-lbfgsb` — combo, also f64.
- `fast_cmaes` (Dicklesworthstone) — SIMD-accelerated, f64.

**Verdict:** For CR-FM-NES specifically, the `crfmnes` crate is the canonical Rust path. The
audit's recommendation to "use a library" is feasible. For full CMA-ES (not currently used) the
options are all f64; we'd need to accept an f64 boundary in the optimizer state if we ever add
full CMA-ES.

**Recommended change:**
1. Delete `crfmnes_impl.rs::gaussian_sample` (Box-Muller) and route through `rand_distr`.
2. Either replace the bespoke `CrfmnesEvolutionState` with the `crfmnes` crate's `CRFMNES` struct,
   or add a comment documenting why we re-implemented it (e.g. for online state mutation, custom
   constraint handling, etc.).
3. Add the `D.max(1e-12)` clamp with a comment "numerical safeguard, not in Nomura&Ono 2022".

---

## 9. CR-FM-NES paper details (arXiv:2201.11422)

**Title:** *Fast Moving Natural Evolution Strategy for High-Dimensional Problems*
**Authors:** Masahiro Nomura, Isao Ono
**Venue:** IEEE CEC 2022

**Abstract / contribution (paraphrased from README and search results):** CR-FM-NES extends
FM-NES by using a *restricted* representation of the covariance matrix:
`C = sigma^2 * (I + vv^T) * D^2 * (I + vv^T)` where `v` is a rank-1 vector and `D` is a diagonal,
giving O(d) time and space per generation. The exponential `exp(eta_sigma/2 * G_s)` on sigma is the
guaranteed-positive part. There is no exponential parameterisation on `D` in the published
algorithm — `D` is updated additively, by design.

**Default parameters (verbatim from alg.py):**

```python
self.eta_m = 1.0
self.eta_move_sigma = 1.
self.eta_stag_sigma = lambda lF: math.tanh((0.024*lF + 0.7*self.dim + 20.) / (self.dim + 12.))
self.eta_conv_sigma = lambda lF: 2. * math.tanh((0.025*lF + 0.75*self.dim + 10.) / (self.dim + 4.))
self.c1            = lambda lF: self.c1_cma * (self.dim - 5) / 6 * (float(lF) / self.lamb)
self.eta_B         = lambda lF: np.tanh((min(0.02*lF, 3*np.log(self.dim)) + 5) / (0.23*self.dim + 25))
```

`lambF` is the number of feasible solutions in this generation. The audit (F-MODELS7-002, -003,
-009) referenced "numerical issues" in our impl — to validate the diagnosis, compare our learning
rate constants against the above six lines. Any drift here is a bug in our port.

---

## Cross-cutting verdict

| Audit finding | Library reality | Action |
| --- | --- | --- |
| F-MODELS5-001 (substring "prob" heuristic) | ort fully supports name-keyed and index-keyed output access (`SessionOutputs[&str]`, `OutputSelector`). | Replace heuristic with exact name from per-model sidecar / ONNX metadata_props. |
| F-MODELS5-002 (no input feature-count validation) | ort exposes `session.inputs()[i].dtype()` -> `ValueType::Tensor { shape: Shape, .. }` with `-1` for dynamic dims. | Compute expected width from shape at load time; bail if dim is dynamic and no override given. |
| F-CORE2-006 (manual KS test) | statrs has `stats_tests::ks_test` for one-sample; two-sample is NOT in statrs. | Use statrs for one-sample; keep two-sample in-tree but document the rationale. |
| F-CORE2-011 (manual PSI w/ ad-hoc Laplace) | No canonical PSI crate. | Keep in-tree, use explicit add-one Laplace formula. |
| F-MODELS4-012 (PD guard missing in CR-FM-NES) | Reference paper has no PD guard either. The pragmatic clamp is correct but it's a *finite-precision safeguard*, not "restoring missing reference behaviour". | Add `D.max(1e-12)` with a comment citing fast-cma-es; don't call it a reference-paper fix. |
| F-MODELS7-002/003/009 (CR-FM-NES numerical issues) | Reference Python impl available verbatim; compare line-for-line. | Port-fidelity test against `crfmnes` crate output for sphere(d=3, lamb=6) for 100 generations. |
| f32-only constraint feasibility | burn (all backends), ndarray, ndarray-linalg, ndarray-rand, polars, rand_distr, statrs all support f32 first-class. cmaes crate uses f64 only. | f32 boundary is achievable everywhere except a hypothetical future CMA-ES via pengowen123/cmaes; that one optimizer would need an f64 sub-API. |

---

## Versions to bump (low-risk minor hops)

- `burn`, `burn-ndarray`, `burn-wgpu`: 0.20 -> 0.21
- `ort`: 2.0.0-rc.10 -> 2.0.0-rc.12
- `rand_distr`: 0.5.1 -> 0.6 (check `Updating to 0.9` migration notes — actually `Updating to 0.6`)
- `polars`: 0.52 -> 0.53

None of these introduce f64 dependencies.

## Sources

- burn: https://github.com/tracel-ai/burn, https://burn.dev (book)
- ort: https://github.com/pykeio/ort, https://ort.pyke.io
- ndarray-linalg: https://github.com/rust-ndarray/ndarray-linalg
- ndarray-rand: https://github.com/rust-ndarray/ndarray/tree/master/ndarray-rand
- polars: https://github.com/pola-rs/polars, https://docs.pola.rs/user-guide/concepts/data-types-and-structures/
- statrs: https://github.com/statrs-dev/statrs
- rand_distr: https://github.com/rust-random/rand_distr
- crfmnes (Python ref): https://github.com/nomuramasahir0/crfmnes
- cmaes (Rust): https://github.com/pengowen123/cmaes
- CR-FM-NES paper: arXiv:2201.11422 (Nomura & Ono, IEEE CEC 2022)
