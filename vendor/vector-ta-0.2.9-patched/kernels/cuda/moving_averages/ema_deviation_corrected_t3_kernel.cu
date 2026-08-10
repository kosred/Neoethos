// ema_deviation_corrected_t3 — CUDA f64 kernel.
//
// WHAT THIS REPLACES
// ------------------
// NOTHING. No `.cu`, no wrapper, no `F64_KERNELS` row: the lane answered
// `CudaF64KernelMissing`. (`tilson_kernel.cu` carries a T3, but a DIFFERENT
// one — plain T3 with no deviation correction — and it is registered under the
// id `tilson`. It is the in-repo precedent for the shape, not for the maths.)
//
// CPU REFERENCE — src/indicators/moving_averages/ema_deviation_corrected_t3.rs
// -----------------------------------------------------------------------------
//   :267 alpha_t3                    :276 correction_variance_scale
//   :281 compute_coefficients        :292 prepare_input
//   :310 validate_params_for_len     :334 ema_deviation_corrected_t3 (the entry
//                                          the brief names)
//   :353 compute_into_slices         — the loop this kernel reproduces
//
// WHICH OUTPUT
// ------------
// Two outputs. registry.rs:537 settles which one this lane emits: "Primary
// output is the corrected line; secondary output is the raw T3 line." This
// kernel writes `corrected` (:420). The `t3` column is not emitted, for the
// same reason the multi-output indicators in this table emit one column: the
// lane's contract is one row per period.
//
// THE PARAMETERS THAT ARE NOT IN THE LANE ABI
// -------------------------------------------
// `hot` and `t3_mode` are not periods, so the sweep cannot carry them. The CPU
// defaults are DEFAULT_HOT = 0.7 (:31) and DEFAULT_T3_MODE = 0 (:32), and with
// mode 0 `alpha_t3` (:269) is `2 / (2 + (period - 1) / 2)`. `period` is swept.
//
// SHAPE — ONE THREAD PER COLUMN, BARS ASCENDING
// ---------------------------------------------
// NINE carried scalars: six T3 cascade stages (t0..t5), two deviation EMAs
// (ema0, ema1) and the correction `corr`, plus the `seeded_ema` flag. Every one
// of them is reset by a non-finite bar (:373-386), so the recurrence restarts
// mid-series and no scan reformulation is available. One thread per column,
// bars ascending.
//
// ARITHMETIC — THE ROUNDING COUNT IS THE SPECIFICATION
// ----------------------------------------------------
//   t0 += alpha_t3 * (value - t0)     (:387)  — subtract, multiply, add: THREE
//                                                roundings. NOT an fma.
//   t3_value = c1*t5 + c2*t4 + c3*t3s + c4*t2  (:394)
//                                             — four multiplies and three adds
//                                               in LEFT-TO-RIGHT order.
//   corr += c * (t3_value - corr)     (:410)  — three again.
//
// `-fmad=false` on this translation unit is what keeps nvcc from contracting
// any of them into an fma behind the source.
//
// `(ema1 - ema0 * ema0).max(0.0)` (:405) is `f64::max`, which returns the
// NON-NaN operand. `fmax` has that same rule; an `if (a > b)` chain does NOT,
// and would let a NaN survive into `variance_sq` and from there into `corr`,
// poisoning every later bar of the recurrence. `fmax` is used.
//
// f64 end to end; no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. Listed in `F64_LANE_SOURCES`, so never `--use_fast_math`.

#include <cmath>
#include <cstdint>

// ema_deviation_corrected_t3.rs:31-32
#define EDCT3_DEFAULT_HOT 0.7
#define EDCT3_DEFAULT_T3_MODE 0

__device__ __forceinline__ double edct3_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void ema_deviation_corrected_t3_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int period_i = periods[r];

    // prepare_input (:292): EmptyInputData, then AllValuesNaN when NO value is
    // finite (:298 — `all(|v| !v.is_finite())`, so an all-infinite series is
    // rejected too), then validate_params_for_len (:310): period == 0 or
    // period > len. `hot` is the finite default and mode 0 is valid, so
    // neither of those two branches can fire here.
    if (n <= 0 || period_i <= 0 || period_i > n) {
        for (int i = 0; i < n; ++i) row[i] = edct3_qnan();
        return;
    }
    {
        bool any_finite = false;
        for (int i = 0; i < n; ++i) {
            if (isfinite(data[i])) { any_finite = true; break; }
        }
        if (!any_finite) {
            for (int i = 0; i < n; ++i) row[i] = edct3_qnan();
            return;
        }
    }

    const double period = (double)period_i;

    // alpha_t3 (:269), mode 0.
    const double alpha_t3 = 2.0 / (2.0 + (period - 1.0) / 2.0);
    const double alpha_ema = 2.0 / (1.0 + period);

    // correction_variance_scale (:276): `period / period.saturating_sub(1).max(1)`.
    const int denom_i = (period_i >= 2) ? (period_i - 1) : 1;
    const double variance_scale = period / (double)denom_i;

    // compute_coefficients (:281). Written term for term, in the CPU's order.
    const double hot = EDCT3_DEFAULT_HOT;
    const double hot2 = hot * hot;
    const double hot3 = hot2 * hot;
    const double c1 = -hot3;
    const double c2 = 3.0 * hot2 + 3.0 * hot3;
    const double c3 = -6.0 * hot2 - 3.0 * hot - 3.0 * hot3;
    const double c4 = 1.0 + 3.0 * hot + hot3 + 3.0 * hot2;

    // The CPU loop starts at index 0 and resets on every non-finite bar
    // (:373); there is no warmup prefix hanging off a first-valid index, which
    // is why the row is registered F64FirstValidRule::Ignored.
    (void)first_valid;

    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3s = 0.0, t4 = 0.0, t5 = 0.0;
    double ema0 = 0.0, ema1 = 0.0, corr = 0.0;
    bool seeded_ema = false;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            t0 = 0.0; t1 = 0.0; t2 = 0.0; t3s = 0.0; t4 = 0.0; t5 = 0.0;
            ema0 = 0.0; ema1 = 0.0; corr = 0.0;
            seeded_ema = false;
            row[i] = edct3_qnan();
            continue;
        }

        t0 += alpha_t3 * (value - t0);
        t1 += alpha_t3 * (t0 - t1);
        t2 += alpha_t3 * (t1 - t2);
        t3s += alpha_t3 * (t2 - t3s);
        t4 += alpha_t3 * (t3s - t4);
        t5 += alpha_t3 * (t4 - t5);

        const double t3_value = c1 * t5 + c2 * t4 + c3 * t3s + c4 * t2;

        const double price_sq = value * value;
        if (seeded_ema) {
            ema0 += alpha_ema * (value - ema0);
            ema1 += alpha_ema * (price_sq - ema1);
        } else {
            ema0 = value;
            ema1 = price_sq;
            seeded_ema = true;
        }

        // :405 — `f64::max`, so `fmax`; see the header.
        const double variance_sq = fmax(ema1 - ema0 * ema0, 0.0) * variance_scale;
        const double v2 = (corr - t3_value) * (corr - t3_value);
        const double c = (v2 < variance_sq || v2 == 0.0) ? 0.0 : (1.0 - variance_sq / v2);
        corr += c * (t3_value - corr);

        row[i] = corr;
    }
}
