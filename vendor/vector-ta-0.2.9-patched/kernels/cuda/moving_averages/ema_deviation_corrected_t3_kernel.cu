// ema_deviation_corrected_t3 — CUDA f64 kernel.
//
// CPU REFERENCE — src/indicators/moving_averages/ema_deviation_corrected_t3.rs
// -----------------------------------------------------------------------------
//   :267 alpha_t3                    :276 correction_variance_scale
//   :281 compute_coefficients        :292 prepare_input
//   :310 validate_params_for_len     :334 ema_deviation_corrected_t3 (the entry
//                                          the brief names)
//   :353 compute_into_slices         — the loop this kernel reproduces
//
// OUTPUT / ABI AUTHORITY
// ----------------------
// `ema_deviation_corrected_t3_row_f64` is the single complete arithmetic
// authority. The preserved primary ABI selects canonical hot=0.7/mode=0 and
// emits corrected; the production pair ABI carries exact period/hot/mode rows
// and emits canonical [corrected,t3] in one launch. Neither wrapper recomputes
// a price-dependent value or delegates to the other ABI.
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

__device__ __forceinline__ void ema_deviation_corrected_t3_row_f64(
    const double* __restrict__ data,
    int n,
    int period_i,
    double hot,
    int t3_mode,
    double* __restrict__ corrected_row,
    double* __restrict__ t3_row)
{
    // prepare_input (:292): EmptyInputData, AllValuesNaN when no value is
    // finite, then the exact period/hot/mode validation.
    if (n <= 0 || period_i <= 0 || period_i > n || !isfinite(hot) ||
        (t3_mode != 0 && t3_mode != 1)) {
        for (int i = 0; i < n; ++i) {
            if (corrected_row != nullptr) corrected_row[i] = edct3_qnan();
            if (t3_row != nullptr) t3_row[i] = edct3_qnan();
        }
        return;
    }
    bool any_finite = false;
    for (int i = 0; i < n; ++i) {
        if (isfinite(data[i])) {
            any_finite = true;
            break;
        }
    }
    if (!any_finite) {
        for (int i = 0; i < n; ++i) {
            if (corrected_row != nullptr) corrected_row[i] = edct3_qnan();
            if (t3_row != nullptr) t3_row[i] = edct3_qnan();
        }
        return;
    }

    const double period = (double)period_i;
    const double alpha_t3 = (t3_mode == 0)
        ? (2.0 / (2.0 + (period - 1.0) / 2.0))
        : (2.0 / (1.0 + period));
    const double alpha_ema = 2.0 / (1.0 + period);
    const int denom_i = (period_i >= 2) ? (period_i - 1) : 1;
    const double variance_scale = period / (double)denom_i;

    // compute_coefficients (:281), term for term in scalar CPU order.
    const double hot2 = hot * hot;
    const double hot3 = hot2 * hot;
    const double c1 = -hot3;
    const double c2 = 3.0 * hot2 + 3.0 * hot3;
    const double c3 = -6.0 * hot2 - 3.0 * hot - 3.0 * hot3;
    const double c4 = 1.0 + 3.0 * hot + hot3 + 3.0 * hot2;

    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3s = 0.0, t4 = 0.0, t5 = 0.0;
    double ema0 = 0.0, ema1 = 0.0, corr = 0.0;
    bool seeded_ema = false;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            t0 = 0.0; t1 = 0.0; t2 = 0.0; t3s = 0.0; t4 = 0.0; t5 = 0.0;
            ema0 = 0.0; ema1 = 0.0; corr = 0.0;
            seeded_ema = false;
            if (corrected_row != nullptr) corrected_row[i] = edct3_qnan();
            if (t3_row != nullptr) t3_row[i] = edct3_qnan();
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

        const double variance_sq = fmax(ema1 - ema0 * ema0, 0.0) * variance_scale;
        const double v2 = (corr - t3_value) * (corr - t3_value);
        const double c = (v2 < variance_sq || v2 == 0.0)
            ? 0.0
            : (1.0 - variance_sq / v2);
        corr += c * (t3_value - corr);

        if (corrected_row != nullptr) corrected_row[i] = corr;
        if (t3_row != nullptr) t3_row[i] = t3_value;
    }
}

extern "C" __global__ void ema_deviation_corrected_t3_outputs_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    const double* __restrict__ hots,
    const int* __restrict__ t3_modes,
    int n_combos,
    double* __restrict__ corrected_out,
    double* __restrict__ t3_out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    double* __restrict__ corrected_row = corrected_out + (size_t)r * (size_t)n;
    double* __restrict__ t3_row = t3_out + (size_t)r * (size_t)n;
    ema_deviation_corrected_t3_row_f64(
        data, n, periods[r], hots[r], t3_modes[r], corrected_row, t3_row);
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
    (void)first_valid;
    double* __restrict__ corrected_row = out + (size_t)r * (size_t)n;
    ema_deviation_corrected_t3_row_f64(
        data,
        n,
        periods[r],
        EDCT3_DEFAULT_HOT,
        EDCT3_DEFAULT_T3_MODE,
        corrected_row,
        nullptr);
}
