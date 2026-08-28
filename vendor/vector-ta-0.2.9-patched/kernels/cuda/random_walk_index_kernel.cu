#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double nz_history_value(const double* src, int idx, int offset) {
    if (idx >= offset) {
        double value = src[idx - offset];
        if (isfinite(value)) {
            return value;
        }
    }
    return 0.0;
}

extern "C" __global__ void random_walk_index_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    int first_valid,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out_high,
    double* __restrict__ out_low
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || first_valid < 0 || first_valid >= len) {
        return;
    }

    int length = lengths[combo_idx];
    double* row_high = out_high + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_low = out_low + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_high[i] = CUDART_NAN;
        row_low[i] = CUDART_NAN;
    }

    if (length <= 0 || length > len) {
        return;
    }

    int warm = first_valid + length - 1;
    double sqrt_length = sqrt(static_cast<double>(length));
    double alpha = 1.0 / static_cast<double>(length);
    double prev_close = close[first_valid];
    double sum_tr = high[first_valid] - low[first_valid];
    double atr = CUDART_NAN;

    if (length == 1) {
        atr = sum_tr;
        double denom = atr * sqrt_length;
        if (isfinite(denom) && denom != 0.0) {
            row_high[first_valid] =
                (high[first_valid] - nz_history_value(low, first_valid, length)) / denom;
            row_low[first_valid] =
                (nz_history_value(high, first_valid, length) - low[first_valid]) / denom;
        }
    }

    for (int i = first_valid + 1; i < len; ++i) {
        double tr = fmax(
            high[i] - low[i],
            fmax(fabs(high[i] - prev_close), fabs(low[i] - prev_close))
        );

        if (i <= warm) {
            sum_tr += tr;
            if (i == warm) {
                atr = sum_tr / static_cast<double>(length);
            }
        } else {
            atr = alpha * (tr - atr) + atr;
        }

        if (i >= warm) {
            double denom = atr * sqrt_length;
            if (isfinite(denom) && denom != 0.0) {
                row_high[i] = (high[i] - nz_history_value(low, i, length)) / denom;
                row_low[i] = (nz_history_value(high, i, length) - low[i]) / denom;
            } else {
                row_high[i] = CUDART_NAN;
                row_low[i] = CUDART_NAN;
            }
        }

        prev_close = close[i];
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `random_walk_index_with_kernel`
// (src/indicators/random_walk_index.rs:463) -> `prepare` (:234) for the
// validity rules and `compute_random_walk_index_into` (:359) for the value.
// `compute_random_walk_index_14_into` (:290) is the length==14 specialisation
// and is the same arithmetic in the same order with `DEFAULT_LENGTH`
// substituted, so one implementation serves both. This kernel emits the HIGH
// series.
//
// WHICH OUTPUT, AND WHY IT IS NAMED HERE. `compute_random_walk_index_batch`
// (cpu_batch.rs:10329) accepts `output_id` "high" or "low" and REJECTS "value"
// (:10359). So a parity check must ask the CPU for "high" explicitly. Stated
// here rather than discovered later.
//
// SHAPE: one thread per column. Wilder-family recurrence -- `atr` is carried
// across bars as `atr = alpha.mul_add(tr - atr, atr)` (:441), ONE rounding,
// and `prev_close` is carried with it. The seed is a plain SUM of the first
// `length` true ranges divided by `length` (:424), not the recurrence, and the
// sum is accumulated in ascending bar order.
//
// ROUNDING COUNT. The CPU line is `alpha.mul_add(tr - atr, atr)`: one
// subtraction, then ONE fused multiply-add. Written below as
// `fma(alpha, tr - atr, atr)` -- the same two roundings. Writing it as
// `atr + alpha * (tr - atr)` would be three.
//
// NaN SEMANTICS. True range is `(h-l).max((h-prev).abs()).max((l-prev).abs())`
// (:412-414). Rust's `f64::max` returns the NON-NaN operand, so `fmax` is used
// below and NOT an if-chain -- a comparison against NaN is false, which would
// let a NaN survive into the carried `atr` and poison every later bar.
//
// `nz_history` (:259) is the other place a non-finite value is absorbed: a
// value more than `length` bars back that is not finite reads as 0.0, and so
// does an index before the start of the series. Reproduced literally.
//
// GUARD, NOT EPSILON. `denom.is_finite() && denom != 0.0` (:426) is an exact
// test, not a tolerance, and is carried across unchanged. No epsilon exists in
// this indicator on the CPU and none was invented -- inventing one here would
// emit a value where the host emits NaN.
//
// WARMUP: `first + length - 1` (:465). `length == 1` is the CPU's own special
// case (:385-409) and is kept: it seeds `atr` with the first bar's range and
// emits from `first`.
//
// f32 -> f64 audit of this file: the f32 entry points above use `fmaxf`,
// `fabsf`, `sqrtf` and `__int_as_float`. Below: `fmax`, `fabs`, `sqrt`, `fma`
// and the f64 quiet-NaN bit pattern. No f32 literal, no f32-suffixed math
// function, no fast-math intrinsic.
// ===========================================================================

static __device__ __forceinline__ double neo_rwi_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// nz_history (:259-270): 0.0 before the start, 0.0 for a non-finite value.
static __device__ __forceinline__ double neo_rwi_nz(const double* src, int idx, int offset) {
    if (idx >= offset) {
        const double v = src[idx - offset];
        return isfinite(v) ? v : 0.0;
    }
    return 0.0;
}

static __device__ __forceinline__ double neo_rwi_tr(double h, double l, double prev_close) {
    return fmax(fmax(h - l, fabs(h - prev_close)), fabs(l - prev_close));
}

extern "C" __global__ void neoethos_random_walk_index_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_rwi_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int length = periods[r];
    const int first  = first_valid;

    if (length <= 0 || length > n) return;          // prepare :241
    if (first < 0 || first >= n) return;
    if ((n - first) < length) return;               // prepare :249

    const double sqrt_length = sqrt(static_cast<double>(length));
    const double alpha       = 1.0 / static_cast<double>(length);

    double prev_close = close[first];
    double sum_tr     = high[first] - low[first];
    double atr;

    if (length == 1) {                              // :385-409
        atr = sum_tr;
        double denom = atr * sqrt_length;
        if (isfinite(denom) && denom != 0.0) {
            row[first] = (high[first] - neo_rwi_nz(low, first, length)) / denom;
        }
        for (int i = first + 1; i < n; ++i) {
            const double tr = neo_rwi_tr(high[i], low[i], prev_close);
            atr   = fma(alpha, tr - atr, atr);
            denom = atr * sqrt_length;
            row[i] = (isfinite(denom) && denom != 0.0)
                         ? (high[i] - neo_rwi_nz(low, i, length)) / denom
                         : nan_d;
            prev_close = close[i];
        }
        return;
    }

    const int warm = first + length - 1;

    for (int i = first + 1; i < warm; ++i) {        // :411-419
        sum_tr += neo_rwi_tr(high[i], low[i], prev_close);
        prev_close = close[i];
    }

    sum_tr += neo_rwi_tr(high[warm], low[warm], prev_close);   // :421-424
    atr = sum_tr / static_cast<double>(length);

    double denom = atr * sqrt_length;
    row[warm] = (isfinite(denom) && denom != 0.0)
                    ? (high[warm] - neo_rwi_nz(low, warm, length)) / denom
                    : nan_d;
    prev_close = close[warm];

    for (int i = warm + 1; i < n; ++i) {            // :437-452
        const double tr = neo_rwi_tr(high[i], low[i], prev_close);
        atr   = fma(alpha, tr - atr, atr);
        denom = atr * sqrt_length;
        row[i] = (isfinite(denom) && denom != 0.0)
                     ? (high[i] - neo_rwi_nz(low, i, length)) / denom
                     : nan_d;
        prev_close = close[i];
    }
}
