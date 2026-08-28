#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef LINREG_NAN
#define LINREG_NAN (__int_as_float(0x7fffffff))
#endif


#ifndef LINREG_LAUNCH_BOUNDS
#define LINREG_LAUNCH_BOUNDS 256, 2
#endif


extern "C" __global__ void linreg_exclusive_prefix_y_yi_f64(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    double* __restrict__ prefix_y,
    double* __restrict__ prefix_yi
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;

    if (first_valid < 0) first_valid = 0;
    if (first_valid > series_len) first_valid = series_len;

    prefix_y[0]  = 0.0;
    prefix_yi[0] = 0.0;

    double acc_y  = 0.0;
    double acc_yi = 0.0;
    for (int t = 0; t < series_len; ++t) {
        const double v = (t < first_valid) ? 0.0 : static_cast<double>(prices[t]);
        acc_y  += v;
        acc_yi  = fma(v, static_cast<double>(t), acc_yi);
        prefix_y[t + 1]  = acc_y;
        prefix_yi[t + 1] = acc_yi;
    }
}

extern "C" __global__
__launch_bounds__(LINREG_LAUNCH_BOUNDS)
void linreg_batch_from_prefix_f64(
    const double* __restrict__ prefix_y,
    const double* __restrict__ prefix_yi,
    const int*   __restrict__ periods,
    const float* __restrict__ x_sums,
    const float* __restrict__ denom_invs,
    const float* __restrict__ inv_periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {
    const int combo = static_cast<int>(blockIdx.y);
    if (combo >= n_combos) return;

    const int period  = periods[combo];
    const int row_off = combo * series_len;

    int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);

    if (series_len <= 0) return;

    if (first_valid < 0 || first_valid >= series_len) {
        while (t < series_len) {
            out[row_off + t] = LINREG_NAN;
            t += stride;
        }
        return;
    }

    if (period <= 0 || period > series_len || (series_len - first_valid) < period) {
        while (t < series_len) {
            out[row_off + t] = LINREG_NAN;
            t += stride;
        }
        return;
    }

    const int warm = first_valid + period - 1;


    if (period == 1) {
        while (t < series_len) {
            if (t < warm) {
                out[row_off + t] = LINREG_NAN;
            } else {
                out[row_off + t] = static_cast<float>(prefix_y[t + 1] - prefix_y[t]);
            }
            t += stride;
        }
        return;
    }

    const double period_f   = static_cast<double>(period);
    const double x_sum      = static_cast<double>(x_sums[combo]);
    const double denom_inv  = static_cast<double>(denom_invs[combo]);
    const double inv_period = static_cast<double>(inv_periods[combo]);

    while (t < series_len) {
        if (t < warm) {
            out[row_off + t] = LINREG_NAN;
        } else {
            const int t1    = t + 1;
            const int start = t1 - period;
            const double sum_y  = prefix_y[t1]  - prefix_y[start];
            const double sum_yi = prefix_yi[t1] - prefix_yi[start];
            const double xy_sum = fma((period_f - static_cast<double>(t)), sum_y, sum_yi);

            const double b_num = fma(period_f, xy_sum, -x_sum * sum_y);
            const double b     = b_num * denom_inv;
            const double a     = (sum_y - b * x_sum) * inv_period;
            out[row_off + t]   = static_cast<float>(a + b * period_f);
        }
        t += stride;
    }
}


extern "C" __global__
__launch_bounds__(LINREG_LAUNCH_BOUNDS)
void linreg_batch_f32(const float* __restrict__ prices,
                      const int*   __restrict__ periods,
                      const float* __restrict__ x_sums,
                      const float* __restrict__ denom_invs,
                      const float* __restrict__ inv_periods,
                      int series_len,
                      int n_combos,
                      int first_valid,
                      float* __restrict__ out)
{
    const int stride = blockDim.x * gridDim.x;

    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += stride)
    {
        const int base   = combo * series_len;
        const int period = periods[combo];


        if (period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LINREG_NAN;
            continue;
        }

        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LINREG_NAN;
            continue;
        }

        const int    warm       = first_valid + period - 1;
        const double period_f   = static_cast<double>(period);
        const double x_sum      = static_cast<double>(x_sums[combo]);
        const double denom_inv  = static_cast<double>(denom_invs[combo]);
        const double inv_period = static_cast<double>(inv_periods[combo]);


        for (int i = 0; i < warm; ++i) out[base + i] = LINREG_NAN;


        if (period == 1) {

            for (int idx = warm; idx < series_len; ++idx) {
                out[base + idx] = prices[idx];
            }
            continue;
        }


        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const double val = static_cast<double>(prices[first_valid + k]);
            const double x   = static_cast<double>(k + 1);
            y_sum  += val;
            xy_sum  = fma(val, x, xy_sum);
        }


        double latest = static_cast<double>(prices[warm]);


        for (int idx = warm; idx < series_len; ++idx) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double a     = (y_sum - b * x_sum) * inv_period;

            out[base + idx] = static_cast<float>(a + b * period_f);

            xy_sum -= y_sum;
            const int oldest = idx - period + 1;
            y_sum  -= static_cast<double>(prices[oldest]);

            if (idx + 1 < series_len)
                latest = static_cast<double>(prices[idx + 1]);
        }
    }
}


static __device__ __forceinline__
int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

extern "C" __global__
__launch_bounds__(LINREG_LAUNCH_BOUNDS)
void linreg_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                      const int*   __restrict__ first_valids,
                                      int num_series,
                                      int series_len,
                                      int period,
                                      float x_sum_f,
                                      float denom_inv_f,
                                      float inv_period_f,
                                      float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;


    const double period_f   = static_cast<double>(period);
    const double x_sum      = static_cast<double>(x_sum_f);
    const double denom_inv  = static_cast<double>(denom_inv_f);
    const double inv_period = static_cast<double>(inv_period_f);

    for (int series = blockIdx.x * blockDim.x + threadIdx.x;
         series < num_series;
         series += stride)
    {

        if (period <= 0 || period > series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = LINREG_NAN;
            continue;
        }

        const int first_valid = first_valids[series];
        if (first_valid < 0 || first_valid >= series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = LINREG_NAN;
            continue;
        }

        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = LINREG_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;


        for (int row = 0; row < warm; ++row)
            out_tm[tm_idx(row, num_series, series)] = LINREG_NAN;


        if (period == 1) {
            for (int row = warm; row < series_len; ++row) {
                out_tm[tm_idx(row, num_series, series)] = prices_tm[tm_idx(row, num_series, series)];
            }
            continue;
        }


        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const int row   = first_valid + k;
            const double v  = static_cast<double>(prices_tm[tm_idx(row, num_series, series)]);
            const double x  = static_cast<double>(k + 1);
            y_sum  += v;
            xy_sum  = fma(v, x, xy_sum);
        }

        double latest = static_cast<double>(prices_tm[tm_idx(warm, num_series, series)]);

        for (int row = warm; row < series_len; ++row) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double a     = (y_sum - b * x_sum) * inv_period;
            out_tm[tm_idx(row, num_series, series)] = static_cast<float>(a + b * period_f);

            xy_sum -= y_sum;
            const int oldest_row = row - period + 1;
            y_sum  -= static_cast<double>(prices_tm[tm_idx(oldest_row, num_series, series)]);

            if (row + 1 < series_len)
                latest = static_cast<double>(prices_tm[tm_idx(row + 1, num_series, series)]);
        }
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/moving_averages/linreg.rs
//   * linreg_prepare (:~230) — first_valid = first non-NaN of the source.
//   * linreg_with_kernel (:248) — warm = first + period, so the NaN prefix runs
//     one bar PAST the first computed bar. The CPU then writes
//     out[first + period - 1] inside that prefix; alloc_with_nan_prefix filled
//     it with NaN first and the loop overwrites it, so the observable series
//     starts at first + period - 1. Reproduced literally: fill to warm, then
//     let the loop start at first + period - 1 and overwrite.
//   * linreg_scalar (:334) — the arithmetic reproduced below.
//
// The x_sum / x2_sum closed forms are INTEGER expressions cast to f64 on the
// CPU. Computed here in 64-bit integers and cast once, so a large period cannot
// lose the low bits to an f64 intermediate.
//
// ROUNDING COUNT. The CPU line is
//     out[idx] = xy_sum * xy_coeff + y_sum * y_coeff;
// — two multiplies and one add, THREE roundings, no mul_add. -fmad=false is
// what stops nvcc contracting the second multiply into the add.
//
// Sequential: y_sum and xy_sum carry across bars. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_linreg() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__
void neoethos_linreg_f64(const double* __restrict__ prices,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const double QNAN = nef_qnan_linreg();

    const int period = periods[r];
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    {
        const long long w = (long long)first_valid + (long long)period;
        const int warm = w > (long long)n ? n : (int)w;
        for (int i = 0; i < warm; ++i) row[i] = QNAN;
        for (int i = warm; i < n; ++i) row[i] = QNAN;
    }

    if (first_valid + period - 1 >= n) return;

    const double period_f = (double)period;
    const long long p = (long long)period;
    const double x_sum  = (double)((p * (p + 1)) / 2);
    const double x2_sum = (double)((p * (p + 1) * (2 * p + 1)) / 6);
    const double denom_inv = 1.0 / (period_f * x2_sum - x_sum * x_sum);
    const double inv_period = 1.0 / period_f;
    const double b_scale = period_f - x_sum * inv_period;
    const double xy_coeff = period_f * denom_inv * b_scale;
    const double y_coeff = inv_period - x_sum * denom_inv * b_scale;

    double y_sum = 0.0;
    double xy_sum = 0.0;
    {
        int k = 1;
        for (int i = first_valid; i < first_valid + period - 1; ++i) {
            const double v = prices[i];
            y_sum += v;
            xy_sum += (double)k * v;
            ++k;
        }
    }

    int idx = first_valid + period - 1;
    int old_idx = first_valid;
    while (idx < n) {
        const double new_val = prices[idx];
        y_sum += new_val;
        xy_sum += new_val * period_f;

        row[idx] = xy_sum * xy_coeff + y_sum * y_coeff;

        xy_sum -= y_sum;
        y_sum -= prices[old_idx];

        ++idx;
        ++old_idx;
    }
}



// ===========================================================================
// S1 f64 LANE  --  linreg
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/moving_averages/linreg.rs -- `linreg_scalar` (:334), `linreg_prepare` (:216), `linreg_with_kernel` (:249)
//
// PERIOD-BASED via `compute_ma_batch`.
//
// THE COEFFICIENTS ARE PRE-FOLDED BY THE CPU AND ARE REPRODUCED IN THAT EXACT
// FOLDING, not re-derived from the textbook least-squares formula:
//   x_sum   = p(p+1)/2          (integer, then converted)
//   x2_sum  = p(p+1)(2p+1)/6    (integer, then converted)
//   denom_inv = 1 / (p*x2_sum - x_sum*x_sum)
//   b_scale   = p - x_sum * (1/p)
//   xy_coeff  = p * denom_inv * b_scale
//   y_coeff   = 1/p - x_sum * denom_inv * b_scale
// Every one of those products is a separate rounding and the order is the
// CPU's. The integer forms are computed in 64-bit integers before conversion,
// as the CPU does -- `p(p+1)(2p+1)/6` in double would round for large p.
//
// THE SLIDE IS ORDER-CRITICAL: after emitting, the CPU does
// `xy_sum -= y_sum` and THEN `y_sum -= data[old_idx]`. Swapping those two
// lines changes every subsequent value. And the entering bar is folded in as
// `xy_sum += new_val * period_f` BEFORE the emit.
//
// The emit is `xy_sum * xy_coeff + y_sum * y_coeff` -- two multiplies and an
// add, THREE roundings, deliberately not `fma`.
//
// WARMUP: `alloc_with_nan_prefix(len, first + period)`, but the compute writes
// from `first + period - 1`, so the first emitted bar is `first + period - 1`
// and the prefix value at that index is overwritten. Both facts matter: the
// prefix length and the first written index are NOT the same number here.
// ===========================================================================

#ifndef NEO_S1_QNAN_DEFINED
#define NEO_S1_QNAN_DEFINED
// The f32 kernels in this crate spell NaN `__int_as_float(0x7fc00000)`. That is
// a 32-bit pattern; widening it is a value change, not a cast. This is the f64
// quiet-NaN pattern, stated once per translation unit.
__device__ __forceinline__ double neo_s1_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}
__device__ __forceinline__ bool neo_s1_isnan(double x) { return x != x; }
#endif

extern "C" __global__ void neoethos_linreg_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int period = periods[r];

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + period;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();

    const double period_f = (double)period;
    const long long p = (long long)period;
    const double x_sum  = (double)((p * (p + 1)) / 2);
    const double x2_sum = (double)((p * (p + 1) * (2 * p + 1)) / 6);
    const double denom_inv = 1.0 / (period_f * x2_sum - x_sum * x_sum);
    const double inv_period = 1.0 / period_f;
    const double b_scale = period_f - x_sum * inv_period;
    const double xy_coeff = period_f * denom_inv * b_scale;
    const double y_coeff = inv_period - x_sum * denom_inv * b_scale;

    double y_sum = 0.0;
    double xy_sum = 0.0;
    {
        int k = 1;
        for (int j = first_valid; j < first_valid + period - 1; ++j) {
            const double v = prices[j];
            y_sum += v;
            xy_sum += (double)k * v;
            ++k;
        }
    }

    int idx = first_valid + period - 1;
    int old_idx = first_valid;
    while (idx < n) {
        const double new_val = prices[idx];
        y_sum += new_val;
        xy_sum += new_val * period_f;

        row[idx] = xy_sum * xy_coeff + y_sum * y_coeff;

        xy_sum -= y_sum;
        y_sum -= prices[old_idx];

        ++idx;
        ++old_idx;
    }
}
