#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef TSF_NAN
#define TSF_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef TSF_LAUNCH_BOUNDS
#define TSF_LAUNCH_BOUNDS 256, 2
#endif


extern "C" __global__ void tsf_exclusive_prefix_y_yi_f64(
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
__launch_bounds__(TSF_LAUNCH_BOUNDS)
void tsf_batch_from_prefix_f64(
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
            out[row_off + t] = TSF_NAN;
            t += stride;
        }
        return;
    }

    if (period <= 1 || period > series_len || (series_len - first_valid) < period) {
        while (t < series_len) {
            out[row_off + t] = TSF_NAN;
            t += stride;
        }
        return;
    }

    const int warm = first_valid + period - 1;

    const double period_f   = static_cast<double>(period);
    const double x_sum      = static_cast<double>(x_sums[combo]);
    const double denom_inv  = static_cast<double>(denom_invs[combo]);
    const double inv_period = static_cast<double>(inv_periods[combo]);
    const double period_next = period_f + 1.0;

    while (t < series_len) {
        if (t < warm) {
            out[row_off + t] = TSF_NAN;
        } else {
            const int t1    = t + 1;
            const int start = t1 - period;
            const double sum_y  = prefix_y[t1]  - prefix_y[start];
            const double sum_yi = prefix_yi[t1] - prefix_yi[start];
            const double xy_sum = fma((period_f - static_cast<double>(t)), sum_y, sum_yi);

            const double b_num = fma(period_f, xy_sum, -x_sum * sum_y);
            const double b     = b_num * denom_inv;
            const double a     = (sum_y - b * x_sum) * inv_period;
            out[row_off + t]   = static_cast<float>(a + b * period_next);
        }
        t += stride;
    }
}


extern "C" __global__
__launch_bounds__(TSF_LAUNCH_BOUNDS)
void tsf_batch_f32(const float* __restrict__ prices,
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


        if (period <= 1 || period > series_len || first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = TSF_NAN;
            continue;
        }

        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int i = 0; i < series_len; ++i) out[base + i] = TSF_NAN;
            continue;
        }

        const int    warm       = first_valid + period - 1;
        const double period_f   = static_cast<double>(period);
        const double x_sum      = static_cast<double>(x_sums[combo]);
        const double denom_inv  = static_cast<double>(denom_invs[combo]);
        const double inv_period = static_cast<double>(inv_periods[combo]);


        for (int i = 0; i < warm; ++i) out[base + i] = TSF_NAN;


        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const double val = static_cast<double>(prices[first_valid + k]);
            const double x   = static_cast<double>(k + 1);
            y_sum  += val;
            xy_sum  = fma(val, x, xy_sum);
        }


        double latest = static_cast<double>(prices[warm]);


        const double period_next = period_f + 1.0;
        for (int idx = warm; idx < series_len; ++idx) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double a     = (y_sum - b * x_sum) * inv_period;
            out[base + idx] = static_cast<float>(a + b * period_next);

            xy_sum -= y_sum;
            const int oldest = idx - period + 1;
            y_sum  -= static_cast<double>(prices[oldest]);

            if (idx + 1 < series_len) {
                latest = static_cast<double>(prices[idx + 1]);
            }
        }
    }
}


static __device__ __forceinline__
int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

extern "C" __global__
__launch_bounds__(TSF_LAUNCH_BOUNDS)
void tsf_many_series_one_param_f32(const float* __restrict__ prices_tm,
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
        if (period <= 1 || period > series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = TSF_NAN;
            continue;
        }

        const int first_valid = first_valids[series];
        if (first_valid < 0 || first_valid >= series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = TSF_NAN;
            continue;
        }

        if (series_len - first_valid < period) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = TSF_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        for (int row = 0; row < warm; ++row)
            out_tm[tm_idx(row, num_series, series)] = TSF_NAN;

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

        const double period_next = period_f + 1.0;
        for (int row = warm; row < series_len; ++row) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double a     = (y_sum - b * x_sum) * inv_period;
            out_tm[tm_idx(row, num_series, series)] = static_cast<float>(a + b * period_next);

            xy_sum -= y_sum;
            const int oldest_row = row - period + 1;
            y_sum  -= static_cast<double>(prices_tm[tm_idx(oldest_row, num_series, series)]);

            if (row + 1 < series_len)
                latest = static_cast<double>(prices_tm[tm_idx(row + 1, num_series, series)]);
        }
    }
}


// ===========================================================================
// S3 f64 LANE — tsf (Time Series Forecast)
// ===========================================================================
// Reference: src/indicators/tsf.rs
//   `tsf_with_kernel` (:260) — first_valid, Err branches,
//                              `alloc_with_nan_prefix(len, first + period - 1)`
//   `tsf_scalar` (:383)      — the arithmetic
// Batch default period 14, source close.
//
// The regression constants use x = 0..p-1 (NOT 1..p as `linearreg_slope` does),
// so `sum_x`/`sum_x2` are accumulated in the CPU's ascending loop rather than
// from the closed form: the closed form is exact for these magnitudes but the
// loop is what the reference runs, and "exact" is an argument, not a guarantee.
//
// NaN BOOKKEEPING is a COUNTER, not a comparison chain. `nan_count` tracks how
// many NaNs sit in the window; while it is non-zero the CPU emits NaN AND
// POISONS s0/s1 to NaN, then rebuilds them from scratch on the first fully
// clean window (`prev_nan != 0` branch, :463). A comparison chain that merely
// skipped NaNs would keep a stale sum and silently emit a wrong number for
// every remaining bar. Transcribed literally.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_tsf_batch_f64(
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
    const int p = periods[r];

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (p < 2) || (p > n) ||
        ((n - first_valid) < p);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + p - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();
    if (warm >= n) return;

    const double pf = (double)p;

    double sum_x = 0.0, sum_x2 = 0.0;
    for (int xi = 0; xi < p; ++xi) {
        const double xf = (double)xi;
        sum_x  += xf;
        sum_x2 += xf * xf;
    }
    const double divisor       = pf * sum_x2 - sum_x * sum_x;
    const double inv_div       = 1.0 / divisor;
    const double inv_pf        = 1.0 / pf;
    const double pf_over_div   = pf * inv_div;
    const double sumx_over_div = sum_x * inv_div;
    const double p_minus_mean_x = pf - sum_x * inv_pf;

    int base = first_valid;
    int i = base + p - 1;

    double s0 = 0.0, s1 = 0.0;
    int nan_count = 0;
    for (int j = 0; j < p; ++j) {
        const double v = data[base + j];
        if (isnan(v)) { nan_count += 1; }
        else { s0 += v; s1 += (double)j * v; }
    }

    if (nan_count == 0) {
        const double m = s1 * pf_over_div - s0 * sumx_over_div;
        row[i] = s0 * inv_pf + m * p_minus_mean_x;
    } else {
        s0 = neo_s3_qnan();
        s1 = neo_s3_qnan();
        row[i] = neo_s3_qnan();
    }

    while (i + 1 < n) {
        const double y_old = data[base];
        const double y_new = data[base + p];
        base += 1;
        i += 1;

        const int prev_nan = nan_count;
        if (isnan(y_old)) nan_count = (nan_count > 0) ? nan_count - 1 : 0;
        if (isnan(y_new)) nan_count = nan_count + 1;

        if (nan_count == 0) {
            if (prev_nan == 0) {
                const double new_s0 = s0 + (y_new - y_old);
                const double new_s1 = pf * y_new + s1 - new_s0;
                s0 = new_s0;
                s1 = new_s1;
            } else {
                double r0 = 0.0, r1 = 0.0;
                for (int j = 0; j < p; ++j) {
                    const double v = data[base + j];
                    r0 += v;
                    r1 += (double)j * v;
                }
                s0 = r0;
                s1 = r1;
            }
            const double m = s1 * pf_over_div - s0 * sumx_over_div;
            row[i] = s0 * inv_pf + m * p_minus_mean_x;
        } else {
            s0 = neo_s3_qnan();
            s1 = neo_s3_qnan();
            row[i] = neo_s3_qnan();
        }
    }
}
