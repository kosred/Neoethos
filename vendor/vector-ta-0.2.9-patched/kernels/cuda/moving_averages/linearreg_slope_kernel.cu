#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef LRS_NAN
#define LRS_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LRS_LAUNCH_BOUNDS
#define LRS_LAUNCH_BOUNDS 256, 2
#endif


extern "C" __global__ void linearreg_slope_exclusive_prefix_y_yi_f64(
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
__launch_bounds__(LRS_LAUNCH_BOUNDS)
void linearreg_slope_batch_from_prefix_f64(
    const double* __restrict__ prefix_y,
    const double* __restrict__ prefix_yi,
    const int*   __restrict__ periods,
    const float* __restrict__ x_sums,
    const float* __restrict__ denom_invs,
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
            out[row_off + t] = LRS_NAN;
            t += stride;
        }
        return;
    }

    if (period < 2 || period > series_len || (series_len - first_valid) < period) {
        while (t < series_len) {
            out[row_off + t] = LRS_NAN;
            t += stride;
        }
        return;
    }

    const int warm = first_valid + period - 1;

    const double period_f  = static_cast<double>(period);
    const double x_sum     = static_cast<double>(x_sums[combo]);
    const double denom_inv = static_cast<double>(denom_invs[combo]);

    while (t < series_len) {
        if (t < warm) {
            out[row_off + t] = LRS_NAN;
        } else {
            const int t1    = t + 1;
            const int start = t1 - period;
            const double sum_y  = prefix_y[t1]  - prefix_y[start];
            const double sum_yi = prefix_yi[t1] - prefix_yi[start];
            const double xy_sum = fma((period_f - static_cast<double>(t)), sum_y, sum_yi);
            const double b_num = fma(period_f, xy_sum, -x_sum * sum_y);
            out[row_off + t] = static_cast<float>(b_num * denom_inv);
        }
        t += stride;
    }
}

extern "C" __global__
__launch_bounds__(LRS_LAUNCH_BOUNDS)
void linearreg_slope_batch_f32(const float* __restrict__ prices,
                               const int*   __restrict__ periods,
                               const float* __restrict__ x_sums,
                               const float* __restrict__ denom_invs,
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

        if (period < 2 || period > series_len || first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LRS_NAN;
            continue;
        }
        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LRS_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        const double period_f  = static_cast<double>(period);
        const double x_sum     = static_cast<double>(x_sums[combo]);
        const double denom_inv = static_cast<double>(denom_invs[combo]);

        for (int i = 0; i < warm; ++i) out[base + i] = LRS_NAN;


        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const double v = static_cast<double>(prices[first_valid + k]);
            const double x = static_cast<double>(k + 1);
            y_sum  += v;
            xy_sum  = fma(v, x, xy_sum);
        }

        double latest = static_cast<double>(prices[warm]);
        for (int idx = warm; idx < series_len; ++idx) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            out[base + idx] = static_cast<float>(b);

            xy_sum -= y_sum;
            const int oldest = idx - period + 1;
            y_sum  -= static_cast<double>(prices[oldest]);
            if (idx + 1 < series_len)
                latest = static_cast<double>(prices[idx + 1]);
        }
    }
}

static __device__ __forceinline__
int tm_idx_lrs(int row, int num_series, int series) {
    return row * num_series + series;
}

extern "C" __global__
__launch_bounds__(LRS_LAUNCH_BOUNDS)
void linearreg_slope_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                               const int*   __restrict__ first_valids,
                                               int num_series,
                                               int series_len,
                                               int period,
                                               float x_sum_f,
                                               float denom_inv_f,
                                               float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;
    const double period_f  = static_cast<double>(period);
    const double x_sum     = static_cast<double>(x_sum_f);
    const double denom_inv = static_cast<double>(denom_inv_f);

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_series; s += stride) {
        if (period < 2 || period > series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;
            continue;
        }
        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;
            continue;
        }
        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        for (int row = 0; row < warm; ++row)
            out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;

        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const int r = first_valid + k;
            const double v = static_cast<double>(prices_tm[tm_idx_lrs(r, num_series, s)]);
            const double x = static_cast<double>(k + 1);
            y_sum  += v;
            xy_sum  = fma(v, x, xy_sum);
        }
        double latest = static_cast<double>(prices_tm[tm_idx_lrs(warm, num_series, s)]);
        for (int row = warm; row < series_len; ++row) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            out_tm[tm_idx_lrs(row, num_series, s)] = static_cast<float>(b);

            xy_sum -= y_sum;
            const int oldest_row = row - period + 1;
            y_sum  -= static_cast<double>(prices_tm[tm_idx_lrs(oldest_row, num_series, s)]);
            if (row + 1 < series_len)
                latest = static_cast<double>(prices_tm[tm_idx_lrs(row + 1, num_series, s)]);
        }
    }
}


// ===========================================================================
// S3 f64 LANE — linearreg_slope
// ===========================================================================
// Reference: src/indicators/linearreg_slope.rs
//   `linearreg_slope_with_kernel` (:220) — first_valid, Err branches, warmup
//   `linearreg_slope_scalar` (:268)        — the NON-finite path (Kahan)
//   `linearreg_slope_scalar_finite` (:358) — the finite path
// Batch default period 14, source close.
//
// TWO ALGORITHMS, NOT ONE. `linearreg_slope_scalar` branches on
// `data[first..].iter().all(|v| v.is_finite())`. The finite path carries plain
// running sums AND REBUILDS THEM EVERY 16 BARS (`if (i & 15) == 0`, :403); the
// non-finite path carries KAHAN-COMPENSATED sums and never rebuilds. Those are
// different numbers, not different speeds. Both are transcribed and selected by
// the same test.
//
// EPSILONS — both are f64 constants read from the f64 CPU, not scaled f32 ones:
//   `denom.abs() < f64::EPSILON`  → 2.220446049250313e-16 (:373)
//   `b.abs() <= 1.1e-8` → 0.0     → the slope deadband (:393, :330)
// The f32 kernels above cannot express either: f32 epsilon is 1.19e-7, which is
// LARGER than the 1.1e-8 deadband, so in f32 the deadband is below the noise it
// is supposed to suppress.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ __forceinline__ void neo_s3_kahan_add(double* sum, double* c, double x) {
    const double y = x - *c;
    const double t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

extern "C" __global__ void neoethos_linearreg_slope_batch_f64(
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
    const int period = periods[r];

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period < 2) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();
    if (warm >= n) return;

    const double p  = (double)period;
    const double x  = 0.5 * p * (p + 1.0);
    const double x2 = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0;
    const double denom = p * x2 - x * x;
    if (fabs(denom) < 2.220446049250313e-16) {   // f64::EPSILON
        for (int i = warm; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }
    const double bd   = 1.0 / denom;
    const double p_bd = p * bd;
    const double x_bd = x * bd;

    bool all_finite = true;
    for (int i = first_valid; i < n; ++i) {
        if (!isfinite(data[i])) { all_finite = false; break; }
    }

    if (all_finite) {
        // ---- linearreg_slope_scalar_finite (:358) ----
        double y = 0.0, xy = 0.0;
        for (int j = 0; j < period; ++j) {
            const double v = data[first_valid + j];
            y  += v;
            xy += v * (double)(j + 1);
        }

        int i = warm;
        for (;;) {
            const double b = xy * p_bd - y * x_bd;
            row[i] = (fabs(b) <= 1.1e-8) ? 0.0 : b;
            if (i + 1 == n) break;

            const double y_in  = data[i + 1];
            const double y_out = data[i + 1 - period];
            const double prev_y = y;
            y  = prev_y + y_in - y_out;
            xy = (xy - prev_y) + p * y_in;
            i += 1;
            // The CPU rebuilds both sums on every index divisible by 16. This
            // is arithmetic, not hygiene: it discards the drift accumulated
            // since the last rebuild, so the emitted value at i changes.
            if ((i & 15) == 0) {
                y = 0.0;
                xy = 0.0;
                const int start = i + 1 - period;
                for (int j = 0; j < period; ++j) {
                    const double v = data[start + j];
                    y  += v;
                    xy += v * (double)(j + 1);
                }
            }
        }
        return;
    }

    // ---- linearreg_slope_scalar, the Kahan path (:268) ----
    double y = 0.0, y_c = 0.0, xy = 0.0, xy_c = 0.0;
    for (int j = 0; j < period - 1; ++j) {
        const double v = data[first_valid + j];
        neo_s3_kahan_add(&y, &y_c, v);
        neo_s3_kahan_add(&xy, &xy_c, v * (double)(j + 1));
    }

    int in_new = first_valid + period - 1;
    int in_old = first_valid;
    for (; in_new < n; ++in_new, ++in_old) {
        const double v = data[in_new];
        neo_s3_kahan_add(&y, &y_c, v);
        neo_s3_kahan_add(&xy, &xy_c, v * p);
        const double b = xy * p_bd - y * x_bd;
        row[in_new] = (fabs(b) <= 1.1e-8) ? 0.0 : b;
        // The CPU performs these two roll-offs only when another bar follows
        // (`while in_new.add(1) < end`); the final bar falls into the tail
        // block, which omits them. Same guard here.
        if (in_new + 1 < n) {
            neo_s3_kahan_add(&xy, &xy_c, -y);
            neo_s3_kahan_add(&y, &y_c, -data[in_old]);
        }
    }
}
