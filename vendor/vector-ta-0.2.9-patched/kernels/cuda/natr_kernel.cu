#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }

static constexpr double NATR_TA_EPSILON = 1.0e-14;

__device__ __forceinline__ double natr_true_range_f32(
    float high, float low, float previous_close) {
    const double high_value = (double)high;
    const double low_value = (double)low;
    const double previous_close_value = (double)previous_close;
    double greatest = high_value - low_value;
    const double high_distance = fabs(previous_close_value - high_value);
    if (high_distance > greatest) greatest = high_distance;
    const double low_distance = fabs(previous_close_value - low_value);
    if (low_distance > greatest) greatest = low_distance;
    return greatest;
}

__device__ __forceinline__ double natr_true_range_f64(
    double high, double low, double previous_close) {
    double greatest = high - low;
    const double high_distance = fabs(previous_close - high);
    if (high_distance > greatest) greatest = high_distance;
    const double low_distance = fabs(previous_close - low);
    if (low_distance > greatest) greatest = low_distance;
    return greatest;
}

__device__ __forceinline__ double natr_wilder_step_f64(
    double previous, double true_range, int period) {
    double next = previous;
    next *= (double)(period - 1);
    next += true_range;
    next /= (double)period;
    return next;
}

__device__ __forceinline__ float natr_output_f32(double atr, float close, int period) {
    if (period <= 1) return (float)atr;
    const double close_value = (double)close;
    if (close_value > -NATR_TA_EPSILON && close_value < NATR_TA_EPSILON) return 0.0f;
    return (float)((atr / close_value) * 100.0);
}

__device__ __forceinline__ float safe_scale_100_over_close(float c) {
    const double close_value = (double)c;
    if (close_value > -NATR_TA_EPSILON && close_value < NATR_TA_EPSILON) return 0.0f;
    return (float)(100.0 / close_value);
}

extern "C" __global__ void natr_tr_from_hlc_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int first_valid,
    float* __restrict__ tr_out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len) return;
    if (t < first_valid) {
        tr_out[t] = 0.0f;
        return;
    }
    const float hi = high[t];
    const float lo = low[t];
    if (t == first_valid) {
        tr_out[t] = 0.0f;
        return;
    }
    tr_out[t] = (float)natr_true_range_f32(hi, lo, close[t - 1]);
}


extern "C" __global__ void natr_make_inv_close100(
    const float* __restrict__ close, int len, float* __restrict__ inv_close100)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < len) {
        inv_close100[t] = safe_scale_100_over_close(close[t]);
    }
}


extern "C" __global__ void natr_batch_f32(
    const float* __restrict__ tr,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;

    const int period = periods[combo];
    const int base = combo * series_len;
    if (period <= 0 || first_valid < 0 || first_valid >= series_len ||
        (series_len - first_valid) <= period) {
        for (int idx = 0; idx < series_len; ++idx) out[base + idx] = dev_nan();
        return;
    }

    const int warm = first_valid + period;
    for (int idx = 0; idx < warm; ++idx) out[base + idx] = dev_nan();

    double sum = 0.0;
    for (int idx = first_valid + 1; idx <= warm; ++idx) sum += (double)tr[idx];
    double atr = sum / (double)period;
    out[base + warm] = natr_output_f32(atr, close[warm], period);

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = natr_wilder_step_f64(atr, trv, period);
        out[base + t] = natr_output_f32(atr, close[t], period);
    }
}


extern "C" __global__ void natr_batch_f32_with_inv(
    const float* __restrict__ tr,
    const float* __restrict__ inv_close100,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;

    const int period = periods[combo];
    const int base = combo * series_len;
    if (period <= 0 || first_valid < 0 || first_valid >= series_len ||
        (series_len - first_valid) <= period) {
        for (int idx = 0; idx < series_len; ++idx) out[base + idx] = dev_nan();
        return;
    }

    const int warm = first_valid + period;
    for (int idx = 0; idx < warm; ++idx) out[base + idx] = dev_nan();

    double sum = 0.0;
    for (int idx = first_valid + 1; idx <= warm; ++idx) sum += (double)tr[idx];
    double atr = sum / (double)period;
    out[base + warm] = period <= 1 ? (float)atr : (float)(atr * (double)inv_close100[warm]);

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = natr_wilder_step_f64(atr, trv, period);
        out[base + t] = period <= 1 ? (float)atr : (float)(atr * (double)inv_close100[t]);
    }
}


extern "C" __global__ void natr_batch_warp_io_f32(
    const float* __restrict__ tr,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    if (blockDim.x != 32) return;
    const int combo = blockIdx.x;
    const int lane = threadIdx.x & (warpSize - 1);
    if (combo >= n_combos || lane != 0) return;

    const int period = periods[combo];
    const int base = combo * series_len;
    if (period <= 0 || first_valid < 0 || first_valid >= series_len ||
        (series_len - first_valid) <= period) {
        for (int idx = 0; idx < series_len; ++idx) out[base + idx] = dev_nan();
        return;
    }

    const int warm = first_valid + period;
    for (int idx = 0; idx < warm; ++idx) out[base + idx] = dev_nan();

    double sum = 0.0;
    for (int idx = first_valid + 1; idx <= warm; ++idx) sum += (double)tr[idx];
    double atr = sum / (double)period;
    out[base + warm] = natr_output_f32(atr, close[warm], period);
    for (int idx = warm + 1; idx < series_len; ++idx) {
        atr = natr_wilder_step_f64(atr, (double)tr[idx], period);
        out[base + idx] = natr_output_f32(atr, close[idx], period);
    }
}

extern "C" __global__ void natr_batch_warp_io_f32_with_inv(
    const float* __restrict__ tr,
    const float* __restrict__ inv_close100,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    if (blockDim.x != 32) return;
    const int combo = blockIdx.x;
    const int lane = threadIdx.x & (warpSize - 1);
    if (combo >= n_combos || lane != 0) return;

    const int period = periods[combo];
    const int base = combo * series_len;
    if (period <= 0 || first_valid < 0 || first_valid >= series_len ||
        (series_len - first_valid) <= period) {
        for (int idx = 0; idx < series_len; ++idx) out[base + idx] = dev_nan();
        return;
    }

    const int warm = first_valid + period;
    for (int idx = 0; idx < warm; ++idx) out[base + idx] = dev_nan();

    double sum = 0.0;
    for (int idx = first_valid + 1; idx <= warm; ++idx) sum += (double)tr[idx];
    double atr = sum / (double)period;
    out[base + warm] = period <= 1 ? (float)atr : (float)(atr * (double)inv_close100[warm]);
    for (int idx = warm + 1; idx < series_len; ++idx) {
        atr = natr_wilder_step_f64(atr, (double)tr[idx], period);
        out[base + idx] = period <= 1 ? (float)atr : (float)(atr * (double)inv_close100[idx]);
    }
}


extern "C" __global__ void natr_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int period,
    int num_series,
    int series_len,
    const int*   __restrict__ first_valids,
    float*       __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int fv = first_valids[s];

        if (fv < 0 || fv >= series_len) {
            for (int t = lane; t < series_len; t += warpSize) {
                out_tm[t * stride + s] = dev_nan();
            }
            continue;
        }

        const int warm = fv + period;
        if (warm >= series_len) {
            for (int t = lane; t < series_len; t += warpSize) {
                out_tm[t * stride + s] = dev_nan();
            }
            continue;
        }

        for (int t = lane; t < warm; t += warpSize) {
            out_tm[t * stride + s] = dev_nan();
        }

        if (lane == 0) {
            double sum = 0.0;
            for (int t = fv + 1; t <= warm; ++t) {
                sum += natr_true_range_f32(
                    high_tm[t * stride + s],
                    low_tm[t * stride + s],
                    close_tm[(t - 1) * stride + s]);
            }
            double atr = sum / (double)period;
            out_tm[warm * stride + s] =
                natr_output_f32(atr, close_tm[warm * stride + s], period);

            for (int t = warm + 1; t < series_len; ++t) {
                const double true_range = natr_true_range_f32(
                    high_tm[t * stride + s],
                    low_tm[t * stride + s],
                    close_tm[(t - 1) * stride + s]);
                atr = natr_wilder_step_f64(atr, true_range, period);
                out_tm[t * stride + s] =
                    natr_output_f32(atr, close_tm[t * stride + s], period);
            }
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — natr
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/natr.rs:288 `natr_scalar`.
 *
 * TA-Lib authority requires input-order seed accumulation and the Wilder
 * multiply/add/divide recurrence as three separate operations.  True Range
 * starts from high-low and conditionally replaces it with the previous-close
 * distances in that exact order.
 *
 * first_valid rule: natr.rs:226-235 takes fh.max(fl).max(fc) — the MAX of
 * three INDEPENDENT first-non-NaN scans, NOT the first index at which all
 * three are simultaneously non-NaN. Registered as HlcMaxOfIndependentFirsts.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void natr_neo_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        const double* __restrict__ close,
                        int series_len,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int period = periods[combo];

    if (period <= 0 || period >= len || first_valid < 0 || first_valid >= len ||
        (len - first_valid) <= period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warm_end = first_valid + period;
    for (int i = 0; i < warm_end && i < len; ++i) o[i] = NEO_F64_NAN;

    double sum_tr = 0.0;
    for (int i = first_valid + 1; i <= warm_end; ++i) {
        sum_tr += natr_true_range_f64(high[i], low[i], close[i - 1]);
    }

    double atr = sum_tr / (double)period;
    const double c_we = close[warm_end];
    if (period <= 1) {
        o[warm_end] = atr;
    } else if (c_we > -NATR_TA_EPSILON && c_we < NATR_TA_EPSILON) {
        o[warm_end] = 0.0;
    } else {
        o[warm_end] = (atr / c_we) * 100.0;
    }

    for (int idx = warm_end + 1; idx < len; ++idx) {
        const double tr = natr_true_range_f64(high[idx], low[idx], close[idx - 1]);
        atr = natr_wilder_step_f64(atr, tr, period);
        const double cv = close[idx];
        if (period <= 1) {
            o[idx] = atr;
        } else if (cv > -NATR_TA_EPSILON && cv < NATR_TA_EPSILON) {
            o[idx] = 0.0;
        } else {
            o[idx] = (atr / cv) * 100.0;
        }
    }
}
