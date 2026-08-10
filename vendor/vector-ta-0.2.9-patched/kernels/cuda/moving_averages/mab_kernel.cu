#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

__device__ __forceinline__ float sma_from_prefix_f32(
    const double* __restrict__ pref_sum,
    const int* __restrict__ pref_nan,
    int t,
    int period
) {
    const int t1 = t + 1;
    const int t0 = t + 1 - period;
    if ((pref_nan[t1] - pref_nan[t0]) != 0) return qnan32();
    const double sum = pref_sum[t1] - pref_sum[t0];
    return (float)(sum / (double)period);
}

extern "C" __global__ void mab_build_prefix_single_f32(
    const float* __restrict__ prices,
    int len,
    double* __restrict__ pref_sum,
    int* __restrict__ pref_nan
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    pref_sum[0] = 0.0;
    pref_nan[0] = 0;
    double acc_s = 0.0;
    int acc_nan = 0;
    for (int i = 0; i < len; ++i) {
        const float x = prices[i];
        if (isnan(x)) {
            ++acc_nan;
        } else {
            acc_s += (double)x;
        }
        pref_sum[i + 1] = acc_s;
        pref_nan[i + 1] = acc_nan;
    }
}


extern "C" __global__ void mab_batch_from_prefix_sma_f32(
    const double* __restrict__ pref_close_sum,
    const int* __restrict__ pref_close_nan,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int len,
    int first_valid,
    int rows,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const int fast_period = fast_periods[row];
    const int slow_period = slow_periods[row];
    const float devup = devups[row];
    const float devdn = devdns[row];

    if (fast_period <= 0 || slow_period <= 0 || len <= 0) return;

    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const int row_off = row * len;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[row_off + t] = nanf;
        out_middle[row_off + t] = nanf;
        out_lower[row_off + t] = nanf;
    }
    if (warm >= len) return;

    const float inv_fast = 1.0f / (float)fast_period;

    float sumsq = 0.0f;
    const int start0 = (warm + 1) - fast_period;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start0 + k;
        const float fm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, idx, fast_period);
        const float sm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, idx, slow_period);
        const float d = fm - sm;
        sumsq = fmaf(d, d, sumsq);
    }

    float dev = sqrtf(sumsq * inv_fast);
    float fm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, warm, fast_period);
    float sm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, warm, slow_period);
    out_middle[row_off + warm] = fm;
    out_upper[row_off + warm] = sm + devup * dev;
    out_lower[row_off + warm] = sm - devdn * dev;

    for (int i = warm + 1; i < len; ++i) {
        const int old_idx = i - fast_period;

        const float fn = sma_from_prefix_f32(pref_close_sum, pref_close_nan, i, fast_period);
        const float sn = sma_from_prefix_f32(pref_close_sum, pref_close_nan, i, slow_period);
        const float fo = sma_from_prefix_f32(pref_close_sum, pref_close_nan, old_idx, fast_period);
        const float so = sma_from_prefix_f32(pref_close_sum, pref_close_nan, old_idx, slow_period);

        const float newd = fn - sn;
        const float oldd = fo - so;
        sumsq = (sumsq + newd * newd) - oldd * oldd;
        if (!isnan(sumsq) && sumsq < 0.0f) sumsq = 0.0f;
        dev = sqrtf(sumsq * inv_fast);

        out_middle[row_off + i] = fn;
        out_upper[row_off + i] = sn + devup * dev;
        out_lower[row_off + i] = sn - devdn * dev;
    }
}


extern "C" __global__ void mab_dev_from_ma_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    int fast_period,
    int first_valid,
    int len,
    float* __restrict__ dev_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0 || fast_period <= 0) return;

    const int first_output = first_valid + max(fast_period, 0) + fast_period - 1;


    for (int t = 0; t < min(first_output, len); ++t) {
        dev_out[t] = qnan32();
    }
    if (first_output >= len) return;

    const int start0 = first_output + 1 - fast_period;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start0 + k;
        const double d = (double)fast[idx] - (double)slow[idx];
        sumsq += d * d;
    }
    dev_out[first_output] = (float)sqrt(sumsq / (double)fast_period);

    for (int i = first_output + 1; i < len; ++i) {
        const int old_idx = i - fast_period;
        const double oldd = (double)fast[old_idx] - (double)slow[old_idx];
        const double newd = (double)fast[i] - (double)slow[i];
        sumsq += newd * newd - oldd * oldd;
        dev_out[i] = (float)sqrt(sumsq / (double)fast_period);
    }
}


extern "C" __global__ void mab_apply_dev_shared_ma_batch_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    const float* __restrict__ dev,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int rows,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const int row_off = row * len;
    const float devup = devups[row];
    const float devdn = devdns[row];

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = qnan32();
    while (t < len) {
        float u = nanf, m = nanf, l = nanf;
        if (t >= warm) {
            const float d = dev[t];
            const float sm = slow[t];
            m = fast[t];
            u = sm + devup * d;
            l = sm - devdn * d;
        }
        out_upper[row_off + t]  = u;
        out_middle[row_off + t] = m;
        out_lower[row_off + t]  = l;
        t += stride;
    }
}


extern "C" __global__ void mab_single_row_from_ma_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    float devup,
    float devdn,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[t] = nanf;
        out_middle[t] = nanf;
        out_lower[t] = nanf;
    }
    if (warm >= len) return;


    int start = (warm + 1) - fast_period;
    if (start < 0) start = 0;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start + k;
        const double d = (double)fast[idx] - (double)slow[idx];
        sumsq += d * d;
    }
    const float dev0 = (float)sqrt(sumsq / (double)fast_period);
    out_middle[warm] = fast[warm];
    out_upper[warm] = slow[warm] + devup * dev0;
    out_lower[warm] = slow[warm] - devdn * dev0;

    for (int i = warm + 1; i < len; ++i) {
        const int old_idx = i - fast_period;
        const double oldd = (double)fast[old_idx] - (double)slow[old_idx];
        const double newd = (double)fast[i] - (double)slow[i];
        sumsq += newd * newd - oldd * oldd;
        const float dev = (float)sqrt(sumsq / (double)fast_period);
        out_middle[i] = fast[i];
        out_upper[i] = slow[i] + devup * dev;
        out_lower[i] = slow[i] - devdn * dev;
    }
}


extern "C" __global__ void mab_many_series_one_param_time_major_f32(
    const float* __restrict__ fast_tm,
    const float* __restrict__ slow_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int fast_period,
    int slow_period,
    float devup,
    float devdn,
    float* __restrict__ out_upper_tm,
    float* __restrict__ out_middle_tm,
    float* __restrict__ out_lower_tm
) {
    const int s = blockIdx.y;
    if (s >= cols) return;
    const int fv = first_valids[s];
    const int warm = fv + max(fast_period, slow_period) + fast_period - 1;

    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int stride = cols;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, rows); ++t) {
        const int idx = t * stride + s;
        out_upper_tm[idx] = nanf;
        out_middle_tm[idx] = nanf;
        out_lower_tm[idx] = nanf;
    }
    if (warm >= rows) return;

    int start = (warm + 1) - fast_period;
    if (start < 0) start = 0;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = (start + k) * stride + s;
        const double d = (double)fast_tm[idx] - (double)slow_tm[idx];
        sumsq += d * d;
    }
    {
        const int i = warm;
        const int idx = i * stride + s;
        const float dev = (float)sqrt(sumsq / (double)fast_period);
        out_middle_tm[idx] = fast_tm[idx];
        out_upper_tm[idx] = slow_tm[idx] + devup * dev;
        out_lower_tm[idx] = slow_tm[idx] - devdn * dev;
    }

    for (int i = warm + 1; i < rows; ++i) {
        const int old_idx = (i - fast_period) * stride + s;
        const int new_idx = i * stride + s;
        const double oldd = (double)fast_tm[old_idx] - (double)slow_tm[old_idx];
        const double newd = (double)fast_tm[new_idx] - (double)slow_tm[new_idx];
        sumsq += newd * newd - oldd * oldd;
        const float dev = (float)sqrt(sumsq / (double)fast_period);
        out_middle_tm[new_idx] = fast_tm[new_idx];
        out_upper_tm[new_idx] = slow_tm[new_idx] + devup * dev;
        out_lower_tm[new_idx] = slow_tm[new_idx] - devdn * dev;
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

__device__ __forceinline__ double qnan32_f64() {
    return __longlong_as_double(0x7fffffffffffffffULL);
}
__device__ __forceinline__ double sma_from_prefix_f64(
    const double* __restrict__ pref_sum,
    const int* __restrict__ pref_nan,
    int t,
    int period
) {
    const int t1 = t + 1;
    const int t0 = t + 1 - period;
    if ((pref_nan[t1] - pref_nan[t0]) != 0) return qnan32_f64();
    const double sum = pref_sum[t1] - pref_sum[t0];
    return (double)(sum / (double)period);
}
extern "C" __global__ void mab_build_prefix_single_f64(
    const double* __restrict__ prices,
    int len,
    double* __restrict__ pref_sum,
    int* __restrict__ pref_nan
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    pref_sum[0] = 0.0;
    pref_nan[0] = 0;
    double acc_s = 0.0;
    int acc_nan = 0;
    for (int i = 0; i < len; ++i) {
        const double x = prices[i];
        if (isnan(x)) {
            ++acc_nan;
        } else {
            acc_s += (double)x;
        }
        pref_sum[i + 1] = acc_s;
        pref_nan[i + 1] = acc_nan;
    }
}
extern "C" __global__ void mab_batch_from_prefix_sma_f64(
    const double* __restrict__ pref_close_sum,
    const int* __restrict__ pref_close_nan,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const double* __restrict__ devups,
    const double* __restrict__ devdns,
    int len,
    int first_valid,
    int rows,
    double* __restrict__ out_upper,
    double* __restrict__ out_middle,
    double* __restrict__ out_lower
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const int fast_period = fast_periods[row];
    const int slow_period = slow_periods[row];
    const double devup = devups[row];
    const double devdn = devdns[row];

    if (fast_period <= 0 || slow_period <= 0 || len <= 0) return;

    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const int row_off = row * len;
    const double nan = qnan32_f64();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[row_off + t] = nan;
        out_middle[row_off + t] = nan;
        out_lower[row_off + t] = nan;
    }
    if (warm >= len) return;

    const double inv_fast = 1.0 / (double)fast_period;

    double sumsq = 0.0;
    const int start0 = (warm + 1) - fast_period;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start0 + k;
        const double fm = sma_from_prefix_f64(pref_close_sum, pref_close_nan, idx, fast_period);
        const double sm = sma_from_prefix_f64(pref_close_sum, pref_close_nan, idx, slow_period);
        const double d = fm - sm;
        sumsq = fma(d, d, sumsq);
    }

    double dev = sqrt(sumsq * inv_fast);
    double fm = sma_from_prefix_f64(pref_close_sum, pref_close_nan, warm, fast_period);
    double sm = sma_from_prefix_f64(pref_close_sum, pref_close_nan, warm, slow_period);
    out_middle[row_off + warm] = fm;
    out_upper[row_off + warm] = sm + devup * dev;
    out_lower[row_off + warm] = sm - devdn * dev;

    for (int i = warm + 1; i < len; ++i) {
        const int old_idx = i - fast_period;

        const double fn = sma_from_prefix_f64(pref_close_sum, pref_close_nan, i, fast_period);
        const double sn = sma_from_prefix_f64(pref_close_sum, pref_close_nan, i, slow_period);
        const double fo = sma_from_prefix_f64(pref_close_sum, pref_close_nan, old_idx, fast_period);
        const double so = sma_from_prefix_f64(pref_close_sum, pref_close_nan, old_idx, slow_period);

        const double newd = fn - sn;
        const double oldd = fo - so;
        sumsq = (sumsq + newd * newd) - oldd * oldd;
        if (!isnan(sumsq) && sumsq < 0.0) sumsq = 0.0;
        dev = sqrt(sumsq * inv_fast);

        out_middle[row_off + i] = fn;
        out_upper[row_off + i] = sn + devup * dev;
        out_lower[row_off + i] = sn - devdn * dev;
    }
}
extern "C" __global__ void mab_dev_from_ma_f64(
    const double* __restrict__ fast,
    const double* __restrict__ slow,
    int fast_period,
    int first_valid,
    int len,
    double* __restrict__ dev_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0 || fast_period <= 0) return;

    const int first_output = first_valid + max(fast_period, 0) + fast_period - 1;


    for (int t = 0; t < min(first_output, len); ++t) {
        dev_out[t] = qnan32_f64();
    }
    if (first_output >= len) return;

    const int start0 = first_output + 1 - fast_period;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start0 + k;
        const double d = (double)fast[idx] - (double)slow[idx];
        sumsq += d * d;
    }
    dev_out[first_output] = (double)sqrt(sumsq / (double)fast_period);

    for (int i = first_output + 1; i < len; ++i) {
        const int old_idx = i - fast_period;
        const double oldd = (double)fast[old_idx] - (double)slow[old_idx];
        const double newd = (double)fast[i] - (double)slow[i];
        sumsq += newd * newd - oldd * oldd;
        dev_out[i] = (double)sqrt(sumsq / (double)fast_period);
    }
}
extern "C" __global__ void mab_apply_dev_shared_ma_batch_f64(
    const double* __restrict__ fast,
    const double* __restrict__ slow,
    const double* __restrict__ dev,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    const double* __restrict__ devups,
    const double* __restrict__ devdns,
    int rows,
    double* __restrict__ out_upper,
    double* __restrict__ out_middle,
    double* __restrict__ out_lower
) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const int row_off = row * len;
    const double devup = devups[row];
    const double devdn = devdns[row];

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const double nan = qnan32_f64();
    while (t < len) {
        double u = nan, m = nan, l = nan;
        if (t >= warm) {
            const double d = dev[t];
            const double sm = slow[t];
            m = fast[t];
            u = sm + devup * d;
            l = sm - devdn * d;
        }
        out_upper[row_off + t]  = u;
        out_middle[row_off + t] = m;
        out_lower[row_off + t]  = l;
        t += stride;
    }
}
extern "C" __global__ void mab_single_row_from_ma_f64(
    const double* __restrict__ fast,
    const double* __restrict__ slow,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    double devup,
    double devdn,
    double* __restrict__ out_upper,
    double* __restrict__ out_middle,
    double* __restrict__ out_lower
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const double nan = qnan32_f64();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[t] = nan;
        out_middle[t] = nan;
        out_lower[t] = nan;
    }
    if (warm >= len) return;


    int start = (warm + 1) - fast_period;
    if (start < 0) start = 0;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start + k;
        const double d = (double)fast[idx] - (double)slow[idx];
        sumsq += d * d;
    }
    const double dev0 = (double)sqrt(sumsq / (double)fast_period);
    out_middle[warm] = fast[warm];
    out_upper[warm] = slow[warm] + devup * dev0;
    out_lower[warm] = slow[warm] - devdn * dev0;

    for (int i = warm + 1; i < len; ++i) {
        const int old_idx = i - fast_period;
        const double oldd = (double)fast[old_idx] - (double)slow[old_idx];
        const double newd = (double)fast[i] - (double)slow[i];
        sumsq += newd * newd - oldd * oldd;
        const double dev = (double)sqrt(sumsq / (double)fast_period);
        out_middle[i] = fast[i];
        out_upper[i] = slow[i] + devup * dev;
        out_lower[i] = slow[i] - devdn * dev;
    }
}
extern "C" __global__ void mab_many_series_one_param_time_major_f64(
    const double* __restrict__ fast_tm,
    const double* __restrict__ slow_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int fast_period,
    int slow_period,
    double devup,
    double devdn,
    double* __restrict__ out_upper_tm,
    double* __restrict__ out_middle_tm,
    double* __restrict__ out_lower_tm
) {
    const int s = blockIdx.y;
    if (s >= cols) return;
    const int fv = first_valids[s];
    const int warm = fv + max(fast_period, slow_period) + fast_period - 1;

    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int stride = cols;
    const double nan = qnan32_f64();

    for (int t = 0; t < min(warm, rows); ++t) {
        const int idx = t * stride + s;
        out_upper_tm[idx] = nan;
        out_middle_tm[idx] = nan;
        out_lower_tm[idx] = nan;
    }
    if (warm >= rows) return;

    int start = (warm + 1) - fast_period;
    if (start < 0) start = 0;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = (start + k) * stride + s;
        const double d = (double)fast_tm[idx] - (double)slow_tm[idx];
        sumsq += d * d;
    }
    {
        const int i = warm;
        const int idx = i * stride + s;
        const double dev = (double)sqrt(sumsq / (double)fast_period);
        out_middle_tm[idx] = fast_tm[idx];
        out_upper_tm[idx] = slow_tm[idx] + devup * dev;
        out_lower_tm[idx] = slow_tm[idx] - devdn * dev;
    }

    for (int i = warm + 1; i < rows; ++i) {
        const int old_idx = (i - fast_period) * stride + s;
        const int new_idx = i * stride + s;
        const double oldd = (double)fast_tm[old_idx] - (double)slow_tm[old_idx];
        const double newd = (double)fast_tm[new_idx] - (double)slow_tm[new_idx];
        sumsq += newd * newd - oldd * oldd;
        const double dev = (double)sqrt(sumsq / (double)fast_period);
        out_middle_tm[new_idx] = fast_tm[new_idx];
        out_upper_tm[new_idx] = slow_tm[new_idx] + devup * dev;
        out_lower_tm[new_idx] = slow_tm[new_idx] - devdn * dev;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3   (mab)
 *
 * CPU reference: `mab_scalar` (src/indicators/mab.rs:698), reached because
 *   `mab_prepare2` (:252) maps `Kernel::Auto` to `Kernel::Scalar` OUTRIGHT
 *   (:282-285). There is no auto-detected AVX path for this indicator, so the
 *   scalar association is the whole oracle and the 1-ULP seed question that
 *   dogged `wilders`/`vwap` does not arise here.
 *
 * Column: output_id "value" -> `out.upperband` (cpu_batch.rs:15316-15321).
 *
 * PERIOD-INVARIANT: `compute_mab_batch` reads `fast_period` (10),
 *   `slow_period` (50), `devup` (1.0), `devdn` (1.0), `fast_ma_type` ("sma")
 *   and `slow_ma_type` ("sma") (cpu_batch.rs:15294-15299) and NEVER `period`.
 *
 * Input: ONE price series -- `extract_slice_input("mab", req.data, "close")`
 *   (cpu_batch.rs:15291) -> F64InputKind::CloseSlice.
 *
 * FIRST-VALID: `mab_prepare2` :269-272 is `position(|x| !x.is_nan())` over the
 *   single series -- F64FirstValidRule::AllInputsNonNan.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Three accumulators are carried
 *   subtract-then-add (the two SMA sums and `sum_sq`), and `sum_sq` at bar i
 *   is a function of its own value at bar i-1, so there is no bar-parallel
 *   form that preserves the rounding.
 *
 * Roundings, counted against the CPU lines:
 *   `sum += *dp.add(first + k);`                (sma.rs:334)  plain add
 *   `sum += *dp.add(i) - *dp.add(i - period);`  (sma.rs:341)  ONE sub then add
 *   `*op.add(i) = sum * inv;`                   (sma.rs:342)  multiply by 1/p
 *   `sum_sq += diff * diff;`                    (mab.rs:718)  NO fma
 *   `sum_sq += new * new - old * old;`          (mab.rs:734)  NO fma, sub first
 *   `let dev = (sum_sq / fast_period as f64).sqrt();` (:735)
 *   `upper[i] = slow_ma[i] + devup * dev;`      (:738)
 *   Not one of these is a `mul_add` on the CPU, so not one of them is an
 *   `fma` here. Writing `fma(devup, dev, slow_ma)` would drop a rounding the
 *   reference performs.
 *
 * NaN semantics: `mab_scalar` contains no max/min at all -- rule 4 does not
 *   bite on this column, and a NaN in the window propagates exactly as the
 *   CPU lets it.
 *
 * The two MIXED stages in this file (`mab_build_prefix_single_f32` writing a
 *   `double*`, `mab_batch_from_prefix_sma_f32` taking one) are left standing
 *   for the f32 wrappers that call them. This lane never crosses that
 *   boundary: no `float*` appears in this entry point, and the resolution
 *   table -- not a name suffix -- is what routes `F64Kernel::Mab` here.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:15294-15297. */
#define NEO_MAB_FAST_PERIOD 10
#define NEO_MAB_SLOW_PERIOD 50
#define NEO_MAB_DEVUP       1.0
#define NEO_MAB_DEVDN       1.0

extern "C" __global__
void mab_neo_batch_f64(const double* __restrict__ prices,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods; /* period-invariant -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    fast  = NEO_MAB_FAST_PERIOD;
    const int    slow  = NEO_MAB_SLOW_PERIOD;
    const double devup = NEO_MAB_DEVUP;

    /* :262 -- InvalidPeriod. */
    if (fast > n || slow > n) return;

    const int first = first_valid;
    if (first < 0 || first >= n) return;

    const int need       = (fast > slow ? fast : slow);
    const int need_total = need + fast - 1;          /* :275 */
    if (n - first < need_total) return;              /* :276 NotEnoughValidData */

    const int warmup       = first + need_total - 1; /* :286 */
    const int first_output = warmup + 1;             /* mab.rs:401 passes warmup+1 */
    if (first_output >= n) return;

    /* --- fast/slow SMA, exactly `sma_scalar` (sma.rs:317) ----------------- */
    const double inv_fast = 1.0 / (double)fast;
    const double inv_slow = 1.0 / (double)slow;

    double sum_fast = 0.0;
    for (int k = 0; k < fast; ++k) sum_fast += prices[first + k];
    double sum_slow = 0.0;
    for (int k = 0; k < slow; ++k) sum_slow += prices[first + k];

    /* `diff[i] = fast_ma[i] - slow_ma[i]` is needed again `fast` bars later,
     * so the row keeps a ring that deep. `fast_period` is a CPU DEFAULT here,
     * not a swept parameter (this indicator is period-invariant), so the ring
     * is sized at exactly that default and no caller-supplied number can reach
     * it. */
    double diff_ring[NEO_MAB_FAST_PERIOD];
    for (int k = 0; k < fast; ++k) diff_ring[k] = 0.0;

    /* `start_idx` (mab.rs:709) = first_output - fast + 1. */
    const int start_idx = (first_output >= fast) ? (first_output - fast + 1) : 0;

    double sum_sq = 0.0;
    int    ring_pos = 0;

    for (int i = first; i < n; ++i) {
        /* Advance the two SMA sums to bar i in CPU order. */
        if (i >= first + fast) sum_fast += prices[i] - prices[i - fast];
        if (i >= first + slow) sum_slow += prices[i] - prices[i - slow];

        const int have_fast = (i >= first + fast - 1);
        const int have_slow = (i >= first + slow - 1);
        if (!have_fast || !have_slow) continue;

        const double fast_ma = sum_fast * inv_fast;
        const double slow_ma = sum_slow * inv_slow;
        const double diff    = fast_ma - slow_ma;

        /* Seed window [start_idx, start_idx + fast) -- mab.rs:716-719. */
        if (i >= start_idx && i < start_idx + fast) {
            sum_sq += diff * diff;
        } else if (i > first_output) {
            /* mab.rs:731-734: subtract the diff `fast` bars back, add this one. */
            const double old = diff_ring[(i - fast) % fast];
            sum_sq += diff * diff - old * old;
        }

        diff_ring[i % fast] = diff;

        if (i >= first_output) {
            const double dev = sqrt(sum_sq / (double)fast);
            o[i] = slow_ma + devup * dev;
        }
    }
}
