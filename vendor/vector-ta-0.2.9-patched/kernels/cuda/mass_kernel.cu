#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mass_nan() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ float2 two_sum_f32(float a, float b) {
    float s = a + b;
    float z = s - a;
    float e = (a - (s - z)) + (b - z);
    return make_float2(s, e);
}


__device__ __forceinline__ float2 two_diff_f32(float a, float b) {
    float s = a - b;
    float z = s - a;
    float e = (a - (s - z)) - (b + z);
    return make_float2(s, e);
}


__device__ __forceinline__ float ds_diff_to_f32(const float2 A, const float2 B) {
    float2 d  = two_diff_f32(A.x, B.x);
    float2 s1 = two_sum_f32(d.x, A.y - B.y);
    float2 s2 = two_sum_f32(s1.x, d.y + s1.y);
    return s2.x + s2.y;
}

extern "C" __global__ void mass_build_prefix_one_series_ds_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    float2* __restrict__ prefix_ratio_ds,
    int* __restrict__ prefix_nan)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0 || first_valid < 0 || first_valid >= len) return;

    prefix_ratio_ds[0] = make_float2(0.0f, 0.0f);
    prefix_nan[0] = 0;

    const float alpha = 2.0f / 10.0f;
    const float inv_alpha = 1.0f - alpha;
    float ema1 = high[first_valid] - low[first_valid];
    float ema2 = ema1;
    const int start_ema2 = first_valid + 8;
    const int start_ratio = first_valid + 16;
    float acc_hi = 0.0f;
    float acc_lo = 0.0f;

    for (int i = 0; i < len; ++i) {
        if (i < first_valid) {
            prefix_ratio_ds[i + 1] = make_float2(acc_hi, acc_lo);
            prefix_nan[i + 1] = prefix_nan[i];
            continue;
        }

        const float hl = high[i] - low[i];
        ema1 = fmaf(alpha, hl, inv_alpha * ema1);
        if (i == start_ema2) {
            ema2 = ema1;
        }

        float ratio = mass_nan();
        if (i >= start_ema2) {
            ema2 = fmaf(alpha, ema1, inv_alpha * ema2);
            if (i >= start_ratio) {
                ratio = ema1 / ema2;
            }
        }

        const bool is_nan = !isfinite(ratio);
        if (!is_nan) {
            float2 s = two_sum_f32(acc_hi, ratio);
            float2 s2 = two_sum_f32(s.x, acc_lo);
            float2 s3 = two_sum_f32(s2.x, s.y + s2.y);
            acc_hi = s3.x;
            acc_lo = s3.y;
            prefix_nan[i + 1] = prefix_nan[i];
        } else {
            prefix_nan[i + 1] = prefix_nan[i] + 1;
        }
        prefix_ratio_ds[i + 1] = make_float2(acc_hi, acc_lo);
    }
}


extern "C" __global__ void mass_batch_f32(
    const float2* __restrict__ prefix_ratio_ds,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    int n_combos,
    float*        __restrict__ out
) {
    const int row = blockIdx.y;
    if (row >= n_combos) return;

    const int period = periods[row];
    if (period <= 0) return;

    const int warm = first_valid + 16 + period - 1;
    const int row_off = row * len;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    int t = t0;
    int start = t + 1 - period;
    while (t < len) {
        float out_val = mass_nan();
        if (t >= warm) {
            const int p1 = t + 1;
            const int bad = prefix_nan[p1] - prefix_nan[start];
            if (bad == 0) {
                const float2 a = prefix_ratio_ds[p1];
                const float2 b = prefix_ratio_ds[start];
                out_val = ds_diff_to_f32(a, b);
            }
        }
        out[row_off + t] = out_val;
        t     += stride;
        start += stride;
    }
}


extern "C" __global__ void mass_many_series_one_param_time_major_f32(
    const double* __restrict__ prefix_ratio_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float*        __restrict__ out_tm
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + 16 + period - 1;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    int t   = t0;

    while (t < series_len) {
        const int idx = t * num_series + series;
        float out_val = mass_nan();
        if (t >= warm) {
            const int start = (t + 1 - period) * num_series + series;
            const int bad = prefix_nan_tm[idx + 1] - prefix_nan_tm[start];
            if (bad == 0) {
                const double sum = prefix_ratio_tm[idx + 1] - prefix_ratio_tm[start];
                out_val = static_cast<float>(sum);
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — mass (mass index)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/mass.rs:353 `mass_scalar`.
 *
 * FIXED WINDOW BOUND, REFUSED NOT TRUNCATED. `sum_ratio` is a running
 * add-and-subtract over a ring of `period` EMA ratios (mass.rs:416-418), and a
 * running total cannot be recovered by recomputation without changing its
 * accumulation order. The ring therefore lives in a per-thread local array of
 * NEO_MASS_MAX_PERIOD, exactly the contract `MFI_MAX_PERIOD` already carries
 * in this lane: a larger period is REFUSED BY NAME by the host wrapper, never
 * silently truncated and never moved to the CPU.
 *
 * The EMA constants are the CPU's: ALPHA = 2.0/10.0 written as a division, not
 * as the decimal 0.2 — `2.0/10.0` and `0.2` happen to round to the same
 * double, but the crate spells it as the ratio and so does this.
 *
 * The double-EMA seeding at mass.rs:395-401 is NOT a loop iteration: at
 * i == first + 8 the code updates ema1, then sets ema2 = ema1, and only THEN
 * runs one ema2 update. Starting ema2 from zero or from hl would shift every
 * later bar. Reproduced literally.
 *
 * `__int_as_float(0x7f...)` in the f32 file is an f32 NaN bit pattern.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Must match NEO_MASS_MAX_PERIOD in neoethos_f64_wrapper.rs. */
#define NEO_MASS_MAX_PERIOD 512

extern "C" __global__
void mass_neo_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
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

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (period <= 0 || period > NEO_MASS_MAX_PERIOD || period > len ||
        first_valid < 0 || first_valid >= len) return;

    const double ALPHA = 2.0 / 10.0;
    const double INV_ALPHA = 1.0 - ALPHA;

    const int start_ema2  = first_valid + 8;
    const int start_ratio = first_valid + 16;
    const int start_out   = start_ratio + (period - 1);
    if (start_ema2 >= len) return;

    double ring[NEO_MASS_MAX_PERIOD];
    for (int t = 0; t < period; ++t) ring[t] = 0.0;
    int ring_index = 0;
    double sum_ratio = 0.0;

    double ema1 = high[first_valid] - low[first_valid];
    double ema2 = ema1;

    int i = first_valid;
    while (i < start_ema2) {
        const double hl = high[i] - low[i];
        ema1 = fma(ema1, INV_ALPHA, hl * ALPHA);
        i += 1;
    }
    {   // mass.rs:395-401 — ema2 is re-seeded FROM ema1 here, then stepped once
        const double hl = high[i] - low[i];
        ema1 = fma(ema1, INV_ALPHA, hl * ALPHA);
        ema2 = ema1;
        ema2 = fma(ema2, INV_ALPHA, ema1 * ALPHA);
        i += 1;
    }
    while (i < start_ratio && i < len) {
        const double hl = high[i] - low[i];
        ema1 = fma(ema1, INV_ALPHA, hl * ALPHA);
        ema2 = fma(ema2, INV_ALPHA, ema1 * ALPHA);
        i += 1;
    }
    while (i < start_out && i < len) {
        const double hl = high[i] - low[i];
        ema1 = fma(ema1, INV_ALPHA, hl * ALPHA);
        ema2 = fma(ema2, INV_ALPHA, ema1 * ALPHA);

        const double ratio = ema1 / ema2;
        sum_ratio -= ring[ring_index];
        ring[ring_index] = ratio;
        sum_ratio += ratio;

        ring_index += 1;
        if (ring_index == period) ring_index = 0;
        i += 1;
    }
    while (i < len) {
        const double hl = high[i] - low[i];
        ema1 = fma(ema1, INV_ALPHA, hl * ALPHA);
        ema2 = fma(ema2, INV_ALPHA, ema1 * ALPHA);

        const double ratio = ema1 / ema2;
        sum_ratio -= ring[ring_index];
        ring[ring_index] = ratio;
        sum_ratio += ratio;

        ring_index += 1;
        if (ring_index == period) ring_index = 0;

        o[i] = sum_ratio;
        i += 1;
    }
}
