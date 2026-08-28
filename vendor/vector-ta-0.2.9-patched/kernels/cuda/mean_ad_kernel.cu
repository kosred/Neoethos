#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static __forceinline__ __device__ float qnan32() {
    return __int_as_float(0x7fffffff);
}


static __forceinline__ __device__ float warp_sum(float v) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) v += __shfl_down_sync(mask, v, ofs);
    return v;
}


static __forceinline__ __device__ void kahan_add(float &sum, float &c, float x) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}


extern "C" __global__
void mean_ad_batch_f32(const float* __restrict__ prices,
                       const int*   __restrict__ periods,
                       const int*   __restrict__ warm_indices,
                       int first_valid,
                       int series_len,
                       int n_combos,
                       int max_period,
                       float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;


    const int lane             = threadIdx.x & 31;
    const int warp_in_block    = threadIdx.x >> 5;
    const int warps_per_block  = blockDim.x >> 5;
    const int warp_global      = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps_grid = gridDim.x * warps_per_block;


    extern __shared__ float s_ring[];
    float* ring = s_ring + (size_t)warp_in_block * (size_t)max_period;

    for (int combo = warp_global; combo < n_combos; combo += total_warps_grid) {

        const int period = periods[combo];
        if (period <= 0) continue;

        const int warm = warm_indices[combo];
        const size_t base = (size_t)combo * (size_t)series_len;


        const int nan_end = (warm < series_len ? warm : series_len);
        for (int t = lane; t < nan_end; t += 32) {
            out[base + t] = qnan32();
        }

        if (warm >= series_len) continue;
        if (first_valid + period > series_len) continue;


        float partial = 0.0f;
        for (int k = lane; k < period; k += 32) {
            partial += prices[first_valid + k];
        }
        float sum = warp_sum(partial);
        sum = __shfl_sync(__activemask(), sum, 0);

        const float inv_p = 1.0f / (float)period;
        float sma = sum * inv_p;


        if (lane == 0) {
            int head = 0;
            float residual_sum = 0.0f, c_res = 0.0f;
            float c_sum = 0.0f;

            const int start_t = first_valid + period - 1;
            const int fill_end = min(start_t + period - 1, series_len - 1);


            for (int t = start_t; t <= fill_end; ++t) {
                const float r = fabsf(prices[t] - sma);
                ring[head++] = r; if (head == period) head = 0;
                kahan_add(residual_sum, c_res, r);

                if (t + 1 < series_len) {
                    const float in_next  = prices[t + 1];
                    const float out_prev = prices[t + 1 - period];
                    kahan_add(sum, c_sum,  in_next);
                    kahan_add(sum, c_sum, -out_prev);
                    sma = sum * inv_p;
                }
            }


            out[base + warm] = residual_sum * inv_p;


            int t = start_t + period;
            int idx = head;
            while (t < series_len) {
                const float r   = fabsf(prices[t] - sma);
                const float old = ring[idx];
                ring[idx] = r;
                idx += 1; if (idx == period) idx = 0;


                kahan_add(residual_sum, c_res,  r);
                kahan_add(residual_sum, c_res, -old);

                out[base + t] = residual_sum * inv_p;

                if (t + 1 < series_len) {
                    const float in_next  = prices[t + 1];
                    const float out_prev = prices[t + 1 - period];
                    kahan_add(sum, c_sum,  in_next);
                    kahan_add(sum, c_sum, -out_prev);
                    sma = sum * inv_p;
                }
                ++t;
            }
        }
    }
}


#ifndef SMALL_PERIOD_MAX
#define SMALL_PERIOD_MAX 64
#endif

extern "C" __global__
void mean_ad_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                       int period,
                                       int num_series,
                                       int series_len,
                                       const int* __restrict__ first_valids,
                                       float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    const int first = first_valids[series_idx];
    if (first < 0 || first >= series_len) return;

    const int warm = first + 2 * period - 2;
    const int stride = num_series;


    const int nan_end = (warm < series_len ? warm : series_len);
    for (int t = 0; t < nan_end; ++t) {
        out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = qnan32();
    }
    if (warm >= series_len) return;


    float sum = 0.0f, c_sum = 0.0f;
    size_t p = (size_t)first * (size_t)stride + (size_t)series_idx;
    for (int k = 0; k < period; ++k) {
        kahan_add(sum, c_sum, prices_tm[p]);
        p += (size_t)stride;
    }
    const float inv_p = 1.0f / (float)period;
    float sma = sum * inv_p;


    float residual_sum = 0.0f, c_res = 0.0f;
    int head = 0;


    float ring_reg[SMALL_PERIOD_MAX];
    float* ring = nullptr;

    if (period <= SMALL_PERIOD_MAX) {
        const int start_t = first + period - 1;
        const int fill_end = min(start_t + period - 1, series_len - 1);
        for (int t = start_t; t <= fill_end; ++t) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r = fabsf(price_t - sma);
            ring_reg[head++] = r; if (head == period) head = 0;
            kahan_add(residual_sum, c_res, r);

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
        }
        out_tm[(size_t)warm * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

        int t = first + 2 * period - 1;
        int idx = head;
        while (t < series_len) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r   = fabsf(price_t - sma);
            const float old = ring_reg[idx];
            ring_reg[idx] = r;
            idx += 1; if (idx == period) idx = 0;

            kahan_add(residual_sum, c_res,  r);
            kahan_add(residual_sum, c_res, -old);

            out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
            ++t;
        }
    } else {

        extern __shared__ float smem[];
        ring = smem + (size_t)threadIdx.x * (size_t)period;

        const int start_t = first + period - 1;
        const int fill_end = min(start_t + period - 1, series_len - 1);
        for (int t = start_t; t <= fill_end; ++t) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r = fabsf(price_t - sma);
            ring[head++] = r; if (head == period) head = 0;
            kahan_add(residual_sum, c_res, r);

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
        }
        out_tm[(size_t)warm * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

        int t = first + 2 * period - 1;
        int idx = head;
        while (t < series_len) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r   = fabsf(price_t - sma);
            const float old = ring[idx];
            ring[idx] = r;
            idx += 1; if (idx == period) idx = 0;

            kahan_add(residual_sum, c_res,  r);
            kahan_add(residual_sum, c_res, -old);

            out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
            ++t;
        }
    }
}

// ===========================================================================
// S3 f64 LANE — mean_ad
// ===========================================================================
// Reference: src/indicators/mean_ad.rs
//   `mean_ad_with_kernel` (:186) — first_valid + the three Err branches
//   `mean_ad_scalar` (:240)      — warmup_end = first + (period<<1) - 2
//   `mean_ad_row_scalar` (:683)  — the arithmetic
// Batch default period 5 (`cpu_batch.rs`, `get_usize_param("mean_ad", …, 5)`),
// source close.
//
// THE RING BUFFER, AND WHY THERE IS NONE HERE
//
// The CPU keeps `residual_buffer: Vec<f64>` of length `period` so it can
// subtract the residual that falls out of the window:
//   residual_sum += residual - old      (mean_ad.rs:727)
// A per-thread array of length `period` is not available on the device —
// `period` is a runtime value, so it would have to be a global scratch
// allocation the whole lane does not have.
//
// It is not needed. `old` is `|data[t-period] - sma_{t-period}|`, and
// `sma_j` is produced by ONE deterministic accumulator recurrence seeded at
// `first`:  sum += data[j+1] - data[j+1-period];  sma = sum * inv_p.
// A SECOND accumulator seeded identically and advanced one step per bar of the
// main loop is, at every step, bit-for-bit the state the first accumulator had
// `period` bars earlier — same seed, same operations, same order. So the lagged
// sma is recomputed rather than remembered, and `old` is exact, not
// approximately equal. That is the whole trick: two scalars instead of an
// array, with no change to the arithmetic.
//
// ROUNDING COUNT. `residual_sum += residual - old` is TWO roundings
// (the subtract, then the add) and is written that way below — not as
// `residual_sum = residual_sum + residual - old`, which associates
// differently. `sum += data[a] - data[b]` likewise.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_mean_ad_batch_f64(
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
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    // `mean_ad_scalar` :260 — first + period > n emits an all-NaN series.
    // Unreachable given the check above, kept because the CPU has it.
    if (first_valid + period > n) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    // `alloc_with_nan_prefix(n, warmup_end.min(n))`, warmup_end :265.
    const int warmup_end = first_valid + (period << 1) - 2;
    const int prefix = warmup_end < n ? warmup_end : n;
    for (int i = 0; i < prefix; ++i) row[i] = neo_s3_qnan();
    // Bars at and after `warmup_end` that the row loop never reaches stay NaN
    // only if the loop writes them; it writes every one from `first_output` on.

    const double inv_p = 1.0 / (double)period;

    // Seed: sum over [first, first+period). CPU accumulates ascending.
    double seed = 0.0;
    for (int i = first_valid; i < first_valid + period; ++i) seed += data[i];

    double sumA = seed;
    double smaA = sumA * inv_p;   // sma valid at bar `start_t`
    double sumB = seed;
    double smaB = sumB * inv_p;   // the same state, replayed `period` bars late

    const int start_t = first_valid + period - 1;
    int fill_end = start_t + period - 1;
    if (fill_end > n - 1) fill_end = n - 1;

    double residual_sum = 0.0;
    for (int t = start_t; t <= fill_end; ++t) {
        const double residual = fabs(data[t] - smaA);
        residual_sum += residual;
        if (t + 1 < n) {
            sumA += data[t + 1] - data[t + 1 - period];
            smaA = sumA * inv_p;
        }
    }

    const int first_output = first_valid + (period << 1) - 2;
    if (first_output < n) row[first_output] = residual_sum * inv_p;

    for (int t = start_t + period; t < n; ++t) {
        const double residual = fabs(data[t] - smaA);
        const int j = t - period;                       // the bar leaving the window
        const double old = fabs(data[j] - smaB);
        residual_sum += residual - old;
        row[t] = residual_sum * inv_p;

        if (t + 1 < n) {
            sumA += data[t + 1] - data[t + 1 - period];
            smaA = sumA * inv_p;
        }
        // Replay the identical update one lag behind. j+1-period >= first_valid
        // holds for every t in this loop, so the read is in range.
        sumB += data[j + 1] - data[j + 1 - period];
        smaB = sumB * inv_p;
    }
}
