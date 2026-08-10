#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef OUT_PER_THREAD


#define OUT_PER_THREAD 8
#endif

static __device__ __forceinline__ float sqwma_weight_sum(int period) {

    double p = static_cast<double>(period);
    double sum = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0 - 1.0;
    return static_cast<float>(sum);
}

static __device__ __forceinline__ float sqwma_eval_f32(
    float P_f, float S0, float S1, float S2, float inv_weight_sum)
{

    float P2 = P_f * P_f;
    float acc = fmaf(P2, S0, fmaf(-2.0f * P_f, S1, S2));
    return acc * inv_weight_sum;
}

static __device__ __forceinline__ void sqwma_init_sums_series_major(
    const float* __restrict__ x, int t, int P, float& S0, float& S1, float& S2)
{

    S0 = 0.0f; S1 = 0.0f; S2 = 0.0f;
    const int last = P - 2;
#pragma unroll 4
    for (int r = 0; r <= last; ++r) {
        float xr = x[t - r];
        float rf = static_cast<float>(r);
        S0 = S0 + xr;
        S1 = fmaf(rf, xr, S1);
        S2 = fmaf(rf * rf, xr, S2);
    }
}

static __device__ __forceinline__ void sqwma_advance_sums_series_major(
    const float* __restrict__ x, int t, int P, float& S0, float& S1, float& S2)
{

    const int oldest_idx = t - (P - 2);
    const float x_old = x[oldest_idx];
    const float x_new = x[t + 1];

    float S1_old = S1;

    S0 = (S0 - x_old) + x_new;

    S1 = fmaf(static_cast<float>(-(P - 1)), x_old, S1_old + (S0 - x_new + x_old));

    const float Pm1 = static_cast<float>(P - 1);
    const float S0_old = S0 - x_new + x_old;
    S2 = S2 + 2.0f * S1_old + S0_old - (Pm1 * Pm1) * x_old;
}

extern "C" __global__
void sqwma_batch_f32(const float* __restrict__ prices,
                     const int*   __restrict__ periods,
                     int series_len,
                     int n_combos,
                     int first_valid,
                     float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    const int base_out = combo * series_len;


    if (period <= 1) {
        for (int t = blockIdx.x * blockDim.x + threadIdx.x;
             t < series_len;
             t += gridDim.x * blockDim.x)
        {
            out[base_out + t] = NAN;
        }
        return;
    }

    const int warm = first_valid + period + 1;
    const float inv_ws = 1.0f / sqwma_weight_sum(period);
    const float P_f    = static_cast<float>(period);


    const int warm_cap = warm < series_len ? warm : series_len;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < warm_cap;
         t += gridDim.x * blockDim.x)
    {
        out[base_out + t] = NAN;
    }
    if (warm >= series_len) return;


    const int tile_size = blockDim.x * OUT_PER_THREAD;
    for (int tile = warm + blockIdx.x * tile_size;
         tile < series_len;
         tile += gridDim.x * tile_size)
    {
        int t0 = tile + threadIdx.x * OUT_PER_THREAD;
        if (t0 >= series_len) continue;


        float S0, S1, S2;
        sqwma_init_sums_series_major(prices, t0, period, S0, S1, S2);


#pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) {
            int t = t0 + i;
            if (t >= series_len) break;

            float value = sqwma_eval_f32(P_f, S0, S1, S2, inv_ws);
            out[base_out + t] = value;


            if ((i + 1) < OUT_PER_THREAD && (t + 1) < series_len) {
                sqwma_advance_sums_series_major(prices, t, period, S0, S1, S2);
            }
        }
    }
}

extern "C" __global__
void sqwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) return;

    if (period <= 1) {
        for (int t = blockIdx.x * blockDim.x + threadIdx.x;
             t < series_len;
             t += gridDim.x * blockDim.x)
        {
            out_tm[t * num_series + series_idx] = NAN;
        }
        return;
    }

    const int warm = first_valids[series_idx] + period + 1;
    const float inv_ws = 1.0f / sqwma_weight_sum(period);
    const float P_f    = static_cast<float>(period);


    const int warm_cap = warm < series_len ? warm : series_len;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < warm_cap;
         t += gridDim.x * blockDim.x)
    {
        out_tm[t * num_series + series_idx] = NAN;
    }
    if (warm >= series_len) return;


    const int tile_size = blockDim.x * OUT_PER_THREAD;
    for (int tile = warm + blockIdx.x * tile_size;
         tile < series_len;
         tile += gridDim.x * tile_size)
    {
        int t0 = tile + threadIdx.x * OUT_PER_THREAD;
        if (t0 >= series_len) continue;


        auto load_tm = [&](int t) {
            return prices_tm[t * num_series + series_idx];
        };
        auto store_tm = [&](int t, float v) {
            out_tm[t * num_series + series_idx] = v;
        };


        float S0 = 0.f, S1 = 0.f, S2 = 0.f;
        const int last = period - 2;
#pragma unroll 4
        for (int r = 0; r <= last; ++r) {
            float xr = load_tm(t0 - r);
            float rf = static_cast<float>(r);
            S0 = S0 + xr;
            S1 = fmaf(rf, xr, S1);
            S2 = fmaf(rf * rf, xr, S2);
        }

#pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) {
            int t = t0 + i;
            if (t >= series_len) break;

            float value = sqwma_eval_f32(P_f, S0, S1, S2, inv_ws);
            store_tm(t, value);

            if ((i + 1) < OUT_PER_THREAD && (t + 1) < series_len) {

                const int oldest_t = t - (period - 2);
                const float x_old = load_tm(oldest_t);
                const float x_new = load_tm(t + 1);

                float S1_old = S1;
                const float Pm1 = static_cast<float>(period - 1);
                const float S0_old = S0;


                S0 = (S0 - x_old) + x_new;

                S1 = fmaf(-Pm1, x_old, S1_old + S0_old);

                S2 = S2 + 2.0f * S1_old + S0_old - (Pm1 * Pm1) * x_old;
            }
        }
    }
}


// ===========================================================================
// S2 f64 LANE — sqwma
// ===========================================================================
// Reference: src/indicators/moving_averages/sqwma.rs
//   `sqwma_scalar` (:286) + `build_sqwma_weights` (:129)
//   + `sqwma_with_kernel` (:228) + `sqwma_scalar_default_14` (:339)
//
// WHAT THE f32 KERNEL ABOVE GETS WRONG, AND IS FIXED HERE
//
//   1. A DIFFERENT ALGORITHM, NOT A LOWER PRECISION. `sqwma_batch_f32` keeps
//      three rolling moments S0/S1/S2 and rebuilds the weighted sum from them
//      (`sqwma_eval_f32`: `P^2*S0 - 2P*S1 + S2`). That is the square weight
//      expanded algebraically and slid forward incrementally. The CPU computes
//      the dot product afresh at every bar, in one order, with `mul_add`. The
//      two agree in exact arithmetic and disagree at every bar in floating
//      point — and the moment recurrence additionally accumulates cancellation
//      error across the whole series, which is why it is not merely an ULP.
//      This kernel does the CPU's dot product.
//   2. WEIGHT SUM. `sqwma_weight_sum` used the closed form
//      `p(p+1)(2p+1)/6 - 1` in double and then NARROWED TO FLOAT. The CPU
//      accumulates `sum += (period - i)^2` ascending. Both are exact in f64
//      for every period below ~2^17, but the accumulation is reproduced
//      literally here rather than assumed equal.
//   3. `sqwma_scalar_default_14` (the period == 14 fast path) unrolls the same
//      13 `mul_add`s with the same integer weights in the same order, and
//      DEFAULT_WEIGHT_SUM == 1015 exactly, so ONE kernel serves both CPU
//      paths. No special case is needed and none is written.
//
// WARMUP: `warm = first_valid + period + 1` — note the `+ 1`, which is one bar
// LATER than the `first + period - 1` most of this crate's moving averages
// use, because the CPU's emit loop is `for j in (first + period + 1)..n`.
// Getting this wrong shifts the whole series rather than perturbing it.
//
// One thread per column. The dot product itself is embarrassingly parallel
// across bars, but the row is walked serially so the launch geometry matches
// the rest of the lane and the weights are built once per row.
// ===========================================================================

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_sqwma_batch_f64(
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

    // `sqwma_with_kernel` — every branch that returns Err.
    const bool declined =
        (n <= 0) ||
        (period < 2) || (period > n) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    // `build_sqwma_weights`: weight[i] = (period - i)^2 for i in 0..period-1,
    // accumulated ascending. Integers, so exact — but summed in the CPU's
    // order regardless, because "exact" is an argument, not a guarantee.
    const int p_minus_1 = period - 1;
    double weight_sum = 0.0;
    for (int i = 0; i < p_minus_1; ++i) {
        const double w = (double)(period - i);
        weight_sum += w * w;
    }
    const double inv_ws = 1.0 / weight_sum;

    const int warm = first_valid + period + 1;
    const int stop = warm < n ? warm : n;
    for (int i = 0; i < stop; ++i) row[i] = neo_s2_qnan();
    if (warm >= n) return;

    const int p4 = p_minus_1 & ~3;

    for (int j = warm; j < n; ++j) {
        double sum = 0.0;
        int k = 0;
        // The CPU's 4-at-a-time unroll. Unrolling does not change the order:
        // d[j-k] is folded in with ascending k either way, so this loop and
        // the tail below reproduce it exactly.
        for (; k < p4; k += 4) {
            const double w0 = (double)(period - (k + 0));
            const double w1 = (double)(period - (k + 1));
            const double w2 = (double)(period - (k + 2));
            const double w3 = (double)(period - (k + 3));
            sum = fma(prices[j - (k + 0)], w0 * w0, sum);
            sum = fma(prices[j - (k + 1)], w1 * w1, sum);
            sum = fma(prices[j - (k + 2)], w2 * w2, sum);
            sum = fma(prices[j - (k + 3)], w3 * w3, sum);
        }
        for (; k < p_minus_1; ++k) {
            const double w = (double)(period - k);
            sum = fma(prices[j - k], w * w, sum);
        }
        row[j] = sum * inv_ws;
    }
}
