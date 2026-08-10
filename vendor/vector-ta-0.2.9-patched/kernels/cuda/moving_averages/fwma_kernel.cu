#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef FWMA_TILE_T

#define FWMA_TILE_T 256
#endif


extern "C" __global__
void fwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;


    extern __shared__ float smem[];
    float* __restrict__ s_w = smem;
    float* __restrict__ s_x = s_w + max_period;


    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm     = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f  = __int_as_float(0x7fffffff);


    const int tile_t0 = blockIdx.x * blockDim.x;
    const int tile_t1 = min(series_len, tile_t0 + blockDim.x);


    if (tile_t1 <= warm) {
        const int t = tile_t0 + threadIdx.x;
        if (t < tile_t1) out[base_out + t] = nan_f;
        return;
    }


    const int load_base = tile_t0 - period + 1;
    const int load_len  = (tile_t1 - tile_t0) + period - 1;


    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        const int g = load_base + i;
        s_x[i] = (unsigned(g) < (unsigned)series_len) ? prices[g] : 0.0f;
    }
    __syncthreads();


    const int t = tile_t0 + threadIdx.x;
    if (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {

            const int offset = (t - period + 1) - load_base;
            float acc = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(s_x[offset + k], s_w[k], acc);
            }
            out[base_out + t] = acc;
        }
    }
}


#ifndef FWMA_TIME_STEPS_PER_BLOCK
#define FWMA_TIME_STEPS_PER_BLOCK 4
#endif

extern "C" __global__
void fwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {

    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);


    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;


    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {

                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}


extern "C" __global__
void fwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const float* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {

    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);

    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;

    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `moving_averages/fwma.rs::fwma_scalar` (l.560), weights built
// at `fwma.rs:331-341`:
//     fib[0] = fib[1] = 1.0 ; fib[i] = fib[i-1] + fib[i-2]  for i in 2..period
//     fib_sum = fib.iter().sum()          <- ASCENDING, left-associated
//     fib[i] /= fib_sum                   <- a DIVIDE per weight, not a
//                                            multiply by a reciprocal
//     for i in (first + period - 1)..len:
//         window = data[i+1-period ..= i]              (index 0 = OLDEST bar)
//         sum accumulated in GROUPS OF FOUR:
//             sum += d0*w0 + d1*w1 + d2*w2 + d3*w3
//         then the remainder ONE AT A TIME: sum += d*w
//         out[i] = sum
//
// NO FMA. The CPU writes `d*w` and then `+`, so each tap is two roundings; an
// `fma(d, w, sum)` would be one and a different number. The group of four is a
// left-associated 4-term product sum added to `sum` in a SINGLE add — not four
// adds — so the grouping is reproduced literally.
//
// The normalised weight table is built once per thread into a fixed local
// array, which is why the compiled kernel carries a period bound and the host
// refuses an oversized period by name.
//
// f32 -> f64 audit: pointers/locals widened; `0.0f`/`1.0f` widened;
// `__int_as_float` NaN x3 -> the f64 quiet-NaN bit pattern; no fast-math
// intrinsic survives; no epsilon (`fib_sum == 0.0` is an exact test on the CPU
// and stays exact); no min/max chain.
// ---------------------------------------------------------------------------

#ifndef FWMA_MAX_PERIOD_F64
#define FWMA_MAX_PERIOD_F64 512
#endif

static __device__ __forceinline__ double fwma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void fwma_batch_f64(const double* __restrict__ prices,
                    int n,
                    const int*   __restrict__ periods,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = fwma_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const long long warm_ll =
        static_cast<long long>(first_valid) + static_cast<long long>(period) - 1;
    if (period <= 0 || period > FWMA_MAX_PERIOD_F64 || warm_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    const int warm = static_cast<int>(warm_ll);

    double fib[FWMA_MAX_PERIOD_F64];
    for (int k = 0; k < period; ++k) {
        fib[k] = (k < 2) ? 1.0 : (fib[k - 1] + fib[k - 2]);
    }
    double fib_sum = 0.0;
    for (int k = 0; k < period; ++k) fib_sum += fib[k];
    if (fib_sum == 0.0) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    for (int k = 0; k < period; ++k) fib[k] /= fib_sum;

    for (int t = 0; t < warm; ++t) row[t] = nan_d;

    const int p4 = period & ~3;

    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - period;
        double sum = 0.0;

        int k = 0;
        while (k < p4) {
            sum += prices[start + k + 0] * fib[k + 0]
                 + prices[start + k + 1] * fib[k + 1]
                 + prices[start + k + 2] * fib[k + 2]
                 + prices[start + k + 3] * fib[k + 3];
            k += 4;
        }
        while (k < period) {
            sum += prices[start + k] * fib[k];
            ++k;
        }

        row[i] = sum;
    }
}
