#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#ifndef PWMA_TILE_TX
#define PWMA_TILE_TX 128
#endif

__device__ __forceinline__ size_t pwma_align_up_sz(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

extern "C" __global__
void pwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int* __restrict__ periods,
                    const int* __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > max_period) {
        return;
    }

    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights_flat[combo * max_period + idx];
    }
    __syncthreads();

    const int warm = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], shared_weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}

extern "C" __global__
void pwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,


                                     float ,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights[idx];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int warm = first_valids[series_idx] + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
            const int in_idx = (start + k) * num_series + series_idx;
            acc = fmaf(prices_tm[in_idx], shared_weights[k], acc);
        }
        out_tm[out_idx] = acc;
    }
    t += stride;
}
}


extern "C" __global__
void pwma_batch_tiled_async_f32(const float* __restrict__ prices,
                                const float* __restrict__ weights_flat,
                                const int* __restrict__ periods,
                                const int* __restrict__ warm_indices,
                                int series_len,
                                int n_combos,
                                int max_period,
                                float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int TILE = PWMA_TILE_TX;
    const int wlen = period;
    const int total = TILE + wlen - 1;

    const int warm = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = pwma_align_up_sz(off + size_t(max_period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);


    const float* wsrc = weights_flat + combo * max_period;
    for (int i = threadIdx.x; i < wlen; i += blockDim.x) {
        w[i] = wsrc[i];
    }
    __syncthreads();

    auto cta = cg::this_thread_block();
    constexpr int STAGES = 2;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
    auto pipe = cuda::make_pipeline(cta, &pss);

    const int lane = threadIdx.x;
    const int grid_tile_stride = gridDim.x * TILE;

    int t_base = blockIdx.x * TILE;
    int stage  = 0;


    for (int s = 0; s < STAGES; ++s) {
        pipe.producer_acquire();
        const int t0 = t_base + s * grid_tile_stride;
        const int p0 = t0 - (wlen - 1);
        for (int dt = lane; dt < total; dt += blockDim.x) {
            const int tcur = p0 + dt;
            if (tcur >= 0 && tcur < series_len) {
                cuda::memcpy_async(&tile[s * total + dt], &prices[tcur], sizeof(float), pipe);
            } else {
                tile[s * total + dt] = 0.0f;
            }
        }
        pipe.producer_commit();
    }


    while (t_base < series_len) {
        pipe.consumer_wait();
        __syncthreads();


        const float* tbuf = &tile[stage * total];
        const int t = t_base + lane;
        if (t < series_len) {
            if (t < warm) {
                out[base_out + t] = nan_f;
            } else {
                int start = lane;
                const float* xptr = &tbuf[start];
                float acc = 0.0f;
#pragma unroll 8
                for (int k = 0; k < wlen; ++k) {
                    acc = fmaf(xptr[k], w[k], acc);
                }
                out[base_out + t] = acc;
            }
        }

        __syncthreads();
        pipe.consumer_release();


        pipe.producer_acquire();
        const int next_t0 = t_base + STAGES * grid_tile_stride;
        const int next_p0 = next_t0 - (wlen - 1);
        const int next_stage = stage;

        for (int dt = lane; dt < total; dt += blockDim.x) {
            const int tcur = next_p0 + dt;
            if (tcur >= 0 && tcur < series_len) {
                cuda::memcpy_async(&tile[next_stage * total + dt], &prices[tcur], sizeof(float), pipe);
            } else {
                tile[next_stage * total + dt] = 0.0f;
            }
        }
        pipe.producer_commit();

        t_base += grid_tile_stride;
        stage = (stage + 1) % STAGES;
    }
}


__device__ __forceinline__ size_t pwma_align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

template<int TX, int TY>
__device__ __forceinline__
void pwma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float ,
                          int num_series,
                          int series_len,
                          const int* __restrict__ first_valids,
                          float* __restrict__ out_tm) {
    const int t0 = blockIdx.x * TX;
    const int s0 = blockIdx.y * TY;
    if (t0 >= series_len || s0 >= num_series) return;


    const int total = TX + period - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = pwma_align_up(off + size_t(period) * sizeof(float), 16);
    const int LD = TY + 1;
    float* tile = reinterpret_cast<float*>(shraw + off);


    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2;
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += blockDim.x * blockDim.y) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += blockDim.x * blockDim.y) {
            w[i] = weights[i];
        }
    }
    __syncthreads();


    const bool vec_ok = (TY == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);
    const int p0 = t0 - (period - 1);
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            if (vec_ok && threadIdx.y == 0) {
                const float4* src4 = reinterpret_cast<const float4*>(&prices_tm[t * num_series + s0]);
                float4 v = src4[0];
                tile[dt * LD + 0] = v.x;
                tile[dt * LD + 1] = v.y;
                tile[dt * LD + 2] = v.z;
                tile[dt * LD + 3] = v.w;
            } else {
                int s = s0 + threadIdx.y;
                float val = 0.f;
                if (s < num_series) val = prices_tm[t * num_series + s];
                tile[dt * LD + threadIdx.y] = val;
            }
        } else {
            int idx = dt * LD + threadIdx.y;
            if (idx < total * LD) tile[idx] = 0.f;
        }
    }
    __syncthreads();


    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;

    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) {
        out_tm[out_idx] = __int_as_float(0x7fffffff);
        return;
    }

    int start = threadIdx.x;
    const float* xptr = &tile[start * LD + threadIdx.y];
    float acc = 0.f;
#pragma unroll 8
    for (int i = 0; i < period; ++i) {
        acc = fmaf(xptr[i * LD], w[i], acc);
    }

    out_tm[out_idx] = acc;
}

#define DEFINE_PWMA_MS1P_TILED(NAME, TX, TY)                                    \
extern "C" __global__ void NAME(                                                \
  const float* __restrict__ prices_tm,                                          \
  const float* __restrict__ weights,                                            \
  int period, float inv_norm, int num_series, int series_len,                   \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {           \
  pwma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,            \
                               num_series, series_len, first_valids, out_tm);   \
}


DEFINE_PWMA_MS1P_TILED(pwma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_PWMA_MS1P_TILED(pwma_ms1p_tiled_f32_tx128_ty4, 128, 4)


#ifndef PWMA_MAX_PERIOD_CONST
#define PWMA_MAX_PERIOD_CONST 4096
#endif

__constant__ float pwma_const_w[PWMA_MAX_PERIOD_CONST];

extern "C" __global__
void pwma_ms1p_const_f32(const float* __restrict__ prices_tm,
                         int period,
                         int num_series, int series_len,
                         const int* __restrict__ first_valids,
                         float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;
    const int warm = first_valids[series_idx] + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices_tm[(start + k) * num_series + series_idx], pwma_const_w[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}


// ===========================================================================
// S2 f64 LANE — pwma  (Pascal weighted moving average)
// ===========================================================================
// Reference: src/indicators/moving_averages/pwma.rs
//   `pwma_prepare`          (:248) — first_valid, refusal, weight choice
//   `pwma_with_kernel`      (:327) — warm = first + period - 1
//   `pwma_scalar_dispatch`  (:384) — period == 5 takes a DIFFERENT path
//   `pwma_scalar_period5`   (:399) — hardcoded weights, hardcoded association
//   `pwma_scalar`           (:425) — 4 partial sums, then (s0+s1)+(s2+s3)
//   `pascal_weights`        (:1447) + `combination_f64` (:1471)
//   Batch route: `ma_batch.rs:1234` sweeps `period` and takes nothing else.
//
// THE ASSOCIATION IS THE SPECIFICATION, TWICE OVER.
//  1. `pwma_scalar` does NOT accumulate one running sum. It keeps FOUR
//     independent accumulators, folds every 4th element into each with
//     `mul_add`, and combines them as `(s0 + s1) + (s2 + s3)` — a balanced
//     tree, not a chain. A single-accumulator loop is a different number at
//     every bar. Reproduced literally.
//  2. `period == 5` is not the same algorithm with a fixed weight vector: it
//     computes `((d0*w0 + d1*w1) + (d2*w2 + d3*w3))` with PLAIN multiplies and
//     then ONE `mul_add` for the last term. Different rounding count, different
//     tree. It gets its own branch here, exactly as on the CPU.
//
// THE WEIGHTS. `combination_f64` alternates multiply-then-divide
// (`result *= (n-i); result /= (i+1)`) rather than computing a factorial
// ratio, which keeps the running value small; then the row is summed in order
// and each entry divided by that sum. Reproduced step for step — a closed-form
// binomial would agree in exact arithmetic and differ in the last bits, and
// these weights multiply every bar of the output.
//
// NaN. `__int_as_float(0x7fc00000)` in the f32 kernels above is an f32 bit
// pattern; the f64 quiet NaN is a different 64-bit value.
// ===========================================================================

#define PWMA_MAX_PERIOD 512

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// `combination_f64` (pwma.rs:1471), operation for operation.
__device__ __forceinline__ double neo_s2_combination(int n, int r) {
    const int rr = (r < (n - r)) ? r : (n - r);
    if (rr == 0) return 1.0;
    double result = 1.0;
    for (int i = 0; i < rr; ++i) {
        result *= (double)(n - i);
        result /= (double)(i + 1);
    }
    return result;
}

extern "C" __global__ void neoethos_pwma_batch_f64(
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
        (period <= 0) || (period > n) || (period > PWMA_MAX_PERIOD) ||
        (first_valid < 0) || (first_valid >= n);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    // `alloc_with_nan_prefix(len, warm)`; the compute writes warm..n-1. Filling
    // the whole row first also defines the tail when warm >= n.
    const int warm = first_valid + period - 1;
    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
    if (warm >= n) return;

    if (period == 5) {
        // `pwma_scalar_period5` — its own association, not `pwma_scalar` with
        // PWMA_PERIOD5_WEIGHTS substituted in.
        for (int i = first_valid + 4; i < n; ++i) {
            const double d0 = prices[i - 4];
            const double d1 = prices[i - 3];
            const double d2 = prices[i - 2];
            const double d3 = prices[i - 1];
            const double d4 = prices[i];
            const double sum = ((d0 * 0.0625) + (d1 * 0.25)) + ((d2 * 0.375) + (d3 * 0.25));
            row[i] = fma(d4, 0.0625, sum);
        }
        return;
    }

    // `pascal_weights(period)`: the binomial row, summed ascending, then each
    // entry divided by that sum.
    double w[PWMA_MAX_PERIOD];
    const int nn = period - 1;
    double wsum = 0.0;
    for (int k = 0; k <= nn; ++k) {
        w[k] = neo_s2_combination(nn, k);
    }
    for (int k = 0; k <= nn; ++k) {
        wsum += w[k];
    }
    if (wsum == 0.0) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }
    for (int k = 0; k <= nn; ++k) {
        w[k] /= wsum;
    }

    const int k_end = period & ~3;

    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - period;

        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int k = 0;
        for (; k < k_end; k += 4) {
            s0 = fma(prices[start + k + 0], w[k + 0], s0);
            s1 = fma(prices[start + k + 1], w[k + 1], s1);
            s2 = fma(prices[start + k + 2], w[k + 2], s2);
            s3 = fma(prices[start + k + 3], w[k + 3], s3);
        }
        double sum = (s0 + s1) + (s2 + s3);
        for (; k < period; ++k) {
            sum = fma(prices[start + k], w[k], sum);
        }
        row[i] = sum;
    }
}
