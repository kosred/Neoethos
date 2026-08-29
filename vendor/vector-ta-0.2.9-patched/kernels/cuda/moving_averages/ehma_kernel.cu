#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#ifndef EHMA_USE_ASYNC
#define EHMA_USE_ASYNC 1
#endif

__device__ __forceinline__ float ehma_hann_weight(int period, int idx) {


    const float i = static_cast<float>(period - idx);
    const float x = i / (static_cast<float>(period) + 1.0f);
    const float s = sinpif(x);
    return 2.0f * s * s;
}


extern "C" __global__
void ehma_batch_f32(const float* __restrict__ prices,
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

    extern __shared__ float weights[];


    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        weights[idx] = ehma_hann_weight(period, idx);
    }
    __syncthreads();

    const float inv_norm = 1.0f / (static_cast<float>(period) + 1.0f);

    const int warm = warm_indices[combo];
    const int first = warm - period + 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm || (t - period + 1) < first) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int k = 0;
            #pragma unroll 4
            for (; k + 3 < period; k += 4) {
                s0 = __fmaf_rn(prices[start + k],     weights[k],     s0);
                s1 = __fmaf_rn(prices[start + k + 1], weights[k + 1], s1);
                s2 = __fmaf_rn(prices[start + k + 2], weights[k + 2], s2);
                s3 = __fmaf_rn(prices[start + k + 3], weights[k + 3], s3);
            }
            float acc = (s0 + s1) + (s2 + s3);
            for (; k < period; ++k) {
                acc = __fmaf_rn(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc * inv_norm;
        }
        t += stride;
    }
}


extern "C" __global__
void ehma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
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

    const int first = first_valids[series_idx];
    const int warm = first + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int k = 0;
            #pragma unroll 4
            for (; k + 3 < period; k += 4) {
                int idx0 = (start + k) * num_series + series_idx;
                int idx1 = (start + k + 1) * num_series + series_idx;
                int idx2 = (start + k + 2) * num_series + series_idx;
                int idx3 = (start + k + 3) * num_series + series_idx;
                s0 = __fmaf_rn(prices_tm[idx0], shared_weights[k],     s0);
                s1 = __fmaf_rn(prices_tm[idx1], shared_weights[k + 1], s1);
                s2 = __fmaf_rn(prices_tm[idx2], shared_weights[k + 2], s2);
                s3 = __fmaf_rn(prices_tm[idx3], shared_weights[k + 3], s3);
            }
            float acc = (s0 + s1) + (s2 + s3);
            for (; k < period; ++k) {
                int in_idx = (start + k) * num_series + series_idx;
                acc = __fmaf_rn(prices_tm[in_idx], shared_weights[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}


__device__ __forceinline__ size_t ehma_align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

__device__ __forceinline__ float ehma_dot_uncomp(const float* __restrict__ x,
                                                 const float* __restrict__ w,
                                                 int n) {

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    #pragma unroll 4
    for (; i + 3 < n; i += 4) {
        s0 = __fmaf_rn(x[i],     w[i],     s0);
        s1 = __fmaf_rn(x[i + 1], w[i + 1], s1);
        s2 = __fmaf_rn(x[i + 2], w[i + 2], s2);
        s3 = __fmaf_rn(x[i + 3], w[i + 3], s3);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s = __fmaf_rn(x[i], w[i], s);
    return s;
}

__device__ __forceinline__ void ehma_dot2_shared(const float* __restrict__ buf,
                                                  int b,
                                                  const float* __restrict__ w,
                                                  int n,
                                                  float& s0_out,
                                                  float& s1_out) {

    float s00 = 0.f, s01 = 0.f, s02 = 0.f, s03 = 0.f;
    float s10 = 0.f, s11 = 0.f, s12 = 0.f, s13 = 0.f;
    int i = 0;
    #pragma unroll 4
    for (; i + 3 < n; i += 4) {
        float w0 = w[i];     float w1 = w[i + 1];
        float w2 = w[i + 2]; float w3 = w[i + 3];
        s00 = __fmaf_rn(buf[b + i],     w0, s00);
        s10 = __fmaf_rn(buf[b + i + 1], w0, s10);
        s01 = __fmaf_rn(buf[b + i + 1], w1, s01);
        s11 = __fmaf_rn(buf[b + i + 2], w1, s11);
        s02 = __fmaf_rn(buf[b + i + 2], w2, s02);
        s12 = __fmaf_rn(buf[b + i + 3], w2, s12);
        s03 = __fmaf_rn(buf[b + i + 3], w3, s03);
        s13 = __fmaf_rn(buf[b + i + 4], w3, s13);
    }
    float s0 = (s00 + s01) + (s02 + s03);
    float s1 = (s10 + s11) + (s12 + s13);
    for (; i < n; ++i) {
        float wi = w[i];
        s0 = __fmaf_rn(buf[b + i],     wi, s0);
        s1 = __fmaf_rn(buf[b + i + 1], wi, s1);
    }
    s0_out = s0; s1_out = s1;
}

template<int TILE>
struct EhmaBatchTiledPrecomputed2X {
    static __device__ __forceinline__
    void run(const float* __restrict__ prices,
             const float* __restrict__ weights_flat,
             const int*   __restrict__ periods,
             const float* __restrict__ inv_norms,
             int max_period,
             int series_len,
             int n_combos,
             int first_valid,
             float* __restrict__ out) {
        const int THREADS = TILE / 2;
        if (blockDim.x != THREADS) return;

        const int combo = blockIdx.y;
        if (combo >= n_combos) return;
        const int period = periods[combo];

        const int t0 = blockIdx.x * TILE;
        if (t0 >= series_len) return;

        const int total = TILE + period - 1;
        extern __shared__ __align__(16) unsigned char shraw[];
        size_t off = 0;
        float* w = reinterpret_cast<float*>(shraw + off);
        off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
        float* buf = reinterpret_cast<float*>(shraw + off);


        const float* wsrc = weights_flat + combo * max_period;
        uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
        if ((waddr & 0xF) == 0) {
            int ve = period >> 2;
            for (int vi = threadIdx.x; vi < ve; vi += THREADS) {
                reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(wsrc)[vi];
            }
            if ((threadIdx.x == 0) && ((period & 3) != 0)) {
                int base = ve << 2;
                for (int r = 0; r < (period & 3); ++r) w[base + r] = wsrc[base + r];
            }
        } else {
            for (int i = threadIdx.x; i < period; i += THREADS) w[i] = wsrc[i];
        }
        __syncthreads();


        const int p_base0 = t0 - (period - 1);
        bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);
        if (in_bounds) {
            const float* src = prices + p_base0;
            uintptr_t addr = reinterpret_cast<uintptr_t>(src);
            if ((addr & 0xF) == 0) {
                int vec_elems = total >> 2;
                for (int vi = threadIdx.x; vi < vec_elems; vi += THREADS) {
                    reinterpret_cast<float4*>(buf)[vi] = reinterpret_cast<const float4*>(src)[vi];
                }
                if ((threadIdx.x == 0) && ((total & 3) != 0)) {
                    int base = vec_elems << 2;
                    for (int r = 0; r < (total & 3); ++r) buf[base + r] = src[base + r];
                }
            } else {
                for (int i = threadIdx.x; i < total; i += THREADS) buf[i] = src[i];
            }
        } else {
            for (int i = threadIdx.x; i < total; i += THREADS) {
                int idx = p_base0 + i;
                buf[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
            }
        }
        __syncthreads();

        const int warm = first_valid + period - 1;
        const int combo_base = combo * series_len;


        int b = 2 * threadIdx.x;
        int t = t0 + b;
        float out0 = NAN, out1 = NAN;
        if (t < series_len) {
            const bool can0 = (t >= warm);
            const bool can1 = ((t + 1) < series_len) && ((t + 1) >= warm);
            if (can0 && can1) {
                float s0, s1;
                ehma_dot2_shared(buf, b, w, period, s0, s1);
                out0 = s0;
                out1 = s1;
            } else if (can0) {
                out0 = ehma_dot_uncomp(&buf[b], w, period);
            } else if (can1) {
                out1 = ehma_dot_uncomp(&buf[b + 1], w, period);
            }
            out[combo_base + t] = out0;
            if ((t + 1) < series_len) out[combo_base + t + 1] = out1;
        }
    }
};

#define DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(NAME, TILE_OUT)                         \
extern "C" __global__ void NAME(                                                  \
  const float* __restrict__ prices,                                               \
  const float* __restrict__ weights_flat,                                         \
  const int*   __restrict__ periods,                                              \
  const float* __restrict__ inv_norms,                                            \
  int max_period, int series_len, int n_combos, int first_valid,                  \
  float* __restrict__ out) {                                                      \
  EhmaBatchTiledPrecomputed2X<TILE_OUT>::run(                                     \
    prices, weights_flat, periods, inv_norms, max_period,                         \
    series_len, n_combos, first_valid, out);                                      \
}

DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(ehma_batch_tiled_f32_2x_tile128, 128)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(ehma_batch_tiled_f32_2x_tile256, 256)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(ehma_batch_tiled_f32_2x_tile512, 512)


template<int TILE>
struct EhmaBatchTiledPrecomputed2X_Async {
    static __device__ __forceinline__
    void run(const float* __restrict__ prices,
             const float* __restrict__ weights_flat,
             const int*   __restrict__ periods,
             const float* __restrict__ inv_norms,
             int max_period, int series_len, int n_combos, int first_valid,
             float* __restrict__ out) {

        const int THREADS = TILE / 2;
        if (blockDim.x != THREADS) return;

        const int combo = blockIdx.y;
        if (combo >= n_combos) return;

        const int period = periods[combo];
        if (period <= 0 || period > max_period) return;


        extern __shared__ __align__(16) unsigned char shraw[];
        size_t off = 0;
        float* w = reinterpret_cast<float*>(shraw + off);
        off = ehma_align_up(off + size_t(period) * sizeof(float), 16);

        float* buf = reinterpret_cast<float*>(shraw + off);
        const int total = TILE + period - 1;


        const float* wsrc = weights_flat + combo * max_period;
        uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
        if ((waddr & 0xF) == 0) {
            int ve = period >> 2;
            for (int vi = threadIdx.x; vi < ve; vi += THREADS) {
                reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(wsrc)[vi];
            }
            if ((threadIdx.x == 0) && ((period & 3) != 0)) {
                int base = ve << 2;
                for (int r = 0; r < (period & 3); ++r) w[base + r] = wsrc[base + r];
            }
        } else {
            for (int i = threadIdx.x; i < period; i += THREADS) w[i] = wsrc[i];
        }
        __syncthreads();

        const int warm = first_valid + period - 1;
        const int combo_base = combo * series_len;


        for (int t0 = blockIdx.x * TILE; t0 < series_len; t0 += gridDim.x * TILE) {


            const int p_base0 = t0 - (period - 1);
            const bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);

#if EHMA_USE_ASYNC && (__CUDA_ARCH__ >= 800)
            if (in_bounds) {

                auto block = cg::this_thread_block();
                cg::memcpy_async(block, buf, prices + p_base0, sizeof(float) * total);
                cg::wait(block);
                __syncthreads();
            } else
#endif
            {

                for (int i = threadIdx.x; i < total; i += THREADS) {
                    int idx = p_base0 + i;
                    buf[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
                }
                __syncthreads();
            }


            int b = 2 * threadIdx.x;
            int t = t0 + b;

            if (t < series_len) {
                float out0 = NAN, out1 = NAN;
                const bool can0 = (t >= warm);
                const bool can1 = ((t + 1) < series_len) && ((t + 1) >= warm);
                if (can0 && can1) {
                    float s0, s1;
                    ehma_dot2_shared(buf, b, w, period, s0, s1);
                    out0 = s0; out1 = s1;
                } else if (can0) {
                    out0 = ehma_dot_uncomp(&buf[b], w, period);
                } else if (can1) {
                    out1 = ehma_dot_uncomp(&buf[b + 1], w, period);
                }
                out[combo_base + t] = out0;
                if ((t + 1) < series_len) out[combo_base + t + 1] = out1;
            }
            __syncthreads();
        }
    }
};

#define DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(NAME, TILE_OUT)                               \
extern "C" __global__ void NAME(                                                               \
  const float* __restrict__ prices, const float* __restrict__ weights_flat,                    \
  const int*   __restrict__ periods, const float* __restrict__ inv_norms,                      \
  int max_period, int series_len, int n_combos, int first_valid,                               \
  float* __restrict__ out) {                                                                   \
  EhmaBatchTiledPrecomputed2X_Async<TILE_OUT>::run(                                            \
    prices, weights_flat, periods, inv_norms, max_period,                                      \
    series_len, n_combos, first_valid, out);                                                   \
}

DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(ehma_batch_tiled_f32_2x_tile128_async, 128)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(ehma_batch_tiled_f32_2x_tile256_async, 256)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(ehma_batch_tiled_f32_2x_tile512_async, 512)


__device__ __forceinline__
float ehma_dot_stride_uncomp(const float* __restrict__ x, int stride,
                             const float* __restrict__ w, int n) {

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    #pragma unroll 4
    for (; i + 3 < n; i += 4) {
        s0 = __fmaf_rn(x[(i + 0) * stride], w[i + 0], s0);
        s1 = __fmaf_rn(x[(i + 1) * stride], w[i + 1], s1);
        s2 = __fmaf_rn(x[(i + 2) * stride], w[i + 2], s2);
        s3 = __fmaf_rn(x[(i + 3) * stride], w[i + 3], s3);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s = __fmaf_rn(x[i * stride], w[i], s);
    return s;
}

template<int TX, int TY>
__device__ __forceinline__
void ehma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float inv_norm,
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
    off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);


    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    const int THREADS = blockDim.x * blockDim.y;
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2;
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += THREADS) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += THREADS) {
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
                tile[dt * TY + 0] = v.x;
                tile[dt * TY + 1] = v.y;
                tile[dt * TY + 2] = v.z;
                tile[dt * TY + 3] = v.w;
            } else {
                int s = s0 + threadIdx.y;
                float val = 0.f;
                if (s < num_series) val = prices_tm[t * num_series + s];
                tile[dt * TY + threadIdx.y] = val;
            }
        } else {
            int idx = dt * TY + threadIdx.y;
            if (idx < total * TY) tile[idx] = 0.f;
        }
    }
    __syncthreads();

    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;
    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) { out_tm[out_idx] = NAN; return; }

    int start = threadIdx.x;
    const float* xptr = &tile[start * TY + threadIdx.y];
    float acc = ehma_dot_stride_uncomp(xptr, TY, w, period);
    out_tm[out_idx] = acc;
}

#define DEFINE_EHMA_MS1P_TILED(NAME, TX, TY)                                       \
extern "C" __global__ void NAME(                                                  \
  const float* __restrict__ prices_tm,                                            \
  const float* __restrict__ weights,                                              \
  int period, float inv_norm, int num_series, int series_len,                     \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {             \
  ehma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,              \
                               num_series, series_len, first_valids, out_tm);     \
}

DEFINE_EHMA_MS1P_TILED(ehma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_EHMA_MS1P_TILED(ehma_ms1p_tiled_f32_tx128_ty4, 128, 4)


template<int TX, int TY>
__device__ __forceinline__
void ehma_ms1p_tiled_core_async(const float* __restrict__ prices_tm,
                                const float* __restrict__ weights,
                                int period, float inv_norm,
                                int num_series, int series_len,
                                const int* __restrict__ first_valids,
                                float* __restrict__ out_tm) {
    const int t0 = blockIdx.x * TX;
    const int s0 = blockIdx.y * TY;
    if (t0 >= series_len || s0 >= num_series) return;

    const int total = TX + period - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);


    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    const int THREADS = blockDim.x * blockDim.y;
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2;
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += THREADS) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if (threadIdx.x == 0 && threadIdx.y == 0 && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += THREADS) w[i] = weights[i];
    }
    __syncthreads();


    const int p0 = t0 - (period - 1);
#if EHMA_USE_ASYNC && (__CUDA_ARCH__ >= 800)


    const bool vec_ok = (TY == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);
    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pss;
    auto pipe = cuda::make_pipeline(block, &pss);

    pipe.producer_acquire();
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            if (vec_ok) {
                if (threadIdx.y == 0) {
                    const float* src = &prices_tm[t * num_series + s0];
                    float* dst = &tile[dt * TY];
                    cuda::memcpy_async(dst, src, sizeof(float4), pipe);
                }
            } else {
                int s = s0 + threadIdx.y;
                float* dst = &tile[dt * TY + threadIdx.y];
                if (s < num_series) {
                    const float* src = &prices_tm[t * num_series + s];
                    cuda::memcpy_async(dst, src, sizeof(float), pipe);
                } else {
                    *dst = 0.f;
                }
            }
        } else {
            int idx = dt * TY + threadIdx.y;
            if (idx < total * TY) tile[idx] = 0.f;
        }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();


    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s < num_series && t < series_len) {
        int warm = first_valids[s] + period - 1;
        int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            int start = threadIdx.x;
            const float* xptr = &tile[start * TY + threadIdx.y];
            float acc = ehma_dot_stride_uncomp(xptr, TY, w, period);
            out_tm[out_idx] = acc;
        }
    }
    __syncthreads();
    pipe.consumer_release();
#else
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            const float* src = &prices_tm[t * num_series + s0];
            float* dst = &tile[dt * TY];
            for (int j = threadIdx.y; j < TY; j += blockDim.y) {
                int s = s0 + j;
                dst[j] = (s < num_series) ? src[j] : 0.f;
            }
        } else {
            for (int j = threadIdx.y; j < TY; j += blockDim.y) {
                int idx = dt * TY + j;
                if (idx < total * TY) tile[idx] = 0.f;
            }
        }
    }
    __syncthreads();


    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;
    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) { out_tm[out_idx] = NAN; return; }

    int start = threadIdx.x;
    const float* xptr = &tile[start * TY + threadIdx.y];
    float acc = ehma_dot_stride_uncomp(xptr, TY, w, period);
    out_tm[out_idx] = acc;
#endif
}

#define DEFINE_EHMA_MS1P_TILED_ASYNC(NAME, TX, TY)                                            \
extern "C" __global__ void NAME(                                                              \
  const float* __restrict__ prices_tm, const float* __restrict__ weights,                     \
  int period, float inv_norm, int num_series, int series_len,                                 \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {                         \
  ehma_ms1p_tiled_core_async<TX, TY>(prices_tm, weights, period, inv_norm,                    \
                                     num_series, series_len, first_valids, out_tm);           \
}

DEFINE_EHMA_MS1P_TILED_ASYNC(ehma_ms1p_tiled_f32_tx128_ty2_async, 128, 2)
DEFINE_EHMA_MS1P_TILED_ASYNC(ehma_ms1p_tiled_f32_tx128_ty4_async, 128, 4)

/* ===========================================================================
 * S4 f64 LANE — ehma (Ehlers Hann moving average)
 * ---------------------------------------------------------------------------
 * Accuracy identity shared byte-for-byte in intent with ehma.rs:
 *   ehma_hann_f64_msun_ddangle_symmetric_pow2_anchored_dot2_v2
 *
 * John F. Ehlers' Hann moving average uses
 *   weight(k) = 1 - cos(2*pi*k/(period+1)), k = 1..period,
 * followed by sum(weight*price)/sum(weight).  This f64 lane evaluates the
 * equivalent cancellation-safe identity `2*sin(pi*k/(period+1))^2`.
 *
 * The half-angle is built with a double-double residual, sine is the vendored
 * FreeBSD-msun polynomial/reduction (not host libm/libdevice), and only the
 * first half of the exactly symmetric window is evaluated.  The chronological
 * dot product is anchored after power-of-two normalization and accumulated as
 * product plus FMA residual plus TwoSum residual.  Explicit CUDA RN intrinsics
 * mirror every intentional CPU rounding point; no AVX reassociation defines
 * the result.  The f32 kernels above retain their independent contract.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

// Must equal `EHMA_MAX_PERIOD` in src/cuda/neoethos_f64_wrapper.rs.
#define NEO_EHMA_MAX_PERIOD 512

static __device__ __forceinline__ unsigned long long neo_ehma_bits_v2(double value)
{
    return (unsigned long long)__double_as_longlong(value);
}

static __device__ __forceinline__ bool neo_ehma_is_nan_bits_v2(
    unsigned long long bits)
{
    const unsigned long long absolute = bits & 0x7fffffffffffffffULL;
    return absolute > 0x7ff0000000000000ULL;
}

/* FreeBSD msun k_sin/k_cos and medium pi/2 reduction.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft. Permission to use, copy, modify, and
 * distribute this software is freely granted, provided this notice is
 * preserved.
 */
static __device__ __forceinline__ double neo_ehma_msun_k_cos_v2(
    double x,
    double y)
{
    const double c1 = 0x1.555555555554cp-5;
    const double c2 = -0x1.6c16c16c15177p-10;
    const double c3 = 0x1.a01a019cb1590p-16;
    const double c4 = -0x1.27e4f809c52adp-22;
    const double c5 = 0x1.1ee9ebdb4b1c4p-29;
    const double c6 = -0x1.8fae9be8838d4p-37;
    const double z = __dmul_rn(x, x);
    const double w2 = __dmul_rn(z, z);
    const double low = __dadd_rn(c2, __dmul_rn(z, c3));
    const double left = __dmul_rn(z, __dadd_rn(c1, __dmul_rn(z, low)));
    const double high = __dadd_rn(c5, __dmul_rn(z, c6));
    const double right = __dmul_rn(
        __dmul_rn(w2, w2),
        __dadd_rn(c4, __dmul_rn(z, high)));
    const double r = __dadd_rn(left, right);
    const double hz = __dmul_rn(0.5, z);
    const double w = __dsub_rn(1.0, hz);
    const double rounding = __dsub_rn(__dsub_rn(1.0, w), hz);
    const double tail = __dsub_rn(__dmul_rn(z, r), __dmul_rn(x, y));
    return __dadd_rn(w, __dadd_rn(rounding, tail));
}

static __device__ __forceinline__ double neo_ehma_msun_k_sin_v2(
    double x,
    double y,
    bool has_tail)
{
    const double s1 = -0x1.5555555555549p-3;
    const double s2 = 0x1.111111110f8a6p-7;
    const double s3 = -0x1.a01a019c161d5p-13;
    const double s4 = 0x1.71de357b1fe7dp-19;
    const double s5 = -0x1.ae5e68a2b9cebp-26;
    const double s6 = 0x1.5d93a5acfd57cp-33;
    const double z = __dmul_rn(x, x);
    const double w = __dmul_rn(z, z);
    const double first = __dadd_rn(
        s2,
        __dmul_rn(z, __dadd_rn(s3, __dmul_rn(z, s4))));
    const double second = __dmul_rn(
        __dmul_rn(z, w),
        __dadd_rn(s5, __dmul_rn(z, s6)));
    const double r = __dadd_rn(first, second);
    const double v = __dmul_rn(z, x);
    if (has_tail) {
        const double inner = __dsub_rn(
            __dmul_rn(z, __dsub_rn(__dmul_rn(0.5, y), __dmul_rn(v, r))),
            y);
        return __dsub_rn(x, __dsub_rn(inner, __dmul_rn(v, s1)));
    }
    return __dadd_rn(x, __dmul_rn(v, __dadd_rn(s1, __dmul_rn(z, r))));
}

static __device__ __forceinline__ int neo_ehma_reduce_pio2_v2(
    double x,
    double* y0_out,
    double* y1_out)
{
    const double inv_pio2 = 0x1.45f306dc9c883p-1;
    const double to_int = 0x1.8p+52;
    const double pio2_1 = 0x1.921fb54400000p+0;
    const double pio2_1t = 0x1.0b4611a626331p-34;
    const double pio2_2 = 0x1.0b4611a600000p-34;
    const double pio2_2t = 0x1.3198a2e037073p-69;
    const double pio2_3 = 0x1.3198a2e000000p-69;
    const double pio2_3t = 0x1.b839a252049c1p-104;

    const double tmp = __dadd_rn(__dmul_rn(x, inv_pio2), to_int);
    const double f_n = __dsub_rn(tmp, to_int);
    const int n = (int)f_n;
    double r = __dsub_rn(x, __dmul_rn(f_n, pio2_1));
    double w = __dmul_rn(f_n, pio2_1t);
    double y0 = __dsub_rn(r, w);
    const int ex = (int)((neo_ehma_bits_v2(x) >> 52) & 0x7ffULL);
    int ey = (int)((neo_ehma_bits_v2(y0) >> 52) & 0x7ffULL);
    if (ex - ey > 16) {
        const double t = r;
        w = __dmul_rn(f_n, pio2_2);
        r = __dsub_rn(t, w);
        w = __dsub_rn(
            __dmul_rn(f_n, pio2_2t),
            __dsub_rn(__dsub_rn(t, r), w));
        y0 = __dsub_rn(r, w);
        ey = (int)((neo_ehma_bits_v2(y0) >> 52) & 0x7ffULL);
        if (ex - ey > 49) {
            const double t2 = r;
            w = __dmul_rn(f_n, pio2_3);
            r = __dsub_rn(t2, w);
            w = __dsub_rn(
                __dmul_rn(f_n, pio2_3t),
                __dsub_rn(__dsub_rn(t2, r), w));
            y0 = __dsub_rn(r, w);
        }
    }
    *y0_out = y0;
    *y1_out = __dsub_rn(__dsub_rn(r, y0), w);
    return n;
}

static __device__ __forceinline__ double neo_ehma_deterministic_sin_v2(double x)
{
    const unsigned int high =
        (unsigned int)((neo_ehma_bits_v2(x) >> 32) & 0x7fffffffULL);
    if (high <= 0x3fe921fbU) {
        return neo_ehma_msun_k_sin_v2(x, 0.0, false);
    }

    double y0;
    double y1;
    const int quadrant = neo_ehma_reduce_pio2_v2(x, &y0, &y1);
    const double sine = neo_ehma_msun_k_sin_v2(y0, y1, true);
    const double cosine = neo_ehma_msun_k_cos_v2(y0, y1);
    switch (quadrant & 3) {
        case 0: return sine;
        case 1: return cosine;
        case 2: return -sine;
        default: return -cosine;
    }
}

static __device__ __forceinline__ double neo_ehma_half_angle_v2(
    int period,
    int k)
{
    const double pi_hi = 0x1.921fb54442d18p+1;
    const double pi_lo = 0x1.1a62633145c07p-53;
    const double denominator = __dadd_rn((double)period, 1.0);
    const double numerator = (double)k;
    const double quotient = __ddiv_rn(numerator, denominator);
    const double quotient_remainder = __fma_rn(-quotient, denominator, numerator);
    const double product = __dmul_rn(quotient, pi_hi);
    const double product_error = __fma_rn(quotient, pi_hi, -product);
    const double correction = __dadd_rn(
        __dadd_rn(product_error, __dmul_rn(quotient, pi_lo)),
        __dmul_rn(__ddiv_rn(quotient_remainder, denominator), pi_hi));
    return __dadd_rn(product, correction);
}

static __device__ __forceinline__ void neo_ehma_dot2_add_product_v2(
    double left,
    double right,
    double* sum,
    double* correction)
{
    const double product = __dmul_rn(left, right);
    const double product_error = __fma_rn(left, right, -product);
    const double updated = __dadd_rn(*sum, product);
    const double recovered = __dsub_rn(updated, *sum);
    const double addition_error = __dadd_rn(
        __dsub_rn(*sum, __dsub_rn(updated, recovered)),
        __dsub_rn(product, recovered));
    *sum = updated;
    *correction = __dadd_rn(
        *correction,
        __dadd_rn(product_error, addition_error));
}

static __device__ __forceinline__ double neo_ehma_build_weights_v2(
    int period,
    double* weights)
{
    const int half = (period + 1) / 2;
    for (int k = 1; k <= half; ++k) {
        const double sine = neo_ehma_deterministic_sin_v2(
            neo_ehma_half_angle_v2(period, k));
        const double weight = __dmul_rn(2.0, __dmul_rn(sine, sine));
        weights[k - 1] = weight;
        weights[period - k] = weight;
    }

    double sum = 0.0;
    double correction = 0.0;
    for (int k = 0; k < period; ++k) {
        neo_ehma_dot2_add_product_v2(1.0, weights[k], &sum, &correction);
    }
    return __dadd_rn(sum, correction);
}

static __device__ __forceinline__ double neo_ehma_floor_power_of_two_scale_v2(
    double max_abs_input)
{
    const unsigned long long bits = neo_ehma_bits_v2(max_abs_input);
    const unsigned long long exponent = (bits >> 52) & 0x7ffULL;
    if (exponent != 0ULL) {
        return __longlong_as_double((long long)(exponent << 52));
    }
    const unsigned long long fraction = bits & ((1ULL << 52) - 1ULL);
    const int highest_bit = 63 - __clzll(fraction);
    return __longlong_as_double((long long)(1ULL << highest_bit));
}

static __device__ __forceinline__ double neo_ehma_stable_window_v2(
    const double* values,
    int period,
    const double* weights,
    double coefficient)
{
    double max_abs_input = 0.0;
    bool has_infinite = false;
    for (int index = 0; index < period; ++index) {
        const unsigned long long bits = neo_ehma_bits_v2(values[index]);
        const unsigned long long absolute_bits = bits & 0x7fffffffffffffffULL;
        if (absolute_bits > 0x7ff0000000000000ULL) {
            return NEO_F64_NAN;
        }
        if (absolute_bits == 0x7ff0000000000000ULL) {
            has_infinite = true;
        } else {
            const double absolute = __longlong_as_double((long long)absolute_bits);
            if (absolute > max_abs_input) max_abs_input = absolute;
        }
    }

    if (has_infinite) {
        double sum = 0.0;
        for (int index = 0; index < period; ++index) {
            sum = __fma_rn(values[index], weights[index], sum);
        }
        const double result = __ddiv_rn(sum, coefficient);
        return neo_ehma_is_nan_bits_v2(neo_ehma_bits_v2(result))
            ? NEO_F64_NAN
            : result;
    }
    if (max_abs_input == 0.0) return 0.0;

    const double scale = neo_ehma_floor_power_of_two_scale_v2(max_abs_input);
    const double anchor = __ddiv_rn(values[0], scale);
    double sum = 0.0;
    double correction = 0.0;
    for (int index = 0; index < period; ++index) {
        const double normalized = __ddiv_rn(values[index], scale);
        neo_ehma_dot2_add_product_v2(
            __dsub_rn(normalized, anchor),
            weights[index],
            &sum,
            &correction);
    }
    const double shifted = __dadd_rn(sum, correction);
    const double result = __dmul_rn(
        scale,
        __dadd_rn(anchor, __ddiv_rn(shifted, coefficient)));
    return result == 0.0 ? 0.0 : result;
}

extern "C" __global__
void ehma_neo_batch_f64(const double* __restrict__ data,
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

    if (len <= 0 || period <= 0 || period > len ||
        period > NEO_EHMA_MAX_PERIOD ||
        first_valid < 0 || first_valid >= len ||
        (len - first_valid) < period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    double w[NEO_EHMA_MAX_PERIOD];
    const double coefficient = neo_ehma_build_weights_v2(period, w);

    const int warm = first_valid + period - 1;
    for (int i = 0; i < len && i < warm; ++i) o[i] = NEO_F64_NAN;
    if (warm >= len) return;

    for (int i = warm; i < len; ++i) {
        const int start = i + 1 - period;
        o[i] = neo_ehma_stable_window_v2(data + start, period, w, coefficient);
    }
}
