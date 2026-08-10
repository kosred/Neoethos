#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#endif
#include <math.h>


#ifndef TRIMA_TILE
#define TRIMA_TILE 256
#endif
#ifndef TRIMA_TS
#define TRIMA_TS 128
#endif
#ifndef TRIMA_TT
#define TRIMA_TT 64
#endif


extern "C" __global__
void trima_batch_f32(const float* __restrict__ prices,
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

    const int warm = warm_indices[combo];

    extern __shared__ float weights[];
    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const float inv_norm = 1.0f / float(m1 * m2);
    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        int w = (idx < m1) ? (idx + 1) : (idx < m2 ? m1 : (m1 + m2 - 1) - idx);
        if (w < 0) w = 0;
        weights[idx] = float(w) * inv_norm;
    }
    __syncthreads();

    const int base_out = combo * series_len;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}


extern "C" __global__
void trima_batch_f32_tiled(const float* __restrict__ prices,
                           const int* __restrict__ periods,
                           const int* __restrict__ warm_indices,
                           int series_len,
                           int n_combos,
                           int max_period,
                           float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int warm = warm_indices[combo];


    extern __shared__ float smem[];
    float* __restrict__ weights = smem;
    float* __restrict__ tile    = smem + max_period;


    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const float inv_norm = 1.0f / float(m1 * m2);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        int w;
        if (i < m1)       w = i + 1;
        else if (i < m2)  w = m1;
        else              w = (m1 + m2 - 1) - i;
        weights[i] = float(w > 0 ? w : 0) * inv_norm;
    }
    __syncthreads();


    const int TILE = blockDim.x;
    const int t0   = blockIdx.x * TILE;
    if (t0 >= series_len) return;
    const int t1   = min(t0 + TILE, series_len);

    const int tile_base = max(t0 - period + 1, 0);
    const int tile_end  = t1 - 1;
    const int tile_len  = tile_end - tile_base + 1;


    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        tile[i] = prices[tile_base + i];
    }
    __syncthreads();


    const int t = t0 + threadIdx.x;
    if (t < t1) {
        const int out_idx = combo * series_len + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            const int start_global = t - period + 1;
            const int start_local  = start_global - tile_base;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(tile[start_local + k], weights[k], acc);
            }
            out[out_idx] = acc;
        }
    }
}


extern "C" __global__
void trima_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                      const float* __restrict__ weights,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ first_valids,
                                      float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) shared_weights[i] = weights[i];
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int warm = first_valids[series_idx] + period - 1;
    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
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
void trima_multi_series_one_param_f32_tm_tiled(const float* __restrict__ prices_tm,
                                               const float* __restrict__ weights_in,
                                               int period,
                                               int num_series,
                                               int series_len,
                                               const int* __restrict__ first_valids,
                                               float* __restrict__ out_tm) {

    extern __shared__ float smem[];
    float* __restrict__ w    = smem;
    float* __restrict__ tile = smem + period;


    for (int i = threadIdx.x; i < period; i += blockDim.x) w[i] = weights_in[i];
    __syncthreads();


    const int s0 = blockIdx.x * TRIMA_TS;
    const int s  = s0 + threadIdx.x;
    if (s >= num_series) return;

    const int t0 = blockIdx.y * TRIMA_TT;
    if (t0 >= series_len) return;
    const int t1 = min(t0 + TRIMA_TT, series_len);


    const int base  = max(t0 - period + 1, 0);
    const int rows  = t1 - base;


    for (int r = 0; r < rows; ++r) {
        const int t = base + r;
        if (s < num_series) {
            tile[r * TRIMA_TS + threadIdx.x] = prices_tm[t * num_series + s];
        }
    }
    __syncthreads();


    const int warm = first_valids[s] + period - 1;
    for (int t = t0; t < t1; ++t) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start_row = (t - period + 1) - base;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(tile[(start_row + k) * TRIMA_TS + threadIdx.x], w[k], acc);
            }
            out_tm[out_idx] = acc;
        }
    }
}


extern "C" __global__
void sma_from_prefix_exclusive_f32(const float* __restrict__ P,
                                   int series_len,
                                   int m1,
                                   int warm_first_valid,
                                   float* __restrict__ A) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= series_len) return;
    int warm = warm_first_valid + (m1 - 1);
    if (t < warm) {
        A[t] = NAN;
    } else {
        float sum = P[t + 1] - P[t + 1 - m1];
        A[t] = sum * (1.0f / float(m1));
    }
}

extern "C" __global__
void trima_from_prefix_exclusive_f32(const float* __restrict__ PA,
                                     int series_len,
                                     int m2,
                                     int warm_after_first_sma,
                                     float* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= series_len) return;
    int warm = warm_after_first_sma + (m2 - 1);
    if (t < warm) {
        out[t] = NAN;
    } else {
        float sum = PA[t + 1] - PA[t + 1 - m2];
        out[t] = sum * (1.0f / float(m2));
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

extern "C" __global__
void trima_batch_f64(const double* __restrict__ prices,
                     const int* __restrict__ periods,
                     const int* __restrict__ warm_indices,
                     int series_len,
                     int n_combos,
                     int max_period,
                     double* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int warm = warm_indices[combo];

    extern __shared__ double weights[];
    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const double inv_norm = 1.0 / double(m1 * m2);
    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        int w = (idx < m1) ? (idx + 1) : (idx < m2 ? m1 : (m1 + m2 - 1) - idx);
        if (w < 0) w = 0;
        weights[idx] = double(w) * inv_norm;
    }
    __syncthreads();

    const int base_out = combo * series_len;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            double acc = 0.0;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fma(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}
extern "C" __global__
void trima_batch_f64_tiled(const double* __restrict__ prices,
                           const int* __restrict__ periods,
                           const int* __restrict__ warm_indices,
                           int series_len,
                           int n_combos,
                           int max_period,
                           double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int warm = warm_indices[combo];


    extern __shared__ double smem[];
    double* __restrict__ weights = smem;
    double* __restrict__ tile    = smem + max_period;


    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const double inv_norm = 1.0 / double(m1 * m2);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        int w;
        if (i < m1)       w = i + 1;
        else if (i < m2)  w = m1;
        else              w = (m1 + m2 - 1) - i;
        weights[i] = double(w > 0 ? w : 0) * inv_norm;
    }
    __syncthreads();


    const int TILE = blockDim.x;
    const int t0   = blockIdx.x * TILE;
    if (t0 >= series_len) return;
    const int t1   = min(t0 + TILE, series_len);

    const int tile_base = max(t0 - period + 1, 0);
    const int tile_end  = t1 - 1;
    const int tile_len  = tile_end - tile_base + 1;


    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        tile[i] = prices[tile_base + i];
    }
    __syncthreads();


    const int t = t0 + threadIdx.x;
    if (t < t1) {
        const int out_idx = combo * series_len + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            const int start_global = t - period + 1;
            const int start_local  = start_global - tile_base;
            double acc = 0.0;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fma(tile[start_local + k], weights[k], acc);
            }
            out[out_idx] = acc;
        }
    }
}
extern "C" __global__
void trima_multi_series_one_param_f64(const double* __restrict__ prices_tm,
                                      const double* __restrict__ weights,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ first_valids,
                                      double* __restrict__ out_tm) {
    extern __shared__ double shared_weights[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) shared_weights[i] = weights[i];
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int warm = first_valids[series_idx] + period - 1;
    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            double acc = 0.0;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                acc = fma(prices_tm[in_idx], shared_weights[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}
extern "C" __global__
void trima_multi_series_one_param_f64_tm_tiled(const double* __restrict__ prices_tm,
                                               const double* __restrict__ weights_in,
                                               int period,
                                               int num_series,
                                               int series_len,
                                               const int* __restrict__ first_valids,
                                               double* __restrict__ out_tm) {

    extern __shared__ double smem[];
    double* __restrict__ w    = smem;
    double* __restrict__ tile = smem + period;


    for (int i = threadIdx.x; i < period; i += blockDim.x) w[i] = weights_in[i];
    __syncthreads();


    const int s0 = blockIdx.x * TRIMA_TS;
    const int s  = s0 + threadIdx.x;
    if (s >= num_series) return;

    const int t0 = blockIdx.y * TRIMA_TT;
    if (t0 >= series_len) return;
    const int t1 = min(t0 + TRIMA_TT, series_len);


    const int base  = max(t0 - period + 1, 0);
    const int rows  = t1 - base;


    for (int r = 0; r < rows; ++r) {
        const int t = base + r;
        if (s < num_series) {
            tile[r * TRIMA_TS + threadIdx.x] = prices_tm[t * num_series + s];
        }
    }
    __syncthreads();


    const int warm = first_valids[s] + period - 1;
    for (int t = t0; t < t1; ++t) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start_row = (t - period + 1) - base;
            double acc = 0.0;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fma(tile[(start_row + k) * TRIMA_TS + threadIdx.x], w[k], acc);
            }
            out_tm[out_idx] = acc;
        }
    }
}
extern "C" __global__
void sma_from_prefix_exclusive_f64(const double* __restrict__ P,
                                   int series_len,
                                   int m1,
                                   int warm_first_valid,
                                   double* __restrict__ A) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= series_len) return;
    int warm = warm_first_valid + (m1 - 1);
    if (t < warm) {
        A[t] = NAN;
    } else {
        double sum = P[t + 1] - P[t + 1 - m1];
        A[t] = sum * (1.0 / double(m1));
    }
}
extern "C" __global__
void trima_from_prefix_exclusive_f64(const double* __restrict__ PA,
                                     int series_len,
                                     int m2,
                                     int warm_after_first_sma,
                                     double* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= series_len) return;
    int warm = warm_after_first_sma + (m2 - 1);
    if (t < warm) {
        out[t] = NAN;
    } else {
        double sum = PA[t + 1] - PA[t + 1 - m2];
        out[t] = sum * (1.0 / double(m2));
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — trima                                       (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/moving_averages/trima.rs
 *   :277 `trima_scalar_optimized`   <- the body reproduced here
 *   :193 `trima_prepare`            m1 = (period+1)/2, m2 = period - m1 + 1,
 *                                   period <= 3 is an ERROR, not a shorter MA
 *   :425 `trima_with_kernel`        warm = first + period - 1
 *
 * PERIOD-SWEPT, unlike most of this closer's set: `trima` has NO arm in
 * `cpu_batch.rs`, so its oracle is the single-series function, which reads
 * `period` from the input. Every row of the sweep is a DIFFERENT column.
 *
 * WHY A SECOND f64 ENTRY POINT. `trima_batch_f64` above is the crate's own
 * prefix-sum shape and takes a different argument list; the f64 lane launches
 * one fixed six-argument signature.
 *
 * SEED ASSOCIATION IS LOAD-BEARING. trima.rs:301-310 sums the first `m1`
 * values in 4-WIDE GROUPS -- `sum1 += a + b + c + d` -- and only then walks
 * the remainder one at a time. That is a different rounding from a plain
 * ascending `sum1 += x`, so it is reproduced group for group.
 *
 * SEQUENTIAL, one thread per column: both sums are incremental
 * (`sum1 += new - old`, `sum2 += new_s1 - old_s1`), so a parallel recompute
 * would be a different number, not a faster one.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Must match `TRIMA_MAX_PERIOD` in src/cuda/neoethos_f64_wrapper.rs. The ring
 * is `m2 = period - (period+1)/2 + 1` deep, so 512 bounds it at 257 slots. An
 * oversized period is REFUSED BY NAME by the host rather than truncated. */
#define NEO_TRIMA_MAX_PERIOD 512
#define NEO_TRIMA_MAX_RING   257

extern "C" __global__
void trima_neo_batch_f64(const double* __restrict__ data,
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

    // trima_prepare: period == 0 || period > len -> InvalidPeriod;
    // period <= 3 -> PeriodTooSmall; (len - first) < period -> NotEnoughValidData.
    // Every one of those is an Err on the CPU, i.e. no column at all, so the
    // device answer is a NaN column rather than a shorter moving average.
    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        period <= 3 || period > len || period > NEO_TRIMA_MAX_PERIOD ||
        (len - first_valid) < period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const int warm = first_valid + period - 1;

    for (int i = 0; i < len && i < warm; ++i) o[i] = NEO_F64_NAN;
    if (warm >= len) return;

    const double inv_m1 = 1.0 / (double)m1;
    const double inv_m2 = 1.0 / (double)m2;

    // trima.rs:301-310 -- 4-wide groups first, then the tail.
    double sum1 = 0.0;
    {
        int j = 0;
        const int end_unroll = m1 & ~3;
        while (j < end_unroll) {
            sum1 += data[first_valid + j]     + data[first_valid + j + 1]
                  + data[first_valid + j + 2] + data[first_valid + j + 3];
            j += 4;
        }
        while (j < m1) {
            sum1 += data[first_valid + j];
            j += 1;
        }
    }

    double ring[NEO_TRIMA_MAX_RING];
    int ring_len = 0;
    double sum2 = 0.0;

    int t = first_valid + m1 - 1;
    int p_new = first_valid + m1;
    int p_old = first_valid;

    {
        const double s1 = sum1 * inv_m1;
        ring[ring_len++] = s1;
        sum2 += s1;
    }

    while (ring_len < m2) {
        t += 1;
        sum1 += data[p_new] - data[p_old];
        p_new += 1;
        p_old += 1;
        const double s1 = sum1 * inv_m1;
        ring[ring_len++] = s1;
        sum2 += s1;
    }

    o[warm] = sum2 * inv_m2;

    int head = 0;
    t += 1;
    while (t < len) {
        sum1 += data[p_new] - data[p_old];
        p_new += 1;
        p_old += 1;

        const double new_s1 = sum1 * inv_m1;
        const double old_s1 = ring[head];
        sum2 += new_s1 - old_s1;
        ring[head] = new_s1;

        head += 1;
        if (head == m2) head = 0;

        o[t] = sum2 * inv_m2;
        t += 1;
    }
}
