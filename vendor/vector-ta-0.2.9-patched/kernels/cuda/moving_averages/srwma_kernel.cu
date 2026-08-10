#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SRWMA_USE_ASYNC_COPY
#define SRWMA_USE_ASYNC_COPY 0
#endif

#if SRWMA_USE_ASYNC_COPY
  #include <cuda/pipeline>
#endif


extern "C" __global__
void srwma_batch_f32(const float* __restrict__ prices,
                     const float* __restrict__ weights_flat,
                     const int*   __restrict__ periods,
                     const int*   __restrict__ warm_indices,
                     const float* __restrict__ inv_norms,
                     int max_wlen,
                     int series_len,
                     int n_combos,
                     float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const int wlen = period - 1;
    const int warm = warm_indices[combo];
    const int start_t = max(warm, wlen - 1);
    const int row_offset = combo * series_len;
    const float inv_norm = inv_norms[combo];

    extern __shared__ float smem[];
    float* __restrict__ w_rev = smem;
    float* __restrict__ tile  = smem + max_wlen;


    const int wbase = combo * max_wlen;
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {

        w_rev[k] = weights_flat[wbase + (wlen - 1 - k)];
    }
    __syncthreads();


    const int tile_span = blockDim.x + wlen - 1;
    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {

        const int t0 = base - (wlen - 1);


        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src = t0 + i;
            float v = 0.0f;
            if (static_cast<unsigned>(src) < static_cast<unsigned>(series_len))
                v = prices[src];
            tile[i] = v;
        }


#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)


#endif

        __syncthreads();


        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int out_idx = row_offset + t;
            if (t < start_t) {
                out[out_idx] = NAN;
            } else {

                const float* __restrict__ win = tile + threadIdx.x;
                float acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fmaf_rn(win[k], w_rev[k], acc);
                }
                out[out_idx] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}


#ifndef SRWMA_USE_CONST_WEIGHTS
#define SRWMA_USE_CONST_WEIGHTS 0
#endif
#if SRWMA_USE_CONST_WEIGHTS
__constant__ float srwma_const_w[4096];
#endif

extern "C" __global__
void srwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
#if SRWMA_USE_CONST_WEIGHTS
                                     const float* __restrict__ weights_unused,
#else
                                     const float* __restrict__ weights,
#endif
                                     int period,
                                     float inv_norm,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) return;
    if (period <= 1) return;

    const int wlen = period - 1;
    const int first_valid = first_valids[series_idx];

    const int warm = first_valid + period + 1;
    const int start_t = max(warm, wlen - 1);

    const int stride = num_series;

    extern __shared__ float smem[];
    float* __restrict__ w_rev = smem;
    float* __restrict__ tile  = smem + wlen;


#if SRWMA_USE_CONST_WEIGHTS


    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = srwma_const_w[wlen - 1 - k];
    }
#else
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = weights[wlen - 1 - k];
    }
#endif
    __syncthreads();

    const int tile_span = blockDim.x + wlen - 1;

    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {
        const int t0 = base - (wlen - 1);


        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src_t = t0 + i;
            float v = 0.0f;
            if (static_cast<unsigned>(src_t) < static_cast<unsigned>(series_len)) {
                v = prices_tm[src_t * stride + series_idx];
            }
            tile[i] = v;
        }

#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)


#endif

        __syncthreads();

        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int offset = t * stride + series_idx;
            if (t < start_t) {
                out_tm[offset] = NAN;
            } else {
                const float* __restrict__ win = tile + threadIdx.x;
                float acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fmaf_rn(win[k], w_rev[k], acc);
                }
                out_tm[offset] = acc * inv_norm;
            }
        }
        __syncthreads();
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
void srwma_batch_f64(const double* __restrict__ prices,
                     const double* __restrict__ weights_flat,
                     const int*   __restrict__ periods,
                     const int*   __restrict__ warm_indices,
                     const double* __restrict__ inv_norms,
                     int max_wlen,
                     int series_len,
                     int n_combos,
                     double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const int wlen = period - 1;
    const int warm = warm_indices[combo];
    const int start_t = max(warm, wlen - 1);
    const int row_offset = combo * series_len;
    const double inv_norm = inv_norms[combo];

    extern __shared__ double smem_f64[];
    double* __restrict__ w_rev = smem_f64;
    double* __restrict__ tile  = smem_f64 + max_wlen;


    const int wbase = combo * max_wlen;
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {

        w_rev[k] = weights_flat[wbase + (wlen - 1 - k)];
    }
    __syncthreads();


    const int tile_span = blockDim.x + wlen - 1;
    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {

        const int t0 = base - (wlen - 1);


        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src = t0 + i;
            double v = 0.0;
            if (static_cast<unsigned>(src) < static_cast<unsigned>(series_len))
                v = prices[src];
            tile[i] = v;
        }


#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)


#endif

        __syncthreads();


        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int out_idx = row_offset + t;
            if (t < start_t) {
                out[out_idx] = NAN;
            } else {

                const double* __restrict__ win = tile + threadIdx.x;
                double acc = 0.0;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fma_rn(win[k], w_rev[k], acc);
                }
                out[out_idx] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}
__constant__ double srwma_const_w[4096];
extern "C" __global__
void srwma_many_series_one_param_f64(const double* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
#if SRWMA_USE_CONST_WEIGHTS
                                     const double* __restrict__ weights_unused,
#else
                                     const double* __restrict__ weights,
#endif
                                     int period,
                                     double inv_norm,
                                     int num_series,
                                     int series_len,
                                     double* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) return;
    if (period <= 1) return;

    const int wlen = period - 1;
    const int first_valid = first_valids[series_idx];

    const int warm = first_valid + period + 1;
    const int start_t = max(warm, wlen - 1);

    const int stride = num_series;

    extern __shared__ double smem_f64[];
    double* __restrict__ w_rev = smem_f64;
    double* __restrict__ tile  = smem_f64 + wlen;


#if SRWMA_USE_CONST_WEIGHTS


    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = srwma_const_w[wlen - 1 - k];
    }
#else
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = weights[wlen - 1 - k];
    }
#endif
    __syncthreads();

    const int tile_span = blockDim.x + wlen - 1;

    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {
        const int t0 = base - (wlen - 1);


        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src_t = t0 + i;
            double v = 0.0;
            if (static_cast<unsigned>(src_t) < static_cast<unsigned>(series_len)) {
                v = prices_tm[src_t * stride + series_idx];
            }
            tile[i] = v;
        }

#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)


#endif

        __syncthreads();

        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int offset = t * stride + series_idx;
            if (t < start_t) {
                out_tm[offset] = NAN;
            } else {
                const double* __restrict__ win = tile + threadIdx.x;
                double acc = 0.0;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fma_rn(win[k], w_rev[k], acc);
                }
                out_tm[offset] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `srwma_with_kernel` (src/indicators/moving_averages/
// srwma.rs:190) for the weights, the norm and the validity rules;
// `srwma_scalar` (:287) for the value. `Kernel::Auto` resolves through
// `detect_best_kernel`, but every non-scalar arm at :244-256 delegates to
// `srwma_scalar` for `period <= 32` and `Kernel::ScalarBatch` -- the path
// `hpc_ta` takes -- always does, so `srwma_scalar` is the oracle.
//
// WEIGHTS (:220-228). `wlen = period - 1`; `w[i] = sqrt(period - i)` for
// i in 0..wlen, so the weights run sqrt(period) down to sqrt(2). `norm` is
// accumulated in ASCENDING i and `inv_norm = 1.0 / norm` -- a reciprocal, then
// ONE multiply at the end of each bar, never a divide per bar.
//
// ROUNDING STRUCTURE -- this is the part that cannot be simplified. The CPU
// keeps EIGHT independent accumulators s0..s7 and folds them only at the end:
//   sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))          (:388)
// Every term enters through `mul_add` -- ONE rounding per term, not two -- and
// which accumulator a term lands in depends on `k % 8` in the 8-wide loop,
// `k % 4` in the 4-wide loop and s0 alone in the tail. Collapsing this to a
// single accumulator, or to `+ x*w` instead of `fma`, changes the answer. The
// loop structure below is the same three loops in the same order.
//
// SHAPE: one thread per column. Not because of a carried recurrence -- each
// bar's window is independent -- but because the eight-accumulator fold is
// per-bar work that is already O(period), and splitting a bar across threads
// would require a reduction whose order is not the CPU's.
//
// WARMUP: `first + period + 1` (:229), TWO bars later than the
// `first + period - 1` most weighted moving averages in this crate use,
// because `srwma_scalar` starts its emit loop at `start_idx = first_val +
// period + 1` (:306). The validity rule matches: `(len - first) < period + 1`
// is refused (:216).
//
// PERIOD == 1 is passed through faithfully rather than special-cased: the CPU
// gets `wlen == 0`, so `norm == 0.0`, `inv_norm == +inf`, every `sum == 0.0`,
// and `0.0 * inf == NaN`. The kernel reproduces that exactly instead of
// inventing a value the host never produces.
//
// f32 -> f64 audit of this file: the f32 entry points use `sqrtf`,
// `__fmaf_rn` and f32 literals. Below: `sqrt`, `fma`, the f64 quiet-NaN bit
// pattern. No f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. No epsilon exists in this indicator on the CPU and none was
// invented.
// ===========================================================================

static __device__ __forceinline__ double neo_srwma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

static __device__ __forceinline__ double neo_srwma_weight(int period, int k) {
    // `((period - i) as f64).sqrt()` (:224).
    return sqrt(static_cast<double>(period - k));
}

extern "C" __global__ void neoethos_srwma_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_srwma_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int period = periods[r];
    const int first  = first_valid;

    if (period <= 0 || period > n) return;              // :208
    if (first < 0 || first >= n) return;
    if ((n - first) < period + 1) return;               // :216

    const int wlen = period - 1;

    double norm = 0.0;                                   // :223-227, ascending
    for (int k = 0; k < wlen; ++k) norm += neo_srwma_weight(period, k);
    const double inv_norm = 1.0 / norm;                  // :229

    const int start_idx = first + period + 1;            // :306

    for (int i = start_idx; i < n; ++i) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        double s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;

        int k = 0;
        while (k + 8 <= wlen) {                          // :325-357
            s0 = fma(data[i - (k + 0)], neo_srwma_weight(period, k + 0), s0);
            s1 = fma(data[i - (k + 1)], neo_srwma_weight(period, k + 1), s1);
            s2 = fma(data[i - (k + 2)], neo_srwma_weight(period, k + 2), s2);
            s3 = fma(data[i - (k + 3)], neo_srwma_weight(period, k + 3), s3);
            s4 = fma(data[i - (k + 4)], neo_srwma_weight(period, k + 4), s4);
            s5 = fma(data[i - (k + 5)], neo_srwma_weight(period, k + 5), s5);
            s6 = fma(data[i - (k + 6)], neo_srwma_weight(period, k + 6), s6);
            s7 = fma(data[i - (k + 7)], neo_srwma_weight(period, k + 7), s7);
            k += 8;
        }
        while (k + 4 <= wlen) {                          // :360-378
            s0 = fma(data[i - (k + 0)], neo_srwma_weight(period, k + 0), s0);
            s1 = fma(data[i - (k + 1)], neo_srwma_weight(period, k + 1), s1);
            s2 = fma(data[i - (k + 2)], neo_srwma_weight(period, k + 2), s2);
            s3 = fma(data[i - (k + 3)], neo_srwma_weight(period, k + 3), s3);
            k += 4;
        }
        while (k < wlen) {                               // :380-385
            s0 = fma(data[i - k], neo_srwma_weight(period, k), s0);
            ++k;
        }

        const double sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));  // :388
        row[i] = sum * inv_norm;                         // :389
    }
}
