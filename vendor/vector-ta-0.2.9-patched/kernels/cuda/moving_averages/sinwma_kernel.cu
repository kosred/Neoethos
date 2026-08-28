#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef SINWMA_BLOCK_X
#define SINWMA_BLOCK_X 256
#endif


static __device__ __forceinline__ float sinwma_inv_norm(int period) {


    const double theta = CUDART_PI / (double(period) + 1.0);
    const double shalf = sin(0.5 * theta);
    const double sn    = sin(0.5 * theta * double(period));
    const double denom = (fabs(shalf) > 1e-20) ? (sn / shalf) : double(period);
    const double inv   = (denom > 0.0) ? (1.0 / denom) : 0.0;
    return (float)inv;
}


static __device__ __forceinline__
void compute_weights_pre_normalized(float* __restrict__ weights, int period) {
    const float theta = CUDART_PI_F / (float(period) + 1.0f);
    const float inv_norm = sinwma_inv_norm(period);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const float angle = (float(i + 1)) * theta;
        weights[i] = sinf(angle) * inv_norm;
    }
}


extern "C" __global__
void sinwma_batch_f32(const float* __restrict__ prices,
                      const int* __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int first_valid,
                      float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;


    extern __shared__ float shmem[];
    float* __restrict__ weights = shmem;
    float* __restrict__ tile    = weights + period;

    const int warm     = first_valid + period - 1;
    const int base_out = combo * series_len;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop   = min(warm, series_len);
        for (; t < stop; t += stride) {
            out[base_out + t] = NAN;
        }
    }


    compute_weights_pre_normalized(weights, period);
    __syncthreads();


    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {


        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + blockDim.x - 1, series_len - 1);

        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len      = (t_end - t_begin + 1) + (period - 1);


            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                tile[i] = prices[tile_in_start + i];
            }
            __syncthreads();


            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start_in_tile = t - t_begin;
                float acc = 0.0f;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(tile[start_in_tile + k], weights[k], acc);
                }
                out[base_out + t] = acc;
            }
            __syncthreads();
        }
    }
}


extern "C" __global__
void sinwma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    if (period <= 0) return;

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    extern __shared__ float shmem[];
    float* __restrict__ weights = shmem;
    float* __restrict__ tile    = weights + period;

    const int warm = first_valids[series_idx] + period - 1;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop   = min(warm, series_len);
        for (; t < stop; t += stride) {
            out_tm[t * num_series + series_idx] = NAN;
        }
    }


    compute_weights_pre_normalized(weights, period);
    __syncthreads();


    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {
        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + blockDim.x - 1, series_len - 1);

        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len      = (t_end - t_begin + 1) + (period - 1);


            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                const int tt = tile_in_start + i;
                tile[i] = prices_tm[tt * num_series + series_idx];
            }
            __syncthreads();

            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start_in_tile = t - t_begin;
                float acc = 0.0f;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(tile[start_in_tile + k], weights[k], acc);
                }
                out_tm[t * num_series + series_idx] = acc;
            }
            __syncthreads();
        }
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

static __device__ __forceinline__ double sinwma_inv_norm_f64(int period) {


    const double theta = CUDART_PI / (double(period) + 1.0);
    const double shalf = sin(0.5 * theta);
    const double sn    = sin(0.5 * theta * double(period));
    const double denom = (fabs(shalf) > 1e-20) ? (sn / shalf) : double(period);
    const double inv   = (denom > 0.0) ? (1.0 / denom) : 0.0;
    return (double)inv;
}
static __device__ __forceinline__
void compute_weights_pre_normalized_f64(double* __restrict__ weights, int period) {
    const double theta = CUDART_PI_F / (double(period) + 1.0);
    const double inv_norm = sinwma_inv_norm_f64(period);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const double angle = (double(i + 1)) * theta;
        weights[i] = sin(angle) * inv_norm;
    }
}
extern "C" __global__
void sinwma_batch_f64(const double* __restrict__ prices,
                      const int* __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;


    extern __shared__ double shmem_f64[];
    double* __restrict__ weights = shmem_f64;
    double* __restrict__ tile    = weights + period;

    const int warm     = first_valid + period - 1;
    const int base_out = combo * series_len;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop   = min(warm, series_len);
        for (; t < stop; t += stride) {
            out[base_out + t] = NAN;
        }
    }


    compute_weights_pre_normalized_f64(weights, period);
    __syncthreads();


    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {


        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + blockDim.x - 1, series_len - 1);

        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len      = (t_end - t_begin + 1) + (period - 1);


            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                tile[i] = prices[tile_in_start + i];
            }
            __syncthreads();


            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start_in_tile = t - t_begin;
                double acc = 0.0;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    // S5 CORRECTION -- ROUNDING COUNT: `sinwma.rs:520-527` is
                    // `sum += d * w`, a separate multiply and add (TWO
                    // roundings). `fma` is ONE. `-fmad=false` keeps this
                    // uncontracted.
                    acc = acc + tile[start_in_tile + k] * weights[k];
                }
                out[base_out + t] = acc;
            }
            __syncthreads();
        }
    }
}
extern "C" __global__
void sinwma_many_series_one_param_time_major_f64(
    const double* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm)
{
    if (period <= 0) return;

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    extern __shared__ double shmem_f64[];
    double* __restrict__ weights = shmem_f64;
    double* __restrict__ tile    = weights + period;

    const int warm = first_valids[series_idx] + period - 1;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop   = min(warm, series_len);
        for (; t < stop; t += stride) {
            out_tm[t * num_series + series_idx] = NAN;
        }
    }


    compute_weights_pre_normalized_f64(weights, period);
    __syncthreads();


    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {
        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + blockDim.x - 1, series_len - 1);

        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len      = (t_end - t_begin + 1) + (period - 1);


            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                const int tt = tile_in_start + i;
                tile[i] = prices_tm[tt * num_series + series_idx];
            }
            __syncthreads();

            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start_in_tile = t - t_begin;
                double acc = 0.0;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    // S5 CORRECTION -- ROUNDING COUNT: `sinwma.rs:520-527` is
                    // `sum += d * w`, a separate multiply and add (TWO
                    // roundings). `fma` is ONE. `-fmad=false` keeps this
                    // uncontracted.
                    acc = acc + tile[start_in_tile + k] * weights[k];
                }
                out_tm[t * num_series + series_idx] = acc;
            }
            __syncthreads();
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `sinwma_with_kernel` (src/indicators/moving_averages/
// sinwma.rs:210) -> `sinwma_prepare` (:307) for the validity rules,
// `build_sinwma_weights` (:273) for the weights and `sinwma_scalar` (:494) for
// the value.
//
// WHY `sinwma_scalar` AND NOT `sinwma_scalar_14`. The period==14
// specialisation (:401) is not a second answer: `sinwma_scalar` with
// period==14 has `p4 == 12`, so it runs three 4-wide chunks associated
// `(((m0+m1)+m2)+m3)` and then adds elements 12 and 13 one at a time -- which
// is term for term what `sinwma_scalar_14` does at :421-425. The only
// difference is that `sinwma_scalar` starts from `sum = 0.0` and adds the
// first chunk, where `_14` assigns it; `0.0 + x == x` for every double except
// that it turns -0.0 into +0.0, which no weighted price sum can be. One
// implementation is therefore faithful to both.
//
// WEIGHTS (:273-292). `angle = (k + 1) * PI / (period + 1)`, `w[k] = sin(angle)`,
// `sum_sines` accumulated in ASCENDING k, then every weight multiplied by
// `inv_sum = 1.0 / sum_sines` -- one multiply per weight, not a divide. The
// sum is a per-ROW constant, so it is computed once before the bar loop; the
// individual sines are recomputed per bar rather than stored, which keeps this
// kernel free of any per-thread array and therefore of any `max_period` bound.
//
// KNOWN NON-BIT-EXACTNESS, stated rather than hidden: `sin` is the one place
// this indicator depends on a transcendental. CUDA's double-precision `sin`
// and Rust's `f64::sin` are both faithfully rounded but are not the same
// implementation, so a weight may differ in the last ULP. That is a property
// of the indicator's definition, not of this transcription -- there is no
// accumulation order or epsilon that could remove it.
//
// WARMUP: `first + period - 1` (:511).
//
// f32 -> f64 audit of this file: the f32 entry points use `sinf`/`__fmaf_rn`
// and f32 literals. Below: `sin`, plain `+`/`*`, the f64 quiet-NaN bit
// pattern, and the f64 PI constant written out to 21 significant digits. No
// f32 literal, no f32-suffixed math function, no fast-math intrinsic. The one
// epsilon on the CPU (:284, `sum_sines.abs() < f64::EPSILON`) is ALREADY an
// f64 epsilon and is carried across as `2.2204460492503131e-16` rather than
// being re-sized or dropped.
// ===========================================================================

static __device__ __forceinline__ double neo_sinwma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

#define NEO_SINWMA_PI  3.14159265358979323846
#define NEO_SINWMA_EPS 2.2204460492503131e-16

extern "C" __global__ void neoethos_sinwma_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_sinwma_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int period = periods[r];
    const int first  = first_valid;

    if (period <= 0 || period > n) return;          // :325
    if (first < 0 || first >= n) return;
    if ((n - first) < period) return;               // :331

    const double denom_angle = static_cast<double>(period) + 1.0;

    // build_sinwma_weights :276-282 -- ascending k, one accumulator.
    double sum_sines = 0.0;
    for (int k = 0; k < period; ++k) {
        const double angle = (static_cast<double>(k) + 1.0) * NEO_SINWMA_PI / denom_angle;
        sum_sines += sin(angle);
    }
    if (fabs(sum_sines) < NEO_SINWMA_EPS) return;   // :284-286 -> ZeroSumSines
    const double inv_sum = 1.0 / sum_sines;         // :287

    const int p4 = period & ~3;

    for (int i = first + period - 1; i < n; ++i) {  // :508
        const int start = i + 1 - period;
        double sum = 0.0;

        // :511-517 -- 4-wide chunks, `(((m0+m1)+m2)+m3)` added to `sum`.
        for (int k = 0; k < p4; k += 4) {
            const double w0 = sin((static_cast<double>(k)       + 1.0) * NEO_SINWMA_PI / denom_angle) * inv_sum;
            const double w1 = sin((static_cast<double>(k + 1)   + 1.0) * NEO_SINWMA_PI / denom_angle) * inv_sum;
            const double w2 = sin((static_cast<double>(k + 2)   + 1.0) * NEO_SINWMA_PI / denom_angle) * inv_sum;
            const double w3 = sin((static_cast<double>(k + 3)   + 1.0) * NEO_SINWMA_PI / denom_angle) * inv_sum;
            sum = sum + ((((data[start + k]     * w0)
                         + (data[start + k + 1] * w1))
                         + (data[start + k + 2] * w2))
                         + (data[start + k + 3] * w3));
        }

        // :519-521 -- tail, one term at a time.
        for (int k = p4; k < period; ++k) {
            const double w = sin((static_cast<double>(k) + 1.0) * NEO_SINWMA_PI / denom_angle) * inv_sum;
            sum = sum + data[start + k] * w;
        }

        row[i] = sum;                               // :523
    }
}
