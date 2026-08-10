#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


#if !defined(__CUDACC_VER_MAJOR__)
#define __CUDACC_VER_MAJOR__ 0
#endif
#if __CUDACC_VER_MAJOR__ >= 12
#include <cuda/annotated_ptr>
#endif


static __device__ __forceinline__ float qnan() {

    return __int_as_float(0x7fffffff);
}


static __device__ __forceinline__ void prefetch_L2(const void* p) {
#if __CUDA_ARCH__ >= 800
    asm volatile ("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}


extern "C" __global__
void mwdx_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ facs,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }


#if __CUDACC_VER_MAJOR__ >= 12
    const float* __restrict__ prices_persist =
        cuda::associate_access_property(prices, cuda::access_property::persisting{});
#else
    const float* __restrict__ prices_persist = prices;
#endif

    const float fac = facs[combo];
    const float beta = 1.0f - fac;
    const int row_offset = combo * series_len;


    if (first_valid < 0 || first_valid >= series_len) {
        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[row_offset + idx] = qnan();
        }
        return;
    }


    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[row_offset + idx] = qnan();
    }


    if (threadIdx.x == 0) {
        float prev = prices_persist[first_valid];
        out[row_offset + first_valid] = prev;


        const int PDIST = 64;
        for (int t = first_valid + 1; t < series_len; ++t) {
#if __CUDA_ARCH__ >= 800
            int pf = t + PDIST;
            if (pf < series_len) prefetch_L2(prices_persist + pf);
#endif
            const float price = prices_persist[t];
            prev = __fmaf_rn(price, fac, beta * prev);
            out[row_offset + t] = prev;
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    float fac,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[series_idx];


    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + series_idx] = qnan();
        }
        return;
    }


    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * stride + series_idx] = qnan();
    }


    if (threadIdx.x == 0) {
        int offset = first_valid * stride + series_idx;
        float prev = prices_tm[offset];
        out_tm[offset] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            offset = t * stride + series_idx;
            const float price = prices_tm[offset];
            prev = __fmaf_rn(price, fac, beta * prev);
            out_tm[offset] = prev;
        }
    }
}


template<int TX, int TY>
__device__ void mwdx_many_series_one_param_tiled2d_f32_core(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    const int s_base = blockIdx.y * TY;
    const int s_local = s_base + threadIdx.y;
    if (s_local >= num_series || series_len <= 0) return;

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[s_local];

    if (first_valid < 0 || first_valid >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += TX) {
            out_tm[t * stride + s_local] = qnan();
        }
        return;
    }


    for (int t = threadIdx.x; t < first_valid; t += TX) {
        out_tm[t * stride + s_local] = qnan();
    }

    if (threadIdx.x == 0) {
        int off0 = first_valid * stride + s_local;
        float prev = prices_tm[off0];
        out_tm[off0] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            const int off = t * stride + s_local;
            const float price = prices_tm[off];
            prev = __fmaf_rn(price, fac, beta * prev);
            out_tm[off] = prev;
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty2(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 2>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty4(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 4>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
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

static __device__ __forceinline__ double qnan_f64() {

    return __longlong_as_double(0x7fffffffffffffffULL);
}
extern "C" __global__
void mwdx_batch_f64(const double* __restrict__ prices,
                    const double* __restrict__ facs,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    double* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }


#if __CUDACC_VER_MAJOR__ >= 12
    const double* __restrict__ prices_persist =
        cuda::associate_access_property(prices, cuda::access_property::persisting{});
#else
    const double* __restrict__ prices_persist = prices;
#endif

    const double fac = facs[combo];
    const double beta = 1.0 - fac;
    const int row_offset = combo * series_len;


    if (first_valid < 0 || first_valid >= series_len) {
        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[row_offset + idx] = qnan_f64();
        }
        return;
    }


    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[row_offset + idx] = qnan_f64();
    }


    if (threadIdx.x == 0) {
        double prev = prices_persist[first_valid];
        out[row_offset + first_valid] = prev;


        const int PDIST = 64;
        for (int t = first_valid + 1; t < series_len; ++t) {
#if __CUDA_ARCH__ >= 800
            int pf = t + PDIST;
            if (pf < series_len) prefetch_L2(prices_persist + pf);
#endif
            const double price = prices_persist[t];
            prev = __fma_rn(price, fac, beta * prev);
            out[row_offset + t] = prev;
        }
    }
}
extern "C" __global__
void mwdx_many_series_one_param_f64(const double* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    double fac,
                                    int num_series,
                                    int series_len,
                                    double* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    const double beta = 1.0 - fac;
    const int stride = num_series;
    const int first_valid = first_valids[series_idx];


    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + series_idx] = qnan_f64();
        }
        return;
    }


    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * stride + series_idx] = qnan_f64();
    }


    if (threadIdx.x == 0) {
        int offset = first_valid * stride + series_idx;
        double prev = prices_tm[offset];
        out_tm[offset] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            offset = t * stride + series_idx;
            const double price = prices_tm[offset];
            prev = __fma_rn(price, fac, beta * prev);
            out_tm[offset] = prev;
        }
    }
}
template<int TX, int TY>
__device__ void mwdx_many_series_one_param_tiled2d_f64_core(
    const double* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    double fac,
    int num_series,
    int series_len,
    double* __restrict__ out_tm) {
    const int s_base = blockIdx.y * TY;
    const int s_local = s_base + threadIdx.y;
    if (s_local >= num_series || series_len <= 0) return;

    const double beta = 1.0 - fac;
    const int stride = num_series;
    const int first_valid = first_valids[s_local];

    if (first_valid < 0 || first_valid >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += TX) {
            out_tm[t * stride + s_local] = qnan_f64();
        }
        return;
    }


    for (int t = threadIdx.x; t < first_valid; t += TX) {
        out_tm[t * stride + s_local] = qnan_f64();
    }

    if (threadIdx.x == 0) {
        int off0 = first_valid * stride + s_local;
        double prev = prices_tm[off0];
        out_tm[off0] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            const int off = t * stride + s_local;
            const double price = prices_tm[off];
            prev = __fma_rn(price, fac, beta * prev);
            out_tm[off] = prev;
        }
    }
}
extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f64_tx128_ty2(
    const double* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    double fac,
    int num_series,
    int series_len,
    double* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f64_core<128, 2>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}
extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f64_tx128_ty4(
    const double* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    double fac,
    int num_series,
    int series_len,
    double* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f64_core<128, 4>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                                      mwdx
 * ---------------------------------------------------------------------------
 * CPU reference: `mwdx_scalar`, src/indicators/moving_averages/mwdx.rs:284,
 * reached through `mwdx_prepare` (:206) and `mwdx_compute_into` (:239).
 *
 * PERIOD-INVARIANT, and that is the indicator, not a shortcut. `mwdx` has NO
 * period parameter at all: `MwdxParams` (:80) carries a single `factor`, whose
 * default is 0.2 (`get_factor`, :118-119). The lane sweeps `periods`; every row
 * of this kernel is therefore the same series, exactly as the CPU batch would
 * produce it for a period list. The `(void)periods` below is the honest
 * statement of that, and it matches how `ewma_volatility_neo_batch_f64` in
 * kernels/cuda/ewma_volatility_kernel.cu handles its own fixed `lambda`.
 *
 * SEQUENTIAL, one thread per column. The recurrence is
 * `y = (x - prev).mul_add(fac, prev)` -- ONE rounding -- so it becomes
 * `fma(x - prev, fac, prev)`. Writing it as `(x - prev) * fac + prev` would be
 * two roundings and would drift across a 200k-bar series.
 *
 * The CPU unrolls that recurrence two bars at a time (:312-323) but does NOT
 * change its shape: `y0` is formed from the carried `prev` and `y1` from `y0`,
 * so the accumulation is strictly bar-by-bar and this single-step loop is
 * bit-identical to it.
 *
 * NaN: the CPU writes NaN for the leading NaN run (:293-302), seeds from the
 * first non-NaN bar (:308-309) and then NEVER tests for NaN again. A NaN
 * appearing later therefore POISONS every subsequent bar through `prev`. That
 * is reproduced deliberately -- adding a mid-stream guard here would compute a
 * different series than the crate does.
 *
 * `first_valid` is the lane's `AllInputsNonNan` over the close slice, which is
 * exactly the CPU's `data.iter().position(|x| !x.is_nan())` (:228).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* MwdxParams::default -> factor 0.2, mwdx.rs:85 and :119. */
#define NEO_MWDX_FACTOR 0.2

extern "C" __global__
void mwdx_neo_batch_f64(const double* __restrict__ data,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;   /* mwdx has no period -- see the header above. */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    if (first_valid < 0 || first_valid >= n) return;

    const double fac = NEO_MWDX_FACTOR;

    /* mwdx.rs:308-310 -- the first non-NaN bar is written through unchanged. */
    double prev = data[first_valid];
    o[first_valid] = prev;

    /* mwdx.rs:312-328 -- one fma per bar, prev carried. */
    for (int i = first_valid + 1; i < n; ++i) {
        const double y = fma(data[i] - prev, fac, prev);
        o[i] = y;
        prev = y;
    }
}
