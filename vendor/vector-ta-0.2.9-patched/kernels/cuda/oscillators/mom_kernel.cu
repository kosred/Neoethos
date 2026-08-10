#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void mom_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base   = combo * series_len;
    const int period = periods[combo];


    if (series_len <= 0 || first_valid >= series_len || period <= 0) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    int warm = first_valid + period;
    if (warm >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }


    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }


    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float cur  = __ldg(&prices[t]);
        const float prev = __ldg(&prices[t - period]);
        out[base + t] = cur - prev;
    }
}


extern "C" __global__
void mom_batch_tiled_f32(const float* __restrict__ prices,
                         const int*   __restrict__ periods,
                         int series_len,
                         int first_valid,
                         int n_combos,
                         float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    const int period = periods[combo];
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (series_len <= 0 || first_valid >= series_len || period <= 0) {
        for (int t = offset; t < series_len; t += stride) {
            out[base + t] = NAN;
        }
        return;
    }

    int warm = first_valid + period;
    if (warm >= series_len) {
        for (int t = offset; t < series_len; t += stride) {
            out[base + t] = NAN;
        }
        return;
    }

    for (int t = offset; t < warm; t += stride) {
        out[base + t] = NAN;
    }

    for (int t = warm + offset; t < series_len; t += stride) {
        const float cur  = __ldg(&prices[t]);
        const float prev = __ldg(&prices[t - period]);
        out[base + t] = cur - prev;
    }
}


extern "C" __global__
void mom_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    if (cols <= 0 || rows <= 0) return;
    if (period <= 0) return;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv   = first_valids[s];
        const int warm = fv + period;


        if (fv < 0 || fv >= rows || warm >= rows) {
            for (int t = 0; t < rows; ++t) {
                out_tm[t * cols + s] = NAN;
            }
            continue;
        }


        for (int t = 0; t < warm; ++t) {
            out_tm[t * cols + s] = NAN;
        }


        for (int t = warm; t < rows; ++t) {
            const int idx = t * cols + s;
            out_tm[idx] = __ldg(&prices_tm[idx]) - __ldg(&prices_tm[(t - period) * cols + s]);
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

extern "C" __global__
void mom_batch_f64(const double* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   double* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base   = combo * series_len;
    const int period = periods[combo];


    if (series_len <= 0 || first_valid >= series_len || period <= 0) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    int warm = first_valid + period;
    if (warm >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }


    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }


    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const double cur  = __ldg(&prices[t]);
        const double prev = __ldg(&prices[t - period]);
        out[base + t] = cur - prev;
    }
}
extern "C" __global__
void mom_batch_tiled_f64(const double* __restrict__ prices,
                         const int*   __restrict__ periods,
                         int series_len,
                         int first_valid,
                         int n_combos,
                         double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    const int period = periods[combo];
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (series_len <= 0 || first_valid >= series_len || period <= 0) {
        for (int t = offset; t < series_len; t += stride) {
            out[base + t] = NAN;
        }
        return;
    }

    int warm = first_valid + period;
    if (warm >= series_len) {
        for (int t = offset; t < series_len; t += stride) {
            out[base + t] = NAN;
        }
        return;
    }

    for (int t = offset; t < warm; t += stride) {
        out[base + t] = NAN;
    }

    for (int t = warm + offset; t < series_len; t += stride) {
        const double cur  = __ldg(&prices[t]);
        const double prev = __ldg(&prices[t - period]);
        out[base + t] = cur - prev;
    }
}
extern "C" __global__
void mom_many_series_one_param_f64(const double* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   double* __restrict__ out_tm)
{
    if (cols <= 0 || rows <= 0) return;
    if (period <= 0) return;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv   = first_valids[s];
        const int warm = fv + period;


        if (fv < 0 || fv >= rows || warm >= rows) {
            for (int t = 0; t < rows; ++t) {
                out_tm[t * cols + s] = NAN;
            }
            continue;
        }


        for (int t = 0; t < warm; ++t) {
            out_tm[t * cols + s] = NAN;
        }


        for (int t = warm; t < rows; ++t) {
            const int idx = t * cols + s;
            out_tm[idx] = __ldg(&prices_tm[idx]) - __ldg(&prices_tm[(t - period) * cols + s]);
        }
    }
}
