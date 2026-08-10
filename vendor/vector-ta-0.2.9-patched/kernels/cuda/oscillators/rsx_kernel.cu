#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


static __device__ __forceinline__ float clamp_0_100(float x) {
    x = fminf(x, 100.0f);
    x = fmaxf(x, 0.0f);
    return x;
}


extern "C" __global__
void transpose_tm_to_rm_f32(const float* __restrict__ in_tm,
                            int rows, int cols,
                            float* __restrict__ out_rm)
{
    __shared__ float tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;


    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int yy = y + j;
        if (x < cols && yy < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in_tm[yy * cols + x];
        }
    }
    __syncthreads();


    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;


    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int yy = y + j;
        if (x < rows && yy < cols) {
            out_rm[yy * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


extern "C" __global__
void rsx_batch_tm_f32(const float* __restrict__ prices,
                      const int*   __restrict__ periods,
                      int series_len,
                      int first_valid,
                      int n_combos,
                      float* __restrict__ out_tm) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) {
        for (int t = 0; t < series_len; ++t) out_tm[(size_t)t * (size_t)n_combos + combo] = NAN;
        return;
    }

    const int warm = first_valid + period - 1;


    float f0  = 0.0f;
    float f8  = 0.0f;
    bool  have_init = false;
    const float alpha = 3.0f / (float(period) + 2.0f);
    const float beta  = 1.0f - alpha;
    float f28 = 0.0f, f30 = 0.0f;
    float f38 = 0.0f, f40 = 0.0f;
    float f48 = 0.0f, f50 = 0.0f;
    float f58 = 0.0f, f60 = 0.0f;
    float f68 = 0.0f, f70 = 0.0f;
    float f78 = 0.0f, f80 = 0.0f;
    const float f88 = (period >= 6) ? float(period - 1) : 5.0f;
    float f90 = 1.0f;


    #pragma unroll 1
    for (int t = 0; t < series_len; ++t) {

        unsigned mask = __activemask();
        float p = 0.0f;
        if ((threadIdx.x & 31) == 0) {
            p = __ldg(prices + t);
        }
        p = __shfl_sync(mask, p, 0);
        const float p100 = 100.0f * p;


        if (t <= warm) {
            out_tm[(size_t)t * (size_t)n_combos + combo] = NAN;
            if (t == warm) { f8 = p100; have_init = true; }
            continue;
        }


        f90 = (f88 <= f90) ? (f88 + 1.0f) : (f90 + 1.0f);
        const float prev = f8;
        f8 = p100;
        const float v8 = f8 - prev;


        f28 = beta * f28 + alpha * v8;
        f30 = alpha * f28 + beta * f30;
        const float v_c = 1.5f * f28 - 0.5f * f30;

        f38 = beta * f38 + alpha * v_c;
        f40 = alpha * f38 + beta * f40;
        const float v10 = 1.5f * f38 - 0.5f * f40;

        f48 = beta * f48 + alpha * v10;
        f50 = alpha * f48 + beta * f50;
        const float v14 = 1.5f * f48 - 0.5f * f50;

        const float av = fabsf(v8);
        f58 = beta * f58 + alpha * av;
        f60 = alpha * f58 + beta * f60;
        const float v18 = 1.5f * f58 - 0.5f * f60;

        f68 = beta * f68 + alpha * v18;
        f70 = alpha * f68 + beta * f70;
        const float v1c = 1.5f * f68 - 0.5f * f70;

        f78 = beta * f78 + alpha * v1c;
        f80 = alpha * f78 + beta * f80;
        const float v20_ = 1.5f * f78 - 0.5f * f80;

        if (f88 >= f90 && f8 != prev) { f0 = 1.0f; }
        if (fabsf(f88 - f90) <= 1e-12f && f0 == 0.0f) { f90 = 0.0f; }

        float y = 50.0f;
        if (f88 < f90 && v20_ > 1e-10f && have_init) {
            y = clamp_0_100((v14 / v20_ + 1.0f) * 50.0f);
        }
        out_tm[(size_t)t * (size_t)n_combos + combo] = y;
    }
}


extern "C" __global__
void rsx_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    if (period <= 0) {
        for (int t = 0; t < series_len; ++t) out[base + t] = NAN;
        return;
    }

    const int warm = first_valid + period - 1;


    float f0  = 0.0f;
    float f8  = 0.0f;
    bool  have_init = false;
    const float alpha = 3.0f / (float(period) + 2.0f);
    const float beta  = 1.0f - alpha;
    float f28 = 0.0f, f30 = 0.0f;
    float f38 = 0.0f, f40 = 0.0f;
    float f48 = 0.0f, f50 = 0.0f;
    float f58 = 0.0f, f60 = 0.0f;
    float f68 = 0.0f, f70 = 0.0f;
    float f78 = 0.0f, f80 = 0.0f;
    const float f88 = (period >= 6) ? float(period - 1) : 5.0f;
    float f90 = 1.0f;


    #pragma unroll 1
    for (int t = 0; t < series_len; ++t) {

        unsigned mask = __activemask();
        float p = 0.0f;
        if ((threadIdx.x & 31) == 0) {
            p = __ldg(prices + t);
        }
        p = __shfl_sync(mask, p, 0);
        const float p100 = 100.0f * p;


        if (t <= warm) {
            out[base + t] = NAN;
            if (t == warm) { f8 = p100; have_init = true; }
            continue;
        }


        f90 = (f88 <= f90) ? (f88 + 1.0f) : (f90 + 1.0f);
        const float prev = f8;
        f8 = p100;
        const float v8 = f8 - prev;


        f28 = beta * f28 + alpha * v8;
        f30 = alpha * f28 + beta * f30;
        const float v_c = 1.5f * f28 - 0.5f * f30;

        f38 = beta * f38 + alpha * v_c;
        f40 = alpha * f38 + beta * f40;
        const float v10 = 1.5f * f38 - 0.5f * f40;

        f48 = beta * f48 + alpha * v10;
        f50 = alpha * f48 + beta * f50;
        const float v14 = 1.5f * f48 - 0.5f * f50;

        const float av = fabsf(v8);
        f58 = beta * f58 + alpha * av;
        f60 = alpha * f58 + beta * f60;
        const float v18 = 1.5f * f58 - 0.5f * f60;

        f68 = beta * f68 + alpha * v18;
        f70 = alpha * f68 + beta * f70;
        const float v1c = 1.5f * f68 - 0.5f * f70;

        f78 = beta * f78 + alpha * v1c;
        f80 = alpha * f78 + beta * f80;
        const float v20_ = 1.5f * f78 - 0.5f * f80;

        if (f88 >= f90 && f8 != prev) { f0 = 1.0f; }
        if (fabsf(f88 - f90) <= 1e-12f && f0 == 0.0f) { f90 = 0.0f; }

        float y = 50.0f;
        if (f88 < f90 && v20_ > 1e-10f && have_init) {
            y = clamp_0_100((v14 / v20_ + 1.0f) * 50.0f);
        }
        out[base + t] = y;
    }
}


extern "C" __global__
void rsx_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    if (period <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }
    const int warm = fv + period - 1;


    for (int t = 0; t <= warm && t < rows; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;


    float f0 = 0.0f;
    float f8 = 100.0f * prices_tm[warm * cols + s];
    const float alpha = 3.0f / (float(period) + 2.0f);
    const float beta  = 1.0f - alpha;
    float f28 = 0.0f, f30 = 0.0f;
    float f38 = 0.0f, f40 = 0.0f;
    float f48 = 0.0f, f50 = 0.0f;
    float f58 = 0.0f, f60 = 0.0f;
    float f68 = 0.0f, f70 = 0.0f;
    float f78 = 0.0f, f80 = 0.0f;
    const float f88 = (period >= 6) ? float(period - 1) : 5.0f;
    float f90 = 1.0f;

    for (int t = warm + 1; t < rows; ++t) {
        f90 = (f88 <= f90) ? (f88 + 1.0f) : (f90 + 1.0f);

        const float prev = f8;
        const float cur  = prices_tm[t * cols + s];
        f8 = 100.0f * cur;
        const float v8 = f8 - prev;

        f28 = beta * f28 + alpha * v8;
        f30 = alpha * f28 + beta * f30;
        const float v_c = 1.5f * f28 - 0.5f * f30;

        f38 = beta * f38 + alpha * v_c;
        f40 = alpha * f38 + beta * f40;
        const float v10 = 1.5f * f38 - 0.5f * f40;

        f48 = beta * f48 + alpha * v10;
        f50 = alpha * f48 + beta * f50;
        const float v14 = 1.5f * f48 - 0.5f * f50;

        const float av = fabsf(v8);
        f58 = beta * f58 + alpha * av;
        f60 = alpha * f58 + beta * f60;
        const float v18 = 1.5f * f58 - 0.5f * f60;

        f68 = beta * f68 + alpha * v18;
        f70 = alpha * f68 + beta * f70;
        const float v1c = 1.5f * f68 - 0.5f * f70;

        f78 = beta * f78 + alpha * v1c;
        f80 = alpha * f78 + beta * f80;
        const float v20_ = 1.5f * f78 - 0.5f * f80;

        if (f88 >= f90 && f8 != prev) {
            f0 = 1.0f;
        }
        if (fabsf(f88 - f90) <= 1e-12f && f0 == 0.0f) {
            f90 = 0.0f;
        }

        float y = 50.0f;
        if (f88 < f90 && v20_ > 1e-10f) {
            y = clamp_0_100((v14 / v20_ + 1.0f) * 50.0f);
        }
        out_tm[t * cols + s] = y;
    }
}


// ===========================================================================
// S2 f64 LANE — rsx  (Jurik relative strength)
// ===========================================================================
// Reference: src/indicators/rsx.rs
//   `rsx_prepare`     (:193) — first_valid + refusals
//   `rsx_with_kernel` (:233) — alloc_with_nan_prefix(len, first + period - 1)
//   `rsx_scalar`      (:305) — six cascaded one-pole pairs
//
// SIX COUPLED RECURRENCES, TWELVE CARRIED SCALARS. f28/f30, f38/f40, f48/f50,
// f58/f60, f68/f70, f78/f80 each form a two-pole stage, and each stage's
// output feeds the next. There is no parallel reformulation of this that
// preserves the arithmetic: one thread per column, bars ascending.
//
// TWO EPSILONS, AND BOTH ARE f64-SIZED ON THE CPU — DO NOT COPY AN f32 ONE.
//   * `(f88 - f90).abs() < f64::EPSILON` — 2.220446049250313e-16. `f88` and
//     `f90` are small integers held in doubles, so this is an equality test
//     written as a tolerance; with `FLT_EPSILON` (1.19e-7) it would still test
//     equality here, but the constant would be wrong by nine orders of
//     magnitude and would be copied onward by the next person.
//   * `v20_ > 1e-10` — a genuine magnitude floor on a smoothed absolute
//     change. 1e-10 is representable and meaningful in f64; in f32 it is only
//     three decimal digits above the subnormal boundary for typical `v20_`
//     scales, which is why the f32 kernel's branch flips on bars the CPU's
//     does not. Carried across unchanged, because it is the CPU's number.
//
// THE CLAMPS ARE IFs, NOT fmin/fmax. `if v4 > 100.0 { v4 = 100.0 }` then
// `if v4 < 0.0 { v4 = 0.0 }`: a NaN `v4` passes BOTH and is written out as
// NaN. `fmin(fmax(v4, 0.0), 100.0)` would emit 0.0 instead — a plausible-
// looking number where the CPU says "no answer".
//
// f90's UPDATE IS A COUNTER WITH A CEILING, expressed as a comparison on
// doubles (`if f88 <= f90 { f88 + 1.0 } else { f90 + 1.0 }`). Reproduced as
// written rather than as integer arithmetic.
//
// out[start] IS EXPLICITLY NaN — one bar past the warmup prefix, set by the
// compute itself, not by the allocator.
// ===========================================================================

#define NEO_RSX_F64_EPSILON 2.220446049250313e-16

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_rsx_batch_f64(
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
        (period <= 0) || (period > n) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();

    const int start = first_valid + period - 1;
    if (start >= n) return;

    double f0 = 0.0;
    double f28 = 0.0, f30 = 0.0;
    double f38 = 0.0, f40 = 0.0;
    double f48 = 0.0, f50 = 0.0;
    double f58 = 0.0, f60 = 0.0;
    double f68 = 0.0, f70 = 0.0;
    double f78 = 0.0, f80 = 0.0;

    double f90 = 1.0;
    const double f88 = (period >= 6) ? (double)(period - 1) : 5.0;
    double f8 = 100.0 * prices[start];
    const double f18 = 3.0 / ((double)period + 2.0);
    const double f20 = 1.0 - f18;

    row[start] = neo_s2_qnan();

    for (int i = start + 1; i < n; ++i) {
        f90 = (f88 <= f90) ? (f88 + 1.0) : (f90 + 1.0);

        const double prev = f8;
        f8 = 100.0 * prices[i];
        const double v8 = f8 - prev;

        f28 = f20 * f28 + f18 * v8;
        f30 = f18 * f28 + f20 * f30;
        const double v_c = f28 * 1.5 - f30 * 0.5;

        f38 = f20 * f38 + f18 * v_c;
        f40 = f18 * f38 + f20 * f40;
        const double v10 = f38 * 1.5 - f40 * 0.5;

        f48 = f20 * f48 + f18 * v10;
        f50 = f18 * f48 + f20 * f50;
        const double v14 = f48 * 1.5 - f50 * 0.5;

        const double av = fabs(v8);
        f58 = f20 * f58 + f18 * av;
        f60 = f18 * f58 + f20 * f60;
        const double v18 = f58 * 1.5 - f60 * 0.5;

        f68 = f20 * f68 + f18 * v18;
        f70 = f18 * f68 + f20 * f70;
        const double v1c = f68 * 1.5 - f70 * 0.5;

        f78 = f20 * f78 + f18 * v1c;
        f80 = f18 * f78 + f20 * f80;
        const double v20_ = f78 * 1.5 - f80 * 0.5;

        if (f88 >= f90 && f8 != prev) {
            f0 = 1.0;
        }
        if (fabs(f88 - f90) < NEO_RSX_F64_EPSILON && f0 == 0.0) {
            f90 = 0.0;
        }

        if (f88 < f90 && v20_ > 1e-10) {
            double v4 = (v14 / v20_ + 1.0) * 50.0;
            if (v4 > 100.0) v4 = 100.0;
            if (v4 < 0.0) v4 = 0.0;
            row[i] = v4;
        } else {
            row[i] = 50.0;
        }
    }
}
