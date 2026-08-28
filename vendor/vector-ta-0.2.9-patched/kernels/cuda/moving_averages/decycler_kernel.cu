#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__
void decycler_batch_f32(const float* __restrict__ prices,
                        const int*   __restrict__ periods,
                        const float* __restrict__ c_vals,
                        const float* __restrict__ two_1m_vals,
                        const float* __restrict__ neg_oma_sq_vals,
                        int series_len,
                        int n_combos,
                        int first_valid,
                        float* __restrict__ out)
{
    (void)periods;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int combo = tid; combo < n_combos; combo += blockDim.x * gridDim.x) {
        float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

        if (series_len <= 0) continue;


        if (first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out_row[i] = CUDART_NAN_F;
            continue;
        }

        const float c          = c_vals[combo];
        const float two_1m     = two_1m_vals[combo];
        const float neg_oma_sq = neg_oma_sq_vals[combo];


        const int warm = min(series_len, first_valid + 2);
        for (int i = 0; i < warm; ++i) out_row[i] = CUDART_NAN_F;

        if (first_valid + 1 >= series_len) continue;


        float hp_im2 = prices[first_valid];
        float hp_im1 = prices[first_valid + 1];

        for (int t = first_valid + 2; t < series_len; ++t) {
            const float x = prices[t];
            const float x1 = prices[t - 1];
            const float x2 = prices[t - 2];
            const float diff = x - 2.0f * x1 + x2;
            const float s3 = __fmaf_rn(two_1m, hp_im1, c * diff);
            const float hp = __fmaf_rn(neg_oma_sq, hp_im2, s3);

            out_row[t] = x - hp;

            hp_im2 = hp_im1;
            hp_im1 = hp;
        }
    }
}


extern "C" __global__
void decycler_batch_warp_scan_f32(const float* __restrict__ prices,
                                 const int*   __restrict__ periods,
                                 const float* __restrict__ c_vals,
                                 const float* __restrict__ two_1m_vals,
                                 const float* __restrict__ neg_oma_sq_vals,
                                 int series_len,
                                 int n_combos,
                                 int first_valid,
                                 float* __restrict__ out) {
    (void)periods;
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0) return;
    if (threadIdx.x >= 32) return;

    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;

    float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = lane; t < series_len; t += 32) out_row[t] = CUDART_NAN_F;
        return;
    }

    const int warm = min(series_len, first_valid + 2);
    for (int t = lane; t < warm; t += 32) out_row[t] = CUDART_NAN_F;

    if (first_valid + 1 >= series_len) return;

    const float c      = c_vals[combo];
    const float a1     = two_1m_vals[combo];
    const float a2     = neg_oma_sq_vals[combo];


    float s0_prev = 0.0f;
    float s1_prev = 0.0f;
    if (lane == 0) {
        s1_prev = prices[first_valid];
        s0_prev = prices[first_valid + 1];
    }
    s0_prev = __shfl_sync(mask, s0_prev, 0);
    s1_prev = __shfl_sync(mask, s1_prev, 0);


    const float m00 = a1;
    const float m01 = a2;
    const float m10 = 1.0f;
    const float m11 = 0.0f;

    const int t0 = first_valid + 2;
    if (t0 >= series_len) return;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        float u = 0.0f;
        if (valid) {
            const float x = prices[t];
            const float x1 = prices[t - 1];
            const float x2 = prices[t - 2];
            u = c * (x - 2.0f * x1 + x2);
        }


        float p00 = valid ? m00 : 1.0f;
        float p01 = valid ? m01 : 0.0f;
        float p10 = valid ? m10 : 0.0f;
        float p11 = valid ? m11 : 1.0f;
        float v0  = valid ? u   : 0.0f;
        float v1  = 0.0f;


        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float p00_prev = __shfl_up_sync(mask, p00, offset);
            const float p01_prev = __shfl_up_sync(mask, p01, offset);
            const float p10_prev = __shfl_up_sync(mask, p10, offset);
            const float p11_prev = __shfl_up_sync(mask, p11, offset);
            const float v0_prev  = __shfl_up_sync(mask, v0,  offset);
            const float v1_prev  = __shfl_up_sync(mask, v1,  offset);
            if (lane >= offset) {
                const float c00 = p00, c01 = p01, c10 = p10, c11 = p11;
                const float cv0 = v0,  cv1 = v1;

                const float n00 = fmaf(c00, p00_prev, c01 * p10_prev);
                const float n01 = fmaf(c00, p01_prev, c01 * p11_prev);
                const float n10 = fmaf(c10, p00_prev, c11 * p10_prev);
                const float n11 = fmaf(c10, p01_prev, c11 * p11_prev);

                const float nv0 = fmaf(c00, v0_prev, fmaf(c01, v1_prev, cv0));
                const float nv1 = fmaf(c10, v0_prev, fmaf(c11, v1_prev, cv1));

                p00 = n00; p01 = n01; p10 = n10; p11 = n11;
                v0  = nv0; v1  = nv1;
            }
        }


        const float hp0 = fmaf(p00, s0_prev, fmaf(p01, s1_prev, v0));
        const float hp1 = fmaf(p10, s0_prev, fmaf(p11, s1_prev, v1));

        if (valid) {
            out_row[t] = prices[t] - hp0;
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        s0_prev = __shfl_sync(mask, hp0, last_lane);
        s1_prev = __shfl_sync(mask, hp1, last_lane);
    }
}


extern "C" __global__
void decycler_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                        const int*   __restrict__ first_valids,
                                        int period,
                                        float c,
                                        float two_1m,
                                        float neg_oma_sq,
                                        int num_series,
                                        int series_len,
                                        float* __restrict__ out_tm)
{
    (void)period;
    const int stride = num_series;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = tid; s < num_series; s += blockDim.x * gridDim.x) {
        if (series_len <= 0) continue;
        const int fv = first_valids[s];
        if (fv < 0 || fv >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[(size_t)t * (size_t)stride + s] = CUDART_NAN_F;
            continue;
        }


        const int warm = min(series_len, fv + 2);
        for (int t = 0; t < warm; ++t) {
            out_tm[(size_t)t * (size_t)stride + s] = CUDART_NAN_F;
        }
        if (fv + 1 >= series_len) continue;


        float hp_im2 = prices_tm[(size_t)fv * (size_t)stride + s];
        float hp_im1 = prices_tm[(size_t)(fv + 1) * (size_t)stride + s];

        for (int t = fv + 2; t < series_len; ++t) {
            const int idx = (size_t)t * (size_t)stride + s;
            const float x = prices_tm[idx];
            const float s3 = __fmaf_rn(two_1m, hp_im1, c * (x - 2.0f * prices_tm[idx - stride] + prices_tm[idx - 2 * stride]));
            const float hp = __fmaf_rn(neg_oma_sq, hp_im2, s3);
            out_tm[idx] = x - hp;
            hp_im2 = hp_im1;
            hp_im1 = hp;
        }
    }
}

// ============================================================================
// f64 LANE — NeoEthos conversion, 2026-08-09
// ----------------------------------------------------------------------------
// CPU reference: src/indicators/decycler.rs:506 decycler_scalar
//                src/indicators/decycler.rs:477 first = first non-NaN of source
//                hp_period default 125, k default 0.707 (decycler.rs:224/228)
//
// THE WARP SCAN ABOVE IS THE THING BEING REPLACED, NOT PORTED.
// `decycler_batch_warp_scan_f32` reformulates this 2-pole high-pass as an
// associative scan so a warp can do it in log steps. That is a DIFFERENT
// accumulation order for a recurrence whose output feeds a threshold
// comparison, so it is forbidden here by construction: this kernel is ONE
// THREAD PER COLUMN walking bars ascending, exactly as decycler_scalar does.
//
// THE FIVE-STEP FMA CHAIN IS THE SPECIFICATION. The CPU writes
//   s0 = current * c;                       // 1 rounding
//   s1 = x1.mul_add(-2.0 * c, s0);          // -2.0*c is a rounding, fma is one
//   s2 = x2.mul_add(c, s1);                 // one
//   s3 = hp_prev1.mul_add(2.0*one_minus_alpha, s2);
//   hp = hp_prev2.mul_add(-one_minus_alpha_sq, s3);
// Collapsing this into a single polynomial evaluation would be algebraically
// identical and numerically different. Each step is reproduced as its own fma.
//
// EPSILON: `const EPSILON: f64 = 1e-10` with `EPSILON.copysign(cos_val)`
// (decycler.rs:522) is an f64 constant from the f64 CPU source and is carried
// over unchanged. The f32 kernel above has no such guard at all — it has
// `__fmaf_rn` x4 and no cos protection — so this is a correction as well as a
// conversion.
//
// WARMUP: `first + 2` (decycler.rs:515). The first two bars seed hp_prev2 and
// hp_prev1 from the DATA, not from zero; the `hp_prev2 = 0.0` initialisers at
// decycler.rs:517-518 are overwritten whenever the series is long enough, which
// it always is here because `prepare` already required `len - first >= period`.
//
// SWEPT PARAMETER: `hp_period`. `k` stays 0.707.
// ============================================================================

#include "../neo_f64_common.cuh"

extern "C" __global__ void neoethos_decycler_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* row = out + (size_t)r * (size_t)n;
    const int hp_period = periods[r];

    // decycler_with_kernel: hp_period < 2 || hp_period > len, len-first < period.
    if (first_valid < 0 || first_valid >= n || hp_period < 2 || hp_period > n ||
        (n - first_valid) < hp_period) {
        neo_fill_all(row, n);
        return;
    }

    const int warm = first_valid + 2;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    const double PI_D = 3.14159265358979323846;
    const double k = 0.707;
    const double angle = (2.0 * PI_D * k) * (1.0 / (double)hp_period);
    double sin_val, cos_val;
    sincos(angle, &sin_val, &cos_val);

    const double EPSILON = 1e-10;
    const double cos_safe = (fabs(cos_val) < EPSILON) ? copysign(EPSILON, cos_val) : cos_val;

    const double alpha = 1.0 + ((sin_val - 1.0) / cos_safe);
    const double one_minus_alpha_half = 1.0 - alpha / 2.0;
    const double c = one_minus_alpha_half * one_minus_alpha_half;
    const double one_minus_alpha = 1.0 - alpha;
    const double one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    double hp_prev2 = data[first_valid];
    double hp_prev1 = data[first_valid + 1];
    double x2 = hp_prev2;
    double x1 = hp_prev1;

    const double m2c = -2.0 * c;
    const double t2oma = 2.0 * one_minus_alpha;
    const double moas = -one_minus_alpha_sq;

    for (int i = first_valid + 2; i < n; ++i) {
        const double current = data[i];

        const double s0 = current * c;
        const double s1 = fma(x1, m2c, s0);
        const double s2 = fma(x2, c, s1);
        const double s3 = fma(hp_prev1, t2oma, s2);
        const double hp_val = fma(hp_prev2, moas, s3);

        hp_prev2 = hp_prev1;
        hp_prev1 = hp_val;
        x2 = x1;
        x1 = current;

        row[i] = current - hp_val;
    }
}
