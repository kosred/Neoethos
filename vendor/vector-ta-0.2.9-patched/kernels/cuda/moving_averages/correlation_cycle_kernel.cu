#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ float sanitize_nan(float x) { return isnan(x) ? 0.f : x; }


extern "C" __global__ void correlation_cycle_batch_f32_ria(
    const float* __restrict__ prices,
    const float* __restrict__ cos_flat,
    const float* __restrict__ sin_flat,
    const int*   __restrict__ periods,
    const float* __restrict__ sum_cos_arr,
    const float* __restrict__ sum_sin_arr,
    const float* __restrict__ sqrt_t2_arr,
    const float* __restrict__ sqrt_t4_arr,
    int max_period,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_offset,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    float* __restrict__ out_angle)
{
    const int combo = combo_offset + blockIdx.y;
    if (combo >= n_combos) return;

    const int period   = periods[combo];
    const float n      = (float)period;
    const int warm_ria = first_valid + period;


    const float sum_cos = sum_cos_arr[combo];
    const float sum_sin = sum_sin_arr[combo];
    const float sqrt_t2 = sqrt_t2_arr[combo];
    const float sqrt_t4 = sqrt_t4_arr[combo];
    const int   base    = combo * series_len;

    extern __shared__ float sh[];
    float* wcos = sh;
    float* wsin = sh + period;

    const float* wcos_src = cos_flat + combo * max_period;
    const float* wsin_src = sin_flat + combo * max_period;
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        wcos[i] = wcos_src[i];
        wsin[i] = wsin_src[i];
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float r_out = NAN, i_out = NAN, ang_out = NAN;
        if (t >= warm_ria) {
            float mean = 0.f, m2 = 0.f;
            float sum_xc = 0.f, sum_xs = 0.f;
            int k = 0;

            #pragma unroll 4
            for (int j = 0; j < period; ++j) {
                int idx = t - (j + 1);
                float x = sanitize_nan(prices[idx]);
                float c = wcos[j];
                float s = wsin[j];

                ++k;
                float delta = x - mean;
                mean += delta / (float)k;
                float delta2 = x - mean;
                m2 = fmaf(delta, delta2, m2);
                sum_xc  = fmaf(x, c, sum_xc);
                sum_xs  = fmaf(x, s, sum_xs);
            }
            float sum_x = mean * n;
            float t1 = n * m2;
            if (t1 < 0.f) t1 = 0.f;
            float r_val = 0.f, i_val = 0.f;
            if (t1 > 0.f) {
                float root = sqrtf(t1);
                if (sqrt_t2 > 0.f) {
                    float denom_r = root * sqrt_t2;
                    if (denom_r > 0.f)
                        r_val = (fmaf(n, sum_xc, -(sum_x * sum_cos))) / denom_r;
                }
                if (sqrt_t4 > 0.f) {
                    float denom_i = root * sqrt_t4;
                    if (denom_i > 0.f)
                        i_val = (fmaf(n, sum_xs, -(sum_x * sum_sin))) / denom_i;
                }
            }
            r_out = r_val;
            i_out = i_val;
            ang_out = (i_val == 0.f) ? 0.f : atan2f(-i_val, r_val) * (180.f / (float)M_PI);
        }
        out_real[base + t]  = r_out;
        out_imag[base + t]  = i_out;
        out_angle[base + t] = ang_out;
        t += stride;
    }
}


extern "C" __global__ void correlation_cycle_state_batch_f32(
    const float* __restrict__ angle_flat,
    const float* __restrict__ thresholds,
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_offset,
    float* __restrict__ out_state)
{
    const int combo = combo_offset + blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float thr  = thresholds[combo];
    const int warm_s = first_valid + period + 1;
    const int base   = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        float st = NAN;
        if (t >= warm_s) {
            float a  = angle_flat[base + t];
            float ap = angle_flat[base + t - 1];
            if (!isnan(ap) && fabsf(a - ap) < thr) {
                st = (a >= 0.f) ? 1.f : -1.f;
            } else {
                st = 0.f;
            }
        }
        out_state[base + t] = st;
        t += stride;
    }
}


extern "C" __global__ void correlation_cycle_many_series_one_param_f32_ria(
    const float* __restrict__ prices_tm,
    const float* __restrict__ wcos,
    const float* __restrict__ wsin,
    const float  sum_cos,
    const float  sum_sin,
    const float  sqrt_t2,
    const float  sqrt_t4,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_real_tm,
    float* __restrict__ out_imag_tm,
    float* __restrict__ out_angle_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || t0 >= rows) return;

    const int warm_ria = first_valids[s] + period;
    const float n = (float)period;
    const int stride_t = gridDim.x * blockDim.x;
    const int stride_s = gridDim.y * blockDim.y;

    for (int t = t0; t < rows; t += stride_t) {
        const int out_idx = t * cols + s;
        float r_out = NAN, i_out = NAN, ang_out = NAN;
        if (t >= warm_ria) {
            float mean = 0.f, m2 = 0.f;
            float sum_xc = 0.f, sum_xs = 0.f;
            int k = 0;
            #pragma unroll 4
            for (int j = 0; j < period; ++j) {
                int tt = t - (j + 1);
                float x = sanitize_nan(prices_tm[tt * cols + s]);
                float c = wcos[j];
                float si = wsin[j];
                ++k;
                float delta = x - mean;
                mean += delta / (float)k;
                float delta2 = x - mean;
                m2 = fmaf(delta, delta2, m2);
                sum_xc  = fmaf(x, c, sum_xc);
                sum_xs  = fmaf(x, si, sum_xs);
            }
            float sum_x = mean * n;
            float t1 = n * m2;
            if (t1 < 0.f) t1 = 0.f;
            float r_val = 0.f, i_val = 0.f;
            if (t1 > 0.f) {
                float root = sqrtf(t1);
                if (sqrt_t2 > 0.f) {
                    float denom_r = root * sqrt_t2;
                    if (denom_r > 0.f) r_val = (fmaf(n, sum_xc, -(sum_x * sum_cos))) / denom_r;
                }
                if (sqrt_t4 > 0.f) {
                    float denom_i = root * sqrt_t4;
                    if (denom_i > 0.f) i_val = (fmaf(n, sum_xs, -(sum_x * sum_sin))) / denom_i;
                }
            }
            r_out = r_val;
            i_out = i_val;
            ang_out = (i_val == 0.f) ? 0.f : atan2f(-i_val, r_val) * (180.f / (float)M_PI);
        }
        out_real_tm[out_idx]  = r_out;
        out_imag_tm[out_idx]  = i_out;
        out_angle_tm[out_idx] = ang_out;
    }
}


extern "C" __global__ void correlation_cycle_state_many_series_one_param_f32(
    const float* __restrict__ angle_tm,
    const float  threshold,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_state_tm)
{
    const int s  = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || t0 >= rows) return;

    const int warm_s = first_valids[s] + period + 1;
    const int stride_t = gridDim.x * blockDim.x;
    for (int t = t0; t < rows; t += stride_t) {
        int idx = t * cols + s;
        float st = NAN;
        if (t >= warm_s) {
            float a  = angle_tm[idx];
            float ap = angle_tm[idx - cols];
            if (!isnan(ap) && fabsf(a - ap) < threshold) {
                st = (a >= 0.f) ? 1.f : -1.f;
            } else {
                st = 0.f;
            }
        }
        out_state_tm[idx] = st;
    }
}

// ===========================================================================
// S3 f64 LANE — correlation_cycle (real / imag / angle / state)
// ===========================================================================
// Reference: src/indicators/correlation_cycle.rs
//   correlation_cycle_with_kernel (:230)      — first_valid + Err branches
//   correlation_cycle_compute_into (:565)     — the arithmetic
//   correlation_cycle_window_sums (:494)      — the O(period) rebase
// Batch defaults: period 20, threshold 9.0, source close.
//
// WHICH OUTPUT. The preserved primary ABI emits REAL. The production ABI emits
// the canonical REAL / IMAG / ANGLE / STATE matrices from this same row
// authority in one launch.
//
// THE TRIG TABLES ARE RECOMPUTED, NOT ALLOCATED. Each cos_table/sin_table pair
// comes from the same deterministic FreeBSD-msun weight authority as the CPU.
// The four-wide angle construction, argument reduction, polynomial constants
// and operation order are mirrored exactly, with no device scratch.
//
// THE SEED SUMS GROUP BY FOUR, AND THE mul_add CHAIN NESTS RIGHT-TO-LEFT.
//   sum_x2 = x0.mul_add(x0, x1.mul_add(x1, x2.mul_add(x2, x3.mul_add(x3, sum))))
// That is FOUR fmas applied innermost-first — x3 is folded in before x0. A
// left-to-right loop over the same four terms is a different association and a
// different result. Reproduced exactly, including the scalar tail for the
// remainder (:546).
//
// NaN IS ZEROED, NOT PROPAGATED. `if x != x { x = 0.0 }` (:518-529, :549) —
// the reference substitutes zero for a NaN sample rather than poisoning the
// window. Written as isnan() here, which has the same truth value.
//
// THE REBASE INTERVAL IS DATA-DEPENDENT (:669): 1 if ANY value from `first` on
// is infinite, otherwise 256. That is not a performance knob — at interval 1
// every bar is recomputed from the window and the incremental rotation is never
// used, so the two settings produce different numbers. The scan is reproduced.
//
// THE COMPLEX ROTATION. Between rebases the CPU advances (sum_xc, sum_xs) by
// multiplying by the unit phasor (z_re, z_im) = (cos w, -sin w):
//   s        = sum_xc + dx
//   next_xc  = z_re.mul_add(s, -z_im * sum_xs)
//   next_xs  = z_im.mul_add(s,  z_re * sum_xs)
// Two roundings each. This is an exact algebraic identity for the shifted
// window, and it is also where drift accumulates — which is why the rebase
// exists and why its period must be 256 exactly.
//
// ANGLE. The full signed f64 ratio passes through the mirrored FreeBSD-msun
// s_atan reduction/polynomial, then exact-bit pi/2 and 180/pi constants. The
// production f64 lane does not call host libm or CUDA libdevice transcendental
// functions, so exact CPU-bit parity remains an executable contract.
//
// One thread per column.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

/* FreeBSD msun k_sin/k_cos, medium pi/2 reduction and s_atan.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft. Permission to use, copy, modify, and
 * distribute this software is freely granted, provided this notice is
 * preserved.
 */
static __device__ __forceinline__ double neo_s3_cc_ms_k_cos(double x, double y) {
    const double c1 = 0x1.555555555554cp-5;
    const double c2 = -0x1.6c16c16c15177p-10;
    const double c3 = 0x1.a01a019cb1590p-16;
    const double c4 = -0x1.27e4f809c52adp-22;
    const double c5 = 0x1.1ee9ebdb4b1c4p-29;
    const double c6 = -0x1.8fae9be8838d4p-37;
    const double z = x * x;
    const double w2 = z * z;
    const double r = z * (c1 + z * (c2 + z * c3))
        + w2 * w2 * (c4 + z * (c5 + z * c6));
    const double hz = 0.5 * z;
    const double w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + (z * r - x * y));
}

static __device__ __forceinline__ double neo_s3_cc_ms_k_sin(
    double x,
    double y,
    bool has_tail) {
    const double s1 = -0x1.5555555555549p-3;
    const double s2 = 0x1.111111110f8a6p-7;
    const double s3 = -0x1.a01a019c161d5p-13;
    const double s4 = 0x1.71de357b1fe7dp-19;
    const double s5 = -0x1.ae5e68a2b9cebp-26;
    const double s6 = 0x1.5d93a5acfd57cp-33;
    const double z = x * x;
    const double w = z * z;
    const double r = s2 + z * (s3 + z * s4) + z * w * (s5 + z * s6);
    const double v = z * x;
    if (has_tail) {
        return x - ((z * (0.5 * y - v * r) - y) - v * s1);
    }
    return x + v * (s1 + z * r);
}

static __device__ __forceinline__ int neo_s3_cc_reduce_pio2(
    double x,
    double* y0_out,
    double* y1_out) {
    const double inv_pio2 = 0x1.45f306dc9c883p-1;
    const double to_int = 0x1.8p+52;
    const double pio2_1 = 0x1.921fb54400000p+0;
    const double pio2_1t = 0x1.0b4611a626331p-34;
    const double pio2_2 = 0x1.0b4611a600000p-34;
    const double pio2_2t = 0x1.3198a2e037073p-69;
    const double pio2_3 = 0x1.3198a2e000000p-69;
    const double pio2_3t = 0x1.b839a252049c1p-104;

    const double tmp = x * inv_pio2 + to_int;
    const double f_n = tmp - to_int;
    const int n = static_cast<int>(f_n);
    double r = x - f_n * pio2_1;
    double w = f_n * pio2_1t;
    double y0 = r - w;
    const unsigned long long x_bits =
        static_cast<unsigned long long>(__double_as_longlong(x));
    const int ex = static_cast<int>((x_bits >> 52) & 0x7ffULL);
    unsigned long long y_bits =
        static_cast<unsigned long long>(__double_as_longlong(y0));
    int ey = static_cast<int>((y_bits >> 52) & 0x7ffULL);
    if (ex - ey > 16) {
        const double t = r;
        w = f_n * pio2_2;
        r = t - w;
        w = f_n * pio2_2t - ((t - r) - w);
        y0 = r - w;
        y_bits = static_cast<unsigned long long>(__double_as_longlong(y0));
        ey = static_cast<int>((y_bits >> 52) & 0x7ffULL);
        if (ex - ey > 49) {
            const double t2 = r;
            w = f_n * pio2_3;
            r = t2 - w;
            w = f_n * pio2_3t - ((t2 - r) - w);
            y0 = r - w;
        }
    }
    *y0_out = y0;
    *y1_out = (r - y0) - w;
    return n;
}

static __device__ __forceinline__ void neo_s3_cc_deterministic_sin_cos(
    double x,
    double* sin_out,
    double* cos_out) {
    const unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(x));
    const unsigned int high = static_cast<unsigned int>((bits >> 32) & 0x7fffffffULL);
    if (high <= 0x3fe921fbU) {
        *sin_out = neo_s3_cc_ms_k_sin(x, 0.0, false);
        *cos_out = neo_s3_cc_ms_k_cos(x, 0.0);
        return;
    }

    double y0;
    double y1;
    const int quadrant = neo_s3_cc_reduce_pio2(x, &y0, &y1);
    const double sin = neo_s3_cc_ms_k_sin(y0, y1, true);
    const double cos = neo_s3_cc_ms_k_cos(y0, y1);
    switch (quadrant & 3) {
        case 0: *sin_out = sin;  *cos_out = cos;  return;
        case 1: *sin_out = cos;  *cos_out = -sin; return;
        case 2: *sin_out = -sin; *cos_out = -cos; return;
        default: *sin_out = -cos; *cos_out = sin; return;
    }
}

static __device__ __forceinline__ void neo_s3_cc_deterministic_weight(
    double w,
    int j,
    double* cos_out,
    double* neg_sin_out) {
    const int group_start = j & ~3;
    double angle = w * (static_cast<double>(group_start) + 1.0);
    int offset = j & 3;
    while (offset != 0) {
        angle += w;
        --offset;
    }
    double sin;
    double cos;
    neo_s3_cc_deterministic_sin_cos(angle, &sin, &cos);
    *cos_out = cos;
    *neg_sin_out = -sin;
}

static __device__ __forceinline__ double neo_s3_cc_deterministic_atan(double input) {
    const double atan_hi[4] = {
        0x1.dac670561bb4fp-2,
        0x1.921fb54442d18p-1,
        0x1.f730bd281f69bp-1,
        0x1.921fb54442d18p+0,
    };
    const double atan_lo[4] = {
        0x1.a2b7f222f65e2p-56,
        0x1.1a62633145c07p-55,
        0x1.007887af0cbbdp-56,
        0x1.1a62633145c07p-54,
    };
    const double coefficients[11] = {
        0x1.555555555550dp-2,
        -0x1.999999998ebc4p-3,
        0x1.24924920083ffp-3,
        -0x1.c71c6fe231671p-4,
        0x1.745cdc54c206ep-4,
        -0x1.3b0f2af749a6dp-4,
        0x1.10d66a0d03d51p-4,
        -0x1.dde2d52defd9ap-5,
        0x1.97b4b24760debp-5,
        -0x1.2b4442c6a6c2fp-5,
        0x1.0ad3ae322da11p-6,
    };
    const double half_pi = 0x1.921fb54442d18p+0;

    double x = input;
    const unsigned long long input_bits =
        static_cast<unsigned long long>(__double_as_longlong(x));
    unsigned int high = static_cast<unsigned int>(input_bits >> 32);
    const unsigned int sign = high >> 31;
    high &= 0x7fffffffU;
    if (high >= 0x44100000U) {
        if (isnan(x)) {
            return x;
        }
        return sign != 0U ? -half_pi : half_pi;
    }

    int reduction;
    if (high < 0x3fdc0000U) {
        if (high < 0x3e400000U) {
            return x;
        }
        reduction = -1;
    } else {
        x = __longlong_as_double(static_cast<long long>(
            input_bits & 0x7fffffffffffffffULL));
        if (high < 0x3ff30000U) {
            if (high < 0x3fe60000U) {
                x = (2.0 * x - 1.0) / (2.0 + x);
                reduction = 0;
            } else {
                x = (x - 1.0) / (x + 1.0);
                reduction = 1;
            }
        } else if (high < 0x40038000U) {
            x = (x - 1.5) / (1.0 + 1.5 * x);
            reduction = 2;
        } else {
            x = -1.0 / x;
            reduction = 3;
        }
    }

    const double z = x * x;
    const double w = z * z;
    const double s1 = z * (
        coefficients[0] + w * (
            coefficients[2] + w * (
                coefficients[4] + w * (
                    coefficients[6] + w * (
                        coefficients[8] + w * coefficients[10])))));
    const double s2 = w * (
        coefficients[1] + w * (
            coefficients[3] + w * (
                coefficients[5] + w * (
                    coefficients[7] + w * coefficients[9]))));
    if (reduction < 0) {
        return x - x * (s1 + s2);
    }

    const double result = atan_hi[reduction]
        - ((x * (s1 + s2) - atan_lo[reduction]) - x);
    return sign != 0U ? -result : result;
}

static __device__ __forceinline__ double neo_s3_cc_deterministic_angle(
    double real,
    double imag) {
    if (imag == 0.0) {
        return 0.0;
    }
    const double half_pi = 0x1.921fb54442d18p+0;
    const double radians_to_degrees = 0x1.ca5dc1a63c1f8p+5;
    double angle = (neo_s3_cc_deterministic_atan(real / imag) + half_pi)
        * radians_to_degrees;
    if (imag > 0.0) {
        angle -= 180.0;
    }
    return angle;
}

// correlation_cycle_window_sums (:494) — 4-wide, right-nested fma chains.
__device__ __forceinline__ void neo_s3_cc_window_sums(
    const double* __restrict__ d, double w, int i, int period,
    double* o_sx, double* o_sx2, double* o_sxc, double* o_sxs)
{
    double sum_x = 0.0, sum_x2 = 0.0, sum_xc = 0.0, sum_xs = 0.0;

    int j = 0;
    while (j + 4 <= period) {
        const int idx0 = i - (j + 1);
        const int idx1 = idx0 - 1;
        const int idx2 = idx1 - 1;
        const int idx3 = idx2 - 1;

        double x0 = d[idx0]; if (isnan(x0)) x0 = 0.0;
        double x1 = d[idx1]; if (isnan(x1)) x1 = 0.0;
        double x2 = d[idx2]; if (isnan(x2)) x2 = 0.0;
        double x3 = d[idx3]; if (isnan(x3)) x3 = 0.0;

        double c0, s0;
        double c1, s1;
        double c2, s2;
        double c3, s3;
        neo_s3_cc_deterministic_weight(w, j, &c0, &s0);
        neo_s3_cc_deterministic_weight(w, j + 1, &c1, &s1);
        neo_s3_cc_deterministic_weight(w, j + 2, &c2, &s2);
        neo_s3_cc_deterministic_weight(w, j + 3, &c3, &s3);

        sum_x += x0 + x1 + x2 + x3;
        sum_x2 = fma(x0, x0, fma(x1, x1, fma(x2, x2, fma(x3, x3, sum_x2))));
        sum_xc = fma(x0, c0, fma(x1, c1, fma(x2, c2, fma(x3, c3, sum_xc))));
        sum_xs = fma(x0, s0, fma(x1, s1, fma(x2, s2, fma(x3, s3, sum_xs))));
        j += 4;
    }
    while (j < period) {
        const int idx = i - (j + 1);
        double x = d[idx]; if (isnan(x)) x = 0.0;
        double c, s;
        neo_s3_cc_deterministic_weight(w, j, &c, &s);
        sum_x  += x;
        sum_x2 = fma(x, x, sum_x2);
        sum_xc = fma(x, c, sum_xc);
        sum_xs = fma(x, s, sum_xs);
        j += 1;
    }

    *o_sx = sum_x; *o_sx2 = sum_x2; *o_sxc = sum_xc; *o_sxs = sum_xs;
}

__device__ __forceinline__ void neo_s3_correlation_cycle_row_f64(
    const double* __restrict__ data,
    int n,
    int period,
    double threshold,
    int first_valid,
    double* out_real,
    double* out_imag,
    double* out_angle,
    double* out_state)
{
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) {
            if (out_real != nullptr) out_real[i] = neo_s3_qnan();
            if (out_imag != nullptr) out_imag[i] = neo_s3_qnan();
            if (out_angle != nullptr) out_angle[i] = neo_s3_qnan();
            if (out_state != nullptr) out_state[i] = neo_s3_qnan();
        }
        return;
    }

    for (int i = 0; i < n; ++i) {
        if (out_real != nullptr) out_real[i] = neo_s3_qnan();
        if (out_imag != nullptr) out_imag[i] = neo_s3_qnan();
        if (out_angle != nullptr) out_angle[i] = neo_s3_qnan();
        if (out_state != nullptr) out_state[i] = neo_s3_qnan();
    }

    const double two_pi = 0x1.921fb54442d18p+2;
    const double nn = (double)period;
    const double w  = two_pi / nn;

    // Seed constants, in the CPU's 4-wide association (:591-645).
    double sum_cos = 0.0, sum_sin = 0.0, sum_cos2 = 0.0, sum_sin2 = 0.0;
    {
        int j = 0;
        while (j + 4 <= period) {
            for (int q = 0; q < 4; ++q) {
                double c, ys;
                neo_s3_cc_deterministic_weight(w, j + q, &c, &ys);
                sum_cos += c;
                sum_sin += ys;
                sum_cos2 += c * c;
                sum_sin2 += ys * ys;
            }
            j += 4;
        }
        while (j < period) {
            double c, ys;
            neo_s3_cc_deterministic_weight(w, j, &c, &ys);
            sum_cos += c;
            sum_sin += ys;
            sum_cos2 += c * c;
            sum_sin2 += ys * ys;
            j += 1;
        }
    }

    const double t2_const = fma(nn, sum_cos2, -(sum_cos * sum_cos));
    const double t4_const = fma(nn, sum_sin2, -(sum_sin * sum_sin));
    const bool has_t2 = t2_const > 0.0;
    const bool has_t4 = t4_const > 0.0;
    const double sqrt_t2c = has_t2 ? sqrt(t2_const) : 0.0;
    const double sqrt_t4c = has_t4 ? sqrt(t4_const) : 0.0;

    const int start_ria = first_valid + period;
    if (start_ria >= n) return;
    const int start_state = start_ria + 1;

    int rebase_interval = 256;
    for (int i = first_valid; i < n; ++i) {
        if (isinf(data[i])) { rebase_interval = 1; break; }
    }

    double z_re, z_im;
    neo_s3_cc_deterministic_weight(w, 0, &z_re, &z_im);
    int last_rebase = start_ria;

    double sum_x, sum_x2, sum_xc, sum_xs;
    neo_s3_cc_window_sums(data, w, start_ria, period, &sum_x, &sum_x2, &sum_xc, &sum_xs);
    double prev_angle = neo_s3_qnan();
    const bool needs_angle = (out_angle != nullptr) || (out_state != nullptr);

    for (int i = start_ria; i < n; ++i) {
        const double t1 = fma(nn, sum_x2, -(sum_x * sum_x));
        double r_val = 0.0;
        double i_val = 0.0;

        if (t1 > 0.0) {
            const double sqrt_t1 = sqrt(t1);
            if (has_t2) {
                const double denom = sqrt_t1 * sqrt_t2c;
                if (denom > 0.0) r_val = fma(nn, sum_xc, -(sum_x * sum_cos)) / denom;
            }
            if (has_t4) {
                const double denom = sqrt_t1 * sqrt_t4c;
                if (denom > 0.0) i_val = fma(nn, sum_xs, -(sum_x * sum_sin)) / denom;
            }
        }

        if (out_real != nullptr) out_real[i] = r_val;
        if (out_imag != nullptr) out_imag[i] = i_val;

        if (needs_angle) {
            const double angle = neo_s3_cc_deterministic_angle(r_val, i_val);
            if (out_angle != nullptr) out_angle[i] = angle;
            if (out_state != nullptr && i >= start_state) {
                const double delta = fabs(angle - prev_angle);
                out_state[i] = !isnan(prev_angle) && delta < threshold
                    ? (angle >= 0.0 ? 1.0 : -1.0)
                    : 0.0;
            }
            prev_angle = angle;
        }

        const int next_i = i + 1;
        if (next_i < n) {
            if (next_i - last_rebase >= rebase_interval) {
                neo_s3_cc_window_sums(data, w, next_i, period,
                                      &sum_x, &sum_x2, &sum_xc, &sum_xs);
                last_rebase = next_i;
            } else {
                double x_new = data[i];
                double x_old = data[i - period];
                if (isnan(x_new)) x_new = 0.0;
                if (isnan(x_old)) x_old = 0.0;
                const double dx = x_new - x_old;
                sum_x += dx;
                sum_x2 += fma(x_new, x_new, -(x_old * x_old));
                const double s = sum_xc + dx;
                const double next_xc = fma(z_re, s, -z_im * sum_xs);
                const double next_xs = fma(z_im, s,  z_re * sum_xs);
                sum_xc = next_xc;
                sum_xs = next_xs;
            }
        }
    }
}

extern "C" __global__ void neoethos_correlation_cycle_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    neo_s3_correlation_cycle_row_f64(
        data,
        n,
        periods[r],
        9.0,
        first_valid,
        row,
        nullptr,
        nullptr,
        nullptr);
}

extern "C" __global__ void correlation_cycle_outputs_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    const double* __restrict__ thresholds,
    int n_combos,
    int first_valid,
    double* __restrict__ out_real,
    double* __restrict__ out_imag,
    double* __restrict__ out_angle,
    double* __restrict__ out_state)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const size_t offset = (size_t)r * (size_t)n;
    neo_s3_correlation_cycle_row_f64(
        data,
        n,
        periods[r],
        thresholds[r],
        first_valid,
        out_real + offset,
        out_imag + offset,
        out_angle + offset,
        out_state + offset);
}
