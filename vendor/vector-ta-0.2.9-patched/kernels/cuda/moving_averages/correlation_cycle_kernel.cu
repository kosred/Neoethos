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
// S3 f64 LANE — correlation_cycle (real component)
// ===========================================================================
// Reference: src/indicators/correlation_cycle.rs
//   correlation_cycle_with_kernel (:230)      — first_valid + Err branches
//   correlation_cycle_compute_into (:565)     — the arithmetic
//   correlation_cycle_window_sums (:494)      — the O(period) rebase
// Batch defaults: period 20, threshold 9.0, source close.
//
// WHICH OUTPUT. Multi-output (real / imag / angle / state); compute_
// correlation_cycle_batch maps "value" to REAL, so this kernel is the real
// component.
//
// THE TRIG TABLES ARE RECOMPUTED, NOT ALLOCATED. cos_table[j] = cos(w*(j+1))
// and sin_table[j] = -sin(w*(j+1)) are pure functions of j, so the two
// period-length Vecs the CPU builds are replaced by evaluation at the point of
// use. Same values, no device scratch.
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
// ANGLE. a = atan(r/i) + asin(1.0), then to_degrees, then -180 if i_val > 0.
// half_pi is spelled f64::asin(1.0) by the reference rather than a PI/2
// literal; asin(1.0) is exactly PI/2 rounded, so the constant below is written
// the same way. to_degrees multiplies by 180/PI as a single folded constant.
//
// TRIG PARITY CAVEAT, STATED RATHER THAN HIDDEN: CUDA's double sin/cos/atan are
// specified to <= 2 ulp, glibc's are effectively correctly rounded. Every value
// here is derived from those, so this kernel matches the reference to a few ulp
// rather than bit-for-bit. That is a property of the transcendental library, not
// of this transcription, and it cannot be removed by writing the arithmetic
// differently.
//
// One thread per column.
// ===========================================================================

#define NEO_S3_CC_THRESHOLD 9.0

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ __forceinline__ double neo_s3_cc_cos(double w, int j) {
    return cos(w * ((double)j + 1.0));
}
__device__ __forceinline__ double neo_s3_cc_sin(double w, int j) {
    return -sin(w * ((double)j + 1.0));
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

        const double c0 = neo_s3_cc_cos(w, j),     s0 = neo_s3_cc_sin(w, j);
        const double c1 = neo_s3_cc_cos(w, j + 1), s1 = neo_s3_cc_sin(w, j + 1);
        const double c2 = neo_s3_cc_cos(w, j + 2), s2 = neo_s3_cc_sin(w, j + 2);
        const double c3 = neo_s3_cc_cos(w, j + 3), s3 = neo_s3_cc_sin(w, j + 3);

        sum_x += x0 + x1 + x2 + x3;
        sum_x2 = fma(x0, x0, fma(x1, x1, fma(x2, x2, fma(x3, x3, sum_x2))));
        sum_xc = fma(x0, c0, fma(x1, c1, fma(x2, c2, fma(x3, c3, sum_xc))));
        sum_xs = fma(x0, s0, fma(x1, s1, fma(x2, s2, fma(x3, s3, sum_xs))));
        j += 4;
    }
    while (j < period) {
        const int idx = i - (j + 1);
        double x = d[idx]; if (isnan(x)) x = 0.0;
        const double c = neo_s3_cc_cos(w, j);
        const double s = neo_s3_cc_sin(w, j);
        sum_x  += x;
        sum_x2 = fma(x, x, sum_x2);
        sum_xc = fma(x, c, sum_xc);
        sum_xs = fma(x, s, sum_xs);
        j += 1;
    }

    *o_sx = sum_x; *o_sx2 = sum_x2; *o_sxc = sum_xc; *o_sxs = sum_xs;
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
    const int period = periods[r];

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();

    const double half_pi = asin(1.0);
    const double two_pi  = 4.0 * asin(1.0);
    const double nn = (double)period;
    const double w  = two_pi / nn;

    // Seed constants, in the CPU's 4-wide association (:591-645).
    double sum_cos = 0.0, sum_sin = 0.0, sum_cos2 = 0.0, sum_sin2 = 0.0;
    {
        int j = 0;
        while (j + 4 <= period) {
            for (int q = 0; q < 4; ++q) {
                const double c = neo_s3_cc_cos(w, j + q);
                const double ys = neo_s3_cc_sin(w, j + q);
                sum_cos += c;
                sum_sin += ys;
                sum_cos2 += c * c;
                sum_sin2 += ys * ys;
            }
            j += 4;
        }
        while (j < period) {
            const double c = neo_s3_cc_cos(w, j);
            const double ys = neo_s3_cc_sin(w, j);
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

    int rebase_interval = 256;
    for (int i = first_valid; i < n; ++i) {
        if (isinf(data[i])) { rebase_interval = 1; break; }
    }

    const double z_re = neo_s3_cc_cos(w, 0);
    const double z_im = neo_s3_cc_sin(w, 0);
    int last_rebase = start_ria;

    double sum_x, sum_x2, sum_xc, sum_xs;
    neo_s3_cc_window_sums(data, w, start_ria, period, &sum_x, &sum_x2, &sum_xc, &sum_xs);

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

        row[i] = r_val;
        (void)half_pi;   // the angle/state outputs are not this entry point's

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
