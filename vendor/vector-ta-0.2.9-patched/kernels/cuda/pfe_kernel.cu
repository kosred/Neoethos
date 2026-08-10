#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>


static __device__ __forceinline__ void two_sumf(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


static __device__ __forceinline__ void quick_two_sumf(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}


static __device__ __forceinline__ void f2_add_scalar(float &a_hi, float &a_lo, float x) {
    float s, e1; two_sumf(a_hi, x, s, e1);
    float e = a_lo + e1;
    quick_two_sumf(s, e, a_hi, a_lo);
}


static __device__ __forceinline__ float sqrt1p_squaref(float d) {
    return sqrtf(fmaf(d, d, 1.0f));
}

extern "C" __global__
void pfe_prepare_data_f32(const float* __restrict__ data,
                          int len,
                          int first_valid,
                          float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    if (first_valid > 0 && i <= first_valid) {
        out[i] = data[first_valid];
    } else {
        out[i] = data[i];
    }
}


extern "C" __global__
void pfe_batch_f32(const float* __restrict__ data,
                   int len,
                   int first_valid,
                   const int* __restrict__ periods,
                   const int* __restrict__ smoothings,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int smoothing = smoothings[combo];
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int row_off = combo * len;
    const int start = first_valid + period;


    for (int t = 0; t < ((start < len) ? start : len); ++t) out[row_off + t] = CUDART_NAN_F;
    if (start >= len) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;


    float denom = 0.0f;
    const bool use_ring = period <= 256;
    float ring[256];
    int head = 0;
    for (int j = start - period; j < start; ++j) {
        const float d = data[j + 1] - data[j];
        const float s = sqrt1p_squaref(d);
        denom += s;
        if (use_ring) { ring[head++] = s; }
    }
    head = 0;

    bool  ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < len; ++t) {
        const float cur  = data[t];
        const float past = data[t - period];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));

        float raw = 0.0f;
        if (denom > 0.0f) raw = 100.0f * (long_leg / denom);
        const float signed_val = copysignf(raw, diff);

        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out[row_off + t] = ema;

        if (t + 1 == len) break;

        const float add_d = data[t + 1] - data[t];
        const float add_s = sqrt1p_squaref(add_d);
        float sub_s;
        if (use_ring) {
            sub_s = ring[head];
            ring[head] = add_s;
            head = (head + 1) % period;
        } else {
            const int oldest = t - period + 1;
            const float sd = data[oldest + 1] - data[oldest];
            sub_s = sqrt1p_squaref(sd);
        }
        denom += add_s - sub_s;
    }
}


extern "C" __global__
void pfe_batch_prefix_f32(const float* __restrict__ data,
                          const double* __restrict__ prefix,
                          int len,
                          int first_valid,
                          const int* __restrict__ periods,
                          const int* __restrict__ smoothings,
                          int n_combos,
                          float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int smoothing = smoothings[combo];
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int row_off = combo * len;
    const int start = first_valid + period;
    for (int t = 0; t < ((start < len) ? start : len); ++t) out[row_off + t] = CUDART_NAN_F;
    if (start >= len) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    bool ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < len; ++t) {
        const float cur  = data[t];
        const float past = data[t - period];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));


        const double denom_d = prefix[t] - prefix[t - period];
        const float denom = (float)denom_d;

        if (!(denom > 0.0f)) {
            out[row_off + t] = CUDART_NAN_F;
            continue;
        }

        const float raw = 100.0f * (long_leg / denom);
        const float signed_val = copysignf(raw, diff);
        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out[row_off + t] = ema;
    }
}


extern "C" __global__
void pfe_many_series_one_param_time_major_f32(const float* __restrict__ data_tm,
                                              const int*   __restrict__ first_valids,
                                              int cols,
                                              int rows,
                                              int period,
                                              int smoothing,
                                              float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || period <= 0 || smoothing <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = CUDART_NAN_F;
        return;
    }

    const int start = fv + period;
    for (int t = 0; t < ((start < rows) ? start : rows); ++t) out_tm[t * cols + s] = CUDART_NAN_F;
    if (start >= rows) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    float denom = 0.0f;
    for (int j = fv; j < start; ++j) {
        const float d = data_tm[(j + 1) * cols + s] - data_tm[j * cols + s];
        denom += sqrt1p_squaref(d);
    }
    int oldest = fv;

    bool  ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < rows; ++t) {
        const float cur  = data_tm[t * cols + s];
        const float past = data_tm[(t - period) * cols + s];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));
        const float raw = (denom > 0.0f) ? (100.0f * (long_leg / denom)) : 0.0f;
        const float signed_val = copysignf(raw, diff);

        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out_tm[t * cols + s] = ema;

        if (t + 1 == rows) break;
        const float add_d = data_tm[(t + 1) * cols + s] - data_tm[t * cols + s];
        const float sub_d = data_tm[(oldest + 1) * cols + s] - data_tm[oldest * cols + s];
        denom += sqrt1p_squaref(add_d) - sqrt1p_squaref(sub_d);
        ++oldest;
    }
}


extern "C" __global__
void pfe_build_steps_f32(const float* __restrict__ data,
                         int len,
                         float* __restrict__ steps_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) { steps_out[0] = 0.0f; }
    for (int t = i + 1; t < len; t += gridDim.x * blockDim.x) {
        const float d = data[t] - data[t - 1];
        steps_out[t] = sqrt1p_squaref(d);
    }
}


extern "C" __global__
void pfe_build_prefix_float2_serial(const float* __restrict__ steps,
                                    int len,
                                    float* __restrict__ pref_hi,
                                    float* __restrict__ pref_lo) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float s_hi = 0.0f, s_lo = 0.0f;
    pref_hi[0] = 0.0f; pref_lo[0] = 0.0f;
    for (int t = 1; t < len; ++t) {
        f2_add_scalar(s_hi, s_lo, steps[t]);
        pref_hi[t] = s_hi;
        pref_lo[t] = s_lo;
    }
}


extern "C" __global__
void pfe_many_params_prefix_f32(const float* __restrict__ data,
                                const float* __restrict__ pref_hi,
                                const float* __restrict__ pref_lo,
                                int len,
                                int first_valid,
                                const int* __restrict__ periods,
                                const int* __restrict__ smoothings,
                                int n_combos,
                                float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period    = periods[combo];
    const int smoothing = smoothings[combo];
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int row_off = combo * len;
    const int start = first_valid + period;
    for (int t = 0; t < ((start < len) ? start : len); ++t) out[row_off + t] = CUDART_NAN_F;
    if (start >= len) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    bool  ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < len; ++t) {
        const float cur  = data[t];
        const float past = data[t - period];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));

        const float d_hi = pref_hi[t] - pref_hi[t - period];
        const float d_lo = pref_lo[t] - pref_lo[t - period];
        const float denom = d_hi + d_lo;

        if (!(denom > 0.0f)) { out[row_off + t] = CUDART_NAN_F; continue; }

        const float raw = 100.0f * (long_leg / denom);
        const float signed_val = copysignf(raw, diff);
        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out[row_off + t] = ema;
    }
}


// ===========================================================================
// S3 f64 LANE — pfe (Polarized Fractal Efficiency)
// ===========================================================================
// Reference: src/indicators/pfe.rs
//   pfe_prepare (:216)       — first_valid + every Err branch
//   pfe_with_kernel (:430)   — alloc_with_nan_prefix(len, first + period)
//   pfe_row_scalar (:864)    — the arithmetic
// Batch defaults: period 10, smoothing 5, source close. `smoothing` is not
// swept, so it is the constant below; `period` IS the swept parameter.
//
// THE RING BUFFER IS NOT NEEDED, AND THIS TIME IT IS TRIVIAL.
// The CPU keeps seg[0..period] of segment lengths so it can subtract the one
// leaving the window (:947, :951). Unlike a running mean, each stored value is
// a PURE FUNCTION OF TWO ADJACENT SAMPLES:
//     s(k) = sqrt(fma(data[k]-data[k-1], data[k]-data[k-1], 1.0))
// and the element retired at bar t is exactly s(t - period + 1) — verified from
// the head walk: at t == start, head == 0 and seg[0] was written as s(base+0)
// with base == first + 1 == start - period + 1. So it is recomputed here rather
// than stored, which removes the only reason this kernel would need device
// scratch, and it is EXACT, not approximate: the same expression on the same
// two samples.
//
// THE SEED SUM GROUPS BY FOUR. :907 accumulates denom += s0 + s1 + s2 + s3 for
// as many whole groups of four as fit, then one at a time. Different
// association from a plain ascending sum, reproduced literally.
//
// THE EPSILON IS f64::EPSILON, USED AS A DENOMINATOR FLOOR (:928):
//     if denom <= f64::EPSILON { 0.0 } else { 100.0 * long_leg / denom }
// 2.220446049250313e-16. An f32 port cannot express this test — f32 epsilon is
// 1.1920929e-7, nine orders of magnitude larger — so the f32 kernels above zero
// out windows the reference scores normally.
//
// ROUNDING.
//   d.mul_add(d, 1.0)      → ONE fma, then sqrt. Not d*d + 1.0 (two).
//   diff.mul_add(diff, p2) → ONE fma over p*p. Not diff*diff + p*p.
//   alpha.mul_add(signed, one_minus_alpha * ema_val) → product then fma: TWO.
//
// SIGN, NOT abs. signed = if diff > 0.0 { raw } else { -raw } — a diff of
// exactly 0.0 takes the NEGATIVE branch and emits -0.0-signed raw. Keeping the
// comparison as written preserves that.
//
// One thread per column: denom and the EMA are carried across bars.
// ===========================================================================

#define NEO_S3_PFE_SMOOTHING 5

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// s(k) = sqrt(1 + (data[k] - data[k-1])^2), one fma then one sqrt.
__device__ __forceinline__ double neo_s3_pfe_seg(const double* __restrict__ d, int k) {
    const double dd = d[k] - d[k - 1];
    return sqrt(fma(dd, dd, 1.0));
}

extern "C" __global__ void neoethos_pfe_batch_f64(
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
    const int smoothing = NEO_S3_PFE_SMOOTHING;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period + 1) ||
        (smoothing == 0);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + period;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();
    if (warm >= n) return;

    const int start = first_valid + period;
    const double p  = (double)period;
    const double p2 = p * p;
    const double alpha = 2.0 / ((double)smoothing + 1.0);
    const double one_minus_alpha = 1.0 - alpha;

    const int base = start - period + 1;   // == first_valid + 1

    // Seed denom in the CPU's 4-wide association, then the tail.
    double denom = 0.0;
    int j = 0;
    const int stop = period & ~3;
    while (j < stop) {
        const double s0 = neo_s3_pfe_seg(data, base + j);
        const double s1 = neo_s3_pfe_seg(data, base + j + 1);
        const double s2 = neo_s3_pfe_seg(data, base + j + 2);
        const double s3 = neo_s3_pfe_seg(data, base + j + 3);
        denom += s0 + s1 + s2 + s3;
        j += 4;
    }
    while (j < period) {
        denom += neo_s3_pfe_seg(data, base + j);
        j += 1;
    }

    bool ema_started = false;
    double ema_val = 0.0;

    for (int t = start; t < n; ++t) {
        const double cur  = data[t];
        const double past = data[t - period];
        const double diff = cur - past;

        const double long_leg = sqrt(fma(diff, diff, p2));
        const double raw = (denom <= 2.220446049250313e-16)   // f64::EPSILON
            ? 0.0
            : (100.0 * (long_leg / denom));
        const double sgn = (diff > 0.0) ? raw : -raw;

        double val;
        if (!ema_started) {
            ema_started = true;
            ema_val = sgn;
            val = sgn;
        } else {
            ema_val = fma(alpha, sgn, one_minus_alpha * ema_val);
            val = ema_val;
        }
        row[t] = val;

        if (t + 1 < n) {
            // The retired element is s(t - period + 1); the arriving one is
            // s(t + 1). Both recomputed, neither stored — see the header.
            const double old = neo_s3_pfe_seg(data, t - period + 1);
            const double new_d = data[t + 1] - cur;
            const double new_s = sqrt(fma(new_d, new_d, 1.0));
            denom += new_s - old;
        }
    }
}
