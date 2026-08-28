#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef DTI_QNAN
#define DTI_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __device__ __forceinline__ void ema_kahan_step(const float alpha,
                                                      const float x,
                                                      float &e, float &c)
{

    const float diff   = x - e;
    const float delta  = fmaf(alpha, diff, 0.0f);
    const float y      = delta - c;
    const float t      = e + y;
    c                  = (t - e) - y;
    e                  = t;
}


extern "C" __global__ void dti_build_x_ax_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int series_len,
    int start,
    float* __restrict__ x,
    float* __restrict__ ax
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= series_len) return;

    if (idx < start || idx == 0) {
        x[idx] = 0.0f;
        ax[idx] = 0.0f;
        return;
    }

    const float dh = high[idx] - high[idx - 1];
    const float dl = low[idx] - low[idx - 1];
    const float x_hmu = fmaxf(dh, 0.0f);
    const float x_lmd = fmaxf(-dl, 0.0f);
    const float v = x_hmu - x_lmd;
    x[idx] = v;
    ax[idx] = fabsf(v);
}


extern "C" __global__ void dti_batch_f32(
    const float* __restrict__ x,
    const float* __restrict__ ax,
    const int*   __restrict__ r_arr,
    const int*   __restrict__ s_arr,
    const int*   __restrict__ u_arr,
    int series_len,
    int n_combos,
    int start,
    float* __restrict__ out
){

    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_combos;
         row += blockDim.x * gridDim.x)
    {
        const int r = r_arr[row];
        const int s = s_arr[row];
        const int u = u_arr[row];
        float* out_row = out + (size_t)row * series_len;

        if (UNLIKELY(r <= 0 || s <= 0 || u <= 0 || start < 1 || start > series_len)) {
            for (int i = 0; i < series_len; ++i) out_row[i] = DTI_QNAN;
            continue;
        }


        for (int i = 0; i < start; ++i) out_row[i] = DTI_QNAN;


        const float ar = 2.0f / (float(r) + 1.0f);
        const float as_ = 2.0f / (float(s) + 1.0f);
        const float au = 2.0f / (float(u) + 1.0f);


        float e0_r = 0.0f, e0_s = 0.0f, e0_u = 0.0f;
        float e1_r = 0.0f, e1_s = 0.0f, e1_u = 0.0f;
        float c0_r = 0.0f, c0_s = 0.0f, c0_u = 0.0f;
        float c1_r = 0.0f, c1_s = 0.0f, c1_u = 0.0f;


        for (int i = start; i < series_len; ++i) {
            const float xi  = x[i];
            const float axi = ax[i];


            ema_kahan_step(ar,  xi,   e0_r, c0_r);
            ema_kahan_step(as_, e0_r, e0_s, c0_s);
            ema_kahan_step(au,  e0_s, e0_u, c0_u);

            ema_kahan_step(ar,  axi,  e1_r, c1_r);
            ema_kahan_step(as_, e1_r, e1_s, c1_s);
            ema_kahan_step(au,  e1_s, e1_u, c1_u);

            const float den = e1_u;
            out_row[i] = (den == den && den != 0.0f) ? (100.0f * (e0_u / den)) : 0.0f;
        }
    }
}


extern "C" __global__ void dti_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int r,
    int s,
    int u,
    float* __restrict__ out_tm
){
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(fv < 0 || fv >= series_len || r <= 0 || s <= 0 || u <= 0)) {

        for (int t = 0; t < series_len; ++t)
            out_tm[(size_t)t * num_series + series] = DTI_QNAN;
        return;
    }

    const int start = fv + 1;
    if (UNLIKELY(start >= series_len)) {
        for (int t = 0; t < series_len; ++t)
            out_tm[(size_t)t * num_series + series] = DTI_QNAN;
        return;
    }


    for (int t = 0; t < start; ++t)
        out_tm[(size_t)t * num_series + series] = DTI_QNAN;


    const float ar  = 2.0f / (float(r) + 1.0f);
    const float as_ = 2.0f / (float(s) + 1.0f);
    const float au  = 2.0f / (float(u) + 1.0f);


    float e0_r = 0.0f, e0_s = 0.0f, e0_u = 0.0f;
    float e1_r = 0.0f, e1_s = 0.0f, e1_u = 0.0f;
    float c0_r = 0.0f, c0_s = 0.0f, c0_u = 0.0f;
    float c1_r = 0.0f, c1_s = 0.0f, c1_u = 0.0f;

    const size_t stride = (size_t)num_series;


    size_t idx_prev = (size_t)fv * stride + series;
    float prev_h = high_tm[idx_prev];
    float prev_l = low_tm [idx_prev];


    size_t idx = (size_t)start * stride + series;

    for (int t = start; t < series_len; ++t, idx += stride) {
        const float h  = high_tm[idx];
        const float l  = low_tm[idx];
        const float dh = h - prev_h;
        const float dl = l - prev_l;
        prev_h = h;
        prev_l = l;


        const float up  = fmaxf(dh, 0.0f);
        const float dn  = fmaxf(-dl, 0.0f);
        const float xi  = up - dn;
        const float axi = fabsf(xi);


        ema_kahan_step(ar,  xi,   e0_r, c0_r);
        ema_kahan_step(as_, e0_r, e0_s, c0_s);
        ema_kahan_step(au,  e0_s, e0_u, c0_u);

        ema_kahan_step(ar,  axi,  e1_r, c1_r);
        ema_kahan_step(as_, e1_r, e1_s, c1_s);
        ema_kahan_step(au,  e1_s, e1_u, c1_u);

        const float den = e1_u;
        out_tm[idx] = (den == den && den != 0.0f) ? (100.0f * (e0_u / den)) : 0.0f;
    }
}


// ===========================================================================
// S3 f64 LANE — dti (Directional Trend Index)
// ===========================================================================
// Reference: src/indicators/dti.rs
//   dti_with_kernel (:308) — first_valid, Err branches,
//                            alloc_with_nan_prefix(len, first_valid + 1)
//   dti_scalar (:385)      — the six chained EMAs
//
// PERIOD-INVARIANT. compute_dti_batch reads r (14), s (10) and u (5) and never
// reads period, so every row of a period sweep is identical.
//
// FIRST-VALID. (0..len).find(|i| !high[i].is_nan() && !low[i].is_nan()) — the
// first bar at which BOTH high and low are non-NaN, i.e.
// F64FirstValidRule::AllInputsNonNan over a HighLow input. Close is never
// scanned and must not be part of the shape, or a late close would shift the
// seed.
//
// THE RECURRENCE IS HIDDEN BEHIND SIX SCALARS, NOT ONE. e0_r/e0_s/e0_u and
// e1_r/e1_s/e1_u are all carried across bars and all seeded to 0.0 — NOT to the
// first sample. That seeding is the whole warm-up behaviour: the series decays
// toward the true EMA rather than starting on it, so shortening or lengthening
// the prefix changes every early value.
//
// ROUNDING. alpha * x + one_minus_alpha * prev is TWO products and ONE add —
// three roundings. The CPU does not use mul_add here (:419-425) and neither
// does this kernel; fma() would be two and would disagree in the last bits of
// every bar, compounded six deep.
//
// THE ZERO/NaN GUARD. out[i] = if !e1_u.is_nan() && e1_u != 0.0 { 100*e0_u/e1_u }
// else { 0.0 }. Note it emits 0.0, NOT NaN, and note that a NaN denominator
// produces 0.0 rather than propagating. Transcribed literally; !isnan(x) &&
// x != 0.0 has the same truth table.
//
// One thread per column.
// ===========================================================================

#define NEO_S3_DTI_R 14
#define NEO_S3_DTI_S 10
#define NEO_S3_DTI_U 5

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_dti_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int rr = blockIdx.x * blockDim.x + threadIdx.x;
    if (rr >= n_combos) return;
    (void)periods;   // PERIOD-INVARIANT — see the header.

    double* __restrict__ row = out + (size_t)rr * (size_t)n;

    const int r = NEO_S3_DTI_R;
    const int s = NEO_S3_DTI_S;
    const int u = NEO_S3_DTI_U;

    bool bad_period = false;
    {
        const int ps[3] = { r, s, u };
        for (int q = 0; q < 3; ++q) {
            const int p = ps[q];
            if (p == 0 || p > n) { bad_period = true; break; }
            if ((n - first_valid) < p) { bad_period = true; break; }
        }
    }

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        bad_period;
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int prefix = first_valid + 1;
    for (int i = 0; i < prefix && i < n; ++i) row[i] = neo_s3_qnan();

    const double alpha_r = 2.0 / ((double)r + 1.0);
    const double alpha_s = 2.0 / ((double)s + 1.0);
    const double alpha_u = 2.0 / ((double)u + 1.0);
    const double alpha_r_1 = 1.0 - alpha_r;
    const double alpha_s_1 = 1.0 - alpha_s;
    const double alpha_u_1 = 1.0 - alpha_u;

    double e0_r = 0.0, e0_s = 0.0, e0_u = 0.0;
    double e1_r = 0.0, e1_s = 0.0, e1_u = 0.0;

    row[first_valid] = neo_s3_qnan();   // dti.rs:410, written by the scalar

    for (int i = first_valid + 1; i < n; ++i) {
        const double dh = high[i] - high[i - 1];
        const double dl = low[i] - low[i - 1];
        const double x_hmu = (dh > 0.0) ? dh : 0.0;
        const double x_lmd = (dl < 0.0) ? -dl : 0.0;
        const double x_price = x_hmu - x_lmd;
        const double x_price_abs = fabs(x_price);

        e0_r = alpha_r * x_price + alpha_r_1 * e0_r;
        e0_s = alpha_s * e0_r + alpha_s_1 * e0_s;
        e0_u = alpha_u * e0_s + alpha_u_1 * e0_u;

        e1_r = alpha_r * x_price_abs + alpha_r_1 * e1_r;
        e1_s = alpha_s * e1_r + alpha_s_1 * e1_s;
        e1_u = alpha_u * e1_s + alpha_u_1 * e1_u;

        row[i] = (!isnan(e1_u) && e1_u != 0.0) ? (100.0 * e0_u / e1_u) : 0.0;
    }
}
