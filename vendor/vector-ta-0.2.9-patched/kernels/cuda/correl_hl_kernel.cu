#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "ds_float2.cuh"


extern "C" __global__ void correl_hl_build_prefix_ds_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    float2* __restrict__ ps_h,
    float2* __restrict__ ps_h2,
    float2* __restrict__ ps_l,
    float2* __restrict__ ps_l2,
    float2* __restrict__ ps_hl,
    int* __restrict__ ps_nan)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    double sum_h = 0.0;
    double sum_h2 = 0.0;
    double sum_l = 0.0;
    double sum_l2 = 0.0;
    double sum_hl = 0.0;
    int nan_count = 0;

    ps_h[0] = make_float2(0.0f, 0.0f);
    ps_h2[0] = make_float2(0.0f, 0.0f);
    ps_l[0] = make_float2(0.0f, 0.0f);
    ps_l2[0] = make_float2(0.0f, 0.0f);
    ps_hl[0] = make_float2(0.0f, 0.0f);
    ps_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float h = high[i];
            const float l = low[i];
            if (isnan(h) || isnan(l)) {
                nan_count += 1;
            } else {
                const double hd = (double)h;
                const double ld = (double)l;
                sum_h += hd;
                sum_h2 += hd * hd;
                sum_l += ld;
                sum_l2 += ld * ld;
                sum_hl += hd * ld;
            }
        }

        const float sum_h_hi = (float)sum_h;
        const float sum_h2_hi = (float)sum_h2;
        const float sum_l_hi = (float)sum_l;
        const float sum_l2_hi = (float)sum_l2;
        const float sum_hl_hi = (float)sum_hl;
        ps_h[i + 1] = make_float2((float)(sum_h - (double)sum_h_hi), sum_h_hi);
        ps_h2[i + 1] = make_float2((float)(sum_h2 - (double)sum_h2_hi), sum_h2_hi);
        ps_l[i + 1] = make_float2((float)(sum_l - (double)sum_l_hi), sum_l_hi);
        ps_l2[i + 1] = make_float2((float)(sum_l2 - (double)sum_l2_hi), sum_l2_hi);
        ps_hl[i + 1] = make_float2((float)(sum_hl - (double)sum_hl_hi), sum_hl_hi);
        ps_nan[i + 1] = nan_count;
    }
}

extern "C" __global__ void correl_hl_build_prefix_f64(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    double* __restrict__ ps_h,
    double* __restrict__ ps_h2,
    double* __restrict__ ps_l,
    double* __restrict__ ps_l2,
    double* __restrict__ ps_hl,
    int* __restrict__ ps_nan)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    double sum_h = 0.0;
    double sum_h2 = 0.0;
    double sum_l = 0.0;
    double sum_l2 = 0.0;
    double sum_hl = 0.0;
    int nan_count = 0;

    ps_h[0] = 0.0;
    ps_h2[0] = 0.0;
    ps_l[0] = 0.0;
    ps_l2[0] = 0.0;
    ps_hl[0] = 0.0;
    ps_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float h = high[i];
            const float l = low[i];
            if (isnan(h) || isnan(l)) {
                nan_count += 1;
            } else {
                const double hd = (double)h;
                const double ld = (double)l;
                sum_h += hd;
                sum_h2 += hd * hd;
                sum_l += ld;
                sum_l2 += ld * ld;
                sum_hl += hd * ld;
            }
        }

        ps_h[i + 1] = sum_h;
        ps_h2[i + 1] = sum_h2;
        ps_l[i + 1] = sum_l;
        ps_l2[i + 1] = sum_l2;
        ps_hl[i + 1] = sum_hl;
        ps_nan[i + 1] = nan_count;
    }
}

extern "C" __global__ void correl_hl_batch_f32(
    const double* __restrict__ ps_h,
    const double* __restrict__ ps_h2,
    const double* __restrict__ ps_l,
    const double* __restrict__ ps_l2,
    const double* __restrict__ ps_hl,
    const int* __restrict__ ps_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out
){
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double inv_pf = 1.0 / (double)period;

    while (t < len) {
        float out_val = nan_f;
        if (t >= warm) {
            const int end = t + 1;
            int start = end - period;
            if (start < 0) start = 0;
            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {
                const double sum_h  = ps_h[end]  - ps_h[start];
                const double sum_l  = ps_l[end]  - ps_l[start];
                const double sum_h2 = ps_h2[end] - ps_h2[start];
                const double sum_l2 = ps_l2[end] - ps_l2[start];
                const double sum_hl = ps_hl[end] - ps_hl[start];
                const double cov  = sum_hl - (sum_h * sum_l) * inv_pf;
                const double varh = sum_h2 - (sum_h * sum_h) * inv_pf;
                const double varl = sum_l2 - (sum_l * sum_l) * inv_pf;
                if (varh > 0.0 && varl > 0.0) {
                    const double denom = sqrt(varh) * sqrt(varl);
                    if (denom > 0.0 && !isnan(denom)) {
                        out_val = (float)(cov / denom);
                    } else {
                        out_val = 0.0f;
                    }
                } else {
                    out_val = 0.0f;
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void correl_hl_batch_f32ds(
    const float2* __restrict__ ps_h,
    const float2* __restrict__ ps_h2,
    const float2* __restrict__ ps_l,
    const float2* __restrict__ ps_l2,
    const float2* __restrict__ ps_hl,
    const int*    __restrict__ ps_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    int n_combos,
    float*        __restrict__ out
){
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int   warm     = first_valid + period - 1;
    const int   row_off  = combo * len;
    const float nan_f    = __int_as_float(0x7fffffff);
    const float inv_pf   = 1.0f / (float)period;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    auto rsqrt_refine = [](float x){
        float y = rsqrtf(x);
        y = y * (1.5f - 0.5f * x * y * y);
        return y;
    };

    while (t < len) {
        float out_val = nan_f;

        if (t >= warm) {
            const int end   = t + 1;
            int       start = end - period;
            if (start < 0) start = 0;

            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {

                float2 ah = ps_h[end];
                float2 bh = ps_h[start];
                float2 al = ps_l[end];
                float2 bl = ps_l[start];
                float2 ah2 = ps_h2[end];
                float2 bh2 = ps_h2[start];
                float2 al2 = ps_l2[end];
                float2 bl2 = ps_l2[start];
                float2 ahl = ps_hl[end];
                float2 bhl = ps_hl[start];

                dsf sum_h  = ds_sub(ds_make(ah.y, ah.x), ds_make(bh.y, bh.x));
                dsf sum_l  = ds_sub(ds_make(al.y, al.x), ds_make(bl.y, bl.x));
                dsf sum_h2 = ds_sub(ds_make(ah2.y, ah2.x), ds_make(bh2.y, bh2.x));
                dsf sum_l2 = ds_sub(ds_make(al2.y, al2.x), ds_make(bl2.y, bl2.x));
                dsf sum_hl = ds_sub(ds_make(ahl.y, ahl.x), ds_make(bhl.y, bhl.x));


                dsf cov  = ds_sub(sum_hl, ds_scale(ds_mul(sum_h, sum_l), inv_pf));

                dsf varh = ds_sub(sum_h2, ds_scale(ds_mul(sum_h, sum_h), inv_pf));
                dsf varl = ds_sub(sum_l2, ds_scale(ds_mul(sum_l, sum_l), inv_pf));

                float cov_f  = ds_to_f(cov);
                float varh_f = ds_to_f(varh);
                float varl_f = ds_to_f(varl);

                if (varh_f > 0.0f && varl_f > 0.0f) {
                    float prod = varh_f * varl_f;
                    if (prod > 0.0f && isfinite(prod)) {
                        float inv = rsqrt_refine(prod);
                        out_val = cov_f * inv;
                    } else {
                        out_val = 0.0f;
                    }
                } else {
                    out_val = 0.0f;
                }
            }
        }

        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void correl_hl_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm
){
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int stride = num_series;


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;


    const int init_start = first_valid;
    const int init_end = min(first_valid + period, series_len);
    double sum_h = 0.0, sum_l = 0.0, sum_h2 = 0.0, sum_l2 = 0.0, sum_hl = 0.0;
    int nan_in_win = 0;
    for (int i = init_start; i < init_end; ++i) {
        const float h = high_tm[i * stride + series];
        const float l = low_tm[i * stride + series];
        if (isnan(h) || isnan(l)) {
            nan_in_win += 1;
        } else {
            const double hd = (double)h;
            const double ld = (double)l;
            sum_h += hd;
            sum_l += ld;
            sum_h2 += hd * hd;
            sum_l2 += ld * ld;
            sum_hl += hd * ld;
        }
    }

    const double inv_pf = 1.0 / (double)period;
    const int warm = first_valid + period - 1;
    if (warm < series_len && nan_in_win == 0) {
        const double cov  = sum_hl - (sum_h * sum_l) * inv_pf;
        const double varh = sum_h2 - (sum_h * sum_h) * inv_pf;
        const double varl = sum_l2 - (sum_l * sum_l) * inv_pf;
        float out0 = 0.0f;
        if (varh > 0.0 && varl > 0.0) {
            const double denom = sqrt(varh) * sqrt(varl);
            out0 = (float)((denom > 0.0) ? (cov / denom) : 0.0);
        }
        out_tm[warm * stride + series] = out0;
    }


    for (int t = warm + 1; t < series_len; ++t) {
        const int old_idx = t - period;
        const float old_h = high_tm[old_idx * stride + series];
        const float old_l = low_tm[old_idx * stride + series];
        const float new_h = high_tm[t * stride + series];
        const float new_l = low_tm[t * stride + series];

        if (isnan(old_h) || isnan(old_l) || isnan(new_h) || isnan(new_l)) {

            sum_h = sum_l = sum_h2 = sum_l2 = sum_hl = 0.0;
            nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float hh = high_tm[k * stride + series];
                const float ll = low_tm[k * stride + series];
                if (isnan(hh) || isnan(ll)) {
                    nan_in_win += 1;
                } else {
                    const double hd = (double)hh;
                    const double ld = (double)ll;
                    sum_h += hd;
                    sum_l += ld;
                    sum_h2 += hd * hd;
                    sum_l2 += ld * ld;
                    sum_hl += hd * ld;
                }
            }
        } else {

            const double oh = (double)old_h, ol = (double)old_l;
            const double nh = (double)new_h, nl = (double)new_l;
            sum_h += nh - oh;
            sum_l += nl - ol;
            sum_h2 += nh * nh - oh * oh;
            sum_l2 += nl * nl - ol * ol;
            sum_hl = fma(nh, nl, sum_hl - oh * ol);
        }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            const double cov  = sum_hl - (sum_h * sum_l) * inv_pf;
            const double varh = sum_h2 - (sum_h * sum_h) * inv_pf;
            const double varl = sum_l2 - (sum_l * sum_l) * inv_pf;
            float outv = 0.0f;
            if (varh > 0.0 && varl > 0.0) {
                const double denom = sqrt(varh) * sqrt(varl);
                outv = (float)((denom > 0.0) ? (cov / denom) : 0.0);
            }
            out_tm[t * stride + series] = outv;
        }
    }
}


// ===========================================================================
// f64 LANE  --  shard S6
//
// CPU reference: `correl_hl_scalar` (src/indicators/correl_hl.rs:321), reached
// from `correl_hl_with_kernel` (:267). NOTE `correl_hl_prepare` (:236-242)
// pins `Kernel::Auto -> Kernel::Scalar` on x86_64+nightly-avx, so the scalar
// path IS the CPU answer here; there is no AVX/scalar ambiguity to settle.
//
// Contract reproduced line for line:
//   * first_valid  = first index where high AND low are both non-NaN
//                    (correl_hl.rs:223-227) -> F64FirstValidRule::AllInputsNonNan.
//   * warm         = first + period - 1 (:271). Everything before it is NaN.
//   * seed window  = [first, first + period), summed FOUR-WIDE
//                    (`sum_h += h0 + h1 + h2 + h3`, :353-357) with a 1-wide
//                    tail. The association is load-bearing: a plain 1-wide
//                    loop gives a different sum and therefore a different
//                    correlation in the last ULPs.
//   * steady state = O(1) update, EXCEPT that a NaN at either end of the
//                    window forces a full 4-wide recompute of the whole window
//                    (:391-427). Reproduced exactly, including that the
//                    recompute zeroes the accumulators first.
//   * sum_hl update uses `new_h.mul_add(new_l, sum_hl - old_hl)` (:434) -- ONE
//                    rounding -> `fma(new_h, new_l, sum_hl - old_hl)`. The four
//                    other accumulators use plain +/- and stay plain.
//
// NaN semantics: `corr_from_sums` guards with `var_h <= 0.0 || var_l <= 0.0`.
// A NaN variance makes BOTH comparisons false in Rust and in C, so the
// division runs and NaN propagates. That is the CPU's behaviour and it is
// preserved deliberately -- this is one comparison chain that must NOT become
// fmax/fmin, because turning NaN into 0.0 here would invent a correlation.
//
// f32 -> f64 audit for this file's new code: no f32 literal, no f32-suffixed
// math function (`sqrtf` -> `sqrt`), no fast-math intrinsic (the f32 lane
// above uses `rsqrtf` and `__int_as_float`; neither appears below --
// `rsqrt(var_h*var_l)` is NOT the same expression as
// `cov / (sqrt(var_h) * sqrt(var_l))` and the CPU writes the latter).
// The f32 NaN pattern `__int_as_float(0x7fffffff)` becomes the f64 quiet-NaN
// bit pattern `0x7ff8000000000000`, which is also the pattern
// `correl_hl_into_slice` writes (:296).
// No epsilon exists in this indicator, so none was carried over.
// ===========================================================================

static __device__ __forceinline__ double correl_hl_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `corr_from_sums`, correl_hl.rs:329-346.
static __device__ __forceinline__ double correl_hl_corr_from_sums_f64(
    double sum_h, double sum_h2, double sum_l, double sum_l2, double sum_hl, double inv_pf)
{
    const double cov   = sum_hl - (sum_h * sum_l) * inv_pf;
    const double var_h = sum_h2 - (sum_h * sum_h) * inv_pf;
    const double var_l = sum_l2 - (sum_l * sum_l) * inv_pf;
    if (var_h <= 0.0 || var_l <= 0.0) {
        return 0.0;
    }
    return cov / (sqrt(var_h) * sqrt(var_l));
}

// The 4-wide window sum of correl_hl.rs:350-373 and :402-427, shared so the
// seed and the NaN-recompute cannot drift apart.
static __device__ __forceinline__ void correl_hl_window_sums_f64(
    const double* __restrict__ high, const double* __restrict__ low,
    int start, int end,
    double* sum_h, double* sum_l, double* sum_h2, double* sum_l2, double* sum_hl)
{
    double sh = 0.0, sl = 0.0, sh2 = 0.0, sl2 = 0.0, shl = 0.0;
    int j = start;
    while (j + 4 <= end) {
        const double h0 = high[j + 0], l0 = low[j + 0];
        const double h1 = high[j + 1], l1 = low[j + 1];
        const double h2 = high[j + 2], l2 = low[j + 2];
        const double h3 = high[j + 3], l3 = low[j + 3];
        sh  += h0 + h1 + h2 + h3;
        sl  += l0 + l1 + l2 + l3;
        sh2 += h0 * h0 + h1 * h1 + h2 * h2 + h3 * h3;
        sl2 += l0 * l0 + l1 * l1 + l2 * l2 + l3 * l3;
        shl += h0 * l0 + h1 * l1 + h2 * l2 + h3 * l3;
        j += 4;
    }
    while (j < end) {
        const double h = high[j], l = low[j];
        sh  += h;
        sl  += l;
        sh2 += h * h;
        sl2 += l * l;
        shl += h * l;
        j += 1;
    }
    *sum_h = sh; *sum_l = sl; *sum_h2 = sh2; *sum_l2 = sl2; *sum_hl = shl;
}

extern "C" __global__
void correl_hl_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = correl_hl_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const int first  = (first_valid < 0) ? 0 : first_valid;

    // `correl_hl_prepare` rejects period == 0, period > len and
    // len - first < period. The device cannot return an Err, so it emits the
    // all-NaN column those errors would have produced no series for.
    if (period <= 0 || period > n || first >= n || (n - first) < period) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const int init_end = first + period;   // exclusive
    const int warm     = init_end - 1;     // correl_hl.rs:271

    for (int t = 0; t < warm; ++t) row[t] = nan_d;

    const double inv_pf = 1.0 / static_cast<double>(period);

    double sum_h, sum_l, sum_h2, sum_l2, sum_hl;
    correl_hl_window_sums_f64(high, low, first, init_end,
                              &sum_h, &sum_l, &sum_h2, &sum_l2, &sum_hl);

    row[warm] = correl_hl_corr_from_sums_f64(sum_h, sum_h2, sum_l, sum_l2, sum_hl, inv_pf);

    for (int i = init_end; i < n; ++i) {
        const int old_idx = i - period;
        const double old_h = high[old_idx], old_l = low[old_idx];
        const double new_h = high[i],       new_l = low[i];

        if (isnan(old_h) || isnan(old_l) || isnan(new_h) || isnan(new_l)) {
            correl_hl_window_sums_f64(high, low, i + 1 - period, i + 1,
                                      &sum_h, &sum_l, &sum_h2, &sum_l2, &sum_hl);
        } else {
            sum_h  += new_h - old_h;
            sum_l  += new_l - old_l;
            sum_h2 += new_h * new_h - old_h * old_h;
            sum_l2 += new_l * new_l - old_l * old_l;
            const double old_hl = old_h * old_l;
            sum_hl = fma(new_h, new_l, sum_hl - old_hl);   // correl_hl.rs:434
        }

        row[i] = correl_hl_corr_from_sums_f64(sum_h, sum_h2, sum_l, sum_l2, sum_hl, inv_pf);
    }
}
