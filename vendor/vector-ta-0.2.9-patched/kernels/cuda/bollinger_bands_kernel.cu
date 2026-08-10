#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}


struct dsf { float hi, lo; };

__device__ __forceinline__ dsf ds_make(float hi, float lo) { dsf r; r.hi = hi; r.lo = lo; return r; }
__device__ __forceinline__ dsf ds_from_f(float a) { return ds_make(a, 0.0f); }


__device__ __forceinline__ dsf ds_from_double(double a) {
    float hi = (float)a;
    float lo = (float)(a - (double)hi);
    return ds_make(hi, lo);
}


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b; float bb = s - a; e = (a - (s - bb)) + (b - bb);
}

__device__ __forceinline__ void two_prod(float a, float b, float &p, float &err) {
    p = a * b; err = __fmaf_rn(a, b, -p);
}
__device__ __forceinline__ dsf ds_add(dsf a, dsf b) {
    float s, e; two_sum(a.hi, b.hi, s, e); e += a.lo + b.lo; float t, lo; two_sum(s, e, t, lo); return ds_make(t, lo);
}
__device__ __forceinline__ dsf ds_sub(dsf a, dsf b) { return ds_add(a, ds_make(-b.hi, -b.lo)); }
__device__ __forceinline__ dsf ds_mul_f(dsf a, float b) {
    float p, err; two_prod(a.hi, b, p, err); err += a.lo * b; float t, lo; two_sum(p, err, t, lo); return ds_make(t, lo);
}
__device__ __forceinline__ dsf ds_mul(dsf a, dsf b) {
    float p, err; two_prod(a.hi, b.hi, p, err); err += a.hi * b.lo + a.lo * b.hi; err += a.lo * b.lo; float t, lo; two_sum(p, err, t, lo); return ds_make(t, lo);
}
__device__ __forceinline__ float ds_to_f(dsf a) { return a.hi + a.lo; }


__device__ __forceinline__ dsf load_dsf(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}

extern "C" __global__ void bollinger_bands_build_prefix_f32(
    const float* __restrict__ data,
    int len,
    float2* __restrict__ prefix_sum,
    float2* __restrict__ prefix_sum_sq,
    int* __restrict__ prefix_nan) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    dsf sum = ds_make(0.0f, 0.0f);
    dsf sum_sq = ds_make(0.0f, 0.0f);
    int nan_count = 0;

    prefix_sum[0] = make_float2(0.0f, 0.0f);
    prefix_sum_sq[0] = make_float2(0.0f, 0.0f);
    prefix_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        const float v = data[i];
        if (isnan(v)) {
            ++nan_count;
        } else {
            const dsf x = ds_make(v, 0.0f);
            sum = ds_add(sum, x);
            sum_sq = ds_add(sum_sq, ds_mul(x, x));
        }
        prefix_sum[i + 1] = make_float2(sum.hi, sum.lo);
        prefix_sum_sq[i + 1] = make_float2(sum_sq.hi, sum_sq.lo);
        prefix_nan[i + 1] = nan_count;
    }
}

extern "C" __global__ void bollinger_bands_sma_prefix_f32(
    const float* __restrict__ data,
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int* __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int n_combos,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const float devup = devups[combo];
    const float devdn = devdns[combo];

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nanf = qnan32();
    const float invP = 1.0f / (float)period;


    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float u = nanf, m = nanf, l = nanf;
        if (t >= warm) {
            bool ok = true;
            if (any_nan_since_first) {
                const int nan_since_first = prefix_nan[t + 1] - nan_base;
                ok = (nan_since_first == 0);
            }
            if (ok) {

                const int t1 = t + 1;
                const int s = t1 - period;


                const float2 ps_e  = prefix_sum[t1];
                const float2 ps_s  = prefix_sum[s];
                const float2 ps2_e = prefix_sum_sq[t1];
                const float2 ps2_s = prefix_sum_sq[s];

                const float sum  = (ps_e.x  - ps_s.x)  + (ps_e.y  - ps_s.y);
                const float sum2 = (ps2_e.x - ps2_s.x) + (ps2_e.y - ps2_s.y);

                const float mean = sum * invP;
                const float ex2  = sum2 * invP;
                float var = fmaf(-mean, mean, ex2);
                if (var < 0.0f) var = 0.0f;
                const float sd = sqrtf(var);

                m = mean;
                u = mean + devup * sd;
                l = mean - devdn * sd;
            }
        }
        out_upper[row_off + t]  = u;
        out_middle[row_off + t] = m;
        out_lower[row_off + t]  = l;
        t += stride;
    }
}


extern "C" __global__ void bollinger_bands_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int* __restrict__ prefix_nan_tm,
    int period,
    float devup,
    float devdn,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_upper_tm,
    float* __restrict__ out_middle_tm,
    float* __restrict__ out_lower_tm) {
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (period <= 0) return;
    const int fv = first_valids[s];
    const int warm = fv + period - 1;
    const int stride = num_series;
    const float invP = 1.0f / (float)period;
    const int nan_base = prefix_nan_tm[fv * stride + s];
    const bool any_nan_since_first = (prefix_nan_tm[series_len * stride + s] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + s;
        float u = qnan32(), m = qnan32(), l = qnan32();
        if (t >= warm) {
            bool ok = true;
            if (any_nan_since_first) {
                const int p_idx_t1 = (t + 1) * stride + s;
                const int nan_since_first = prefix_nan_tm[p_idx_t1] - nan_base;
                ok = (nan_since_first == 0);
            }
            if (ok) {
                const int t1 = t + 1;
                const int p_idx = t1 * stride + s;
                const int s_idx = (t1 - period) * stride + s;

                const dsf sum_ds  = ds_sub(load_dsf(prefix_sum_tm,    p_idx), load_dsf(prefix_sum_tm,    s_idx));
                const dsf sum2_ds = ds_sub(load_dsf(prefix_sum_sq_tm, p_idx), load_dsf(prefix_sum_sq_tm, s_idx));


                const double sum_d  = (double)sum_ds.hi  + (double)sum_ds.lo;
                const double sum2_d = (double)sum2_ds.hi + (double)sum2_ds.lo;
                const double invPd = 1.0 / (double)period;
                const double mean_d = sum_d * invPd;
                double var_d = (sum2_d * invPd) - mean_d * mean_d;
                if (var_d < 0.0) var_d = 0.0;
                const float sd = (float)sqrt(var_d);

                const float mean_f = (float)mean_d;
                m = mean_f;
                u = mean_f + devup * sd;
                l = mean_f - devdn * sd;
            }
        }
        out_upper_tm[out_idx]  = u;
        out_middle_tm[out_idx] = m;
        out_lower_tm[out_idx]  = l;
        t += step;
    }
}

/* ===========================================================================
 * S4 f64 LANE — bollinger_bands
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/bollinger_bands.rs
 *   `bb_prepare`                 (:269) — first_valid and the Err branches
 *   `bollinger_bands_with_kernel`(:380) — warm = first + period - 1, three
 *                                         NaN-prefixed outputs
 *   `bollinger_bands_compute_into`(:313)— matype "sma" takes the classic path
 *   `bb_row_scalar_classic_sma`  (:582) — the rolling mean/variance recurrence
 *
 * WHICH BAND THIS EMITS. `compute_bollinger_batch` (cpu_batch.rs:5514) maps
 * output_id "value" -> `upper_band`. The f64 lane carries ONE matrix per
 * indicator, so this kernel emits the UPPER band, which is what a caller
 * asking for `bollinger_bands` with no output_id gets from the CPU. Emitting
 * `middle` would be the more "natural" choice and would silently disagree
 * with the reference.
 *
 * PARAMETERS: the swept int is `period` (CPU default 20). `devup`/`devdn` are
 * held at 2.0, `matype` at "sma" and `devtype` at 0 — the cpu_batch defaults
 * (:5491-5495). devtype 0 is the only value that takes the fused recurrence
 * below; any other routes the CPU through `deviation()`, which is a different
 * indicator's kernel and not this file's business.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. CATASTROPHIC CANCELLATION IN f32. `var = sum_sq/n - mean*mean` subtracts
 *     two nearly equal quantities. On FX closes near 1.1, sum_sq/n and
 *     mean*mean agree to ~7 significant digits — i.e. ALL of them in f32 — so
 *     the f32 kernel's variance is dominated by rounding noise and the band
 *     width is noise times `devup`. This is the single strongest reason this
 *     indicator must not be served in f32, and it is not a tolerance question.
 *  2. `sqrtf` -> `sqrt`; `__fmaf_rn` -> the CPU's plain `+`/`*`. Note the CPU
 *     writes `mean + devup * std0` for the SMA path (:653) — NOT a mul_add —
 *     while the devtype != 0 path at :619 DOES use `mul_add`. Two roundings
 *     here, one there; the classic path is what devtype 0 takes, so two.
 *  3. `__int_as_float(0x7f...)` -> `__longlong_as_double(0x7ff8...)`.
 *  4. `var.max(0.0)` is `f64::max`, which returns the NON-NaN operand: a NaN
 *     variance becomes 0.0, not NaN. `fmax` matches; `var < 0 ? 0 : var` does
 *     not.
 *
 * THE TWO ROLLING SUMS DO NOT SHARE AN ASSOCIATION. `sum += new - old` but
 * `sum_sq = (sum_sq - old*old) + new*new` (:665) — parenthesised, not
 * `sum_sq += new*new - old*old` the way `stddev_kernel.cu` does it. The two
 * differ in the last place at every bar. Copied literally.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_BB_DEVUP 2.0
#define NEO_BB_DEVDN 2.0

extern "C" __global__
void bollinger_bands_neo_batch_f64(const double* __restrict__ data,
                                   int series_len,
                                   const int* __restrict__ periods,
                                   int n_combos,
                                   int first_valid,
                                   double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int period = periods[combo];

    if (len <= 0 || period <= 0 || period > len ||
        first_valid < 0 || first_valid >= len ||
        (len - first_valid) < period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < len && i < warm; ++i) o[i] = NEO_F64_NAN;
    if (warm >= len) return;

    const double devup = NEO_BB_DEVUP;
    const double devdn = NEO_BB_DEVDN;
    (void)devdn;  // the lower band is not this matrix; kept so the constant
                  // pair reads as the CPU's pair rather than half of it.

    const double inv_n = 1.0 / (double)period;

    double sum = 0.0;
    double sum_sq = 0.0;
    for (int k = 0; k < period; ++k) {
        const double v = data[first_valid + k];
        sum += v;
        sum_sq += v * v;
    }

    const double mean = sum * inv_n;
    // `use_sample` is a hard `false` at bollinger_bands.rs:627, so the sample
    // branch is dead on this path and is not written.
    const double var0 = (sum_sq * inv_n) - (mean * mean);
    const double std0 = sqrt(fmax(var0, 0.0));

    o[warm] = mean + devup * std0;

    for (int i = warm + 1; i < len; ++i) {
        const double nv = data[i];
        const double ov = data[i - period];

        sum += nv - ov;
        sum_sq = (sum_sq - ov * ov) + nv * nv;   // :665, exactly this shape

        const double m = sum * inv_n;
        const double var = (sum_sq * inv_n) - (m * m);
        const double sd = sqrt(fmax(var, 0.0));

        o[i] = m + devup * sd;
    }
}
