#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }


struct twof { float hi, lo; };


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}

__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}

__device__ __forceinline__ twof make_twof(float hi, float lo) { return {hi, lo}; }

__device__ __forceinline__ twof twof_add(twof x, twof y) {
    float s, e; two_sum(x.hi, y.hi, s, e);
    float t = x.lo + y.lo;
    float sh, sl; quick_two_sum(s, e + t, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_sub(twof x, twof y) {
    float s, e; two_sum(x.hi, -y.hi, s, e);
    float t = x.lo - y.lo;
    float sh, sl; quick_two_sum(s, e + t, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_scale(twof x, float k) {
    float p, e; two_prod(x.hi, k, p, e);
    e = fmaf(x.lo, k, e);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_sqr(twof x) {

    float p, e; two_prod(x.hi, x.hi, p, e);
    e = fmaf(2.0f * x.hi, x.lo, e) + (x.lo * x.lo);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ float twof_to_f(twof x) { return x.hi + x.lo; }


__device__ __forceinline__ twof ld_twof(const float2* __restrict__ a, int idx) {
    float2 v = a[idx];
    return make_twof(v.x, v.y);
}

extern "C" __global__ void deviation_build_prefix_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    float2* __restrict__ prefix_sum,
    float2* __restrict__ prefix_sum_sq,
    int* __restrict__ prefix_nan)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    twof sum = make_twof(0.0f, 0.0f);
    twof sum_sq = make_twof(0.0f, 0.0f);
    int nan_count = 0;

    prefix_sum[0] = make_float2(0.0f, 0.0f);
    prefix_sum_sq[0] = make_float2(0.0f, 0.0f);
    prefix_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float v = data[i];
            if (isnan(v)) {
                nan_count += 1;
            } else {
                const twof x = make_twof(v, 0.0f);
                sum = twof_add(sum, x);
                sum_sq = twof_add(sum_sq, twof_sqr(x));
            }
        }
        prefix_sum[i + 1] = make_float2(sum.hi, sum.lo);
        prefix_sum_sq[i + 1] = make_float2(sum_sq.hi, sum_sq.lo);
        prefix_nan[i + 1] = nan_count;
    }
}


extern "C" __global__ void deviation_batch_f32(
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const size_t row_off = static_cast<size_t>(combo) * static_cast<size_t>(len);
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);
    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = dev_nan();
        if (t >= warm) {
            const int start = t + 1 - period;
            bool ok = true;
            if (any_nan_since_first) {
                ok = (prefix_nan[t + 1] - prefix_nan[start]) == 0;
            }
            if (ok) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    const float2 ps_e  = prefix_sum[t + 1];
                    const float2 ps_s  = prefix_sum[start];
                    const float2 ps2_e = prefix_sum_sq[t + 1];
                    const float2 ps2_s = prefix_sum_sq[start];

                    const float sum  = (ps_e.x  - ps_s.x)  + (ps_e.y  - ps_s.y);
                    const float sum2 = (ps2_e.x - ps2_s.x) + (ps2_e.y - ps2_s.y);

                    const float mean = sum * inv_den;
                    const float ex2  = sum2 * inv_den;
                    float var_f = fmaf(-mean, mean, ex2);
                    if (var_f < 0.0f) var_f = 0.0f;
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void deviation_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float*        __restrict__ out_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + period - 1;
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series;
        float out_val = dev_nan();
        if (t >= warm) {
            const int wr = idx + 1;
            const int wl = wr - period * num_series;
            const int bad = prefix_nan_tm[wr] - prefix_nan_tm[wl];
            if (bad == 0) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    twof s1  = twof_sub(ld_twof(prefix_sum_tm,    wr),
                                         ld_twof(prefix_sum_tm,    wl));
                    twof s2  = twof_sub(ld_twof(prefix_sum_sq_tm, wr),
                                         ld_twof(prefix_sum_sq_tm, wl));

                    twof mean  = twof_scale(s1, inv_den);
                    twof mean2 = twof_scale(s2, inv_den);
                    twof var_ds = twof_sub(mean2, twof_sqr(mean));

                    float var_f = twof_to_f(var_ds);
                    if (var_f < 0.0f) var_f = 0.0f;
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}

// ===========================================================================
// S3 f64 LANE — deviation
// ===========================================================================
// Reference: src/indicators/deviation.rs
//   `deviation_prepare` (:256)  — first_valid + the four Err branches
//   `deviation_with_kernel` (:354) — `alloc_with_nan_prefix(len, first+period-1)`
//   `deviation_compute_into` (:296) — devtype 0 is the batch default
//     (`cpu_batch.rs:3628` reads `devtype` with default 0), which routes to
//   `standard_deviation_rolling_into` (:1055), which itself delegates to
//   `standard_deviation_rolling_finite_into` (:1211) WHENEVER every value from
//   `first` on is finite. Both paths are transcribed below and selected by the
//   SAME test, because they are not the same arithmetic: the finite path
//   rebuilds `sum`/`sumsq` from scratch only when the running pair goes
//   non-finite, whereas the general path carries a `bad` counter and emits NaN
//   while it is non-zero.
//
// WHAT THE f32 KERNELS ABOVE GET WRONG
//   1. f32 throughout, including `sqrtf` and a `__int_as_float` NaN.
//   2. The catastrophic-cancellation guard. The CPU re-derives the variance
//      from `(x-mean)^2` when `|var| / max(|sumsq/n|, 1e-30) < 1e-10`. Those
//      two constants are sized for DOUBLE — 1e-30 is far below f32's smallest
//      normal (1.18e-38 is close, but 1e-10 is *above* f32 epsilon 1.19e-7),
//      so in f32 the guard fires on essentially every bar or never, depending
//      on scale. They are taken verbatim from the f64 CPU here, which is the
//      only place they are meaningful.
//   3. `sumsq = v.mul_add(v, sumsq)` is ONE rounding on the CPU. Written as
//      `sumsq += v*v` it is two. `fma` below reproduces the CPU exactly.
//
// One thread per column: the rolling sums are a cross-bar recurrence.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// `var.abs() / scale.max(1e-30) < 1e-10` → recompute from (x-mean)^2.
// Constants lifted from deviation.rs:1237 / :1110 unchanged, because the CPU
// they came from is already f64 and IS the oracle.
__device__ __forceinline__ double neo_s3_dev_var(
    const double* __restrict__ d, int start, int end, double sum, double sumsq, double n)
{
    const double mean = sum / n;
    double var = (sumsq / n) - mean * mean;
    const double scale = fabs(sumsq / n);
    if (fabs(var) / fmax(scale, 1e-30) < 1e-10) {
        double v2 = 0.0;
        for (int k = start; k <= end; ++k) {
            const double dd = d[k] - mean;
            v2 = fma(dd, dd, v2);
        }
        var = v2 / n;
    }
    if (var < 0.0) var = 0.0;
    return var;
}

extern "C" __global__ void neoethos_deviation_batch_f64(
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

    // Every `deviation_prepare` branch that returns Err → the CPU produces no
    // series at all, which this lane represents as an all-NaN row.
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();
    if (warm >= n) return;

    // `standard_deviation_rolling_into` :1061 — period == 1 writes 0.0 from
    // `first`, not from `warm`. With period == 1 the two coincide.
    if (period == 1) {
        for (int i = first_valid; i < n; ++i) row[i] = 0.0;
        return;
    }

    const double nd = (double)period;

    // `data[first..].iter().all(|x| x.is_finite())` — deviation.rs:1079.
    bool all_finite = true;
    for (int i = first_valid; i < n; ++i) {
        if (!isfinite(data[i])) { all_finite = false; break; }
    }

    if (all_finite) {
        // ---- standard_deviation_rolling_finite_into (:1211) ----
        double sum = 0.0, sumsq = 0.0;
        for (int j = first_valid; j < first_valid + period; ++j) {
            const double v = data[j];
            sum += v;
            sumsq = fma(v, v, sumsq);
        }
        if (!isfinite(sum) || !isfinite(sumsq)) {
            row[warm] = neo_s3_qnan();
        } else {
            row[warm] = sqrt(neo_s3_dev_var(data, first_valid, warm, sum, sumsq, nd));
        }

        for (int i = warm + 1; i < n; ++i) {
            const double v_in = data[i];
            const double v_out = data[i - period];
            sum += v_in;
            sumsq = fma(v_in, v_in, sumsq);
            sum -= v_out;
            sumsq -= v_out * v_out;

            const int start = i + 1 - period;
            if (!isfinite(sum) || !isfinite(sumsq)) {
                sum = 0.0;
                sumsq = 0.0;
                for (int k = start; k <= i; ++k) {
                    const double x = data[k];
                    sum += x;
                    sumsq = fma(x, x, sumsq);
                }
                if (!isfinite(sum) || !isfinite(sumsq)) {
                    row[i] = neo_s3_qnan();
                    continue;
                }
            }
            row[i] = sqrt(neo_s3_dev_var(data, start, i, sum, sumsq, nd));
        }
        return;
    }

    // ---- standard_deviation_rolling_into, the `bad`-counter path (:1083) ----
    double sum = 0.0, sumsq = 0.0;
    int bad = 0;
    for (int j = first_valid; j < first_valid + period; ++j) {
        const double v = data[j];
        if (!isfinite(v)) { bad += 1; }
        else { sum += v; sumsq = fma(v, v, sumsq); }
    }

    if (bad > 0 || !isfinite(sum) || !isfinite(sumsq)) {
        row[warm] = neo_s3_qnan();
    } else {
        row[warm] = sqrt(neo_s3_dev_var(data, warm + 1 - period, warm, sum, sumsq, nd));
    }

    for (int i = warm + 1; i < n; ++i) {
        const double v_in = data[i];
        const double v_out = data[i - period];
        if (!isfinite(v_in)) { bad += 1; }
        else { sum += v_in; sumsq = fma(v_in, v_in, sumsq); }
        if (!isfinite(v_out)) { bad = (bad > 0) ? bad - 1 : 0; }
        else { sum -= v_out; sumsq -= v_out * v_out; }

        if (bad > 0 || !isfinite(sum) || !isfinite(sumsq)) {
            if (bad == 0) {
                const int start = i + 1 - period;
                double s = 0.0, s2 = 0.0;
                for (int k = start; k <= i; ++k) {
                    const double v = data[k];
                    s += v;
                    s2 = fma(v, v, s2);
                }
                if (isfinite(s) && isfinite(s2)) {
                    row[i] = sqrt(neo_s3_dev_var(data, start, i, s, s2, nd));
                } else {
                    row[i] = neo_s3_qnan();
                }
            } else {
                row[i] = neo_s3_qnan();
            }
        } else {
            row[i] = sqrt(neo_s3_dev_var(data, i + 1 - period, i, sum, sumsq, nd));
        }
    }
}
