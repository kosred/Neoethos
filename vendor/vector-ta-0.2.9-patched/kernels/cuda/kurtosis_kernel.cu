#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


struct __align__(8) dsf { float hi, lo; };

__device__ __forceinline__ dsf ds_from_float(float a) {
    return {a, 0.0f};
}


__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


__device__ __forceinline__ dsf ds_add(dsf a, dsf b) {
    float s, e;
    two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return {hi, lo};
}

__device__ __forceinline__ dsf ds_neg(dsf a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ dsf ds_sub(dsf a, dsf b) { return ds_add(a, ds_neg(b)); }


__device__ __forceinline__ dsf ds_mul(dsf a, dsf b) {
    float p  = a.hi * b.hi;
    float e  = fmaf(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    e += a.lo * b.lo;
    float hi = p + e;
    float lo = e - (hi - p);
    return {hi, lo};
}

__device__ __forceinline__ dsf ds_scale(dsf a, float s) {
    float p  = a.hi * s;
    float e  = fmaf(a.hi, s, -p);
    e += a.lo * s;
    float hi = p + e;
    float lo = e - (hi - p);
    return {hi, lo};
}

__device__ __forceinline__ dsf ds_square(dsf a) { return ds_mul(a, a); }
__device__ __forceinline__ float ds_to_f32(dsf a) { return a.hi + a.lo; }


__device__ __forceinline__ dsf ld_ds(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return {v.x, v.y};
}


__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fffffff); }


extern "C" __global__ void kurtosis_build_prefix_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    float2* __restrict__ ps_x,
    float2* __restrict__ ps_x2,
    float2* __restrict__ ps_x3,
    float2* __restrict__ ps_x4,
    int* __restrict__ ps_nan
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    dsf s1 = ds_from_float(0.0f);
    dsf s2 = ds_from_float(0.0f);
    dsf s3 = ds_from_float(0.0f);
    dsf s4 = ds_from_float(0.0f);
    int nan_count = 0;

    ps_x[0] = make_float2(0.0f, 0.0f);
    ps_x2[0] = make_float2(0.0f, 0.0f);
    ps_x3[0] = make_float2(0.0f, 0.0f);
    ps_x4[0] = make_float2(0.0f, 0.0f);
    ps_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float v = data[i];
            if (isnan(v)) {
                nan_count += 1;
            } else {
                const float d2 = fmaf(v, v, 0.0f);
                s1 = ds_add(s1, ds_from_float(v));
                s2 = ds_add(s2, ds_from_float(d2));
                s3 = ds_add(s3, ds_from_float(d2 * v));
                s4 = ds_add(s4, ds_from_float(d2 * d2));
            }
        }

        ps_x[i + 1] = make_float2(s1.hi, s1.lo);
        ps_x2[i + 1] = make_float2(s2.hi, s2.lo);
        ps_x3[i + 1] = make_float2(s3.hi, s3.lo);
        ps_x4[i + 1] = make_float2(s4.hi, s4.lo);
        ps_nan[i + 1] = nan_count;
    }
}

extern "C" __global__ void kurtosis_batch_f32(
    const float2* __restrict__ ps_x,
    const float2* __restrict__ ps_x2,
    const float2* __restrict__ ps_x3,
    const float2* __restrict__ ps_x4,
    const int*    __restrict__ ps_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float inv_n = 1.0f / (float)period;

    while (t < len) {
        float out_val = qnan_f32();
        if (t >= warm) {
            const int end   = t + 1;
            int start = end - period;
            if (start < 0) start = 0;

            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {

                const float2 px_e  = ps_x[end];
                const float2 px_s  = ps_x[start];
                const float2 px2_e = ps_x2[end];
                const float2 px2_s = ps_x2[start];
                const float2 px3_e = ps_x3[end];
                const float2 px3_s = ps_x3[start];
                const float2 px4_e = ps_x4[end];
                const float2 px4_s = ps_x4[start];

                const float sum1 = (px_e.x  - px_s.x)  + (px_e.y  - px_s.y);
                const float sum2 = (px2_e.x - px2_s.x) + (px2_e.y - px2_s.y);
                const float sum3 = (px3_e.x - px3_s.x) + (px3_e.y - px3_s.y);
                const float sum4 = (px4_e.x - px4_s.x) + (px4_e.y - px4_s.y);

                const float mean = sum1 * inv_n;
                const float Ex2  = sum2 * inv_n;
                const float Ex3  = sum3 * inv_n;
                const float Ex4  = sum4 * inv_n;

                const float mean2 = mean * mean;
                const float m2 = Ex2 - mean2;

                if (m2 > 0.0f) {

                    const float term1 = fmaf(-4.0f * mean, Ex3, Ex4);
                    const float term2 = fmaf(6.0f * mean2, Ex2, term1);
                    const float mean4 = mean2 * mean2;
                    const float m4 = fmaf(-3.0f, mean4, term2);

                    const float denom = m2 * m2;
                    if (denom > 0.0f && !isnan(denom)) {
                        out_val = (m4 / denom) - 3.0f;
                    }
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void kurtosis_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    const int*   __restrict__ first_valids,
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;
    const int stride = num_series;


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = qnan_f32();
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int warm = first_valid + period - 1;
    const float inv_n = 1.0f / (float)period;


    dsf s1 = ds_from_float(0.0f), s2 = ds_from_float(0.0f),
        s3 = ds_from_float(0.0f), s4 = ds_from_float(0.0f);
    int nan_in_win = 0;

    const int init_end = (warm + 1 < series_len) ? (warm + 1) : series_len;
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else {
            const float d  = v;
            const float d2 = fmaf(d, d, 0.0f);
            const float d3 = d2 * d;
            const float d4 = d2 * d2;
            s1 = ds_add(s1, ds_from_float(d));
            s2 = ds_add(s2, ds_from_float(d2));
            s3 = ds_add(s3, ds_from_float(d3));
            s4 = ds_add(s4, ds_from_float(d4));
        }
    }

    if (warm < series_len && nan_in_win == 0) {
        const dsf mean = ds_scale(s1, inv_n);
        const dsf Ex2  = ds_scale(s2, inv_n);
        const dsf Ex3  = ds_scale(s3, inv_n);
        const dsf Ex4  = ds_scale(s4, inv_n);

        const dsf mean2 = ds_square(mean);
        const dsf m2_ds = ds_sub(Ex2, mean2);
        const float m2  = ds_to_f32(m2_ds);

        float out0 = qnan_f32();
        if (m2 > 0.0f) {
            const dsf term1 = ds_sub(Ex4, ds_scale(ds_mul(mean, Ex3), 4.0f));
            const dsf term2 = ds_add(term1, ds_scale(ds_mul(mean2, Ex2), 6.0f));
            const dsf m4_ds = ds_sub(term2, ds_scale(ds_square(mean2), 3.0f));
            const float m4  = ds_to_f32(m4_ds);
            const float denom = m2 * m2;
            if (denom > 0.0f && !isnan(denom)) {
                out0 = (m4 / denom) - 3.0f;
            }
        }
        out_tm[warm * stride + series] = out0;
    }


    for (int t = warm + 1; t < series_len; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        if (isnan(old_v) || isnan(new_v)) {

            s1 = ds_from_float(0.0f); s2 = ds_from_float(0.0f);
            s3 = ds_from_float(0.0f); s4 = ds_from_float(0.0f);
            nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else {
                    const float d  = vv;
                    const float d2 = fmaf(d, d, 0.0f);
                    const float d3 = d2 * d;
                    const float d4 = d2 * d2;
                    s1 = ds_add(s1, ds_from_float(d));
                    s2 = ds_add(s2, ds_from_float(d2));
                    s3 = ds_add(s3, ds_from_float(d3));
                    s4 = ds_add(s4, ds_from_float(d4));
                }
            }
        } else {

            const float od  = old_v;
            const float nd  = new_v;
            const float od2 = fmaf(od, od, 0.0f);
            const float nd2 = fmaf(nd, nd, 0.0f);

            s1 = ds_add(s1, ds_from_float(nd - od));
            s2 = ds_add(s2, ds_from_float(nd2 - od2));
            s3 = ds_add(s3, ds_from_float(nd2 * nd - od2 * od));
            s4 = ds_add(s4, ds_from_float(nd2 * nd2 - od2 * od2));
        }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = qnan_f32();
        } else {
            const dsf mean = ds_scale(s1, inv_n);
            const dsf Ex2  = ds_scale(s2, inv_n);
            const dsf Ex3  = ds_scale(s3, inv_n);
            const dsf Ex4  = ds_scale(s4, inv_n);
            const dsf mean2 = ds_square(mean);
            const dsf m2_ds = ds_sub(Ex2, mean2);
            const float m2  = ds_to_f32(m2_ds);

            float outv = qnan_f32();
            if (m2 > 0.0f) {
                const dsf term1 = ds_sub(Ex4, ds_scale(ds_mul(mean, Ex3), 4.0f));
                const dsf term2 = ds_add(term1, ds_scale(ds_mul(mean2, Ex2), 6.0f));
                const dsf m4_ds = ds_sub(term2, ds_scale(ds_square(mean2), 3.0f));
                const float m4  = ds_to_f32(m4_ds);
                const float denom = m2 * m2;
                if (denom > 0.0f && !isnan(denom)) {
                    outv = (m4 / denom) - 3.0f;
                }
            }
            out_tm[t * stride + series] = outv;
        }
    }
}


// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/kurtosis.rs
//   * `kurtosis_scalar`            (:372) — the general window path
//   * `kurtosis_period5_value`     (:464) — the period == 5 CLOSED FORM, which
//     multiplies by 0.2 instead of dividing by 5.0. `x * 0.2` and `x / 5.0` are
//     NOT the same double (0.2 is not representable), so the branch is
//     reproduced rather than folded into the general path.
//   * warmup prefix: `alloc_with_nan_prefix(len, first + period - 1)` (:238).
//
// Windowed, not recursive: every bar reads its own [i+1-period, i] window and
// nothing carries across bars, so this is parallel over (combo, bar).
//
// The epsilon is `f64::EPSILON` = 2^-52 = 2.220446049250313e-16 — DERIVED for
// f64, not the f32 1.1920929e-7 that the legacy lane would have used.
// =============================================================================

__device__ __forceinline__ double nef_qnan_kurt() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__
void neoethos_kurtosis_f64(const double* __restrict__ prices,
                           int n,
                           const int* __restrict__ periods,
                           int n_combos,
                           int first_valid,
                           double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int bar = blockIdx.x * blockDim.x + threadIdx.x;
    if (bar >= n || n <= 0) return;

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double QNAN = nef_qnan_kurt();

    const int period = periods[combo];
    if (period <= 0 || first_valid < 0 || first_valid >= n) { row[bar] = QNAN; return; }

    // warm = first + period - 1; everything before it is the CPU's NaN prefix.
    const int first_out = first_valid + period - 1;
    if (bar < first_out) { row[bar] = QNAN; return; }

    const int start = bar + 1 - period;

    // f64::EPSILON, exactly. NOT the f32 epsilon.
    const double EPS_F64 = 2.220446049250313e-16;

    if (period == 5) {
        // kurtosis_period5_value — association order preserved literally.
        const double a = prices[start];
        const double b = prices[start + 1];
        const double c = prices[start + 2];
        const double d = prices[start + 3];
        const double e = prices[start + 4];
        if (isnan(a) || isnan(b) || isnan(c) || isnan(d) || isnan(e)) { row[bar] = QNAN; return; }

        const double mean = ((((a + b) + c) + d) + e) * 0.2;
        const double da = a - mean;
        const double db = b - mean;
        const double dc = c - mean;
        const double dd = d - mean;
        const double de = e - mean;

        const double da2 = da * da;
        const double db2 = db * db;
        const double dc2 = dc * dc;
        const double dd2 = dd * dd;
        const double de2 = de * de;

        const double m2 = ((((da2 + db2) + dc2) + dd2) + de2) * 0.2;
        if (fabs(m2) < EPS_F64) { row[bar] = QNAN; return; }
        const double m4 =
            (((((da2 * da2) + (db2 * db2)) + (dc2 * dc2)) + (dd2 * dd2)) + (de2 * de2)) * 0.2;
        row[bar] = (m4 / (m2 * m2)) - 3.0;
        return;
    }

    // General path: forward sum, then a second forward pass for m2/m4.
    double sum = 0.0;
    for (int k = 0; k < period; ++k) {
        const double v = prices[start + k];
        if (isnan(v)) { row[bar] = QNAN; return; }
        sum += v;
    }
    const double nn = (double)period;
    const double mean = sum / nn;
    double m2 = 0.0;
    double m4 = 0.0;
    for (int k = 0; k < period; ++k) {
        const double diff = prices[start + k] - mean;
        const double d2 = diff * diff;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= nn;
    m4 /= nn;

    if (fabs(m2) < EPS_F64) { row[bar] = QNAN; return; }
    row[bar] = (m4 / (m2 * m2)) - 3.0;
}


// ===========================================================================
// S1 f64 LANE  --  kurtosis
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/kurtosis.rs -- `kurtosis_scalar` (:372), `kurtosis_period5_value` (:463), `kurtosis_with_kernel` (:207)
//
// SOURCE SERIES IS hl2, NOT close. `compute_kurtosis_batch`
// (cpu_batch.rs:3522) calls `extract_slice_input("kurtosis", req.data, "hl2")`.
// Feeding this kernel close computes a different indicator and passes every
// shape check on the way, which is why the lane declares `Hl2Slice` for it --
// the same reason `cci`/`mfi` declare hlc3.
//
// PERIOD-BASED: `period` (default 5) is the swept parameter.
//
// TWO CPU PATHS, BOTH REPRODUCED, BECAUSE THEY ARE NOT THE SAME ARITHMETIC.
// `kurtosis_scalar` branches at period == 5 to a closed form whose mean is
// `sum * 0.2` where the general path computes `sum / n`. 0.2 is NOT
// representable in binary, so `sum * 0.2` and `sum / 5.0` differ in the last
// place for most inputs -- this is a genuine fork, not a fast path that
// happens to agree, and the default period IS 5, so it is the branch this lane
// will actually take. Both are written out below.
// The `_clean` variant (kurtosis.rs:421) differs from `kurtosis_scalar_period5`
// (:440) only by skipping the per-bar NaN guard; with a NaN input the guarded
// path stores NaN explicitly and the unguarded one propagates NaN through the
// arithmetic to the same NaN, so one body serves both.
//
// EPSILON: the CPU compares `m2.abs() < f64::EPSILON`. That is ALREADY an
// f64-sized constant (2^-52 = 2.220446049250313e-16) and is carried over
// unchanged -- unlike an f32 epsilon, which would have had to be re-derived.
// It is written as the literal rather than as `DBL_EPSILON` so the value is
// visible at the comparison.
//
// SUMMATION ORDER: the general path accumulates `sum += v` over the window
// ascending, then a second ascending pass for m2/m4. The period-5 path is an
// explicit left-nested tree `((((a+b)+c)+d)+e)`, which is the SAME association
// as an ascending loop -- but it is written out literally rather than looped,
// because "the same" is an argument and this is a fixed five-term expression.
//
// WARMUP: `alloc_with_nan_prefix(len, first + period - 1)`.
// ===========================================================================

#ifndef NEO_S1_QNAN_DEFINED
#define NEO_S1_QNAN_DEFINED
// The f32 kernels in this crate spell NaN `__int_as_float(0x7fc00000)`. That is
// a 32-bit pattern; widening it is a value change, not a cast. This is the f64
// quiet-NaN pattern, stated once per translation unit.
__device__ __forceinline__ double neo_s1_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}
__device__ __forceinline__ bool neo_s1_isnan(double x) { return x != x; }
#endif

__device__ __forceinline__ double neo_s1_kurtosis_p5(double a, double b, double c,
                                                     double d, double e) {
    const double mean = ((((a + b) + c) + d) + e) * 0.2;
    const double da = a - mean;
    const double db = b - mean;
    const double dc = c - mean;
    const double dd = d - mean;
    const double de = e - mean;

    const double da2 = da * da;
    const double db2 = db * db;
    const double dc2 = dc * dc;
    const double dd2 = dd * dd;
    const double de2 = de * de;

    const double m2 = ((((da2 + db2) + dc2) + dd2) + de2) * 0.2;
    // f64::EPSILON, spelled out.
    if (fabs(m2) < 2.220446049250313e-16) return neo_s1_qnan();
    const double m4 =
        (((((da2 * da2) + (db2 * db2)) + (dc2 * dc2)) + (dd2 * dd2)) + (de2 * de2)) * 0.2;
    return (m4 / (m2 * m2)) - 3.0;
}

extern "C" __global__ void neoethos_kurtosis_batch_f64(
    const double* __restrict__ hl2,
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
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();
    if (warm >= n) return;

    if (period == 5) {
        for (int i = first_valid + 4; i < n; ++i) {
            const double a = hl2[i - 4];
            const double b = hl2[i - 3];
            const double c = hl2[i - 2];
            const double d = hl2[i - 1];
            const double e = hl2[i];
            if (neo_s1_isnan(a) || neo_s1_isnan(b) || neo_s1_isnan(c) ||
                neo_s1_isnan(d) || neo_s1_isnan(e)) {
                row[i] = neo_s1_qnan();
                continue;
            }
            row[i] = neo_s1_kurtosis_p5(a, b, c, d, e);
        }
        return;
    }

    const double nf = (double)period;
    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - period;

        bool has_nan = false;
        double sum = 0.0;
        for (int k = 0; k < period; ++k) {
            const double v = hl2[start + k];
            if (neo_s1_isnan(v)) { has_nan = true; break; }
            sum += v;
        }
        if (has_nan) { row[i] = neo_s1_qnan(); continue; }

        const double mean = sum / nf;
        double m2 = 0.0;
        double m4 = 0.0;
        for (int k = 0; k < period; ++k) {
            const double diff = hl2[start + k] - mean;
            const double d2 = diff * diff;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= nf;
        m4 /= nf;

        if (fabs(m2) < 2.220446049250313e-16) {
            row[i] = neo_s1_qnan();
        } else {
            row[i] = (m4 / (m2 * m2)) - 3.0;
        }
    }
}
