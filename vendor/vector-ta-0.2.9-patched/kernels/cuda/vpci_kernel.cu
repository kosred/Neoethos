#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include "ds_float2.cuh"

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ dsf load_dsf_f2(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}


__device__ __forceinline__ dsf ds_div(dsf num, dsf den) {
    if (den.hi == 0.0f && den.lo == 0.0f) return ds_make(nan_f32(), 0.0f);
    float q1 = num.hi / den.hi;
    dsf t = ds_scale(den, q1);
    dsf r = ds_sub(num, t);
    float q2 = r.hi / den.hi;

    float s = q1 + q2;
    float e = q2 - (s - q1);
    return ds_norm(s, e);
}


__device__ __forceinline__ void kahan_add(float x, float& sum, float& c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}


__device__ __forceinline__ float warp_bcast_f32_first(float v_any) {
    unsigned mask = __activemask();
    int first = __ffs(mask) - 1;
    return __shfl_sync(mask, v_any, first);
}
__device__ __forceinline__ dsf warp_bcast_dsf_first(dsf v_any) {
    unsigned mask = __activemask();
    int first = __ffs(mask) - 1;
    float hi = __shfl_sync(mask, v_any.hi, first);
    float lo = __shfl_sync(mask, v_any.lo, first);
    return ds_make(hi, lo);
}

extern "C" __global__ void vpci_build_prefix_single_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int series_len,
    int first_valid,
    float2* __restrict__ pfx_c,
    float2* __restrict__ pfx_v,
    float2* __restrict__ pfx_cv
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    for (int i = 0; i < first_valid; ++i) {
        pfx_c[i] = make_float2(0.0f, 0.0f);
        pfx_v[i] = make_float2(0.0f, 0.0f);
        pfx_cv[i] = make_float2(0.0f, 0.0f);
    }

    dsf sc = ds_make(0.0f, 0.0f);
    dsf sv = ds_make(0.0f, 0.0f);
    dsf scv = ds_make(0.0f, 0.0f);
    for (int i = first_valid; i < series_len; ++i) {
        const float c = isfinite(close[i]) ? close[i] : 0.0f;
        const float v = isfinite(volume[i]) ? volume[i] : 0.0f;
        sc = ds_add(sc, ds_set(c));
        sv = ds_add(sv, ds_set(v));
        scv = ds_add(scv, ds_set(c * v));
        pfx_c[i] = make_float2(sc.hi, sc.lo);
        pfx_v[i] = make_float2(sv.hi, sv.lo);
        pfx_cv[i] = make_float2(scv.hi, scv.lo);
    }
}

extern "C" __global__ void vpci_batch_f32(
    const float2* __restrict__ pfx_c,
    const float2* __restrict__ pfx_v,
    const float2* __restrict__ pfx_cv,
    const float*  __restrict__ volume,
    const int*    __restrict__ shorts,
    const int*    __restrict__ longs,
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_vpci,
    float* __restrict__ out_vpcis
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int short_p = shorts[row];
    const int long_p  = longs[row];
    const int base    = row * series_len;
    float* __restrict__ y_vpci  = out_vpci  + base;
    float* __restrict__ y_vpcis = out_vpcis + base;

    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p > long_p ||
                 long_p > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < long_p)) {
        for (int i = 0; i < series_len; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }
        return;
    }

    const int warm = first_valid + long_p - 1;


    for (int i = 0; i < warm; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }

    const float inv_long  = 1.0f / (float)long_p;
    const float inv_short = 1.0f / (float)short_p;

    float sum_vpci_vol_short = 0.0f;
    float sum_comp           = 0.0f;

    for (int i = warm; i < series_len; ++i) {
        const int idx_long_prev  = i - long_p;
        const int idx_short_prev = i - short_p;


        dsf c_cur  = load_dsf_f2(pfx_c,  i);
        dsf v_cur  = load_dsf_f2(pfx_v,  i);
        dsf cv_cur = load_dsf_f2(pfx_cv, i);
        float vol_i = volume[i];


        const dsf zero = ds_make(0.0f, 0.0f);
        const dsf c_prev_l  = (idx_long_prev < first_valid) ? zero : load_dsf_f2(pfx_c,  idx_long_prev);
        const dsf v_prev_l  = (idx_long_prev < first_valid) ? zero : load_dsf_f2(pfx_v,  idx_long_prev);
        const dsf cv_prev_l = (idx_long_prev < first_valid) ? zero : load_dsf_f2(pfx_cv, idx_long_prev);
        const dsf c_prev_s  = (idx_short_prev < first_valid) ? zero : load_dsf_f2(pfx_c,  idx_short_prev);
        const dsf v_prev_s  = (idx_short_prev < first_valid) ? zero : load_dsf_f2(pfx_v,  idx_short_prev);
        const dsf cv_prev_s = (idx_short_prev < first_valid) ? zero : load_dsf_f2(pfx_cv, idx_short_prev);


        const dsf sc_l  = ds_sub(c_cur,  c_prev_l);
        const dsf sv_l  = ds_sub(v_cur,  v_prev_l);
        const dsf scv_l = ds_sub(cv_cur, cv_prev_l);
        const dsf sc_s  = ds_sub(c_cur,  c_prev_s);
        const dsf sv_s  = ds_sub(v_cur,  v_prev_s);
        const dsf scv_s = ds_sub(cv_cur, cv_prev_s);


        const dsf sma_l   = ds_scale(sc_l,  inv_long);
        const dsf sma_s   = ds_scale(sc_s,  inv_short);
        const dsf sma_v_l = ds_scale(sv_l,  inv_long);
        const dsf sma_v_s = ds_scale(sv_s,  inv_short);


        const dsf vwma_l = ds_div(scv_l, sv_l);
        const dsf vwma_s = ds_div(scv_s, sv_s);

        const dsf vpc_ds = ds_sub(vwma_l, sma_l);
        const dsf vpr_ds = ds_div(vwma_s, sma_s);
        const dsf vm_ds  = ds_div(sma_v_s, sma_v_l);

        const float vpc = ds_to_f(vpc_ds);
        const float vpr = ds_to_f(vpr_ds);
        const float vm  = ds_to_f(vm_ds);

        const float vpci = vpc * vpr * vm;

        y_vpci[i] = vpci;


        const float contrib = isfinite(vpci) ? (vpci * vol_i) : 0.0f;
        kahan_add(contrib, sum_vpci_vol_short, sum_comp);
        if (i >= warm + short_p) {
            const int rm = i - short_p;
            const float vpci_rm = y_vpci[rm];
            const float rm_contrib = isfinite(vpci_rm) ? (vpci_rm * volume[rm]) : 0.0f;
            kahan_add(-rm_contrib, sum_vpci_vol_short, sum_comp);
        }


        const float denom = ds_to_f(sma_v_s);
        if (denom != 0.0f && isfinite(denom)) {
            y_vpcis[i] = (sum_vpci_vol_short * inv_short) / denom;
        } else {
            y_vpcis[i] = nan_f32();
        }
    }
}


extern "C" __global__ void vpci_many_series_one_param_f32(
    const float2* __restrict__ pfx_c_tm,
    const float2* __restrict__ pfx_v_tm,
    const float2* __restrict__ pfx_cv_tm,
    const float*  __restrict__ volume_tm,
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_vpci_tm,
    float* __restrict__ out_vpcis_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int first = first_valids[series];
    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p > long_p ||
                 long_p > rows || first < 0 || first >= rows)) {
        for (int r = 0; r < rows; ++r) {
            const int idx = r * cols + series;
            out_vpci_tm[idx]  = nan_f32();
            out_vpcis_tm[idx] = nan_f32();
        }
        return;
    }

    const int warm = first + long_p - 1;
    for (int r = 0; r < warm; ++r) {
        const int idx = r * cols + series;
        out_vpci_tm[idx]  = nan_f32();
        out_vpcis_tm[idx] = nan_f32();
    }

    const float inv_long  = 1.0f / (float)long_p;
    const float inv_short = 1.0f / (float)short_p;

    float sum_vpci_vol_short = 0.0f;
    float sum_comp           = 0.0f;

    for (int r = warm; r < rows; ++r) {
        const int idx          = r * cols + series;
        const int idx_long_pr  = (r - long_p) * cols + series;
        const int idx_short_pr = (r - short_p) * cols + series;

        const dsf c_cur  = load_dsf_f2(pfx_c_tm,  idx);
        const dsf v_cur  = load_dsf_f2(pfx_v_tm,  idx);
        const dsf cv_cur = load_dsf_f2(pfx_cv_tm, idx);

        const dsf zero = ds_make(0.0f, 0.0f);
        const dsf sc_l  = ds_sub(c_cur,  (idx_long_pr < first * cols + series) ? zero : load_dsf_f2(pfx_c_tm,  idx_long_pr));
        const dsf sv_l  = ds_sub(v_cur,  (idx_long_pr < first * cols + series) ? zero : load_dsf_f2(pfx_v_tm,  idx_long_pr));
        const dsf scv_l = ds_sub(cv_cur, (idx_long_pr < first * cols + series) ? zero : load_dsf_f2(pfx_cv_tm, idx_long_pr));
        const dsf sc_s  = ds_sub(c_cur,  (idx_short_pr < first * cols + series) ? zero : load_dsf_f2(pfx_c_tm,  idx_short_pr));
        const dsf sv_s  = ds_sub(v_cur,  (idx_short_pr < first * cols + series) ? zero : load_dsf_f2(pfx_v_tm,  idx_short_pr));
        const dsf scv_s = ds_sub(cv_cur, (idx_short_pr < first * cols + series) ? zero : load_dsf_f2(pfx_cv_tm, idx_short_pr));

        const dsf sma_l   = ds_scale(sc_l,  inv_long);
        const dsf sma_s   = ds_scale(sc_s,  inv_short);
        const dsf sma_v_l = ds_scale(sv_l,  inv_long);
        const dsf sma_v_s = ds_scale(sv_s,  inv_short);

        const dsf vwma_l = ds_div(scv_l, sv_l);
        const dsf vwma_s = ds_div(scv_s, sv_s);

        const dsf vpc_ds = ds_sub(vwma_l, sma_l);
        const dsf vpr_ds = ds_div(vwma_s, sma_s);
        const dsf vm_ds  = ds_div(sma_v_s, sma_v_l);

        const float vpci = ds_to_f(vpc_ds) * ds_to_f(vpr_ds) * ds_to_f(vm_ds);
        out_vpci_tm[idx] = vpci;

        float contrib = isfinite(vpci) ? (vpci * volume_tm[idx]) : 0.0f;
        kahan_add(contrib, sum_vpci_vol_short, sum_comp);

        if (r >= warm + short_p) {
            const int rm = (r - short_p) * cols + series;
            const float vpci_rm = out_vpci_tm[rm];
            const float rm_contrib = isfinite(vpci_rm) ? (vpci_rm * volume_tm[rm]) : 0.0f;
            kahan_add(-rm_contrib, sum_vpci_vol_short, sum_comp);
        }

        const float denom = ds_to_f(sma_v_s);
        out_vpcis_tm[idx] = (denom != 0.0f && isfinite(denom))
                          ? (sum_vpci_vol_short * inv_short) / denom
                          : nan_f32();
    }
}

/* ===========================================================================
 * S4 f64 LANE — vpci (volume price confirmation indicator)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/vpci.rs
 *   `first_valid_both`             (:389) — first index where close AND volume
 *                                           are both non-NaN
 *   `build_prefix_sums`            (:408) — three prefix sums from index 0,
 *                                           with non-finite inputs zeroed
 *   `vpci_with_kernel`             (:819) — warmup = first + long - 1
 *   `vpci_scalar_into_from_psums`  (:489) — the window algebra
 *
 * WHICH SERIES THIS EMITS. `compute_vpci_batch` (cpu_batch.rs:5776) maps
 * output_id "value" -> `VpciOutputField::Vpci`. One matrix, so this is the
 * VPCI line. The smoothed `vpcis` line is a separate output and is NOT
 * computed here — its rolling `sum_vpci_vol_short` accumulator feeds nothing
 * that this matrix reads, so carrying it would be dead work, not fidelity.
 *
 * PERIOD-INVARIANT, AND THAT IS FAITHFUL. `compute_vpci_batch` reads
 * `short_range` (5) and `long_range` (25) — cpu_batch.rs:5797-5798 — and never
 * `period`. Identical CPU columns, identical rows here, declared through
 * `is_period_invariant`. Because both ranges are fixed the prefix ring below
 * is a compile-time 26 slots and needs no `max_period`.
 *
 * WHY A PREFIX-SUM RING AND NOT A WINDOW SUM. The CPU does NOT sum the window;
 * it DIFFERENCES two prefix sums that were accumulated from index 0 over the
 * whole series (:536-542). Those are different numbers: the prefix at bar
 * 90 000 of an FX close series is ~1e5 and subtracting two such values loses
 * the low bits that a fresh 25-term window sum would keep. Summing the window
 * directly would be MORE accurate and WRONG — it is not what the reference
 * computes. So this kernel accumulates the same running prefixes from index 0
 * and keeps the last `long + 1` of each in a ring, which reproduces the
 * reference's exact operands with O(long) state instead of O(n).
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. THE PREFIX SUMS ARE THE WHOLE PROBLEM IN f32. `ps_cv` accumulates
 *     close*volume over the entire series; on FX volume that reaches 1e9-1e12,
 *     at which point an f32 accumulator's ulp EXCEEDS a single bar's
 *     contribution and the running sum stops advancing. The window difference
 *     is then 0 or garbage. This is not a precision tolerance, it is a
 *     silently frozen indicator, and it is why the file's `vpci_build_prefix_
 *     single_f32` cannot be rescued by a wider tolerance.
 *  2. `__int_as_float(0x7f...)` -> `__longlong_as_double(0x7ff8...)`.
 *  3. The zero-guards are `!= 0.0` EXACTLY, not `fabs(x) < eps`. An epsilon
 *     here would be an f32-sized constant applied to an f64 quantity — the
 *     hazard the brief names — and it would also change the answer: the CPU
 *     divides by any non-zero denominator however small and emits NaN only on
 *     an exact zero.
 *  4. `zf()` is `is_finite ? x : 0.0`, which zeroes BOTH NaN and infinity.
 *     `isfinite` is the f64 spelling; a `!isnan` test would let an infinity
 *     through and poison the prefix for the rest of the series.
 *
 * ONE THREAD PER COLUMN, walking from index 0 because the prefix sums do.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_VPCI_SHORT 5
#define NEO_VPCI_LONG  25
#define NEO_VPCI_RING  (NEO_VPCI_LONG + 1)

__device__ __forceinline__ double neo_vpci_zf(double x) {
    return isfinite(x) ? x : 0.0;
}

extern "C" __global__
void vpci_neo_batch_f64(const double* __restrict__ close,
                        const double* __restrict__ volume,
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
    (void)periods;   /* period-invariant — see the header. */

    const int shortr = NEO_VPCI_SHORT;
    const int longr  = NEO_VPCI_LONG;

    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        longr > len || shortr > len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warmup = first_valid + longr - 1;
    for (int i = 0; i < len && i < warmup; ++i) o[i] = NEO_F64_NAN;
    if (warmup >= len) return;

    const double inv_long  = 1.0 / (double)longr;
    const double inv_short = 1.0 / (double)shortr;

    /* ps_*[k] for k in [i+1-longr .. i+1], keyed by k % NEO_VPCI_RING. The
     * CPU's ps arrays are length n+1 with ps[0] = 0. */
    double ring_c[NEO_VPCI_RING];
    double ring_v[NEO_VPCI_RING];
    double ring_cv[NEO_VPCI_RING];

    double acc_c = 0.0, acc_v = 0.0, acc_cv = 0.0;
    ring_c[0] = 0.0; ring_v[0] = 0.0; ring_cv[0] = 0.0;

    for (int i = 0; i < len; ++i) {
        const double c_val = neo_vpci_zf(close[i]);
        const double v_val = neo_vpci_zf(volume[i]);
        acc_c  = acc_c  + c_val;
        acc_v  = acc_v  + v_val;
        acc_cv = acc_cv + c_val * v_val;

        const int end = i + 1;
        const int slot = end % NEO_VPCI_RING;
        ring_c[slot]  = acc_c;
        ring_v[slot]  = acc_v;
        ring_cv[slot] = acc_cv;

        if (i < warmup) continue;

        /* `end.saturating_sub(long)` — end >= longr here because
         * i >= warmup >= longr - 1, so end >= longr. */
        const int long_start  = end - longr;
        const int short_start = end - shortr;
        const int ls = long_start  % NEO_VPCI_RING;
        const int ss = short_start % NEO_VPCI_RING;

        const double sc_l  = ring_c[slot]  - ring_c[ls];
        const double sv_l  = ring_v[slot]  - ring_v[ls];
        const double scv_l = ring_cv[slot] - ring_cv[ls];

        const double sc_s  = ring_c[slot]  - ring_c[ss];
        const double sv_s  = ring_v[slot]  - ring_v[ss];
        const double scv_s = ring_cv[slot] - ring_cv[ss];

        const double sma_l   = sc_l * inv_long;
        const double sma_s   = sc_s * inv_short;
        const double sma_v_l = sv_l * inv_long;
        const double sma_v_s = sv_s * inv_short;

        const double vwma_l = (sv_l != 0.0) ? (scv_l / sv_l) : NEO_F64_NAN;
        const double vwma_s = (sv_s != 0.0) ? (scv_s / sv_s) : NEO_F64_NAN;

        const double vpc = vwma_l - sma_l;
        const double vpr = (sma_s   != 0.0) ? (vwma_s  / sma_s)   : NEO_F64_NAN;
        const double vm  = (sma_v_l != 0.0) ? (sma_v_s / sma_v_l) : NEO_F64_NAN;

        /* :564 — `vpc * vpr * vm`, left to right. */
        o[i] = vpc * vpr * vm;
    }
}
