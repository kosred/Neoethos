#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


struct dsf32 { float hi, lo; };

__device__ __forceinline__ void two_sum_f(float a, float b, float& s, float& e) {

    float t = a + b;
    float bp = t - a;
    e = (a - (t - bp)) + (b - bp);
    s = t;
}

__device__ __forceinline__ dsf32 ds_from_sum(float s, float c) {
    dsf32 r;
    two_sum_f(s, c, r.hi, r.lo);
    return r;
}

__device__ __forceinline__ dsf32 ds_fma(const dsf32 y, float a, float x) {

    float p1 = a * y.hi;
    float e1 = fmaf(a, y.hi, -p1);
    float p2 = a * y.lo;
    float e2 = fmaf(a, y.lo, -p2);

    float s, es;       two_sum_f(p1, p2, s, es);
    float s2, ex;      two_sum_f(s,  x,  s2, ex);

    float corr = (e1 + e2) + (es + ex);
    float hi, lo;      two_sum_f(s2, corr, hi, lo);
    return {hi, lo};
}


__device__ __forceinline__ float cand_long(float prev_low, float mult, const dsf32 dm) {

    float t = fmaf(-mult, dm.hi, prev_low);
    return fmaf(-mult, dm.lo, t);
}
__device__ __forceinline__ float cand_short(float prev_high, float mult, const dsf32 dm) {
    float t = fmaf( mult, dm.hi, prev_high);
    return fmaf( mult, dm.lo, t);
}


__device__ __forceinline__ float reduce_max4(const float r[4], int n) {
    float m = r[0];
    if (n > 1) m = fmaxf(m, r[1]);
    if (n > 2) m = fmaxf(m, r[2]);
    if (n > 3) m = fmaxf(m, r[3]);
    return m;
}
__device__ __forceinline__ float reduce_min4(const float r[4], int n) {
    float m = r[0];
    if (n > 1) m = fminf(m, r[1]);
    if (n > 2) m = fminf(m, r[2]);
    if (n > 3) m = fminf(m, r[3]);
    return m;
}

extern "C" __global__
void safezonestop_build_dm_raw_f32(const float* __restrict__ high,
                                   const float* __restrict__ low,
                                   int len,
                                   int first,
                                   int dir_long,
                                   float* __restrict__ dm_raw)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    if (i <= first) {
        dm_raw[i] = 0.0f;
        return;
    }

    const float up = high[i] - high[i - 1];
    const float dn = low[i - 1] - low[i];
    const float up_pos = (up > 0.0f) ? up : 0.0f;
    const float dn_pos = (dn > 0.0f) ? dn : 0.0f;

    dm_raw[i] = dir_long ? ((dn_pos > up_pos) ? dn_pos : 0.0f)
                         : ((up_pos > dn_pos) ? up_pos : 0.0f);
}

extern "C" __global__
void safezonestop_batch_f32(const float* __restrict__ high,
                            const float* __restrict__ low,
                            const float* __restrict__ dm_raw,
                            int len,
                            int first,
                            const int*  __restrict__ periods,
                            const float* __restrict__ mults,
                            const int*  __restrict__ lookbacks,
                            int n_rows,
                            int dir_long,


                            int*   __restrict__ q_idx,
                            float* __restrict__ q_val,
                            int lb_cap,
                            float* __restrict__ out)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridDim.x == 1 && blockDim.x == 1) { row = blockIdx.y; }
    if (row >= n_rows) return;

    const int   period = periods[row];
    const float mult_f = mults[row];
    const int   lb     = lookbacks[row];
    const float nan_f  = CUDART_NAN_F;

    const int base = row * len;


    if (len <= 0 || period <= 0 || lb <= 0 || first < 0 || first >= len) {
        for (int i = 0; i < len; ++i) out[base + i] = nan_f;
        return;
    }

    const int end0 = first + period;
    const int warm = first + ((period > lb) ? period : lb) - 1;
    if (end0 >= len) {
        for (int i = 0; i < len; ++i) out[base + i] = nan_f;
        return;
    }


    for (int i = 0; i <= warm && i < len; ++i) out[base + i] = nan_f;
    if (warm >= len - 1) return;


    float sum = 0.0f, c = 0.0f;
    for (int j = first + 1; j <= end0; ++j) {
        float x = dm_raw[j];
        float t = sum + x;

        if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
        else                        c += (x   - t) + sum;
        sum = t;
    }
    dsf32 dm = ds_from_sum(sum, c);


    const float invp = 1.0f / (float)period;
    const float alpha_f = fmaf(-invp, 1.0f, 1.0f);


    const bool have_deque = (q_idx != nullptr) && (q_val != nullptr) && (lb_cap >= (lb + 1));
    const bool small_win  = (lb <= 4);


    int *qidx = nullptr; float *qv = nullptr;
    int q_head = 0, q_tail = 0, q_len = 0;
    if (have_deque) {
        qidx = q_idx + row * lb_cap;
        qv   = q_val + row * lb_cap;
    }
    auto ring_inc = [&](int x) { int y = x + 1; return (y == lb_cap) ? 0 : y; };
    auto ring_dec = [&](int x) { return (x == 0) ? (lb_cap - 1) : (x - 1); };


    float ringv[4]; int rpos = 0, rcount = 0;


    if (dir_long) {

        float prev_lm1 = low[end0 - 1];
        float cand = cand_long(prev_lm1, mult_f, dm);

        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
            if (end0 >= warm) out[base + end0] = reduce_max4(ringv, rcount);
        } else if (have_deque) {
            int start = end0 + 1 - lb;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand) { q_tail = last; --q_len; } else break; }
            qidx[q_tail] = end0; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (end0 >= warm && q_len > 0) out[base + end0] = qv[q_head];
        } else {
            if (end0 >= warm) out[base + end0] = cand;
        }

        float prev_l = low[end0];


        for (int i = end0 + 1; i < len; ++i) {
            float drm = dm_raw[i];
            dm = ds_fma(dm, alpha_f, drm);

            float cand_i = cand_long(prev_l, mult_f, dm);

            if (small_win) {
                ringv[rpos] = cand_i; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
                if (i >= warm) out[base + i] = reduce_max4(ringv, rcount);
            } else if (have_deque) {
                int start = i + 1 - lb;
                while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
                while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand_i) { q_tail = last; --q_len; } else break; }
                qidx[q_tail] = i; qv[q_tail] = cand_i; q_tail = ring_inc(q_tail); ++q_len;
                if (i >= warm && q_len > 0) out[base + i] = qv[q_head];
            } else {
                if (i >= warm) out[base + i] = cand_i;
            }

            prev_l = low[i];
        }
    } else {

        float prev_hm1 = high[end0 - 1];
        float cand = cand_short(prev_hm1, mult_f, dm);

        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
            if (end0 >= warm) out[base + end0] = reduce_min4(ringv, rcount);
        } else if (have_deque) {
            int start = end0 + 1 - lb;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand) { q_tail = last; --q_len; } else break; }
            qidx[q_tail] = end0; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (end0 >= warm && q_len > 0) out[base + end0] = qv[q_head];
        } else {
            if (end0 >= warm) out[base + end0] = cand;
        }

        float prev_h = high[end0];

        for (int i = end0 + 1; i < len; ++i) {
            float drm = dm_raw[i];
            dm = ds_fma(dm, alpha_f, drm);

            float cand_i = cand_short(prev_h, mult_f, dm);

            if (small_win) {
                ringv[rpos] = cand_i; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
                if (i >= warm) out[base + i] = reduce_min4(ringv, rcount);
            } else if (have_deque) {
                int start = i + 1 - lb;
                while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
                while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand_i) { q_tail = last; --q_len; } else break; }
                qidx[q_tail] = i; qv[q_tail] = cand_i; q_tail = ring_inc(q_tail); ++q_len;
                if (i >= warm && q_len > 0) out[base + i] = qv[q_head];
            } else {
                if (i >= warm) out[base + i] = cand_i;
            }

            prev_h = high[i];
        }
    }
}

extern "C" __global__
void safezonestop_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    int cols,
    int rows,
    int period,
    float mult,
    int max_lookback,
    const int* __restrict__ first_valids,
    int dir_long,

    int*   __restrict__ q_idx_tm,
    float* __restrict__ q_val_tm,
    int lb_cap,
    float* __restrict__ out_tm)
{

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int first = first_valids[s];
    const int len   = rows;
    const float nan_f = CUDART_NAN_F;

    auto at = [cols](const float* buf, int t, int ss) { return buf[t * cols + ss]; };

    if (first < 0 || first >= len || period <= 0 || max_lookback <= 0) {
        for (int t = 0; t < len; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    const int end0 = first + period;
    const int warm = first + ((period > max_lookback) ? period : max_lookback) - 1;
    if (end0 >= len) {
        for (int t = 0; t < len; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    for (int t = 0; t <= warm && t < len; ++t) out_tm[t * cols + s] = nan_f;
    if (warm >= len - 1) return;


    float sum = 0.0f, c = 0.0f;
    float prev_h = at(high_tm, first, s);
    float prev_l = at(low_tm,  first, s);
    for (int t = first + 1; t <= end0; ++t) {
        float h = at(high_tm, t, s);
        float l = at(low_tm,  t, s);
        float up = h - prev_h;
        float dn = prev_l - l;
        float up_pos = (up > 0.0f) ? up : 0.0f;
        float dn_pos = (dn > 0.0f) ? dn : 0.0f;
        float drm = dir_long ? ((dn_pos > up_pos) ? dn_pos : 0.0f)
                             : ((up_pos > dn_pos) ? up_pos : 0.0f);
        float tsum = sum + drm;
        if (fabsf(sum) >= fabsf(drm)) c += (sum - tsum) + drm; else c += (drm - tsum) + sum;
        sum = tsum;
        prev_h = h; prev_l = l;
    }
    dsf32 dm = ds_from_sum(sum, c);

    const float invp = 1.0f / (float)period;
    const float alpha_f = fmaf(-invp, 1.0f, 1.0f);


    const bool have_deque = (q_idx_tm != nullptr) && (q_val_tm != nullptr) && (lb_cap >= (max_lookback + 1));
    const bool small_win  = (max_lookback <= 4);

    int *qidx = nullptr; float *qv = nullptr; int q_head = 0, q_tail = 0, q_len = 0;
    if (have_deque) {
        qidx = q_idx_tm + s * lb_cap;
        qv   = q_val_tm + s * lb_cap;
    }
    auto ring_inc = [&](int x) { int y = x + 1; return (y == lb_cap) ? 0 : y; };
    auto ring_dec = [&](int x) { return (x == 0) ? (lb_cap - 1) : (x - 1); };

    float ringv[4]; int rpos = 0, rcount = 0;


    {
        int i = end0;
        float cand = dir_long ? cand_long(at(low_tm, i - 1, s),  mult, dm)
                              : cand_short(at(high_tm, i - 1, s), mult, dm);
        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == max_lookback ? 0 : rpos + 1); if (rcount < max_lookback) ++rcount;
            if (i >= warm) out_tm[i * cols + s] = dir_long ? reduce_max4(ringv, rcount)
                                                           : reduce_min4(ringv, rcount);
        } else if (have_deque) {
            int start = i + 1 - max_lookback;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            if (dir_long) { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand) { q_tail = last; --q_len; } else break; } }
            else          { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand) { q_tail = last; --q_len; } else break; } }
            qidx[q_tail] = i; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (i >= warm && q_len > 0) out_tm[i * cols + s] = qv[q_head];
        } else {
            if (i >= warm) out_tm[i * cols + s] = cand;
        }
    }


    float prev_h_i = at(high_tm, end0, s);
    float prev_l_i = at(low_tm,  end0, s);

    for (int i = end0 + 1; i < len; ++i) {
        float h = at(high_tm, i, s);
        float l = at(low_tm,  i, s);
        float up = h - prev_h_i;
        float dn = prev_l_i - l;
        float up_pos = (up > 0.0f) ? up : 0.0f;
        float dn_pos = (dn > 0.0f) ? dn : 0.0f;
        float drm = dir_long ? ((dn_pos > up_pos) ? dn_pos : 0.0f)
                             : ((up_pos > dn_pos) ? up_pos : 0.0f);

        dm = ds_fma(dm, alpha_f, drm);

        float cand = dir_long ? cand_long(prev_l_i, mult, dm)
                              : cand_short(prev_h_i, mult, dm);

        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == max_lookback ? 0 : rpos + 1); if (rcount < max_lookback) ++rcount;
            if (i >= warm) out_tm[i * cols + s] = dir_long ? reduce_max4(ringv, rcount)
                                                           : reduce_min4(ringv, rcount);
        } else if (have_deque) {
            int start = i + 1 - max_lookback;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            if (dir_long) { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand) { q_tail = last; --q_len; } else break; } }
            else          { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand) { q_tail = last; --q_len; } else break; } }
            qidx[q_tail] = i; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (i >= warm && q_len > 0) out_tm[i * cols + s] = qv[q_head];
        } else {
            if (i >= warm) out_tm[i * cols + s] = cand;
        }

        prev_h_i = h; prev_l_i = l;
    }
}


// ===========================================================================
// S1 f64 LANE  --  safezonestop
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/safezonestop.rs -- `safezonestop_scalar` (:459), `warm_len` (:39), `safezonestop_with_kernel` (:237)
//
// INPUT SHAPE: high and low ONLY. `compute_safezonestop_batch`
// (cpu_batch.rs:15008) calls `extract_high_low_input`, and first_valid is
// `first_valid_pair(high, low)`. Declared `HighLow`; an Ohlc ref would adopt
// close's first-valid and shift the series.
//
// PERIOD-BASED: `period` (default 22) is the swept parameter. `mult` (2.5),
// `max_lookback` (3) and `direction` ("long") are the CPU defaults.
//
// `safezonestop_with_kernel` maps `Auto` to `Kernel::Scalar` (:284-287).
//
// max_lookback DEFAULTS TO 3, AND THE CPU'S THRESHOLD IS ZERO
// (`LB_DEQUE_THRESHOLD = 0`), so `max_lookback > 0` always holds and the
// DEQUE branch is the only reachable one. The other branch (a pre-materialised
// `dm_smooth` vector) is dead at the defaults and is not written here; if a
// caller ever passes `max_lookback == 0` this kernel declines by name rather
// than quietly taking a branch it was not checked against.
//
// ARITHMETIC ORDER -- two `mul_add`s, ONE rounding each:
//   `dm_prev = alpha.mul_add(dm_prev, dm_raw)` with
//   `alpha = 1.0 - 1.0/period`. NOT `dm_prev*alpha + dm_raw` as two roundings,
//   and NOT the Wilder `(x - y).mul_add(1/p, y)` form -- this is the
//   accumulate-then-decay variant and the CPU spells it that way.
//   `cand = (-mult).mul_add(dm_prev, prev_low)` for the long direction. The
//   negation is folded into the MULTIPLIER, not applied to the product, which
//   matters for the sign of the rounding.
//   The seed is a plain ascending `boot_sum += dm_raw` over `period` bars and
//   is used UNDIVIDED (`dm_prev = boot_sum`) -- it is a sum, not a mean, and
//   dividing it here would be the obvious and wrong "correction".
//
// NaN / COMPARISON SEMANTICS: `up_pos`/`dn_pos` use `if up > 0.0 { up } else
// { 0.0 }`, which maps NaN to 0.0 -- that is NOT `fmax(up, 0.0)`, which would
// return NaN. The CPU's if-chain is reproduced literally; substituting fmax
// here would change the value at every gapped bar. This is the mirror image of
// the adx bug the brief names, and it is why the rule is "match the CPU", not
// "always use fmax".
//
// THE MONOTONIC DEQUE over the candidate stops is reproduced with the same
// ring, the same capacity (`max_lookback + 1`), the same expiry test
// (`idx_front < i + 1 - max_lookback`) and the same pop predicate
// (`back <= cand` for long). It is small by construction, so it is simulated
// directly rather than rescanned.
//
// WARMUP: `warm_len` gives `first + max(period, max_lookback - 1)` for the
// allocation, while the compute uses `first + max(period, max_lookback) - 1`
// as its emission gate. Those two are NOT the same expression; the emission
// gate is what decides which bars carry a value, and it is what is used here.
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

#define NEO_S1_SZS_MAX_LOOKBACK 64

extern "C" __global__ void neoethos_safezonestop_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
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

    // `SafeZoneStopParams` defaults as read by cpu_batch.rs:15016-15019.
    const double mult = 2.5;
    const int max_lookback = 3;
    const bool dir_long = true;

    const int needed_a = period + 1;
    const int needed = (needed_a > max_lookback) ? needed_a : max_lookback;
    const int cap = ((max_lookback > 1) ? max_lookback : 1) + 1;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        (max_lookback <= 0) ||
        (cap > NEO_S1_SZS_MAX_LOOKBACK) ||
        ((n - first_valid) < needed);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + ((period > max_lookback) ? period : max_lookback) - 1;
    const int warm_end = (warm < n) ? warm : n;
    for (int k = 0; k < warm_end; ++k) row[k] = neo_s1_qnan();
    if (warm >= n) return;

    int   q_idx[NEO_S1_SZS_MAX_LOOKBACK];
    double q_val[NEO_S1_SZS_MAX_LOOKBACK];
    int q_head = 0, q_tail = 0, q_len = 0;

    double prev_high = high[first_valid];
    double prev_low = low[first_valid];
    double dm_prev = 0.0;
    bool dm_ready = false;
    int boot_n = 0;
    double boot_sum = 0.0;
    const double alpha = 1.0 - 1.0 / (double)period;

    for (int i = first_valid + 1; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double up = h - prev_high;
        const double dn = prev_low - l;
        // Deliberately an if-chain, not fmax: a NaN must become 0.0 here.
        const double up_pos = (up > 0.0) ? up : 0.0;
        const double dn_pos = (dn > 0.0) ? dn : 0.0;
        double dm_raw;
        if (dir_long) {
            dm_raw = (dn_pos > up_pos) ? dn_pos : 0.0;
        } else {
            dm_raw = (up_pos > dn_pos) ? up_pos : 0.0;
        }

        if (!dm_ready) {
            boot_n += 1;
            boot_sum += dm_raw;
            if (boot_n == period) {
                dm_prev = boot_sum;
                dm_ready = true;
            }
        } else {
            dm_prev = fma(alpha, dm_prev, dm_raw);
        }

        if (dm_ready) {
            const double cand = dir_long ? fma(-mult, dm_prev, prev_low)
                                         : fma(mult, dm_prev, prev_high);

            const int start = (i + 1 >= max_lookback) ? (i + 1 - max_lookback) : 0;
            while (q_len > 0) {
                const int idx_front = q_idx[q_head];
                if (idx_front < start) {
                    q_head = (q_head + 1 == cap) ? 0 : (q_head + 1);
                    q_len -= 1;
                } else {
                    break;
                }
            }
            while (q_len > 0) {
                const int last = (q_tail == 0) ? (cap - 1) : (q_tail - 1);
                const double back_val = q_val[last];
                const bool pop = dir_long ? (back_val <= cand) : (back_val >= cand);
                if (pop) {
                    q_tail = last;
                    q_len -= 1;
                } else {
                    break;
                }
            }
            q_idx[q_tail] = i;
            q_val[q_tail] = cand;
            q_tail = (q_tail + 1 == cap) ? 0 : (q_tail + 1);
            q_len += 1;
        }

        if (i >= warm) {
            row[i] = (q_len > 0) ? q_val[q_head] : neo_s1_qnan();
        }

        prev_high = h;
        prev_low = l;
    }
}
