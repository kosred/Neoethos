#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


__device__ __forceinline__ float qnan_f32() {
    return __int_as_float(0x7FC00000u);
}


struct f2 { float hi, lo; };

__device__ __forceinline__ f2 make_f2(float hi, float lo) {

    float s = hi + lo;
    float z = s - hi;
    float e = (hi - (s - z)) + (lo - z);
    f2 r; r.hi = s; r.lo = e; return r;
}


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


__device__ __forceinline__ void two_prod_fma(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}


__device__ __forceinline__ f2 f2_add(const f2 &x, const f2 &y) {
    float s, e;
    two_sum(x.hi, y.hi, s, e);
    e += (x.lo + y.lo);
    return make_f2(s, e);
}


__device__ __forceinline__ f2 f2_mul_scalar(const f2 &x, float b) {
    float p1, e1; two_prod_fma(x.hi, b, p1, e1);
    float p2 = x.lo * b;
    float s, e; two_sum(p1, p2, s, e);
    e += e1;
    return make_f2(s, e);
}


__device__ __forceinline__ f2 ema_update_f2(float a, float x, float b, const f2 &y) {

    float p1, pe1; two_prod_fma(a, x, p1, pe1);
    float p2, pe2; two_prod_fma(b, y.hi, p2, pe2);
    float s, e; two_sum(p1, p2, s, e);
    e += (pe1 + pe2 + y.lo * b);
    return make_f2(s, e);
}


__device__ __forceinline__ float warp_broadcast_load(const float* __restrict__ prices, int idx) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31;
    float v = 0.0f;
    if (lane == 0) {

        v = prices[idx];
    }
    v = __shfl_sync(mask, v, 0);
    return v;
}


template <bool UseWarpBroadcast>
__device__ __forceinline__
void range_filter_scan_one_combo(
    int combo,
    const float* __restrict__ prices,
    const float* __restrict__ range_sizes,
    const int*   __restrict__ range_periods,
    const int*   __restrict__ smooth_flags,
    const int*   __restrict__ smooth_periods,
    int series_len,
    int first_valid,

    float* __restrict__ filter_out,
    float* __restrict__ high_out,
    float* __restrict__ low_out
){
    if (combo < 0) return;
    const float rs_f = range_sizes[combo];
    const int   rp   = range_periods[combo];
    const int   sflag= smooth_flags[combo];
    const int   sp   = smooth_periods[combo];
    if (series_len <= 0 || rp <= 0) return;

    float* __restrict__ f_row = filter_out + combo * series_len;
    float* __restrict__ h_row = high_out   + combo * series_len;
    float* __restrict__ l_row = low_out    + combo * series_len;


    const int warm_extra = sflag ? (rp > sp ? rp : sp) : rp;
    const int warm_end   = first_valid + warm_extra;
    const int capped_warm_end = (warm_end < series_len ? warm_end : series_len);


    const float qnan = qnan_f32();
    for (int i = 0; i < capped_warm_end; ++i) {
        f_row[i] = qnan; h_row[i] = qnan; l_row[i] = qnan;
    }
    if (first_valid >= series_len - 1) return;


    float prev_filter = prices[first_valid];
    float prev_price  = prev_filter;

    bool ac_initialized = false;
    f2   ac_ema = {0.f, 0.f};

    bool range_initialized = false;
    f2   range_ema = {0.f, 0.f};

    const float alpha_ac = 2.0f / (float(rp) + 1.0f);
    const float one_minus_alpha_ac = 1.0f - alpha_ac;
    const float alpha_range = sflag ? (2.0f / (float(sp) + 1.0f)) : 0.0f;
    const float one_minus_alpha_range = 1.0f - alpha_range;


    for (int t = first_valid + 1; t < series_len; ++t) {
        float price = UseWarpBroadcast ? warp_broadcast_load(prices, t) : prices[t];
        const float d = price - prev_price;
        const float abs_change = fabsf(d);

        if (!isnan(abs_change)) {
            if (!ac_initialized) {
                ac_ema = make_f2(abs_change, 0.0f);
                ac_initialized = true;
            } else {
                ac_ema = ema_update_f2(alpha_ac, abs_change, one_minus_alpha_ac, ac_ema);
            }
        }
        if (!ac_initialized) {
            prev_price = price;
            continue;
        }


        float range_unsmoothed = fmaf(ac_ema.hi, rs_f, ac_ema.lo * rs_f);

        float range_cur;
        if (sflag) {
            if (!range_initialized) {
                range_ema = make_f2(range_unsmoothed, 0.0f);
                range_initialized = true;
            } else {
                range_ema = ema_update_f2(alpha_range, range_unsmoothed, one_minus_alpha_range, range_ema);
            }
            range_cur = range_ema.hi + range_ema.lo;
        } else {
            range_cur = range_unsmoothed;
        }

        const float min_b = price - range_cur;
        const float max_b = price + range_cur;

        float current = fminf(fmaxf(prev_filter, min_b), max_b);

        if (t >= capped_warm_end) {
            f_row[t] = current;
            h_row[t] = current + range_cur;
            l_row[t] = current - range_cur;
        }

        prev_filter = current;
        prev_price  = price;
    }
}


extern "C" __global__
void range_filter_batch_f32(const float* __restrict__ prices,
                            const float* __restrict__ range_sizes,
                            const int*   __restrict__ range_periods,
                            const int*   __restrict__ smooth_flags,
                            const int*   __restrict__ smooth_periods,
                            int series_len,
                            int n_combos,
                            int first_valid,
                            float* __restrict__ filter_out,
                            float* __restrict__ high_out,
                            float* __restrict__ low_out) {


    if (gridDim.y > 1) {
        const int combo = blockIdx.y;
        if (combo >= n_combos || threadIdx.x != 0) return;
        range_filter_scan_one_combo<false>(combo, prices, range_sizes, range_periods, smooth_flags, smooth_periods,
                                           series_len, first_valid, filter_out, high_out, low_out);
        return;
    }


    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int combo = tid; combo < n_combos; combo += stride) {
        range_filter_scan_one_combo<true>(combo, prices, range_sizes, range_periods, smooth_flags, smooth_periods,
                                          series_len, first_valid, filter_out, high_out, low_out);
    }
}


extern "C" __global__
void range_filter_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                            float range_size_f,
                                            int   range_period,
                                            int   smooth_flag,
                                            int   smooth_period,
                                            int   num_series,
                                            int   series_len,
                                            const int* __restrict__ first_valids,
                                            float* __restrict__ filter_tm,
                                            float* __restrict__ high_tm,
                                            float* __restrict__ low_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || threadIdx.x != 0) return;
    if (series_len <= 0 || range_period <= 0) return;

    const int first_valid = first_valids[series];
    const int warm_extra = smooth_flag ? (range_period > smooth_period ? range_period : smooth_period) : range_period;
    const int warm_end   = first_valid + warm_extra;
    const int capped_warm_end = (warm_end < series_len ? warm_end : series_len);

    const float qnan = qnan_f32();
    for (int t = 0; t < capped_warm_end; ++t) {
        const int idx = t * num_series + series;
        filter_tm[idx] = qnan; high_tm[idx] = qnan; low_tm[idx] = qnan;
    }
    if (first_valid >= series_len - 1) return;

    float prev_filter = prices_tm[first_valid * num_series + series];
    float prev_price  = prev_filter;

    bool ac_initialized = false;
    f2   ac_ema = {0.f, 0.f};

    bool range_initialized = false;
    f2   range_ema = {0.f, 0.f};

    const float alpha_ac = 2.0f / (float(range_period) + 1.0f);
    const float one_minus_alpha_ac = 1.0f - alpha_ac;
    const float alpha_range = smooth_flag ? (2.0f / (float(smooth_period) + 1.0f)) : 0.0f;
    const float one_minus_alpha_range = 1.0f - alpha_range;

    for (int t = first_valid + 1; t < series_len; ++t) {
        const int idx = t * num_series + series;
        const float price = prices_tm[idx];
        const float d = price - prev_price;
        const float abs_change = fabsf(d);

        if (!isnan(abs_change)) {
            if (!ac_initialized) {
                ac_ema = make_f2(abs_change, 0.0f);
                ac_initialized = true;
            } else {
                ac_ema = ema_update_f2(alpha_ac, abs_change, one_minus_alpha_ac, ac_ema);
            }
        }
        if (!ac_initialized) {
            prev_price = price; continue;
        }

        float range_unsmoothed = fmaf(ac_ema.hi, range_size_f, ac_ema.lo * range_size_f);

        float range_cur;
        if (smooth_flag) {
            if (!range_initialized) {
                range_ema = make_f2(range_unsmoothed, 0.0f);
                range_initialized = true;
            } else {
                range_ema = ema_update_f2(alpha_range, range_unsmoothed, one_minus_alpha_range, range_ema);
            }
            range_cur = range_ema.hi + range_ema.lo;
        } else {
            range_cur = range_unsmoothed;
        }

        const float min_b = price - range_cur;
        const float max_b = price + range_cur;
        const float current = fminf(fmaxf(prev_filter, min_b), max_b);

        if (t >= capped_warm_end) {
            filter_tm[idx] = current;
            high_tm[idx]   = current + range_cur;
            low_tm[idx]    = current - range_cur;
        }

        prev_filter = current;
        prev_price  = price;
    }
}


// ===========================================================================
// S3 f64 LANE — range_filter (filter line)
// ===========================================================================
// Reference: src/indicators/range_filter.rs
//   range_filter_prepare (:506)     — first_valid + the Err branches
//   range_filter_with_kernel (:465) — warmup_end
//   range_filter_scalar (:645)      — the smooth_range == true branch
// Batch defaults: range_size 2.618, range_period 14, smooth_range TRUE,
// smooth_period 27, source close. PERIOD-INVARIANT — no `period` is read.
//
// WHICH OUTPUT. Multi-output (filter / high_band / low_band); compute_range_
// filter_batch maps "value" to FILTER, so this kernel is the filter line.
//
// WHICH BRANCH. get_smooth_range defaults to true (:135) and the batch passes
// that default through, so the SECOND loop (:716) is what runs — the one with
// the extra range EMA. The first loop is a different series, not a faster one.
//
// THE WARMUP PREFIX IS NOT WHERE THE FILTER STARTS. warmup_end is
//   first + max(range_period, smooth_period) = first + 27
// but the scalar begins writing at first + 1 and keeps writing through the
// whole series; the prefix simply overwrites the first 27 of those with NaN.
// So the state at bar warmup_end depends on bars the output never shows —
// starting the recursion at warmup_end instead would produce a different
// series from the first visible bar on.
//
// TWO LAZY SEEDS, NOT ONE. ac_ema and range_ema are each initialised on their
// FIRST finite sample rather than at a fixed index (:689-694, :736-741), and
// while ac_ema is uninitialised the loop `continue`s WITHOUT writing output —
// so a leading NaN run inside the series shifts where the filter begins.
// Both flags are carried here.
//
// NaN GUARD. The update is gated on !abs_change.is_nan() (:721): a NaN bar
// leaves ac_ema untouched instead of poisoning it. Note this is the OPPOSITE
// polarity from a comparison chain that lets NaN through — matching the
// reference means matching where it refuses to update.
//
// clamp IS NOT fmin/fmax. Rust f64::clamp is
//     if self < min { min } else if self > max { max } else { self }
// so a NaN prev_filter passes THROUGH unchanged, whereas fmax(fmin(x,max),min)
// would return a number. Transcribed as the same comparison chain.
//
// ROUNDING. ac_ema = alpha*abs_change + one_minus_alpha*ac_ema is TWO products
// and ONE add — three roundings. The CPU does not use mul_add here (:726, :740)
// and neither does this kernel.
//
// One thread per column.
// ===========================================================================

#define NEO_S3_RF_RANGE_SIZE    2.618
#define NEO_S3_RF_RANGE_PERIOD  14
#define NEO_S3_RF_SMOOTH_PERIOD 27

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_range_filter_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;   // PERIOD-INVARIANT — see the header.

    double* __restrict__ row = out + (size_t)r * (size_t)n;

    const double range_size   = NEO_S3_RF_RANGE_SIZE;
    const int range_period    = NEO_S3_RF_RANGE_PERIOD;
    const int smooth_period   = NEO_S3_RF_SMOOTH_PERIOD;

    const int needed = (range_period > smooth_period) ? range_period : smooth_period;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (range_period == 0) || (range_period > n) ||
        (smooth_period == 0) || (smooth_period > n) ||
        ((n - first_valid) < needed);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warmup_end = first_valid + needed;

    // Every bar the scalar does not write stays NaN, and the first
    // `warmup_end` are overwritten with NaN regardless.
    for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
    if (first_valid + 1 >= n) return;

    const double alpha_ac = 2.0 / ((double)range_period + 1.0);
    const double one_minus_alpha_ac = 1.0 - alpha_ac;
    const double alpha_range = 2.0 / ((double)smooth_period + 1.0);
    const double one_minus_alpha_range = 1.0 - alpha_range;

    double ac_ema = 0.0;
    bool ac_initialized = false;
    double range_ema = 0.0;
    bool range_initialized = false;

    double prev_filter = data[first_valid];
    double prev_price  = prev_filter;

    for (int i = first_valid + 1; i < n; ++i) {
        const double price = data[i];

        const double d = price - prev_price;
        const double abs_change = fabs(d);
        if (!isnan(abs_change)) {
            if (!ac_initialized) {
                ac_ema = abs_change;
                ac_initialized = true;
            } else {
                ac_ema = alpha_ac * abs_change + one_minus_alpha_ac * ac_ema;
            }
        }

        if (!ac_initialized) {
            prev_price = price;
            continue;
        }

        double range = ac_ema * range_size;
        if (!range_initialized) {
            range_ema = range;
            range_initialized = true;
        } else {
            range_ema = alpha_range * range + one_minus_alpha_range * range_ema;
        }
        range = range_ema;

        const double min_b = price - range;
        const double max_b = price + range;
        // f64::clamp — NOT fmin/fmax: NaN passes through.
        double current;
        if (prev_filter < min_b)      current = min_b;
        else if (prev_filter > max_b) current = max_b;
        else                          current = prev_filter;

        if (i >= warmup_end) row[i] = current;

        prev_filter = current;
        prev_price = price;
    }
}
