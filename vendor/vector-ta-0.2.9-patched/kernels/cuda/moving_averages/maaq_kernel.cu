#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


static __forceinline__ __device__
float sc_from_er_poly(float er, float a, float b, float c) {
    float er2 = er * er;
    return fmaf(a, er2, fmaf(b, er, c));
}

extern "C" __global__
void maaq_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    const float* __restrict__ fast_scs,
                    const float* __restrict__ slow_scs,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0 || max_period <= 0) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const float fast_sc = fast_scs[combo];
    const float slow_sc = slow_scs[combo];

    int first = first_valid;
    if (first < 0) first = 0;
    if (first >= series_len) return;

    const int warm = first + period - 1;
    if (warm >= series_len) return;

    extern __shared__ float diffs[];

    const int base_out = combo * series_len;
    const float nan_f = CUDART_NAN_F;
    const float anchor = prices[first];
    const float EPS = 1.0e-12f;


    if (threadIdx.x == 0) {

        for (int idx = 0; idx < first; ++idx) {
            out[base_out + idx] = nan_f;
        }

        for (int idx = first; idx <= warm; ++idx) {
            out[base_out + idx] = prices[idx];
        }
        if (warm + 1 >= series_len) return;


        float vol_sum = 0.0f;
        for (int j = 1; j < period; ++j) {
            const int cur = first + j;
            const float diff = fabsf(prices[cur] - prices[cur - 1]);
            diffs[j] = diff;
            vol_sum += diff;
        }


        const int i0 = warm + 1;
        float prev = prices[warm];
        float prev_input = prices[warm];

        const float newest = prices[i0];
        const float newest_diff = fabsf(newest - prev_input);
        diffs[0] = newest_diff;
        vol_sum += newest_diff;
        prev_input = newest;


        const float a = fast_sc * fast_sc;
        const float b = 2.0f * fast_sc * slow_sc;
        const float c = slow_sc * slow_sc;

        float er = 0.0f;
        if (vol_sum > EPS) {
            er = fabsf(newest - anchor) / vol_sum;
        }
        float sc = sc_from_er_poly(er, a, b, c);
        prev = fmaf(sc, newest - prev, prev);
        out[base_out + i0] = prev;


        int head = 1;
        for (int t = i0 + 1; t < series_len; ++t) {

            vol_sum -= diffs[head];

            const float cur_price = prices[t];
            const float nd = fabsf(cur_price - prev_input);
            diffs[head] = nd;
            vol_sum += nd;
            prev_input = cur_price;

            ++head; if (head == period) head = 0;

            float er_t = 0.0f;
            if (vol_sum > EPS) {
                er_t = fabsf(cur_price - prices[t - period]) / vol_sum;
            }
            const float sc_t = sc_from_er_poly(er_t, a, b, c);
            prev = fmaf(sc_t, cur_price - prev, prev);
            out[base_out + t] = prev;
        }
    }
}

extern "C" __global__
void maaq_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     float fast_sc,
                                     float slow_sc,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;
    if (period <= 0 || series_len <= 0) return;

    extern __shared__ float diffs[];

    int first = first_valids[series_idx];
    if (first < 0) first = 0;
    if (first >= series_len) return;

    const int warm = first + period - 1;
    if (warm >= series_len) return;

    const int stride = num_series;
    const float nan_f = CUDART_NAN_F;
    const float EPS = 1.0e-12f;

    if (threadIdx.x == 0) {

        for (int t = 0; t < warm; ++t) {
            out_tm[t * stride + series_idx] = nan_f;
        }

        const int warm_idx = warm * stride + series_idx;
        out_tm[warm_idx] = prices_tm[warm_idx];

        if (warm + 1 >= series_len) return;


        float vol_sum = 0.0f;
        for (int j = 1; j < period; ++j) {
            const int cur = first + j;
            const int idx = cur * stride + series_idx;
            const int prev_idx = (cur - 1) * stride + series_idx;
            const float diff = fabsf(prices_tm[idx] - prices_tm[prev_idx]);
            diffs[j] = diff;
            vol_sum += diff;
        }

        const int i0 = warm + 1;
        const int prev_idx = warm * stride + series_idx;
        float prev       = prices_tm[prev_idx];
        float prev_input = prices_tm[prev_idx];

        const int cur_idx = i0 * stride + series_idx;
        const float newest = prices_tm[cur_idx];
        const float newest_diff = fabsf(newest - prev_input);
        diffs[0] = newest_diff;
        vol_sum += newest_diff;
        prev_input = newest;

        const float anchor = prices_tm[first * stride + series_idx];


        const float a = fast_sc * fast_sc;
        const float b = 2.0f * fast_sc * slow_sc;
        const float c = slow_sc * slow_sc;

        float er = 0.0f;
        if (vol_sum > EPS) {
            er = fabsf(newest - anchor) / vol_sum;
        }
        float sc = sc_from_er_poly(er, a, b, c);
        prev = fmaf(sc, newest - prev, prev);
        out_tm[cur_idx] = prev;

        int head = 1;
        for (int t = i0 + 1; t < series_len; ++t) {
            vol_sum -= diffs[head];

            const int idx_curr = t * stride + series_idx;
            const float cur_price = prices_tm[idx_curr];
            const float nd = fabsf(cur_price - prev_input);
            diffs[head] = nd;
            vol_sum += nd;
            prev_input = cur_price;

            ++head; if (head == period) head = 0;

            float er_t = 0.0f;
            if (vol_sum > EPS) {
                const int idx_old = (t - period) * stride + series_idx;
                er_t = fabsf(cur_price - prices_tm[idx_old]) / vol_sum;
            }
            const float sc_t = sc_from_er_poly(er_t, a, b, c);
            prev = fmaf(sc_t, cur_price - prev, prev);
            out_tm[idx_curr] = prev;
        }
    }
}


// ===========================================================================
// S2 f64 LANE — maaq  (moving average adaptive Q)
// ===========================================================================
// Reference: src/indicators/moving_averages/maaq.rs
//   `maaq_prepare`     (:279) — first_valid, refusals (note `period >= len`,
//                                NOT `period > len`)
//   `maaq_with_kernel` (:326) — warm = first + period - 1, NaN before it
//   `maaq_scalar`      (:349) — the ring, the efficiency ratio, the recurrence
//   Defaults: fast_period = 2 (:117), slow_period = 30 (:121).
//
// THE EPSILON. The CPU gates the efficiency ratio on `vol_sum > f64::EPSILON`
// — 2.220446049250313e-16. An f32 port of this line would carry
// `FLT_EPSILON` (1.19e-7), which is NINE ORDERS OF MAGNITUDE larger and would
// zero the ratio on any instrument quoted in small numbers. This kernel uses
// `NEO_F64_EPSILON`, which is the CPU's constant, not a widened copy of the f32
// one.
//
// ROUNDINGS PER BAR:
//   nd   = fabs(d[i] - d[i-1])                    -> sub                (1)
//   er   = fabs(d[i] - d[i-period]) / vol_sum     -> sub + div          (2)
//   sc   = fast_sc.mul_add(er, slow_sc); sc *= sc -> fma + mul          (2)
//   prev = sc.mul_add(d[i] - prev, prev)          -> sub + fma          (2)
// Matched below.
//
// NaN SEMANTICS. `vol_sum > NEO_F64_EPSILON` is false when `vol_sum` is NaN, and
// the CPU's `if` has exactly the same behaviour — `f64 > f64` with a NaN
// operand is false in Rust too. So the comparison chain is faithful as written
// and does NOT need fmax/fmin: both sides agree that a NaN volatility sum
// yields `er = 0.0`.
//
// WARMUP AND THE SEED COPY. `maaq_scalar` copies data[first .. first+period]
// straight into the output, and `maaq_with_kernel` then overwrites
// [0 .. first+period-1] with NaN. The net effect is that index
// `first + period - 1` alone survives the copy, carrying `data[first+period-1]`
// — this is reproduced literally rather than simplified, because "simplifying"
// it would drop a real output bar.
// ===========================================================================

// `f64::EPSILON` = 2^-52, written out so no <float.h> ordering question
// arises and so it can never be confused with the f32 epsilon the f32
// kernel above would have used.
#define NEO_F64_EPSILON 2.220446049250313e-16

#define MAAQ_MAX_PERIOD 512

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_maaq_batch_f64(
    const double* __restrict__ prices,
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
    const int fast_p = 2;    // MaaqInput::get_fast_period -> unwrap_or(2)
    const int slow_p = 30;   // MaaqInput::get_slow_period -> unwrap_or(30)

    const bool declined =
        (n <= 0) ||
        (period <= 0) || (period >= n) || (period > MAAQ_MAX_PERIOD) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    const double fast_sc = 2.0 / ((double)fast_p + 1.0);
    const double slow_sc = 2.0 / ((double)slow_p + 1.0);

    double diffs[MAAQ_MAX_PERIOD];
    for (int k = 0; k < period; ++k) diffs[k] = 0.0;

    double vol_sum = 0.0;
    for (int j = 1; j < period; ++j) {
        const double d = fabs(prices[first_valid + j] - prices[first_valid + j - 1]);
        diffs[j] = d;
        vol_sum += d;
    }

    // NaN warmup FIRST, then the one surviving seed bar, so the order matches
    // `maaq_with_kernel` (compute, then overwrite the prefix with NaN).
    const int warm = first_valid + period - 1;
    const int stop = warm < n ? warm : n;
    for (int i = 0; i < stop; ++i) row[i] = neo_s2_qnan();
    if (warm < n) row[warm] = prices[warm];

    const int i0 = first_valid + period;
    if (i0 >= n) {
        for (int i = (warm + 1 > 0 ? warm + 1 : 0); i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    const double new_diff = fabs(prices[i0] - prices[i0 - 1]);
    diffs[0] = new_diff;
    vol_sum += new_diff;

    double prev_val = prices[i0 - 1];
    const double er0 = (vol_sum > NEO_F64_EPSILON)
        ? (fabs(prices[i0] - prices[first_valid]) / vol_sum)
        : 0.0;
    double sc = fma(fast_sc, er0, slow_sc);
    sc *= sc;

    prev_val = fma(sc, prices[i0] - prev_val, prev_val);
    row[i0] = prev_val;

    int head = (period > 1) ? 1 : 0;

    for (int i = i0 + 1; i < n; ++i) {
        vol_sum -= diffs[head];
        const double nd = fabs(prices[i] - prices[i - 1]);
        diffs[head] = nd;
        vol_sum += nd;
        head += 1;
        if (head == period) head = 0;

        const double er = (vol_sum > NEO_F64_EPSILON)
            ? (fabs(prices[i] - prices[i - period]) / vol_sum)
            : 0.0;
        double s = fma(fast_sc, er, slow_sc);
        s *= s;

        prev_val = fma(s, prices[i] - prev_val, prev_val);
        row[i] = prev_val;
    }
}
