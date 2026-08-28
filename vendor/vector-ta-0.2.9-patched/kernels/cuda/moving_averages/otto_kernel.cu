#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float nzf(float x) {
    return isfinite(x) ? x : 0.0f;
}

extern "C" __global__
void otto_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ ott_periods,
    const float* __restrict__ ott_percents,
    const int*   __restrict__ fast_vidyas,
    const int*   __restrict__ slow_vidyas,
    const float* __restrict__ cocos,
    int series_len,
    int n_combos,
    int ,
    float* __restrict__ hott_out,
    float* __restrict__ lott_out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;


    const int slow = max(__ldg(slow_vidyas + combo), 1);
    const int fast = max(__ldg(fast_vidyas + combo), 1);
    const int p1 = max(slow / 2, 1);
    const int p2 = slow;
    const int p3 = max(slow * fast, 1);

    const float a1_base = 2.0f / (static_cast<float>(p1) + 1.0f);
    const float a2_base = 2.0f / (static_cast<float>(p2) + 1.0f);
    const float a3_base = 2.0f / (static_cast<float>(p3) + 1.0f);

    const int ott_p = max(__ldg(ott_periods + combo), 1);
    const float a_base_lott = 2.0f / (static_cast<float>(ott_p) + 1.0f);
    const float ott_percent = __ldg(ott_percents + combo);
    const float coco = __ldg(cocos + combo);

    const float fark = ott_percent * 0.01f;
    const float scale_up = (200.0f + ott_percent) / 200.0f;
    const float scale_dn = (200.0f - ott_percent) / 200.0f;

    float* __restrict__ hott_row = hott_out + combo * series_len;
    float* __restrict__ lott_row = lott_out + combo * series_len;


    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;

    const int CMO_P = 9;
    float ring_up_price[CMO_P];
    float ring_dn_price[CMO_P];
    float ring_up_lott[CMO_P];
    float ring_dn_lott[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) {
        ring_up_price[k] = 0.0f; ring_dn_price[k] = 0.0f;
        ring_up_lott[k] = 0.0f; ring_dn_lott[k] = 0.0f;
    }
    float sum_up_price = 0.0f, sum_dn_price = 0.0f;
    float sum_up_lott = 0.0f, sum_dn_lott = 0.0f;
    int head_price = 0;
    int head_lott = 0;
    float prev_price = 0.0f;

    float prev_lott = 0.0f;
    float ma_prev = 0.0f;
    float long_stop_prev = NAN, short_stop_prev = NAN;
    int dir_prev = 1;

    for (int i = 0; i < series_len; ++i) {
        const float price_raw = __ldg(prices + i);
        const float x = nzf(price_raw);
        if (i > 0) {
            float d = price_raw - prev_price;
            if (!isfinite(price_raw) || !isfinite(prev_price)) {
                d = 0.0f;
            }
            if (i >= CMO_P) {
                sum_up_price -= ring_up_price[head_price];
                sum_dn_price -= ring_dn_price[head_price];
            }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up_price[head_price] = up;
            ring_dn_price[head_price] = dn;
            sum_up_price += up;
            sum_dn_price += dn;
            head_price = (head_price + 1) == CMO_P ? 0 : (head_price + 1);
        }
        prev_price = price_raw;

        const float denom_price = sum_up_price + sum_dn_price;
        const float c_abs =
            (i >= CMO_P && denom_price != 0.0f) ? fabsf((sum_up_price - sum_dn_price) / denom_price) : 0.0f;


        const float a1 = a1_base * c_abs;
        const float a2 = a2_base * c_abs;
        const float a3 = a3_base * c_abs;


        v1 = fmaf(a1, x, (1.0f - a1) * v1);
        v2 = fmaf(a2, x, (1.0f - a2) * v2);
        v3 = fmaf(a3, x, (1.0f - a3) * v3);


        const float denom_l = (v2 - v3) + coco;
        const float lott = denom_l != 0.0f ? (v1 / denom_l) : 0.0f;
        lott_row[i] = lott;


        if (i > 0) {
            const float d = lott - prev_lott;
            if (i >= CMO_P) {
                sum_up_lott -= ring_up_lott[head_lott];
                sum_dn_lott -= ring_dn_lott[head_lott];
            }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up_lott[head_lott] = up;
            ring_dn_lott[head_lott] = dn;
            sum_up_lott += up;
            sum_dn_lott += dn;
            head_lott = (head_lott + 1) == CMO_P ? 0 : (head_lott + 1);
        }
        prev_lott = lott;

        const float denom = sum_up_lott + sum_dn_lott;
        const float c2 =
            (i >= CMO_P && denom != 0.0f) ? fabsf((sum_up_lott - sum_dn_lott) / denom) : 0.0f;
        const float a_lott = a_base_lott * c2;
        const float ma = fmaf(a_lott, lott, (1.0f - a_lott) * ma_prev);
        ma_prev = ma;

        if (i == 0) {
            long_stop_prev = ma * (1.0f - fark);
            short_stop_prev = ma * (1.0f + fark);
            const float mt = long_stop_prev;
            hott_row[i] = (ma > mt ? mt * scale_up : mt * scale_dn);
        } else {
            const float ls = ma * (1.0f - fark);
            const float ss = ma * (1.0f + fark);
            const float long_stop = (ma > long_stop_prev) ? fmaxf(ls, long_stop_prev) : ls;
            const float short_stop = (ma < short_stop_prev) ? fminf(ss, short_stop_prev) : ss;
            const int dir = (dir_prev == -1 && ma > short_stop_prev)
                                ? 1
                                : ((dir_prev == 1 && ma < long_stop_prev) ? -1 : dir_prev);
            const float mt = (dir == 1) ? long_stop : short_stop;
            hott_row[i] = (ma > mt ? mt * scale_up : mt * scale_dn);
            long_stop_prev = long_stop;
            short_stop_prev = short_stop;
            dir_prev = dir;
        }
    }
}

extern "C" __global__
void otto_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int ott_period,
    float ott_percent_f,
    int fast_vidya,
    int slow_vidya,
    float coco_f,
    float* __restrict__ hott_tm,
    float* __restrict__ lott_tm
) {
    const int series = blockIdx.x;
    if (series >= rows || threadIdx.x != 0) return;

    const int p1 = max(slow_vidya / 2, 1);
    const int p2 = max(slow_vidya, 1);
    const int p3 = max(slow_vidya * max(fast_vidya, 1), 1);
    const float a1_base = 2.0f / (static_cast<float>(p1) + 1.0f);
    const float a2_base = 2.0f / (static_cast<float>(p2) + 1.0f);
    const float a3_base = 2.0f / (static_cast<float>(p3) + 1.0f);
    const float a_base_lott = 2.0f / (static_cast<float>(max(ott_period, 1)) + 1.0f);
    const float coco = coco_f;
    const float ott_percent = ott_percent_f;
    const float fark = ott_percent * 0.01f;
    const float scale_up = (200.0f + ott_percent) / 200.0f;
    const float scale_dn = (200.0f - ott_percent) / 200.0f;


    const int CMO_P = 9;
    float ring_up_p[CMO_P];
    float ring_dn_p[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up_p[k] = 0.0f; ring_dn_p[k] = 0.0f; }
    float sum_up_p = 0.0f, sum_dn_p = 0.0f; int head_p = 0;

    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    float prev_price = 0.0f;


    float ring_up_l[CMO_P];
    float ring_dn_l[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up_l[k] = 0.0f; ring_dn_l[k] = 0.0f; }
    float sum_up_l = 0.0f, sum_dn_l = 0.0f; int head_l = 0;
    float prev_lott = 0.0f;
    float ma_prev = 0.0f;
    float long_stop_prev = NAN, short_stop_prev = NAN; int dir_prev = 1;

    for (int t = 0; t < cols; ++t) {
        const float x = nzf(prices_tm[t * rows + series]);
        if (t > 0) {
            const float d = x - prev_price;
            if (t >= CMO_P) { sum_up_p -= ring_up_p[head_p]; sum_dn_p -= ring_dn_p[head_p]; }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up_p[head_p] = up; ring_dn_p[head_p] = dn;
            sum_up_p += up; sum_dn_p += dn; head_p = (head_p + 1) == CMO_P ? 0 : (head_p + 1);
        }
        prev_price = x;
        const float denom_p = sum_up_p + sum_dn_p;
        const float c_abs = (t >= CMO_P && denom_p != 0.0f) ? fabsf((sum_up_p - sum_dn_p) / denom_p) : 0.0f;

        const float a1 = a1_base * c_abs;
        const float a2 = a2_base * c_abs;
        const float a3 = a3_base * c_abs;
        v1 = fmaf(a1, x, (1.0f - a1) * v1);
        v2 = fmaf(a2, x, (1.0f - a2) * v2);
        v3 = fmaf(a3, x, (1.0f - a3) * v3);
        const float denom_l = (v2 - v3) + coco;
        const float lott = denom_l != 0.0f ? (v1 / denom_l) : 0.0f;
        lott_tm[t * rows + series] = lott;

        if (t > 0) {
            const float d = lott - prev_lott;
            if (t >= CMO_P) { sum_up_l -= ring_up_l[head_l]; sum_dn_l -= ring_dn_l[head_l]; }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up_l[head_l] = up; ring_dn_l[head_l] = dn;
            sum_up_l += up; sum_dn_l += dn; head_l = (head_l + 1) == CMO_P ? 0 : (head_l + 1);
        }
        prev_lott = lott;
        const float denom_lc = sum_up_l + sum_dn_l;
        const float c2 = (t >= CMO_P && denom_lc != 0.0f) ? fabsf((sum_up_l - sum_dn_l) / denom_lc) : 0.0f;
        const float a_lott = a_base_lott * c2;
        const float ma = fmaf(a_lott, lott, (1.0f - a_lott) * ma_prev);
        ma_prev = ma;

        if (t == 0) {
            long_stop_prev = ma * (1.0f - fark);
            short_stop_prev = ma * (1.0f + fark);
            const float mt = long_stop_prev;
            hott_tm[t * rows + series] = (ma > mt ? mt * scale_up : mt * scale_dn);
        } else {
            const float ls = ma * (1.0f - fark);
            const float ss = ma * (1.0f + fark);
            const float long_stop = (ma > long_stop_prev) ? fmaxf(ls, long_stop_prev) : ls;
            const float short_stop = (ma < short_stop_prev) ? fminf(ss, short_stop_prev) : ss;
            const int dir = (dir_prev == -1 && ma > short_stop_prev)
                                ? 1
                                : ((dir_prev == 1 && ma < long_stop_prev) ? -1 : dir_prev);
            const float mt = (dir == 1) ? long_stop : short_stop;
            hott_tm[t * rows + series] = (ma > mt ? mt * scale_up : mt * scale_dn);
            long_stop_prev = long_stop; short_stop_prev = short_stop; dir_prev = dir;
        }
    }
}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: otto_into_slices (src/indicators/otto.rs:1154-1420), the
// general VAR arm, reached through otto_with_kernel (:1599).
//
// The all_finite fast path at :1183 (otto_var_clean_two_pass_into_slices,
// :1427) is NOT a second formula: it is the general arm with the two NaN
// guards removed -- `val = if x.is_nan() { 0.0 } else { x }` (:1217) and
// `if !x.is_finite() || !prev_x.is_finite() { d = 0.0 }` (:1221). On a frame
// where every bar is finite those guards are no-ops, so the general body
// below reproduces BOTH paths bar for bar.
//
// OUTPUT: the HOTT column -- compute_otto_batch (cpu_batch.rs:15680) resolves
// output_id == "value" to out.hott.
//
// PERIOD-INVARIANT: that batch reads ott_period (2), ott_percent (0.6),
// fast_vidya_length (10), slow_vidya_length (25), correcting_constant
// (100000.0) and ma_type ("VAR") and NEVER `period` (cpu_batch.rs:15657-15662).
// A sweep of five periods gets five identical CPU columns, so this kernel
// writes five identical rows.
//
// FIRST-VALID: Ignored, and that is the contract. Both otto arms walk from
// bar 0 and write every bar; otto_with_kernel allocates with
// alloc_with_nan_prefix(len, 0) (:1605), so there is no warmup prefix at all
// and a first-valid index would name a bar the CPU never skips.
//
// SHAPE: one thread per combo, TWO ascending passes. The first builds lott
// from three variable-alpha EMAs driven by a nine-wide CMO ring; the second
// reads lott back, drives a second nine-wide CMO ring and a fourth EMA, and
// ratchets the band. The second pass reads the first pass's whole output, so
// lott is written into the row and read back before the row is overwritten.
// ===========================================================================

#define OTTO_NEO_CMO_P 9

static __forceinline__ __device__ double otto_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void otto_neo_batch_f64(const double* __restrict__ data,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;
    (void)periods;      // PERIOD-INVARIANT -- see the header.
    (void)first_valid;  // Ignored -- see the header.

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = otto_neo_qnan();

    const int ott_p = 2;                    // cpu_batch.rs:15657
    const double ott_percent = 0.6;         // :15658
    const int fast_vidya = 10;              // :15659
    const int slow_vidya = 25;              // :15660
    const double coco = 100000.0;           // :15661

    // otto_prepare, :1121-1140. first_valid_idx_and_all_finite errors when
    // every bar is NaN; `needed` is (slow * fast).max(10).
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(data[i])) { first = i; break; }
    }
    int needed = slow_vidya * fast_vidya;
    if (needed < 10) needed = 10;

    bool refused = false;
    if (first < 0) refused = true;
    if (ott_p <= 0 || ott_p > n) refused = true;
    if (!refused && (n - first) < needed) refused = true;

    const int p1 = slow_vidya / 2;
    const int p2 = slow_vidya;
    const int p3 = slow_vidya * fast_vidya;
    if (p1 == 0 || p2 == 0 || p3 == 0) refused = true;   // :1174

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    const double a1_base = 2.0 / ((double)p1 + 1.0);
    const double a2_base = 2.0 / ((double)p2 + 1.0);
    const double a3_base = 2.0 / ((double)p3 + 1.0);

    double ring_up[OTTO_NEO_CMO_P];
    double ring_dn[OTTO_NEO_CMO_P];
    for (int k = 0; k < OTTO_NEO_CMO_P; ++k) { ring_up[k] = 0.0; ring_dn[k] = 0.0; }
    double sum_up = 0.0, sum_dn = 0.0;
    int head = 0;

    double v1 = 0.0, v2 = 0.0, v3 = 0.0;
    double prev_x = data[0];

    // ---------------------------------------------------------------- pass 1
    // :1215-1263 -- lott. Written into the row and read back by pass 2.
    for (int i = 0; i < n; ++i) {
        const double x = data[i];
        const double val = isnan(x) ? 0.0 : x;

        if (i > 0) {
            double d = x - prev_x;
            if (!isfinite(x) || !isfinite(prev_x)) d = 0.0;

            if (i >= OTTO_NEO_CMO_P) {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }

            const double up = (d > 0.0) ? d : 0.0;
            const double dn = (d > 0.0) ? 0.0 : -d;
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;

            head += 1;
            if (head == OTTO_NEO_CMO_P) head = 0;

            prev_x = x;
        }

        double cmo_abs = 0.0;
        if (i >= OTTO_NEO_CMO_P) {
            const double denom = sum_up + sum_dn;
            cmo_abs = (denom != 0.0) ? fabs((sum_up - sum_dn) / denom) : 0.0;
        }

        const double a1 = a1_base * cmo_abs;
        const double a2 = a2_base * cmo_abs;
        const double a3 = a3_base * cmo_abs;
        v1 = a1 * val + (1.0 - a1) * v1;
        v2 = a2 * val + (1.0 - a2) * v2;
        v3 = a3 * val + (1.0 - a3) * v3;

        const double denom_l = (v2 - v3) + coco;
        row[i] = v1 / denom_l;
    }

    // ---------------------------------------------------------------- pass 2
    // :1268-1360 -- the VAR moving average over lott, then the band ratchet.
    const double fark = ott_percent * 0.01;
    const double scale_up = (200.0 + ott_percent) / 200.0;
    const double scale_dn = (200.0 - ott_percent) / 200.0;

    double ring_up2[OTTO_NEO_CMO_P];
    double ring_dn2[OTTO_NEO_CMO_P];
    for (int k = 0; k < OTTO_NEO_CMO_P; ++k) { ring_up2[k] = 0.0; ring_dn2[k] = 0.0; }
    double sum_up2 = 0.0, sum_dn2 = 0.0;
    int head2 = 0;
    double prev_lott = row[0];

    const double a_base = 2.0 / ((double)ott_p + 1.0);
    double ma_prev = 0.0;

    double long_stop_prev = nn;
    double short_stop_prev = nn;
    int dir_prev = 1;

    for (int i = 0; i < n; ++i) {
        const double lott_i = row[i];

        if (i > 0) {
            double d = lott_i - prev_lott;
            if (!isfinite(lott_i) || !isfinite(prev_lott)) d = 0.0;
            if (i >= OTTO_NEO_CMO_P) {
                sum_up2 -= ring_up2[head2];
                sum_dn2 -= ring_dn2[head2];
            }
            const double up = (d > 0.0) ? d : 0.0;
            const double dn = (d > 0.0) ? 0.0 : -d;
            ring_up2[head2] = up;
            ring_dn2[head2] = dn;
            sum_up2 += up;
            sum_dn2 += dn;
            head2 += 1;
            if (head2 == OTTO_NEO_CMO_P) head2 = 0;
            prev_lott = lott_i;
        }

        double c_abs = 0.0;
        if (i >= OTTO_NEO_CMO_P) {
            const double denom = sum_up2 + sum_dn2;
            c_abs = (denom != 0.0) ? fabs((sum_up2 - sum_dn2) / denom) : 0.0;
        }

        const double a = a_base * c_abs;
        const double ma = a * lott_i + (1.0 - a) * ma_prev;
        ma_prev = ma;

        double hott;
        if (i == 0) {
            long_stop_prev = ma * (1.0 - fark);
            short_stop_prev = ma * (1.0 + fark);
            const double mt = long_stop_prev;
            hott = (ma > mt) ? (mt * scale_up) : (mt * scale_dn);
        } else {
            const double ls = ma * (1.0 - fark);
            const double ss = ma * (1.0 + fark);
            // f64::max / f64::min at :1569 and :1574 -- they return the
            // non-NaN operand, so fmax / fmin, never an if-chain.
            const double long_stop = (ma > long_stop_prev) ? fmax(ls, long_stop_prev) : ls;
            const double short_stop = (ma < short_stop_prev) ? fmin(ss, short_stop_prev) : ss;
            int dir;
            if (dir_prev == -1 && ma > short_stop_prev) dir = 1;
            else if (dir_prev == 1 && ma < long_stop_prev) dir = -1;
            else dir = dir_prev;
            const double mt = (dir == 1) ? long_stop : short_stop;
            hott = (ma > mt) ? (mt * scale_up) : (mt * scale_dn);
            long_stop_prev = long_stop;
            short_stop_prev = short_stop;
            dir_prev = dir;
        }
        row[i] = hott;
    }
}
