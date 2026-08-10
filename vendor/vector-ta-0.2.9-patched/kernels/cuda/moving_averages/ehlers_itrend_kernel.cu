#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif


__device__ __forceinline__ float lerp_fma(float prev, float x, float a) {
    return __fmaf_rn(a, x - prev, prev);
}

template <typename T>
__device__ __forceinline__ T clampT(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

extern "C" __global__
void ehlers_itrend_batch_f32(const float* __restrict__ prices,
                             const int* __restrict__ warmups,
                             const int* __restrict__ max_dcs,
                             int series_len,
                             int first_valid,
                             int n_combos,
                             int max_shared_dc,
                             float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int warmup = warmups[combo];
    const int max_dc = max_dcs[combo];
    if (warmup <= 0 || max_dc <= 0 || max_shared_dc <= 0) return;
    if (max_shared_dc < max_dc) return;


    if (threadIdx.x != 0) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    float* __restrict__ pfx = reinterpret_cast<float*>(shraw);
    const int cap = max_dc;
    for (int i = 0; i < cap; ++i) pfx[i] = 0.0f;

    const int row_offset = combo * series_len;


    float fir_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float det_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float i1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float q1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float prev_i2 = 0.0f, prev_q2 = 0.0f;
    float prev_re = 0.0f, prev_im = 0.0f;
    float prev_mesa = 0.0f, prev_smooth = 0.0f;
    float prev_it1 = 0.0f, prev_it2 = 0.0f, prev_it3 = 0.0f;

    int ring_ptr = 0;
    int pidx = 0;
    float pcur = 0.0f;

    const int warm_threshold = first_valid + warmup;
    const float c0962 = 0.0962f;
    const float c5769 = 0.5769f;

    for (int i = 0; i < series_len; ++i) {
        const float x0 = prices[i];
        const float x1 = (i >= 1) ? prices[i - 1] : 0.0f;
        const float x2 = (i >= 2) ? prices[i - 2] : 0.0f;
        const float x3 = (i >= 3) ? prices[i - 3] : 0.0f;

        const float fir_val = (4.0f * x0 + 3.0f * x1 + 2.0f * x2 + x3) * 0.1f;
        fir_buf[ring_ptr] = fir_val;


        const int c  = ring_ptr;
        const int c2 = (c >= 2) ? (c - 2) : (c + 5);
        const int c4 = (c >= 4) ? (c - 4) : (c + 3);
        const int c6 = (c >= 6) ? (c - 6) : (c + 1);
        const int c3 = (c >= 3) ? (c - 3) : (c + 4);

        const float fir_0 = fir_buf[c];
        const float fir_2 = fir_buf[c2];
        const float fir_4 = fir_buf[c4];
        const float fir_6 = fir_buf[c6];

        const float period_mult = 0.075f * prev_mesa + 0.54f;
        const float h_in = c0962 * fir_0 + c5769 * fir_2 - c5769 * fir_4 - c0962 * fir_6;

        const float det_val = h_in * period_mult;
        det_buf[c] = det_val;

        const float i1_val = det_buf[c3];
        i1_buf[c] = i1_val;

        const float det_0 = det_buf[c];
        const float det_2 = det_buf[c2];
        const float det_4 = det_buf[c4];
        const float det_6 = det_buf[c6];

        const float h_in_q1 = c0962 * det_0 + c5769 * det_2 - c5769 * det_4 - c0962 * det_6;
        const float q1_val = h_in_q1 * period_mult;
        q1_buf[c] = q1_val;

        const float i1_0 = i1_buf[c];
        const float i1_2 = i1_buf[c2];
        const float i1_4 = i1_buf[c4];
        const float i1_6 = i1_buf[c6];
        const float j_i_val = (c0962 * i1_0 + c5769 * i1_2 - c5769 * i1_4 - c0962 * i1_6) * period_mult;

        const float q1_0 = q1_buf[c];
        const float q1_2 = q1_buf[c2];
        const float q1_4 = q1_buf[c4];
        const float q1_6 = q1_buf[c6];
        const float j_q_val = (c0962 * q1_0 + c5769 * q1_2 - c5769 * q1_4 - c0962 * q1_6) * period_mult;

        const float i2_cur = 0.2f * (i1_val - j_q_val) + 0.8f * prev_i2;
        const float q2_cur = 0.2f * (q1_val + j_i_val) + 0.8f * prev_q2;

        const float re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const float im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const float re_smooth = prev_re + 0.2f * (re_val - prev_re);
        const float im_smooth = prev_im + 0.2f * (im_val - prev_im);
        prev_re = re_smooth;
        prev_im = im_smooth;

        float new_mesa = 0.0f;
        if (re_smooth != 0.0f || im_smooth != 0.0f) {
            const float phase = atan2f(im_smooth, re_smooth);
            if (phase != 0.0f) new_mesa = (2.0f * CUDART_PI_F) / phase;
        }

        const float up_lim  = 1.5f * prev_mesa;
        const float low_lim = 0.67f * prev_mesa;
        new_mesa = clampT(new_mesa, low_lim, up_lim);
        new_mesa = clampT(new_mesa, 6.0f, 50.0f);
        const float final_mesa = prev_mesa + 0.2f * (new_mesa - prev_mesa);
        prev_mesa = final_mesa;
        const float sp_val = prev_smooth + 0.33f * (final_mesa - prev_smooth);
        prev_smooth = sp_val;

        int dcp = __float2int_rn(sp_val);
        dcp = clampT(dcp, 1, max_dc);


        float old = pfx[pidx];
        pidx += 1; if (pidx >= cap) pidx = 0;
        pcur += x0;
        int pback = pidx - dcp; if (pback < 0) pback += cap;
        const float prev_prefix = (pback == pidx) ? old : pfx[pback];
        const float sum_src = pcur - prev_prefix;
        pfx[pidx] = pcur;
        const float it_val  = sum_src / (float)dcp;

        const float eit_val = (i < warmup)
            ? x0
            : (4.0f * it_val + 3.0f * prev_it1 + 2.0f * prev_it2 + prev_it3) * 0.1f;

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out[row_offset + i] = (i >= warm_threshold) ? eit_val : CUDART_NAN_F;

        ring_ptr = (c == 6) ? 0 : (c + 1);
    }
}

extern "C" __global__
void ehlers_itrend_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int warmup,
    int max_dc,
    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) return;
    if (warmup <= 0 || max_dc <= 0) return;

    const int stride = num_series;


    if (threadIdx.x != 0) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    float* __restrict__ pfx = reinterpret_cast<float*>(shraw);
    const int cap = max_dc;
    for (int i = 0; i < cap; ++i) pfx[i] = 0.0f;

    float fir_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float det_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float i1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float q1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float prev_i2 = 0.0f, prev_q2 = 0.0f;
    float prev_re = 0.0f, prev_im = 0.0f;
    float prev_mesa = 0.0f, prev_smooth = 0.0f;
    float prev_it1 = 0.0f, prev_it2 = 0.0f, prev_it3 = 0.0f;
    int ring_ptr = 0;
    int pidx = 0;
    float pcur = 0.0f;

    const int first_valid = first_valids[series_idx];
    const int warm_threshold = first_valid + warmup;
    const float c0962 = 0.0962f;
    const float c5769 = 0.5769f;

    for (int t = 0; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const float x0 = prices_tm[idx];
        const float x1 = (t >= 1) ? prices_tm[(t - 1) * stride + series_idx] : 0.0f;
        const float x2 = (t >= 2) ? prices_tm[(t - 2) * stride + series_idx] : 0.0f;
        const float x3 = (t >= 3) ? prices_tm[(t - 3) * stride + series_idx] : 0.0f;

        const float fir_val = (4.0f * x0 + 3.0f * x1 + 2.0f * x2 + x3) * 0.1f;
        fir_buf[ring_ptr] = fir_val;

        const int c  = ring_ptr;
        const int c2 = (c >= 2) ? (c - 2) : (c + 5);
        const int c4 = (c >= 4) ? (c - 4) : (c + 3);
        const int c6 = (c >= 6) ? (c - 6) : (c + 1);
        const int c3 = (c >= 3) ? (c - 3) : (c + 4);

        const float fir_0 = fir_buf[c];
        const float fir_2 = fir_buf[c2];
        const float fir_4 = fir_buf[c4];
        const float fir_6 = fir_buf[c6];

        const float period_mult = 0.075f * prev_mesa + 0.54f;
        const float h_in = c0962 * fir_0 + c5769 * fir_2 - c5769 * fir_4 - c0962 * fir_6;

        const float det_val = h_in * period_mult;
        det_buf[c] = det_val;

        const float i1_val = det_buf[c3];
        i1_buf[c] = i1_val;

        const float det_0 = det_buf[c];
        const float det_2 = det_buf[c2];
        const float det_4 = det_buf[c4];
        const float det_6 = det_buf[c6];

        const float h_in_q1 = c0962 * det_0 + c5769 * det_2 - c5769 * det_4 - c0962 * det_6;
        const float q1_val = h_in_q1 * period_mult;
        q1_buf[c] = q1_val;

        const float i1_0 = i1_buf[c];
        const float i1_2 = i1_buf[c2];
        const float i1_4 = i1_buf[c4];
        const float i1_6 = i1_buf[c6];
        const float j_i_val = (c0962 * i1_0 + c5769 * i1_2 - c5769 * i1_4 - c0962 * i1_6) * period_mult;

        const float q1_0 = q1_buf[c];
        const float q1_2 = q1_buf[c2];
        const float q1_4 = q1_buf[c4];
        const float q1_6 = q1_buf[c6];
        const float j_q_val = (c0962 * q1_0 + c5769 * q1_2 - c5769 * q1_4 - c0962 * q1_6) * period_mult;

        const float i2_cur = 0.2f * (i1_val - j_q_val) + 0.8f * prev_i2;
        const float q2_cur = 0.2f * (q1_val + j_i_val) + 0.8f * prev_q2;

        const float re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const float im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const float re_smooth = prev_re + 0.2f * (re_val - prev_re);
        const float im_smooth = prev_im + 0.2f * (im_val - prev_im);
        prev_re = re_smooth;
        prev_im = im_smooth;

        float new_mesa = 0.0f;
        if (re_smooth != 0.0f || im_smooth != 0.0f) {
            const float phase = atan2f(im_smooth, re_smooth);
            if (phase != 0.0f) new_mesa = (2.0f * CUDART_PI_F) / phase;
        }
        const float up_lim  = 1.5f * prev_mesa;
        const float low_lim = 0.67f * prev_mesa;
        new_mesa = clampT(new_mesa, low_lim, up_lim);
        new_mesa = clampT(new_mesa, 6.0f, 50.0f);
        const float final_mesa = prev_mesa + 0.2f * (new_mesa - prev_mesa);
        prev_mesa = final_mesa;
        const float sp_val = prev_smooth + 0.33f * (final_mesa - prev_smooth);
        prev_smooth = sp_val;

        int dcp = __float2int_rn(sp_val);
        dcp = clampT(dcp, 1, max_dc);


        float old = pfx[pidx];
        pidx += 1; if (pidx >= cap) pidx = 0;
        pcur += x0;
        int pback = pidx - dcp; if (pback < 0) pback += cap;
        const float prev_prefix = (pback == pidx) ? old : pfx[pback];
        const float sum_src = pcur - prev_prefix;
        pfx[pidx] = pcur;
        const float it_val  = sum_src / (float)dcp;

        const float eit_val = (t < warmup)
            ? x0
            : (4.0f * it_val + 3.0f * prev_it1 + 2.0f * prev_it2 + prev_it3) * 0.1f;

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out_tm[idx] = (t >= warm_threshold) ? eit_val : CUDART_NAN_F;

        ring_ptr = (c == 6) ? 0 : (c + 1);
    }
}


// ===========================================================================
// S2 f64 LANE — ehlers_itrend
// ===========================================================================
// Reference: src/indicators/moving_averages/ehlers_itrend.rs
//   `ehlers_itrend_prepare`     (:167) — first_valid, refusals, warm
//   `ehlers_itrend_with_kernel` (:445) — alloc_with_nan_prefix(len, warm)
//   `ehlers_itrend_scalar_tail` (:216) — the Hilbert transform, the MESA
//                                         period estimate and the trendline
//   Batch route: `ma_batch.rs:1143` sweeps `max_dc_period` and pins
//   `warmup_bars` to 20 — NOT the struct default of 12 (:114). The batch
//   default is the one this lane must reproduce, so 20 is what is written
//   below, with the discrepancy named rather than silently resolved.
//
// WHY THE SWEPT `period` IS `max_dc_period`. `EhlersITrendBatchRange` puts
// `period_range` on `max_dc_period` (`ma_batch.rs:1147`). So a row's "period"
// is the CAP on the dominant-cycle window, not a smoothing length.
//
// THREE CLAMPS, THREE DIFFERENT SEMANTICS — AND NONE OF THEM IS fmin/fmax.
//   * `if new_mesa > up_lim { new_mesa = up_lim }` and the matching lower
//     bound are plain ifs: a NaN `new_mesa` survives both.
//   * `new_mesa.clamp(6.0, 50.0)` is Rust's `f64::clamp`, which is
//     `if self < min { min } else if self > max { max } else { self }` — so a
//     NaN passes THROUGH. `fmin(fmax(v, 6.0), 50.0)` would return 6.0 for NaN,
//     which is a different indicator on any bar where the phase estimate
//     degenerates. Written as the if-chain.
//   * `dcp.clamp(1, max_dc)` is on usize, AFTER `(sp_val + 0.5).floor() as
//     usize`, and Rust's float->int cast SATURATES: negative and NaN both
//     become 0, which the clamp then lifts to 1. Reproduced with
//     `if (!(f >= 1.0)) dcp = 1;` — the negated comparison is what makes NaN
//     land on 1 rather than on max_dc.
//
// THE TWO SUMMATION PATHS ARE NOT EQUIVALENT, AND BOTH ARE HERE.
//   The CPU builds a PREFIX SUM when `first_valid == 0` and every value is
//   finite, and otherwise walks a ring adding `dcp` terms. `prefix[end] -
//   prefix[start]` and a `dcp`-term sum are different roundings of the same
//   exact quantity, so choosing one for both cases would diverge from the CPU
//   on exactly the frames the other path handles. Both are implemented, under
//   the same condition, and the "all finite" scan is done up front.
//   The prefix is not materialised for the whole series — only the last
//   `max_dc + 1` prefix values are ever read (`start >= end - dcp >=
//   end - max_dc`), so a ring of that size is exact, not an approximation.
//
// f32 HAZARDS FIXED: `atan2f` -> `atan2`; `__fmaf_rn` -> the exact
// multiply/add the CPU writes (the CPU line `0.0962*a + 0.5769*b - 0.5769*c -
// 0.0962*d` is FOUR multiplies and THREE add/subs, NOT a chain of fmas —
// fusing it would remove three roundings the CPU makes); 142 f32 literals ->
// f64; the f32 NaN bit pattern -> the f64 one.
// ===========================================================================

#define ITREND_MAX_DC 512

#ifndef NEO_S2_PI
// Rust's `std::f64::consts::PI`, written out rather than relying on `M_PI`
// (which MSVC hides behind _USE_MATH_DEFINES) or on `CUDART_PI` being in scope.
#define NEO_S2_PI 3.14159265358979323846264338327950288
#endif

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// `ring_get(buf, center, off)` from ehlers_itrend.rs:266.
__device__ __forceinline__ double neo_s2_ring7(const double* buf, int center, int off) {
    int idx = center + 7 - off;
    if (idx >= 7) idx -= 7;
    return buf[idx];
}

extern "C" __global__ void neoethos_ehlers_itrend_batch_f64(
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
    const int max_dc = periods[r];
    const int warmup_bars = 20;   // ma_batch.rs:1144 -> unwrap_or(20)

    const bool declined =
        (n <= 0) ||
        (warmup_bars == 0) || (max_dc == 0) || (max_dc > ITREND_MAX_DC) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < warmup_bars);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    const int warm = first_valid + warmup_bars;
    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();

    // `prefix_sum` is Some only when first_valid == 0 AND every value is
    // finite. The scan is the CPU's `for &x in src { if !x.is_finite() ... }`.
    bool use_prefix = (first_valid == 0);
    if (use_prefix) {
        for (int i = 0; i < n; ++i) {
            if (!isfinite(prices[i])) { use_prefix = false; break; }
        }
    }

    double fir_buf[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double det_buf[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double i1_buf [7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_buf [7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    double prev_i2 = 0.0, prev_q2 = 0.0;
    double prev_re = 0.0, prev_im = 0.0;
    double prev_mesa = 0.0, prev_smooth = 0.0;
    double prev_it1 = 0.0, prev_it2 = 0.0, prev_it3 = 0.0;
    double src_l1 = 0.0, src_l2 = 0.0, src_l3 = 0.0;
    int ring_ptr = 0;

    // One backing store, two meanings: prefix sums (path A) or raw values
    // (path B). Path A needs max_dc + 1 entries, path B needs max_dc.
    double sring[ITREND_MAX_DC + 1];
    for (int k = 0; k <= max_dc; ++k) sring[k] = 0.0;
    const int pcap = max_dc + 1;
    double acc = 0.0;          // running prefix, path A
    int sum_idx = 0;           // ring cursor, path B

    for (int i = 0; i < n; ++i) {
        const double x0 = prices[i];

        const double fir_val = (4.0 * x0 + 3.0 * src_l1 + 2.0 * src_l2 + src_l3) / 10.0;
        fir_buf[ring_ptr] = fir_val;

        const double fir_0 = neo_s2_ring7(fir_buf, ring_ptr, 0);
        const double fir_2 = neo_s2_ring7(fir_buf, ring_ptr, 2);
        const double fir_4 = neo_s2_ring7(fir_buf, ring_ptr, 4);
        const double fir_6 = neo_s2_ring7(fir_buf, ring_ptr, 6);

        const double h_in = 0.0962 * fir_0 + 0.5769 * fir_2 - 0.5769 * fir_4 - 0.0962 * fir_6;
        const double period_mult = 0.075 * prev_mesa + 0.54;
        const double det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        const double i1_val = neo_s2_ring7(det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        const double det_0 = neo_s2_ring7(det_buf, ring_ptr, 0);
        const double det_2 = neo_s2_ring7(det_buf, ring_ptr, 2);
        const double det_4 = neo_s2_ring7(det_buf, ring_ptr, 4);
        const double det_6 = neo_s2_ring7(det_buf, ring_ptr, 6);
        const double h_in_q1 = 0.0962 * det_0 + 0.5769 * det_2 - 0.5769 * det_4 - 0.0962 * det_6;
        const double q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        const double i1_0 = neo_s2_ring7(i1_buf, ring_ptr, 0);
        const double i1_2 = neo_s2_ring7(i1_buf, ring_ptr, 2);
        const double i1_4 = neo_s2_ring7(i1_buf, ring_ptr, 4);
        const double i1_6 = neo_s2_ring7(i1_buf, ring_ptr, 6);
        const double j_i_val =
            (0.0962 * i1_0 + 0.5769 * i1_2 - 0.5769 * i1_4 - 0.0962 * i1_6) * period_mult;

        const double q1_0 = neo_s2_ring7(q1_buf, ring_ptr, 0);
        const double q1_2 = neo_s2_ring7(q1_buf, ring_ptr, 2);
        const double q1_4 = neo_s2_ring7(q1_buf, ring_ptr, 4);
        const double q1_6 = neo_s2_ring7(q1_buf, ring_ptr, 6);
        const double j_q_val =
            (0.0962 * q1_0 + 0.5769 * q1_2 - 0.5769 * q1_4 - 0.0962 * q1_6) * period_mult;

        const double i2_cur = 0.2 * (i1_val - j_q_val) + 0.8 * prev_i2;
        const double q2_cur = 0.2 * (q1_val + j_i_val) + 0.8 * prev_q2;

        const double re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const double im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const double re_smooth = 0.2 * re_val + 0.8 * prev_re;
        const double im_smooth = 0.2 * im_val + 0.8 * prev_im;
        prev_re = re_smooth;
        prev_im = im_smooth;

        double new_mesa = 0.0;
        if (re_smooth != 0.0 && im_smooth != 0.0) {
            const double angle = atan2(im_smooth, re_smooth);
            if (angle != 0.0) {
                new_mesa = (2.0 * NEO_S2_PI) / angle;
            }
        }
        const double up_lim = 1.5 * prev_mesa;
        if (new_mesa > up_lim) new_mesa = up_lim;
        const double low_lim = 0.67 * prev_mesa;
        if (new_mesa < low_lim) new_mesa = low_lim;
        // f64::clamp — NaN passes through, deliberately not fmin/fmax.
        if (new_mesa < 6.0) new_mesa = 6.0;
        else if (new_mesa > 50.0) new_mesa = 50.0;

        const double final_mesa = 0.2 * new_mesa + 0.8 * prev_mesa;
        prev_mesa = final_mesa;
        const double sp_val = 0.33 * final_mesa + 0.67 * prev_smooth;
        prev_smooth = sp_val;

        // `(sp_val + 0.5).floor() as usize` then `.clamp(1, max_dc)`.
        const double f = floor(sp_val + 0.5);
        int dcp;
        if (!(f >= 1.0))                 dcp = 1;
        else if (f >= (double)max_dc)    dcp = max_dc;
        else                             dcp = (int)f;

        double sum_src;
        if (use_prefix) {
            // prefix[end] - prefix[start], end = i + 1.
            acc += x0;
            const int end = i + 1;
            sring[end % pcap] = acc;
            const int start = (end > dcp) ? (end - dcp) : 0;
            const double p_end = acc;
            const double p_start = (start == 0) ? 0.0 : sring[start % pcap];
            sum_src = p_end - p_start;
        } else {
            sring[sum_idx] = x0;
            sum_idx += 1;
            if (sum_idx == max_dc) sum_idx = 0;
            sum_src = 0.0;
            int idx2 = sum_idx;
            for (int k = 0; k < dcp; ++k) {
                idx2 = (idx2 == 0) ? (max_dc - 1) : (idx2 - 1);
                sum_src += sring[idx2];
            }
        }
        const double it_val = sum_src / (double)dcp;

        const double eit_val = (i < warmup_bars)
            ? x0
            : ((4.0 * it_val + 3.0 * prev_it1 + 2.0 * prev_it2 + prev_it3) / 10.0);

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if (i >= warm) row[i] = eit_val;

        src_l3 = src_l2;
        src_l2 = src_l1;
        src_l1 = x0;
        ring_ptr += 1;
        if (ring_ptr == 7) ring_ptr = 0;
    }
}
