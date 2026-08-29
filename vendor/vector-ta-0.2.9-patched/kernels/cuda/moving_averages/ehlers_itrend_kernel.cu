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
// f64 NUMERICAL AUTHORITY V1.
// ehlers_itrend_f64_pow2_scaled_chronological_neumaier_window_explicit_wma4_rn_v1
//   The adaptive SMA is reduced chronologically (oldest -> newest) after exact
//   power-of-two normalization with Neumaier compensation. A prefix difference
//   is deliberately forbidden: on long, large-offset series it subtracts two
//   accumulated quantities and can lose many low bits. CPU scalar/Auto/AVX,
//   batch, legacy and stream use the same versioned schedule. The following
//   4-3-2-1 WMA and Hilbert transform are also spelled as explicit operations;
//   build.rs compiles this f64 lane with -fmad=false.
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

__device__ __forceinline__ double neo_s2_floor_power_of_two_v1(double max_abs) {
    const unsigned long long bits =
        ((unsigned long long)__double_as_longlong(max_abs)) & 0x7fffffffffffffffULL;
    const unsigned long long exponent = bits & 0x7ff0000000000000ULL;
    if (exponent != 0ULL) {
        return __longlong_as_double((long long)exponent);
    }
    const int highest_mantissa_bit = 63 - __clzll(bits);
    return __longlong_as_double((long long)(1ULL << highest_mantissa_bit));
}

__device__ __forceinline__ void neo_s2_neumaier_add_v1(
    double value,
    double* sum,
    double* correction)
{
    const double next = *sum + value;
    if (fabs(*sum) >= fabs(value)) {
        *correction += (*sum - next) + value;
    } else {
        *correction += (value - next) + *sum;
    }
    *sum = next;
}

__device__ __forceinline__ double neo_s2_stable_window_mean_v1(
    const double* ring,
    int oldest,
    int count,
    int capacity)
{
    double max_abs = 0.0;
    int index = oldest;
    for (int offset = 0; offset < count; ++offset) {
        const double value = ring[index];
        if (!isfinite(value)) return neo_s2_qnan();
        const double magnitude = fabs(value);
        if (magnitude > max_abs) max_abs = magnitude;
        index += 1;
        if (index == capacity) index = 0;
    }
    if (max_abs == 0.0) return 0.0;

    const double scale = neo_s2_floor_power_of_two_v1(max_abs);
    double sum = 0.0;
    double correction = 0.0;
    index = oldest;
    for (int offset = 0; offset < count; ++offset) {
        neo_s2_neumaier_add_v1(ring[index] / scale, &sum, &correction);
        index += 1;
        if (index == capacity) index = 0;
    }
    const double result = ((sum + correction) / (double)count) * scale;
    if (!isfinite(result)) return neo_s2_qnan();
    return (result == 0.0) ? 0.0 : result;
}

__device__ __forceinline__ double neo_s2_wma4_v1(
    double current,
    double lag1,
    double lag2,
    double lag3)
{
    const double weighted_0 = 4.0 * current;
    const double weighted_1 = weighted_0 + 3.0 * lag1;
    const double weighted_2 = weighted_1 + 2.0 * lag2;
    const double weighted_3 = weighted_2 + lag3;
    return weighted_3 / 10.0;
}

__device__ __forceinline__ double neo_s2_hilbert4_v1(
    double current,
    double lag2,
    double lag4,
    double lag6)
{
    const double term_0 = 0.0962 * current;
    const double term_1 = term_0 + 0.5769 * lag2;
    const double term_2 = term_1 - 0.5769 * lag4;
    return term_2 - 0.0962 * lag6;
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

    // Raw values in insertion order; `sum_idx` always names the next slot.
    double sring[ITREND_MAX_DC + 1];
    for (int k = 0; k < max_dc; ++k) sring[k] = 0.0;
    int sum_idx = 0;

    for (int i = 0; i < n; ++i) {
        const double x0 = prices[i];

        const double fir_val = neo_s2_wma4_v1(x0, src_l1, src_l2, src_l3);
        fir_buf[ring_ptr] = fir_val;

        const double fir_0 = neo_s2_ring7(fir_buf, ring_ptr, 0);
        const double fir_2 = neo_s2_ring7(fir_buf, ring_ptr, 2);
        const double fir_4 = neo_s2_ring7(fir_buf, ring_ptr, 4);
        const double fir_6 = neo_s2_ring7(fir_buf, ring_ptr, 6);

        const double h_in = neo_s2_hilbert4_v1(fir_0, fir_2, fir_4, fir_6);
        const double period_mult = 0.075 * prev_mesa + 0.54;
        const double det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        const double i1_val = neo_s2_ring7(det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        const double det_0 = neo_s2_ring7(det_buf, ring_ptr, 0);
        const double det_2 = neo_s2_ring7(det_buf, ring_ptr, 2);
        const double det_4 = neo_s2_ring7(det_buf, ring_ptr, 4);
        const double det_6 = neo_s2_ring7(det_buf, ring_ptr, 6);
        const double h_in_q1 = neo_s2_hilbert4_v1(det_0, det_2, det_4, det_6);
        const double q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        const double i1_0 = neo_s2_ring7(i1_buf, ring_ptr, 0);
        const double i1_2 = neo_s2_ring7(i1_buf, ring_ptr, 2);
        const double i1_4 = neo_s2_ring7(i1_buf, ring_ptr, 4);
        const double i1_6 = neo_s2_ring7(i1_buf, ring_ptr, 6);
        const double j_i_val = neo_s2_hilbert4_v1(i1_0, i1_2, i1_4, i1_6) * period_mult;

        const double q1_0 = neo_s2_ring7(q1_buf, ring_ptr, 0);
        const double q1_2 = neo_s2_ring7(q1_buf, ring_ptr, 2);
        const double q1_4 = neo_s2_ring7(q1_buf, ring_ptr, 4);
        const double q1_6 = neo_s2_ring7(q1_buf, ring_ptr, 6);
        const double j_q_val = neo_s2_hilbert4_v1(q1_0, q1_2, q1_4, q1_6) * period_mult;

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

        sring[sum_idx] = x0;
        sum_idx += 1;
        if (sum_idx == max_dc) sum_idx = 0;
        const int oldest = (sum_idx + max_dc - dcp) % max_dc;
        const double it_val =
            neo_s2_stable_window_mean_v1(sring, oldest, dcp, max_dc);

        const double eit_val = (i < warmup_bars)
            ? x0
            : neo_s2_wma4_v1(it_val, prev_it1, prev_it2, prev_it3);

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
