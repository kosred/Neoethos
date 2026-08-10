#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef TRENDFLEX_NAN
#define TRENDFLEX_NAN (__int_as_float(0x7fffffff))
#endif


#ifndef TRENDFLEX_ASSUME_OUT_PREFILLED
#define TRENDFLEX_ASSUME_OUT_PREFILLED 0
#endif


#ifndef TRENDFLEX_USE_RSQRT_NR
#define TRENDFLEX_USE_RSQRT_NR 0
#endif


static __device__ __forceinline__ float trendflex_round_half(float v) {
    return roundf(v);
}

static __device__ __forceinline__ float inv_sqrt_pos(float x) {
#if TRENDFLEX_USE_RSQRT_NR


    float y = rsqrtf(x);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
#else
    return 1.0f / sqrtf(x);
#endif
}


extern "C" __global__ void trendflex_batch_f32(const float* __restrict__ prices,
                                               const int*   __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               int max_period,
                                               float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];

    if (series_len <= 0 || period <= 0 || period >= series_len) return;
    if (first_valid < 0 || first_valid >= series_len) return;
    if (max_period <= 0 || period > max_period) return;

    const int tail_len = series_len - first_valid;
    if (tail_len < period) return;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) ss_period = 1;
    if (tail_len < ss_period) return;


    const double PI    = 3.1415926535897932384626433832795;
    const double ROOT2 = 1.4142135623730951;

    const double inv_ss = 1.0 / (double)ss_period;
    const double k      = ROOT2 * PI * inv_ss;
    const double a      = exp(-k);
    const double a_sq   = a * a;
    const double b      = 2.0 * a * cos(k);
    const double c      = 0.5 * (1.0 + a_sq - b);

    const int warm = first_valid + period;
    const int warm_clamped = (warm < series_len) ? warm : series_len;

    const size_t base = (size_t)combo * (size_t)series_len;
    float* __restrict__ row_out = out + base;

#if !TRENDFLEX_ASSUME_OUT_PREFILLED
    for (int i = 0; i < warm_clamped; ++i) {
        row_out[i] = TRENDFLEX_NAN;
    }
#endif
    if (warm >= series_len) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    float* __restrict__ sh = reinterpret_cast<float*>(shraw);
    float* __restrict__ ring = sh + (size_t)threadIdx.x * (size_t)max_period;

    const int fidx = first_valid;


    double prev2 = (double)prices[fidx];
    ring[0] = (float)prev2;
    double rolling_sum = (double)ring[0];

    double prev1 = prev2;
    double prev_price = prev2;
    if (period >= 2) {
        const double p1 = (double)prices[fidx + 1];
        prev1 = p1;
        prev_price = p1;
        ring[1] = (float)p1;
        rolling_sum += (double)ring[1];
    }

    for (int t = 2; t < period; ++t) {
        const double cur_price = (double)prices[fidx + t];
        const double ss = fma(c, (cur_price + prev_price),
                              fma(b, prev1, -a_sq * prev2));
        const float ss_f = (float)ss;
        ring[t] = ss_f;
        rolling_sum += (double)ss_f;
        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }

    const double tp_f   = (double)period;
    const double inv_tp = 1.0 / tp_f;
    double ms_prev = 0.0;

    for (int row = warm; row < series_len; ++row) {
        const double cur_price = (double)prices[row];
        const double ss = fma(c, (cur_price + prev_price),
                              fma(b, prev1, -a_sq * prev2));

        const float ss_f = (float)ss;
        const double ss_q = (double)ss_f;
        const double my_sum  = (tp_f * ss_q - rolling_sum) * inv_tp;
        const double ms_current = fma(0.04, my_sum * my_sum, 0.96 * ms_prev);
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0) {
            out_val = (float)(my_sum / sqrt(ms_current));
        }
        row_out[row] = out_val;

        const int pos = (row - fidx) % period;
        const double ss_old = (double)ring[pos];
        ring[pos] = ss_f;
        rolling_sum += ss_q - ss_old;

        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }
}


extern "C" __global__ void trendflex_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ ssf_tm,
    float* __restrict__ out_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (series_len <= 0 || period <= 0 || period >= series_len) return;

    const int stride = num_series;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int tail_len = series_len - first_valid;
    if (tail_len < period) return;

    const double PI    = 3.1415926535897932384626433832795;
    const double ROOT2 = 1.4142135623730951;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) ss_period = 1;
    if (tail_len < ss_period) return;

    const double inv_ss = 1.0 / (double)ss_period;
    const double k = ROOT2 * PI * inv_ss;
    const double a_d    = exp(-k);
    const double a_sq_d = a_d * a_d;
    const double b_d    = 2.0 * a_d * cos(k);
    const double c_d    = 0.5 * (1.0 + a_sq_d - b_d);
    const float  a      = (float)a_d;
    const float  a_sq   = (float)a_sq_d;
    const float  b      = (float)b_d;
    const float  c      = (float)c_d;


    auto at = [stride, series](int row) { return row * stride + series; };

    const int warm = first_valid + period;
#if !TRENDFLEX_ASSUME_OUT_PREFILLED
    const int nan_end = warm < series_len ? warm : series_len;
    for (int row = 0; row < nan_end; ++row) {
        out_tm[at(row)] = TRENDFLEX_NAN;
    }
#endif
    if (warm >= series_len) return;


    const int fidx = first_valid;


    float prev2 = prices_tm[at(fidx)];
    ssf_tm[at(fidx)] = prev2;
    float rolling_sum = prev2;


    float prev1, prev_price;
    if (tail_len > 1) {
        const float p1 = prices_tm[at(fidx + 1)];
        prev1 = p1;
        ssf_tm[at(fidx + 1)] = prev1;
        rolling_sum += prev1;
        prev_price = p1;
    } else {
        return;
    }


    for (int t = 2; t < period; ++t) {
        const float cur_price = prices_tm[at(fidx + t)];
        const float ss = fmaf(c, (cur_price + prev_price),
                              fmaf(b, prev1, -a_sq * prev2));
        ssf_tm[at(fidx + t)] = ss;
        rolling_sum += ss;
        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }


    const float tp_f   = (float)period;
    const float inv_tp = 1.0f / tp_f;
    float ms_prev = 0.0f;

    for (int row = warm; row < series_len; ++row) {
        const float cur_price = prices_tm[at(row)];
        const float ss = fmaf(c, (cur_price + prev_price),
                              fmaf(b, prev1, -a_sq * prev2));

        const float my_sum  = (tp_f * ss - rolling_sum) * inv_tp;
        const float my_sum2 = my_sum * my_sum;
        const float ms_current = fmaf(0.04f, my_sum2, 0.96f * ms_prev);
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0f) {
            out_val = my_sum * inv_sqrt_pos(ms_current);
        }
        out_tm[at(row)] = out_val;

        const float ss_old = ssf_tm[at(row - period)];
        rolling_sum += ss - ss_old;
        ssf_tm[at(row)] = ss;

        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `moving_averages/trendflex.rs::trendflex_scalar_into` (l.440).
// `ss_period = round(period / 2.0)` (`trendflex.rs:390`); default period 20.
//
//     warm = first_valid + period           -> NaN prefix
//     a    = exp(-1.414 * PI / ss_period)
//     b    = 2.0 * a * cos(1.414 * PI / ss_period)
//     c    = (1.0 + a*a - b) * 0.5
//     x    = data[first_valid..], m = len - first_valid
//     prev2 = x[0] ; prev1 = if m > 1 { x[1] } else { x[0] }
//     ring[period] seeded with prev2 (and prev1 when m > 1), sum likewise
//     for i in 2..m:
//         cur = (-a_sq).mul_add(prev2, b.mul_add(prev1, c * (x[i] + x[i-1])))
//         ... once i >= period:
//         my_sum     = (period * cur - sum) * inv_tp
//         ms_current = 0.04.mul_add(my_sum*my_sum, 0.96 * ms_prev)
//         out        = if ms_current != 0.0 { my_sum / sqrt(ms_current) } else { 0.0 }
//         sum       += cur - old ; ring rotates
//
// THE 1.414 IS A LITERAL, NOT sqrt(2). Ehlers' published constant is 1.414 and
// the reference writes `1.414_f64`; substituting `sqrt(2.0)` changes `a` in the
// 4th significant figure and every bar with it. It is carried across verbatim.
//
// The two nested mul_adds are ONE rounding each; written unfused the inner
// `b*prev1 + c*(x[i]+x[i-1])` would round twice. `0.04.mul_add(q, 0.96*ms_prev)`
// is likewise fused, with `0.96 * ms_prev` rounded first.
//
// `ms_current != 0.0` is an EXACT zero test on the CPU — not a tolerance — so
// no epsilon appears here and none may be invented. `sqrt` is the IEEE double
// square root, never `sqrtf` and never `rsqrtf`: the f32 file used `rsqrtf`,
// a fast reciprocal-square-root approximation, which is not even correctly
// rounded in f32.
//
// Serial recurrence with two carried scalars plus a ring -> ONE THREAD PER
// COLUMN, bars ascending. The ring is `period` entries in a fixed local array,
// so the compiled kernel carries the period bound.
//
// f32 -> f64 audit: pointers/locals widened; `sqrtf`->`sqrt`, `rsqrtf` removed,
// `roundf`->`round`; every literal widened; the f32 NaN constant replaced by
// the f64 quiet-NaN bit pattern; no fast-math intrinsic survives; no min/max
// chain.
// ---------------------------------------------------------------------------

#ifndef TRENDFLEX_MAX_PERIOD_F64
#define TRENDFLEX_MAX_PERIOD_F64 512
#endif

static __device__ __forceinline__ double trendflex_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void trendflex_batch_f64(const double* __restrict__ prices,
                         int n,
                         const int*   __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = trendflex_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const long long warm_ll =
        static_cast<long long>(first_valid) + static_cast<long long>(period);
    if (period <= 0 || period > TRENDFLEX_MAX_PERIOD_F64 || first_valid >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const int nan_end = (warm_ll >= n) ? n : static_cast<int>(warm_ll);
    for (int t = 0; t < nan_end; ++t) row[t] = nan_d;

    const int m = n - first_valid;
    if (m < period) return;

    // ss_period = (period as f64 / 2.0).round() as usize
    const int ss_period = static_cast<int>(round(static_cast<double>(period) / 2.0));
    if (ss_period <= 0 || m < ss_period) return;

    const double PI_D = 3.14159265358979323846;
    const double ang = 1.414 * PI_D / static_cast<double>(ss_period);
    const double a = exp(-ang);
    const double a_sq = a * a;
    const double b = 2.0 * a * cos(ang);
    const double c = (1.0 + a_sq - b) * 0.5;

    const double* __restrict__ x = prices + first_valid;

    double prev2 = x[0];
    double prev1 = (m > 1) ? x[1] : x[0];

    double ring[TRENDFLEX_MAX_PERIOD_F64];
    for (int k = 0; k < period; ++k) ring[k] = 0.0;
    int head = 0;
    double sum = 0.0;

    ring[head] = prev2;
    sum += prev2;
    head = (head + 1) % period;
    if (m > 1) {
        ring[head] = prev1;
        sum += prev1;
        head = (head + 1) % period;
    }

    const double tp_f = static_cast<double>(period);
    const double inv_tp = 1.0 / tp_f;
    double ms_prev = 0.0;

    int i = 2;
    while (i < m && i < period) {
        const double cur = fma(-a_sq, prev2, fma(b, prev1, c * (x[i] + x[i - 1])));
        prev2 = prev1;
        prev1 = cur;

        sum += cur;
        ring[head] = cur;
        head = (head + 1) % period;
        ++i;
    }

    while (i < m) {
        const double cur = fma(-a_sq, prev2, fma(b, prev1, c * (x[i] + x[i - 1])));
        prev2 = prev1;
        prev1 = cur;

        const double my_sum = (tp_f * cur - sum) * inv_tp;
        const double ms_current = fma(0.04, my_sum * my_sum, 0.96 * ms_prev);
        ms_prev = ms_current;

        row[first_valid + i] = (ms_current != 0.0) ? (my_sum / sqrt(ms_current)) : 0.0;

        const double old = ring[head];
        sum += cur - old;
        ring[head] = cur;
        head = (head + 1) % period;

        ++i;
    }
}
