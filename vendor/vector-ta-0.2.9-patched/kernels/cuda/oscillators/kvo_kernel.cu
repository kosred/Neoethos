#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void two_diff(float a, float b, float &s, float &e) {
    s = a - b;
    float bb = s - a;
    e = (a - (s - bb)) - b;
}
__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}
__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}

struct f2 { float hi, lo; };

__device__ __forceinline__ f2 f2_make(float x) { f2 r; r.hi = x; r.lo = 0.0f; return r; }


__device__ __forceinline__ void ema_update_f2(f2 &ema, float x, float alpha)
{
    float s, s_err; two_sum(ema.hi, ema.lo, s, s_err);
    float d_hi, d_err; two_diff(x, s, d_hi, d_err);
    float delta_hi = d_hi;
    float delta_lo = d_err - s_err;

    float p_hi, p_lo; two_prod(alpha, delta_hi, p_hi, p_lo);
    p_lo = fmaf(alpha, delta_lo, p_lo);

    float y_hi, y_lo; two_sum(s, p_hi, y_hi, y_lo);
    y_lo += p_lo;
    quick_two_sum(y_hi, y_lo, ema.hi, ema.lo);
}


__device__ __forceinline__ float rcp_nr(float c)
{
    float r = __fdividef(1.0f, c);
    r = r * fmaf(-c, r, 2.0f);
    return r;
}

extern "C" __global__ void kvo_build_vf_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ vf_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0 || first_valid < 0 || first_valid >= len) return;

    const float nanv = f32_nan();
    for (int i = 0; i < len; ++i) {
        vf_out[i] = nanv;
    }
    if (len <= first_valid + 1) return;

    double prev_h = static_cast<double>(high[first_valid]);
    double prev_l = static_cast<double>(low[first_valid]);
    double prev_c = static_cast<double>(close[first_valid]);
    double prev_hlc = prev_h + prev_l + prev_c;
    double prev_dm = prev_h - prev_l;
    int trend = -1;
    double cm = 0.0;

    for (int i = first_valid + 1; i < len; ++i) {
        const double h = static_cast<double>(high[i]);
        const double l = static_cast<double>(low[i]);
        const double c = static_cast<double>(close[i]);
        const double v = static_cast<double>(volume[i]);
        const double hlc = h + l + c;
        const double dm = h - l;

        if (hlc > prev_hlc && trend != 1) {
            trend = 1;
            cm = prev_dm;
        } else if (hlc < prev_hlc && trend != 0) {
            trend = 0;
            cm = prev_dm;
        }

        cm += dm;
        const double temp = fabs(((dm / cm) * 2.0) - 1.0);
        const double sign = (trend == 1) ? 1.0 : -1.0;
        vf_out[i] = static_cast<float>(v * temp * 100.0 * sign);

        prev_hlc = hlc;
        prev_dm = dm;
    }
}


__device__ __forceinline__ void warp_inclusive_scan_affine(float &A, float &B, unsigned lane, unsigned mask) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float A_prev = __shfl_up_sync(mask, A, offset);
        const float B_prev = __shfl_up_sync(mask, B, offset);
        if (lane >= static_cast<unsigned>(offset)) {
            const float A_cur = A;
            const float B_cur = B;
            A = A_cur * A_prev;
            B = __fmaf_rn(A_cur, B_prev, B_cur);
        }
    }
}

extern "C" __global__ void kvo_batch_f32(
    const float* __restrict__ vf,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    int n_combos,
    float* __restrict__ out)
{
    if (len <= 0 || n_combos <= 0) return;


    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    for (int combo = blockIdx.x * warps_per_block + warp_id;
         combo < n_combos;
         combo += gridDim.x * warps_per_block)
    {
        const int s = shorts[combo];
        const int l = longs[combo];
        if (s <= 0 || l < s) continue;

        const int warm = first_valid + 1;
        float* __restrict__ row_out = out + (size_t)combo * (size_t)len;

        const float nanv = f32_nan();
        const int warm_end = (warm < len ? warm : len);
        for (int t = lane; t < warm_end; t += 32) row_out[t] = nanv;
        if (warm >= len) continue;

        const float alpha_s = 2.0f / (float)(s + 1);
        const float alpha_l = 2.0f / (float)(l + 1);
        const float beta_s = 1.0f - alpha_s;
        const float beta_l = 1.0f - alpha_l;

        const float seed = vf[warm];
        float ema_s_prev = seed;
        float ema_l_prev = seed;

        if (lane == 0) row_out[warm] = 0.0f;

        for (int t0 = warm + 1; t0 < len; t0 += 32) {
            const int t = t0 + lane;
            float x = 0.0f;
            if (t < len) x = vf[t];

            float As = beta_s;
            float Bs = alpha_s * x;
            float Al = beta_l;
            float Bl = alpha_l * x;

            warp_inclusive_scan_affine(As, Bs, lane, mask);
            warp_inclusive_scan_affine(Al, Bl, lane, mask);

            const float ema_s = __fmaf_rn(As, ema_s_prev, Bs);
            const float ema_l = __fmaf_rn(Al, ema_l_prev, Bl);

            if (t < len) row_out[t] = ema_s - ema_l;

            const int remain = len - 1 - t0;
            const int last_lane = (remain < 31 ? remain : 31);
            ema_s_prev = __shfl_sync(mask, ema_s, last_lane);
            ema_l_prev = __shfl_sync(mask, ema_l, last_lane);
        }
    }
}


extern "C" __global__ void kvo_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{

    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s];
        if (fv < 0 || fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[(size_t)t * (size_t)cols + s] = f32_nan();
            continue;
        }

        const int warm = fv + 1;

        const int warm_end = (warm < rows ? warm : rows);
        for (int t = 0; t < warm_end; ++t) out_tm[(size_t)t * (size_t)cols + s] = f32_nan();
        if (warm >= rows) continue;

        const float alpha_s = 2.0f / (float)(short_p + 1);
        const float alpha_l = 2.0f / (float)(long_p + 1);

        const size_t idx0 = (size_t)fv * (size_t)cols + s;
        double prev_h = (double)high_tm[idx0];
        double prev_l = (double)low_tm[idx0];
        double prev_c = (double)close_tm[idx0];
        double prev_hlc = prev_h + prev_l + prev_c;
        double prev_dm  = prev_h - prev_l;
        int    trend    = -1;
        double cm       = 0.0;


        {
            const size_t idx = (size_t)warm * (size_t)cols + s;
            const double h = (double)high_tm[idx];
            const double l = (double)low_tm[idx];
            const double c = (double)close_tm[idx];
            const double v = (double)volume_tm[idx];
            const double hlc = h + l + c;
            const double dm  = h - l;

            if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
            else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
            cm += dm;

            const double ratio = dm / cm;
            const double temp  = fabs((ratio * 2.0) - 1.0);
            const double sign  = (trend == 1) ? 1.0 : -1.0;
            const float vf     = (float)(v * temp * 100.0 * sign);

            float ema_s = vf;
            float ema_l = vf;
            out_tm[idx] = 0.0f;

            prev_hlc = hlc;
            prev_dm  = dm;

            #pragma unroll 1
            for (int t = warm + 1; t < rows; ++t) {
                const size_t j = (size_t)t * (size_t)cols + s;
                const double h2 = (double)high_tm[j];
                const double l2 = (double)low_tm[j];
                const double c2 = (double)close_tm[j];
                const double v2 = (double)volume_tm[j];
                const double hlc2 = h2 + l2 + c2;
                const double dm2  = h2 - l2;

                if (hlc2 > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
                else if (hlc2 < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
                cm += dm2;

                const double ratio2 = dm2 / cm;
                const double temp2  = fabs((ratio2 * 2.0) - 1.0);
                const double sign2  = (trend == 1) ? 1.0 : -1.0;
                const float vf2     = (float)(v2 * temp2 * 100.0 * sign2);


                ema_s = fmaf(alpha_s, (vf2 - ema_s), ema_s);
                ema_l = fmaf(alpha_l, (vf2 - ema_l), ema_l);
                out_tm[j] = ema_s - ema_l;

                prev_hlc = hlc2;
                prev_dm  = dm2;
            }
        }
    }
}


// ===========================================================================
// S1 f64 LANE  --  kvo
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/kvo.rs -- `kvo_scalar` (:469), `kvo_scalar_default_2_5` (:575), `kvo_with_kernel` (:279)
//
// PERIOD-INVARIANT. `compute_kvo_batch` (cpu_batch.rs:2985) reads
// `short_period` (default 2) and `long_period` (default 5); there is no
// `period` parameter, so a sweep's rows are byte-identical.
//
// ONE KERNEL SERVES BOTH CPU PATHS. `kvo_scalar` branches to
// `kvo_scalar_default_2_5` when (short, long) == (2, 5), which is the default
// pair. The two bodies are identical except that the fast path spells the
// smoothing constants `2.0/3.0` and `1.0/3.0` where the general path computes
// `2.0/(2.0+1.0)` and `2.0/(5.0+1.0)`. IEEE division is correctly rounded, so
// 2/3 and 2/3 are the same double and 2/6 and 1/3 are the same double: the two
// paths are bit-identical and no special case is written here. This was
// CHECKED rather than assumed, because a divergent fast path at the default
// parameters would be the only path this lane ever exercises.
//
// ARITHMETIC ORDER: `short_ema += (vf - short_ema) * short_alpha` -- subtract,
// multiply, add: THREE roundings, and deliberately NOT `mul_add`. (Contrast
// `wilders`, whose CPU line IS `mul_add` and therefore has one.)
// `vf = v * temp * 100.0 * sign` is left to right, three multiplies.
//
// NaN SEMANTICS: the comparisons `hlc > prev_hlc` / `hlc < prev_hlc` drive an
// int state machine, not a max. The CPU uses the same bare comparisons, and a
// NaN makes BOTH false there and here, so `trend` simply holds -- there is no
// `f64::max` to mirror and no fmax to substitute. Stated because rule 4 asks
// every comparison chain to be checked, not because one was found.
//
// The `trend`/`cm`/`sign` state carries across bars, so this is one thread per
// column walking bars ascending; no scan reformulation preserves it.
//
// WARMUP: `alloc_with_nan_prefix(len, first + 1)`, and first_valid is the
// first index at which high, low, close AND volume are simultaneously non-NaN
// (kvo.rs:292-297).
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

extern "C" __global__ void neoethos_kvo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    double* __restrict__ row = out + (size_t)r * (size_t)n;
    (void)periods;

    // `KvoParams::default()`.
    const int short_period = 2;
    const int long_period = 5;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (short_period < 1) || (long_period < short_period) ||
        ((n - first_valid) < 2);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const double short_alpha = 2.0 / ((double)short_period + 1.0);
    const double long_alpha  = 2.0 / ((double)long_period + 1.0);

    const int warm = first_valid + 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();

    int trend = -1;
    double sign = -1.0;
    double cm = 0.0;

    double prev_hlc = high[first_valid] + low[first_valid] + close[first_valid];
    double prev_dm  = high[first_valid] - low[first_valid];

    double short_ema = 0.0;
    double long_ema = 0.0;

    int i = warm;
    if (i < n) {
        const double h = high[i], l = low[i], c = close[i], v = volume[i];
        const double hlc = h + l + c;
        const double dm = h - l;

        if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; sign = 1.0; }
        else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; sign = -1.0; }
        cm += dm;

        const double temp = fabs((dm / cm) * 2.0 - 1.0);
        const double vf = v * temp * 100.0 * sign;

        short_ema = vf;
        long_ema = vf;
        row[i] = short_ema - long_ema;

        prev_hlc = hlc;
        prev_dm = dm;
        ++i;
    }

    for (; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i], v = volume[i];
        const double hlc = h + l + c;
        const double dm = h - l;

        if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; sign = 1.0; }
        else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; sign = -1.0; }
        cm += dm;

        const double temp = fabs((dm / cm) * 2.0 - 1.0);
        const double vf = v * temp * 100.0 * sign;

        short_ema += (vf - short_ema) * short_alpha;
        long_ema  += (vf - long_ema)  * long_alpha;

        row[i] = short_ema - long_ema;

        prev_hlc = hlc;
        prev_dm = dm;
    }
}
