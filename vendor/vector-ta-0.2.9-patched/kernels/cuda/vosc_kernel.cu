#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef VOSC_NAN
#define VOSC_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


struct ds {
    float hi;
    float lo;
};

__device__ __forceinline__ ds ds_make(float hi, float lo) { ds r{hi, lo}; return r; }


__device__ __forceinline__ ds ds_from_double(double d) {
    float hi = (float)d;
    float lo = (float)(d - (double)hi);
    return ds_make(hi, lo);
}

__device__ __forceinline__ ds ds_add(ds a, ds b) {
    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float e  = (a.hi - (s - bb)) + (b.hi - bb);
    float t  = e + a.lo + b.lo;
    float hi = s + t;
    float lo = t - (hi - s);
    return ds_make(hi, lo);
}
__device__ __forceinline__ ds ds_neg(ds a) { return ds_make(-a.hi, -a.lo); }
__device__ __forceinline__ ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }

__device__ __forceinline__ ds ds_mul_f(ds a, float k) {

    float p  = a.hi * k;
    float e  = fmaf(a.hi, k, -p) + a.lo * k;
    float hi = p + e;
    float lo = e - (hi - p);
    return ds_make(hi, lo);
}

__device__ __forceinline__ float ds_to_float(ds a) { return a.hi + a.lo; }

extern "C" __global__ void vosc_build_prefix_f32_ds(
    const float* __restrict__ data,
    int len,
    float2* __restrict__ prefix_f2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_f2[0] = make_float2(0.0f, 0.0f);
    double acc = 0.0;
    for (int i = 0; i < len; ++i) {
        acc += (double)data[i];
        float hi = (float)acc;
        float lo = (float)(acc - (double)hi);
        prefix_f2[i + 1] = make_float2(hi, lo);
    }
}


extern "C" __global__ void vosc_batch_prefix_f32(
    const double* __restrict__ prefix_sum,
    int len,
    int first_valid,
    const int* __restrict__ short_periods,
    const int* __restrict__ long_periods,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;

    const int warm = first_valid + L - 1;
    const int row_off = combo * len;

    const float inv_S = __fdividef(1.0f, (float)S);
    const float inv_L = __fdividef(1.0f, (float)L);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        float out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;

            ds PT = ds_from_double(prefix_sum[t1]);
            ds PS = ds_from_double(prefix_sum[sS]);
            ds PL = ds_from_double(prefix_sum[sL]);


            ds short_sum = ds_sub(PT, PS);
            ds long_sum  = ds_sub(PT, PL);
            ds savg_ds = ds_mul_f(short_sum, inv_S);
            ds lavg_ds = ds_mul_f(long_sum,  inv_L);
            float lavg = ds_to_float(lavg_ds);
            float num  = ds_to_float(ds_sub(savg_ds, lavg_ds));
            float v = 100.0f * num * __fdividef(1.0f, lavg);
            out_val = v;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void vosc_many_series_one_param_f32(
    const double* __restrict__ prefix_tm,
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;

    const int warm = first_valids[series] + long_period - 1;
    const int stride = num_series;
    const double inv_S = 1.0 / static_cast<double>(short_period);
    const double inv_L = 1.0 / static_cast<double>(long_period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = t * stride + series;
        float out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_idx_t  = t1 * stride + series;
            const int p_idx_sS = sS * stride + series;
            const int p_idx_sL = sL * stride + series;
            const double short_sum = prefix_tm[p_idx_t] - prefix_tm[p_idx_sS];
            const double long_sum  = prefix_tm[p_idx_t] - prefix_tm[p_idx_sL];
            const double lavg = long_sum * inv_L;
            const double savg = short_sum * inv_S;
            const double v = 100.0 * (savg - lavg) / lavg;
            out_val = static_cast<float>(v);
        }
        out_tm[out_idx] = out_val;
        t += step;
    }
}


extern "C" __global__ void pack_double_to_float2(
    const double* __restrict__ in, float2* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (i < n) {
        double d = in[i];
        float hi = (float)d;
        float lo = (float)(d - (double)hi);
        out[i] = make_float2(hi, lo);
        i += stride;
    }
}

extern "C" __global__ void vosc_batch_prefix_f32_ds(
    const float2* __restrict__ prefix_f2,
    int len,
    int first_valid,
    const int* __restrict__ short_periods,
    const int* __restrict__ long_periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;
    const int warm = first_valid + L - 1;
    const int row_off = combo * len;
    const float invS = __fdividef(1.0f, (float)S);
    const float invL = __fdividef(1.0f, (float)L);
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        float out_val = VOSC_NAN;
        if (LIKELY(t >= warm)) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;
            float2 pt = prefix_f2[t1];
            float2 pS = prefix_f2[sS];
            float2 pL = prefix_f2[sL];
            ds PT = ds_make(pt.x, pt.y);
            ds PS = ds_make(pS.x, pS.y);
            ds PL = ds_make(pL.x, pL.y);
            ds short_sum = ds_sub(PT, PS);
            ds long_sum  = ds_sub(PT, PL);
            ds savg_ds = ds_mul_f(short_sum, invS);
            ds lavg_ds = ds_mul_f(long_sum,  invL);
            float lavg = ds_to_float(lavg_ds);
            float num  = ds_to_float(ds_sub(savg_ds, lavg_ds));
            float v = 100.0f * num * __fdividef(1.0f, lavg);
            out_val = v;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

extern "C" __global__ void vosc_many_series_one_param_f32_ds_tm_coalesced(
    const float2* __restrict__ prefix_tm,
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm,
    int row_base)
{
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;
    const int t_global = row_base + blockIdx.y;
    if (t_global >= series_len) return;
    const float invS = __fdividef(1.0f, (float)short_period);
    const float invL = __fdividef(1.0f, (float)long_period);
    const int stride = num_series;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    const int t1 = t_global + 1;
    const int p_t1 = t1 * stride;
    while (s < num_series) {
        const int warm = first_valids[s] + long_period - 1;
        float out_val = VOSC_NAN;
        if (t_global >= warm) {
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_sS = sS * stride;
            const int p_sL = sL * stride;
            float2 pt = prefix_tm[p_t1 + s];
            float2 ps = prefix_tm[p_sS + s];
            float2 pl = prefix_tm[p_sL + s];
            ds PT = ds_make(pt.x, pt.y);
            ds PS = ds_make(ps.x, ps.y);
            ds PL = ds_make(pl.x, pl.y);
            ds short_sum = ds_sub(PT, PS);
            ds long_sum  = ds_sub(PT, PL);
            ds savg_ds = ds_mul_f(short_sum, invS);
            ds lavg_ds = ds_mul_f(long_sum,  invL);
            float lavg = ds_to_float(lavg_ds);
            float num  = ds_to_float(ds_sub(savg_ds, lavg_ds));
            float v = 100.0f * num * __fdividef(1.0f, lavg);
            out_val = v;
        }
        out_tm[t_global * stride + s] = out_val;
        s += step;
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

struct ds_f64 {
    double hi;
    double lo;
};
__device__ __forceinline__ ds_f64 ds_make_f64(double hi, double lo) { ds_f64 r{hi, lo}; return r; }
__device__ __forceinline__ ds_f64 ds_from_double_f64(double d) {
    double hi = (double)d;
    double lo = (double)(d - (double)hi);
    return ds_make_f64(hi, lo);
}
__device__ __forceinline__ ds_f64 ds_add_f64(ds_f64 a, ds_f64 b) {
    double s  = a.hi + b.hi;
    double bb = s - a.hi;
    double e  = (a.hi - (s - bb)) + (b.hi - bb);
    double t  = e + a.lo + b.lo;
    double hi = s + t;
    double lo = t - (hi - s);
    return ds_make_f64(hi, lo);
}
__device__ __forceinline__ ds_f64 ds_mul_f_f64(ds_f64 a, double k) {

    double p  = a.hi * k;
    double e  = fma(a.hi, k, -p) + a.lo * k;
    double hi = p + e;
    double lo = e - (hi - p);
    return ds_make_f64(hi, lo);
}
__device__ __forceinline__ double ds_to_float_f64(ds_f64 a) { return a.hi + a.lo; }
extern "C" __global__ void vosc_build_prefix_f64_ds(
    const double* __restrict__ data,
    int len,
    double2* __restrict__ prefix_f2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_f2[0] = make_double2(0.0, 0.0);
    double acc = 0.0;
    for (int i = 0; i < len; ++i) {
        acc += (double)data[i];
        double hi = (double)acc;
        double lo = (double)(acc - (double)hi);
        prefix_f2[i + 1] = make_double2(hi, lo);
    }
}
extern "C" __global__ void vosc_batch_prefix_f64(
    const double* __restrict__ prefix_sum,
    int len,
    int first_valid,
    const int* __restrict__ short_periods,
    const int* __restrict__ long_periods,
    int n_combos,
    double* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;

    const int warm = first_valid + L - 1;
    const int row_off = combo * len;

    const double inv_S = __ddiv_rn(1.0, (double)S);
    const double inv_L = __ddiv_rn(1.0, (double)L);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        double out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;

            ds_f64 PT = ds_from_double_f64(prefix_sum[t1]);
            ds_f64 PS = ds_from_double_f64(prefix_sum[sS]);
            ds_f64 PL = ds_from_double_f64(prefix_sum[sL]);


            ds_f64 short_sum = ds_sub(PT, PS);
            ds_f64 long_sum  = ds_sub(PT, PL);
            ds_f64 savg_ds = ds_mul_f_f64(short_sum, inv_S);
            ds_f64 lavg_ds = ds_mul_f_f64(long_sum,  inv_L);
            double lavg = ds_to_float_f64(lavg_ds);
            double num  = ds_to_float_f64(ds_sub(savg_ds, lavg_ds));
            // S5 CORRECTION -- ROUNDING COUNT. `vosc.rs:437` is
            // `100.0 * (savg - lavg) / lavg`, which Rust parses left to
            // right as `(100.0 * diff) / lavg`: ONE multiply and ONE
            // DIVIDE, two roundings. The original multiplied by a
            // separately-formed reciprocal -- `(100.0 * num) * (1/lavg)` --
            // which is THREE roundings and a different number. The two
            // reciprocals above stay as they are, because `vosc.rs:412-413`
            // genuinely does form `1.0 / period` once and multiply by it.
            double v = 100.0 * num / lavg;
            out_val = v;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}
extern "C" __global__ void vosc_many_series_one_param_f64(
    const double* __restrict__ prefix_tm,
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;

    const int warm = first_valids[series] + long_period - 1;
    const int stride = num_series;
    const double inv_S = 1.0 / static_cast<double>(short_period);
    const double inv_L = 1.0 / static_cast<double>(long_period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = t * stride + series;
        double out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_idx_t  = t1 * stride + series;
            const int p_idx_sS = sS * stride + series;
            const int p_idx_sL = sL * stride + series;
            const double short_sum = prefix_tm[p_idx_t] - prefix_tm[p_idx_sS];
            const double long_sum  = prefix_tm[p_idx_t] - prefix_tm[p_idx_sL];
            const double lavg = long_sum * inv_L;
            const double savg = short_sum * inv_S;
            const double v = 100.0 * (savg - lavg) / lavg;
            out_val = static_cast<double>(v);
        }
        out_tm[out_idx] = out_val;
        t += step;
    }
}
extern "C" __global__ void pack_double_to_float2_f64(
    const double* __restrict__ in, double2* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (i < n) {
        double d = in[i];
        double hi = (double)d;
        double lo = (double)(d - (double)hi);
        out[i] = make_double2(hi, lo);
        i += stride;
    }
}
extern "C" __global__ void vosc_batch_prefix_f64_ds(
    const double2* __restrict__ prefix_f2,
    int len,
    int first_valid,
    const int* __restrict__ short_periods,
    const int* __restrict__ long_periods,
    int n_combos,
    double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;
    const int warm = first_valid + L - 1;
    const int row_off = combo * len;
    const double invS = __ddiv_rn(1.0, (double)S);
    const double invL = __ddiv_rn(1.0, (double)L);
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        double out_val = VOSC_NAN;
        if (LIKELY(t >= warm)) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;
            double2 pt = prefix_f2[t1];
            double2 pS = prefix_f2[sS];
            double2 pL = prefix_f2[sL];
            ds_f64 PT = ds_make_f64(pt.x, pt.y);
            ds_f64 PS = ds_make_f64(pS.x, pS.y);
            ds_f64 PL = ds_make_f64(pL.x, pL.y);
            ds_f64 short_sum = ds_sub(PT, PS);
            ds_f64 long_sum  = ds_sub(PT, PL);
            ds_f64 savg_ds = ds_mul_f_f64(short_sum, invS);
            ds_f64 lavg_ds = ds_mul_f_f64(long_sum,  invL);
            double lavg = ds_to_float_f64(lavg_ds);
            double num  = ds_to_float_f64(ds_sub(savg_ds, lavg_ds));
            // S5 CORRECTION -- ROUNDING COUNT. `vosc.rs:437` is
            // `100.0 * (savg - lavg) / lavg`, which Rust parses left to
            // right as `(100.0 * diff) / lavg`: ONE multiply and ONE
            // DIVIDE, two roundings. The original multiplied by a
            // separately-formed reciprocal -- `(100.0 * num) * (1/lavg)` --
            // which is THREE roundings and a different number. The two
            // reciprocals above stay as they are, because `vosc.rs:412-413`
            // genuinely does form `1.0 / period` once and multiply by it.
            double v = 100.0 * num / lavg;
            out_val = v;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}
extern "C" __global__ void vosc_many_series_one_param_f64_ds_tm_coalesced(
    const double2* __restrict__ prefix_tm,
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm,
    int row_base)
{
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;
    const int t_global = row_base + blockIdx.y;
    if (t_global >= series_len) return;
    const double invS = __ddiv_rn(1.0, (double)short_period);
    const double invL = __ddiv_rn(1.0, (double)long_period);
    const int stride = num_series;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    const int t1 = t_global + 1;
    const int p_t1 = t1 * stride;
    while (s < num_series) {
        const int warm = first_valids[s] + long_period - 1;
        double out_val = VOSC_NAN;
        if (t_global >= warm) {
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_sS = sS * stride;
            const int p_sL = sL * stride;
            double2 pt = prefix_tm[p_t1 + s];
            double2 ps = prefix_tm[p_sS + s];
            double2 pl = prefix_tm[p_sL + s];
            ds_f64 PT = ds_make_f64(pt.x, pt.y);
            ds_f64 PS = ds_make_f64(ps.x, ps.y);
            ds_f64 PL = ds_make_f64(pl.x, pl.y);
            ds_f64 short_sum = ds_sub(PT, PS);
            ds_f64 long_sum  = ds_sub(PT, PL);
            ds_f64 savg_ds = ds_mul_f_f64(short_sum, invS);
            ds_f64 lavg_ds = ds_mul_f_f64(long_sum,  invL);
            double lavg = ds_to_float_f64(lavg_ds);
            double num  = ds_to_float_f64(ds_sub(savg_ds, lavg_ds));
            // S5 CORRECTION -- ROUNDING COUNT. `vosc.rs:437` is
            // `100.0 * (savg - lavg) / lavg`, which Rust parses left to
            // right as `(100.0 * diff) / lavg`: ONE multiply and ONE
            // DIVIDE, two roundings. The original multiplied by a
            // separately-formed reciprocal -- `(100.0 * num) * (1/lavg)` --
            // which is THREE roundings and a different number. The two
            // reciprocals above stay as they are, because `vosc.rs:412-413`
            // genuinely does form `1.0 / period` once and multiply by it.
            double v = 100.0 * num / lavg;
            out_val = v;
        }
        out_tm[t_global * stride + s] = out_val;
        s += step;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — vosc                                        (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/vosc.rs
 *   :405 vosc_scalar        <- the body reproduced here
 *   :327 vosc_with_kernel   warmup = first + long_period - 1
 *
 * PERIOD-INVARIANT. cpu_batch.rs:3032 reads "short_period" (2) and
 * "long_period" (5) and NEVER "period", so a sweep of [7,21,50,...] produces
 * one column repeated. Inventing a mapping from the swept int onto one of the
 * two named windows would compute something the CPU never computes.
 *
 * WHY NOT vosc_batch_prefix_f64. That entry point takes the crate prefix-sum
 * argument list; the f64 lane launches (volume, n, periods, n_combos,
 * first_valid, out). Also note pack_double_to_float2 at :166 in this file is
 * an f32 double-single packer left over from the f32 lane -- unrelated.
 *
 * SEQUENTIAL, one thread per column: both window sums are INCREMENTAL
 * (sum += new; sum -= old), which is a different rounding from a fresh
 * window sum at each bar. That is the whole reason this is not bar-parallel.
 *
 * NaN SEMANTICS: none needed. The CPU has no comparison chain here -- it
 * divides by lavg unguarded, so a zero long average yields +/-inf on both
 * sides, and a NaN input propagates identically.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_VOSC_SHORT 2
#define NEO_VOSC_LONG  5

extern "C" __global__
void vosc_neo_batch_f64(const double* __restrict__ data,
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
    (void)periods;  // PERIOD-INVARIANT

    const int short_period = NEO_VOSC_SHORT;
    const int long_period  = NEO_VOSC_LONG;

    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        long_period > len || (len - first_valid) < long_period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const double short_div = 1.0 / (double)short_period;
    const double long_div  = 1.0 / (double)long_period;

    const int start = first_valid;
    const int end_init = start + long_period;
    const int short_start = end_init - short_period;

    const int warm = end_init - 1;
    for (int i = 0; i < len && i < warm; ++i) o[i] = NEO_F64_NAN;

    double short_sum = 0.0, long_sum = 0.0;
    for (int i = start; i < end_init; ++i) {
        const double v = data[i];
        long_sum += v;
        if (i >= short_start) short_sum += v;
    }

    int idx = end_init - 1;
    double lavg = long_sum * long_div;
    double savg = short_sum * short_div;
    o[idx] = 100.0 * (savg - lavg) / lavg;

    int t_s = end_init - short_period;
    int t_l = start;

    for (int j = end_init; j < len; ++j) {
        const double x_new = data[j];

        short_sum += x_new;
        short_sum -= data[t_s];

        long_sum += x_new;
        long_sum -= data[t_l];

        t_s += 1;
        t_l += 1;
        idx += 1;

        lavg = long_sum * long_div;
        savg = short_sum * short_div;
        o[idx] = 100.0 * (savg - lavg) / lavg;
    }
}
