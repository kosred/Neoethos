#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef AO_NAN_F
#define AO_NAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


#include "../ds_float2.cuh"


__device__ __forceinline__ dsf load_dsf(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}

extern "C" __global__ void ao_build_prefix_dsf_serial_f32(
    const float* __restrict__ hl2,
    int len,
    int first_valid,
    float2* __restrict__ prefix_ds)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_ds[0] = make_float2(0.0f, 0.0f);
    dsf acc = ds_set(0.0f);
    for (int i = 0; i < len; ++i) {
        const float v = (i >= first_valid && !isnan(hl2[i])) ? hl2[i] : 0.0f;
        acc = ds_add(acc, ds_set(v));
        prefix_ds[i + 1] = make_float2(acc.hi, acc.lo);
    }
}


extern "C" __global__ void ao_batch_f32(const float2* __restrict__ prefix_ds,
                                         int len,
                                         int first_valid,
                                         const int* __restrict__ shorts,
                                         const int* __restrict__ longs,
                                         int n_combos,
                                         float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int s = shorts[combo];
    const int l = longs[combo];
    if (UNLIKELY(s <= 0 || l <= 0 || s >= l)) {

        const int base = combo * len;
        for (int t = 0; t < len; ++t) out[base + t] = AO_NAN_F;
        return;
    }

    const int warm = first_valid + l - 1;
    const int row_off = combo * len;


    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const float inv_s = 1.0f / (float)s;
    const float inv_l = 1.0f / (float)l;

    while (t < len) {
        float out_val = AO_NAN_F;
        if (t >= warm) {
            int start_s = t + 1 - s;
            int start_l = t + 1 - l;
            if (start_s < 0) start_s = 0;
            if (start_l < 0) start_l = 0;

            dsf head   = load_dsf(prefix_ds, t + 1);
            dsf tail_s = load_dsf(prefix_ds, start_s);
            dsf tail_l = load_dsf(prefix_ds, start_l);
            dsf sum_s = ds_sub(head, tail_s);
            dsf sum_l = ds_sub(head, tail_l);
            dsf ao_ds = ds_sub(ds_scale(sum_s, inv_s), ds_scale(sum_l, inv_l));
            out_val = ds_to_f(ao_ds);
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

extern "C" __global__ void ao_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;


    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p >= long_p)) {
        float* o = out_tm + series;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = AO_NAN_F;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* o = out_tm + series;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = AO_NAN_F;
        return;
    }

    const int warm = first_valid + long_p - 1;


    if (UNLIKELY(warm >= series_len)) {
        float* o = out_tm + series;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = AO_NAN_F;
        return;
    }


    {
        float* o = out_tm + series;
        for (int row = 0; row < warm; ++row, o += num_series) *o = AO_NAN_F;
    }


    dsf sum_s = ds_set(0.0f);
    dsf sum_l = ds_set(0.0f);

    const float* pl = prices_tm + (size_t)first_valid * (size_t)num_series + series;
    for (int k = 0; k < long_p; ++k) {
        const float v = *pl;
        sum_l = ds_add(sum_l, ds_set(v));
        if (k >= long_p - short_p) sum_s = ds_add(sum_s, ds_set(v));
        pl += num_series;
    }

    const float inv_s = 1.0f / (float)short_p;
    const float inv_l = 1.0f / (float)long_p;


    *(out_tm + (size_t)warm * (size_t)num_series + series) =
        ds_to_f(ds_sub(ds_scale(sum_s, inv_s), ds_scale(sum_l, inv_l)));


    const float* cur   = prices_tm + ((size_t)warm + 1) * (size_t)num_series + series;
    const float* old_s = prices_tm + ((size_t)first_valid + (long_p - short_p)) * (size_t)num_series + series;
    const float* old_l = prices_tm + ((size_t)first_valid) * (size_t)num_series + series;
    float*       dst   = out_tm   + ((size_t)warm + 1) * (size_t)num_series + series;

    for (int row = warm + 1; row < series_len; ++row) {
        const float c  = *cur;
        const float os = *old_s;
        const float ol = *old_l;

        sum_s = ds_add(sum_s, ds_set(c));
        sum_s = ds_sub(sum_s, ds_set(os));
        sum_l = ds_add(sum_l, ds_set(c));
        sum_l = ds_sub(sum_l, ds_set(ol));

        *dst = ds_to_f(ds_sub(ds_scale(sum_s, inv_s), ds_scale(sum_l, inv_l)));

        cur   += num_series;
        old_s += num_series;
        old_l += num_series;
        dst   += num_series;
    }
}

// ===========================================================================
// S3 f64 LANE — ao (Awesome Oscillator)
// ===========================================================================
// Reference: src/indicators/ao.rs
//   `ao_prepare` (:289) — first_valid + the four Err branches
//   `ao_with_kernel` (:318) — `alloc_with_nan_prefix(len, first + long - 1)`
//   `ao_scalar` (:367) — the arithmetic, including the 2x unroll
//
// PERIOD-INVARIANT. `compute_ao_batch` reads `short_period` (default 5) and
// `long_period` (default 34) and NEVER reads `period`, so a sweep over periods
// produces `n_combos` byte-identical rows. `(void)periods` below is that fact,
// not an oversight — the same contract `neoethos_tsi_batch_f64` documents.
//
// SOURCE. `compute_ao_batch` resolves `source.unwrap_or("hl2")`. The single
// price series this kernel receives MUST be hl2, not close; feeding close
// computes a different indicator that every length check would pass. That is
// declared upstream by `F64InputKind::Hl2Slice`.
//
// ROUNDING. `short_sum.mul_add(inv_s, -long_sum * inv_l)` is ONE fused
// multiply-add over a separately-rounded product — two roundings total. Written
// as `short_sum*inv_s - long_sum*inv_l` it would be three. `fma(short_sum,
// inv_s, -(long_sum * inv_l))` reproduces the CPU exactly.
//
// The CPU's 2x unrolled loop and its scalar tail perform the same operations in
// the same order on the same values, so one loop reproduces both.
// ===========================================================================

#define NEO_S3_AO_SHORT 5
#define NEO_S3_AO_LONG  34

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_ao_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;  // PERIOD-INVARIANT — see the header.

    double* __restrict__ row = out + (size_t)r * (size_t)n;

    const int shortp = NEO_S3_AO_SHORT;
    const int longp  = NEO_S3_AO_LONG;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (shortp == 0) || (longp == 0) ||
        (shortp >= longp) ||
        ((n - first_valid) < longp);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + longp - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();
    if (warm >= n) return;

    const double inv_s = 1.0 / (double)shortp;
    const double inv_l = 1.0 / (double)longp;

    // ao.rs:383-395 — long_sum over the first (long-1) bars from `first`,
    // short_sum over the (short-1) bars ending just before `warm`.
    double long_sum = 0.0;
    for (int i = 0; i < longp - 1; ++i) long_sum += data[first_valid + i];

    double short_sum = 0.0;
    for (int i = 0; i < shortp - 1; ++i) short_sum += data[first_valid + longp - shortp + i];

    int tail_long  = first_valid;
    int tail_short = first_valid + longp - shortp;

    for (int i = warm; i < n; ++i) {
        const double v = data[i];
        long_sum  += v;
        short_sum += v;
        row[i] = fma(short_sum, inv_s, -(long_sum * inv_l));
        long_sum  -= data[tail_long];
        short_sum -= data[tail_short];
        ++tail_long;
        ++tail_short;
    }
}
