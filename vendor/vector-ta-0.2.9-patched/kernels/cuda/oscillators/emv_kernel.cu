#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef EMV_NAN
#define EMV_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __device__ __forceinline__ void two_diff_f32(float a, float b, float &s, float &e) {
    s = a - b;
    float bb = s - a;
    e = (a - (s - bb)) - (b + bb);
}


extern "C" __global__ void emv_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ volume,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* row = out + (size_t)combo * series_len;

    if (UNLIKELY(series_len <= 0 || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) row[i] = EMV_NAN;
        return;
    }

    const int warm = first_valid + 1;


    for (int i = 0; i < warm && i < series_len; ++i) row[i] = EMV_NAN;


    const unsigned mask = __activemask();
    const int src_lane = __ffs(mask) - 1;


    float h0 = 0.0f, l0 = 0.0f;
    if ((threadIdx.x & 31) == src_lane) {
        h0 = high[first_valid];
        l0 = low[first_valid];
    }
    h0 = __shfl_sync(mask, h0, src_lane);
    l0 = __shfl_sync(mask, l0, src_lane);
    float last_mid = 0.5f * (h0 + l0);

    for (int i = warm; i < series_len; ++i) {

        float hf = 0.0f, lf = 0.0f, vf = 0.0f;
        if ((threadIdx.x & 31) == src_lane) {
            hf = high[i];
            lf = low[i];
            vf = volume[i];
        }
        hf = __shfl_sync(mask, hf, src_lane);
        lf = __shfl_sync(mask, lf, src_lane);
        vf = __shfl_sync(mask, vf, src_lane);

        if (UNLIKELY(isnan(hf) || isnan(lf) || isnan(vf))) {
            row[i] = EMV_NAN;
            continue;
        }

        const float range = hf - lf;
        const float current_mid = 0.5f * (hf + lf);

        if (UNLIKELY(range == 0.0f)) {
            row[i] = EMV_NAN;
            last_mid = current_mid;
            continue;
        }


        float s, e;
        two_diff_f32(current_mid, last_mid, s, e);


        const float k = range * (10000.0f / vf);


        row[i] = fmaf(s, k, e * k);

        last_mid = current_mid;
    }
}


extern "C" __global__ void emv_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int fv = first_valids[s];
    if (UNLIKELY(series_len <= 0 || fv < 0 || fv >= series_len)) {
        float* o = out_tm + s;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = EMV_NAN;
        return;
    }

    const int warm = fv + 1;

    {
        float* o = out_tm + s;
        for (int r = 0; r < warm && r < series_len; ++r, o += num_series) *o = EMV_NAN;
    }


    const size_t idx0 = (size_t)fv * num_series + s;
    float last_mid = 0.5f * (high_tm[idx0] + low_tm[idx0]);

    for (int r = warm; r < series_len; ++r) {
        const size_t idx = (size_t)r * num_series + s;
        const float hf = high_tm[idx];
        const float lf = low_tm[idx];
        const float vf = volume_tm[idx];
        float* out_elem = out_tm + idx;

        if (UNLIKELY(isnan(hf) || isnan(lf) || isnan(vf))) {
            *out_elem = EMV_NAN;
            continue;
        }
        const float current_mid = 0.5f * (hf + lf);
        const float range = hf - lf;
        if (UNLIKELY(range == 0.0f)) {
            *out_elem = EMV_NAN;
            last_mid = current_mid;
            continue;
        }

        float s_hi, s_lo;
        two_diff_f32(current_mid, last_mid, s_hi, s_lo);

        const float k = range * (10000.0f / vf);
        *out_elem = fmaf(s_hi, k, s_lo * k);

        last_mid = current_mid;
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/emv.rs
//   * emv_with_kernel (:219) — first_valid is the first index at which HIGH,
//     LOW and VOLUME are ALL non-NaN simultaneously, and the warmup prefix is
//     first + 1 (:235) because the first bar has no previous midpoint.
//   * emv_scalar (:335) — the arithmetic reproduced below.
//
// PERIOD-INVARIANT. compute_emv_batch (cpu_batch.rs:2825) takes |_params| — it
// reads no parameter at all, so every row of a period sweep is identical.
//
// INPUTS ARE (high, low, VOLUME) — three series, and the third is volume, not
// close. Feeding close here computes a different indicator while passing every
// length check on the way in.
//
// NaN and zero-range branches are EXACT tests, not tolerances: the CPU writes
// NaN when any of h/l/v is NaN, and NaN again when range == 0.0 exactly while
// still ADVANCING last_mid. There is no epsilon here to re-derive; inventing
// one would change which bars emit a value.
//
// Sequential: last_mid carries across bars. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_emv() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__
void neoethos_emv_f64(const double* __restrict__ high,
                      const double* __restrict__ low,
                      const double* __restrict__ volume,
                      int n,
                      const int* __restrict__ periods,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;  // PERIOD-INVARIANT: see the header.

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const double QNAN = nef_qnan_emv();

    if (first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    {
        const int warm = (first_valid + 1) < n ? (first_valid + 1) : n;
        for (int i = 0; i < warm; ++i) row[i] = QNAN;
        for (int i = warm; i < n; ++i) row[i] = QNAN;
    }

    double last_mid = 0.5 * (high[first_valid] + low[first_valid]);

    for (int i = first_valid + 1; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double v = volume[i];

        if (isnan(h) || isnan(l) || isnan(v)) { row[i] = QNAN; continue; }

        const double current_mid = 0.5 * (h + l);
        const double range = h - l;
        if (range == 0.0) { row[i] = QNAN; last_mid = current_mid; continue; }

        const double dmid = current_mid - last_mid;
        row[i] = dmid * range * 10000.0 / v;
        last_mid = current_mid;
    }
}



// ===========================================================================
// S1 f64 LANE  --  emv
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/emv.rs -- `emv_scalar` (:335), `emv_with_kernel` (:195)
//
// PERIOD-INVARIANT. `compute_emv_batch` (cpu_batch.rs:2818) takes
// `|_params|`; emv has no period parameter.
//
// INPUT SHAPE: high, low, VOLUME -- NOT high/low/close. `emv_with_kernel`
// destructures close as `_close` (emv.rs:196) and never reads it, and
// first_valid is the first index at which HIGH, LOW and VOLUME are
// simultaneously non-NaN (emv.rs:219); close is never scanned. Handing this
// kernel an (high, low, close) triple would compute a different indicator AND
// adopt a different first-valid, which is why the lane declares a distinct
// `HighLowVolume` input kind for it rather than reusing `Hlc`.
//
// ARITHMETIC ORDER: `dmid * range * 10_000.0 / v` is LEFT TO RIGHT --
// ((dmid*range)*10000)/v, three roundings. Regrouping it as
// `dmid * (range * 10000.0 / v)` would be a different number. The 10_000.0 is
// an exact scale factor, not an epsilon, and does not change with precision.
//
// WARMUP: `alloc_with_nan_prefix(len, first + 1)` -- one bar LATER than the
// usual `first`, because the first output needs a previous midpoint. The two
// in-loop NaN emissions are reproduced exactly: a bar with any NaN input, and
// a bar whose range is EXACTLY zero. The CPU tests `range == 0.0`, not
// `|range| < eps`; substituting an epsilon would change WHICH bars are
// emitted, not merely their value.
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

extern "C" __global__ void neoethos_emv_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
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

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < 2);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    // `alloc_with_nan_prefix(len, first + 1)`.
    const int warm = first_valid + 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();

    double last_mid = 0.5 * (high[first_valid] + low[first_valid]);

    for (int i = warm; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double v = volume[i];

        if (neo_s1_isnan(h) || neo_s1_isnan(l) || neo_s1_isnan(v)) {
            row[i] = neo_s1_qnan();
            continue;
        }

        const double current_mid = 0.5 * (h + l);
        const double range = h - l;
        if (range == 0.0) {
            row[i] = neo_s1_qnan();
            last_mid = current_mid;
            continue;
        }

        const double dmid = current_mid - last_mid;
        row[i] = dmid * range * 10000.0 / v;
        last_mid = current_mid;
    }
}
