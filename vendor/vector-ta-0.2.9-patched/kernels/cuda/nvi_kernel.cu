#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void nvi_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    if (len <= 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;
    const float nan_f = CUDART_NAN_F;
    for (int i = 0; i < fv && i < len; ++i) out[i] = nan_f;
    if (fv >= len) return;

    out[fv] = 1000.0f;
    if (fv + 1 >= len) return;

    double nvi = 1000.0;
    double prev_close = (double)close[fv];
    double prev_volume = (double)volume[fv];
    for (int i = fv + 1; i < len; ++i) {
        const double current_close = (double)close[i];
        const double current_volume = (double)volume[i];
        if (current_volume < prev_volume && prev_close != 0.0) {
            double candidate = nvi;
            candidate += (current_close - prev_close) / prev_close * candidate;
            if (isfinite(candidate)) nvi = candidate;
        }
        out[i] = (float)nvi;
        prev_close = current_close;
        prev_volume = current_volume;
    }
}


extern "C" __global__ void nvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    if (rows <= 0 || cols <= 0) return;
    const float nan_f = CUDART_NAN_F;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s] < 0 ? 0 : first_valids[s];


        if (fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
            continue;
        }


        for (int t = 0; t < fv; ++t) out_tm[t * cols + s] = nan_f;


        double nvi = 1000.0;
        out_tm[fv * cols + s] = (float)nvi;
        if (fv + 1 >= rows) continue;

        double prev_close  = (double)close_tm[fv * cols + s];
        double prev_volume = (double)volume_tm[fv * cols + s];

        for (int t = fv + 1; t < rows; ++t) {
            const double c = (double)close_tm[t * cols + s];
            const double v = (double)volume_tm[t * cols + s];

            if (v < prev_volume && prev_close != 0.0) {
                double candidate = nvi;
                candidate += (c - prev_close) / prev_close * candidate;
                if (isfinite(candidate)) nvi = candidate;
            }
            out_tm[t * cols + s] = (float)nvi;
            prev_close  = c;
            prev_volume = v;
        }
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/nvi.rs
//   * nvi_with_kernel (:216) — first_valid is the first index at which CLOSE and
//     VOLUME are both non-NaN, and the warmup prefix is exactly `first` (:236),
//     because out[first] is the 1000.0 seed.
//   * nvi_scalar (:327) — the arithmetic reproduced below.
//
// PERIOD-INVARIANT. compute_nvi_batch (cpu_batch.rs:3944) takes |_params|.
//
// COMPARISON SEMANTICS, deliberately NOT fmax/fmin. The CPU gate is a bare
//     if v < prev_volume { ... }
// so a NaN volume makes the comparison FALSE and nvi_val is carried forward
// unchanged. Rewriting this as fmin would change which bars update. Rule 4 is
// "match the CPU", and here the CPU is a raw comparison — the fmax/fmin
// rewrite applies where the CPU itself calls f64::max, not everywhere a
// comparison appears.
//
// Sequential: nvi_val, prev_close and prev_volume carry across bars.
// One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_nvi() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__
void neoethos_nvi_f64(const double* __restrict__ close,
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
    const double QNAN = nef_qnan_nvi();

    if (first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = QNAN;
    for (int i = first_valid; i < n; ++i) row[i] = QNAN;

    double nvi_val = 1000.0;
    row[first_valid] = nvi_val;

    if (first_valid + 1 >= n) return;

    double prev_close = close[first_valid];
    double prev_volume = volume[first_valid];

    for (int i = first_valid + 1; i < n; ++i) {
        const double c = close[i];
        const double v = volume[i];

        if (v < prev_volume && prev_close != 0.0) {
            double candidate = nvi_val;
            candidate += (c - prev_close) / prev_close * candidate;
            if (isfinite(candidate)) nvi_val = candidate;
        }

        row[i] = nvi_val;

        prev_close = c;
        prev_volume = v;
    }
}



// ===========================================================================
// S1 f64 LANE  --  nvi
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/nvi.rs -- `nvi_scalar` (:327), `nvi_with_kernel` (:198)
//
// PERIOD-INVARIANT. `compute_nvi_batch` (cpu_batch.rs:3937) takes
// `|_params|` -- nvi has no period parameter at all -- so every row of a sweep
// is byte-identical, as with `obv`.
//
// `nvi_with_kernel` collapses EVERY `Kernel` variant to `Kernel::Scalar`
// (nvi.rs:237-246), so `nvi_scalar` is the only CPU answer on any host and
// there is no scalar/AVX disagreement to settle for this indicator.
//
// ARITHMETIC ORDER: `pct = (c - prev_close) / prev_close`, then
// `nvi_val += nvi_val * pct` -- a multiply and an add, TWO roundings, no
// `mul_add`. Reproduced literally.
//
// WARMUP: `alloc_with_nan_prefix(len, first)` then `out[first] = 1000.0`.
// first_valid is the first index at which close AND volume are both non-NaN
// (nvi.rs:219-222) -- the common `AllInputsNonNan` rule over a (close, volume)
// pair.
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

extern "C" __global__ void neoethos_nvi_batch_f64(
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

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = neo_s1_qnan();

    double nvi_val = 1000.0;
    row[first_valid] = nvi_val;

    double prev_close = close[first_valid];
    double prev_volume = volume[first_valid];

    for (int i = first_valid + 1; i < n; ++i) {
        const double c = close[i];
        const double v = volume[i];
        if (v < prev_volume && prev_close != 0.0) {
            double candidate = nvi_val;
            candidate += (c - prev_close) / prev_close * candidate;
            if (isfinite(candidate)) nvi_val = candidate;
        }
        row[i] = nvi_val;
        prev_close = c;
        prev_volume = v;
    }
}
