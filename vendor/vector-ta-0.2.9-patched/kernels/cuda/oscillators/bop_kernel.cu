#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef BOP_NAN_F
#define BOP_NAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __forceinline__ __device__ float bop_core(float o, float h, float l, float c) {
    const float den = h - l;
    return (den <= 0.0f) ? 0.0f : (c - o) / den;
}


extern "C" __global__ void bop_batch_f32(const float* __restrict__ open,
                                         const float* __restrict__ high,
                                         const float* __restrict__ low,
                                         const float* __restrict__ close,
                                         int len,
                                         int first_valid,
                                         float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (UNLIKELY(combo > 0)) return;

    constexpr int ILP = 8;

    const int tid   = threadIdx.x;
    const int bdim  = blockDim.x;
    const int gdim  = gridDim.x;

    int base = blockIdx.x * bdim * ILP;
    const int step = gdim * bdim * ILP;

    for (; base < len; base += step) {

        #pragma unroll
        for (int k = 0; k < ILP; ++k) {
            const int t = base + tid + k * bdim;
            if (t >= len) continue;

            if (LIKELY(t >= first_valid)) {
                const float o = open[t];
                const float h = high[t];
                const float l = low[t];
                const float c = close[t];
                out[t] = bop_core(o, h, l, c);
            } else {
                out[t] = BOP_NAN_F;
            }
        }
    }
}


extern "C" __global__ void bop_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int fv = first_valids[s];
    if (UNLIKELY(fv < 0 || fv >= series_len)) {

        float* o = out_tm + s;
        for (int t = 0; t < series_len; ++t, o += num_series) { *o = BOP_NAN_F; }
        return;
    }


    {
        float* o = out_tm + s;
        for (int t = 0; t < fv; ++t, o += num_series) { *o = BOP_NAN_F; }
    }


    const float* po = open_tm  + (size_t)fv * num_series + s;
    const float* ph = high_tm  + (size_t)fv * num_series + s;
    const float* pl = low_tm   + (size_t)fv * num_series + s;
    const float* pc = close_tm + (size_t)fv * num_series + s;
    float*       pd = out_tm   + (size_t)fv * num_series + s;


    #pragma unroll 4
    for (int t = fv; t < series_len; ++t) {
        const float v = bop_core(*po, *ph, *pl, *pc);
        *pd = v;

        po += num_series; ph += num_series; pl += num_series; pc += num_series; pd += num_series;
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `bop.rs::bop_scalar_from` (l.263) —
//   `denom = h - l; out = if denom <= 0.0 { 0.0 } else { (c - o) / denom }`
// PERIOD-INVARIANT: bop takes no period, so every row of a sweep is identical.
// first_valid is the first bar at which open/high/low/close are ALL non-NaN
// (`bop_scalar`, l.299).
//
// f32 -> f64 audit: pointers and locals widened; `0.0f` -> `0.0`;
// `BOP_NAN_F` (an `__int_as_float` f32 bit pattern) -> `__longlong_as_double`
// of the f64 quiet-NaN pattern. No math intrinsics, no epsilon, and the only
// comparison is `den <= 0.0`, which is FALSE for a NaN den and therefore takes
// the divide branch and propagates NaN — the same thing the CPU `if` does. No
// fmax/fmin substitution is needed here because there is no max/min.
// ---------------------------------------------------------------------------

#ifndef BOP_NAN_D
#define BOP_NAN_D (__longlong_as_double(0x7ff8000000000000ULL))
#endif

static __forceinline__ __device__ double bop_core_f64(double o, double h, double l, double c) {
    const double den = h - l;
    return (den <= 0.0) ? 0.0 : (c - o) / den;
}

extern "C" __global__ void bop_batch_f64(const double* __restrict__ open,
                                         const double* __restrict__ high,
                                         const double* __restrict__ low,
                                         const double* __restrict__ close,
                                         int n,
                                         const int* __restrict__ periods,
                                         int n_combos,
                                         int first_valid,
                                         double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;
    (void)periods;   // period-invariant, see above

    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int t = 0; t < n; ++t) {
        row[t] = (t < first_valid) ? BOP_NAN_D : bop_core_f64(open[t], high[t], low[t], close[t]);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — bop (balance of power)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/bop.rs:263 `bop_scalar_from`, entered from
 *             `bop_with_kernel` (:180) whose `first` is at :209.
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:2765 `expect_value_output`).
 *
 * PERIOD-INVARIANT AND PARAMETERLESS: `compute_bop_batch` builds
 * `BopParams::default()` inside a `|_params|` closure (cpu_batch.rs:2811), so
 * every row of a sweep is byte-identical.
 *
 * FIRST-VALID: `!is_nan` on open, high, low and close SIMULTANEOUSLY
 * (bop.rs:209-211). Deliberately `!is_nan` and NOT `is_finite` — an infinite
 * bar is ACCEPTED by the CPU here, unlike `accumulation_swing_index`, which
 * scans the same four series with `is_finite`. Registered as
 * `F64FirstValidRule::Ohlc4AllNonNan`; folding the two into one rule would
 * shift one of them on any frame carrying an infinity.
 *
 * WARMUP: `alloc_with_nan_prefix(len, first)` (:218) — NaN strictly before
 * `first`, and NO per-bar validity test after it. A NaN high after `first`
 * therefore propagates into the output as NaN through the arithmetic itself,
 * which is what the CPU does; there is no reset and no skip.
 *
 * `denom <= 0.0` is the CPU's exact guard (:284): a zero-or-inverted range
 * emits 0.0. It is a BRANCH on the sign, not an epsilon, so no f64-sized
 * tolerance is introduced. Note that the comparison is false when `denom` is
 * NaN, so a NaN bar takes the divide arm and yields NaN — matching the CPU's
 * `if denom <= 0.0 { 0.0 } else { .. }` exactly.
 *
 * BAR-PARALLEL IN PRINCIPLE, sequential here: the lane launches one thread per
 * combo column, and bop carries no state, so this loop is a plain per-bar map
 * with no accumulation order to preserve.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void bop_neo_batch_f64(const double* __restrict__ open,
                       const double* __restrict__ high,
                       const double* __restrict__ low,
                       const double* __restrict__ close,
                       int series_len,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    if (first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    for (int i = 0; i < first_valid; ++i) o[i] = NEO_F64_NAN;

    for (int i = first_valid; i < len; ++i) {
        const double denom = high[i] - low[i];
        o[i] = (denom <= 0.0) ? 0.0 : (close[i] - open[i]) / denom;
    }
}
