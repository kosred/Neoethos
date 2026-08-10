#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void apo_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ short_periods,
                   const float* __restrict__ short_alphas,
                   const int*   __restrict__ long_periods,
                   const float* __restrict__ long_alphas,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int  sp    = short_periods[combo];
    const int  lp    = long_periods[combo];
    if (sp <= 0 || lp <= 0 || sp >= lp) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const float a_s  = short_alphas[combo];
    const float a_l  = long_alphas[combo];
    const float oma_s= 1.0f - a_s;
    const float oma_l= 1.0f - a_l;

    const size_t base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);


    for (int i = threadIdx.x; i < first_valid; i += blockDim.x) {
        out[base + static_cast<size_t>(i)] = NAN;
    }

    if (threadIdx.x >= 32) return;

    const unsigned lane = static_cast<unsigned>(threadIdx.x);
    const unsigned mask = 0xffffffffu;

    float se_prev = prices[first_valid];
    float le_prev = se_prev;
    if (lane == 0) {
        out[base + static_cast<size_t>(first_valid)] = 0.0f;
    }

    int t0 = first_valid + 1;
    const int full_chunks = (series_len - t0) >> 5;
    for (int chunk = 0; chunk < full_chunks; ++chunk, t0 += 32) {
        const int t = t0 + static_cast<int>(lane);

        const float x = prices[t];
        float A_s = oma_s;
        float B_s = a_s * x;
        float A_l = oma_l;
        float B_l = a_l * x;

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_s_prev = __shfl_up_sync(mask, A_s, offset);
            const float B_s_prev = __shfl_up_sync(mask, B_s, offset);
            const float A_l_prev = __shfl_up_sync(mask, A_l, offset);
            const float B_l_prev = __shfl_up_sync(mask, B_l, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const float A_s_cur = A_s;
                const float B_s_cur = B_s;
                const float A_l_cur = A_l;
                const float B_l_cur = B_l;
                A_s = A_s_cur * A_s_prev;
                B_s = __fmaf_rn(A_s_cur, B_s_prev, B_s_cur);
                A_l = A_l_cur * A_l_prev;
                B_l = __fmaf_rn(A_l_cur, B_l_prev, B_l_cur);
            }
        }

        const float se = __fmaf_rn(A_s, se_prev, B_s);
        const float le = __fmaf_rn(A_l, le_prev, B_l);
        out[base + static_cast<size_t>(t)] = se - le;

        se_prev = __shfl_sync(mask, se, 31);
        le_prev = __shfl_sync(mask, le, 31);
    }

    if (t0 < series_len) {
        const int t = t0 + static_cast<int>(lane);
        float A_s = 1.0f;
        float B_s = 0.0f;
        float A_l = 1.0f;
        float B_l = 0.0f;
        if (t < series_len) {
            const float x = prices[t];
            A_s = oma_s;
            B_s = a_s * x;
            A_l = oma_l;
            B_l = a_l * x;
        }

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_s_prev = __shfl_up_sync(mask, A_s, offset);
            const float B_s_prev = __shfl_up_sync(mask, B_s, offset);
            const float A_l_prev = __shfl_up_sync(mask, A_l, offset);
            const float B_l_prev = __shfl_up_sync(mask, B_l, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const float A_s_cur = A_s;
                const float B_s_cur = B_s;
                const float A_l_cur = A_l;
                const float B_l_cur = B_l;
                A_s = A_s_cur * A_s_prev;
                B_s = __fmaf_rn(A_s_cur, B_s_prev, B_s_cur);
                A_l = A_l_cur * A_l_prev;
                B_l = __fmaf_rn(A_l_cur, B_l_prev, B_l_cur);
            }
        }

        const float se = __fmaf_rn(A_s, se_prev, B_s);
        const float le = __fmaf_rn(A_l, le_prev, B_l);
        if (t < series_len) {
            out[base + static_cast<size_t>(t)] = se - le;
        }
        const int last_lane = (series_len - t0) - 1;
        se_prev = __shfl_sync(mask, se, last_lane);
        le_prev = __shfl_sync(mask, le, last_lane);
    }
}


extern "C" __global__
void apo_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int short_period,
                                   float short_alpha,
                                   int long_period,
                                   float long_alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) return;
    if (short_period <= 0 || long_period <= 0 || short_period >= long_period) return;

    const int stride = num_series;
    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;

    const float a_s   = short_alpha;
    const float a_l   = long_alpha;
    const float oma_s = 1.0f - a_s;
    const float oma_l = 1.0f - a_l;


    for (int t = threadIdx.x; t < fv; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    if (threadIdx.x != 0) return;


    float se = prices_tm[fv * stride + series_idx];
    float le = se;
    out_tm[fv * stride + series_idx] = 0.0f;

    for (int t = fv + 1; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        se = a_s * x + oma_s * se;
        le = a_l * x + oma_l * le;
        out_tm[t * stride + series_idx] = se - le;
    }
}


// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/apo.rs
//   * `apo_prepare`  (:209) — first_valid = first non-NaN of the source series.
//   * `apo_with_kernel` (:278) — warmup prefix is `first`, NOT first + long.
//   * `apo_scalar`   (:311) — the arithmetic this reproduces.
//
// PERIOD-INVARIANT. `compute_apo_batch` (cpu_batch.rs:3374) reads
// `short_period` (default 10) and `long_period` (default 20) and NEVER reads
// `period`, so a period sweep produces identical rows on the CPU and must
// produce identical rows here. The defaults are baked in for the same reason
// `neoethos_tsi_batch_f64` bakes in 25/13.
//
// ROUNDING COUNT. The CPU line is
//     se = alpha_s * p0 + oma_s * se;
// — TWO multiplies and ONE add, three roundings, and it is NOT `mul_add`.
// Reproduced literally; `-fmad=false` on this translation unit is what stops
// nvcc contracting the multiply-add behind our back and silently producing a
// DIFFERENT number from the CPU.
//
// The CPU unrolls the loop two bars at a time (`while i + 1 < n`) plus a tail.
// Unrolling does not change the accumulation order, so a single ascending loop
// is bit-identical.
//
// Sequential: `se`/`le` carry across bars. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_apo() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

#define NEF_APO_SHORT 10
#define NEF_APO_LONG  20

extern "C" __global__
void neoethos_apo_f64(const double* __restrict__ prices,
                      int n,
                      const int* __restrict__ periods,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;  // PERIOD-INVARIANT: see the header.

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const double QNAN = nef_qnan_apo();

    if (n <= 0) return;
    if (first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    // warmup prefix = first_valid (apo.rs:281 `let warmup_period = first;`)
    for (int i = 0; i < first_valid; ++i) row[i] = QNAN;

    const double alpha_s = 2.0 / ((double)NEF_APO_SHORT + 1.0);
    const double alpha_l = 2.0 / ((double)NEF_APO_LONG + 1.0);
    const double oma_s = 1.0 - alpha_s;
    const double oma_l = 1.0 - alpha_l;

    double se = prices[first_valid];
    double le = se;
    row[first_valid] = 0.0;

    for (int i = first_valid + 1; i < n; ++i) {
        const double p = prices[i];
        se = alpha_s * p + oma_s * se;
        le = alpha_l * p + oma_l * le;
        row[i] = se - le;
    }
}


// ===========================================================================
// S1 f64 LANE  --  apo
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/apo.rs -- `apo_scalar` (:311), `apo_prepare` (:209), `apo_with_kernel` (:278)
//
// PERIOD-INVARIANT. `compute_apo_batch` (cpu_batch.rs:3357) reads
// `short_period` (default 10) and `long_period` (default 20) and NEVER reads
// `period`, so every row of a period sweep is byte-identical -- exactly as
// `tsi`/`obv` already are in this lane. The swept `periods[r]` is deliberately
// not consulted; consulting it would compute something the CPU never computes.
//
// ARITHMETIC ORDER: the CPU line is `se = alpha_s * p0 + oma_s * se` -- two
// multiplies and one add, THREE roundings, and NO `mul_add`. It is reproduced
// literally; an `fma` here would be one rounding and a different number. The
// CPU's two-at-a-time unroll is reproduced too: it does not change the order
// (bar i is folded before bar i+1 either way) but keeping it makes the tail
// (`if i < n`) match, so no bar is written twice or missed.
//
// WARMUP: `alloc_with_nan_prefix(len, first)` -- NaN strictly BEFORE
// first_valid, and `out[first] = 0.0`, not NaN. `apo_prepare` additionally
// forces `Kernel::Scalar` for `Auto` even when `nightly-avx` is on
// (apo.rs:238-246), so `apo_scalar` is the ONLY CPU answer on any host and
// there is no scalar/AVX disagreement to settle for this indicator.
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

extern "C" __global__ void neoethos_apo_batch_f64(
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

    // `ApoParams::default()` -- apo.rs:73-74.
    const int short_p = 10;
    const int long_p  = 20;
    (void)periods;

    // Every branch of `apo_prepare` that returns Err, in its order.
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (short_p == 0) || (long_p == 0) ||
        (short_p >= long_p) ||
        ((n - first_valid) < long_p);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const double alpha_s = 2.0 / ((double)short_p + 1.0);
    const double alpha_l = 2.0 / ((double)long_p + 1.0);
    const double oma_s = 1.0 - alpha_s;
    const double oma_l = 1.0 - alpha_l;

    for (int i = 0; i < first_valid; ++i) row[i] = neo_s1_qnan();

    double se = prices[first_valid];
    double le = se;
    row[first_valid] = 0.0;

    int i = first_valid + 1;
    while (i + 1 < n) {
        const double p0 = prices[i];
        se = alpha_s * p0 + oma_s * se;
        le = alpha_l * p0 + oma_l * le;
        row[i] = se - le;

        const double p1 = prices[i + 1];
        se = alpha_s * p1 + oma_s * se;
        le = alpha_l * p1 + oma_l * le;
        row[i + 1] = se - le;

        i += 2;
    }
    if (i < n) {
        const double p = prices[i];
        se = alpha_s * p + oma_s * se;
        le = alpha_l * p + oma_l * le;
        row[i] = se - le;
    }
}
