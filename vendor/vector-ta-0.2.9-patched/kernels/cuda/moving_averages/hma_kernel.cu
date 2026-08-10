#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif


#ifndef HMA_ASSUME_OUT_PREFILLED
#define HMA_ASSUME_OUT_PREFILLED 0
#endif


#ifndef HMA_RING_IN_SHARED
#define HMA_RING_IN_SHARED 1
#endif


#include <cuda_runtime.h>
#include <math.h>

#ifndef HMA_NAN
#define HMA_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ int clamp_positive(int v) { return v > 0 ? v : 0; }

extern "C" __global__
void hma_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   int max_sqrt_len,
                   float* __restrict__ x_buf,
                   float* __restrict__ out) {

#if HMA_RING_IN_SHARED
    extern __shared__ float sh_ring[];
#endif


    const int stride = blockDim.x * gridDim.x;
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x; combo < n_combos; combo += stride) {

        const int base = combo * series_len;

        const int period = periods[combo];
        const int half   = period >> 1;
#if !HMA_ASSUME_OUT_PREFILLED
        if (period < 2 || half < 1) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }
#else
        if (period < 2 || half < 1) { continue; }
#endif

        int sqrt_len = (int)sqrtf((float)period);
        if (sqrt_len < 1) sqrt_len = 1;
#if !HMA_ASSUME_OUT_PREFILLED
        if (sqrt_len > max_sqrt_len) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }
#else
        if (sqrt_len > max_sqrt_len) { continue; }
#endif

#if !HMA_ASSUME_OUT_PREFILLED
        if ((unsigned)first_valid >= (unsigned)series_len) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }
#else
        if ((unsigned)first_valid >= (unsigned)series_len) { continue; }
#endif

        const int tail_len = series_len - first_valid;
#if !HMA_ASSUME_OUT_PREFILLED
        if (tail_len < period + sqrt_len - 1) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }


        int warmup_end = first_valid + period + sqrt_len - 2;
        if (warmup_end > series_len) warmup_end = series_len;
        for (int i = 0; i < warmup_end; ++i) { out[base + i] = HMA_NAN; }
#else
        if (tail_len < period + sqrt_len - 1) { continue; }
#endif


        const float f_half   = (float)half;
        const float f_full   = (float)period;
        const float f_sqrt   = (float)sqrt_len;

        const float ws_half  = 0.5f * f_half * (f_half + 1.0f);
        const float ws_full  = 0.5f * f_full * (f_full + 1.0f);
        const float ws_sqrt  = 0.5f * f_sqrt * (f_sqrt + 1.0f);

        const float inv_ws_half = 1.0f / ws_half;
        const float inv_ws_full = 1.0f / ws_full;
        const float inv_ws_sqrt = 1.0f / ws_sqrt;


        float sum_half = 0.0f, wsum_half = 0.0f;
        float sum_full = 0.0f, wsum_full = 0.0f;


        float sum_x = 0.0f, wsum_x = 0.0f;
        int   ring_head = 0;
        int   ring_count = 0;


#if HMA_RING_IN_SHARED
        float* ring = sh_ring + threadIdx.x * max_sqrt_len;
#else
        float* ring = x_buf + combo * max_sqrt_len;
#endif


        for (int j = 0; j < tail_len; ++j) {
            const int idx = first_valid + j;

            const float val = prices[idx];


            if (j < period) {
                const float jf = (float)(j + 1);
                wsum_full = fmaf(jf, val, wsum_full);
                sum_full  += val;
            } else {
                const float old = prices[idx - period];
                const float prev_sum = sum_full;
                sum_full = prev_sum + val - old;
                wsum_full = fmaf((float)period, val, wsum_full - prev_sum);
            }


            if (j < half) {
                const float jf = (float)(j + 1);
                wsum_half = fmaf(jf, val, wsum_half);
                sum_half  += val;
            } else {
                const float old = prices[idx - half];
                const float prev_sum = sum_half;
                sum_half = prev_sum + val - old;
                wsum_half = fmaf((float)half, val, wsum_half - prev_sum);
            }


            if (j + 1 < period) { continue; }

            const float wma_full = wsum_full * inv_ws_full;
            const float wma_half = wsum_half * inv_ws_half;
            const float x_val    = 2.0f * wma_half - wma_full;

            if (ring_count < sqrt_len) {
                ring[ring_count] = x_val;
                const float rc1 = (float)(ring_count + 1);
                wsum_x = fmaf(rc1, x_val, wsum_x);
                sum_x  += x_val;
                ++ring_count;

                if (ring_count == sqrt_len) {
                    out[base + idx] = wsum_x * inv_ws_sqrt;
                }
            } else {
                const float old_x = ring[ring_head];
                ring[ring_head] = x_val;
                ++ring_head; if (ring_head == sqrt_len) ring_head = 0;

                const float prev_sum = sum_x;
                sum_x   = prev_sum + x_val - old_x;
                wsum_x  = fmaf((float)sqrt_len, x_val, wsum_x - prev_sum);

                out[base + idx] = wsum_x * inv_ws_sqrt;
            }
        }
    }
}

extern "C" __global__
void hma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int num_series,
                                   int series_len,
                                   int period,
                                   int max_sqrt_len,
                                   float* __restrict__ x_buf,
                                   float* __restrict__ out_tm) {

#if HMA_RING_IN_SHARED
    extern __shared__ float sh_ring[];
#endif

    if (period < 2) return;
    const int half = period >> 1;
    if (half < 1) return;

    int sqrt_len = (int)sqrtf((float)period);
    if (sqrt_len < 1) sqrt_len = 1;
    if (sqrt_len > max_sqrt_len) return;

    const float f_half   = (float)half;
    const float f_full   = (float)period;
    const float f_sqrt   = (float)sqrt_len;

    const float ws_half  = 0.5f * f_half * (f_half + 1.0f);
    const float ws_full  = 0.5f * f_full * (f_full + 1.0f);
    const float ws_sqrt  = 0.5f * f_sqrt * (f_sqrt + 1.0f);

    const float inv_ws_half = 1.0f / ws_half;
    const float inv_ws_full = 1.0f / ws_full;
    const float inv_ws_sqrt = 1.0f / ws_sqrt;

    const int stride = blockDim.x * gridDim.x;

    for (int series = blockIdx.x * blockDim.x + threadIdx.x; series < num_series; series += stride) {

        const int first_valid = first_valids[series];
#if !HMA_ASSUME_OUT_PREFILLED
        if ((unsigned)first_valid >= (unsigned)series_len) {
            for (int row = 0; row < series_len; ++row) {
                out_tm[row * num_series + series] = HMA_NAN;
            }
            continue;
        }
#else
        if ((unsigned)first_valid >= (unsigned)series_len) { continue; }
#endif

        const int tail_len = series_len - first_valid;
#if !HMA_ASSUME_OUT_PREFILLED
        if (tail_len < period + sqrt_len - 1) {
            for (int row = 0; row < series_len; ++row) {
                out_tm[row * num_series + series] = HMA_NAN;
            }
            continue;
        }

        int warmup_end = first_valid + period + sqrt_len - 2;
        if (warmup_end > series_len) warmup_end = series_len;
        for (int row = 0; row < warmup_end; ++row) {
            out_tm[row * num_series + series] = HMA_NAN;
        }
#else
        if (tail_len < period + sqrt_len - 1) { continue; }
#endif


        float sum_half = 0.0f, wsum_half = 0.0f;
        float sum_full = 0.0f, wsum_full = 0.0f;

        float sum_x = 0.0f, wsum_x = 0.0f;
        int   ring_head = 0;
        int   ring_count = 0;

#if HMA_RING_IN_SHARED
        float* ring = sh_ring + threadIdx.x * max_sqrt_len;
#else
        float* ring = x_buf + series * max_sqrt_len;
#endif


        for (int j = 0; j < tail_len; ++j) {
            const int row = first_valid + j;
            const int a   = row * num_series + series;
            const float val = prices_tm[a];

            if (j < period) {
                const float jf = (float)(j + 1);
                wsum_full = fmaf(jf, val, wsum_full);
                sum_full  += val;
            } else {
                const float old = prices_tm[(row - period) * num_series + series];
                const float prev_sum = sum_full;
                sum_full = prev_sum + val - old;
                wsum_full = fmaf(f_full, val, wsum_full - prev_sum);
            }

            if (j < half) {
                const float jf = (float)(j + 1);
                wsum_half = fmaf(jf, val, wsum_half);
                sum_half  += val;
            } else {
                const float old = prices_tm[(row - half) * num_series + series];
                const float prev_sum = sum_half;
                sum_half = prev_sum + val - old;
                wsum_half = fmaf(f_half, val, wsum_half - prev_sum);
            }

            if (j + 1 < period) { continue; }

            const float wma_full = wsum_full * inv_ws_full;
            const float wma_half = wsum_half * inv_ws_half;
            const float x_val    = 2.0f * wma_half - wma_full;

            if (ring_count < sqrt_len) {
                ring[ring_count] = x_val;
                const float rc1 = (float)(ring_count + 1);
                wsum_x = fmaf(rc1, x_val, wsum_x);
                sum_x  += x_val;
                ++ring_count;

                if (ring_count == sqrt_len) {
                    out_tm[a] = wsum_x * inv_ws_sqrt;
                }
            } else {
                const float old_x = ring[ring_head];
                ring[ring_head] = x_val;
                ++ring_head; if (ring_head == sqrt_len) ring_head = 0;

                const float prev_sum = sum_x;
                sum_x  = prev_sum + x_val - old_x;
                wsum_x = fmaf(f_sqrt, x_val, wsum_x - prev_sum);

                out_tm[a] = wsum_x * inv_ws_sqrt;
            }
        }
    }
}


// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/moving_averages/hma.rs
//   * hma_with_kernel (:202) — first_valid = first non-NaN of the source;
//     first_out = first + period + sq - 2; the warmup prefix is first_out.
//   * hma_scalar_period5 (:346) — the period == 5 CLOSED FORM. It is a
//     different expression from the general path, not an optimisation of it, so
//     it is reproduced rather than folded in.
//   * hma_scalar (:368) — the general rolling-WMA path reproduced below.
//
// sq = floor(sqrt(period)) computed in f64 exactly as the CPU does
// ((period as f64).sqrt().floor() as usize) — an integer isqrt would differ on
// the boundary for a period whose square root rounds up.
//
// ROUNDING COUNT. Every weighted-sum update on the CPU is ONE mul_add:
//     ws_full_acc = period_f.mul_add(v, ws_full_acc - prev_f);
// — the subtract rounds, then a single fused multiply-add. Reproduced as
//     ws_full_acc = fma(period_f, v, ws_full_acc - prev_f);
// -fmad=false on this translation unit forbids nvcc from contracting any OTHER
// multiply-add, so the fused steps are exactly the ones the CPU fuses.
//
// Sequential: three rolling accumulators and an sq-deep ring carry across bars.
// One thread per column.
//
// RING BOUND. x_buf is sq = floor(sqrt(period)) deep and lives in per-thread
// local memory, so it is sized at COMPILE time: NEF_HMA_MAX_SQ = 64 admits every
// period up to 4224 (floor(sqrt(4224)) = 64). The host refuses a larger period
// BY NAME rather than truncating the window — see F64Kernel::max_period.
// =============================================================================

__device__ __forceinline__ double nef_qnan_hma() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

#define NEF_HMA_MAX_SQ 64

extern "C" __global__
void neoethos_hma_f64(const double* __restrict__ prices,
                      int n,
                      const int* __restrict__ periods,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const double QNAN = nef_qnan_hma();

    const int period = periods[r];
    if (period < 2 || first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    const int half = period / 2;
    const int sq = (int)floor(sqrt((double)period));
    if (half == 0 || sq == 0 || sq > NEF_HMA_MAX_SQ) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    const int first = first_valid;

    if (period == 5) {
        // hma_scalar_period5: first_out = first + 5, and a bar whose data[i-2]
        // is NaN emits NaN. inv45 is a multiply by 1/45 on the CPU too.
        const int first_out = first + 5;
        const int warm = first_out < n ? first_out : n;
        for (int i = 0; i < warm; ++i) row[i] = QNAN;
        const double inv45 = 1.0 / 45.0;
        for (int i = first_out; i < n; ++i) {
            const double d2 = prices[i - 2];
            if (isnan(d2)) { row[i] = QNAN; continue; }
            row[i] = (30.0 * prices[i] + 27.0 * prices[i - 1]
                      - 7.0 * prices[i - 3]
                      - 4.0 * prices[i - 4]
                      - prices[i - 5]) * inv45;
        }
        return;
    }

    const int first_out = first + period + sq - 2;
    {
        const int warm = first_out < n ? first_out : n;
        for (int i = 0; i < warm; ++i) row[i] = QNAN;
    }
    if (first_out >= n) return;

    const double ws_half  = (double)((long long)half * (long long)(half + 1) / 2);
    const double ws_full  = (double)((long long)period * (long long)(period + 1) / 2);
    const double ws_sqrt  = (double)((long long)sq * (long long)(sq + 1) / 2);
    const double half_f   = (double)half;
    const double period_f = (double)period;
    const double sq_f     = (double)sq;

    double s_half = 0.0, ws_half_acc = 0.0;
    double s_full = 0.0, ws_full_acc = 0.0;
    double wma_half = QNAN, wma_full = QNAN;

    double x_buf[NEF_HMA_MAX_SQ];
    double x_sum = 0.0;
    double x_wsum = 0.0;
    int x_head = 0;

    const int start = first;

    for (int j = 0; j < half; ++j) {
        const double v = prices[start + j];
        const double jf = (double)j + 1.0;
        s_full += v;
        ws_full_acc = fma(jf, v, ws_full_acc);
        s_half += v;
        ws_half_acc = fma(jf, v, ws_half_acc);
    }
    wma_half = ws_half_acc / ws_half;

    if (period > half + 1) {
        for (int j = half; j < period - 1; ++j) {
            const int idx = start + j;
            const double v = prices[idx];

            const double jf = (double)j + 1.0;
            s_full += v;
            ws_full_acc = fma(jf, v, ws_full_acc);

            const double old_h = prices[idx - half];
            const double prev = s_half;
            s_half = prev + v - old_h;
            ws_half_acc = fma(half_f, v, ws_half_acc - prev);
            wma_half = ws_half_acc / ws_half;
        }
    }

    {
        const int j = period - 1;
        const int idx = start + j;
        const double v = prices[idx];

        const double jf = (double)j + 1.0;
        s_full += v;
        ws_full_acc = fma(jf, v, ws_full_acc);
        wma_full = ws_full_acc / ws_full;

        const double old_h = prices[idx - half];
        const double prev = s_half;
        s_half = prev + v - old_h;
        ws_half_acc = fma(half_f, v, ws_half_acc - prev);
        wma_half = ws_half_acc / ws_half;

        const double x = 2.0 * wma_half - wma_full;
        x_buf[0] = x;
        x_sum += x;
        x_wsum = fma(1.0, x, x_wsum);

        if (sq == 1) row[first_out] = x_wsum / ws_sqrt;
    }

    if (sq > 1) {
        for (int j = period; j < period + sq - 1; ++j) {
            const int idx = start + j;
            const double v = prices[idx];

            const double old_f = prices[idx - period];
            const double prev_f = s_full;
            s_full = prev_f + v - old_f;
            ws_full_acc = fma(period_f, v, ws_full_acc - prev_f);
            wma_full = ws_full_acc / ws_full;

            const double old_h = prices[idx - half];
            const double prev_h = s_half;
            s_half = prev_h + v - old_h;
            ws_half_acc = fma(half_f, v, ws_half_acc - prev_h);
            wma_half = ws_half_acc / ws_half;

            const double x = 2.0 * wma_half - wma_full;
            const int pos = j + 1 - period;
            x_buf[pos] = x;
            x_sum += x;
            x_wsum = fma((double)pos + 1.0, x, x_wsum);

            if (pos + 1 == sq) row[first_out] = x_wsum / ws_sqrt;
        }
    }

    for (int j = period + sq - 1; j < n - start; ++j) {
        const int idx = start + j;
        const double v = prices[idx];

        const double old_f = prices[idx - period];
        const double prev_f = s_full;
        s_full = prev_f + v - old_f;
        ws_full_acc = fma(period_f, v, ws_full_acc - prev_f);
        wma_full = ws_full_acc / ws_full;

        const double old_h = prices[idx - half];
        const double prev_h = s_half;
        s_half = prev_h + v - old_h;
        ws_half_acc = fma(half_f, v, ws_half_acc - prev_h);
        wma_half = ws_half_acc / ws_half;

        const double x = 2.0 * wma_half - wma_full;
        const double old_x = x_buf[x_head];
        x_buf[x_head] = x;
        ++x_head;
        if (x_head == sq) x_head = 0;

        const double prev_sum = x_sum;
        x_sum = prev_sum + x - old_x;
        x_wsum = fma(sq_f, x, x_wsum - prev_sum);

        row[idx] = x_wsum / ws_sqrt;
    }
}


// ===========================================================================
// S1 f64 LANE  --  hma
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/moving_averages/hma.rs -- `hma_scalar` (:368), `hma_scalar_period5` (:346), `hma_with_kernel_into` (:268)
//
// PERIOD-BASED, routed through `compute_ma_batch` (cpu_batch.rs:2216), which
// passes the combo's `period` straight through.
//
// TWO CPU PATHS, BOTH WRITTEN OUT, BECAUSE THEY ARE DIFFERENT ARITHMETIC.
// `hma_with_kernel_into` forks at period == 5 -- for EVERY kernel variant,
// scalar and AVX alike (hma.rs:320-328) -- to `hma_scalar_period5`, a closed
// form `(30c[i] + 27c[i-1] - 7c[i-3] - 4c[i-4] - c[i-5]) / 45`. That is the
// algebraic expansion of the three nested WMAs, evaluated in ONE pass instead
// of through three rolling accumulators, so it differs from the general path
// in floating point at every bar. Both are here.
//
// THE GENERAL PATH IS A THREE-STAGE ROLLING RECURRENCE and every stage is
// reproduced in the CPU's exact order:
//   `ws_full_acc = jf.mul_add(v, ws_full_acc)` during the seed -- ONE rounding.
//   `ws_half_acc = half_f.mul_add(v, ws_half_acc - prev)` during the slide --
//     ONE rounding on the fused part, and the `- prev` INSIDE the addend, not
//     factored out. Rewriting it as `ws + half*v - prev` is two roundings and
//     a different accumulator from that bar onward.
//   `x_wsum = sq_f.mul_add(x, x_wsum - prev_sum)` for the sqrt-length stage.
// The weight sums `ws_half`/`ws_full`/`ws_sqrt` are the exact triangular
// numbers `p(p+1)/2` computed in integer arithmetic and then converted, which
// is what the CPU does -- not accumulated, and not the closed form in double.
//
// `sq = (period as f64).sqrt().floor()` is computed with the same `sqrt` then
// `floor`, because for a perfect square a different rounding of `sqrt` would
// change `sq` by one and shift the whole series.
//
// PER-THREAD RING: the third stage needs the last `sq` values of the
// `2*wma_half - wma_full` series, and it is a subtract-then-add sliding sum, so
// the ring is required. `sq = floor(sqrt(period))`, so bounding the ring at 64
// bounds the period at 4095 -- declared to the host as `HMA_MAX_PERIOD` and
// refused BY NAME rather than truncated.
//
// WARMUP: `first + period + sqrt_len - 2`, which is NOT the `first + period -
// 1` most moving averages use. `hma_with_kernel_into` also rejects the input
// outright when `len - first < period + sqrt_len - 1`.
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

#define NEO_S1_HMA_MAX_SQRT 64
#define NEO_S1_HMA_MAX_PERIOD 4095

extern "C" __global__ void neoethos_hma_batch_f64(
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
    const int period = periods[r];

    const int half = period / 2;
    const int sq = (period > 0) ? (int)floor(sqrt((double)period)) : 0;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) || (period > NEO_S1_HMA_MAX_PERIOD) ||
        (half == 0) || (sq == 0) ||
        ((n - first_valid) < period) ||
        ((n - first_valid) < period + sq - 1);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int first_out = first_valid + period + sq - 2;
    for (int i = 0; i < first_out && i < n; ++i) row[i] = neo_s1_qnan();
    if (first_out >= n) return;

    if (period == 5) {
        // `hma_scalar_period5`. Note it starts at `first + 5`, which equals
        // `first_out` here (period 5 -> sq 2 -> first + 5 + 2 - 2).
        const double inv45 = 1.0 / 45.0;
        for (int i = first_valid + 5; i < n; ++i) {
            const double d2 = prices[i - 2];
            if (neo_s1_isnan(d2)) {
                row[i] = neo_s1_qnan();
            } else {
                row[i] = (30.0 * prices[i] + 27.0 * prices[i - 1]
                          - 7.0 * prices[i - 3]
                          - 4.0 * prices[i - 4]
                          - prices[i - 5]) * inv45;
            }
        }
        return;
    }

    // Triangular weight sums, integer-exact then converted -- as the CPU does.
    const double ws_half = (double)((long long)half * (long long)(half + 1) / 2);
    const double ws_full = (double)((long long)period * (long long)(period + 1) / 2);
    const double ws_sqrt = (double)((long long)sq * (long long)(sq + 1) / 2);
    const double half_f = (double)half;
    const double period_f = (double)period;
    const double sq_f = (double)sq;

    double s_half = 0.0, ws_half_acc = 0.0;
    double s_full = 0.0, ws_full_acc = 0.0;
    double wma_half = neo_s1_qnan(), wma_full = neo_s1_qnan();

    double x_buf[NEO_S1_HMA_MAX_SQRT];
    double x_sum = 0.0;
    double x_wsum = 0.0;
    int x_head = 0;

    const int start = first_valid;

    for (int j = 0; j < half; ++j) {
        const double v = prices[start + j];
        const double jf = (double)j + 1.0;
        s_full += v;
        ws_full_acc = fma(jf, v, ws_full_acc);
        s_half += v;
        ws_half_acc = fma(jf, v, ws_half_acc);
    }
    wma_half = ws_half_acc / ws_half;

    if (period > half + 1) {
        for (int j = half; j < period - 1; ++j) {
            const int idx = start + j;
            const double v = prices[idx];

            const double jf = (double)j + 1.0;
            s_full += v;
            ws_full_acc = fma(jf, v, ws_full_acc);

            const double old_h = prices[idx - half];
            const double prev = s_half;
            s_half = prev + v - old_h;
            ws_half_acc = fma(half_f, v, ws_half_acc - prev);
            wma_half = ws_half_acc / ws_half;
        }
    }

    {
        const int j = period - 1;
        const int idx = start + j;
        const double v = prices[idx];

        const double jf = (double)j + 1.0;
        s_full += v;
        ws_full_acc = fma(jf, v, ws_full_acc);
        wma_full = ws_full_acc / ws_full;

        const double old_h = prices[idx - half];
        const double prev = s_half;
        s_half = prev + v - old_h;
        ws_half_acc = fma(half_f, v, ws_half_acc - prev);
        wma_half = ws_half_acc / ws_half;

        const double x = 2.0 * wma_half - wma_full;
        x_buf[0] = x;
        x_sum += x;
        x_wsum = fma(1.0, x, x_wsum);

        if (sq == 1) row[first_out] = x_wsum / ws_sqrt;
    }

    if (sq > 1) {
        for (int j = period; j < period + sq - 1; ++j) {
            const int idx = start + j;
            const double v = prices[idx];

            const double old_f = prices[idx - period];
            const double prev_f = s_full;
            s_full = prev_f + v - old_f;
            ws_full_acc = fma(period_f, v, ws_full_acc - prev_f);
            wma_full = ws_full_acc / ws_full;

            const double old_h = prices[idx - half];
            const double prev_h = s_half;
            s_half = prev_h + v - old_h;
            ws_half_acc = fma(half_f, v, ws_half_acc - prev_h);
            wma_half = ws_half_acc / ws_half;

            const double x = 2.0 * wma_half - wma_full;
            const int pos = j + 1 - period;
            x_buf[pos] = x;
            x_sum += x;
            x_wsum = fma((double)pos + 1.0, x, x_wsum);

            if (pos + 1 == sq) row[first_out] = x_wsum / ws_sqrt;
        }
    }

    for (int j = period + sq - 1; j < n - start; ++j) {
        const int idx = start + j;
        const double v = prices[idx];

        const double old_f = prices[idx - period];
        const double prev_f = s_full;
        s_full = prev_f + v - old_f;
        ws_full_acc = fma(period_f, v, ws_full_acc - prev_f);
        wma_full = ws_full_acc / ws_full;

        const double old_h = prices[idx - half];
        const double prev_h = s_half;
        s_half = prev_h + v - old_h;
        ws_half_acc = fma(half_f, v, ws_half_acc - prev_h);
        wma_half = ws_half_acc / ws_half;

        const double x = 2.0 * wma_half - wma_full;
        const double old_x = x_buf[x_head];
        x_buf[x_head] = x;
        if (++x_head == sq) x_head = 0;

        const double prev_sum = x_sum;
        x_sum = prev_sum + x - old_x;
        x_wsum = fma(sq_f, x, x_wsum - prev_sum);

        row[idx] = x_wsum / ws_sqrt;
    }
}
