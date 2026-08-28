#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {

constexpr int WARP = 32;

__device__ __forceinline__ double kama_const_max() {
    return 2.0 / 31.0;
}

__device__ __forceinline__ double kama_const_diff() {
    return (2.0 / 3.0) - kama_const_max();
}

__device__ __forceinline__ double warp_sum(double v) {
    unsigned m = __activemask();
    #pragma unroll
    for (int off = WARP >> 1; off > 0; off >>= 1) {
        v += __shfl_down_sync(m, v, off);
    }
    return v;
}

}


extern "C" __global__ __launch_bounds__(32)
void kama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;


    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid)) ||
        ((first_valid + period) >= series_len);

    const float nan_f = CUDART_NAN_F;

    if (invalid) {

        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = nan_f;
        }
        return;
    }


    const int initial_idx = first_valid + period;


    for (int i = threadIdx.x; i < initial_idx; i += blockDim.x) {
        out[base + i] = nan_f;
    }


    double sum_roc1 = 0.0;
    if (threadIdx.x < WARP) {
        const int lane = threadIdx.x;
        double local = 0.0;
        const int start = first_valid;
        const int end   = first_valid + period;
        for (int j = start + lane; j < end; j += WARP) {
            const double a = static_cast<double>(prices[j]);
            const double b = static_cast<double>(prices[j + 1]);
            local += fabs(b - a);
        }
        local = warp_sum(local);
        if (lane == 0) sum_roc1 = local;
    }


    if (threadIdx.x != 0) return;


    double prev_price = static_cast<double>(prices[initial_idx]);
    double prev_kama  = prev_price;
    out[base + initial_idx] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(prices[trailing_idx]);

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int i = initial_idx + 1; i < series_len; ++i) {
        const double price         = static_cast<double>(prices[i]);
        const double next_trailing = static_cast<double>(prices[trailing_idx + 1]);


        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);


        trailing_value = next_trailing;
        trailing_idx  += 1;


        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;


        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out[base + i] = static_cast<float>(prev_kama);


        prev_price = price;
    }
}


extern "C" __global__ __launch_bounds__(32)
void kama_batch_prefix_f32(const float* __restrict__ prices,
                           const float* __restrict__ prefix_roc1,
                           const int* __restrict__ periods,
                           int series_len,
                           int n_combos,
                           int first_valid,
                           float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    const int initial_idx = first_valid + period;
    const float nan_f = CUDART_NAN_F;

    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid)) ||
        (initial_idx >= series_len);

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = nan_f;
        }
        return;
    }

    for (int i = threadIdx.x; i < initial_idx; i += blockDim.x) {
        out[base + i] = nan_f;
    }

    if (threadIdx.x == 0) {
        out[base + initial_idx] = prices[initial_idx];
    }


    const int lane = threadIdx.x;
    if (lane >= WARP) return;

    float prev_kama = prices[initial_idx];
    const float cmax = 2.0f / 31.0f;
    const float cdiff = (2.0f / 3.0f) - cmax;

    int chunk_start = initial_idx + 1;
    for (; (chunk_start + (WARP - 1)) < series_len; chunk_start += WARP) {
        const int t = chunk_start + lane;
        const float price = prices[t];
        const float sum_roc1 = prefix_roc1[t] - prefix_roc1[t - period];
        const float direction = fabsf(price - prices[t - period]);
        const float er = (sum_roc1 == 0.0f) ? 0.0f : (direction / sum_roc1);
        const float tmp = fmaf(er, cdiff, cmax);
        const float sc = tmp * tmp;

        float a = 1.0f - sc;
        float b = sc * price;

        const unsigned m = 0xFFFFFFFFu;
        #pragma unroll
        for (int off = 1; off < WARP; off <<= 1) {
            const float a_up = __shfl_up_sync(m, a, off);
            const float b_up = __shfl_up_sync(m, b, off);
            if (lane >= off) {
                b = fmaf(a, b_up, b);
                a = a * a_up;
            }
        }

        const float x = fmaf(a, prev_kama, b);
        out[base + t] = x;

        prev_kama = __shfl_sync(m, x, WARP - 1);
    }


    if (lane == 0) {
        float kama = prev_kama;
        for (int t = chunk_start; t < series_len; ++t) {
            const float price = prices[t];
            const float sum_roc1 = prefix_roc1[t] - prefix_roc1[t - period];
            const float direction = fabsf(price - prices[t - period]);
            const float er = (sum_roc1 == 0.0f) ? 0.0f : (direction / sum_roc1);
            const float tmp = fmaf(er, cdiff, cmax);
            const float sc = tmp * tmp;
            kama = fmaf(price - kama, sc, kama);
            out[base + t] = kama;
        }
    }
}

extern "C" __global__ __launch_bounds__(32)
void kama_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    const int first_valid = first_valids[series];
    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid));

    const int initial_idx = first_valid + period;
    const float nan_f = CUDART_NAN_F;


    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };


    if (invalid || initial_idx >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + series] = nan_f;
        }
        return;
    }


    for (int t = threadIdx.x; t < initial_idx; t += blockDim.x) {
        out_tm[t * num_series + series] = nan_f;
    }


    double sum_roc1 = 0.0;
    if (threadIdx.x < WARP) {
        const int lane = threadIdx.x;
        double local = 0.0;
        const int start = first_valid;
        const int end   = first_valid + period;
        for (int j = start + lane; j < end; j += WARP) {
            const double a = static_cast<double>(at(prices_tm, j,     series));
            const double b = static_cast<double>(at(prices_tm, j + 1, series));
            local += fabs(b - a);
        }
        local = warp_sum(local);
        if (lane == 0) sum_roc1 = local;
    }

    if (threadIdx.x != 0) return;

    double prev_price = static_cast<double>(at(prices_tm, initial_idx, series));
    double prev_kama  = prev_price;
    out_tm[initial_idx * num_series + series] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(at(prices_tm, trailing_idx, series));

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int t = initial_idx + 1; t < series_len; ++t) {
        const double price         = static_cast<double>(at(prices_tm, t, series));
        const double next_trailing = static_cast<double>(at(prices_tm, trailing_idx + 1, series));

        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);

        trailing_value = next_trailing;
        trailing_idx  += 1;

        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;

        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out_tm[t * num_series + series] = static_cast<float>(prev_kama);

        prev_price = price;
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/moving_averages/kama.rs
//   * kama_prepare (:190) — first_valid = first non-NaN of the source.
//   * kama_with_kernel (:255) — warm = first + period; the NaN prefix is that
//     long and out[first + period] is the SEED, written as the raw price.
//   * kama_scalar (:312) — the arithmetic reproduced below.
//
// ROUNDING COUNT. Two fused steps on the CPU, both reproduced as fma():
//     let t = er.mul_add(const_diff, const_max);   -> fma(er, const_diff, const_max)
//     kama = (price - kama).mul_add(sc, kama);     -> fma(price - kama, sc, kama)
// Everything else is a plain operation and -fmad=false keeps it unfused.
//
// const_max and const_diff are the CPU's exact expressions, NOT decimal
// approximations: 2.0/(30.0+1.0) and (2.0/(2.0+1.0)) - const_max. Writing
// 0.06451612903225806 instead would be a different double.
//
// The er guard is an EXACT ZERO test (sum_roc1 == 0.0), not a tolerance. There
// is no epsilon here to re-derive for f64 — inventing one would change which
// bars take the branch.
//
// Sequential: kama, sum_roc1 and the trailing window carry across bars.
// One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_kama() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__
void neoethos_kama_f64(const double* __restrict__ prices,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const double QNAN = nef_qnan_kama();

    const int period = periods[r];
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    // kama.rs:255 — warm = first + period.
    const int warm = (first_valid + period) < n ? (first_valid + period) : n;
    for (int i = 0; i < warm; ++i) row[i] = QNAN;

    const int lookback = period - 1;
    const int initial_idx = first_valid + lookback + 1;
    if (initial_idx >= n) return;

    const double const_max = 2.0 / (30.0 + 1.0);
    const double const_diff = (2.0 / (2.0 + 1.0)) - const_max;

    double sum_roc1 = 0.0;
    const int today = first_valid;
    {
        double prev = prices[today];
        for (int i = 0; i <= lookback; ++i) {
            const double next = prices[today + i + 1];
            sum_roc1 += fabs(next - prev);
            prev = next;
        }
    }

    double kama = prices[initial_idx];
    row[initial_idx] = kama;

    int trailing_idx = today;
    double trailing_value = prices[trailing_idx];

    for (int i = initial_idx + 1; i < n; ++i) {
        const double price_prev = prices[i - 1];
        const double price = prices[i];

        const double next_tail = prices[trailing_idx + 1];
        const double old_diff = fabs(next_tail - trailing_value);
        const double new_diff = fabs(price - price_prev);
        sum_roc1 += new_diff - old_diff;

        trailing_value = next_tail;
        trailing_idx += 1;

        const double direction = fabs(price - next_tail);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);
        const double t = fma(er, const_diff, const_max);
        const double sc = t * t;

        kama = fma(price - kama, sc, kama);
        row[i] = kama;
    }
}



// ===========================================================================
// S1 f64 LANE  --  kama
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/moving_averages/kama.rs -- `kama_scalar` (:312), `kama_prepare` (:191), `kama_with_kernel` (:256)
//
// PERIOD-BASED via `compute_ma_batch`.
//
// ARITHMETIC ORDER -- two `mul_add`s, ONE rounding each:
//   `t = er.mul_add(const_diff, const_max)` then `sc = t * t`.
//   `kama = (price - kama).mul_add(sc, kama)` -- the same one-rounding shape
//   the brief names for `natr`. `kama + (price - kama) * sc` would be two.
// The constants are computed exactly as the CPU spells them:
//   `const_max  = 2.0 / (30.0 + 1.0)`
//   `const_diff = (2.0 / (2.0 + 1.0)) - const_max`
// -- NOT pre-folded to decimal literals, because 2/31 and 2/3 are not
// representable and a decimal literal would be a different double.
//
// THE EFFICIENCY-RATIO SUM slides with `sum_roc1 += new_diff - old_diff`: a
// subtract then an add, in that order, and the trailing value is taken from
// the data BEFORE the pointer advances. Reproduced literally; recomputing the
// window sum would be a different number even though it is the same quantity.
//
// `er = if sum_roc1 == 0.0 { 0.0 } else { direction / sum_roc1 }` is an exact
// zero test, not an epsilon -- an epsilon would change which bars take it.
//
// WARMUP: `alloc_with_nan_prefix(len, first + period)`; the seed value
// `out[first + period]` is then written by the compute, so the first emitted
// bar is `first + period`, one LATER than the usual `first + period - 1`. The
// seed loop reads `data[first + lookback + 1] = data[first + period]`, which is
// why `kama_prepare` rejects `len - first <= period` (strictly).
//
// KERNEL SELECTION CAVEAT: `choose_kama_kernel` maps `Auto` to `Avx2` when the
// host has avx2+fma (kama.rs:341-353). This kernel is written against
// `kama_scalar`; if the two disagree the fix belongs in the CPU.
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

extern "C" __global__ void neoethos_kama_batch_f64(
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

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) <= period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + period;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();

    const int lookback = period - 1;
    const double const_max = 2.0 / (30.0 + 1.0);
    const double const_diff = (2.0 / (2.0 + 1.0)) - const_max;

    double sum_roc1 = 0.0;
    const int today = first_valid;
    double prev = prices[today];
    for (int i = 0; i <= lookback; ++i) {
        const double next = prices[today + i + 1];
        sum_roc1 += fabs(next - prev);
        prev = next;
    }

    const int initial_idx = today + lookback + 1;
    double kama = prices[initial_idx];
    row[initial_idx] = kama;

    int trailing_idx = today;
    double trailing_value = prices[trailing_idx];

    for (int i = initial_idx + 1; i < n; ++i) {
        const double price_prev = prices[i - 1];
        const double price = prices[i];

        const double next_tail = prices[trailing_idx + 1];
        const double old_diff = fabs(next_tail - trailing_value);
        const double new_diff = fabs(price - price_prev);
        sum_roc1 += new_diff - old_diff;

        trailing_value = next_tail;
        trailing_idx += 1;

        const double direction = fabs(price - next_tail);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);
        const double t = fma(er, const_diff, const_max);
        const double sc = t * t;

        kama = fma(price - kama, sc, kama);
        row[i] = kama;
    }
}
