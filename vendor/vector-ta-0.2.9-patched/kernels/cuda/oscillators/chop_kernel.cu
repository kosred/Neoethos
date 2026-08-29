#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


#ifndef CHOP_REG_RING_MAX
#define CHOP_REG_RING_MAX 64
#endif


static __forceinline__ __device__
void kbn_update(float delta, float& sum_hi, float& sum_lo) {
    float t = sum_hi + delta;
    float c = (fabsf(sum_hi) >= fabsf(delta)) ? (sum_hi - t) + delta : (delta - t) + sum_hi;
    sum_hi = t;
    sum_lo += c;
}


extern "C" __global__ void chop_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    const int*   __restrict__ drifts,
    const float* __restrict__ scalars,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    int max_period,
    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ row_out = out + base;

    const int period = periods[combo];
    const int drift  = drifts[combo];
    const float scalar = scalars[combo];

    auto fill_all_nan = [&]() {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            row_out[i] = NAN;
        }
    };

    if (UNLIKELY(period <= 0 || drift <= 0 ||
                 first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan();
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        fill_all_nan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        row_out[i] = NAN;
    }

    __shared__ int sh_k;
    __shared__ int sh_k_ok;
    if (threadIdx.x == 0) {
        sh_k = log2_tbl[period];
        sh_k_ok = (sh_k >= 0 && sh_k < level_count) ? 1 : 0;
    }
    __syncthreads();

    if (UNLIKELY(sh_k_ok == 0)) {

        fill_all_nan();
        return;
    }


    if (threadIdx.x != 0) return;

    const int k = sh_k;


    const float inv_drift = 1.0f / (float)drift;


    const float inv_log2p = 1.0f / log2f((float)period);
    const float scale_over_log2p = scalar * inv_log2p;


    const int offset = 1 << k;
    const int level_base = level_offsets[k];


    const bool series_has_nan = (nan_psum[series_len] - nan_psum[first_valid]) != 0;


    float rma_atr = NAN;
    float sum_tr = 0.0f;


    int ring_idx = 0;
    float sum_hi = 0.0f, sum_lo = 0.0f;


    float ring_reg[CHOP_REG_RING_MAX];
    extern __shared__ unsigned char __smem[];
    float* ring_smem = reinterpret_cast<float*>(__smem);

    if (period <= CHOP_REG_RING_MAX) {
        #pragma unroll
        for (int i = 0; i < CHOP_REG_RING_MAX; ++i) {
            if (i < period) ring_reg[i] = 0.0f;
        }
    } else {
        for (int i = 0; i < period && i < max_period; ++i) ring_smem[i] = 0.0f;
    }


    float prev_close = close[first_valid];

    for (int t = first_valid; t < series_len; ++t) {
        const float hi = high[t];
        const float lo = low[t];
        const float cl = close[t];
        const int rel = t - first_valid;


        float tr;
        if (rel == 0) {
            tr = hi - lo;
        } else {
            const float a = hi - lo;
            const float b = fabsf(hi - prev_close);
            const float c = fabsf(lo - prev_close);
            tr = fmaxf(a, fmaxf(b, c));
        }


        if (rel < drift) {
            sum_tr += tr;
            if (rel == drift - 1) {
                rma_atr = sum_tr * inv_drift;
            }
        } else {

            rma_atr = fmaf(inv_drift, (tr - rma_atr), rma_atr);
        }
        prev_close = cl;


        const float current_atr = (rel < drift) ? ((rel == drift - 1) ? rma_atr : NAN) : rma_atr;
        const float add = (current_atr == current_atr) ? current_atr : 0.0f;


        float oldest = 0.0f;
        if (period <= CHOP_REG_RING_MAX) {
            oldest = ring_reg[ring_idx];
            ring_reg[ring_idx] = add;
        } else {
            oldest = ring_smem[ring_idx];
            ring_smem[ring_idx] = add;
        }

        ring_idx += 1;
        if (ring_idx == period) ring_idx = 0;

        const float delta = add - oldest;
        kbn_update(delta, sum_hi, sum_lo);
        const float rolling_sum_atr = sum_hi + sum_lo;

        if (rel >= period - 1) {
            const int start = t - period + 1;


            if (series_has_nan) {
                if (nan_psum[t + 1] - nan_psum[start] != 0) {
                    row_out[t] = NAN;
                    continue;
                }
            }


            const int idx_a = level_base + start;
            const int idx_b = level_base + (t + 1 - offset);
            const float hmax = fmaxf(st_max[idx_a], st_max[idx_b]);
            const float lmin = fminf(st_min[idx_a], st_min[idx_b]);
            const float range = hmax - lmin;

            if (!(range > 0.0f) || !(rolling_sum_atr > 0.0f)) {
                row_out[t] = NAN;
            } else {

                const float ratio = rolling_sum_atr / range;
                const float y = scale_over_log2p * log2f(ratio);
                row_out[t] = y;
            }
        }
    }
}


extern "C" __global__ void chop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ atr_psum_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;


    const float inv_log2p = 1.0f / log2f((float)period);
    const float scale_over_log2p = scalar * inv_log2p;

    const int first = first_valids[s];
    if (UNLIKELY(first < 0 || first >= rows)) {
        for (int r = 0; r < rows; ++r) out_tm[(size_t)r * cols + s] = NAN;
        return;
    }
    if (UNLIKELY(period <= 0 || period > rows - first)) {
        for (int r = 0; r < rows; ++r) out_tm[(size_t)r * cols + s] = NAN;
        return;
    }

    const int warm = first + period - 1;
    for (int r = 0; r < warm; ++r) out_tm[(size_t)r * cols + s] = NAN;

    for (int r = warm; r < rows; ++r) {

        const float sum_atr = atr_psum_tm[(size_t)(r + 1) * cols + s]
                            - atr_psum_tm[(size_t)(r + 1 - period) * cols + s];
        if (!(sum_atr > 0.0f)) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }


        float hmax = -INFINITY;
        float lmin = INFINITY;
        bool nan_in_window = false;
        const size_t start = (size_t)(r - period + 1) * cols + s;
        for (int k = 0; k < period; ++k) {
            const float h = high_tm[start + (size_t)k * cols];
            const float l = low_tm[start + (size_t)k * cols];
            if (!(h == h) || !(l == l)) { nan_in_window = true; break; }
            hmax = fmaxf(hmax, h);
            lmin = fminf(lmin, l);
        }

        if (nan_in_window) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }

        const float range = hmax - lmin;
        if (!(range > 0.0f)) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }


        const float y = scale_over_log2p * log2f(sum_atr / range);
        out_tm[(size_t)r * cols + s] = y;
    }
}


// ===========================================================================
// S1 f64 LANE  --  chop
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/chop.rs -- `chop_scalar` (:469), `chop_scalar_period_14_drift_1` (:601), `chop_with_kernel` (:245)
//
// PERIOD-BASED. `compute_chop_batch` (cpu_batch.rs:15086) reads `period`
// (default 14), `scalar` (100.0) and `drift` (1).
//
// CHOP_TRADINGVIEW_LOG10_SEMANTICS_V1 follows the published operation shape:
// `100 * LOG10(SUM(ATR(1), n) / range) / LOG10(n)`. The platform libc and
// libdevice logarithms are different implementations: Gate196 observed the
// AMD/Linux CPU bit 0x40569935e3af9c88 while the same RTX row produced
// 0x40569935e3af9c87. Both CPU paths and this CUDA lane therefore use the same
// fixed-order range reduction and 25-term atanh series below. The specialized
// (14, 1) path still owns a true-range ring; only its old algebraic rewrite to
// natural log/log1p was retired.
//
// NaN SEMANTICS -- THE adx-CLASS BUG, CHECKED: true range is
// `hl.max(hc).max(lc)`, which is `f64::max` and therefore returns the NON-NaN
// operand. An if-chain would let a NaN survive and poison every later bar of
// the ATR recursion. Written with `fmax`.
//
// THE MONOTONIC DEQUES ARE REPRODUCED EXACTLY, NOT APPROXIMATED. The CPU keeps
// a max-deque over `high` (pop back while `high[back] <= hi`) and a min-deque
// over `low` (pop back while `low[back] >= lo`), and reads the FRONT. The
// front is the smallest index in the window that no later window element has
// killed, and an element is killed only by a strictly comparable neighbour --
// a NaN kills nothing and is itself never killed, so a NaN inside the window
// can legitimately BE the front and make `range` NaN. A plain `fmax` scan
// would skip the NaN and emit a number where the CPU emits NaN. The helpers
// below therefore simulate the deque with a backward walk: identical front,
// identical NaN behaviour, O(period) per bar instead of amortised O(1). With
// one thread per period-combo (rows are the period list, typically five) that
// cost is irrelevant and the exactness is not.
//
// PER-THREAD RING: the general path slides `rolling_sum_atr` over a ring of
// `period` values with subtract-then-add, so the ring must exist -- the sum is
// accumulation-order dependent and cannot be recomputed. It is a fixed
// per-thread array, so the kernel carries a hard period bound, declared to the
// host as `CHOP_MAX_PERIOD` and REFUSED BY NAME rather than truncated.
//
// WARMUP: `first_valid + period - 1`, and first_valid is the first index where
// high, low and close are simultaneously non-NaN (chop.rs:281-287).
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

#define NEO_S1_CHOP_MAX_PERIOD 1024

__device__ __forceinline__ double neoethos_chop_ln_cpu_exact_v1(double value) {
    if (value == INFINITY) return INFINITY;
    if (!(value > 0.0)) return neo_s1_qnan();

    double normalized = value;
    int exponent_adjustment = 0;
    unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(normalized));
    if (((bits >> 52U) & 0x7ffU) == 0U) {
        normalized *= __longlong_as_double(
            static_cast<long long>(0x4350000000000000ULL));
        exponent_adjustment = -54;
        bits = static_cast<unsigned long long>(__double_as_longlong(normalized));
    }

    const int exponent =
        static_cast<int>((bits >> 52U) & 0x7ffU) - 1023 + exponent_adjustment;
    const unsigned long long mantissa_bits =
        (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    const double mantissa =
        __longlong_as_double(static_cast<long long>(mantissa_bits));
    const double z = (mantissa - 1.0) / (mantissa + 1.0);
    const double z_squared = z * z;
    double term = z;
    double sum = z;
    unsigned int denominator = 3U;
    while (denominator <= 49U) {
        term *= z_squared;
        sum += term / static_cast<double>(denominator);
        denominator += 2U;
    }
    const double ln_2 = __longlong_as_double(
        static_cast<long long>(0x3fe62e42fefa39efULL));
    return static_cast<double>(exponent) * ln_2 + 2.0 * sum;
}

__device__ __forceinline__ double neoethos_chop_log10_cpu_exact_v1(double value) {
    const double ln_10 = __longlong_as_double(
        static_cast<long long>(0x40026bb1bbb55515ULL));
    return neoethos_chop_ln_cpu_exact_v1(value) / ln_10;
}

__device__ __forceinline__ double neoethos_chop_value_from_ratio_exact_v1(
    double ratio, double scalar, double log10_period) {
    return (scalar * neoethos_chop_log10_cpu_exact_v1(ratio)) / log10_period;
}

// Front of the CPU's max-deque over `high` for the window [win_start, i].
// An element k survives iff no LATER window element j has `high[k] <= high[j]`.
// NaN makes that comparison false in both directions, so a NaN is never popped
// and never pops -- which is why this is a simulation and not an fmax scan.
__device__ __forceinline__ double neo_s1_chop_front_high(
    const double* __restrict__ high, int win_start, int i)
{
    double m = -INFINITY;
    int front = i;
    for (int k = i; k >= win_start; --k) {
        const double hk = high[k];
        const bool survives = neo_s1_isnan(hk) || !(hk <= m);
        if (survives) front = k;
        if (!neo_s1_isnan(hk) && hk > m) m = hk;
    }
    return high[front];
}

// Front of the CPU's min-deque over `low`: k survives iff no later j has
// `low[k] >= low[j]`.
__device__ __forceinline__ double neo_s1_chop_front_low(
    const double* __restrict__ low, int win_start, int i)
{
    double m = INFINITY;
    int front = i;
    for (int k = i; k >= win_start; --k) {
        const double lk = low[k];
        const bool survives = neo_s1_isnan(lk) || !(lk >= m);
        if (survives) front = k;
        if (!neo_s1_isnan(lk) && lk < m) m = lk;
    }
    return low[front];
}

extern "C" __global__ void neoethos_chop_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
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

    // `ChopParams::default()` as read by cpu_batch.rs:15093-15095.
    const double scalar = 100.0;
    const int drift = 1;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period <= 0) || (period > NEO_S1_CHOP_MAX_PERIOD) ||
        (drift <= 0) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();
    if (warm >= n) return;

    double atr_ring[NEO_S1_CHOP_MAX_PERIOD];
    for (int k = 0; k < period; ++k) atr_ring[k] = 0.0;
    int atr_ring_idx = 0;
    double rolling_sum_atr = 0.0;

    double prev_close = close[first_valid];
    const double log10_period =
        neoethos_chop_log10_cpu_exact_v1(static_cast<double>(period));

    if (period == 14 && drift == 1) {
        // `chop_scalar_period_14_drift_1`: the ring holds TRUE RANGE, not the
        // smoothed ATR. Its output still uses the published LOG10 form.
        for (int i = first_valid; i < n; ++i) {
            const double hi = high[i];
            const double lo = low[i];
            const double hl = hi - lo;
            double tr;
            if (i == first_valid) {
                tr = hl;
            } else {
                const double hc = fabs(hi - prev_close);
                const double lc = fabs(lo - prev_close);
                tr = fmax(fmax(hl, hc), lc);
            }
            prev_close = close[i];

            rolling_sum_atr -= atr_ring[atr_ring_idx];
            atr_ring[atr_ring_idx] = tr;
            rolling_sum_atr += tr;
            if (++atr_ring_idx == 14) atr_ring_idx = 0;

            if (i - first_valid >= 13) {
                const int win_start = i - 13;
                const double range = neo_s1_chop_front_high(high, win_start, i)
                                   - neo_s1_chop_front_low(low, win_start, i);
                if (range > 0.0 && rolling_sum_atr > 0.0) {
                    const double ratio = rolling_sum_atr / range;
                    row[i] = neoethos_chop_value_from_ratio_exact_v1(
                        ratio, scalar, log10_period);
                } else {
                    row[i] = neo_s1_qnan();
                }
            }
        }
        return;
    }

    // General path.
    const double alpha = 1.0 / (double)drift;
    double rma_atr = neo_s1_qnan();
    double sum_tr = 0.0;

    for (int i = first_valid; i < n; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double hl = hi - lo;
        double tr;
        if (i == first_valid) {
            sum_tr = hl;
            tr = hl;
        } else {
            const double hc = fabs(hi - prev_close);
            const double lc = fabs(lo - prev_close);
            tr = fmax(fmax(hl, hc), lc);
        }

        const int rel = i - first_valid;
        if (rel < drift) {
            if (i != first_valid) sum_tr += tr;
            if (rel == drift - 1) rma_atr = sum_tr / (double)drift;
        } else {
            rma_atr += alpha * (tr - rma_atr);
        }
        prev_close = close[i];

        double current_atr;
        if (rel < drift) {
            current_atr = (rel == drift - 1) ? rma_atr : neo_s1_qnan();
        } else {
            current_atr = rma_atr;
        }

        rolling_sum_atr -= atr_ring[atr_ring_idx];
        const double new_val = neo_s1_isnan(current_atr) ? 0.0 : current_atr;
        atr_ring[atr_ring_idx] = new_val;
        rolling_sum_atr += new_val;
        if (++atr_ring_idx == period) atr_ring_idx = 0;

        if (rel >= period - 1) {
            const int win_start = (i >= period - 1) ? (i - (period - 1)) : 0;
            const double range = neo_s1_chop_front_high(high, win_start, i)
                               - neo_s1_chop_front_low(low, win_start, i);
            if (range > 0.0 && rolling_sum_atr > 0.0) {
                const double ratio = rolling_sum_atr / range;
                row[i] = neoethos_chop_value_from_ratio_exact_v1(
                    ratio, scalar, log10_period);
            } else {
                row[i] = neo_s1_qnan();
            }
        }
    }
}
