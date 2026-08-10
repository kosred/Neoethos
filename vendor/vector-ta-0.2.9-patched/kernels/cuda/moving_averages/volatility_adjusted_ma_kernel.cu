// volatility_adjusted_ma (VAMA) — CUDA f64 kernel.
//
// WHAT THIS REPLACES
// ------------------
// The f64 LANE had nothing: no `F64_KERNELS` row, so a request for
// `volatility_adjusted_ma` answered `CudaF64KernelMissing`.
//
// `kernels/cuda/moving_averages/vama_kernel.cu` DOES exist and is NOT touched
// by this file. It carries two f32 entry points — `vama_batch_f32` (:10) and
// `vama_many_series_one_param_f32` (:200) — which `src/cuda/moving_averages/
// vama_wrapper.rs:146` still loads. Converting that file is shard S1's
// assignment and this file does not overlap it: a new translation unit, a new
// module stem, a new entry point.
//
// CPU REFERENCE — src/indicators/moving_averages/volatility_adjusted_ma.rs
// -------------------------------------------------------------------------
//   :120 VamaInput                      — the line the brief names
//   :705 vama_prepare                   — `first` and every rejection
//   :314 vama_core_into                 — EMA + monotonic deques
//   :442 vama_default_fused_wma_into    — the fused fast path
//   :562 vama_with_kernel               — which of the two runs
//   :422 is_default_smoothed_vama / :437 can_use_default_fused
// and the two indicators it composes:
//   moving_averages/ema.rs:461 ema_scalar_into, :427 ema_into_slice
//   moving_averages/wma.rs:305 wma_scalar,      :223 wma_into_slice
//
// THE PARAMETERS THAT ARE NOT IN THE LANE ABI
// -------------------------------------------
// VAMA takes FIVE parameters and the sweep carries one. `ma.rs:1397-1415` is
// the route that sweeps a period through it, and it is explicit: `base_period:
// Some(period), ..Default::default()`. So base_period is swept and the rest are
// the CPU defaults — vol_period 51 (:41), smoothing true (:42), smooth_type 3 =
// WMA (:43), smooth_period 5 (:44). Source is "close" (:45).
//
// TWO CPU PATHS, ONE KERNEL — AND WHY THAT IS NOT A SHORTCUT
// ----------------------------------------------------------
// `vama_with_kernel` picks the fused path (:566-573) only when base_period is
// exactly 113 AND every value from `first` on is finite. The general path is
// EMA -> core -> `wma_into_slice`. The fused path inlines all three. They are
// the SAME NUMBERS, checked term by term:
//
//  * EMA. Fused :473-481 is `mean = ((vc-1)*mean + x)/vc` then
//    `beta.mul_add(ema_value, alpha*x)` — character for character
//    `ema_scalar_into` :485-505.
//  * DEQUES. Fused :484-528 drops the `!(e.is_nan() || x.is_nan())` guard that
//    :383 applies. Under `can_use_default_fused` nothing is non-finite, so the
//    guard never fires in the general path either.
//  * WMA. `wma_scalar` warms with `weight_sum += v*(k+1)` for k in 0..4 then
//    rolls `weight_sum += v*5; sum += v; out = weight_sum/15;
//    weight_sum -= sum; sum -= old`, with `weights = 5*6*0.5 = 15.0` exactly.
//    Fused :539-550 is the same recurrence with WMA_DEN = 15.0, and its
//    `wma_ring[(k+1) % 5]` is bar `i - 4`, which is `wma_scalar`'s `*in_old`.
//    Its `first` is `warmup`, because `work[..warmup]` is the NaN prefix
//    `alloc_with_nan_prefix(len, warmup)` left and `vama_core_into` writes only
//    from `warmup` on.
//
// So one implementation serves both, and it is the streaming one — which also
// removes the full-length `work` and `ema_values` buffers a literal
// transcription would need per thread.
//
// SHAPE — ONE THREAD PER COLUMN, BARS ASCENDING
// ---------------------------------------------
// Three chained recurrences: the EMA carries `mean`/`ema_value`, the volatility
// window carries two monotonic deques whose contents depend on every earlier
// bar, and the WMA carries a rolling weighted sum. One thread walks the column.
// The deques are `vol_period` slots — 51, the CPU default and the only value
// this ABI can produce — so they are fixed-size local arrays, not an
// unbounded allocation.
//
// ARITHMETIC
// ----------
// `(0.5f64).mul_add(up + dn, e)` (:411) is ONE rounding and is written as
// `fma(0.5, up + dn, e)`. `beta.mul_add(prev, alpha * x)` likewise. Everything
// else is spelled with the CPU's operator order and `-fmad=false` on this
// translation unit stops nvcc contracting any of it.
//
// The deque guard at :383 is `is_nan`, NOT `is_finite`: an INFINITE bar is
// pushed by the CPU (and produces an infinite `d`) while the EMA at :492 skips
// it with `is_finite_fast`. The two tests are deliberately different here too.
//
// f64 end to end; no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. Listed in `F64_LANE_SOURCES`, so never `--use_fast_math`.

#include <cmath>
#include <cstdint>

// volatility_adjusted_ma.rs:41, :43, :44
#define VAMA_VOL_PERIOD 51
#define VAMA_SMOOTH_PERIOD 5
// wma_scalar (:309): period_f * (period_f + 1.0) * 0.5 with period_f = 5.
#define VAMA_WMA_DEN 15.0

__device__ __forceinline__ double vama_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void volatility_adjusted_ma_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int base_period = periods[r];
    const int vol_period = VAMA_VOL_PERIOD;
    const int first = first_valid;

    const int needed = (base_period > vol_period) ? base_period : vol_period;

    // vama_prepare (:705): EmptyInputData, AllValuesNaN (the caller's
    // first-valid scan), InvalidPeriod for either period, and
    // NotEnoughValidData when `len - first < max(base, vol)`. smooth_type 3 is
    // in range so that branch cannot fire.
    //
    // The last clause is `wma_prepare` (:249) applied to `work`, whose first
    // non-NaN index is `warmup`: it rejects `len - first < period`, i.e. fewer
    // than five core values, and a rejection there aborts the whole indicator
    // rather than shortening it.
    const long long warmup_ll = (long long)first + (long long)needed - 1;
    const bool declined =
        (n <= 0) ||
        (first < 0) || (first >= n) ||
        (base_period <= 0) || (base_period > n) ||
        (vol_period > n) ||
        ((long long)(n - first) < (long long)needed) ||
        (warmup_ll >= (long long)n) ||
        ((long long)n - warmup_ll < (long long)VAMA_SMOOTH_PERIOD) ||
        (VAMA_SMOOTH_PERIOD > n);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = vama_qnan();
        return;
    }

    const int warmup = (int)warmup_ll;

    // wma_into_slice (:233) NaNs `dst[..first + period - 1]`, and `first` for
    // the smoothing pass is `warmup`.
    const int emit_from = warmup + VAMA_SMOOTH_PERIOD - 1;
    for (int i = 0; i < emit_from && i < n; ++i) row[i] = vama_qnan();

    const double alpha = 2.0 / ((double)base_period + 1.0);
    const double beta = 1.0 - alpha;

    // ema_scalar_into (:465-470): seeded from data[first], `valid_count` 1.
    double mean = data[first];
    double ema_value = mean;
    int valid_count = 1;
    const int ema_warmup_end = ((first + base_period) < n) ? (first + base_period) : n;

    // vama_core_into (:337-348): two monotonic deques, cap == vol_period.
    // head == tail means EMPTY, which is the CPU's own convention and is why a
    // deque holding all `cap` slots reads as empty — reproduced, not repaired.
    int idx_max[VAMA_VOL_PERIOD];
    double val_max[VAMA_VOL_PERIOD];
    int head_max = 0, tail_max = 0;
    int idx_min[VAMA_VOL_PERIOD];
    double val_min[VAMA_VOL_PERIOD];
    int head_min = 0, tail_min = 0;
    const int cap = vol_period;

    // The rolling WMA over the core series — see "TWO CPU PATHS, ONE KERNEL".
    double wma_ring[VAMA_SMOOTH_PERIOD];
    double wma_sum = 0.0;
    double wma_weight_sum = 0.0;

    for (int i = first; i < n; ++i) {
        const double x = data[i];

        // ---- EMA, ema_scalar_into (:461).
        if (i == first) {
            // Already seeded above; out[first] = mean.
            ema_value = mean;
        } else if (i < ema_warmup_end) {
            if (isfinite(x)) {
                valid_count += 1;
                const double vc = (double)valid_count;
                mean = ((vc - 1.0) * mean + x) / vc;
            }
            ema_value = mean;
        } else {
            if (isfinite(x)) {
                ema_value = fma(beta, ema_value, alpha * x);
            }
        }
        const double e = ema_value;

        // ---- volatility window, vama_core_into (:353).
        const int span = i + 1 - first;
        const int window_len = (vol_period < span) ? vol_period : span;
        const int window_start = i + 1 - window_len;

        while (head_max != tail_max && idx_max[head_max] < window_start) {
            head_max += 1;
            if (head_max == cap) head_max = 0;
        }
        while (head_min != tail_min && idx_min[head_min] < window_start) {
            head_min += 1;
            if (head_min == cap) head_min = 0;
        }

        // :373 — `is_nan`, not `is_finite`. See the header.
        if (!(isnan(e) || isnan(x))) {
            const double d = x - e;

            while (head_max != tail_max) {
                const int last = (tail_max == 0) ? (cap - 1) : (tail_max - 1);
                if (val_max[last] <= d) {
                    tail_max = last;
                } else {
                    break;
                }
            }
            idx_max[tail_max] = i;
            val_max[tail_max] = d;
            tail_max += 1;
            if (tail_max == cap) tail_max = 0;

            while (head_min != tail_min) {
                const int last = (tail_min == 0) ? (cap - 1) : (tail_min - 1);
                if (val_min[last] >= d) {
                    tail_min = last;
                } else {
                    break;
                }
            }
            idx_min[tail_min] = i;
            val_min[tail_min] = d;
            tail_min += 1;
            if (tail_min == cap) tail_min = 0;
        }

        if (i < warmup) continue;

        // :403-413 — the core value the smoothing pass consumes.
        double core;
        if (isnan(e)) {
            core = vama_qnan();
        } else if (head_max != tail_max && head_min != tail_min) {
            core = fma(0.5, val_max[head_max] + val_min[head_min], e);
        } else {
            core = e;
        }

        // ---- WMA(5), wma_scalar (:305) over the core series.
        const int k = i - warmup;
        wma_ring[k % VAMA_SMOOTH_PERIOD] = core;
        if (k < VAMA_SMOOTH_PERIOD - 1) {
            wma_weight_sum += core * ((double)k + 1.0);
            wma_sum += core;
        } else {
            wma_weight_sum += core * (double)VAMA_SMOOTH_PERIOD;
            wma_sum += core;
            row[i] = wma_weight_sum / VAMA_WMA_DEN;
            const double old = wma_ring[(k + 1) % VAMA_SMOOTH_PERIOD];
            wma_weight_sum -= wma_sum;
            wma_sum -= old;
        }
    }
}
