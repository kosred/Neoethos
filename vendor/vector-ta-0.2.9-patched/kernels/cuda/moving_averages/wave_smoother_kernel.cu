// wave_smoother — CUDA f64 kernel.
//
// WHAT THIS REPLACES
// ------------------
// NOTHING. Before this file there was no `.cu` for this indicator at all, no
// wrapper, and no `F64_KERNELS` row, so `resolve_f64_kernel("wave_smoother")`
// returned `CudaF64KernelMissing` for every request.
//
// CPU REFERENCE — src/indicators/moving_averages/wave_smoother.rs
// ---------------------------------------------------------------
//   :233 prepare_input             — validation and `first`
//   :258 build_normalized_weights  — the window+1 weight vector
//   :284 smooth_value              — the 2-bar pre-smoother
//   :293 compute_wave_smoother     — the accumulation this kernel reproduces
//   :354 wave_smoother             — the entry the brief names
//   :358 wave_smoother_with_kernel — NaN prefix = `first`, weights, compute
//
// THE PARAMETERS THAT ARE NOT IN THE LANE ABI
// -------------------------------------------
// `phase` is not a period, so the sweep request cannot carry it. The CPU
// default is DEFAULT_PHASE = 70.0 (:32) and that is what this kernel uses —
// the same rule tilson's `v_factor == 0` lane follows. `period` comes from the
// swept `periods[]`.
//
// SHAPE — ONE THREAD PER COLUMN, BARS ASCENDING
// ---------------------------------------------
// The pre-smoother `smooth_value` reads bar i and bar i-1, and the weighted
// sum reads the last `period + 1` smoothed values, so a bar is a window over
// smoothed values rather than over raw ones. That is still expressible per
// column, and it is written that way: the CPU's ring buffer is NOT emulated,
// because every value it can hold is recomputable in closed form from `data`:
//
//   ring holds smooth(j) for j in [idx - count + 1, idx], count = min(idx - first + 1, window)
//   smooth(j)  = NaN                                    when data[j] is not finite
//              = 0.5 * (data[j] + 0.0)                  when j == first
//              = 0.5 * (data[j] + data[j-1])            when data[j-1] is finite
//              = 0.5 * (data[j] + 0.0)                  otherwise
//
// so `lag < count` is exactly `idx - lag >= first`. Dropping the ring removes
// a `period + 1` local array WITHOUT changing a single rounding: the values
// summed, and the order they are summed in, are identical.
//
// THE ONE LOCAL ARRAY THAT REMAINS, AND ITS CAP
// ---------------------------------------------
// The weights cannot be recomputed inside the lag loop: that would run
// `2 * (period + 1)` sin/cos per BAR instead of per series. They are built
// once per thread into a local array, which forces a compile-time bound.
// WS_MAX_PERIOD is 512 and the array is one longer, because the window is
// `period + 1`.
//
// The bound is REFUSED BY NAME on the host: `F64Kernel::max_period` returns
// `WS_MAX_PERIOD` and `CudaF64Indicators::sweep` (:3864) answers
// `PeriodTooLarge { indicator, period, max }` before any launch. The in-kernel
// guard below is the second lock on the same door — if the two constants ever
// drift, the kernel writes NaN instead of overrunning a local array.
//
// ARITHMETIC
// ----------
// f64 end to end. No f32 literal, no f32-suffixed math function, no fast-math
// intrinsic; the file is listed in `F64_LANE_SOURCES` in build.rs so the whole
// translation unit is compiled `-prec-div=true -prec-sqrt=true -fmad=false
// -ftz=false` and NEVER with `--use_fast_math`.
//
// The epsilon is `f64::EPSILON` (:272), which is DBL_EPSILON — it is already
// the f64-sized constant, not an f32 one carried over, so it is used as is.
//
// `to_radians` is `self * (PI / 180.0)` in core, one multiply by a
// compile-time constant; written the same way here so the two agree bit for
// bit rather than merely in value.

#include <cmath>
#include <cfloat>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The CPU default, wave_smoother.rs:33.
#define WS_DEFAULT_PHASE 70.0

// See "THE ONE LOCAL ARRAY THAT REMAINS" above. MUST equal
// `neoethos_f64_wrapper::WS_MAX_PERIOD`.
#define WS_MAX_PERIOD 512

__device__ __forceinline__ double ws_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// smooth_value (:284). `prev_raw` is the RAW previous bar, not the smoothed
// one, and a non-finite `prev_raw` contributes 0.0 rather than poisoning the
// value — that is the CPU's `if prev_raw.is_finite() { prev_raw } else { 0.0 }`.
__device__ __forceinline__ double ws_smooth(double value, double prev_raw) {
    if (isfinite(value)) {
        return 0.5 * (value + (isfinite(prev_raw) ? prev_raw : 0.0));
    }
    return ws_qnan();
}

extern "C" __global__ void wave_smoother_neo_batch_f64(
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
    const int period = periods[r];
    const int window = period + 1;

    // prepare_input (:233): EmptyInputData, AllValuesNaN (via first_valid < 0
    // from the caller's scan) and InvalidPeriod. `phase` is the CPU default so
    // validate_phase (:225) cannot fail. The window cap is this kernel's own
    // limit and is declared, not hidden — see the header.
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period <= 0) ||
        (period > WS_MAX_PERIOD);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = ws_qnan();
        return;
    }

    // build_normalized_weights (:258). The sum is accumulated ASCENDING, one
    // term at a time, exactly as the CPU loop does.
    double weights[WS_MAX_PERIOD + 1];
    const double phase_rad = WS_DEFAULT_PHASE * (M_PI / 180.0);
    const double period_f = (double)period;
    double sum = 0.0;
    for (int i = 0; i < window; ++i) {
        const double idx = (double)i;
        const double value =
            sin(idx * M_PI / period_f + phase_rad) * cos(idx * M_PI / (2.0 * period_f));
        weights[i] = value;
        sum += value;
    }

    // :271 — `!sum.is_finite() || sum.abs() <= f64::EPSILON` returns None, and
    // the `let Some(weights) = weights else` arm (:329) then writes NaN at
    // every bar from `first` on. The prefix below `first` is NaN either way.
    const bool no_weights = (!isfinite(sum)) || (fabs(sum) <= DBL_EPSILON);
    if (no_weights) {
        for (int i = 0; i < n; ++i) row[i] = ws_qnan();
        return;
    }

    const double inv = 1.0 / sum;
    for (int i = 0; i < window; ++i) {
        weights[i] = weights[i] * inv;
    }

    // compute_wave_smoother (:305) fills `out[..first]` with NaN.
    for (int i = 0; i < first_valid && i < n; ++i) row[i] = ws_qnan();

    // `first_nz` (:311) is the first FINITE smoothed value. `first_valid` is
    // the first index at which `data` is finite (prepare_input :242 scans with
    // `is_finite`, not `!is_nan`), and at that index `prev_raw` is still NaN,
    // so the smoothed value is 0.5 * (data[first] + 0.0) — finite, and
    // therefore `first_nz` is settled on the very first iteration and never
    // changes. Computing it up front is the same number, not an assumption.
    const double first_fill = ws_smooth(data[first_valid], ws_qnan());

    for (int idx = first_valid; idx < n; ++idx) {
        double acc = 0.0;
        for (int lag = 0; lag < window; ++lag) {
            const int j = idx - lag;
            // `lag < count` (:337) with count = min(idx - first + 1, window).
            double hist;
            if (j >= first_valid) {
                const double prev_raw = (j > first_valid) ? data[j - 1] : ws_qnan();
                hist = ws_smooth(data[j], prev_raw);
            } else {
                hist = first_fill;
            }
            // :344 — a non-finite history slot is replaced by `first_fill`,
            // NOT skipped, so the weight is still applied.
            acc += (isfinite(hist) ? hist : first_fill) * weights[lag];
        }
        row[idx] = acc;
    }
}
