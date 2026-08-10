// logarithmic_moving_average — CUDA f64 kernel.
//
// WHAT THIS REPLACES
// ------------------
// NOTHING. No `.cu`, no wrapper, no `F64_KERNELS` row: the lane answered
// `CudaF64KernelMissing`.
//
// CPU REFERENCE — src/indicators/moving_averages/logarithmic_moving_average.rs
// -----------------------------------------------------------------------------
//   :616 logarithmic_moving_average   — the entry the brief names
//  :1273 logarithmic_moving_average_with_kernel
//  :1147 logarithmic_moving_average_compute_into  — `out_lma.fill(NAN)` then
//                                                    compute_lma
//   :758 compute_weights              — the weight vector and its total
//   :775 compute_lma                  — THE FUNCTION THIS KERNEL REPRODUCES
//   :623 longest_finite_run           — the rejection at :700
//   :648 prepare_input / :710 prepare_param_values
//
// WHICH OUTPUT
// ------------
// Four outputs (lma, signal, position, momentum_confirmed) and the PRIMARY is
// `lma` — :1165 fills it from `compute_lma` alone, and everything else is
// derived from it. This kernel writes `lma`. The signal/position/confirmation
// columns are a separate contract (thresholds, a second smoothing choice and
// an ma_type that can demand volume) and are not what a period sweep asks for.
//
// THE PARAMETERS THAT ARE NOT IN THE LANE ABI
// -------------------------------------------
// `steepness` is not a period. The CPU default is DEFAULT_STEEPNESS = 2.5
// (:31), and that is what the weights are built from here. `period` is swept.
// The other defaults (smooth = 10 :33, ma_type "ema" :32, momentum_weight,
// thresholds) matter only through the validation gate, which is reproduced.
//
// SHAPE — ONE THREAD PER COLUMN, NOT BAR-PARALLEL
// -----------------------------------------------
// The brief allows bar-parallel here "only if the CPU sums the window in the
// same association". It does not: :801-819 sums the window in 8-WIDE CHUNKS
//
//     acc += p[i-k]*w[k] + p[i-k-1]*w[k+1] + ... + p[i-k-7]*w[k+7]
//
// which is a left-to-right chain of eight products folded into `acc`, then a
// one-at-a-time tail. That association is reproduced exactly, and a `run`
// counter carries across bars (:786-794) — a finite run shorter than `period`
// emits nothing and a non-finite bar resets it — so the column is walked by one
// thread in ascending bar order.
//
// THE ONE LOCAL ARRAY, AND ITS CAP
// --------------------------------
// The weights depend only on `period` and `steepness`, so they are built once
// per thread rather than per bar — `1 / ln(max(i + steepness, 2))^2` is a
// logarithm per slot and rebuilding them inside the bar loop would run one per
// window position per bar. That forces a compile-time bound: LMA_MAX_PERIOD is
// 512.
//
// The bound is REFUSED BY NAME on the host: `F64Kernel::max_period` returns
// `LMA_MAX_PERIOD` and `CudaF64Indicators::sweep` (:3864) answers
// `PeriodTooLarge { indicator, period, max }` before any launch. The in-kernel
// guard below is the second lock on the same door — if the two constants ever
// drift, the kernel writes NaN instead of overrunning a local array.
//
// ARITHMETIC
// ----------
// `log`, never `logf`. `powi(2)` on an f64 is `x * x` and is written that way.
// `max(2.0)` is `f64::max`, so `fmax` — the NON-NaN operand wins, which an
// `if` chain would not honour.
//
// f64 end to end; no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. Listed in `F64_LANE_SOURCES`, so never `--use_fast_math`; that
// matters here because a fast reciprocal would move `acc / total_weight` on
// every emitted bar.

#include <cmath>
#include <cstdint>

// logarithmic_moving_average.rs:31, :33
#define LMA_DEFAULT_STEEPNESS 2.5
#define LMA_DEFAULT_SMOOTH 10

// See "THE ONE LOCAL ARRAY, AND ITS CAP" above. MUST equal
// `neoethos_f64_wrapper::LMA_MAX_PERIOD`.
#define LMA_MAX_PERIOD 512

__device__ __forceinline__ double lma_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void logarithmic_moving_average_neo_batch_f64(
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

    // prepare_param_values (:710): period == 0 and smooth == 0 are Err; the
    // steepness / momentum_weight / threshold checks pass on the defaults.
    // prepare_input (:648): EmptyInputData, AllValuesNaN, period > len,
    // smooth > len.
    const bool declined =
        (n <= 0) ||
        (period <= 0) ||
        (period > n) ||
        (LMA_DEFAULT_SMOOTH > n) ||
        (period > LMA_MAX_PERIOD);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = lma_qnan();
        return;
    }

    // :663 — `prices.iter().all(|x| x.is_nan())` is AllValuesNaN. Note this is
    // `is_nan`, not `is_finite`: an all-INFINITE series passes this gate and is
    // then rejected by the run check below, which scans with `is_finite`.
    {
        bool any_non_nan = false;
        for (int i = 0; i < n; ++i) {
            if (!isnan(prices[i])) { any_non_nan = true; break; }
        }
        if (!any_non_nan) {
            for (int i = 0; i < n; ++i) row[i] = lma_qnan();
            return;
        }
    }

    // longest_finite_run (:623) < period is NotEnoughValidData (:700), which
    // means the CPU produces no series at all.
    {
        int cur = 0;
        bool ok = false;
        for (int i = 0; i < n; ++i) {
            if (isfinite(prices[i])) {
                cur += 1;
                if (cur >= period) { ok = true; break; }
            } else {
                cur = 0;
            }
        }
        if (!ok) {
            for (int i = 0; i < n; ++i) row[i] = lma_qnan();
            return;
        }
    }

    // compute_weights (:758). `total` is accumulated ASCENDING, one term at a
    // time, exactly as the CPU loop does.
    double weights[LMA_MAX_PERIOD];
    double total_weight = 0.0;
    for (int i = 0; i < period; ++i) {
        const double log_arg = fmax((double)i + LMA_DEFAULT_STEEPNESS, 2.0);
        const double l = log(log_arg);
        const double weight = 1.0 / (l * l);
        weights[i] = weight;
        total_weight += weight;
    }

    // :1156 — `out_lma.fill(f64::NAN)` before compute_lma writes anything, so
    // every bar the run gate skips stays NaN. `first_valid` plays no part: the
    // CPU walks from index 0 with a run counter, which is why the row is
    // registered F64FirstValidRule::Ignored.
    (void)first_valid;
    for (int i = 0; i < n; ++i) row[i] = lma_qnan();

    int run = 0;
    for (int i = 0; i < n; ++i) {
        const double price = prices[i];
        if (isfinite(price)) {
            run += 1;
        } else {
            run = 0;
            continue;
        }
        if (run < period) continue;

        double acc = 0.0;
        int k = 0;
        // :801-816 — eight products folded LEFT TO RIGHT into one value, which
        // is then added to `acc`. Writing this as eight separate `acc +=`
        // statements would be a DIFFERENT summation tree.
        while (k + 7 < period) {
            acc += prices[i - k] * weights[k]
                 + prices[i - k - 1] * weights[k + 1]
                 + prices[i - k - 2] * weights[k + 2]
                 + prices[i - k - 3] * weights[k + 3]
                 + prices[i - k - 4] * weights[k + 4]
                 + prices[i - k - 5] * weights[k + 5]
                 + prices[i - k - 6] * weights[k + 6]
                 + prices[i - k - 7] * weights[k + 7];
            k += 8;
        }
        // :817-820 — the tail, one at a time.
        while (k < period) {
            acc += prices[i - k] * weights[k];
            k += 1;
        }
        row[i] = acc / total_weight;
    }
}
