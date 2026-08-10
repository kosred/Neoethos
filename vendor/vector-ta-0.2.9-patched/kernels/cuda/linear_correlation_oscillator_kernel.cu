#include <cmath>
#include <cstddef>

static __device__ inline double lco_compute_correlation_from_sums(
    double sum_y,
    double sum_y2,
    double weighted_sum,
    int period
) {
    double period_f = static_cast<double>(period);
    double inv_period = 1.0 / period_f;
    double mean_x = 0.5 * (period_f + 1.0);
    double var_x = static_cast<double>(period * period - 1) / 12.0;
    if (!(var_x > 0.0) || !isfinite(var_x)) {
        return NAN;
    }

    double centered = weighted_sum - mean_x * sum_y;
    double mean_y = sum_y * inv_period;
    double var_y = sum_y2 * inv_period - mean_y * mean_y;
    if (var_y < 0.0 && var_y > -1e-12) {
        var_y = 0.0;
    }
    if (!(var_y > 0.0) || !isfinite(var_y)) {
        return NAN;
    }

    double denom = sqrt(var_y * var_x);
    if (!(denom > 0.0) || !isfinite(denom)) {
        return NAN;
    }

    double corr = centered * inv_period / denom;
    if (!isfinite(corr)) {
        return NAN;
    }
    if (corr > 1.0) {
        return 1.0;
    }
    if (corr < -1.0) {
        return -1.0;
    }
    return corr;
}

extern "C" __global__ void linear_correlation_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    int rows,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int period = periods[row];
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (period <= 0 || period > len) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (!isnan(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    int warm = first + period + 1;
    if (warm >= len) {
        return;
    }

    for (int end = warm; end < len; ++end) {
        int start = end + 1 - period;
        double sum_y = 0.0;
        double sum_y2 = 0.0;
        double weighted_sum = 0.0;
        bool has_nan = false;

        for (int offset = 0; offset < period; ++offset) {
            double value = data[start + offset];
            if (isnan(value)) {
                has_nan = true;
                break;
            }
            double weight = static_cast<double>(offset + 1);
            sum_y += value;
            sum_y2 += value * value;
            weighted_sum += weight * value;
        }

        if (!has_nan) {
            row_out[end] = lco_compute_correlation_from_sums(sum_y, sum_y2, weighted_sum, period);
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// CPU REFERENCE
// -------------
//   src/indicators/linear_correlation_oscillator.rs
//     :367 linear_correlation_oscillator_scalar   <- the whole specification
//     :337 recompute_lco_window
//     :302 compute_correlation_from_precomputed
//     :247 prepare -- `first` is the first `!is_nan`, and `valid <= period + 1`
//                     is an Err (so the row is all NaN)
//     :447 into_slice -- NaN prefix is `[..first + period + 1]`
//   dispatch: cpu_batch.rs:3326, param `period` (default 14).
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW walking bars ascending. PERIOD-SWEPT: this is
// one of the few in this closer set whose CPU batch reads a parameter literally
// named `period`, so `periods[row]` is used.
//
// The three running sums are ADD-ON-ENTRY / SUBTRACT-ON-EXIT accumulators whose
// value depends on the order they were built in, and `weighted_sum` is updated
// USING THE OLD `sum_y` (:395-397) before `sum_y` itself moves -- so the three
// updates are not interchangeable and are written in the CPU order. There is no
// per-thread ring: the window is read straight out of `data`, so this kernel
// carries NO `max_period` bound.
//
// ARITHMETIC
// ----------
// f64 end to end, no fast-math, no f32-suffixed function. Three things are
// carried over from the CPU EXACTLY and are not "epsilons" that needed
// rewidening:
//   * `var_y < 0.0 && var_y > -1e-12` (:317) is a SIGN-REPAIR on a
//     mathematically non-negative quantity computed as a difference of two
//     f64 sums. It is already an f64-scale guard -- the f32 rule in the brief
//     is about constants sized for f32 machine epsilon (~1.19e-7), and this is
//     five orders of magnitude below that. Widening it would ACCEPT variances
//     the CPU rejects.
//   * `(sum_y * inv_period).powi(2)` is `x * x` for `powi(2)`.
//   * `corr.clamp(-1.0, 1.0)` is fmin(fmax(...)) so a NaN cannot slip past as
//     it would through an if-chain -- though the CPU guards `is_finite` first,
//     the pair is used anyway so the property does not depend on that ordering.

__device__ __forceinline__ double lco_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// :302 compute_correlation_from_precomputed
__device__ __forceinline__ double lco_neo_corr(double sum_y,
                                               double sum_y2,
                                               double weighted_sum,
                                               double inv_period,
                                               double mean_x,
                                               double var_x) {
    const double nan_d = lco_neo_qnan();
    if (var_x <= 0.0) return nan_d;

    const double centered = weighted_sum - mean_x * sum_y;
    const double m = sum_y * inv_period;
    double var_y = sum_y2 * inv_period - m * m;
    if (var_y < 0.0 && var_y > -1e-12) var_y = 0.0;
    if (var_y <= 0.0 || !isfinite(var_y)) return nan_d;

    const double denom = sqrt(var_y * var_x);
    if (denom == 0.0 || !isfinite(denom)) return nan_d;

    const double corr = centered * inv_period / denom;
    if (isfinite(corr)) return fmin(fmax(corr, -1.0), 1.0);
    return nan_d;
}

// :337 recompute_lco_window over data[start..=end]
__device__ __forceinline__ void lco_neo_recompute(const double* __restrict__ data,
                                                  int start,
                                                  int end,
                                                  double* sum_y,
                                                  double* sum_y2,
                                                  double* weighted_sum,
                                                  int* nan_count) {
    double sy = 0.0, sy2 = 0.0, ws = 0.0;
    int nc = 0;
    for (int off = 0; start + off <= end; ++off) {
        const double value = data[start + off];
        if (isnan(value)) {
            nc += 1;
        } else {
            const double weight = static_cast<double>(off + 1);
            sy += value;
            sy2 += value * value;
            ws += weight * value;
        }
    }
    *sum_y = sy; *sum_y2 = sy2; *weighted_sum = ws; *nan_count = nc;
}

extern "C" __global__ void linear_correlation_oscillator_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = lco_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) o[i] = nan_d;

    const int period = periods[row];
    if (n <= 0 || first_valid < 0 || first_valid >= n) return;
    if (period <= 0 || period > n) return;                     // :253 InvalidPeriod
    if (n - first_valid <= period + 1) return;                 // :259 NotEnoughValidData

    const double period_f = static_cast<double>(period);
    const double inv_period = 1.0 / period_f;
    const double mean_x = 0.5 * (period_f + 1.0);
    // `((period * period - 1) as f64) / 12.0` -- the product is formed in the
    // integer domain first, exactly as the CPU writes it.
    const double var_x = static_cast<double>(
        static_cast<long long>(period) * static_cast<long long>(period) - 1LL) / 12.0;

    const int start0 = first_valid + 2;
    int end = first_valid + period + 1;
    if (end >= n) return;                                      // :372

    double sum_y, sum_y2, weighted_sum;
    int nan_count;
    lco_neo_recompute(data, start0, end, &sum_y, &sum_y2, &weighted_sum, &nan_count);
    int window_start = start0;

    for (;;) {
        o[end] = (nan_count == 0)
            ? lco_neo_corr(sum_y, sum_y2, weighted_sum, inv_period, mean_x, var_x)
            : nan_d;
        if (end + 1 == n) break;

        const double old_v = data[window_start];
        const double new_v = data[end + 1];
        if (nan_count == 0 && !isnan(new_v)) {
            // ORDER IS LOAD-BEARING: `weighted_sum` consumes the OLD `sum_y`.
            weighted_sum = weighted_sum - sum_y + period_f * new_v;
            sum_y += new_v - old_v;
            sum_y2 += new_v * new_v - old_v * old_v;
        } else {
            if (isnan(old_v)) nan_count -= 1;
            if (isnan(new_v)) nan_count += 1;
            if (nan_count == 0) {
                int discard;
                lco_neo_recompute(data, window_start + 1, end + 1,
                                  &sum_y, &sum_y2, &weighted_sum, &discard);
            }
        }
        window_start += 1;
        end += 1;
    }
}
