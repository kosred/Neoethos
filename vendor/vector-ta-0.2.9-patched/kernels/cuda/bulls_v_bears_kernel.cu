#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_EMA = 0;
constexpr int MA_SMA = 1;
constexpr int MA_WMA = 2;
constexpr int METHOD_NORMALIZED = 0;
constexpr int METHOD_RAW = 1;

__device__ inline bool finite3(double a, double b, double c) {
    return isfinite(a) && isfinite(b) && isfinite(c);
}
}

extern "C" __global__ void bulls_v_bears_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* periods,
    const int* normalized_bars_backs,
    const int* raw_rolling_periods,
    const double* raw_threshold_percentiles,
    const double* threshold_levels,
    const int* ma_types,
    const int* calculation_methods,
    int rows,
    double* out_value,
    double* out_bull,
    double* out_bear,
    double* out_ma,
    double* out_upper,
    double* out_lower,
    double* out_bullish_signal,
    double* out_bearish_signal,
    double* out_zero_cross_up,
    double* out_zero_cross_down
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int period = periods[row];
    const int normalized_bars_back = normalized_bars_backs[row];
    const int raw_rolling_period = raw_rolling_periods[row];
    const double raw_threshold_percentile = raw_threshold_percentiles[row];
    const double threshold_level = threshold_levels[row];
    const int ma_type = ma_types[row];
    const int calculation_method = calculation_methods[row];

    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bull = out_bull + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear = out_bear + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_signal =
        out_bullish_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_signal =
        out_bearish_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_zero_cross_up =
        out_zero_cross_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_zero_cross_down =
        out_zero_cross_down + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = NAN;
        row_bull[i] = NAN;
        row_bear[i] = NAN;
        row_ma[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_bullish_signal[i] = NAN;
        row_bearish_signal[i] = NAN;
        row_zero_cross_up[i] = NAN;
        row_zero_cross_down[i] = NAN;
    }

    if (period <= 0 || normalized_bars_back <= 0 || raw_rolling_period <= 0
        || !isfinite(raw_threshold_percentile) || raw_threshold_percentile < 80.0
        || raw_threshold_percentile > 99.0 || !isfinite(threshold_level)
        || threshold_level < 0.0 || threshold_level > 100.0
        || (ma_type != MA_EMA && ma_type != MA_SMA && ma_type != MA_WMA)
        || (calculation_method != METHOD_NORMALIZED && calculation_method != METHOD_RAW)) {
        return;
    }

    const double ema_alpha = 2.0 / (static_cast<double>(period) + 1.0);
    double ema_prev = NAN;
    double window_sum = 0.0;
    int finite_count = 0;
    double wma_weighted = 0.0;
    bool wma_prev_full_valid = false;
    double prev_total = NAN;
    int segment_start = 0;

    for (int i = 0; i < len; ++i) {
        const double c = close[i];

        if (ma_type == MA_EMA) {
            if (!isfinite(c)) {
                ema_prev = NAN;
                row_ma[i] = NAN;
            } else {
                ema_prev = isfinite(ema_prev) ? (ema_prev + ema_alpha * (c - ema_prev)) : c;
                row_ma[i] = ema_prev;
            }
        } else if (ma_type == MA_SMA) {
            if (isfinite(c)) {
                window_sum += c;
                finite_count += 1;
            }
            if (i >= period) {
                const double old = close[i - period];
                if (isfinite(old)) {
                    window_sum -= old;
                    finite_count -= 1;
                }
            }
            if (i + 1 >= period && finite_count == period) {
                row_ma[i] = window_sum / static_cast<double>(period);
            }
        } else {
            const double old_window_sum = window_sum;
            const bool popped = i >= period;
            if (popped) {
                const double old = close[i - period];
                if (isfinite(old)) {
                    window_sum -= old;
                    finite_count -= 1;
                }
            }
            if (isfinite(c)) {
                window_sum += c;
                finite_count += 1;
            }
            const bool full_valid = i + 1 >= period && finite_count == period;
            if (full_valid) {
                if (wma_prev_full_valid && popped && isfinite(c)) {
                    wma_weighted = wma_weighted
                        + static_cast<double>(period) * c
                        - old_window_sum;
                } else {
                    wma_weighted = 0.0;
                    const int start = i + 1 - period;
                    for (int j = start; j <= i; ++j) {
                        wma_weighted +=
                            close[j] * static_cast<double>(j - start + 1);
                    }
                }
                const double denominator =
                    static_cast<double>(period) * (static_cast<double>(period) + 1.0) / 2.0;
                row_ma[i] = wma_weighted / denominator;
                wma_prev_full_valid = true;
            } else {
                wma_weighted = 0.0;
                wma_prev_full_valid = false;
            }
        }

        if (finite3(high[i], low[i], row_ma[i])) {
            row_bull[i] = high[i] - row_ma[i];
            row_bear[i] = row_ma[i] - low[i];
        }

        if (calculation_method == METHOD_NORMALIZED) {
            row_upper[i] = threshold_level;
            row_lower[i] = -threshold_level;
        }

        if (!(isfinite(row_bull[i]) && isfinite(row_bear[i]))) {
            segment_start = i + 1;
            prev_total = NAN;
            continue;
        }

        if (calculation_method == METHOD_NORMALIZED) {
            const int window_start =
                (i + 1 > normalized_bars_back) ? (i + 1 - normalized_bars_back) : 0;
            const int start = window_start > segment_start ? window_start : segment_start;
            double bull_min = NAN;
            double bull_max = NAN;
            double bear_min = NAN;
            double bear_max = NAN;
            for (int j = start; j <= i; ++j) {
                const double bull = row_bull[j];
                const double bear = row_bear[j];
                if (isfinite(bull)) {
                    bull_min = isfinite(bull_min) ? fmin(bull_min, bull) : bull;
                    bull_max = isfinite(bull_max) ? fmax(bull_max, bull) : bull;
                }
                if (isfinite(bear)) {
                    bear_min = isfinite(bear_min) ? fmin(bear_min, bear) : bear;
                    bear_max = isfinite(bear_max) ? fmax(bear_max, bear) : bear;
                }
            }
            const double bull_range = bull_max - bull_min;
            const double bear_range = bear_max - bear_min;
            if (bull_range > 0.0 && bear_range > 0.0) {
                const double norm_bull = ((row_bull[i] - bull_min) / bull_range - 0.5) * 100.0;
                const double norm_bear = ((row_bear[i] - bear_min) / bear_range - 0.5) * 100.0;
                row_value[i] = norm_bull - norm_bear;
            }
        } else {
            row_value[i] = row_bull[i] - row_bear[i];

            const int window_start =
                (i + 1 > raw_rolling_period) ? (i + 1 - raw_rolling_period) : 0;
            const int start = window_start > segment_start ? window_start : segment_start;
            double lowest = NAN;
            double highest = NAN;
            for (int j = start; j <= i; ++j) {
                const double value = row_value[j];
                if (isfinite(value)) {
                    lowest = isfinite(lowest) ? fmin(lowest, value) : value;
                    highest = isfinite(highest) ? fmax(highest, value) : value;
                }
            }
            if (isfinite(lowest) && isfinite(highest)) {
                const double range = highest - lowest;
                row_upper[i] = lowest + range * (raw_threshold_percentile / 100.0);
                row_lower[i] = lowest + range * ((100.0 - raw_threshold_percentile) / 100.0);
            }
        }

        if (isfinite(row_value[i]) && isfinite(row_upper[i]) && isfinite(row_lower[i])) {
            row_bullish_signal[i] = row_value[i] > row_upper[i] ? 1.0 : 0.0;
            row_bearish_signal[i] = row_value[i] < row_lower[i] ? 1.0 : 0.0;
            row_zero_cross_up[i] =
                isfinite(prev_total) && row_value[i] > 0.0 && prev_total <= 0.0 ? 1.0 : 0.0;
            row_zero_cross_down[i] =
                isfinite(prev_total) && row_value[i] < 0.0 && prev_total >= 0.0 ? 1.0 : 0.0;
            prev_total = row_value[i];
        } else {
            prev_total = NAN;
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/bulls_v_bears.rs:852
// (bulls_v_bears_with_kernel). The column this emits is value
// (dispatch/cpu_batch.rs:11121-11122).
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- the moving
// average at the default ma_type is an EMA whose state is carried and RESET to
// NaN on a non-finite close, and the normalisation window is a rolling
// min/max over the bull and bear series the same pass produces.
//
// PERIOD-SWEPT, unlike most of this closer's kernels: compute_bulls_v_bears_
// batch (cpu_batch.rs:11153) reads a parameter literally named period
// (default 14), so periods[combo] binds to it. The other six parameters --
// ma_type "ema", calculation_method "normalized", normalized_bars_back 120,
// raw_rolling_period 50, raw_threshold_percentile 95.0, threshold_level 80.0
// (:11154-11175) -- are not swept and are pinned at their CPU defaults.
//
// ONLY THE NORMALIZED BRANCH IS REACHABLE AT THE DEFAULTS, and it is the only
// one written here: calculation_method is pinned to "normalized", so the raw
// branch could not run. It is named rather than silently dropped so a later
// reader knows this kernel serves the default configuration and not the
// indicator's whole parameter space -- exactly the contract the ten
// period-invariant shard-4 variants carry.
//
// THE TWO RINGS ARE PER-THREAD, so their depth is a property of THIS COMPILED
// KERNEL. The CPU rescans the last normalized_bars_back values of bull and
// bear at every bar; a ring of that depth holds exactly what is reachable, and
// nothing older is ever read. At the pinned 120 the bound below cannot be
// approached; it is checked rather than assumed.
//
// FIRST VALID IS NOT READ: the CPU emits from bar 0, seeds the EMA at the
// first finite close and resets it at every non-finite one, so there is no
// warmup index. The lane row declares F64FirstValidRule::Ignored.
//
// NaN CANNOT SURVIVE: every min/max update is guarded by isfinite on BOTH the
// incoming value and the running extreme before fmin/fmax is called, which is
// what the CPU does -- a bare comparison chain would let a NaN bull value sit
// in bull_min forever and poison every later bar.
//
// f64 END TO END: double literals, double fmin/fmax, no f32-suffixed math
// function, no fast-math intrinsic, and no epsilon -- the range test is the
// CPU's exact > 0.0.
// ---------------------------------------------------------------------------

#define NEO_BVB_NORMALIZED_BARS_BACK 120
#define NEO_BVB_RAW_ROLLING_PERIOD 50
#define NEO_BVB_RAW_THRESHOLD_PERCENTILE 95.0
#define NEO_BVB_THRESHOLD_LEVEL 80.0
#define NEO_BVB_MAX_BARS_BACK 512

extern "C" __global__ void bulls_v_bears_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row_idx >= n_combos || n <= 0) {
        return;
    }
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int period = periods[row_idx];
    const int normalized_bars_back = NEO_BVB_NORMALIZED_BARS_BACK;
    const int raw_rolling_period = NEO_BVB_RAW_ROLLING_PERIOD;
    const double raw_threshold_percentile = NEO_BVB_RAW_THRESHOLD_PERCENTILE;
    const double threshold_level = NEO_BVB_THRESHOLD_LEVEL;

    if (period <= 0 || normalized_bars_back <= 0 || raw_rolling_period <= 0 ||
        !isfinite(raw_threshold_percentile) || raw_threshold_percentile < 80.0 ||
        raw_threshold_percentile > 99.0 || !isfinite(threshold_level) ||
        threshold_level < 0.0 || threshold_level > 100.0 ||
        normalized_bars_back > NEO_BVB_MAX_BARS_BACK) {
        return;
    }

    double bull_ring[NEO_BVB_MAX_BARS_BACK];
    double bear_ring[NEO_BVB_MAX_BARS_BACK];
    for (int j = 0; j < normalized_bars_back; ++j) {
        bull_ring[j] = NAN;
        bear_ring[j] = NAN;
    }

    const double ema_alpha = 2.0 / (static_cast<double>(period) + 1.0);
    double ema_prev = NAN;
    int segment_start = 0;

    for (int i = 0; i < n; ++i) {
        const double c = close[i];

        double ma_value = NAN;
        if (!isfinite(c)) {
            ema_prev = NAN;
        } else {
            ema_prev = isfinite(ema_prev) ? (ema_prev + ema_alpha * (c - ema_prev)) : c;
            ma_value = ema_prev;
        }

        double bull = NAN;
        double bear = NAN;
        if (finite3(high[i], low[i], ma_value)) {
            bull = high[i] - ma_value;
            bear = ma_value - low[i];
        }
        bull_ring[i % normalized_bars_back] = bull;
        bear_ring[i % normalized_bars_back] = bear;

        if (!(isfinite(bull) && isfinite(bear))) {
            segment_start = i + 1;
            continue;
        }

        const int window_start =
            (i + 1 > normalized_bars_back) ? (i + 1 - normalized_bars_back) : 0;
        const int start = window_start > segment_start ? window_start : segment_start;
        double bull_min = NAN;
        double bull_max = NAN;
        double bear_min = NAN;
        double bear_max = NAN;
        for (int j = start; j <= i; ++j) {
            const double b = bull_ring[j % normalized_bars_back];
            const double s = bear_ring[j % normalized_bars_back];
            if (isfinite(b)) {
                bull_min = isfinite(bull_min) ? fmin(bull_min, b) : b;
                bull_max = isfinite(bull_max) ? fmax(bull_max, b) : b;
            }
            if (isfinite(s)) {
                bear_min = isfinite(bear_min) ? fmin(bear_min, s) : s;
                bear_max = isfinite(bear_max) ? fmax(bear_max, s) : s;
            }
        }

        const double bull_range = bull_max - bull_min;
        const double bear_range = bear_max - bear_min;
        if (bull_range > 0.0 && bear_range > 0.0) {
            const double norm_bull = ((bull - bull_min) / bull_range - 0.5) * 100.0;
            const double norm_bear = ((bear - bear_min) / bear_range - 0.5) * 100.0;
            row[i] = norm_bull - norm_bear;
        }
    }
}
