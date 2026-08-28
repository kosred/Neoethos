#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void daily_factor_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ threshold_levels,
    int n_combos,
    double* __restrict__ out_value,
    double* __restrict__ out_ema,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    double threshold_level = threshold_levels[combo_idx];
    double* row_value = out_value + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_ema = out_ema + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = CUDART_NAN;
        row_ema[i] = CUDART_NAN;
        row_signal[i] = CUDART_NAN;
    }

    if (!isfinite(threshold_level) || threshold_level < 0.0 || threshold_level > 1.0) {
        return;
    }

    double alpha = 2.0 / 15.0;
    double prev_open = CUDART_NAN;
    double prev_high = CUDART_NAN;
    double prev_low = CUDART_NAN;
    double prev_close = CUDART_NAN;
    double prev_ema = CUDART_NAN;
    bool has_prev = false;

    for (int i = 0; i < len; ++i) {
        double o = open[i];
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!(isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c))) {
            continue;
        }

        double ema = isfinite(prev_ema) ? prev_ema + alpha * (c - prev_ema) : c;
        double value = 0.0;
        if (has_prev) {
            double range = prev_high - prev_low;
            if (isfinite(range) && range != 0.0) {
                value = fabs(prev_open - prev_close) / range;
            }
        }

        double signal = 0.0;
        if (value > threshold_level && c > ema) {
            signal = 2.0;
        } else if (value > threshold_level && c < ema) {
            signal = -2.0;
        } else if (c > ema) {
            signal = 1.0;
        } else if (c < ema) {
            signal = -1.0;
        }

        row_value[i] = value;
        row_ema[i] = ema;
        row_signal[i] = signal;

        prev_open = o;
        prev_high = h;
        prev_low = l;
        prev_close = c;
        prev_ema = ema;
        has_prev = true;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - daily_factor
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/daily_factor.rs:466
 *             `daily_factor_output_into_slice`, the `Value` arm.
 *
 * COLUMN: `value`. `OUTPUTS_DAILY_FACTOR` is [value, ema, signal]
 * (registry.rs:1460) and cpu_batch.rs:10063 maps "value" onto
 * `OutputField::Value`, so the value column is the RANGE FACTOR, not the EMA
 * and not the discrete signal.
 *
 * PERIOD-INVARIANT. The CPU batch reads `threshold_level` (0.35) and never
 * `period`. `threshold_level` only feeds the `Signal` column, so it does not
 * appear below at all.
 *
 * FIRST-VALID: `first_valid_ohlc` (:258) - open, high, low and close ALL
 * `is_finite` at the same index. Registered as
 * `F64FirstValidRule::Ohlc4AllFinite`, the same rule
 * `accumulation_swing_index` uses and NOT the `!is_nan` rule `bop` uses over
 * the same four series.
 *
 * WARMUP: `dst.fill(NAN)` then the walk starts at `first` (:479, :489).
 *
 * THE HOLE BEHAVIOUR IS `continue`, NOT NaN-AND-RESET (:494-496): an invalid
 * bar leaves `dst[i]` at its pre-filled NaN AND LEAVES `prev_*` UNTOUCHED, so
 * the next valid bar still sees the last valid bar as its predecessor and the
 * EMA carries straight across the gap. That is materially different from the
 * reset every other state machine in this shard performs, and it is
 * reproduced exactly.
 *
 * THE VALUE READS THE PREVIOUS BAR, NOT THIS ONE (:503-511):
 * `|prev_open - prev_close| / (prev_high - prev_low)`. On the first valid bar
 * `has_prev` is false and the value is 0.0, not NaN.
 *
 * `range.is_finite() && range != 0.0` is a branch, not an epsilon - a zero
 * range emits 0.0. No f64 tolerance is invented.
 *
 * The EMA step is written `prev_ema + alpha * (c - prev_ema)` (:499) - TWO
 * roundings, in that association. NOT `fma`, and NOT
 * `alpha * c + (1 - alpha) * prev_ema`; both would be different doubles.
 * `-fmad=false` in build.rs keeps the compiler from contracting it.
 *
 * SEQUENTIAL, one thread per combo column: a 14-period EMA recurrence plus a
 * one-bar lag.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void daily_factor_neo_batch_f64(
    const double* __restrict__ open,
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

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (first_valid < 0 || first_valid >= len) return;

    const double alpha = 2.0 / (14.0 + 1.0);   /* DEFAULT_EMA_PERIOD = 14, :222 */

    double prev_open = NEO_F64_NAN, prev_high = NEO_F64_NAN;
    double prev_low = NEO_F64_NAN, prev_close = NEO_F64_NAN;
    double prev_ema = NEO_F64_NAN;
    bool   has_prev = false;

    for (int i = first_valid; i < len; ++i) {
        const double op = open[i], h = high[i], l = low[i], c = close[i];
        if (!(isfinite(op) && isfinite(h) && isfinite(l) && isfinite(c))) {
            continue;                 /* leaves o[i] NaN and prev_* untouched */
        }

        const double ema = isfinite(prev_ema) ? prev_ema + alpha * (c - prev_ema)
                                              : c;
        double value;
        if (has_prev) {
            const double range = prev_high - prev_low;
            value = (isfinite(range) && range != 0.0)
                        ? fabs(prev_open - prev_close) / range
                        : 0.0;
        } else {
            value = 0.0;
        }

        o[i] = value;

        prev_open = op; prev_high = h; prev_low = l; prev_close = c;
        prev_ema = ema;
        has_prev = true;
    }
}
