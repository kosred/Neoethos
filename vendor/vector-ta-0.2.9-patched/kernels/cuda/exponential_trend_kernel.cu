#include <cmath>
#include <cstddef>

static __device__ inline void exponential_trend_reset_state(
    int* atr_count,
    double* atr_sum,
    double* atr_value,
    double* atr_prev_close,
    bool* atr_have_prev_close,
    double* prev_upper_band,
    double* prev_lower_band,
    double* prev_close,
    bool* prev_atr_ready,
    double* initial_line,
    double* prev_initial_line,
    int* trend,
    int* bars_since_change,
    int* segment_index
) {
    *atr_count = 0;
    *atr_sum = 0.0;
    *atr_value = NAN;
    *atr_prev_close = NAN;
    *atr_have_prev_close = false;
    *prev_upper_band = NAN;
    *prev_lower_band = NAN;
    *prev_close = NAN;
    *prev_atr_ready = false;
    *initial_line = 0.0;
    *prev_initial_line = 0.0;
    *trend = 0;
    *bars_since_change = 0;
    *segment_index = 0;
}

extern "C" __global__ void exponential_trend_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ exp_rates,
    const double* __restrict__ initial_distances,
    const double* __restrict__ width_multipliers,
    int rows,
    double* __restrict__ out_uptrend_base,
    double* __restrict__ out_downtrend_base,
    double* __restrict__ out_uptrend_extension,
    double* __restrict__ out_downtrend_extension,
    double* __restrict__ out_bullish_change,
    double* __restrict__ out_bearish_change
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double exp_rate = exp_rates[row];
    double initial_distance = initial_distances[row];
    double width_multiplier = width_multipliers[row];

    double* row_out_uptrend_base =
        out_uptrend_base + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_downtrend_base =
        out_downtrend_base + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_uptrend_extension =
        out_uptrend_extension + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_downtrend_extension =
        out_downtrend_extension + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bullish_change =
        out_bullish_change + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bearish_change =
        out_bearish_change + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_uptrend_base[i] = NAN;
        row_out_downtrend_base[i] = NAN;
        row_out_uptrend_extension[i] = NAN;
        row_out_downtrend_extension[i] = NAN;
        row_out_bullish_change[i] = NAN;
        row_out_bearish_change[i] = NAN;
    }

    if (!isfinite(exp_rate) || exp_rate < 0.0 || exp_rate > 0.5 ||
        !isfinite(initial_distance) || initial_distance < 0.1 ||
        !isfinite(width_multiplier) || width_multiplier < 0.1) {
        return;
    }

    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = NAN;
    double atr_prev_close = NAN;
    bool atr_have_prev_close = false;
    double prev_upper_band = NAN;
    double prev_lower_band = NAN;
    double prev_close = NAN;
    bool prev_atr_ready = false;
    double initial_line = 0.0;
    double prev_initial_line = 0.0;
    int trend = 0;
    int bars_since_change = 0;
    int segment_index = 0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            exponential_trend_reset_state(
                &atr_count,
                &atr_sum,
                &atr_value,
                &atr_prev_close,
                &atr_have_prev_close,
                &prev_upper_band,
                &prev_lower_band,
                &prev_close,
                &prev_atr_ready,
                &initial_line,
                &prev_initial_line,
                &trend,
                &bars_since_change,
                &segment_index
            );
            continue;
        }

        double tr_prev_close = atr_have_prev_close ? atr_prev_close : c;
        double tr = fmax(h - l, fmax(fabs(h - tr_prev_close), fabs(l - tr_prev_close)));
        atr_prev_close = c;
        atr_have_prev_close = true;

        bool atr_ready = false;
        if (atr_count < 14) {
            atr_count += 1;
            atr_sum += tr;
            if (atr_count == 14) {
                atr_value = atr_sum / 14.0;
                atr_ready = true;
            }
        } else {
            atr_value = ((atr_value * 13.0) + tr) / 14.0;
            atr_ready = true;
        }

        double upper = NAN;
        double lower = NAN;
        int direction = 1;

        if (atr_ready) {
            double src = (h + l) * 0.5;
            double raw_upper = src + initial_distance * atr_value;
            double raw_lower = src - initial_distance * atr_value;
            double prev_lower = isfinite(prev_lower_band) ? prev_lower_band : 0.0;
            double prev_upper = isfinite(prev_upper_band) ? prev_upper_band : 0.0;
            double prev_close_value = isfinite(prev_close) ? prev_close : c;

            lower = (raw_lower > prev_lower || prev_close_value < prev_lower) ? raw_lower : prev_lower;
            upper = (raw_upper < prev_upper || prev_close_value > prev_upper) ? raw_upper : prev_upper;
            direction = !prev_atr_ready ? 1 : ((c < lower) ? 1 : -1);
        }

        int prev_trend = trend;
        double saved_prev_initial = prev_initial_line;
        double saved_prev_close = prev_close;

        if (segment_index == 100 && isfinite(upper) && isfinite(lower)) {
            if (direction < 0) {
                initial_line = lower;
                trend = 1;
            } else {
                initial_line = upper;
                trend = -1;
            }
        }

        bool crossover = isfinite(initial_line) && isfinite(saved_prev_close) &&
            isfinite(saved_prev_initial) && c > initial_line && saved_prev_close <= saved_prev_initial;
        bool crossunder = isfinite(initial_line) && isfinite(saved_prev_close) &&
            isfinite(saved_prev_initial) && c < initial_line && saved_prev_close >= saved_prev_initial;

        if (crossover && isfinite(lower)) {
            initial_line = lower;
            trend = 1;
        } else if (crossunder && isfinite(upper)) {
            initial_line = upper;
            trend = -1;
        }

        if (trend != prev_trend) {
            bars_since_change = 0;
        } else if (trend != 0) {
            bars_since_change += 1;
        }

        if (trend != 0) {
            double exp_multiplier =
                1.0 + static_cast<double>(trend) *
                (1.0 - exp(-exp_rate * static_cast<double>(bars_since_change)));
            if (exp_multiplier > 900.0) {
                exp_multiplier = 900.0;
            }
            initial_line *= exp_multiplier;
        }

        if (atr_ready) {
            double extension = initial_line +
                ((trend > 0) ? atr_value * width_multiplier : -atr_value * width_multiplier);

            if (trend == 1) {
                row_out_uptrend_base[i] = initial_line;
                row_out_uptrend_extension[i] = extension;
            } else if (trend == -1) {
                row_out_downtrend_base[i] = initial_line;
                row_out_downtrend_extension[i] = extension;
            }

            if (crossover) {
                row_out_bullish_change[i] = initial_line - atr_value;
            }
            if (crossunder) {
                row_out_bearish_change[i] = initial_line + atr_value;
            }
        }

        prev_upper_band = upper;
        prev_lower_band = lower;
        prev_close = c;
        prev_initial_line = initial_line;
        prev_atr_ready = atr_ready;
        segment_index += 1;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/exponential_trend.rs:864
// (exponential_trend_with_kernel). The column this emits is uptrend_base,
// which is what output_id == "value" resolves to (dispatch/cpu_batch.rs:4425).
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- a Wilder
// ATR recurrence (mean seed over 14 true ranges, then (atr*13 + tr)/14), a
// ratcheting band pair that reads its own previous value, a trend STATE
// MACHINE, and a bars-since-change counter that drives an exponential
// multiplier. Every one of those is carried, and the CPU RESETS all of them on
// a non-finite bar.
//
// PERIOD-INVARIANT. compute_exponential_trend_batch (cpu_batch.rs:4404-4408)
// reads exp_rate, initial_distance and width_multiplier and NEVER period, so
// five swept periods give five identical CPU columns and this kernel emits
// five identical rows. All three CPU defaults are pinned below.
//
// FIRST VALID IS NOT READ: the CPU walks from bar 0 and RESTARTS its state at
// every non-finite bar, and the 14-bar ATR seed is counted from the restart,
// not from a global warmup index. The lane row declares
// F64FirstValidRule::Ignored.
//
// NaN CANNOT SURVIVE: the true range uses fmax (which is what the CPU's
// f64::max is), and every band comparison is reached only after an explicit
// isfinite guard that substitutes the CPU's nz() value -- so no comparison
// against NaN can silently take the false branch and poison the recurrence.
//
// f64 END TO END: double literals, double exp/fmax/fabs, no f32-suffixed math
// function and no fast-math intrinsic. The 900.0 clamp on the exponential
// multiplier is the CPU's own saturation bound, not a precision guard.
// ---------------------------------------------------------------------------

#define NEO_EXPTREND_EXP_RATE 0.00003
#define NEO_EXPTREND_INITIAL_DISTANCE 4.0
#define NEO_EXPTREND_WIDTH_MULTIPLIER 1.0

extern "C" __global__ void exponential_trend_neo_batch_f64(
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
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const double exp_rate = NEO_EXPTREND_EXP_RATE;
    const double initial_distance = NEO_EXPTREND_INITIAL_DISTANCE;
    const double width_multiplier = NEO_EXPTREND_WIDTH_MULTIPLIER;

    if (!isfinite(exp_rate) || exp_rate < 0.0 || exp_rate > 0.5 ||
        !isfinite(initial_distance) || initial_distance < 0.1 ||
        !isfinite(width_multiplier) || width_multiplier < 0.1) {
        return;
    }

    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = NAN;
    double atr_prev_close = NAN;
    bool atr_have_prev_close = false;
    double prev_upper_band = NAN;
    double prev_lower_band = NAN;
    double prev_close = NAN;
    bool prev_atr_ready = false;
    double initial_line = 0.0;
    double prev_initial_line = 0.0;
    int trend = 0;
    int bars_since_change = 0;
    int segment_index = 0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            exponential_trend_reset_state(
                &atr_count,
                &atr_sum,
                &atr_value,
                &atr_prev_close,
                &atr_have_prev_close,
                &prev_upper_band,
                &prev_lower_band,
                &prev_close,
                &prev_atr_ready,
                &initial_line,
                &prev_initial_line,
                &trend,
                &bars_since_change,
                &segment_index
            );
            continue;
        }

        const double tr_prev_close = atr_have_prev_close ? atr_prev_close : c;
        const double tr = fmax(h - l, fmax(fabs(h - tr_prev_close), fabs(l - tr_prev_close)));
        atr_prev_close = c;
        atr_have_prev_close = true;

        bool atr_ready = false;
        if (atr_count < 14) {
            atr_count += 1;
            atr_sum += tr;
            if (atr_count == 14) {
                atr_value = atr_sum / 14.0;
                atr_ready = true;
            }
        } else {
            atr_value = ((atr_value * 13.0) + tr) / 14.0;
            atr_ready = true;
        }

        double upper = NAN;
        double lower = NAN;
        int direction = 1;

        if (atr_ready) {
            const double src = (h + l) * 0.5;
            const double raw_upper = src + initial_distance * atr_value;
            const double raw_lower = src - initial_distance * atr_value;
            const double prev_lower = isfinite(prev_lower_band) ? prev_lower_band : 0.0;
            const double prev_upper = isfinite(prev_upper_band) ? prev_upper_band : 0.0;
            const double prev_close_value = isfinite(prev_close) ? prev_close : c;

            lower = (raw_lower > prev_lower || prev_close_value < prev_lower) ? raw_lower
                                                                             : prev_lower;
            upper = (raw_upper < prev_upper || prev_close_value > prev_upper) ? raw_upper
                                                                             : prev_upper;
            direction = !prev_atr_ready ? 1 : ((c < lower) ? 1 : -1);
        }

        const int prev_trend = trend;
        const double saved_prev_initial = prev_initial_line;
        const double saved_prev_close = prev_close;

        if (segment_index == 100 && isfinite(upper) && isfinite(lower)) {
            if (direction < 0) {
                initial_line = lower;
                trend = 1;
            } else {
                initial_line = upper;
                trend = -1;
            }
        }

        const bool crossover = isfinite(initial_line) && isfinite(saved_prev_close) &&
            isfinite(saved_prev_initial) && c > initial_line &&
            saved_prev_close <= saved_prev_initial;
        const bool crossunder = isfinite(initial_line) && isfinite(saved_prev_close) &&
            isfinite(saved_prev_initial) && c < initial_line &&
            saved_prev_close >= saved_prev_initial;

        if (crossover && isfinite(lower)) {
            initial_line = lower;
            trend = 1;
        } else if (crossunder && isfinite(upper)) {
            initial_line = upper;
            trend = -1;
        }

        if (trend != prev_trend) {
            bars_since_change = 0;
        } else if (trend != 0) {
            bars_since_change += 1;
        }

        if (trend != 0) {
            double exp_multiplier =
                1.0 + static_cast<double>(trend) *
                (1.0 - exp(-exp_rate * static_cast<double>(bars_since_change)));
            if (exp_multiplier > 900.0) {
                exp_multiplier = 900.0;
            }
            initial_line *= exp_multiplier;
        }

        if (atr_ready && trend == 1) {
            row[i] = initial_line;
        }

        prev_upper_band = upper;
        prev_lower_band = lower;
        prev_close = c;
        prev_initial_line = initial_line;
        prev_atr_ready = atr_ready;
        segment_index += 1;
    }
}
