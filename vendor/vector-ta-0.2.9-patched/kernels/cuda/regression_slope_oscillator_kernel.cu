#include <cmath>
#include <cstddef>

extern "C" __global__ void regression_slope_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ min_ranges,
    const int* __restrict__ max_ranges,
    const int* __restrict__ steps,
    const int* __restrict__ signal_lines,
    int rows,
    double* __restrict__ out_value,
    double* __restrict__ out_signal,
    double* __restrict__ out_bullish_reversal,
    double* __restrict__ out_bearish_reversal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int min_range = min_ranges[row];
    int max_range = max_ranges[row];
    int step = steps[row];
    int signal_line = signal_lines[row];

    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish =
        out_bullish_reversal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish =
        out_bearish_reversal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = NAN;
        row_signal[i] = NAN;
        row_bullish[i] = NAN;
        row_bearish[i] = NAN;
    }

    if (min_range < 2 || max_range < 2 || step <= 0 || signal_line <= 0 || min_range > max_range) {
        return;
    }

    double* signal_queue = new double[signal_line];
    if (signal_queue == nullptr) {
        return;
    }

    for (int i = max_range - 1; i < len; ++i) {
        int max_start = i + 1 - max_range;
        bool max_window_valid = true;
        for (int j = max_start; j <= i; ++j) {
            double sample = data[j];
            if (!(isfinite(sample) && sample > 0.0)) {
                max_window_valid = false;
                break;
            }
        }
        if (!max_window_valid) {
            continue;
        }

        double sum_slopes = 0.0;
        int slope_count = 0;
        bool valid = true;

        for (int length = min_range; length <= max_range; length += step) {
            int start = i + 1 - length;
            double length_f64 = static_cast<double>(length);
            double sum_x = length_f64 * (length_f64 + 1.0) * 0.5;
            double sum_x_sqr =
                length_f64 * (length_f64 + 1.0) * (2.0 * length_f64 + 1.0) / 6.0;
            double denom = length_f64 * sum_x_sqr - sum_x * sum_x;
            double sum_y = 0.0;
            double sum_xy = 0.0;

            for (int j = 0; j < length; ++j) {
                double sample = data[start + j];
                if (!(isfinite(sample) && sample > 0.0)) {
                    valid = false;
                    break;
                }
                double x = static_cast<double>(j + 1);
                double logged = log(sample);
                sum_y += logged;
                sum_xy += x * logged;
            }

            if (!valid) {
                break;
            }

            sum_slopes += (length_f64 * sum_xy - sum_x * sum_y) / denom;
            slope_count += 1;
        }

        if (valid && slope_count > 0) {
            row_value[i] = sum_slopes / static_cast<double>(slope_count);
        }
    }

    double signal_sum = 0.0;
    int signal_count = 0;
    int signal_head = 0;

    for (int i = 0; i < len; ++i) {
        double value = row_value[i];
        if (isfinite(value)) {
            if (signal_count < signal_line) {
                signal_queue[(signal_head + signal_count) % signal_line] = value;
                signal_sum += value;
                signal_count += 1;
            } else {
                signal_sum -= signal_queue[signal_head];
                signal_queue[signal_head] = value;
                signal_sum += value;
                signal_head += 1;
                if (signal_head == signal_line) {
                    signal_head = 0;
                }
            }
        }

        if (signal_count == signal_line) {
            row_signal[i] = signal_sum / static_cast<double>(signal_line);
        }

        if (isfinite(value) && isfinite(row_signal[i])) {
            double prev_value = i > 0 ? row_value[i - 1] : NAN;
            double prev_signal = i > 0 ? row_signal[i - 1] : NAN;
            row_bearish[i] = isfinite(prev_value) && isfinite(prev_signal) &&
                    value < row_signal[i] && prev_value >= prev_signal && value > 0.0
                ? 1.0
                : 0.0;
            row_bullish[i] = isfinite(prev_value) && isfinite(prev_signal) &&
                    value > row_signal[i] && prev_value <= prev_signal && value < 0.0
                ? 1.0
                : 0.0;
        }
    }

    delete[] signal_queue;
}


// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// regression_slope_oscillator_batch_f64 above is genuine double-in/double-out,
// but it takes 11 parameters, writes FOUR output matrices, and calls
// `new double[]` on the DEVICE for its signal ring. The f64 lane launches one
// shape -- (prices, n, periods, n_combos, first_valid, out) -- and allocates
// ONE output matrix, so that entry point cannot be reached from it.
//
// CPU REFERENCE
//   src/indicators/regression_slope_oscillator.rs:564
//     regression_slope_oscillator_with_kernel -> :519
//     regression_slope_oscillator_compute_into
//   build_prefixes      :385   window_has_invalid :425
//   slope_from_prefix   :440   build_length_spec  :261   expand_specs :274
//
// THE COLUMN THIS EMITS is value, which is what output_id == "value" resolves
// to (cpu_batch.rs, first arm).
//
// PINNED CPU DEFAULTS (compute_regression_slope_oscillator_batch):
// min_range 10, max_range 100, step 5, signal_line 7. expand_specs then yields
// the nineteen lengths 10, 15, ... 100, and value_warmup is max_range - 1 = 99.
//
// PERIOD-INVARIANT. The batch reads min_range, max_range, step and signal_line
// and NEVER `period`, so five swept periods give five identical CPU columns and
// this kernel emits five identical rows.
//
// THIS IS THE PREFIX FORM, NOT THE DIRECT FORM -- and that is the whole point
// of writing a second kernel rather than reusing the one above. The CPU does
// NOT recompute a window sum per (bar, length): build_prefixes accumulates ONE
// running `sum_prefix` of ln(price) and ONE running `weighted_prefix` of
// ln(price) * i from index 0, and every slope is a DIFFERENCE of two entries
// (:445-449). The kernel above instead re-sums each window from scratch with
// x = j + 1. Those agree in exact arithmetic and disagree in doubles: the
// prefix form carries a running total that reaches ~1e10 after 800k bars and
// then subtracts, so the cancellation is part of the answer. Reproducing the
// CPU means reproducing the cancellation.
//
// BOUNDED STATE, NOT A FULL-LENGTH ARRAY. A slope at bar i reads prefix entries
// [i + 1 - max_range, i + 1] -- exactly max_range + 1 = 101 consecutive indices
// -- so the three prefixes live in 101-slot rings indexed by (k mod 101) and
// the values read are bit-identical to the full arrays the CPU allocates.
//
// SHAPE: one thread per combo, bars ascending -- forced, because the prefixes
// are running sums.
//
// THE INVALID MASK IS A PREFIX COUNT, matching build_prefixes exactly: a bar
// counts as invalid unless `is_finite() && > 0.0` (:397), and a window is
// rejected when the two prefix counts differ (:429). Note `> 0.0`, not merely
// finite -- the indicator takes ln(price), and a non-positive price has no
// logarithm. `invalid_prefix == None` on the CPU when nothing is invalid, which
// is the same answer as two equal counts.
//
// FIRST VALID IS NOT READ: the CPU fills [0, value_warmup) with NaN itself
// (:527) and writes every later index. The lane row declares
// F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double log/fabs, no f32-suffixed math
// function, no fast-math intrinsic. The file is added to F64_LANE_SOURCES, so
// `log` is never the fast-math approximation -- which matters here because
// every slope is a difference of two large running sums of logarithms.
// ---------------------------------------------------------------------------

#define RSO_NEO_MIN_RANGE 10
#define RSO_NEO_MAX_RANGE 100
#define RSO_NEO_STEP 5
#define RSO_NEO_PREFIX_CAP (RSO_NEO_MAX_RANGE + 1)
// expand_specs(10, 100, 5) -- 10, 15, ... 100.
#define RSO_NEO_MAX_SPECS 19

__device__ __forceinline__ double rso_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void regression_slope_oscillator_neo_batch_f64(
    const double* __restrict__ prices,
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
    const double qnan = rso_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    // expand_specs (:274-289) + build_length_spec (:261-271).
    int spec_length[RSO_NEO_MAX_SPECS];
    double spec_length_f64[RSO_NEO_MAX_SPECS];
    double spec_sum_x[RSO_NEO_MAX_SPECS];
    double spec_denominator[RSO_NEO_MAX_SPECS];
    int spec_count = 0;
    {
        int current = RSO_NEO_MIN_RANGE;
        for (;;) {
            if (spec_count >= RSO_NEO_MAX_SPECS) {
                return;   // the compiled bound, refused rather than truncated
            }
            const double length_f64 = static_cast<double>(current);
            const double sum_x = length_f64 * (length_f64 + 1.0) * 0.5;
            const double sum_x_sqr =
                length_f64 * (length_f64 + 1.0) * (2.0 * length_f64 + 1.0) / 6.0;
            spec_length[spec_count] = current;
            spec_length_f64[spec_count] = length_f64;
            spec_sum_x[spec_count] = sum_x;
            spec_denominator[spec_count] = length_f64 * sum_x_sqr - sum_x * sum_x;
            spec_count += 1;
            if (current >= RSO_NEO_MAX_RANGE) {
                break;
            }
            const int next = current + RSO_NEO_STEP;
            if (next > RSO_NEO_MAX_RANGE) {
                break;
            }
            current = next;
        }
    }
    if (spec_count == 0) {
        return;
    }
    const double spec_count_f64 = static_cast<double>(spec_count);

    // build_prefixes (:385-423), held as 101-slot rings over the prefix index.
    double sum_prefix[RSO_NEO_PREFIX_CAP];
    double weighted_prefix[RSO_NEO_PREFIX_CAP];
    int invalid_prefix[RSO_NEO_PREFIX_CAP];
    double running_sum = 0.0;
    double running_weighted = 0.0;
    int running_invalid = 0;
    sum_prefix[0] = 0.0;
    weighted_prefix[0] = 0.0;
    invalid_prefix[0] = 0;

    const int value_warmup = RSO_NEO_MAX_RANGE - 1;

    for (int i = 0; i < n; ++i) {
        const double value = prices[i];
        if (isfinite(value) && value > 0.0) {
            const double logged = log(value);
            running_sum += logged;
            running_weighted += logged * static_cast<double>(i);
        } else {
            running_invalid += 1;
        }
        const int slot = (i + 1) % RSO_NEO_PREFIX_CAP;
        sum_prefix[slot] = running_sum;
        weighted_prefix[slot] = running_weighted;
        invalid_prefix[slot] = running_invalid;

        if (i < value_warmup) {
            continue;
        }

        const int end_slot = slot;
        const int max_start = i + 1 - RSO_NEO_MAX_RANGE;
        const int max_start_slot = max_start % RSO_NEO_PREFIX_CAP;
        if (invalid_prefix[end_slot] != invalid_prefix[max_start_slot]) {
            continue;   // out_value[i] stays NaN (:534-537)
        }

        const double sum_prefix_end = sum_prefix[end_slot];
        const double weighted_prefix_end = weighted_prefix[end_slot];

        double sum_slopes = 0.0;
        for (int s = 0; s < spec_count; ++s) {
            const int start = i + 1 - spec_length[s];
            const int start_slot = start % RSO_NEO_PREFIX_CAP;
            // slope_from_prefix (:440-452), term for term.
            const double sum_y = sum_prefix_end - sum_prefix[start_slot];
            const double weighted_abs = weighted_prefix_end - weighted_prefix[start_slot];
            const double weighted_rel =
                weighted_abs + (1.0 - static_cast<double>(start)) * sum_y;
            sum_slopes += (spec_length_f64[s] * weighted_rel - spec_sum_x[s] * sum_y) /
                          spec_denominator[s];
        }
        row[i] = sum_slopes / spec_count_f64;
    }
}
