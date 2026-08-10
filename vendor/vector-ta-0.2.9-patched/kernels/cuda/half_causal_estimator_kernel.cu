#include <cmath>
#include <cstddef>

namespace {

constexpr int KERNEL_GAUSSIAN = 0;
constexpr int KERNEL_EPANECHNIKOV = 1;
constexpr int KERNEL_TRIANGULAR = 2;
constexpr int KERNEL_SINC = 3;

constexpr int CONFIDENCE_SYMMETRIC = 0;
constexpr int CONFIDENCE_LINEAR = 1;
constexpr int CONFIDENCE_NONE = 2;

__device__ inline bool finite_value(double value) {
    return isfinite(value);
}

__device__ inline double clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

__device__ bool bucket_mean_icv(
    const double* data,
    int history_end,
    int target_slot,
    int slots_per_day,
    int data_period,
    double maximum_confidence_adjust_factor,
    double* out_mean,
    double* out_icv
) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;

    for (int idx = history_end; idx >= 0; --idx) {
        if ((idx % slots_per_day) != target_slot) {
            continue;
        }
        const double value = data[idx];
        if (!finite_value(value)) {
            continue;
        }
        sum += value;
        sum_sq += value * value;
        count += 1;
        if (data_period > 0 && count >= data_period) {
            break;
        }
    }

    if (count == 0) {
        return false;
    }

    const double mean = sum / static_cast<double>(count);
    double icv = 1.0;
    if (fabs(mean) > DBL_EPSILON) {
        const double variance = sum_sq / static_cast<double>(count) - mean * mean;
        const double stdev = sqrt(fmax(variance, 0.0));
        const double ratio = clamp01(stdev / mean);
        icv = 1.0 - ratio * maximum_confidence_adjust_factor;
    }

    *out_mean = mean;
    *out_icv = icv;
    return true;
}

__device__ bool collect_future(
    const double* data,
    int history_end,
    int slot,
    int slots_per_day,
    int data_period,
    int future_len,
    double maximum_confidence_adjust_factor,
    double* future_values,
    double* future_weights
) {
    if (future_len <= 0) {
        return true;
    }
    if (slots_per_day <= 0) {
        return false;
    }

    int found = 0;
    int offset = 1;
    bool saw_valid = false;
    while (found < future_len) {
        const int next_slot = (slot + offset) % slots_per_day;
        double mean = NAN;
        double icv = 1.0;
        if (bucket_mean_icv(
                data,
                history_end,
                next_slot,
                slots_per_day,
                data_period,
                maximum_confidence_adjust_factor,
                &mean,
                &icv
            )) {
            saw_valid = true;
            future_values[found] = mean;
            future_weights[found] = fmax(icv, 0.0);
            found += 1;
        }
        offset += 1;
        if (offset > slots_per_day * 4 && !saw_valid) {
            return false;
        }
    }

    for (int left = 0, right = future_len - 1; left < right; ++left, --right) {
        const double tmp_value = future_values[left];
        future_values[left] = future_values[right];
        future_values[right] = tmp_value;

        const double tmp_weight = future_weights[left];
        future_weights[left] = future_weights[right];
        future_weights[right] = tmp_weight;
    }
    return true;
}

__device__ double compute_estimate_window(
    const double* data,
    int len,
    int index,
    int slots_per_day,
    int data_period,
    int real_filter_length,
    int window_size,
    int confidence_adjust,
    double maximum_confidence_adjust_factor,
    const double* kernel_row,
    double* future_values,
    double* future_weights
) {
    if (real_filter_length <= 0 || window_size <= 0 || index < 0 || index >= len) {
        return NAN;
    }

    const int future_len = real_filter_length - 1;
    if (!collect_future(
            data,
            index - 1,
            index % slots_per_day,
            slots_per_day,
            data_period,
            future_len,
            maximum_confidence_adjust_factor,
            future_values,
            future_weights
        )) {
        return NAN;
    }

    double acc = 0.0;
    for (int j = 0; j < future_len; ++j) {
        const double value = future_values[j];
        if (!finite_value(value)) {
            return NAN;
        }
        const double confidence =
            confidence_adjust == CONFIDENCE_NONE ? 1.0 : future_weights[j];
        acc += value * confidence * kernel_row[j];
    }

    double future_weight_sum = 0.0;
    for (int j = 0; j < future_len; ++j) {
        future_weight_sum += future_weights[j];
    }

    for (int j = 0; j < real_filter_length; ++j) {
        const int source_index = index - j;
        if (source_index < 0) {
            return NAN;
        }
        const double value = data[source_index];
        if (!finite_value(value)) {
            return NAN;
        }

        double confidence = 1.0;
        if (confidence_adjust == CONFIDENCE_SYMMETRIC) {
            if (j == 0) {
                confidence = 1.0;
            } else {
                confidence = 2.0 - future_weights[future_len - j];
            }
        } else if (confidence_adjust == CONFIDENCE_LINEAR) {
            confidence =
                real_filter_length > 1
                ? 2.0 - future_weight_sum / static_cast<double>(real_filter_length - 1)
                : 1.0;
        }

        acc += value * confidence * kernel_row[future_len + j];
    }

    return acc;
}

__device__ double compute_expected_window(
    const double* data,
    int len,
    int index,
    int slots_per_day,
    int data_period,
    int real_filter_length,
    int window_size,
    double maximum_confidence_adjust_factor,
    const double* kernel_row,
    double* future_values,
    double* future_weights
) {
    if (real_filter_length <= 0 || window_size <= 0 || index < 0 || index >= len) {
        return NAN;
    }

    const int future_len = real_filter_length - 1;
    if (!collect_future(
            data,
            index - 1,
            index % slots_per_day,
            slots_per_day,
            data_period,
            future_len,
            maximum_confidence_adjust_factor,
            future_values,
            future_weights
        )) {
        return NAN;
    }

    double acc = 0.0;
    for (int j = 0; j < future_len; ++j) {
        const double value = future_values[j];
        if (!finite_value(value)) {
            return NAN;
        }
        acc += value * kernel_row[j];
    }

    for (int j = 0; j < real_filter_length; ++j) {
        const int source_index = index - j;
        if (source_index < 0) {
            return NAN;
        }
        const int history_end = source_index - 1;
        double mean = NAN;
        double icv = 1.0;
        const bool ok = bucket_mean_icv(
            data,
            history_end,
            source_index % slots_per_day,
            slots_per_day,
            data_period,
            maximum_confidence_adjust_factor,
            &mean,
            &icv
        );
        (void)icv;
        if (!ok || !finite_value(mean)) {
            return NAN;
        }
        acc += mean * kernel_row[future_len + j];
    }

    return acc;
}

__device__ double wma_update(
    double raw_value,
    int wma_length,
    double* history,
    int* count,
    double* first,
    bool* has_first
) {
    if (!finite_value(raw_value)) {
        return NAN;
    }

    if (!(*has_first)) {
        *first = raw_value;
        *has_first = true;
    }

    if (wma_length <= 1) {
        history[0] = raw_value;
        *count = 1;
        return raw_value;
    }

    const int prev_count = *count;
    const int next_count = prev_count < wma_length ? prev_count + 1 : wma_length;
    for (int i = next_count - 1; i >= 1; --i) {
        history[i] = history[i - 1];
    }
    history[0] = raw_value;
    *count = next_count;

    const double denominator = static_cast<double>(wma_length * (wma_length + 1) / 2);
    double sum = 0.0;
    for (int i = 0; i < wma_length; ++i) {
        const double sample = i < next_count ? history[i] : *first;
        sum += sample * static_cast<double>(wma_length - i);
    }
    return sum / denominator;
}

}

extern "C" __global__ void half_causal_estimator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ slots_per_days,
    const int* __restrict__ data_periods,
    const int* __restrict__ filter_lengths,
    const int* __restrict__ real_filter_lengths,
    const int* __restrict__ window_sizes,
    const double* __restrict__ maximum_confidence_adjust_factors,
    const int* __restrict__ enable_expected_values,
    const int* __restrict__ confidence_adjusts,
    const int* __restrict__ wma_lengths,
    int rows,
    int future_cap,
    int window_cap,
    int wma_cap,
    const double* __restrict__ kernel_matrix,
    double* __restrict__ future_values_scratch,
    double* __restrict__ future_weights_scratch,
    double* __restrict__ wma_history_scratch,
    double* __restrict__ out_estimate,
    double* __restrict__ out_expected_value
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int slots_per_day = slots_per_days[row];
    const int data_period = data_periods[row];
    const int real_filter_length = real_filter_lengths[row];
    const int window_size = window_sizes[row];
    const double maximum_confidence_adjust_factor = maximum_confidence_adjust_factors[row];
    const int enable_expected_value = enable_expected_values[row];
    const int confidence_adjust = confidence_adjusts[row];
    const int wma_length = wma_lengths[row];

    double* row_estimate = out_estimate + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_expected =
        out_expected_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* future_values =
        future_values_scratch + static_cast<size_t>(row) * static_cast<size_t>(future_cap);
    double* future_weights =
        future_weights_scratch + static_cast<size_t>(row) * static_cast<size_t>(future_cap);
    double* wma_history =
        wma_history_scratch + static_cast<size_t>(row) * static_cast<size_t>(wma_cap);
    const double* kernel_row =
        kernel_matrix + static_cast<size_t>(row) * static_cast<size_t>(window_cap);

    for (int i = 0; i < len; ++i) {
        row_estimate[i] = NAN;
        row_expected[i] = NAN;
    }

    if (slots_per_day < 2 || real_filter_length < 2 || window_size <= 0 ||
        !finite_value(maximum_confidence_adjust_factor)) {
        return;
    }

    int wma_count = 0;
    double wma_first = NAN;
    bool wma_has_first = false;
    bool ready = false;

    for (int i = 0; i < len; ++i) {
        const int slot = i % slots_per_day;
        const bool session_start = slot == 0;
        if (!ready && i > window_size && session_start) {
            ready = true;
        }

        if (ready && i + 1 >= real_filter_length) {
            const double estimate_raw = compute_estimate_window(
                data,
                len,
                i,
                slots_per_day,
                data_period,
                real_filter_length,
                window_size,
                confidence_adjust,
                maximum_confidence_adjust_factor,
                kernel_row,
                future_values,
                future_weights
            );
            if (finite_value(estimate_raw)) {
                row_estimate[i] = wma_update(
                    estimate_raw,
                    wma_length,
                    wma_history,
                    &wma_count,
                    &wma_first,
                    &wma_has_first
                );
            }

            if (enable_expected_value != 0) {
                row_expected[i] = compute_expected_window(
                    data,
                    len,
                    i,
                    slots_per_day,
                    data_period,
                    real_filter_length,
                    window_size,
                    maximum_confidence_adjust_factor,
                    kernel_row,
                    future_values,
                    future_weights
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `half_causal_estimator_with_kernel`
// (src/indicators/half_causal_estimator.rs:1391) -> `compute_row` (:1358) ->
// `HalfCausalEstimatorContext::update` (:859) and `compute_window` (:889), with
// `build_kernel` (:1122), `collect_future_into` (:1159), `TimeOfDayBucket`
// (:597), `infer_slots_per_day` (:1207) and `slot_from_timestamp` (:1228).
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `half_causal_estimator_batch_f64` (:325) is double-clean but declares
// twenty-one parameters -- eight `const int*` per-row parameter arrays, a
// host-built kernel-weight matrix, three scratch buffers and TWO output
// matrices. The f64 lane launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// so the lane gets its own entry point here. It builds its own kernel weights
// on the device (39 doubles) and its two collect buffers are fixed-size
// PER-THREAD arrays (19 each): 77 doubles, 616 bytes.
//
// WHICH COLUMN: `estimate`. `compute_half_causal_estimator_batch`
// (cpu_batch.rs:9481) accepts exactly two output ids, `estimate` and
// `expected_value`, and has no `value` alias. `expected_value` is OFF at the
// CPU default (`enable_expected_value` false, cpu_batch.rs:9608), so it is
// all-NaN and `estimate` is the only column with content.
//
// WHY THE INPUT IS (timestamps, close, volume) AND NOT A PLAIN SLICE
//
// The CPU has two doors. The Slice door requires an explicit `slots_per_day`
// and the batch closure passes `None` for it at the default (0 ->
// `MissingSlotsPerDay`, cpu_batch.rs:9573-9577), so with default parameters
// that door ERRORS. The Candles door is the one that works: it INFERS
// `slots_per_day` from the bar timestamps (:1319) and takes `volume` as the
// source (`source` default "volume", cpu_batch.rs:9531; `source_from_candles`
// :1248). So the timestamps are an INPUT to this indicator, exactly as they are
// for `vwap`, and the lane row declares `F64InputKind::TimestampCloseVolume`.
// `close` is passed by that shape and is NOT read here -- the CPU does not read
// it either on this path.
//
// SHAPE: one thread per combo, bars ascending. `ready` latches on the first
// session boundary after `window_size` bars and never clears (:865-867), and
// the time-of-day store is built from every bar before the current one, so bar
// i is not computable without the whole prefix.
//
// PERIOD-INVARIANT: the CPU batch reads `slots_per_day`, `data_period`,
// `filter_length`, `kernel_width`, `maximum_confidence_adjust`,
// `extra_smoothing`, `enable_expected_value`, `source`, `kernel_type` and
// `confidence_adjust` -- and NEVER `period` (cpu_batch.rs:9503-9560), so every
// swept period gives the same CPU column and this kernel writes identical rows.
// Pinned at the CPU defaults: data_period 5, filter_length 20, kernel
// epanechnikov, kernel_width 20.0, maximum_confidence_adjust 100.0 (factor
// 1.0), confidence_adjust symmetric, extra_smoothing 0, expected_value off.
// Hence `real_filter_length` = 20 and `window_size` = 2*20-1 = 39 (:1055-1066).
//
// `extra_smoothing` 0 makes `FillWmaState::length` 1, and `FillWmaState::update`
// (:795-800) then returns the input unchanged for a finite input and `None` for
// a non-finite one -- so the WMA stage is exactly "keep it if it is finite",
// which is what this kernel writes.
//
// ROUNDING: `compute_window` (:889-975) accumulates ONE `sum`, the future half
// first (ascending j) and then the causal half (newest bar first), each term
// `value * confidence * kernel[k]` left to right. Reproduced term for term.
// The CPU writes no `mul_add` on this path, so no `fma` is introduced.
//
// EPSILON: `f64::EPSILON` guards `avg.abs()` in `icv` (:713). That is the
// DOUBLE epsilon, already f64-sized, and is kept verbatim as NEO_HCE_F64_EPSILON. Nothing
// f32-sized is carried in.
//
// NaN SEMANTICS: `collect_future_into` only takes buckets that HAVE values
// (:1183) and the window refuses any non-finite term (:975, :986), so a NaN
// cannot reach the accumulator. `fmax(icv, 0.0)` reproduces the CPU's
// `.max(0.0)` (:1190) -- and it is `fmax`, not a comparison chain, precisely
// because `f64::max` returns the non-NaN operand.
//
// THE ONE DOCUMENTED RESIDUAL -- READ THIS BEFORE TRUSTING THE LAST BIT.
// The CPU's `TimeOfDayBucket` carries `sum` and `sum_sq` as RING ACCUMULATORS:
// every value is added when it arrives and subtracted when it is evicted
// (:625-648), so after k arrivals the accumulator carries rounding from values
// that are no longer in the window. This kernel RECONSTRUCTS each bucket by
// scanning backwards for its most recent `data_period` finite samples and
// summing them, which is the same SET and a different accumulation history --
// so `mean` and `icv`, and therefore the estimate, can differ in the last
// place or two.
//   Holding the CPU's accumulator exactly would mean holding the whole store
// per thread: `slots_per_day` can be 1440 (M1 bars: `infer_slots_per_day`
// returns 1440/minutes, and minutes >= 1), and a bucket is 5 values plus `sum`
// and `sum_sq`, so 1440 * 7 * 8 = 80,640 bytes of LOCAL memory per thread. CUDA
// reserves local memory for the device's maximum resident thread count, not for
// the launched grid, so on an 82-SM card at 1536 threads/SM that reservation is
// ~10 GB. That is why the store is reconstructed rather than carried, and the
// residual is named here rather than hidden.
//   The reconstruction is also what the existing entry point in this file does
// (`bucket_mean_icv`, :29), so this is not a new behaviour -- it is the same
// behaviour, now written down.
//
// COST, HONESTLY: `collect_future` walks back about `data_period *
// slots_per_day` bars per target slot and needs `real_filter_length - 1` = 19
// of them, so a bar costs ~5 * slots_per_day * 19 scanned bars. On M5
// (slots_per_day 288) that is ~27k per bar. This is the same cost the existing
// entry point pays; it is not made worse here, and it is a profiling target,
// not a correctness one.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. `sqrt`, `fabs`, `fmax` are the double overloads. The NaN is a
// DOUBLE quiet-NaN bit pattern.
//
// FIRST VALID IS NOT READ: the CPU's `first_finite` (:1338) is used only to
// reject an all-NaN series, and the warmup is set by `ready`, which latches on
// a SESSION boundary rather than at a fixed index. The lane row declares
// `F64FirstValidRule::Ignored`.
//
// WHEN THE TIMEFRAME CANNOT BE INFERRED the CPU returns
// `UnableToInferMinuteTimeframe` and the whole batch is an `Err`. A kernel has
// no error channel in this lane, so it writes the all-NaN column it already
// initialised. That is the only case in this file where the device is quieter
// than the CPU, and it is a data-shape error the caller sees on the CPU side
// of any parity run.
// ---------------------------------------------------------------------------

#define NEO_HCE_DATA_PERIOD 5
#define NEO_HCE_FILTER_LENGTH 20
#define NEO_HCE_REAL_FILTER_LENGTH 20
#define NEO_HCE_FUTURE_LEN (NEO_HCE_REAL_FILTER_LENGTH - 1)
#define NEO_HCE_WINDOW_SIZE (2 * NEO_HCE_REAL_FILTER_LENGTH - 1)
#define NEO_HCE_KERNEL_WIDTH 20.0
#define NEO_HCE_MAX_CONF_FACTOR 1.0
#define NEO_HCE_DAY_MS 86400000LL
// f64::EPSILON, spelled out so this section does not depend on <cfloat>.
#define NEO_HCE_F64_EPSILON 2.2204460492503131e-16

__device__ inline double neo_hce_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `epanechnikov_kernel`, :1085.
__device__ inline double neo_hce_epanechnikov(double centered_index, double bandwidth) {
    const double ratio = centered_index / bandwidth;
    if (fabs(ratio) <= 1.0) {
        return 0.75 * (1.0 - ratio * ratio);
    }
    return 0.0;
}

// `infer_slots_per_day`, :1207-1225. Returns 0 when the CPU would return
// `UnableToInferMinuteTimeframe`.
__device__ inline int neo_hce_infer_slots_per_day(const long long* ts, int n) {
    long long min_positive = 0x7fffffffffffffffLL;
    for (int i = 1; i < n; ++i) {
        const long long delta = ts[i] - ts[i - 1];
        if (delta > 0 && delta < NEO_HCE_DAY_MS && delta < min_positive) {
            min_positive = delta;
        }
    }
    if (min_positive == 0x7fffffffffffffffLL || (min_positive % 60000LL) != 0) {
        return 0;
    }
    const long long minutes = min_positive / 60000LL;
    if (minutes == 0 || (1440LL % minutes) != 0) {
        return 0;
    }
    return static_cast<int>(1440LL / minutes);
}

// `slot_from_timestamp`, :1228-1239. Returns -1 where the CPU would return
// `InvalidTimestamp`.
__device__ inline int neo_hce_slot(long long timestamp, int slots_per_day) {
    if (timestamp < 0) {
        return -1;
    }
    const long long seconds = timestamp / 1000LL;
    const long long minutes_of_day = (seconds % 86400LL) / 60LL;
    const int minutes_per_slot = 1440 / slots_per_day;
    if (minutes_per_slot <= 0) {
        return -1;
    }
    return static_cast<int>(minutes_of_day) / minutes_per_slot;
}

// `TimeOfDayBucket::mean` (:657) and `::icv` (:707) for ONE slot, rebuilt from
// the bars strictly before `history_end + 1`. Returns false where
// `has_values()` is false.
__device__ bool neo_hce_bucket(
    const double* __restrict__ source,
    const long long* __restrict__ timestamps,
    int history_end,
    int target_slot,
    int slots_per_day,
    double* out_mean,
    double* out_icv
) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    for (int idx = history_end; idx >= 0; --idx) {
        if (neo_hce_slot(timestamps[idx], slots_per_day) != target_slot) {
            continue;
        }
        const double value = source[idx];
        if (!isfinite(value)) {
            continue;
        }
        sum += value;
        sum_sq += value * value;
        count += 1;
        if (count >= NEO_HCE_DATA_PERIOD) {
            break;
        }
    }
    if (count == 0) {
        return false;
    }
    const double mean = sum / static_cast<double>(count);
    double icv = 1.0;
    if (fabs(mean) > NEO_HCE_F64_EPSILON) {
        const double variance = sum_sq / static_cast<double>(count) - mean * mean;
        const double stdev = sqrt(fmax(variance, 0.0));
        double ratio = stdev / mean;
        // `.clamp(0.0, 1.0)`, :718.
        if (ratio < 0.0) {
            ratio = 0.0;
        } else if (ratio > 1.0) {
            ratio = 1.0;
        }
        icv = 1.0 - ratio * NEO_HCE_MAX_CONF_FACTOR;
    }
    *out_mean = mean;
    *out_icv = icv;
    return true;
}

extern "C" __global__ void half_causal_estimator_neo_batch_f64(
    const long long* __restrict__ timestamps,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;
    (void)close;  // the CPU source is `volume`; close is not read on this path

    const double nan_value = neo_hce_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = nan_value;
    }

    const int slots_per_day = neo_hce_infer_slots_per_day(timestamps, n);
    if (slots_per_day < 2) {
        // `resolve_params` (:1023) rejects < 2, and a 0 here means the CPU
        // could not infer the timeframe at all.
        return;
    }

    // `build_kernel`, :1122-1155.
    double kernel_w[NEO_HCE_WINDOW_SIZE];
    {
        const double center = static_cast<double>(NEO_HCE_WINDOW_SIZE - 1) * 0.5;
        double normalization = 0.0;
        for (int i = 0; i < NEO_HCE_WINDOW_SIZE; ++i) {
            const double centered = static_cast<double>(i) - center;
            const double weight = neo_hce_epanechnikov(centered, NEO_HCE_KERNEL_WIDTH);
            normalization += weight;
            kernel_w[i] = weight;
        }
        if (normalization != 0.0) {
            for (int i = 0; i < NEO_HCE_WINDOW_SIZE; ++i) {
                kernel_w[i] /= normalization;
            }
        }
    }

    double future_values[NEO_HCE_FUTURE_LEN];
    double future_weights[NEO_HCE_FUTURE_LEN];

    bool ready = false;
    bool has_prev_slot = false;
    int prev_slot = 0;

    for (int i = 0; i < n; ++i) {
        const int slot = neo_hce_slot(timestamps[i], slots_per_day);
        if (slot < 0) {
            return;  // the CPU errors on an unrepresentable timestamp
        }

        // `update`, :860-867.
        const bool session_start = has_prev_slot ? (slot <= prev_slot) : true;
        prev_slot = slot;
        has_prev_slot = true;
        if (!ready && i > NEO_HCE_WINDOW_SIZE && session_start) {
            ready = true;
        }

        // `source_buffer.is_full()` -- the buffer takes one push per bar, so it
        // is full once `i + 1 >= real_filter_length`.
        if (!ready || i + 1 < NEO_HCE_REAL_FILTER_LENGTH) {
            continue;
        }

        // `collect_future_into`, :1159-1204. The store holds bars < i.
        int found = 0;
        int offset = 1;
        bool saw_valid = false;
        bool failed = false;
        while (found < NEO_HCE_FUTURE_LEN) {
            const int next_slot = (slot + offset) % slots_per_day;
            double mean = nan_value;
            double icv = 1.0;
            if (neo_hce_bucket(volume, timestamps, i - 1, next_slot, slots_per_day,
                               &mean, &icv)) {
                saw_valid = true;
                future_values[found] = mean;
                future_weights[found] = fmax(icv, 0.0);
                found += 1;
            }
            offset += 1;
            if (offset > slots_per_day * 4 && !saw_valid) {
                failed = true;
                break;
            }
        }
        if (failed) {
            continue;
        }
        // `values.reverse()` / `weights.reverse()`, :1199-1202.
        for (int a = 0, b = NEO_HCE_FUTURE_LEN - 1; a < b; ++a, --b) {
            const double tv = future_values[a];
            future_values[a] = future_values[b];
            future_values[b] = tv;
            const double tw = future_weights[a];
            future_weights[a] = future_weights[b];
            future_weights[b] = tw;
        }

        // `compute_window`, :926-974. Future half first, then causal half.
        double sum = 0.0;
        bool ok = true;
        for (int j = 0; j < NEO_HCE_FUTURE_LEN; ++j) {
            const double value = future_values[j];
            if (!isfinite(value)) {
                ok = false;
                break;
            }
            sum += value * future_weights[j] * kernel_w[j];
        }
        if (!ok) {
            continue;
        }
        for (int j = 0; j < NEO_HCE_REAL_FILTER_LENGTH; ++j) {
            // `causal_values.iter()` yields the buffer FRONT first, and the
            // front is the most recent push (`FixedFrontBuffer::push`, :742).
            const double value = volume[i - j];
            if (!isfinite(value)) {
                ok = false;
                break;
            }
            // Symmetric confidence, :959-965.
            const double confidence =
                (j == 0) ? 1.0 : (2.0 - future_weights[NEO_HCE_FUTURE_LEN - j]);
            sum += value * confidence * kernel_w[NEO_HCE_FUTURE_LEN + j];
        }
        if (!ok) {
            continue;
        }

        // `FillWmaState::update` with length 1, :795-800.
        if (isfinite(sum)) {
            row[i] = sum;
        }
    }
}
