#include <cmath>
#include <float.h>   // DBL_EPSILON — the f64 conversion used it without declaring it
#include <cstddef>

namespace {

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
// so the lane gets its own entry point here. It builds each admitted row's
// kernel weights on the device in compile-time capacity 399 and uses two
// compile-time future caches of capacity 199.
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
// SHAPE: one thread walks bars ascending once per requested RegistryRatio row.
// Scratch, cached future windows, and readiness are reset between rows. `ready`
// latches on the first session boundary after that row's `window_size` bars,
// and the time-of-day store contains every finite bar before the current one.
//
// TYPED REGISTRY RATIO: `periods[combo]` is the preserved ABI anchor and maps
// to (D,L) as 7=(2,7), 20=(5,20), 21=(5,21), 50=(13,50), 100=(25,100),
// and 200=(50,200). L21 remains distinct from L20 because its 2L-1 readiness
// boundary can cross a different session start even when one endpoint weight
// is zero. Non-D/L parameters remain fixed at registry defaults:
// Epanechnikov, width 20.0, maximum
// confidence 100%, symmetric confidence, smoothing 0, expected-value off,
// and volume source.
//
// `extra_smoothing` 0 makes `FillWmaState::length` 1, and `FillWmaState::update`
// (:795-800) then returns the input unchanged for a finite input and `None` for
// a non-finite one -- so the WMA stage is exactly "keep it if it is finite",
// which is what this kernel writes.
//
// ROUNDING: Stable Authority V2 forms every weighted term as two explicit
// products (`value * confidence`, then `scaled * kernel[k]`) and feeds future
// terms first, causal newest-first terms second, through Neumaier summation.
// The native f64 build pins `-fmad=false`, so those source operations cannot be
// contracted into an unshared multiply-add schedule.
//
// CREATOR CV FALLBACK: an empty bucket or exact-zero mean uses confidence 1.0.
// A tiny finite non-zero mean still participates in `stdev / mean`; only an
// undefined (NaN) confidence falls back to 1.0. No epsilon approximation and
// no f32-sized constant is carried in.
//
// NaN SEMANTICS: `collect_future_into` only takes buckets that HAVE values
// (:1183) and the window refuses any non-finite term (:975, :986), so a NaN
// cannot reach the accumulator. `fmax(icv, 0.0)` reproduces the CPU's
// `.max(0.0)` (:1190) -- and it is `fmax`, not a comparison chain, precisely
// because `f64::max` returns the non-NaN operand.
//
// STABLE AUTHORITY V2. Each row retains its mapped last D finite observations
// per time-of-day slot and recomputes population `(mean, M2)` in chronological
// oldest-to-newest order after an insertion. The future-reversed then causal-
// newest dot uses the same Neumaier schedule. Scratch is explicit global memory
// owned by the strict wrapper and reused serially, making work O(N*combos)
// without changing the creator's compute-before-TOD-insert schedule.
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

#define NEO_HCE_MAX_SLOTS_PER_DAY 1440
#define NEO_HCE_MAX_DATA_PERIOD 50
#define NEO_HCE_MAX_FILTER_LENGTH 200
#define NEO_HCE_MAX_FUTURE_LEN 199
#define NEO_HCE_MAX_WINDOW_SIZE 399
#define NEO_HCE_KERNEL_WIDTH 20.0
#define NEO_HCE_MAX_CONF_FACTOR 1.0
#define NEO_HCE_DAY_MS 86400000LL
#define NEO_HCE_CHRONO_MIN_TIMESTAMP_MS (-8334601228800000LL)
#define NEO_HCE_CHRONO_MAX_TIMESTAMP_MS 8210266876799999LL
__device__ inline double neo_hce_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// Typed RegistryRatio contract. `periods[combo]` remains the stable ABI anchor;
// only the two creator D/L parameters vary. Kernel family, width, confidence,
// extra smoothing, expected-value, and volume source remain their registry
// defaults. The L21 row is retained because readiness depends on 2L-1.
__device__ inline bool neo_hce_resolve_registry_anchor(
    int anchor,
    int* data_period,
    int* filter_length
) {
    switch (anchor) {
        case 7: *data_period = 2; *filter_length = 7; return true;
        case 20: *data_period = 5; *filter_length = 20; return true;
        case 21: *data_period = 5; *filter_length = 21; return true;
        case 50: *data_period = 13; *filter_length = 50; return true;
        case 100: *data_period = 25; *filter_length = 100; return true;
        case 200: *data_period = 50; *filter_length = 200; return true;
        default: return false;
    }
}

// `epanechnikov_kernel`, :1085.
__device__ inline double neo_hce_epanechnikov(double centered_index, double bandwidth) {
    const double ratio = centered_index / bandwidth;
    if (fabs(ratio) <= 1.0) {
        return 0.75 * (1.0 - ratio * ratio);
    }
    return 0.0;
}

// Strict f64 semantic authority:
// half-causal-estimator-f64-v2-neoethos-canonical-pine6-script24-utc-day-slot-session-proxy-cached-future-windows-stable-f64-registry-ratio-dl
// Creator Pine session state cannot be reconstructed from the lane ABI. The
// named NeoEthos variant therefore uses a UTC calendar-day transition as its
// explicit session proxy, while retaining Pine's cached future-window update.

__device__ inline bool neo_hce_timestamp_is_valid(long long timestamp) {
    return timestamp >= NEO_HCE_CHRONO_MIN_TIMESTAMP_MS &&
           timestamp <= NEO_HCE_CHRONO_MAX_TIMESTAMP_MS;
}

// Every timestamp is admitted before any finite scratch/output write. This
// mirrors host `DateTime::<Utc>::from_timestamp_millis` rather than silently
// treating an unrepresentable timestamp as a different time-of-day slot.
__device__ inline bool neo_hce_validate_all_timestamps(
    const long long* timestamps,
    int n
) {
    for (int i = 0; i < n; ++i) {
        if (!neo_hce_timestamp_is_valid(timestamps[i])) {
            return false;
        }
    }
    return true;
}

__device__ inline long long neo_hce_utc_day(long long timestamp) {
    const long long quotient = timestamp / NEO_HCE_DAY_MS;
    const long long remainder = timestamp % NEO_HCE_DAY_MS;
    return remainder < 0 ? quotient - 1 : quotient;
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
    if (!neo_hce_timestamp_is_valid(timestamp)) {
        return -1;
    }
    long long millis_of_day = timestamp % NEO_HCE_DAY_MS;
    if (millis_of_day < 0) {
        millis_of_day += NEO_HCE_DAY_MS;
    }
    const long long minutes_of_day = millis_of_day / 60000LL;
    const int minutes_per_slot = 1440 / slots_per_day;
    if (minutes_per_slot <= 0) {
        return -1;
    }
    return static_cast<int>(minutes_of_day) / minutes_per_slot;
}

__device__ inline void neo_hce_welford_add(
    double value,
    int* count,
    double* mean,
    double* m2
) {
    const int next_count = *count + 1;
    const double delta = value - *mean;
    *mean += delta / static_cast<double>(next_count);
    const double delta_after_mean = value - *mean;
    *m2 += delta * delta_after_mean;
    *count = next_count;
}

__device__ inline void neo_hce_recompute_slot(
    int slot,
    int data_period,
    const double* __restrict__ tod_values_scratch,
    const int* __restrict__ tod_counts_scratch,
    const int* __restrict__ tod_next_scratch,
    double* __restrict__ tod_means_scratch,
    double* __restrict__ tod_m2_scratch
) {
    const int count = tod_counts_scratch[slot];
    const int start = count == data_period ? tod_next_scratch[slot] : 0;
    double mean = 0.0;
    double m2 = 0.0;
    int moment_count = 0;
    for (int offset = 0; offset < count; ++offset) {
        const int index = (start + offset) % data_period;
        const double value =
            tod_values_scratch[slot * NEO_HCE_MAX_DATA_PERIOD + index];
        neo_hce_welford_add(value, &moment_count, &mean, &m2);
    }
    tod_means_scratch[slot] = mean;
    tod_m2_scratch[slot] = m2;
}

__device__ inline void neo_hce_insert_finite(
    int slot,
    int data_period,
    double value,
    double* __restrict__ tod_values_scratch,
    int* __restrict__ tod_counts_scratch,
    int* __restrict__ tod_next_scratch,
    double* __restrict__ tod_means_scratch,
    double* __restrict__ tod_m2_scratch
) {
    int count = tod_counts_scratch[slot];
    int next = tod_next_scratch[slot];
    tod_values_scratch[slot * NEO_HCE_MAX_DATA_PERIOD + next] = value;
    if (count < data_period) {
        ++count;
    }
    next = (next + 1) % data_period;
    tod_counts_scratch[slot] = count;
    tod_next_scratch[slot] = next;
    neo_hce_recompute_slot(
        slot,
        data_period,
        tod_values_scratch,
        tod_counts_scratch,
        tod_next_scratch,
        tod_means_scratch,
        tod_m2_scratch
    );
}

__device__ inline bool neo_hce_cached_bucket(
    int slot,
    const int* __restrict__ tod_counts_scratch,
    const double* __restrict__ tod_means_scratch,
    const double* __restrict__ tod_m2_scratch,
    double* out_mean,
    double* out_icv
) {
    const int count = tod_counts_scratch[slot];
    if (count == 0) {
        return false;
    }
    const double mean = tod_means_scratch[slot];
    double icv = 1.0;
    if (mean != 0.0) {
        const double variance =
            fmax(tod_m2_scratch[slot] / static_cast<double>(count), 0.0);
        const double stdev = sqrt(variance);
        const double confidence =
            1.0 - fmin(1.0, fmax(0.0, stdev / mean)) * NEO_HCE_MAX_CONF_FACTOR;
        if (!isnan(confidence)) {
            icv = confidence;
        }
    }
    *out_mean = mean;
    *out_icv = icv;
    return true;
}

__device__ inline bool neo_hce_next_cached_bucket(
    int start_key,
    int slots_per_day,
    const int* __restrict__ tod_counts_scratch,
    const double* __restrict__ tod_means_scratch,
    const double* __restrict__ tod_m2_scratch,
    int* out_key,
    double* out_value,
    double* out_weight
) {
    for (int offset = 1; offset <= slots_per_day; ++offset) {
        const int key = (start_key + offset) % slots_per_day;
        double value = neo_hce_qnan();
        double confidence = 1.0;
        if (neo_hce_cached_bucket(
                key,
                tod_counts_scratch,
                tod_means_scratch,
                tod_m2_scratch,
                &value,
                &confidence)) {
            *out_key = key;
            *out_value = value;
            *out_weight = fmax(confidence, 0.0);
            return true;
        }
    }
    return false;
}

// Pine `init_make_window`: scan forward from the current session key and
// unshift each valid bucket. The resulting cache is farthest-first and its
// last discovered key is retained for subsequent one-step maintenance.
__device__ inline bool neo_hce_initialize_future_window(
    int current_key,
    int slots_per_day,
    int future_length,
    const int* __restrict__ tod_counts_scratch,
    const double* __restrict__ tod_means_scratch,
    const double* __restrict__ tod_m2_scratch,
    double* future_values,
    double* future_weights,
    int* future_window_key
) {
    int key = current_key;
    for (int found = 0; found < future_length; ++found) {
        int next_key = key;
        double value = neo_hce_qnan();
        double weight = 1.0;
        if (!neo_hce_next_cached_bucket(
                key,
                slots_per_day,
                tod_counts_scratch,
                tod_means_scratch,
                tod_m2_scratch,
                &next_key,
                &value,
                &weight)) {
            return false;
        }
        for (int index = found; index > 0; --index) {
            future_values[index] = future_values[index - 1];
            future_weights[index] = future_weights[index - 1];
        }
        future_values[0] = value;
        future_weights[0] = weight;
        key = next_key;
    }
    *future_window_key = key;
    return true;
}

// Pine `maintain_window`: pop the nearest cached point, then append exactly
// one next valid key by unshifting it. No current-slot rescan is permitted.
__device__ inline bool neo_hce_maintain_future_window(
    int slots_per_day,
    int future_length,
    const int* __restrict__ tod_counts_scratch,
    const double* __restrict__ tod_means_scratch,
    const double* __restrict__ tod_m2_scratch,
    double* future_values,
    double* future_weights,
    int* future_window_key
) {
    int next_key = *future_window_key;
    double value = neo_hce_qnan();
    double weight = 1.0;
    if (!neo_hce_next_cached_bucket(
            *future_window_key,
            slots_per_day,
            tod_counts_scratch,
            tod_means_scratch,
            tod_m2_scratch,
            &next_key,
            &value,
            &weight)) {
        return false;
    }
    for (int index = future_length - 1; index > 0; --index) {
        future_values[index] = future_values[index - 1];
        future_weights[index] = future_weights[index - 1];
    }
    future_values[0] = value;
    future_weights[0] = weight;
    *future_window_key = next_key;
    return true;
}

__device__ inline void neo_hce_neumaier_add(
    double value,
    double* sum,
    double* correction
) {
    const double next = *sum + value;
    if (fabs(*sum) >= fabs(value)) {
        *correction += (*sum - next) + value;
    } else {
        *correction += (value - next) + *sum;
    }
    *sum = next;
}

__device__ inline void neo_hce_add_weighted(
    double value,
    double confidence,
    double coefficient,
    double* sum,
    double* correction
) {
    const double scaled = value * confidence;
    const double term = scaled * coefficient;
    neo_hce_neumaier_add(term, sum, correction);
}

extern "C" __global__ void half_causal_estimator_neo_batch_f64(
    const long long* __restrict__ timestamps,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ hce_scratch_f64,
    int* __restrict__ hce_scratch_i32,
    double* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || n <= 0 || n_combos <= 0) {
        return;
    }
    (void)first_valid;
    (void)close;  // the CPU source is `volume`; close is not read on this path

    const double nan_value = neo_hce_qnan();
    for (int combo = 0; combo < n_combos; ++combo) {
        double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
        for (int i = 0; i < n; ++i) {
            row[i] = nan_value;
        }
    }

    if (!neo_hce_validate_all_timestamps(timestamps, n)) {
        return;
    }

    const int slots_per_day = neo_hce_infer_slots_per_day(timestamps, n);
    if (slots_per_day < 2 || slots_per_day > NEO_HCE_MAX_SLOTS_PER_DAY) {
        // `resolve_params` (:1023) rejects < 2, and a 0 here means the CPU
        // could not infer the timeframe at all. The explicit upper bound is
        // the maximum minute-resolution creator schedule and the scratch ABI.
        return;
    }

    // Strict HCE-v2 scratch layout. Every slot owns capacity for the largest
    // admitted RegistryRatio data period plus chronological Welford population
    // moments. One serial combo reuses the same scratch after a full reset.
    double* tod_values_scratch = hce_scratch_f64;
    double* tod_means_scratch =
        tod_values_scratch + NEO_HCE_MAX_SLOTS_PER_DAY * NEO_HCE_MAX_DATA_PERIOD;
    double* tod_m2_scratch = tod_means_scratch + NEO_HCE_MAX_SLOTS_PER_DAY;
    int* tod_counts_scratch = hce_scratch_i32;
    int* tod_next_scratch = tod_counts_scratch + NEO_HCE_MAX_SLOTS_PER_DAY;
    double kernel_w[NEO_HCE_MAX_WINDOW_SIZE];
    double future_values[NEO_HCE_MAX_FUTURE_LEN];
    double future_weights[NEO_HCE_MAX_FUTURE_LEN];

    for (int combo = 0; combo < n_combos; ++combo) {
        const int anchor = periods[combo];
        int resolved_data_period = 0;
        int resolved_filter_length = 0;
        if (!neo_hce_resolve_registry_anchor(
                anchor,
                &resolved_data_period,
                &resolved_filter_length)) {
            continue;
        }
        const int data_period = resolved_data_period;
        const int filter_length = resolved_filter_length;
        if (data_period <= 0 || data_period > NEO_HCE_MAX_DATA_PERIOD ||
            filter_length < 2 || filter_length > NEO_HCE_MAX_FILTER_LENGTH) {
            continue;  // defense in depth behind typed host admission
        }
        const int future_length = filter_length - 1;
        const int window_size = filter_length * 2 - 1;
        double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

        // No prior combo may leak retained samples, moments, cache keys, or
        // readiness into this row.
        for (int slot = 0; slot < slots_per_day; ++slot) {
            for (int retained = 0; retained < NEO_HCE_MAX_DATA_PERIOD; ++retained) {
                tod_values_scratch[
                    slot * NEO_HCE_MAX_DATA_PERIOD + retained
                ] = 0.0;
            }
            tod_counts_scratch[slot] = 0;
            tod_next_scratch[slot] = 0;
            tod_means_scratch[slot] = 0.0;
            tod_m2_scratch[slot] = 0.0;
        }
        for (int index = 0; index < NEO_HCE_MAX_FUTURE_LEN; ++index) {
            future_values[index] = nan_value;
            future_weights[index] = 1.0;
        }
        for (int index = 0; index < NEO_HCE_MAX_WINDOW_SIZE; ++index) {
            kernel_w[index] = 0.0;
        }

        // Registry production fixes Epanechnikov, width 20.0, symmetric
        // confidence, maximum confidence factor 1.0, smoothing 0, expected off.
        const double center = static_cast<double>(window_size - 1) * 0.5;
        double normalization = 0.0;
        for (int i = 0; i < window_size; ++i) {
            const double centered = static_cast<double>(i) - center;
            const double weight = neo_hce_epanechnikov(centered, NEO_HCE_KERNEL_WIDTH);
            normalization += weight;
            kernel_w[i] = weight;
        }
        if (normalization != 0.0) {
            for (int i = 0; i < window_size; ++i) {
                kernel_w[i] /= normalization;
            }
        }

        int future_window_key = 0;
        bool future_initialized = false;
        bool ready = false;
        bool has_previous_utc_day = false;
        long long previous_utc_day = 0;

        for (int i = 0; i < n; ++i) {
            const int slot = neo_hce_slot(timestamps[i], slots_per_day);
            if (slot < 0) {
                return;  // defense in depth after whole-input prevalidation
            }

            const long long utc_day = neo_hce_utc_day(timestamps[i]);
            const bool session_start =
                has_previous_utc_day ? (utc_day != previous_utc_day) : true;
            previous_utc_day = utc_day;
            has_previous_utc_day = true;
            if (!ready && i > window_size && session_start) {
                ready = true;
            }

            bool future_ready = false;
            if (ready) {
                if (session_start) {
                    future_ready = neo_hce_initialize_future_window(
                        slot,
                        slots_per_day,
                        future_length,
                        tod_counts_scratch,
                        tod_means_scratch,
                        tod_m2_scratch,
                        future_values,
                        future_weights,
                        &future_window_key
                    );
                    future_initialized = future_ready;
                } else if (future_initialized) {
                    future_ready = neo_hce_maintain_future_window(
                        slots_per_day,
                        future_length,
                        tod_counts_scratch,
                        tod_means_scratch,
                        tod_m2_scratch,
                        future_values,
                        future_weights,
                        &future_window_key
                    );
                }
            }

            // The causal series advances every bar, so it is full once
            // `i + 1 >= filter_length`. Compute before current TOD insertion.
            if (future_ready && i + 1 >= filter_length) {
                double sum = 0.0;
                double correction = 0.0;
                bool ok = true;
                for (int j = 0; j < future_length; ++j) {
                    const double value = future_values[j];
                    if (!isfinite(value)) {
                        ok = false;
                        break;
                    }
                    neo_hce_add_weighted(
                        value,
                        future_weights[j],
                        kernel_w[j],
                        &sum,
                        &correction
                    );
                }
                if (ok) {
                    for (int j = 0; j < filter_length; ++j) {
                        const double value = volume[i - j];
                        if (!isfinite(value)) {
                            ok = false;
                            break;
                        }
                        const double confidence =
                            (j == 0)
                                ? 1.0
                                : (2.0 - future_weights[future_length - j]);
                        neo_hce_add_weighted(
                            value,
                            confidence,
                            kernel_w[future_length + j],
                            &sum,
                            &correction
                        );
                    }
                }
                const double estimate = sum + correction;
                if (ok && isfinite(estimate)) {
                    row[i] = estimate;
                }
            }

            if (isfinite(volume[i])) {
                neo_hce_insert_finite(
                    slot,
                    data_period,
                    volume[i],
                    tod_values_scratch,
                    tod_counts_scratch,
                    tod_next_scratch,
                    tod_means_scratch,
                    tod_m2_scratch
                );
            }
        }
    }
}
