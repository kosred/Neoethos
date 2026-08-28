#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {
constexpr int ATR_LENGTH = 14;
constexpr double ATR_MULTIPLIER = 1.2;
constexpr double VOLATILITY_THRESHOLD = 1.2;
constexpr double BULLISH_LOCATION_WEIGHT = 0.15;
constexpr double BEARISH_LOCATION_WEIGHT = 0.85;

__device__ inline int lower_bound_sorted(const double* sorted, int size, double value) {
    int left = 0;
    int right = size;
    while (left < right) {
        const int mid = left + ((right - left) >> 1);
        if (sorted[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ inline void insert_sorted(double* sorted, int* size, double value, int capacity) {
    if (*size >= capacity) {
        return;
    }
    const int idx = lower_bound_sorted(sorted, *size, value);
    for (int i = *size; i > idx; --i) {
        sorted[i] = sorted[i - 1];
    }
    sorted[idx] = value;
    *size += 1;
}

__device__ inline void remove_sorted_once(double* sorted, int* size, double value) {
    const int idx = lower_bound_sorted(sorted, *size, value);
    if (idx < *size && sorted[idx] == value) {
        for (int i = idx; i + 1 < *size; ++i) {
            sorted[i] = sorted[i + 1];
        }
        *size -= 1;
    }
}

__device__ inline unsigned char bool_window_get_ago(
    const unsigned char* ring,
    int count,
    int head,
    int len,
    int ago
) {
    if (ago >= count) {
        return 0;
    }
    if (count < len) {
        return ring[count - 1 - ago];
    }
    const int latest = head == 0 ? (len - 1) : (head - 1);
    return ring[(latest + len - ago) % len];
}
}

extern "C" __global__ void range_breakout_signals_batch_f64(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
    int len,
    const int* range_lengths,
    const int* confirmation_lengths,
    int rows,
    int max_range_length,
    int max_confirmation_window,
    double* out_range_top,
    double* out_range_bottom,
    double* out_bullish,
    double* out_extra_bullish,
    double* out_bearish,
    double* out_extra_bearish,
    double* dist_ring_buffers,
    double* dist_sorted_buffers,
    double* up_volume_buffers,
    double* down_volume_buffers,
    unsigned char* under_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int range_length = range_lengths[row];
    const int confirmation_length = confirmation_lengths[row];
    const int confirmation_window = confirmation_length + 1;

    double* row_range_top = out_range_top + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_bottom =
        out_range_bottom + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish = out_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_extra_bullish =
        out_extra_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish = out_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_extra_bearish =
        out_extra_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);

    double* row_dist_ring =
        dist_ring_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_range_length);
    double* row_dist_sorted =
        dist_sorted_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_range_length);
    double* row_up_volume =
        up_volume_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_confirmation_window);
    double* row_down_volume = down_volume_buffers
        + static_cast<size_t>(row) * static_cast<size_t>(max_confirmation_window);
    unsigned char* row_under =
        under_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_confirmation_window);

    for (int i = 0; i < len; ++i) {
        row_range_top[i] = NAN;
        row_range_bottom[i] = NAN;
        row_bullish[i] = NAN;
        row_extra_bullish[i] = NAN;
        row_bearish[i] = NAN;
        row_extra_bearish[i] = NAN;
    }

    if (range_length <= 0 || range_length > max_range_length || confirmation_length <= 0
        || confirmation_window > max_confirmation_window) {
        return;
    }

    int dist_head = 0;
    int dist_count = 0;
    int dist_sorted_count = 0;
    double dist_sum = 0.0;

    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = NAN;
    double prev_close = NAN;
    bool have_prev_close = false;

    int volume_head = 0;
    int volume_count = 0;
    double up_volume_sum = 0.0;
    double down_volume_sum = 0.0;

    int under_head = 0;
    int under_count = 0;

    double prev_volatility = NAN;
    bool active_range = false;
    double active_top = NAN;
    double active_bottom = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!isfinite(o) || !isfinite(h) || !isfinite(l) || !isfinite(c) || !isfinite(v)) {
            dist_head = 0;
            dist_count = 0;
            dist_sorted_count = 0;
            dist_sum = 0.0;
            atr_count = 0;
            atr_sum = 0.0;
            atr_value = NAN;
            prev_close = NAN;
            have_prev_close = false;
            volume_head = 0;
            volume_count = 0;
            up_volume_sum = 0.0;
            down_volume_sum = 0.0;
            under_head = 0;
            under_count = 0;
            prev_volatility = NAN;
            active_range = false;
            active_top = NAN;
            active_bottom = NAN;
            continue;
        }

        const double tr_prev_close = have_prev_close ? prev_close : c;
        const double tr = fmax(h - l, fmax(fabs(h - tr_prev_close), fabs(l - tr_prev_close)));
        prev_close = c;
        have_prev_close = true;

        bool atr_ready = false;
        if (atr_count < ATR_LENGTH) {
            atr_count += 1;
            atr_sum += tr;
            if (atr_count == ATR_LENGTH) {
                atr_value = atr_sum / static_cast<double>(ATR_LENGTH);
                atr_ready = true;
            }
        } else {
            atr_value =
                ((atr_value * static_cast<double>(ATR_LENGTH - 1)) + tr) / static_cast<double>(ATR_LENGTH);
            atr_ready = true;
        }

        const double dist_value = fabs(c - o);
        if (dist_count == range_length) {
            const double old = row_dist_ring[dist_head];
            dist_sum -= old;
            remove_sorted_once(row_dist_sorted, &dist_sorted_count, old);
            row_dist_ring[dist_head] = dist_value;
            dist_head += 1;
            if (dist_head == range_length) {
                dist_head = 0;
            }
        } else {
            row_dist_ring[dist_count] = dist_value;
            dist_count += 1;
            if (dist_count == range_length) {
                dist_head = 0;
            }
        }
        dist_sum += dist_value;
        insert_sorted(row_dist_sorted, &dist_sorted_count, dist_value, max_range_length);

        double volatility = NAN;
        if (dist_count == range_length) {
            const double median =
                (range_length & 1) == 1
                    ? row_dist_sorted[range_length >> 1]
                    : (row_dist_sorted[(range_length >> 1) - 1]
                       + row_dist_sorted[range_length >> 1])
                        * 0.5;
            if (median > 0.0) {
                volatility = (dist_sum / static_cast<double>(range_length)) / median;
            }
        }

        const bool current_isunder = isfinite(volatility) && volatility < VOLATILITY_THRESHOLD;
        double up_volume = 0.0;
        double down_volume = 0.0;
        if (c > o) {
            up_volume = v;
        } else if (c < o) {
            down_volume = v;
        } else {
            up_volume = v * 0.5;
            down_volume = v * 0.5;
        }

        if (volume_count == confirmation_window) {
            up_volume_sum -= row_up_volume[volume_head];
            down_volume_sum -= row_down_volume[volume_head];
            row_up_volume[volume_head] = up_volume;
            row_down_volume[volume_head] = down_volume;
            volume_head += 1;
            if (volume_head == confirmation_window) {
                volume_head = 0;
            }
        } else {
            row_up_volume[volume_count] = up_volume;
            row_down_volume[volume_count] = down_volume;
            volume_count += 1;
            if (volume_count == confirmation_window) {
                volume_head = 0;
            }
        }
        up_volume_sum += up_volume;
        down_volume_sum += down_volume;

        if (under_count == confirmation_window) {
            row_under[under_head] = current_isunder ? 1 : 0;
            under_head += 1;
            if (under_head == confirmation_window) {
                under_head = 0;
            }
        } else {
            row_under[under_count] = current_isunder ? 1 : 0;
            under_count += 1;
            if (under_count == confirmation_window) {
                under_head = 0;
            }
        }

        const bool ready = isfinite(volatility) && atr_ready && isfinite(prev_volatility)
            && volume_count == confirmation_window && under_count == confirmation_window;

        double range_top = NAN;
        double range_bottom = NAN;
        double bullish = NAN;
        double extra_bullish = NAN;
        double bearish = NAN;
        double extra_bearish = NAN;

        if (ready) {
            const bool under_ago =
                bool_window_get_ago(
                    row_under,
                    under_count,
                    under_head,
                    confirmation_window,
                    confirmation_length
                )
                != 0;
            const bool crossed_under =
                prev_volatility >= VOLATILITY_THRESHOLD && volatility < VOLATILITY_THRESHOLD;
            if (!active_range && crossed_under && current_isunder && under_ago) {
                const double offset = atr_value * ATR_MULTIPLIER;
                active_top = c + offset;
                active_bottom = c - offset;
                active_range = true;
            }

            if (active_range) {
                range_top = active_top;
                range_bottom = active_bottom;
                if (c > active_top || c < active_bottom) {
                    const bool bullish_break = c > active_top;
                    const double location = active_bottom
                        + (active_top - active_bottom)
                            * (bullish_break ? BULLISH_LOCATION_WEIGHT : BEARISH_LOCATION_WEIGHT);
                    const bool bullish_volume_bias = up_volume_sum > down_volume_sum;
                    if (bullish_break) {
                        bullish = location;
                        if (bullish_volume_bias) {
                            extra_bullish = location;
                        }
                    } else {
                        bearish = location;
                        if (!bullish_volume_bias) {
                            extra_bearish = location;
                        }
                    }
                    active_range = false;
                    active_top = NAN;
                    active_bottom = NAN;
                }
            }

            row_range_top[i] = range_top;
            row_range_bottom[i] = range_bottom;
            row_bullish[i] = bullish;
            row_extra_bullish[i] = extra_bullish;
            row_bearish[i] = bearish;
            row_extra_bearish[i] = extra_bearish;
        }

        prev_volatility = isfinite(volatility) ? volatility : NAN;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// range_breakout_signals_batch_f64 above is genuine double-in/double-out, but
// it takes 22 parameters, writes SIX output matrices and demands five
// caller-allocated scratch matrices. The f64 lane launches one shape -- (open,
// high, low, close, volume, n, periods, n_combos, first_valid, out) -- and
// allocates ONE output matrix, so that entry point cannot be reached from it.
//
// CPU REFERENCE
//   src/indicators/range_breakout_signals.rs:1381
//     range_breakout_signals_with_kernel -> :1298 compute_row
//     -> :874 RangeBreakoutSignalsDefaultState::update (the path every lane row
//     takes: :1319 selects it when range_length and confirmation_length are the
//     defaults, which the pins below are).
//   MedianSmaWindow20::push :641   AtrState::update :476
//   VolumeWindow6::push     :745   BoolWindow6::push/get_ago :796/:814
//   split_volume :1000             location :1013
//
// THE COLUMN THIS EMITS is range_top, which is what output_id == "value"
// resolves to (cpu_batch.rs -- "range_top" || "value").
//
// PERIOD-INVARIANT. compute_range_breakout_signals_batch reads range_length and
// confirmation_length and NEVER `period`, so five swept periods give five
// identical CPU columns and this kernel emits five identical rows. Both CPU
// defaults (20 and 5) are pinned below, along with the four module constants
// ATR_LENGTH 14, ATR_MULTIPLIER 1.2, VOLATILITY_THRESHOLD 1.2 and the two
// location weights (:74-78).
//
// SHAPE: one thread per combo, bars ascending. A BREAKOUT STATE MACHINE with a
// small local state: an active range that is armed by a volatility crossing and
// disarmed by the close leaving it, plus a 20-deep median/mean window, a Wilder
// ATR, a 6-deep signed-volume window and a 6-deep boolean window. Every one of
// those is carried, and a non-finite bar clears all of them (:880-889).
//
// ONE FIDELITY FIX relative to the 22-parameter kernel above. The CPU applies
// the active-range block OUTSIDE the `ready` gate (:929-955 sits after the
// `if ready { ... }` at :904-927 and is not nested in it); only the ARMING is
// gated. The kernel above nests the whole block inside `if (ready)`, so on a
// bar where readiness lapses -- which happens whenever the previous bar's
// volatility was None, since `previous_volatility.is_finite()` is one of the
// five readiness terms -- it FAILS TO DISARM a range the CPU disarms, and every
// later bar then reports a stale range_top. This twin follows the CPU: arm
// under `ready`, run the breakout unconditionally, publish under `ready`.
//
// NaN SEMANTICS: `hl.max(hc).max(lc)` in AtrState (:482-484) is f64::max, which
// returns the non-NaN operand, so fmax is used rather than a comparison chain.
//
// FIRST VALID IS NOT READ: `ready` is derived from the state machine's own
// counters, and compute_row writes every index (NaN where update returns None),
// so the alloc_with_nan_prefix warmup is overwritten wholesale. The lane row
// declares F64FirstValidRule::Ignored.
//
// OPEN AND VOLUME ARE INPUTS -- the bar's body is close - open (:895) and the
// confirmation bias is a signed volume sum (:1000-1008) -- so the lane row
// declares F64InputKind::Ohlcv5.
//
// f64 END TO END: double literals, double fmax/fabs, no f32-suffixed math
// function, no fast-math intrinsic, and no epsilon: the CPU's guards here are
// literal `median > 0.0` and `volatility < 1.2` comparisons.
// ---------------------------------------------------------------------------

#define RBS_NEO_RANGE_LENGTH 20
#define RBS_NEO_CONFIRMATION_LENGTH 5
#define RBS_NEO_CONFIRMATION_WINDOW (RBS_NEO_CONFIRMATION_LENGTH + 1)

__device__ __forceinline__ double rbs_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void range_breakout_signals_neo_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
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
    const double qnan = rbs_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    // MedianSmaWindow20 (:634-640): the CPU's own fixed arrays.
    double dist_ring[RBS_NEO_RANGE_LENGTH];
    double dist_sorted[RBS_NEO_RANGE_LENGTH];
    int dist_head = 0;
    int dist_count = 0;
    double dist_sum = 0.0;

    // AtrState (:444-451).
    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = qnan;
    bool atr_seeded = false;
    double atr_prev_close = qnan;
    bool atr_have_prev_close = false;

    // VolumeWindow6 (:707-714) and BoolWindow6 (:774-778).
    double up_ring[RBS_NEO_CONFIRMATION_WINDOW];
    double down_ring[RBS_NEO_CONFIRMATION_WINDOW];
    int volume_head = 0;
    int volume_count = 0;
    double up_sum = 0.0;
    double down_sum = 0.0;
    unsigned char under_ring[RBS_NEO_CONFIRMATION_WINDOW];
    int under_head = 0;
    int under_count = 0;

    double prev_volatility = qnan;
    bool active_range = false;
    double active_top = qnan;
    double active_bottom = qnan;

    for (int i = 0; i < n; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!isfinite(o) || !isfinite(h) || !isfinite(l) || !isfinite(c) || !isfinite(v)) {
            // :880-889 -- reset() clears every sub-state.
            dist_head = 0;
            dist_count = 0;
            dist_sum = 0.0;
            atr_count = 0;
            atr_sum = 0.0;
            atr_value = qnan;
            atr_seeded = false;
            atr_prev_close = qnan;
            atr_have_prev_close = false;
            volume_head = 0;
            volume_count = 0;
            up_sum = 0.0;
            down_sum = 0.0;
            under_head = 0;
            under_count = 0;
            prev_volatility = qnan;
            active_range = false;
            active_top = qnan;
            active_bottom = qnan;
            continue;
        }

        const double previous_volatility = prev_volatility;

        // AtrState::update (:476-500).
        {
            const double pc = atr_have_prev_close ? atr_prev_close : c;
            const double tr = fmax(h - l, fmax(fabs(h - pc), fabs(l - pc)));
            atr_prev_close = c;
            atr_have_prev_close = true;
            if (atr_count < ATR_LENGTH) {
                atr_count += 1;
                atr_sum += tr;
                if (atr_count == ATR_LENGTH) {
                    atr_value = atr_sum / static_cast<double>(ATR_LENGTH);
                    atr_seeded = true;
                }
            } else {
                atr_value = ((atr_value * static_cast<double>(ATR_LENGTH - 1)) + tr) /
                            static_cast<double>(ATR_LENGTH);
                atr_seeded = true;
            }
        }
        const bool atr_ready = atr_seeded;

        // MedianSmaWindow20::push((close - open).abs()) (:641-704).
        double volatility = qnan;
        bool volatility_ready = false;
        {
            const double value = fabs(c - o);
            int sorted_len;
            if (dist_count == RBS_NEO_RANGE_LENGTH) {
                const double old = dist_ring[dist_head];
                dist_sum -= old;
                // :651-652 -- find the FIRST slot equal to the departing value.
                int index = 0;
                while (dist_sorted[index] != old) {
                    index += 1;
                }
                while (index + 1 < RBS_NEO_RANGE_LENGTH) {
                    dist_sorted[index] = dist_sorted[index + 1];
                    index += 1;
                }
                dist_ring[dist_head] = value;
                dist_head += 1;
                if (dist_head == RBS_NEO_RANGE_LENGTH) {
                    dist_head = 0;
                }
                sorted_len = RBS_NEO_RANGE_LENGTH - 1;
            } else {
                dist_ring[dist_count] = value;
                dist_count += 1;
                if (dist_count == RBS_NEO_RANGE_LENGTH) {
                    dist_head = 0;
                }
                sorted_len = dist_count - 1;
            }
            dist_sum += value;
            // :691-697 -- insert AFTER the last element <= value (upper bound).
            int index = 0;
            while (index < sorted_len && dist_sorted[index] <= value) {
                index += 1;
            }
            for (int j = sorted_len; j > index; --j) {
                dist_sorted[j] = dist_sorted[j - 1];
            }
            dist_sorted[index] = value;

            if (dist_count == RBS_NEO_RANGE_LENGTH) {
                const double median = (dist_sorted[(RBS_NEO_RANGE_LENGTH >> 1) - 1] +
                                       dist_sorted[RBS_NEO_RANGE_LENGTH >> 1]) *
                                      0.5;
                const double mean = dist_sum / static_cast<double>(RBS_NEO_RANGE_LENGTH);
                if (median > 0.0) {
                    volatility = mean / median;
                    volatility_ready = true;
                }
            }
        }

        const bool current_isunder = volatility_ready && volatility < VOLATILITY_THRESHOLD;

        // split_volume (:1000-1008).
        double up_volume;
        double down_volume;
        if (c > o) {
            up_volume = v;
            down_volume = 0.0;
        } else if (c < o) {
            up_volume = 0.0;
            down_volume = v;
        } else {
            const double half = v * 0.5;
            up_volume = half;
            down_volume = half;
        }

        // VolumeWindow6::push (:745-765).
        if (volume_count == RBS_NEO_CONFIRMATION_WINDOW) {
            up_sum -= up_ring[volume_head];
            down_sum -= down_ring[volume_head];
            up_ring[volume_head] = up_volume;
            down_ring[volume_head] = down_volume;
            volume_head += 1;
            if (volume_head == RBS_NEO_CONFIRMATION_WINDOW) {
                volume_head = 0;
            }
        } else {
            up_ring[volume_count] = up_volume;
            down_ring[volume_count] = down_volume;
            volume_count += 1;
            if (volume_count == RBS_NEO_CONFIRMATION_WINDOW) {
                volume_head = 0;
            }
        }
        up_sum += up_volume;
        down_sum += down_volume;

        // BoolWindow6::push (:796-812).
        if (under_count == RBS_NEO_CONFIRMATION_WINDOW) {
            under_ring[under_head] = current_isunder ? 1 : 0;
            under_head += 1;
            if (under_head == RBS_NEO_CONFIRMATION_WINDOW) {
                under_head = 0;
            }
        } else {
            under_ring[under_count] = current_isunder ? 1 : 0;
            under_count += 1;
            if (under_count == RBS_NEO_CONFIRMATION_WINDOW) {
                under_head = 0;
            }
        }

        const bool ready = volatility_ready && atr_ready && isfinite(previous_volatility) &&
                           volume_count == RBS_NEO_CONFIRMATION_WINDOW &&
                           under_count == RBS_NEO_CONFIRMATION_WINDOW;

        if (ready) {
            // BoolWindow6::get_ago(confirmation_length) (:814-833).
            bool under_ago = false;
            if (RBS_NEO_CONFIRMATION_LENGTH < under_count) {
                if (under_count < RBS_NEO_CONFIRMATION_WINDOW) {
                    under_ago =
                        under_ring[under_count - 1 - RBS_NEO_CONFIRMATION_LENGTH] != 0;
                } else {
                    const int latest =
                        (under_head == 0) ? (RBS_NEO_CONFIRMATION_WINDOW - 1) : (under_head - 1);
                    const int index = (latest >= RBS_NEO_CONFIRMATION_LENGTH)
                        ? (latest - RBS_NEO_CONFIRMATION_LENGTH)
                        : (latest + RBS_NEO_CONFIRMATION_WINDOW - RBS_NEO_CONFIRMATION_LENGTH);
                    under_ago = under_ring[index] != 0;
                }
            }
            const bool crossed_under = previous_volatility >= VOLATILITY_THRESHOLD &&
                                       volatility < VOLATILITY_THRESHOLD;
            if (!active_range && crossed_under && current_isunder && under_ago) {
                const double offset = atr_value * ATR_MULTIPLIER;
                active_top = c + offset;
                active_bottom = c - offset;
                active_range = true;
            }
        }

        // :929-955 -- OUTSIDE the readiness gate on the CPU. A range armed on an
        // earlier bar is disarmed by a break even on a bar that is not `ready`.
        double range_top = qnan;
        if (active_range) {
            range_top = active_top;
            if (c > active_top || c < active_bottom) {
                active_range = false;
                active_top = qnan;
                active_bottom = qnan;
            }
        }

        prev_volatility = volatility_ready ? volatility : qnan;

        if (ready) {
            row[i] = range_top;
        }
    }
}
