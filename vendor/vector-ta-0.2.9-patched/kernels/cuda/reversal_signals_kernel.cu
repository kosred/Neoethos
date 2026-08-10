#include <cmath>
#include <cstddef>

namespace {

constexpr int VOLUME_SMA_PERIOD = 20;
constexpr int TREND_MA_SMA = 0;
constexpr int TREND_MA_EMA = 1;
constexpr int TREND_MA_WMA = 2;
constexpr int TREND_MA_VWMA = 3;

__device__ inline bool is_valid_ohlcv(double open, double high, double low, double close, double volume) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close) && isfinite(volume);
}

__device__ inline void reset_queue(int* size) {
    *size = 0;
}

__device__ inline void push_min_queue(
    int* indices,
    double* values,
    int* size,
    int index,
    double value
) {
    while (*size > 0 && values[*size - 1] > value) {
        *size -= 1;
    }
    indices[*size] = index;
    values[*size] = value;
    *size += 1;
}

__device__ inline void push_max_queue(
    int* indices,
    double* values,
    int* size,
    int index,
    double value
) {
    while (*size > 0 && values[*size - 1] < value) {
        *size -= 1;
    }
    indices[*size] = index;
    values[*size] = value;
    *size += 1;
}

__device__ inline void prune_queue_front(
    int* indices,
    double* values,
    int* size,
    int min_index
) {
    int shift = 0;
    while (shift < *size && indices[shift] < min_index) {
        shift += 1;
    }
    if (shift == 0) {
        return;
    }
    for (int i = shift; i < *size; ++i) {
        indices[i - shift] = indices[i];
        values[i - shift] = values[i];
    }
    *size -= shift;
}

__device__ inline double queue_current(const double* values, int size) {
    return size > 0 ? values[0] : NAN;
}

}

extern "C" __global__ void reversal_signals_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lookback_periods,
    const int* __restrict__ confirmation_periods,
    const int* __restrict__ trend_ma_periods,
    const int* __restrict__ ma_step_periods,
    int use_volume_confirmation,
    int trend_ma_kind,
    int rows,
    int max_lookback,
    int max_trend_ma_period,
    double* __restrict__ ma_price_scratch,
    double* __restrict__ ma_volume_scratch,
    double* __restrict__ volume_sma_scratch,
    int* __restrict__ low_idx_scratch,
    double* __restrict__ low_val_scratch,
    int* __restrict__ high_idx_scratch,
    double* __restrict__ high_val_scratch,
    double* __restrict__ out_buy_signal,
    double* __restrict__ out_sell_signal,
    double* __restrict__ out_stepped_ma,
    double* __restrict__ out_state
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int lookback_period = lookback_periods[row];
    const int confirmation_period = confirmation_periods[row];
    const int trend_ma_period = trend_ma_periods[row];
    const int ma_step_period = ma_step_periods[row];
    if (lookback_period <= 0 || confirmation_period < 0 || trend_ma_period <= 0 || ma_step_period <= 0) {
        return;
    }

    double* row_ma_price =
        ma_price_scratch + static_cast<size_t>(row) * static_cast<size_t>(max_trend_ma_period);
    double* row_ma_volume =
        ma_volume_scratch + static_cast<size_t>(row) * static_cast<size_t>(max_trend_ma_period);
    double* row_volume_sma =
        volume_sma_scratch + static_cast<size_t>(row) * static_cast<size_t>(VOLUME_SMA_PERIOD);
    int* row_low_idx =
        low_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(max_lookback);
    double* row_low_val =
        low_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(max_lookback);
    int* row_high_idx =
        high_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(max_lookback);
    double* row_high_val =
        high_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(max_lookback);

    double* row_buy = out_buy_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_sell = out_sell_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_stepped = out_stepped_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_buy[i] = NAN;
        row_sell[i] = NAN;
        row_stepped[i] = NAN;
        row_state[i] = NAN;
    }

    int valid_run = 0;

    double ma_sum = 0.0;
    double ma_weighted_sum = 0.0;
    double ma_pv_sum = 0.0;
    double ma_v_sum = 0.0;
    double ema_alpha = 2.0 / (static_cast<double>(trend_ma_period) + 1.0);
    double ema_value = NAN;
    bool ema_initialized = false;
    int ma_count = 0;
    int ma_head = 0;

    double volume_sma_sum = 0.0;
    int volume_sma_count = 0;
    int volume_sma_head = 0;

    int low_queue_size = 0;
    int high_queue_size = 0;

    bool bull_candidate = false;
    bool bear_candidate = false;
    double bull_low = 0.0;
    double bull_high = 0.0;
    double bear_low = 0.0;
    double bear_high = 0.0;
    bool bull_confirmed = false;
    bool bear_confirmed = false;
    int bull_counter = 0;
    int bear_counter = 0;

    double stepped_ma = NAN;
    int ma_last_update_bar = 0;
    int ma_direction = 1;
    const int prev_span = lookback_period > 0 ? lookback_period - 1 : 0;
    const double ma_norm =
        static_cast<double>(trend_ma_period) * static_cast<double>(trend_ma_period + 1) * 0.5;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!is_valid_ohlcv(o, h, l, c, v)) {
            valid_run = 0;
            ma_sum = 0.0;
            ma_weighted_sum = 0.0;
            ma_pv_sum = 0.0;
            ma_v_sum = 0.0;
            ema_value = NAN;
            ema_initialized = false;
            ma_count = 0;
            ma_head = 0;
            volume_sma_sum = 0.0;
            volume_sma_count = 0;
            volume_sma_head = 0;
            reset_queue(&low_queue_size);
            reset_queue(&high_queue_size);
            bull_candidate = false;
            bear_candidate = false;
            bull_low = 0.0;
            bull_high = 0.0;
            bear_low = 0.0;
            bear_high = 0.0;
            bull_confirmed = false;
            bear_confirmed = false;
            bull_counter = 0;
            bear_counter = 0;
            stepped_ma = NAN;
            ma_last_update_bar = 0;
            ma_direction = 1;
            continue;
        }

        valid_run += 1;
        row_buy[i] = 0.0;
        row_sell[i] = 0.0;

        bool ma_ready = false;
        double ma_current = NAN;
        if (trend_ma_kind == TREND_MA_SMA) {
            if (ma_count == trend_ma_period) {
                ma_sum -= row_ma_price[ma_head];
            } else {
                ma_count += 1;
            }
            row_ma_price[ma_head] = c;
            ma_sum += c;
            ma_head = ma_head + 1 == trend_ma_period ? 0 : ma_head + 1;
            if (ma_count == trend_ma_period) {
                ma_current = ma_sum / static_cast<double>(trend_ma_period);
                ma_ready = true;
            }
        } else if (trend_ma_kind == TREND_MA_EMA) {
            if (!ema_initialized) {
                ema_initialized = true;
                ema_value = c;
            } else {
                ema_value = ema_alpha * c + (1.0 - ema_alpha) * ema_value;
            }
            ma_current = ema_value;
            ma_ready = true;
        } else if (trend_ma_kind == TREND_MA_WMA) {
            if (ma_count == trend_ma_period) {
                const double old_sum = ma_sum;
                const double oldest = row_ma_price[ma_head];
                row_ma_price[ma_head] = c;
                ma_head = ma_head + 1 == trend_ma_period ? 0 : ma_head + 1;
                ma_sum = old_sum - oldest + c;
                ma_weighted_sum = ma_weighted_sum - old_sum + static_cast<double>(trend_ma_period) * c;
                ma_current = ma_weighted_sum / ma_norm;
                ma_ready = true;
            } else {
                row_ma_price[ma_count] = c;
                ma_count += 1;
                ma_sum += c;
                ma_weighted_sum += static_cast<double>(ma_count) * c;
                if (ma_count == trend_ma_period) {
                    ma_current = ma_weighted_sum / ma_norm;
                    ma_ready = true;
                }
            }
        } else if (trend_ma_kind == TREND_MA_VWMA) {
            if (ma_count == trend_ma_period) {
                ma_pv_sum -= row_ma_price[ma_head] * row_ma_volume[ma_head];
                ma_v_sum -= row_ma_volume[ma_head];
            } else {
                ma_count += 1;
            }
            row_ma_price[ma_head] = c;
            row_ma_volume[ma_head] = v;
            ma_pv_sum += c * v;
            ma_v_sum += v;
            ma_head = ma_head + 1 == trend_ma_period ? 0 : ma_head + 1;
            if (ma_count == trend_ma_period && ma_v_sum != 0.0) {
                ma_current = ma_pv_sum / ma_v_sum;
                ma_ready = true;
            }
        }

        bool volume_avg_ready = false;
        double volume_avg = NAN;
        if (use_volume_confirmation != 0) {
            if (volume_sma_count == VOLUME_SMA_PERIOD) {
                volume_sma_sum -= row_volume_sma[volume_sma_head];
            } else {
                volume_sma_count += 1;
            }
            row_volume_sma[volume_sma_head] = v;
            volume_sma_sum += v;
            volume_sma_head = volume_sma_head + 1 == VOLUME_SMA_PERIOD ? 0 : volume_sma_head + 1;
            if (volume_sma_count == VOLUME_SMA_PERIOD) {
                volume_avg = volume_sma_sum / static_cast<double>(VOLUME_SMA_PERIOD);
                volume_avg_ready = true;
            }
        }
        const bool volume_is_high = volume_avg_ready && v > volume_avg;

        const bool has_prev_window = prev_span == 0 || valid_run > prev_span;
        const bool bull_candidate_trigger = prev_span == 0
            ? true
            : (has_prev_window && low_queue_size > 0 && c < queue_current(row_low_val, low_queue_size));
        const bool bear_candidate_trigger = prev_span == 0
            ? true
            : (has_prev_window && high_queue_size > 0 && c > queue_current(row_high_val, high_queue_size));

        if (bear_candidate_trigger) {
            bear_candidate = true;
            bear_low = l;
            bear_high = h;
            bear_confirmed = false;
            bear_counter = 0;
        }

        if (bear_candidate) {
            bear_counter += 1;
            if (c > bear_high) {
                bear_candidate = false;
            }
        }

        bool bear_condition = false;
        if (bear_candidate && c < bear_low && !bear_confirmed &&
            bear_counter <= confirmation_period + 1) {
            bear_confirmed = true;
            bear_condition = true;
        }

        if (bear_condition && (use_volume_confirmation == 0 || volume_is_high)) {
            row_sell[i] = 1.0;
        }

        if (bull_candidate_trigger) {
            bull_candidate = true;
            bull_low = l;
            bull_high = h;
            bull_confirmed = false;
            bull_counter = 0;
        }

        if (bull_candidate) {
            bull_counter += 1;
            if (c < bull_low) {
                bull_candidate = false;
            }
        }

        bool bull_condition = false;
        if (bull_candidate && c > bull_high && !bull_confirmed &&
            bull_counter <= confirmation_period + 1) {
            bull_confirmed = true;
            bull_condition = true;
        }

        if (bull_condition && (use_volume_confirmation == 0 || volume_is_high)) {
            row_buy[i] = 1.0;
        }

        if (ma_ready) {
            if (isnan(stepped_ma)) {
                stepped_ma = ma_current;
                ma_last_update_bar = i;
            } else if (ma_direction == 1) {
                if (c < stepped_ma) {
                    ma_direction = -1;
                    stepped_ma = ma_current;
                    ma_last_update_bar = i;
                } else if (i - ma_last_update_bar >= ma_step_period) {
                    stepped_ma = stepped_ma > ma_current ? stepped_ma : ma_current;
                    ma_last_update_bar = i;
                }
            } else if (c > stepped_ma) {
                ma_direction = 1;
                stepped_ma = ma_current;
                ma_last_update_bar = i;
            } else if (i - ma_last_update_bar >= ma_step_period) {
                stepped_ma = stepped_ma < ma_current ? stepped_ma : ma_current;
                ma_last_update_bar = i;
            }

            row_stepped[i] = stepped_ma;
            row_state[i] = static_cast<double>(ma_direction);
        }

        push_min_queue(row_low_idx, row_low_val, &low_queue_size, i, l);
        push_max_queue(row_high_idx, row_high_val, &high_queue_size, i, h);
        const int min_index = i + 1 - prev_span > 0 ? i + 1 - prev_span : 0;
        prune_queue_front(row_low_idx, row_low_val, &low_queue_size, min_index);
        prune_queue_front(row_high_idx, row_high_val, &high_queue_size, min_index);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/reversal_signals.rs `compute_default_row`
 *   (:1011-1215), reached from `reversal_signals_with_kernel` (:1223) when
 *   `is_default_single_params` holds -- which it does for every combo the
 *   batch dispatcher builds, because that dispatcher reads only NAMED
 *   parameters and this kernel pins all of them at their CPU defaults.
 *   Batch dispatcher: cpu_batch.rs:7274 -- output "value" is an ALIAS OF
 *   "stepped_ma" (:7323), so this kernel emits `stepped_ma`, not the buy or
 *   sell flags and not `state`.
 *
 * WHY A SECOND ENTRY POINT: `reversal_signals_batch_f64` (:76) takes 26
 *   parameters and emits four series. The lane launches
 *   (open, high, low, close, volume, n, periods, n_combos, first_valid, out).
 *
 * INPUT: the FULL bar -- extract_ohlcv_full_input (cpu_batch.rs:7278) -- and
 *   OPEN is a genuine input: `is_valid_ohlcv` (:347) tests it, so a bar with a
 *   non-finite open RESETS the whole state (:1052-1069) even when high, low,
 *   close and volume are all fine. F64InputKind::Ohlcv5, NOT Hlcv: a
 *   four-pointer shape would never see that bar.
 *
 * FIRST-VALID IGNORED: the loop walks EVERY bar from 0 and resets in place on
 *   an invalid one, so there is no single warmup index. `valid_run` is a
 *   per-segment counter, not a global first-valid.
 *
 * PERIOD-INVARIANT: the CPU batch reads `lookback_period`,
 *   `confirmation_period`, `use_volume_confirmation`, `trend_ma_period`,
 *   `trend_ma_type` and `ma_step_period` (cpu_batch.rs:7286-7295) and never
 *   `period`. All are pinned at the CPU defaults (12 / 3 / true / 50 / "EMA"
 *   / 33), so every row of a sweep is byte-identical.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. The stepped MA is a RATCHET:
 *   its direction latch, its last-update bar and its level all carry across
 *   bars, and the level can only move in the latched direction until the close
 *   crosses it. Not bar-parallel under any reformulation.
 *
 * ARITHMETIC taken verbatim:
 *   * the EMA step is `alpha * c + (1 - alpha) * ema` (:1080) -- TWO products
 *     and one add, THREE roundings. NOT a fused `ema + alpha*(c - ema)`, which
 *     is two. The seed is the first valid close itself (:1078), not a mean.
 *   * `alpha` is `2 / (trend_ma_period + 1)` formed once (:1025).
 *   * the volume SMA maintains its sum with `+= v - old` (:1099) once the ring
 *     is full, and with a plain `+= v` while filling (:1087).
 *   * `stepped_ma.max(ema_value)` / `.min(ema_value)` (:1058, :1066) are
 *     f64::max / f64::min, hence fmax / fmin, which return the non-NaN
 *     operand. An if-chain here would let a NaN survive into every later bar
 *     of the ratchet.
 *
 * EPSILON: there is none. Every CPU guard on this path is a strict order
 *   comparison against another price, and no tolerance is imported.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* reversal_signals.rs:28-34, matching cpu_batch.rs:7286-7295. */
#define NEO_RS_LOOKBACK_PERIOD     12
#define NEO_RS_CONFIRMATION_PERIOD 3
#define NEO_RS_TREND_MA_PERIOD     50
#define NEO_RS_MA_STEP_PERIOD      33
#define NEO_RS_VOLUME_SMA_PERIOD   20
#define NEO_RS_PREV_SPAN           (NEO_RS_LOOKBACK_PERIOD - 1)
#define NEO_RS_CONFIRM_LIMIT       (NEO_RS_CONFIRMATION_PERIOD + 1)

extern "C" __global__
void reversal_signals_neo_batch_f64(const double* __restrict__ open,
                                    const double* __restrict__ high,
                                    const double* __restrict__ low,
                                    const double* __restrict__ close,
                                    const double* __restrict__ volume,
                                    int n,
                                    const int* __restrict__ periods,
                                    int n_combos,
                                    int first_valid,
                                    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* handled in place -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const double ema_alpha = 2.0 / ((double)NEO_RS_TREND_MA_PERIOD + 1.0);
    const double volume_denom = (double)NEO_RS_VOLUME_SMA_PERIOD;

    double volume_buf[NEO_RS_VOLUME_SMA_PERIOD];
    double lows[NEO_RS_PREV_SPAN];
    double highs[NEO_RS_PREV_SPAN];

    int    volume_pos = 0, volume_count = 0;
    double volume_sum = 0.0;
    int    prev_pos = 0, prev_count = 0;

    /* ReversalCandidateState::default (:776) */
    bool   bull_candidate = false, bear_candidate = false;
    bool   bull_confirmed = false, bear_confirmed = false;
    double bull_low = 0.0, bull_high = 0.0, bear_low = 0.0, bear_high = 0.0;
    int    bull_counter = 0, bear_counter = 0;

    int    valid_run = 0;
    double ema_value = NEO_F64_NAN;
    bool   ema_initialized = false;
    double stepped_ma = NEO_F64_NAN;
    int    ma_last_update_bar = 0;
    int    ma_direction = 1;

    for (int i = 0; i < n; ++i) {
        const double op = open[i];
        const double h  = high[i];
        const double l  = low[i];
        const double c  = close[i];
        const double v  = volume[i];

        /* is_valid_ohlcv (:347) */
        if (!(isfinite(op) && isfinite(h) && isfinite(l) && isfinite(c) && isfinite(v))) {
            o[i] = NEO_F64_NAN;
            volume_pos = 0; volume_count = 0; volume_sum = 0.0;
            prev_pos = 0; prev_count = 0;
            bull_candidate = false; bear_candidate = false;
            bull_confirmed = false; bear_confirmed = false;
            bull_low = 0.0; bull_high = 0.0; bear_low = 0.0; bear_high = 0.0;
            bull_counter = 0; bear_counter = 0;
            valid_run = 0;
            ema_value = NEO_F64_NAN; ema_initialized = false;
            stepped_ma = NEO_F64_NAN;
            ma_last_update_bar = 0;
            ma_direction = 1;
            continue;
        }

        valid_run += 1;

        if (!ema_initialized) { ema_initialized = true; ema_value = c; }
        else                  { ema_value = ema_alpha * c + (1.0 - ema_alpha) * ema_value; }

        /* The volume SMA (:1085-1101). `volume_is_high` is computed because the
         * CPU computes it; the `stepped_ma` column does not read it, but the
         * ring state it maintains is shared with nothing else, so it is kept
         * exactly rather than elided -- eliding it would be a different
         * program if a later output is ever added. */
        bool have_avg = false; double volume_avg = 0.0;
        if (volume_count < NEO_RS_VOLUME_SMA_PERIOD) {
            volume_buf[volume_count] = v;
            volume_count += 1;
            volume_sum += v;
            if (volume_count == NEO_RS_VOLUME_SMA_PERIOD) {
                volume_avg = volume_sum / volume_denom; have_avg = true;
            }
        } else {
            const double old = volume_buf[volume_pos];
            volume_buf[volume_pos] = v;
            volume_pos += 1; if (volume_pos == NEO_RS_VOLUME_SMA_PERIOD) volume_pos = 0;
            volume_sum += v - old;
            volume_avg = volume_sum / volume_denom; have_avg = true;
        }
        const bool volume_is_high = have_avg && (v > volume_avg);
        (void)volume_is_high;

        double min_prev = INFINITY;
        double max_prev = -INFINITY;
        if (valid_run > NEO_RS_PREV_SPAN) {
            for (int j = 0; j < NEO_RS_PREV_SPAN; ++j) {
                const double prev_low  = lows[j];
                const double prev_high = highs[j];
                if (prev_low < min_prev)  min_prev = prev_low;
                if (prev_high > max_prev) max_prev = prev_high;
            }
        }
        const bool bull_trigger = (valid_run > NEO_RS_PREV_SPAN) && (c < min_prev);
        const bool bear_trigger = (valid_run > NEO_RS_PREV_SPAN) && (c > max_prev);

        if (bear_trigger) {
            bear_candidate = true; bear_low = l; bear_high = h;
            bear_confirmed = false; bear_counter = 0;
        }
        if (bear_candidate) {
            bear_counter += 1;
            if (c > bear_high) bear_candidate = false;
        }
        if (bear_candidate && c < bear_low && !bear_confirmed && bear_counter <= NEO_RS_CONFIRM_LIMIT) {
            bear_confirmed = true;
        }

        if (bull_trigger) {
            bull_candidate = true; bull_low = l; bull_high = h;
            bull_confirmed = false; bull_counter = 0;
        }
        if (bull_candidate) {
            bull_counter += 1;
            if (c < bull_low) bull_candidate = false;
        }
        if (bull_candidate && c > bull_high && !bull_confirmed && bull_counter <= NEO_RS_CONFIRM_LIMIT) {
            bull_confirmed = true;
        }

        /* The stepped MA ratchet (:1179-1197). */
        if (isnan(stepped_ma)) {
            stepped_ma = ema_value;
            ma_last_update_bar = i;
        } else if (ma_direction == 1) {
            if (c < stepped_ma) {
                ma_direction = -1;
                stepped_ma = ema_value;
                ma_last_update_bar = i;
            } else if (i - ma_last_update_bar >= NEO_RS_MA_STEP_PERIOD) {
                stepped_ma = fmax(stepped_ma, ema_value);
                ma_last_update_bar = i;
            }
        } else if (c > stepped_ma) {
            ma_direction = 1;
            stepped_ma = ema_value;
            ma_last_update_bar = i;
        } else if (i - ma_last_update_bar >= NEO_RS_MA_STEP_PERIOD) {
            stepped_ma = fmin(stepped_ma, ema_value);
            ma_last_update_bar = i;
        }

        o[i] = stepped_ma;

        /* The previous-bar ring is written LAST (:1203-1213), so the min/max
         * scan above never sees the current bar. */
        if (prev_count < NEO_RS_PREV_SPAN) {
            lows[prev_count] = l;
            highs[prev_count] = h;
            prev_count += 1;
        } else {
            lows[prev_pos] = l;
            highs[prev_pos] = h;
            prev_pos += 1;
            if (prev_pos == NEO_RS_PREV_SPAN) prev_pos = 0;
        }
    }
}
