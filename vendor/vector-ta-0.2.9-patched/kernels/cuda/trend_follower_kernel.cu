#include <cmath>
#include <cstddef>

namespace {
constexpr int CHANNEL_WINDOW = 280;
constexpr int MATYPE_EMA = 0;
constexpr int MATYPE_SMA = 1;
constexpr int MATYPE_RMA = 2;
constexpr int MATYPE_WMA = 3;
constexpr int MATYPE_VWMA = 4;

struct EmaState {
    int period;
    int valid_count;
    bool has_value;
    double value;
    double alpha;
    double beta;

    __device__ inline void init(int len) {
        period = len;
        valid_count = 0;
        has_value = false;
        value = NAN;
        alpha = 2.0 / (static_cast<double>(len) + 1.0);
        beta = 1.0 - alpha;
    }

    __device__ inline double update(double input) {
        if (!has_value) {
            valid_count = 1;
            has_value = true;
            value = input;
            return value;
        }
        if (valid_count < period) {
            valid_count += 1;
            const double vc = static_cast<double>(valid_count);
            value = ((vc - 1.0) * value + input) / vc;
            return value;
        }
        value = beta * value + alpha * input;
        return value;
    }
};

struct RmaState {
    int period;
    int seed_count;
    bool has_value;
    double seed_sum;
    double value;

    __device__ inline void init(int len) {
        period = len;
        seed_count = 0;
        has_value = false;
        seed_sum = 0.0;
        value = NAN;
    }

    __device__ inline double update(double input) {
        if (has_value) {
            value = value + (input - value) / static_cast<double>(period);
            return value;
        }
        seed_sum += input;
        seed_count += 1;
        if (seed_count == period) {
            value = seed_sum / static_cast<double>(period);
            has_value = true;
            return value;
        }
        return NAN;
    }
};
}

extern "C" __global__ void trend_follower_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
    int len,
    const int* trend_periods,
    const int* ma_periods,
    const double* channel_rate_fractions,
    const int* linear_regression_periods,
    const int* ma_type_ids,
    int use_linear_regression,
    int rows,
    double* out_values,
    double* base_ma_history,
    double* ma_history
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int trend_period = trend_periods[row];
    const int ma_period = ma_periods[row];
    const double channel_rate_fraction = channel_rate_fractions[row];
    const int linear_regression_period = linear_regression_periods[row];
    const int ma_type = ma_type_ids[row];

    double* row_out = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_base_ma_history =
        base_ma_history + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma_history = ma_history + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
        row_base_ma_history[i] = NAN;
        row_ma_history[i] = NAN;
    }

    if (trend_period < 1 || ma_period <= 0 || !isfinite(channel_rate_fraction)
        || channel_rate_fraction <= 0.0 || (use_linear_regression != 0 && linear_regression_period < 2)
        || ma_type < MATYPE_EMA || ma_type > MATYPE_VWMA) {
        return;
    }

    EmaState ema_state;
    RmaState rma_state;
    ema_state.init(ma_period);
    rma_state.init(ma_period);

    int segment_start = 0;
    for (int i = 0; i < len; ++i) {
        const bool needs_volume = ma_type == MATYPE_VWMA;
        if (!(isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i]))
            || (needs_volume && !isfinite(volume[i]))) {
            segment_start = i + 1;
            ema_state.init(ma_period);
            rma_state.init(ma_period);
            continue;
        }

        const int bars_in_segment = i - segment_start + 1;
        double base_ma = NAN;

        if (ma_type == MATYPE_EMA) {
            base_ma = ema_state.update(close[i]);
        } else if (ma_type == MATYPE_RMA) {
            base_ma = rma_state.update(close[i]);
        } else if (ma_type == MATYPE_SMA) {
            if (bars_in_segment >= ma_period) {
                double sum = 0.0;
                for (int j = i - ma_period + 1; j <= i; ++j) {
                    sum += close[j];
                }
                base_ma = sum / static_cast<double>(ma_period);
            }
        } else if (ma_type == MATYPE_WMA) {
            if (bars_in_segment >= ma_period) {
                double weighted_sum = 0.0;
                double weight_sum = 0.0;
                int weight = 1;
                for (int j = i - ma_period + 1; j <= i; ++j, ++weight) {
                    weighted_sum += close[j] * static_cast<double>(weight);
                    weight_sum += static_cast<double>(weight);
                }
                base_ma = weighted_sum / weight_sum;
            }
        } else if (ma_type == MATYPE_VWMA) {
            if (bars_in_segment >= ma_period) {
                double sum_pv = 0.0;
                double sum_v = 0.0;
                for (int j = i - ma_period + 1; j <= i; ++j) {
                    sum_pv += close[j] * volume[j];
                    sum_v += volume[j];
                }
                if (sum_v != 0.0) {
                    base_ma = sum_pv / sum_v;
                }
            }
        }

        row_base_ma_history[i] = base_ma;

        double ma_value = base_ma;
        if (use_linear_regression != 0) {
            ma_value = NAN;
            if (isfinite(base_ma) && bars_in_segment >= linear_regression_period) {
                const int start = i - linear_regression_period + 1;
                double y_sum = 0.0;
                double xy_sum = 0.0;
                bool all_finite = true;
                int x = 1;
                for (int j = start; j <= i; ++j, ++x) {
                    const double y = row_base_ma_history[j];
                    if (!isfinite(y)) {
                        all_finite = false;
                        break;
                    }
                    y_sum += y;
                    xy_sum += y * static_cast<double>(x);
                }
                if (all_finite) {
                    const double pf = static_cast<double>(linear_regression_period);
                    const double x_sum = pf * (pf + 1.0) * 0.5;
                    const double x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
                    const double denom = pf * x2_sum - x_sum * x_sum;
                    if (denom != 0.0) {
                        const double b = (pf * xy_sum - x_sum * y_sum) / denom;
                        const double a = (y_sum - b * x_sum) / pf;
                        ma_value = a + b * pf;
                    }
                }
            }
        }

        row_ma_history[i] = ma_value;
        if (!isfinite(ma_value)) {
            continue;
        }

        const int channel_start =
            (i - CHANNEL_WINDOW + 1 > segment_start) ? (i - CHANNEL_WINDOW + 1) : segment_start;
        double channel_high = high[channel_start];
        double channel_low = low[channel_start];
        for (int j = channel_start + 1; j <= i; ++j) {
            channel_high = fmax(channel_high, high[j]);
            channel_low = fmin(channel_low, low[j]);
        }

        const int ma_start = (i - trend_period + 1 > segment_start) ? (i - trend_period + 1)
                                                                     : segment_start;
        bool have_ma = false;
        double hh = NAN;
        double ll = NAN;
        for (int j = ma_start; j <= i; ++j) {
            const double value = row_ma_history[j];
            if (!isfinite(value)) {
                continue;
            }
            if (!have_ma) {
                hh = value;
                ll = value;
                have_ma = true;
            } else {
                hh = fmax(hh, value);
                ll = fmin(ll, value);
            }
        }
        if (!have_ma) {
            continue;
        }

        const double chan = (channel_high - channel_low) * channel_rate_fraction;
        if (!isfinite(chan) || chan == 0.0) {
            continue;
        }

        const double diff = fabs(hh - ll);
        double trend = 0.0;
        if (diff > chan) {
            if (ma_value > ll + chan) {
                trend = 1.0;
            } else if (ma_value < hh - chan) {
                trend = -1.0;
            }
        }

        row_out[i] = trend * diff / chan;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/trend_follower.rs `trend_follower_compute_into`
 *   (:1063-1087), which BRANCHES on `data_is_clean` (:691) between
 *   `trend_follower_compute_clean_into` (:964-1035) and
 *   `trend_follower_compute_fallback_into` (:1037-1050). The single output is
 *   `values` (:1116), so "value" is unambiguous.
 *
 * WHY A SECOND ENTRY POINT: `trend_follower_batch_f64` (:79) takes 15
 *   parameters and emits eight series. The lane launches
 *   (high, low, close, volume, n, periods, n_combos, first_valid, out).
 *
 * INPUT: (high, low, close, volume) -- F64InputKind::Hlcv. Volume is bound and
 *   read only when `matype == Vwma`; the CPU default is `ema` (:143), so it is
 *   unread here -- but it IS part of the length contract, and `needs_volume`
 *   is false for exactly that reason.
 *
 * BOTH BRANCHES ARE IMPLEMENTED, AND THEY ARE NOT THE SAME ARITHMETIC.
 *   This is the whole difficulty of this indicator and it cannot be skipped:
 *     * the CLEAN branch builds the base EMA with `ema_with_kernel`, whose
 *       warm-up is a RUNNING MEAN `((vc-1)*mean + x)/vc` (ema.rs:482) and
 *       whose steady step is `beta.mul_add(prev, alpha*x)` (ema.rs:493) -- ONE
 *       fused rounding -- and then runs `linreg_with_kernel`, whose y_sum and
 *       xy_sum are maintained by a SLIDING update (linreg.rs:358-370) that
 *       carries accumulated drift forward across the whole series.
 *     * the FALLBACK branch runs `TrendFollowerStream`, whose EmaState has the
 *       same warm-up but whose LinRegStream recomputes a FRESH dot product
 *       over the ring at every bar (linreg.rs:900-914) -- no carried drift,
 *       and a different closed form (`a + b*period` rather than
 *       `xy_sum*xy_coeff + y_sum*y_coeff`).
 *   Collapsing the two would silently answer with the wrong branch on any
 *   frame carrying a hole, which is most real frames.
 *
 * FIRST-VALID IGNORED, AND DERIVED HERE: `first_valid_bar` (:678) scans high,
 *   low AND close together with `is_finite` -- stricter than `!is_nan`, so an
 *   INFINITE bar is skipped by the CPU and would be accepted by
 *   `AllInputsNonNan`. And in the fallback branch there is no global index at
 *   all: the stream restarts at every hole. Derived here for both reasons.
 *
 * PERIOD-INVARIANT: the CPU reads NAMED parameters -- `trend_period`,
 *   `ma_period`, `channel_rate_percent`, `use_linear_regression`,
 *   `linear_regression_period`, `matype` (:147-169) -- and never `period`.
 *   All are pinned at the CPU defaults (20 / 20 / 1.0 / true / 5 / "ema"), so
 *   every row of a sweep is byte-identical.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. Four monotone deques, an EMA
 *   recurrence and a linear-regression accumulator all carry across bars.
 *
 * ARITHMETIC taken verbatim:
 *   * `channel_rate_fraction` is `channel_rate_percent * 0.01` formed ONCE
 *     (:669), not a division by 100 per bar.
 *   * `chan = (channel_high - channel_low) * fraction` (:1017) -- difference
 *     first, then one product.
 *   * `diff = (hh - ll).abs()`; the trend is a three-way branch on
 *     `ma > ll + chan` / `ma < hh - chan` (:1023-1031), and the result is
 *     `trend * diff / chan` (:1033) -- multiply THEN divide, which is not the
 *     same rounding as `trend * (diff / chan)`.
 *   * the monotone deques pop the back on `<=` for max and `>=` for min
 *     (:919, :941), which decides which of two equal values survives.
 *
 * EPSILON: there is none. The CPU guards are `chan == 0.0` and `is_finite`.
 *   No tolerance is imported.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* trend_follower.rs:39 and :147-169 */
#define NEO_TF_CHANNEL_WINDOW 280
#define NEO_TF_TREND_PERIOD   20
#define NEO_TF_MA_PERIOD      20
#define NEO_TF_LR_PERIOD      5
#define NEO_TF_CHANNEL_RATE_PERCENT 1.0

/* Ring capacity for a [head, count) monotone deque. The CPU's MonoQueue uses
 * `window + 1` with an explicit count (:881); two spare slots here so the
 * transient occupancy inside push can never alias. */
#define NEO_TF_CHAN_CAP  (NEO_TF_CHANNEL_WINDOW + 2)
#define NEO_TF_TREND_CAP (NEO_TF_TREND_PERIOD + 2)

/* MonoQueue (:869-963) as a fixed ring with an explicit count. */
typedef struct {
    int    idx[NEO_TF_CHAN_CAP];
    double val[NEO_TF_CHAN_CAP];
    int    head;
    int    count;
    int    cap;
} NeoTfQueue;

__device__ __forceinline__ void neo_tf_q_init(NeoTfQueue* q, int cap)
{
    q->head = 0; q->count = 0; q->cap = cap;
}

__device__ __forceinline__ void neo_tf_q_evict(NeoTfQueue* q, int idx, int window)
{
    const int min_idx = (idx + 1 > window) ? (idx + 1 - window) : 0;
    while (q->count > 0 && q->idx[q->head] < min_idx) {
        q->head += 1; if (q->head == q->cap) q->head = 0;
        q->count -= 1;
    }
}

__device__ __forceinline__ void neo_tf_q_push_max(NeoTfQueue* q, int idx, double value)
{
    while (q->count > 0) {
        int back = q->head + q->count - 1; if (back >= q->cap) back -= q->cap;
        if (q->val[back] <= value) q->count -= 1; else break;
    }
    int slot = q->head + q->count; if (slot >= q->cap) slot -= q->cap;
    q->idx[slot] = idx; q->val[slot] = value; q->count += 1;
}

__device__ __forceinline__ void neo_tf_q_push_min(NeoTfQueue* q, int idx, double value)
{
    while (q->count > 0) {
        int back = q->head + q->count - 1; if (back >= q->cap) back -= q->cap;
        if (q->val[back] >= value) q->count -= 1; else break;
    }
    int slot = q->head + q->count; if (slot >= q->cap) slot -= q->cap;
    q->idx[slot] = idx; q->val[slot] = value; q->count += 1;
}

extern "C" __global__
void trend_follower_neo_batch_f64(const double* __restrict__ high,
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
    (void)volume;      /* matype is `ema`; needs_volume is false -- see header */
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* derived here -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    trend_period = NEO_TF_TREND_PERIOD;
    const int    ma_period    = NEO_TF_MA_PERIOD;
    const int    lr_period    = NEO_TF_LR_PERIOD;
    const double fraction     = NEO_TF_CHANNEL_RATE_PERCENT * 0.01;

    /* resolve_params (:633) refuses ma_period > data_len and, with
     * use_linear_regression, linear_regression_period > data_len. */
    if (ma_period > n || lr_period > n || lr_period < 2) return;

    /* first_valid_bar (:678) -- high, low, close, `is_finite`. */
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i])) { first = i; break; }
    }
    if (first < 0) return;                 /* AllValuesNaN */

    /* data_is_clean (:691) -- from `first` to the end. */
    bool clean = true;
    for (int i = first; i < n; ++i) {
        if (!(isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i]))) { clean = false; break; }
    }

    NeoTfQueue high_max, low_min, ma_max, ma_min;
    neo_tf_q_init(&high_max, NEO_TF_CHAN_CAP);
    neo_tf_q_init(&low_min,  NEO_TF_CHAN_CAP);
    neo_tf_q_init(&ma_max,   NEO_TF_TREND_CAP);
    neo_tf_q_init(&ma_min,   NEO_TF_TREND_CAP);

    /* Shared linreg constants -- linreg_scalar (:335-342) forms them from the
     * integer sums, so they are formed the same way here. */
    const double lp   = (double)lr_period;
    const double x_sum  = (double)((lr_period * (lr_period + 1)) / 2);
    const double x2_sum = (double)((lr_period * (lr_period + 1) * (2 * lr_period + 1)) / 6);

    if (clean) {
        /* ---------------------------------------------------------------
         * CLEAN BRANCH: ema_with_kernel -> linreg_with_kernel -> deques.
         * ------------------------------------------------------------- */
        /* ema_prepare (:305) scans CLOSE ALONE with `!is_nan` -- a different
         * index from `first`, which also required high and low. */
        int ema_first = -1;
        for (int i = 0; i < n; ++i) { if (!isnan(close[i])) { ema_first = i; break; } }
        if (ema_first < 0) return;
        if (n - ema_first < ma_period) return;   /* NotEnoughValidData */
        /* linreg_prepare (:224) scans base_ma, whose first non-NaN is
         * ema_first, and needs `len - first >= period`. */
        if (n - ema_first < lr_period) return;

        const double alpha = 2.0 / ((double)ma_period + 1.0);
        const double beta  = 1.0 - alpha;

        const double denom_inv   = 1.0 / (lp * x2_sum - x_sum * x_sum);
        const double inv_period  = 1.0 / lp;
        const double b_scale     = lp - x_sum * inv_period;
        const double xy_coeff    = lp * denom_inv * b_scale;
        const double y_coeff     = inv_period - x_sum * denom_inv * b_scale;

        /* base_ma ring for the linreg sliding subtraction (`data[old_idx]`). */
        double ma_ring[NEO_TF_LR_PERIOD];
        for (int k = 0; k < lr_period; ++k) ma_ring[k] = 0.0;

        double ema_mean = 0.0;
        int    ema_valid_count = 0;
        const int warmup_end = (ema_first + ma_period < n) ? (ema_first + ma_period) : n;

        double lr_y_sum = 0.0, lr_xy_sum = 0.0;
        const int lr_emit_start = ema_first + lr_period - 1;

        for (int i = 0; i < n; ++i) {
            /* ---- base EMA (ema.rs:461-499) ---- */
            double base_ma = NEO_F64_NAN;
            if (i == ema_first) {
                ema_mean = close[i];
                ema_valid_count = 1;
                base_ma = ema_mean;
            } else if (i > ema_first && i < warmup_end) {
                const double x = close[i];
                if (isfinite(x)) {
                    ema_valid_count += 1;
                    const double vc = (double)ema_valid_count;
                    ema_mean = ((vc - 1.0) * ema_mean + x) / vc;
                }
                base_ma = ema_mean;
            } else if (i >= warmup_end && i > ema_first) {
                const double x = close[i];
                if (isfinite(x)) {
                    ema_mean = fma(beta, ema_mean, alpha * x);
                }
                base_ma = ema_mean;
            }

            /* ---- linreg over base_ma (linreg.rs:334-372) ---- */
            double trend_ma = NEO_F64_NAN;
            if (i >= ema_first) {
                const int slot = (i - ema_first) % lr_period;
                if (i < lr_emit_start) {
                    /* the init_slice pass: k = 1 .. period-1 */
                    const double kk = (double)(i - ema_first + 1);
                    lr_y_sum  += base_ma;
                    lr_xy_sum += kk * base_ma;
                    ma_ring[slot] = base_ma;
                } else {
                    lr_y_sum  += base_ma;
                    lr_xy_sum += base_ma * lp;
                    trend_ma = lr_xy_sum * xy_coeff + lr_y_sum * y_coeff;
                    lr_xy_sum -= lr_y_sum;
                    /* data[old_idx] is base_ma[i - (period - 1)] */
                    const int old_slot = (i - ema_first - (lr_period - 1)) % lr_period;
                    lr_y_sum -= ma_ring[old_slot];
                    ma_ring[slot] = base_ma;
                }
            }

            if (i < first) continue;

            neo_tf_q_evict(&high_max, i, NEO_TF_CHANNEL_WINDOW);
            neo_tf_q_evict(&low_min,  i, NEO_TF_CHANNEL_WINDOW);
            neo_tf_q_evict(&ma_max,   i, trend_period);
            neo_tf_q_evict(&ma_min,   i, trend_period);

            neo_tf_q_push_max(&high_max, i, high[i]);
            neo_tf_q_push_min(&low_min,  i, low[i]);

            const double ma = trend_ma;
            if (isfinite(ma)) {
                neo_tf_q_push_max(&ma_max, i, ma);
                neo_tf_q_push_min(&ma_min, i, ma);
            }

            if (ma_max.count == 0 || ma_min.count == 0) continue;
            if (high_max.count == 0 || low_min.count == 0) continue;

            const double hh = ma_max.val[ma_max.head];
            const double ll = ma_min.val[ma_min.head];
            const double channel_high = high_max.val[high_max.head];
            const double channel_low  = low_min.val[low_min.head];

            const double chan = (channel_high - channel_low) * fraction;
            if (!isfinite(ma) || !isfinite(chan) || chan == 0.0) { o[i] = NEO_F64_NAN; continue; }

            const double diff = fabs(hh - ll);
            double trend;
            if (diff > chan) {
                if (ma > ll + chan)      trend =  1.0;
                else if (ma < hh - chan) trend = -1.0;
                else                     trend =  0.0;
            } else {
                trend = 0.0;
            }
            o[i] = trend * diff / chan;
        }
        return;
    }

    /* -------------------------------------------------------------------
     * FALLBACK BRANCH: TrendFollowerStream::update_reset_on_nan (:1282).
     * ----------------------------------------------------------------- */
    {
        const double alpha = 2.0 / ((double)ma_period + 1.0);
        const double beta  = 1.0 - 2.0 / ((double)ma_period + 1.0);  /* EmaState:249 */
        const double bd    = 1.0 / (lp * x2_sum - x_sum * x_sum);

        double lr_buf[NEO_TF_LR_PERIOD];
        for (int k = 0; k < lr_period; ++k) lr_buf[k] = NEO_F64_NAN;
        int    lr_head = 0;
        bool   lr_filled = false;

        double ema_value = 0.0;
        bool   ema_has = false;
        int    ema_valid_count = 0;
        int    sidx = 0;

        for (int i = 0; i < n; ++i) {
            const double h = high[i], l = low[i], c = close[i];
            if (!(isfinite(h) && isfinite(l) && isfinite(c))) {
                /* reset (:1196) */
                neo_tf_q_init(&high_max, NEO_TF_CHAN_CAP);
                neo_tf_q_init(&low_min,  NEO_TF_CHAN_CAP);
                neo_tf_q_init(&ma_max,   NEO_TF_TREND_CAP);
                neo_tf_q_init(&ma_min,   NEO_TF_TREND_CAP);
                for (int k = 0; k < lr_period; ++k) lr_buf[k] = NEO_F64_NAN;
                lr_head = 0; lr_filled = false;
                ema_value = 0.0; ema_has = false; ema_valid_count = 0;
                sidx = 0;
                o[i] = NEO_F64_NAN;
                continue;
            }

            const int idx = sidx;
            neo_tf_q_evict(&high_max, idx, NEO_TF_CHANNEL_WINDOW);
            neo_tf_q_evict(&low_min,  idx, NEO_TF_CHANNEL_WINDOW);
            neo_tf_q_evict(&ma_max,   idx, trend_period);
            neo_tf_q_evict(&ma_min,   idx, trend_period);

            neo_tf_q_push_max(&high_max, idx, h);
            neo_tf_q_push_min(&low_min,  idx, l);

            /* EmaState::update (:255) */
            double base_ma;
            bool   base_ok = true;
            if (!ema_has) {
                ema_valid_count = 1; ema_value = c; ema_has = true; base_ma = ema_value;
            } else if (ema_valid_count < ma_period) {
                ema_valid_count += 1;
                const double vc = (double)ema_valid_count;
                ema_value = ((vc - 1.0) * ema_value + c) / vc;
                base_ma = ema_value;
            } else {
                ema_value = fma(beta, ema_value, alpha * c);
                base_ma = ema_value;
            }

            /* LinRegStream::update (:887) */
            double ma_value = 0.0;
            bool   ma_ok = false;
            if (base_ok) {
                lr_buf[lr_head] = base_ma;
                lr_head = (lr_head + 1) % lr_period;
                if (!lr_filled && lr_head == 0) lr_filled = true;
                if (lr_filled) {
                    /* dot_ring (:900) -- a FRESH dot product, oldest first. */
                    double y_sum = 0.0, xy_sum = 0.0;
                    for (int k = 0; k < lr_period; ++k) {
                        const double y = lr_buf[(lr_head + k) % lr_period];
                        y_sum  += y;
                        xy_sum += y * (double)(k + 1);
                    }
                    const double b = (lp * xy_sum - x_sum * y_sum) * bd;
                    const double a = (y_sum - b * x_sum) / lp;
                    ma_value = a + b * lp;
                    ma_ok = true;
                }
            }

            sidx = idx + 1;

            if (!ma_ok) { o[i] = NEO_F64_NAN; continue; }
            if (!isfinite(ma_value)) { o[i] = NEO_F64_NAN; continue; }

            neo_tf_q_push_max(&ma_max, idx, ma_value);
            neo_tf_q_push_min(&ma_min, idx, ma_value);

            if (ma_max.count == 0 || ma_min.count == 0) { o[i] = NEO_F64_NAN; continue; }
            if (high_max.count == 0 || low_min.count == 0) { o[i] = NEO_F64_NAN; continue; }

            const double hh = ma_max.val[ma_max.head];
            const double ll = ma_min.val[ma_min.head];
            const double channel_high = high_max.val[high_max.head];
            const double channel_low  = low_min.val[low_min.head];

            const double chan = (channel_high - channel_low) * fraction;
            if (!isfinite(chan) || chan == 0.0) { o[i] = NEO_F64_NAN; continue; }

            const double diff = fabs(hh - ll);
            double trend;
            if (diff > chan) {
                if (ma_value > ll + chan)      trend =  1.0;
                else if (ma_value < hh - chan) trend = -1.0;
                else                           trend =  0.0;
            } else {
                trend = 0.0;
            }
            o[i] = trend * diff / chan;
        }
    }
}
