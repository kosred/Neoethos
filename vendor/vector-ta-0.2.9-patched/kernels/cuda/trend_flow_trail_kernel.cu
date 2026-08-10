#include <cmath>
#include <cstddef>

namespace {

constexpr int MFI_HMA_LENGTH = 7;
constexpr int MFI_HMA_HALF = 3;
constexpr int MFI_HMA_SQRT = 2;

struct LinWmaState {
    double* buffer;
    int period;
    int head;
    int count;
    bool filled;
    double sum;
    double wsum;
    double inv_norm;
    int nan_count;
    bool dirty;

    __device__ void init(double* buffer_ptr, int period_value) {
        buffer = buffer_ptr;
        period = period_value;
        inv_norm = 2.0 / (static_cast<double>(period) * static_cast<double>(period + 1));
        reset();
    }

    __device__ void reset() {
        for (int i = 0; i < period; ++i) {
            buffer[i] = NAN;
        }
        head = 0;
        count = 0;
        filled = false;
        sum = 0.0;
        wsum = 0.0;
        nan_count = 0;
        dirty = false;
    }

    __device__ void rebuild() {
        sum = 0.0;
        wsum = 0.0;
        nan_count = 0;
        int idx = head;
        for (int i = 0; i < period; ++i) {
            const double v = buffer[idx];
            if (isnan(v)) {
                nan_count += 1;
            } else {
                sum += v;
                wsum += static_cast<double>(i + 1) * v;
            }
            idx = idx + 1 == period ? 0 : idx + 1;
        }
        dirty = nan_count != 0;
    }

    __device__ bool update(double value, double* out) {
        const double n = static_cast<double>(period);

        if (!filled) {
            buffer[head] = value;
            head = head + 1 == period ? 0 : head + 1;
            count += 1;
            if (isnan(value)) {
                nan_count += 1;
                dirty = true;
            } else {
                sum += value;
                wsum += static_cast<double>(count) * value;
            }
            if (count == period) {
                filled = true;
                *out = nan_count > 0 ? NAN : wsum * inv_norm;
                return true;
            }
            return false;
        }

        const double old = buffer[head];
        buffer[head] = value;
        head = head + 1 == period ? 0 : head + 1;
        if (isnan(old)) {
            nan_count = nan_count > 0 ? nan_count - 1 : 0;
        }
        if (isnan(value)) {
            nan_count += 1;
        }
        if (nan_count > 0) {
            dirty = true;
            *out = NAN;
            return true;
        }
        if (dirty) {
            rebuild();
            dirty = false;
            *out = wsum * inv_norm;
            return true;
        }

        const double prev_sum = sum;
        sum = prev_sum + value - old;
        wsum = n * value + wsum - prev_sum;
        *out = wsum * inv_norm;
        return true;
    }
};

struct HmaState {
    bool direct;
    LinWmaState wma_half;
    LinWmaState wma_full;
    LinWmaState wma_sqrt;

    __device__ void init(
        int period,
        double* full_ptr,
        double* half_ptr,
        double* sqrt_ptr
    ) {
        direct = period == 1;
        if (!direct) {
            wma_half.init(half_ptr, period / 2);
            wma_full.init(full_ptr, period);
            wma_sqrt.init(sqrt_ptr, static_cast<int>(floor(sqrt(static_cast<double>(period)))));
        }
    }

    __device__ void reset() {
        if (!direct) {
            wma_half.reset();
            wma_full.reset();
            wma_sqrt.reset();
        }
    }

    __device__ bool update(double value, double* out) {
        if (direct) {
            *out = value;
            return true;
        }
        double full = NAN;
        double half = NAN;
        const bool full_ready = wma_full.update(value, &full);
        const bool half_ready = wma_half.update(value, &half);
        if (!(full_ready && half_ready)) {
            return false;
        }
        return wma_sqrt.update(2.0 * half - full, out);
    }
};

struct EmaState {
    int period;
    double alpha;
    double beta;
    int count;
    double mean;
    bool filled;

    __device__ void init(int period_value) {
        period = period_value;
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
        filled = false;
    }

    __device__ bool update(double value, double* out) {
        if (!isfinite(value)) {
            if (filled) {
                *out = mean;
                return true;
            }
            return false;
        }

        count += 1;
        if (count == 1) {
            mean = value;
        } else if (count <= period) {
            mean += (value - mean) / static_cast<double>(count);
        } else {
            mean = beta * mean + alpha * value;
        }

        if (!filled && count >= period) {
            filled = true;
        }
        if (filled) {
            *out = mean;
            return true;
        }
        return false;
    }
};

struct MoneyFlowRawState {
    int len;
    double* pos;
    double* neg;
    int head;
    int count;
    double pos_sum;
    double neg_sum;
    double prev_src;
    bool has_prev_src;

    __device__ void init(double* pos_ptr, double* neg_ptr, int len_value) {
        len = len_value;
        pos = pos_ptr;
        neg = neg_ptr;
        reset();
    }

    __device__ void reset() {
        for (int i = 0; i < len; ++i) {
            pos[i] = 0.0;
            neg[i] = 0.0;
        }
        head = 0;
        count = 0;
        pos_sum = 0.0;
        neg_sum = 0.0;
        prev_src = NAN;
        has_prev_src = false;
    }

    __device__ bool update(double src, double volume, double* out) {
        const double delta = has_prev_src ? (src - prev_src) : 0.0;
        prev_src = src;
        has_prev_src = true;

        const double pos_flow = delta > 0.0 ? volume * src : 0.0;
        const double neg_flow = delta < 0.0 ? volume * src : 0.0;

        if (count == len) {
            pos_sum -= pos[head];
            neg_sum -= neg[head];
        } else {
            count += 1;
        }

        pos[head] = pos_flow;
        neg[head] = neg_flow;
        pos_sum += pos_flow;
        neg_sum += neg_flow;
        head = (head + 1) % len;

        if (count < len) {
            return false;
        }

        const double ratio = pos_sum / neg_sum;
        *out = 100.0 - (100.0 / (1.0 + ratio));
        return true;
    }
};

__device__ inline bool crossover(bool has_prev_left, double prev_left, double left, double right) {
    return has_prev_left && isfinite(prev_left) && isfinite(left) && prev_left <= right &&
        left > right;
}

__device__ inline bool crossunder(bool has_prev_left, double prev_left, double left, double right) {
    return has_prev_left && isfinite(prev_left) && isfinite(left) && prev_left >= right &&
        left < right;
}

__device__ inline bool cross_pair(
    bool has_prev_left,
    double prev_left,
    double left,
    bool has_prev_right,
    double prev_right,
    double right
) {
    return has_prev_left && has_prev_right && isfinite(prev_left) && isfinite(prev_right) &&
        isfinite(left) && isfinite(right) && prev_left <= prev_right && left > right;
}

__device__ inline bool crossunder_pair(
    bool has_prev_left,
    double prev_left,
    double left,
    bool has_prev_right,
    double prev_right,
    double right
) {
    return has_prev_left && has_prev_right && isfinite(prev_left) && isfinite(prev_right) &&
        isfinite(left) && isfinite(right) && prev_left >= prev_right && left < right;
}

}

extern "C" __global__ void trend_flow_trail_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ alpha_lengths,
    const double* __restrict__ alpha_multipliers,
    const int* __restrict__ mfi_lengths,
    int rows,
    int alpha_full_cap,
    int alpha_half_cap,
    int alpha_sqrt_cap,
    int mfi_cap,
    double* __restrict__ alpha_full_scratch,
    double* __restrict__ alpha_half_scratch,
    double* __restrict__ alpha_sqrt_scratch,
    double* __restrict__ mfi_pos_scratch,
    double* __restrict__ mfi_neg_scratch,
    double* __restrict__ mfi_full_scratch,
    double* __restrict__ mfi_half_scratch,
    double* __restrict__ mfi_sqrt_scratch,
    double* __restrict__ out_alpha_trail,
    double* __restrict__ out_alpha_trail_bullish,
    double* __restrict__ out_alpha_trail_bearish,
    double* __restrict__ out_alpha_dir,
    double* __restrict__ out_mfi,
    double* __restrict__ out_tp_upper,
    double* __restrict__ out_tp_lower,
    double* __restrict__ out_alpha_trail_bullish_switch,
    double* __restrict__ out_alpha_trail_bearish_switch,
    double* __restrict__ out_mfi_overbought,
    double* __restrict__ out_mfi_oversold,
    double* __restrict__ out_mfi_cross_up_mid,
    double* __restrict__ out_mfi_cross_down_mid,
    double* __restrict__ out_price_cross_alpha_trail_up,
    double* __restrict__ out_price_cross_alpha_trail_down,
    double* __restrict__ out_mfi_above_90,
    double* __restrict__ out_mfi_below_10
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_alpha_trail =
        out_alpha_trail + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_alpha_trail_bullish =
        out_alpha_trail_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_alpha_trail_bearish =
        out_alpha_trail_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_alpha_dir = out_alpha_dir + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi = out_mfi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_tp_upper = out_tp_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_tp_lower = out_tp_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_alpha_trail_bullish_switch =
        out_alpha_trail_bullish_switch + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_alpha_trail_bearish_switch =
        out_alpha_trail_bearish_switch + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi_overbought =
        out_mfi_overbought + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi_oversold =
        out_mfi_oversold + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi_cross_up_mid =
        out_mfi_cross_up_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi_cross_down_mid =
        out_mfi_cross_down_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_price_cross_alpha_trail_up =
        out_price_cross_alpha_trail_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_price_cross_alpha_trail_down =
        out_price_cross_alpha_trail_down + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi_above_90 =
        out_mfi_above_90 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mfi_below_10 =
        out_mfi_below_10 + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_alpha_trail[i] = NAN;
        row_alpha_trail_bullish[i] = NAN;
        row_alpha_trail_bearish[i] = NAN;
        row_alpha_dir[i] = NAN;
        row_mfi[i] = NAN;
        row_tp_upper[i] = NAN;
        row_tp_lower[i] = NAN;
        row_alpha_trail_bullish_switch[i] = NAN;
        row_alpha_trail_bearish_switch[i] = NAN;
        row_mfi_overbought[i] = NAN;
        row_mfi_oversold[i] = NAN;
        row_mfi_cross_up_mid[i] = NAN;
        row_mfi_cross_down_mid[i] = NAN;
        row_price_cross_alpha_trail_up[i] = NAN;
        row_price_cross_alpha_trail_down[i] = NAN;
        row_mfi_above_90[i] = NAN;
        row_mfi_below_10[i] = NAN;
    }

    const int alpha_length = alpha_lengths[row];
    const double alpha_multiplier = alpha_multipliers[row];
    const int mfi_length = mfi_lengths[row];
    if (alpha_length <= 0 || !isfinite(alpha_multiplier) || alpha_multiplier < 0.1 ||
        mfi_length <= 0) {
        return;
    }

    HmaState basis_state;
    basis_state.init(
        alpha_length,
        alpha_full_scratch + static_cast<size_t>(row) * static_cast<size_t>(alpha_full_cap),
        alpha_half_scratch + static_cast<size_t>(row) * static_cast<size_t>(alpha_half_cap),
        alpha_sqrt_scratch + static_cast<size_t>(row) * static_cast<size_t>(alpha_sqrt_cap)
    );
    EmaState spread_state;
    spread_state.init(alpha_length > 0 ? alpha_length : 1);
    MoneyFlowRawState money_flow_state;
    money_flow_state.init(
        mfi_pos_scratch + static_cast<size_t>(row) * static_cast<size_t>(mfi_cap),
        mfi_neg_scratch + static_cast<size_t>(row) * static_cast<size_t>(mfi_cap),
        mfi_length
    );
    HmaState mfi_state;
    mfi_state.init(
        MFI_HMA_LENGTH,
        mfi_full_scratch + static_cast<size_t>(row) * static_cast<size_t>(MFI_HMA_LENGTH),
        mfi_half_scratch + static_cast<size_t>(row) * static_cast<size_t>(MFI_HMA_HALF),
        mfi_sqrt_scratch + static_cast<size_t>(row) * static_cast<size_t>(MFI_HMA_SQRT)
    );

    bool has_prev_upper = false;
    double prev_upper = NAN;
    bool has_prev_lower = false;
    double prev_lower = NAN;
    bool has_prev_trail = false;
    double prev_trail = NAN;
    bool has_prev_alpha_dir = false;
    double prev_alpha_dir = NAN;
    bool has_prev_close = false;
    double prev_close = NAN;
    bool has_prev_mfi = false;
    double prev_mfi = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!(isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) && isfinite(v))) {
            basis_state.reset();
            spread_state.reset();
            money_flow_state.reset();
            mfi_state.reset();
            has_prev_upper = false;
            prev_upper = NAN;
            has_prev_lower = false;
            prev_lower = NAN;
            has_prev_trail = false;
            prev_trail = NAN;
            has_prev_alpha_dir = false;
            prev_alpha_dir = NAN;
            has_prev_close = false;
            prev_close = NAN;
            has_prev_mfi = false;
            prev_mfi = NAN;
            continue;
        }

        const bool prev_close_ready = has_prev_close;
        const double prev_close_value = prev_close;
        const bool prev_alpha_dir_ready = has_prev_alpha_dir;
        const double prev_alpha_dir_value = prev_alpha_dir;
        const bool prev_trail_ready = has_prev_trail;
        const double prev_trail_value = prev_trail;
        const bool prev_mfi_ready = has_prev_mfi;
        const double prev_mfi_value = prev_mfi;

        prev_close = c;
        has_prev_close = true;

        double basis = NAN;
        double spread = NAN;
        double raw_mfi = NAN;
        double mfi = NAN;
        const bool basis_ready = basis_state.update(c, &basis);
        const bool spread_ready = spread_state.update(fabs(h - l), &spread);
        const bool raw_ready = money_flow_state.update((h + l + c) / 3.0, v, &raw_mfi);
        const bool mfi_ready = raw_ready && mfi_state.update(raw_mfi, &mfi);

        if (!(basis_ready && spread_ready && mfi_ready && isfinite(mfi))) {
            continue;
        }

        const double spread_scaled = spread * alpha_multiplier;
        double upper = basis + spread_scaled;
        double lower = basis - spread_scaled;
        const double prev_upper_value = has_prev_upper ? prev_upper : 0.0;
        const double prev_lower_value = has_prev_lower ? prev_lower : 0.0;
        const double prev_close_for_band = prev_close_ready ? prev_close_value : 0.0;

        lower = (lower > prev_lower_value || prev_close_for_band < prev_lower_value)
            ? lower
            : prev_lower_value;
        upper = (upper < prev_upper_value || prev_close_for_band > prev_upper_value)
            ? upper
            : prev_upper_value;

        double alpha_dir = 1.0;
        if (!prev_trail_ready) {
            alpha_dir = 1.0;
        } else if (prev_alpha_dir_ready && prev_alpha_dir_value > 0.0) {
            alpha_dir = c > upper ? -1.0 : 1.0;
        } else if (c < lower) {
            alpha_dir = 1.0;
        } else {
            alpha_dir = -1.0;
        }

        const double alpha_trail = alpha_dir < 0.0 ? lower : upper;
        prev_upper = upper;
        has_prev_upper = true;
        prev_lower = lower;
        has_prev_lower = true;
        prev_trail = alpha_trail;
        has_prev_trail = true;
        prev_alpha_dir = alpha_dir;
        has_prev_alpha_dir = true;
        prev_mfi = mfi;
        has_prev_mfi = true;

        row_alpha_trail[i] = alpha_trail;
        row_alpha_trail_bullish[i] = alpha_dir < 0.0 ? alpha_trail : NAN;
        row_alpha_trail_bearish[i] = alpha_dir > 0.0 ? alpha_trail : NAN;
        row_alpha_dir[i] = alpha_dir;
        row_mfi[i] = mfi;
        row_tp_upper[i] =
            crossover(prev_mfi_ready, prev_mfi_value, mfi, 80.0) && alpha_dir == -1.0
            ? 1.0
            : NAN;
        row_tp_lower[i] =
            crossunder(prev_mfi_ready, prev_mfi_value, mfi, 20.0) && alpha_dir == 1.0
            ? 1.0
            : NAN;
        row_alpha_trail_bullish_switch[i] =
            crossover(prev_alpha_dir_ready, prev_alpha_dir_value, alpha_dir, 0.0) ? 1.0 : NAN;
        row_alpha_trail_bearish_switch[i] =
            crossunder(prev_alpha_dir_ready, prev_alpha_dir_value, alpha_dir, 0.0) ? 1.0 : NAN;
        row_mfi_overbought[i] =
            crossover(prev_mfi_ready, prev_mfi_value, mfi, 80.0) ? 1.0 : NAN;
        row_mfi_oversold[i] =
            crossunder(prev_mfi_ready, prev_mfi_value, mfi, 20.0) ? 1.0 : NAN;
        row_mfi_cross_up_mid[i] =
            crossover(prev_mfi_ready, prev_mfi_value, mfi, 50.0) ? 1.0 : NAN;
        row_mfi_cross_down_mid[i] =
            crossunder(prev_mfi_ready, prev_mfi_value, mfi, 50.0) ? 1.0 : NAN;
        row_price_cross_alpha_trail_up[i] = cross_pair(
            prev_close_ready,
            prev_close_value,
            c,
            prev_trail_ready,
            prev_trail_value,
            alpha_trail
        )
            ? 1.0
            : NAN;
        row_price_cross_alpha_trail_down[i] = crossunder_pair(
            prev_close_ready,
            prev_close_value,
            c,
            prev_trail_ready,
            prev_trail_value,
            alpha_trail
        )
            ? 1.0
            : NAN;
        row_mfi_above_90[i] =
            crossover(prev_mfi_ready, prev_mfi_value, mfi, 90.0) ? 1.0 : NAN;
        row_mfi_below_10[i] =
            crossunder(prev_mfi_ready, prev_mfi_value, mfi, 10.0) ? 1.0 : NAN;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — trend_flow_trail
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/trend_flow_trail.rs:497
 *   TrendFlowTrailState::update, driving HmaLikeStream::update (:384) ->
 *   HmaStream::update (moving_averages/hma.rs:844) -> LinWma::update (:750),
 *   EmaStream::update (moving_averages/ema.rs:592) and
 *   MoneyFlowRawState::update (:423).
 *
 * Column: output_id "value" resolves to out.alpha_trail —
 *   cpu_batch.rs:4487 accepts "alpha_trail"/"value". The other sixteen
 *   columns are separate output ids; the state they need (prev_mfi,
 *   prev_alpha_dir, prev_close) is still CARRIED here because the alpha_trail
 *   recurrence reads prev_upper, prev_lower, prev_trail and prev_close.
 *
 * PERIOD-INVARIANT: compute_trend_flow_trail_batch reads alpha_length (33),
 *   alpha_multiplier (3.3) and mfi_length (14) and NEVER period
 *   (cpu_batch.rs:4465-4468).
 *
 * FIRST-VALID IGNORED: update RESETS every stream on a bar where ANY of open,
 *   high, low, close or volume is non-finite (:506-516) and compute_row walks
 *   every bar from index 0, writing NaN for a None.
 *
 * Input: (open, high, low, close, volume) — extract_ohlcv_full_input
 *   (cpu_batch.rs:4457) — F64InputKind::Ohlcv5. OPEN is read ONLY by the
 *   validity gate, and that is precisely why it cannot be dropped: a bar with
 *   a non-finite open and finite h/l/c/v RESETS the whole cascade on the CPU,
 *   and a kernel that never saw open would carry straight through it.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Four stateful stages — a
 *   Hull MA on close, an EMA on the bar range, a 14-deep money-flow ring, and
 *   a second Hull MA on the money-flow ratio — feed a band pair that clamps
 *   against ITS OWN previous values and a latched trail.
 *
 * ARITHMETIC taken verbatim:
 *   * LinWma keeps sum and wsum incrementally as
 *     sum = prev_sum + value - old and wsum = n.mul_add(value, wsum - prev_sum)
 *     (hma.rs:803-805) — ONE fma over a pre-formed difference — and multiplies
 *     by a PRECOMPUTED inv_norm rather than dividing (:806).
 *   * The Hull combination is 2.0.mul_add(half, -full) (hma.rs:849) — ONE fma,
 *     not 2*half - full.
 *   * The EMA seeds on the first value, runs a RUNNING MEAN for bars
 *     2..=period as (x - mean).mul_add(1/count, mean), and only then switches
 *     to beta.mul_add(mean, alpha * x) (ema.rs:600-607). Three regimes.
 *   * The money-flow ratio is pos_sum / neg_sum and the index is
 *     100 - (100 / (1 + ratio)) (:437-438) — reproduced as written, including
 *     the 0/0 that yields NaN and the finite/0 that yields the exact 100.
 *   * The band clamps are written as IF-CHAINS on the CPU (:539-549), not as
 *     min/max, and they are reproduced as if-chains for that reason: the CPU
 *     keeps the PREVIOUS band when the test fails, which is not what fmin
 *     would return for a NaN operand.
 *   * There is no epsilon anywhere in this column.
 *
 * NaN semantics: LinWma tracks a nan_count and returns NaN for the whole
 *   window while any member is NaN, then REBUILDS the accumulators from the
 *   ring once the window is clean again (hma.rs:783-800). That rebuild is not
 *   an optimisation — it is what stops a NaN-poisoned incremental sum from
 *   persisting — so it is reproduced exactly, including its own accumulation
 *   order (oldest first, weight i+1).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:4465-4468, plus MFI_HMA_LENGTH (:33). Each sizes
 * a per-thread ring, so the bounds belong to the COMPILED kernel. */
#define NEO_TFT_ALPHA_LENGTH      33
#define NEO_TFT_ALPHA_MULTIPLIER  3.3
#define NEO_TFT_MFI_LENGTH        14
#define NEO_TFT_MFI_HMA_LENGTH     7
#define NEO_TFT_WMA_MAX           NEO_TFT_ALPHA_LENGTH

/* LinWma (moving_averages/hma.rs:696). */
struct NeoTftWma {
    int    period;
    double inv_norm;
    double buffer[NEO_TFT_WMA_MAX];
    int    head;
    bool   filled;
    int    count;
    double sum;
    double wsum;
    int    nan_count;
    bool   dirty;
};

static __device__ inline void neo_tft_wma_init(NeoTftWma* w, int period) {
    const double norm = (double)period * ((double)period + 1.0) * 0.5;
    w->period   = period;
    w->inv_norm = 1.0 / norm;
    for (int i = 0; i < period; ++i) w->buffer[i] = NEO_F64_NAN;
    w->head = 0; w->filled = false; w->count = 0;
    w->sum = 0.0; w->wsum = 0.0; w->nan_count = 0; w->dirty = false;
}

static __device__ inline void neo_tft_wma_rebuild(NeoTftWma* w) {
    w->sum = 0.0; w->wsum = 0.0; w->nan_count = 0;
    int idx = w->head;
    for (int i = 0; i < w->period; ++i) {
        const double v = w->buffer[idx];
        if (isnan(v)) {
            ++w->nan_count;
        } else {
            w->sum += v;
            w->wsum = fma((double)i + 1.0, v, w->wsum);
        }
        idx = (idx + 1 == w->period) ? 0 : (idx + 1);
    }
    w->dirty = (w->nan_count != 0);
}

static __device__ inline bool neo_tft_wma_update(NeoTftWma* w, double value, double* outv) {
    const double n = (double)w->period;

    const double old = w->buffer[w->head];
    w->buffer[w->head] = value;
    w->head = (w->head + 1 == w->period) ? 0 : (w->head + 1);

    if (!w->filled) {
        ++w->count;
        if (isnan(value)) {
            ++w->nan_count;
            w->dirty = true;
        } else {
            w->sum += value;
            w->wsum = fma((double)w->count, value, w->wsum);
        }
        if (w->count == w->period) {
            w->filled = true;
            *outv = (w->nan_count > 0) ? NEO_F64_NAN : (w->wsum * w->inv_norm);
            return true;
        }
        return false;
    }

    if (isnan(old))   { if (w->nan_count > 0) --w->nan_count; }
    if (isnan(value)) ++w->nan_count;

    if (w->nan_count > 0) { w->dirty = true; *outv = NEO_F64_NAN; return true; }

    if (w->dirty) {
        neo_tft_wma_rebuild(w);
        w->dirty = false;
        *outv = w->wsum * w->inv_norm;
        return true;
    }

    const double prev_sum = w->sum;
    w->sum  = prev_sum + value - old;
    w->wsum = fma(n, value, w->wsum - prev_sum);
    *outv = w->wsum * w->inv_norm;
    return true;
}

/* HmaStream (moving_averages/hma.rs:812). */
struct NeoTftHma { NeoTftWma half, full, sqrtw; };

static __device__ inline void neo_tft_hma_init(NeoTftHma* s, int period) {
    const int half = period / 2;
    int sqrt_len = (int)floor(sqrt((double)period));
    neo_tft_wma_init(&s->half, half);
    neo_tft_wma_init(&s->full, period);
    neo_tft_wma_init(&s->sqrtw, sqrt_len);
}

static __device__ inline bool neo_tft_hma_update(NeoTftHma* s, double value, double* outv) {
    double f, h;
    const bool have_f = neo_tft_wma_update(&s->full, value, &f);
    const bool have_h = neo_tft_wma_update(&s->half, value, &h);
    if (have_f && have_h) {
        const double x = fma(2.0, h, -f);
        return neo_tft_wma_update(&s->sqrtw, x, outv);
    }
    return false;
}

extern "C" __global__
void trend_flow_trail_neo_batch_f64(const double* __restrict__ open,
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
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* the mid-series reset reproduces it — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    alpha_length = NEO_TFT_ALPHA_LENGTH;
    const double alpha_mult   = NEO_TFT_ALPHA_MULTIPLIER;
    const int    mfi_length   = NEO_TFT_MFI_LENGTH;

    NeoTftHma basis, mfi_hma;
    neo_tft_hma_init(&basis, alpha_length);
    neo_tft_hma_init(&mfi_hma, NEO_TFT_MFI_HMA_LENGTH);

    /* EmaStream(alpha_length) */
    const double ema_alpha = 2.0 / ((double)alpha_length + 1.0);
    const double ema_beta  = 1.0 - ema_alpha;
    int    ema_count = 0;
    double ema_mean  = NEO_F64_NAN;
    bool   ema_filled = false;

    /* MoneyFlowRawState(mfi_length) */
    double mf_pos[NEO_TFT_MFI_LENGTH], mf_neg[NEO_TFT_MFI_LENGTH];
    for (int k = 0; k < mfi_length; ++k) { mf_pos[k] = 0.0; mf_neg[k] = 0.0; }
    int    mf_head = 0, mf_count = 0;
    double mf_pos_sum = 0.0, mf_neg_sum = 0.0;
    bool   mf_have_prev = false;
    double mf_prev_src  = 0.0;

    bool   have_prev_upper = false, have_prev_lower = false, have_prev_trail = false;
    bool   have_prev_close = false;
    double prev_upper_s = 0.0, prev_lower_s = 0.0, prev_trail_s = 0.0, prev_close_s = 0.0;

    for (int i = 0; i < n; ++i) {
        const double op = open[i], h = high[i], l = low[i], c = close[i], v = volume[i];

        if (!(isfinite(op) && isfinite(h) && isfinite(l) && isfinite(c) && isfinite(v))) {
            neo_tft_hma_init(&basis, alpha_length);
            neo_tft_hma_init(&mfi_hma, NEO_TFT_MFI_HMA_LENGTH);
            ema_count = 0; ema_mean = NEO_F64_NAN; ema_filled = false;
            for (int k = 0; k < mfi_length; ++k) { mf_pos[k] = 0.0; mf_neg[k] = 0.0; }
            mf_head = 0; mf_count = 0; mf_pos_sum = 0.0; mf_neg_sum = 0.0;
            mf_have_prev = false; mf_prev_src = 0.0;
            have_prev_upper = false; have_prev_lower = false;
            have_prev_trail = false; have_prev_close = false;
            prev_upper_s = 0.0; prev_lower_s = 0.0; prev_trail_s = 0.0; prev_close_s = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        double basis_v;
        const bool have_basis = neo_tft_hma_update(&basis, c, &basis_v);

        /* EmaStream::update over |high - low|, scaled by alpha_multiplier. */
        const double range_in = fabs(h - l);
        ++ema_count;
        if (ema_count == 1) {
            ema_mean = range_in;
        } else if (ema_count <= alpha_length) {
            const double inv = 1.0 / (double)ema_count;
            ema_mean = fma(range_in - ema_mean, inv, ema_mean);
        } else {
            ema_mean = fma(ema_beta, ema_mean, ema_alpha * range_in);
        }
        if (!ema_filled && ema_count >= alpha_length) ema_filled = true;
        const bool   have_spread = ema_filled;
        const double spread_v    = ema_mean * alpha_mult;

        /* MoneyFlowRawState::update on (hlc3, volume), then the MFI Hull MA. */
        const double src   = (h + l + c) / 3.0;
        const double delta = mf_have_prev ? (src - mf_prev_src) : 0.0;
        mf_have_prev = true;
        mf_prev_src  = src;
        const double pos_flow = (delta > 0.0) ? (v * src) : 0.0;
        const double neg_flow = (delta < 0.0) ? (v * src) : 0.0;
        if (mf_count == mfi_length) {
            mf_pos_sum -= mf_pos[mf_head];
            mf_neg_sum -= mf_neg[mf_head];
        } else {
            ++mf_count;
        }
        mf_pos[mf_head] = pos_flow;
        mf_neg[mf_head] = neg_flow;
        mf_pos_sum += pos_flow;
        mf_neg_sum += neg_flow;
        mf_head = (mf_head + 1) % mfi_length;

        bool   have_mfi = false;
        double mfi_v    = NEO_F64_NAN;
        if (mf_count >= mfi_length) {
            const double ratio = mf_pos_sum / mf_neg_sum;
            const double raw   = 100.0 - (100.0 / (1.0 + ratio));
            have_mfi = neo_tft_hma_update(&mfi_hma, raw, &mfi_v);
        }

        const bool   had_prev_close = have_prev_close;
        const double prev_close_v   = prev_close_s;
        have_prev_close = true;
        prev_close_s    = c;

        const bool   had_prev_trail = have_prev_trail;
        const double prev_trail_v   = prev_trail_s;

        if (!(have_basis && have_spread && have_mfi && isfinite(mfi_v))) {
            o[i] = NEO_F64_NAN;
            continue;
        }

        double upper = basis_v + spread_v;
        double lower = basis_v - spread_v;
        const double prev_upper = have_prev_upper ? prev_upper_s : 0.0;
        const double prev_lower = have_prev_lower ? prev_lower_s : 0.0;
        const double prev_close_value = had_prev_close ? prev_close_v : 0.0;

        lower = (lower > prev_lower || prev_close_value < prev_lower) ? lower : prev_lower;
        upper = (upper < prev_upper || prev_close_value > prev_upper) ? upper : prev_upper;

        double alpha_dir;
        if (!had_prev_trail)                    alpha_dir = 1.0;
        else if (prev_trail_v == prev_upper)    alpha_dir = (c > upper) ? -1.0 : 1.0;
        else if (c < lower)                     alpha_dir = 1.0;
        else                                    alpha_dir = -1.0;

        const double alpha_trail = (alpha_dir < 0.0) ? lower : upper;

        have_prev_upper = true; prev_upper_s = upper;
        have_prev_lower = true; prev_lower_s = lower;
        have_prev_trail = true; prev_trail_s = alpha_trail;

        o[i] = alpha_trail;
    }
}
