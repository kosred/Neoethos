#include <cmath>
#include <cstddef>

namespace {
constexpr double DEFAULT_TREND = 1.0;
constexpr double MIN_MULTIPLIER = 0.1;
constexpr double MIN_ALPHA_PERCENT = 0.1;
constexpr double MAX_ALPHA_PERCENT = 100.0;

struct AtrState {
    int length;
    int count;
    double sum;
    double value;

    __device__ void init(int period) {
        length = period;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
    }

    __device__ double update(double tr, bool* ready) {
        if (count < length) {
            count += 1;
            sum += tr;
            if (count == length) {
                value = sum / static_cast<double>(length);
                *ready = true;
                return value;
            }
            *ready = false;
            return NAN;
        }

        value = ((value * static_cast<double>(length - 1)) + tr) / static_cast<double>(length);
        *ready = true;
        return value;
    }
};

__device__ inline double true_range(double high, double low, double prev_close) {
    return fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)));
}
}

extern "C" __global__ void supertrend_recovery_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* atr_lengths,
    const double* multipliers,
    const double* alpha_percents,
    const double* threshold_atrs,
    int rows,
    double* out_band,
    double* out_switch_price,
    double* out_trend,
    double* out_changed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int atr_length = atr_lengths[row];
    const double multiplier = multipliers[row];
    const double alpha_percent = alpha_percents[row];
    const double threshold_atr = threshold_atrs[row];

    double* row_band = out_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_switch_price =
        out_switch_price + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_band[i] = NAN;
        row_switch_price[i] = NAN;
        row_trend[i] = NAN;
        row_changed[i] = NAN;
    }

    if (atr_length <= 0 || atr_length > len || !isfinite(multiplier) || multiplier < MIN_MULTIPLIER
        || !isfinite(alpha_percent) || alpha_percent < MIN_ALPHA_PERCENT
        || alpha_percent > MAX_ALPHA_PERCENT || !isfinite(threshold_atr)
        || threshold_atr < 0.0) {
        return;
    }

    const double alpha = alpha_percent * 0.01;

    AtrState atr;
    atr.init(atr_length);

    double prev_close = NAN;
    double band = NAN;
    double switch_price = NAN;
    double trend = DEFAULT_TREND;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            atr.reset();
            prev_close = NAN;
            band = NAN;
            switch_price = NAN;
            trend = DEFAULT_TREND;
            continue;
        }

        if (!isfinite(switch_price)) {
            switch_price = c;
        }

        const double tr = isfinite(prev_close) ? true_range(h, l, prev_close) : (h - l);
        prev_close = c;

        bool atr_ready = false;
        const double atr_value = atr.update(tr, &atr_ready);
        if (!atr_ready) {
            continue;
        }

        const double src = 0.5 * (h + l);
        const double upper_base = src + multiplier * atr_value;
        const double lower_base = src - multiplier * atr_value;
        const double deviation = threshold_atr * atr_value;
        const bool is_at_loss =
            (trend == 1.0 && (switch_price - c) > deviation)
            || (trend == -1.0 && (c - switch_price) > deviation);
        const double prev_band = isfinite(band) ? band : (trend == 1.0 ? lower_base : upper_base);

        double changed = 0.0;

        if (trend == 1.0) {
            const double target_band =
                is_at_loss ? fma(alpha, c, (1.0 - alpha) * prev_band) : lower_base;
            band = fmax(target_band, prev_band);
            if (c < band) {
                trend = -1.0;
                band = upper_base;
                switch_price = c;
                changed = 1.0;
            }
        } else {
            const double target_band =
                is_at_loss ? fma(alpha, c, (1.0 - alpha) * prev_band) : upper_base;
            band = fmin(target_band, prev_band);
            if (c > band) {
                trend = 1.0;
                band = lower_base;
                switch_price = c;
                changed = 1.0;
            }
        }

        row_band[i] = band;
        row_switch_price[i] = switch_price;
        row_trend[i] = trend;
        row_changed[i] = changed;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — supertrend_recovery
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/supertrend_recovery.rs:776 compute_row, driving
 *   SuperTrendRecoveryState::update (:575) with AtrState::update (:521),
 *   true_range (:100) and hl2 (:95).
 *
 * Column: output_id "value" resolves to out.band — cpu_batch.rs:4934 accepts
 *   "band"/"value". switch_price, trend and changed are separate output ids;
 *   switch_price and trend are still CARRIED here because the band recurrence
 *   reads both.
 *
 * PERIOD-INVARIANT: compute_supertrend_recovery_batch reads atr_length (10),
 *   multiplier (3.0), alpha_percent (5.0) and threshold_atr (1.0) and NEVER
 *   period (cpu_batch.rs:4913-4916).
 *
 * FIRST-VALID IGNORED: update RESETS the whole state on any non-finite bar
 *   (:576-579) and compute_row walks EVERY bar from index 0. The caller's
 *   first-valid index only sizes a NaN prefix the row then overwrites.
 *
 * Input: high / low / close — extract_ohlc_input (cpu_batch.rs:4905) —
 *   F64InputKind::Hlc.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. The ATR, the latched trend,
 *   the switch price and the band itself all carry across bars, and the band
 *   is clamped against ITS OWN previous value.
 *
 * THE ALPHA ROUND TRIP IS REPRODUCED, NOT SIMPLIFIED. prepare_input stores
 *   alpha = alpha_percent * 0.01 (:488); supertrend_recovery_with_kernel then
 *   passes `prepared.alpha / 0.01` to compute_row (:847); compute_row passes
 *   `alpha_percent * 0.01` to the state (:805). That is multiply, divide,
 *   multiply — three roundings on the same constant, and collapsing it to
 *   0.05 is not guaranteed to be the same double. The chain is written out
 *   below exactly as the CPU performs it.
 *
 * ARITHMETIC taken verbatim:
 *   * true_range is (h-l).max(|h-pc|).max(|l-pc|) (:100-104) — f64::max, so
 *     fmax is used: it returns the non-NaN operand where an if-chain would
 *     let a NaN through.
 *   * the ATR seed is sum / length (:526) — a DIVIDE, not a multiply by a
 *     reciprocal.
 *   * the ATR step is ((value * (length - 1)) + tr) / length (:532) — THREE
 *     roundings in that exact shape. It is deliberately NOT rewritten as the
 *     Wilder fma (tr - value).mul_add(1/length, value), which is one.
 *   * the recovery band is alpha.mul_add(close, (1 - alpha) * prev_band)
 *     (:611, :624) — ONE fma whose addend is itself a product.
 *   * the band clamps are f64::max / f64::min (:615, :628) — fmax/fmin.
 *   * there is no epsilon anywhere: every guard is an exact comparison.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:4913-4916. */
#define NEO_STR_ATR_LENGTH     10
#define NEO_STR_MULTIPLIER     3.0
#define NEO_STR_ALPHA_PERCENT  5.0
#define NEO_STR_THRESHOLD_ATR  1.0

extern "C" __global__
void supertrend_recovery_neo_batch_f64(const double* __restrict__ high,
                                       const double* __restrict__ low,
                                       const double* __restrict__ close,
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

    const int atr_length = NEO_STR_ATR_LENGTH;
    if (atr_length > n) return;   /* validate_params / NotEnoughValidData */

    const double multiplier    = NEO_STR_MULTIPLIER;
    const double threshold_atr = NEO_STR_THRESHOLD_ATR;
    /* multiply, divide, multiply — see header. */
    const double prepared_alpha = NEO_STR_ALPHA_PERCENT * 0.01;
    const double alpha          = (prepared_alpha / 0.01) * 0.01;

    const double len_f = (double)atr_length;

    int    atr_count = 0;
    double atr_sum   = 0.0;
    double atr_value = NEO_F64_NAN;

    double prev_close   = NEO_F64_NAN;
    double band         = NEO_F64_NAN;
    double switch_price = NEO_F64_NAN;
    int    trend        = 1;      /* DEFAULT_TREND (:74) */

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            atr_count = 0; atr_sum = 0.0; atr_value = NEO_F64_NAN;
            prev_close = NEO_F64_NAN; band = NEO_F64_NAN;
            switch_price = NEO_F64_NAN; trend = 1;
            o[i] = NEO_F64_NAN;
            continue;
        }

        if (!isfinite(switch_price)) switch_price = c;

        double tr;
        if (isfinite(prev_close)) {
            tr = fmax(fmax(h - l, fabs(h - prev_close)), fabs(l - prev_close));
        } else {
            tr = h - l;
        }
        prev_close = c;

        /* AtrState::update (:521) */
        double atr;
        if (atr_count < atr_length) {
            ++atr_count;
            atr_sum += tr;
            if (atr_count == atr_length) {
                atr_value = atr_sum / len_f;
                atr = atr_value;
            } else {
                o[i] = NEO_F64_NAN;
                continue;                       /* the `?` on update (:592) */
            }
        } else {
            atr_value = ((atr_value * (len_f - 1.0)) + tr) / len_f;
            atr = atr_value;
        }

        const double src        = 0.5 * (h + l);
        const double upper_base = src + multiplier * atr;
        const double lower_base = src - multiplier * atr;
        const double deviation  = threshold_atr * atr;

        const bool is_at_loss = (trend ==  1 && (switch_price - c) > deviation) ||
                                (trend == -1 && (c - switch_price) > deviation);

        double prev_band;
        if (isfinite(band))    prev_band = band;
        else if (trend == 1)   prev_band = lower_base;
        else                   prev_band = upper_base;

        if (trend == 1) {
            const double target_band = is_at_loss
                ? fma(alpha, c, (1.0 - alpha) * prev_band)
                : lower_base;
            band = fmax(target_band, prev_band);
            if (c < band) {
                trend = -1;
                band = upper_base;
                switch_price = c;
            }
        } else {
            const double target_band = is_at_loss
                ? fma(alpha, c, (1.0 - alpha) * prev_band)
                : upper_base;
            band = fmin(target_band, prev_band);
            if (c > band) {
                trend = 1;
                band = lower_base;
                switch_price = c;
            }
        }

        o[i] = band;
    }
}
