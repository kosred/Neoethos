#include <cmath>
#include <cstddef>

namespace {

constexpr int WMA_PERIOD = 200;

struct KalmanState {
    double alpha_mul_period;
    double beta_div_period;
    double value;
    double covariance;
    bool has_value;

    __device__ void init(double alpha, int period, double beta) {
        alpha_mul_period = alpha * static_cast<double>(period);
        beta_div_period = beta / static_cast<double>(period);
        reset();
    }

    __device__ void reset() {
        value = NAN;
        covariance = 1.0;
        has_value = false;
    }

    __device__ bool update(double input, bool has_prev_input, double prev_input, double* out) {
        const double gain = covariance / (covariance + alpha_mul_period);
        if (!has_value && has_prev_input) {
            value = prev_input;
            has_value = true;
        }
        bool ready = false;
        if (has_value) {
            const double next = value + gain * (input - value);
            value = next;
            *out = next;
            ready = true;
        }
        covariance = (1.0 - gain) * covariance + beta_div_period;
        return ready;
    }
};

struct AtrState {
    int period;
    int count;
    double sum;
    double value;
    double prev_close;
    bool seeded;
    bool has_prev_close;

    __device__ void init(int period_value) {
        period = period_value;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        prev_close = NAN;
        seeded = false;
        has_prev_close = false;
    }

    __device__ bool update(double high, double low, double close, double* out) {
        const double tr = has_prev_close
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        has_prev_close = true;

        if (seeded) {
            value = ((value * static_cast<double>(period - 1)) + tr) / static_cast<double>(period);
            *out = value;
            return true;
        }

        count += 1;
        sum += tr;
        if (count == period) {
            value = sum / static_cast<double>(period);
            seeded = true;
            *out = value;
            return true;
        }
        return false;
    }
};

struct WmaState {
    double* buffer;
    int head;
    int len;
    double sum;
    double weighted_sum;
    double divisor;

    __device__ void init(double* buffer_ptr) {
        buffer = buffer_ptr;
        divisor = static_cast<double>(WMA_PERIOD * (WMA_PERIOD + 1) / 2);
        reset();
    }

    __device__ void reset() {
        head = 0;
        len = 0;
        sum = 0.0;
        weighted_sum = 0.0;
    }

    __device__ bool update(double value, double* out) {
        if (!isfinite(value)) {
            reset();
            return false;
        }
        if (len < WMA_PERIOD) {
            const int pos = (head + len) % WMA_PERIOD;
            buffer[pos] = value;
            len += 1;
            sum += value;
            weighted_sum += static_cast<double>(len) * value;
            if (len == WMA_PERIOD) {
                *out = weighted_sum / divisor;
                return true;
            }
            return false;
        }

        const double oldest = buffer[head];
        const double old_sum = sum;
        buffer[head] = value;
        head = (head + 1) % WMA_PERIOD;
        weighted_sum = weighted_sum - old_sum + static_cast<double>(WMA_PERIOD) * value;
        sum = old_sum - oldest + value;
        *out = weighted_sum / divisor;
        return true;
    }
};

struct SuperTrendState {
    double factor;
    double prev_lower_band;
    double prev_upper_band;
    double prev_k;
    bool has_prev_lower_band;
    bool has_prev_upper_band;
    bool has_prev_k;
    bool prev_atr_ready;
    int prev_direction;

    __device__ void init(double factor_value) {
        factor = factor_value;
        reset();
    }

    __device__ void reset() {
        prev_lower_band = NAN;
        prev_upper_band = NAN;
        prev_k = NAN;
        has_prev_lower_band = false;
        has_prev_upper_band = false;
        has_prev_k = false;
        prev_atr_ready = false;
        prev_direction = 1;
    }

    __device__ void update(double k, double atr, double* supertrend, int* direction) {
        double upper_band = k + factor * atr;
        double lower_band = k - factor * atr;
        const double prev_lower = has_prev_lower_band ? prev_lower_band : lower_band;
        const double prev_upper = has_prev_upper_band ? prev_upper_band : upper_band;
        const double prev_k_value = has_prev_k ? prev_k : k;

        if (!(lower_band > prev_lower || prev_k_value < prev_lower)) {
            lower_band = prev_lower;
        }
        if (!(upper_band < prev_upper || prev_k_value > prev_upper)) {
            upper_band = prev_upper;
        }

        if (!prev_atr_ready) {
            *direction = 1;
        } else if (prev_direction == 1) {
            *direction = k > upper_band ? -1 : 1;
        } else if (k < lower_band) {
            *direction = 1;
        } else {
            *direction = -1;
        }

        *supertrend = *direction == -1 ? lower_band : upper_band;
        prev_lower_band = lower_band;
        prev_upper_band = upper_band;
        prev_k = k;
        has_prev_lower_band = true;
        has_prev_upper_band = true;
        has_prev_k = true;
        prev_atr_ready = true;
        prev_direction = *direction;
    }
};

}

extern "C" __global__ void range_filtered_trend_signals_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ kalman_alphas,
    const double* __restrict__ kalman_betas,
    const int* __restrict__ kalman_periods,
    const double* __restrict__ devs,
    const double* __restrict__ supertrend_factors,
    const int* __restrict__ supertrend_atr_periods,
    int rows,
    double* __restrict__ wma_scratch,
    double* __restrict__ out_kalman,
    double* __restrict__ out_supertrend,
    double* __restrict__ out_upper_band,
    double* __restrict__ out_lower_band,
    double* __restrict__ out_trend,
    double* __restrict__ out_kalman_trend,
    double* __restrict__ out_state,
    double* __restrict__ out_market_trending,
    double* __restrict__ out_market_ranging,
    double* __restrict__ out_short_term_bullish,
    double* __restrict__ out_short_term_bearish,
    double* __restrict__ out_long_term_bullish,
    double* __restrict__ out_long_term_bearish
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_kalman = out_kalman + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_supertrend =
        out_supertrend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_band =
        out_upper_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_band =
        out_lower_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_kalman_trend =
        out_kalman_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_market_trending =
        out_market_trending + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_market_ranging =
        out_market_ranging + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_term_bullish =
        out_short_term_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_term_bearish =
        out_short_term_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_term_bullish =
        out_long_term_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_term_bearish =
        out_long_term_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_kalman[i] = NAN;
        row_supertrend[i] = NAN;
        row_upper_band[i] = NAN;
        row_lower_band[i] = NAN;
        row_trend[i] = NAN;
        row_kalman_trend[i] = NAN;
        row_state[i] = NAN;
        row_market_trending[i] = NAN;
        row_market_ranging[i] = NAN;
        row_short_term_bullish[i] = NAN;
        row_short_term_bearish[i] = NAN;
        row_long_term_bullish[i] = NAN;
        row_long_term_bearish[i] = NAN;
    }

    const double kalman_alpha = kalman_alphas[row];
    const double kalman_beta = kalman_betas[row];
    const int kalman_period = kalman_periods[row];
    const double dev = devs[row];
    const double supertrend_factor = supertrend_factors[row];
    const int supertrend_atr_period = supertrend_atr_periods[row];
    if (!isfinite(kalman_alpha) || kalman_alpha <= 0.0 || !isfinite(kalman_beta) ||
        kalman_beta < 0.0 || kalman_period <= 0 || !isfinite(dev) || dev < 0.0 ||
        !isfinite(supertrend_factor) || supertrend_factor < 0.0 ||
        supertrend_atr_period <= 0) {
        return;
    }

    KalmanState kalman_state;
    kalman_state.init(kalman_alpha, kalman_period, kalman_beta);
    AtrState atr_state;
    atr_state.init(supertrend_atr_period);
    WmaState wma_state;
    wma_state.init(wma_scratch + static_cast<size_t>(row) * static_cast<size_t>(WMA_PERIOD));
    SuperTrendState supertrend_state;
    supertrend_state.init(supertrend_factor);

    bool has_prev_close = false;
    double prev_close = NAN;
    double trend_state = 0.0;
    bool has_prev_trend = false;
    double prev_trend = NAN;
    bool has_prev_kalman_trend = false;
    double prev_kalman_trend = NAN;
    bool has_prev_state = false;
    double prev_state = NAN;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            kalman_state.reset();
            atr_state.reset();
            wma_state.reset();
            supertrend_state.reset();
            has_prev_close = false;
            prev_close = NAN;
            trend_state = 0.0;
            has_prev_trend = false;
            prev_trend = NAN;
            has_prev_kalman_trend = false;
            prev_kalman_trend = NAN;
            has_prev_state = false;
            prev_state = NAN;
            continue;
        }

        double kalman_value = NAN;
        const bool kalman_ready =
            kalman_state.update(c, has_prev_close, prev_close, &kalman_value);
        prev_close = c;
        has_prev_close = true;

        double atr_value = NAN;
        const bool atr_ready = atr_state.update(h, l, c, &atr_value);

        double vola = NAN;
        const bool vola_ready = wma_state.update(h - l, &vola);

        double supertrend = NAN;
        int direction = 0;
        bool supertrend_ready = false;
        if (kalman_ready && atr_ready) {
            supertrend_state.update(kalman_value, atr_value, &supertrend, &direction);
            supertrend_ready = true;
        }

        if (!(kalman_ready && atr_ready && vola_ready && supertrend_ready)) {
            continue;
        }

        const double upper_band = kalman_value + vola * dev;
        const double lower_band = kalman_value - vola * dev;
        if (c > upper_band) {
            trend_state = 1.0;
        } else if (c < lower_band) {
            trend_state = -1.0;
        }

        const double kalman_trend = direction < 0 ? 1.0 : -1.0;
        const double state = kalman_trend * trend_state;
        const double market_trending =
            has_prev_state && state > 0.0 && prev_state <= 0.0 ? 1.0 : 0.0;
        const double market_ranging =
            has_prev_state && state < 0.0 && prev_state >= 0.0 ? 1.0 : 0.0;
        const double short_term_bullish =
            has_prev_trend && trend_state > 0.0 && prev_trend <= 0.0 ? 1.0 : 0.0;
        const double short_term_bearish =
            has_prev_trend && trend_state < 0.0 && prev_trend >= 0.0 ? 1.0 : 0.0;
        const double long_term_bullish =
            has_prev_kalman_trend && kalman_trend > 0.0 && prev_kalman_trend <= 0.0 ? 1.0 : 0.0;
        const double long_term_bearish =
            has_prev_kalman_trend && kalman_trend < 0.0 && prev_kalman_trend >= 0.0 ? 1.0 : 0.0;

        row_kalman[i] = kalman_value;
        row_supertrend[i] = supertrend;
        row_upper_band[i] = upper_band;
        row_lower_band[i] = lower_band;
        row_trend[i] = trend_state;
        row_kalman_trend[i] = kalman_trend;
        row_state[i] = state;
        row_market_trending[i] = market_trending;
        row_market_ranging[i] = market_ranging;
        row_short_term_bullish[i] = short_term_bullish;
        row_short_term_bearish[i] = short_term_bearish;
        row_long_term_bullish[i] = long_term_bullish;
        row_long_term_bearish[i] = long_term_bearish;

        prev_trend = trend_state;
        has_prev_trend = true;
        prev_kalman_trend = kalman_trend;
        has_prev_kalman_trend = true;
        prev_state = state;
        has_prev_state = true;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// range_filtered_trend_signals_batch_f64 above is genuine double-in/double-out,
// but it takes 25 parameters, writes THIRTEEN output matrices and demands a
// caller-allocated WMA scratch matrix. The f64 lane launches one shape --
// (high, low, close, n, periods, n_combos, first_valid, out) -- and allocates
// ONE output matrix, so that entry point cannot be reached from it.
//
// CPU REFERENCE
//   src/indicators/range_filtered_trend_signals.rs:744
//     range_filtered_trend_signals_with_kernel -> RangeFilteredTrendSignalsCore
//     ::update :588
//   KalmanState::update :335   AtrState::update :375   WmaState::update :438
//
// THE COLUMN THIS EMITS is kalman. This indicator's CPU batch has NO "value"
// output -- compute_range_filtered_trend_signals_batch accepts kalman,
// supertrend, upper_band/upper, lower_band/lower, trend, kalman_trend/
// long_trend, state, market_trending, market_ranging and the four signal
// columns, and REJECTS "value" outright -- so the lane declares the primary
// series, which is the filtered price the whole indicator is named after and
// the first arm of the CPU's own output match.
//
// PINNED CPU DEFAULTS (compute_range_filtered_trend_signals_batch):
// kalman_alpha 0.01, kalman_beta 0.1, kalman_period 77, dev 1.2,
// supertrend_factor 0.7, supertrend_atr_period 7; WMA_PERIOD is the module
// constant 200 (:36).
//
// PERIOD-INVARIANT. The batch reads those six names and NEVER `period`, so five
// swept periods give five identical CPU columns and five identical kernel rows.
//
// SHAPE: one thread per combo, bars ascending. The range filter carries a
// smoothed range and a direction across bars: the Kalman gain is driven by a
// covariance that is UPDATED FROM ITSELF every bar (:344, and note it advances
// even on the seeding bar where no output is produced), the ATR is a Wilder
// recursion, and the 200-deep volatility WMA is a rolling weighted sum whose
// two accumulators are carried.
//
// WHY THE SUPERTREND IS NOT COMPUTED HERE, stated rather than skipped. In
// RangeFilteredTrendSignalsCore::update (:604-609) the supertrend enters the
// emitted `kalman` column through exactly one term: `supertrend_out` is
// `Some(..)` if and only if `kalman` and `atr` are both `Some(..)`, and the
// early return at :610-613 is the only thing it gates. Its VALUE feeds the
// supertrend, trend, state and signal columns, none of which is this one. So
// readiness is reproduced exactly and the band ratchet is not carried -- which
// also side-steps a divergence in the 25-parameter kernel above, where
// `prev_direction == 1` stands in for the CPU's `prev_supertrend ==
// Some(prev_upper_band)` (:394); those disagree whenever a zero ATR makes the
// two bands equal.
//
// ARITHMETIC ORDER:
//   * KalmanState: `gain = covariance / (covariance + alpha*period)` is formed
//     BEFORE the seed check (:336), and `covariance = (1 - gain)*covariance +
//     beta/period` runs on EVERY bar including the one that returns None.
//   * `next = prior + gain * (input - prior)` -- subtract, multiply, add. NOT
//     an fma; -fmad=false stops the compiler contracting it into one.
//   * AtrState seeds from a simple mean of the first `period` true ranges and
//     then rolls `((prev * (period - 1)) + tr) / period` (:392) -- that is the
//     CPU's literal expression, three roundings, and it is written as such.
//     Rewriting it as a one-rounding fma would be a DIFFERENT number.
//   * The first bar of a run has no previous close, and its true range is
//     `high - low` alone, not the three-way max (:377-384).
//   * WmaState on a full ring (:453-454) computes
//     `weighted_sum = weighted_sum - old_sum + period*value` then
//     `sum = old_sum - oldest + value`, both reading the PRE-update sum.
//
// FIRST VALID IS NOT READ: the core derives its own readiness from its window
// counters and a non-finite bar resets everything (:594-597); the row writes
// every index. The lane row declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double fmax/fabs, no f32-suffixed math
// function, no fast-math intrinsic, and no epsilon -- the CPU has none on this
// path.
// ---------------------------------------------------------------------------

#define RFTS_NEO_KALMAN_ALPHA 0.01
#define RFTS_NEO_KALMAN_BETA 0.1
#define RFTS_NEO_KALMAN_PERIOD 77
#define RFTS_NEO_ST_ATR_PERIOD 7

__device__ __forceinline__ double rfts_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void range_filtered_trend_signals_neo_batch_f64(
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
    const double qnan = rfts_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    // KalmanState::new (:318-320).
    const double alpha_mul_period =
        RFTS_NEO_KALMAN_ALPHA * static_cast<double>(RFTS_NEO_KALMAN_PERIOD);
    const double beta_div_period =
        RFTS_NEO_KALMAN_BETA / static_cast<double>(RFTS_NEO_KALMAN_PERIOD);
    double kalman_value = 0.0;
    bool kalman_has_value = false;
    double covariance = 1.0;

    // AtrState::new (:355-362).
    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = 0.0;
    bool atr_seeded = false;
    double atr_prev_close = 0.0;
    bool atr_have_prev_close = false;

    // WmaState::new(WMA_PERIOD) (:412-421). WMA_PERIOD is a module constant, so
    // the ring depth is a property of the compiled kernel.
    double wma_ring[WMA_PERIOD];
    int wma_pos = 0;
    int wma_len = 0;
    double wma_sum = 0.0;
    double wma_weighted_sum = 0.0;
    const double wma_divisor =
        static_cast<double>(WMA_PERIOD * (WMA_PERIOD + 1) / 2);

    double prev_close = 0.0;
    bool has_prev_close = false;

    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            // :594-597 -- reset() clears every sub-state and the carried
            // trend/prev values.
            kalman_value = 0.0;
            kalman_has_value = false;
            covariance = 1.0;
            atr_count = 0;
            atr_sum = 0.0;
            atr_value = 0.0;
            atr_seeded = false;
            atr_prev_close = 0.0;
            atr_have_prev_close = false;
            wma_pos = 0;
            wma_len = 0;
            wma_sum = 0.0;
            wma_weighted_sum = 0.0;
            has_prev_close = false;
            prev_close = 0.0;
            continue;
        }

        // KalmanState::update(close, prev_close) (:335-345).
        double kalman_out = qnan;
        bool kalman_ready = false;
        {
            const double gain = covariance / (covariance + alpha_mul_period);
            if (!kalman_has_value && has_prev_close) {
                kalman_value = prev_close;
                kalman_has_value = true;
            }
            if (kalman_has_value) {
                const double prior = kalman_value;
                const double next = prior + gain * (c - prior);
                kalman_value = next;
                kalman_out = next;
                kalman_ready = true;
            }
            // :344 -- the covariance advances on EVERY bar, including the one
            // that produced no output.
            covariance = (1.0 - gain) * covariance + beta_div_period;
        }
        prev_close = c;
        has_prev_close = true;

        // AtrState::update (:375-400).
        bool atr_ready = false;
        {
            const double tr = atr_have_prev_close
                ? fmax(h - l, fmax(fabs(h - atr_prev_close), fabs(l - atr_prev_close)))
                : (h - l);
            atr_prev_close = c;
            atr_have_prev_close = true;
            if (atr_seeded) {
                atr_value =
                    ((atr_value * (static_cast<double>(RFTS_NEO_ST_ATR_PERIOD) - 1.0)) + tr) /
                    static_cast<double>(RFTS_NEO_ST_ATR_PERIOD);
                atr_ready = true;
            } else {
                atr_count += 1;
                atr_sum += tr;
                if (atr_count == RFTS_NEO_ST_ATR_PERIOD) {
                    atr_value = atr_sum / static_cast<double>(RFTS_NEO_ST_ATR_PERIOD);
                    atr_seeded = true;
                    atr_ready = true;
                }
            }
        }

        // WmaState::update(high - low) (:438-461).
        bool vola_ready = false;
        {
            const double value = h - l;
            if (wma_len < WMA_PERIOD) {
                wma_ring[wma_pos] = value;
                wma_pos = (wma_pos + 1) % WMA_PERIOD;
                wma_len += 1;
                wma_sum += value;
                wma_weighted_sum += static_cast<double>(wma_len) * value;
                if (wma_len == WMA_PERIOD) {
                    vola_ready = true;
                }
            } else {
                const double oldest = wma_ring[wma_pos];
                const double old_sum = wma_sum;
                wma_ring[wma_pos] = value;
                wma_pos = (wma_pos + 1) % WMA_PERIOD;
                wma_weighted_sum =
                    wma_weighted_sum - old_sum + static_cast<double>(WMA_PERIOD) * value;
                wma_sum = old_sum - oldest + value;
                vola_ready = true;
            }
        }
        // wma_divisor is read only through the columns this kernel does not
        // emit; naming it keeps the transliteration complete.
        (void)wma_divisor;

        // :610-613 -- all three must be ready or the bar is NaN. supertrend_out
        // is Some iff kalman and atr are, so it adds no term of its own.
        if (!(kalman_ready && atr_ready && vola_ready)) {
            continue;
        }

        row[i] = kalman_out;
    }
}
