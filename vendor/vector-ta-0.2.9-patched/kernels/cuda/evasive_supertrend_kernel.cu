#include <cmath>
#include <cstddef>

namespace {

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

struct AtrTrackerDevice {
    int period;
    int count;
    double tr_sum;
    double prev_close;
    bool has_prev_close;
    double atr;

    __device__ void init(int period_value) {
        period = period_value;
        reset();
    }

    __device__ void reset() {
        count = 0;
        tr_sum = 0.0;
        prev_close = 0.0;
        has_prev_close = false;
        atr = NAN;
    }

    __device__ bool update(double high, double low, double close, double* out_atr) {
        const double tr = has_prev_close
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        has_prev_close = true;

        if (count < period) {
            count += 1;
            tr_sum += tr;
            if (count == period) {
                atr = tr_sum / static_cast<double>(period);
                *out_atr = atr;
                return true;
            }
            return false;
        }

        atr = ((atr * static_cast<double>(period - 1)) + tr) / static_cast<double>(period);
        *out_atr = atr;
        return true;
    }
};

}

extern "C" __global__ void evasive_supertrend_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ atr_lengths,
    const double* __restrict__ base_multipliers,
    const double* __restrict__ noise_thresholds,
    const double* __restrict__ expansion_alphas,
    int rows,
    double* __restrict__ out_band,
    double* __restrict__ out_state,
    double* __restrict__ out_noisy,
    double* __restrict__ out_changed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int atr_length = atr_lengths[row];
    const double base_multiplier = base_multipliers[row];
    const double noise_threshold = noise_thresholds[row];
    const double expansion_alpha = expansion_alphas[row];

    double* row_band = out_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_noisy = out_noisy + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_band[i] = NAN;
        row_state[i] = NAN;
        row_noisy[i] = NAN;
        row_changed[i] = NAN;
    }

    if (atr_length <= 0 || !isfinite(base_multiplier) || base_multiplier < 0.1 ||
        !isfinite(noise_threshold) || noise_threshold < 0.1 || !isfinite(expansion_alpha) ||
        expansion_alpha < 0.0) {
        return;
    }

    AtrTrackerDevice tracker;
    tracker.init(atr_length);
    int trend = 1;
    double band = NAN;

    for (int i = 0; i < len; ++i) {
        if (!is_valid_ohlc(open[i], high[i], low[i], close[i])) {
            tracker.reset();
            trend = 1;
            band = NAN;
            continue;
        }

        double atr = NAN;
        if (!tracker.update(high[i], low[i], close[i], &atr)) {
            continue;
        }

        const double src = (high[i] + low[i]) * 0.5;
        const double upper_base = src + base_multiplier * atr;
        const double lower_base = src - base_multiplier * atr;
        const double prev_band = isnan(band) ? (trend == 1 ? lower_base : upper_base) : band;
        const bool is_noisy = fabs(close[i] - prev_band) < atr * noise_threshold;
        const int prev_trend = trend;
        double next_band = NAN;

        if (prev_trend == 1) {
            next_band = is_noisy ? (prev_band - atr * expansion_alpha) : fmax(lower_base, prev_band);
            if (close[i] < next_band) {
                trend = -1;
                next_band = upper_base;
            }
        } else {
            next_band = is_noisy ? (prev_band + atr * expansion_alpha) : fmin(upper_base, prev_band);
            if (close[i] > next_band) {
                trend = 1;
                next_band = lower_base;
            }
        }

        band = next_band;
        row_band[i] = next_band;
        row_state[i] = static_cast<double>(trend);
        row_noisy[i] = is_noisy ? 1.0 : 0.0;
        row_changed[i] = trend != prev_trend ? 1.0 : 0.0;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — evasive_supertrend
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/evasive_supertrend.rs:521 `compute_row`, driving
 *   `compute_point` (:462) over an `AtrTracker` (:325). Walks every bar from 0;
 *   an invalid bar RESETS the tracker, forces `trend = 1` and `band = NaN` and
 *   writes NOTHING (the outputs keep their NaN prefill), so `first_valid` is
 *   not read. A bar where the ATR is not yet seeded also writes nothing.
 *
 * Column: output_id "value" / "band" -> `out.band` (cpu_batch.rs:7254).
 *
 * PERIOD-INVARIANT: `compute_evasive_supertrend_batch` (cpu_batch.rs:7229-7235)
 *   reads `atr_length` (10), `base_multiplier` (3.0), `noise_threshold` (1.0)
 *   and `expansion_alpha` (0.5) — and NEVER `period`.
 *
 * Input: open / high / low / close — F64InputKind::Ohlc4. `open` never enters
 *   the arithmetic but it DOES gate validity (`is_valid_ohlc`, :354), so a bar
 *   with a NaN open resets the trend state on the CPU. Dropping open would
 *   carry state across a gap the CPU breaks.
 *
 * The ATR seed is the MEAN of the first `atr_length` true ranges and the
 *   recursion is `((atr * (p - 1)) + tr) / p` (:347) — TWO roundings in that
 *   exact shape. The Wilder form `(tr - atr).mul_add(1/p, atr)` is ONE and
 *   would drift; this is the same trap the brief names for natr.
 *
 * `band` starts as NaN and the FIRST emitted bar substitutes `lower_base` or
 *   `upper_base` for it depending on the trend seed of +1 (:477-485). That
 *   substitution happens once per reset, not once per series.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:7229-7235. */
#define NEO_EVST_ATR_LENGTH      10
#define NEO_EVST_BASE_MULTIPLIER 3.0
#define NEO_EVST_NOISE_THRESHOLD 1.0
#define NEO_EVST_EXPANSION_ALPHA 0.5

extern "C" __global__
void evasive_supertrend_neo_batch_f64(const double* __restrict__ open,
                                      const double* __restrict__ high,
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
    (void)periods;
    (void)first_valid;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    P  = NEO_EVST_ATR_LENGTH;
    const double Pf = (double)P;
    const double base_multiplier = NEO_EVST_BASE_MULTIPLIER;
    const double noise_threshold = NEO_EVST_NOISE_THRESHOLD;
    const double expansion_alpha = NEO_EVST_EXPANSION_ALPHA;

    /* AtrTracker (:296) */
    int    count = 0;
    double tr_sum = 0.0;
    double atr = 0.0;
    bool   have_prev_close = false;
    double prev_close = 0.0;

    int    trend = 1;
    double band  = NEO_F64_NAN;

    for (int i = 0; i < n; ++i) {
        const double op = open[i], hi = high[i], lo = low[i], cl = close[i];
        if (!(isfinite(op) && isfinite(hi) && isfinite(lo) && isfinite(cl))) {
            count = 0; tr_sum = 0.0; atr = 0.0;
            have_prev_close = false; prev_close = 0.0;
            trend = 1; band = NEO_F64_NAN;
            continue;                       /* nothing written -- stays NaN */
        }

        /* tracker.update (:325) */
        double tr;
        if (have_prev_close) {
            const double hl = hi - lo;
            const double hc = fabs(hi - prev_close);
            const double lc = fabs(lo - prev_close);
            tr = fmax(fmax(hl, hc), lc);
        } else {
            tr = hi - lo;
        }
        prev_close = cl; have_prev_close = true;

        bool atr_ready;
        if (count < P) {
            ++count;
            tr_sum += tr;
            if (count == P) { atr = tr_sum / Pf; atr_ready = true; }
            else            { atr_ready = false; }
        } else {
            atr = ((atr * (Pf - 1.0)) + tr) / Pf;
            atr_ready = true;
        }
        if (!atr_ready) continue;           /* compute_point returned None */

        /* compute_point (:462) */
        const double src = (hi + lo) * 0.5;
        const double upper_base = src + base_multiplier * atr;
        const double lower_base = src - base_multiplier * atr;
        const double prev_band = isnan(band) ? ((trend == 1) ? lower_base : upper_base)
                                             : band;
        const bool is_noisy = fabs(cl - prev_band) < atr * noise_threshold;
        double next_band;

        if (trend == 1) {
            next_band = is_noisy ? (prev_band - atr * expansion_alpha)
                                 : fmax(lower_base, prev_band);
            if (cl < next_band) { trend = -1; next_band = upper_base; }
        } else {
            next_band = is_noisy ? (prev_band + atr * expansion_alpha)
                                 : fmin(upper_base, prev_band);
            if (cl > next_band) { trend = 1; next_band = lower_base; }
        }

        band = next_band;
        o[i] = next_band;
    }
}
