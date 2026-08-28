#include <cmath>
#include <cstddef>

namespace {

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline void evasive_supertrend_row_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    int atr_length,
    double base_multiplier,
    double noise_threshold,
    double expansion_alpha,
    double* __restrict__ out_band,
    double* __restrict__ out_state,
    double* __restrict__ out_noisy,
    double* __restrict__ out_changed
) {
    for (int i = 0; i < len; ++i) {
        if (out_band != nullptr) out_band[i] = NAN;
        if (out_state != nullptr) out_state[i] = NAN;
        if (out_noisy != nullptr) out_noisy[i] = NAN;
        if (out_changed != nullptr) out_changed[i] = NAN;
    }
    if (atr_length <= 0 || !isfinite(base_multiplier) || base_multiplier < 0.1 ||
        !isfinite(noise_threshold) || noise_threshold < 0.1 ||
        !isfinite(expansion_alpha) || expansion_alpha < 0.0) {
        return;
    }

    const double period_f64 = static_cast<double>(atr_length);
    int count = 0;
    double tr_sum = 0.0;
    double prev_close = 0.0;
    bool has_prev_close = false;
    double atr = NAN;
    int trend = 1;
    double band = NAN;

    for (int i = 0; i < len; ++i) {
        const double open_value = open[i];
        const double high_value = high[i];
        const double low_value = low[i];
        const double close_value = close[i];
        if (!is_valid_ohlc(open_value, high_value, low_value, close_value)) {
            count = 0;
            tr_sum = 0.0;
            prev_close = 0.0;
            has_prev_close = false;
            atr = NAN;
            trend = 1;
            band = NAN;
            continue;
        }

        const double tr = has_prev_close
            ? fmax(
                fmax(high_value - low_value, fabs(high_value - prev_close)),
                fabs(low_value - prev_close)
            )
            : (high_value - low_value);
        prev_close = close_value;
        has_prev_close = true;

        if (count < atr_length) {
            count += 1;
            tr_sum += tr;
            if (count != atr_length) continue;
            atr = tr_sum / period_f64;
        } else {
            atr = ((atr * (period_f64 - 1.0)) + tr) / period_f64;
        }

        const double src = (high_value + low_value) * 0.5;
        const double upper_base = src + base_multiplier * atr;
        const double lower_base = src - base_multiplier * atr;
        const double prev_band = isnan(band)
            ? ((trend == 1) ? lower_base : upper_base)
            : band;
        const bool is_noisy = fabs(close_value - prev_band) < atr * noise_threshold;
        const int prev_trend = trend;
        double next_band;

        if (prev_trend == 1) {
            next_band = is_noisy
                ? (prev_band - atr * expansion_alpha)
                : fmax(lower_base, prev_band);
            if (close_value < next_band) {
                trend = -1;
                next_band = upper_base;
            }
        } else {
            next_band = is_noisy
                ? (prev_band + atr * expansion_alpha)
                : fmin(upper_base, prev_band);
            if (close_value > next_band) {
                trend = 1;
                next_band = lower_base;
            }
        }

        band = next_band;
        if (out_band != nullptr) out_band[i] = next_band;
        if (out_state != nullptr) out_state[i] = static_cast<double>(trend);
        if (out_noisy != nullptr) out_noisy[i] = is_noisy ? 1.0 : 0.0;
        if (out_changed != nullptr) out_changed[i] = trend != prev_trend ? 1.0 : 0.0;
    }
}

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

    double* row_band = out_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_noisy = out_noisy + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);
    evasive_supertrend_row_f64(
        open,
        high,
        low,
        close,
        len,
        atr_lengths[row],
        base_multipliers[row],
        noise_thresholds[row],
        expansion_alphas[row],
        row_band,
        row_state,
        row_noisy,
        row_changed
    );
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
 * Canonical primary column: output_id "band". The requested period vector is
 * the registry's `atr_length`; the other three parameters retain their exact
 * scalar defaults in this preserved primary ABI.
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

/* Defaults from cpu_batch.rs:7229-7235. */
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
    (void)first_valid;
    double* __restrict__ row_band = out + (size_t)combo * (size_t)n;
    evasive_supertrend_row_f64(
        open,
        high,
        low,
        close,
        n,
        periods[combo],
        NEO_EVST_BASE_MULTIPLIER,
        NEO_EVST_NOISE_THRESHOLD,
        NEO_EVST_EXPANSION_ALPHA,
        row_band,
        nullptr,
        nullptr,
        nullptr
    );
}
