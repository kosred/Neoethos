#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void volume_zone_oscillator_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ noise_filters,
    const int* __restrict__ intraday_flags,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int noise_filter = noise_filters[combo_idx];
    bool intraday_smoothing = intraday_flags[combo_idx] != 0;
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length < 2 || noise_filter < 2) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(length) + 1.0);
    double beta = 1.0 - alpha;
    double smooth_alpha = 2.0 / (static_cast<double>(noise_filter) + 1.0);
    double smooth_beta = 1.0 - smooth_alpha;

    double prev_close = CUDART_NAN;
    double ema_direction = 0.0;
    double ema_total = 0.0;
    double smooth = 0.0;
    bool smooth_valid = false;
    bool started = false;

    for (int i = 0; i < len; ++i) {
        double vol = volume[i];
        if (!started) {
            if (!isfinite(vol)) {
                continue;
            }
            started = true;
        }

        double raw = CUDART_NAN;
        bool raw_valid = false;
        if (!isfinite(vol)) {
            if (ema_total != 0.0) {
                raw = 100.0 * ema_direction / ema_total;
                raw_valid = true;
            }
        } else {
            double current_close = close[i];
            double directed =
                (isfinite(current_close) && isfinite(prev_close) && current_close > prev_close)
                    ? vol
                    : -vol;
            ema_direction = beta * ema_direction + alpha * directed;
            ema_total = beta * ema_total + alpha * vol;
            if (ema_total != 0.0) {
                raw = 100.0 * ema_direction / ema_total;
                raw_valid = true;
            }
        }

        if (isfinite(close[i])) {
            prev_close = close[i];
        }

        if (intraday_smoothing) {
            if (raw_valid) {
                smooth = smooth_beta * smooth + smooth_alpha * raw;
                smooth_valid = true;
                row[i] = smooth;
            } else if (smooth_valid) {
                row[i] = smooth;
            }
        } else if (raw_valid) {
            row[i] = raw;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — volume_zone_oscillator                      (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/volume_zone_oscillator.rs
 *   :314 compute_volume_zone_oscillator_into  <- the per-bar body
 *   :279 compute_vzo_value                    (the two EMA recurrences)
 *   :247 ema_alpha  = 2 / (period + 1)
 *   :252 extract_close_volume                 first_valid scans VOLUME ALONE
 *
 * PERIOD-INVARIANT. cpu_batch.rs:11552 reads "length" (14),
 * "intraday_smoothing" (true) and "noise_filter" (4); never "period".
 *
 * FIRST-VALID SCANS VOLUME ONLY (:271). Close is deliberately not part of the
 * scan -- a NaN close is handled INSIDE the loop by the `directed` branch,
 * which treats a non-finite close as "not an up bar" and therefore signs the
 * volume NEGATIVE. Adopting close into first_valid would skip bars the CPU
 * counts as down bars, shifting both EMAs. Registered as VolumeFiniteOnly.
 *
 * NON-FINITE VOLUME IS NOT A RESET. compute_vzo_value returns the PREVIOUS
 * ratio (or None) without touching either EMA, so the state survives the hole.
 *
 * ROUNDING: three mul_add sites, one per EMA and one for the noise filter,
 * each `beta.mul_add(state, alpha * x)` -- one pre-rounded product then ONE
 * fma. Reproduced with fma in the same operand order.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_VZO_LENGTH       14
#define NEO_VZO_NOISE_FILTER 4
/* get_intraday_smoothing default is TRUE (cpu_batch.rs:11555). */
#define NEO_VZO_INTRADAY     1

extern "C" __global__
void volume_zone_oscillator_neo_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;

    if (len <= 0 || first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const double alpha = 2.0 / ((double)NEO_VZO_LENGTH + 1.0);
    const double beta = 1.0 - alpha;
    const double smooth_alpha = 2.0 / ((double)NEO_VZO_NOISE_FILTER + 1.0);
    const double smooth_beta = 1.0 - smooth_alpha;

    for (int i = 0; i < first_valid; ++i) o[i] = NEO_F64_NAN;

    double prev_close = NEO_F64_NAN;
    double ema_direction = 0.0;
    double ema_total = 0.0;
    double smooth = 0.0;
    bool smooth_valid = false;

    for (int i = first_valid; i < len; ++i) {
        const double c = close[i];
        const double v = volume[i];

        bool have_raw;
        double raw = 0.0;

        if (!isfinite(v)) {
            // :288 -- the EMAs are NOT advanced; the previous ratio is reused.
            if (ema_total != 0.0) {
                have_raw = true;
                raw = 100.0 * ema_direction / ema_total;
            } else {
                have_raw = false;
            }
        } else {
            const double directed =
                (isfinite(c) && isfinite(prev_close) && c > prev_close) ? v : -v;
            ema_direction = fma(beta, ema_direction, alpha * directed);
            ema_total     = fma(beta, ema_total,     alpha * v);
            if (ema_total != 0.0) {
                have_raw = true;
                raw = 100.0 * ema_direction / ema_total;
            } else {
                have_raw = false;
            }
        }

        if (isfinite(c)) prev_close = c;

#if NEO_VZO_INTRADAY
        if (have_raw) {
            smooth = fma(smooth_beta, smooth, smooth_alpha * raw);
            smooth_valid = true;
            o[i] = smooth;
        } else if (smooth_valid) {
            o[i] = smooth;
        } else {
            o[i] = NEO_F64_NAN;
        }
#else
        o[i] = have_raw ? raw : NEO_F64_NAN;
#endif
    }
}
