#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool vw_rsi_valid_pair(double close, double volume) {
    return isfinite(close) && isfinite(volume);
}

__device__ inline double vw_rsi_from_components(double avg_up, double avg_down) {
    double denom = avg_up + avg_down;
    if (denom == 0.0) {
        return 50.0;
    }
    return 100.0 * avg_up / denom;
}

extern "C" __global__ void volume_weighted_rsi_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    double inv_period = 1.0 / static_cast<double>(period);
    double beta = 1.0 - inv_period;
    double prev_close = CUDART_NAN;
    bool has_prev = false;
    int seeded = 0;
    double sum_up = 0.0;
    double sum_down = 0.0;
    double avg_up = 0.0;
    double avg_down = 0.0;

    for (int i = 0; i < len; ++i) {
        if (!vw_rsi_valid_pair(close[i], volume[i])) {
            prev_close = CUDART_NAN;
            has_prev = false;
            seeded = 0;
            sum_up = 0.0;
            sum_down = 0.0;
            avg_up = 0.0;
            avg_down = 0.0;
            continue;
        }

        double up = 0.0;
        double down = 0.0;
        if (has_prev) {
            if (close[i] > prev_close) {
                up = volume[i];
            } else if (close[i] < prev_close) {
                down = volume[i];
            }
        }

        prev_close = close[i];
        has_prev = true;

        if (seeded < period) {
            sum_up += up;
            sum_down += down;
            seeded += 1;
            if (seeded < period) {
                continue;
            }
            avg_up = sum_up * inv_period;
            avg_down = sum_down * inv_period;
            row[i] = vw_rsi_from_components(avg_up, avg_down);
            continue;
        }

        avg_up = beta * avg_up + inv_period * up;
        avg_down = beta * avg_down + inv_period * down;
        row[i] = vw_rsi_from_components(avg_up, avg_down);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — volume_weighted_rsi                         (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/volume_weighted_rsi.rs
 *   :389 compute_row            <- the per-bar body reproduced here
 *   :307 rsi_from_components    denom == 0 -> 50.0, else 100*up/denom
 *   :302 is_valid_pair          BOTH close and volume finite
 *   :354 validate_common
 *
 * PERIOD-SWEPT, and one of the few in this closer set that genuinely is:
 * cpu_batch.rs:6787 calls combo_periods(.., "period", 14), so every row of a
 * sweep is a different column.
 *
 * WHY A SECOND f64 ENTRY POINT. volume_weighted_rsi_batch_f64 above already
 * carries the crate six-argument batch shape (close, volume, len, periods,
 * n_combos, out) and is bar-parallel over combos in the same way -- but the
 * f64 lane launches (a, b, n, periods, n_combos, first_valid, out), i.e. with
 * first_valid between n_combos and out. Reusing the existing symbol would
 * slide first_valid into the out pointer. Hence a second, lane-shaped entry.
 *
 * FIRST-VALID IGNORED: compute_row walks from index 0 over the WHOLE series
 * and an invalid (close, volume) pair RESETS the Wilder state rather than
 * being skipped. There is no NaN prefix to align.
 *
 * ROUNDING: avg = avg.mul_add(beta, inv_period * up) is ONE fma over a
 * pre-rounded product. fma(avg, beta, inv_period * up) reproduces exactly
 * that; writing avg*beta + inv_period*up would add a rounding per bar and
 * compound through the recursion.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void volume_weighted_rsi_neo_batch_f64(
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
    const int period = periods[combo];
    (void)first_valid;  // see FIRST-VALID IGNORED above

    if (len <= 0) return;
    // validate_common: period == 0 || period > len is an Err, i.e. no column.
    if (period <= 0 || period > len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const double inv_period = 1.0 / (double)period;
    const double beta = 1.0 - inv_period;

    double prev_close = NEO_F64_NAN;
    bool has_prev = false;
    int seeded = 0;
    double sum_up = 0.0, sum_down = 0.0;
    double avg_up = 0.0, avg_down = 0.0;

    for (int i = 0; i < len; ++i) {
        const double c = close[i];
        const double vol = volume[i];
        if (!isfinite(c) || !isfinite(vol)) {
            prev_close = NEO_F64_NAN;
            has_prev = false;
            seeded = 0;
            sum_up = 0.0; sum_down = 0.0;
            avg_up = 0.0; avg_down = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        double up = 0.0, down = 0.0;
        if (has_prev) {
            if (c > prev_close)      { up = vol;  down = 0.0; }
            else if (c < prev_close) { up = 0.0;  down = vol; }
        }
        prev_close = c;
        has_prev = true;

        if (seeded < period) {
            sum_up += up;
            sum_down += down;
            seeded += 1;
            if (seeded < period) {
                o[i] = NEO_F64_NAN;
                continue;
            }
            avg_up = sum_up * inv_period;
            avg_down = sum_down * inv_period;
            const double denom = avg_up + avg_down;
            o[i] = (denom == 0.0) ? 50.0 : (100.0 * avg_up / denom);
            continue;
        }

        avg_up   = fma(avg_up,   beta, inv_period * up);
        avg_down = fma(avg_down, beta, inv_period * down);
        const double denom = avg_up + avg_down;
        o[i] = (denom == 0.0) ? 50.0 : (100.0 * avg_up / denom);
    }
}
