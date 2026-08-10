#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

constexpr int VELOCITY_MAX_LENGTH = 60;
constexpr int VELOCITY_MAX_SMOOTH_LENGTH = 9;

extern "C" __global__ void velocity_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooth_lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int smooth_length = smooth_lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (length < 2 || length > VELOCITY_MAX_LENGTH || smooth_length < 1 ||
        smooth_length > VELOCITY_MAX_SMOOTH_LENGTH) {
        return;
    }

    double harmonic = 0.0;
    for (int lag = 1; lag <= length; ++lag) {
        harmonic += 1.0 / static_cast<double>(lag);
    }
    double harmonic_over_length = harmonic / static_cast<double>(length);
    double smooth_denom = static_cast<double>(smooth_length * (smooth_length + 1) / 2);

    double history[VELOCITY_MAX_LENGTH];
    double raw_ring[VELOCITY_MAX_SMOOTH_LENGTH];
    for (int i = 0; i < VELOCITY_MAX_LENGTH; ++i) {
        history[i] = CUDART_NAN;
    }
    for (int i = 0; i < VELOCITY_MAX_SMOOTH_LENGTH; ++i) {
        raw_ring[i] = CUDART_NAN;
    }

    int history_head = 0;
    int history_count = 0;
    int raw_head = 0;
    int raw_count = 0;
    bool started = false;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!started) {
            if (isnan(value)) {
                continue;
            }
            started = true;
        }

        double raw;
        if (isfinite(value)) {
            double weighted_past = 0.0;
            for (int lag = 1; lag <= length; ++lag) {
                double past = 0.0;
                if (lag <= history_count) {
                    int idx = (history_head + length - lag) % length;
                    double hist_value = history[idx];
                    if (isfinite(hist_value)) {
                        past = hist_value;
                    }
                }
                weighted_past += past / static_cast<double>(lag);
            }
            raw = value * harmonic_over_length - weighted_past / static_cast<double>(length);
        } else {
            raw = CUDART_NAN;
        }

        history[history_head] = value;
        history_head += 1;
        if (history_head == length) {
            history_head = 0;
        }
        if (history_count < length) {
            history_count += 1;
        }

        raw_ring[raw_head] = raw;
        raw_head += 1;
        if (raw_head == smooth_length) {
            raw_head = 0;
        }
        if (raw_count < smooth_length) {
            raw_count += 1;
        }
        if (raw_count < smooth_length) {
            continue;
        }

        double weighted = 0.0;
        bool valid = true;
        for (int offset = 0; offset < smooth_length; ++offset) {
            int idx = (raw_head + offset) % smooth_length;
            double raw_value = raw_ring[idx];
            if (!isfinite(raw_value)) {
                valid = false;
                break;
            }
            weighted += static_cast<double>(offset + 1) * raw_value;
        }

        if (valid) {
            row[t] = weighted / smooth_denom;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — velocity                                    (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/velocity.rs
 *   :409 `compute_velocity_default_into`  <- the per-bar body reproduced here
 *   :236 `VelocityCore`                   (identical arithmetic, generic form)
 *   :519 `velocity_with_kernel`           warm = first_valid + smooth - 1
 *
 * WHY A SECOND f64 ENTRY POINT BESIDE `velocity_batch_f64`.
 * `velocity_batch_f64` above takes (data, len, lengths, smooth_lengths,
 * n_combos, out) -- the crate's own six-argument batch shape. The f64 LANE
 * launches ONE fixed signature (prices, n, periods, n_combos, first_valid,
 * out); handing it the six-argument kernel would slide `n_combos` into
 * `smooth_lengths` and read a period array as a pointer. So the lane gets its
 * own entry point in THIS file, beside the one the crate's wrapper calls.
 *
 * PERIOD-INVARIANT. `compute_velocity_batch` (cpu_batch.rs:4177) reads
 * `length` (default 21) and `smooth_length` (default 5) and NEVER `period`,
 * so a sweep of [7,21,50,...] produces one column repeated. The kernel emits
 * that same column on every row and `is_period_invariant` says so, rather
 * than inventing a mapping the CPU never performs.
 *
 * SEQUENTIAL, one thread per column: the raw value at bar i reads a 21-deep
 * ring of PAST sources and the smoother reads a 5-deep ring of PAST raws.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_VELOCITY_LENGTH 21
#define NEO_VELOCITY_SMOOTH 5

extern "C" __global__
void velocity_neo_batch_f64(const double* __restrict__ data,
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
    // Deliberately unread -- see PERIOD-INVARIANT above.
    (void)periods;

    if (len <= 0 || first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    // velocity.rs:466 -- `valid < smooth_length` is NotEnoughValidData, and the
    // CPU batch turns that into no column at all. A NaN column is the honest
    // device answer; a partial series would be a different indicator.
    if (len - first_valid < NEO_VELOCITY_SMOOTH) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    for (int i = 0; i < first_valid; ++i) o[i] = NEO_F64_NAN;

    // velocity.rs:410-414 -- ascending accumulation, then ONE divide.
    double harmonic = 0.0;
    for (int lag = 1; lag <= NEO_VELOCITY_LENGTH; ++lag) {
        harmonic += 1.0 / (double)lag;
    }
    const double harmonic_over_length = harmonic / (double)NEO_VELOCITY_LENGTH;

    double history[NEO_VELOCITY_LENGTH];
    for (int k = 0; k < NEO_VELOCITY_LENGTH; ++k) history[k] = NEO_F64_NAN;
    int history_head = 0;
    int history_count = 0;

    double raw_ring[NEO_VELOCITY_SMOOTH];
    for (int k = 0; k < NEO_VELOCITY_SMOOTH; ++k) raw_ring[k] = NEO_F64_NAN;
    int raw_head = 0;
    int raw_count = 0;

    // velocity.rs:421 -- (5 * 6 / 2) as f64, an exact integer.
    const double raw_denom =
        (double)((NEO_VELOCITY_SMOOTH * (NEO_VELOCITY_SMOOTH + 1)) / 2);

    for (int idx = first_valid; idx < len; ++idx) {
        const double value = data[idx];

        double raw;
        if (isfinite(value)) {
            double weighted_past = 0.0;
            for (int lag = 1; lag <= NEO_VELOCITY_LENGTH; ++lag) {
                double hist = 0.0;
                if (lag <= history_count) {
                    int hist_idx = history_head + NEO_VELOCITY_LENGTH - lag;
                    if (hist_idx >= NEO_VELOCITY_LENGTH) hist_idx -= NEO_VELOCITY_LENGTH;
                    const double past = history[hist_idx];
                    hist = isfinite(past) ? past : 0.0;
                }
                weighted_past += hist / (double)lag;
            }
            raw = value * harmonic_over_length
                - weighted_past / (double)NEO_VELOCITY_LENGTH;
        } else {
            raw = NEO_F64_NAN;
        }

        history[history_head] = value;
        history_head += 1;
        if (history_head == NEO_VELOCITY_LENGTH) history_head = 0;
        if (history_count < NEO_VELOCITY_LENGTH) history_count += 1;

        raw_ring[raw_head] = raw;
        raw_head += 1;
        if (raw_head == NEO_VELOCITY_SMOOTH) raw_head = 0;
        if (raw_count < NEO_VELOCITY_SMOOTH) raw_count += 1;

        if (raw_count < NEO_VELOCITY_SMOOTH) {
            o[idx] = NEO_F64_NAN;
            continue;
        }

        double weighted = 0.0;
        bool valid = true;
        for (int offset = 0; offset < NEO_VELOCITY_SMOOTH; ++offset) {
            int raw_idx = raw_head + offset;
            if (raw_idx >= NEO_VELOCITY_SMOOTH) raw_idx -= NEO_VELOCITY_SMOOTH;
            const double raw_value = raw_ring[raw_idx];
            if (!isfinite(raw_value)) { valid = false; break; }
            weighted += (double)(offset + 1) * raw_value;
        }
        o[idx] = valid ? (weighted / raw_denom) : NEO_F64_NAN;
    }
}
