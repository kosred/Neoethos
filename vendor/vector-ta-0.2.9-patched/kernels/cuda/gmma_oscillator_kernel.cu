#include <cmath>
#include <cstdint>

__device__ __forceinline__ int gmma_fast_count(int mode_flag) {
    return mode_flag == 0 ? 6 : 11;
}

__device__ __forceinline__ int gmma_slow_count(int mode_flag) {
    return mode_flag == 0 ? 6 : 16;
}

__device__ __forceinline__ int gmma_fast_period(int mode_flag, int idx) {
    if (mode_flag == 0) {
        switch (idx) {
            case 0: return 3;
            case 1: return 5;
            case 2: return 8;
            case 3: return 10;
            case 4: return 12;
            default: return 15;
        }
    }

    switch (idx) {
        case 0: return 3;
        case 1: return 5;
        case 2: return 7;
        case 3: return 9;
        case 4: return 11;
        case 5: return 13;
        case 6: return 15;
        case 7: return 17;
        case 8: return 19;
        case 9: return 21;
        default: return 23;
    }
}

__device__ __forceinline__ int gmma_slow_period(int mode_flag, int idx) {
    if (mode_flag == 0) {
        switch (idx) {
            case 0: return 30;
            case 1: return 35;
            case 2: return 40;
            case 3: return 45;
            case 4: return 50;
            default: return 60;
        }
    }

    switch (idx) {
        case 0: return 25;
        case 1: return 28;
        case 2: return 31;
        case 3: return 34;
        case 4: return 37;
        case 5: return 40;
        case 6: return 43;
        case 7: return 46;
        case 8: return 49;
        case 9: return 52;
        case 10: return 55;
        case 11: return 58;
        case 12: return 61;
        case 13: return 64;
        case 14: return 67;
        default: return 70;
    }
}

extern "C" __global__ void gmma_oscillator_batch_f64(
    const double* data,
    int len,
    int mode_flag,
    int multiplier,
    const int* smooth_lengths,
    const int* signal_lengths,
    int rows,
    int max_smooth_length,
    double* raw_windows,
    double* out_oscillator,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int smooth_length = smooth_lengths[row];
    int signal_length = signal_lengths[row];
    if (smooth_length <= 0 || signal_length <= 0 || multiplier <= 0 || max_smooth_length <= 0) {
        return;
    }

    const double nan = NAN;
    double* row_window = raw_windows + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    double fast_state[16] = {0.0};
    double slow_state[16] = {0.0};
    double fast_alpha[16] = {0.0};
    double slow_alpha[16] = {0.0};
    int fast_count = gmma_fast_count(mode_flag);
    int slow_count = gmma_slow_count(mode_flag);
    for (int i = 0; i < fast_count; ++i) {
        int effective = gmma_fast_period(mode_flag, i) * multiplier;
        if (effective < 1) {
            effective = 1;
        }
        fast_alpha[i] = 2.0 / (static_cast<double>(effective) + 1.0);
    }
    for (int i = 0; i < slow_count; ++i) {
        int effective = gmma_slow_period(mode_flag, i) * multiplier;
        if (effective < 1) {
            effective = 1;
        }
        slow_alpha[i] = 2.0 / (static_cast<double>(effective) + 1.0);
    }

    double smooth_sum = 0.0;
    int smooth_count = 0;
    int smooth_index = 0;
    double signal_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);
    double signal_state = 0.0;
    bool signal_seeded = false;
    bool initialized = false;

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = nan;
        row_signal[i] = nan;

        double value = data[i];
        if (!isfinite(value)) {
            for (int j = 0; j < fast_count; ++j) {
                fast_state[j] = 0.0;
            }
            for (int j = 0; j < slow_count; ++j) {
                slow_state[j] = 0.0;
            }
            smooth_sum = 0.0;
            smooth_count = 0;
            smooth_index = 0;
            signal_state = 0.0;
            signal_seeded = false;
            initialized = false;
            continue;
        }

        if (!initialized) {
            for (int j = 0; j < fast_count; ++j) {
                fast_state[j] = value;
            }
            for (int j = 0; j < slow_count; ++j) {
                slow_state[j] = value;
            }
            initialized = true;
        } else {
            for (int j = 0; j < fast_count; ++j) {
                double prev = fast_state[j];
                fast_state[j] = prev + fast_alpha[j] * (value - prev);
            }
            for (int j = 0; j < slow_count; ++j) {
                double prev = slow_state[j];
                slow_state[j] = prev + slow_alpha[j] * (value - prev);
            }
        }

        double fast_sum = 0.0;
        double slow_sum = 0.0;
        for (int j = 0; j < fast_count; ++j) {
            fast_sum += fast_state[j];
        }
        for (int j = 0; j < slow_count; ++j) {
            slow_sum += slow_state[j];
        }

        double fast_avg = fast_sum / static_cast<double>(fast_count);
        double slow_avg = slow_sum / static_cast<double>(slow_count);
        if (!isfinite(slow_avg) || slow_avg == 0.0) {
            for (int j = 0; j < fast_count; ++j) {
                fast_state[j] = 0.0;
            }
            for (int j = 0; j < slow_count; ++j) {
                slow_state[j] = 0.0;
            }
            smooth_sum = 0.0;
            smooth_count = 0;
            smooth_index = 0;
            signal_state = 0.0;
            signal_seeded = false;
            initialized = false;
            continue;
        }

        double raw = ((fast_avg - slow_avg) / slow_avg) * 100.0;
        if (!isfinite(raw)) {
            for (int j = 0; j < fast_count; ++j) {
                fast_state[j] = 0.0;
            }
            for (int j = 0; j < slow_count; ++j) {
                slow_state[j] = 0.0;
            }
            smooth_sum = 0.0;
            smooth_count = 0;
            smooth_index = 0;
            signal_state = 0.0;
            signal_seeded = false;
            initialized = false;
            continue;
        }

        double signal = raw;
        if (signal_seeded) {
            signal_state += signal_alpha * (raw - signal_state);
            signal = signal_state;
        } else {
            signal_state = raw;
            signal_seeded = true;
        }
        row_signal[i] = signal;

        if (smooth_length == 1) {
            row_oscillator[i] = raw;
            continue;
        }

        if (smooth_count < smooth_length) {
            row_window[smooth_index] = raw;
            smooth_sum += raw;
            smooth_count += 1;
            smooth_index += 1;
            if (smooth_index == smooth_length) {
                smooth_index = 0;
            }
            if (smooth_count == smooth_length) {
                row_oscillator[i] = smooth_sum / static_cast<double>(smooth_length);
            }
            continue;
        }

        double old = row_window[smooth_index];
        row_window[smooth_index] = raw;
        smooth_sum += raw - old;
        smooth_index += 1;
        if (smooth_index == smooth_length) {
            smooth_index = 0;
        }
        row_oscillator[i] = smooth_sum / static_cast<double>(smooth_length);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — gmma_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/gmma_oscillator.rs:561 `GmmaCore::update`, driven
 *   bar by bar from index 0; a non-finite value RESETS the whole fan (:562-565)
 *   and so does a zero or non-finite slow average (:596-599), so `first_valid`
 *   is not read.
 *
 * Column: output_id "value" / "oscillator" -> the SMA of the raw spread
 *   (cpu_batch.rs:15610). With the default `smooth_length` of 1 that SMA is the
 *   identity (:540) — the raw value passes through — but the branch is kept so
 *   a non-default smooth stays correct rather than silently ignored.
 *
 * PERIOD-INVARIANT: `compute_gmma_oscillator_batch` (cpu_batch.rs:7577-7580)
 *   reads `gmma_type` ("guppy"), `smooth_length` (1), `signal_length` (13) and
 *   `anchor_minutes` (0) — and NEVER `period`. With anchor_minutes = 0,
 *   `resolve_multiplier` (:404) returns 1, so the fan periods are the literal
 *   guppy tables.
 *
 * The fan is TWELVE EMAs — six fast, six slow — all seeded to the first valid
 *   value on the same bar (:568-573), not warmed independently. The averages
 *   are plain sums divided by the count, accumulated in table order (:586-595);
 *   that order is preserved because six additions of numbers of similar
 *   magnitude still round differently under a different association.
 *
 * Shape: ONE THREAD PER COLUMN. Twelve recurrences plus a signal EMA carry
 *   across every bar.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* GUPPY_FAST_PERIODS / GUPPY_SLOW_PERIODS, gmma_oscillator.rs:28-29. */
#define NEO_GMMA_FAST_COUNT 6
#define NEO_GMMA_SLOW_COUNT 6
/* Defaults from cpu_batch.rs:7578-7579. */
#define NEO_GMMA_SMOOTH_LENGTH 1
#define NEO_GMMA_SIGNAL_LENGTH 13

extern "C" __global__
void gmma_oscillator_neo_batch_f64(const double* __restrict__ data,
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

    const int fast_periods[NEO_GMMA_FAST_COUNT] = { 3, 5, 8, 10, 12, 15 };
    const int slow_periods[NEO_GMMA_SLOW_COUNT] = { 30, 35, 40, 45, 50, 60 };
    const int multiplier = 1;   /* resolve_multiplier(0, _) -> 1 (:408) */

    double fast_alpha[NEO_GMMA_FAST_COUNT], slow_alpha[NEO_GMMA_SLOW_COUNT];
    for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) {
        int eff = fast_periods[i] * multiplier; if (eff < 1) eff = 1;
        fast_alpha[i] = 2.0 / ((double)eff + 1.0);
    }
    for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) {
        int eff = slow_periods[i] * multiplier; if (eff < 1) eff = 1;
        slow_alpha[i] = 2.0 / ((double)eff + 1.0);
    }
    const double signal_alpha = 2.0 / ((double)NEO_GMMA_SIGNAL_LENGTH + 1.0);
    (void)signal_alpha;   /* the `signal` column, not this one -- kept so the
                           * state machine below reads the same as the CPU */

    double fast_state[NEO_GMMA_FAST_COUNT];
    double slow_state[NEO_GMMA_SLOW_COUNT];
    for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) fast_state[i] = 0.0;
    for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) slow_state[i] = 0.0;
    bool initialized = false;

    /* update_sma (:539) */
    const int SL = NEO_GMMA_SMOOTH_LENGTH;
    double smooth_window[NEO_GMMA_SMOOTH_LENGTH];
    for (int i = 0; i < SL; ++i) smooth_window[i] = 0.0;
    double smooth_sum = 0.0;
    int    smooth_count = 0, smooth_index = 0;

    bool   signal_set = false;
    double signal_state = 0.0;

    for (int bar = 0; bar < n; ++bar) {
        const double value = data[bar];
        if (!isfinite(value)) {
            for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) fast_state[i] = 0.0;
            for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) slow_state[i] = 0.0;
            for (int i = 0; i < SL; ++i) smooth_window[i] = 0.0;
            smooth_sum = 0.0; smooth_count = 0; smooth_index = 0;
            signal_set = false; signal_state = 0.0;
            initialized = false;
            o[bar] = NEO_F64_NAN;
            continue;
        }

        if (!initialized) {
            for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) fast_state[i] = value;
            for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) slow_state[i] = value;
            initialized = true;
        } else {
            for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) {
                const double prev = fast_state[i];
                fast_state[i] = prev + fast_alpha[i] * (value - prev);
            }
            for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) {
                const double prev = slow_state[i];
                slow_state[i] = prev + slow_alpha[i] * (value - prev);
            }
        }

        double fast_sum = 0.0;
        for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) fast_sum += fast_state[i];
        double slow_sum = 0.0;
        for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) slow_sum += slow_state[i];
        const double fast_avg = fast_sum / (double)NEO_GMMA_FAST_COUNT;
        const double slow_avg = slow_sum / (double)NEO_GMMA_SLOW_COUNT;

        if (!isfinite(slow_avg) || slow_avg == 0.0) {
            for (int i = 0; i < NEO_GMMA_FAST_COUNT; ++i) fast_state[i] = 0.0;
            for (int i = 0; i < NEO_GMMA_SLOW_COUNT; ++i) slow_state[i] = 0.0;
            for (int i = 0; i < SL; ++i) smooth_window[i] = 0.0;
            smooth_sum = 0.0; smooth_count = 0; smooth_index = 0;
            signal_set = false; signal_state = 0.0;
            initialized = false;
            o[bar] = NEO_F64_NAN;
            continue;
        }

        const double raw = ((fast_avg - slow_avg) / slow_avg) * 100.0;

        /* signal EMA -- not this column, but its state must advance in step. */
        if (signal_set) signal_state = signal_state + signal_alpha * (raw - signal_state);
        else { signal_state = raw; signal_set = true; }

        double oscillator;
        if (SL == 1) {
            oscillator = raw;
        } else if (smooth_count < SL) {
            smooth_window[smooth_index] = raw;
            smooth_sum += raw;
            ++smooth_count;
            smooth_index = (smooth_index + 1) % NEO_GMMA_SMOOTH_LENGTH;
            oscillator = (smooth_count < SL) ? NEO_F64_NAN
                                             : (smooth_sum / (double)SL);
        } else {
            const double old = smooth_window[smooth_index];
            smooth_window[smooth_index] = raw;
            smooth_sum += raw - old;
            smooth_index = (smooth_index + 1) % NEO_GMMA_SMOOTH_LENGTH;
            oscillator = smooth_sum / (double)SL;
        }

        o[bar] = oscillator;
    }
}
