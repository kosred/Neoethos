#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void absolute_strength_index_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ signal_lengths,
    int n_combos,
    double* __restrict__ out_oscillator,
    double* __restrict__ out_signal,
    double* __restrict__ out_histogram
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int ema_length = ema_lengths[combo_idx];
    int signal_length = signal_lengths[combo_idx];
    double* row_oscillator =
        out_oscillator + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal =
        out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_histogram =
        out_histogram + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = CUDART_NAN;
        row_signal[i] = CUDART_NAN;
        row_histogram[i] = CUDART_NAN;
    }

    if (ema_length <= 0 || signal_length <= 1) {
        return;
    }

    double ema_alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);
    double signal_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);
    double signal_beta = 1.0 - signal_alpha;

    bool have_prev_close = false;
    bool have_ema_abssi = false;
    double prev_close = CUDART_NAN;
    double a = 0.0;
    double m = 0.0;
    double d = 0.0;
    double ema_abssi = CUDART_NAN;
    double mt = 0.0;
    double ut = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            have_prev_close = false;
            have_ema_abssi = false;
            prev_close = CUDART_NAN;
            a = 0.0;
            m = 0.0;
            d = 0.0;
            ema_abssi = CUDART_NAN;
            mt = 0.0;
            ut = 0.0;
            continue;
        }

        double abssi = 1.0;
        if (have_prev_close) {
            if (value > prev_close) {
                if (prev_close != 0.0) {
                    a += value / prev_close - 1.0;
                }
            } else if (value < prev_close) {
                if (value != 0.0) {
                    d += prev_close / value - 1.0;
                }
            } else {
                m += 0.1;
            }

            double denom = d + m * 0.5;
            if (denom == 0.0) {
                abssi = 1.0;
            } else {
                abssi = 1.0 - 1.0 / (1.0 + (a + m * 0.5) / denom);
            }
        }

        prev_close = value;
        have_prev_close = true;

        if (have_ema_abssi) {
            ema_abssi = ema_alpha * abssi + (1.0 - ema_alpha) * ema_abssi;
        } else {
            ema_abssi = abssi;
            have_ema_abssi = true;
        }

        double oscillator = abssi - ema_abssi;
        mt = signal_alpha * oscillator + signal_beta * mt;
        ut = signal_alpha * mt + signal_beta * ut;

        double signal = ((2.0 - signal_alpha) * mt - ut) / signal_beta;
        double histogram = oscillator - signal;

        row_oscillator[i] = oscillator;
        row_signal[i] = signal;
        row_histogram[i] = histogram;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — absolute_strength_index_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/absolute_strength_index_oscillator.rs:521
 *             `absolute_strength_index_oscillator_row_field_from_slice`,
 *             driving `AbsoluteStrengthIndexOscillatorStream::update` (:297).
 *
 * COLUMN: the lane emits `oscillator`. cpu_batch.rs:8531 maps output_id
 * "value" (and "indicator") onto `OutputField::Oscillator`, so that is the
 * column the CPU produces for a `value` request — never `signal`, never
 * `histogram`.
 *
 * PERIOD-INVARIANT. `compute_absolute_strength_index_oscillator_batch`
 * (cpu_batch.rs:8525) reads `ema_length` (default 21) and `signal_length`
 * (default 34) and NEVER `period`, so a sweep of [7,21,50,100,200] yields
 * five identical CPU columns and this kernel emits five identical rows.
 *
 * FIRST-VALID IGNORED. The CPU row function constructs a fresh stream and
 * walks from index 0; `prepare`'s `first` is never handed to it. A non-finite
 * bar RESETS the whole stream and emits NaN, so the series restarts after
 * every hole rather than carrying state across it. Registered as
 * `F64FirstValidRule::Ignored` so the table states that rather than declaring
 * a warmup the kernel does not honour.
 *
 * `m += 1.0 / OSL_DIVISOR` with OSL_DIVISOR = 10.0 (:33, :313). `1.0/10.0`
 * and the literal `0.1` are the same binary64 value — the correctly rounded
 * quotient IS the nearest double to one tenth — so the literal is used and
 * the divide is not repeated per bar.
 *
 * SEQUENTIAL, one thread per combo column: `ema_abssi`, `mt` and `ut` are
 * each a first-order recurrence and `a`/`d`/`m` are running sums, so no
 * bar-parallel or scan reformulation reproduces the CPU's rounding.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void absolute_strength_index_oscillator_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods; (void)first_valid;   // period-invariant; first-valid ignored

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    // Defaults from absolute_strength_index_oscillator.rs:31-32, resolved by
    // `resolve_params` (:453) into the two alphas at :469-470.
    const double ema_alpha    = 2.0 / (21.0 + 1.0);
    const double signal_alpha = 2.0 / (34.0 + 1.0);

    bool   have_prev  = false;
    bool   have_ema   = false;
    double prev_close = 0.0;
    double a = 0.0, m = 0.0, d = 0.0;
    double ema_abssi = 0.0;
    double mt = 0.0, ut = 0.0;

    for (int i = 0; i < len; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {
            // `Stream::reset` (:299) — every carried scalar returns to its
            // constructed value, not merely `prev_close`.
            have_prev = false; have_ema = false;
            prev_close = 0.0;
            a = 0.0; m = 0.0; d = 0.0;
            ema_abssi = 0.0;
            mt = 0.0; ut = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        double abssi;
        if (have_prev) {
            if (v > prev_close) {
                if (prev_close != 0.0) a += v / prev_close - 1.0;
            } else if (v < prev_close) {
                if (v != 0.0) d += prev_close / v - 1.0;
            } else {
                m += 0.1;               // 1.0 / OSL_DIVISOR, OSL_DIVISOR = 10.0
            }
            const double denom = d + m * 0.5;
            abssi = (denom == 0.0) ? 1.0
                                   : 1.0 - 1.0 / (1.0 + (a + m * 0.5) / denom);
        } else {
            abssi = 1.0;
        }
        prev_close = v;
        have_prev = true;

        if (have_ema) {
            ema_abssi = ema_alpha * abssi + (1.0 - ema_alpha) * ema_abssi;
        } else {
            ema_abssi = abssi;
            have_ema = true;
        }

        const double oscillator = abssi - ema_abssi;
        mt = signal_alpha * oscillator + (1.0 - signal_alpha) * mt;
        ut = signal_alpha * mt         + (1.0 - signal_alpha) * ut;

        o[i] = oscillator;
    }
}
