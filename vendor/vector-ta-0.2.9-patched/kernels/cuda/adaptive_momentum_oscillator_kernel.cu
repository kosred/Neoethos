#include <cmath>
#include <cstddef>

static __device__ inline double amo_linreg_from_ring(
    const double* ring,
    int head,
    int period
) {
    double x_sum = 0.0;
    double x2_sum = 0.0;
    for (int i = 1; i <= period; ++i) {
        double x = static_cast<double>(i);
        x_sum += x;
        x2_sum += x * x;
    }

    double period_f = static_cast<double>(period);
    double denom = period_f * x2_sum - x_sum * x_sum;
    if (denom == 0.0 || !isfinite(denom)) {
        return NAN;
    }

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int i = 0; i < period; ++i) {
        double y = ring[(head + i) % period];
        y_sum += y;
        xy_sum += y * static_cast<double>(i + 1);
    }

    double b = (period_f * xy_sum - x_sum * y_sum) / denom;
    double a = (y_sum - b * x_sum) / period_f;
    return a + b * period_f;
}

extern "C" __global__ void adaptive_momentum_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smoothing_lengths,
    int rows,
    int max_length,
    int max_smoothing_length,
    double* __restrict__ raw_ring_buf,
    double* __restrict__ change_ring_buf,
    double* __restrict__ linreg_ring_buf,
    double* __restrict__ out_amo,
    double* __restrict__ out_ama
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    int smoothing_length = smoothing_lengths[row];
    double* raw_ring =
        raw_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* change_ring =
        change_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* linreg_ring =
        linreg_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smoothing_length);
    double* row_out_amo = out_amo + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_ama = out_ama + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_amo[i] = NAN;
        row_out_ama[i] = NAN;
    }

    if (length <= 0 || smoothing_length <= 0 || length > max_length ||
        smoothing_length > max_smoothing_length) {
        return;
    }

    int raw_head = 0;
    int raw_count = 0;
    int linreg_head = 0;
    bool linreg_filled = false;
    int change_head = 0;
    int change_count = 0;
    double change_sum = 0.0;
    bool avg_have_prev = false;
    double avg_prev = NAN;
    double ama_value = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];

        double raw = NAN;
        if (isfinite(value) && raw_count >= length) {
            bool valid = true;
            double best_abs = -1.0;
            double best_delta = NAN;
            for (int lag = 1; lag <= length; ++lag) {
                int hist_idx = (raw_head + length - lag) % length;
                double past = raw_ring[hist_idx];
                if (!isfinite(past)) {
                    valid = false;
                    break;
                }
                double delta = value - past;
                double abs_delta = fabs(delta);
                if (abs_delta >= best_abs) {
                    best_abs = abs_delta;
                    best_delta = delta;
                }
            }
            if (valid) {
                raw = best_delta;
            }
        }

        raw_ring[raw_head] = value;
        raw_head += 1;
        if (raw_head == length) {
            raw_head = 0;
        }
        if (raw_count < length) {
            raw_count += 1;
        }

        linreg_ring[linreg_head] = raw;
        linreg_head += 1;
        if (linreg_head == smoothing_length) {
            linreg_head = 0;
            linreg_filled = true;
        }

        double amo = NAN;
        if (linreg_filled) {
            amo = amo_linreg_from_ring(linreg_ring, linreg_head, smoothing_length);
        }

        double change = 0.0;
        if (avg_have_prev && isfinite(amo) && isfinite(avg_prev)) {
            change = fabs(amo - avg_prev);
        }
        double normalized_change = isfinite(change) ? change : 0.0;
        if (change_count < length) {
            change_ring[change_head] = normalized_change;
            change_sum += normalized_change;
            change_count += 1;
        } else {
            double old = change_ring[change_head];
            change_ring[change_head] = normalized_change;
            change_sum += normalized_change - old;
        }
        change_head += 1;
        if (change_head == length) {
            change_head = 0;
        }

        if (isfinite(amo) && change_sum > 0.0) {
            double efficiency_ratio = fabs(amo) / change_sum;
            double delta = efficiency_ratio * (amo - ama_value);
            if (isfinite(delta)) {
                ama_value += delta;
            }
        }

        avg_prev = amo;
        avg_have_prev = true;

        if (isfinite(amo)) {
            row_out_amo[i] = amo;
            row_out_ama[i] = ama_value;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE - adaptive_momentum_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/adaptive_momentum_oscillator.rs:573
 *             `compute_output_into_slice`, driving `Core::update` (:524),
 *             `AmoRawState::update` (:391) and `LinRegStream::update`
 *             (moving_averages/linreg.rs:887, dot at :900).
 *
 * COLUMN: `amo`. cpu_batch.rs:4206 maps output_id "value" onto
 * `OutputField::Amo`; `ama` is the other column and is computed here only
 * because the CPU threads it through the same state machine.
 *
 * PERIOD-INVARIANT. The CPU batch reads `length` (14) and `smoothing_length`
 * (9) and never `period`.
 *
 * FIRST-VALID IGNORED. `compute_output_into_slice` fills NaN and walks from
 * index 0 with a fresh core; `prepare_input`s `first_valid` only rejects an
 * all-NaN frame and sizes a validity check. NOTHING in this state machine is
 * reset by a non-finite bar - `AmoRawState::update` PUSHES the non-finite
 * value into its ring (:418) and the linreg ring likewise stores it (:888),
 * so a hole propagates for exactly `length` and `smoothing_length` bars and
 * then clears itself. That is the CPU behaviour and it is reproduced rather
 * than "fixed" with a reset.
 *
 * WINDOW BOUND: both rings are fixed-size per-thread arrays - 14 doubles for
 * the raw lag ring, 9 for the linreg ring, 14 for the efficiency-ratio change
 * ring - because `length` and `smoothing_length` are pinned at the CPU
 * defaults. No dynamic allocation.
 *
 * THE `>=` IN THE LAG SCAN IS LOAD-BEARING (:404): `abs_delta >= best_abs`
 * keeps the LAST maximal delta, not the first. A `>` would pick a different
 * bar whenever two lags tie, which is common on flat data.
 *
 * `LinRegStream::dot_ring` (:900) walks the ring OLDEST-FIRST from `head` and
 * accumulates `y_sum` and `xy_sum` in that order. The order is reproduced
 * exactly; a reversed or tree-reduced sum would round differently.
 *
 * NaN SEMANTICS: the raw scan uses an explicit `is_finite` guard rather than
 * a max, so there is no `f64::max` to mistranslate here. `change_sum > 0.0`
 * (:477) is false for NaN, which is the CPU behaviour.
 *
 * SEQUENTIAL, one thread per combo column: three chained stateful stages.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define AMO_NEO_LENGTH    14   /* adaptive_momentum_oscillator.rs DEFAULT_LENGTH */
#define AMO_NEO_SMOOTHING  9   /* DEFAULT_SMOOTHING_LENGTH */

extern "C" __global__
void adaptive_momentum_oscillator_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods; (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    const int L = AMO_NEO_LENGTH;
    const int S = AMO_NEO_SMOOTHING;

    /* AmoRawState (:354): ring seeded with NaN, head 0, count 0. */
    double raw_ring[AMO_NEO_LENGTH];
    #pragma unroll
    for (int k = 0; k < AMO_NEO_LENGTH; ++k) raw_ring[k] = NEO_F64_NAN;
    int raw_head = 0, raw_count = 0;

    /* LinRegStream (:851): buffer seeded with NaN; x_sum / x2_sum built by
       the same ascending loop the CPU uses (:871). */
    double lr_buf[AMO_NEO_SMOOTHING];
    #pragma unroll
    for (int k = 0; k < AMO_NEO_SMOOTHING; ++k) lr_buf[k] = NEO_F64_NAN;
    int  lr_head = 0;
    bool lr_filled = false;
    double x_sum = 0.0, x2_sum = 0.0;
    for (int k = 1; k <= S; ++k) {
        const double xi = (double)k;
        x_sum  += xi;
        x2_sum += xi * xi;
    }

    /* AdaptiveAverageState (:437): change ring of zeros, running sum. */
    double ch_ring[AMO_NEO_LENGTH];
    #pragma unroll
    for (int k = 0; k < AMO_NEO_LENGTH; ++k) ch_ring[k] = 0.0;
    int    ch_head = 0, ch_count = 0;
    double change_sum = 0.0;
    double avg_prev = NEO_F64_NAN;
    bool   avg_have_prev = false;
    double avg_value = 0.0;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];

        /* ---- stage 1: AmoRawState::update (:391) ---------------------- */
        double raw = NEO_F64_NAN;
        if (isfinite(value) && raw_count >= L) {
            double best_abs = -1.0;
            double best_delta = NEO_F64_NAN;
            bool   valid = true;
            for (int lag = 1; lag <= L; ++lag) {
                /* history_value (:373): (head + length - lag) % length */
                const int idx = (raw_head + L - lag) % L;
                const double past = raw_ring[idx];
                if (!isfinite(past)) { valid = false; break; }
                const double delta = value - past;
                const double abs_delta = fabs(delta);
                if (abs_delta >= best_abs) {   /* `>=` keeps the LAST maximum */
                    best_abs = abs_delta;
                    best_delta = delta;
                }
            }
            raw = valid ? best_delta : NEO_F64_NAN;
        }
        raw_ring[raw_head] = value;
        raw_head += 1; if (raw_head == L) raw_head = 0;
        if (raw_count < L) raw_count += 1;

        /* ---- stage 2: LinRegStream::update (:887) ---------------------- */
        lr_buf[lr_head] = raw;
        lr_head = (lr_head + 1) % S;
        if (!lr_filled && lr_head == 0) lr_filled = true;

        double amo = NEO_F64_NAN;
        if (lr_filled) {
            double y_sum = 0.0, xy_sum = 0.0;
            for (int k = 0; k < S; ++k) {
                const double y = lr_buf[(lr_head + k) % S];   /* oldest first */
                y_sum  += y;
                xy_sum += y * (double)(k + 1);
            }
            const double pf = (double)S;
            const double bd = 1.0 / (pf * x2_sum - x_sum * x_sum);
            const double b  = (pf * xy_sum - x_sum * y_sum) * bd;
            const double a  = (y_sum - b * x_sum) / pf;
            amo = a + b * pf;
        }

        /* ---- stage 3: AdaptiveAverageState::update (:469) -------------- */
        double change;
        if (avg_have_prev && isfinite(amo) && isfinite(avg_prev)) {
            change = fabs(amo - avg_prev);
        } else {
            change = 0.0;
        }
        {
            const double normalized = isfinite(change) ? change : 0.0;
            if (ch_count < L) {
                ch_ring[ch_head] = normalized;
                change_sum += normalized;
                ch_count += 1;
            } else {
                const double old = ch_ring[ch_head];
                ch_ring[ch_head] = normalized;
                change_sum += normalized - old;
            }
            ch_head += 1; if (ch_head == L) ch_head = 0;
        }
        if (isfinite(amo) && change_sum > 0.0) {
            const double efficiency_ratio = fabs(amo) / change_sum;
            const double delta = efficiency_ratio * (amo - avg_value);
            if (isfinite(delta)) avg_value += delta;
        }
        avg_prev = amo;
        avg_have_prev = true;

        /* `Core::update` (:524) emits only when `amo` is finite. */
        o[i] = isfinite(amo) ? amo : NEO_F64_NAN;
    }
}
