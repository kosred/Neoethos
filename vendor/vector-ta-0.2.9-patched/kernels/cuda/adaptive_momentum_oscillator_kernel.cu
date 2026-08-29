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

    // Gate157: direct division produced 0xbec5c12413ce0b60 at fixture row 23,
    // while CPU LinRegStream's reciprocal-then-multiply authority produced
    // 0xbec5c12413ce0b68. The operation order is part of the exact f64 ABI.
    double bd = 1.0 / denom;
    double b = (period_f * xy_sum - x_sum * y_sum) * bd;
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
    int linreg_head = 0;
    bool linreg_filled = false;
    int change_head = 0;
    int change_count = 0;
    double change_sum = 0.0;
    bool avg_have_prev = false;
    double avg_prev = NAN;
    double ama_value = 0.0;

    for (int k = 0; k < length; ++k) {
        raw_ring[k] = NAN;
    }

    for (int i = 0; i < len; ++i) {
        double value = data[i];

        double max_momentum = 0.0;
        double selected_delta = 0.0;
        for (int lag = 1; lag <= length; ++lag) {
            int hist_idx = (raw_head + length - lag) % length;
            double past = raw_ring[hist_idx];
            double delta = value - past;
            double absolute_momentum = fabs(delta);
            if (isnan(max_momentum) || isnan(absolute_momentum)) {
                max_momentum = NAN;
            } else {
                max_momentum = fmax(max_momentum, absolute_momentum);
            }
            if (max_momentum == absolute_momentum) {
                selected_delta = delta;
            }
        }
        double raw = selected_delta;

        raw_ring[raw_head] = value;
        raw_head += 1;
        if (raw_head == length) {
            raw_head = 0;
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

        double change = avg_have_prev ? fabs(amo - avg_prev) : NAN;
        if (!isnan(change)) {
            if (change_count < length) {
                change_ring[change_head] = change;
                change_sum += change;
                change_count += 1;
            } else {
                double old = change_ring[change_head];
                change_ring[change_head] = change;
                change_sum += change - old;
            }
            change_head += 1;
            if (change_head == length) {
                change_head = 0;
            }
        }

        double rolling_sum = change_count == length ? change_sum : NAN;
        double efficiency_ratio = fabs(amo) / rolling_sum;
        double delta = efficiency_ratio * (amo - ama_value);
        if (isnan(delta)) {
            delta = 0.0;
        }
        ama_value += delta;

        avg_prev = amo;
        avg_have_prev = true;

        row_out_amo[i] = amo;
        row_out_ama[i] = ama_value;
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
 * CREATOR AUTHORITY:
 * https://pine-facade.tradingview.com/pine-facade/get/PUB%3B1763d63e649c4be4baf7fe86bee776b8/last
 * Raw momentum starts from zero and uses whatever history exists. A missing
 * lag poisons Pine `math.max`, but does not erase the delta selected before
 * that lag. This keeps raw momentum finite from the first bar, so only the
 * linear-regression window controls AMO warmup. `math.sum` ignores `na`
 * changes and waits for `length` non-na changes before updating the zero-
 * seeded AMA.
 *
 * TIES ARE LOAD-BEARING. Pine assigns `max = max(max, absolute_momentum)` and
 * then selects the delta when `max == absolute_momentum`, so every equal
 * maximum replaces the prior selection. The explicit equality reproduces
 * that ordering without relying on C `fmax`'s different NaN behavior.
 *
 * `LinRegStream::dot_ring` (:900) walks the ring OLDEST-FIRST from `head` and
 * accumulates `y_sum` and `xy_sum` in that order. The order is reproduced
 * exactly; a reversed or tree-reduced sum would round differently.
 *
 * NaN SEMANTICS: C/CUDA `fmax` ignores one NaN operand, while Pine `math.max`
 * returns `na`. Both CUDA routes therefore propagate NaN explicitly before
 * applying the equality selection. The adaptive recurrence implements Pine
 * `nz(delta)` by replacing NaN delta with exactly zero.
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
    int raw_head = 0;

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
        double max_momentum = 0.0;
        double selected_delta = 0.0;
        for (int lag = 1; lag <= L; ++lag) {
            /* history_value: (head + length - lag) % length */
            const int idx = (raw_head + L - lag) % L;
            const double past = raw_ring[idx];
            const double delta = value - past;
            const double absolute_momentum = fabs(delta);
            if (isnan(max_momentum) || isnan(absolute_momentum)) {
                max_momentum = NEO_F64_NAN;
            } else {
                max_momentum = fmax(max_momentum, absolute_momentum);
            }
            if (max_momentum == absolute_momentum) {
                selected_delta = delta;
            }
        }
        const double raw = selected_delta;
        raw_ring[raw_head] = value;
        raw_head += 1; if (raw_head == L) raw_head = 0;

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
        const double change = avg_have_prev ? fabs(amo - avg_prev) : NEO_F64_NAN;
        if (!isnan(change)) {
            if (ch_count < L) {
                ch_ring[ch_head] = change;
                change_sum += change;
                ch_count += 1;
            } else {
                const double old = ch_ring[ch_head];
                ch_ring[ch_head] = change;
                change_sum += change - old;
            }
            ch_head += 1; if (ch_head == L) ch_head = 0;
        }
        const double rolling_sum = ch_count == L ? change_sum : NEO_F64_NAN;
        const double efficiency_ratio = fabs(amo) / rolling_sum;
        double delta = efficiency_ratio * (amo - avg_value);
        if (isnan(delta)) delta = 0.0;
        avg_value += delta;
        avg_prev = amo;
        avg_have_prev = true;

        o[i] = amo;
    }
}
