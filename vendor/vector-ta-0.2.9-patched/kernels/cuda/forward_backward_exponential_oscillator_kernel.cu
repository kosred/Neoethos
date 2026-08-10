#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline double ring_at(
    const double* ring,
    int head,
    int period,
    int idx
) {
    int pos = head + idx;
    if (pos >= period) {
        pos -= period;
    }
    return ring[pos];
}

__device__ inline double compute_forward_backward_value(
    const double* ema1_ring,
    int ema1_head,
    int length,
    double alpha
) {
    double current = ring_at(ema1_ring, ema1_head, length, length - 1);
    double ema2 = current;
    double prev = ema2;
    double num = 0.0;
    double den = 0.0;

    for (int idx = length - 2; idx >= 0; --idx) {
        double value = ring_at(ema1_ring, ema1_head, length, idx);
        ema2 += alpha * (value - ema2);
        double dt = prev - ema2;
        num += dt;
        den += fabs(dt);
        prev = ema2;
    }

    if (den == 0.0) {
        return CUDART_NAN;
    }
    return num / den * 50.0 + 50.0;
}
}

extern "C" __global__ void forward_backward_exponential_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooths,
    int n_combos,
    int max_length,
    double* __restrict__ ema1_buffer,
    double* __restrict__ diff_buffer,
    double* __restrict__ out_forward_backward,
    double* __restrict__ out_backward,
    double* __restrict__ out_histogram
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int smooth = smooths[combo_idx];
    double* ema1_ring =
        ema1_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* diff_ring =
        diff_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* row_forward_backward =
        out_forward_backward + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_backward =
        out_backward + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_histogram =
        out_histogram + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_forward_backward[i] = CUDART_NAN;
        row_backward[i] = CUDART_NAN;
        row_histogram[i] = CUDART_NAN;
    }

    if (length <= 0 || length > max_length || smooth <= 0) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(smooth) + 1.0);
    double beta = 1.0 - alpha;

    bool have_ema1_state = false;
    bool have_ema2_state = false;
    bool have_prev_ema2 = false;
    double ema1_state = CUDART_NAN;
    double ema2_state = CUDART_NAN;
    double prev_ema2 = CUDART_NAN;

    int ema1_count = 0;
    int ema1_head = 0;

    int diff_count = 0;
    int diff_head = 0;
    double diff_sum = 0.0;
    double diff_abs_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            have_ema1_state = false;
            have_ema2_state = false;
            have_prev_ema2 = false;
            ema1_state = CUDART_NAN;
            ema2_state = CUDART_NAN;
            prev_ema2 = CUDART_NAN;
            ema1_count = 0;
            ema1_head = 0;
            diff_count = 0;
            diff_head = 0;
            diff_sum = 0.0;
            diff_abs_sum = 0.0;
            continue;
        }

        if (have_ema1_state) {
            ema1_state = alpha * value + beta * ema1_state;
        } else {
            ema1_state = value;
            have_ema1_state = true;
        }

        if (ema1_count < length) {
            ema1_ring[ema1_count] = ema1_state;
            ema1_count += 1;
        } else {
            ema1_ring[ema1_head] = ema1_state;
            ema1_head += 1;
            if (ema1_head == length) {
                ema1_head = 0;
            }
        }

        if (ema1_count == length) {
            row_forward_backward[i] =
                compute_forward_backward_value(ema1_ring, ema1_head, length, alpha);
        }

        if (have_ema2_state) {
            ema2_state = alpha * ema1_state + beta * ema2_state;
        } else {
            ema2_state = ema1_state;
            have_ema2_state = true;
        }

        if (have_prev_ema2) {
            double diff = ema2_state - prev_ema2;
            if (diff_count < length) {
                diff_ring[diff_count] = diff;
                diff_count += 1;
                diff_sum += diff;
                diff_abs_sum += fabs(diff);
            } else {
                double removed = diff_ring[diff_head];
                diff_sum -= removed;
                diff_abs_sum -= fabs(removed);
                diff_ring[diff_head] = diff;
                diff_sum += diff;
                diff_abs_sum += fabs(diff);
                diff_head += 1;
                if (diff_head == length) {
                    diff_head = 0;
                }
            }

            if (diff_count == length && diff_abs_sum != 0.0) {
                double backward = diff_sum / diff_abs_sum * 50.0 + 50.0;
                row_backward[i] = backward;
                double forward_backward = row_forward_backward[i];
                if (isfinite(forward_backward)) {
                    row_histogram[i] = (forward_backward - backward) * 0.25 + 50.0;
                }
            }
        }

        prev_ema2 = ema2_state;
        have_prev_ema2 = true;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — forward_backward_exponential_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/forward_backward_exponential_oscillator.rs:488
 *   `compute_into_slices`, with `ema_step` (:314),
 *   `compute_forward_backward_value` (:332) and `RollingDiffWindow` (:276).
 *   The driver walks EVERY bar from 0 and CLEARS both windows and both EMA
 *   states on a non-finite value (:550-554), so `first_valid` is not read.
 *
 * Column: output_id "value" / "forward_backward" / "fb" -> `out.forward_backward`
 *   (cpu_batch.rs:15993).
 *
 * PERIOD-INVARIANT: `compute_forward_backward_exponential_oscillator_batch`
 *   (cpu_batch.rs:15968, :15974) reads `length` (20) and `smooth` (10) and
 *   NEVER `period`. `alpha = 2 / (smooth + 1)` (:481), i.e. 2/11 — note that
 *   the alpha comes from SMOOTH, not from LENGTH, so a kernel that derived it
 *   from the window size would be smoothing at the wrong rate.
 *
 * The forward-backward value is a BACKWARD pass over the last `length` EMA1
 *   values, re-run from scratch at every bar (:332-354): it seeds `ema2` at the
 *   most recent value and walks the window in REVERSE, accumulating the signed
 *   and absolute increments. That is O(length) per bar and it is not a
 *   recurrence that can be slid forward — the seed changes every bar. Kept as
 *   written.
 *
 * Shape: ONE THREAD PER COLUMN. `ema1` carries across bars, and the window is a
 *   per-thread ring of `length` doubles.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:15968/:15974. Per-thread rings, so the bound
 * belongs to the compiled kernel. */
#define NEO_FBEO_LENGTH 20
#define NEO_FBEO_SMOOTH 10

extern "C" __global__
void forward_backward_exponential_oscillator_neo_batch_f64(const double* __restrict__ data,
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

    const int    L     = NEO_FBEO_LENGTH;
    const double alpha = 2.0 / ((double)NEO_FBEO_SMOOTH + 1.0);

    /* ema1_window: a FIFO of the last L ema1 values, oldest at index `head`. */
    double win[NEO_FBEO_LENGTH];
    int win_head = 0, win_len = 0;

    bool   e1_set = false;
    double e1 = 0.0;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        double fb = NEO_F64_NAN;

        if (isfinite(value)) {
            /* ema_step (:314): alpha * value + (1 - alpha) * last */
            const double ema1 = e1_set ? (alpha * value + (1.0 - alpha) * e1) : value;
            e1 = ema1; e1_set = true;

            /* push_window (:324): drop the front once full, then push back. */
            if (win_len == L) {
                win[win_head] = ema1;
                win_head = win_head + 1; if (win_head == L) win_head = 0;
            } else {
                win[(win_head + win_len) % NEO_FBEO_LENGTH] = ema1;
                ++win_len;
            }

            if (win_len == L) {
                /* compute_forward_backward_value (:332): seed at the BACK, walk
                 * the window in reverse skipping the first (the seed itself). */
                const double current = win[(win_head + win_len - 1) % NEO_FBEO_LENGTH];
                double ema2 = current;
                double prev = ema2;
                double num = 0.0, den = 0.0;
                for (int k = win_len - 2; k >= 0; --k) {
                    const double v = win[(win_head + k) % NEO_FBEO_LENGTH];
                    ema2 += alpha * (v - ema2);
                    const double dt = prev - ema2;
                    num += dt;
                    den += fabs(dt);
                    prev = ema2;
                }
                if (den != 0.0) {
                    const double v = num / den * 50.0 + 50.0;
                    if (isfinite(v)) fb = v;
                }
            }
        } else {
            e1_set = false; e1 = 0.0;
            win_head = 0; win_len = 0;
        }

        o[i] = fb;
    }
}
