#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double ema_seeded_update(
    int period,
    double alpha,
    double beta,
    int* count,
    double* mean,
    bool* filled,
    double value,
    bool* produced
) {
    *count += 1;
    int current = *count;
    if (current == 1) {
        *mean = value;
    } else if (current <= period) {
        double inv = 1.0 / static_cast<double>(current);
        *mean = (value - *mean) * inv + *mean;
    } else {
        *mean = beta * (*mean) + alpha * value;
    }
    if (!*filled && current >= period) {
        *filled = true;
    }
    *produced = *filled;
    return *mean;
}

extern "C" __global__ void trend_direction_force_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    int max_norm_window,
    int* __restrict__ deque_indices,
    double* __restrict__ deque_values,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_norm_window <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    int* dq_idx = deque_indices + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_norm_window);
    double* dq_val = deque_values + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_norm_window);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length <= 0) {
        return;
    }

    int half = length / 2;
    if (half < 1) {
        half = 1;
    }
    int norm_window = length * 3;
    if (norm_window < 1 || norm_window > max_norm_window) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(half) + 1.0);
    double beta = 1.0 - alpha;
    int ema1_count = 0;
    int ema2_count = 0;
    double ema1_mean = CUDART_NAN;
    double ema2_mean = CUDART_NAN;
    bool ema1_filled = false;
    bool ema2_filled = false;
    double prev_ema1 = CUDART_NAN;
    double prev_ema2 = CUDART_NAN;
    bool have_prev_emas = false;
    int next_index = 0;
    int dq_head = 0;
    int dq_size = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            ema1_count = 0;
            ema2_count = 0;
            ema1_mean = CUDART_NAN;
            ema2_mean = CUDART_NAN;
            ema1_filled = false;
            ema2_filled = false;
            prev_ema1 = CUDART_NAN;
            prev_ema2 = CUDART_NAN;
            have_prev_emas = false;
            next_index = 0;
            dq_head = 0;
            dq_size = 0;
            continue;
        }

        int idx = next_index;
        next_index += 1;

        bool ema1_ready = false;
        double ema1 = ema_seeded_update(
            half,
            alpha,
            beta,
            &ema1_count,
            &ema1_mean,
            &ema1_filled,
            value * 1000.0,
            &ema1_ready
        );
        if (!ema1_ready) {
            continue;
        }

        bool ema2_ready = false;
        double ema2 = ema_seeded_update(
            half,
            alpha,
            beta,
            &ema2_count,
            &ema2_mean,
            &ema2_filled,
            ema1,
            &ema2_ready
        );
        if (!ema2_ready) {
            continue;
        }

        if (!have_prev_emas) {
            prev_ema1 = ema1;
            prev_ema2 = ema2;
            have_prev_emas = true;
            continue;
        }

        double ema_diff_avg = ((ema1 - prev_ema1) + (ema2 - prev_ema2)) * 0.5;
        double tdf = fabs(ema1 - ema2) * ema_diff_avg * ema_diff_avg * ema_diff_avg;
        prev_ema1 = ema1;
        prev_ema2 = ema2;

        double abs_tdf = fabs(tdf);
        int window_start = idx + 1 - norm_window;
        if (window_start < 0) {
            window_start = 0;
        }
        while (dq_size > 0 && dq_idx[dq_head] < window_start) {
            dq_head += 1;
            if (dq_head == norm_window) {
                dq_head = 0;
            }
            dq_size -= 1;
        }

        while (dq_size > 0) {
            int back_pos = dq_head + dq_size - 1;
            if (back_pos >= norm_window) {
                back_pos -= norm_window;
            }
            if (dq_val[back_pos] <= abs_tdf) {
                dq_size -= 1;
            } else {
                break;
            }
        }
        int insert_pos = dq_head + dq_size;
        if (insert_pos >= norm_window) {
            insert_pos -= norm_window;
        }
        dq_idx[insert_pos] = idx;
        dq_val[insert_pos] = abs_tdf;
        dq_size += 1;

        double max_abs = dq_size > 0 ? dq_val[dq_head] : 0.0;
        row[i] = max_abs == 0.0 ? 0.0 : tdf / max_abs;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — trend_direction_force_index                 (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/trend_direction_force_index.rs
 *   :470 `compute_row_with_buffers`   <- the per-bar body reproduced here
 *   :246 `EmaSeededStream`            (mean-seeded EMA, TWO fma sites)
 *   :394 `half_length`  = max(length/2, 1)
 *   :404 `normalization_window` = max(length*3, 1)
 *   :454 `compute_row`  -- length == 10 takes the fixed [31] buffers
 *
 * PERIOD-INVARIANT (cpu_batch.rs:8269 reads `length`, default 10), so the
 * deque cap is the CPU's own fixed 31 and lives in a per-thread array.
 *
 * FIRST-VALID IGNORED: the CPU walks from index 0 and a non-finite bar RESETS
 * every accumulator (including the deque and `next_index`) rather than being
 * skipped, so there is no warmup prefix to align against.
 *
 * ROUNDING. `EmaSeededStream::update` is TWO different fused forms:
 *   count <= period : mean = (value - mean).mul_add(inv, mean)   -- ONE rounding
 *   count >  period : mean = beta.mul_add(mean, alpha * value)   -- alpha*value
 *                                                                  rounds, then
 *                                                                  ONE fma
 * Reproduced with `fma` on both, in the same operand order. Writing either as
 * `a*b + c*d` would add a rounding and shift every later bar of the recursion.
 *
 * `powi(3)` is `(x*x)*x`, not `exp(3*log x)`.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_TDFI_LENGTH      10
#define NEO_TDFI_NORM_WINDOW 30   /* length * 3 */
#define NEO_TDFI_CAP         31   /* norm_window + 1 */

extern "C" __global__
void trend_direction_force_index_neo_batch_f64(
    const double* __restrict__ data,
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
    (void)first_valid;

    if (len <= 0) return;

    const int half = (NEO_TDFI_LENGTH / 2) > 1 ? (NEO_TDFI_LENGTH / 2) : 1;
    const double alpha = 2.0 / ((double)half + 1.0);
    const double beta  = 1.0 - alpha;

    // Two independent EmaSeededStream instances.
    int   c1 = 0, c2 = 0;
    double m1 = NEO_F64_NAN, m2 = NEO_F64_NAN;
    bool  f1 = false, f2 = false;

    int next_index = 0;
    double prev_ema1 = NEO_F64_NAN, prev_ema2 = NEO_F64_NAN;
    bool have_prev_emas = false;

    int    max_idx[NEO_TDFI_CAP];
    double max_vals[NEO_TDFI_CAP];
    int head = 0, tail = 0, count = 0;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            c1 = 0; m1 = NEO_F64_NAN; f1 = false;
            c2 = 0; m2 = NEO_F64_NAN; f2 = false;
            next_index = 0;
            prev_ema1 = NEO_F64_NAN;
            prev_ema2 = NEO_F64_NAN;
            have_prev_emas = false;
            head = 0; tail = 0; count = 0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const int idx = next_index;
        next_index += 1;

        // ema1_stream.update(value * 1000.0)
        {
            const double x = value * 1000.0;
            c1 += 1;
            if (c1 == 1) {
                m1 = x;
            } else if (c1 <= half) {
                const double inv = 1.0 / (double)c1;
                m1 = fma(x - m1, inv, m1);
            } else {
                m1 = fma(beta, m1, alpha * x);
            }
            if (!f1 && c1 >= half) f1 = true;
        }
        if (!f1) { o[i] = NEO_F64_NAN; continue; }
        const double ema1 = m1;

        // ema2_stream.update(ema1)
        {
            c2 += 1;
            if (c2 == 1) {
                m2 = ema1;
            } else if (c2 <= half) {
                const double inv = 1.0 / (double)c2;
                m2 = fma(ema1 - m2, inv, m2);
            } else {
                m2 = fma(beta, m2, alpha * ema1);
            }
            if (!f2 && c2 >= half) f2 = true;
        }
        if (!f2) { o[i] = NEO_F64_NAN; continue; }
        const double ema2 = m2;

        if (!have_prev_emas) {
            prev_ema1 = ema1;
            prev_ema2 = ema2;
            have_prev_emas = true;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double ema_diff_avg = ((ema1 - prev_ema1) + (ema2 - prev_ema2)) * 0.5;
        // `.powi(3)` == (x*x)*x.
        const double cube = (ema_diff_avg * ema_diff_avg) * ema_diff_avg;
        const double tdf = fabs(ema1 - ema2) * cube;
        prev_ema1 = ema1;
        prev_ema2 = ema2;

        const double abs_tdf = fabs(tdf);
        while (count > 0) {
            const int back = (tail == 0) ? (NEO_TDFI_CAP - 1) : (tail - 1);
            if (max_vals[back] <= abs_tdf) {
                tail = back;
                count -= 1;
            } else {
                break;
            }
        }

        max_idx[tail] = idx;
        max_vals[tail] = abs_tdf;
        tail += 1;
        if (tail == NEO_TDFI_CAP) tail = 0;
        count += 1;

        // idx.saturating_add(1).saturating_sub(norm_window)
        int window_start = idx + 1 - NEO_TDFI_NORM_WINDOW;
        if (window_start < 0) window_start = 0;
        while (count > 0 && max_idx[head] < window_start) {
            head += 1;
            if (head == NEO_TDFI_CAP) head = 0;
            count -= 1;
        }

        const double max_abs = (count == 0) ? 0.0 : max_vals[head];
        o[i] = (max_abs == 0.0) ? 0.0 : (tdf / max_abs);
    }
}
