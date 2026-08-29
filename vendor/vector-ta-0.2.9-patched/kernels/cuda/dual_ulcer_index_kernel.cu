#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void dual_ulcer_index_build_squares_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out_long_sq,
    double* __restrict__ out_short_sq
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row_long_sq = out_long_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_short_sq = out_short_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_long_sq[t] = CUDART_NAN;
        row_short_sq[t] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    int close_count = 0;

    for (int t = 0; t < len; ++t) {
        double close = data[t];
        if (!isfinite(close) || close <= 0.0) {
            close_count = 0;
            continue;
        }

        if (close_count < period) {
            close_count += 1;
        }
        if (close_count < period) {
            continue;
        }

        int window_start = t + 1 - period;
        double highest = -CUDART_INF;
        double lowest = CUDART_INF;
        bool valid = true;

        for (int i = window_start; i <= t; ++i) {
            double value = data[i];
            if (!isfinite(value) || value <= 0.0) {
                valid = false;
                break;
            }
            if (value > highest) {
                highest = value;
            }
            if (value < lowest) {
                lowest = value;
            }
        }

        if (!valid) {
            close_count = 0;
            continue;
        }

        double long_ret = 100.0 * (close - highest) / highest;
        double short_ret = 100.0 * (close - lowest) / lowest;
        row_long_sq[t] = long_ret * long_ret;
        row_short_sq[t] = short_ret * short_ret;
    }
}

extern "C" __global__ void dual_ulcer_index_finalize_f64(
    const double* __restrict__ long_sq,
    const double* __restrict__ short_sq,
    int len,
    const int* __restrict__ periods,
    const double* __restrict__ thresholds,
    int auto_threshold,
    int n_combos,
    double* __restrict__ out_long_ulcer,
    double* __restrict__ out_short_ulcer,
    double* __restrict__ out_threshold
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double custom_threshold = thresholds[combo_idx];
    const double* row_long_sq = long_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    const double* row_short_sq = short_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_long = out_long_ulcer + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_short = out_short_ulcer + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_threshold = out_threshold + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_long[t] = CUDART_NAN;
        row_short[t] = CUDART_NAN;
        row_threshold[t] = CUDART_NAN;
    }

    if (period <= 0 || !isfinite(custom_threshold) || custom_threshold < 0.0) {
        return;
    }

    int sq_count = 0;
    double long_sq_sum = 0.0;
    double short_sq_sum = 0.0;
    double diff_sum = 0.0;
    int diff_count = 0;

    for (int t = 0; t < len; ++t) {
        double current_long_sq = row_long_sq[t];
        double current_short_sq = row_short_sq[t];
        if (!isfinite(current_long_sq) || !isfinite(current_short_sq)) {
            sq_count = 0;
            long_sq_sum = 0.0;
            short_sq_sum = 0.0;
            continue;
        }

        if (sq_count == period) {
            long_sq_sum -= row_long_sq[t - period];
            short_sq_sum -= row_short_sq[t - period];
        } else {
            sq_count += 1;
        }

        long_sq_sum += current_long_sq;
        short_sq_sum += current_short_sq;

        if (sq_count < period) {
            continue;
        }

        double denom = static_cast<double>(period);
        double long_ulcer = sqrt(long_sq_sum) / denom;
        double short_ulcer = sqrt(short_sq_sum) / denom;
        double threshold_value;

        if (auto_threshold != 0) {
            double diff = fabs(long_ulcer - short_ulcer);
            diff_sum += diff;
            diff_count += 1;
            threshold_value = diff_sum / static_cast<double>(diff_count);
        } else {
            threshold_value = custom_threshold;
        }

        row_long[t] = long_ulcer;
        row_short[t] = short_ulcer;
        row_threshold[t] = threshold_value;
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                        dual_ulcer_index
 * ---------------------------------------------------------------------------
 * CPU reference: `compute_dual_ulcer_index_row` and its operation-identical
 * selected-row sibling in src/indicators/dual_ulcer_index.rs, reached from the
 * canonical batch dispatcher once per output and parameter tuple.
 *
 * `period` IS the swept parameter (cpu_batch.rs:6723, default 5).
 * `auto_threshold` is true and `threshold` is 0.1 by default. The registry and
 * CPU dispatcher expose exactly long_ulcer/short_ulcer/threshold; the retired
 * value/uulcer/dulcer spellings are not production identities.
 *
 * WHY THE EXISTING ENTRY POINTS COULD NOT BE REUSED. This file already carries
 * `dual_ulcer_index_build_squares_f64` and `dual_ulcer_index_finalize_f64`, a
 * public TWO-PASS pair retained for its standalone ABI. Production cannot use
 * it because it allocates intermediate matrices and launches twice. The typed
 * shared-session ABI below fuses the work into one thread body and one launch;
 * the preserved generic primary delegates to that same row authority.
 *
 * SEQUENTIAL, one thread per column: `long_sq_sum` is a SLIDING sum, updated as
 * `sum -= leaving; sum += arriving` in that order (:646, :653), and its value
 * therefore depends on every bar before it. A fresh window sum per bar would be
 * a different double.
 *
 * NO PER-THREAD RING AND NO DEQUE, so no `max_period` bound and NEVER-OOM by
 * construction. The CPU keeps `long_sq_ring`/`short_sq_ring` of `period`
 * doubles and two monotonic deques (:577-581). Neither is needed on the card:
 *   * `highest`/`lowest` are an exact max/min over the window, and max/min have
 *     no accumulation order to preserve, so rescanning the `period` bars gives
 *     the identical double the deque front would have given. The deque only
 *     ever holds bars since the last reset, and the emit condition is
 *     `close_count >= period` -- i.e. the whole window is valid -- so the two
 *     windows coincide;
 *   * the long and short values LEAVING their sliding sums are the two squared
 *     returns at bar `t - period`. They are recomputed rather than stored,
 *     which gives the same doubles because the operation sequence is fixed.
 * The cost is a second O(period) rescan per bar. The benefit is that an
 * oversized period is not refused by name and no local array exists to size.
 *
 * `is_valid_price` is `is_finite() && > 0.0` (:389-391) -- a zero or negative
 * close RESETS the run (:591-599), it is not merely skipped, because the ulcer
 * ratio divides by `highest`.
 *
 * FIRST-VALID IS `Ignored`, and that is read off the CPU rather than assumed:
 * `compute_dual_ulcer_index_selected_row` fills the row with NaN (:574) and
 * then iterates `for i in 0..len` (:589). It never consults a first-valid
 * index; the `warmup` computed at :701 belongs to a DIFFERENT entry point
 * (`dual_ulcer_index_with_kernel`) that allocates its own prefix. So this
 * kernel starts at bar 0, exactly as the CPU does.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* dual_ulcer_index.rs:389-391. */
__device__ __forceinline__
static bool neo_dui_valid(double v) { return isfinite(v) && v > 0.0; }

/* The exact long/short squared returns at bar `t` for a window of `period`
 * bars ending there, dual_ulcer_index.rs:619-642. Returns false when the
 * complete window is not valid. */
__device__ __forceinline__
static bool neo_dui_squares(const double* __restrict__ data,
                            int t, int period,
                            double* out_long_sq, double* out_short_sq)
{
    const int window_start = t + 1 - period;
    if (window_start < 0) return false;
    double close = data[t];
    if (!neo_dui_valid(close)) return false;

    double highest = data[window_start];
    double lowest = highest;
    if (!neo_dui_valid(highest)) return false;
    for (int i = window_start + 1; i <= t; ++i) {
        const double v = data[i];
        if (!neo_dui_valid(v)) return false;
        if (v > highest) highest = v;
        if (v < lowest) lowest = v;
    }

    const double long_ret = 100.0 * (close - highest) / highest;
    const double short_ret = 100.0 * (close - lowest) / lowest;
    *out_long_sq = long_ret * long_ret;
    *out_short_sq = short_ret * short_ret;
    return true;
}

/* One production arithmetic authority for both the canonical triple-output
 * launch and the preserved long-ulcer primary ABI. Null output pointers mean
 * "do not materialize this matrix"; every scalar operation still executes in
 * the same order as compute_dual_ulcer_index_selected_row. */
__device__ __forceinline__
static void dual_ulcer_index_row_f64(
    const double* __restrict__ data,
    int n,
    int period,
    bool auto_threshold,
    double custom_threshold,
    double* __restrict__ out_long_ulcer,
    double* __restrict__ out_short_ulcer,
    double* __restrict__ out_threshold)
{
    for (int i = 0; i < n; ++i) {
        if (out_long_ulcer != nullptr) out_long_ulcer[i] = NEO_F64_NAN;
        if (out_short_ulcer != nullptr) out_short_ulcer[i] = NEO_F64_NAN;
        if (out_threshold != nullptr) out_threshold[i] = NEO_F64_NAN;
    }
    if (period <= 0 || period > n || !isfinite(custom_threshold) ||
        custom_threshold < 0.0) {
        return;
    }

    int close_count = 0;
    int sq_count = 0;
    double long_sq_sum = 0.0;
    double short_sq_sum = 0.0;
    double diff_sum = 0.0;
    int diff_count = 0;

    for (int t = 0; t < n; ++t) {
        const double close = data[t];
        if (!neo_dui_valid(close)) {
            close_count = 0;
            sq_count = 0;
            long_sq_sum = 0.0;
            short_sq_sum = 0.0;
            continue;
        }

        if (close_count < period) close_count += 1;
        if (close_count < period) continue;

        double long_sq;
        double short_sq;
        if (!neo_dui_squares(data, t, period, &long_sq, &short_sq)) {
            close_count = 0;
            sq_count = 0;
            long_sq_sum = 0.0;
            short_sq_sum = 0.0;
            continue;
        }

        if (sq_count == period) {
            double leaving_long_sq;
            double leaving_short_sq;
            if (!neo_dui_squares(data, t - period, period,
                                 &leaving_long_sq, &leaving_short_sq)) {
                return;
            }
            long_sq_sum -= leaving_long_sq;
            short_sq_sum -= leaving_short_sq;
        } else {
            sq_count += 1;
        }
        long_sq_sum += long_sq;
        short_sq_sum += short_sq;

        if (sq_count < period) continue;

        const double denom = (double)period;
        const double long_ulcer = sqrt(long_sq_sum) / denom;
        const double short_ulcer = sqrt(short_sq_sum) / denom;
        const double diff = fabs(long_ulcer - short_ulcer);
        double threshold_value;
        if (auto_threshold) {
            diff_sum += diff;
            diff_count += 1;
            threshold_value = diff_sum / (double)diff_count;
        } else {
            threshold_value = custom_threshold;
        }

        if (out_long_ulcer != nullptr) out_long_ulcer[t] = long_ulcer;
        if (out_short_ulcer != nullptr) out_short_ulcer[t] = short_ulcer;
        if (out_threshold != nullptr) out_threshold[t] = threshold_value;
    }
}

extern "C" __global__
void dual_ulcer_index_all_outputs_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    const int* __restrict__ auto_thresholds,
    const double* __restrict__ thresholds,
    int n_combos,
    double* __restrict__ out_long_ulcer,
    double* __restrict__ out_short_ulcer,
    double* __restrict__ out_threshold)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    const size_t base = (size_t)combo * (size_t)n;
    dual_ulcer_index_row_f64(
        data,
        n,
        periods[combo],
        auto_thresholds[combo] != 0,
        thresholds[combo],
        out_long_ulcer + base,
        out_short_ulcer + base,
        out_threshold + base);
}

extern "C" __global__
void dual_ulcer_index_neo_batch_f64(const double* __restrict__ data,
                                    int n,
                                    const int* __restrict__ periods,
                                    int n_combos,
                                    int first_valid,
                                    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)first_valid;
    dual_ulcer_index_row_f64(
        data,
        n,
        periods[combo],
        true,
        0.1,
        out + (size_t)combo * (size_t)n,
        nullptr,
        nullptr);
}
