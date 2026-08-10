#include <cmath>
#include <cstddef>

static __device__ inline double adaptive_macd_corr_sq_window(
    const double* data,
    int end_idx,
    int length
) {
    int start = end_idx + 1 - length;
    double sum_x = static_cast<double>((length - 1) * length) * 0.5;
    double sum_x2 = static_cast<double>((length - 1) * length * (2 * length - 1)) / 6.0;
    double n = static_cast<double>(length);
    double denom_x = n * sum_x2 - sum_x * sum_x;
    double sum_y = 0.0;
    double sum_y2 = 0.0;
    double sum_xy = 0.0;

    for (int i = 0; i < length; ++i) {
        double value = data[start + i];
        if (!isfinite(value)) {
            return NAN;
        }
        sum_y += value;
        sum_y2 += value * value;
        sum_xy += static_cast<double>(i) * value;
    }

    double denom_y = n * sum_y2 - sum_y * sum_y;
    if (denom_y <= 1e-12) {
        return 0.0;
    }
    double num = n * sum_xy - sum_x * sum_y;
    double corr_sq = (num * num) / (denom_x * denom_y);
    if (corr_sq < 0.0) {
        return 0.0;
    }
    if (corr_sq > 1.0) {
        return 1.0;
    }
    return corr_sq;
}

extern "C" __global__ void adaptive_macd_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const int* __restrict__ signal_periods,
    int rows,
    double* __restrict__ out_macd,
    double* __restrict__ out_signal,
    double* __restrict__ out_hist
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    int fast_period = fast_periods[row];
    int slow_period = slow_periods[row];
    int signal_period = signal_periods[row];

    double* row_out_macd = out_macd + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_hist = out_hist + static_cast<size_t>(row) * static_cast<size_t>(len);

    if (length < 2 ||
        fast_period < 2 ||
        slow_period < 2 ||
        signal_period < 2 ||
        length > len ||
        fast_period > len ||
        slow_period > len ||
        signal_period > len) {
        for (int i = 0; i < len; ++i) {
            row_out_macd[i] = NAN;
            row_out_signal[i] = NAN;
            row_out_hist[i] = NAN;
        }
        return;
    }

    double a1 = 2.0 / (static_cast<double>(fast_period) + 1.0);
    double a2 = 2.0 / (static_cast<double>(slow_period) + 1.0);
    double delta_coeff = a1 - a2;
    double recur_coeff = 2.0 - a1 - a2;
    double trend_coeff = (1.0 - a1) * (1.0 - a2);
    double cycle_coeff = (1.0 - a1) / (1.0 - a2);
    double alpha = 2.0 / (static_cast<double>(signal_period) + 1.0);
    double beta = 1.0 - alpha;

    bool signal_started = false;
    int signal_count = 0;
    double signal_sum = 0.0;
    double signal_value = NAN;
    double prev_close = NAN;
    double prev_macd1 = NAN;
    double prev_macd2 = NAN;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        double current_macd = NAN;
        if (isfinite(value) && i + 1 >= length) {
            bool valid_window = true;
            for (int j = i + 1 - length; j <= i; ++j) {
                if (!isfinite(data[j])) {
                    valid_window = false;
                    break;
                }
            }
            if (valid_window && isfinite(prev_close)) {
                double corr_sq = adaptive_macd_corr_sq_window(data, i, length);
                if (isfinite(corr_sq)) {
                    double r2 = 0.5 * corr_sq + 0.5;
                    double k = r2 * trend_coeff + (1.0 - r2) * cycle_coeff;
                    double prev1 = isfinite(prev_macd1) ? prev_macd1 : 0.0;
                    double prev2 = isfinite(prev_macd2) ? prev_macd2 : 0.0;
                    current_macd =
                        (value - prev_close) * delta_coeff + recur_coeff * prev1 - k * prev2;
                }
            }
        }

        prev_close = value;
        prev_macd2 = prev_macd1;
        prev_macd1 = current_macd;

        double signal = NAN;
        if (isfinite(current_macd)) {
            if (!signal_started) {
                signal_started = true;
                signal_count = 1;
                signal_sum = current_macd;
                signal_value = current_macd;
            } else if (signal_count < signal_period) {
                signal_count += 1;
                signal_sum += current_macd;
                signal_value = signal_sum / static_cast<double>(signal_count);
            } else {
                signal_value = beta * signal_value + alpha * current_macd;
            }
            signal = signal_value;
        } else if (signal_started) {
            signal = signal_value;
        }

        row_out_macd[i] = current_macd;
        row_out_signal[i] = signal;
        row_out_hist[i] =
            (isfinite(current_macd) && isfinite(signal)) ? current_macd - signal : NAN;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE - adaptive_macd
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/adaptive_macd.rs:789 `compute_output_row`,
 *             driving `AdaptiveMacdState::update` (:700), the rolling
 *             correlation at :580/:591 and `EmaLikeState::update` (:648).
 *
 * COLUMN: `macd`. cpu_batch.rs:4800 maps output_id "value" onto
 * `OutputField::Macd`. `signal` and `hist` are other columns.
 *
 * PERIOD-INVARIANT. The CPU batch reads `length` (20), `fast_period` (10),
 * `slow_period` (20) and `signal_period` (9) and NEVER `period`, so a sweep
 * of [7,21,50,100,200] gets five identical CPU columns and this kernel emits
 * five identical rows.
 *
 * FIRST-VALID IGNORED. `compute_output_row` builds a fresh state and walks
 * from index 0. `prepare_input`s `warmup` only sizes a NaN prefix in the
 * Vec-returning path; the into-slice path the batch takes fills NaN and then
 * overwrites every bar the state emits. A non-finite bar resets ONLY the
 * correlation ring (:593) - `prev_close`, `prev_macd1`, `prev_macd2` and the
 * signal EMA are deliberately NOT reset, and that asymmetry is reproduced.
 *
 * WINDOW BOUND: the correlation ring is `length` doubles and `length` is
 * pinned at the CPU default 20, so the ring is a fixed 20-element per-thread
 * array. No dynamic allocation, and nothing for the sweep to overflow.
 *
 * ROUNDING: three `mul_add`s are reproduced as explicit `fma` - `denom_y`
 * (:582), `num` (:586) and the EMA step `beta.mul_add(value, alpha * value)`
 * (:667). Each is ONE rounding on the CPU and one here. `denom_x` (:566) is
 * likewise an fma, computed once per thread from the same integer-derived
 * `sum_x` / `sum_x2`.
 *
 * EPSILON: CORR_EPSILON = 1e-12 (:66) guards the variance denominator. It is
 * an f64-sized guard already - not an f32 machine epsilon - and is carried
 * across unchanged so the zero-variance branch fires on exactly the CPU
 * bars.
 *
 * `.clamp(0.0, 1.0)` on `corr_sq` is the Rust `f64::clamp`: `< min -> min`,
 * `> max -> max`, otherwise SELF, so a NaN stays NaN. Written as the same two
 * comparisons rather than `fmin(fmax(..))`, which would return a bound.
 *
 * SEQUENTIAL, one thread per combo column: a two-lag MACD recurrence over a
 * sliding correlation whose sums are updated incrementally in the CPU order.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define AMACD_NEO_LENGTH 20      /* adaptive_macd.rs:61 DEFAULT_LENGTH */
#define AMACD_NEO_SIGNAL 9       /* adaptive_macd.rs:64 DEFAULT_SIGNAL_PERIOD */

extern "C" __global__
void adaptive_macd_neo_batch_f64(
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

    const int    L   = AMACD_NEO_LENGTH;
    const double n   = (double)L;
    const double CORR_EPSILON = 1e-12;

    /* build_spec (:497-506) with fast_period = 10, slow_period = 20. */
    const double a1 = 2.0 / (10.0 + 1.0);
    const double a2 = 2.0 / (20.0 + 1.0);
    const double delta_coeff = a1 - a2;
    const double recur_coeff = 2.0 - a1 - a2;
    const double trend_coeff = (1.0 - a1) * (1.0 - a2);
    const double cycle_coeff = (1.0 - a1) / (1.0 - a2);

    /* RollingCorrelationState::new (:553-567). The two sums are formed from
       the same integer expressions the CPU uses, so they are exact. */
    const double sum_x  = (double)((L - 1) * L) * 0.5;
    const double sum_x2 = (double)((L - 1) * L * (2 * L - 1)) / 6.0;
    const double denom_x = fma(n, sum_x2, -(sum_x * sum_x));

    double ring[AMACD_NEO_LENGTH];
    #pragma unroll
    for (int k = 0; k < AMACD_NEO_LENGTH; ++k) ring[k] = 0.0;
    int    head = 0, count = 0;
    double sum_y = 0.0, sum_y2 = 0.0, sum_xy = 0.0;

    double prev_close = NEO_F64_NAN;
    double prev_macd1 = NEO_F64_NAN;
    double prev_macd2 = NEO_F64_NAN;

    /* EmaLikeState for the signal line. Carried because `hist` and `signal`
       share this state with `macd` on the CPU; the signal value itself is not
       emitted, but advancing it keeps the state machine identical bar for
       bar should this file later serve those columns. */
    const double s_alpha = 2.0 / ((double)AMACD_NEO_SIGNAL + 1.0);
    const double s_beta  = 1.0 - s_alpha;
    int    s_count = 0;
    double s_sum = 0.0, s_value = NEO_F64_NAN;
    bool   s_started = false;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];

        double current_macd;
        if (isfinite(value)) {
            /* RollingCorrelationState::push (:591). */
            bool   have_corr = false;
            double corr_sq = 0.0;
            if (count < L) {
                const double idx = (double)count;
                ring[head] = value;
                head += 1; if (head == L) head = 0;
                count += 1;
                sum_y  += value;
                sum_y2 += value * value;
                sum_xy += idx * value;
                have_corr = (count == L);
            } else {
                const double old = ring[head];
                const double prev_sum_y  = sum_y;
                const double prev_sum_xy = sum_xy;
                ring[head] = value;
                head += 1; if (head == L) head = 0;
                sum_y  = prev_sum_y - old + value;
                sum_y2 = sum_y2 - old * old + value * value;
                sum_xy = prev_sum_xy - (prev_sum_y - old) + (n - 1.0) * value;
                have_corr = true;
            }

            if (have_corr) {
                const double denom_y = fma(n, sum_y2, -(sum_y * sum_y));
                if (denom_y <= CORR_EPSILON) {
                    corr_sq = 0.0;
                } else {
                    const double num = fma(n, sum_xy, -(sum_x * sum_y));
                    double r = (num * num) / (denom_x * denom_y);
                    if (r < 0.0)      r = 0.0;       /* f64::clamp, NaN stays NaN */
                    else if (r > 1.0) r = 1.0;
                    corr_sq = r;
                }
            }

            if (isfinite(prev_close) && have_corr) {
                const double r2 = 0.5 * corr_sq + 0.5;
                const double k  = r2 * trend_coeff + (1.0 - r2) * cycle_coeff;
                const double p1 = isfinite(prev_macd1) ? prev_macd1 : 0.0;
                const double p2 = isfinite(prev_macd2) ? prev_macd2 : 0.0;
                current_macd = (value - prev_close) * delta_coeff
                               + recur_coeff * p1
                               - k * p2;
            } else {
                current_macd = NEO_F64_NAN;
            }
        } else {
            /* reset() touches the ring bookkeeping ONLY (:571-577). */
            head = 0; count = 0;
            sum_y = 0.0; sum_y2 = 0.0; sum_xy = 0.0;
            current_macd = NEO_F64_NAN;
        }

        prev_close = value;
        prev_macd2 = prev_macd1;
        prev_macd1 = current_macd;

        /* EmaLikeState::update (:648) fed with `current_macd`. */
        if (!isfinite(current_macd)) {
            /* returns the held value if started, else None - no state change */
        } else if (!s_started) {
            s_started = true; s_count = 1; s_sum = current_macd; s_value = current_macd;
        } else if (s_count < AMACD_NEO_SIGNAL) {
            s_count += 1; s_sum += current_macd;
            s_value = s_sum / (double)s_count;
        } else {
            s_value = fma(s_beta, s_value, s_alpha * current_macd);
        }

        o[i] = current_macd;
    }
}
