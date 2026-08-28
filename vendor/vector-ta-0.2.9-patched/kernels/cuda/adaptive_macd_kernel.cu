#include <cmath>
#include <cstddef>

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

/* One formula implementation serves both the registry's default primary
   route and the true multi-parameter, three-output route. `ring` is local
   storage for the primary launch and caller-owned device scratch for the
   all-output launch; no host value participates once the frame is resident. */
static __device__ inline void adaptive_macd_neo_row_f64(
    const double* __restrict__ data,
    const int len,
    const int length,
    const int fast_period,
    const int slow_period,
    const int signal_period,
    double* __restrict__ ring,
    double* __restrict__ out_macd,
    double* __restrict__ out_signal,
    double* __restrict__ out_hist)
{
    const int L = length;
    const double n = (double)L;
    const double CORR_EPSILON = 1e-12;

    /* build_spec (:497-506), with the exact caller-owned parameter tuple. */
    const double a1 = 2.0 / ((double)fast_period + 1.0);
    const double a2 = 2.0 / ((double)slow_period + 1.0);
    const double delta_coeff = a1 - a2;
    const double recur_coeff = 2.0 - a1 - a2;
    const double trend_coeff = (1.0 - a1) * (1.0 - a2);
    const double cycle_coeff = (1.0 - a1) / (1.0 - a2);

    /* RollingCorrelationState::new (:553-567). The two sums are formed from
       the same integer expressions the CPU uses, so they are exact. */
    const unsigned long long l_u64 = (unsigned long long)L;
    const double sum_x = (double)((l_u64 - 1ULL) * l_u64) * 0.5;
    const double sum_x2 =
        (double)((l_u64 - 1ULL) * l_u64 * (2ULL * l_u64 - 1ULL)) / 6.0;
    const double denom_x = fma(n, sum_x2, -(sum_x * sum_x));

    for (int k = 0; k < L; ++k) ring[k] = 0.0;
    int    head = 0, count = 0;
    double sum_y = 0.0, sum_y2 = 0.0, sum_xy = 0.0;

    double prev_close = NEO_F64_NAN;
    double prev_macd1 = NEO_F64_NAN;
    double prev_macd2 = NEO_F64_NAN;

    /* EmaLikeState for the signal line. `hist` and `signal` share this state
       with `macd` on the CPU, so the all-output route emits them from this
       exact pass instead of reconstructing either series on the host. */
    const double s_alpha = 2.0 / ((double)signal_period + 1.0);
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
        } else if (s_count < signal_period) {
            s_count += 1; s_sum += current_macd;
            s_value = s_sum / (double)s_count;
        } else {
            s_value = fma(s_beta, s_value, s_alpha * current_macd);
        }

        const double signal = s_started ? s_value : NEO_F64_NAN;
        if (out_macd != nullptr) out_macd[i] = current_macd;
        if (out_signal != nullptr) out_signal[i] = signal;
        if (out_hist != nullptr) {
            out_hist[i] = (isfinite(current_macd) && isfinite(signal))
                ? current_macd - signal
                : NEO_F64_NAN;
        }
    }
}

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

    double ring[AMACD_NEO_LENGTH];
    adaptive_macd_neo_row_f64(
        data,
        series_len,
        AMACD_NEO_LENGTH,
        10,
        20,
        AMACD_NEO_SIGNAL,
        ring,
        out + (size_t)combo * (size_t)series_len,
        nullptr,
        nullptr);
}

extern "C" __global__
void adaptive_macd_neo_all_outputs_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ lengths,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const int* __restrict__ signal_periods,
    int n_combos,
    int max_length,
    double* __restrict__ ring_scratch,
    double* __restrict__ out_macd,
    double* __restrict__ out_signal,
    double* __restrict__ out_hist)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int L = lengths[combo];
    double* __restrict__ macd = out_macd + (size_t)combo * (size_t)series_len;
    double* __restrict__ signal = out_signal + (size_t)combo * (size_t)series_len;
    double* __restrict__ hist = out_hist + (size_t)combo * (size_t)series_len;
    if (series_len <= 0 || L < 2 || L > max_length || L > series_len
        || fast_periods[combo] < 2 || fast_periods[combo] > series_len
        || slow_periods[combo] < 2 || slow_periods[combo] > series_len
        || signal_periods[combo] < 2 || signal_periods[combo] > series_len) {
        for (int i = 0; i < series_len; ++i) {
            macd[i] = NEO_F64_NAN;
            signal[i] = NEO_F64_NAN;
            hist[i] = NEO_F64_NAN;
        }
        return;
    }

    adaptive_macd_neo_row_f64(
        data,
        series_len,
        L,
        fast_periods[combo],
        slow_periods[combo],
        signal_periods[combo],
        ring_scratch + (size_t)combo * (size_t)max_length,
        macd,
        signal,
        hist);
}
