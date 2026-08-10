#include <cmath>
#include <cstddef>

namespace {

constexpr int ATR_PERIOD = 200;

__device__ inline bool valid_bar(double high, double low, double source) {
    return isfinite(high) && isfinite(low) && isfinite(source) && high >= low;
}

__device__ inline double pine_sign(double value) {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

__device__ inline double true_range(double high, double low, double prev_close) {
    if (isfinite(prev_close)) {
        const double a = high - low;
        const double b = fabs(high - prev_close);
        const double c = fabs(low - prev_close);
        return fmax(a, fmax(b, c));
    }
    return high - low;
}

}

extern "C" __global__ void hypertrend_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ source,
    int len,
    const double* __restrict__ factors,
    const double* __restrict__ slopes,
    const double* __restrict__ width_ratios,
    int rows,
    double* __restrict__ out_upper,
    double* __restrict__ out_average,
    double* __restrict__ out_lower,
    double* __restrict__ out_trend,
    double* __restrict__ out_changed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const double factor = factors[row];
    const double slope = slopes[row];
    const double width_ratio = width_ratios[row];

    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_average = out_average + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_upper[i] = NAN;
        row_average[i] = NAN;
        row_lower[i] = NAN;
        row_trend[i] = NAN;
        row_changed[i] = NAN;
    }

    if (!isfinite(factor) || factor <= 0.0 || !isfinite(slope) || slope <= 0.0 ||
        !isfinite(width_ratio) || width_ratio < 0.0 || width_ratio > 1.0) {
        return;
    }

    bool initialized = false;
    double avg = 0.0;
    double hold = 0.0;
    double os = 1.0;

    double prev_close = NAN;
    double seed_sum = 0.0;
    int seed_count = 0;
    double atr = NAN;

    for (int i = 0; i < len; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double src = source[i];

        if (!valid_bar(hi, lo, src)) {
            initialized = false;
            avg = 0.0;
            hold = 0.0;
            os = 1.0;
            prev_close = NAN;
            seed_sum = 0.0;
            seed_count = 0;
            atr = NAN;
            continue;
        }

        const double tr = true_range(hi, lo, prev_close);
        prev_close = src;

        double atr_value = 0.0;
        if (seed_count < ATR_PERIOD) {
            seed_sum += tr;
            seed_count += 1;
            if (seed_count == ATR_PERIOD) {
                atr = seed_sum / static_cast<double>(ATR_PERIOD);
                atr_value = atr;
            }
        } else {
            atr = ((atr * static_cast<double>(ATR_PERIOD - 1)) + tr) / static_cast<double>(ATR_PERIOD);
            atr_value = atr;
        }

        if (!initialized) {
            avg = src;
            hold = 0.0;
            os = 1.0;
            row_average[i] = avg;
            row_upper[i] = avg;
            row_lower[i] = avg;
            row_trend[i] = os;
            row_changed[i] = 0.0;
            initialized = true;
            continue;
        }

        const double atr_band = atr_value * factor;
        const double next_avg = fabs(src - avg) > atr_band
            ? 0.5 * (src + avg)
            : avg + os * (hold / factor / slope);
        const double next_os = pine_sign(next_avg - avg);
        const double changed = next_os != os ? 1.0 : 0.0;
        const double next_hold = changed != 0.0 ? atr_band : hold;
        const double upper = next_avg + width_ratio * next_hold;
        const double lower = next_avg - width_ratio * next_hold;

        row_upper[i] = upper;
        row_average[i] = next_avg;
        row_lower[i] = lower;
        row_trend[i] = next_os;
        row_changed[i] = changed;

        avg = next_avg;
        hold = next_hold;
        os = next_os;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — hypertrend
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/hypertrend.rs:389 `hypertrend_row_scalar`, fed by
 *   `compute_atr_zeroed` (:351). That ATR pass is a SEPARATE full sweep over
 *   the series before the trend walk, and it writes 0.0 — not NaN — for every
 *   bar before its 200-bar seed completes (:378 falls through with out[i]
 *   already 0.0). The trend walk then multiplies that zero by `factor`, which
 *   makes `(src - avg).abs() > atr` true on the very first comparison. Reading
 *   the ATR as NaN during warmup would take the other branch and produce a
 *   different series from bar 2 onward.
 *
 * Column: output_id "value" / "average" -> `out.average` (cpu_batch.rs:12765).
 *
 * PERIOD-INVARIANT: `compute_hypertrend_batch` (cpu_batch.rs:12785-12788) reads
 *   `source`, `factor` (5.0), `slope` (14.0) and `width_percent` (80.0) and
 *   NEVER `period`. ATR_PERIOD is a constant 200 (:32), not a parameter.
 *
 * Input: high / low / close — F64InputKind::Hlc; the CPU source default is
 *   "close" (cpu_batch.rs:12785).
 *
 * `valid_bar` (:281) is STRICTER than "all three finite": it also requires
 *   high >= low. A bar with an inverted range resets both the ATR seed and the
 *   trend state, so the gate is reproduced exactly rather than relaxed to
 *   isfinite.
 *
 * `pine_sign` (:329) returns 0.0 at exactly zero — not 1.0 — and `next_os` is
 *   compared to `os` by equality to decide `changed`, so the three-way form is
 *   kept instead of a copysign.
 *
 * Shape: ONE THREAD PER COLUMN, two sequential passes fused into one walk. The
 *   ATR is a Wilder recursion and `avg`/`hold`/`os` carry across every bar.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* ATR_PERIOD, hypertrend.rs:32. */
#define NEO_HT_ATR_PERIOD 200

extern "C" __global__
void hypertrend_neo_batch_f64(const double* __restrict__ high,
                              const double* __restrict__ low,
                              const double* __restrict__ close,
                              int n,
                              const int* __restrict__ periods,
                              int n_combos,
                              int first_valid,
                              double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;
    (void)first_valid;   /* alloc_with_nan_prefix only pre-fills; the row walk
                          * itself starts at 0 and the invalid-bar branch
                          * writes NaN for every bar before the first valid one */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    const double factor        = 5.0;
    const double slope         = 14.0;
    const double width_percent = 80.0;
    const double width_ratio   = width_percent * 0.01;   /* :543 */
    (void)width_ratio;   /* upper/lower only; the "value" column is `average` */

    const double ap = (double)NEO_HT_ATR_PERIOD;

    /* compute_atr_zeroed state (:351) */
    double atr_prev_close = NEO_F64_NAN;
    double seed_sum = 0.0;
    int    seed_count = 0;
    double atr_state = NEO_F64_NAN;

    /* hypertrend_row_scalar state (:403) */
    bool   initialized = false;
    double avg = 0.0, hold = 0.0, os = 1.0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], src = close[i];
        const bool ok = isfinite(h) && isfinite(l) && isfinite(src) && (h >= l);

        double atr_i;
        if (!ok) {
            atr_i = 0.0;
            atr_prev_close = NEO_F64_NAN;
            seed_sum = 0.0; seed_count = 0; atr_state = NEO_F64_NAN;
        } else {
            /* true_range (:340): NaN prev_close -> high - low */
            double tr;
            if (isfinite(atr_prev_close)) {
                const double a = h - l;
                const double b = fabs(h - atr_prev_close);
                const double c = fabs(l - atr_prev_close);
                tr = fmax(fmax(a, b), c);
            } else {
                tr = h - l;
            }
            atr_prev_close = src;

            atr_i = 0.0;
            if (seed_count < NEO_HT_ATR_PERIOD) {
                seed_sum += tr; ++seed_count;
                if (seed_count == NEO_HT_ATR_PERIOD) {
                    atr_state = seed_sum / ap;
                    atr_i = atr_state;
                }
            } else {
                atr_state = ((atr_state * (ap - 1.0)) + tr) / ap;
                atr_i = atr_state;
            }
        }

        if (!ok) {
            o[i] = NEO_F64_NAN;
            initialized = false; avg = 0.0; hold = 0.0; os = 1.0;
            continue;
        }

        if (!initialized) {
            avg = src; hold = 0.0; os = 1.0;
            o[i] = avg;
            initialized = true;
            continue;
        }

        const double atr = atr_i * factor;
        const double next_avg = (fabs(src - avg) > atr)
                                    ? (0.5 * (src + avg))
                                    : (avg + os * (hold / factor / slope));
        const double d = next_avg - avg;
        const double next_os = (d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0);
        const double changed = (next_os != os) ? 1.0 : 0.0;
        const double next_hold = (changed != 0.0) ? atr : hold;

        o[i] = next_avg;

        avg = next_avg; hold = next_hold; os = next_os;
    }
}
