#include <cmath>
#include <cstddef>

static __device__ inline bool finite_pair(
    const double* main,
    const double* compare,
    int idx
) {
    return isfinite(main[idx]) && isfinite(compare[idx]);
}

static __device__ inline bool finite_return_pair(
    const double* main,
    const double* compare,
    int idx
) {
    return idx > 0 && finite_pair(main, compare, idx - 1) && finite_pair(main, compare, idx);
}

static __device__ inline double return_value(const double* values, int idx) {
    return values[idx] - values[idx - 1];
}

extern "C" __global__ void spearman_correlation_batch_f64(
    const double* __restrict__ main,
    const double* __restrict__ compare,
    int len,
    const int* __restrict__ lookbacks,
    const int* __restrict__ smoothing_lengths,
    int rows,
    double* __restrict__ out_raw,
    double* __restrict__ out_smoothed
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int lookback = lookbacks[row];
    int smoothing_length = smoothing_lengths[row];

    double* row_out_raw = out_raw + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_smoothed = out_smoothed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_raw[i] = NAN;
        row_out_smoothed[i] = NAN;
    }

    if (lookback <= 0 || smoothing_length <= 0 || lookback >= len) {
        return;
    }

    double mean_rank = (static_cast<double>(lookback) + 1.0) * 0.5;

    for (int i = 0; i < len; ++i) {
        int start = i + 1 - lookback;
        if (start < 1) {
            continue;
        }

        bool valid_window = true;
        for (int idx = start; idx <= i; ++idx) {
            if (!finite_return_pair(main, compare, idx)) {
                valid_window = false;
                break;
            }
        }
        if (!valid_window) {
            continue;
        }

        double cov = 0.0;
        double var_main = 0.0;
        double var_compare = 0.0;

        for (int a = start; a <= i; ++a) {
            double main_a = return_value(main, a);
            double compare_a = return_value(compare, a);

            int main_less = 0;
            int main_equal = 0;
            int compare_less = 0;
            int compare_equal = 0;

            for (int b = start; b <= i; ++b) {
                double main_b = return_value(main, b);
                double compare_b = return_value(compare, b);

                if (main_b < main_a) {
                    main_less += 1;
                } else if (main_b == main_a) {
                    main_equal += 1;
                }

                if (compare_b < compare_a) {
                    compare_less += 1;
                } else if (compare_b == compare_a) {
                    compare_equal += 1;
                }
            }

            double main_rank =
                1.0 + static_cast<double>(main_less) + 0.5 * static_cast<double>(main_equal - 1);
            double compare_rank = 1.0 +
                                  static_cast<double>(compare_less) +
                                  0.5 * static_cast<double>(compare_equal - 1);
            double dx = main_rank - mean_rank;
            double dy = compare_rank - mean_rank;
            cov += dx * dy;
            var_main += dx * dx;
            var_compare += dy * dy;
        }

        double denom = sqrt(var_main * var_compare);
        if (!isfinite(denom) || denom == 0.0) {
            continue;
        }

        double raw = cov / denom;
        row_out_raw[i] = raw;

        int smooth_start = i + 1 - smoothing_length;
        if (smooth_start < 0) {
            continue;
        }

        bool smooth_valid = true;
        double smooth_sum = 0.0;
        for (int j = smooth_start; j <= i; ++j) {
            double value = row_out_raw[j];
            if (!isfinite(value)) {
                smooth_valid = false;
                break;
            }
            smooth_sum += value;
        }

        if (smooth_valid) {
            row_out_smoothed[i] = smooth_sum / static_cast<double>(smoothing_length);
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — spearman_correlation
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/spearman_correlation.rs:451 `compute_raw_into`,
 *   with `rank_average` (:360) and `rank_pearson_correlation` (:383).
 *   `first_return` comes from `first_valid_return_idx` (:308) via `prepare`
 *   (:336).
 *
 * Column: output_id "value" resolves to `out.raw` — cpu_batch.rs:10685 accepts
 *   "raw"/"value" and returns the RAW series. The `smoothed` column
 *   (`compute_smoothed_into`, :525) is a different output id and is not
 *   computed here.
 *
 * PERIOD-INVARIANT: `compute_spearman_correlation_batch`
 *   (cpu_batch.rs:10658-10663) reads `source` (close), `comparison_source`
 *   (open), `lookback` (30) and `smoothing_length` (3) and NEVER `period`. A
 *   sweep of five periods produces five identical CPU columns, so the kernel
 *   emits five identical rows.
 *
 * FIRST-VALID IGNORED: the CPU's start index is `first_return`, the first
 *   i >= 1 at which main[i-1], main[i], compare[i-1] AND compare[i] are all
 *   FINITE (:308-315) — a two-series consecutive-pair rule over (close, open)
 *   that no `F64FirstValidRule` variant expresses. The kernel derives it
 *   itself rather than adopting a rule it does not honour.
 *
 * Input: (open, high, low, close) — F64InputKind::Ohlc4, of which only OPEN
 *   (the comparison source) and CLOSE (the main source) are read. High and low
 *   are never touched; the shape is Ohlc4 because the resident upload already
 *   carries all four and the four-pointer launch arm exists.
 *
 * Shape: ONE THREAD PER COLUMN. Each bar re-ranks its own trailing window, so
 *   the per-bar value carries no state; the thread body is the loop over bars
 *   because that is the shape this lane launches, and the window is copied
 *   into two per-thread arrays whose length is the compiled `lookback` bound.
 *
 * ORDER STATISTIC, computed by COUNTING rather than sorting. `rank_average`
 *   sorts with `total_cmp`, then gives every member of a tie group the average
 *   rank `(start + 1 + end) * 0.5` where `start` is the number of strictly
 *   smaller values and `end` is `start + tie_count` (:374). Counting produces
 *   exactly those two integers without a sort, and both are small integers, so
 *   the expression is exact in f64 either way. It is also why `-0.0` vs `0.0`
 *   cannot diverge: the CPU groups ties with `==`, which already merges them.
 *
 * NaN CANNOT REACH THE RANKING. The CPU emits NaN outright whenever
 *   `valid_pairs != lookback` (:508), i.e. whenever any return pair in the
 *   window is non-finite, and index 0 (whose return is NaN by construction) is
 *   never inside a window because `warm >= lookback` for `first_return >= 1`.
 *   The counting loop therefore only ever sees finite values, exactly as the
 *   sort does.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:10661-10663. `lookback` bounds two per-thread
 * window copies, so the bound belongs to the COMPILED kernel. */
#define NEO_SPEARMAN_LOOKBACK 30

extern "C" __global__
void spearman_correlation_neo_batch_f64(const double* __restrict__ open,
                                        const double* __restrict__ high,
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
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* derived below — see header */
    (void)high;        /* spearman_correlation never reads high or low */
    (void)low;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int lookback = NEO_SPEARMAN_LOOKBACK;

    /* `prepare` (:326) refuses `lookback >= len` before producing anything. */
    if (lookback >= n) return;

    /* `first_valid_return_idx` — the first i >= 1 whose return pair is formed
     * from four FINITE values. */
    int first_return = -1;
    for (int i = 1; i < n; ++i) {
        if (isfinite(close[i]) && isfinite(close[i - 1]) &&
            isfinite(open[i])  && isfinite(open[i - 1])) {
            first_return = i;
            break;
        }
    }
    if (first_return < 0) return;                 /* AllValuesNaN on the CPU */
    if (n - first_return < lookback) return;      /* NotEnoughValidData */

    const int warm = first_return + lookback - 1;
    if (warm >= n) return;

    const double rank_mean = ((double)lookback + 1.0) * 0.5;

    double wm[NEO_SPEARMAN_LOOKBACK];
    double wc[NEO_SPEARMAN_LOOKBACK];

    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - lookback;

        /* Materialise the window's return pairs and count the finite ones.
         * `main_returns[j]`/`compare_returns[j]` are BOTH NaN unless all four
         * prices behind them are finite (:487-497), so one flag serves both. */
        int valid_pairs = 0;
        for (int k = 0; k < lookback; ++k) {
            const int j = start + k;
            const double m0 = close[j - 1];
            const double m1 = close[j];
            const double c0 = open[j - 1];
            const double c1 = open[j];
            if (isfinite(m0) && isfinite(m1) && isfinite(c0) && isfinite(c1)) {
                wm[k] = m1 - m0;
                wc[k] = c1 - c0;
                ++valid_pairs;
            } else {
                wm[k] = NEO_F64_NAN;
                wc[k] = NEO_F64_NAN;
            }
        }
        if (valid_pairs != lookback) {
            o[i] = NEO_F64_NAN;
            continue;
        }

        /* Rank by counting: `less` is the sort position of the tie group's
         * first member, `less + ties` is one past its last. */
        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (int k = 0; k < lookback; ++k) {
            int less_m = 0, ties_m = 0, less_c = 0, ties_c = 0;
            for (int q = 0; q < lookback; ++q) {
                if (wm[q] <  wm[k]) ++less_m;
                if (wm[q] == wm[k]) ++ties_m;
                if (wc[q] <  wc[k]) ++less_c;
                if (wc[q] == wc[k]) ++ties_c;
            }
            const double rx = ((double)less_m + 1.0 + (double)(less_m + ties_m)) * 0.5;
            const double ry = ((double)less_c + 1.0 + (double)(less_c + ties_c)) * 0.5;
            const double dx = rx - rank_mean;
            const double dy = ry - rank_mean;
            cov   += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        const double denom = sqrt(var_x * var_y);
        o[i] = (denom == 0.0 || !isfinite(denom)) ? NEO_F64_NAN : (cov / denom);
    }
}
