#include <cmath>
#include <cstddef>

extern "C" __global__ void emd_trend_batch_f64(
    const double* __restrict__ src,
    int len,
    const double* __restrict__ mults,
    const double* __restrict__ averages,
    const double* __restrict__ deviations,
    int rows,
    double* __restrict__ out_direction,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const double mult = mults[row];
    const double* row_avg = averages + static_cast<size_t>(row) * static_cast<size_t>(len);
    const double* row_dev = deviations + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction =
        out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);

    double direction = 0.0;
    for (int i = 0; i < len; ++i) {
        const double avg = row_avg[i];
        const double dev = row_dev[i];
        if (isfinite(avg) && isfinite(dev)) {
            row_upper[i] = avg + dev * mult;
            row_lower[i] = avg - dev * mult;
        } else {
            row_upper[i] = NAN;
            row_lower[i] = NAN;
        }

        if (i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(row_upper[i]) &&
            isfinite(row_upper[i - 1]) && src[i] > row_upper[i] &&
            src[i - 1] <= row_upper[i - 1]) {
            direction = 1.0;
        } else if (
            i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(row_lower[i]) &&
            isfinite(row_lower[i - 1]) && src[i] < row_lower[i] &&
            src[i - 1] >= row_lower[i - 1]
        ) {
            direction = -1.0;
        }
        row_direction[i] = direction;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — emd_trend
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/emd_trend.rs:724 `compute_from_source_into`.
 *
 * Column: output_id "value" / "average" -> `out.average` (cpu_batch.rs:14604),
 *   and `average_out.copy_from_slice(&average)` (:764) where `average` is
 *   `compute_average_series` (:684) = `ma_with_kernel("sma", src, length)`.
 *   So the "value" column of emd_trend IS a simple moving average of the
 *   source — the envelope, the direction state machine and the deviation EMA
 *   feed the OTHER four outputs and never this one. Emitting the envelope here
 *   would be a different series that passes every shape check.
 *
 * PERIOD-INVARIANT: `compute_emd_trend_batch` (cpu_batch.rs:14579-14582) reads
 *   `source`, `avg_type`, `length` (28) and `mult`, and NEVER `period`.
 *
 * Source: the CPU default is "close" (:30) — F64InputKind::CloseSlice.
 *   `avg_type` defaults to "SMA" (:31), which is why the average is an SMA and
 *   not one of the other 70 MAs the dispatcher can route to.
 *
 * Arithmetic: the seed is a plain left-to-right sum of the first `length`
 *   values and every later bar is `sum += x_new - x_old` — the same sliding
 *   accumulation `neoethos_sma_batch_f64` already runs for the `sma` id, so
 *   both lanes agree bit for bit on a column that is literally the same
 *   computation.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* DEFAULT_LENGTH, emd_trend.rs:32. */
#define NEO_EMD_TREND_LENGTH 28

extern "C" __global__
void emd_trend_neo_batch_f64(const double* __restrict__ prices,
                             int n,
                             const int* __restrict__ periods,
                             int n_combos,
                             int first_valid,
                             double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = NEO_EMD_TREND_LENGTH;
    if (first_valid < 0 || first_valid >= n) return;

    const int warm = first_valid + period - 1;
    if (warm >= n) return;

    double sum = 0.0;
    for (int k = 0; k < period; ++k) sum += prices[first_valid + k];
    const double inv = 1.0 / (double)period;

    o[warm] = sum * inv;
    for (int i = first_valid + period; i < n; ++i) {
        sum += prices[i] - prices[i - period];
        o[i] = sum * inv;
    }
}
