#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void donchian_channel_width_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    for (int t = 0; t < len; ++t) {
        double h = high[t];
        double l = low[t];
        if (!isfinite(h) || !isfinite(l)) {
            continue;
        }

        double max_h = -CUDART_INF;
        double min_l = CUDART_INF;
        int count = 0;

        for (int i = t; i >= 0 && count < period; --i) {
            double hh = high[i];
            double ll = low[i];
            if (!isfinite(hh) || !isfinite(ll)) {
                break;
            }
            if (hh > max_h) {
                max_h = hh;
            }
            if (ll < min_l) {
                min_l = ll;
            }
            count += 1;
        }

        if (count == period) {
            row[t] = max_h - min_l;
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - donchian_channel_width
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/donchian_channel_width.rs:354 `compute_row`
 *             (and :438 `compute_row_no_nan`, the all-valid fast path),
 *             entered from `..._into_slice` (:526).
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:6948 `expect_value_output`).
 *
 * PERIOD-SWEPT: `compute_donchian_channel_width_batch` reads `period`
 * (default 20), so `periods[combo]` is honoured.
 *
 * FIRST-VALID IGNORED. `compute_row` walks from index 0 and derives its own
 * segment boundaries: an invalid pair emits NaN, clears both monotone deques
 * and ENDS THE SEGMENT, so the `period`-bar window restarts from the next
 * valid bar (`seg_start`, :379). A single first-valid index cannot express
 * that - the indicator has one warmup per segment, not one per series.
 * Registered as `F64FirstValidRule::Ignored`.
 *
 * NO DEQUE, AND THAT IS NOT AN APPROXIMATION. The CPU keeps two monotone
 * deques of size `period + 1` purely as an O(1) amortisation; the VALUE it
 * emits is `max(high[w..=i]) - min(low[w..=i])` over a window that is always
 * exactly `period` bars wide at emit time (`i + 1 >= seg_start + period`
 * implies `window_start == i + 1 - period`). max and min are exact
 * SELECTIONS of input values - they introduce no rounding and have no
 * accumulation order - so an O(period) rescan produces the identical double.
 * Doing it this way removes the only thing that would have needed a
 * `max_period` bound and a per-thread array, so this kernel refuses no period.
 *
 * The subtraction `upper - lower` is the ONLY arithmetic in the indicator and
 * is a single rounding on both sides.
 *
 * VALIDITY: `is_valid_pair` (:283) is `high.is_finite() && low.is_finite()` -
 * `is_finite`, not `!is_nan`, so an infinite high breaks the segment. Matched
 * exactly; the weaker test would extend a segment the CPU ends.
 *
 * SEQUENTIAL PER COLUMN because the segment state is carried across bars.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void donchian_channel_width_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int period = periods[combo];

    if (period <= 0 || period > len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    int  seg_start = 0;
    bool in_segment = false;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            o[i] = NEO_F64_NAN;
            in_segment = false;
            continue;
        }
        if (!in_segment) { seg_start = i; in_segment = true; }

        if (i + 1 >= seg_start + period) {
            /* window_start == max(i + 1 - period, seg_start) == i + 1 - period
               once the segment is at least `period` bars old (:429). */
            const int ws = i + 1 - period;
            double upper = high[ws];
            double lower = low[ws];
            for (int k = ws + 1; k <= i; ++k) {
                upper = fmax(upper, high[k]);
                lower = fmin(lower, low[k]);
            }
            o[i] = upper - lower;
        } else {
            o[i] = NEO_F64_NAN;
        }
    }
}
