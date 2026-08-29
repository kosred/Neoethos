#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool atr_percentile_valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline bool atr_percentile_compute_atr(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int t,
    int atr_length,
    double* out_atr
) {
    if (atr_length <= 0 || t < atr_length - 1) {
        return false;
    }

    int start = t + 1 - atr_length;
    double sum = 0.0;

    for (int i = start; i <= t; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!atr_percentile_valid_bar(h, l, c)) {
            return false;
        }

        double tr = h - l;
        if (i > 0) {
            double prev_h = high[i - 1];
            double prev_l = low[i - 1];
            double prev_c = close[i - 1];
            if (atr_percentile_valid_bar(prev_h, prev_l, prev_c)) {
                double hc = fabs(h - prev_c);
                double lc = fabs(l - prev_c);
                if (hc > tr) {
                    tr = hc;
                }
                if (lc > tr) {
                    tr = lc;
                }
            }
        }

        sum += tr;
    }

    *out_atr = sum / static_cast<double>(atr_length);
    return true;
}

extern "C" __global__ void atr_percentile_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ atr_lengths,
    const int* __restrict__ percentile_lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int atr_length = atr_lengths[combo_idx];
    int percentile_length = percentile_lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (atr_length <= 0 || percentile_length <= 0) {
        return;
    }

    int first_output = atr_length + percentile_length - 1;
    if (first_output >= len) {
        return;
    }

    for (int t = first_output; t < len; ++t) {
        double current_atr = 0.0;
        if (!atr_percentile_compute_atr(high, low, close, t, atr_length, &current_atr)) {
            continue;
        }

        bool valid = true;
        int below = 0;
        for (int offset = 1; offset <= percentile_length; ++offset) {
            double prev_atr = 0.0;
            if (!atr_percentile_compute_atr(high, low, close, t - offset, atr_length, &prev_atr)) {
                valid = false;
                break;
            }
            below += static_cast<int>(current_atr > prev_atr);
        }

        if (valid) {
            row[t] = 100.0 * static_cast<double>(below) / static_cast<double>(percentile_length);
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - atr_percentile
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/atr_percentile.rs:447
 *             `atr_percentile_row_from_slices`, with the true range at :388.
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:8388 `expect_value_output`).
 *
 * WINDOW-ANCHORED. NeoEthos' versioned scalar ABI carries the largest member
 * of the registry tuple.  V1 is the declared (atr_length=10,
 * percentile_length=50) shape, scaled with positive integer half-up rounding:
 * percentile_length=anchor and atr_length=round(10*anchor/50).  This is the
 * same mapping used to build the CPU authority request; it is not an invented
 * `period` alias.
 *
 * FIRST-VALID IGNORED. `atr_percentile_into_slice` fills NaN and calls the
 * row function, which walks from index 0 and does NOT take `first`
 * (`_first` is discarded at :641). Validity is per bar, tracked inside the
 * two rings by a parallel `valid` flag array, so a hole does not reset the
 * rings - it marks slots invalid and they age out. That is materially
 * different from a reset and is reproduced literally.
 *
 * THE ATR IS A SIMPLE MOVING AVERAGE OF TRUE RANGE, NOT A WILDER SMOOTHING
 * (:499 `tr_sum / atr_length`). Substituting the Wilder recurrence, which is
 * what "ATR" usually means in this crate, would be a different indicator.
 *
 * THE ROLLING SUM IS AN ACCUMULATOR: `tr_sum += value` on push and
 * `tr_sum -= tr_values[old]` on eviction (:465, :479). Not a fresh window
 * sum. Reproduced in that order.
 *
 * PERCENTILE: `below` counts strict `atr_now > prev` over the
 * `percentile_length` ATRs ALREADY in the ring - the scan happens BEFORE this
 * bar is pushed (:505-514, push at :528), so the current ATR is not compared
 * with itself. The result is `100.0 * below / percentile_length`.
 * This is an ORDER STATISTIC - a count of comparisons - so it has no
 * accumulation order to preserve and no rounding to match beyond the final
 * divide.
 *
 * NaN SEMANTICS: `hl.max(hc).max(lc)` (:404) is `f64::max`, which returns the
 * NON-NaN operand. `fmax` matches. The guard above it already rejects a
 * non-finite bar, but the nesting is kept identical so the two agree even on
 * an infinity.
 *
 * SEQUENTIAL, one thread per combo column.  The output row is first used as
 * ATR scratch, then converted in DESCENDING index order so every historical
 * ATR is still present when compared.  True range eviction is recomputed from
 * the immutable HLC inputs, eliminating a fixed-size per-thread ring without
 * changing the CPU subtract-then-add accumulator order.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* V1 of the registry-ratio ABI: round-half-up(anchor * 10 / 50).  For a
 * positive integral anchor this is quotient + (remainder >= 3).  Keeping the
 * quotient and remainder separate avoids signed overflow at INT_MAX.
 */
__device__ __forceinline__ int anchor_atr_length_v1(int anchor) {
    const int quotient = anchor / 5;
    const int remainder = anchor % 5;
    const int scaled = quotient + (remainder >= 3 ? 1 : 0);
    return scaled > 0 ? scaled : 1;
}

__device__ __forceinline__ bool atrp_neo_valid_bar(double high,
                                                   double low,
                                                   double close) {
    return isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ __forceinline__ bool atrp_neo_true_range_at(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int i,
    double* __restrict__ value)
{
    const double h = high[i], l = low[i], c = close[i];
    if (!atrp_neo_valid_bar(h, l, c)) return false;

    const double hl = h - l;
    if (i == 0 || !atrp_neo_valid_bar(high[i - 1], low[i - 1], close[i - 1])) {
        *value = hl;
        return true;
    }

    const double prev_close = close[i - 1];
    const double hc = fabs(h - prev_close);
    const double lc = fabs(l - prev_close);
    *value = fmax(fmax(hl, hc), lc);
    return true;
}

/* Scratch must retain the CPU's validity flag independently of its numeric
 * ATR.  Finite HLC values can still overflow an arithmetic expression to NaN;
 * such a value is VALID in the CPU ring and comparisons against it are simply
 * false.  A dedicated payload represents only an invalid HLC window.
 */
#define ATRP_NEO_INVALID_SCRATCH (__longlong_as_double(0x7ff8000000000001ULL))
__device__ __forceinline__ bool atrp_neo_scratch_valid(double value) {
    return (unsigned long long)__double_as_longlong(value) != 0x7ff8000000000001ULL;
}

extern "C" __global__
void atr_percentile_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
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

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (len <= 0) return;

    const int anchor = periods[combo];
    if (anchor <= 0) return;
    const int AL = anchor_atr_length_v1(anchor);
    const int PL = anchor;
    if (AL > len || PL > len || PL > len - AL) return;

    int tr_count = 0;
    int tr_valid_count = 0;
    double tr_sum = 0.0;

    /* Pass 1: identical subtract-old, add-new accumulator; o[i] temporarily
     * holds the ATR and its validity payload.
     */
    for (int i = 0; i < len; ++i) {
        if (tr_count >= AL) {
            double old_tr = 0.0;
            if (atrp_neo_true_range_at(high, low, close, i - AL, &old_tr)) {
                if (tr_valid_count > 0) tr_valid_count -= 1;
                tr_sum -= old_tr;
            }
        } else {
            tr_count += 1;
        }

        double tr = 0.0;
        if (atrp_neo_true_range_at(high, low, close, i, &tr)) {
            tr_valid_count += 1;
            tr_sum += tr;
        }

        if (tr_count >= AL) {
            o[i] = tr_valid_count == AL
                ? tr_sum / (double)AL
                : ATRP_NEO_INVALID_SCRATCH;
        } else {
            o[i] = ATRP_NEO_INVALID_SCRATCH;
        }
    }

    /* Pass 2 descends so o[i-k] is still the historical ATR scratch.  The CPU
     * compares against exactly the PL prior ATRs, never against the current
     * value itself.
     */
    const int first_output = AL + PL - 1;
    for (int i = len - 1; i >= first_output; --i) {
        const double current = o[i];
        bool valid = atrp_neo_scratch_valid(current);
        int below = 0;
        for (int k = 1; valid && k <= PL; ++k) {
            const double previous = o[i - k];
            if (!atrp_neo_scratch_valid(previous)) {
                valid = false;
            } else if (current > previous) {
                below += 1;
            }
        }
        o[i] = valid ? 100.0 * (double)below / (double)PL : NEO_F64_NAN;
    }
    for (int i = 0; i < first_output; ++i) o[i] = NEO_F64_NAN;
}
