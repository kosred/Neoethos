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
 * PERIOD-INVARIANT. The CPU batch reads `atr_length` (10) and
 * `percentile_length` (50) and never `period`.
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
 * SEQUENTIAL, one thread per combo column. Both rings are fixed-size
 * per-thread arrays at the CPU defaults (10 and 50 doubles plus their flags).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define ATRP_NEO_ATR_LEN   10
#define ATRP_NEO_PCT_LEN   50

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
    (void)periods; (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    const int AL = ATRP_NEO_ATR_LEN;
    const int PL = ATRP_NEO_PCT_LEN;

    double tr_values[ATRP_NEO_ATR_LEN];
    unsigned char tr_valid[ATRP_NEO_ATR_LEN];
    int    tr_idx = 0, tr_count = 0, tr_valid_count = 0;
    double tr_sum = 0.0;
    #pragma unroll
    for (int k = 0; k < ATRP_NEO_ATR_LEN; ++k) { tr_values[k] = 0.0; tr_valid[k] = 0; }

    double atr_values[ATRP_NEO_PCT_LEN];
    unsigned char atr_valid[ATRP_NEO_PCT_LEN];
    int    atr_idx = 0, atr_count = 0, atr_valid_count = 0;
    #pragma unroll
    for (int k = 0; k < ATRP_NEO_PCT_LEN; ++k) { atr_values[k] = 0.0; atr_valid[k] = 0; }

    double prev_close = NEO_F64_NAN;
    bool   has_prev_close = false;

    for (int i = 0; i < len; ++i) {
        o[i] = NEO_F64_NAN;              /* dst.fill(NAN) before the walk */

        if (tr_count >= AL) {
            const int old_idx = tr_idx;
            if (tr_valid[old_idx] != 0) {
                if (tr_valid_count > 0) tr_valid_count -= 1;
                tr_sum -= tr_values[old_idx];
            }
        } else {
            tr_count += 1;
        }

        const double h = high[i], l = low[i], c = close[i];
        const bool bar_ok = isfinite(h) && isfinite(l) && isfinite(c);
        bool   have_tr = false;
        double tr = 0.0;
        if (bar_ok) {
            const double hl = h - l;
            if (!has_prev_close || !isfinite(prev_close)) {
                tr = hl;
            } else {
                const double hc = fabs(h - prev_close);
                const double lc = fabs(l - prev_close);
                tr = fmax(fmax(hl, hc), lc);
            }
            have_tr = true;
        }

        if (have_tr) {
            tr_values[tr_idx] = tr;
            tr_valid[tr_idx] = 1;
            tr_valid_count += 1;
            tr_sum += tr;
        } else {
            tr_values[tr_idx] = 0.0;
            tr_valid[tr_idx] = 0;
        }
        tr_idx += 1; if (tr_idx == AL) tr_idx = 0;

        if (tr_count >= AL) {
            const bool   atr_valid_now = (tr_valid_count == AL);
            const double atr_now = atr_valid_now ? tr_sum / (double)AL : 0.0;

            if (atr_count >= PL) {
                if (atr_valid_now && atr_valid_count == PL) {
                    int below = 0;
                    for (int k = 0; k < PL; ++k) {
                        if (atr_now > atr_values[k]) below += 1;
                    }
                    o[i] = 100.0 * (double)below / (double)PL;
                } else {
                    o[i] = NEO_F64_NAN;
                }
                const int old_idx = atr_idx;
                if (atr_valid[old_idx] != 0) {
                    if (atr_valid_count > 0) atr_valid_count -= 1;
                }
            } else {
                atr_count += 1;
            }

            if (atr_valid_now) {
                atr_values[atr_idx] = atr_now;
                atr_valid[atr_idx] = 1;
                atr_valid_count += 1;
            } else {
                atr_values[atr_idx] = 0.0;
                atr_valid[atr_idx] = 0;
            }
            atr_idx += 1; if (atr_idx == PL) atr_idx = 0;
        }

        if (bar_ok) { prev_close = c; has_prev_close = true; }
        else        { prev_close = NEO_F64_NAN; has_prev_close = false; }
    }
}
