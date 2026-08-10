#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void volatility_quality_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ fast_lengths,
    const int* __restrict__ slow_lengths,
    int n_combos,
    double* __restrict__ out_vqi_sum,
    double* __restrict__ out_fast_sma,
    double* __restrict__ out_slow_sma
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int fast_length = fast_lengths[combo_idx];
    int slow_length = slow_lengths[combo_idx];
    double* row_vqi = out_vqi_sum + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_fast = out_fast_sma + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_slow = out_slow_sma + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    double prev_close = CUDART_NAN;
    double prev_vqi_t = 0.0;
    double cumulative = 0.0;

    for (int i = 0; i < len; ++i) {
        double o = open[i];
        double h = high[i];
        double l = low[i];
        double c = close[i];
        double range = h - l;

        double tr;
        if (isfinite(h) && isfinite(l)) {
            if (isfinite(prev_close)) {
                tr = range;
                double hc = fabs(h - prev_close);
                double lc = fabs(l - prev_close);
                if (hc > tr) {
                    tr = hc;
                }
                if (lc > tr) {
                    tr = lc;
                }
            } else {
                tr = range;
            }
        } else {
            tr = CUDART_NAN;
        }

        double vqi_t;
        if (isfinite(prev_close) &&
            isfinite(o) &&
            isfinite(h) &&
            isfinite(l) &&
            isfinite(c) &&
            isfinite(tr) &&
            tr != 0.0 &&
            isfinite(range) &&
            range != 0.0) {
            vqi_t = 0.5 * (((c - prev_close) / tr) + ((c - o) / range));
        } else {
            vqi_t = prev_vqi_t;
        }

        double raw;
        if (isfinite(prev_close) && isfinite(o) && isfinite(c)) {
            raw = fabs(vqi_t) * 0.5 * ((c - prev_close) + (c - o));
        } else {
            raw = 0.0;
        }

        prev_vqi_t = vqi_t;
        prev_close = c;
        cumulative += raw;
        row_vqi[i] = cumulative;
        row_fast[i] = CUDART_NAN;
        row_slow[i] = CUDART_NAN;
    }

    if (fast_length > 0 && fast_length <= len) {
        double sum = 0.0;
        for (int i = 0; i < fast_length; ++i) {
            sum += row_vqi[i];
        }
        row_fast[fast_length - 1] = sum / static_cast<double>(fast_length);
        for (int i = fast_length; i < len; ++i) {
            sum += row_vqi[i] - row_vqi[i - fast_length];
            row_fast[i] = sum / static_cast<double>(fast_length);
        }
    }

    if (slow_length > 0 && slow_length <= len) {
        double sum = 0.0;
        for (int i = 0; i < slow_length; ++i) {
            sum += row_vqi[i];
        }
        row_slow[slow_length - 1] = sum / static_cast<double>(slow_length);
        for (int i = slow_length; i < len; ++i) {
            sum += row_vqi[i] - row_vqi[i - slow_length];
            row_slow[i] = sum / static_cast<double>(slow_length);
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — volatility_quality_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/volatility_quality_index.rs:424
 *   compute_vqi_sum_series_into, driving compute_vqi_point (:364).
 *
 * Column: output_id "value" resolves to out.vqi_sum — cpu_batch.rs:10985
 *   accepts "vqi_sum"/"value". The fast_sma / slow_sma columns are separate
 *   output ids computed by sma_into (:445) over this same series; neither
 *   feeds back into it, so neither is computed here.
 *
 * PERIOD-INVARIANT: compute_volatility_quality_index_batch reads fast_length
 *   (9) and slow_length (200) and NEVER period (cpu_batch.rs:10966-10968), and
 *   the vqi_sum column does not read either of them at all.
 *
 * FIRST-VALID IGNORED: the series is a CUMULATIVE SUM walked from index 0 with
 *   no reset and no warmup prefix — every bar contributes, and a bar whose
 *   prices are not finite contributes raw = 0.0 (:407-411) rather than
 *   truncating the series. Skipping to a first-valid index would start the
 *   cumulative sum at a different place and shift every later value.
 *
 * Input: open / high / low / close — extract_ohlc_full_input
 *   (cpu_batch.rs:10957) — F64InputKind::Ohlc4. All four are read.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `cumulative` is a running sum
 *   and `prev_vqi_t` is carried forward whenever the current bar cannot form a
 *   new one, so the value at bar i depends on every bar before it.
 *
 * ARITHMETIC taken verbatim:
 *   * the true range is built by three COMPARISONS in order — start from
 *     high - low, replace with |high - prev_close| if larger, then with
 *     |low - prev_close| if larger (:373-386). Kept as comparisons rather
 *     than fmax so the exact tie behaviour (`>` not `>=`) is preserved; the
 *     operands here are all finite because the branch is guarded by
 *     is_finite on high, low and prev_close.
 *   * vqi_t is 0.5 * ((close - prev_close)/tr + (close - open)/range) (:402)
 *     — the two quotients are summed FIRST and the product by 0.5 is last.
 *   * raw is |vqi_t| * 0.5 * ((close - prev_close) + (close - open)) (:409),
 *     associating left to right exactly as written.
 *   * there is no tolerance anywhere in this column: the guards are the exact
 *     tests tr != 0.0 and range != 0.0, reproduced as written. No epsilon is
 *     introduced, because introducing one would change the answer.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void volatility_quality_index_neo_batch_f64(const double* __restrict__ open,
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
    (void)first_valid; /* cumulative from index 0 — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    double prev_close = NEO_F64_NAN;
    double prev_vqi_t = 0.0;
    double cumulative = 0.0;

    for (int i = 0; i < n; ++i) {
        const double op = open[i];
        const double hi = high[i];
        const double lo = low[i];
        const double cl = close[i];

        const double range = hi - lo;

        double tr;
        if (isfinite(hi) && isfinite(lo)) {
            if (isfinite(prev_close)) {
                tr = range;
                const double hc = fabs(hi - prev_close);
                if (hc > tr) tr = hc;
                const double lc = fabs(lo - prev_close);
                if (lc > tr) tr = lc;
            } else {
                tr = range;
            }
        } else {
            tr = NEO_F64_NAN;
        }

        double vqi_t;
        if (isfinite(prev_close) && isfinite(op) && isfinite(hi) && isfinite(lo) &&
            isfinite(cl) && isfinite(tr) && tr != 0.0 && isfinite(range) && range != 0.0) {
            vqi_t = 0.5 * (((cl - prev_close) / tr) + ((cl - op) / range));
        } else {
            vqi_t = prev_vqi_t;
        }

        double raw;
        if (isfinite(prev_close) && isfinite(op) && isfinite(cl)) {
            raw = fabs(vqi_t) * 0.5 * ((cl - prev_close) + (cl - op));
        } else {
            raw = 0.0;
        }

        prev_vqi_t = vqi_t;
        prev_close = cl;
        cumulative += raw;
        o[i] = cumulative;
    }
}
