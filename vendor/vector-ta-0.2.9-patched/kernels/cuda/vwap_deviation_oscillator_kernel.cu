#include <cmath>
#include <cstddef>

extern "C" __global__ void vwap_deviation_oscillator_batch_f64(
    const double* source_values,
    int len,
    const int* modes,
    const int* windows,
    const double* guards,
    int rows,
    int max_window,
    double* scratch_values,
    double* out_osc,
    double* out_std1,
    double* out_std2,
    double* out_std3
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0 || max_window <= 0) {
        return;
    }

    const int mode = modes[row];
    const int window = windows[row];
    const double guard = guards[row];
    if (window <= 0 || window > max_window) {
        return;
    }

    const double* row_source = source_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_scratch =
        scratch_values + static_cast<size_t>(row) * static_cast<size_t>(max_window);
    double* row_osc = out_osc + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_std1 = out_std1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_std2 = out_std2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_std3 = out_std3 + static_cast<size_t>(row) * static_cast<size_t>(len);

    const double nan = NAN;
    int head = 0;
    int count = 0;
    double sum = 0.0;
    double sumsq = 0.0;

    for (int i = 0; i < len; ++i) {
        const double value = row_source[i];
        row_osc[i] = value;
        if (mode == 2) {
            row_std1[i] = 1.0;
            row_std2[i] = 2.0;
            row_std3[i] = 3.0;
        } else {
            row_std1[i] = nan;
            row_std2[i] = nan;
            row_std3[i] = nan;
        }

        if (isfinite(value)) {
            if (count < window) {
                row_scratch[count] = value;
                count += 1;
            } else {
                const double old = row_scratch[head];
                sum -= old;
                sumsq -= old * old;
                row_scratch[head] = value;
                head += 1;
                if (head == window) {
                    head = 0;
                }
            }
            sum += value;
            sumsq += value * value;
        }

        if (count < window) {
            if (mode == 2) {
                row_osc[i] = nan;
            }
            continue;
        }

        const double mean = sum / static_cast<double>(window);
        double variance = sumsq / static_cast<double>(window) - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        const double std = sqrt(variance);

        if (mode == 2) {
            if (!isfinite(value) || !isfinite(std) || std <= 0.0) {
                row_osc[i] = nan;
            } else {
                row_osc[i] = (value - mean) / std;
            }
            continue;
        }

        if (!isfinite(std)) {
            continue;
        }

        const double std1 = std > guard ? std : guard;
        row_std1[i] = std1;
        row_std2[i] = std1 * 2.0;
        row_std3[i] = std1 * 3.0;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — vwap_deviation_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/vwap_deviation_oscillator.rs:763
 *   compute_base_series_filtered, RollingBars arm (:786-828), with price_ref
 *   (:585). With the batch defaults the osc column IS resid_abs — the
 *   Absolute arm of compute_output_field_from_base (:981) is a straight
 *   copy_from_slice of it — so no volatility window is involved in this
 *   column at all.
 *
 * Column: output_id "value" resolves to osc — cpu_batch.rs:11024 accepts
 *   "osc"/"value". std1/std2/std3 are separate output ids driven by
 *   RollingFiniteWindow over this same series; none of them feeds back.
 *
 * PERIOD-INVARIANT: compute_vwap_deviation_oscillator_batch reads
 *   session_mode ("rolling_bars"), rolling_period (20), rolling_days (30),
 *   use_close (false), deviation_mode ("absolute"), z_window (50),
 *   pct_vol_lookback (100), pct_min_sigma (0.1) and abs_vol_lookback (100) —
 *   and NEVER period (cpu_batch.rs:11050-11085). Five swept periods give five
 *   identical CPU columns, so this kernel emits five identical rows.
 *
 * THE DEFAULTS SELECT A TIMESTAMP-FREE PATH, and that is why this kernel does
 *   not need the bar timestamps the CPU signature carries. session_mode is
 *   "rolling_bars", whose arm never reads `timestamps` — only the RollingDays
 *   and the FourHours/Daily/Weekly arms do (:830, :871). Pinning the default
 *   is the same thing every period-invariant registration in this lane does.
 *
 * FIRST-VALID IGNORED: the series is walked from index 0 and a bar whose price
 *   or volume is non-finite contributes (0.0, 0.0) to the ring (:794-798)
 *   rather than being skipped, so the window still advances. Starting later
 *   would build a different window.
 *
 * Input: (high, low, close, volume) — F64InputKind::Hlcv. use_close is FALSE
 *   by default, so the price reference is hlc3 = (h + l + c) / 3 and it is
 *   NaN unless all three are finite (:588-591); volume is read for the weight.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. The (sum_pv, sum_vol) pair is
 *   an INCREMENTAL ring sum, so the accumulation order is load-bearing.
 *
 * ARITHMETIC taken verbatim:
 *   * the ring SUBTRACTS THE OUTGOING ENTRY FIRST, before the incoming one is
 *     stored and added (:799-812).
 *   * the vwap guard is the exact test sum_vol != 0.0 (:814) — no epsilon is
 *     introduced, because introducing one would change the answer.
 *   * the residual is pr - vwap (:820), written only when BOTH pr and vwap are
 *     finite; otherwise the pre-filled NaN stands.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:11063. rolling_period sizes the per-thread ring,
 * so the bound belongs to the COMPILED kernel. */
#define NEO_VDO_ROLLING_PERIOD 20

extern "C" __global__
void vwap_deviation_oscillator_neo_batch_f64(const double* __restrict__ high,
                                             const double* __restrict__ low,
                                             const double* __restrict__ close,
                                             const double* __restrict__ volume,
                                             int n,
                                             const int* __restrict__ periods,
                                             int n_combos,
                                             int first_valid,
                                             double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* walked from index 0 — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = NEO_VDO_ROLLING_PERIOD;

    double ring_pv[NEO_VDO_ROLLING_PERIOD], ring_vol[NEO_VDO_ROLLING_PERIOD];
    for (int k = 0; k < period; ++k) { ring_pv[k] = 0.0; ring_vol[k] = 0.0; }
    int head = 0, count = 0;
    double sum_pv = 0.0, sum_vol = 0.0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i], v = volume[i];

        /* price_ref with use_close == false */
        const double pr = (isfinite(h) && isfinite(l) && isfinite(c))
            ? ((h + l + c) / 3.0)
            : NEO_F64_NAN;

        double cpv = 0.0, cvol = 0.0;
        if (isfinite(pr) && isfinite(v)) { cpv = pr * v; cvol = v; }

        if (count == period) {
            sum_pv  -= ring_pv[head];
            sum_vol -= ring_vol[head];
        } else {
            ++count;
        }
        ring_pv[head]  = cpv;
        ring_vol[head] = cvol;
        sum_pv  += cpv;
        sum_vol += cvol;
        ++head;
        if (head == period) head = 0;

        const double vwap = (sum_vol != 0.0) ? (sum_pv / sum_vol) : NEO_F64_NAN;
        if (isfinite(pr) && isfinite(vwap)) {
            o[i] = pr - vwap;
        }
    }
}
