#include <cmath>
#include <cstdint>

extern "C" __global__ void vwap_zscore_with_signals_batch_f64(
    const double* close,
    const double* volume,
    int len,
    const int* lengths,
    const double* upper_bottoms,
    const double* lower_bottoms,
    int rows,
    int max_length,
    double* pv_values,
    double* vol_values,
    int* pv_valid,
    double* dev_values,
    int* dev_valid,
    double* out_zvwap,
    double* out_support,
    double* out_resistance
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    double upper_bottom = upper_bottoms[row];
    double lower_bottom = lower_bottoms[row];
    if (length <= 0 || max_length <= 0 || !isfinite(upper_bottom) || !isfinite(lower_bottom)) {
        return;
    }

    const double nan = NAN;
    double* row_pv_values =
        pv_values + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_vol_values =
        vol_values + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* row_pv_valid = pv_valid + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_dev_values =
        dev_values + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* row_dev_valid = dev_valid + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_zvwap = out_zvwap + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_support = out_support + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_resistance =
        out_resistance + static_cast<size_t>(row) * static_cast<size_t>(len);

    int idx = 0;
    int count = 0;
    int valid_count = 0;
    double pv_sum = 0.0;
    double vol_sum = 0.0;

    int dev_idx = 0;
    int dev_count = 0;
    int dev_valid_count = 0;
    double dev_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        row_zvwap[i] = nan;
        row_support[i] = nan;
        row_resistance[i] = nan;

        if (count >= length) {
            int old_idx = idx;
            if (row_pv_valid[old_idx] != 0) {
                valid_count -= 1;
                pv_sum -= row_pv_values[old_idx];
                vol_sum -= row_vol_values[old_idx];
            }
        } else {
            count += 1;
        }

        double c = close[i];
        double v = volume[i];
        if (isfinite(c) && isfinite(v) && v >= 0.0) {
            double pv = c * v;
            row_pv_values[idx] = pv;
            row_vol_values[idx] = v;
            row_pv_valid[idx] = 1;
            valid_count += 1;
            pv_sum += pv;
            vol_sum += v;
        } else {
            row_pv_values[idx] = 0.0;
            row_vol_values[idx] = 0.0;
            row_pv_valid[idx] = 0;
        }
        idx += 1;
        if (idx == length) {
            idx = 0;
        }

        if (dev_count >= length) {
            int old_idx = dev_idx;
            if (row_dev_valid[old_idx] != 0) {
                dev_valid_count -= 1;
                dev_sum -= row_dev_values[old_idx];
            }
        } else {
            dev_count += 1;
        }

        double mean = nan;
        if (count >= length && valid_count == length && vol_sum > 0.0) {
            mean = pv_sum / vol_sum;
            double dev = (c - mean) * (c - mean);
            row_dev_values[dev_idx] = dev;
            row_dev_valid[dev_idx] = 1;
            dev_valid_count += 1;
            dev_sum += dev;
        } else {
            row_dev_values[dev_idx] = 0.0;
            row_dev_valid[dev_idx] = 0;
        }
        dev_idx += 1;
        if (dev_idx == length) {
            dev_idx = 0;
        }

        if (dev_count < length || dev_valid_count != length || !isfinite(mean)) {
            continue;
        }

        double variance = dev_sum / static_cast<double>(length);
        if (variance < 0.0) {
            variance = 0.0;
        }
        double sd = sqrt(variance);
        if (!isfinite(sd) || sd <= 0.0) {
            continue;
        }

        double zvwap = (c - mean) / sd;
        row_zvwap[i] = zvwap;
        row_support[i] = zvwap < lower_bottom ? 1.0 : 0.0;
        row_resistance[i] = zvwap > upper_bottom ? 1.0 : 0.0;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — vwap_zscore_with_signals
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/vwap_zscore_with_signals.rs:553
 *   vwap_zscore_with_signals_output_row_from_slices with field = Zvwap. That
 *   is the function the batch actually drives (cpu_batch.rs:13491 ->
 *   vwap_zscore_with_signals_output_into_slice, :803), not the three-output
 *   row; it writes a value at EVERY index, NaN included.
 *
 * Column: output_id "value" resolves to zvwap — cpu_batch.rs:13454 accepts
 *   "zvwap"/"value".
 *
 * PERIOD-INVARIANT: the batch reads length (20), upper_bottom (2.5) and
 *   lower_bottom (-2.5) and NEVER period (cpu_batch.rs:13477-13479). The
 *   zvwap column does not read either bound at all — they only select the two
 *   signal columns.
 *
 * FIRST-VALID IGNORED: the row walks EVERY bar from index 0. An invalid bar is
 *   not skipped — it is entered into the ring as a zero with its valid flag
 *   clear (:499-503), which is what keeps valid_count from reaching `length`
 *   and suppresses the output for the next `length` bars. Starting at a
 *   first-valid index would give a different ring and a different series.
 *
 * Input: (close, volume) — extract_close_volume_input(.., "close")
 *   (cpu_batch.rs:13451) — F64InputKind::CloseVolume.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Two INCREMENTAL rings carry
 *   state: (pv_sum, vol_sum) over the last `length` bars, and (dev_sum) over
 *   the last `length` squared deviations — and the deviation ring is fed by
 *   the mean the first ring produced, so the two are chained.
 *
 * ARITHMETIC taken verbatim:
 *   * both rings SUBTRACT THE OUTGOING ENTRY FIRST, at the top of the bar,
 *     and only then add the incoming one (:483-489, :510-517). That is the
 *     CPU's rounding order and reversing it drifts.
 *   * variance is (dev_sum / length).max(0.0) (:540) — f64::max, so fmax is
 *     used: a NaN there must not survive into the sqrt.
 *   * the validity test on a bar is close finite AND volume finite AND
 *     volume >= 0.0 (:403) — the sign test is part of it, not an add-on.
 *   * there is no epsilon in this column: the guards are the exact tests
 *     vol_sum > 0.0 and sd > 0.0, reproduced as written.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Default from cpu_batch.rs:13477. It sizes two per-thread rings, so the bound
 * belongs to the COMPILED kernel. */
#define NEO_VZS_LENGTH 20

extern "C" __global__
void vwap_zscore_with_signals_neo_batch_f64(const double* __restrict__ close,
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
    (void)first_valid; /* the ring itself encodes the warmup — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = NEO_VZS_LENGTH;

    /* vwap_zscore_with_signals_prepare (:695) refuses length > len. */
    if (length > n) return;

    double pv_values[NEO_VZS_LENGTH], vol_values[NEO_VZS_LENGTH];
    unsigned char pv_valid[NEO_VZS_LENGTH];
    for (int k = 0; k < length; ++k) { pv_values[k] = 0.0; vol_values[k] = 0.0; pv_valid[k] = 0; }
    int idx = 0, count = 0, valid_count = 0;
    double pv_sum = 0.0, vol_sum = 0.0;

    double dev_values[NEO_VZS_LENGTH];
    unsigned char dev_valid[NEO_VZS_LENGTH];
    for (int k = 0; k < length; ++k) { dev_values[k] = 0.0; dev_valid[k] = 0; }
    int dev_idx = 0, dev_count = 0, dev_valid_count = 0;
    double dev_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        if (count >= length) {
            const int old_idx = idx;
            if (pv_valid[old_idx] != 0) {
                if (valid_count > 0) --valid_count;
                pv_sum  -= pv_values[old_idx];
                vol_sum -= vol_values[old_idx];
            }
        } else {
            ++count;
        }

        const double c = close[i];
        const double v = volume[i];
        if (isfinite(c) && isfinite(v) && v >= 0.0) {
            const double pv = c * v;
            pv_values[idx]  = pv;
            vol_values[idx] = v;
            pv_valid[idx]   = 1;
            ++valid_count;
            pv_sum  += pv;
            vol_sum += v;
        } else {
            pv_values[idx]  = 0.0;
            vol_values[idx] = 0.0;
            pv_valid[idx]   = 0;
        }
        ++idx;
        if (idx == length) idx = 0;

        if (dev_count >= length) {
            const int old_idx = dev_idx;
            if (dev_valid[old_idx] != 0) {
                if (dev_valid_count > 0) --dev_valid_count;
                dev_sum -= dev_values[old_idx];
            }
        } else {
            ++dev_count;
        }

        double mean = NEO_F64_NAN;
        if (count >= length && valid_count == length && vol_sum > 0.0) {
            mean = pv_sum / vol_sum;
            const double dev = (c - mean) * (c - mean);
            dev_values[dev_idx] = dev;
            dev_valid[dev_idx]  = 1;
            ++dev_valid_count;
            dev_sum += dev;
        } else {
            dev_values[dev_idx] = 0.0;
            dev_valid[dev_idx]  = 0;
        }
        ++dev_idx;
        if (dev_idx == length) dev_idx = 0;

        double value = NEO_F64_NAN;
        if (dev_count >= length && dev_valid_count == length && isfinite(mean)) {
            const double variance = fmax(dev_sum / (double)length, 0.0);
            const double sd = sqrt(variance);
            if (isfinite(sd) && sd > 0.0) {
                value = (c - mean) / sd;
            }
        }
        o[i] = value;
    }
}
