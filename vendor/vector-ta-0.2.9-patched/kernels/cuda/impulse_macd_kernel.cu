#include <cmath>
#include <cstdint>

extern "C" __global__ void impulse_macd_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* length_mas,
    const int* length_signals,
    int rows,
    int max_signal_length,
    double* signal_buf,
    double* out_md,
    double* out_hist,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length_ma = length_mas[row];
    int length_signal = length_signals[row];
    if (length_ma <= 0 || length_signal <= 0 || max_signal_length <= 0) {
        return;
    }

    const double nan = NAN;
    double* row_signal_buf =
        signal_buf + static_cast<size_t>(row) * static_cast<size_t>(max_signal_length);
    double* row_md = out_md + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_hist = out_hist + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int hi_count = 0;
    double hi_sum = 0.0;
    double hi_value = nan;
    bool hi_ready = false;

    int lo_count = 0;
    double lo_sum = 0.0;
    double lo_value = nan;
    bool lo_ready = false;

    double ema_alpha = 2.0 / (static_cast<double>(length_ma) + 1.0);
    double ema1_value = 0.0;
    bool ema1_has = false;
    double ema2_value = 0.0;
    bool ema2_has = false;

    int signal_head = 0;
    int signal_len = 0;
    double signal_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        row_md[i] = nan;
        row_hist[i] = nan;
        row_signal[i] = nan;

        if (!(isfinite(h) && isfinite(l) && isfinite(c)) || h < l) {
            hi_count = 0;
            hi_sum = 0.0;
            hi_value = nan;
            hi_ready = false;
            lo_count = 0;
            lo_sum = 0.0;
            lo_value = nan;
            lo_ready = false;
            ema1_value = 0.0;
            ema1_has = false;
            ema2_value = 0.0;
            ema2_has = false;
            signal_head = 0;
            signal_len = 0;
            signal_sum = 0.0;
            continue;
        }

        double src = (h + l + c) / 3.0;

        if (length_ma == 1) {
            hi_value = h;
            hi_ready = true;
        } else if (!hi_ready) {
            hi_sum += h;
            hi_count += 1;
            if (hi_count == length_ma) {
                hi_value = hi_sum / static_cast<double>(length_ma);
                hi_ready = true;
            }
        } else {
            double p = static_cast<double>(length_ma);
            hi_value = (hi_value * (p - 1.0) + h) / p;
        }

        if (length_ma == 1) {
            lo_value = l;
            lo_ready = true;
        } else if (!lo_ready) {
            lo_sum += l;
            lo_count += 1;
            if (lo_count == length_ma) {
                lo_value = lo_sum / static_cast<double>(length_ma);
                lo_ready = true;
            }
        } else {
            double p = static_cast<double>(length_ma);
            lo_value = (lo_value * (p - 1.0) + l) / p;
        }

        double ema1 = ema1_has ? ema_alpha * src + (1.0 - ema_alpha) * ema1_value : src;
        ema1_value = ema1;
        ema1_has = true;

        double ema2 = ema2_has ? ema_alpha * ema1 + (1.0 - ema_alpha) * ema2_value : ema1;
        ema2_value = ema2;
        ema2_has = true;

        double mi = ema1 + (ema1 - ema2);
        double md = 0.0;
        if (hi_ready && lo_ready) {
            if (mi > hi_value) {
                md = mi - hi_value;
            } else if (mi < lo_value) {
                md = mi - lo_value;
            }
        }

        double signal_value = nan;
        if (length_signal == 1) {
            row_signal_buf[0] = md;
            signal_len = 1;
            signal_sum = md;
            signal_value = md;
        } else if (signal_len < length_signal) {
            row_signal_buf[signal_len] = md;
            signal_len += 1;
            signal_sum += md;
            if (signal_len == length_signal) {
                signal_value = signal_sum / static_cast<double>(length_signal);
            }
        } else {
            double old = row_signal_buf[signal_head];
            row_signal_buf[signal_head] = md;
            signal_head += 1;
            if (signal_head == length_signal) {
                signal_head = 0;
            }
            signal_sum += md - old;
            signal_value = signal_sum / static_cast<double>(length_signal);
        }

        row_md[i] = md;
        row_signal[i] = signal_value;
        row_hist[i] = isfinite(signal_value) ? (md - signal_value) : nan;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — impulse_macd
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/impulse_macd.rs:527 `impulse_macd_compute_into`,
 *   driving `ImpulseMacdStream::update_reset_on_nan` (:463) -> `update` (:440).
 *   The driver walks EVERY bar from 0 and RESETS the whole cascade on an
 *   invalid bar, so `first_valid` is not read here — the reset reproduces it.
 *
 * Column: output_id "value" / "impulse_macd" -> `out.impulse_macd`, which the
 *   stream calls `md` (cpu_batch.rs:13044).
 *
 * PERIOD-INVARIANT: `compute_impulse_macd_batch` (cpu_batch.rs:13026-13027)
 *   reads `length_ma` (34) and `length_signal` (9) and NEVER `period`.
 *
 * Input: high / low / close — F64InputKind::Hlc.
 *
 * Shape: ONE THREAD PER COLUMN. Five recurrences run in lockstep — two Wilder
 *   SMMAs on high and low, two chained EMAs on hlc3, and an SMA on the result —
 *   and the SMA input is the previous stage output, so nothing here is
 *   bar-parallel.
 *
 * Two arithmetic details taken verbatim rather than tidied:
 *   * SMMA (:281) seeds on the MEAN of the first `period` values, then
 *     `(value*(p-1) + x) / p`. That is TWO roundings in a specific shape; the
 *     Wilder form `(x - v).mul_add(1/p, v)` is one and would drift.
 *   * The EMA (:324) is `alpha.mul_add(x, (1 - alpha) * prev)` — ONE fma with
 *     the product on the addend side. Not `prev + alpha*(x - prev)`.
 *   The `md` selector (:447) requires BOTH SMMAs to be ready; while either is
 *   still warming, md is 0.0, not NaN.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:13026-13027. `signal_sma` keeps a ring of
 * `length_signal` values, so its bound belongs to the compiled kernel. */
#define NEO_IMACD_LENGTH_MA     34
#define NEO_IMACD_LENGTH_SIGNAL 9

extern "C" __global__
void impulse_macd_neo_batch_f64(const double* __restrict__ high,
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
    (void)first_valid;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    const int    pma   = NEO_IMACD_LENGTH_MA;
    const int    psig  = NEO_IMACD_LENGTH_SIGNAL;
    const double pma_f = (double)pma;
    const double alpha = 2.0 / (pma_f + 1.0);

    /* SmmaState (:260) x2 */
    int    hi_count = 0, lo_count = 0;
    double hi_sum = 0.0, lo_sum = 0.0;
    double hi_val = NEO_F64_NAN, lo_val = NEO_F64_NAN;
    bool   hi_ready = false, lo_ready = false;
    /* EmaState (:309) x2 */
    bool   e1_set = false, e2_set = false;
    double e1_val = 0.0, e2_val = 0.0;
    /* SmaState (:343) */
    double sig_buf[NEO_IMACD_LENGTH_SIGNAL];
    for (int i = 0; i < psig; ++i) sig_buf[i] = 0.0;
    int    sig_head = 0, sig_len = 0;
    double sig_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i];
        /* valid_bar for impulse_macd is the three-way finite test its
         * `first_valid_bar` uses; a hole resets the cascade (:469). */
        if (!(isfinite(h) && isfinite(l) && isfinite(c))) {
            hi_count = lo_count = 0; hi_sum = lo_sum = 0.0;
            hi_val = lo_val = NEO_F64_NAN; hi_ready = lo_ready = false;
            e1_set = e2_set = false; e1_val = e2_val = 0.0;
            for (int k = 0; k < psig; ++k) sig_buf[k] = 0.0;
            sig_head = 0; sig_len = 0; sig_sum = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double src = (h + l + c) / 3.0;

        /* hi_smma.update(high) */
        bool hi_some = false;
        if (pma == 1) { hi_val = h; hi_ready = true; hi_some = true; }
        else if (!hi_ready) {
            hi_sum += h; ++hi_count;
            if (hi_count == pma) { hi_val = hi_sum / pma_f; hi_ready = true; hi_some = true; }
        } else {
            hi_val = (hi_val * (pma_f - 1.0) + h) / pma_f; hi_some = true;
        }

        /* lo_smma.update(low) */
        bool lo_some = false;
        if (pma == 1) { lo_val = l; lo_ready = true; lo_some = true; }
        else if (!lo_ready) {
            lo_sum += l; ++lo_count;
            if (lo_count == pma) { lo_val = lo_sum / pma_f; lo_ready = true; lo_some = true; }
        } else {
            lo_val = (lo_val * (pma_f - 1.0) + l) / pma_f; lo_some = true;
        }

        /* ema1 over src, ema2 over ema1 */
        const double ema1 = e1_set ? fma(alpha, src, (1.0 - alpha) * e1_val) : src;
        e1_val = ema1; e1_set = true;
        const double ema2 = e2_set ? fma(alpha, ema1, (1.0 - alpha) * e2_val) : ema1;
        e2_val = ema2; e2_set = true;

        const double mi = ema1 + (ema1 - ema2);

        double md;
        if (hi_some && lo_some && mi > hi_val)      md = mi - hi_val;
        else if (hi_some && lo_some && mi < lo_val) md = mi - lo_val;
        else                                        md = 0.0;

        /* signal_sma.update(md) — value not emitted, but its ring must advance
         * in step so the OTHER outputs of this stream stay reproducible. */
        if (psig == 1) { sig_buf[0] = md; sig_len = 1; sig_sum = md; }
        else if (sig_len < psig) {
            sig_buf[sig_len] = md; ++sig_len; sig_sum += md;
        } else {
            const double old = sig_buf[sig_head];
            sig_buf[sig_head] = md;
            sig_head = sig_head + 1; if (sig_head == psig) sig_head = 0;
            sig_sum += md - old;
        }

        o[i] = md;
    }
}
