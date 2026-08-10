#include <cmath>
#include <cstddef>

extern "C" __global__ void andean_oscillator_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ signal_lengths,
    int rows,
    double* __restrict__ out_bull,
    double* __restrict__ out_bear,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int signal_length = signal_lengths[row];

    double* row_bull = out_bull + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear = out_bear + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_bull[i] = NAN;
        row_bear[i] = NAN;
        row_signal[i] = NAN;
    }

    if (length <= 0 || signal_length <= 0) {
        return;
    }

    const double alpha = 2.0 / (static_cast<double>(length) + 1.0);
    const double signal_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);

    bool initialized = false;
    double up1 = NAN;
    double up2 = NAN;
    double dn1 = NAN;
    double dn2 = NAN;
    double signal = NAN;

    for (int i = 0; i < len; ++i) {
        const double open_i = open[i];
        const double close_i = close[i];
        if (!isfinite(open_i) || !isfinite(close_i)) {
            continue;
        }

        const double close_sq = close_i * close_i;
        const double open_sq = open_i * open_i;

        if (!initialized) {
            up1 = close_i;
            up2 = close_sq;
            dn1 = close_i;
            dn2 = close_sq;
            signal = 0.0;
            initialized = true;
            row_bull[i] = 0.0;
            row_bear[i] = 0.0;
            row_signal[i] = 0.0;
            continue;
        }

        const double up1_next = up1 - (up1 - close_i) * alpha;
        const double up2_next = up2 - (up2 - close_sq) * alpha;
        const double dn1_next = dn1 + (close_i - dn1) * alpha;
        const double dn2_next = dn2 + (close_sq - dn2) * alpha;

        up1 = fmax(close_i, fmax(open_i, up1_next));
        up2 = fmax(close_sq, fmax(open_sq, up2_next));
        dn1 = fmin(close_i, fmin(open_i, dn1_next));
        dn2 = fmin(close_sq, fmin(open_sq, dn2_next));

        const double bull = sqrt(fmax(dn2 - dn1 * dn1, 0.0));
        const double bear = sqrt(fmax(up2 - up1 * up1, 0.0));
        const double signal_input = fmax(bull, bear);
        signal = isfinite(signal)
            ? signal_alpha * signal_input + (1.0 - signal_alpha) * signal
            : signal_input;

        row_bull[i] = bull;
        row_bear[i] = bear;
        row_signal[i] = signal;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE - andean_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/andean_oscillator.rs:341
 *             `compute_andean_oscillator_into`, driving `AndeanCore::update`
 *             (:299).
 *
 * COLUMN: `bull`. This indicator has NO "value" output - its registry row
 * declares `OUTPUTS_ANDEAN_OSCILLATOR = [bull, bear, signal]`
 * (registry.rs:1457) and the CPU batch matches only those three
 * (cpu_batch.rs:10046). The lane emits the FIRST declared output, `bull`,
 * the same convention shard 3 used for `di -> plus` and `aso -> bulls`.
 *
 * INPUT: (open, close). No high, no low. The lane carries this on the Ohlc4
 * shape - four `const double*` in the order open/high/low/close - and this
 * kernel reads the first and the fourth. A two-pointer shape would have been
 * indistinguishable at the ABI from (high, low) and (price, volume), so the
 * four-pointer shape is used and the two unread series are named `(void)`
 * below rather than silently ignored.
 *
 * FIRST-VALID: `first_valid_pair` (:244) - OPEN and CLOSE both `is_finite` at
 * the same index. Not the Hlc rule and not the Ohlc4 rule: high and low are
 * never scanned, so a frame whose high starts late must NOT shift this
 * series. Registered as `F64FirstValidRule::OpenCloseFinite`.
 *
 * WARMUP: `alloc_with_nan_prefix(len, first)` (:392) - NaN strictly before
 * `first`. After it there is no warmup window at all: the first valid bar
 * seeds the four extremes and emits 0.0.
 *
 * PERIOD-INVARIANT. The CPU batch reads `length` (50) and `signal_length` (9)
 * and never `period`.
 *
 * NaN SEMANTICS: every one of the eight `max`/`min` calls at :323-330 is
 * `f64::max` / `f64::min`, which return the NON-NaN operand, and `.max(0.0)`
 * at :328-329 is what keeps a tiny negative variance out of `sqrt`. `fmax` /
 * `fmin` match. A comparison chain would let a NaN survive into `up1`, and
 * because `up1` feeds itself every later bar, one NaN would poison the rest
 * of the series.
 *
 * SEQUENTIAL, one thread per combo column: four exponential extremes plus a
 * signal EMA, all first-order recurrences.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void andean_oscillator_neo_batch_f64(
    const double* __restrict__ open,
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
    (void)periods;              /* period-invariant */
    (void)high; (void)low;      /* andean reads open and close only */

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    const double alpha        = 2.0 / (50.0 + 1.0);  /* length 50, :238 */
    const double signal_alpha = 2.0 / (9.0 + 1.0);   /* signal_length 9, :239 */

    if (first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    for (int i = 0; i < first_valid; ++i) o[i] = NEO_F64_NAN;

    bool   initialized = false;
    double up1 = NEO_F64_NAN, up2 = NEO_F64_NAN;
    double dn1 = NEO_F64_NAN, dn2 = NEO_F64_NAN;
    double sig = NEO_F64_NAN;

    for (int i = first_valid; i < len; ++i) {
        const double op = open[i];
        const double cl = close[i];
        if (!isfinite(op) || !isfinite(cl)) {
            /* `update` returns the NaN triple WITHOUT touching the state
               (:300) - the extremes are carried, not reset. */
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double close_sq = cl * cl;
        const double open_sq  = op * op;

        if (!initialized) {
            up1 = cl; up2 = close_sq;
            dn1 = cl; dn2 = close_sq;
            sig = 0.0;
            initialized = true;
            o[i] = 0.0;             /* bull on the seed bar is 0.0 (:314) */
            continue;
        }

        const double nup1 = up1 - (up1 - cl) * alpha;
        const double nup2 = up2 - (up2 - close_sq) * alpha;
        const double ndn1 = dn1 + (cl - dn1) * alpha;
        const double ndn2 = dn2 + (close_sq - dn2) * alpha;

        up1 = fmax(cl,       fmax(op,      nup1));
        up2 = fmax(close_sq, fmax(open_sq, nup2));
        dn1 = fmin(cl,       fmin(op,      ndn1));
        dn2 = fmin(close_sq, fmin(open_sq, ndn2));

        const double bull = sqrt(fmax(dn2 - dn1 * dn1, 0.0));
        const double bear = sqrt(fmax(up2 - up1 * up1, 0.0));
        const double signal_input = fmax(bull, bear);
        sig = isfinite(sig)
                  ? signal_alpha * signal_input + (1.0 - signal_alpha) * sig
                  : signal_input;

        o[i] = bull;
    }
}
