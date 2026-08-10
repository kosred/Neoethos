#include <cmath>
#include <cstddef>

namespace {
constexpr double OUTPUT_SCALE = 100.0;

struct AtrStreamState {
    int length;
    double alpha;
    double prev_close;
    double rma;
    double warm_sum;
    int warm_count;
    bool seeded;

    __device__ void init(int value) {
        length = value;
        alpha = 1.0 / static_cast<double>(value);
        reset();
    }

    __device__ void reset() {
        prev_close = NAN;
        rma = NAN;
        warm_sum = 0.0;
        warm_count = 0;
        seeded = false;
    }

    __device__ double update(double high, double low, double close, bool* ready) {
        const double tr = isnan(prev_close) ? (high - low) : (fmax(high, prev_close) - fmin(low, prev_close));
        prev_close = close;

        if (!seeded) {
            warm_sum += tr;
            warm_count += 1;
            if (warm_count == length) {
                rma = warm_sum * alpha;
                seeded = true;
                *ready = true;
                return rma;
            }
            *ready = false;
            return NAN;
        }

        rma = fma(alpha, tr - rma, rma);
        *ready = true;
        return rma;
    }
};

__device__ inline bool valid_bar(double high, double low, double source) {
    return isfinite(high) && isfinite(low) && isfinite(source) && high >= low;
}

__device__ inline double clamp_unit(double value) {
    return value < -1.0 ? -1.0 : (value > 1.0 ? 1.0 : value);
}
}

extern "C" __global__ void supertrend_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* source,
    int len,
    const int* lengths,
    const double* mults,
    const int* smooths,
    int rows,
    double* out_oscillator,
    double* out_signal,
    double* out_histogram
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];
    const int smooth = smooths[row];

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_histogram = out_histogram + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = NAN;
        row_signal[i] = NAN;
        row_histogram[i] = NAN;
    }

    if (length <= 0 || smooth <= 0 || !isfinite(mult) || mult <= 0.0) {
        return;
    }

    const double hist_alpha = 2.0 / (static_cast<double>(smooth) + 1.0);
    const double length_f64 = static_cast<double>(length);

    AtrStreamState atr;
    atr.init(length);

    double prev_source = NAN;
    double prev_upper = NAN;
    double prev_lower = NAN;
    double prev_trend = 0.0;
    double ama_prev = NAN;
    bool have_ama = false;
    double hist_prev = NAN;
    bool have_hist = false;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double src = source[i];

        if (!valid_bar(h, l, src)) {
            atr.reset();
            prev_source = NAN;
            prev_upper = NAN;
            prev_lower = NAN;
            prev_trend = 0.0;
            ama_prev = NAN;
            have_ama = false;
            hist_prev = NAN;
            have_hist = false;
            continue;
        }

        bool atr_ready = false;
        const double atr_value = atr.update(h, l, src, &atr_ready);
        if (!atr_ready) {
            prev_source = src;
            continue;
        }

        const double mid = 0.5 * (h + l);
        const double band = atr_value * mult;
        const double up = mid + band;
        const double dn = mid - band;

        const double upper =
            (isfinite(prev_source) && isfinite(prev_upper) && prev_source < prev_upper)
            ? fmin(up, prev_upper)
            : up;
        const double lower =
            (isfinite(prev_source) && isfinite(prev_lower) && prev_source > prev_lower)
            ? fmax(dn, prev_lower)
            : dn;

        const double trend =
            (isfinite(prev_upper) && src > prev_upper) ? 1.0
            : ((isfinite(prev_lower) && src < prev_lower) ? 0.0 : prev_trend);
        const double supertrend = trend * lower + (1.0 - trend) * upper;
        const double width = upper - lower;
        const double osc =
            (isfinite(width) && width != 0.0) ? clamp_unit((src - supertrend) / width) : 0.0;
        const double alpha = (osc * osc) / length_f64;
        const double ama = have_ama ? (ama_prev + alpha * (osc - ama_prev)) : osc;
        const double diff = osc - ama;
        const double hist = have_hist ? (hist_prev + hist_alpha * (diff - hist_prev)) : diff;

        row_oscillator[i] = osc * OUTPUT_SCALE;
        row_signal[i] = ama * OUTPUT_SCALE;
        row_histogram[i] = hist * OUTPUT_SCALE;

        prev_source = src;
        prev_upper = upper;
        prev_lower = lower;
        prev_trend = trend;
        ama_prev = ama;
        have_ama = true;
        hist_prev = hist;
        have_hist = true;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — supertrend_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/supertrend_oscillator.rs:640
 *   supertrend_oscillator_row_fused_checked — the arm
 *   supertrend_oscillator_row_fused (:474) takes whenever the frame is not
 *   all-valid, and the one whose expressions the all-valid twin (:514)
 *   repeats verbatim. Reproducing the checked form therefore reproduces both.
 *
 * Column: output_id "value" resolves to out.oscillator — cpu_batch.rs:12301
 *   accepts "oscillator"/"value". signal and histogram are separate output
 *   ids; their two recurrences are still ADVANCED here, because `ama` is
 *   subtracted from `osc` to form the histogram input and the reset must
 *   clear a consistent state — but only the oscillator is written.
 *
 * PERIOD-INVARIANT: compute_supertrend_oscillator_batch reads source
 *   ("close"), length (10), mult (2.0) and smooth (72) and NEVER period
 *   (cpu_batch.rs:12215-12218). Five swept periods give five identical CPU
 *   columns, so this kernel emits five identical rows.
 *
 * FIRST-VALID IGNORED: the row walks EVERY bar from index 0 and RESETS the
 *   ATR warm-up, both bands, the trend state and both smoothers on a bar that
 *   fails valid_bar (:292 — all three finite AND high >= low). The caller's
 *   first-valid index is never read.
 *
 * Input: high / low / source, and the CPU default source is close
 *   (cpu_batch.rs:12215) — F64InputKind::Hlc.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. A Wilder ATR, two bands that
 *   clamp against their own previous values, a latched trend state and two
 *   EMA-shaped smoothers all carry across bars.
 *
 * ARITHMETIC taken verbatim:
 *   * the true range is built by two COMPARISONS (`>` and `<`) selecting the
 *     wider of the current bar and the previous close (:701-704), NOT by
 *     fmax/fmin — the CPU writes it as an if-chain on values it has already
 *     proved finite, and the tie behaviour is what is preserved.
 *   * the ATR seed is warm_sum * atr_alpha (:711) — a SUM then ONE multiply
 *     by the reciprocal, not a divide.
 *   * the ATR step is atr_alpha.mul_add(true_range - atr, atr) (:720) — ONE
 *     fma over a pre-formed difference. Two roundings total, and writing it
 *     as (atr*(1-a) + a*tr) would be three.
 *   * the band clamps ARE f64::min / f64::max (:730, :736), so fmin/fmax are
 *     used: they return the non-NaN operand where an if-chain would let a NaN
 *     through into the next bar's band.
 *   * clamp_unit is f64::clamp(-1, 1) (:390). `supertrend_oscillator_batch_f64`
 *     writes it as the if-chain at :57-59; the neo kernel below writes it as
 *     fmin(fmax(x,-1),1) at :351, and THAT SUBSTITUTION NEEDS AN ARGUMENT.
 *
 *     The argument this comment used to give was that `f64::clamp` "PANICS on
 *     a NaN bound". True, and irrelevant: the bounds here are the literals
 *     -1.0 and 1.0, so no bound can ever be NaN and the panic can never fire.
 *     Worse, it pointed at the wrong property and would license the same
 *     rewrite somewhere it is NOT safe.
 *
 *     The property that actually differs is the NaN VALUE.
 *     `x.clamp(-1.0, 1.0)` RETURNS NaN for a NaN x — its `if self < min` and
 *     `else if self > max` both compare false, so it falls through to `self`.
 *     `fmin(fmax(NaN, -1.0), 1.0)` returns -1.0, because IEEE fmax/fmin
 *     DISCARD a NaN operand and return the other one. On a NaN input the two
 *     forms disagree by a whole value, not by an ULP.
 *
 *     The substitution is sound here ONLY because `raw` cannot be NaN. :349
 *     admits the divide only when `isfinite(width) && width != 0.0`, and the
 *     bar reached that line through the validity gate above, which requires hi,
 *     lo and src all finite; `atr` is a finite accumulation of finite true
 *     ranges, so `supertrend`, and therefore `src - supertrend`, is finite.
 *     A finite numerator over a finite non-zero denominator is finite, or ±inf
 *     if it overflows — and on ±inf the two forms still agree (both give ±1).
 *     0/0 is the only route to NaN and `width != 0.0` closes it.
 *
 *     IF A FUTURE EDIT LETS A NaN REACH :351, fmin/fmax will silently emit
 *     -1.0 where the CPU emits NaN. Use the :57 if-chain there instead.
 *   * the two smoothers are prev + alpha * (x - prev) (:757, :764) — NOT an
 *     fma, because the CPU line contains none.
 *   * OUTPUT_SCALE is 100.0 (:33) and multiplies the value last (:770 ->
 *     write_oscillator_values).
 *   * there is no epsilon in this column: the only guard is the exact test
 *     width != 0.0.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:12216-12218 (:30-33). */
#define NEO_STO_LENGTH        10
#define NEO_STO_MULT          2.0
#define NEO_STO_SMOOTH        72
#define NEO_STO_OUTPUT_SCALE  100.0

extern "C" __global__
void supertrend_oscillator_neo_batch_f64(const double* __restrict__ high,
                                         const double* __restrict__ low,
                                         const double* __restrict__ source,
                                         int n,
                                         const int* __restrict__ periods,
                                         int n_combos,
                                         int first_valid,
                                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* the mid-series reset reproduces it — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = NEO_STO_LENGTH;
    /* validate_params (:865) refuses length > data_len. */
    if (length > n) return;

    const double mult       = NEO_STO_MULT;
    const double atr_alpha  = 1.0 / (double)length;
    const double hist_alpha = 2.0 / ((double)NEO_STO_SMOOTH + 1.0);
    const double length_f64 = (double)length;

    double prev_close = NEO_F64_NAN;
    double atr = NEO_F64_NAN;
    double warm_sum = 0.0;
    int    warm_count = 0;
    bool   seeded = false;

    double prev_source = NEO_F64_NAN;
    double prev_upper  = NEO_F64_NAN;
    double prev_lower  = NEO_F64_NAN;
    double prev_trend  = 0.0;
    bool   ama_seeded = false, hist_seeded = false;
    double ama_prev = 0.0, hist_prev = 0.0;

    for (int i = 0; i < n; ++i) {
        const double hi  = high[i];
        const double lo  = low[i];
        const double src = source[i];

        if (!(isfinite(hi) && isfinite(lo) && isfinite(src) && hi >= lo)) {
            o[i] = NEO_F64_NAN;
            prev_close = NEO_F64_NAN;
            atr = NEO_F64_NAN;
            warm_sum = 0.0; warm_count = 0; seeded = false;
            prev_source = NEO_F64_NAN;
            prev_upper  = NEO_F64_NAN;
            prev_lower  = NEO_F64_NAN;
            prev_trend  = 0.0;
            ama_seeded = false; hist_seeded = false;
            ama_prev = 0.0; hist_prev = 0.0;
            continue;
        }

        double true_range;
        if (isnan(prev_close)) {
            true_range = hi - lo;
        } else {
            const double up = (hi > prev_close) ? hi : prev_close;
            const double dn = (lo < prev_close) ? lo : prev_close;
            true_range = up - dn;
        }
        prev_close = src;

        if (!seeded) {
            warm_sum += true_range;
            ++warm_count;
            if (warm_count == length) {
                atr = warm_sum * atr_alpha;
                seeded = true;
            } else {
                o[i] = NEO_F64_NAN;
                prev_source = src;
                continue;
            }
        } else {
            atr = fma(atr_alpha, true_range - atr, atr);
        }

        const double mid  = 0.5 * (hi + lo);
        const double band = atr * mult;
        const double up_b = mid + band;
        const double dn_b = mid - band;

        const double upper =
            (isfinite(prev_source) && isfinite(prev_upper) && prev_source < prev_upper)
                ? fmin(up_b, prev_upper) : up_b;
        const double lower =
            (isfinite(prev_source) && isfinite(prev_lower) && prev_source > prev_lower)
                ? fmax(dn_b, prev_lower) : dn_b;

        double trend;
        if      (isfinite(prev_upper) && src > prev_upper) trend = 1.0;
        else if (isfinite(prev_lower) && src < prev_lower) trend = 0.0;
        else                                               trend = prev_trend;

        const double supertrend = trend * lower + (1.0 - trend) * upper;
        const double width = upper - lower;
        double osc;
        if (isfinite(width) && width != 0.0) {
            const double raw = (src - supertrend) / width;
            osc = fmin(fmax(raw, -1.0), 1.0);   /* clamp_unit (:390) */
        } else {
            osc = 0.0;
        }

        const double alpha = (osc * osc) / length_f64;
        double ama;
        if (ama_seeded) ama = ama_prev + alpha * (osc - ama_prev);
        else            { ama = osc; ama_seeded = true; }

        const double diff = osc - ama;
        double hist;
        if (hist_seeded) hist = hist_prev + hist_alpha * (diff - hist_prev);
        else             { hist = diff; hist_seeded = true; }

        o[i] = osc * NEO_STO_OUTPUT_SCALE;

        prev_source = src;
        prev_upper  = upper;
        prev_lower  = lower;
        prev_trend  = trend;
        ama_prev    = ama;
        hist_prev   = hist;
    }
}
