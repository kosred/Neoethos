#include <cmath>
#include <cstdint>

__device__ __forceinline__ bool crossed(double prev_a, double curr_a, double prev_b, double curr_b) {
    return (curr_a > curr_b && prev_a <= prev_b) || (curr_a < curr_b && prev_a >= prev_b);
}

extern "C" __global__ void hull_butterfly_oscillator_batch_f64(
    const double* data,
    int len,
    const int* coeff_lens,
    const double* mults,
    const double* coeffs,
    int max_coeff_len,
    int rows,
    double* out_oscillator,
    double* out_cumulative_mean,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int coeff_len = coeff_lens[row];
    double mult = mults[row];
    if (coeff_len < 2 || coeff_len > max_coeff_len || !isfinite(mult)) {
        return;
    }

    const double nan = NAN;
    const double* row_coeffs = coeffs + static_cast<size_t>(row) * static_cast<size_t>(max_coeff_len);
    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_cumulative_mean =
        out_cumulative_mean + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int segment_index = 0;
    double cumulative_abs = 0.0;
    double prev_hso = 0.0;
    double prev_cmean = 0.0;
    double signal_state = 0.0;
    bool has_prev = false;

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = nan;
        row_cumulative_mean[i] = nan;
        row_signal[i] = nan;

        double value = data[i];
        if (!isfinite(value)) {
            segment_index = 0;
            cumulative_abs = 0.0;
            prev_hso = 0.0;
            prev_cmean = 0.0;
            signal_state = 0.0;
            has_prev = false;
            continue;
        }

        int current_index = segment_index;
        segment_index += 1;
        if (segment_index < coeff_len) {
            continue;
        }

        int window_start = i - coeff_len + 1;
        double hma = 0.0;
        double inv_hma = 0.0;
        for (int j = 0; j < coeff_len; ++j) {
            double coeff = row_coeffs[j];
            hma += data[i - j] * coeff;
            inv_hma += data[window_start + j] * coeff;
        }

        double hso = hma - inv_hma;
        cumulative_abs += fabs(hso);
        if (current_index == 0) {
            continue;
        }

        double cmean = cumulative_abs / static_cast<double>(current_index) * mult;
        if (has_prev) {
            if (crossed(prev_hso, hso, prev_cmean, cmean)
                || crossed(prev_hso, hso, -prev_cmean, -cmean)) {
                signal_state = 0.0;
            } else if (hso < prev_hso && hso > cmean) {
                signal_state = -1.0;
            } else if (hso > prev_hso && hso < -cmean) {
                signal_state = 1.0;
            }
        }

        prev_hso = hso;
        prev_cmean = cmean;
        has_prev = true;
        row_oscillator[i] = hso;
        row_cumulative_mean[i] = cmean;
        row_signal[i] = signal_state;
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2               hull_butterfly_oscillator
 * ---------------------------------------------------------------------------
 * CPU reference: `HullButterflyOscillatorStream::update`,
 * src/indicators/hull_butterfly_oscillator.rs:404, driven bar by bar from
 * `hull_butterfly_oscillator_selected_row_from_slice` (:517), with the
 * coefficient construction `compute_hull_coeffs` (:282).
 *
 * `length` IS the swept parameter (cpu_batch.rs:8769, default 14); `mult` is
 * 2.0 (:8770). The lane emits the OSCILLATOR series -- this indicator's batch
 * does NOT accept `output_id == "value"` (:8751-8762 accepts only
 * "oscillator", "cumulative_mean" and "signal"), so a parity run must ask the
 * CPU for "oscillator" explicitly.
 *
 * WHY THE EXISTING ENTRY POINT COULD NOT BE REUSED. This file already carries
 * `hull_butterfly_oscillator_batch_f64`, which takes `coeff_lens`, `mults`, a
 * `coeffs` MATRIX and `max_coeff_len` from the host and writes THREE output
 * matrices. The lane launches (series, n, periods, n_combos, first_valid, out)
 * and allocates one matrix, so that signature cannot be launched from it.
 *
 * NO PER-THREAD COEFFICIENT ARRAY, so no `max_period` bound and NEVER-OOM by
 * construction. `compute_hull_coeffs` builds a vector of
 * `length + hull_len - 1` doubles by repeated `insert(0, ..)`; every element of
 * it is a closed form, so `neo_hbo_coeff` below evaluates one on demand:
 *
 *   short_len = length / 2,  hull_len = max(floor(sqrt(length)), 1)
 *   den1 = short_len*(short_len+1)/2, den2 = length*(length+1)/2,
 *   den3 = hull_len*(hull_len+1)/2
 *   v_i  = 2.0 * (saturating_sub(short_len, i) / den1) - ((length - i) / den2)
 *   the prepends leave lcwa[j] = 0 for j < hull_len - 1,
 *                       lcwa[j] = v_{(length + hull_len - 2) - j} for the next
 *                                 `length` entries, and 0 for the last
 *                                 `hull_len`
 *   size = length + 2*hull_len - 1, and hull_coeffs[k] is the i = size-1-k
 *   entry of `sum3 / den3` where sum3 walks j = i-hull_len .. i-1 ASCENDING,
 *   weighting by (i - j) -- that order is reproduced because it is a sum of
 *   `hull_len` doubles and the association is load-bearing.
 *
 * SEQUENTIAL, one thread per column: `cumulative_abs` is a running sum over the
 * whole segment (:436), `signal_state` is a carried state machine (:442-452),
 * and `prev_hso`/`prev_cmean` feed the crossing test. None of that is
 * bar-parallel.
 *
 * NO RING BUFFER either: the CPU's ring holds the last `coeff_len` values, and
 * `recent_idx`/`inverse_idx` (:429-430) resolve to `data[t - i]` and
 * `data[t - coeff_len + 1 + i]`. Those are read straight out of the resident
 * series.
 *
 * `!value.is_finite()` RESETS the whole segment (:405-408, reset at :388-396),
 * so an infinity is a reset here and not merely a NaN.
 *
 * FIRST-VALID IS `Ignored`, read off the CPU rather than assumed:
 * `hull_butterfly_oscillator_selected_row_from_slice` fills the row with NaN
 * (:521) and then zips the stream over `data.iter()` from index 0 (:524). The
 * `warmup` at :551 belongs to `hull_butterfly_oscillator_with_kernel`, a
 * different entry point that allocates its own prefix.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:8770 -- `mult` default. */
#define NEO_HBO_MULT 2.0

/* hull_butterfly_oscillator.rs:290-293 -- one raw LCWA term. */
__device__ __forceinline__
static double neo_hbo_v(int i, int short_len, int length, double den1, double den2)
{
    const double sum1 = (double)(short_len > i ? (short_len - i) : 0);  /* saturating_sub */
    const double sum2 = (double)(length - i);
    return 2.0 * (sum1 / den1) - (sum2 / den2);
}

/* hull_butterfly_oscillator.rs:289-297 -- one element of the padded LCWA
 * vector, after both prepend passes. */
__device__ __forceinline__
static double neo_hbo_lcwa(int j, int short_len, int length, int hull_len,
                           double den1, double den2)
{
    if (j < hull_len - 1) return 0.0;
    const int back = j - (hull_len - 1);
    if (back >= length) return 0.0;
    return neo_hbo_v((length + hull_len - 2) - j, short_len, length, den1, den2);
}

/* hull_butterfly_oscillator.rs:300-307 -- coefficient `k` of the final vector,
 * which the CPU builds by prepending, hence i = size - 1 - k. */
__device__ __forceinline__
static double neo_hbo_coeff(int k, int short_len, int length, int hull_len,
                            double den1, double den2, double den3)
{
    const int size = length + 2 * hull_len - 1;
    const int i    = size - 1 - k;
    double sum3 = 0.0;
    for (int j = i - hull_len; j < i; ++j) {
        sum3 += neo_hbo_lcwa(j, short_len, length, hull_len, den1, den2)
                * (double)(i - j);
    }
    return sum3 / den3;
}

extern "C" __global__
void hull_butterfly_oscillator_neo_batch_f64(const double* __restrict__ data,
                                             int n,
                                             const int* __restrict__ periods,
                                             int n_combos,
                                             int first_valid,
                                             double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)first_valid;   /* the CPU row starts at bar 0 -- see the header. */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = periods[combo];
    /* resolve_params, :324-337 -- length < 2 or longer than the data errors. */
    if (length < 2 || length > n) return;

    const int short_len = length / 2;
    int hull_len = (int)floor(sqrt((double)length));
    if (hull_len < 1) hull_len = 1;

    const double den1 = (double)(short_len * (short_len + 1) / 2);
    const double den2 = (double)(length * (length + 1) / 2);
    const double den3 = (double)(hull_len * (hull_len + 1) / 2);

    const int coeff_len = length + hull_len - 1;
    if (coeff_len < 2 || coeff_len > n) return;

    /* prepare, :475-480 -- NotEnoughValidData when no run reaches
     * warmup_period + 1 = coeff_len. The CPU errors and the whole row is NaN;
     * reproduced by scanning the longest finite run first. */
    {
        int run = 0, best = 0;
        for (int i = 0; i < n; ++i) {
            if (isfinite(data[i])) { run += 1; if (run > best) best = run; }
            else run = 0;
        }
        if (best < coeff_len) return;
    }

    double cumulative_abs = 0.0;
    int    segment_index  = 0;
    int    count          = 0;
    double prev_hso       = 0.0;
    double prev_cmean     = 0.0;
    double signal_state   = 0.0;    /* carried; emitted only via "signal" */
    bool   has_prev       = false;

    for (int t = 0; t < n; ++t) {
        const double value = data[t];
        if (!isfinite(value)) {
            /* reset(), :388-396 */
            cumulative_abs = 0.0;
            segment_index  = 0;
            count          = 0;
            prev_hso       = 0.0;
            prev_cmean     = 0.0;
            signal_state   = 0.0;
            has_prev       = false;
            continue;
        }

        if (count < coeff_len) count += 1;            /* :416-418 */
        const int current_index = segment_index;      /* :419 */
        segment_index += 1;                           /* :420 */

        if (count < coeff_len) continue;              /* :422-424 */

        /* :426-433 -- i ascending, two independent accumulators. */
        const int window_start = t - coeff_len + 1;
        double hma = 0.0, inv_hma = 0.0;
        for (int i = 0; i < coeff_len; ++i) {
            const double coeff =
                neo_hbo_coeff(i, short_len, length, hull_len, den1, den2, den3);
            hma     += data[t - i] * coeff;
            inv_hma += data[window_start + i] * coeff;
        }

        const double hso = hma - inv_hma;             /* :435 */
        cumulative_abs += fabs(hso);                  /* :436 */
        if (current_index == 0) continue;             /* :437-439 */

        const double cmean =
            cumulative_abs / (double)current_index * NEO_HBO_MULT;   /* :440 */

        if (has_prev) {
            /* crossed(), :346-348 */
            const bool x1 = (hso > cmean && prev_hso <= prev_cmean)
                         || (hso < cmean && prev_hso >= prev_cmean);
            const bool x2 = (hso > -cmean && prev_hso <= -prev_cmean)
                         || (hso < -cmean && prev_hso >= -prev_cmean);
            if (x1 || x2) {
                signal_state = 0.0;
            } else if (hso < prev_hso && hso > cmean) {
                signal_state = -1.0;
            } else if (hso > prev_hso && hso < -cmean) {
                signal_state = 1.0;
            }
        }

        prev_hso   = hso;
        prev_cmean = cmean;
        has_prev   = true;

        o[t] = hso;                                   /* the "oscillator" field */
    }

    /* `signal_state` drives the "signal" column, which this lane does not
     * emit. It is carried here so the state machine above reads exactly as the
     * CPU's does rather than as a pruned copy of it. */
    (void)signal_state;
}
