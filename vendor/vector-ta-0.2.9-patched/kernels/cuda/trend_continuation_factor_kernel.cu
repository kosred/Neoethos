#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void trend_continuation_factor_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    int max_length,
    double* __restrict__ plus_buffer,
    double* __restrict__ minus_buffer,
    double* __restrict__ out_plus,
    double* __restrict__ out_minus
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row_plus = out_plus + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_minus = out_minus + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* plus_ring =
        plus_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* minus_ring =
        minus_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row_plus[i] = CUDART_NAN;
        row_minus[i] = CUDART_NAN;
    }

    if (length <= 0 || length > max_length) {
        return;
    }

    bool have_prev = false;
    bool have_plus_cf = false;
    bool have_minus_cf = false;
    double prev = CUDART_NAN;
    double plus_cf = 0.0;
    double minus_cf = 0.0;
    int comparisons_seen = 0;
    int head = 0;
    double sum_plus = 0.0;
    double sum_minus = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            have_prev = false;
            have_plus_cf = false;
            have_minus_cf = false;
            prev = CUDART_NAN;
            plus_cf = 0.0;
            minus_cf = 0.0;
            comparisons_seen = 0;
            head = 0;
            sum_plus = 0.0;
            sum_minus = 0.0;
            continue;
        }

        if (!have_prev) {
            prev = value;
            have_prev = true;
            continue;
        }

        double change = value - prev;
        double plus_change = change > 0.0 ? change : 0.0;
        double minus_change = change < 0.0 ? -change : 0.0;

        double next_plus_cf = plus_change == 0.0
            ? 0.0
            : plus_change + (have_plus_cf ? plus_cf : 1.0);
        double next_minus_cf = minus_change == 0.0
            ? 0.0
            : minus_change + (have_minus_cf ? minus_cf : 1.0);

        have_plus_cf = true;
        have_minus_cf = true;
        plus_cf = next_plus_cf;
        minus_cf = next_minus_cf;
        prev = value;

        double plus = plus_change - next_minus_cf;
        double minus = minus_change - next_plus_cf;

        if (comparisons_seen < length) {
            plus_ring[comparisons_seen] = plus;
            minus_ring[comparisons_seen] = minus;
            sum_plus += plus;
            sum_minus += minus;
            comparisons_seen += 1;
            if (comparisons_seen == length) {
                row_plus[i] = sum_plus;
                row_minus[i] = sum_minus;
            }
            continue;
        }

        double old_plus = plus_ring[head];
        double old_minus = minus_ring[head];
        plus_ring[head] = plus;
        minus_ring[head] = minus;
        sum_plus += plus - old_plus;
        sum_minus += minus - old_minus;
        head += 1;
        if (head == length) {
            head = 0;
        }

        row_plus[i] = sum_plus;
        row_minus[i] = sum_minus;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — trend_continuation_factor                   (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/trend_continuation_factor.rs
 *   :412 trend_continuation_factor_compute_with_buffers  <- the per-bar body
 *   :224 trend_continuation_factor_prepare
 *   :471 trend_continuation_factor_with_kernel
 *
 * OUTPUT COLUMN: cpu_batch.rs:12370 maps output_id == "value" onto plus_tcf.
 * This kernel emits plus_tcf.
 *
 * PERIOD-INVARIANT (cpu_batch.rs:12345 reads "length", default 35).
 *
 * FIRST-VALID IGNORED, and that is the CPU behaviour: the compute loop runs
 * for i in 0..data.len() and OVERWRITES the NaN prefix alloc_with_nan_prefix
 * laid down, with a non-finite bar taking the reset branch. Starting at
 * first_valid instead would agree today and diverge the moment a hole appears
 * mid-series.
 *
 * SEQUENTIAL, one thread per column: plus_cf / minus_cf are a recurrence
 * (plus_change + previous cf) and the window sum is incremental.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_TCF_LENGTH 35

extern "C" __global__
void trend_continuation_factor_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;
    (void)first_valid;

    if (len <= 0) return;

    const int length = NEO_TCF_LENGTH;

    double plus_buffer[NEO_TCF_LENGTH];
    for (int k = 0; k < NEO_TCF_LENGTH; ++k) plus_buffer[k] = 0.0;

    double prev = 0.0;
    bool has_prev = false;
    double plus_cf = 0.0, minus_cf = 0.0;
    bool has_cf = false;
    int comparisons_seen = 0;
    int head = 0;
    double sum_plus = 0.0;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            has_prev = false;
            has_cf = false;
            comparisons_seen = 0;
            head = 0;
            sum_plus = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        if (!has_prev) {
            prev = value;
            has_prev = true;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double change = value - prev;
        prev = value;

        const double plus_change  = (change > 0.0) ?  change : 0.0;
        const double minus_change = (change < 0.0) ? -change : 0.0;
        const double cf_seed_plus  = has_cf ? plus_cf  : 1.0;
        const double cf_seed_minus = has_cf ? minus_cf : 1.0;

        const double next_plus_cf  =
            (plus_change  == 0.0) ? 0.0 : (plus_change  + cf_seed_plus);
        const double next_minus_cf =
            (minus_change == 0.0) ? 0.0 : (minus_change + cf_seed_minus);

        plus_cf  = next_plus_cf;
        minus_cf = next_minus_cf;
        has_cf = true;

        // plus_tcf only -- minus_change - next_plus_cf feeds the minus_tcf
        // column, which is NOT what output_id "value" selects.
        const double plus = plus_change - next_minus_cf;

        if (comparisons_seen < length) {
            plus_buffer[comparisons_seen] = plus;
            sum_plus += plus;
            comparisons_seen += 1;
            o[i] = (comparisons_seen < length) ? NEO_F64_NAN : sum_plus;
            continue;
        }

        const double old_plus = plus_buffer[head];
        plus_buffer[head] = plus;
        sum_plus += plus - old_plus;
        head += 1;
        if (head == length) head = 0;

        o[i] = sum_plus;
    }
}
