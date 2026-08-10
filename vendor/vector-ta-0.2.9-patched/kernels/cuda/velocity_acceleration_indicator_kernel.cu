#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double vai_weighted_past_sum(
    const double* values,
    int length,
    int next,
    int count
) {
    int upto = count < length ? count : length;
    double sum = 0.0;
    for (int lag = 1; lag <= upto; ++lag) {
        int idx = next >= lag ? next - lag : length + next - lag;
        sum += values[idx] / static_cast<double>(lag);
    }
    return sum;
}

__device__ inline void vai_push(double* values, int length, int* next, int* count, double value) {
    if (length <= 0) {
        return;
    }
    values[*next] = value;
    *next += 1;
    if (*next == length) {
        *next = 0;
    }
    if (*count < length) {
        *count += 1;
    }
}

extern "C" __global__ void velocity_acceleration_indicator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooth_lengths,
    int n_combos,
    int max_length,
    int max_smooth_length,
    double* __restrict__ source_histories,
    double* __restrict__ acceleration_histories,
    double* __restrict__ wma_values,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0 || max_smooth_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int smooth_length = smooth_lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* source_history =
        source_histories + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* acceleration_history =
        acceleration_histories + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* wma_history =
        wma_values + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_smooth_length);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length < 2 || length > max_length || smooth_length <= 0 ||
        smooth_length > max_smooth_length) {
        return;
    }

    double harmonic_sum = 0.0;
    for (int lag = 1; lag <= length; ++lag) {
        harmonic_sum += 1.0 / static_cast<double>(lag);
    }
    double inv_length = 1.0 / static_cast<double>(length);
    double wma_denominator =
        static_cast<double>(smooth_length * (smooth_length + 1) / 2);

    int source_next = 0;
    int source_count = 0;
    int acceleration_next = 0;
    int acceleration_count = 0;
    int wma_next = 0;
    int wma_count = 0;
    double wma_sum = 0.0;
    double wma_weighted_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            source_next = 0;
            source_count = 0;
            acceleration_next = 0;
            acceleration_count = 0;
            wma_next = 0;
            wma_count = 0;
            wma_sum = 0.0;
            wma_weighted_sum = 0.0;
            continue;
        }

        double velocity =
            (value * harmonic_sum -
             vai_weighted_past_sum(source_history, length, source_next, source_count)) *
            inv_length;
        vai_push(source_history, length, &source_next, &source_count, value);

        double velocity_avg = CUDART_NAN;
        bool have_velocity_avg = false;
        if (smooth_length == 1) {
            wma_history[0] = velocity;
            wma_next = 0;
            wma_count = 1;
            wma_sum = velocity;
            wma_weighted_sum = velocity;
            velocity_avg = velocity;
            have_velocity_avg = true;
        } else if (wma_count < smooth_length) {
            wma_history[wma_next] = velocity;
            wma_count += 1;
            wma_next += 1;
            if (wma_next == smooth_length) {
                wma_next = 0;
            }
            wma_sum += velocity;
            wma_weighted_sum += static_cast<double>(wma_count) * velocity;
            if (wma_count == smooth_length) {
                velocity_avg = wma_weighted_sum / wma_denominator;
                have_velocity_avg = true;
            }
        } else {
            double old = wma_history[wma_next];
            double previous_sum = wma_sum;
            wma_history[wma_next] = velocity;
            wma_next += 1;
            if (wma_next == smooth_length) {
                wma_next = 0;
            }
            wma_sum = previous_sum - old + velocity;
            wma_weighted_sum =
                wma_weighted_sum - previous_sum + static_cast<double>(smooth_length) * velocity;
            velocity_avg = wma_weighted_sum / wma_denominator;
            have_velocity_avg = true;
        }

        if (!have_velocity_avg) {
            continue;
        }

        double acceleration =
            (velocity_avg * harmonic_sum -
             vai_weighted_past_sum(
                 acceleration_history,
                 length,
                 acceleration_next,
                 acceleration_count
             )) *
            inv_length;
        vai_push(
            acceleration_history,
            length,
            &acceleration_next,
            &acceleration_count,
            velocity_avg
        );
        row[i] = acceleration;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — velocity_acceleration_indicator             (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/velocity_acceleration_indicator.rs
 *   :638 `compute_row_default`            <- the per-bar body reproduced here
 *   :465 `fixed_lag_weighted_past_sum`    (the two-segment ring walk)
 *   :496 `fixed_velocity_value`           (current*H - past) * (1/length)
 *   :508 `fixed_wma_update`               (the incremental 5-tap WMA)
 *
 * WHY A SECOND f64 ENTRY POINT. `velocity_acceleration_indicator_batch_f64`
 * above takes eleven arguments including four scratch pointers. The f64 LANE
 * launches ONE fixed six-argument signature, so it gets its own entry point
 * here rather than a reinterpretation of that one.
 *
 * PERIOD-INVARIANT. cpu_batch.rs:9292 reads `length` (21) and `smooth_length`
 * (5); it never reads `period`.
 *
 * FIRST-VALID IGNORED, and that is the CPU's own behaviour rather than an
 * omission: `compute_row_default` walks the WHOLE series from index 0 and a
 * non-finite bar RESETS every accumulator instead of being skipped, so there
 * is no warmup prefix to align. Passing first_valid in would shift the series.
 *
 * SEQUENTIAL, one thread per column: three rings carry across bars and the
 * WMA sum is incremental, so the accumulation order is load-bearing.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_VAI_LENGTH 21
#define NEO_VAI_SMOOTH 5

// velocity_acceleration_indicator.rs:465 `fixed_lag_weighted_past_sum`.
// Split into the two segments the CPU uses (the `next - lag` run, then the
// wrapped `N + next - lag` run) so the ADDITION ORDER is identical.
static __device__ __forceinline__ double neo_vai_past_sum(
    const double* __restrict__ values, int next, int count)
{
    const int upto = count < NEO_VAI_LENGTH ? count : NEO_VAI_LENGTH;
    const int direct = upto < next ? upto : next;
    double sum = 0.0;
    for (int lag = 1; lag <= direct; ++lag) {
        sum += values[next - lag] / (double)lag;
    }
    for (int lag = direct + 1; lag <= upto; ++lag) {
        sum += values[NEO_VAI_LENGTH + next - lag] / (double)lag;
    }
    return sum;
}

extern "C" __global__
void velocity_acceleration_indicator_neo_batch_f64(
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
    (void)periods;      // PERIOD-INVARIANT
    (void)first_valid;  // FIRST-VALID IGNORED -- see header

    if (len <= 0) return;

    double harmonic_sum = 0.0;
    for (int i = 1; i <= NEO_VAI_LENGTH; ++i) harmonic_sum += 1.0 / (double)i;
    const double inv_length = 1.0 / (double)NEO_VAI_LENGTH;

    // The CPU initialises every ring to 0.0, NOT NaN -- and it matters,
    // because `fixed_lag_weighted_past_sum` reads only the first `count`
    // slots, so a NaN fill would be equivalent here but a 0.0 fill is what
    // the reference states and is what a future `count` bug would expose.
    double source_history[NEO_VAI_LENGTH];
    for (int k = 0; k < NEO_VAI_LENGTH; ++k) source_history[k] = 0.0;
    int source_next = 0, source_count = 0;

    double wma_values[NEO_VAI_SMOOTH];
    for (int k = 0; k < NEO_VAI_SMOOTH; ++k) wma_values[k] = 0.0;
    int wma_next = 0, wma_count = 0;
    double wma_sum = 0.0, wma_weighted_sum = 0.0;

    double acceleration_history[NEO_VAI_LENGTH];
    for (int k = 0; k < NEO_VAI_LENGTH; ++k) acceleration_history[k] = 0.0;
    int acceleration_next = 0, acceleration_count = 0;

    // velocity_acceleration_indicator.rs:33 -- (5 * 6 / 2) as f64.
    const double wma_denom = (double)((NEO_VAI_SMOOTH * (NEO_VAI_SMOOTH + 1)) / 2);

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            source_next = 0;  source_count = 0;
            wma_next = 0;     wma_count = 0;
            wma_sum = 0.0;    wma_weighted_sum = 0.0;
            acceleration_next = 0; acceleration_count = 0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double velocity =
            (value * harmonic_sum
             - neo_vai_past_sum(source_history, source_next, source_count))
            * inv_length;

        source_history[source_next] = value;
        source_next += 1;
        if (source_next == NEO_VAI_LENGTH) source_next = 0;
        if (source_count < NEO_VAI_LENGTH) source_count += 1;

        // fixed_wma_update, both branches, in the CPU's exact order.
        bool have_avg;
        double velocity_avg = 0.0;
        if (wma_count < NEO_VAI_SMOOTH) {
            wma_values[wma_next] = velocity;
            wma_count += 1;
            wma_next += 1;
            if (wma_next == NEO_VAI_SMOOTH) wma_next = 0;
            wma_sum += velocity;
            wma_weighted_sum += (double)wma_count * velocity;
            if (wma_count < NEO_VAI_SMOOTH) {
                have_avg = false;
            } else {
                have_avg = true;
                velocity_avg = wma_weighted_sum / wma_denom;
            }
        } else {
            const double old = wma_values[wma_next];
            const double prev_sum = wma_sum;
            wma_values[wma_next] = velocity;
            wma_next += 1;
            if (wma_next == NEO_VAI_SMOOTH) wma_next = 0;
            wma_sum = prev_sum - old + velocity;
            wma_weighted_sum =
                wma_weighted_sum - prev_sum + (double)NEO_VAI_SMOOTH * velocity;
            have_avg = true;
            velocity_avg = wma_weighted_sum / wma_denom;
        }

        if (!have_avg) {
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double acceleration =
            (velocity_avg * harmonic_sum
             - neo_vai_past_sum(acceleration_history, acceleration_next,
                                acceleration_count))
            * inv_length;

        acceleration_history[acceleration_next] = velocity_avg;
        acceleration_next += 1;
        if (acceleration_next == NEO_VAI_LENGTH) acceleration_next = 0;
        if (acceleration_count < NEO_VAI_LENGTH) acceleration_count += 1;

        o[i] = acceleration;
    }
}
