#include <cmath>
#include <cstdint>

static __device__ inline double vacd_ring_get(
    const double* ring,
    int head,
    int count,
    int capacity,
    int lookback
) {
    if (lookback <= 0 || lookback > count) {
        return 0.0;
    }
    int idx = head - lookback;
    while (idx < 0) {
        idx += capacity;
    }
    return ring[idx];
}

static __device__ inline void vacd_ring_push(
    double* ring,
    int capacity,
    int* head,
    int* count,
    double value
) {
    ring[*head] = value;
    *head += 1;
    if (*head >= capacity) {
        *head = 0;
    }
    if (*count < capacity) {
        *count += 1;
    }
}

static __device__ inline double vacd_compute_velocity_current(
    const double* history,
    int head,
    int count,
    int capacity,
    double current,
    int length
) {
    double sum = 0.0;
    for (int i = 1; i <= length; ++i) {
        double prev =
            i <= count ? vacd_ring_get(history, head, count, capacity, i) : 0.0;
        sum += (current - prev) / static_cast<double>(i);
    }
    return sum / static_cast<double>(length);
}

static __device__ inline double vacd_compute_wma_tail(
    const double* history,
    int head,
    int capacity,
    int period
) {
    double numerator = 0.0;
    double denominator = 0.0;
    int start = head - period;
    while (start < 0) {
        start += capacity;
    }
    for (int offset = 0; offset < period; ++offset) {
        int idx = start + offset;
        if (idx >= capacity) {
            idx -= capacity;
        }
        double weight = static_cast<double>(offset + 1);
        numerator += history[idx] * weight;
        denominator += weight;
    }
    return numerator / denominator;
}

static __device__ inline double vacd_classify_signal(double vacd, double prev_vacd_nz) {
    if (vacd > 0.0) {
        return vacd > prev_vacd_nz ? 2.0 : 1.0;
    }
    if (vacd < 0.0) {
        return vacd < prev_vacd_nz ? -2.0 : -1.0;
    }
    return 0.0;
}

extern "C" __global__ void velocity_acceleration_convergence_divergence_indicator_batch_f64(
    const double* data,
    int len,
    const int* lengths,
    const int* smooth_lengths,
    int rows,
    int max_length,
    int max_smooth_length,
    double* source_history,
    double* raw_velocity_history,
    double* velocity_avg_history,
    double* out_vacd,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int smooth_length = smooth_lengths[row];
    if (length < 2 || smooth_length <= 0 || max_length <= 0 || max_smooth_length <= 0) {
        return;
    }

    const double nan = NAN;

    double* row_source = source_history + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_raw = raw_velocity_history
        + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_velocity_avg =
        velocity_avg_history + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_vacd = out_vacd + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int source_head = 0;
    int source_count = 0;
    int raw_head = 0;
    int raw_count = 0;
    int velocity_avg_head = 0;
    int velocity_avg_count = 0;
    double prev_vacd = nan;
    bool has_prev_vacd = false;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            source_head = 0;
            source_count = 0;
            raw_head = 0;
            raw_count = 0;
            velocity_avg_head = 0;
            velocity_avg_count = 0;
            prev_vacd = nan;
            has_prev_vacd = false;
            row_vacd[i] = nan;
            row_signal[i] = nan;
            continue;
        }

        double raw_velocity = vacd_compute_velocity_current(
            row_source,
            source_head,
            source_count,
            max_length,
            value,
            length);
        vacd_ring_push(row_source, max_length, &source_head, &source_count, value);
        vacd_ring_push(row_raw, max_smooth_length, &raw_head, &raw_count, raw_velocity);

        if (raw_count < smooth_length) {
            row_vacd[i] = nan;
            row_signal[i] = nan;
            continue;
        }

        double velocity_avg =
            vacd_compute_wma_tail(row_raw, raw_head, max_smooth_length, smooth_length);
        double acceleration = vacd_compute_velocity_current(
            row_velocity_avg,
            velocity_avg_head,
            velocity_avg_count,
            max_length,
            velocity_avg,
            length);
        double vacd = velocity_avg - acceleration;
        double signal = vacd_classify_signal(vacd, has_prev_vacd ? prev_vacd : 0.0);

        vacd_ring_push(
            row_velocity_avg, max_length, &velocity_avg_head, &velocity_avg_count, velocity_avg);
        prev_vacd = vacd;
        has_prev_vacd = true;

        row_vacd[i] = vacd;
        row_signal[i] = signal;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — velocity_acceleration_convergence_divergence_indicator
 *                                                                 (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle:
 *   src/indicators/velocity_acceleration_convergence_divergence_indicator.rs
 *   :610 `compute_row_default`               <- the per-bar body
 *   :571 `compute_velocity_default_current`  (count-guarded lag walk)
 *   :586 `compute_velocity_default_current_full`
 *   :600 `compute_wma_default_tail_full`
 *   :544 `fixed_history_at_full`
 *
 * OUTPUT COLUMN. cpu_batch.rs:8256 maps `output_id == "value"` onto `vacd`,
 * not onto `signal`. This kernel emits `vacd`.
 *
 * PERIOD-INVARIANT (cpu_batch.rs:8229 reads `length`/`smooth_length`).
 * FIRST-VALID IGNORED: `compute_row_default` walks from index 0 and RESETS on
 * a non-finite bar, exactly like its `velocity_acceleration_indicator` twin.
 *
 * THE COUNT-GUARDED / FULL SPLIT IS NOT COSMETIC. `compute_velocity_default_
 * current` reads `fixed_history_at`, which returns 0.0 when `count < lag`;
 * `..._full` skips that test entirely. The CPU picks between them on
 * `count == DEFAULT_LENGTH`, so at steady state both are the same walk -- but
 * during warmup they differ, and folding them would change the first 21 bars.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_VACD_LENGTH 21
#define NEO_VACD_SMOOTH 5

// fixed_history_at_full (:544): idx = next >= lag ? next - lag : N + next - lag.
static __device__ __forceinline__ double neo_vacd_at_full(
    const double* __restrict__ v, int n, int next, int lag)
{
    const int idx = (next >= lag) ? (next - lag) : (n + next - lag);
    return v[idx];
}

// compute_velocity_default_current (:571) and ..._full (:586). `full` is the
// `count == N` case, where `fixed_history_at` degenerates to `..._full`.
static __device__ __forceinline__ double neo_vacd_velocity(
    const double* __restrict__ history, int next, int count, double current,
    bool full)
{
    double sum = 0.0;
    for (int i = 1; i <= NEO_VACD_LENGTH; ++i) {
        const double prev = (full || count >= i)
            ? neo_vacd_at_full(history, NEO_VACD_LENGTH, next, i)
            : 0.0;
        sum += (current - prev) / (double)i;
    }
    return sum / (double)NEO_VACD_LENGTH;
}

// classify_signal (:374) is not emitted here -- "value" is `vacd`.

extern "C" __global__
void velocity_acceleration_convergence_divergence_indicator_neo_batch_f64(
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

    double source_history[NEO_VACD_LENGTH];
    for (int k = 0; k < NEO_VACD_LENGTH; ++k) source_history[k] = 0.0;
    int source_next = 0, source_count = 0;

    double raw_history[NEO_VACD_SMOOTH];
    for (int k = 0; k < NEO_VACD_SMOOTH; ++k) raw_history[k] = 0.0;
    int raw_next = 0, raw_count = 0;

    double velocity_avg_history[NEO_VACD_LENGTH];
    for (int k = 0; k < NEO_VACD_LENGTH; ++k) velocity_avg_history[k] = 0.0;
    int velocity_avg_next = 0, velocity_avg_count = 0;

    // DEFAULT_WMA_DENOMINATOR (:30) -- (5 * 6 / 2) as f64.
    const double wma_denom = (double)((NEO_VACD_SMOOTH * (NEO_VACD_SMOOTH + 1)) / 2);

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            source_next = 0; source_count = 0;
            raw_next = 0;    raw_count = 0;
            velocity_avg_next = 0; velocity_avg_count = 0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double raw_velocity = neo_vacd_velocity(
            source_history, source_next, source_count, value,
            source_count == NEO_VACD_LENGTH);

        source_history[source_next] = value;
        source_next += 1;
        if (source_next == NEO_VACD_LENGTH) source_next = 0;
        if (source_count < NEO_VACD_LENGTH) source_count += 1;

        raw_history[raw_next] = raw_velocity;
        raw_next += 1;
        if (raw_next == NEO_VACD_SMOOTH) raw_next = 0;
        if (raw_count < NEO_VACD_SMOOTH) raw_count += 1;

        if (raw_count < NEO_VACD_SMOOTH) {
            o[i] = NEO_F64_NAN;
            continue;
        }

        // compute_wma_default_tail_full (:600): offset 0..4, weight offset+1,
        // value at lag (SMOOTH - offset). Oldest tap first.
        double numerator = 0.0;
        for (int offset = 0; offset < NEO_VACD_SMOOTH; ++offset) {
            const double weight = (double)(offset + 1);
            const double v = neo_vacd_at_full(raw_history, NEO_VACD_SMOOTH,
                                              raw_next, NEO_VACD_SMOOTH - offset);
            numerator += v * weight;
        }
        const double velocity_avg = numerator / wma_denom;

        const double acceleration = neo_vacd_velocity(
            velocity_avg_history, velocity_avg_next, velocity_avg_count,
            velocity_avg, velocity_avg_count == NEO_VACD_LENGTH);

        o[i] = velocity_avg - acceleration;

        velocity_avg_history[velocity_avg_next] = velocity_avg;
        velocity_avg_next += 1;
        if (velocity_avg_next == NEO_VACD_LENGTH) velocity_avg_next = 0;
        if (velocity_avg_count < NEO_VACD_LENGTH) velocity_avg_count += 1;
    }
}
