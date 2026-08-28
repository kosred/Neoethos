#include <cuda_runtime.h>
#include <math.h>

/* ===========================================================================
 * NEOETHOS f64 Ehlers Undersampled Double Moving Average
 *
 * Scalar authority: `compute_eudma_into` -> `EudmaCore::update` ->
 * `HannFilterState::update`. Production supplies the exact CPU-built Hann
 * weights/norms, owns runtime-sized fast/slow rings, and emits both canonical
 * outputs from one sequential thread per admitted RegistryRatio tuple.
 *
 * The preserved generic primary ABI remains fixed at CPU defaults and calls
 * the same complete row function. It is compatibility-only; canonical
 * production always enters the dynamic two-output ABI below.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#ifndef NEO_F64_PI
#define NEO_F64_PI 3.14159265358979323846
#endif

#define NEO_EUDMA_FAST_LENGTH   6
#define NEO_EUDMA_SLOW_LENGTH   12
#define NEO_EUDMA_SAMPLE_LENGTH 5

__device__ __forceinline__ double ehlers_undersampled_double_moving_average_hann_update_f64(
    double sample,
    int length,
    const double* __restrict__ weights,
    double norm,
    double* __restrict__ ring,
    int* __restrict__ head,
    int* __restrict__ count)
{
    const bool full = (*count == length);
    ring[*head] = sample;
    *head += 1;
    if (*head == length) *head = 0;
    if (!full) *count += 1;

    double acc = 0.0;
    int idx = (*head == 0) ? (length - 1) : (*head - 1);
    if (full) {
        for (int offset = 0; offset < length; ++offset) {
            const double current = ring[idx];
            const double value = isfinite(current) ? current : 0.0;
            acc += weights[offset] * value;
            idx = (idx == 0) ? (length - 1) : (idx - 1);
        }
    } else {
        for (int offset = 0; offset < length; ++offset) {
            double value;
            if (offset < *count) {
                const double current = ring[idx];
                value = isfinite(current) ? current : 0.0;
            } else {
                value = 0.0;
            }
            acc += weights[offset] * value;
            idx = (idx == 0) ? (length - 1) : (idx - 1);
        }
    }
    return (norm == 0.0) ? 0.0 : (acc / norm);
}

__device__ __forceinline__ void ehlers_undersampled_double_moving_average_row_f64(
    const double* __restrict__ prices,
    int n,
    int first_valid,
    int fast_length,
    int slow_length,
    int sample_length,
    const double* __restrict__ fast_weights,
    double fast_norm,
    const double* __restrict__ slow_weights,
    double slow_norm,
    double* __restrict__ fast_ring,
    double* __restrict__ slow_ring,
    double* __restrict__ fast_out,
    double* __restrict__ slow_out)
{
    for (int i = 0; i < n; ++i) {
        if (fast_out != nullptr) fast_out[i] = NEO_F64_NAN;
        if (slow_out != nullptr) slow_out[i] = NEO_F64_NAN;
    }
    if (n <= 0 || first_valid < 0 || first_valid >= n ||
        fast_length <= 0 || slow_length <= 0 || sample_length <= 0) {
        return;
    }

    for (int i = 0; i < fast_length; ++i) fast_ring[i] = 0.0;
    for (int i = 0; i < slow_length; ++i) slow_ring[i] = 0.0;
    int fast_head = 0;
    int fast_count = 0;
    int slow_head = 0;
    int slow_count = 0;
    int sample_countdown = 0;
    double last_sample = NEO_F64_NAN;

    for (int i = 0; i < n; ++i) {
        const double value = prices[i];
        double sampled;
        if (sample_countdown == 0) {
            sample_countdown = sample_length - 1;
            sampled = value;
        } else if (isfinite(last_sample)) {
            sample_countdown -= 1;
            sampled = last_sample;
        } else {
            sample_countdown -= 1;
            sampled = 0.0;
        }
        last_sample = sampled;

        const double fast = ehlers_undersampled_double_moving_average_hann_update_f64(
            sampled,
            fast_length,
            fast_weights,
            fast_norm,
            fast_ring,
            &fast_head,
            &fast_count);
        const double slow = ehlers_undersampled_double_moving_average_hann_update_f64(
            sampled,
            slow_length,
            slow_weights,
            slow_norm,
            slow_ring,
            &slow_head,
            &slow_count);
        if (i >= first_valid) {
            if (fast_out != nullptr) fast_out[i] = fast;
            if (slow_out != nullptr) slow_out[i] = slow;
        }
    }
}

extern "C" __global__
void ehlers_undersampled_double_moving_average_outputs_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ parameter_rows,
    const double* __restrict__ norms,
    const int* __restrict__ weight_offsets,
    const double* __restrict__ weights,
    int n_rows,
    int first_valid,
    int max_fast_length,
    int max_slow_length,
    double* __restrict__ fast_scratch,
    double* __restrict__ slow_scratch,
    double* __restrict__ fast_out,
    double* __restrict__ slow_out)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_rows || n <= 0) return;

    const int* __restrict__ tuple = parameter_rows + (size_t)row * 3U;
    const int fast_length = tuple[0];
    const int slow_length = tuple[1];
    const int sample_length = tuple[2];
    double* __restrict__ fast_row = fast_out + (size_t)row * (size_t)n;
    double* __restrict__ slow_row = slow_out + (size_t)row * (size_t)n;

    if (fast_length <= 0 || slow_length <= 0 || sample_length <= 0 ||
        fast_length > max_fast_length || slow_length > max_slow_length) {
        for (int i = 0; i < n; ++i) {
            fast_row[i] = NEO_F64_NAN;
            slow_row[i] = NEO_F64_NAN;
        }
        return;
    }

    const int fast_offset = weight_offsets[(size_t)row * 2U];
    const int slow_offset = weight_offsets[(size_t)row * 2U + 1U];
    ehlers_undersampled_double_moving_average_row_f64(
        prices,
        n,
        first_valid,
        fast_length,
        slow_length,
        sample_length,
        weights + fast_offset,
        norms[(size_t)row * 2U],
        weights + slow_offset,
        norms[(size_t)row * 2U + 1U],
        fast_scratch + (size_t)row * (size_t)max_fast_length,
        slow_scratch + (size_t)row * (size_t)max_slow_length,
        fast_row,
        slow_row);
}

extern "C" __global__
void ehlers_undersampled_double_moving_average_neo_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;

    double fast_weights[NEO_EUDMA_FAST_LENGTH];
    double slow_weights[NEO_EUDMA_SLOW_LENGTH];
    double fast_norm = 0.0;
    double slow_norm = 0.0;
    for (int i = 1; i <= NEO_EUDMA_FAST_LENGTH; ++i) {
        const double weight =
            1.0 - cos(2.0 * NEO_F64_PI * (double)i /
                      (double)(NEO_EUDMA_FAST_LENGTH + 1));
        fast_weights[i - 1] = weight;
        fast_norm += weight;
    }
    for (int i = 1; i <= NEO_EUDMA_SLOW_LENGTH; ++i) {
        const double weight =
            1.0 - cos(2.0 * NEO_F64_PI * (double)i /
                      (double)(NEO_EUDMA_SLOW_LENGTH + 1));
        slow_weights[i - 1] = weight;
        slow_norm += weight;
    }
    double fast_ring[NEO_EUDMA_FAST_LENGTH];
    double slow_ring[NEO_EUDMA_SLOW_LENGTH];
    ehlers_undersampled_double_moving_average_row_f64(
        prices,
        n,
        first_valid,
        NEO_EUDMA_FAST_LENGTH,
        NEO_EUDMA_SLOW_LENGTH,
        NEO_EUDMA_SAMPLE_LENGTH,
        fast_weights,
        fast_norm,
        slow_weights,
        slow_norm,
        fast_ring,
        slow_ring,
        out + (size_t)combo * (size_t)n,
        nullptr);
}
