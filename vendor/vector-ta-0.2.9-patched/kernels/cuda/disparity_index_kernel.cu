#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define DISP_NEO_EMA_PERIOD 14
#define DISP_NEO_LOOKBACK 14
#define DISP_NEO_SMOOTHING 9

__device__ __forceinline__ void disparity_index_fill_nan_f64(
    double* __restrict__ row,
    int len) {
    for (int index = 0; index < len; ++index) {
        row[index] = NEO_F64_NAN;
    }
}

/*
 * Exact f64 authority shared by the dynamic full ABI and the preserved
 * default primary ABI. Operation order mirrors DisparityIndexStream::update:
 * both EMA recurrences use one fused rounding, the rolling extrema use
 * f64::max/min semantics, and the SMA replacement advances its index before
 * applying `scaled - old` to the sum.
 */
__device__ __forceinline__ void disparity_index_row_f64(
    const double* __restrict__ data,
    int len,
    int ema_period,
    int lookback_period,
    int smoothing_period,
    int smoothing_flag,
    double* __restrict__ disparity_ring,
    double* __restrict__ sma_ring,
    double* __restrict__ row) {
    disparity_index_fill_nan_f64(row, len);
    if (ema_period <= 0 || lookback_period <= 0 || smoothing_period <= 0 ||
        (smoothing_flag != 0 && smoothing_flag != 1)) {
        return;
    }

    const double ema_alpha = 2.0 / (static_cast<double>(ema_period) + 1.0);
    const double ema_beta = 1.0 - ema_alpha;
    const double smoothing_alpha =
        2.0 / (static_cast<double>(smoothing_period) + 1.0);
    const double smoothing_beta = 1.0 - smoothing_alpha;
    const double double_epsilon =
        2.2204460492503130808472633361816e-16;

    int ema_seed_count = 0;
    double ema_seed_sum = 0.0;
    double ema = NEO_F64_NAN;
    bool ema_ready = false;

    int disparity_count = 0;
    int disparity_index = 0;

    int smoothing_seed_count = 0;
    double smoothing_seed_sum = 0.0;
    double smoothed = NEO_F64_NAN;
    bool smoothed_ready = false;

    int sma_count = 0;
    int sma_index = 0;
    double sma_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            ema_seed_count = 0;
            ema_seed_sum = 0.0;
            ema = NEO_F64_NAN;
            ema_ready = false;
            disparity_count = 0;
            disparity_index = 0;
            smoothing_seed_count = 0;
            smoothing_seed_sum = 0.0;
            smoothed = NEO_F64_NAN;
            smoothed_ready = false;
            sma_count = 0;
            sma_index = 0;
            sma_sum = 0.0;
            continue;
        }

        if (!ema_ready) {
            ema_seed_sum += value;
            ema_seed_count += 1;
            if (ema_seed_count < ema_period) {
                continue;
            }
            ema = ema_seed_sum / static_cast<double>(ema_period);
            ema_ready = true;
        } else {
            ema = fma(ema, ema_beta, ema_alpha * value);
        }

        double disparity = NEO_F64_NAN;
        if (!isfinite(ema)) {
            continue;
        }
        if (fabs(ema) <= double_epsilon) {
            if (fabs(value) <= double_epsilon) {
                disparity = 0.0;
            } else {
                continue;
            }
        } else {
            disparity = (value - ema) / ema * 100.0;
        }

        disparity_ring[disparity_index] = disparity;
        disparity_index += 1;
        if (disparity_index == lookback_period) {
            disparity_index = 0;
        }
        if (disparity_count < lookback_period) {
            disparity_count += 1;
        }
        if (disparity_count < lookback_period) {
            continue;
        }

        double high = -CUDART_INF;
        double low = CUDART_INF;
        for (int index = 0; index < lookback_period; ++index) {
            const double window_value = disparity_ring[index];
            high = fmax(high, window_value);
            low = fmin(low, window_value);
        }
        const double scaled =
            !(high > low) ? 50.0 : (disparity - low) / (high - low) * 100.0;

        if (smoothing_flag == 0) {
            if (!smoothed_ready) {
                smoothing_seed_sum += scaled;
                smoothing_seed_count += 1;
                if (smoothing_seed_count < smoothing_period) {
                    continue;
                }
                smoothed =
                    smoothing_seed_sum / static_cast<double>(smoothing_period);
                smoothed_ready = true;
            } else {
                smoothed =
                    fma(smoothed, smoothing_beta, smoothing_alpha * scaled);
            }
            row[i] = smoothed;
        } else if (sma_count < smoothing_period) {
            sma_ring[sma_count] = scaled;
            sma_sum += scaled;
            sma_count += 1;
            if (sma_count == smoothing_period) {
                row[i] = sma_sum / static_cast<double>(smoothing_period);
            }
        } else {
            const double old = sma_ring[sma_index];
            sma_ring[sma_index] = scaled;
            sma_index += 1;
            if (sma_index == smoothing_period) {
                sma_index = 0;
            }
            sma_sum += scaled - old;
            row[i] = sma_sum / static_cast<double>(smoothing_period);
        }
    }
}

extern "C" __global__ void disparity_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ ema_periods,
    const int* __restrict__ lookback_periods,
    const int* __restrict__ smoothing_periods,
    const int* __restrict__ smoothing_flags,
    int n_combos,
    int max_lookback,
    int max_smoothing,
    double* __restrict__ disparity_buffer,
    double* __restrict__ sma_buffer,
    double* __restrict__ out) {
    const int combo_idx =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    double* row =
        out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    if (max_lookback <= 0 || max_smoothing <= 0) {
        disparity_index_fill_nan_f64(row, len);
        return;
    }

    const int ema_period = ema_periods[combo_idx];
    const int lookback_period = lookback_periods[combo_idx];
    const int smoothing_period = smoothing_periods[combo_idx];
    const int smoothing_flag = smoothing_flags[combo_idx];
    if (lookback_period > max_lookback || smoothing_period > max_smoothing) {
        disparity_index_fill_nan_f64(row, len);
        return;
    }
    double* disparity_ring =
        disparity_buffer + static_cast<size_t>(combo_idx) *
                               static_cast<size_t>(max_lookback);
    double* sma_ring =
        sma_buffer + static_cast<size_t>(combo_idx) *
                         static_cast<size_t>(max_smoothing);
    disparity_index_row_f64(
        data,
        len,
        ema_period,
        lookback_period,
        smoothing_period,
        smoothing_flag,
        disparity_ring,
        sma_ring,
        row);
}

/*
 * Preserved generic primary ABI. Its `periods` and `first_valid` arguments
 * remain intentionally ignored because this ABI denotes only the canonical
 * 14:14:9:ema value column. Production RegistryRatio points use the dynamic
 * full entry point above.
 */
extern "C" __global__ void disparity_index_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out) {
    const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || series_len <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;

    double disparity_ring[DISP_NEO_LOOKBACK];
    double sma_ring[DISP_NEO_SMOOTHING];
    double* row =
        out + static_cast<size_t>(combo) * static_cast<size_t>(series_len);
    disparity_index_row_f64(
        data,
        series_len,
        DISP_NEO_EMA_PERIOD,
        DISP_NEO_LOOKBACK,
        DISP_NEO_SMOOTHING,
        0,
        disparity_ring,
        sma_ring,
        row);
}
