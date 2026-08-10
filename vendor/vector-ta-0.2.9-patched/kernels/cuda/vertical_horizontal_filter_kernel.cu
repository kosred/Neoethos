#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void vertical_horizontal_filter_batch_f32(
    const float* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    float* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    float* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        if (t + 1 < length) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t + 1 - length;
        bool valid = true;
        float highest = -CUDART_INF_F;
        float lowest = CUDART_INF_F;
        float denom = 0.0f;

        for (int i = start; i <= t; ++i) {
            float value = data[i];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            if (value > highest) {
                highest = value;
            }
            if (value < lowest) {
                lowest = value;
            }
        }

        if (valid) {
            for (int i = start; i <= t; ++i) {
                if (i == 0) {
                    valid = false;
                    break;
                }
                float prev = data[i - 1];
                float curr = data[i];
                if (!isfinite(prev) || !isfinite(curr)) {
                    valid = false;
                    break;
                }
                denom += fabsf(curr - prev);
            }
        }

        if (!valid || !(denom > 0.0f) || !isfinite(highest) || !isfinite(lowest)) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        row[t] = fabsf(highest - lowest) / denom;
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `vertical_horizontal_filter.rs` — `vhf_value()` (l.428) over a
// `length`-bar window, with `valid_value` / `valid_change` deciding validity.
// Warmup is `first_valid + length - 1`, matching `get_warmup_period()`.
//
// f32 -> f64 audit for this file:
//   * `float`     -> `double` on every data pointer and every local.
//   * `0.0f`      -> `0.0`.
//   * `fabsf`     -> `fabs`.
//   * `CUDART_NAN_F` -> a double quiet NaN built from the f64 bit pattern; the
//     f32 payload `0x7fc00000` is NOT a double NaN and reinterpreting it would
//     produce a denormal, not a NaN.
//   * `-CUDART_INF_F` / `CUDART_INF_F` -> `-INFINITY` / `INFINITY` (double).
//   * No epsilon in this file: the guard is `denom > 0.0`, an exact zero test,
//     which is precision-independent. Nothing to re-derive.
//   * The `> highest` / `< lowest` chain is guarded by an explicit
//     `isfinite(value)` break BEFORE the comparison, so no NaN can reach the
//     comparison and survive it. That is why this file does not need fmax/fmin.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double vhf_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void vertical_horizontal_filter_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    const double nan_d = vhf_qnan_f64();
    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = nan_d;
        }
        return;
    }

    // CPU warmup: `first + length - 1`.
    const int warm = first_valid + length - 1;

    for (int t = 0; t < len; ++t) {
        if (t < warm) {
            row[t] = nan_d;
            continue;
        }

        int start = t + 1 - length;
        bool valid = true;
        double highest = -INFINITY;
        double lowest = INFINITY;
        double denom = 0.0;

        for (int i = start; i <= t; ++i) {
            double value = data[i];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            if (value > highest) {
                highest = value;
            }
            if (value < lowest) {
                lowest = value;
            }
        }

        if (valid) {
            for (int i = start; i <= t; ++i) {
                if (i == 0) {
                    valid = false;
                    break;
                }
                double prev = data[i - 1];
                double curr = data[i];
                if (!isfinite(prev) || !isfinite(curr)) {
                    valid = false;
                    break;
                }
                denom += fabs(curr - prev);
            }
        }

        if (!valid || !(denom > 0.0) || !isfinite(highest) || !isfinite(lowest)) {
            row[t] = nan_d;
            continue;
        }

        row[t] = fabs(highest - lowest) / denom;
    }
}
