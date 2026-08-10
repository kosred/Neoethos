#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void historical_volatility_batch_f32(
    const float* __restrict__ data,
    int len,
    const int* __restrict__ lookbacks,
    const float* __restrict__ annualization_scales,
    int n_combos,
    float* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int lookback = lookbacks[combo_idx];
    float annualization_scale = annualization_scales[combo_idx];
    float* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (lookback <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        if (t < lookback) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t + 1 - lookback;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = start; i <= t; ++i) {
            float prev = data[i - 1];
            float curr = data[i];
            if (!isfinite(prev) || !isfinite(curr) || prev == 0.0f) {
                valid = false;
                break;
            }
            double ret = ((static_cast<double>(curr) / static_cast<double>(prev)) - 1.0) * 100.0;
            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        double inv_lb = 1.0 / static_cast<double>(lookback);
        double mean = sum * inv_lb;
        double variance = sumsq * inv_lb - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        row[t] = static_cast<float>(sqrt(variance) * static_cast<double>(annualization_scale));
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `historical_volatility.rs::hv_row_from_prefix` (l.393).
//   warmup      = first + lookback - 1              (NOT `t < lookback`)
//   valid_count = lookback returns in the window, else NaN
//   mean        = sum * inv_lb                       (reciprocal, not divide)
//   variance    = (sumsq * inv_lb - mean*mean).max(0.0)
//   value       = sqrt(variance) * annualization_scale
// `annualization_scale` is `annualization_days.sqrt()` and the CPU default for
// `annualization_days` is 250.0 (`historical_volatility.rs:111`), so the
// registry entry point bakes `sqrt(250.0)` — computed at run time in f64 rather
// than written as a decimal literal, so it is the correctly rounded double.
//
// f32 -> f64 audit: the f32 kernel already accumulated in `double`, so the
// arithmetic body is unchanged; what changed is (a) the INPUT is now double, so
// `curr/prev` is a true f64 ratio instead of an f32 ratio widened after the
// fact, (b) the `prev == 0.0f` test is now `prev == 0.0`, (c) the output is no
// longer truncated to float, (d) the warmup was WRONG — `t < lookback` ignores
// `first_valid` entirely and emits values over a leading NaN region, and
// (e) `variance < 0.0 -> 0.0` becomes `fmax(variance, 0.0)` to match
// `f64::max`, which returns the non-NaN operand; the if-chain lets a NaN
// through.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double hv_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void historical_volatility_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookbacks,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    const double nan_d = hv_qnan_f64();
    // CPU default: annualization_days = 250.0, scale = sqrt(250.0).
    const double annualization_scale = sqrt(250.0);

    int lookback = lookbacks[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (lookback <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = nan_d;
        }
        return;
    }

    const int warm = first_valid + lookback - 1;
    const double inv_lb = 1.0 / static_cast<double>(lookback);

    for (int t = 0; t < len; ++t) {
        if (t < warm) {
            row[t] = nan_d;
            continue;
        }

        int start = t + 1 - lookback;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = start; i <= t; ++i) {
            if (i == 0) {          // no return is defined at index 0
                valid = false;
                break;
            }
            double prev = data[i - 1];
            double curr = data[i];
            if (!isfinite(prev) || !isfinite(curr) || prev == 0.0) {
                valid = false;
                break;
            }
            double ret = ((curr / prev) - 1.0) * 100.0;
            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            row[t] = nan_d;
            continue;
        }

        double mean = sum * inv_lb;
        double variance = fmax(sumsq * inv_lb - mean * mean, 0.0);
        row[t] = sqrt(variance) * annualization_scale;
    }
}
