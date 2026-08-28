#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ bool valid_high_low(float high, float low) {
    return isfinite(high) && isfinite(low) && high > 0.0f && low > 0.0f;
}

extern "C" __global__ void parkinson_volatility_build_prefix_f64(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    double* __restrict__ prefix_sum,
    int* __restrict__ prefix_invalid
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    prefix_sum[0] = 0.0;
    prefix_invalid[0] = 0;

    double sum = 0.0;
    int invalid = 0;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float h = high[i];
            const float l = low[i];
            if (valid_high_low(h, l)) {
                const double x = log((double)h / (double)l);
                sum += x * x;
            } else {
                invalid += 1;
            }
        }
        prefix_sum[i + 1] = sum;
        prefix_invalid[i + 1] = invalid;
    }
}

extern "C" __global__ void parkinson_volatility_batch_f32(
    const double* __restrict__ prefix_sum,
    const int* __restrict__ prefix_invalid,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out_volatility,
    float* __restrict__ out_variance
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > len) {
        return;
    }

    const int warmup = first_valid + period - 1;
    const int base = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);
    const double denom = ((double)period) * (4.0 * 0.69314718055994530942);

    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < len;
         t += (int)gridDim.x * (int)blockDim.x) {
        float vol_out = nan_f;
        float var_out = nan_f;

        if (t >= warmup) {
            const int end = t + 1;
            const int start = end - period;
            const int invalid = prefix_invalid[end] - prefix_invalid[start];
            if (invalid == 0) {
                double variance = (prefix_sum[end] - prefix_sum[start]) / denom;
                if (variance < 0.0) {
                    variance = 0.0;
                }
                var_out = (float)variance;
                vol_out = sqrtf((float)variance);
            }
        }

        out_volatility[base + t] = vol_out;
        out_variance[base + t] = var_out;
    }
}

extern "C" __global__ void parkinson_volatility_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int period,
    int cols,
    int rows,
    float* __restrict__ out_volatility_tm,
    float* __restrict__ out_variance_tm
) {
    const int s = (int)blockIdx.x;
    if (s >= cols) {
        return;
    }

    const float nan_f = __int_as_float(0x7fffffff);
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        const int idx = t * cols + s;
        out_volatility_tm[idx] = nan_f;
        out_variance_tm[idx] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }
    if (period <= 0 || period > rows) {
        return;
    }

    const int first_valid = first_valids[s];
    if (first_valid < 0 || first_valid >= rows) {
        return;
    }

    const int warmup = first_valid + period - 1;
    if (warmup >= rows) {
        return;
    }

    const double denom = ((double)period) * (4.0 * 0.69314718055994530942);
    double sum = 0.0;
    int invalid = 0;

    for (int t = first_valid; t <= warmup; ++t) {
        const int idx = t * cols + s;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        if (valid_high_low(h, l)) {
            const double x = log((double)h / (double)l);
            sum += x * x;
        } else {
            invalid += 1;
        }
    }

    if (invalid == 0) {
        double variance = sum / denom;
        if (variance < 0.0) {
            variance = 0.0;
        }
        const int idx = warmup * cols + s;
        out_variance_tm[idx] = (float)variance;
        out_volatility_tm[idx] = sqrtf((float)variance);
    }

    for (int t = warmup + 1; t < rows; ++t) {
        const int old_idx = (t - period) * cols + s;
        const float old_h = high_tm[old_idx];
        const float old_l = low_tm[old_idx];
        if (valid_high_low(old_h, old_l)) {
            const double x = log((double)old_h / (double)old_l);
            sum -= x * x;
        } else {
            invalid -= 1;
        }

        const int idx = t * cols + s;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        if (valid_high_low(h, l)) {
            const double x = log((double)h / (double)l);
            sum += x * x;
        } else {
            invalid += 1;
        }

        if (invalid == 0) {
            double variance = sum / denom;
            if (variance < 0.0) {
                variance = 0.0;
            }
            out_variance_tm[idx] = (float)variance;
            out_volatility_tm[idx] = sqrtf((float)variance);
        }
    }
}


// ===========================================================================
// f64 LANE  --  shard S6
//
// CPU reference: `parkinson_compute_into`
// (src/indicators/parkinson_volatility.rs:274), reached from
// `parkinson_volatility_with_kernel` (:403). `parkinson_prepare` pins
// `Kernel::Auto -> Kernel::Scalar` (:266-269), so there is exactly one CPU
// answer and no AVX association to settle.
//
// OUTPUT: `volatility`, which is `OUTPUTS_PARKINSON[0]`
// (registry.rs:1302 -> [OUTPUT_VOLATILITY, OUTPUT_VARIANCE]). `variance` is
// `volatility * volatility` by construction (:232-235); the single-matrix f64
// lane carries one output per indicator, so the primary is what this emits.
//
// FIRST VALID IS NOT THE COMMON RULE. `first_valid_high_low` (:219-223) scans
// for `is_valid_high_low` = `h.is_finite() && l.is_finite() && h > 0 && l > 0`
// (:214-216) -- FINITE AND STRICTLY POSITIVE, not merely non-NaN. An infinite
// high, or a zero/negative price, is skipped by the CPU and would be accepted
// by `AllInputsNonNan`, which would seed the window at a different bar and
// shift the whole series. Declared as
// `F64FirstValidRule::HighLowFiniteAndPositive`.
//
// warm = first + period - 1 (:282). Everything before it is NaN.
//
// THE RING IS NOT NEEDED ON THE DEVICE. The CPU keeps `ring[period]` holding
// `log_range_sq` for the bars in the window purely to subtract the departing
// one (:314-319). The departing bar at step i is i-period, and
// `log_range_sq` is a pure function of `high[i-period]` / `low[i-period]`, so
// recomputing it is BIT-IDENTICAL to reading it back and costs one extra
// log per bar instead of an unbounded per-thread array. That is why this
// kernel declares no `max_period`: there is no per-thread ring to overrun.
// The same argument covers the `invalid` counter, which the CPU maintains
// incrementally and which is likewise a function of the window's bars.
//
// f32 -> f64 audit: the f32 lane above uses `sqrtf` x3 and
// `__int_as_float` x2. Below: `sqrt`, `log`, and the f64 quiet-NaN bit
// pattern. FOUR_LN_2 is `4.0 * std::f64::consts::LN_2`
// (parkinson_volatility.rs:38) written out to full f64 precision rather than
// as a decimal literal rounded for f32. No epsilon exists in this indicator.
// The `variance.max(0.0)` on line 233 is `f64::max`, which returns the
// non-NaN operand -- reproduced with `fmax`, NOT with a comparison, because
// a NaN variance must survive as NaN through the sqrt and not be clamped to
// zero by a comparison that is false against NaN.
// ===========================================================================

static __device__ __forceinline__ double parkinson_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// parkinson_volatility.rs:214-216
static __device__ __forceinline__ bool parkinson_is_valid_hl_f64(double h, double l) {
    return isfinite(h) && isfinite(l) && h > 0.0 && l > 0.0;
}

// parkinson_volatility.rs:226-229
static __device__ __forceinline__ double parkinson_log_range_sq_f64(double h, double l) {
    const double x = log(h / l);
    return x * x;
}

extern "C" __global__
void parkinson_volatility_batch_f64(const double* __restrict__ high,
                                    const double* __restrict__ low,
                                    int n,
                                    const int* __restrict__ periods,
                                    int n_combos,
                                    int first_valid,
                                    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = parkinson_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const int first  = (first_valid < 0) ? 0 : first_valid;

    if (period <= 0 || period > n || first >= n || (n - first) < period) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const int warm = first + period - 1;      // :282
    for (int t = 0; t < n; ++t) row[t] = nan_d;
    if (warm >= n) return;                    // :283-285, CPU leaves the prefix

    // 4.0 * LN_2, f64-exact.
    const double FOUR_LN_2 = 4.0 * 0.693147180559945309417232121458176568;
    // DIVISION, not a reciprocal multiply: the CPU writes
    // `(sum_log_sq / (period as f64)) / FOUR_LN_2` (:233). `sum * (1/period)`
    // is a different rounding for every period that is not a power of two.
    const double period_f   = static_cast<double>(period);

    // Seed: the window [first, first + period), accumulated in ascending bar
    // order exactly as :291-301 / :346-356 do.
    int    invalid     = 0;
    double sum_log_sq  = 0.0;
    for (int j = 0; j < period; ++j) {
        const int i = first + j;
        if (parkinson_is_valid_hl_f64(high[i], low[i])) {
            sum_log_sq += parkinson_log_range_sq_f64(high[i], low[i]);
        } else {
            invalid += 1;
        }
    }

    // :232-235 -- `((sum / period) / FOUR_LN_2).max(0.0)`, then sqrt.
    if (invalid == 0) {
        const double variance = fmax((sum_log_sq / period_f) / FOUR_LN_2, 0.0);
        row[warm] = sqrt(variance);
    } else {
        row[warm] = nan_d;
    }

    for (int i = warm + 1; i < n; ++i) {
        // The bar leaving the window. The CPU reads it from `ring`; this
        // recomputes the identical expression from the identical inputs.
        const int    old_i = i - period;
        const double old_h = high[old_i], old_l = low[old_i];
        if (parkinson_is_valid_hl_f64(old_h, old_l)) {
            sum_log_sq -= parkinson_log_range_sq_f64(old_h, old_l);
        } else {
            invalid -= 1;
        }

        if (parkinson_is_valid_hl_f64(high[i], low[i])) {
            sum_log_sq += parkinson_log_range_sq_f64(high[i], low[i]);
        } else {
            invalid += 1;
        }

        if (invalid == 0) {
            const double variance = fmax((sum_log_sq / period_f) / FOUR_LN_2, 0.0);
            row[i] = sqrt(variance);
        } else {
            row[i] = nan_d;
        }
    }
}
