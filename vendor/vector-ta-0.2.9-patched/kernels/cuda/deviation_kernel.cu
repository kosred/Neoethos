#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }


struct twof { float hi, lo; };


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}

__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}

__device__ __forceinline__ twof make_twof(float hi, float lo) { return {hi, lo}; }

__device__ __forceinline__ twof twof_add(twof x, twof y) {
    float s, e; two_sum(x.hi, y.hi, s, e);
    float t = x.lo + y.lo;
    float sh, sl; quick_two_sum(s, e + t, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_sub(twof x, twof y) {
    float s, e; two_sum(x.hi, -y.hi, s, e);
    float t = x.lo - y.lo;
    float sh, sl; quick_two_sum(s, e + t, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_scale(twof x, float k) {
    float p, e; two_prod(x.hi, k, p, e);
    e = fmaf(x.lo, k, e);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_sqr(twof x) {

    float p, e; two_prod(x.hi, x.hi, p, e);
    e = fmaf(2.0f * x.hi, x.lo, e) + (x.lo * x.lo);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ float twof_to_f(twof x) { return x.hi + x.lo; }


__device__ __forceinline__ twof ld_twof(const float2* __restrict__ a, int idx) {
    float2 v = a[idx];
    return make_twof(v.x, v.y);
}

extern "C" __global__ void deviation_build_prefix_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    float2* __restrict__ prefix_sum,
    float2* __restrict__ prefix_sum_sq,
    int* __restrict__ prefix_nan)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    twof sum = make_twof(0.0f, 0.0f);
    twof sum_sq = make_twof(0.0f, 0.0f);
    int nan_count = 0;

    prefix_sum[0] = make_float2(0.0f, 0.0f);
    prefix_sum_sq[0] = make_float2(0.0f, 0.0f);
    prefix_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float v = data[i];
            if (isnan(v)) {
                nan_count += 1;
            } else {
                const twof x = make_twof(v, 0.0f);
                sum = twof_add(sum, x);
                sum_sq = twof_add(sum_sq, twof_sqr(x));
            }
        }
        prefix_sum[i + 1] = make_float2(sum.hi, sum.lo);
        prefix_sum_sq[i + 1] = make_float2(sum_sq.hi, sum_sq.lo);
        prefix_nan[i + 1] = nan_count;
    }
}


extern "C" __global__ void deviation_batch_f32(
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const size_t row_off = static_cast<size_t>(combo) * static_cast<size_t>(len);
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);
    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = dev_nan();
        if (t >= warm) {
            const int start = t + 1 - period;
            bool ok = true;
            if (any_nan_since_first) {
                ok = (prefix_nan[t + 1] - prefix_nan[start]) == 0;
            }
            if (ok) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    const float2 ps_e  = prefix_sum[t + 1];
                    const float2 ps_s  = prefix_sum[start];
                    const float2 ps2_e = prefix_sum_sq[t + 1];
                    const float2 ps2_s = prefix_sum_sq[start];

                    const float sum  = (ps_e.x  - ps_s.x)  + (ps_e.y  - ps_s.y);
                    const float sum2 = (ps2_e.x - ps2_s.x) + (ps2_e.y - ps2_s.y);

                    const float mean = sum * inv_den;
                    const float ex2  = sum2 * inv_den;
                    float var_f = fmaf(-mean, mean, ex2);
                    if (var_f < 0.0f) var_f = 0.0f;
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void deviation_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float*        __restrict__ out_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + period - 1;
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series;
        float out_val = dev_nan();
        if (t >= warm) {
            const int wr = idx + 1;
            const int wl = wr - period * num_series;
            const int bad = prefix_nan_tm[wr] - prefix_nan_tm[wl];
            if (bad == 0) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    twof s1  = twof_sub(ld_twof(prefix_sum_tm,    wr),
                                         ld_twof(prefix_sum_tm,    wl));
                    twof s2  = twof_sub(ld_twof(prefix_sum_sq_tm, wr),
                                         ld_twof(prefix_sum_sq_tm, wl));

                    twof mean  = twof_scale(s1, inv_den);
                    twof mean2 = twof_scale(s2, inv_den);
                    twof var_ds = twof_sub(mean2, twof_sqr(mean));

                    float var_f = twof_to_f(var_ds);
                    if (var_f < 0.0f) var_f = 0.0f;
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}

// ===========================================================================
// S3 f64 LANE — deviation
// ===========================================================================
// Stable Authority V2 semantic identity:
// deviation_population_f64_global_pow2_anchored_neumaier_two_pass_fma_sqrt_rn_v2
//
// Every output window is independent.  One CUDA thread scans that window in
// chronological order, exactly matching `stable_population_deviation_window_v2`
// in src/indicators/deviation.rs.  Parallelism is across output windows and
// parameter rows, never across observations inside one reduction.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ __forceinline__ void neo_s3_neumaier_add_v2(
    double value,
    double* __restrict__ sum,
    double* __restrict__ correction)
{
    const double previous = *sum;
    const double updated = __dadd_rn(previous, value);
    double residual;
    if (fabs(previous) >= fabs(value)) {
        residual = __dadd_rn(__dsub_rn(previous, updated), value);
    } else {
        residual = __dadd_rn(__dsub_rn(value, updated), previous);
    }
    *correction = __dadd_rn(*correction, residual);
    *sum = updated;
}

__device__ __forceinline__ double neo_s3_floor_power_of_two_input_scale_v2(
    unsigned long long max_abs_bits)
{
    const unsigned long long exponent_bits = max_abs_bits & 0x7ff0000000000000ULL;
    if (exponent_bits != 0ULL) {
        return __longlong_as_double(static_cast<long long>(exponent_bits));
    }

    // A finite non-zero subnormal has at least one fraction bit.  Its highest
    // set bit is itself the exact floor power of two; no log/exp/libm enters
    // the authority.
    const int highest_fraction_bit = 63 - __clzll(max_abs_bits);
    const unsigned long long scale_bits = 1ULL << highest_fraction_bit;
    return __longlong_as_double(static_cast<long long>(scale_bits));
}

__device__ __forceinline__ double neo_s3_stable_population_deviation_window_v2(
    const double* __restrict__ data,
    int start,
    int period)
{
    unsigned long long max_abs_bits = 0ULL;
    for (int offset = 0; offset < period; ++offset) {
        const double value = data[start + offset];
        if (!isfinite(value)) {
            return neo_s3_qnan();
        }
        const unsigned long long magnitude_bits =
            static_cast<unsigned long long>(__double_as_longlong(value)) &
            0x7fffffffffffffffULL;
        if (magnitude_bits > max_abs_bits) {
            max_abs_bits = magnitude_bits;
        }
    }

    if (max_abs_bits == 0ULL) {
        return 0.0;
    }

    const double scale = neo_s3_floor_power_of_two_input_scale_v2(max_abs_bits);
    const double anchor = __ddiv_rn(data[start], scale);

    double shifted_sum = 0.0;
    double shifted_correction = 0.0;
    for (int offset = 0; offset < period; ++offset) {
        const double value = data[start + offset];
        const double normalized_value = __ddiv_rn(value, scale);
        const double delta = __dsub_rn(normalized_value, anchor);
        neo_s3_neumaier_add_v2(delta, &shifted_sum, &shifted_correction);
    }

    const double count = static_cast<double>(period);
    const double mean_delta =
        __ddiv_rn(__dadd_rn(shifted_sum, shifted_correction), count);

    double square_sum = 0.0;
    double square_correction = 0.0;
    for (int offset = 0; offset < period; ++offset) {
        const double value = data[start + offset];
        const double normalized_value = __ddiv_rn(value, scale);
        const double centered =
            __dsub_rn(__dsub_rn(normalized_value, anchor), mean_delta);
        const double square = __fma_rn(centered, centered, 0.0);
        neo_s3_neumaier_add_v2(square, &square_sum, &square_correction);
    }

    const double corrected_square_sum = __dadd_rn(square_sum, square_correction);
    if (!isfinite(corrected_square_sum) || corrected_square_sum < 0.0) {
        return neo_s3_qnan();
    }

    const double normalized_deviation =
        __dsqrt_rn(__ddiv_rn(corrected_square_sum, count));
    if (!isfinite(normalized_deviation)) {
        return neo_s3_qnan();
    }

    const double result = __dmul_rn(scale, normalized_deviation);
    if (!isfinite(result)) {
        return neo_s3_qnan();
    }
    return (result == 0.0) ? 0.0 : result;
}

extern "C" __global__ void neoethos_deviation_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = blockIdx.y;
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || output_index >= n) return;

    const size_t output_offset =
        static_cast<size_t>(combo) * static_cast<size_t>(n) +
        static_cast<size_t>(output_index);
    const int period = periods[combo];

    // Every `deviation_prepare` branch that returns Err is represented by an
    // all-NaN row.  With one thread per cell, each thread writes its own NaN.
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period <= 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        out[output_offset] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    if (output_index < warm) {
        out[output_offset] = neo_s3_qnan();
        return;
    }

    const int start = output_index + 1 - period;
    out[output_offset] =
        neo_s3_stable_population_deviation_window_v2(data, start, period);
}
