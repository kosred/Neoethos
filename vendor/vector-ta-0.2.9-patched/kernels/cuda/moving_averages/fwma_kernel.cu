#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef FWMA_TILE_T

#define FWMA_TILE_T 256
#endif


extern "C" __global__
void fwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;


    extern __shared__ float smem[];
    float* __restrict__ s_w = smem;
    float* __restrict__ s_x = s_w + max_period;


    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm     = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f  = __int_as_float(0x7fffffff);


    const int tile_t0 = blockIdx.x * blockDim.x;
    const int tile_t1 = min(series_len, tile_t0 + blockDim.x);


    if (tile_t1 <= warm) {
        const int t = tile_t0 + threadIdx.x;
        if (t < tile_t1) out[base_out + t] = nan_f;
        return;
    }


    const int load_base = tile_t0 - period + 1;
    const int load_len  = (tile_t1 - tile_t0) + period - 1;


    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        const int g = load_base + i;
        s_x[i] = (unsigned(g) < (unsigned)series_len) ? prices[g] : 0.0f;
    }
    __syncthreads();


    const int t = tile_t0 + threadIdx.x;
    if (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {

            const int offset = (t - period + 1) - load_base;
            float acc = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(s_x[offset + k], s_w[k], acc);
            }
            out[base_out + t] = acc;
        }
    }
}


#ifndef FWMA_TIME_STEPS_PER_BLOCK
#define FWMA_TIME_STEPS_PER_BLOCK 4
#endif

extern "C" __global__
void fwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {

    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);


    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;


    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {

                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}


extern "C" __global__
void fwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const float* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {

    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);

    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;

    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Strict f64 lane.
//
// Semantic identity:
// fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery
//
// The Fibonacci integers are generated exactly in three 64-bit limbs.  The
// p<=254 bound keeps F_254 and its exact denominator within 192 bits.  Each
// finite window is globally power-of-two scaled, then accumulated oldest to
// newest with the same DD / TwoSum / FMA-product-tail schedule as the host.
// Nonfinite or uncertifiable scale/quotient underflow fails closed to the
// canonical quiet NaN.  Compile this lane with FTZ disabled and precise div.
// ---------------------------------------------------------------------------

#ifndef FWMA_MAX_PERIOD_F64
#define FWMA_MAX_PERIOD_F64 254
#endif

struct fwma_dd_f64_v2 {
    double hi;
    double lo;
};

struct fwma_u192_f64_v2 {
    unsigned long long lo;
    unsigned long long mid;
    unsigned long long hi;
};

static __device__ __forceinline__ double fwma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

static __device__ __forceinline__ double fwma_canonical_zero_f64_v2(
    double value)
{
    return value == 0.0 ? 0.0 : value;
}

static __device__ __forceinline__ fwma_dd_f64_v2 fwma_two_sum_f64_v2(
    double a,
    double b)
{
    const double hi = __dadd_rn(a, b);
    const double b_virtual = __dsub_rn(hi, a);
    const double lo = __dadd_rn(
        __dsub_rn(a, __dsub_rn(hi, b_virtual)),
        __dsub_rn(b, b_virtual));
    return {hi, lo};
}

static __device__ __forceinline__ fwma_dd_f64_v2 fwma_dd_add_f64_v2(
    fwma_dd_f64_v2 a,
    fwma_dd_f64_v2 b)
{
    const fwma_dd_f64_v2 high = fwma_two_sum_f64_v2(a.hi, b.hi);
    const fwma_dd_f64_v2 low = fwma_two_sum_f64_v2(a.lo, b.lo);
    const fwma_dd_f64_v2 middle = fwma_two_sum_f64_v2(high.lo, low.hi);
    const fwma_dd_f64_v2 normalized =
        fwma_two_sum_f64_v2(high.hi, middle.hi);
    const double tail = __dadd_rn(
        __dadd_rn(normalized.lo, middle.lo),
        low.lo);
    return fwma_two_sum_f64_v2(normalized.hi, tail);
}

static __device__ __forceinline__ fwma_dd_f64_v2 fwma_dd_sub_f64_v2(
    fwma_dd_f64_v2 a,
    fwma_dd_f64_v2 b)
{
    b.hi = -b.hi;
    b.lo = -b.lo;
    return fwma_dd_add_f64_v2(a, b);
}

static __device__ __forceinline__ fwma_dd_f64_v2 fwma_dd_mul_f64_v2(
    double value,
    fwma_dd_f64_v2 weight)
{
    const double product = __dmul_rn(value, weight.hi);
    const double product_tail = __fma_rn(value, weight.hi, -product);
    return fwma_two_sum_f64_v2(
        product,
        __fma_rn(value, weight.lo, product_tail));
}

static __device__ __forceinline__ fwma_dd_f64_v2 fwma_dd_mul_scalar_f64_v2(
    fwma_dd_f64_v2 value,
    double scalar)
{
    const double product = __dmul_rn(value.hi, scalar);
    const double product_tail = __fma_rn(value.hi, scalar, -product);
    return fwma_two_sum_f64_v2(
        product,
        __fma_rn(value.lo, scalar, product_tail));
}

static __device__ __forceinline__ bool fwma_u192_add_f64_v2(
    fwma_u192_f64_v2 a,
    fwma_u192_f64_v2 b,
    fwma_u192_f64_v2* out)
{
    const unsigned long long lo = a.lo + b.lo;
    const bool carry0 = lo < a.lo;
    const unsigned long long mid0 = a.mid + b.mid;
    const bool carry1 = mid0 < a.mid;
    const unsigned long long mid =
        mid0 + static_cast<unsigned long long>(carry0);
    const bool carry2 = mid < mid0;
    const unsigned long long hi0 = a.hi + b.hi;
    const bool carry3 = hi0 < a.hi;
    const unsigned long long hi =
        hi0 + static_cast<unsigned long long>(carry1 || carry2);
    const bool carry4 = hi < hi0;
    out->lo = lo;
    out->mid = mid;
    out->hi = hi;
    return !(carry3 || carry4);
}

static __device__ __forceinline__ unsigned int fwma_u192_chunk_f64_v2(
    fwma_u192_f64_v2 value,
    int index)
{
    if (index == 0) return static_cast<unsigned int>(value.lo);
    if (index == 1) return static_cast<unsigned int>(value.lo >> 32);
    if (index == 2) return static_cast<unsigned int>(value.mid);
    if (index == 3) return static_cast<unsigned int>(value.mid >> 32);
    if (index == 4) return static_cast<unsigned int>(value.hi);
    return static_cast<unsigned int>(value.hi >> 32);
}

static __device__ __forceinline__ fwma_dd_f64_v2 fwma_u192_to_dd_f64_v2(
    fwma_u192_f64_v2 value)
{
    fwma_dd_f64_v2 result = {0.0, 0.0};
    for (int index = 5; index >= 0; --index) {
        result.hi = __dmul_rn(result.hi, 4294967296.0);
        result.lo = __dmul_rn(result.lo, 4294967296.0);
        const fwma_dd_f64_v2 chunk = {
            __uint2double_rn(fwma_u192_chunk_f64_v2(value, index)),
            0.0};
        result = fwma_dd_add_f64_v2(result, chunk);
    }
    return result;
}

static __device__ __forceinline__ bool fwma_build_weights_f64_v2(
    int period,
    fwma_dd_f64_v2* weights,
    fwma_dd_f64_v2* denominator_dd)
{
    const fwma_u192_f64_v2 zero = {0ULL, 0ULL, 0ULL};
    const fwma_u192_f64_v2 one = {1ULL, 0ULL, 0ULL};
    fwma_u192_f64_v2 previous = one;
    fwma_u192_f64_v2 current = one;
    fwma_u192_f64_v2 denominator = zero;
    for (int index = 0; index < period; ++index) {
        fwma_u192_f64_v2 weight = one;
        if (index >= 2) {
            if (!fwma_u192_add_f64_v2(previous, current, &weight)) return false;
            previous = current;
            current = weight;
        }
        if (!fwma_u192_add_f64_v2(denominator, weight, &denominator)) return false;
        weights[index] = fwma_u192_to_dd_f64_v2(weight);
    }
    *denominator_dd = fwma_u192_to_dd_f64_v2(denominator);
    return true;
}

static __device__ __forceinline__ bool fwma_unbiased_exponent_f64_v2(
    double value,
    int* exponent_out)
{
    const unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(value))
        & 0x7fffffffffffffffULL;
    if (bits == 0ULL) return false;
    const int exponent = static_cast<int>((bits >> 52) & 0x7ffULL);
    if (exponent != 0) {
        *exponent_out = exponent - 1023;
    } else {
        const unsigned long long fraction = bits & 0x000fffffffffffffULL;
        const int highest = 63 - __clzll(fraction);
        *exponent_out = highest - 1074;
    }
    return true;
}

static __device__ __forceinline__ double fwma_pow2_f64_v2(int exponent) {
    return __longlong_as_double(
        static_cast<unsigned long long>(exponent + 1023) << 52);
}

static __device__ __forceinline__ bool fwma_scale_pow2_checked_f64_v2(
    double value,
    int exponent,
    double* result_out)
{
    while (exponent != 0) {
        const int step =
            exponent > 512 ? 512 : (exponent < -512 ? -512 : exponent);
        const double scaled = __dmul_rn(value, fwma_pow2_f64_v2(step));
        if (!isfinite(scaled) || (scaled == 0.0 && value != 0.0)) return false;
        value = scaled;
        exponent -= step;
    }
    *result_out = value;
    return true;
}

static __device__ __forceinline__ bool fwma_compensated_quotient_f64_v2(
    fwma_dd_f64_v2 numerator,
    fwma_dd_f64_v2 denominator,
    double* result_out)
{
    if (denominator.hi == 0.0
        || !isfinite(denominator.hi)
        || !isfinite(numerator.hi)) return false;
    const double q0 = __ddiv_rn(numerator.hi, denominator.hi);
    if (!isfinite(q0)) return false;
    const fwma_dd_f64_v2 residual0 = fwma_dd_sub_f64_v2(
        numerator,
        fwma_dd_mul_scalar_f64_v2(denominator, q0));
    const double q1 = __ddiv_rn(residual0.hi, denominator.hi);
    const fwma_dd_f64_v2 residual1 = fwma_dd_sub_f64_v2(
        residual0,
        fwma_dd_mul_scalar_f64_v2(denominator, q1));
    const double q2 = __ddiv_rn(residual1.hi, denominator.hi);
    const fwma_dd_f64_v2 quotient = fwma_dd_add_f64_v2(
        fwma_two_sum_f64_v2(q0, q1),
        fwma_dd_f64_v2{q2, 0.0});
    const double result = __dadd_rn(quotient.hi, quotient.lo);
    if (!isfinite(result)
        || (result == 0.0
            && (numerator.hi != 0.0 || numerator.lo != 0.0))) return false;
    *result_out = fwma_canonical_zero_f64_v2(result);
    return true;
}

static __device__ __forceinline__ bool fwma_window_authority_f64_v2(
    const double* prices,
    int start,
    int period,
    const fwma_dd_f64_v2* weights,
    fwma_dd_f64_v2 denominator,
    double* result_out)
{
    if (period == 1) {
        const double value = prices[start];
        if (!isfinite(value)) return false;
        *result_out = fwma_canonical_zero_f64_v2(value);
        return true;
    }

    bool any_nonzero = false;
    int maximum_exponent = -1075;
    for (int index = 0; index < period; ++index) {
        const double value = prices[start + index];
        if (!isfinite(value)) return false;
        int exponent = 0;
        if (fwma_unbiased_exponent_f64_v2(value, &exponent)) {
            any_nonzero = true;
            if (exponent > maximum_exponent) maximum_exponent = exponent;
        }
    }
    if (!any_nonzero) {
        *result_out = 0.0;
        return true;
    }

    fwma_dd_f64_v2 numerator = {0.0, 0.0};
    for (int index = 0; index < period; ++index) {
        double scaled = 0.0;
        if (!fwma_scale_pow2_checked_f64_v2(
                prices[start + index], -maximum_exponent, &scaled)) return false;
        numerator = fwma_dd_add_f64_v2(
            numerator,
            fwma_dd_mul_f64_v2(scaled, weights[index]));
    }
    double scaled_result = 0.0;
    if (!fwma_compensated_quotient_f64_v2(
            numerator, denominator, &scaled_result)) return false;
    double result = 0.0;
    if (!fwma_scale_pow2_checked_f64_v2(
            scaled_result, maximum_exponent, &result)) return false;
    *result_out = fwma_canonical_zero_f64_v2(result);
    return true;
}

extern "C" __global__
void fwma_batch_f64(const double* __restrict__ prices,
                    int n,
                    const int*   __restrict__ periods,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    __shared__ fwma_dd_f64_v2 weights[FWMA_MAX_PERIOD_F64];
    __shared__ fwma_dd_f64_v2 denominator;
    __shared__ int shared_period;
    __shared__ int shared_warm;
    __shared__ int shared_success;

    const int combo = blockIdx.y;
    if (threadIdx.x == 0) {
        shared_period = 0;
        shared_warm = 0;
        shared_success = 0;
        denominator = {0.0, 0.0};
        if (combo < n_combos && n > 0) {
            const int period = periods[combo];
            const long long warm_ll =
                static_cast<long long>(first_valid)
                + static_cast<long long>(period)
                - 1LL;
            if (period > 0
                && period <= FWMA_MAX_PERIOD_F64
                && first_valid >= 0
                && warm_ll >= 0
                && warm_ll < n
                && fwma_build_weights_f64_v2(period, weights, &denominator)) {
                shared_period = period;
                shared_warm = static_cast<int>(warm_ll);
                shared_success = 1;
            }
        }
    }
    __syncthreads();

    if (combo >= n_combos || n <= 0) return;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    double* __restrict__ row =
        out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    row[index] = fwma_qnan_f64();
    if (!shared_success || index < shared_warm) return;

    const int start = index + 1 - shared_period;
    double value = 0.0;
    if (fwma_window_authority_f64_v2(
            prices,
            start,
            shared_period,
            weights,
            denominator,
            &value)) {
        row[index] = value;
    }
}
