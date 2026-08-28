#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

namespace {


    __device__ __forceinline__ constexpr int kTile() { return 8; }


    __device__ __forceinline__ double epma_weight_sum(int p1, int offset) {
        return 0.5 * static_cast<double>(p1) *
               (static_cast<double>(p1) + 3.0 - 2.0 * static_cast<double>(offset));
    }
}

extern "C" __global__
void epma_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ offsets,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int offset = offsets[combo];
    const int p1     = period - 1;
    if (p1 <= 0) return;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);


    const int warm = first_valid + period + offset + 1;

    const int base_out = combo * series_len;

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            out[base_out + t] = NAN;
        }
        if (t_end <= warm) continue;


        int t0 = (t_start < warm ? warm : t_start);


        int a = t0 + 1 - p1;
        int b = t0;


        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = static_cast<double>(prices[idx]);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }


        out[base_out + t0] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);


        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = static_cast<double>(prices[old_a]);
            double entering = static_cast<double>(prices[b]);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            out[base_out + t] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}

extern "C" __global__
void epma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int offset,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int p1 = period - 1;
    if (p1 <= 0) return;

    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period + offset + 1;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    auto load_tm = [&](int t) -> double {
        long long in_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
        return static_cast<double>(prices_tm[in_idx]);
    };

    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = NAN;
        }
        if (t_end <= warm) continue;

        int t0 = (t_start < warm ? warm : t_start);

        int a = t0 + 1 - p1;
        int b = t0;

        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = load_tm(idx);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }

        {
            long long out_idx = static_cast<long long>(t0) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }

        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = load_tm(old_a);
            double entering = load_tm(b);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

extern "C" __global__
void epma_batch_f64(const double* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ offsets,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int offset = offsets[combo];
    const int p1     = period - 1;
    if (p1 <= 0) return;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);


    const int warm = first_valid + period + offset + 1;

    const int base_out = combo * series_len;

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            out[base_out + t] = NAN;
        }
        if (t_end <= warm) continue;


        int t0 = (t_start < warm ? warm : t_start);


        int a = t0 + 1 - p1;
        int b = t0;


        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = static_cast<double>(prices[idx]);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }


        out[base_out + t0] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);


        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = static_cast<double>(prices[old_a]);
            double entering = static_cast<double>(prices[b]);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            out[base_out + t] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}
extern "C" __global__
void epma_many_series_one_param_time_major_f64(
    const double* __restrict__ prices_tm,
    int period,
    int offset,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm)
{
    const int p1 = period - 1;
    if (p1 <= 0) return;

    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period + offset + 1;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    auto load_tm = [&](int t) -> double {
        long long in_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
        return static_cast<double>(prices_tm[in_idx]);
    };

    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = NAN;
        }
        if (t_end <= warm) continue;

        int t0 = (t_start < warm ? warm : t_start);

        int a = t0 + 1 - p1;
        int b = t0;

        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = load_tm(idx);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }

        {
            long long out_idx = static_cast<long long>(t0) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }

        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = load_tm(old_a);
            double entering = load_tm(b);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}

/* ===========================================================================
 * NEOETHOS strict f64 lane — EPMA bounded-faithful v1
 *
 * Authority: epma/jesse-period-minus-one-ramp/c0=2-offset/period-max=260/absolute-1024-segments/pow2-scaled-dd-rolling-dot2-fallback/compensated-quotient/bounded-faithful/f64/v1
 *
 * This mirrors the Rust operation and branch schedule.  It is not a claim of
 * universal correctly-rounded binary64 arithmetic.  One CUDA thread owns one
 * absolute 1024-output segment for one combo, so every segment starts from its
 * canonical chronological Dot2 checkpoint and no thread-local window array is
 * required.  Severe conditioning or exponent loss fails closed to canonical
 * qNaN and recovers at the first certifiable finite window.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_EPMA_OFFSET 4
#define NEO_EPMA_MAX_PERIOD_V1 260
#define NEO_EPMA_SEGMENT_OUTPUTS_V1 1024
#define NEO_EPMA_CONDITION_REBASE_V1 0x1p-5
#define NEO_EPMA_FALLBACK_CONDITION_V1 0x1p-32

struct NeoEpmaDdV1 {
    double hi;
    double lo;
};

struct NeoEpmaStateV1 {
    double scale;
    double minimum_nonzero_abs;
    NeoEpmaDdV1 sum;
    NeoEpmaDdV1 weighted;
};

__device__ __forceinline__ bool neo_epma_is_normal_v1(double value)
{
    const unsigned long long exponent =
        ((unsigned long long)__double_as_longlong(fabs(value))) & 0x7ff0000000000000ULL;
    return exponent != 0ULL && exponent != 0x7ff0000000000000ULL;
}

__device__ __forceinline__ bool neo_epma_is_subnormal_v1(double value)
{
    const unsigned long long bits =
        ((unsigned long long)__double_as_longlong(fabs(value)));
    return (bits & 0x7ff0000000000000ULL) == 0ULL
        && (bits & 0x000fffffffffffffULL) != 0ULL;
}

__device__ __forceinline__ void neo_epma_two_sum_v1(
    double a, double b, double* sum, double* error)
{
    *sum = __dadd_rn(a, b);
    const double recovered = __dsub_rn(*sum, a);
    *error = __dadd_rn(
        __dsub_rn(a, __dsub_rn(*sum, recovered)),
        __dsub_rn(b, recovered));
}

__device__ __forceinline__ void neo_epma_dd_add_v1(NeoEpmaDdV1* value, double addend)
{
    double sum, error, tail, tail_error, hi, lo, final_hi, final_lo;
    neo_epma_two_sum_v1(value->hi, addend, &sum, &error);
    neo_epma_two_sum_v1(error, value->lo, &tail, &tail_error);
    neo_epma_two_sum_v1(sum, tail, &hi, &lo);
    neo_epma_two_sum_v1(hi, __dadd_rn(lo, tail_error), &final_hi, &final_lo);
    value->hi = final_hi;
    value->lo = final_lo;
}

__device__ __forceinline__ void neo_epma_dd_add_dd_v1(
    NeoEpmaDdV1* value, const NeoEpmaDdV1 other, double sign)
{
    neo_epma_dd_add_v1(value, __dmul_rn(sign, other.hi));
    neo_epma_dd_add_v1(value, __dmul_rn(sign, other.lo));
}

__device__ __forceinline__ void neo_epma_dd_add_product_v1(
    NeoEpmaDdV1* value, double a, double b)
{
    const double product = __dmul_rn(a, b);
    neo_epma_dd_add_v1(value, product);
    neo_epma_dd_add_v1(value, __fma_rn(a, b, -product));
}

__device__ __forceinline__ bool neo_epma_dd_scale_v1(NeoEpmaDdV1* value, double ratio)
{
    const double hi = __dmul_rn(value->hi, ratio);
    const double lo = __dmul_rn(value->lo, ratio);
    if (!isfinite(hi) || !isfinite(lo)
        || (value->hi != 0.0 && hi == 0.0)
        || (value->lo != 0.0 && lo == 0.0)) return false;
    value->hi = hi;
    value->lo = lo;
    return true;
}

__device__ __forceinline__ double neo_epma_dd_value_v1(const NeoEpmaDdV1 value)
{
    return __dadd_rn(value.hi, value.lo);
}

__device__ __forceinline__ double neo_epma_dd_magnitude_v1(const NeoEpmaDdV1 value)
{
    return __dadd_rn(fabs(value.hi), fabs(value.lo));
}

__device__ __forceinline__ bool neo_epma_dd_finite_v1(const NeoEpmaDdV1 value)
{
    return isfinite(value.hi) && isfinite(value.lo);
}

__device__ __forceinline__ double neo_epma_floor_power_of_two_v1(double value)
{
    const unsigned long long bits =
        ((unsigned long long)__double_as_longlong(fabs(value)));
    const unsigned long long exponent = bits & 0x7ff0000000000000ULL;
    if (exponent != 0ULL) return __longlong_as_double((long long)exponent);
    if (bits == 0ULL) return 1.0;
    return __longlong_as_double((long long)(1ULL << (63 - __clzll(bits))));
}

__device__ __forceinline__ bool neo_epma_compensated_quotient_v1(
    const NeoEpmaDdV1 numerator, double denominator, double* corrected)
{
    const double quotient = __ddiv_rn(numerator.hi, denominator);
    const bool numerator_nonzero = numerator.hi != 0.0 || numerator.lo != 0.0;
    if (!isfinite(quotient) || neo_epma_is_subnormal_v1(quotient)
        || (numerator_nonzero && quotient == 0.0)) return false;
    const double product_remainder = __fma_rn(-quotient, denominator, numerator.hi);
    const double remainder = __dadd_rn(product_remainder, numerator.lo);
    if (!isfinite(product_remainder) || !isfinite(remainder)) return false;
    const double correction = __ddiv_rn(remainder, denominator);
    if (!isfinite(correction)
        || (remainder != 0.0
            && (correction == 0.0 || neo_epma_is_subnormal_v1(correction)))) return false;
    *corrected = __dadd_rn(quotient, correction);
    return isfinite(*corrected) && !neo_epma_is_subnormal_v1(*corrected)
        && !(numerator_nonzero && *corrected == 0.0);
}

__device__ __forceinline__ bool neo_epma_rescale_result_v1(
    const NeoEpmaDdV1 numerator, double denominator, double scale, double* result)
{
    double normalized;
    if (!neo_epma_compensated_quotient_v1(numerator, denominator, &normalized)) return false;
    *result = __dmul_rn(normalized, scale);
    if (!isfinite(*result) || (normalized != 0.0 && *result == 0.0)) return false;
    return true;
}

__device__ __forceinline__ double neo_epma_abs_weight_sum_v1(int width, int c0)
{
    long long total = 0;
    for (int index = 0; index < width; ++index) {
        const long long weight = (long long)c0 + (long long)index;
        total += weight < 0 ? -weight : weight;
    }
    return (double)total;
}

__device__ __forceinline__ bool neo_epma_build_state_v1(
    const double* __restrict__ data,
    int window_start,
    int width,
    int c0,
    double weight_sum,
    NeoEpmaStateV1* state,
    double* result)
{
    double maximum_abs = 0.0;
    double minimum_nonzero_abs = __longlong_as_double(0x7ff0000000000000ULL);
    for (int index = 0; index < width; ++index) {
        const double value = data[window_start + index];
        if (!isfinite(value)) return false;
        const double magnitude = fabs(value);
        maximum_abs = maximum_abs > magnitude ? maximum_abs : magnitude;
        if (magnitude != 0.0) {
            minimum_nonzero_abs = minimum_nonzero_abs < magnitude
                ? minimum_nonzero_abs : magnitude;
        }
    }

    const double scale = neo_epma_floor_power_of_two_v1(maximum_abs);
    NeoEpmaDdV1 sum = {0.0, 0.0};
    NeoEpmaDdV1 weighted = {0.0, 0.0};
    NeoEpmaDdV1 absolute_products = {0.0, 0.0};
    for (int index = 0; index < width; ++index) {
        const double value = data[window_start + index];
        const double normalized = __ddiv_rn(value, scale);
        if (value != 0.0 && !neo_epma_is_normal_v1(normalized)) return false;
        neo_epma_dd_add_v1(&sum, normalized);
        const double weight = (double)(c0 + index);
        const double product = __dmul_rn(normalized, weight);
        if (value != 0.0 && weight != 0.0 && !neo_epma_is_normal_v1(product)) return false;
        neo_epma_dd_add_product_v1(&weighted, normalized, weight);
        neo_epma_dd_add_v1(&absolute_products, fabs(product));
    }
    if (!neo_epma_dd_finite_v1(sum)
        || !neo_epma_dd_finite_v1(weighted)
        || !neo_epma_dd_finite_v1(absolute_products)) return false;

    const double weighted_value = neo_epma_dd_value_v1(weighted);
    const double absolute_value = neo_epma_dd_value_v1(absolute_products);
    if (absolute_value != 0.0
        && fabs(weighted_value)
            <= __dmul_rn(absolute_value, NEO_EPMA_FALLBACK_CONDITION_V1)) return false;
    if (absolute_value == 0.0) {
        *result = 0.0;
    } else if (!neo_epma_rescale_result_v1(weighted, weight_sum, scale, result)) {
        return false;
    }

    state->scale = scale;
    state->minimum_nonzero_abs = minimum_nonzero_abs;
    state->sum = sum;
    state->weighted = weighted;
    return true;
}

__device__ __forceinline__ bool neo_epma_roll_state_v1(
    NeoEpmaStateV1* rolling,
    double leaving,
    double entering,
    int width,
    int c0,
    double weight_sum,
    double absolute_weight_sum,
    double* result)
{
    if (!isfinite(leaving) || !isfinite(entering)) return false;
    const double entering_abs = fabs(entering);
    const double entering_scale = neo_epma_floor_power_of_two_v1(entering_abs);
    if (entering_scale > rolling->scale) {
        const double ratio = __ddiv_rn(rolling->scale, entering_scale);
        if (!neo_epma_is_normal_v1(ratio)
            || !neo_epma_dd_scale_v1(&rolling->sum, ratio)
            || !neo_epma_dd_scale_v1(&rolling->weighted, ratio)) return false;
        rolling->scale = entering_scale;
    }
    if (entering_abs != 0.0) {
        rolling->minimum_nonzero_abs = rolling->minimum_nonzero_abs < entering_abs
            ? rolling->minimum_nonzero_abs : entering_abs;
    }

    const double leaving_normalized = __ddiv_rn(leaving, rolling->scale);
    const double entering_normalized = __ddiv_rn(entering, rolling->scale);
    const double minimum_normalized =
        __ddiv_rn(rolling->minimum_nonzero_abs, rolling->scale);
    if ((isfinite(rolling->minimum_nonzero_abs)
            && !neo_epma_is_normal_v1(minimum_normalized))
        || (leaving != 0.0 && !neo_epma_is_normal_v1(leaving_normalized))
        || (entering != 0.0 && !neo_epma_is_normal_v1(entering_normalized))) return false;

    const double previous_sum_magnitude = neo_epma_dd_magnitude_v1(rolling->sum);
    const double previous_weighted_magnitude = neo_epma_dd_magnitude_v1(rolling->weighted);
    const double leaving_weight = (double)(1 - c0);
    const double entering_weight = (double)(c0 + width - 1);
    const double leaving_product = __dmul_rn(leaving_weight, leaving_normalized);
    const double entering_product = __dmul_rn(entering_weight, entering_normalized);
    if ((leaving_weight != 0.0 && leaving_normalized != 0.0
            && !neo_epma_is_normal_v1(leaving_product))
        || (entering_weight != 0.0 && entering_normalized != 0.0
            && !neo_epma_is_normal_v1(entering_product))) return false;

    neo_epma_dd_add_dd_v1(&rolling->weighted, rolling->sum, -1.0);
    neo_epma_dd_add_product_v1(&rolling->weighted, leaving_weight, leaving_normalized);
    neo_epma_dd_add_product_v1(&rolling->weighted, entering_weight, entering_normalized);
    neo_epma_dd_add_v1(&rolling->sum, -leaving_normalized);
    neo_epma_dd_add_v1(&rolling->sum, entering_normalized);

    double weighted_bound = __dadd_rn(previous_weighted_magnitude, previous_sum_magnitude);
    weighted_bound = __dadd_rn(weighted_bound, fabs(leaving_product));
    weighted_bound = __dadd_rn(weighted_bound, fabs(entering_product));
    double sum_bound = __dadd_rn(previous_sum_magnitude, fabs(leaving_normalized));
    sum_bound = __dadd_rn(sum_bound, fabs(entering_normalized));
    const double weighted_value_abs = fabs(neo_epma_dd_value_v1(rolling->weighted));
    if (!neo_epma_dd_finite_v1(rolling->sum)
        || !neo_epma_dd_finite_v1(rolling->weighted)
        || weighted_value_abs
            <= __dmul_rn(weighted_bound, NEO_EPMA_CONDITION_REBASE_V1)
        || fabs(neo_epma_dd_value_v1(rolling->sum))
            <= __dmul_rn(sum_bound, NEO_EPMA_CONDITION_REBASE_V1)
        || weighted_value_abs
            <= __dmul_rn(__dmul_rn(2.0, absolute_weight_sum),
                         NEO_EPMA_FALLBACK_CONDITION_V1)) return false;

    return neo_epma_rescale_result_v1(
        rolling->weighted, weight_sum, rolling->scale, result);
}

extern "C" __global__
void epma_neo_batch_f64(const double* __restrict__ data,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)blockIdx.y;
    const long long segment =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long segment_start_ll = segment * (long long)NEO_EPMA_SEGMENT_OUTPUTS_V1;
    if (combo >= n_combos || n <= 0 || segment_start_ll >= (long long)n) return;
    const int segment_start = (int)segment_start_ll;
    const long long segment_limit_ll =
        segment_start_ll + (long long)NEO_EPMA_SEGMENT_OUTPUTS_V1;
    const long long segment_end_ll = (long long)n < segment_limit_ll
        ? (long long)n : segment_limit_ll;
    const int segment_end = (int)segment_end_ll;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int index = segment_start; index < segment_end; ++index) o[index] = NEO_F64_NAN;

    const int period = periods[combo];
    const int offset = NEO_EPMA_OFFSET;
    if (period < 2 || period > NEO_EPMA_MAX_PERIOD_V1 || offset >= period) return;
    if (first_valid < 0 || first_valid >= n) return;
    const int width = period - 1;
    const int c0 = 2 - offset;
    const long long twice_sum =
        (long long)width * (2LL * (long long)c0 + (long long)width - 1LL);
    if ((twice_sum & 1LL) != 0LL) return;
    const long long integer_weight_sum = twice_sum / 2LL;
    if (integer_weight_sum == 0LL) return;
    const double weight_sum = (double)integer_weight_sum;
    const double absolute_weight_sum = neo_epma_abs_weight_sum_v1(width, c0);

    const long long warmup_ll = (long long)first_valid + (long long)period
        + (long long)offset + 1LL;
    const long long output_start_ll = segment_start_ll > warmup_ll
        ? segment_start_ll : warmup_ll;
    if (output_start_ll >= segment_end_ll) return;
    const int output_start = (int)output_start_ll;

    NeoEpmaStateV1 state;
    bool state_valid = false;
    for (int output_index = output_start; output_index < segment_end; ++output_index) {
        const int window_start = output_index + 1 - width;
        double result;
        if (!state_valid || output_index == output_start) {
            state_valid = neo_epma_build_state_v1(
                data, window_start, width, c0, weight_sum, &state, &result);
        } else {
            NeoEpmaStateV1 rolled = state;
            state_valid = neo_epma_roll_state_v1(
                &rolled,
                data[window_start - 1],
                data[output_index],
                width,
                c0,
                weight_sum,
                absolute_weight_sum,
                &result);
            if (state_valid) state = rolled;
            else {
                state_valid = neo_epma_build_state_v1(
                    data, window_start, width, c0, weight_sum, &state, &result);
            }
        }
        o[output_index] = state_valid ? result : NEO_F64_NAN;
    }
}
