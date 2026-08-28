#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

extern "C" __global__ void fisher_build_hl2_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    float* __restrict__ hl)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        hl[idx] = 0.5f * (high[idx] + low[idx]);
    }
}


__device__ __forceinline__ int rb_dec(int x, int cap) { return (x == 0) ? (cap - 1) : (x - 1); }
__device__ __forceinline__ int rb_inc(int x, int cap) { return (x + 1 == cap) ? 0 : (x + 1); }


extern "C" __global__ void fisher_batch_f32(const float* __restrict__ hl,
                                             const int*   __restrict__ periods,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out_fisher,
                                             float* __restrict__ out_signal) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;


    auto fill_all_nan = [&](int len){
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            out_fisher[base + i] = NAN;
            out_signal[base + i] = NAN;
        }
    };

    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan(series_len);
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        fill_all_nan(series_len);
        return;
    }

    const int warm = first_valid + period - 1;


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_fisher[base + i] = NAN;
        out_signal[base + i] = NAN;
    }
    __syncthreads();


    if (threadIdx.x != 0) return;


    extern __shared__ int s[];
    int* dq_min = s;
    int* dq_max = s + period;

    int hmin = 0, tmin = 0;
    int hmax = 0, tmax = 0;
    int cmin = 0, cmax = 0;


    float prev_fish = 0.0f;
    float val1 = 0.0f;


    for (int i = first_valid; i < series_len; ++i) {
        const float xi = hl[i];


        if (i >= warm) {
            const int window_start = i - period + 1;
            while (cmin > 0 && dq_min[hmin] < window_start) { hmin = rb_inc(hmin, period); --cmin; }
            while (cmax > 0 && dq_max[hmax] < window_start) { hmax = rb_inc(hmax, period); --cmax; }
        }


        while (cmin > 0) {
            int last = rb_dec(tmin, period);
            if (xi <= hl[dq_min[last]]) {
                tmin = last;
                --cmin;
            } else {
                break;
            }
        }
        dq_min[tmin] = i;
        tmin = rb_inc(tmin, period);
        ++cmin;


        while (cmax > 0) {
            int last = rb_dec(tmax, period);
            if (xi >= hl[dq_max[last]]) {
                tmax = last;
                --cmax;
            } else {
                break;
            }
        }
        dq_max[tmax] = i;
        tmax = rb_inc(tmax, period);
        ++cmax;


        if (i >= warm) {
            const float minv  = hl[dq_min[hmin]];
            const float maxv  = hl[dq_max[hmax]];
            const float range = fmaxf(maxv - minv, 1.0e-3f);
            const float norm  = (xi - minv) / range - 0.5f;


            val1 = fmaf(0.67f, val1, 0.66f * norm);
            val1 = clampf(val1, -0.999f, 0.999f);

            out_signal[base + i] = prev_fish;

            const float fish = atanhf(val1) + 0.5f * prev_fish;
            out_fisher[base + i] = fish;
            prev_fish = fish;
        }
    }
}


extern "C" __global__ void fisher_many_series_one_param_f32(
    const float* __restrict__ hl_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ fisher_tm,
    float* __restrict__ signal_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (UNLIKELY(first_valid >= series_len || (series_len - first_valid) < period)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int r = 0; r < warm; ++r) {
        const int idx = r * num_series + series;
        fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
    }

    float prev_fish = 0.0f;
    float val1 = 0.0f;
    for (int r = warm; r < series_len; ++r) {
        const int start = r + 1 - period;
        float minv = CUDART_INF_F;
        float maxv = -CUDART_INF_F;

        for (int k = 0; k < period; ++k) {
            const int idx = (start + k) * num_series + series;
            const float x = hl_tm[idx];
            minv = fminf(minv, x);
            maxv = fmaxf(maxv, x);
        }
        const float range = fmaxf(maxv - minv, 1.0e-3f);
        const float x = hl_tm[r * num_series + series];
        const float norm = (x - minv) / range - 0.5f;

        val1 = fmaf(0.67f, val1, 0.66f * norm);
        val1 = clampf(val1, -0.999f, 0.999f);

        const int idxo = r * num_series + series;
        signal_tm[idxo] = prev_fish;
        const float fish = atanhf(val1) + 0.5f * prev_fish;
        fisher_tm[idxo] = fish;
        prev_fish = fish;
    }
}

// =============================================================================
// NeoEthos f64 lane
// Fisher v2: deterministic, bounded-faithful CPU/CUDA authority.
//
// The creator coefficient/order semantics and VectorTA's established 0.001
// range floor are preserved. Sun fdlibm/OpenLibm e_log is mirrored below with
// explicit RN arithmetic; no platform-native transcendental participates.
// Immutable authority receipt:
// commit=82e90aef0657289192efe77be89791c07dea0775
// source=https://raw.githubusercontent.com/JuliaMath/openlibm/82e90aef0657289192efe77be89791c07dea0775/src/e_log.c
// license=https://raw.githubusercontent.com/JuliaMath/openlibm/82e90aef0657289192efe77be89791c07dea0775/LICENSE.md
// sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD
// Every finite H/L/midpoint segment is a fresh recurrence. A non-finite bar or
// arithmetic-domain failure emits canonical qNaN, clears extrema/state, and
// requires `period` consecutive finite bars before the next output. The first
// signal in every segment is exact +0.
//
// Bounded-faithful audit receipts, not a universal RN or ULP guarantee:
// FISHER_F64_V2_FIXTURE_MAX_ULP=2
// FISHER_F64_V2_FIXTURE_MAX_ABS=8.881784197001252e-16
// FISHER_F64_V2_ADVERSARIAL_MAX_ABS=1.7763568394002505e-15
// The fixture bound is against a correctly-rounded transform while retaining
// the established binary64 coefficient/floor schedule: 24,195 primary cells,
// 1,327 nonzero differences, 28 above one ULP. Exact-real normalization is a
// separate authority question and is deliberately not claimed by this v2.
//
// One CUDA block owns one period tuple. Thread zero walks the sequential
// recurrence; the block cooperatively initializes outputs. Strict f64 periods
// through 1024 use O(N) monotone deques in dynamic shared memory. Any invalid
// or larger period remains canonical qNaN; the shared wrapper must reject it
// before allocation/upload/module lookup/launch. ABI symbols and argument order
// remain unchanged; the wrapper must launch grid.x=n_combos, one warp per block,
// and 2*(max_period+1) shared int slots. That hunk is handed off separately.
// The release gate is an RTX 1M-row x 250-period receipt: all periods <=1024
// must stay on this O(N) deque body, report zero local-array spill, and show no
// regression versus the frozen native strict-f64 production launch.
// =============================================================================

#define NEO_FISHER_F64_MAX_PERIOD 1024

__device__ __forceinline__ double fisher_qnan_f64_v2() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ __forceinline__ double fisher_add_rn_f64_v2(double left, double right) {
    return __dadd_rn(left, right);
}

__device__ __forceinline__ double fisher_sub_rn_f64_v2(double left, double right) {
    return __dsub_rn(left, right);
}

__device__ __forceinline__ double fisher_mul_rn_f64_v2(double left, double right) {
    return __dmul_rn(left, right);
}

__device__ __forceinline__ double fisher_div_rn_f64_v2(double left, double right) {
    return __ddiv_rn(left, right);
}

__device__ __forceinline__ double fisher_fma_rn_f64_v2(double left,
                                                        double right,
                                                        double addend) {
    return __fma_rn(left, right, addend);
}

__device__ __forceinline__ double fisher_with_high_word_f64_v2(double value,
                                                                unsigned int high) {
    unsigned long long bits = (unsigned long long)__double_as_longlong(value);
    bits = (bits & 0x00000000ffffffffULL) | ((unsigned long long)high << 32);
    return __longlong_as_double((long long)bits);
}

// Literal Sun fdlibm/OpenLibm e_log schedule. Every arithmetic edge is an
// explicit round-to-nearest-even operation so host and CUDA share one graph.
__device__ __forceinline__ bool fisher_log_f64_v2(double value, double* output) {
    const double TWO54 = 1.80143985094819840000e+16;
    const double LN2_HI = 6.93147180369123816490e-01;
    const double LN2_LO = 1.90821492927058770002e-10;
    const double LG1 = 6.666666666666735130e-01;
    const double LG2 = 3.999999999940941908e-01;
    const double LG3 = 2.857142874366239149e-01;
    const double LG4 = 2.222219843214978396e-01;
    const double LG5 = 1.818357216161805012e-01;
    const double LG6 = 1.531383769920937332e-01;
    const double LG7 = 1.479819860511658591e-01;

    if (!isfinite(value) || !(value > 0.0)) return false;
    unsigned long long raw = (unsigned long long)__double_as_longlong(value);
    int high = (int)(raw >> 32);
    const unsigned int low = (unsigned int)raw;
    int exponent = 0;
    if (high < 0x00100000) {
        if ((((unsigned int)high & 0x7fffffffU) | low) == 0U) return false;
        exponent -= 54;
        value = fisher_mul_rn_f64_v2(value, TWO54);
        raw = (unsigned long long)__double_as_longlong(value);
        high = (int)(raw >> 32);
    }
    if (high >= 0x7ff00000) return false;

    exponent += (high >> 20) - 1023;
    high &= 0x000fffff;
    const int normalize = (high + 0x00095f64) & 0x00100000;
    value = fisher_with_high_word_f64_v2(
        value, (unsigned int)(high | (normalize ^ 0x3ff00000)));
    exponent += normalize >> 20;

    const double fraction = fisher_sub_rn_f64_v2(value, 1.0);
    if ((0x000fffff & (2 + high)) < 3) {
        if (fraction == 0.0) {
            if (exponent == 0) {
                *output = 0.0;
                return true;
            }
            const double exponent_f64 = (double)exponent;
            *output = fisher_add_rn_f64_v2(
                fisher_mul_rn_f64_v2(exponent_f64, LN2_HI),
                fisher_mul_rn_f64_v2(exponent_f64, LN2_LO));
            return isfinite(*output);
        }
        const double square = fisher_mul_rn_f64_v2(fraction, fraction);
        const double inner = fisher_sub_rn_f64_v2(
            0.5, fisher_mul_rn_f64_v2(0.33333333333333333, fraction));
        const double remainder = fisher_mul_rn_f64_v2(square, inner);
        if (exponent == 0) {
            *output = fisher_sub_rn_f64_v2(fraction, remainder);
            return isfinite(*output);
        }
        const double exponent_f64 = (double)exponent;
        const double correction = fisher_sub_rn_f64_v2(
            fisher_sub_rn_f64_v2(
                remainder, fisher_mul_rn_f64_v2(exponent_f64, LN2_LO)),
            fraction);
        *output = fisher_sub_rn_f64_v2(
            fisher_mul_rn_f64_v2(exponent_f64, LN2_HI), correction);
        return isfinite(*output);
    }

    const double scaled = fisher_div_rn_f64_v2(
        fraction, fisher_add_rn_f64_v2(2.0, fraction));
    const double exponent_f64 = (double)exponent;
    const double square = fisher_mul_rn_f64_v2(scaled, scaled);
    const int selector = (high - 0x0006147a) | (0x0006b851 - high);
    const double fourth = fisher_mul_rn_f64_v2(square, square);
    const double even_inner = fisher_add_rn_f64_v2(
        LG4, fisher_mul_rn_f64_v2(fourth, LG6));
    const double even = fisher_mul_rn_f64_v2(
        fourth,
        fisher_add_rn_f64_v2(LG2, fisher_mul_rn_f64_v2(fourth, even_inner)));
    const double odd_inner = fisher_add_rn_f64_v2(
        LG5, fisher_mul_rn_f64_v2(fourth, LG7));
    const double odd_middle = fisher_add_rn_f64_v2(
        LG3, fisher_mul_rn_f64_v2(fourth, odd_inner));
    const double odd = fisher_mul_rn_f64_v2(
        square,
        fisher_add_rn_f64_v2(LG1, fisher_mul_rn_f64_v2(fourth, odd_middle)));
    const double remainder = fisher_add_rn_f64_v2(odd, even);

    if (selector > 0) {
        const double half_square = fisher_mul_rn_f64_v2(
            fisher_mul_rn_f64_v2(0.5, fraction), fraction);
        const double scaled_sum = fisher_mul_rn_f64_v2(
            scaled, fisher_add_rn_f64_v2(half_square, remainder));
        if (exponent == 0) {
            *output = fisher_sub_rn_f64_v2(
                fraction, fisher_sub_rn_f64_v2(half_square, scaled_sum));
            return isfinite(*output);
        }
        const double low_term = fisher_mul_rn_f64_v2(exponent_f64, LN2_LO);
        const double correction = fisher_sub_rn_f64_v2(
            fisher_sub_rn_f64_v2(
                half_square, fisher_add_rn_f64_v2(scaled_sum, low_term)),
            fraction);
        *output = fisher_sub_rn_f64_v2(
            fisher_mul_rn_f64_v2(exponent_f64, LN2_HI), correction);
        return isfinite(*output);
    }

    const double scaled_remainder = fisher_mul_rn_f64_v2(
        scaled, fisher_sub_rn_f64_v2(fraction, remainder));
    if (exponent == 0) {
        *output = fisher_sub_rn_f64_v2(fraction, scaled_remainder);
        return isfinite(*output);
    }
    const double correction = fisher_sub_rn_f64_v2(
        fisher_sub_rn_f64_v2(
            scaled_remainder, fisher_mul_rn_f64_v2(exponent_f64, LN2_LO)),
        fraction);
    *output = fisher_sub_rn_f64_v2(
        fisher_mul_rn_f64_v2(exponent_f64, LN2_HI), correction);
    return isfinite(*output);
}

__device__ __forceinline__ bool fisher_midpoint_f64_v2(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int index,
    double* midpoint) {
    const double high_value = high[index];
    const double low_value = low[index];
    if (!isfinite(high_value) || !isfinite(low_value)) return false;
    *midpoint = fisher_mul_rn_f64_v2(
        0.5, fisher_add_rn_f64_v2(high_value, low_value));
    return isfinite(*midpoint);
}

__device__ __forceinline__ bool fisher_transition_f64_v2(double midpoint,
                                                          double minimum,
                                                          double maximum,
                                                          double* value1,
                                                          double* previous_fisher,
                                                          double* fisher,
                                                          double* signal) {
    const double range_delta = fisher_sub_rn_f64_v2(maximum, minimum);
    if (!isfinite(range_delta)) return false;
    const double range = range_delta > 0.001 ? range_delta : 0.001;
    const double normalized = fisher_sub_rn_f64_v2(
        fisher_div_rn_f64_v2(
            fisher_sub_rn_f64_v2(midpoint, minimum), range),
        0.5);
    const double weighted = fisher_mul_rn_f64_v2(0.66, normalized);
    if (!isfinite(normalized) || !isfinite(weighted)) return false;

    double next_value1 = fisher_fma_rn_f64_v2(0.67, *value1, weighted);
    if (!isfinite(next_value1)) return false;
    if (next_value1 > 0.99) {
        next_value1 = 0.999;
    } else if (next_value1 < -0.99) {
        next_value1 = -0.999;
    }

    const double numerator = fisher_add_rn_f64_v2(1.0, next_value1);
    const double denominator = fisher_sub_rn_f64_v2(1.0, next_value1);
    const double ratio = fisher_div_rn_f64_v2(numerator, denominator);
    double logarithm = 0.0;
    if (!fisher_log_f64_v2(ratio, &logarithm)) return false;
    const double next_signal = *previous_fisher;
    const double next_fisher = fisher_fma_rn_f64_v2(
        0.5, logarithm, fisher_mul_rn_f64_v2(0.5, next_signal));
    if (!isfinite(next_fisher)) return false;

    *value1 = next_value1;
    *previous_fisher = next_fisher;
    *fisher = next_fisher;
    *signal = next_signal;
    return true;
}

struct NeoFisherDequeF64V2 {
    int* indices;
    int head;
    int length;
    int capacity;
};

__device__ __forceinline__ void fisher_deque_init_f64_v2(
    NeoFisherDequeF64V2* deque, int* indices, int capacity) {
    deque->indices = indices;
    deque->head = 0;
    deque->length = 0;
    deque->capacity = capacity;
}

__device__ __forceinline__ void fisher_deque_clear_f64_v2(
    NeoFisherDequeF64V2* deque) {
    deque->head = 0;
    deque->length = 0;
}

__device__ __forceinline__ int fisher_deque_front_f64_v2(
    const NeoFisherDequeF64V2* deque) {
    return deque->indices[deque->head];
}

__device__ __forceinline__ int fisher_deque_back_f64_v2(
    const NeoFisherDequeF64V2* deque) {
    const int slot = (deque->head + deque->length - 1) % deque->capacity;
    return deque->indices[slot];
}

__device__ __forceinline__ void fisher_deque_pop_front_f64_v2(
    NeoFisherDequeF64V2* deque) {
    deque->head = (deque->head + 1) % deque->capacity;
    --deque->length;
}

__device__ __forceinline__ void fisher_deque_pop_back_f64_v2(
    NeoFisherDequeF64V2* deque) {
    --deque->length;
}

__device__ __forceinline__ void fisher_deque_push_back_f64_v2(
    NeoFisherDequeF64V2* deque, int index) {
    const int slot = (deque->head + deque->length) % deque->capacity;
    deque->indices[slot] = index;
    ++deque->length;
}

__device__ __forceinline__ bool fisher_admit_midpoint_f64_v2(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int index,
    int period,
    double midpoint,
    NeoFisherDequeF64V2* minimum,
    NeoFisherDequeF64V2* maximum) {
    while (minimum->length > 0) {
        double last = 0.0;
        if (!fisher_midpoint_f64_v2(
                high, low, fisher_deque_back_f64_v2(minimum), &last)) return false;
        if (!(last >= midpoint)) break;
        fisher_deque_pop_back_f64_v2(minimum);
    }
    fisher_deque_push_back_f64_v2(minimum, index);

    while (maximum->length > 0) {
        double last = 0.0;
        if (!fisher_midpoint_f64_v2(
                high, low, fisher_deque_back_f64_v2(maximum), &last)) return false;
        if (!(last <= midpoint)) break;
        fisher_deque_pop_back_f64_v2(maximum);
    }
    fisher_deque_push_back_f64_v2(maximum, index);

    const int start = index + 1 - period;
    while (minimum->length > 0 && fisher_deque_front_f64_v2(minimum) < start) {
        fisher_deque_pop_front_f64_v2(minimum);
    }
    while (maximum->length > 0 && fisher_deque_front_f64_v2(maximum) < start) {
        fisher_deque_pop_front_f64_v2(maximum);
    }
    return true;
}

__device__ __forceinline__ void fisher_reset_deques_f64_v2(
    NeoFisherDequeF64V2* minimum,
    NeoFisherDequeF64V2* maximum,
    int* finite_bars,
    double* value1,
    double* previous_fisher) {
    fisher_deque_clear_f64_v2(minimum);
    fisher_deque_clear_f64_v2(maximum);
    *finite_bars = 0;
    *value1 = 0.0;
    *previous_fisher = 0.0;
}

__device__ __forceinline__ void fisher_row_deque_f64_v2(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    int period,
    int first_valid,
    double* __restrict__ fisher_out,
    double* __restrict__ signal_out,
    int* __restrict__ storage) {
    const int capacity = period + 1;
    NeoFisherDequeF64V2 minimum;
    NeoFisherDequeF64V2 maximum;
    fisher_deque_init_f64_v2(&minimum, storage, capacity);
    fisher_deque_init_f64_v2(&maximum, storage + capacity, capacity);
    int finite_bars = 0;
    double previous_fisher = 0.0;
    double value1 = 0.0;

    for (int index = first_valid; index < n; ++index) {
        double midpoint = 0.0;
        if (!fisher_midpoint_f64_v2(high, low, index, &midpoint)) {
            fisher_reset_deques_f64_v2(
                &minimum, &maximum, &finite_bars, &value1, &previous_fisher);
            continue;
        }
        if (!fisher_admit_midpoint_f64_v2(
                high, low, index, period, midpoint, &minimum, &maximum)) {
            fisher_reset_deques_f64_v2(
                &minimum, &maximum, &finite_bars, &value1, &previous_fisher);
            continue;
        }
        if (finite_bars < period) ++finite_bars;
        if (finite_bars < period) continue;

        double minimum_value = 0.0;
        double maximum_value = 0.0;
        if (!fisher_midpoint_f64_v2(
                high, low, fisher_deque_front_f64_v2(&minimum), &minimum_value)
            || !fisher_midpoint_f64_v2(
                high, low, fisher_deque_front_f64_v2(&maximum), &maximum_value)) {
            fisher_reset_deques_f64_v2(
                &minimum, &maximum, &finite_bars, &value1, &previous_fisher);
            continue;
        }
        double fisher = 0.0;
        double signal = 0.0;
        if (!fisher_transition_f64_v2(
                midpoint,
                minimum_value,
                maximum_value,
                &value1,
                &previous_fisher,
                &fisher,
                &signal)) {
            fisher_reset_deques_f64_v2(
                &minimum, &maximum, &finite_bars, &value1, &previous_fisher);
            continue;
        }
        if (fisher_out != nullptr) fisher_out[index] = fisher;
        if (signal_out != nullptr) signal_out[index] = signal;
    }
}

// All threads enter so the qNaN initialization and barrier are block-uniform.
// Thread zero then owns the recurrence for the block's single period tuple.
__device__ __forceinline__ void fisher_row_f64_v2(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    int period,
    int first_valid,
    double* __restrict__ fisher_out,
    double* __restrict__ signal_out,
    int* __restrict__ deque_storage) {
    const double qnan = fisher_qnan_f64_v2();
    for (int index = (int)threadIdx.x; index < n; index += (int)blockDim.x) {
        if (fisher_out != nullptr) fisher_out[index] = qnan;
        if (signal_out != nullptr) signal_out[index] = qnan;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (period <= 0 || period > NEO_FISHER_F64_MAX_PERIOD) return;
    if (first_valid < 0 || first_valid >= n) return;
    fisher_row_deque_f64_v2(
        high,
        low,
        n,
        period,
        first_valid,
        fisher_out,
        signal_out,
        deque_storage);
}

extern "C" __global__
void neoethos_fisher_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = (int)blockIdx.x;
    if (combo >= n_combos || n <= 0) return;
    extern __shared__ int fisher_deque_storage[];
    fisher_row_f64_v2(
        high,
        low,
        n,
        periods[combo],
        first_valid,
        out + (size_t)combo * (size_t)n,
        nullptr,
        fisher_deque_storage);
}

extern "C" __global__
void neoethos_fisher_signal_f64(const double* __restrict__ high,
                                const double* __restrict__ low,
                                int n,
                                const int* __restrict__ periods,
                                int n_combos,
                                int first_valid,
                                double* __restrict__ out)
{
    const int combo = (int)blockIdx.x;
    if (combo >= n_combos || n <= 0) return;
    extern __shared__ int fisher_deque_storage[];
    fisher_row_f64_v2(
        high,
        low,
        n,
        periods[combo],
        first_valid,
        nullptr,
        out + (size_t)combo * (size_t)n,
        fisher_deque_storage);
}

// Production full-pair ABI. One block owns one period tuple and writes both
// matrices from the same recurrence; no second launch replays the state.
extern "C" __global__
void fisher_outputs_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        const int* __restrict__ periods,
                        int n,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ fisher_out,
                        double* __restrict__ signal_out)
{
    const int combo = (int)blockIdx.x;
    if (combo >= n_combos || n <= 0) return;
    extern __shared__ int fisher_deque_storage[];
    fisher_row_f64_v2(
        high,
        low,
        n,
        periods[combo],
        first_valid,
        fisher_out + (size_t)combo * (size_t)n,
        signal_out + (size_t)combo * (size_t)n,
        fisher_deque_storage);
}


// Compatibility ABI for callers that request only the primary matrix. It is
// deliberately routed through the same v2 body as the full-pair production
// ABI, including finite-segment reset, canonical qNaN, and deterministic log.

extern "C" __global__ void neoethos_fisher_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)blockIdx.x;
    if (combo >= n_combos || n <= 0) return;
    extern __shared__ int fisher_deque_storage[];
    fisher_row_f64_v2(
        high,
        low,
        n,
        periods[combo],
        first_valid,
        out + (size_t)combo * (size_t)n,
        nullptr,
        fisher_deque_storage);
}
