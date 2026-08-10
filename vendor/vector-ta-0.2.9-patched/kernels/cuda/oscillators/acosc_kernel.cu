#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


__device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
__device__ __forceinline__ void kahan_add_sub(float add, float sub, float &sum, float &c) {
    kahan_add(add, sum, c);
    kahan_add(-sub, sum, c);
}


static __device__ __forceinline__ float acosc_median(const float* __restrict__ high,
                                                     const float* __restrict__ low,
                                                     int idx) {
    return (high[idx] + low[idx]) * 0.5f;
}

static __device__ __forceinline__ float acosc_ao_at(const float* __restrict__ high,
                                                    const float* __restrict__ low,
                                                    int idx,
                                                    int first_valid) {
    float sum34 = 0.0f;
    float sum5 = 0.0f;
    const int rel = idx - first_valid;
#pragma unroll
    for (int k = 0; k < 34; ++k) {
        const float med = acosc_median(high, low, idx - k);
        sum34 += med;
        if (rel >= 38 && k < 5) sum5 += med;
    }
    if (rel < 38) {


        if (rel == 34) {
            sum5 = acosc_median(high, low, first_valid + 1) +
                   acosc_median(high, low, first_valid + 2) +
                   acosc_median(high, low, first_valid + 3) +
                   acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx);
        } else if (rel == 35) {
            sum5 = acosc_median(high, low, first_valid + 2) +
                   acosc_median(high, low, first_valid + 3) +
                   acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx - 1) +
                   acosc_median(high, low, idx);
        } else if (rel == 36) {
            sum5 = acosc_median(high, low, first_valid + 3) +
                   acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx - 2) +
                   acosc_median(high, low, idx - 1) +
                   acosc_median(high, low, idx);
        } else {

            sum5 = acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx - 3) +
                   acosc_median(high, low, idx - 2) +
                   acosc_median(high, low, idx - 1) +
                   acosc_median(high, low, idx);
        }
    }
    return sum5 * (1.0f / 5.0f) - sum34 * (1.0f / 34.0f);
}

static __device__ __forceinline__ void acosc_ao_chain_fast(const float* __restrict__ high,
                                                            const float* __restrict__ low,
                                                            int idx,
                                                            float& ao0,
                                                            float& ao1,
                                                            float& ao2,
                                                            float& ao3,
                                                            float& ao4,
                                                            float& ao5) {
    constexpr float INV5 = 1.0f / 5.0f;
    constexpr float INV34 = 1.0f / 34.0f;

    int t = idx - 5;
    float sum34 = 0.0f;
    float sum5 = 0.0f;

#pragma unroll
    for (int k = 0; k < 34; ++k) {
        const float med = acosc_median(high, low, t - k);
        sum34 += med;
        if (k < 5) {
            sum5 += med;
        }
    }
    ao5 = sum5 * INV5 - sum34 * INV34;

    t = idx - 4;
    float med = acosc_median(high, low, t);
    sum34 += med - acosc_median(high, low, t - 34);
    sum5 += med - acosc_median(high, low, t - 5);
    ao4 = sum5 * INV5 - sum34 * INV34;

    t = idx - 3;
    med = acosc_median(high, low, t);
    sum34 += med - acosc_median(high, low, t - 34);
    sum5 += med - acosc_median(high, low, t - 5);
    ao3 = sum5 * INV5 - sum34 * INV34;

    t = idx - 2;
    med = acosc_median(high, low, t);
    sum34 += med - acosc_median(high, low, t - 34);
    sum5 += med - acosc_median(high, low, t - 5);
    ao2 = sum5 * INV5 - sum34 * INV34;

    t = idx - 1;
    med = acosc_median(high, low, t);
    sum34 += med - acosc_median(high, low, t - 34);
    sum5 += med - acosc_median(high, low, t - 5);
    ao1 = sum5 * INV5 - sum34 * INV34;

    t = idx;
    med = acosc_median(high, low, t);
    sum34 += med - acosc_median(high, low, t - 34);
    sum5 += med - acosc_median(high, low, t - 5);
    ao0 = sum5 * INV5 - sum34 * INV34;
}


extern "C" __global__ __launch_bounds__(256, 4)
void acosc_batch_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     int series_len,
                     int first_valid,
                     float* __restrict__ out_osc,
                     float* __restrict__ out_change) {
    if (series_len <= 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;
    const int warm = fv + 34 + 5 - 1;
    const float nn = CUDART_NAN_F;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < series_len; i += stride) {
        if (i < warm) {
            out_osc[i] = nn;
            out_change[i] = nn;
            continue;
        }


        float ao0, ao1, ao2, ao3, ao4, ao5 = 0.0f;
        const int rel = i - fv;
        if (rel > 43) {
            acosc_ao_chain_fast(high, low, i, ao0, ao1, ao2, ao3, ao4, ao5);
        } else {
            ao0 = acosc_ao_at(high, low, i, fv);
            ao1 = acosc_ao_at(high, low, i - 1, fv);
            ao2 = acosc_ao_at(high, low, i - 2, fv);
            ao3 = acosc_ao_at(high, low, i - 3, fv);
            ao4 = acosc_ao_at(high, low, i - 4, fv);
            if (i != warm) {
                ao5 = acosc_ao_at(high, low, i - 5, fv);
            }
        }

        const float sma5_ao = (ao0 + ao1 + ao2 + ao3 + ao4) * (1.0f / 5.0f);
        const float res = ao0 - sma5_ao;

        float prev_res = 0.0f;
        if (i != warm) {
            const float sma5_ao_prev = (ao1 + ao2 + ao3 + ao4 + ao5) * (1.0f / 5.0f);
            prev_res = ao1 - sma5_ao_prev;
        }

        out_osc[i] = res;
        out_change[i] = res - prev_res;
    }
}


extern "C" __global__
void acosc_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const int* __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_osc_tm,
                                     float* __restrict__ out_change_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0) return;

    const int stride = num_series;
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_osc_tm[t * stride + s] = CUDART_NAN_F;
        out_change_tm[t * stride + s] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (fv >= series_len) return;
    if ((series_len - fv) < 39) return;

    const int P5 = 5;
    const int P34 = 34;
    const float INV5 = 1.0f / 5.0f;
    const float INV34 = 1.0f / 34.0f;
    float q5[5];
    float q34[34];
    float q5ao[5];
    float sum5 = 0.0f, c5 = 0.0f;
    float sum34 = 0.0f, c34 = 0.0f;
    float sum5ao = 0.0f, c5ao = 0.0f;
    int i5 = 0, i34 = 0, i5ao = 0;

    #pragma unroll
    for (int k = 0; k < P34; ++k) {
        const int t = fv + k;
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        kahan_add(med, sum34, c34);
        q34[k] = med;
        if (k < P5) {
            kahan_add(med, sum5, c5);
            q5[k] = med;
        }
    }

    for (int t = fv + P34; t < fv + P34 + P5 - 1; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[i34];
        kahan_add_sub(med, old34, sum34, c34);
        q34[i34] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[i5];
        kahan_add_sub(med, old5, sum5, c5);
        q5[i5] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        kahan_add(ao, sum5ao, c5ao);
        q5ao[i5ao] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;
    }

    float prev_res = 0.0f;
    for (int t = fv + P34 + P5 - 1; t < series_len; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[i34];
        kahan_add_sub(med, old34, sum34, c34);
        q34[i34] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[i5];
        kahan_add_sub(med, old5, sum5, c5);
        q5[i5] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        const float old_ao = q5ao[i5ao];
        kahan_add_sub(ao, old_ao, sum5ao, c5ao);
        q5ao[i5ao] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;

        const float sma5ao = sum5ao * INV5;
        const float res = ao - sma5ao;
        const float mom = res - prev_res;
        prev_res = res;
        out_osc_tm[t * stride + s] = res;
        out_change_tm[t * stride + s] = mom;
    }
}


extern "C" __global__
void acosc_many_series_one_param_f32_warp(const float* __restrict__ high_tm,
                                          const float* __restrict__ low_tm,
                                          const int* __restrict__ first_valids,
                                          int num_series,
                                          int series_len,
                                          float* __restrict__ out_osc_tm,
                                          float* __restrict__ out_change_tm) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_idx = blockIdx.x;
    const int s = warp_idx * WARP_SIZE + lane;
    if (s >= num_series || series_len <= 0) return;

    const int stride = num_series;
    int fv = first_valids[s]; if (fv < 0) fv = 0;


    for (int t = 0; t < series_len; ++t) {
        out_osc_tm[t * stride + s] = CUDART_NAN_F;
        out_change_tm[t * stride + s] = CUDART_NAN_F;
    }

    if (fv >= series_len) return;
    const int P5 = 5;
    const int P34 = 34;
    if ((series_len - fv) < (P34 + P5)) return;

    const float INV5 = 1.0f / 5.0f;
    const float INV34 = 1.0f / 34.0f;

    extern __shared__ float smem[];
    float* q34  = smem;
    float* q5   = q34  + P34 * WARP_SIZE;
    float* q5ao = q5   + P5  * WARP_SIZE;

    auto SM = [&](int k) { return k * WARP_SIZE + lane; };

    float sum34 = 0.0f, c34 = 0.0f;
    float sum5  = 0.0f, c5  = 0.0f;
    float sum5ao= 0.0f, c5ao= 0.0f;
    int i34 = 0, i5 = 0, i5ao = 0;


    #pragma unroll
    for (int k = 0; k < P34; ++k) {
        const int t = fv + k;
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        kahan_add(med, sum34, c34);
        q34[SM(k)] = med;
        if (k < P5) {
            kahan_add(med, sum5, c5);
            q5[SM(k)] = med;
        }
    }


    for (int t = fv + P34; t < fv + P34 + P5 - 1; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[SM(i34)];
        kahan_add_sub(med, old34, sum34, c34);
        q34[SM(i34)] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[SM(i5)];
        kahan_add_sub(med, old5, sum5, c5);
        q5[SM(i5)] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        kahan_add(ao, sum5ao, c5ao);
        q5ao[SM(i5ao)] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;
    }

    float prev_res = 0.0f;
    for (int t = fv + P34 + P5 - 1; t < series_len; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[SM(i34)];
        kahan_add_sub(med, old34, sum34, c34);
        q34[SM(i34)] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[SM(i5)];
        kahan_add_sub(med, old5, sum5, c5);
        q5[SM(i5)] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        const float old_ao = q5ao[SM(i5ao)];
        kahan_add_sub(ao, old_ao, sum5ao, c5ao);
        q5ao[SM(i5ao)] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;

        const float sma5ao = sum5ao * INV5;
        const float res = ao - sma5ao;
        const float mom = res - prev_res;
        prev_res = res;
        out_osc_tm[t * stride + s] = res;
        out_change_tm[t * stride + s] = mom;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — acosc (output "osc", index 0 of
 * fallback_outputs &["osc","change"])
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/acosc.rs:196 `acosc_scalar`.
 *
 * PERIOD-INVARIANT and FIRST-VALID-IGNORED, both by construction, not by
 * omission: `acosc_scalar` takes no period at all (the windows are the fixed
 * 5 and 34 of the Awesome Oscillator) and it starts at absolute index 0.
 * Declared F64FirstValidRule::Ignored so the table does not claim a warmup the
 * kernel does not honour.
 *
 * THE THREE RUNNING SUMS ARE SLIDING, NOT RECOMPUTED. `sum34 += med - queue34[idx]`
 * accumulates rounding across the whole series; recomputing the mean over the
 * window each bar would be more accurate and WRONG — it is a different number
 * from the reference. Same for sum5 and for the sum of the last five AO values.
 *
 * INV5/INV34 are `1.0/5.0` and `1.0/34.0` as the CPU spells them (acosc.rs:199).
 * 1/34 is not representable; forming it once and multiplying is not the same as
 * dividing by 34 each bar, and the CPU multiplies.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void acosc_neo_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int series_len,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods; (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    const int P5 = 5, P34 = 34;
    const double INV5  = 1.0 / 5.0;
    const double INV34 = 1.0 / 34.0;

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    // acosc.rs:205 debug_asserts len >= 39; below that the CPU produces nothing.
    if (len < P34 + P5) return;

    double queue5[5], queue34[34], queue5_ao[5];
    double sum5 = 0.0, sum34 = 0.0, sum5_ao = 0.0;
    int idx5 = 0, idx34 = 0, idx5_ao = 0;
    for (int t = 0; t < 5; ++t) { queue5[t] = 0.0; queue5_ao[t] = 0.0; }
    for (int t = 0; t < 34; ++t) queue34[t] = 0.0;

    for (int i = 0; i < P34; ++i) {
        const double med = (high[i] + low[i]) * 0.5;
        sum34 += med;
        queue34[i] = med;
        if (i < P5) { sum5 += med; queue5[i] = med; }
    }
    for (int i = P34; i < P34 + P5 - 1; ++i) {
        const double med = (high[i] + low[i]) * 0.5;
        sum34 += med - queue34[idx34];
        queue34[idx34] = med;
        idx34 += 1; if (idx34 == P34) idx34 = 0;
        const double sma34 = sum34 * INV34;

        sum5 += med - queue5[idx5];
        queue5[idx5] = med;
        idx5 += 1; if (idx5 == P5) idx5 = 0;
        const double sma5 = sum5 * INV5;

        const double ao = sma5 - sma34;
        sum5_ao += ao;
        queue5_ao[idx5_ao] = ao;
        idx5_ao += 1;
    }
    if (idx5_ao == P5) idx5_ao = 0;

    double prev_res = 0.0;
    for (int i = P34 + P5 - 1; i < len; ++i) {
        const double med = (high[i] + low[i]) * 0.5;
        sum34 += med - queue34[idx34];
        queue34[idx34] = med;
        idx34 += 1; if (idx34 == P34) idx34 = 0;
        const double sma34 = sum34 * INV34;

        sum5 += med - queue5[idx5];
        queue5[idx5] = med;
        idx5 += 1; if (idx5 == P5) idx5 = 0;
        const double sma5 = sum5 * INV5;

        const double ao = sma5 - sma34;
        const double old_ao = queue5_ao[idx5_ao];
        sum5_ao += ao - old_ao;
        queue5_ao[idx5_ao] = ao;
        idx5_ao += 1; if (idx5_ao == P5) idx5_ao = 0;

        const double sma5_ao = sum5_ao * INV5;
        const double res = ao - sma5_ao;
        prev_res = res;                 // `change` (output 1) is res - prev_res
        o[i] = res;                     // output "osc"
    }
    (void)prev_res;
}
