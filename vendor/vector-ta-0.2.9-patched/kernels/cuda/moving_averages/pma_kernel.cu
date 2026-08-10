#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float nan32() { return nanf(""); }


struct lwma7_recur_f32 {
    float buf[7];
    int   head;
    int   count;
    int   ticks;
    float s1, c1;
    float s2, c2;

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = 0.f;
        head = 0; count = 0; ticks = 0; s1 = c1 = 0.f; s2 = c2 = 0.f;
    }

    __device__ __forceinline__ void kahan_add(float y, float &s, float &c) {
        float t = __fadd_rn(s, __fsub_rn(y, c));
        c = __fsub_rn(__fsub_rn(t, s), __fsub_rn(y, c));
        s = t;
    }

    __device__ __forceinline__ void push(float x) {
        const float old = buf[head];
        buf[head] = x;
        head++; if (head == 7) head = 0;
        if (count < 7) count++;

        const float s1_old = s1;

        kahan_add(__fmaf_rn(7.f, x, -s1_old), s2, c2);

        kahan_add(x, s1, c1);
        kahan_add(-old, s1, c1);


        ticks++;
        if ((ticks & 0x3FF) == 0) {
            float ns1 = 0.f, nc1 = 0.f, ns2 = 0.f, nc2 = 0.f;
            #pragma unroll
            for (int i = 0; i < 7; ++i) {
                const int idx = (head + i) % 7;
                const float v = buf[idx];
                kahan_add(v, ns1, nc1);
                kahan_add(__fmul_rn((float)(i + 1), v), ns2, nc2);
            }
            s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
        }
    }

    __device__ __forceinline__ float value() const { return __fmul_rn(s2, 1.f / 28.f); }


    __device__ __forceinline__ void seed_from7(const float x[7]) {
        #pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = x[i];
        head = 0; count = 7; ticks = 0;
        float sum = 0.f; float wsum = 0.f;
        #pragma unroll
        for (int i = 0; i < 7; ++i) { sum = __fadd_rn(sum, x[i]); wsum = __fadd_rn(wsum, __fmul_rn((float)(i+1), x[i])); }
        s1 = sum; c1 = 0.f; s2 = wsum; c2 = 0.f;
    }
};


struct lwma4_recur_f32 {
    float buf[4];
    int   head;
    int   count;
    int   ticks;
    float s1, c1, s2, c2;

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < 4; ++i) buf[i] = 0.f;
        head = 0; count = 0; ticks = 0; s1 = c1 = s2 = c2 = 0.f;
    }
    __device__ __forceinline__ void kahan_add(float y, float &s, float &c) {
        float t = __fadd_rn(s, __fsub_rn(y, c));
        c = __fsub_rn(__fsub_rn(t, s), __fsub_rn(y, c));
        s = t;
    }
    __device__ __forceinline__ void push(float x) {
        if (count < 4) {
            buf[head] = x; head++; if (head == 4) head = 0; count++;
            kahan_add(x, s1, c1);
            kahan_add(__fmul_rn((float)count, x), s2, c2);
        } else {
            const float old = buf[head]; buf[head] = x; head++; if (head == 4) head = 0;
            const float s1_old = s1;
            kahan_add(__fmaf_rn(4.f, x, -s1_old), s2, c2);
            kahan_add(x, s1, c1);
            kahan_add(-old, s1, c1);
            ticks++;
            if ((ticks & 0x3FF) == 0) {
                float ns1 = 0.f, nc1 = 0.f, ns2 = 0.f, nc2 = 0.f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int idx = (head + i) % 4;
                    const float v = buf[idx];
                    kahan_add(v, ns1, nc1);
                    kahan_add(__fmul_rn((float)(i + 1), v), ns2, nc2);
                }
                s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
            }
        }
    }
    __device__ __forceinline__ float value() const { return __fmul_rn(s2, 0.1f); }
};

static __device__ __forceinline__ void pma_batch_core(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return;

    const float nan_f = nan32();
    if (series_len <= 0) return;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_predict = first_valid + 6;
    const int warm_trigger = first_valid + 9;

    float* predict_row = out_predict + combo * series_len;
    float* trigger_row = out_trigger + combo * series_len;


    {
        int stop = (series_len < warm_predict) ? series_len : warm_predict;
        for (int i = 0; i < stop; ++i) predict_row[i] = nan_f;
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int i = 0; i < stop; ++i) trigger_row[i] = nan_f;
    }

    if (warm_predict >= series_len) return;


    const int j0 = warm_predict;

    lwma7_recur_f32 wma1; wma1.init();
    float seed7[7];
    #pragma unroll
    for (int k = 0; k < 7; ++k) seed7[k] = prices[j0 - 6 + k];
    wma1.seed_from7(seed7);


    lwma7_recur_f32 wma2; wma2.init();


    lwma4_recur_f32 trig; trig.init();

    float w1 = wma1.value();
    wma2.push(w1);
    float w2 = wma2.value();
    float pr = 2.f * w1 - w2;
    predict_row[j0] = pr;
    trig.push(pr);


    for (int j = j0 + 1; j < series_len; ++j) {
        const float x_new = prices[j];
        wma1.push(x_new);
        w1 = wma1.value();

        wma2.push(w1);
        w2 = wma2.value();

        pr = 2.f * w1 - w2;
        predict_row[j] = pr;

        trig.push(pr);
        if (j >= warm_trigger) {
            trigger_row[j] = trig.value();
        } else {
            trigger_row[j] = nan_f;
        }
    }
}

extern "C" __global__ void pma_batch_f32(const float* __restrict__ prices,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          float* __restrict__ out_predict,
                                          float* __restrict__ out_trigger) {
    pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}


extern "C" __global__ void pma_batch_tiled_f32_tile128(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

extern "C" __global__ void pma_batch_tiled_f32_tile256(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

static __device__ __forceinline__ void pma_many_series_core(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm)
{
    const int series = (blockIdx.y > 0) ? (blockIdx.x * blockDim.y + threadIdx.y)
                                        : (blockIdx.x);

    if (series >= num_series) return;
    if (threadIdx.x != 0) return;

    const int stride = num_series;
    const float nan_f = nan32();

    int fv = first_valids ? first_valids[series] : 0;
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;

    const int warm_predict = fv + 6;
    const int warm_trigger = fv + 9;


    {
        int stop = (series_len < warm_predict) ? series_len : warm_predict;
        for (int row = 0; row < stop; ++row) out_predict_tm[row * stride + series] = nan_f;
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) out_trigger_tm[row * stride + series] = nan_f;
    }

    if (warm_predict >= series_len) return;


    const int j0 = warm_predict;
    lwma7_recur_f32 wma1; wma1.init();
    float seed7tm[7];
    #pragma unroll
    for (int k = 0; k < 7; ++k) seed7tm[k] = prices_tm[(j0 - 6 + k) * stride + series];
    wma1.seed_from7(seed7tm);
    lwma7_recur_f32 wma2; wma2.init();
    lwma4_recur_f32 trig; trig.init();

    float w1 = wma1.value();
    wma2.push(w1);
    float w2 = wma2.value();
    float pr = 2.f * w1 - w2;
    out_predict_tm[j0 * stride + series] = pr;
    trig.push(pr);

    for (int row = j0 + 1; row < series_len; ++row) {
        const float x_new = prices_tm[row * stride + series];
        wma1.push(x_new);
        w1 = wma1.value();
        wma2.push(w1);
        w2 = wma2.value();
        pr = 2.f * w1 - w2;
        const int idx = row * stride + series;
        out_predict_tm[idx] = pr;
        trig.push(pr);
        if (row >= warm_trigger) {
            out_trigger_tm[idx] = trig.value();
        } else {
            out_trigger_tm[idx] = nan_f;
        }
    }
}

extern "C" __global__ void pma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    pma_many_series_core(prices_tm, num_series, series_len, first_valids, out_predict_tm, out_trigger_tm);
}

extern "C" __global__ void pma_ms1p_tiled_f32_tx1_ty2(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    pma_many_series_core(prices_tm, num_series, series_len, first_valids, out_predict_tm, out_trigger_tm);
}

extern "C" __global__ void pma_ms1p_tiled_f32_tx1_ty4(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    pma_many_series_core(prices_tm, num_series, series_len, first_valids, out_predict_tm, out_trigger_tm);
}


// ===========================================================================
// S1 f64 LANE  --  pma
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/pma.rs -- `pma_scalar` (:214), `pma_with_kernel` (:187)
//
// PERIOD-INVARIANT. `compute_pma_batch` (cpu_batch.rs:15763) takes
// `|_params|` and constructs `PmaParams::default()`; pma has no parameters at
// all, so every row of a sweep is byte-identical.
//
// `pma_with_kernel` maps `Auto` to `Kernel::Scalar` (pma.rs:192-195).
//
// PRIMARY OUTPUT: `predict`. cpu_batch.rs:15776 maps "value" to `out.predict`;
// `trigger` is a second series. Unlike the other multi-output indicators in
// this shard, `trigger` is DOWNSTREAM of `predict` (it is a 4-tap sum of the
// predict series), so dropping it cannot perturb what is emitted -- the
// dependency runs one way only. Its accumulators are therefore not carried.
//
// ARITHMETIC ORDER -- this indicator is nothing BUT accumulation order:
//   The seed sum is the explicit tree `((x0+x1)+(x2+x3)) + ((x4+x5)+x6)` --
//   a BALANCED tree, not the ascending chain an ordinary loop would produce.
//   Written out literally.
//   The weighted seed is `((s01 + s23) + s45) + 7*x6` where
//   `s01 = x0.mul_add(1.0, 2.0*x1)`, `s23 = 3*x2 + 4*x3`, `s45 = 5*x4 + 6*x5`.
//   Note `s01` is a `mul_add` and `s23`/`s45` are not -- an inconsistency in
//   the CPU that is nonetheless the reference, so it is reproduced exactly.
//   The slides are `S = 7.0.mul_add(x_new, S) - old_A` and
//   `S1 = 7.0.mul_add(w1, S1) - old_A1`: ONE fused rounding then a subtract.
//   The output is `pr = 2.0.mul_add(w1, -w2)` -- ONE rounding; `2*w1 - w2`
//   would be two.
//   `INV_28 = 1.0/28.0` is formed once and MULTIPLIED, never divided by 28 at
//   the use site. 1/28 is not representable, so the two differ.
//
// WARMUP: `alloc_with_nan_prefix(n, first + 7)`, but `predict[first + 6]` is
// then written by the seed block, so the first emitted bar is `first + 6` and
// the prefix at that index is overwritten. The input is declined outright when
// `n <= first + 6`.
// ===========================================================================

#ifndef NEO_S1_QNAN_DEFINED
#define NEO_S1_QNAN_DEFINED
// The f32 kernels in this crate spell NaN `__int_as_float(0x7fc00000)`. That is
// a 32-bit pattern; widening it is a value change, not a cast. This is the f64
// quiet-NaN pattern, stated once per translation unit.
__device__ __forceinline__ double neo_s1_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}
__device__ __forceinline__ bool neo_s1_isnan(double x) { return x != x; }
#endif

extern "C" __global__ void neoethos_pma_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    double* __restrict__ row = out + (size_t)r * (size_t)n;
    (void)periods;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (n <= first_valid + 6);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + 7;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();

    const double INV_28 = 1.0 / 28.0;

    double x_ring[7];
    double w_ring[7];
    for (int k = 0; k < 7; ++k) { x_ring[k] = 0.0; w_ring[k] = 0.0; }
    int x_head = 0, w_head = 0;

    double A = 0.0, S = 0.0, A1 = 0.0, S1 = 0.0;

    const int j0 = first_valid + 6;

    const double x0 = prices[j0 - 6];
    const double x1 = prices[j0 - 5];
    const double x2 = prices[j0 - 4];
    const double x3 = prices[j0 - 3];
    const double x4 = prices[j0 - 2];
    const double x5 = prices[j0 - 1];
    const double x6 = prices[j0];

    x_ring[0] = x0; x_ring[1] = x1; x_ring[2] = x2; x_ring[3] = x3;
    x_ring[4] = x4; x_ring[5] = x5; x_ring[6] = x6;

    A = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + x6);

    {
        const double s01 = fma(x0, 1.0, 2.0 * x1);
        const double s23 = (3.0 * x2) + (4.0 * x3);
        const double s45 = (5.0 * x4) + (6.0 * x5);
        S = (s01 + s23) + s45 + 7.0 * x6;
    }

    double w1 = S * INV_28;

    {
        const double old_A1 = A1;
        const double old_w = w_ring[w_head];
        S1 = fma(7.0, w1, S1) - old_A1;
        A1 = A1 + w1 - old_w;
        w_ring[w_head] = w1;
        if (++w_head == 7) w_head = 0;
    }

    double w2 = S1 * INV_28;
    double pr = fma(2.0, w1, -w2);
    row[j0] = pr;

    for (int j = j0 + 1; j < n; ++j) {
        const double x_new = prices[j];
        const double x_old = x_ring[x_head];
        const double old_A = A;

        A = A + x_new - x_old;
        S = fma(7.0, x_new, S) - old_A;

        x_ring[x_head] = x_new;
        if (++x_head == 7) x_head = 0;

        w1 = S * INV_28;

        const double old_A1 = A1;
        const double w_old = w_ring[w_head];
        S1 = fma(7.0, w1, S1) - old_A1;
        A1 = A1 + w1 - w_old;

        w_ring[w_head] = w1;
        if (++w_head == 7) w_head = 0;

        w2 = S1 * INV_28;

        pr = fma(2.0, w1, -w2);
        row[j] = pr;
    }
}
