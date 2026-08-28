#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#ifndef TRIX_QNAN_U32
#define TRIX_QNAN_U32 0x7fc00000u
#endif

static __device__ __forceinline__ float trix_qnan() {
    return __int_as_float((int)TRIX_QNAN_U32);
}


static __device__ __forceinline__ float ema_step(float prev, float x, float a) {
    return fmaf(a, x - prev, prev);
}


static __device__ __forceinline__ double ema_step_d(double prev, double x, double a) {
    return fma(a, x - prev, prev);
}

extern "C" __global__
void trix_build_logs_f32(const float* __restrict__ prices,
                         int series_len,
                         int first_valid,
                         float* __restrict__ logs)
{
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;
    for (int i = idx; i < series_len; i += stride) {
        if (i < first_valid) {
            logs[i] = 0.0f;
        } else {
            logs[i] = logf(prices[i]);
        }
    }
}


extern "C" __global__
void trix_batch_f32(const float* __restrict__ logs,
                    const int*   __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;

    const int period = periods[combo];
    if (period <= 0 || series_len <= 0) return;

    float* __restrict__ out_row = out + combo * series_len;

    const int warmup_end = first_valid + 3 * (period - 1) + 1;
    const int nan_to = warmup_end < series_len ? warmup_end : series_len;
    const float qn = trix_qnan();
    for (int i = 0; i < nan_to; ++i) out_row[i] = qn;
    if (warmup_end >= series_len) return;

    const float a = 2.0f / (float(period) + 1.0f);
    const float inv_n = 1.0f / float(period);
    const float SCALE = 10000.0f;
    const double a_d = (double)a;
    const double inv_n_d = (double)inv_n;


    float sum1 = 0.0f;
    for (int i = first_valid; i < first_valid + period; ++i) {
        sum1 += logs[i];
    }
    float ema1 = sum1 * inv_n;


    float sum_ema1 = ema1;
    int end2 = first_valid + 2 * period - 1;
    for (int i = first_valid + period; i < end2; ++i) {
        ema1 = ema_step(ema1, logs[i], a);
        sum_ema1 += ema1;
    }


    float ema2 = sum_ema1 * inv_n;


    double sum_ema2 = (double)ema2;
    int end3 = first_valid + 3 * period - 2;
    for (int i = end2; i < end3; ++i) {
        ema1 = ema_step(ema1, logs[i], a);
        ema2 = ema_step(ema2, ema1, a);
        sum_ema2 += (double)ema2;
    }


    double ema3_prev = sum_ema2 * inv_n_d;


    int t = warmup_end;
    double ema3 = ema3_prev;
    {
        const float lv = logs[t];
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step_d(ema3_prev, (double)ema2, a_d);
        out_row[t] = (float)((ema3 - ema3_prev) * (double)SCALE);
        ema3_prev = ema3;
        ++t;
    }


    for (; t < series_len; ++t) {
        const float lv = logs[t];
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step_d(ema3_prev, (double)ema2, a_d);
        out_row[t] = (float)((ema3 - ema3_prev) * (double)SCALE);
        ema3_prev = ema3;
    }
}


extern "C" __global__
void trix_batch_warp_scan_f32(const float* __restrict__ logs,
                              const int*   __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const unsigned mask = 0xffffffffu;
    const int lane = (int)(threadIdx.x & 31);
    const int warp_id = (int)(threadIdx.x >> 5);
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int combo = (int)blockIdx.x * warps_per_block + warp_id;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float qn = trix_qnan();
    float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

    if (period <= 0 || series_len <= 0) {
        for (int i = lane; i < series_len; i += 32) out_row[i] = qn;
        return;
    }

    int fv = first_valid;
    if (fv < 0) fv = 0;

    const int warmup_end = fv + 3 * (period - 1) + 1;
    const int nan_to = warmup_end < series_len ? warmup_end : series_len;
    for (int i = lane; i < nan_to; i += 32) out_row[i] = qn;
    if (warmup_end >= series_len) return;

    const float a = 2.0f / (float(period) + 1.0f);
    const float one_minus_a = 1.0f - a;
    const float inv_n = 1.0f / float(period);
    const float SCALE = 10000.0f;

    float ema1 = 0.0f;
    float ema2 = 0.0f;
    float ema3_prev = 0.0f;
    if (lane == 0) {
        float sum1 = 0.0f;
        for (int i = fv; i < fv + period; ++i) sum1 += logs[i];
        ema1 = sum1 * inv_n;

        float sum_ema1 = ema1;
        const int end2 = fv + 2 * period - 1;
        for (int i = fv + period; i < end2; ++i) {
            ema1 = ema_step(ema1, logs[i], a);
            sum_ema1 += ema1;
        }
        ema2 = sum_ema1 * inv_n;

        float sum_ema2 = ema2;
        const int end3 = fv + 3 * period - 2;
        for (int i = end2; i < end3; ++i) {
            ema1 = ema_step(ema1, logs[i], a);
            ema2 = ema_step(ema2, ema1, a);
            sum_ema2 += ema2;
        }
        ema3_prev = sum_ema2 * inv_n;
    }

    ema1 = __shfl_sync(mask, ema1, 0);
    ema2 = __shfl_sync(mask, ema2, 0);
    ema3_prev = __shfl_sync(mask, ema3_prev, 0);

    for (int t0 = warmup_end; t0 < series_len; t0 += 32) {
        const int t = t0 + lane;
        const float lv = (t < series_len) ? logs[t] : 0.0f;


        float A1 = one_minus_a;
        float B1 = a * lv;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A1, offset);
            const float B_prev = __shfl_up_sync(mask, B1, offset);
            if (lane >= offset) {
                const float A_cur = A1;
                const float B_cur = B1;
                A1 = A_cur * A_prev;
                B1 = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }
        const float ema1_lane = __fmaf_rn(A1, ema1, B1);


        float A2 = one_minus_a;
        float B2 = a * ema1_lane;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A2, offset);
            const float B_prev = __shfl_up_sync(mask, B2, offset);
            if (lane >= offset) {
                const float A_cur = A2;
                const float B_cur = B2;
                A2 = A_cur * A_prev;
                B2 = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }
        const float ema2_lane = __fmaf_rn(A2, ema2, B2);


        float A3 = one_minus_a;
        float B3 = a * ema2_lane;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A3, offset);
            const float B_prev = __shfl_up_sync(mask, B3, offset);
            if (lane >= offset) {
                const float A_cur = A3;
                const float B_cur = B3;
                A3 = A_cur * A_prev;
                B3 = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }
        const float ema3_lane = __fmaf_rn(A3, ema3_prev, B3);

        const float ema3_prev_lane =
            (lane == 0) ? ema3_prev : __shfl_up_sync(mask, ema3_lane, 1);
        if (t < series_len) out_row[t] = (ema3_lane - ema3_prev_lane) * SCALE;

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        ema1 = __shfl_sync(mask, ema1_lane, last_lane);
        ema2 = __shfl_sync(mask, ema2_lane, last_lane);
        ema3_prev = __shfl_sync(mask, ema3_lane, last_lane);
    }
}


extern "C" __global__
void trix_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm)
{
    const int sidx = blockIdx.x;
    if (sidx >= num_series || threadIdx.x != 0) return;
    if (period <= 0 || series_len <= 0) return;

    int fv = first_valids[sidx];
    if (fv < 0) fv = 0;

    const int warmup_end = fv + 3 * (period - 1) + 1;
    const int nan_to = warmup_end < series_len ? warmup_end : series_len;
    const float qn = trix_qnan();
    for (int t = 0; t < nan_to; ++t) {
        out_tm[t * num_series + sidx] = qn;
    }
    if (warmup_end >= series_len) return;

    const float a = 2.0f / (float(period) + 1.0f);
    const float inv_n = 1.0f / float(period);
    const float SCALE = 10000.0f;


    float sum1 = 0.0f;
    for (int i = fv; i < fv + period; ++i) {
        const float px = prices_tm[i * num_series + sidx];
        sum1 += logf(px);
    }
    float ema1 = sum1 * inv_n;


    float sum_ema1 = ema1;
    int end2 = fv + 2 * period - 1;
    for (int i = fv + period; i < end2; ++i) {
        const float lv = logf(prices_tm[i * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        sum_ema1 += ema1;
    }


    float ema2 = sum_ema1 * inv_n;


    float sum_ema2 = ema2;
    int end3 = fv + 3 * period - 2;
    for (int i = end2; i < end3; ++i) {
        const float lv = logf(prices_tm[i * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        sum_ema2 += ema2;
    }


    float ema3_prev = sum_ema2 * inv_n;


    int t = warmup_end;
    float ema3 = ema3_prev;
    {
        const float lv = logf(prices_tm[t * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step(ema3_prev, ema2, a);
        out_tm[t * num_series + sidx] = (ema3 - ema3_prev) * SCALE;
        ema3_prev = ema3;
        ++t;
    }

    for (; t < series_len; ++t) {
        const float lv = logf(prices_tm[t * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step(ema3_prev, ema2, a);
        out_tm[t * num_series + sidx] = (ema3 - ema3_prev) * SCALE;
        ema3_prev = ema3;
    }
}


// ===========================================================================
// S2 f64 LANE — trix
// ===========================================================================
// Reference: src/indicators/trix.rs
//   `trix_prepare`              (:213) — first_valid, needed length, warmup
//   `trix_with_kernel`          (:395) — alloc_with_nan_prefix(len, warmup_end)
//   `trix_compute_into_scalar`  (:251) — three cascaded EMAs over ln(price)
//
// THE WARMUP IS A THREE-STAGE SEED, NOT A SINGLE OFFSET.
//   warmup_end = first + 3*(period - 1) + 1
//   stage 1: SMA of ln(price) over [first, first+period)      -> ema1 seed
//   stage 2: EMA of ln(price) over [end1, first+2*period-1),
//            averaged (with the seed included) -> ema2 seed
//   stage 3: EMA-of-EMA over [end2, first+3*period-2),
//            averaged (with the seed included) -> ema3 seed
//   Each stage's running SUM INCLUDES ITS OWN SEED as its first term
//   (`sum_ema1 = ema1` before the loop). Dropping that term shifts every later
//   bar. Reproduced literally.
//
// SCALE = 10000.0 IS PART OF THE OUTPUT, not a display convenience:
//   out[i] = (ema3 - ema3_prev) * 10000.0.
//
// LOGARITHMS. The f32 kernels above use `logf` (6 call sites). `logf` is not
// `log` at reduced precision — it is a different polynomial with ~1e-7
// relative error, and its output is then differenced against the previous
// bar's, so the error is amplified by the reciprocal of a quantity that is
// itself ~1e-4. This is the single largest f32 error source in the file.
//
// ROUNDINGS PER STAGE PER BAR: `(lv - ema).mul_add(alpha, ema)` -> sub + fma,
// TWO. The f32 kernels use `__fmaf_rn` x6, which fuses correctly but at f32
// width; the widening is the fix, the structure was already right.
//
// THE CPU'S UNROLL BY FOUR CHANGES NOTHING — identical, strictly dependent
// bodies — so one loop per stage here.
// ===========================================================================

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_trix_batch_f64(
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
    const int period = periods[r];

    // `trix_needed_len(period)` is the span the three seeds consume:
    // 3*(period - 1) + 2, i.e. warmup_end - first + 1.
    const long long needed = 3LL * (long long)(period - 1) + 2LL;

    const bool declined =
        (n <= 0) ||
        (period <= 0) || (period > n) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((long long)(n - first_valid) < needed);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    const int warmup_end = first_valid + 3 * (period - 1) + 1;
    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
    if (warmup_end >= n) return;

    const double alpha = 2.0 / ((double)period + 1.0);
    const double inv_n = 1.0 / (double)period;
    const double SCALE = 10000.0;

    // Stage 1 — SMA of ln over [first, first + period).
    double sum1 = 0.0;
    const int end1 = first_valid + period;
    for (int i = first_valid; i < end1; ++i) {
        sum1 += log(prices[i]);
    }
    double ema1 = sum1 * inv_n;

    // Stage 2 — the running sum SEEDED WITH ema1 itself.
    double sum_ema1 = ema1;
    const int end2 = first_valid + 2 * period - 1;
    for (int i = end1; i < end2; ++i) {
        const double lv = log(prices[i]);
        ema1 = fma(lv - ema1, alpha, ema1);
        sum_ema1 += ema1;
    }
    double ema2 = sum_ema1 * inv_n;

    // Stage 3 — likewise seeded with ema2.
    double sum_ema2 = ema2;
    const int end3 = first_valid + 3 * period - 2;
    for (int i = end2; i < end3; ++i) {
        const double lv = log(prices[i]);
        ema1 = fma(lv - ema1, alpha, ema1);
        ema2 = fma(ema1 - ema2, alpha, ema2);
        sum_ema2 += ema2;
    }
    double ema3_prev = sum_ema2 * inv_n;

    for (int src = warmup_end; src < n; ++src) {
        const double lv = log(prices[src]);
        ema1 = fma(lv - ema1, alpha, ema1);
        ema2 = fma(ema1 - ema2, alpha, ema2);
        const double ema3 = fma(ema2 - ema3_prev, alpha, ema3_prev);
        row[src] = (ema3 - ema3_prev) * SCALE;
        ema3_prev = ema3;
    }
}
