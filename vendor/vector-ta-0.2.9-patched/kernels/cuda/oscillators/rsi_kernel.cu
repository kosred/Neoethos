#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef RSI_NAN
#define RSI_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ float clamp_rsi(float x) {
    x = fminf(100.0f, x);
    x = fmaxf(0.0f, x);
    return x;
}


extern "C" __global__
void rsi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * (size_t)series_len;


    if (period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
        return;
    }
    const int fv = first_valid;
    const int tail = series_len - fv;
    if (tail <= period) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
        return;
    }

    const int warm = fv + period;


    for (int i = (int)lane; i < warm; i += 32) out_row[i] = RSI_NAN;


    const float inv_p = 1.0f / (float)period;
    const float beta  = inv_p;
    const float alpha = 1.0f - inv_p;


    float avg_g = 0.0f;
    float avg_l = 0.0f;
    int dead_i = 0;
    if (lane == 0) {
        float prev = prices[fv];
        float sum_g = 0.0f;
        float sum_l = 0.0f;
        for (int i = fv + 1; i <= warm; ++i) {
            const float curr = prices[i];
            const float d = curr - prev;
            prev = curr;
            if (!isfinite(d)) {
                dead_i = 1;
                break;
            }
            if (d > 0.0f) sum_g += d;
            else if (d < 0.0f) sum_l -= d;
        }
        if (!dead_i) {
            avg_g = sum_g * beta;
            avg_l = sum_l * beta;
            const float denom = avg_g + avg_l;
            float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
            out_row[warm] = clamp_rsi(rsi);
        }
    }

    const unsigned mask = 0xffffffffu;
    avg_g = __shfl_sync(mask, avg_g, 0);
    avg_l = __shfl_sync(mask, avg_l, 0);
    dead_i = __shfl_sync(mask, dead_i, 0);

    if (dead_i) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
        return;
    }


    for (int t0 = warm + 1; t0 < series_len; t0 += 32) {
        const int t = t0 + (int)lane;

        float A  = 1.0f;
        float Bg = 0.0f;
        float Bl = 0.0f;
        bool ok = true;
        if (t < series_len) {
            const float p1 = prices[t];
            const float p0 = prices[t - 1];
            const float d = p1 - p0;
            ok = isfinite(d);
            if (ok) {
                const float g = fmaxf(d, 0.0f);
                const float l = fmaxf(-d, 0.0f);
                A  = alpha;
                Bg = beta * g;
                Bl = beta * l;
            }
        }

        const unsigned invalid_mask = __ballot_sync(mask, (t < series_len) && (!ok));


        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev  = __shfl_up_sync(mask, A, offset);
            const float Bg_prev = __shfl_up_sync(mask, Bg, offset);
            const float Bl_prev = __shfl_up_sync(mask, Bl, offset);
            if (lane >= (unsigned)offset) {
                const float A_cur  = A;
                const float Bg_cur = Bg;
                const float Bl_cur = Bl;
                A  = A_cur * A_prev;
                Bg = __fmaf_rn(A_cur, Bg_prev, Bg_cur);
                Bl = __fmaf_rn(A_cur, Bl_prev, Bl_cur);
            }
        }

        const float yg = __fmaf_rn(A, avg_g, Bg);
        const float yl = __fmaf_rn(A, avg_l, Bl);

        if (t < series_len) {
            if (invalid_mask) {
                const int first_bad = __ffs(invalid_mask) - 1;
                if ((int)lane >= first_bad) {
                    out_row[t] = RSI_NAN;
                } else {
                    const float denom = yg + yl;
                    float rsi = (denom == 0.0f) ? 50.0f : (100.0f * yg / denom);
                    out_row[t] = clamp_rsi(rsi);
                }
            } else {
                const float denom = yg + yl;
                float rsi = (denom == 0.0f) ? 50.0f : (100.0f * yg / denom);
                out_row[t] = clamp_rsi(rsi);
            }
        }


        if (invalid_mask) {
            const int remaining = series_len - t0;
            const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
            const int first_bad = __ffs(invalid_mask) - 1;
            if (first_bad <= last_lane) {
                for (int i = t0 + 32 + (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
                return;
            }
        }


        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        avg_g = __shfl_sync(mask, yg, last_lane);
        avg_l = __shfl_sync(mask, yl, last_lane);
    }
}


extern "C" __global__
void rsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    if (period <= 0) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }

    const int fv   = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }

    const int warm = fv + period;
    for (int t = 0; t <= warm && t < rows; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;

    const float inv_p = 1.0f / (float)period;
    const float beta  = 1.0f - inv_p;


    float avg_g = 0.0f, avg_l = 0.0f;
    float sum_g = 0.0f, sum_l = 0.0f;
    bool  has_nan = false;

    for (int t = fv + 1; t <= warm; ++t) {
        const float d = prices_tm[t * cols + s] - prices_tm[(t - 1) * cols + s];
        if (!isfinite(d)) { has_nan = true; break; }
        if (d > 0.0f) sum_g += d;
        else if (d < 0.0f) sum_l -= d;
    }

    if (has_nan) {
        out_tm[warm * cols + s] = NAN;
        avg_g = avg_l = NAN;
    } else {
        avg_g = sum_g * inv_p;
        avg_l = sum_l * inv_p;
        const float denom = avg_g + avg_l;
        float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
        out_tm[warm * cols + s] = clamp_rsi(rsi);
    }


    for (int t = warm + 1; t < rows; ++t) {
        const float d = prices_tm[t * cols + s] - prices_tm[(t - 1) * cols + s];
        const float g = (d > 0.0f) ? d : 0.0f;
        const float l = (d < 0.0f) ? -d : 0.0f;
        avg_g = fmaf(beta, avg_g, inv_p * g);
        avg_l = fmaf(beta, avg_l, inv_p * l);
        const float denom = avg_g + avg_l;
        float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
        out_tm[t * cols + s] = clamp_rsi(rsi);
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — rsi
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/rsi.rs:327 `rsi_compute_into_scalar` (Kernel::Auto
 * resolves to Scalar for rsi at rsi.rs:219, so this IS the reference on every
 * host).
 *
 * Differences from `rsi_batch_f32` above, all of them bugs in the f32 lane:
 *   * f32 clamps the result into [0,100] via `clamp_rsi`. The CPU does NOT
 *     clamp. Clamping hides a wrong value instead of showing it.
 *   * f32 seeds with `sum_g * beta` where `beta = inv_p`; the CPU multiplies
 *     the accumulated sums by `inv_p` in the same place but then recurses with
 *     `beta = 1 - inv_p`. Names collide, meanings do not.
 *   * `RSI_NAN` was `__int_as_float(0x7fffffff)` — an f32 bit pattern. In f64
 *     that is a denormal, not a NaN.
 *
 * ABI: the neoethos sequential lane
 *   (prices, n, periods, n_combos, first_valid, out) — one thread per column,
 *   bars ascending, so the accumulation order is the CPU's exactly.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void rsi_neo_batch_f64(const double* __restrict__ prices,
                       int series_len,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int period = periods[combo];

    // rsi_with_kernel (rsi.rs:189-215): period == 0 || period > len -> Err,
    // and (len - first) < period -> Err. An Err row is all-NaN here rather
    // than a partial series, because the CPU produces no series at all.
    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len ||
        (len - first_valid) < period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int idx0 = first_valid + period;          // alloc_with_nan_prefix(len, first + period)
    const int warm = idx0 < len ? idx0 : len;
    for (int i = 0; i < warm; ++i) o[i] = NEO_F64_NAN;

    const double inv_p = 1.0 / (double)period;
    const double beta  = 1.0 - inv_p;               // rsi.rs:330

    double avg_gain = 0.0;
    double avg_loss = 0.0;
    bool has_nan = false;

    // warm_last = min(first + period, len - 1)  (rsi.rs:336)
    const int warm_last = (idx0 < len - 1) ? idx0 : (len - 1);
    for (int i = first_valid + 1; i <= warm_last; ++i) {
        const double delta = prices[i] - prices[i - 1];
        if (!isfinite(delta)) { has_nan = true; break; }
        if (delta > 0.0)      avg_gain += delta;
        else if (delta < 0.0) avg_loss -= delta;
    }

    if (has_nan) {
        avg_gain = NEO_F64_NAN;
        avg_loss = NEO_F64_NAN;
        if (idx0 < len) o[idx0] = NEO_F64_NAN;
    } else {
        avg_gain *= inv_p;
        avg_loss *= inv_p;
        if (idx0 < len) {
            const double denom = avg_gain + avg_loss;
            o[idx0] = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
        }
    }

    // rsi.rs:372-413. The CPU unrolls by two; the arithmetic per bar is
    // identical and strictly sequential, so a plain loop is bit-equal.
    for (int j = idx0 + 1; j < len; ++j) {
        const double d = prices[j] - prices[j - 1];
        const double g = (d > 0.0) ? d : 0.0;
        const double l = (d < 0.0) ? -d : 0.0;
        avg_gain = fma(avg_gain, beta, inv_p * g);   // avg_gain.mul_add(beta, inv_p * g)
        avg_loss = fma(avg_loss, beta, inv_p * l);
        const double denom = avg_gain + avg_loss;
        o[j] = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
    }
}
