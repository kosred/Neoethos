#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    float block_sum = 0.0f;
    if (wid == 0 && lane == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

        float c = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < num_warps; ++i) {
            float y = warp_sums[i] - c;
            float t = block_sum + y;
            c = (t - block_sum) - y;
            block_sum = t;
        }
        block_sum += c;
    }

    return (wid == 0 && lane == 0) ? block_sum : 0.0f;
}


__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ float safe_scale_100_over_close(float c) {
    return (isfinite(c) && c != 0.0f) ? (100.0f / c) : dev_nan();
}


__device__ __forceinline__ void ema_update_kahan(float& atr, float& c, float alpha, float x) {

    float y = __fmaf_rn(alpha, x - (atr + c), 0.0f);
    float t = atr + y;
    c = (t - atr) - y;
    atr = t;
}

extern "C" __global__ void natr_tr_from_hlc_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int first_valid,
    float* __restrict__ tr_out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len) return;
    if (t < first_valid) {
        tr_out[t] = 0.0f;
        return;
    }
    const float hi = high[t];
    const float lo = low[t];
    if (t == first_valid) {
        tr_out[t] = hi - lo;
        return;
    }
    const float pc = close[t - 1];
    const float hl = hi - lo;
    const float hc = fabsf(hi - pc);
    const float lc = fabsf(lo - pc);
    tr_out[t] = fmaxf(hl, fmaxf(hc, lc));
}


extern "C" __global__ void natr_make_inv_close100(
    const float* __restrict__ close, int len, float* __restrict__ inv_close100)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < len) {
        inv_close100[t] = safe_scale_100_over_close(close[t]);
    }
}


extern "C" __global__ void natr_batch_f32(
    const float* __restrict__ tr,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;


    if (first_valid >= series_len || warm >= series_len) {
        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[base + idx] = dev_nan();
        }
        return;
    }


    for (int idx = threadIdx.x; idx < warm; idx += blockDim.x) {
        out[base + idx] = dev_nan();
    }
    __syncthreads();


    const int start = first_valid;
    float local_sum = 0.0f;
    float local_c   = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const float v = tr[start + k];
        float y = v - local_c;
        float t = local_sum + y;
        local_c = (t - local_sum) - y;
        local_sum = t;
    }
    local_sum += local_c;
    const float sum_f = block_reduce_sum(local_sum);

    if (threadIdx.x != 0) return;

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = static_cast<double>(sum_f) * inv_p;


    {
        float c = close[warm];
        float scale = safe_scale_100_over_close(c);
        out[base + warm] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = (trv - atr) * inv_p + atr;
        float c = close[t];
        float scale = safe_scale_100_over_close(c);
        out[base + t] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }
}


extern "C" __global__ void natr_batch_f32_with_inv(
    const float* __restrict__ tr,
    const float* __restrict__ inv_close100,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    if (first_valid >= series_len || warm >= series_len) {
        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[base + idx] = dev_nan();
        }
        return;
    }

    for (int idx = threadIdx.x; idx < warm; idx += blockDim.x) {
        out[base + idx] = dev_nan();
    }
    __syncthreads();

    const int start = first_valid;
    float local_sum = 0.0f;
    float local_c   = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const float v = tr[start + k];
        float y = v - local_c;
        float t = local_sum + y;
        local_c = (t - local_sum) - y;
        local_sum = t;
    }
    local_sum += local_c;
    const float sum_f = block_reduce_sum(local_sum);

    if (threadIdx.x != 0) return;

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = static_cast<double>(sum_f) * inv_p;

    {
        float scale = inv_close100[warm];
        out[base + warm] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = (trv - atr) * inv_p + atr;
        float scale = inv_close100[t];
        out[base + t] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }
}


extern "C" __global__ void natr_batch_warp_io_f32(
    const float* __restrict__ tr,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    if (blockDim.x != 32) return;
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    const int lane = threadIdx.x & (warpSize - 1);
    const unsigned mask = 0xFFFFFFFFu;

    if (first_valid >= series_len || warm >= series_len) {
        for (int idx = lane; idx < series_len; idx += warpSize) {
            out[base + idx] = dev_nan();
        }
        return;
    }

    for (int idx = lane; idx < warm; idx += warpSize) {
        out[base + idx] = dev_nan();
    }


    const int start = first_valid;
    float local_sum = 0.0f;
    float local_c   = 0.0f;
    for (int k = lane; k < period; k += warpSize) {
        const float v = tr[start + k];
        float y = v - local_c;
        float t = local_sum + y;
        local_c = (t - local_sum) - y;
        local_sum = t;
    }
    local_sum += local_c;
    const float sum_f = warp_reduce_sum(local_sum);

    __shared__ float sh_tr[32];
    __shared__ float sh_scale[32];
    __shared__ double sh_atr[32];

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = 0.0;
    if (lane == 0) {
        atr = static_cast<double>(sum_f) * inv_p;
    }

    for (int tile = warm; tile < series_len; tile += warpSize) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        if (valid) {
            sh_tr[lane] = tr[t];
            sh_scale[lane] = safe_scale_100_over_close(close[t]);
        }
        __syncwarp(mask);

        if (lane == 0) {
            #pragma unroll
            for (int o = 0; o < 32; ++o) {
                const int tt = tile + o;
                if (tt >= series_len) break;
                if (tt != warm) {
                    const double trv = static_cast<double>(sh_tr[o]);
                    atr = (trv - atr) * inv_p + atr;
                }
                sh_atr[o] = atr;
            }
        }
        __syncwarp(mask);

        if (valid) {
            const double a = sh_atr[lane];
            const float scale = sh_scale[lane];
            out[base + t] = (scale == scale) ? static_cast<float>(a * static_cast<double>(scale)) : dev_nan();
        }
        __syncwarp(mask);
    }
}

extern "C" __global__ void natr_batch_warp_io_f32_with_inv(
    const float* __restrict__ tr,
    const float* __restrict__ inv_close100,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    if (blockDim.x != 32) return;
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    const int lane = threadIdx.x & (warpSize - 1);
    const unsigned mask = 0xFFFFFFFFu;

    if (first_valid >= series_len || warm >= series_len) {
        for (int idx = lane; idx < series_len; idx += warpSize) {
            out[base + idx] = dev_nan();
        }
        return;
    }

    for (int idx = lane; idx < warm; idx += warpSize) {
        out[base + idx] = dev_nan();
    }

    const int start = first_valid;
    float local_sum = 0.0f;
    float local_c   = 0.0f;
    for (int k = lane; k < period; k += warpSize) {
        const float v = tr[start + k];
        float y = v - local_c;
        float t = local_sum + y;
        local_c = (t - local_sum) - y;
        local_sum = t;
    }
    local_sum += local_c;
    const float sum_f = warp_reduce_sum(local_sum);

    __shared__ float sh_tr[32];
    __shared__ float sh_scale[32];
    __shared__ double sh_atr[32];

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = 0.0;
    if (lane == 0) {
        atr = static_cast<double>(sum_f) * inv_p;
    }

    for (int tile = warm; tile < series_len; tile += warpSize) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        if (valid) {
            sh_tr[lane] = tr[t];
            sh_scale[lane] = inv_close100[t];
        }
        __syncwarp(mask);

        if (lane == 0) {
            #pragma unroll
            for (int o = 0; o < 32; ++o) {
                const int tt = tile + o;
                if (tt >= series_len) break;
                if (tt != warm) {
                    const double trv = static_cast<double>(sh_tr[o]);
                    atr = (trv - atr) * inv_p + atr;
                }
                sh_atr[o] = atr;
            }
        }
        __syncwarp(mask);

        if (valid) {
            const double a = sh_atr[lane];
            const float scale = sh_scale[lane];
            out[base + t] = (scale == scale) ? static_cast<float>(a * static_cast<double>(scale)) : dev_nan();
        }
        __syncwarp(mask);
    }
}


extern "C" __global__ void natr_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int period,
    int num_series,
    int series_len,
    const int*   __restrict__ first_valids,
    float*       __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int fv = first_valids[s];

        if (fv < 0 || fv >= series_len) {
            for (int t = lane; t < series_len; t += warpSize) {
                out_tm[t * stride + s] = dev_nan();
            }
            continue;
        }

        const int warm_end = fv + period;
        if (warm_end > series_len) {
            for (int t = lane; t < series_len; t += warpSize) {
                out_tm[t * stride + s] = dev_nan();
            }
            continue;
        }

        const int warm = warm_end - 1;
        for (int t = lane; t < warm; t += warpSize) {
            out_tm[t * stride + s] = dev_nan();
        }


        float local = 0.0f, csum = 0.0f;
        #pragma unroll 1
        for (int k = lane; k < period; k += warpSize) {
            const int t = fv + k;
            const float h = high_tm[t * stride + s];
            const float l = low_tm[t * stride + s];
            float trv;
            if (t == fv) {
                trv = h - l;
            } else {
                const float pc = close_tm[(t - 1) * stride + s];
                const float hl = h - l;
                const float hc = fabsf(h - pc);
                const float lc = fabsf(l - pc);
                trv = fmaxf(hl, fmaxf(hc, lc));
            }
            float y = trv - csum;
            float tmp = local + y;
            csum = (tmp - local) - y;
            local = tmp;
        }
        local += csum;
        float sum = warp_reduce_sum(local);

        if (lane == 0) {
            const double inv_p = 1.0 / static_cast<double>(period);
            double atr = static_cast<double>(sum) * inv_p;

            {
                float c = close_tm[warm * stride + s];
                float scale = safe_scale_100_over_close(c);
                out_tm[warm * stride + s] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
            }

            for (int t = warm + 1; t < series_len; ++t) {
                const float h = high_tm[t * stride + s];
                const float l = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                const float hl = h - l;
                const float hc = fabsf(h - pc);
                const float lc = fabsf(l - pc);
                const double trv = static_cast<double>(fmaxf(hl, fmaxf(hc, lc)));

                atr = (trv - atr) * inv_p + atr;

                float c = close_tm[t * stride + s];
                float scale = safe_scale_100_over_close(c);
                out_tm[t * stride + s] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
            }
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — natr
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/natr.rs:288 `natr_scalar`.
 *
 * THE BUG THE BRIEF NAMES, fixed here. The f32 kernels above carry
 *     atr = (atr * pm1 + tr) * inv_p      // three roundings
 * while the CPU carries
 *     atr = (tr - atr).mul_add(inv_p, atr)   // ONE rounding
 * Those are different numbers, and natr feeds a threshold comparison.
 *
 * NaN: `tr1.max(tr2).max(tr3)` is `f64::max`, which RETURNS THE NON-NaN
 * OPERAND. An `a > b ? a : b` chain does the opposite — a comparison against
 * NaN is false, so the NaN survives and poisons every later bar of the
 * recurrence. `fmax` has f64::max's semantics; the chain does not.
 *
 * first_valid rule: natr.rs:226-235 takes fh.max(fl).max(fc) — the MAX of
 * three INDEPENDENT first-non-NaN scans, NOT the first index at which all
 * three are simultaneously non-NaN. Registered as HlcMaxOfIndependentFirsts.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void natr_neo_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        const double* __restrict__ close,
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

    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len ||
        (len - first_valid) < period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warm_end = first_valid + period - 1;
    for (int i = 0; i < warm_end && i < len; ++i) o[i] = NEO_F64_NAN;

    const double inv_p = 1.0 / (double)period;
    const double k100  = 100.0;

    // natr.rs:307 — the FIRST bar's TR is the bare range, with no previous
    // close. Seeding it as a three-way max against close[first-1] would read a
    // bar the CPU never looks at.
    double sum_tr = high[first_valid] - low[first_valid];
    for (int i = first_valid + 1; i <= warm_end; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double pc = close[i - 1];
        const double tr = fmax(fmax(hi - lo, fabs(hi - pc)), fabs(lo - pc));
        sum_tr += tr;
    }

    double atr = sum_tr * inv_p;
    const double c_we = close[warm_end];
    o[warm_end] = (isfinite(c_we) && c_we != 0.0) ? (atr / c_we) * k100 : NEO_F64_NAN;

    for (int idx = warm_end + 1; idx < len; ++idx) {
        const double hi = high[idx];
        const double lo = low[idx];
        const double pc = close[idx - 1];
        const double tr = fmax(fmax(hi - lo, fabs(hi - pc)), fabs(lo - pc));
        atr = fma(tr - atr, inv_p, atr);            // ONE rounding, as the CPU
        const double cv = close[idx];
        o[idx] = (isfinite(cv) && cv != 0.0) ? (atr / cv) * k100 : NEO_F64_NAN;
    }
}
