#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


struct ds {
    float hi, lo;
    __device__ __forceinline__ ds() {}
    __device__ __forceinline__ ds(float h, float l=0.f): hi(h), lo(l) {}
};

__device__ __forceinline__ ds ds_from_f(float x) { return ds(x, 0.f); }
__device__ __forceinline__ float ds_to_f(ds a)  { return a.hi + a.lo; }
__device__ __forceinline__ ds ds_neg(ds a)      { return ds(-a.hi, -a.lo); }


__device__ __forceinline__ ds ds_from_d(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return ds(hi, lo);
}


__device__ __forceinline__ ds ds_add(ds a, ds b) {
    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float e  = (a.hi - (s - bb)) + (b.hi - bb);
    e += a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return ds(hi, lo);
}
__device__ __forceinline__ ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }


__device__ __forceinline__ ds ds_mul(ds a, ds b) {
    float p   = a.hi * b.hi;
    float err = fmaf(a.hi, b.hi, -p);
    err += a.hi * b.lo + a.lo * b.hi;
    float hi  = p + err;
    float lo  = err - (hi - p);
    return ds(hi, lo);
}

__device__ __forceinline__ ds ds_scale(ds a, float s) {
    float p   = a.hi * s;
    float err = fmaf(a.hi, s, -p) + a.lo * s;
    float hi  = p + err;
    float lo  = err - (hi - p);
    return ds(hi, lo);
}

__device__ __forceinline__ ds ds_square(ds a) { return ds_mul(a, a); }


#ifndef STDDEV_COMBO_TILE
#define STDDEV_COMBO_TILE 4
#endif

extern "C" __global__ void stddev_build_prefix_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    float2* __restrict__ ps_x,
    float2* __restrict__ ps_x2,
    int* __restrict__ ps_nan
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    ds sum = ds(0.0f, 0.0f);
    ds sum_sq = ds(0.0f, 0.0f);
    int nan_count = 0;

    ps_x[0] = make_float2(0.0f, 0.0f);
    ps_x2[0] = make_float2(0.0f, 0.0f);
    ps_nan[0] = 0;

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float v = data[i];
            if (isnan(v)) {
                nan_count += 1;
            } else {
                const ds x = ds(v, 0.0f);
                sum = ds_add(sum, x);
                sum_sq = ds_add(sum_sq, ds_square(x));
            }
        }
        ps_x[i + 1] = make_float2(sum.hi, sum.lo);
        ps_x2[i + 1] = make_float2(sum_sq.hi, sum_sq.lo);
        ps_nan[i + 1] = nan_count;
    }
}

extern "C" __global__ void stddev_batch_f32(
    const float2* __restrict__ ps_x,
    const float2* __restrict__ ps_x2,
    const int*    __restrict__ ps_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ nbdevs,
    int n_combos,
    float* __restrict__ out
) {
    const int group = blockIdx.y;
    const int co_base = group * STDDEV_COMBO_TILE;

    __shared__ int s_period[STDDEV_COMBO_TILE];
    __shared__ int s_warm[STDDEV_COMBO_TILE];
    __shared__ float s_nb[STDDEV_COMBO_TILE];
    __shared__ double s_inv_n[STDDEV_COMBO_TILE];

    if (threadIdx.x < STDDEV_COMBO_TILE) {
        const int c = co_base + threadIdx.x;
        if (c < n_combos) {
            const int p = periods[c];
            s_period[threadIdx.x] = p;
            s_warm[threadIdx.x] = first_valid + p - 1;
            s_nb[threadIdx.x] = nbdevs[c];
            s_inv_n[threadIdx.x] = (p > 0) ? (1.0 / (double)p) : 0.0;
        } else {
            s_period[threadIdx.x] = 0;
            s_warm[threadIdx.x] = INT_MAX;
            s_nb[threadIdx.x] = 0.0f;
            s_inv_n[threadIdx.x] = 0.0;
        }
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nan_f = __int_as_float(0x7fffffff);

    while (t < len) {
        const int end = t + 1;
        const float2 ex = ps_x[end];
        const float2 ex2 = ps_x2[end];
        const int end_bad = ps_nan[end];

#pragma unroll
        for (int k = 0; k < STDDEV_COMBO_TILE; ++k) {
            const int combo = co_base + k;
            if (combo >= n_combos) break;

            const int period = s_period[k];
            float outv = nan_f;
            if (period > 0) {
                const int warm = s_warm[k];
                const float nb = s_nb[k];
                const double inv_n = s_inv_n[k];

                if (nb == 0.0f) {
                    outv = (t >= warm) ? 0.0f : nan_f;
                } else if (t >= warm) {
                    int start = end - period;
                    if (start < 0) start = 0;
                    const int nan_count = end_bad - ps_nan[start];
                    if (nan_count == 0) {
                        float2 sx = ps_x[start];
                        float2 sx2 = ps_x2[start];
                        const double s1 =
                            ((double)ex.x + (double)ex.y) - ((double)sx.x + (double)sx.y);
                        const double s2 =
                            ((double)ex2.x + (double)ex2.y) - ((double)sx2.x + (double)sx2.y);
                        const double mean = s1 * inv_n;
                        const double var = (s2 * inv_n) - (mean * mean);
                        outv = (var > 0.0) ? sqrtf((float)var) * nb : 0.0f;
                    }
                }
            }
            out[combo * len + t] = outv;
        }

        t += stride;
    }
}


extern "C" __global__ void stddev_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    const int*  __restrict__ first_valids,
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;


    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm     = first_valid + period - 1;
    const double inv_n = 1.0 / (double)period;


    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else { double d = (double)v; s1 += d; s2 += d * d; }
    }

    if (warm < rows) {
        if (nan_in_win == 0) {
            double mean = s1 * inv_n;
            double var  = (s2 * inv_n) - (mean * mean);
            out_tm[warm * stride + series] = (var > 0.0 && nbdev != 0.0f)
                ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        } else {
            out_tm[warm * stride + series] = __int_as_float(0x7fffffff);
        }
    }


    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];


        if (!isnan(old_v)) { double od = (double)old_v; s1 -= od; s2 -= od * od; }
        else { nan_in_win--; }
        if (!isnan(new_v)) { double nd = (double)new_v; s1 += nd; s2 += nd * nd; }
        else { nan_in_win++; }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            double mean = s1 * inv_n;
            double var  = (s2 * inv_n) - (mean * mean);
            out_tm[t * stride + series] = (var > 0.0 && nbdev != 0.0f)
                ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — stddev
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/stddev.rs:359 `stddev_scalar`, which for the
 * DEFAULT nbdev (1.0, stddev.rs:115 / :527) delegates to
 * `stddev_scalar_nbdev1` (stddev.rs:417). Our lane sweeps `period` with
 * default params, so nbdev is 1.0 and the two paths are bit-identical anyway
 * (`x * 1.0 == x`).
 *
 * SUM OF SQUARES, NOT WELFORD. The CPU keeps `sum` and `sum_sqr` and forms
 * `var = sum_sqr*inv - mean*mean`, then clamps `var <= 0 -> 0`. That
 * cancellation is numerically poor and IS THE REFERENCE; replacing it with a
 * stable two-pass or Welford update would produce a different number, which is
 * the failure this lane exists to prevent.
 *
 * The rolling update is `sum_sqr += new*new - old*old` — ONE add of a
 * difference, not two separate updates. Reproduced literally.
 *
 * `__int_as_float(0x7f...)` x4 in the f32 file is an f32 NaN bit pattern.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void stddev_neo_batch_f64(const double* __restrict__ data,
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

    const int idx0 = first_valid + period - 1;      // stddev.rs:253 warmup
    for (int i = 0; i < idx0 && i < len; ++i) o[i] = NEO_F64_NAN;

    const double inv_den = 1.0 / (double)period;

    double sum = 0.0, sum_sqr = 0.0;
    for (int i = first_valid; i < first_valid + period; ++i) {
        const double v = data[i];
        sum     += v;
        sum_sqr += v * v;
    }

    const double mean0 = sum * inv_den;
    const double var0  = (sum_sqr * inv_den) - (mean0 * mean0);
    o[idx0] = (var0 <= 0.0) ? 0.0 : sqrt(var0);

    for (int i = idx0 + 1; i < len; ++i) {
        const double old_v = data[i - period];
        const double new_v = data[i];
        sum     += new_v - old_v;
        sum_sqr += new_v * new_v - old_v * old_v;

        const double mean = sum * inv_den;
        const double var  = (sum_sqr * inv_den) - (mean * mean);
        o[i] = (var <= 0.0) ? 0.0 : sqrt(var);
    }
}
