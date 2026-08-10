#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }


extern "C" __global__
void cvi_batch_f32(const float* __restrict__ high,
                   const float* __restrict__ low,
                   const int*   __restrict__ periods,
                   const float* __restrict__ alphas,
                   const int*   __restrict__ warm_indices,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int total_threads = blockDim.x * gridDim.x;
    int combo = blockIdx.x * blockDim.x + threadIdx.x;

    for (; combo < n_combos; combo += total_threads) {
        const int   period = periods[combo];
        const float alpha  = alphas[combo];
        const int   warm   = warm_indices[combo];
        if (period <= 0 || warm >= series_len) continue;

        const int base = combo * series_len;


        float y = high[first_valid] - low[first_valid];
        out[base + first_valid] = y;


        for (int t = first_valid + 1; t < series_len; ++t) {
            const float r = high[t] - low[t];
            y = __fmaf_rn((r - y), alpha, y);
            out[base + t] = y;
        }


        for (int t = series_len - 1; t >= warm; --t) {
            const float curr = out[base + t];
            const float old  = out[base + (t - period)];
            out[base + t] = 100.0f * (curr - old) / old;
        }


        for (int t = 0; t < warm; ++t) {
            out[base + t] = NAN;
        }
    }
}


extern "C" __global__
void cvi_batch_from_range_f32(const float* __restrict__ range,
                              const int*   __restrict__ periods,
                              const float* __restrict__ alphas,
                              const int*   __restrict__ warm_indices,
                              int series_len,
                              int first_valid,
                              int n_combos,
                              float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int total_threads = blockDim.x * gridDim.x;
    int combo = blockIdx.x * blockDim.x + threadIdx.x;

    for (; combo < n_combos; combo += total_threads) {
        const int   period = periods[combo];
        const float alpha  = alphas[combo];
        const int   warm   = warm_indices[combo];
        if (period <= 0 || warm >= series_len) continue;

        const int base = combo * series_len;


        float y = range[first_valid];
        out[base + first_valid] = y;
        for (int t = first_valid + 1; t < series_len; ++t) {
            const float r = range[t];
            y = __fmaf_rn((r - y), alpha, y);
            out[base + t] = y;
        }


        for (int t = series_len - 1; t >= warm; --t) {
            const float curr = out[base + t];
            const float old  = out[base + (t - period)];
            out[base + t] = 100.0f * (curr - old) / old;
        }


        for (int t = 0; t < warm; ++t) {
            out[base + t] = NAN;
        }
    }
}


extern "C" __global__
void cvi_many_series_one_param_f32(const float* __restrict__ high_tm,
                                   const float* __restrict__ low_tm,
                                   const int*   __restrict__ first_valids,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_series;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s];
        if (fv < 0 || fv >= series_len) {
            continue;
        }

        const int warm = fv + (2 * period - 1);
        if (warm >= series_len) {
            continue;
        }


        float y = high_tm[fv * stride + s] - low_tm[fv * stride + s];
        out_tm[fv * stride + s] = y;

        for (int t = fv + 1; t < series_len; ++t) {
            const float r = high_tm[t * stride + s] - low_tm[t * stride + s];
            y = __fmaf_rn((r - y), alpha, y);
            out_tm[t * stride + s] = y;
        }


        for (int t = series_len - 1; t >= warm; --t) {
            const float curr = out_tm[t * stride + s];
            const float old  = out_tm[(t - period) * stride + s];
            out_tm[t * stride + s] = 100.0f * (curr - old) / old;
        }


        for (int t = 0; t < warm; ++t) {
            out_tm[t * stride + s] = NAN;
        }
    }
}


extern "C" __global__
void range_from_high_low_f32(const float* __restrict__ high,
                             const float* __restrict__ low,
                             int series_len,
                             float* __restrict__ range)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < series_len;
         t += blockDim.x * gridDim.x)
    {
        range[t] = high[t] - low[t];
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `cvi.rs::cvi_scalar` (l.354). Inputs are (high, low).
//   alpha  = 2.0 / (period + 1)
//   val    = high[fv] - low[fv]              (the ring's element 0)
//   warmup = fv+1 .. fv + (2*period - 1), each bar doing
//            `val += (range - val) * alpha`  — TWO roundings (the multiply and
//            the compound add). The CPU does NOT use mul_add here, so this
//            kernel must NOT use fma: an fma would be ONE rounding and a
//            different number.
//   emit   = `100.0 * (val - old) / old`, where `old` is the ring entry
//            `period` bars back — evaluated left to right, so the multiply by
//            100 happens BEFORE the divide.
//
// The `lag` ring is `period` entries. It lives in a fixed per-thread local
// array, which makes the bound a property of the COMPILED kernel, so the host
// refuses an oversized period by name (`F64Kernel::max_period`) rather than
// truncating the window or moving the sweep to the host.
//
// f32 -> f64 audit: the f32 file's `__fmaf_rn` x3 are gone — they were a fused
// form the CPU does not use; every literal widened; NaN constant is the f64
// quiet-NaN bit pattern. No epsilon here; `old` is a raw divisor exactly as on
// the CPU, and a guard would make the device disagree. No min/max chain.
// ---------------------------------------------------------------------------

#ifndef CVI_MAX_PERIOD_F64
#define CVI_MAX_PERIOD_F64 512
#endif

static __device__ __forceinline__ double cvi_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void cvi_batch_f64(const double* __restrict__ high,
                   const double* __restrict__ low,
                   int n,
                   const int*   __restrict__ periods,
                   int n_combos,
                   int first_valid,
                   double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = cvi_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    if (period <= 0 || period > CVI_MAX_PERIOD_F64 || first_valid >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const long long needed = 2LL * static_cast<long long>(period) - 1;
    const long long end_warm_ll = static_cast<long long>(first_valid) + needed;
    if (end_warm_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    const int end_warm = static_cast<int>(end_warm_ll);

    for (int t = 0; t < end_warm; ++t) row[t] = nan_d;

    const double alpha = 2.0 / (static_cast<double>(period) + 1.0);

    double lag[CVI_MAX_PERIOD_F64];
    double val = high[first_valid] - low[first_valid];
    lag[0] = val;

    int head = 1;
    for (int i = first_valid + 1; i < end_warm; ++i) {
        const double range = high[i] - low[i];
        val += (range - val) * alpha;
        lag[head] = val;
        ++head;
        if (head == period) head = 0;
    }

    for (int j = end_warm; j < n; ++j) {
        const double range = high[j] - low[j];
        val += (range - val) * alpha;
        const double old = lag[head];
        row[j] = 100.0 * (val - old) / old;
        lag[head] = val;
        ++head;
        if (head == period) head = 0;
    }
}
