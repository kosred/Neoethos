#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


#ifndef ROCP_QNAN

__device__ __forceinline__ float rocp_qnan() {
    return __int_as_float(0x7fc00000);
}
#define ROCP_QNAN rocp_qnan()
#endif


extern "C" __global__
void rocp_build_reciprocals_f32(const float* __restrict__ data,
                                int len,
                                float* __restrict__ inv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    inv[idx] = 1.0f / data[idx];
}

extern "C" __global__
void rocp_batch_f32(const float* __restrict__ data,
                    const float* __restrict__ inv,
                    const int* __restrict__ periods,
                    int len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= n_combos) return;
    const int period = periods[row];
    if (period <= 0) return;

    const int base = row * len;

    const int start = first_valid + period;


    const int warm = (start < len) ? start : len;
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = ROCP_QNAN;
    }

    if (start >= len) return;


    int t = start + threadIdx.x;
    const int stride = blockDim.x;


    for (; t + 3*stride < len; t += 4*stride) {
        const float c0  = data[t];
        const float ip0 = inv[t - period];
        out[base + t] = fmaf(c0, ip0, -1.0f);

        const int t1 = t + stride;
        const float c1  = data[t1];
        const float ip1 = inv[t1 - period];
        out[base + t1] = fmaf(c1, ip1, -1.0f);

        const int t2 = t + 2*stride;
        const float c2  = data[t2];
        const float ip2 = inv[t2 - period];
        out[base + t2] = fmaf(c2, ip2, -1.0f);

        const int t3 = t + 3*stride;
        const float c3  = data[t3];
        const float ip3 = inv[t3 - period];
        out[base + t3] = fmaf(c3, ip3, -1.0f);
    }


    for (; t < len; t += stride) {
        const float c  = data[t];
        const float ip = inv[t - period];
        out[base + t] = fmaf(c, ip, -1.0f);
    }
}


extern "C" __global__
void rocp_batch_tiled_f32(const float* __restrict__ data,
                          const float* __restrict__ inv,
                          const int* __restrict__ periods,
                          int len,
                          int first_valid,
                          int n_combos,
                          float* __restrict__ out) {
    const int row = blockIdx.y;
    if (row >= n_combos) return;
    const int period = periods[row];
    if (period <= 0) return;

    const int base = row * len;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int start = first_valid + period;

    const int warm = (start < len) ? start : len;
    for (int t = offset; t < warm; t += stride) {
        out[base + t] = ROCP_QNAN;
    }

    if (start >= len) return;

    int t = start + offset;

    for (; t + 3*stride < len; t += 4*stride) {
        const float c0  = data[t];
        const float ip0 = inv[t - period];
        out[base + t] = fmaf(c0, ip0, -1.0f);

        const int t1 = t + stride;
        const float c1  = data[t1];
        const float ip1 = inv[t1 - period];
        out[base + t1] = fmaf(c1, ip1, -1.0f);

        const int t2 = t + 2*stride;
        const float c2  = data[t2];
        const float ip2 = inv[t2 - period];
        out[base + t2] = fmaf(c2, ip2, -1.0f);

        const int t3 = t + 3*stride;
        const float c3  = data[t3];
        const float ip3 = inv[t3 - period];
        out[base + t3] = fmaf(c3, ip3, -1.0f);
    }

    for (; t < len; t += stride) {
        const float c  = data[t];
        const float ip = inv[t - period];
        out[base + t] = fmaf(c, ip, -1.0f);
    }
}


extern "C" __global__
void rocp_many_series_one_param_f32(const float* __restrict__ data_tm,
                                    const int* __restrict__ firsts,
                                    int cols,
                                    int rows,
                                    int period,
                                    float* __restrict__ out) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || period <= 0) return;

    const int first = firsts[s];
    if (first >= rows) {

        for (int t = 0; t < rows; ++t) {
            out[t * cols + s] = ROCP_QNAN;
        }
        return;
    }

    const int warm = first + period;

    const int limit = (warm < rows) ? warm : rows;
    for (int t = 0; t < limit; ++t) {
        out[t * cols + s] = ROCP_QNAN;
    }
    if (warm >= rows) return;


    for (int t = warm; t < rows; ++t) {
        const float c = data_tm[t * cols + s];
        const float p = data_tm[(t - period) * cols + s];
        out[t * cols + s] = (c - p) / p;
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `rocp.rs::rocp_scalar` (l.320) —
//   `for i in (first_val + period)..len { out[i] = (data[i] - data[i-period]) / data[i-period] }`
//
// f32 -> f64 audit: pointers/locals widened; the f32 file's
// `__int_as_float(0x7fc00000)` NaN becomes the f64 quiet-NaN bit pattern; no
// fast-math divide (`__fdividef`) survives — this is a plain IEEE divide, one
// rounding, matching the single rounding in the CPU line. No epsilon: a zero
// `prev` divides to +/-Inf on both sides, which is what the CPU produces, and
// substituting a guard here would make the device disagree with the reference.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double rocp_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void rocp_batch_f64(const double* __restrict__ prices,
                    int n,
                    const int*   __restrict__ periods,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = rocp_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    if (period <= 0) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const long long start_ll = static_cast<long long>(first_valid) + static_cast<long long>(period);
    const int start = (start_ll >= n) ? n : static_cast<int>(start_ll);

    for (int t = 0; t < start; ++t) row[t] = nan_d;
    for (int t = start; t < n; ++t) {
        const double p = prices[t - period];
        row[t] = (prices[t] - p) / p;
    }
}
