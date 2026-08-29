#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


static __device__ __forceinline__ float f32_qnan() {
    return __int_as_float(0x7fffffff);
}


extern "C" __global__
void cora_wave_batch_f32(const float* __restrict__ prices,
                         const float* __restrict__ weights_flat,
                         const int*   __restrict__ periods,
                         const float* __restrict__ inv_norms,
                         int max_period,
                         int series_len,
                         int n_combos,
                         int first_valid,
                         float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    extern __shared__ float shared_weights[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        shared_weights[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = f32_qnan();
        } else {
            const int start = t - period + 1;
            float s = 0.f;

            float c = 0.f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                float term = __fmaf_rn(prices[start + k], shared_weights[k], 0.f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
            out[out_idx] = __fmul_rn(s, inv_norms[combo]);
        }
        t += stride;
    }
}


extern "C" __global__
void cora_wave_batch_wma_from_y_f32(const float* __restrict__ y,
                                     const int*   __restrict__ smooth_periods,
                                     const int*   __restrict__ warm0s,
                                     int series_len,
                                     int n_combos,
                                     float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int m = smooth_periods[combo];
    if (m <= 1) {

        const int base = combo * series_len;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            out[base + t] = y[base + t];
            t += stride;
        }
        return;
    }

    const float inv_norm = 2.0f / (float(m) * (float(m) + 1.0f));
    const int warm = warm0s[combo] + (m - 1);
    const int base = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = base + t;
        if (t < warm) {
            out[out_idx] = f32_qnan();
        } else {
            const int start = t - m + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < m; ++k) {
                acc = __fmaf_rn(y[base + start + k], float(k + 1), acc);
            }
            out[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}


extern "C" __global__
void cora_wave_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ weights,
    int period,
    float inv_norm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    if (period <= 0) return;

    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = f32_qnan();
        } else {
            const int start = (t - period + 1) * num_series + s;
            float sacc = 0.f, c = 0.f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                float x = prices_tm[start + k * num_series];
                float term = __fmaf_rn(x, weights[k], 0.f);
                float y = term - c;
                float u = sacc + y;
                c = (u - sacc) - y;
                sacc = u;
            }
            out_tm[out_idx] = sacc * inv_norm;
        }
        t += stride;
    }
}


extern "C" __global__
void cora_wave_ms1p_wma_time_major_f32(const float* __restrict__ y_tm,
                                       int wma_period,
                                       int num_series,
                                       int series_len,
                                       const int* __restrict__ warm0s,
                                       float* __restrict__ out_tm) {
    if (wma_period <= 1) {

        int s = blockIdx.y;
        if (s >= num_series) return;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            out_tm[t * num_series + s] = y_tm[t * num_series + s];
            t += stride;
        }
        return;
    }
    const float inv_norm = 2.0f / (float(wma_period) * (float(wma_period) + 1.0f));
    const int s = blockIdx.y;
    if (s >= num_series) return;
    const int warm = warm0s[s] + (wma_period - 1);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = f32_qnan();
        } else {
            const int start = (t - wma_period + 1) * num_series + s;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < wma_period; ++k) {
                float y = y_tm[start + k * num_series];
                acc = __fmaf_rn(y, float(k + 1), acc);
            }
            out_tm[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: cora_wave_scalar_with_weights
// (src/indicators/cora_wave.rs:422-712), reached through
// cora_wave_with_kernel (:246) -> cora_wave_compute_into (:390).
//
// PERIOD-SWEPT: ma_batch.rs:959-963 assigns sweep.period = period_range and
// pins r_multi at 2.0 and smooth at true, so the swept int is the window and
// the other two are the CPU defaults at every row.
//
// SHAPE: one thread per combo walking bars ASCENDING. The convolution is
// rolled with S = S*inv_R - a_old*x_old + w_last*x_new (:503) rather than
// rebuilt, and the smoothing WMA is a second carried accumulator, so the
// accumulation order is load-bearing throughout.
//
// THE n < 100_000 FORK IS REAL AND IS REPRODUCED. At :551 and :655 the CPU
// takes a DIFFERENT smoothing formulation on long inputs -- it rolls
// wsum = wsum - ssum + m*y_new instead of re-summing the ring each bar -- and
// the two round differently. A kernel implementing only one of them would
// disagree with the CPU on exactly one side of that threshold, so both are
// here, keyed on the same n.
//
// Gate203 caught the first valid row differing by one bit:
// CPU 0x3ff1333a931db83d versus CUDA 0x3ff1333a931db83e. The cause was separate
// host powf and libdevice pow coefficient construction. The exact RedK
// compound-ratio weights are now built once by the CPU authority and retained
// by the resident launch. The only per-thread array here is the smoothing ring,
// whose length is round(sqrt(period)); CORA_WAVE_NEO_MAX_M bounds it and
// F64Kernel::max_period REFUSES a larger period by name.
// ===========================================================================

#define CORA_WAVE_NEO_MAX_M 64
#define CORA_WAVE_NEO_MAX_PERIOD 4160

static __forceinline__ __device__ double cora_wave_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// Gate208 advanced past the coefficient repair and caught the next rolling
// output at CPU 0x3ff1334448be5cc8 versus CUDA 0x3ff1334448be5cc7.
// Make both contractions explicit instead of leaving the recurrence to the
// host and device compilers.
static __forceinline__ __device__ double cora_wave_roll_forward_exact_v1(
    double previous,
    double inv_r,
    double a_old,
    double x_old,
    double w_last,
    double x_new) {
    const double without_old = fma(-a_old, x_old, previous * inv_r);
    return fma(w_last, x_new, without_old);
}

extern "C" __global__
void cora_wave_neo_batch_f64(const double* __restrict__ data,
                             int n,
                             const int* __restrict__ periods,
                             int n_combos,
                             int first_valid,
                             const double* __restrict__ exact_coefficients,
                             const int coefficient_stride,
                             double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = cora_wave_neo_qnan();

    const int p = periods[combo];

    // cora_wave_prepare, :325-328 -- !is_nan, which is what
    // F64FirstValidRule::AllInputsNonNan resolves to for a single close
    // series, so the caller value is used directly.
    int first = first_valid;
    if (first < 0) first = 0;

    bool refused = false;
    if (first >= n) refused = true;
    if (p <= 0 || p > n) refused = true;                     // :339
    if (!refused && (n - first) < p) refused = true;         // :345
    if (p > CORA_WAVE_NEO_MAX_PERIOD) refused = true;
    if (coefficient_stride <= p) refused = true;

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    // Exact oldest-to-newest weights and normalization come from the one CPU
    // authority. The final stride element is reserved for inv_wsum.
    const double* __restrict__ weights =
        exact_coefficients + (size_t)combo * (size_t)coefficient_stride;
    const double inv_wsum = weights[coefficient_stride - 1];
    const double w0 = weights[0];
    const double w1 = p == 1 ? 1.0 : weights[1];
    const double w_last = weights[p - 1];

    // :377-381 -- smooth is true at every row, so m = max(round(sqrt(p)), 1).
    int m = (int)round(sqrt((double)p));
    if (m < 1) m = 1;
    if (m > CORA_WAVE_NEO_MAX_M) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    // :257 -- warm = first + period - 1 + (smooth_period - 1).
    long long warm_ll = (long long)first + (long long)p - 1 + (long long)(m - 1);
    const int nan_end = warm_ll < (long long)n ? (int)warm_ll : n;
    for (int i = 0; i < nan_end; ++i) row[i] = nn;
    if (warm_ll >= (long long)n) return;

    const double wma_sum = (double)m * ((double)m + 1.0) * 0.5;
    const bool short_input = (n < 100000);

    // ---------------------------------------------------------------- m == 1
    if (m == 1) {
        if (p == 1) {
            // :436-448
            for (int i = first; i < n; ++i) row[i] = data[i] * inv_wsum;
            return;
        }
        const double inv_R = w0 / w1;
        const double a_old = w0 * inv_R;
        const int warm0 = first + p - 1;
        if (warm0 >= n) return;
        const int start0 = warm0 + 1 - p;

        // :462-495 -- four independent accumulators, then
        // (acc0+acc1)+(acc2+acc3). The pairing is load-bearing.
        double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
        const int end4 = p & ~3;
        int j = 0;
        while (j < end4) {
            const double x0 = data[start0 + j];
            const double x1 = data[start0 + j + 1];
            const double x2 = data[start0 + j + 2];
            const double x3 = data[start0 + j + 3];
            const double y0 = weights[j];
            const double y1 = weights[j + 1];
            const double y2 = weights[j + 2];
            const double y3 = weights[j + 3];
            acc0 = fma(x0, y0, acc0);
            acc1 = fma(x1, y1, acc1);
            acc2 = fma(x2, y2, acc2);
            acc3 = fma(x3, y3, acc3);
            j += 4;
        }
        double S = (acc0 + acc1) + (acc2 + acc3);
        while (j < p) {
            const double x = data[start0 + j];
            const double y = weights[j];
            S = fma(x, y, S);
            ++j;
        }

        row[warm0] = S * inv_wsum;
        int i = warm0;
        while (i + 1 < n) {
            const double x_old = data[i + 1 - p];
            const double x_new = data[i + 1];
            S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
            row[i + 1] = S * inv_wsum;
            ++i;
        }
        return;
    }

    // ---------------------------------------------------------------- m > 1
    double ring[CORA_WAVE_NEO_MAX_M];

    if (p == 1) {
        // :515-585 -- the ring holds RAW data, not filtered values.
        const int warm0 = first;
        if (warm0 >= n) return;
        const int warm_total = warm0 + m - 1;

        if (short_input) {
            int head = 0;
            for (int i = warm0; i < n; ++i) {
                ring[head] = data[i];
                head = (head + 1) % m;
                if (i >= warm_total) {
                    double acc = 0.0;
                    for (int k = 0; k < m; ++k) {
                        const double v = ring[(head + k) % m];
                        acc += v * (double)(k + 1);
                    }
                    row[i] = acc / wma_sum;
                }
            }
            return;
        }

        int fill = 0;
        int i = warm0;
        while (i <= warm_total && i < n) {
            ring[fill] = data[i];
            ++fill;
            ++i;
        }
        if (warm_total >= n) return;

        double ssum = 0.0, wsum = 0.0;
        for (int k = 0; k < m; ++k) {
            const double v = ring[k];
            ssum += v;
            wsum += v * (double)(k + 1);
        }
        int head = 0;
        int t = warm_total;
        row[t] = wsum / wma_sum;
        while (t + 1 < n) {
            const double y_old = ring[head];
            const double y_new = data[t + 1];
            wsum = wsum - ssum + (double)m * y_new;
            ring[head] = y_new;
            ssum = ssum + y_new - y_old;
            head = (head + 1) % m;
            row[t + 1] = wsum / wma_sum;
            ++t;
        }
        return;
    }

    // p > 1 and m > 1 -- :588-712.
    const double inv_R = w0 / w1;
    const double a_old = w0 * inv_R;
    const int warm0 = first + p - 1;
    if (warm0 >= n) return;
    const int start0 = warm0 + 1 - p;

    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    const int end4 = p & ~3;
    int j = 0;
    while (j < end4) {
        const double x0 = data[start0 + j];
        const double x1 = data[start0 + j + 1];
        const double x2 = data[start0 + j + 2];
        const double x3 = data[start0 + j + 3];
        const double y0 = weights[j];
        const double y1 = weights[j + 1];
        const double y2 = weights[j + 2];
        const double y3 = weights[j + 3];
        acc0 = fma(x0, y0, acc0);
        acc1 = fma(x1, y1, acc1);
        acc2 = fma(x2, y2, acc2);
        acc3 = fma(x3, y3, acc3);
        j += 4;
    }
    double S = (acc0 + acc1) + (acc2 + acc3);
    while (j < p) {
        const double x = data[start0 + j];
        const double y = weights[j];
        S = fma(x, y, S);
        ++j;
    }

    int fill = 0;
    double y = S * inv_wsum;
    ring[fill] = y;
    ++fill;

    const int warm_total = warm0 + m - 1;
    int i = warm0;
    while (i + 1 <= warm_total && i + 1 < n) {
        const double x_old = data[i + 1 - p];
        const double x_new = data[i + 1];
        S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
        y = S * inv_wsum;
        ring[fill] = y;
        ++fill;
        ++i;
    }
    if (warm_total >= n) return;

    if (short_input) {
        int head = 0;
        {
            double acc = 0.0;
            for (int k = 0; k < m; ++k) {
                const double v = ring[(head + k) % m];
                acc += v * (double)(k + 1);
            }
            row[warm_total] = acc / wma_sum;
        }
        while (i + 1 < n) {
            const double x_old = data[i + 1 - p];
            const double x_new = data[i + 1];
            S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
            const double y_new = S * inv_wsum;
            ring[head] = y_new;
            head = (head + 1) % m;
            double acc = 0.0;
            for (int k = 0; k < m; ++k) {
                const double v = ring[(head + k) % m];
                acc += v * (double)(k + 1);
            }
            row[i + 1] = acc / wma_sum;
            ++i;
        }
        return;
    }

    int head = 0;
    double ssum = 0.0, wsum = 0.0;
    for (int k = 0; k < m; ++k) {
        const double v = ring[k];
        ssum += v;
        wsum += v * (double)(k + 1);
    }
    row[warm_total] = wsum / wma_sum;
    while (i + 1 < n) {
        const double x_old = data[i + 1 - p];
        const double x_new = data[i + 1];
        S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
        const double y_new = S * inv_wsum;
        wsum = wsum - ssum + (double)m * y_new;
        const double y_old = ring[head];
        ring[head] = y_new;
        ssum = ssum + y_new - y_old;
        head = (head + 1) % m;
        row[i + 1] = wsum / wma_sum;
        ++i;
    }
}
