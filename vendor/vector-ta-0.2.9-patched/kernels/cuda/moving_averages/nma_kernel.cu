#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <math.h>
namespace cg = cooperative_groups;


#ifndef NMA_NAN
#define NMA_NAN (__int_as_float(0x7fffffff))
#endif


#ifndef NMA_MAX_PERIOD
#define NMA_MAX_PERIOD 4096
#endif


#ifndef NMA_MAX_TILE
#define NMA_MAX_TILE 512
#endif


extern "C" __constant__ float c_sqrt_diffs[NMA_MAX_PERIOD];


extern "C" __global__ void nma_fill_nan_f32(float* __restrict__ out, int total_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) out[idx] = NMA_NAN;
}

extern "C" __global__ void nma_abs_log_diffs_f32(const float* __restrict__ prices,
                                                 int series_len,
                                                 int first_valid,
                                                 float* __restrict__ abs_diffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= series_len) return;

    if (idx <= first_valid) {
        abs_diffs[idx] = 0.0f;
        return;
    }

    const float p0 = prices[idx - 1];
    const float p1 = prices[idx];
    const float ln0 = logf(fmaxf(p0, 1e-10f));
    const float ln1 = logf(fmaxf(p1, 1e-10f));
    abs_diffs[idx] = fabsf(ln1 - ln0);
}

extern "C" __global__ __launch_bounds__(256, 2)
void nma_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ abs_diffs,
                   const int*   __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int base = combo * series_len;


    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        for (int t = idx; t < series_len; t += stride) {
            out[base + t] = NMA_NAN;
        }
    }

    const int period = periods[combo];
    if (period <= 0 || period >= series_len) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int tail_len = series_len - first_valid;
    if (tail_len <= period) return;

    const int warm = first_valid + period;


    __shared__ float tile[NMA_MAX_TILE + NMA_MAX_PERIOD];

    __shared__ float wbuf[NMA_MAX_PERIOD];


    const bool use_const = (c_sqrt_diffs[1] > 0.0f);
    if (!use_const) {
        for (int i = threadIdx.x; i < period; i += blockDim.x) {
            const float s0 = sqrtf((float)i);
            const float s1 = sqrtf((float)(i + 1));
            wbuf[i] = s1 - s0;
        }
        __syncthreads();
    }

    const int TILE = blockDim.x;
    auto block = cg::this_thread_block();

    for (int tileStart = warm + blockIdx.x * TILE; tileStart < series_len; tileStart += TILE * gridDim.x) {
        const int L = min(TILE, series_len - tileStart);
        const int g_start = tileStart - (period - 1);
        const int load_elems = L + (period - 1);


        cg::memcpy_async(block, tile, abs_diffs + g_start, sizeof(float) * load_elems);
        cg::wait(block);
        block.sync();

        const int lane = threadIdx.x;
        if (lane < L) {
            const int t = tileStart + lane;
            const int cur_idx = (period - 1) + lane;

            float num = 0.0f;
            float denom = 0.0f;

            #pragma unroll 4
            for (int k = 0; k < period; ++k) {
                const float oi = tile[cur_idx - k];
                const float w  = use_const ? c_sqrt_diffs[k] : wbuf[k];
                num   = fmaf(oi, w, num);
                denom += oi;
            }

            const float ratio  = (denom > 0.0f) ? (num / denom) : 0.0f;
            const int   anchor = t - period + 1;
            const float latest = prices[anchor];
            const float prev   = prices[anchor - 1];

            out[base + t] = latest * ratio + prev * (1.0f - ratio);
        }
        block.sync();
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void nma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const float* __restrict__ abs_diffs_tm,
                                   const int*   __restrict__ first_valids,
                                   int num_series,
                                   int series_len,
                                   int period,
                                   float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;


    const int stride = num_series;
    for (int row = 0; row < series_len; ++row) {
        out_tm[row * stride + series] = NMA_NAN;
    }

    if (period <= 0 || period >= series_len) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int tail_len = series_len - first_valid;
    if (tail_len <= period) return;

    const int warm = first_valid + period;


    __shared__ float wbuf[NMA_MAX_PERIOD];
    const bool use_const = (c_sqrt_diffs[1] > 0.0f);
    if (!use_const) {
        for (int i = threadIdx.x; i < period; i += blockDim.x) {
            const float s0 = sqrtf((float)i);
            const float s1 = sqrtf((float)(i + 1));
            wbuf[i] = s1 - s0;
        }
        __syncthreads();
    }

    for (int row = warm; row < series_len; ++row) {
        float num = 0.0f;
        float denom = 0.0f;
        int cur = row;

        #pragma unroll 4
        for (int k = 0; k < period; ++k, --cur) {
            const float oi = abs_diffs_tm[cur * stride + series];
            const float w  = use_const ? c_sqrt_diffs[k] : wbuf[k];
            num   = fmaf(oi, w, num);
            denom += oi;
        }

        const float ratio  = (denom > 0.0f) ? (num / denom) : 0.0f;
        const int   anchor = row - period + 1;
        const float latest = prices_tm[anchor * stride + series];
        const float prev   = prices_tm[(anchor - 1) * stride + series];
        out_tm[row * stride + series] = latest * ratio + prev * (1.0f - ratio);
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `moving_averages/nma.rs::nma_scalar` (l.307).
//     ln_values[i] = data[i].max(1e-10).ln()
//     sqrt_diffs[i]= sqrt(i+1) - sqrt(i)                for i in 0..period
//     for j in (first + period)..len:
//         for i in 0..period:
//             oi     = |ln[j-i] - ln[j-i-1]|
//             num   += oi * sqrt_diffs[i]
//             denom += oi
//         ratio = if denom == 0.0 { 0.0 } else { num / denom }
//         out[j]= data[j-(period-1)]*ratio + data[j-period]*(1.0 - ratio)
//
// THE 1e-10 IS NOT AN EPSILON. It is a DOMAIN CLAMP that keeps log off the
// non-positive axis, and it is already an f64-sized quantity in the reference,
// so it is carried across UNCHANGED. Re-deriving it for f64 would move the
// clamp point and change every bar of a series that touches it. (Contrast the
// f32 machine epsilons elsewhere in this shard, which must be re-derived.)
//
// data[i].max(1e-10) is f64::max, which returns the NON-NaN operand: a NaN
// input therefore yields 1e-10, not NaN. fmax has exactly that semantics; an
// if (x > 1e-10) chain does not, and would let the NaN through into log and
// poison every window that touches the bar. This kernel uses fmax.
//
// The accumulators num and denom are a single flat ascending sum on the CPU --
// no grouping, no fma -- so this kernel does the same: num += oi * sd is TWO
// roundings, and writing it as fma(oi, sd, num) would be one.
//
// f32 -> f64 audit: pointers/locals widened; fabsf->fabs, logf->log,
// sqrtf->sqrt (x4), fmaxf->fmax (x2); __int_as_float NaN -> f64 quiet-NaN
// pattern; no fast-math intrinsic survives.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double nma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void nma_batch_f64(const double* __restrict__ prices,
                   int n,
                   const int*   __restrict__ periods,
                   int n_combos,
                   int first_valid,
                   double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = nma_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const long long start_ll =
        static_cast<long long>(first_valid) + static_cast<long long>(period);
    if (period <= 0 || start_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    const int start = static_cast<int>(start_ll);

    for (int t = 0; t < start; ++t) row[t] = nan_d;

    for (int j = start; j < n; ++j) {
        double num = 0.0;
        double denom = 0.0;

        for (int i = 0; i < period; ++i) {
            const double a = log(fmax(prices[j - i], 1e-10));
            const double b = log(fmax(prices[j - i - 1], 1e-10));
            const double oi = fabs(a - b);
            const double sd = sqrt(static_cast<double>(i) + 1.0) - sqrt(static_cast<double>(i));
            num += oi * sd;
            denom += oi;
        }

        const double ratio = (denom == 0.0) ? 0.0 : (num / denom);
        const int i_off = period - 1;
        row[j] = prices[j - i_off] * ratio + prices[j - i_off - 1] * (1.0 - ratio);
    }
}
