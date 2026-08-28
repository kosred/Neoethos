#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


#ifndef LDG
#  if __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif


__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }

extern "C" __global__
void vwma_prefix_pv_vol_f64_f32(const float* __restrict__ prices,
                                const float* __restrict__ volumes,
                                int series_len,
                                int first_valid,
                                double* __restrict__ pv_prefix,
                                double* __restrict__ vol_prefix) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;

    if (first_valid < 0) first_valid = 0;
    if (first_valid > series_len) first_valid = series_len;

    double acc_pv = 0.0;
    double acc_vol = 0.0;

    for (int i = 0; i < first_valid; ++i) {
        pv_prefix[i] = 0.0;
        vol_prefix[i] = 0.0;
    }

    for (int i = first_valid; i < series_len; ++i) {
        const float p = prices[i];
        const float v = volumes[i];
        if (isnan(p) || isnan(v) || isnan(acc_pv) || isnan(acc_vol)) {
            acc_pv = NAN;
            acc_vol = NAN;
        } else {
            acc_pv += (double)p * (double)v;
            acc_vol += (double)v;
        }
        pv_prefix[i] = acc_pv;
        vol_prefix[i] = acc_vol;
    }
}

extern "C" __global__
void vwma_prefix_pv_vol_time_major_f64_f32(const float* __restrict__ prices_tm,
                                           const float* __restrict__ volumes_tm,
                                           const int* __restrict__ first_valids,
                                           int num_series,
                                           int series_len,
                                           double* __restrict__ pv_prefix_tm,
                                           double* __restrict__ vol_prefix_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series || series_len <= 0) return;

    int first_valid = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid > series_len) first_valid = series_len;

    double acc_pv = 0.0;
    double acc_vol = 0.0;

    for (int row = 0; row < series_len; ++row) {
        const int idx = row * num_series + series_idx;
        if (row >= first_valid) {
            const float p = prices_tm[idx];
            const float v = volumes_tm[idx];
            if (isnan(p) || isnan(v) || isnan(acc_pv) || isnan(acc_vol)) {
                acc_pv = NAN;
                acc_vol = NAN;
            } else {
                acc_pv += (double)p * (double)v;
                acc_vol += (double)v;
            }
        }
        pv_prefix_tm[idx] = acc_pv;
        vol_prefix_tm[idx] = acc_vol;
    }
}

extern "C" __global__
void vwma_batch_f32(const double* __restrict__ pv_prefix,
                    const double* __restrict__ vol_prefix,
                    const int*    __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int warm   = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float value;
        if (t < warm) {
            value = nan_f32();
        } else {

            const int prev = t - period;


            double sum_pv  = LDG(&pv_prefix[t]);
            double sum_vol = LDG(&vol_prefix[t]);

            if (prev >= 0) {
                sum_pv  -= LDG(&pv_prefix[prev]);
                sum_vol -= LDG(&vol_prefix[prev]);
            }

            value = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                     : nan_f32();
        }
        out[base_out + t] = value;
        t += stride;
    }
}

extern "C" __global__
void vwma_multi_series_one_param_f32(const double* __restrict__ pv_prefix_tm,
                                     const double* __restrict__ vol_prefix_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f32();
        } else {
            const int prev = t - period;
            const int idx  = out_idx;

            double sum_pv  = LDG(&pv_prefix_tm[idx]);
            double sum_vol = LDG(&vol_prefix_tm[idx]);

            if (prev >= 0) {
                const int prev_idx = prev * num_series + series_idx;
                sum_pv  -= LDG(&pv_prefix_tm[prev_idx]);
                sum_vol -= LDG(&vol_prefix_tm[prev_idx]);
            }

            out_tm[out_idx] = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                               : nan_f32();
        }
        t += stride;
    }
}


extern "C" __global__
void vwma_multi_series_one_param_tm_coalesced_f32(const double* __restrict__ pv_prefix_tm,
                                                  const double* __restrict__ vol_prefix_tm,
                                                  int period,
                                                  int num_series,
                                                  int series_len,
                                                  const int* __restrict__ first_valids,
                                                  float* __restrict__ out_tm)
{

    const int series_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;


    const int warm = first_valids[series_idx] + period - 1;


    for (int t = blockIdx.x * blockDim.y + threadIdx.y;
         t < series_len;
         t += gridDim.x * blockDim.y)
    {
        const int out_idx = t * num_series + series_idx;

        if (t < warm) {
            out_tm[out_idx] = nan_f32();
            continue;
        }

        const int prev = t - period;


        double sum_pv  = LDG(&pv_prefix_tm[out_idx]);
        double sum_vol = LDG(&vol_prefix_tm[out_idx]);

        if (prev >= 0) {
            const int prev_idx = prev * num_series + series_idx;
            sum_pv  -= LDG(&pv_prefix_tm[prev_idx]);
            sum_vol -= LDG(&vol_prefix_tm[prev_idx]);
        }

        out_tm[out_idx] = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                           : nan_f32();
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `moving_averages/vwma.rs::vwma_scalar` (l.377). Inputs are
// (price, volume).
//   seed   = for i in 0..period: `sum += p*v; vsum += v` — ONE ELEMENT AT A
//            TIME, no grouping. (Contrast `qstick`/`var` in this same shard,
//            whose seeds DO group by four; the two are not interchangeable.)
//   emit0  = out[first + period - 1] = `sum / vsum` — a DIVIDE, not a
//            reciprocal multiply.
//   roll   = `sum += pn*vn; sum -= po*vo; vsum += vn - vo` in exactly that
//            order — three roundings — then `sum / vsum`. The CPU's 4-wide
//            unroll performs the identical operations per element in the
//            identical order, so a flat loop reproduces it exactly.
//
// f32 -> f64 audit: the f32 file was already `mix` (f32 in, f64 accumulator);
// this makes the whole path f64 — inputs, accumulator and output. Literals
// widened, `__int_as_float` NaN -> f64 quiet-NaN pattern, no fast-math
// intrinsic, no epsilon (a zero `vsum` divides to Inf/NaN exactly as it does on
// the CPU), no min/max chain.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double vwma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void vwma_batch_f64(const double* __restrict__ price,
                    const double* __restrict__ volume,
                    int n,
                    const int*   __restrict__ periods,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = vwma_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const long long warm_ll =
        static_cast<long long>(first_valid) + static_cast<long long>(period) - 1;
    if (period <= 0 || warm_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    const int warm = static_cast<int>(warm_ll);
    const int base = first_valid;

    for (int t = 0; t < warm; ++t) row[t] = nan_d;

    double sum = 0.0;
    double vsum = 0.0;
    for (int i = 0; i < period; ++i) {
        const double p = price[base + i];
        const double v = volume[base + i];
        sum += p * v;
        vsum += v;
    }
    row[warm] = sum / vsum;

    int new_idx = base + period;
    int old_idx = base;
    while (new_idx < n) {
        const double pn = price[new_idx];
        const double vn = volume[new_idx];
        const double po = price[old_idx];
        const double vo = volume[old_idx];
        sum += pn * vn;
        sum -= po * vo;
        vsum += vn - vo;
        row[new_idx] = sum / vsum;
        ++new_idx;
        ++old_idx;
    }
}
