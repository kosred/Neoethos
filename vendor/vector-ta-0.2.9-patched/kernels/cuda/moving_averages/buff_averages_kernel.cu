#include <cuda_runtime.h>
#include <math.h>


#ifndef BA_ENABLE_L2_PREFETCH
#define BA_ENABLE_L2_PREFETCH 1
#endif

#ifndef BA_EXP2_NR_STEPS
#define BA_EXP2_NR_STEPS 1
#endif

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

extern "C" __global__ void buff_averages_build_prefix_f32(
    const float* __restrict__ prices,
    const float* __restrict__ volumes,
    int len,
    float* __restrict__ prefix_pv,
    float* __restrict__ prefix_vv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    prefix_pv[0] = 0.0f;
    prefix_vv[0] = 0.0f;
    double acc_pv = 0.0;
    double acc_vv = 0.0;
    for (int i = 0; i < len; ++i) {
        const float p = prices[i];
        const float v = volumes[i];
        const double pv = (isnan(p) || isnan(v)) ? 0.0 : (double)p * (double)v;
        const double vv = isnan(v) ? 0.0 : (double)v;
        acc_pv += pv;
        acc_vv += vv;
        prefix_pv[i + 1] = (float)acc_pv;
        prefix_vv[i + 1] = (float)acc_vv;
    }
}


__device__ __forceinline__ float ratio_from_prefix(float pv_t, float pv_s,
                                                   float vv_t, float vv_s) {
    float den = vv_t - vv_s;
    if (den != 0.0f) {
        float rcp = __frcp_rn(den);
        rcp = fmaf(rcp, (2.0f - den * rcp), 0.0f);
        return (pv_t - pv_s) * rcp;
    }
    return 0.0f;
}


struct f2 { float hi, lo; };

__device__ __forceinline__ f2 two_sum(float a, float b) {
    float s = a + b;
    float bp = s - a;
    float e = (a - (s - bp)) + (b - bp);
    f2 r; r.hi = s; r.lo = e; return r;
}

__device__ __forceinline__ f2 add_f2(f2 x, f2 y) {
    f2 s = two_sum(x.hi, y.hi);
    float t = x.lo + y.lo;
    f2 r = two_sum(s.hi, s.lo + t);
    return r;
}

__device__ __forceinline__ f2 sub_f2(f2 x, f2 y) {
    f2 s = two_sum(x.hi, -y.hi);
    float t = x.lo - y.lo;
    f2 r = two_sum(s.hi, s.lo + t);
    return r;
}

__device__ __forceinline__ float div_f2(f2 n, f2 d) {

    if (d.hi == 0.0f && d.lo == 0.0f) return 0.0f;

    float rcp = __frcp_rn(d.hi);
#if BA_EXP2_NR_STEPS >= 1
    rcp = fmaf(rcp, (2.0f - d.hi * rcp), 0.0f);
#endif

    float q0 = n.hi * rcp;
    float r  = fmaf(-q0, d.hi, n.hi);
    r        = fmaf(-q0, d.lo, r);
    r       += n.lo;
    float q1 = r * rcp;
#if BA_EXP2_NR_STEPS >= 2

    float r2 = fmaf(-(q0 + q1), d.hi, n.hi);
    r2       = fmaf(-(q0 + q1), d.lo, r2);
    r2      += n.lo;
    q1      += r2 * rcp;
#endif
    return q0 + q1;
}

extern "C" __global__ void buff_averages_batch_prefix_f32(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (; t < len; t += stride) {
        float fo = qnan32();
        float so = qnan32();

#if BA_ENABLE_L2_PREFETCH

        int t_pref = t + stride;
        if (t_pref + 1 < len + 1) {
            const float* p0 = prefix_pv + (t_pref + 1);
            const float* p1 = prefix_vv + (t_pref + 1);
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
            asm volatile ("prefetch.global.L2 [%0];" :: "l"(p0));
            asm volatile ("prefetch.global.L2 [%0];" :: "l"(p1));
#endif
        }
#endif

        if (t >= warm) {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            const float pv_t = prefix_pv[t1];
            const float vv_t = prefix_vv[t1];
            so = ratio_from_prefix(pv_t, prefix_pv[sstart], vv_t, prefix_vv[sstart]);
            fo = ratio_from_prefix(pv_t, prefix_pv[fstart], vv_t, prefix_vv[fstart]);
        }

        fast_out[row_offset + t] = fo;
        slow_out[row_offset + t] = so;
    }
}


template<int TILE>
__device__ __forceinline__ void buff_averages_batch_prefix_tiled_f32_impl(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    const int t0 = blockIdx.x * TILE;
    const int t  = t0 + threadIdx.x;
    if (t >= len) return;

    float fast_val = nan_f;
    float slow_val = nan_f;

    if (t >= warm) {
        int fast_start = t + 1 - fast_period;
        if (fast_start < 0) fast_start = 0;
        int slow_start = t + 1 - slow_period;
        if (slow_start < 0) slow_start = 0;

        const float slow_num = prefix_pv[t + 1] - prefix_pv[slow_start];
        const float slow_den = prefix_vv[t + 1] - prefix_vv[slow_start];
        if (slow_den != 0.0f) {
            float rcp = __frcp_rn(slow_den);
            rcp = fmaf(rcp, (2.0f - slow_den * rcp), 0.0f);
            slow_val = slow_num * rcp;
        } else {
            slow_val = 0.0f;
        }

        const float fast_num = prefix_pv[t + 1] - prefix_pv[fast_start];
        const float fast_den = prefix_vv[t + 1] - prefix_vv[fast_start];
        if (fast_den != 0.0f) {
            float rcp = __frcp_rn(fast_den);
            rcp = fmaf(rcp, (2.0f - fast_den * rcp), 0.0f);
            fast_val = fast_num * rcp;
        } else {
            fast_val = 0.0f;
        }
    }

    fast_out[row_offset + t] = fast_val;
    slow_out[row_offset + t] = slow_val;
}

extern "C" __global__ void buff_averages_batch_prefix_tiled_f32_tile128(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    buff_averages_batch_prefix_tiled_f32_impl<128>(
        prefix_pv, prefix_vv, len, first_valid,
        fast_periods, slow_periods, n_combos, fast_out, slow_out);
}

extern "C" __global__ void buff_averages_batch_prefix_tiled_f32_tile256(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    buff_averages_batch_prefix_tiled_f32_impl<256>(
        prefix_pv, prefix_vv, len, first_valid,
        fast_periods, slow_periods, n_combos, fast_out, slow_out);
}


extern "C" __global__ void buff_averages_batch_prefix_tiled_f32_tile512(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    buff_averages_batch_prefix_tiled_f32_impl<512>(
        prefix_pv, prefix_vv, len, first_valid,
        fast_periods, slow_periods, n_combos, fast_out, slow_out);
}


extern "C" __global__ void buff_averages_many_series_one_param_f32(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valids[series] + slow_period - 1;
    const int stride = num_series;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + series;
        if (t < warm) {
            fast_out_tm[out_idx] = qnan32();
            slow_out_tm[out_idx] = qnan32();
        } else {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            const int p_idx = t1 * stride + series;
            const int f_idx = fstart * stride + series;
            const int s_idx = sstart * stride + series;

            const float pv_t = pv_prefix_tm[p_idx];
            const float vv_t = vv_prefix_tm[p_idx];

            fast_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[f_idx],
                                                     vv_t, vv_prefix_tm[f_idx]);
            slow_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[s_idx],
                                                     vv_t, vv_prefix_tm[s_idx]);
        }
        t += step;
    }
}

template<int TX, int TY>
__device__ __forceinline__ void buff_averages_many_series_one_param_tiled2d_impl(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    const int s = blockIdx.y * TY + threadIdx.y;
    if (s >= num_series) return;
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valids[s] + slow_period - 1;
    const int stride = num_series;

    const int t0 = blockIdx.x * TX;
    const int t = t0 + threadIdx.x;
    if (t >= series_len) return;

    const int out_idx = t * stride + s;
    if (t < warm) {
        fast_out_tm[out_idx] = qnan32();
        slow_out_tm[out_idx] = qnan32();
        return;
    }

    const int t1 = t + 1;
    int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
    int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

    const int p_idx = t1 * stride + s;
    const int f_idx = fstart * stride + s;
    const int s_idx = sstart * stride + s;

    const float pv_t = pv_prefix_tm[p_idx];
    const float vv_t = vv_prefix_tm[p_idx];

    fast_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[f_idx],
                                             vv_t, vv_prefix_tm[f_idx]);
    slow_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[s_idx],
                                             vv_t, vv_prefix_tm[s_idx]);
}

extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_tx128_ty2(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_impl<128, 2>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}

extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_tx128_ty4(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_impl<128, 4>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}


template<int SX, int TY>
__device__ __forceinline__ void buff_averages_many_series_one_param_tiled2d_swizzled_f32(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {

    if (fast_period <= 0 || slow_period <= 0) return;

    const int s = blockIdx.y * SX + threadIdx.x;
    if (s >= num_series) return;

    const int t = blockIdx.x * TY + threadIdx.y;
    if (t >= series_len) return;

    const int warm = first_valids[s] + slow_period - 1;
    const int stride = num_series;

    const int out_idx = t * stride + s;
    if (t < warm) {
        fast_out_tm[out_idx] = qnan32();
        slow_out_tm[out_idx] = qnan32();
        return;
    }

    const int t1 = t + 1;
    int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
    int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

    const int p_idx = t1 * stride + s;
    const int f_idx = fstart * stride + s;
    const int s_idx = sstart * stride + s;

    const float pv_t = pv_prefix_tm[p_idx];
    const float vv_t = vv_prefix_tm[p_idx];

    fast_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[f_idx],
                                             vv_t, vv_prefix_tm[f_idx]);
    slow_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[s_idx],
                                             vv_t, vv_prefix_tm[s_idx]);
}


extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_sx128_ty1(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_swizzled_f32<128,1>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}

extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_sx128_ty2(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_swizzled_f32<128,2>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}


extern "C" __global__ void buff_averages_batch_prefix_exp2_f32(
    const float* __restrict__ pv_hi,
    const float* __restrict__ pv_lo,
    const float* __restrict__ vv_hi,
    const float* __restrict__ vv_lo,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float fast_val = nan_f;
        float slow_val = nan_f;
        if (t >= warm) {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            f2 pv_t  = { pv_hi[t1],  pv_lo[t1] };
            f2 pv_f0 = { pv_hi[fstart], pv_lo[fstart] };
            f2 pv_s0 = { pv_hi[sstart], pv_lo[sstart] };
            f2 vv_t  = { vv_hi[t1],  vv_lo[t1] };
            f2 vv_f0 = { vv_hi[fstart], vv_lo[fstart] };
            f2 vv_s0 = { vv_hi[sstart], vv_lo[sstart] };

            f2 fast_num = sub_f2(pv_t, pv_f0);
            f2 fast_den = sub_f2(vv_t, vv_f0);
            f2 slow_num = sub_f2(pv_t, pv_s0);
            f2 slow_den = sub_f2(vv_t, vv_s0);

            fast_val = div_f2(fast_num, fast_den);
            slow_val = div_f2(slow_num, slow_den);
        }
        fast_out[row_offset + t] = fast_val;
        slow_out[row_offset + t] = slow_val;
        t += stride;
    }
}


extern "C" __global__ void buff_averages_many_series_one_param_exp2_f32(
    const float* __restrict__ pv_hi_tm,
    const float* __restrict__ pv_lo_tm,
    const float* __restrict__ vv_hi_tm,
    const float* __restrict__ vv_lo_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valids[s] + slow_period - 1;
    const int stride = num_series;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + s;
        if (t < warm) {
            fast_out_tm[out_idx] = qnan32();
            slow_out_tm[out_idx] = qnan32();
        } else {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            const int p = t1 * stride + s;
            const int f = fstart * stride + s;
            const int q = sstart * stride + s;

            f2 pv_t  = { pv_hi_tm[p], pv_lo_tm[p] };
            f2 pv_f0 = { pv_hi_tm[f], pv_lo_tm[f] };
            f2 pv_s0 = { pv_hi_tm[q], pv_lo_tm[q] };
            f2 vv_t  = { vv_hi_tm[p], vv_lo_tm[p] };
            f2 vv_f0 = { vv_hi_tm[f], vv_lo_tm[f] };
            f2 vv_s0 = { vv_hi_tm[q], vv_lo_tm[q] };

            f2 fast_num = sub_f2(pv_t, pv_f0);
            f2 fast_den = sub_f2(vv_t, vv_f0);
            f2 slow_num = sub_f2(pv_t, pv_s0);
            f2 slow_den = sub_f2(vv_t, vv_s0);

            fast_out_tm[out_idx] = div_f2(fast_num, fast_den);
            slow_out_tm[out_idx] = div_f2(slow_num, slow_den);
        }
        t += step;
    }
}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: `buff_averages_scalar`
// (src/indicators/moving_averages/buff_averages.rs:599-691), reached through
// `buff_averages_with_kernel` (:342) -> `buff_averages_compute_into` (:524).
//
// OUTPUT: the FAST buff. `ma_batch.rs:627-637` defaults `output` to "fast",
// so that is the column a caller asking for `value` gets.
//
// PERIOD-SWEPT, and the swept int is the SLOW period: `ma_batch.rs:593`
// assigns `sweep.slow_period = period_range` while `fast_period` stays at its
// default 5 (`BuffAveragesBatchRange::default`, :1429). Mapping the swept int
// onto `fast_period` instead would compute a different indicator.
//
// SHAPE: one thread per combo walking bars ASCENDING. The numerator and
// denominator are carried across bars with subtract-then-add (:616-651), so
// the accumulation order is load-bearing and a bar-parallel rebuild would
// round differently. The window is read straight out of global memory --
// no per-thread ring, hence no `max_period` and NEVER-OOM by construction.
//
// FIRST-VALID: declared `Ignored` and derived here. `buff_averages_prepare`
// (:470) scans PRICE ALONE with `!is_nan`; volume is never scanned, so the
// `AllInputsNonNan` rule over a (close, volume) pair would name a later bar
// on any frame whose volume has a hole, and first-valid sets both the NaN
// prefix and the seed window.
// ===========================================================================

static __forceinline__ __device__ double buff_averages_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void buff_averages_neo_batch_f64(const double* __restrict__ price,
                                 const double* __restrict__ volume,
                                 int n,
                                 const int* __restrict__ periods,
                                 int n_combos,
                                 int first_valid,
                                 double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;
    (void)first_valid;  // derived below -- see the header.

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = buff_averages_neo_qnan();

    const int slow_period = periods[combo];
    const int fast_period = 5;  // buff_averages.rs:1429 / :220

    // buff_averages_prepare, :470-473 -- price alone, `!is_nan`.
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(price[i])) { first = i; break; }
    }

    bool refused = false;
    if (first < 0) refused = true;                                   // AllValuesNaN
    if (fast_period <= 0 || fast_period > n) refused = true;          // :476
    if (slow_period <= 0 || slow_period > n) refused = true;          // :483
    if (!refused && (n - first) < slow_period) refused = true;        // :491

    long long warm_ll = 0;
    if (!refused) {
        warm_ll = (long long)first + (long long)slow_period - 1;
        // :613 -- `if warm >= len { return; }` leaves an all-NaN column.
        if (warm_ll >= (long long)n) refused = true;
        // The CPU forms `warm + 1 - fast_period` as a usize (:634). When the
        // slow window is shorter than the fast one that expression underflows
        // rather than producing a window, so the row is refused by name here
        // instead of reading out of bounds.
        else if (warm_ll + 1 < (long long)fast_period) refused = true;
    }

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    const int warm = (int)warm_ll;
    for (int i = 0; i < warm; ++i) row[i] = nn;

    // :633-640 -- the fast seed window, ascending, skipping any bar whose
    // price OR volume is NaN.
    double num = 0.0;
    double den = 0.0;
    const int fast_start = warm + 1 - fast_period;
    for (int i = fast_start; i <= warm; ++i) {
        const double p = price[i];
        const double v = volume[i];
        if (!isnan(p) && !isnan(v)) {
            num += p * v;
            den += v;
        }
    }
    row[warm] = (den != 0.0) ? (num / den) : 0.0;

    // :615-651 -- subtract the leaving bar, then add the arriving one, in
    // that order. Reversing them changes the rounding of `num`.
    for (int i = warm + 1; i < n; ++i) {
        const double new_p = price[i];
        const double new_v = volume[i];
        const int old_i = i - fast_period;
        const double old_p = price[old_i];
        const double old_v = volume[old_i];

        if (!isnan(old_p) && !isnan(old_v)) {
            num -= old_p * old_v;
            den -= old_v;
        }
        if (!isnan(new_p) && !isnan(new_v)) {
            num += new_p * new_v;
            den += new_v;
        }

        row[i] = (den != 0.0) ? (num / den) : 0.0;
    }
}
