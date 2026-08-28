#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef DCH_NAN
#define DCH_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


#if __CUDA_ARCH__ >= 350
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

__device__ __forceinline__ int floor_log2_u32(unsigned int x) {
    return 31 - __clz(x);
}


extern "C" __global__ void rmq_init_level0_f32(const float* __restrict__ in,
                                               float* __restrict__ st_lvl0,
                                               int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) st_lvl0[i] = in[i];
}


extern "C" __global__ void rmq_init_nan_mask_u8(const float* __restrict__ high,
                                                const float* __restrict__ low,
                                                int N,
                                                int first_valid,
                                                unsigned char* __restrict__ mask_lvl0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (i >= first_valid) {
            const float h = high[i];
            const float l = low[i];
            mask_lvl0[i] = (isnan(h) || isnan(l)) ? 1u : 0u;
        } else {
            mask_lvl0[i] = 0u;
        }
    }
}


extern "C" __global__ void rmq_build_level_max_f32(const float* __restrict__ prev,
                                                   float* __restrict__ curr,
                                                   int N, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int j = i + offset;
        const int limit = N - (offset << 1) + 1;
        float a = prev[i];
        float b = (j < N) ? prev[j] : -CUDART_INF_F;
        curr[i] = (i < limit) ? fmaxf(a, b) : -CUDART_INF_F;
    }
}


extern "C" __global__ void rmq_build_level_min_f32(const float* __restrict__ prev,
                                                   float* __restrict__ curr,
                                                   int N, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int j = i + offset;
        const int limit = N - (offset << 1) + 1;
        float a = prev[i];
        float b = (j < N) ? prev[j] : CUDART_INF_F;
        curr[i] = (i < limit) ? fminf(a, b) : CUDART_INF_F;
    }
}


extern "C" __global__ void rmq_build_level_or_u8(const unsigned char* __restrict__ prev,
                                                 unsigned char* __restrict__ curr,
                                                 int N, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int j = i + offset;
        const int limit = N - (offset << 1) + 1;
        const unsigned char a = prev[i];
        const unsigned char b = (j < N) ? prev[j] : 0u;
        curr[i] = (i < limit) ? (unsigned char)(a | b) : (unsigned char)0u;
    }
}

extern "C" __global__ void donchian_batch_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const int*   __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               float* __restrict__ out_upper,
                                               float* __restrict__ out_middle,
                                               float* __restrict__ out_lower) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    float* uo = out_upper + base;
    float* mo = out_middle + base;
    float* lo = out_lower + base;


    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) {
            uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN;
        }
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < series_len; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }

    if (period == 1) {
        for (int i = first_valid; i < series_len; ++i) {
            const float h = high[i];
            const float l = low[i];
            if (isnan(h) || isnan(l)) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
            else { uo[i] = h; lo[i] = l; mo[i] = 0.5f * (h + l); }
        }
        return;
    }


    for (int i = warm; i < series_len; ++i) {
        const int start = i + 1 - period;
        float maxv = -CUDART_INF_F;
        float minv =  CUDART_INF_F;
        bool any_nan = false;
        for (int k = 0; k < period; ++k) {
            const float h = high[start + k];
            const float l = low[start + k];
            if (UNLIKELY(isnan(h) || isnan(l))) { any_nan = true; break; }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        if (any_nan) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        else { uo[i] = maxv; lo[i] = minv; mo[i] = 0.5f * (maxv + minv); }
    }
}


extern "C" __global__ void donchian_batch_from_rmq_f32(
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    const float* __restrict__ st_high,
    const float* __restrict__ st_low,
    const unsigned char* __restrict__ st_nan,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower) {

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int N = series_len;
    const int period = periods[combo];
    const int base   = combo * N;

    float* __restrict__ uo = out_upper  + base;
    float* __restrict__ mo = out_middle + base;
    float* __restrict__ lo = out_lower  + base;

    if (UNLIKELY(period <= 0 || period > N || first_valid < 0 || first_valid >= N)) {
        for (int i = 0; i < N; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }
    const int tail_len = N - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < N; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }

    const int k    = floor_log2_u32((unsigned)period);
    const int len2 = 1 << k;
    const size_t off = (size_t)k * (size_t)N;

    const float* __restrict__ hi_lvl = st_high + off;
    const float* __restrict__ lo_lvl = st_low  + off;
    const unsigned char* __restrict__ nm_lvl = st_nan + off;

    for (int i = warm; i < N; ++i) {
        const int L = i + 1 - period;
        const int R = i;
        const int R2 = R - len2 + 1;

        const unsigned char nm = (unsigned char)(LDG(nm_lvl + L) | LDG(nm_lvl + R2));
        if (UNLIKELY(nm)) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; continue; }

        const float uh = fmaxf(LDG(hi_lvl + L),  LDG(hi_lvl + R2));
        const float ll = fminf(LDG(lo_lvl + L),  LDG(lo_lvl + R2));
        uo[i] = uh; lo[i] = ll; mo[i] = 0.5f * (uh + ll);
    }
}


extern "C" __global__ void donchian_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ upper_tm,
    float* __restrict__ middle_tm,
    float* __restrict__ lower_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len || period <= 0 || period > series_len || (series_len - first_valid) < period) {

        int idx = series;
        for (int row = 0; row < series_len; ++row, idx += num_series) {
            upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    int idx = series;
    for (int row = 0; row < warm; ++row, idx += num_series) {
        upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN;
    }

    if (period == 1) {
        for (int row = first_valid; row < series_len; ++row, idx += num_series) {
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            if (UNLIKELY(isnan(h) || isnan(l))) { upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN; }
            else { upper_tm[idx] = h; lower_tm[idx] = l; middle_tm[idx] = 0.5f * (h + l); }
        }
        return;
    }

    for (int row = warm; row < series_len; ++row, idx += num_series) {
        const int start = row + 1 - period;
        float maxv = -CUDART_INF_F;
        float minv =  CUDART_INF_F;
        bool any_nan = false;

        int idxk = (start * num_series) + series;
        for (int k = 0; k < period; ++k, idxk += num_series) {
            const float h = high_tm[idxk];
            const float l = low_tm[idxk];
            if (UNLIKELY(isnan(h) || isnan(l))) { any_nan = true; break; }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        if (any_nan) { upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN; }
        else { upper_tm[idx] = maxv; lower_tm[idx] = minv; middle_tm[idx] = 0.5f * (maxv + minv); }
    }
}


// ===========================================================================
// f64 LANE  --  shard S6
//
// CPU reference: `donchian_scalar` (src/indicators/donchian.rs:425).
//
// OUTPUTS: canonical `[upper, middle, lower]`. The preserved generic primary
// ABI still returns `upper`; production enters the full three-output ABI so a
// family tuple is launched once. Both entry points call one exact row authority.
//
// FIRST VALID IS NOT THE COMMON RULE. donchian.rs:183-188 scans high and low
// INDEPENDENTLY and takes `h.max(l)` -- the same construction `adx` uses over
// three series. It is NOT "the first index at which both are non-NaN": if low
// starts at 0 and high has a hole at 3 that clears by 5, the independent-max
// answer is 0 and the simultaneous answer is 0 too, but move the hole to
// before low's first bar and they part company. Declared as
// `F64FirstValidRule::MaxOfIndependentFirsts`.
//
// THE CPU HAS TWO DIFFERENT VALIDITY TESTS AND THEY DISAGREE ON INFINITIES.
// This is reproduced, not silently unified, because the CPU is the oracle:
//   * period <= 32 (:470-506) rejects a window on `h.is_nan() || l.is_nan()`.
//     An INFINITE high therefore flows straight through and becomes the upper
//     band.
//   * period  > 32 (:509-548) builds its block max/min from
//     `ok = h.is_finite() & l.is_finite()`, so an infinite high is treated as
//     MISSING and the bar becomes NaN.
// So `donchian(period=32)` and `donchian(period=33)` answer differently for
// the same infinite bar. That is a defect in vector-ta worth fixing on the CPU
// side, but fixing it here alone would put the device out of parity with the
// host, which is strictly worse. Reported rather than papered over.
//
// PARALLEL OVER (combo, bar), not one thread per column: donchian carries no
// state across bars -- every output is a fresh max/min over its own window --
// so `F64Kernel::is_sequential` is false for it and the launcher uses the
// (bars x rows) grid. For period <= 32 the row scans forward exactly like the
// CPU. For period > 32 it reconstructs the CPU's reverse suffix, forward
// prefix, and suffix-vs-prefix selection order. Equal signed zeroes make that
// order observable, so a simpler full-window scan is not exact-bit authority.
//
// f32 -> f64 audit: the f32 lane above uses `fmaxf` x2, `fminf` x2 and
// `__int_as_float` x2. Below there is no f32-suffixed function and no
// fast-math intrinsic. The comparison chains `if (hj > maxv)` / `if (lj < minv)`
// are kept as comparisons and NOT converted to fmax/fmin -- that is deliberate
// and it is the CPU's own structure (:487-493): NaN can never reach them
// because `has_nan` breaks out first, and in the period > 32 arm an invalid
// bar has already been mapped to -INFINITY / +INFINITY exactly as :531-532
// does. Using fmax here would change nothing for valid data and would MASK the
// infinity behaviour the CPU has. No epsilon exists in this indicator.
// ===========================================================================

static __device__ __forceinline__ double donchian_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

static __device__ __forceinline__
void donchian_row_f64(const double* __restrict__ high,
                      const double* __restrict__ low,
                      int n,
                      int period,
                      int first_valid,
                      int i,
                      double* upper,
                      double* middle,
                      double* lower)
{
    const double nan_d = donchian_qnan_f64();
    if (period <= 0 || period > n || first_valid < 0) {
        *upper = nan_d;
        *middle = nan_d;
        *lower = nan_d;
        return;
    }

    const int warmup = first_valid + period - 1;
    if (i < warmup) {
        *upper = nan_d;
        *middle = nan_d;
        *lower = nan_d;
        return;
    }

    if (period == 1) {
        const double h = high[i];
        const double l = low[i];
        if (isnan(h) || isnan(l)) {
            *upper = nan_d;
            *middle = nan_d;
            *lower = nan_d;
        } else {
            *upper = h;
            *middle = fma(h - l, 0.5, l);
            *lower = l;
        }
        return;
    }

    const int start = i + 1 - period;
    double maxv = -INFINITY;
    double minv = INFINITY;

    if (period <= 32) {
        bool has_nan = false;
        for (int k = 0; k < period; ++k) {
            const double h = high[start + k];
            const double l = low[start + k];
            if (isnan(h) || isnan(l)) {
                has_nan = true;
                break;
            }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        if (has_nan) {
            *upper = nan_d;
            *middle = nan_d;
            *lower = nan_d;
            return;
        }
    } else {
        bool all_valid = true;
        for (int k = 0; k < period; ++k) {
            const double h = high[start + k];
            const double l = low[start + k];
            const bool ok = isfinite(h) && isfinite(l);
            if (!ok) all_valid = false;
        }
        if (!all_valid) {
            *upper = nan_d;
            *middle = nan_d;
            *lower = nan_d;
            return;
        }

        const int suffix_end = min(((start / period) + 1) * period - 1, n - 1);
        double suffix_max = high[suffix_end];
        double suffix_min = low[suffix_end];
        for (int k = suffix_end - 1; k >= start; --k) {
            const double h = high[k];
            const double l = low[k];
            if (h > suffix_max) suffix_max = h;
            if (l < suffix_min) suffix_min = l;
        }

        const int prefix_start = (i / period) * period;
        double prefix_max = high[prefix_start];
        double prefix_min = low[prefix_start];
        for (int k = prefix_start + 1; k <= i; ++k) {
            const double h = high[k];
            const double l = low[k];
            if (h > prefix_max) prefix_max = h;
            if (l < prefix_min) prefix_min = l;
        }

        maxv = suffix_max > prefix_max ? suffix_max : prefix_max;
        minv = suffix_min < prefix_min ? suffix_min : prefix_min;
    }

    *upper = maxv;
    *middle = fma(maxv - minv, 0.5, minv);
    *lower = minv;
}

extern "C" __global__
void donchian_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    double upper;
    double middle;
    double lower;
    donchian_row_f64(high, low, n, periods[r], first_valid, i, &upper, &middle, &lower);
    row[i] = upper;
}

extern "C" __global__
void donchian_all_outputs_batch_f64(const double* __restrict__ high,
                                    const double* __restrict__ low,
                                    int n,
                                    const int* __restrict__ periods,
                                    int n_combos,
                                    int first_valid,
                                    double* __restrict__ upper_out,
                                    double* __restrict__ middle_out,
                                    double* __restrict__ lower_out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const size_t offset = static_cast<size_t>(r) * static_cast<size_t>(n) +
                          static_cast<size_t>(i);
    double upper;
    double middle;
    double lower;
    donchian_row_f64(high, low, n, periods[r], first_valid, i, &upper, &middle, &lower);
    upper_out[offset] = upper;
    middle_out[offset] = middle;
    lower_out[offset] = lower;
}
