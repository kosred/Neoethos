#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


__device__ __forceinline__
void max_earliest_update(float v, int i, float &best_v, int &best_i) {
    if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
}


__device__ __forceinline__
void min_earliest_update(float v, int i, float &best_v, int &best_i) {
    if (v < best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
}


__device__ __forceinline__
void warp_argmaxmin_earliest(float &max_v, int &max_i, float &min_v, int &min_i, unsigned mask) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float mv = __shfl_down_sync(mask, max_v, offset);
        int   mi = __shfl_down_sync(mask, max_i, offset);
        if (mv > max_v || (mv == max_v && mi < max_i)) { max_v = mv; max_i = mi; }

        float nv = __shfl_down_sync(mask, min_v, offset);
        int   ni = __shfl_down_sync(mask, min_i, offset);
        if (nv < min_v || (nv == min_v && ni < min_i)) { min_v = nv; min_i = ni; }
    }
}


extern "C" __global__
void aroonosc_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        const int*   __restrict__ lengths,
                        int series_len,
                        int first_valid,
                        int n_combos,
                        float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int base = combo * series_len;

    const int L = lengths[combo];
    if (L <= 0 || first_valid < 0 || first_valid >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }

    const int warm = first_valid + L;
    if (warm >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }

    const float scale = 100.0f / (float)L;


    const unsigned mask = __activemask();
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x / WARP_SIZE;


    for (int t = warm + warp_id; t < series_len; t += warps_per_block) {
        const int start = t - L;


        float max_v = high[start];
        int   max_i = start;
        float min_v = low[start];
        int   min_i = start;


        for (int j = start + lane; j <= t; j += WARP_SIZE) {
            const float h = high[j];
            const float l = low[j];
            max_earliest_update(h, j, max_v, max_i);
            min_earliest_update(l, j, min_v, min_i);
        }


        warp_argmaxmin_earliest(max_v, max_i, min_v, min_i, mask);

        if (lane == 0) {
            float v = (float)(max_i - min_i) * scale;

            v = fminf(100.0f, fmaxf(-100.0f, v));
            out[base + t] = v;
        }
    }
}


extern "C" __global__
void aroonosc_many_series_one_param_f32(const float* __restrict__ high_tm,
                                        const float* __restrict__ low_tm,
                                        const int*   __restrict__ first_valids,
                                        int num_series,
                                        int series_len,
                                        int length,
                                        float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0) return;

    if (length <= 0) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + s] = CUDART_NAN_F;
        }
        return;
    }

    const int fv   = first_valids[s] < 0 ? 0 : first_valids[s];
    const int warm = fv + length;
    if (warm >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + s] = CUDART_NAN_F;
        }
        return;
    }


    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_tm[t * num_series + s] = CUDART_NAN_F;
    }

    const float scale  = 100.0f / (float)length;
    const int   stride = num_series;

    if (threadIdx.x != 0) return;

    for (int t = warm; t < series_len; ++t) {
        const int start = t - length;
        int   hi_idx = start,  lo_idx = start;
        float hi_val = high_tm[start * stride + s];
        float lo_val =  low_tm[start * stride + s];

        for (int j = start + 1; j <= t; ++j) {
            const float h = high_tm[j * stride + s];
            if (h > hi_val) { hi_val = h; hi_idx = j; }
            const float l = low_tm[j * stride + s];
            if (l < lo_val) { lo_val = l; lo_idx = j; }
        }
        float v = (float)(hi_idx - lo_idx) * scale;
        v = fminf(100.0f, fmaxf(-100.0f, v));
        out_tm[t * stride + s] = v;
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `aroonosc.rs::aroon_osc_scalar_highlow_into` (l.313).
// Inputs are (high, low); there is no close.
//     start_i = first + length          <- warmup is +length, NOT +length-1
//     scale   = 100.0 / length
//     window  = [i - length, i]  (length + 1 bars)
//     maxi    = the EARLIEST index in the window attaining the window max
//     mini    = the EARLIEST index in the window attaining the window min
//     v       = (maxi - mini) * scale
//     out[i]  = v.max(-100.0).min(100.0)
//
// "EARLIEST index" is not a detail. The CPU keeps a running (max, maxi) and
// only replaces it on a STRICT `>` — so on a tie the older index wins — and
// when the held index falls out of the window it rescans from the window start
// with the same strict `>`. A scan that used `>=` would pick the LATEST index
// and change the oscillator by up to 100 points on any flat stretch. This
// kernel rescans the window with strict `>` / `<` from the window start, which
// reproduces the invariant exactly.
//
// NaN: `hv > max` is false for a NaN `hv`, so the CPU silently skips NaN bars
// and keeps the last real extreme. Using fmax/fmin here would be WRONG in the
// other direction — it would also have to track an index, and fmax carries
// none. The comparison chain is the faithful form and it is retained; what is
// converted to fmax/fmin is the FINAL clamp, `v.max(-100.0).min(100.0)`, which
// really is `f64::max`/`f64::min` on the CPU.
//
// f32 -> f64 audit: pointers/locals widened; `fmaxf` x2 / `fminf` x2 -> `fmax`
// / `fmin`; `100.0f`/`-100.0f` widened; the f32 NaN constant replaced by the
// f64 quiet-NaN bit pattern. No fast-math intrinsic in this file, no epsilon.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double aroonosc_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void aroonosc_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        int n,
                        const int*   __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = aroonosc_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int length = periods[combo];
    const long long start_ll =
        static_cast<long long>(first_valid) + static_cast<long long>(length);
    if (length <= 0 || start_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    const int start_i = static_cast<int>(start_ll);
    const double scale = 100.0 / static_cast<double>(length);

    for (int t = 0; t < start_i; ++t) row[t] = nan_d;

    for (int i = start_i; i < n; ++i) {
        const int wstart = i - length;

        int maxi = wstart;
        int mini = wstart;
        double mx = high[wstart];
        double mn = low[wstart];
        for (int k = wstart + 1; k <= i; ++k) {
            const double hv = high[k];
            if (hv > mx) { mx = hv; maxi = k; }
            const double lv = low[k];
            if (lv < mn) { mn = lv; mini = k; }
        }

        const double v = (static_cast<double>(maxi) - static_cast<double>(mini)) * scale;
        row[i] = fmin(fmax(v, -100.0), 100.0);
    }
}
