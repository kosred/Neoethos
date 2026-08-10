#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ bool both_finite(float h, float l) {
    return isfinite(h) && isfinite(l);
}


extern "C" __global__
void aroon_batch_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     const int*   __restrict__ lengths,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out_up,
                     float* __restrict__ out_down) {

    const int combo = blockIdx.x + blockIdx.y * gridDim.x;
    if (combo >= n_combos) return;

    const int length = lengths[combo];
    if (length <= 0 || first_valid < 0 || first_valid >= series_len) return;

    const int base = combo * series_len;
    const int W = length + 1;
    const int warm = first_valid + length;
    if (warm >= series_len) {

        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_up  [base + i] = NAN;
            out_down[base + i] = NAN;
        }
        return;
    }


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_up  [base + i] = NAN;
        out_down[base + i] = NAN;
    }
    __syncthreads();


    extern __shared__ unsigned char s_mem[];
    int* __restrict__ dq_max_idx = reinterpret_cast<int*>(s_mem);
    int* __restrict__ dq_min_idx = dq_max_idx + W;
    float* __restrict__ dq_max_val = reinterpret_cast<float*>(dq_min_idx + W);
    float* __restrict__ dq_min_val = dq_max_val + W;


    if (threadIdx.x != 0) return;


    int h_head = 0, h_tail = 0;
    int h_head_idx = 0, h_tail_idx = 0;
    int l_head = 0, l_tail = 0;
    int l_head_idx = 0, l_tail_idx = 0;

    const float scale = 100.0f / (float)length;
    int last_bad = -0x3fffffff;


    for (int t = 0; t < series_len; ++t) {
        const int start = t - length;

        while (h_tail > h_head && dq_max_idx[h_head_idx] < start) {
            ++h_head;
            h_head_idx = (h_head_idx + 1 == W) ? 0 : (h_head_idx + 1);
        }
        while (l_tail > l_head && dq_min_idx[l_head_idx] < start) {
            ++l_head;
            l_head_idx = (l_head_idx + 1 == W) ? 0 : (l_head_idx + 1);
        }

        const float h = high[t];
        const float l = low[t];


        if (!both_finite(h, l)) {
            last_bad = t;
        } else {

            while (h_tail > h_head) {
                const int last_slot = (h_tail_idx == 0) ? (W - 1) : (h_tail_idx - 1);
                if (dq_max_val[last_slot] < h) {
                    --h_tail;
                    h_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_max_idx[h_tail_idx] = t;
            dq_max_val[h_tail_idx] = h;
            ++h_tail;
            h_tail_idx = (h_tail_idx + 1 == W) ? 0 : (h_tail_idx + 1);

            while (l_tail > l_head) {
                const int last_slot = (l_tail_idx == 0) ? (W - 1) : (l_tail_idx - 1);
                if (dq_min_val[last_slot] > l) {
                    --l_tail;
                    l_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_min_idx[l_tail_idx] = t;
            dq_min_val[l_tail_idx] = l;
            ++l_tail;
            l_tail_idx = (l_tail_idx + 1 == W) ? 0 : (l_tail_idx + 1);
        }

        if (t >= warm) {
            if (last_bad >= start) {
                out_up  [base + t] = NAN;
                out_down[base + t] = NAN;
            } else {
                const int idx_hi = (h_tail > h_head) ? dq_max_idx[h_head_idx] : -1;
                const int idx_lo = (l_tail > l_head) ? dq_min_idx[l_head_idx] : -1;
                if (idx_hi < 0 || idx_lo < 0) {
                    out_up  [base + t] = NAN;
                    out_down[base + t] = NAN;
                } else {
                    const int dist_hi = t - idx_hi;
                    const int dist_lo = t - idx_lo;

                    const float up = (dist_hi == 0) ? 100.0f
                                     : (dist_hi >= length ? 0.0f
                                     : fmaf(-(float)dist_hi, scale, 100.0f));
                    const float dn = (dist_lo == 0) ? 100.0f
                                     : (dist_lo >= length ? 0.0f
                                     : fmaf(-(float)dist_lo, scale, 100.0f));
                    out_up  [base + t] = up;
                    out_down[base + t] = dn;
                }
            }
        }
    }
}


extern "C" __global__
void aroon_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const int*   __restrict__ first_valids,
                                     int length,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_up_tm,
                                     float* __restrict__ out_down_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || length <= 0) return;

    const int first = first_valids[s];
    if (first < 0 || first >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_up_tm  [t * num_series + s] = NAN;
            out_down_tm[t * num_series + s] = NAN;
        }
        return;
    }

    const int W = length + 1;
    const int warm = first + length;
    const int stride = num_series;


    for (int t = threadIdx.x; t < (warm < series_len ? warm : series_len); t += blockDim.x) {
        out_up_tm  [t * stride + s] = NAN;
        out_down_tm[t * stride + s] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    extern __shared__ unsigned char s_mem[];
    int* __restrict__ dq_max_idx = reinterpret_cast<int*>(s_mem);
    int* __restrict__ dq_min_idx = dq_max_idx + W;
    float* __restrict__ dq_max_val = reinterpret_cast<float*>(dq_min_idx + W);
    float* __restrict__ dq_min_val = dq_max_val + W;

    int h_head = 0, h_tail = 0;
    int h_head_idx = 0, h_tail_idx = 0;
    int l_head = 0, l_tail = 0;
    int l_head_idx = 0, l_tail_idx = 0;
    const float scale = 100.0f / (float)length;
    int last_bad = -0x3fffffff;

    for (int t = 0; t < series_len; ++t) {
        const int start = t - length;
        while (h_tail > h_head && dq_max_idx[h_head_idx] < start) {
            ++h_head;
            h_head_idx = (h_head_idx + 1 == W) ? 0 : (h_head_idx + 1);
        }
        while (l_tail > l_head && dq_min_idx[l_head_idx] < start) {
            ++l_head;
            l_head_idx = (l_head_idx + 1 == W) ? 0 : (l_head_idx + 1);
        }

        const float h = high_tm[t * stride + s];
        const float l = low_tm [t * stride + s];

        if (!both_finite(h, l)) {
            last_bad = t;
        } else {

            while (h_tail > h_head) {
                const int last_slot = (h_tail_idx == 0) ? (W - 1) : (h_tail_idx - 1);
                if (dq_max_val[last_slot] < h) {
                    --h_tail;
                    h_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_max_idx[h_tail_idx] = t;
            dq_max_val[h_tail_idx] = h;
            ++h_tail;
            h_tail_idx = (h_tail_idx + 1 == W) ? 0 : (h_tail_idx + 1);

            while (l_tail > l_head) {
                const int last_slot = (l_tail_idx == 0) ? (W - 1) : (l_tail_idx - 1);
                if (dq_min_val[last_slot] > l) {
                    --l_tail;
                    l_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_min_idx[l_tail_idx] = t;
            dq_min_val[l_tail_idx] = l;
            ++l_tail;
            l_tail_idx = (l_tail_idx + 1 == W) ? 0 : (l_tail_idx + 1);
        }

        if (t >= warm) {
            if (last_bad >= start) {
                out_up_tm  [t * stride + s] = NAN;
                out_down_tm[t * stride + s] = NAN;
            } else {
                const int idx_hi = (h_tail > h_head) ? dq_max_idx[h_head_idx] : -1;
                const int idx_lo = (l_tail > l_head) ? dq_min_idx[l_head_idx] : -1;
                if (idx_hi < 0 || idx_lo < 0) {
                    out_up_tm  [t * stride + s] = NAN;
                    out_down_tm[t * stride + s] = NAN;
                } else {
                    const int dist_hi = t - idx_hi;
                    const int dist_lo = t - idx_lo;
                    const float up = (dist_hi == 0) ? 100.0f
                                     : (dist_hi >= length ? 0.0f
                                     : fmaf(-(float)dist_hi, scale, 100.0f));
                    const float dn = (dist_lo == 0) ? 100.0f
                                     : (dist_lo >= length ? 0.0f
                                     : fmaf(-(float)dist_lo, scale, 100.0f));
                    out_up_tm  [t * stride + s] = up;
                    out_down_tm[t * stride + s] = dn;
                }
            }
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — aroon (output "first" = aroon UP, index 0 of
 * fallback_outputs &["first","second"], cuda_non_ma_generated.rs)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/aroon.rs:523 `aroon_scalar`.
 *
 * `aroon_scalar` has an all-finite FAST path and a NaN-tolerant SLOW path.
 * They are PROVABLY the same function on finite input: with every bar finite
 * the slow path's `invalid_count` is 0 at every bar, `have_extremes` is false
 * exactly once (at i == length), and the seed scan and the incremental
 * max/min updates are then identical statement for statement. Only the slow
 * path is implemented here, so there is one code path to be right about
 * instead of two to keep in step.
 *
 * `first_valid` is IGNORED: `aroon_scalar` starts at absolute index `length`
 * and never reads `first`. Declared F64FirstValidRule::Ignored.
 *
 * `pair_is_finite` tests the EXPONENT FIELD, so it rejects infinities as well
 * as NaN. `isfinite()` has the same meaning; a bare `!isnan()` would not.
 *
 * `aroon_percent` ends in `.max(0.0)` — f64::max, so a NaN distance would
 * yield 0.0. It cannot be NaN here (it is an integer distance) but fmax is
 * used anyway so the semantics do not depend on that staying true.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void aroon_neo_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int series_len,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int length = periods[combo];

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (length < 1 || length > len) return;      // aroon_scalar asserts this range

    const double scale_100 = 100.0 / (double)length;
    const int window = length + 1;

    int invalid_count = 0;
    bool have_extremes = false;
    int maxi = 0, mini = 0;
    double mx = 0.0, mn = 0.0;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        if (!(isfinite(h) && isfinite(l))) invalid_count += 1;
        if (i >= window) {
            const int leave = i - window;
            if (!(isfinite(high[leave]) && isfinite(low[leave]))) invalid_count -= 1;
        }

        if (i < length) continue;

        if (invalid_count != 0) {
            o[i] = NEO_F64_NAN;
            have_extremes = false;
            continue;
        }

        const int start = i - length;

        if (!have_extremes) {
            maxi = start; mini = start;
            mx = high[start]; mn = low[start];
            for (int j = start + 1; j <= i; ++j) {
                const double hv = high[j];
                if (hv > mx) { mx = hv; maxi = j; }
                const double lv = low[j];
                if (lv < mn) { mn = lv; mini = j; }
            }
            have_extremes = true;
        } else {
            if (maxi < start) {
                maxi = start; mx = high[maxi];
                for (int j = start + 1; j <= i; ++j) {
                    const double hv = high[j];
                    if (hv > mx) { mx = hv; maxi = j; }
                }
            } else if (h > mx) { maxi = i; mx = h; }

            if (mini < start) {
                mini = start; mn = low[mini];
                for (int j = start + 1; j <= i; ++j) {
                    const double lv = low[j];
                    if (lv < mn) { mn = lv; mini = j; }
                }
            } else if (l < mn) { mini = i; mn = l; }
        }

        const double dist_hi = (double)(i - maxi);
        o[i] = fmax(100.0 - dist_hi * scale_100, 0.0);     // aroon UP
    }
}
