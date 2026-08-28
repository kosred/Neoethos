#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void sama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ lengths,
                    const float* __restrict__ min_alphas,
                    const float* __restrict__ maj_alphas,
                    const int* __restrict__ first_valids,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const float min_alpha = min_alphas[combo];
    const float maj_alpha = maj_alphas[combo];
    const int first_valid = first_valids[combo];

    if (length < 0 || first_valid >= series_len || series_len <= 0) {
        return;
    }

    const int row_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float prev = NAN;

    for (int t = first_valid; t < series_len; ++t) {
        const float price = prices[t];
        if (!isfinite(price)) {
            out[row_offset + t] = NAN;
            continue;
        }

        int start = t - length;
        if (start < 0) {
            start = 0;
        }
        float hh = -CUDART_INF_F;
        float ll = CUDART_INF_F;
        for (int j = start; j <= t; ++j) {
            const float v = prices[j];
            if (!isfinite(v)) {
                continue;
            }
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        float mult = 0.0f;
        if (hh != ll) {
            const float numer = fabsf(2.0f * price - ll - hh);
            const float denom = hh - ll;
            if (denom != 0.0f) {
                mult = numer / denom;
            }
        }
        float alpha = (mult * (min_alpha - maj_alpha) + maj_alpha);
        alpha = alpha * alpha;

        if (!isfinite(prev)) {
            prev = price;
        } else {
            prev = __fmaf_rn(price - prev, alpha, prev);
        }

        out[row_offset + t] = prev;
    }
}

extern "C" __global__
void sama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    int length,
                                    float min_alpha,
                                    float maj_alpha,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (length < 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float prev = NAN;

    for (int t = first_valid; t < series_len; ++t) {
        const int offset = t * stride + series_idx;
        const float price = prices_tm[offset];
        if (!isfinite(price)) {
            out_tm[offset] = NAN;
            continue;
        }

        int start = t - length;
        if (start < 0) {
            start = 0;
        }

        float hh = -CUDART_INF_F;
        float ll = CUDART_INF_F;
        for (int j = start; j <= t; ++j) {
            const float v = prices_tm[j * stride + series_idx];
            if (!isfinite(v)) {
                continue;
            }
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        float mult = 0.0f;
        if (hh != ll) {
            const float numer = fabsf(2.0f * price - ll - hh);
            const float denom = hh - ll;
            if (denom != 0.0f) {
                mult = numer / denom;
            }
        }
        float alpha = (mult * (min_alpha - maj_alpha) + maj_alpha);
        alpha = alpha * alpha;

        if (!isfinite(prev)) {
            prev = price;
        } else {
            prev = __fmaf_rn(price - prev, alpha, prev);
        }

        out_tm[offset] = prev;
    }
}


#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#if !defined(CUDA_HAS_LDG_WRAPPER)
#define CUDA_HAS_LDG_WRAPPER

static __device__ __forceinline__ float ldgf(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
#endif


static __device__ __forceinline__ int clamp_start(int t, int length) {
    int s = t - length;
    return s < 0 ? 0 : s;
}


static __device__ __forceinline__
void pop_outdated_front(int*& q, int& front, int& size, const int cap, int start) {
    while (size > 0) {
        int idx = q[front];
        if (idx >= start) break;
        front = (front + 1);
        if (front == cap) front = 0;
        --size;
    }
}


static __device__ __forceinline__
void push_max_idx(const float* base, int*& q, int& back, int& size, const int cap, int k) {
    float vk = ldgf(base + k);
    if (!isfinite(vk)) return;
    while (size > 0) {
        int back_pos = (back == 0 ? cap - 1 : back - 1);
        float vb = ldgf(base + q[back_pos]);

        if (vb > vk) break;
        back = back_pos;
        --size;
    }
    q[back] = k;
    back = (back + 1);
    if (back == cap) back = 0;
    ++size;
}


static __device__ __forceinline__
void push_min_idx(const float* base, int*& q, int& back, int& size, const int cap, int k) {
    float vk = ldgf(base + k);
    if (!isfinite(vk)) return;
    while (size > 0) {
        int back_pos = (back == 0 ? cap - 1 : back - 1);
        float vb = ldgf(base + q[back_pos]);

        if (vb < vk) break;
        back = back_pos;
        --size;
    }
    q[back] = k;
    back = (back + 1);
    if (back == cap) back = 0;
    ++size;
}


extern "C" __global__
void sama_batch_f32_opt(const float* __restrict__ prices,
                        const int*   __restrict__ lengths,
                        const float* __restrict__ min_alphas,
                        const float* __restrict__ maj_alphas,
                        const int*   __restrict__ first_valids,
                        int series_len,
                        int n_combos,
                        int max_window,
                        float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int length      = lengths[combo];
    const float min_alpha = min_alphas[combo];
    const float maj_alpha = maj_alphas[combo];
    const int first_valid = first_valids[combo];

    if (length < 0 || first_valid >= series_len || series_len <= 0) return;

    const int row_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;


    const float dalpha = min_alpha - maj_alpha;

    float prev = NAN;


    const bool use_deque = (max_window >= length);
    if (!use_deque) {
        for (int t = first_valid; t < series_len; ++t) {
            const float price_t = ldgf(prices + t);
            if (!isfinite(price_t)) {
                out[row_offset + t] = NAN;
                continue;
            }
            const int start = clamp_start(t, length);

            float hh = -CUDART_INF_F, ll = CUDART_INF_F;
            bool any = false;
            #pragma unroll 1
            for (int j = start; j <= t; ++j) {
                const float v = ldgf(prices + j);
                if (!isfinite(v)) continue;
                any = true;
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            float mult = 0.0f;
            if (any) {
                const float denom = hh - ll;
                if (denom != 0.0f) {
                    const float numer = fabsf(2.0f * price_t - ll - hh);
                    mult = numer / denom;
                }
            }
            float alpha = __fmaf_rn(mult, dalpha, maj_alpha);
            alpha = alpha * alpha;

            prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
            out[row_offset + t] = prev;
        }
        return;
    }


    extern __shared__ int shmem[];
    const int cap = max_window + 1;
    int* dq_max = shmem;
    int* dq_min = shmem + cap;

    int fmax = 0, bmax = 0, szmax = 0;
    int fmin = 0, bmin = 0, szmin = 0;

    for (int t = first_valid; t < series_len; ++t) {
        const int start = clamp_start(t, length);
        const float price_t = ldgf(prices + t);


        pop_outdated_front(dq_max, fmax, szmax, cap, start);
        pop_outdated_front(dq_min, fmin, szmin, cap, start);

        if (!isfinite(price_t)) {
            out[row_offset + t] = NAN;
            continue;
        }


        while (szmax > 0) {
            int back_pos = (bmax == 0 ? cap - 1 : bmax - 1);
            float vb = ldgf(prices + dq_max[back_pos]);
            if (vb > price_t) break;
            bmax = back_pos; --szmax;
        }
        dq_max[bmax] = t; bmax = (bmax + 1 == cap ? 0 : bmax + 1); ++szmax;

        while (szmin > 0) {
            int back_pos = (bmin == 0 ? cap - 1 : bmin - 1);
            float vb = ldgf(prices + dq_min[back_pos]);
            if (vb < price_t) break;
            bmin = back_pos; --szmin;
        }
        dq_min[bmin] = t; bmin = (bmin + 1 == cap ? 0 : bmin + 1); ++szmin;


        const float hh = ldgf(prices + dq_max[fmax]);
        const float ll = ldgf(prices + dq_min[fmin]);
        const float denom = hh - ll;
        float mult = 0.0f;
        if (denom != 0.0f) {
            const float numer = fabsf(2.0f * price_t - ll - hh);
            mult = numer / denom;
        }

        float alpha = __fmaf_rn(mult, (min_alpha - maj_alpha), maj_alpha);
        alpha = alpha * alpha;

        prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
        out[row_offset + t] = prev;
    }
}


extern "C" __global__
void sama_many_series_one_param_f32_opt(const float* __restrict__ prices_tm,
                                        const int*   __restrict__ first_valids,
                                        int length,
                                        float min_alpha,
                                        float maj_alpha,
                                        int num_series,
                                        int series_len,
                                        int max_window,
                                        float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;
    if (length < 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float dalpha = min_alpha - maj_alpha;

    float prev = NAN;


    const bool use_deque = (max_window >= length);
    if (!use_deque) {
        for (int t = first_valid; t < series_len; ++t) {
            const int off = t * stride + series_idx;
            const float price_t = ldgf(prices_tm + off);
            if (!isfinite(price_t)) {
                out_tm[off] = NAN;
                continue;
            }
            const int start = clamp_start(t, length);
            float hh = -CUDART_INF_F, ll = CUDART_INF_F;
            bool any = false;
            #pragma unroll 1
            for (int j = start; j <= t; ++j) {
                const float v = ldgf(prices_tm + j * stride + series_idx);
                if (!isfinite(v)) continue;
                any = true;
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            float mult = 0.0f;
            if (any) {
                const float denom = hh - ll;
                if (denom != 0.0f) {
                    const float numer = fabsf(2.0f * price_t - ll - hh);
                    mult = numer / denom;
                }
            }
            float alpha = __fmaf_rn(mult, dalpha, maj_alpha);
            alpha = alpha * alpha;

            prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
            out_tm[off] = prev;
        }
        return;
    }


    extern __shared__ int shmem[];
    const int cap = max_window + 1;
    int* dq_max = shmem;
    int* dq_min = shmem + cap;

    int fmax = 0, bmax = 0, szmax = 0;
    int fmin = 0, bmin = 0, szmin = 0;

    auto load_tm = [&](int t)->float {
        return ldgf(prices_tm + t * stride + series_idx);
    };

    for (int t = first_valid; t < series_len; ++t) {
        const int start = clamp_start(t, length);
        const int off   = t * stride + series_idx;
        const float price_t = load_tm(t);


        pop_outdated_front(dq_max, fmax, szmax, cap, start);
        pop_outdated_front(dq_min, fmin, szmin, cap, start);

        if (!isfinite(price_t)) {
            out_tm[off] = NAN;
            continue;
        }


        while (szmax > 0) {
            int back_pos = (bmax == 0 ? cap - 1 : bmax - 1);
            float vb = load_tm(dq_max[back_pos]);
            if (vb > price_t) break;
            bmax = back_pos; --szmax;
        }
        dq_max[bmax] = t; bmax = (bmax + 1 == cap ? 0 : bmax + 1); ++szmax;

        while (szmin > 0) {
            int back_pos = (bmin == 0 ? cap - 1 : bmin - 1);
            float vb = load_tm(dq_min[back_pos]);
            if (vb < price_t) break;
            bmin = back_pos; --szmin;
        }
        dq_min[bmin] = t; bmin = (bmin + 1 == cap ? 0 : bmin + 1); ++szmin;

        const float hh = load_tm(dq_max[fmax]);
        const float ll = load_tm(dq_min[fmin]);
        const float denom = hh - ll;
        float mult = 0.0f;
        if (denom != 0.0f) {
            const float numer = fabsf(2.0f * price_t - ll - hh);
            mult = numer / denom;
        }

        float alpha = __fmaf_rn(mult, (min_alpha - maj_alpha), maj_alpha);
        alpha = alpha * alpha;

        prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
        out_tm[off] = prev;
    }
}


// ===========================================================================
// S2 f64 LANE — sama  (slope adaptive moving average)
// ===========================================================================
// Reference: src/indicators/moving_averages/sama.rs
//   `sama_prepare`      (:305) — first_valid, `length + 1 > n` refusal
//   `sama_with_kernel`  (:253) — alloc_with_nan_prefix(len, FIRST), i.e. the
//                                 NaN prefix is `first_valid`, NOT
//                                 `first + length - 1`
//   `sama_scalar`       (:513) — the general path this kernel mirrors
//   `sama_scalar_default_200_14_6` (:397) — the same arithmetic with the
//                                 defaults folded in; it is a specialisation,
//                                 not a different algorithm, so ONE kernel
//                                 serves both and no special case is written
//   Batch route: `ma_batch.rs:1895` sweeps `length`; `maj_length` and
//   `min_length` stay at `SamaBatchRange::default()` = 14 (:127) and 6 (:132).
//
// WARMUP IS UNUSUAL AND IS NOT A TYPO. Every other moving average in this
// shard NaNs `first + period - 1` bars. `sama` NaNs only `first_valid` bars
// and emits a value from the very first valid bar (`sama_val = p` on the first
// non-NaN). Getting this wrong would shift the series by `length - 1` bars —
// 199 at the default — which no tolerance would forgive.
//
// PER-BAR NaN PASS-THROUGH. `if p.is_nan() { out[i] = NAN; continue; }` — a
// NaN bar does NOT update the deques, does NOT update `sama_val`, and does NOT
// end the series. Reproduced exactly; an f32 kernel that let the NaN into the
// recurrence would poison every later bar, which is the failure rule 4 names.
//
// `mult` GATES ON `denom > 0.0`, NOT `!= 0.0`. A negative denominator cannot
// happen (hh >= ll by construction) but a NaN one can, and `NaN > 0.0` is
// false on both sides, so the guard is faithful as an if.
//
// ROUNDINGS.
//   c     = (p + p) - (hh + ll)            -> add + add + sub   (3)
//   mult  = fabs(c) / denom                -> div               (1)
//   a     = mult.mul_add(delta, maj_alpha) -> fma               (1)
//   alpha = a * a                          -> mul               (1)
//   sama  = (p - sama).mul_add(alpha, sama)-> sub + fma         (2)
// `(p + p)` is NOT `2.0 * p`; identical in binary floating point, but written
// as the CPU writes it so no one has to re-derive that.
// ===========================================================================

#define SAMA_MAX_LENGTH 512

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_sama_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int length = periods[r];
    const int maj_length = 14;   // SamaParams::get_maj_length -> unwrap_or(14)
    const int min_length = 6;    // SamaParams::get_min_length -> unwrap_or(6)

    const bool declined =
        (n <= 0) ||
        (length == 0) || ((length + 1) > n) || (length > SAMA_MAX_LENGTH) ||
        (maj_length == 0) || (min_length == 0) ||
        (first_valid < 0) || (first_valid >= n);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    // alloc_with_nan_prefix(len, first) — only `first_valid` bars are NaN.
    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();

    const double maj_alpha = 2.0 / ((double)maj_length + 1.0);
    const double min_alpha = 2.0 / ((double)min_length + 1.0);
    const double delta = min_alpha - maj_alpha;

    const int cap = length + 1;
    int max_idx[SAMA_MAX_LENGTH + 1];
    int min_idx[SAMA_MAX_LENGTH + 1];
    int max_head = 0, min_head = 0;
    int max_len = 0, min_len = 0;

    double sama_val = neo_s2_qnan();
    bool have_sama = false;

    for (int i = first_valid; i < n; ++i) {
        const double p = prices[i];
        if (isnan(p)) {
            row[i] = neo_s2_qnan();
            continue;
        }

        const int wstart = (i > length) ? (i - length) : 0;   // saturating_sub

        while (max_len > 0) {
            const int idx = max_idx[max_head];
            if (idx >= wstart) break;
            max_head += 1;
            if (max_head == cap) max_head = 0;
            max_len -= 1;
        }
        while (min_len > 0) {
            const int idx = min_idx[min_head];
            if (idx >= wstart) break;
            min_head += 1;
            if (min_head == cap) min_head = 0;
            min_len -= 1;
        }

        while (max_len > 0) {
            int last_pos = max_head + max_len - 1;
            if (last_pos >= cap) last_pos -= cap;
            if (prices[max_idx[last_pos]] <= p) max_len -= 1;
            else break;
        }
        int ins_pos_max = max_head + max_len;
        if (ins_pos_max >= cap) ins_pos_max -= cap;
        max_idx[ins_pos_max] = i;
        max_len += 1;

        while (min_len > 0) {
            int last_pos = min_head + min_len - 1;
            if (last_pos >= cap) last_pos -= cap;
            if (prices[min_idx[last_pos]] >= p) min_len -= 1;
            else break;
        }
        int ins_pos_min = min_head + min_len;
        if (ins_pos_min >= cap) ins_pos_min -= cap;
        min_idx[ins_pos_min] = i;
        min_len += 1;

        const double hh = prices[max_idx[max_head]];
        const double ll = prices[min_idx[min_head]];

        const double denom = hh - ll;
        const double c = (p + p) - (hh + ll);
        const double mult = (denom > 0.0) ? (fabs(c) / denom) : 0.0;

        const double a = fma(mult, delta, maj_alpha);
        const double alpha = a * a;

        if (!have_sama) {
            sama_val = p;
            have_sama = true;
        } else {
            const double diff = p - sama_val;
            sama_val = fma(diff, alpha, sama_val);
        }

        row[i] = sama_val;
    }
}
