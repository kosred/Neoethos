#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef KDJ_QNAN
#define KDJ_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {

    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}


__device__ __forceinline__ float stoch_from_tables_fk(
    int t, int fk, int k_log2, int offset, int level_base,
    const float* __restrict__ close,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum
){
    const int start = t - fk + 1;

    if (nan_psum[t + 1] - nan_psum[start]) return KDJ_QNAN;

    const int idx_a = level_base + start;
    const int idx_b = level_base + (t + 1 - offset);
    const float h = fmaxf(st_max[idx_a], st_max[idx_b]);
    const float l = fminf(st_min[idx_a], st_min[idx_b]);
    const float c = close[t];
    const float den = h - l;

    if (!(h==h) || !(l==l) || !(c==c) || den <= 0.0f) return KDJ_QNAN;
    return (c - l) * (100.0f / den);
}

extern "C" __global__ void kdj_batch_f32(

    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    const int*   __restrict__ fast_k_arr,
    const int*   __restrict__ slow_k_arr,
    const int*   __restrict__ slow_d_arr,
    const int*   __restrict__ k_ma_types,
    const int*   __restrict__ d_ma_types,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,

    float* __restrict__ out_k,
    float* __restrict__ out_d,
    float* __restrict__ out_j
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;


    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_k[base + i] = KDJ_QNAN;
            out_d[base + i] = KDJ_QNAN;
            out_j[base + i] = KDJ_QNAN;
        }
        return;
    }

    const int fk = fast_k_arr[combo];
    const int sk = slow_k_arr[combo];
    const int sd = slow_d_arr[combo];
    if (UNLIKELY(fk <= 0 || sk <= 0 || sd <= 0 || level_count <= 0)) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_k[base + i] = KDJ_QNAN;
            out_d[base + i] = KDJ_QNAN;
            out_j[base + i] = KDJ_QNAN;
        }
        return;
    }

    const int stoch_warm = first_valid + fk - 1;
    if (UNLIKELY(stoch_warm >= series_len)) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_k[base + i] = KDJ_QNAN;
            out_d[base + i] = KDJ_QNAN;
            out_j[base + i] = KDJ_QNAN;
        }
        return;
    }

    const int k_ma = k_ma_types[combo];
    const int d_ma = d_ma_types[combo];


    const int k_log2 = log2_tbl[fk];
    if (UNLIKELY(k_log2 < 0 || k_log2 >= level_count)) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_k[base + i] = KDJ_QNAN;
            out_d[base + i] = KDJ_QNAN;
            out_j[base + i] = KDJ_QNAN;
        }
        return;
    }
    const int offset     = 1 << k_log2;
    const int level_base = level_offsets[k_log2];


    const int lane = threadIdx.x & 31;
    if ((threadIdx.x >> 5) == 0) {

        for (int t = lane; t < stoch_warm; t += 32) {
            out_j[base + t] = KDJ_QNAN;
        }
        for (int t = stoch_warm + lane; t < series_len; t += 32) {
            out_j[base + t] = stoch_from_tables_fk(
                t, fk, k_log2, offset, level_base, close, st_max, st_min, nan_psum
            );
        }
    }
    __syncthreads();

    const int k_warm = stoch_warm + sk - 1;
    const int d_warm = k_warm     + sd - 1;


    if (k_ma == 0 && d_ma == 0) {
        if (UNLIKELY(k_warm >= series_len)) {
            for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
                out_k[base + i] = KDJ_QNAN;
                out_d[base + i] = KDJ_QNAN;
                out_j[base + i] = KDJ_QNAN;
            }
            return;
        }


        for (int i = threadIdx.x; i < k_warm; i += blockDim.x) out_k[base + i] = KDJ_QNAN;
        for (int i = threadIdx.x; i < d_warm && i < series_len; i += blockDim.x) out_d[base + i] = KDJ_QNAN;


        for (int t = k_warm + threadIdx.x; t < series_len; t += blockDim.x) {
            float sum = 0.0f;
            int cnt = 0;
            const int start = t - sk + 1;
            for (int ti = start; ti <= t; ++ti) {
                const float s = out_j[base + ti];
                if (s == s) { sum += s; ++cnt; }
            }
            out_k[base + t] = (cnt > 0) ? (sum / (float)cnt) : KDJ_QNAN;
        }
        __syncthreads();


        for (int i = threadIdx.x; i < d_warm && i < series_len; i += blockDim.x) out_j[base + i] = KDJ_QNAN;


        if (d_warm < series_len) {
            for (int t = d_warm + threadIdx.x; t < series_len; t += blockDim.x) {
                float sum = 0.0f;
                int cnt = 0;
                const int start = t - sd + 1;
                for (int ti = start; ti <= t; ++ti) {
                    const float kv = out_k[base + ti];
                    if (kv == kv) { sum += kv; ++cnt; }
                }
                const float dv = (cnt > 0) ? (sum / (float)cnt) : KDJ_QNAN;
                out_d[base + t] = dv;

                const float kv = out_k[base + t];
                out_j[base + t] = (kv == kv && dv == dv) ? (3.0f * kv - 2.0f * dv) : KDJ_QNAN;
            }
        }
        return;
    }


    if (threadIdx.x != 0) return;


    if (k_ma == 1 && d_ma == 1) {
        if (k_warm >= series_len) return;

        const float alpha_k = 2.0f / (float(sk) + 1.0f);
        const float one_mk  = 1.0f - alpha_k;
        const float alpha_d = 2.0f / (float(sd) + 1.0f);
        const float one_md  = 1.0f - alpha_d;


        float sum0 = 0.0f, c0 = 0.0f; int cnt0 = 0;
        {
            const int start = k_warm - sk + 1;
            for (int ti = start; ti <= k_warm; ++ti) {
                const float s = out_j[base + ti];
                if (s == s) { kahan_add(s, sum0, c0); ++cnt0; }
            }
        }
        float ema_k = (cnt0 > 0) ? (sum0 / (float)cnt0) : KDJ_QNAN;
        out_k[base + k_warm] = ema_k;


        float sum_d = (ema_k == ema_k) ? ema_k : 0.0f;
        int   cnt_d = (ema_k == ema_k) ? 1 : 0;


        for (int t = k_warm + 1; t <= d_warm && t < series_len; ++t) {
            const float s = out_j[base + t];
            if (s == s && ema_k == ema_k) {
                ema_k = fmaf(alpha_k, s, one_mk * ema_k);
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = s;
            }
            out_k[base + t] = ema_k;
            if (ema_k == ema_k) { sum_d += ema_k; ++cnt_d; }
        }


        float ema_d = (d_warm < series_len && cnt_d > 0) ? (sum_d / (float)cnt_d) : KDJ_QNAN;
        if (d_warm < series_len) {
            out_d[base + d_warm] = ema_d;
            const float kv = out_k[base + d_warm];
            out_j[base + d_warm] = (kv == kv && ema_d == ema_d) ? (3.0f * kv - 2.0f * ema_d) : KDJ_QNAN;
        }


        for (int t = d_warm + 1; t < series_len; ++t) {
            const float s = out_j[base + t];
            if (s == s && ema_k == ema_k) {
                ema_k = fmaf(alpha_k, s, one_mk * ema_k);
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = s;
            }
            out_k[base + t] = ema_k;

            if (ema_k == ema_k && ema_d == ema_d) {
                ema_d = fmaf(alpha_d, ema_k, one_md * ema_d);
            } else if (ema_k == ema_k && !(ema_d == ema_d)) {
                ema_d = ema_k;
            }
            out_d[base + t] = ema_d;
            out_j[base + t] = (ema_k == ema_k && ema_d == ema_d) ? (3.0f * ema_k - 2.0f * ema_d) : KDJ_QNAN;
        }
        return;
    }


}


extern "C" __global__ void kdj_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int fast_k,
    int slow_k,
    int slow_d,
    int k_ma_type,
    int d_ma_type,
    float* __restrict__ k_out_tm,
    float* __restrict__ d_out_tm,
    float* __restrict__ j_out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];

    for (int t = 0; t < series_len; ++t) {
        *(k_out_tm + (size_t)t * num_series + series) = KDJ_QNAN;
        *(d_out_tm + (size_t)t * num_series + series) = KDJ_QNAN;
        *(j_out_tm + (size_t)t * num_series + series) = KDJ_QNAN;
    }
    if (UNLIKELY(fv < 0 || fv >= series_len || fast_k <= 0 || slow_k <= 0 || slow_d <= 0)) return;

    const int stoch_warm = fv + fast_k - 1;
    const int k_warm     = stoch_warm + slow_k - 1;
    const int d_warm     = k_warm + slow_d - 1;
    if (UNLIKELY(stoch_warm >= series_len)) return;

    auto load_tm = [num_series, series](const float* base, int t)->float {
        return *(base + (size_t)t * num_series + series);
    };

    auto stoch_naive = [&](int t)->float {
    const int start = t - fast_k + 1;
    float h = -INFINITY, l = INFINITY;


        size_t idx = (size_t)start * (size_t)num_series + (size_t)series;
        const float* __restrict__ ph = high_tm;
        const float* __restrict__ pl = low_tm;

        for (int i = start; i <= t; ++i, idx += (size_t)num_series) {
            const float hi = ph[idx];
            const float lo = pl[idx];
            if (!(hi==hi) || !(lo==lo)) return KDJ_QNAN;
            h = fmaxf(h, hi);
            l = fminf(l, lo);
        }
        const float c = *(close_tm + (size_t)t * (size_t)num_series + series);
        if (!(c==c)) return KDJ_QNAN;
        const float den = h - l;
        return (den > 0.0f) ? (c - l) * (100.0f / den) : KDJ_QNAN;
    };

    if (k_ma_type == 0 && d_ma_type == 0) {
        for (int t = k_warm; t < series_len; ++t) {
            double sum_k = 0.0; int cnt_k = 0;
            const int k_start = t - slow_k + 1;
            for (int u = 0; u < slow_k; ++u) {
                const int ti = k_start + u;
                float s = stoch_naive(ti);
                if (s == s) { sum_k += (double)s; cnt_k += 1; }
            }
            const float kv = (cnt_k > 0) ? (float)(sum_k / (double)cnt_k) : KDJ_QNAN;
            *(k_out_tm + (size_t)t * num_series + series) = kv;

            if (t >= d_warm) {
                double sum_d = 0.0; int cnt_d = 0;
                const int d_start = t - slow_d + 1;
                for (int v = 0; v < slow_d; ++v) {
                    const int tj = d_start + v;
                    float kk = *(k_out_tm + (size_t)tj * num_series + series);
                    if (kk == kk) { sum_d += (double)kk; cnt_d += 1; }
                }
                const float dv = (cnt_d > 0) ? (float)(sum_d / (double)cnt_d) : KDJ_QNAN;
                *(d_out_tm + (size_t)t * num_series + series) = dv;
                *(j_out_tm + (size_t)t * num_series + series) = (kv == kv && dv == dv) ? (3.0f * kv - 2.0f * dv) : KDJ_QNAN;
            }
        }
        return;
    }

    if (k_ma_type == 1 && d_ma_type == 1) {
        const double ak = 2.0 / (double(slow_k) + 1.0);
        const double ok = 1.0 - ak;
        const double ad = 2.0 / (double(slow_d) + 1.0);
        const double od = 1.0 - ad;
        double ema_k = NAN, ema_d = NAN;
        double sum_init_k = 0.0; int cnt_init_k = 0;
        double sum_init_d = 0.0; int cnt_init_d = 0;

        if (k_warm < series_len) {
            const int ks = k_warm - slow_k + 1;
            for (int ti = ks; ti <= k_warm; ++ti) {
                float s = stoch_naive(ti);
                if (s == s) { sum_init_k += s; cnt_init_k += 1; }
            }
            ema_k = (cnt_init_k > 0) ? (sum_init_k / (double)cnt_init_k) : NAN;
            *(k_out_tm + (size_t)k_warm * num_series + series) = (float)ema_k;
            if (ema_k == ema_k) { sum_init_d += ema_k; cnt_init_d += 1; }
        }

        for (int t = k_warm + 1; t <= d_warm && t < series_len; ++t) {
            float s = stoch_naive(t);
            if (s == s && ema_k == ema_k) {
                ema_k = ak * (double)s + ok * ema_k;
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = (double)s;
            }
            *(k_out_tm + (size_t)t * num_series + series) = (float)ema_k;
            if (ema_k == ema_k) { sum_init_d += ema_k; cnt_init_d += 1; }
        }

        if (d_warm < series_len) {
            ema_d = (cnt_init_d > 0) ? (sum_init_d / (double)cnt_init_d) : NAN;
            *(d_out_tm + (size_t)d_warm * num_series + series) = (float)ema_d;
            const float kv = *(k_out_tm + (size_t)d_warm * num_series + series);
            *(j_out_tm + (size_t)d_warm * num_series + series) = (kv == kv && ema_d == ema_d) ? (3.0f * kv - 2.0f * (float)ema_d) : KDJ_QNAN;
        }

        for (int t = d_warm + 1; t < series_len; ++t) {
            float s = stoch_naive(t);
            if (s == s && ema_k == ema_k) {
                ema_k = ak * (double)s + ok * ema_k;
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = (double)s;
            }
            *(k_out_tm + (size_t)t * num_series + series) = (float)ema_k;

            if (ema_k == ema_k && ema_d == ema_d) {
                ema_d = ad * ema_k + od * ema_d;
            } else if (ema_k == ema_k && !(ema_d == ema_d)) {
                ema_d = ema_k;
            }
            *(d_out_tm + (size_t)t * num_series + series) = (float)ema_d;
            *(j_out_tm + (size_t)t * num_series + series) = (ema_k == ema_k && ema_d == ema_d) ? (3.0f * (float)ema_k - 2.0f * (float)ema_d) : KDJ_QNAN;
        }
        return;
    }
}


// ===========================================================================
// S3 f64 LANE — kdj (K line)
// ===========================================================================
// Reference: src/indicators/kdj.rs
//   kdj_with_kernel (:161)             — first_valid + the Err branches
//   kdj_compute_into_scalar (:449)     — the ma_type / period fork
//   kdj_default_sma_9_3_3_into (:857)  — the path this lane takes
//
// WHICH OUTPUT. kdj is multi-output (k / d / j) and the lane emits ONE matrix.
// compute_kdj_batch maps output_id "value" — what the f64 lane asks for — to
// `out.k`. So this kernel is the K line. d and j are separate entry points'
// work, not a silent substitution.
//
// WHICH PATH. The batch defaults are fast_k 9, slow_k 3, slow_d 3, both MA
// types "sma", which is EXACTLY the fast-path guard at :476. So
// kdj_default_sma_9_3_3_into is what runs, its ring sizes are compile-time 3
// and its monotone deques are compile-time 10 — all of which fit in registers,
// so this kernel needs no device scratch and no dynamic array.
//
// PERIOD-INVARIANT. No `period` parameter is read; a period sweep produces
// identical rows.
//
// FIRST-VALID is the simultaneous rule (kdj.rs:202-206): the first i where
// high, low AND close are all non-NaN → AllInputsNonNan.
//
// THE MONOTONE DEQUES ARE TRANSCRIBED, NOT REPLACED BY fmax/fmin.
// The CPU pops while max_val[back] <= hi and while min_val[back] >= lo. Under
// NaN those comparisons are false, so a NaN high is never evicted and can reach
// the head, making `hh` NaN and the bar's stochastic NaN. An fmax/fmin rolling
// extreme would return the non-NaN operand and emit a NUMBER where the
// reference emits NaN. The deque is reproduced literally in a 10-slot register
// array — the same size the CPU uses.
//
// THE MEANS COUNT NON-NaN MEMBERS. sum_k/cnt_k and sum_d/cnt_d skip NaN entries
// on both insert and evict (:961-969), so the K line is the mean of however
// many of the last 3 stochastics were finite — NOT a fixed /3. Getting this
// wrong produces a plausible-looking series that is wrong on exactly the bars
// that follow a flat window.
//
// NO EPSILON. denom == 0.0 is tested exactly and separately from
// denom.is_nan() (:953). No tolerance exists in the reference and none is
// invented; an f32 port is tempted to add one because f32 collapses hh-ll to
// zero far more often, and that temptation is the bug.
//
// One thread per column.
// ===========================================================================

#define NEO_S3_KDJ_FAST_K 9
#define NEO_S3_KDJ_SLOW_K 3
#define NEO_S3_KDJ_SLOW_D 3
#define NEO_S3_KDJ_CAP    10

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_kdj_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;   // PERIOD-INVARIANT — see the header.

    double* __restrict__ row = out + (size_t)r * (size_t)n;

    const int fast_k = NEO_S3_KDJ_FAST_K;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (fast_k > n) ||
        ((n - first_valid) < fast_k);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int stoch_warm = first_valid + 8;    // first + fast_k - 1
    const int k_warm     = stoch_warm + 2;     // + slow_k - 1

    for (int i = 0; i < k_warm && i < n; ++i) row[i] = neo_s3_qnan();

    int    max_idx[NEO_S3_KDJ_CAP];
    double max_val[NEO_S3_KDJ_CAP];
    int    min_idx[NEO_S3_KDJ_CAP];
    double min_val[NEO_S3_KDJ_CAP];
    int max_head = 0, max_tail = 0, max_cnt = 0;
    int min_head = 0, min_tail = 0, min_cnt = 0;

    double stoch_ring[NEO_S3_KDJ_SLOW_K];
    for (int q = 0; q < NEO_S3_KDJ_SLOW_K; ++q) stoch_ring[q] = neo_s3_qnan();
    double sum_k = 0.0;
    int cnt_k = 0;

    int pos_k = stoch_warm % NEO_S3_KDJ_SLOW_K;

    for (int i = first_valid; i < n; ++i) {
        const double hi = high[i];
        while (max_cnt > 0) {
            const int back = (max_tail == 0) ? (NEO_S3_KDJ_CAP - 1) : (max_tail - 1);
            if (max_val[back] <= hi) { max_tail = back; max_cnt -= 1; }
            else break;
        }
        max_val[max_tail] = hi;
        max_idx[max_tail] = i;
        max_tail += 1; if (max_tail == NEO_S3_KDJ_CAP) max_tail = 0;
        max_cnt += 1;
        while (max_cnt > 0 && max_idx[max_head] + 9 <= i) {
            max_head += 1; if (max_head == NEO_S3_KDJ_CAP) max_head = 0;
            max_cnt -= 1;
        }

        const double lo = low[i];
        while (min_cnt > 0) {
            const int back = (min_tail == 0) ? (NEO_S3_KDJ_CAP - 1) : (min_tail - 1);
            if (min_val[back] >= lo) { min_tail = back; min_cnt -= 1; }
            else break;
        }
        min_val[min_tail] = lo;
        min_idx[min_tail] = i;
        min_tail += 1; if (min_tail == NEO_S3_KDJ_CAP) min_tail = 0;
        min_cnt += 1;
        while (min_cnt > 0 && min_idx[min_head] + 9 <= i) {
            min_head += 1; if (min_head == NEO_S3_KDJ_CAP) min_head = 0;
            min_cnt -= 1;
        }

        if (i < stoch_warm) continue;

        const double hh = max_val[max_head];
        const double ll = min_val[min_head];
        const double denom = hh - ll;
        double stoch_i;
        if (denom == 0.0 || isnan(denom)) {
            stoch_i = neo_s3_qnan();
        } else {
            const double c = close[i];
            stoch_i = 100.0 * ((c - ll) / denom);
        }

        const double old_st = stoch_ring[pos_k];
        if (!isnan(old_st)) { sum_k -= old_st; cnt_k -= 1; }
        stoch_ring[pos_k] = stoch_i;
        if (!isnan(stoch_i)) { sum_k += stoch_i; cnt_k += 1; }
        pos_k += 1; if (pos_k == NEO_S3_KDJ_SLOW_K) pos_k = 0;

        if (i >= k_warm) {
            row[i] = (cnt_k > 0) ? (sum_k / (double)cnt_k) : neo_s3_qnan();
        }
    }
}
