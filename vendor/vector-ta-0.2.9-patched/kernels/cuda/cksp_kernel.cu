#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


__device__ __forceinline__ int rb_inc(int idx, int cap) { return (idx + 1) >= cap ? 0 : idx + 1; }
__device__ __forceinline__ int rb_dec(int idx, int cap) { return (idx == 0) ? (cap - 1) : (idx - 1); }


extern "C" __global__
void tr_from_hlc_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     const float* __restrict__ close,
                     int series_len,
                     int first_valid,
                     float* __restrict__ tr_out) {
    if (series_len <= 0) return;
    const int start = (first_valid < 0 ? 0 : first_valid);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += blockDim.x * gridDim.x) {
        const float hi = high[i];
        const float lo = low[i];
        if (i <= start) {
            tr_out[i] = hi - lo;
        } else {
            const float pc = close[i - 1];
            const float hl = hi - lo;
            const float hc = fabsf(hi - pc);
            const float lc = fabsf(lo - pc);
            tr_out[i] = fmaxf(hl, fmaxf(hc, lc));
        }
    }
}


template<bool UsePrecomputedTR>
__device__ void cksp_core_row(const float* __restrict__ high,
                              const float* __restrict__ low,
                              const float* __restrict__ close,
                              const float* __restrict__ tr_opt,
                              int series_len,
                              int first_valid,
                              int p,
                              float x,
                              int q,
                              int cap_max,
                              float* __restrict__ out_long_row,
                              float* __restrict__ out_short_row) {
    if (series_len <= 0 || p <= 0 || q <= 0) return;
    const int start = (first_valid < 0 ? 0 : first_valid);
    if (start >= series_len) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    int*   h_idx  = (int*)shraw;
    int*   l_idx  = h_idx + cap_max;
    int*   ls_idx = l_idx + cap_max;
    int*   ss_idx = ls_idx + cap_max;
    float* ls_val = (float*)(ss_idx + cap_max);
    float* ss_val = ls_val + cap_max;

    const int cap  = q + 1;
    const int warm = start + p + q - 1;

    if (threadIdx.x != 0) return;


    const int warm_end = (warm < series_len) ? warm : series_len;
    for (int i = 0; i < warm_end; ++i) {
        out_long_row[i]  = CUDART_NAN_F;
        out_short_row[i] = CUDART_NAN_F;
    }

    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;


    float rma = 0.0f;
    const float alpha = 1.0f / (float)p;

    float sum_tr = 0.0f, c_tr = 0.0f;


    {
        const int i = start;
        const float hi = high[i];
        const float lo = low[i];
        const float tr = UsePrecomputedTR ? tr_opt[i] : (hi - lo);


        float y = tr - c_tr;
        float t = sum_tr + y;
        c_tr = (t - sum_tr) - y;
        sum_tr = t;
        if (p == 1) rma = tr;


        if (q > 1) {

            while (h_head != h_tail) {
                const int last = rb_dec(h_tail, cap);
                const int last_i = h_idx[last];
                if (high[last_i] <= hi) h_tail = last; else break;
            }
            int next_tail = rb_inc(h_tail, cap);
            if (next_tail == h_head) h_head = rb_inc(h_head, cap);
            h_idx[h_tail] = i; h_tail = next_tail;


            while (l_head != l_tail) {
                const int last = rb_dec(l_tail, cap);
                const int last_i = l_idx[last];
                if (low[last_i] >= lo) l_tail = last; else break;
            }
            next_tail = rb_inc(l_tail, cap);
            if (next_tail == l_head) l_head = rb_inc(l_head, cap);
            l_idx[l_tail] = i; l_tail = next_tail;
        }
    }


    if (q == 1) {

        int k = 1;
        float prev_close = close[start];
        for (int i = start + 1; i < series_len; ++i, ++k) {
            const float hi = high[i];
            const float lo = low[i];
            float tr = UsePrecomputedTR ? tr_opt[i]
                                        : fmaxf(hi - lo, fmaxf(fabsf(hi - prev_close), fabsf(lo - prev_close)));
            prev_close = close[i];

            if (k < p) {

                float y = tr - c_tr;
                float t = sum_tr + y;
                c_tr = (t - sum_tr) - y;
                sum_tr = t;
                if (k == p - 1) rma = sum_tr / (float)p;
            } else {
                rma = fmaf(alpha, tr - rma, rma);
            }

            if (i >= warm) {
                out_long_row[i]  = fmaf(-x, rma, hi);
                out_short_row[i] = fmaf(+x, rma, lo);
            }
        }
        return;
    }


    int k = 1;
    float prev_close = close[start];
    for (int i = start + 1; i < series_len; ++i, ++k) {
        const float hi = high[i];
        const float lo = low[i];
        float tr = UsePrecomputedTR ? tr_opt[i]
                                    : fmaxf(hi - lo, fmaxf(fabsf(hi - prev_close), fabsf(lo - prev_close)));
        prev_close = close[i];


        if (k < p) {
            float y = tr - c_tr;
            float t = sum_tr + y;
            c_tr = (t - sum_tr) - y;
            sum_tr = t;
            if (k == p - 1) rma = sum_tr / (float)p;
        } else {
            rma = fmaf(alpha, tr - rma, rma);
        }


        while (h_head != h_tail) {
            const int last = rb_dec(h_tail, cap);
            const int last_i = h_idx[last];
            if (high[last_i] <= hi) h_tail = last; else break;
        }
        int next_tail = rb_inc(h_tail, cap);
        if (next_tail == h_head) h_head = rb_inc(h_head, cap);
        h_idx[h_tail] = i; h_tail = next_tail;
        while (h_head != h_tail) {
            const int front_i = h_idx[h_head];
            if (front_i + q <= i) h_head = rb_inc(h_head, cap); else break;
        }
        const float mh = high[h_idx[h_head]];


        while (l_head != l_tail) {
            const int last = rb_dec(l_tail, cap);
            const int last_i = l_idx[last];
            if (low[last_i] >= lo) l_tail = last; else break;
        }
        next_tail = rb_inc(l_tail, cap);
        if (next_tail == l_head) l_head = rb_inc(l_head, cap);
        l_idx[l_tail] = i; l_tail = next_tail;
        while (l_head != l_tail) {
            const int front_i = l_idx[l_head];
            if (front_i + q <= i) l_head = rb_inc(l_head, cap); else break;
        }
        const float ml = low[l_idx[l_head]];

        if (i >= warm) {
            const float ls0 = fmaf(-x, rma, mh);
            const float ss0 = fmaf(+x, rma, ml);


            while (ls_head != ls_tail) {
                const int last = rb_dec(ls_tail, cap);
                if (ls_val[last] <= ls0) ls_tail = last; else break;
            }
            next_tail = rb_inc(ls_tail, cap);
            if (next_tail == ls_head) ls_head = rb_inc(ls_head, cap);
            ls_idx[ls_tail] = i; ls_val[ls_tail] = ls0; ls_tail = next_tail;
            while (ls_head != ls_tail) {
                const int front_i = ls_idx[ls_head];
                if (front_i + q <= i) ls_head = rb_inc(ls_head, cap); else break;
            }
            out_long_row[i] = ls_val[ls_head];


            while (ss_head != ss_tail) {
                const int last = rb_dec(ss_tail, cap);
                if (ss_val[last] >= ss0) ss_tail = last; else break;
            }
            next_tail = rb_inc(ss_tail, cap);
            if (next_tail == ss_head) ss_head = rb_inc(ss_head, cap);
            ss_idx[ss_tail] = i; ss_val[ss_tail] = ss0; ss_tail = next_tail;
            while (ss_head != ss_tail) {
                const int front_i = ss_idx[ss_head];
                if (front_i + q <= i) ss_head = rb_inc(ss_head, cap); else break;
            }
            out_short_row[i] = ss_val[ss_head];
        }
    }
}

extern "C" __global__
void cksp_batch_f32(const float* __restrict__ high,
                    const float* __restrict__ low,
                    const float* __restrict__ close,
                    int series_len,
                    int first_valid,
                    const int* __restrict__ p_list,
                    const float* __restrict__ x_list,
                    const int* __restrict__ q_list,
                    int n_combos,
                    int cap_max,
                    float* __restrict__ out_long,
                    float* __restrict__ out_short) {
    const int row = blockIdx.y;
    if (row >= n_combos || series_len <= 0) return;
    if (blockIdx.x != 0) return;

    const int base = row * series_len;
    cksp_core_row<false>(
        high, low, close, nullptr, series_len, first_valid,
        p_list[row], x_list[row], q_list[row], cap_max,
         out_long + base, out_short + base
    );
}


extern "C" __global__
void cksp_batch_f32_pretr(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          const float* __restrict__ tr,
                          int series_len,
                          int first_valid,
                          const int* __restrict__ p_list,
                          const float* __restrict__ x_list,
                          const int* __restrict__ q_list,
                          int n_combos,
                          int cap_max,
                          float* __restrict__ out_long,
                          float* __restrict__ out_short) {
    const int row = blockIdx.y;
    if (row >= n_combos || series_len <= 0) return;
    if (blockIdx.x != 0) return;

    const int base = row * series_len;
    cksp_core_row<true>(
        high, low, close, tr, series_len, first_valid,
        p_list[row], x_list[row], q_list[row], cap_max,
        out_long + base, out_short + base
    );
}


extern "C" __global__
void cksp_many_series_one_param_f32(const float* __restrict__ high_tm,
                                    const float* __restrict__ low_tm,
                                    const float* __restrict__ close_tm,
                                    const int* __restrict__ first_valids,
                                    int num_series,
                                    int series_len,
                                    int p,
                                    float x,
                                    int q,
                                    int cap_max,
                                    float* __restrict__ out_long_tm,
                                    float* __restrict__ out_short_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0 || p <= 0 || q <= 0) return;
    const int stride = num_series;
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    if (fv >= series_len) return;
    const int warm = fv + p + q - 1;
    const int cap  = q + 1;


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_long_tm[t * stride + s]  = CUDART_NAN_F;
        out_short_tm[t * stride + s] = CUDART_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    extern __shared__ __align__(16) unsigned char shraw[];
    int*   h_idx  = (int*)shraw;
    int*   l_idx  = h_idx + cap_max;
    int*   ls_idx = l_idx + cap_max;
    int*   ss_idx = ls_idx + cap_max;
    float* ls_val = (float*)(ss_idx + cap_max);
    float* ss_val = ls_val + cap_max;

    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;

    float rma = 0.0f;
    const float alpha = 1.0f / (float)p;
    float sum_tr = 0.0f, c_tr = 0.0f;


    {
        const int t = fv;
        const float hi = high_tm[t * stride + s];
        const float lo = low_tm [t * stride + s];
        const float tr = hi - lo;

        float y = tr - c_tr; float tt = sum_tr + y; c_tr = (tt - sum_tr) - y; sum_tr = tt;
        if (p == 1) rma = tr;

        if (q > 1) {
            while (h_head != h_tail) {
                const int last = rb_dec(h_tail, cap);
                const int last_t = h_idx[last];
                const float last_v = high_tm[last_t * stride + s];
                if (last_v <= hi) h_tail = last; else break;
            }
            int next_tail = rb_inc(h_tail, cap);
            if (next_tail == h_head) h_head = rb_inc(h_head, cap);
            h_idx[h_tail] = t; h_tail = next_tail;

            while (l_head != l_tail) {
                const int last = rb_dec(l_tail, cap);
                const int last_t = l_idx[last];
                const float last_v = low_tm[last_t * stride + s];
                if (last_v >= lo) l_tail = last; else break;
            }
            next_tail = rb_inc(l_tail, cap);
            if (next_tail == l_head) l_head = rb_inc(l_head, cap);
            l_idx[l_tail] = t; l_tail = next_tail;
        }
    }


    if (q == 1) {
        int k = 1;
        float prev_close = close_tm[fv * stride + s];
        for (int t = fv + 1; t < series_len; ++t, ++k) {
            const float hi = high_tm [t * stride + s];
            const float lo = low_tm  [t * stride + s];
            const float clp= prev_close;
            prev_close     = close_tm[t * stride + s];

            const float tr = fmaxf(hi - lo, fmaxf(fabsf(hi - clp), fabsf(lo - clp)));
            if (k < p) {
                float y = tr - c_tr; float tt = sum_tr + y; c_tr = (tt - sum_tr) - y; sum_tr = tt;
                if (k == p - 1) rma = sum_tr / (float)p;
            } else {
                rma = fmaf(alpha, tr - rma, rma);
            }
            if (t >= warm) {
                out_long_tm [t * stride + s] = fmaf(-x, rma, hi);
                out_short_tm[t * stride + s] = fmaf(+x, rma, lo);
            }
        }
        return;
    }

    int k = 1;
    float prev_close = close_tm[fv * stride + s];
    for (int t = fv + 1; t < series_len; ++t, ++k) {
        const float hi = high_tm[t * stride + s];
        const float lo = low_tm [t * stride + s];
        const float clp= prev_close;
        prev_close     = close_tm[t * stride + s];

        const float tr = fmaxf(hi - lo, fmaxf(fabsf(hi - clp), fabsf(lo - clp)));
        if (k < p) {
            float y = tr - c_tr; float tt = sum_tr + y; c_tr = (tt - sum_tr) - y; sum_tr = tt;
            if (k == p - 1) rma = sum_tr / (float)p;
        } else {
            rma = fmaf(alpha, tr - rma, rma);
        }


        while (h_head != h_tail) {
            const int last = rb_dec(h_tail, cap);
            const int last_t = h_idx[last];
            const float last_v = high_tm[last_t * stride + s];
            if (last_v <= hi) h_tail = last; else break;
        }
        int next_tail = rb_inc(h_tail, cap);
        if (next_tail == h_head) h_head = rb_inc(h_head, cap);
        h_idx[h_tail] = t; h_tail = next_tail;
        while (h_head != h_tail) {
            const int front_t = h_idx[h_head];
            if (front_t + q <= t) h_head = rb_inc(h_head, cap); else break;
        }
        const float mh = high_tm[h_idx[h_head] * stride + s];


        while (l_head != l_tail) {
            const int last = rb_dec(l_tail, cap);
            const int last_t = l_idx[last];
            const float last_v = low_tm[last_t * stride + s];
            if (last_v >= lo) l_tail = last; else break;
        }
        next_tail = rb_inc(l_tail, cap);
        if (next_tail == l_head) l_head = rb_inc(l_head, cap);
        l_idx[l_tail] = t; l_tail = next_tail;
        while (l_head != l_tail) {
            const int front_t = l_idx[l_head];
            if (front_t + q <= t) l_head = rb_inc(l_head, cap); else break;
        }
        const float ml = low_tm[l_idx[l_head] * stride + s];

        if (t >= warm) {
            const float ls0 = fmaf(-x, rma, mh);
            const float ss0 = fmaf(+x, rma, ml);

            while (ls_head != ls_tail) {
                const int last = rb_dec(ls_tail, cap);
                if (ls_val[last] <= ls0) ls_tail = last; else break;
            }
            next_tail = rb_inc(ls_tail, cap);
            if (next_tail == ls_head) ls_head = rb_inc(ls_head, cap);
            ls_idx[ls_tail] = t; ls_val[ls_tail] = ls0; ls_tail = next_tail;
            while (ls_head != ls_tail) {
                const int front_t = ls_idx[ls_head];
                if (front_t + q <= t) ls_head = rb_inc(ls_head, cap); else break;
            }
            out_long_tm[t * stride + s] = ls_val[ls_head];

            while (ss_head != ss_tail) {
                const int last = rb_dec(ss_tail, cap);
                if (ss_val[last] >= ss0) ss_tail = last; else break;
            }
            next_tail = rb_inc(ss_tail, cap);
            if (next_tail == ss_head) ss_head = rb_inc(ss_head, cap);
            ss_idx[ss_tail] = t; ss_val[ss_tail] = ss0; ss_tail = next_tail;
            while (ss_head != ss_tail) {
                const int front_t = ss_idx[ss_head];
                if (front_t + q <= t) ss_head = rb_inc(ss_head, cap); else break;
            }
            out_short_tm[t * stride + s] = ss_val[ss_head];
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — cksp (Chande Kroll stop)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/cksp.rs
 *   `cksp_with_kernel` (:255) — first_valid scans CLOSE ONLY, warmup = p+q-1
 *   `cksp_scalar`      (:608) — the RMA recurrence and the four monotonic
 *                               deques, verbatim
 *
 * OUTPUT CONTRACT. The canonical registry owns `long_values` and
 * `short_values`. The preserved generic primary entry emits the first one for
 * the admitted default `(p=10,x=1,q=9)`. `cksp_outputs_f64` emits the complete
 * pair from explicit p/x/q rows and runtime-sized resident deque scratch.
 *
 * DEFAULT-ONLY FEATURE ADMISSION. NeoEthos currently classifies p/q outside
 * its generic `period` vocabulary, so only the exact registry default tuple is
 * admitted. The primary ABI remains period-invariant for that reason; the
 * typed pair ABI is nevertheless fully parameterized and never pretends an
 * inert synthetic period changed the formula.
 *
 * FIRST-VALID IS CLOSE-ONLY. cksp.rs:281 is
 * `close.iter().position(|v| !v.is_nan())`; high and low are never scanned.
 * That is the `HlcCloseOnly` rule, the same one `adxr` uses and a DIFFERENT
 * index from the one `atr` computes on any gapped symbol. Declared in
 * `F64_KERNELS`, not assumed.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. `fabsf` x10 and `fmaxf` x10 -> `fabs`, and the max/min are NOT `fmax`:
 *     see 3.
 *  2. THE RMA IS ONE ROUNDING, NOT THREE. cksp.rs:731 is
 *     `alpha.mul_add(tr - rma, rma)`. Written here as
 *     `fma(alpha, tr - rma, rma)`. A `rma*(1-alpha) + tr*alpha` shape is three
 *     roundings and drifts over a 100k-bar series.
 *  3. TRUE RANGE USES A COMPARISON CHAIN, NOT `max`. cksp.rs:709-721 is
 *     `if hl >= hc { if hl >= lc {hl} else {lc} } else { ... }`. A comparison
 *     against NaN is FALSE, so a NaN `hl` falls through to the else arm and
 *     the result can be `hc` or `lc` — whereas `fmax` would return the
 *     non-NaN operand and give a DIFFERENT answer. This is the inverse of the
 *     rule that applies almost everywhere else in this shard, and it is
 *     written the CPU way on purpose.
 *  4. THE DEQUES ARE REPRODUCED, NOT REPLACED BY A WINDOW SCAN. Their pop
 *     conditions are `<=` / `>=`, which a NaN also fails, so a NaN entry is
 *     never evicted by value and can reach the front. A brute-force `fmax`
 *     over the window would silently skip it. Eviction by INDEX
 *     (`front_i + q <= i`) is the only thing that removes it, exactly as on
 *     the CPU.
 *
 * ONE THREAD PER COLUMN. Carried state: sum_tr, rma, and the ring buffers.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_CKSP_P 10
#define NEO_CKSP_Q 9
#define NEO_CKSP_X 1.0
#define NEO_CKSP_CAP (NEO_CKSP_Q + 1)

__device__ __forceinline__ int neo_cksp_rb_dec(int idx, int cap) {
    return (idx == 0) ? (cap - 1) : (idx - 1);
}
__device__ __forceinline__ int neo_cksp_rb_inc(int idx, int cap) {
    int t = idx + 1;
    return (t == cap) ? 0 : t;
}

__device__ __forceinline__ void cksp_row_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    int p,
    double x,
    int q,
    int first_valid,
    int* __restrict__ h_idx,
    int* __restrict__ l_idx,
    int* __restrict__ ls_idx,
    int* __restrict__ ss_idx,
    double* __restrict__ ls_val,
    double* __restrict__ ss_val,
    int scratch_cap,
    double* __restrict__ row_long_values,
    double* __restrict__ row_short_values)
{
    const bool emit_short = row_short_values != nullptr;
    for (int i = 0; i < len; ++i) {
        row_long_values[i] = NEO_F64_NAN;
        if (emit_short) row_short_values[i] = NEO_F64_NAN;
    }

    const long long warm_span = (long long)p + (long long)q - 1LL;
    if (len <= 0 || p <= 0 || q <= 0 || first_valid < 0 || first_valid >= len ||
        warm_span < 0LL || warm_span >= (long long)(len - first_valid) ||
        scratch_cap < q + 1 || h_idx == nullptr || l_idx == nullptr ||
        ls_idx == nullptr || ls_val == nullptr ||
        (emit_short && (ss_idx == nullptr || ss_val == nullptr))) {
        return;
    }

    const int cap = q + 1;
    const int warmup = first_valid + (int)warm_span;
    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;

    double sum_tr = 0.0;
    double rma = 0.0;
    const double alpha = 1.0 / (double)p;

    for (int i = first_valid; i < len; ++i) {
        const double hi = high[i];
        const double lo = low[i];

        double tr;
        if (i == first_valid) {
            tr = hi - lo;
        } else {
            const double cprev = close[i - 1];
            const double hl = hi - lo;
            const double hc = fabs(hi - cprev);
            const double lc = fabs(lo - cprev);
            /* cksp.rs:709-721 — comparison chain, NOT fmax. Header note 3. */
            if (hl >= hc) { tr = (hl >= lc) ? hl : lc; }
            else          { tr = (hc >= lc) ? hc : lc; }
        }

        const int k = i - first_valid;
        if (k < p) {
            sum_tr += tr;
            if (k == p - 1) rma = sum_tr / (double)p;
        } else {
            rma = fma(alpha, tr - rma, rma);
        }

        /* --- max-high deque over the last q bars ------------------------- */
        while (h_head != h_tail) {
            const int last = neo_cksp_rb_dec(h_tail, cap);
            if (high[h_idx[last]] <= hi) h_tail = last; else break;
        }
        {
            int next_tail = neo_cksp_rb_inc(h_tail, cap);
            if (next_tail == h_head) h_head = neo_cksp_rb_inc(h_head, cap);
            h_idx[h_tail] = i;
            h_tail = next_tail;
        }
        while (h_head != h_tail) {
            if (h_idx[h_head] + q <= i) h_head = neo_cksp_rb_inc(h_head, cap); else break;
        }
        const double mh = high[h_idx[h_head]];

        /* --- min-low deque over the last q bars -------------------------- */
        while (l_head != l_tail) {
            const int last = neo_cksp_rb_dec(l_tail, cap);
            if (low[l_idx[last]] >= lo) l_tail = last; else break;
        }
        {
            int next_tail = neo_cksp_rb_inc(l_tail, cap);
            if (next_tail == l_head) l_head = neo_cksp_rb_inc(l_head, cap);
            l_idx[l_tail] = i;
            l_tail = next_tail;
        }
        while (l_head != l_tail) {
            if (l_idx[l_head] + q <= i) l_head = neo_cksp_rb_inc(l_head, cap); else break;
        }
        const double ml = low[l_idx[l_head]];

        if (i >= warmup) {
            /* cksp.rs:785 — `(-x).mul_add(rma, mh)`, one rounding. */
            const double ls0 = fma(-x, rma, mh);
            const double ss0 = fma(x, rma, ml);

            while (ls_head != ls_tail) {
                const int last = neo_cksp_rb_dec(ls_tail, cap);
                if (ls_val[last] <= ls0) ls_tail = last; else break;
            }
            {
                int next_tail = neo_cksp_rb_inc(ls_tail, cap);
                if (next_tail == ls_head) ls_head = neo_cksp_rb_inc(ls_head, cap);
                ls_idx[ls_tail] = i;
                ls_val[ls_tail] = ls0;
                ls_tail = next_tail;
            }
            while (ls_head != ls_tail) {
                if (ls_idx[ls_head] + q <= i) ls_head = neo_cksp_rb_inc(ls_head, cap); else break;
            }
            row_long_values[i] = ls_val[ls_head];

            if (emit_short) {
                while (ss_head != ss_tail) {
                    const int last = neo_cksp_rb_dec(ss_tail, cap);
                    if (ss_val[last] >= ss0) ss_tail = last; else break;
                }
                {
                    int next_tail = neo_cksp_rb_inc(ss_tail, cap);
                    if (next_tail == ss_head) ss_head = neo_cksp_rb_inc(ss_head, cap);
                    ss_idx[ss_tail] = i;
                    ss_val[ss_tail] = ss0;
                    ss_tail = next_tail;
                }
                while (ss_head != ss_tail) {
                    if (ss_idx[ss_head] + q <= i) ss_head = neo_cksp_rb_inc(ss_head, cap); else break;
                }
                row_short_values[i] = ss_val[ss_head];
            }
        }
    }
}

extern "C" __global__ void cksp_outputs_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int series_len,
    const int* __restrict__ p_values,
    const double* __restrict__ x_values,
    const int* __restrict__ q_values,
    int rows,
    int first_valid,
    int deque_scratch_cap,
    int* __restrict__ h_index_scratch,
    int* __restrict__ l_index_scratch,
    int* __restrict__ long_index_scratch,
    int* __restrict__ short_index_scratch,
    double* __restrict__ long_value_scratch,
    double* __restrict__ short_value_scratch,
    double* __restrict__ out_long_values,
    double* __restrict__ out_short_values)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || series_len <= 0) return;

    const size_t output_offset = (size_t)row * (size_t)series_len;
    const size_t scratch_offset = (size_t)row * (size_t)deque_scratch_cap;
    cksp_row_f64(
        high,
        low,
        close,
        series_len,
        p_values[row],
        x_values[row],
        q_values[row],
        first_valid,
        h_index_scratch + scratch_offset,
        l_index_scratch + scratch_offset,
        long_index_scratch + scratch_offset,
        short_index_scratch + scratch_offset,
        long_value_scratch + scratch_offset,
        short_value_scratch + scratch_offset,
        deque_scratch_cap,
        out_long_values + output_offset,
        out_short_values + output_offset);
}

extern "C" __global__
void cksp_neo_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        const double* __restrict__ close,
                        int series_len,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;   /* exact default-only primary ABI — see the header. */

    int h_idx[NEO_CKSP_CAP];
    int l_idx[NEO_CKSP_CAP];
    int ls_idx[NEO_CKSP_CAP];
    double ls_val[NEO_CKSP_CAP];
    cksp_row_f64(
        high,
        low,
        close,
        series_len,
        NEO_CKSP_P,
        NEO_CKSP_X,
        NEO_CKSP_Q,
        first_valid,
        h_idx,
        l_idx,
        ls_idx,
        nullptr,
        ls_val,
        nullptr,
        NEO_CKSP_CAP,
        out + (size_t)combo * (size_t)series_len,
        nullptr);
}
