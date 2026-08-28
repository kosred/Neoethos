#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef TTM_QNAN_F
#define TTM_QNAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }


struct NeumaierSumF {
    double s, c;
    __device__ __forceinline__ void reset() { s = 0.0; c = 0.0; }
    __device__ __forceinline__ void add(double x) {
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    __device__ __forceinline__ double val() const { return s + c; }
};


struct DequeI {
    int *buf; int cap; int head; int tail; int len;
    __device__ __forceinline__ DequeI(int* p, int c): buf(p), cap(c), head(0), tail(0), len(0) {}
    __device__ __forceinline__ bool empty() const { return len == 0; }
    __device__ __forceinline__ int  size()  const { return len; }
    __device__ __forceinline__ int  front() const { int i = head; return buf[i]; }
    __device__ __forceinline__ int  back()  const { int i = tail - 1; if (i < 0) i += cap; return buf[i]; }
    __device__ __forceinline__ void pop_front() { head = (head + 1 == cap) ? 0 : head + 1; --len; }
    __device__ __forceinline__ void pop_back()  { tail = (tail == 0) ? cap - 1 : tail - 1; --len; }
    __device__ __forceinline__ void push_back(int v) { buf[tail] = v; tail = (tail + 1 == cap) ? 0 : tail + 1; ++len; }
};


static __device__ __forceinline__ float true_range_idx_f32(
    int i,
    int first_valid,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close
){
    const float h = high[i];
    const float l = low[i];
    if (i == first_valid) return fabsf(h - l);
    const float pc  = close[i - 1];
    const float tr1 = fabsf(h - l);
    const float tr2 = fabsf(h - pc);
    const float tr3 = fabsf(l - pc);
    return fmaxf(fmaxf(tr1, tr2), tr3);
}


extern "C" __global__ void ttm_squeeze_batch_f32(

    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,

    const int*   __restrict__ length_arr,
    const float* __restrict__ bb_mult_arr,
    const float* __restrict__ kc_high_arr,
    const float* __restrict__ kc_mid_arr,
    const float* __restrict__ kc_low_arr,

    int series_len,
    int n_combos,
    int first_valid,

    float* __restrict__ out_momentum,
    float* __restrict__ out_squeeze
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int base = combo * series_len;

    const int   L = length_arr[combo];
    auto fill_all_nan = [&]() {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_momentum[base + i] = TTM_QNAN_F;
            out_squeeze[base + i]  = TTM_QNAN_F;
        }
    };

    if (UNLIKELY(L <= 0 || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan();
        return;
    }
    const int warm  = first_valid + L - 1;
    if (UNLIKELY(warm >= series_len)) {
        fill_all_nan();
        return;
    }


    __shared__ int seed_ok_i;
    if (threadIdx.x == 0) {
        bool seed_ok = true;
        for (int j = first_valid; j < first_valid + L && j < series_len; ++j) {
            if (!is_finite_f(close[j]) || !is_finite_f(high[j]) || !is_finite_f(low[j])) { seed_ok = false; break; }
        }
        seed_ok_i = seed_ok ? 1 : 0;
    }
    __syncthreads();
    if (UNLIKELY(seed_ok_i == 0)) {
        fill_all_nan();
        return;
    }


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_momentum[base + i] = TTM_QNAN_F;
        out_squeeze[base + i]  = TTM_QNAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;


    const double n    = (double)L;
    const double invL = 1.0 / n;
    const double sx   = 0.5 * n * (n - 1.0);
    const double sx2  = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    const double den  = n * sx2 - sx * sx;
    const double inv_den = (den != 0.0) ? (1.0 / den) : 0.0;

    const double bb_sq = (double)bb_mult_arr[combo] * (double)bb_mult_arr[combo];
    const double kh_sq = (double)kc_high_arr[combo] * (double)kc_high_arr[combo];
    const double km_sq = (double)kc_mid_arr[combo]  * (double)kc_mid_arr[combo];
    const double kl_sq = (double)kc_low_arr[combo]  * (double)kc_low_arr[combo];


    extern __shared__ unsigned char __ttm_smem[];
    int   *dq_max_buf = (int*)  (__ttm_smem);
    int   *dq_min_buf = dq_max_buf + L;
    float *ring_c     = (float*)(dq_min_buf + L);
    float *ring_tr    = ring_c     + L;
    unsigned char *v_in = (unsigned char*)(ring_tr + L);
    unsigned char *v_tr = v_in + L;

    DequeI dq_max(dq_max_buf, L);
    DequeI dq_min(dq_min_buf, L);


    const int start0 = warm - L + 1;
    NeumaierSumF sumc;  sumc.reset();
    NeumaierSumF sumc2; sumc2.reset();
    NeumaierSumF sumtr; sumtr.reset();
    double sumxc = 0.0;

    int bad_in_window = 0;
    int bad_tr_window = 0;

    for (int k = 0; k < L; ++k) {
        const int idx = start0 + k;
        const float h = high[idx];
        const float l = low[idx];
        const float c = close[idx];

        const unsigned char fin = (unsigned char)(is_finite_f(h) & is_finite_f(l) & is_finite_f(c));
        v_in[k] = fin;
        if (!fin) ++bad_in_window;

        ring_c[k] = c;
        if (fin) {
            const double cd = (double)c;
            sumc.add(cd);
            sumc2.add(cd * cd);
            sumxc += (double)k * cd;
        }


        const float tr = true_range_idx_f32(idx, first_valid, high, low, close);
        const unsigned char ftr = (unsigned char)is_finite_f(tr);
        v_tr[k] = ftr;
        if (!ftr) ++bad_tr_window;
        ring_tr[k] = tr;
        if (ftr) sumtr.add(tr);


        while (!dq_max.empty() && high[dq_max.back()] <= h) dq_max.pop_back();
        dq_max.push_back(idx);
        while (!dq_min.empty() && low[dq_min.back()] >= l) dq_min.pop_back();
        dq_min.push_back(idx);
    }


    int ring_head = 0;


    if (bad_in_window == 0 && bad_tr_window == 0) {
        const double mean = sumc.val() * invL;
        const double var  = fmax(sumc2.val() * invL - mean * mean, 0.0);
        const double dkc  = sumtr.val() * invL;
        const double dkc2 = dkc * dkc;


        const double bbv = bb_sq * var;
        const double t_low  = kl_sq * dkc2;
        const double t_mid  = km_sq * dkc2;
        const double t_high = kh_sq * dkc2;
        out_squeeze[base + warm] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));


        const double highest = (double)high[dq_max.front()];
        const double lowest  = (double)low [dq_min.front()];
        const double midpoint = 0.5 * (highest + lowest);
        const double avg = 0.5 * (midpoint + mean);
        const double S0 = sumc.val() - n * avg;
        const double S1 = sumxc - avg * sx;
        const double slope = (den != 0.0) ? ((n * S1 - sx * S0) * inv_den) : 0.0;
        const double intercept = (S0 - slope * sx) / n;
        const double yhat_last = intercept + slope * (n - 1.0);
        out_momentum[base + warm] = (float)yhat_last;
    } else {
        out_squeeze[base + warm]  = TTM_QNAN_F;
        out_momentum[base + warm] = TTM_QNAN_F;
    }


    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_new = i;
        const int idx_old = i - L;
        const int slot    = ring_head;


        while (!dq_max.empty() && dq_max.front() <= idx_old) dq_max.pop_front();
        while (!dq_min.empty() && dq_min.front() <= idx_old) dq_min.pop_front();


        const float h_new = high[idx_new];
        const float l_new = low [idx_new];
        const float c_new = close[idx_new];
        const unsigned char fin_new = (unsigned char)(is_finite_f(h_new) & is_finite_f(l_new) & is_finite_f(c_new));

        const float tr_new = true_range_idx_f32(idx_new, first_valid, high, low, close);
        const unsigned char ftr_new = (unsigned char)is_finite_f(tr_new);


        const float c_old = ring_c[slot];
        const float tr_old = ring_tr[slot];
        const unsigned char fin_old = v_in[slot];
        const unsigned char ftr_old = v_tr[slot];


        bad_in_window += (int)!fin_new - (int)!fin_old;
        bad_tr_window += (int)!ftr_new - (int)!ftr_old;


        const double sumc_before = sumc.val();
        if (fin_old) {
            const double oldd = (double)c_old;
            sumc.add(-oldd);
            sumc2.add(-(oldd * oldd));
        }
        if (fin_new) {
            const double newd = (double)c_new;
            sumc.add(newd);
            sumc2.add(newd * newd);
        }

        const double adj_old = fin_old ? (double)c_old : 0.0;
        const double adj_new = fin_new ? (double)c_new : 0.0;
        sumxc -= (sumc_before - adj_old);
        sumxc += (double)(L - 1) * adj_new;

        if (ftr_old) sumtr.add(-tr_old);
        if (ftr_new) sumtr.add( tr_new);


        ring_c[slot] = c_new; v_in[slot] = fin_new;
        ring_tr[slot] = tr_new; v_tr[slot] = ftr_new;
        ring_head = (ring_head + 1 == L) ? 0 : ring_head + 1;


        while (!dq_max.empty() && high[dq_max.back()] <= h_new) dq_max.pop_back();
        dq_max.push_back(idx_new);
        while (!dq_min.empty() && low[dq_min.back()] >= l_new) dq_min.pop_back();
        dq_min.push_back(idx_new);

        if (bad_in_window == 0 && bad_tr_window == 0) {
            const double mean = sumc.val() * invL;
            const double var  = fmax(sumc2.val() * invL - mean * mean, 0.0);
            const double dkc  = sumtr.val() * invL;
            const double dkc2 = dkc * dkc;

            const double bbv = bb_sq * var;
            const double t_low  = kl_sq * dkc2;
            const double t_mid  = km_sq * dkc2;
            const double t_high = kh_sq * dkc2;
            out_squeeze[base + i] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

            const double highest = (double)high[dq_max.front()];
            const double lowest  = (double)low [dq_min.front()];
            const double midpoint = 0.5 * (highest + lowest);
            const double avg = 0.5 * (midpoint + mean);
            const double S0 = sumc.val() - n * avg;
            const double S1 = sumxc - avg * sx;
            const double slope = (den != 0.0) ? ((n * S1 - sx * S0) * inv_den) : 0.0;
            const double intercept = (S0 - slope * sx) / n;
            const double yhat_last = intercept + slope * (n - 1.0);
            out_momentum[base + i] = (float)yhat_last;
        } else {
            out_squeeze[base + i]  = TTM_QNAN_F;
            out_momentum[base + i] = TTM_QNAN_F;
        }
    }
}


extern "C" __global__ void ttm_squeeze_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int length,
    float bb_mult,
    float kc_high,
    float kc_mid,
    float kc_low,
    float* __restrict__ out_momentum_tm,
    float* __restrict__ out_squeeze_tm
) {
    const int s   = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;
    if (tid != 0) return;


    float* mo = out_momentum_tm + s;
    float* sq = out_squeeze_tm + s;
    auto fill_all_nan = [&]() {
        for (int t = 0; t < series_len; ++t) {
            mo[t * num_series] = TTM_QNAN_F;
            sq[t * num_series] = TTM_QNAN_F;
        }
    };

    const int L = length;
    const int fv = first_valids[s];
    if (UNLIKELY(L <= 0 || fv < 0 || fv >= series_len)) {
        fill_all_nan();
        return;
    }
    const int warm = fv + L - 1;
    if (UNLIKELY(warm >= series_len)) {
        fill_all_nan();
        return;
    }


    for (int t = 0; t < warm; ++t) {
        mo[t * num_series] = TTM_QNAN_F;
        sq[t * num_series] = TTM_QNAN_F;
    }


    auto H = [&](int t){ return high_tm[(size_t)t * num_series + s]; };
    auto Lw= [&](int t){ return  low_tm[(size_t)t * num_series + s]; };
    auto C = [&](int t){ return close_tm[(size_t)t * num_series + s]; };

    auto TR = [&](int t){
        if (t == fv) return fabsf(H(t) - Lw(t));
        const float pc = C(t - 1);
        const float tr1 = fabsf(H(t) - Lw(t));
        const float tr2 = fabsf(H(t) - pc);
        const float tr3 = fabsf(Lw(t) - pc);
        return fmaxf(fmaxf(tr1, tr2), tr3);
    };


    const float n    = (float)L;
    const float invL = 1.0f / n;
    const float sx   = 0.5f * n * (n - 1.0f);
    const float sx2  = (n - 1.0f) * n * (2.0f * n - 1.0f) / 6.0f;
    const float den  = n * sx2 - sx * sx;
    const float inv_den = (den != 0.0f) ? (1.0f / den) : 0.0f;

    const float bb_sq = bb_mult * bb_mult;
    const float kh_sq = kc_high * kc_high;
    const float km_sq = kc_mid  * kc_mid;
    const float kl_sq = kc_low  * kc_low;


    extern __shared__ unsigned char __ttm_smem[];
    int   *dq_max_buf = (int*)  (__ttm_smem);
    int   *dq_min_buf = dq_max_buf + L;
    float *ring_c     = (float*)(dq_min_buf + L);
    float *ring_tr    = ring_c     + L;
    unsigned char *v_in = (unsigned char*)(ring_tr + L);
    unsigned char *v_tr = v_in + L;

    DequeI dq_max(dq_max_buf, L);
    DequeI dq_min(dq_min_buf, L);


    bool seed_ok = true;

    for (int j = fv; j < fv + L && j < series_len; ++j) {
        float ch = H(j), cl = Lw(j), cc = C(j);
        if (!is_finite_f(ch) || !is_finite_f(cl) || !is_finite_f(cc)) { seed_ok = false; break; }
    }
    if (UNLIKELY(!seed_ok)) {
        fill_all_nan();
        return;
    }


    const int start0 = warm - L + 1;
    NeumaierSumF sumc;  sumc.reset();
    NeumaierSumF sumc2; sumc2.reset();
    NeumaierSumF sumtr; sumtr.reset();
    float sumxc = 0.0f;

    int bad_in_window = 0, bad_tr_window = 0;
    for (int k = 0; k < L; ++k) {
        const int idx = start0 + k;
        const float h = H(idx);
        const float l = Lw(idx);
        const float c = C(idx);
        const unsigned char fin = (unsigned char)(is_finite_f(h) & is_finite_f(l) & is_finite_f(c));
        v_in[k] = fin; ring_c[k] = c;
        if (!fin) ++bad_in_window;
        else { sumc.add(c); sumc2.add(c * c); sumxc = fmaf((float)k, c, sumxc); }

        const float tr = TR(idx);
        const unsigned char ftr = (unsigned char)is_finite_f(tr);
        v_tr[k] = ftr; ring_tr[k] = tr;
        if (!ftr) ++bad_tr_window;
        else sumtr.add(tr);

        while (!dq_max.empty() && H(dq_max.back()) <= h) dq_max.pop_back();
        dq_max.push_back(idx);
        while (!dq_min.empty() && Lw(dq_min.back()) >= l) dq_min.pop_back();
        dq_min.push_back(idx);
    }
    int ring_head = 0;


    if (bad_in_window == 0 && bad_tr_window == 0) {
        const float mean = sumc.val() * invL;
        const float var  = fmaxf(sumc.val() * invL * mean * 0.f , 0.f);
        (void)var;

        const float highest = H(dq_max.front());
        const float lowest  = Lw(dq_min.front());
        const float midpoint = 0.5f * (highest + lowest);
        const float mean_c = sumc.val() * invL;
        const float var_c  = fmaxf(sumc2.val() * invL - mean_c * mean_c, 0.0f);
        const float dkc    = sumtr.val() * invL;
        const float dkc2   = dkc * dkc;

        const float bbv = bb_sq * var_c;
        const float t_low  = kl_sq * dkc2;
        const float t_mid  = km_sq * dkc2;
        const float t_high = kh_sq * dkc2;
        sq[warm * num_series] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

        const float avg = 0.5f * (midpoint + mean_c);
        const float S0  = sumc.val() - n * avg;
        const float S1  = sumxc - avg * sx;
        const float slope = (den != 0.0f) ? ( (n * S1 - sx * S0) * inv_den ) : 0.0f;
        const float intercept = (S0 - slope * sx) * (1.0f / n);
        const float yhat_last = intercept + slope * (n - 1.0f);
        mo[warm * num_series] = yhat_last;
    } else {
        mo[warm * num_series] = TTM_QNAN_F;
        sq[warm * num_series] = TTM_QNAN_F;
    }

    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_new = i;
        const int idx_old = i - L;
        const int slot = ring_head;

        while (!dq_max.empty() && dq_max.front() <= idx_old) dq_max.pop_front();
        while (!dq_min.empty() && dq_min.front() <= idx_old) dq_min.pop_front();

        const float h_new = H(idx_new);
        const float l_new = Lw(idx_new);
        const float c_new = C(idx_new);
        const unsigned char fin_new = (unsigned char)(is_finite_f(h_new) & is_finite_f(l_new) & is_finite_f(c_new));
        const float tr_new = TR(idx_new);
        const unsigned char ftr_new = (unsigned char)is_finite_f(tr_new);

        const float c_old = ring_c[slot];
        const float tr_old = ring_tr[slot];
        const unsigned char fin_old = v_in[slot];
        const unsigned char ftr_old = v_tr[slot];

        bad_in_window += (int)!fin_new - (int)!fin_old;
        bad_tr_window += (int)!ftr_new - (int)!ftr_old;

        const float sumc_before = sumc.val();
        if (fin_old) { sumc.add(-c_old); sumc2.add(-(c_old * c_old)); }
        if (fin_new) { sumc.add( c_new); sumc2.add(  c_new * c_new ); }

        float adj_old = (fin_old ? c_old : 0.0f);
        float adj_new = (fin_new ? c_new : 0.0f);
        sumxc = fmaf(-1.0f, (sumc_before - adj_old), sumxc);
        sumxc = fmaf((float)(L - 1), adj_new,        sumxc);

        if (ftr_old) sumtr.add(-tr_old);
        if (ftr_new) sumtr.add( tr_new);

        ring_c[slot] = c_new; v_in[slot] = fin_new;
        ring_tr[slot] = tr_new; v_tr[slot] = ftr_new;
        ring_head = (ring_head + 1 == L) ? 0 : ring_head + 1;

        while (!dq_max.empty() && H(dq_max.back()) <= h_new) dq_max.pop_back();
        dq_max.push_back(idx_new);
        while (!dq_min.empty() && Lw(dq_min.back()) >= l_new) dq_min.pop_back();
        dq_min.push_back(idx_new);

        if (bad_in_window == 0 && bad_tr_window == 0) {
            const float mean_c = sumc.val() * invL;
            const float var_c  = fmaxf(sumc2.val() * invL - mean_c * mean_c, 0.0f);
            const float dkc    = sumtr.val() * invL;
            const float dkc2   = dkc * dkc;

            const float bbv = bb_sq * var_c;
            const float t_low  = kl_sq * dkc2;
            const float t_mid  = km_sq * dkc2;
            const float t_high = kh_sq * dkc2;
            sq[i * num_series] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

            const float highest = H(dq_max.front());
            const float lowest  = Lw(dq_min.front());
            const float avg = 0.5f * (0.5f * (highest + lowest) + mean_c);
            const float S0  = sumc.val() - n * avg;
            const float S1  = sumxc - avg * sx;
            const float slope = (den != 0.0f) ? ( (n * S1 - sx * S0) * inv_den ) : 0.0f;
            const float intercept = (S0 - slope * sx) * (1.0f / n);
            const float yhat_last = intercept + slope * (n - 1.0f);
            mo[i * num_series] = yhat_last;
        } else {
            mo[i * num_series] = TTM_QNAN_F;
            sq[i * num_series] = TTM_QNAN_F;
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — ttm_squeeze
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ttm_squeeze.rs
 *   `ttm_squeeze_with_kernel`    (:342)  — first_valid scans CLOSE ONLY,
 *                                          warmup = first + length - 1, and
 *                                          the classic-path predicate
 *   `ttm_squeeze_scalar_classic` (:1124) — the path the defaults take
 *
 * WHICH SERIES THIS EMITS. `compute_ttm_squeeze_batch` (cpu_batch.rs:5913)
 * maps output_id "value" -> `out.momentum`. One matrix, so this is the
 * MOMENTUM line. The `squeeze` state (0/1/2/3) is a separate output; its
 * arithmetic is still carried below because it costs three multiplies and
 * keeps this a line-for-line mirror, but it is not written out.
 *
 * PERIOD-INVARIANT, AND THAT IS FAITHFUL. `compute_ttm_squeeze_batch` reads
 * `length` (20), `bb_mult` (2.0) and three kc multipliers (1.0/1.5/2.0) —
 * cpu_batch.rs:5883-5893. It never reads `period`. Identical CPU columns,
 * identical rows here, declared through `is_period_invariant`. And because
 * `length` is fixed at 20 the ring buffers are a compile-time 20 slots, so
 * this kernel needs no `max_period`.
 *
 * THE CLASSIC PATH IS ONLY REACHED AT THE DEFAULTS. ttm_squeeze.rs:402-408
 * requires `length == 20 && bb_mult == 2.0 && kc == (1.0, 1.5, 2.0)` AND
 * `Kernel::Scalar`. `compute_ttm_squeeze_batch` passes
 * `req.kernel.to_non_batch()`, and `Auto` resolves to `Scalar` at :398. So the
 * defaults take this path and this is the reference. The generic path
 * (:432-459) computes the same indicator through `sma_with_kernel` twice and
 * is a DIFFERENT rounding; it is not what a default call runs.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. `fabsf` x8, `fmaxf` x7 -> `fabs`, and `fmax` ONLY where the CPU uses
 *     `f64::max`. See 2 — this file uses BOTH forms and they are not
 *     interchangeable.
 *  2. TRUE RANGE IS COMPUTED TWICE, WITH TWO DIFFERENT NaN SEMANTICS, AND
 *     THAT IS DELIBERATE IN THE REFERENCE.
 *       - warm-up loop, :1189-1201: an explicit `if hl >= hc {...} else {...}`
 *         chain. A comparison against NaN is false, so NaN falls through.
 *       - steady loop, :1364: `hl.max(hc).max(lc)`, i.e. `f64::max`, which
 *         RETURNS THE NON-NaN OPERAND.
 *     The two disagree the moment any of high/low/close is NaN. Both are
 *     reproduced exactly as written. Collapsing them to one form — either
 *     form — is a wrong kernel that passes every clean-data test.
 *  3. THE VARIANCE IS ONE FUSED ROUNDING. `(-m).mul_add(m, sumsq * inv_n)` is
 *     `fma(-m, m, sumsq * inv_n)`: one rounding on the fused pair, one on the
 *     scaled sum. `sumsq/n - m*m` is three and drifts.
 *  4. THE ROLLING sum1 UPDATE READS THE OLD sum0. :1348 is
 *     `sum1 - sum0_old + old + (n-1)*new` where `sum0_old` was captured BEFORE
 *     `sum0 += new - old`. Using the new sum0 gives a plausible slope that is
 *     wrong at every bar.
 *  5. `sumsq = new.mul_add(new, sumsq - old*old)` — the subtraction happens
 *     INSIDE the fma addend, so it is `fma(new, new, sumsq - old*old)`, two
 *     roundings, not `sumsq + new*new - old*old`, which is three.
 *  6. `__int_as_float(0x7f...)` -> `__longlong_as_double(0x7ff8...)`.
 *
 * ONE THREAD PER COLUMN. Carried: four rolling sums, two monotonic deques and
 * two value rings.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_TTMSQ_LENGTH 20
#define NEO_TTMSQ_BB     2.0
#define NEO_TTMSQ_KC_HI  1.0
#define NEO_TTMSQ_KC_MID 1.5
#define NEO_TTMSQ_KC_LO  2.0

extern "C" __global__
void ttm_squeeze_neo_batch_f64(const double* __restrict__ high,
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

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;   /* period-invariant — see the header. */

    const int length = NEO_TTMSQ_LENGTH;

    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        length > len || (len - first_valid) < length) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warmup = first_valid + length - 1;
    for (int i = 0; i < len && i < warmup; ++i) o[i] = NEO_F64_NAN;
    if (warmup >= len) return;

    const double n = (double)length;
    const double sx = 0.5 * n * (n - 1.0);
    const double sx2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    const double den = n * sx2 - sx * sx;
    const double inv_den = 1.0 / den;
    const double inv_n = 1.0 / n;
    const double half_nm1 = 0.5 * (n - 1.0);

    const double bb_sq      = NEO_TTMSQ_BB     * NEO_TTMSQ_BB;
    const double kc_low_sq  = NEO_TTMSQ_KC_LO  * NEO_TTMSQ_KC_LO;
    const double kc_mid_sq  = NEO_TTMSQ_KC_MID * NEO_TTMSQ_KC_MID;
    const double kc_high_sq = NEO_TTMSQ_KC_HI  * NEO_TTMSQ_KC_HI;

    double cbuf[NEO_TTMSQ_LENGTH];
    double trbuf[NEO_TTMSQ_LENGTH];
    int    max_q[NEO_TTMSQ_LENGTH];
    int    min_q[NEO_TTMSQ_LENGTH];
    const int cap = length;

    int cpos = 0, trpos = 0;
    int max_head = 0, max_tail = 0, max_len_ = 0;
    int min_head = 0, min_tail = 0, min_len_ = 0;

    double sum0 = 0.0, sum1 = 0.0, sumsq = 0.0, tr_sum = 0.0;

    /* ---------------------------------------------------------- warm-up ---
     * ttm_squeeze.rs:1172-1249. NOTE the comparison-chain true range. */
    {
        int r = 0;
        for (int i = first_valid; i <= warmup; ++i) {
            const double c = close[i];
            cbuf[cpos] = c;
            sum0 += c;
            sumsq = fma(c, c, sumsq);
            sum1 += (double)r * c;

            double tr_val;
            if (i == first_valid) {
                tr_val = high[i] - low[i];
            } else {
                const double pc = close[i - 1];
                const double hl = high[i] - low[i];
                const double hc = fabs(high[i] - pc);
                const double lc = fabs(low[i] - pc);
                if (hl >= hc) { tr_val = (hl >= lc) ? hl : lc; }
                else          { tr_val = (hc >= lc) ? hc : lc; }
            }
            trbuf[trpos] = tr_val;
            tr_sum += tr_val;

            while (max_len_ > 0) {
                const int back_pos = (max_tail == 0) ? (cap - 1) : (max_tail - 1);
                if (high[i] <= high[max_q[back_pos]]) break;
                max_tail = back_pos;
                max_len_ -= 1;
            }
            max_q[max_tail] = i;
            max_tail += 1; if (max_tail == cap) max_tail = 0;
            max_len_ += 1;

            while (min_len_ > 0) {
                const int back_pos = (min_tail == 0) ? (cap - 1) : (min_tail - 1);
                if (low[i] >= low[min_q[back_pos]]) break;
                min_tail = back_pos;
                min_len_ -= 1;
            }
            min_q[min_tail] = i;
            min_tail += 1; if (min_tail == cap) min_tail = 0;
            min_len_ += 1;

            cpos += 1;  if (cpos == length)  cpos = 0;
            trpos += 1; if (trpos == length) trpos = 0;
            r += 1;
        }
    }

    /* ------------------------------------------------ the seeded bar ----- */
    {
        const double m = sum0 * inv_n;
        const double var = fma(-m, m, sumsq * inv_n);
        const double var_pos = (var > 0.0) ? var : 0.0;
        const double dkc = tr_sum * inv_n;
        const double dkc2 = dkc * dkc;

        /* The squeeze state is a separate output; computed to keep the mirror
         * exact and to keep the compiler from reordering what follows. */
        const double bbv = bb_sq * var_pos;
        const double t_low = kc_low_sq * dkc2;
        const double t_mid = kc_mid_sq * dkc2;
        const double t_high = kc_high_sq * dkc2;
        (void)bbv; (void)t_low; (void)t_mid; (void)t_high;

        const double highest = high[max_q[max_head]];
        const double lowest  = low[min_q[min_head]];

        const double midpoint = 0.5 * (highest + lowest);
        const double avg = 0.5 * (midpoint + m);
        const double sy = sum0 - avg * n;
        const double sxy = sum1 - avg * sx;
        const double slope = fma(n, sxy, -(sx * sy)) * inv_den;
        o[warmup] = sy * inv_n + slope * half_nm1;
    }

    /* ------------------------------------------------- the steady loop ---- */
    for (int i = warmup + 1; i < len; ++i) {
        const int start_idx = i + 1 - length;

        while (max_len_ > 0) {
            if (max_q[max_head] >= start_idx) break;
            max_head += 1; if (max_head == cap) max_head = 0;
            max_len_ -= 1;
        }
        while (min_len_ > 0) {
            if (min_q[min_head] >= start_idx) break;
            min_head += 1; if (min_head == cap) min_head = 0;
            min_len_ -= 1;
        }

        while (max_len_ > 0) {
            const int back_pos = (max_tail == 0) ? (cap - 1) : (max_tail - 1);
            if (high[i] <= high[max_q[back_pos]]) break;
            max_tail = back_pos;
            max_len_ -= 1;
        }
        max_q[max_tail] = i;
        max_tail += 1; if (max_tail == cap) max_tail = 0;
        max_len_ += 1;

        while (min_len_ > 0) {
            const int back_pos = (min_tail == 0) ? (cap - 1) : (min_tail - 1);
            if (low[i] >= low[min_q[back_pos]]) break;
            min_tail = back_pos;
            min_len_ -= 1;
        }
        min_q[min_tail] = i;
        min_tail += 1; if (min_tail == cap) min_tail = 0;
        min_len_ += 1;

        const double old_c = cbuf[cpos];
        const double new_c = close[i];
        const double sum0_old = sum0;
        sum0 += new_c - old_c;
        sumsq = fma(new_c, new_c, sumsq - old_c * old_c);
        sum1 = sum1 - sum0_old + old_c + (n - 1.0) * new_c;
        cbuf[cpos] = new_c;
        cpos += 1; if (cpos == length) cpos = 0;

        const double old_tr = trbuf[trpos];
        const double pc = close[i - 1];
        const double hi_i = high[i];
        const double lo_i = low[i];
        const double hl = hi_i - lo_i;
        const double hc = fabs(hi_i - pc);
        const double lc = fabs(lo_i - pc);
        /* :1364 — `f64::max`, NOT the warm-up loop's comparison chain. */
        const double tr_new = fmax(fmax(hl, hc), lc);
        tr_sum += tr_new - old_tr;
        trbuf[trpos] = tr_new;
        trpos += 1; if (trpos == length) trpos = 0;

        const double m = sum0 * inv_n;
        const double var = fma(-m, m, sumsq * inv_n);
        const double var_pos = (var > 0.0) ? var : 0.0;
        const double dkc = tr_sum * inv_n;
        const double dkc2 = dkc * dkc;
        const double bbv = bb_sq * var_pos;
        const double t_low = kc_low_sq * dkc2;
        const double t_mid = kc_mid_sq * dkc2;
        const double t_high = kc_high_sq * dkc2;
        (void)bbv; (void)t_low; (void)t_mid; (void)t_high;

        const double highest = high[max_q[max_head]];
        const double lowest  = low[min_q[min_head]];

        const double midpoint = 0.5 * (highest + lowest);
        const double avg = 0.5 * (midpoint + m);
        const double sy = sum0 - avg * n;
        const double sxy = sum1 - avg * sx;
        const double slope = fma(n, sxy, -(sx * sy)) * inv_den;
        o[i] = sy * inv_n + slope * half_nm1;
    }
}
