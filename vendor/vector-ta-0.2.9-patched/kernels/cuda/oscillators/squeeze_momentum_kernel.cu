#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SMI_QNAN_F
#define SMI_QNAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }


static __device__ __forceinline__ double true_range_idx(
    int i, const float* __restrict__ high, const float* __restrict__ low, const float* __restrict__ close
) {
    const double h = (double)high[i];
    const double l = (double)low[i];
    if (i == 0) {
        return fabs(h - l);
    } else {
        const double pc = (double)close[i - 1];
        const double tr1 = fabs(h - l);
        const double tr2 = fabs(h - pc);
        const double tr3 = fabs(l - pc);
        return fmax(fmax(tr1, tr2), tr3);
    }
}

extern "C" __global__ void squeeze_momentum_batch_f32(

    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ lbb_arr,
    const float* __restrict__ mbb_arr,
    const int*   __restrict__ lkc_arr,
    const float* __restrict__ mkc_arr,
    int series_len,
    int n_combos,
    int first_valid,

    float* __restrict__ out_sq,
    float* __restrict__ out_mo,
    float* __restrict__ out_si
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;


    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_sq[base + i] = SMI_QNAN_F;
        out_mo[base + i] = SMI_QNAN_F;
        out_si[base + i] = SMI_QNAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const int lbb = lbb_arr[combo];
    const double mbb = (double)mbb_arr[combo];
    const int lkc = lkc_arr[combo];
    const double mkc = (double)mkc_arr[combo];

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    if (UNLIKELY(lbb <= 0 || lkc <= 0)) return;

    const int warm_sq = max(lbb, lkc) - 1;
    const int warm_mo = lkc - 1;
    const int warm_si = warm_mo + 1;


    const double inv_lbb = 1.0 / (double)lbb;
    const double inv_lkc = 1.0 / (double)lkc;
    const double p = (double)lkc;
    const double sum_x  = 0.5 * p * (p + 1.0);
    const double sum_x2 = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
    const double denom  = p * sum_x2 - sum_x * sum_x;
    const double inv_den= 1.0 / denom;
    const double x_last_minus_xbar = p - sum_x * inv_lkc;


    int start_bb = first_valid + lbb - 1;
    int start_kc = first_valid + lkc - 1;
    if (start_bb < series_len) {
        double sum_bb = 0.0, sumsq_bb = 0.0;
        for (int j = start_bb + 1 - lbb; j <= start_bb; ++j) {
            const double v = (double)close[j];
            sum_bb += v;
            sumsq_bb = fma(v, v, sumsq_bb);
        }


    }
    double sum_bb = 0.0, sumsq_bb = 0.0;
    if (start_bb < series_len) {
        for (int j = start_bb + 1 - lbb; j <= start_bb; ++j) {
            const double v = (double)close[j];
            sum_bb += v;
            sumsq_bb = fma(v, v, sumsq_bb);
        }
    }
    double sum_kc = 0.0, sum_tr = 0.0;
    if (start_kc < series_len) {
        for (int j = start_kc + 1 - lkc; j <= start_kc; ++j) {
            sum_kc += (double)close[j];
            sum_tr += true_range_idx(j, high, low, close);
        }
    }


    bool bb_seed_ok = true; for (int j = 0; j < lbb && j < series_len; ++j) { if (!is_finite_f(close[j])) { bb_seed_ok = false; break; } }
    bool kc_seed_ok = true; for (int j = 0; j < lkc && j < series_len; ++j) { if (!is_finite_f(close[j]) || !is_finite_f(high[j]) || !is_finite_f(low[j])) { kc_seed_ok = false; break; } }


    double prev_momentum = NAN;
    for (int i = first_valid; i < series_len; ++i) {

        if (i > start_bb) {
            const double c_new = (double)close[i];
            const double c_old = (double)close[i - lbb];
            sum_bb += c_new - c_old;
            sumsq_bb = fma(c_new, c_new, sumsq_bb - c_old * c_old);
        }
        if (i > start_kc) {
            const double c_new = (double)close[i];
            const double c_old = (double)close[i - lkc];
            sum_kc += c_new - c_old;

            const double tr_new = true_range_idx(i, high, low, close);
            const int old_idx = i - lkc;
            const double tr_old = true_range_idx(old_idx, high, low, close);
            sum_tr += tr_new - tr_old;
        }


        if (i >= start_bb && i >= start_kc && i >= warm_sq) {


            if (bb_seed_ok && kc_seed_ok) {
            const double mean_bb = sum_bb * inv_lbb;
            const double var_bb = fma(sumsq_bb * inv_lbb, 1.0, -mean_bb * mean_bb);
            const double dev_bb = sqrt(fmax(0.0, var_bb));
            const double upper_bb = mean_bb + mbb * dev_bb;
            const double lower_bb = mean_bb - mbb * dev_bb;

            const double kc_mid = sum_kc * inv_lkc;
            const double tr_avg = sum_tr * inv_lkc;
            const double upper_kc = kc_mid + mkc * tr_avg;
            const double lower_kc = kc_mid - mkc * tr_avg;

            const bool on  = (lower_bb > lower_kc) && (upper_bb < upper_kc);
            const bool off = (lower_bb < lower_kc) && (upper_bb > upper_kc);
            out_sq[base + i] = on ? -1.0f : (off ? 1.0f : 0.0f);
            } else {
                out_sq[base + i] = SMI_QNAN_F;
            }
        }


        if (i >= start_kc && kc_seed_ok) {

            double highest = -INFINITY, lowest = INFINITY;
            const int win_start = i - lkc + 1;
            for (int j = win_start; j <= i; ++j) {
                const double h = (double)high[j];
                const double l = (double)low[j];
                if (h > highest) highest = h;
                if (l < lowest) lowest = l;
            }
            const double kc_mid = sum_kc * inv_lkc;
            const double c_i = (double)close[i];
            const double raw_i = c_i - 0.25 * (highest + lowest) - 0.5 * kc_mid;


            double S0 = 0.0, S1 = 0.0;
            double j = 1.0;
            for (int t = win_start; t <= i; ++t, j += 1.0) {
                const double y = (double)close[t] - 0.25 * ((double)high[t] + (double)low[t]) - 0.5 * (sum_kc );

            }


            S0 = 0.0; S1 = 0.0; j = 1.0;
            for (int t = win_start; t <= i; ++t, j += 1.0) {

                double sum_c = 0.0;
                const int ts = t - lkc + 1;
                for (int u = ts; u <= t; ++u) sum_c += (double)close[u];
                const double kc_mid_t = sum_c * inv_lkc;

                double hh = -INFINITY, ll = INFINITY;
                for (int u = ts; u <= t; ++u) { double hhv=(double)high[u]; double llv=(double)low[u]; if (hhv>hh) hh=hhv; if (llv<ll) ll=llv; }
                const double raw_t = (double)close[t] - 0.25 * (hh + ll) - 0.5 * kc_mid_t;
                S0 += raw_t;
                S1 = fma(j, raw_t, S1);
            }
            const double b = (-sum_x * S0 + p * S1) * inv_den;
            const double ybar = S0 * inv_lkc;
            const double yhat_last = fma(b, x_last_minus_xbar, ybar);
            out_mo[base + i] = (float)yhat_last;


            if (i >= 1) {
                const double prev = (double)out_mo[base + (i - 1)];
                if (isfinite(prev) && isfinite(yhat_last)) {
                    float sig = 0.0f;
                    if (yhat_last > 0.0) {
                        sig = (yhat_last > prev) ? 1.0f : 2.0f;
                    } else {
                        sig = (yhat_last < prev) ? -1.0f : -2.0f;
                    }
                    out_si[base + i] = sig;
                } else if (i >= warm_si) {
                    out_si[base + i] = SMI_QNAN_F;
                }
            }
        }
    }
}


#ifndef SMI_QNAN_F
#define SMI_QNAN_F (__int_as_float(0x7fffffff))
#endif


static __device__ __forceinline__ void kbn_add(float x, float &sum, float &c) {
    float t = sum + x;
    if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
    else                        c += (x - t) + sum;
    sum = t;
}

static __device__ __forceinline__ float fmaxf2(float a, float b){ return fmaxf(a,b); }
static __device__ __forceinline__ float fminf2(float a, float b){ return fminf(a,b); }


extern "C" __global__ void smi_precompute_shared_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,

    float* __restrict__ tr,
    float* __restrict__ ps_close,
    float* __restrict__ ps_close2,
    float* __restrict__ ps_tr,
    int*   __restrict__ log2_tbl,
    float* __restrict__ st_max,
    float* __restrict__ st_min,
    int K
){
    const int N = series_len;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;


    for (int i = tid; i < N; i += nthreads) {
        float h = high[i], l = low[i];
        float tr1 = fabsf(h - l);
        float tr2 = (i>0) ? fabsf(h - close[i-1]) : 0.0f;
        float tr3 = (i>0) ? fabsf(l - close[i-1]) : 0.0f;
        tr[i] = fmaxf(tr1, fmaxf(tr2, tr3));
    }
    __syncthreads();


    if (tid == 0) {
        log2_tbl[0] = 0;
        log2_tbl[1] = 0;
        for (int i = 2; i <= N; ++i) log2_tbl[i] = log2_tbl[i >> 1] + 1;

        float s1=0.0f,c1=0.0f;
        float s2=0.0f,c2=0.0f;
        float s3=0.0f,c3=0.0f;
        for (int i = 0; i < N; ++i) {
            const float ci = close[i];
            const float ci2 = ci * ci;
            const float tri = tr[i];
            kbn_add(ci,  s1, c1); ps_close[i]  = s1 + c1;
            kbn_add(ci2, s2, c2); ps_close2[i] = s2 + c2;
            kbn_add(tri, s3, c3); ps_tr[i]     = s3 + c3;
        }
    }
    __syncthreads();


    for (int i = tid; i < N; i += nthreads) {
        st_max[0 * N + i] = high[i];
        st_min[0 * N + i] = low[i];
    }
    __syncthreads();


    for (int k = 1; k < K; ++k) {
        const int span = 1 << k;
        const int half = span >> 1;
        for (int i = tid; i + span - 1 < N; i += nthreads) {
            const int off0 = (k-1) * N + i;
            const int off1 = (k-1) * N + i + half;
            st_max[k * N + i] = fmaxf2(st_max[off0], st_max[off1]);
            st_min[k * N + i] = fminf2(st_min[off0], st_min[off1]);
        }
        __syncthreads();
    }
}


static __device__ __forceinline__ float rmq_max(
    const float* __restrict__ st_max, const int* __restrict__ log2_tbl,
    int N, int , int l, int r)
{
    const int len = r - l + 1;
    const int k = log2_tbl[len];
    const float a = st_max[k * N + l];
    const float b = st_max[k * N + (r - (1 << k) + 1)];
    return fmaxf(a, b);
}
static __device__ __forceinline__ float rmq_min(
    const float* __restrict__ st_min, const int* __restrict__ log2_tbl,
    int N, int , int l, int r)
{
    const int len = r - l + 1;
    const int k = log2_tbl[len];
    const float a = st_min[k * N + l];
    const float b = st_min[k * N + (r - (1 << k) + 1)];
    return fminf(a, b);
}

static __device__ __forceinline__ float win_sum_ps(
    const float* __restrict__ ps, int i, int len)
{
    const int j = i - len;
    const float si = ps[i];
    return (j >= 0) ? (si - ps[j]) : si;
}


extern "C" __global__ void squeeze_momentum_batch_f32_opt(

    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ lbb_arr,
    const float* __restrict__ mbb_arr,
    const int*   __restrict__ lkc_arr,
    const float* __restrict__ mkc_arr,
    int series_len,
    int n_combos,
    int first_valid,

    const float* __restrict__ tr,
    const float* __restrict__ ps_close,
    const float* __restrict__ ps_close2,
    const float* __restrict__ ps_tr,
    const int*   __restrict__ log2_tbl,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    int K,

    float* __restrict__ out_sq,
    float* __restrict__ out_mo,
    float* __restrict__ out_si
){
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    auto fill_all_nan = [&]() {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_sq[base + i] = SMI_QNAN_F;
            out_mo[base + i] = SMI_QNAN_F;
            out_si[base + i] = SMI_QNAN_F;
        }
    };

    __shared__ int sh_valid;
    __shared__ int sh_lbb;
    __shared__ int sh_lkc;
    __shared__ int sh_start_bb;
    __shared__ int sh_start_kc;
    __shared__ int sh_pref_sq;
    __shared__ int sh_pref_mo;
    __shared__ int sh_pref_si;
    __shared__ float sh_mbb;
    __shared__ float sh_mkc;
    __shared__ int sh_bb_seed_ok;

    if (threadIdx.x == 0) {
        sh_valid = 1;
        if (first_valid < 0 || first_valid >= series_len) sh_valid = 0;

        sh_lbb = lbb_arr[combo];
        sh_mbb = mbb_arr[combo];
        sh_lkc = lkc_arr[combo];
        sh_mkc = mkc_arr[combo];

        if (sh_lbb <= 0 || sh_lkc <= 0 || sh_lbb > series_len || sh_lkc > series_len) {
            sh_valid = 0;
        }

        sh_start_bb = 0;
        sh_start_kc = 0;
        sh_bb_seed_ok = 0;
        if (sh_valid) {
            sh_start_bb = first_valid + sh_lbb - 1;
            sh_start_kc = first_valid + sh_lkc - 1;
            if (sh_start_bb >= series_len || sh_start_kc >= series_len) {
                sh_valid = 0;
            }
        }


        if (sh_valid) {
            bool bb_ok = true;
            for (int j = 0; j < sh_lbb && j < series_len; ++j) {
                if (!is_finite_f(close[j])) { bb_ok = false; break; }
            }
            bool kc_ok = true;
            for (int j = 0; j < sh_lkc && j < series_len; ++j) {
                if (!is_finite_f(close[j]) || !is_finite_f(high[j]) || !is_finite_f(low[j])) { kc_ok = false; break; }
            }
            sh_bb_seed_ok = bb_ok ? 1 : 0;
            if (!kc_ok) sh_valid = 0;
        }


        sh_pref_sq = 0;
        sh_pref_mo = 0;
        sh_pref_si = 0;
        if (sh_valid) {
            const int warm_sq = max(sh_lbb, sh_lkc) - 1;
            const int warm_si = sh_lkc;

            int pref_sq = sh_start_bb;
            if (sh_start_kc > pref_sq) pref_sq = sh_start_kc;
            if (warm_sq > pref_sq) pref_sq = warm_sq;
            sh_pref_sq = pref_sq;

            sh_pref_mo = sh_start_kc;

            int pref_si = warm_si;
            if (sh_start_kc > pref_si) pref_si = sh_start_kc;
            sh_pref_si = pref_si;
        }
    }
    __syncthreads();

    if (!sh_valid) {
        fill_all_nan();
        return;
    }


    for (int i = threadIdx.x; i < sh_pref_sq; i += blockDim.x) out_sq[base + i] = SMI_QNAN_F;
    for (int i = threadIdx.x; i < sh_pref_mo; i += blockDim.x) out_mo[base + i] = SMI_QNAN_F;
    for (int i = threadIdx.x; i < sh_pref_si; i += blockDim.x) out_si[base + i] = SMI_QNAN_F;
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int   lbb = sh_lbb;
    const float mbb = sh_mbb;
    const int   lkc = sh_lkc;
    const float mkc = sh_mkc;
    const int   start_bb = sh_start_bb;
    const int   start_kc = sh_start_kc;
    const bool  bb_seed_ok = (sh_bb_seed_ok != 0);
    const bool  kc_seed_ok = true;

    const int N = series_len;
    const int warm_sq = max(lbb, lkc) - 1;
    const int warm_mo = lkc - 1;
    const int warm_si = warm_mo + 1;

    const float inv_lbb = 1.0f / float(lbb);
    const float inv_lkc = 1.0f / float(lkc);
    const float p       = float(lkc);
    const float sum_x   = 0.5f * p * (p + 1.0f);
    const float sum_x2  = p * (p + 1.0f) * (2.0f * p + 1.0f) / 6.0f;
    const float denom   = fmaf(p, sum_x2, -sum_x * sum_x);
    const float inv_den = 1.0f / denom;
    const float x_last_minus_xbar = p - sum_x * inv_lkc;

    extern __shared__ float s_ring[];
    float* raw_ring = s_ring;

    float S0 = 0.0f, S1 = 0.0f;
    bool   ols_seeded = false;

    for (int i = first_valid; i < N; ++i) {
        if (i >= start_bb && i >= start_kc && i >= warm_sq) {
            if (bb_seed_ok && kc_seed_ok) {
                const float sum_bb   = win_sum_ps(ps_close,  i, lbb);
                const float sum2_bb  = win_sum_ps(ps_close2, i, lbb);
                const float mean_bb  = sum_bb * inv_lbb;
                float var_bb = fmaf(sum2_bb * inv_lbb, 1.0f, -mean_bb * mean_bb);
                var_bb = fmaxf(0.0f, var_bb);
                const float dev_bb = sqrtf(var_bb);
                const float upper_bb = mean_bb + mbb * dev_bb;
                const float lower_bb = mean_bb - mbb * dev_bb;

                const float kc_mid = win_sum_ps(ps_close, i, lkc) * inv_lkc;
                const float tr_avg = win_sum_ps(ps_tr,    i, lkc) * inv_lkc;
                const float upper_kc = kc_mid + mkc * tr_avg;
                const float lower_kc = kc_mid - mkc * tr_avg;

                const bool on  = (lower_bb > lower_kc) && (upper_bb < upper_kc);
                const bool off = (lower_bb < lower_kc) && (upper_bb > upper_kc);
                out_sq[base + i] = on ? -1.0f : (off ? 1.0f : 0.0f);
            } else {
                out_sq[base + i] = SMI_QNAN_F;
            }
        }

        if (i >= start_kc && kc_seed_ok) {
            const int l = i - lkc + 1;
            const float highest = rmq_max(st_max, log2_tbl, N, K, l, i);
            const float lowest  = rmq_min(st_min, log2_tbl, N, K, l, i);
            const float kc_mid_i= win_sum_ps(ps_close, i, lkc) * inv_lkc;
            const float raw_i   = close[i] - 0.25f * (highest + lowest) - 0.5f * kc_mid_i;

            if (!ols_seeded) {
                float j = 1.0f;
                for (int t = l; t <= i; ++t, j += 1.0f) {
                    const int lt = t - lkc + 1;
                    const float h = rmq_max(st_max, log2_tbl, N, K, lt, t);
                    const float lw= rmq_min(st_min, log2_tbl, N, K, lt, t);
                    const float kc_mid_t = win_sum_ps(ps_close, t, lkc) * inv_lkc;
                    const float raw_t = close[t] - 0.25f * (h + lw) - 0.5f * kc_mid_t;
                    raw_ring[(t - l) % lkc] = raw_t;
                    S0 += raw_t;
                    S1 = fmaf(j, raw_t, S1);
                }
                ols_seeded = true;
            } else {
                const float y_new = raw_i;
                const float y_old = raw_ring[(i - lkc) % lkc];
                S1 = fmaf(p, y_new, (S1 - S0));
                S0 = fmaf(1.0f, y_new, S0 - y_old);
                raw_ring[i % lkc] = y_new;
            }

            const float b    = fmaf(-sum_x, S0, p * S1) * inv_den;
            const float ybar = S0 * inv_lkc;
            const float yhat = fmaf(b, x_last_minus_xbar, ybar);
            out_mo[base + i] = yhat;

            if (i >= 1) {
                const float prev = out_mo[base + (i - 1)];
                if (is_finite_f(prev) && is_finite_f(yhat)) {
                    float sig = 0.0f;
                    if (yhat > 0.0f) sig = (yhat > prev) ?  1.0f :  2.0f;
                    else             sig = (yhat < prev) ? -1.0f : -2.0f;
                    out_si[base + i] = sig;
                } else if (i >= warm_si) {
                    out_si[base + i] = SMI_QNAN_F;
                }
            }
        }
    }
}


extern "C" __global__ void squeeze_momentum_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int lbb,
    float mbb,
    int lkc,
    float mkc,
    float* __restrict__ out_sq_tm,
    float* __restrict__ out_mo_tm,
    float* __restrict__ out_si_tm
) {
    const int series = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (tid != 0) return;

    const int fv = first_valids[series];
    if (UNLIKELY(fv < 0 || fv >= series_len || lbb <= 0 || lkc <= 0)) {

        float* sq = out_sq_tm + series;
        float* mo = out_mo_tm + series;
        float* si = out_si_tm + series;
        for (int r = 0; r < series_len; ++r) {
            *sq = SMI_QNAN_F; *mo = SMI_QNAN_F; *si = SMI_QNAN_F;
            sq += num_series; mo += num_series; si += num_series;
        }
        return;
    }

    const int warm_sq = max(lbb, lkc) - 1;
    const int warm_mo = lkc - 1;
    const int warm_si = warm_mo + 1;
    const double inv_lbb = 1.0 / (double)lbb;
    const double inv_lkc = 1.0 / (double)lkc;
    const double p = (double)lkc;
    const double sum_x  = 0.5 * p * (p + 1.0);
    const double sum_x2 = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
    const double inv_den= 1.0 / (p * sum_x2 - sum_x * sum_x);
    const double x_last_minus_xbar = p - sum_x * inv_lkc;


    auto H = [&](int t){ return (double)high_tm[(size_t)t * num_series + series]; };
    auto L = [&](int t){ return (double)low_tm[(size_t)t * num_series + series]; };
    auto C = [&](int t){ return (double)close_tm[(size_t)t * num_series + series]; };
    auto TR = [&](int t){
        if (t == 0) return fabs(H(t) - L(t));
        const double pc = C(t - 1);
        const double tr1 = fabs(H(t) - L(t));
        const double tr2 = fabs(H(t) - pc);
        const double tr3 = fabs(L(t) - pc);
        return fmax(fmax(tr1, tr2), tr3);
    };


    double sum_bb = 0.0, sumsq_bb = 0.0;
    double sum_kc = 0.0, sum_tr = 0.0;
    const int start_bb = fv + lbb - 1;
    const int start_kc = fv + lkc - 1;
    if (start_bb < series_len) {
        for (int j = start_bb + 1 - lbb; j <= start_bb; ++j) {
            const double v = C(j);
            sum_bb += v; sumsq_bb = fma(v, v, sumsq_bb);
        }
    }
    if (start_kc < series_len) {
        for (int j = start_kc + 1 - lkc; j <= start_kc; ++j) {
            sum_kc += C(j);
            sum_tr += TR(j);
        }
    }

    float* sq = out_sq_tm + series;
    float* mo = out_mo_tm + series;
    float* si = out_si_tm + series;


    for (int r = 0; r < warm_sq && r < series_len; ++r) { sq[r * num_series] = SMI_QNAN_F; }
    for (int r = 0; r < warm_mo && r < series_len; ++r) { mo[r * num_series] = SMI_QNAN_F; }
    for (int r = 0; r < warm_si && r < series_len; ++r) { si[r * num_series] = SMI_QNAN_F; }


    bool bb_seed_ok = true; for (int j = 0; j < lbb && j < series_len; ++j) { float cc = (float)C(j); if (!is_finite_f(cc)) { bb_seed_ok=false; break; } }
    bool kc_seed_ok = true; for (int j = 0; j < lkc && j < series_len; ++j) { float ch=(float)H(j); float cl=(float)L(j); float cc=(float)C(j); if (!is_finite_f(ch)||!is_finite_f(cl)||!is_finite_f(cc)) { kc_seed_ok=false; break; } }

    for (int t = fv; t < series_len; ++t) {
        if (t > start_bb) {
            const double c_new = C(t);
            const double c_old = C(t - lbb);
            sum_bb += c_new - c_old;
            sumsq_bb = fma(c_new, c_new, sumsq_bb - c_old * c_old);
        }
        if (t > start_kc) {
            sum_kc += C(t) - C(t - lkc);
            sum_tr += TR(t) - TR(t - lkc);
        }

        if (t >= start_bb && t >= start_kc && t >= warm_sq) {
            const double mean_bb = sum_bb * inv_lbb;
            const double var_bb = fma(sumsq_bb * inv_lbb, 1.0, -mean_bb * mean_bb);
            const double dev_bb = sqrt(fmax(0.0, var_bb));
            const double upper_bb = mean_bb + (double)mbb * dev_bb;
            const double lower_bb = mean_bb - (double)mbb * dev_bb;
            const double kc_mid = sum_kc * inv_lkc;
            const double tr_avg = sum_tr * inv_lkc;
            const double upper_kc = kc_mid + (double)mkc * tr_avg;
            const double lower_kc = kc_mid - (double)mkc * tr_avg;
            const bool on = (lower_bb > lower_kc) && (upper_bb < upper_kc);
            const bool off= (lower_bb < lower_kc) && (upper_bb > upper_kc);
            sq[t * num_series] = on ? -1.0f : (off ? 1.0f : 0.0f);
        }

        if (t >= start_kc) {

            const int ws = t - lkc + 1;
            double highest = -INFINITY, lowest = INFINITY;
            for (int j = ws; j <= t; ++j) { const double h=H(j), l=L(j); if (h>highest) highest=h; if (l<lowest) lowest=l; }


            double S0 = 0.0, S1 = 0.0, j = 1.0;
            for (int u = ws; u <= t; ++u, j += 1.0) {

                double sum_c = 0.0; const int us = u - lkc + 1;
                for (int v = us; v <= u; ++v) sum_c += C(v);
                const double kc_mid_u = sum_c * inv_lkc;
                double hh=-INFINITY, ll=INFINITY; for (int v=us; v<=u; ++v){ double hv=H(v), lv=L(v); if (hv>hh) hh=hv; if (lv<ll) ll=lv; }
                const double raw_u = C(u) - 0.25 * (hh + ll) - 0.5 * kc_mid_u;
                S0 += raw_u; S1 = fma(j, raw_u, S1);
            }
            const double b = (-sum_x * S0 + p * S1) * inv_den;
            const double ybar = S0 * inv_lkc;
            const double yhat_last = fma(b, x_last_minus_xbar, ybar);
            mo[t * num_series] = (float)yhat_last;
            if (t >= 1) {
                const double prev = (double)mo[(t - 1) * num_series];
                if (isfinite(prev) && isfinite(yhat_last)) {
                    float sig = 0.0f;
                    if (yhat_last > 0.0) sig = (yhat_last > prev) ? 1.0f : 2.0f; else sig = (yhat_last < prev) ? -1.0f : -2.0f;
                    si[t * num_series] = sig;
                } else if (t >= warm_si) {
                    si[t * num_series] = SMI_QNAN_F;
                }
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3   (squeeze_momentum)
 *
 * CPU reference: `squeeze_momentum_scalar_classic`
 *   (src/indicators/squeeze_momentum.rs:1051). `squeeze_momentum_into_slices`
 *   discards the kernel selector outright (`let _ = kern;`, :292) and calls
 *   this function unconditionally, so there is ONE CPU path and therefore one
 *   oracle -- no scalar-vs-AVX seed question arises here.
 *
 * Column: output_id "value" -> `out.momentum` (cpu_batch.rs:6281-6284).
 *   `squeeze` and `momentum_signal` are different columns and are deliberately
 *   not computed: `momentum` depends on `length_kc` alone, never on `mult_bb`
 *   or `mult_kc`, so the Bollinger/Keltner comparison is dead weight for this
 *   output.
 *
 * PERIOD-INVARIANT: `compute_squeeze_momentum_batch` reads `length_bb` (20),
 *   `mult_bb` (2.0), `length_kc` (20) and `mult_kc` (1.5)
 *   (cpu_batch.rs:6260-6263) and NEVER `period`.
 *
 * Input: high / low / close -- `extract_ohlc_input("squeeze_momentum",
 *   req.data)` (cpu_batch.rs:6252) -> F64InputKind::Hlc.
 *
 * FIRST-VALID: :280-282 is the first index at which high, low AND close are
 *   ALL non-NaN SIMULTANEOUSLY -- F64FirstValidRule::AllInputsNonNan. It is
 *   NOT `HlcMaxOfIndependentFirsts`: this scan is a single `find` over the
 *   triple, so on a gapped symbol it names a different bar from the adx/natr
 *   rule, and the index sets both the NaN prefix and the seed window.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `S0`/`S1` are updated
 *   incrementally (`(S1 - S0) + p * y_new`, :1372) so bar i's rolling linear
 *   regression depends on bar i-1's pair; and the windowed high/low use
 *   monotonic deques whose contents depend on the arrival order.
 *
 * The deques are reproduced EXACTLY rather than replaced by a window scan.
 *   A scan would return the same max VALUE on clean data, but the CPU's pop
 *   condition is `high[idx] <= hi` (:1228) -- a comparison that is FALSE
 *   against a NaN -- so on a frame carrying a NaN high the two disagree about
 *   which index survives, and `highest` is read from that index.
 *
 * Roundings, counted against the CPU lines:
 *   :1105  sumsq_bb = f64::mul_add(v, v, sumsq_bb)      -- (squeeze only)
 *   :1338  s1 = f64::mul_add(j, y, s1)                  -- ONE fma
 *   :1346  let b = f64::mul_add(-sum_x, S0, p * S1) * inv_denom  -- ONE fma
 *   :1348  let yhat_last = f64::mul_add(b, x_last_minus_xbar, ybar) -- ONE fma
 *   :1372  let new_S1 = (S1 - S0) + p * y_new           -- plain, NO fma
 *   :1373  let new_S0 = (S0 - y_old) + y_new            -- plain, NO fma
 *   :1317  let raw_i = c_i - 0.25 * (highest + lowest) - 0.5 * kc_mid -- plain
 *   Three fmas and no more. Folding :1372 into an fma would REMOVE a rounding
 *   the reference performs, and this column feeds a sign test.
 *
 * NaN semantics: `tr1.max(tr2).max(tr3)` (:1134) is `f64::max`, which returns
 *   the NON-NaN operand -- `fmax` is used here for exactly that reason. The
 *   true range feeds `sum_tr`, which is carried subtract-then-add, so a NaN
 *   admitted by an if-chain would poison every later bar of the row.
 *
 * One honest divergence, stated rather than hidden: on the CPU, momentum at
 *   indices in [length_kc - 1, first_valid + length_kc - 1) is never written
 *   at all -- `squeeze_momentum_with_kernel` allocates with
 *   `alloc_with_nan_prefix(len, 0)` (:311) and `scalar_classic` fills only
 *   `[..warm_m]` (:1074). Those bars are uninitialised memory whenever
 *   `first_valid > 0`. This kernel writes NaN there, which is the only
 *   defensible value; it is a divergence from garbage, not from a number.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:6260-6263. `length_kc` bounds three per-thread
 * rings, and it is a CPU DEFAULT rather than a swept parameter (this indicator
 * is period-invariant), so no caller-supplied number reaches them. */
#define NEO_SQM_LKC 20

extern "C" __global__
void squeeze_momentum_neo_batch_f64(const double* __restrict__ high,
                                    const double* __restrict__ low,
                                    const double* __restrict__ close,
                                    int n,
                                    const int* __restrict__ periods,
                                    int n_combos,
                                    int first_valid,
                                    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods; /* period-invariant -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int lkc = NEO_SQM_LKC;
    const int lbb = NEO_SQM_LKC;   /* length_bb, same default; unused by this column */

    /* :266-278 -- InvalidPeriod. */
    if (lbb == 0 || lbb > n || lkc == 0 || lkc > n) return;

    const int first = first_valid;
    if (first < 0 || first >= n) return;

    const int need = (lbb > lkc) ? lbb : lkc;
    if (n - first < need) return;                /* :284 NotEnoughValidData */

    const double inv_lkc = 1.0 / (double)lkc;

    /* :1088-1093 -- the fixed linear-regression geometry. */
    const double p                 = (double)lkc;
    const double sum_x             = 0.5 * p * (p + 1.0);
    const double sum_x2            = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
    const double denom             = p * sum_x2 - sum_x * sum_x;
    const double inv_denom         = 1.0 / denom;
    const double x_last_minus_xbar = p - sum_x * inv_lkc;

    const int start_kc = first + (lkc - 1);      /* :1096 */

    /* :1120-1139 -- seed `sum_kc` over the first window. */
    double sum_kc = 0.0;
    if (start_kc < n) {
        const int s = start_kc + 1 - lkc;
        for (int j = s; j <= start_kc; ++j) sum_kc += close[j];
    }

    /* Monotonic deques over `high` (max) and `low` (min), plus the raw ring. */
    int    dq_max_idx[NEO_SQM_LKC];
    int    dq_min_idx[NEO_SQM_LKC];
    double raw_buf[NEO_SQM_LKC];
    for (int k = 0; k < lkc; ++k) { dq_max_idx[k] = 0; dq_min_idx[k] = 0; raw_buf[k] = 0.0; }

    int max_head = 0, max_len = 0, min_head = 0, min_len = 0;
    int rb_pos = 0, raw_count = 0;
    double S0 = 0.0, S1 = 0.0;
    double prev_mom = NEO_F64_NAN;   /* only the momentum column is emitted */

    for (int i = first; i < n; ++i) {
        const double hi = high[i];
        const double lo = low[i];

        /* Expire indices that have left the window -- :1199-1223. */
        while (max_len > 0) {
            const int idx = dq_max_idx[max_head];
            if (idx + lkc <= i) { max_head += 1; if (max_head == lkc) max_head = 0; max_len -= 1; }
            else break;
        }
        while (min_len > 0) {
            const int idx = dq_min_idx[min_head];
            if (idx + lkc <= i) { min_head += 1; if (min_head == lkc) min_head = 0; min_len -= 1; }
            else break;
        }

        /* Pop-back while dominated -- :1225-1249. The comparison is FALSE
         * against a NaN, exactly as on the CPU. */
        while (max_len > 0) {
            int back = max_head + max_len - 1; if (back >= lkc) back -= lkc;
            const int idx = dq_max_idx[back];
            if (high[idx] <= hi) max_len -= 1; else break;
        }
        { int pos = max_head + max_len; if (pos >= lkc) pos -= lkc;
          dq_max_idx[pos] = i; max_len += 1; }

        while (min_len > 0) {
            int back = min_head + min_len - 1; if (back >= lkc) back -= lkc;
            const int idx = dq_min_idx[back];
            if (low[idx] >= lo) min_len -= 1; else break;
        }
        { int pos = min_head + min_len; if (pos >= lkc) pos -= lkc;
          dq_min_idx[pos] = i; min_len += 1; }

        /* :1261-1263 -- slide `sum_kc`. The true-range ring feeds `squeeze`
         * only, so it is not carried here. */
        if (i > start_kc) sum_kc += close[i] - close[i - lkc];

        if (i >= start_kc) {
            const int    hi_idx  = dq_max_idx[max_head];
            const int    lo_idx  = dq_min_idx[min_head];
            const double highest = high[hi_idx];
            const double lowest  = low[lo_idx];

            const double kc_mid = sum_kc * inv_lkc;
            const double c_i    = close[i];
            /* :1317 -- plain, no fma. */
            const double raw_i  = c_i - 0.25 * (highest + lowest) - 0.5 * kc_mid;

            const double y_old = raw_buf[rb_pos];
            raw_buf[rb_pos] = raw_i;
            rb_pos += 1; if (rb_pos == lkc) rb_pos = 0;
            if (raw_count < lkc) raw_count += 1;

            double yhat_last = NEO_F64_NAN;

            if (raw_count == lkc && i == start_kc + lkc - 1) {
                /* :1329-1345 -- the ONE full pass that seeds S0/S1. */
                double s0 = 0.0, s1 = 0.0;
                int    idx = rb_pos;
                double j   = 1.0;
                for (int k = 0; k < lkc; ++k) {
                    const double y = raw_buf[idx];
                    s0 += y;
                    s1  = fma(j, y, s1);          /* :1338 */
                    j  += 1.0;
                    idx += 1; if (idx == lkc) idx = 0;
                }
                S0 = s0; S1 = s1;

                const double b    = fma(-sum_x, S0, p * S1) * inv_denom;  /* :1346 */
                const double ybar = S0 * inv_lkc;
                yhat_last = fma(b, x_last_minus_xbar, ybar);              /* :1348 */
                o[i] = yhat_last;
            } else if (raw_count == lkc) {
                /* :1370-1377 -- incremental, and deliberately NOT fma'd. */
                const double y_new  = raw_i;
                const double new_S1 = (S1 - S0) + p * y_new;
                const double new_S0 = (S0 - y_old) + y_new;
                S1 = new_S1; S0 = new_S0;

                const double b    = fma(-sum_x, S0, p * S1) * inv_denom;
                const double ybar = S0 * inv_lkc;
                yhat_last = fma(b, x_last_minus_xbar, ybar);
                o[i] = yhat_last;
            } else {
                o[i] = NEO_F64_NAN;               /* :1397 */
            }

            prev_mom = yhat_last;
        }
    }
    (void)prev_mom;   /* the signal column, which this output is not, reads it */
}
