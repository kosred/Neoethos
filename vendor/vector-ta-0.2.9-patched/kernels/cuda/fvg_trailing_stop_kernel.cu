#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ __forceinline__ bool finite3(float a, float b, float c) {
    return !isnan(a) && !isnan(b) && !isnan(c) && !isinf(a) && !isinf(b) && !isinf(c);
}


struct NeumaierAcc {
    float s, c;
    __device__ __forceinline__ void reset() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ __forceinline__ void sub(float x) { add(-x); }
    __device__ __forceinline__ float total() const { return s + c; }
};

template <int MAXL>
__device__ __forceinline__ void push_with_lookback(float* buf, int& cur_len, const int lookback, float v) {
    if (cur_len < lookback) {
        if (cur_len < MAXL) { buf[cur_len] = v; cur_len += 1; }
    } else if (lookback > 0) {
        const int L = (lookback < MAXL ? lookback : MAXL);
        #pragma unroll
        for (int k = 1; k < L; ++k) { buf[k - 1] = buf[k]; }
        buf[L - 1] = v;
        cur_len = L;
    }
}


template <int MAXL>
__device__ __forceinline__ float compact_and_avg_bull_f32_comp(float* buf, int& len, float close_v, int& new_len_out) {
    NeumaierAcc acc; acc.reset();
    int new_len = 0;
    const int L = len < MAXL ? len : MAXL;
    #pragma unroll 1
    for (int k = 0; k < L; ++k) {
        float v = buf[k];
        if (!(v > close_v)) { buf[new_len] = v; new_len += 1; acc.add(v); }
    }
    len = new_len; new_len_out = new_len;
    return (new_len > 0) ? acc.total() / (float)new_len : CUDART_NAN_F;
}


template <int MAXL>
__device__ __forceinline__ float compact_and_avg_bear_f32_comp(float* buf, int& len, float close_v, int& new_len_out) {
    NeumaierAcc acc; acc.reset();
    int new_len = 0;
    const int L = len < MAXL ? len : MAXL;
    #pragma unroll 1
    for (int k = 0; k < L; ++k) {
        float v = buf[k];
        if (!(v < close_v)) { buf[new_len] = v; new_len += 1; acc.add(v); }
    }
    len = new_len; new_len_out = new_len;
    return (new_len > 0) ? acc.total() / (float)new_len : CUDART_NAN_F;
}


template <int MAXW, bool USE_SHMEM>
struct RingAvg;

template <int MAXW>
struct RingAvg<MAXW, false> {
    float* buf;  int w, idx, count; int bad; NeumaierAcc acc; float local_buf[MAXW];
    __device__ __forceinline__ void init(int _w, float* ) {
        w=_w; idx=0; count=0; bad=0; acc.reset(); buf = local_buf;
    }
    __device__ __forceinline__ void push(float x) {
        if (count < w) {
            buf[idx] = x; if (!isfinite(x)) bad++; else acc.add(x); idx = (idx + 1 == w) ? 0 : (idx + 1); ++count;
        } else {
            float old = buf[idx]; if (!isfinite(old)) bad--; else acc.sub(old);
            buf[idx] = x; if (!isfinite(x)) bad++; else acc.add(x); idx = (idx + 1 == w) ? 0 : (idx + 1);
        }
    }
    __device__ __forceinline__ float avg() const { if (count < w || bad > 0) return CUDART_NAN_F; return acc.total() / (float)w; }
};

template <int MAXW>
struct RingAvg<MAXW, true> {
    float* buf;  int w, idx, count; int bad; NeumaierAcc acc;
    __device__ __forceinline__ void init(int _w, float* external_buf) {
        w=_w; idx=0; count=0; bad=0; acc.reset(); buf = external_buf;
    }
    __device__ __forceinline__ void push(float x) {
        if (count < w) {
            buf[idx] = x; if (!isfinite(x)) bad++; else acc.add(x); idx = (idx + 1 == w) ? 0 : (idx + 1); ++count;
        } else {
            float old = buf[idx]; if (!isfinite(old)) bad--; else acc.sub(old);
            buf[idx] = x; if (!isfinite(x)) bad++; else acc.add(x); idx = (idx + 1 == w) ? 0 : (idx + 1);
        }
    }
    __device__ __forceinline__ float avg() const { if (count < w || bad > 0) return CUDART_NAN_F; return acc.total() / (float)w; }
};


template <int MAXL>
__device__ inline float compact_and_avg_bull(float* buf, int& len, float close_v, int& new_len_out) {
    double acc = 0.0;
    int new_len = 0;
    const int L = len < MAXL ? len : MAXL;
    for (int k = 0; k < L; ++k) {
        float v = buf[k];
        if (!(v > close_v)) { buf[new_len] = v; new_len += 1; acc += (double)v; }
    }
    len = new_len; new_len_out = new_len;
    return (new_len > 0) ? (float)(acc / (double)new_len) : CUDART_NAN_F;
}
template <int MAXL>
__device__ inline float compact_and_avg_bear(float* buf, int& len, float close_v, int& new_len_out) {
    double acc = 0.0;
    int new_len = 0;
    const int L = len < MAXL ? len : MAXL;
    for (int k = 0; k < L; ++k) {
        float v = buf[k];
        if (!(v < close_v)) { buf[new_len] = v; new_len += 1; acc += (double)v; }
    }
    len = new_len; new_len_out = new_len;
    return (new_len > 0) ? (float)(acc / (double)new_len) : CUDART_NAN_F;
}
__device__ inline float sma_last_bs_over_close(const float* close, int i, int bs) {
    if (bs <= 0) return CUDART_NAN_F;
    int start = i + 1 - bs; if (start < 0) return CUDART_NAN_F;
    double s = 0.0;
    for (int j = start; j <= i; ++j) { float v = close[j]; if (isnan(v) || isinf(v)) return CUDART_NAN_F; s += (double)v; }
    return (float)(s / (double)bs);
}

template <int MAXW>
__device__ inline float disp_last_w(const float* hist, int count, int w) {
    if (w <= 0 || count < w) return CUDART_NAN_F;
    double s = 0.0;
    for (int k = 0; k < w; ++k) {
        float v = hist[(count - 1) - k]; if (isnan(v) || isinf(v)) return CUDART_NAN_F; s += (double)v; }
    return (float)(s / (double)w);
}
}


#define FVGTS_MAX_LOOKBACK 256
#define FVGTS_MAX_SMOOTH   256


extern "C" __global__ void fvg_trailing_stop_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    const int*   __restrict__ lookbacks,
    const int*   __restrict__ smoothings,
    const int*   __restrict__ resets,
    int n_combos,
    float* __restrict__ upper_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_ts_out,
    float* __restrict__ lower_ts_out,
    int use_shmem_rings,
    int smem_stride
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;


    __shared__ int first_valid_sh;
    if (threadIdx.x == 0) {
        int fv = len;
        for (int i = 0; i < len; ++i) { if (finite3(high[i], low[i], close[i])) { fv = i; break; } }
        first_valid_sh = fv;
    }
    __syncthreads();
    if (first_valid_sh >= len) {
        for (int row = tid0; row < n_combos; row += stride) {
            float* U = upper_out    + (size_t)row * len;
            float* L = lower_out    + (size_t)row * len;
            float* UT= upper_ts_out + (size_t)row * len;
            float* LT= lower_ts_out + (size_t)row * len;
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
        }
        return;
    }


    extern __shared__ float shmem[];

    for (int row = tid0; row < n_combos; row += stride) {
        const int look = lookbacks[row];
        const int w    = smoothings[row];
        const bool rst = (resets[row] != 0);

        float* U  = upper_out    + (size_t)row * len;
        float* L  = lower_out    + (size_t)row * len;
        float* UT = upper_ts_out + (size_t)row * len;
        float* LT = lower_ts_out + (size_t)row * len;

        if (look <= 0 || look > FVGTS_MAX_LOOKBACK || w <= 0 || w > FVGTS_MAX_SMOOTH) {
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
            continue;
        }


        float bull_buf[FVGTS_MAX_LOOKBACK]; int bull_len = 0;
        float bear_buf[FVGTS_MAX_LOOKBACK]; int bear_len = 0;


        float* close_ring_buf = nullptr;
        float* xbull_ring_buf = nullptr;
        float* xbear_ring_buf = nullptr;
        if (use_shmem_rings != 0) {
            const int stride_per_ring = smem_stride * blockDim.x;
            const int t_off = threadIdx.x * smem_stride;
            close_ring_buf = shmem + 0 * stride_per_ring + t_off;
            xbull_ring_buf = shmem + 1 * stride_per_ring + t_off;
            xbear_ring_buf = shmem + 2 * stride_per_ring + t_off;
        }

        RingAvg<FVGTS_MAX_SMOOTH, false> ring_close_w; ring_close_w.init(w, close_ring_buf);
        RingAvg<FVGTS_MAX_SMOOTH, false> ring_xbull_w; ring_xbull_w.init(w, xbull_ring_buf);
        RingAvg<FVGTS_MAX_SMOOTH, false> ring_xbear_w; ring_xbear_w.init(w, xbear_ring_buf);


        NeumaierAcc bs_acc_bull; bs_acc_bull.reset(); int bs_len_bull = 0; bool bs_bad_bull = false;
        NeumaierAcc bs_acc_bear; bs_acc_bear.reset(); int bs_len_bear = 0; bool bs_bad_bear = false;

        int last_bull_non_na = -1;
        int last_bear_non_na = -1;
        int os = 0;
        float ts = CUDART_NAN_F;
        float ts_prev = CUDART_NAN_F;

        for (int i = 0; i < len; ++i) {

            if (i >= 2) {
                float hi2 = high[i-2];
                float lo2 = low [i-2];
                float cm1 = close[i-1];
                float hi  = high[i];
                float lo  = low [i];
                if (finite3(hi2, lo2, cm1) && finite3(hi, lo, cm1)) {
                    if (lo > hi2 && cm1 > hi2) { push_with_lookback<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, look, hi2); }
                    if (hi < lo2 && cm1 < lo2) { push_with_lookback<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, look, lo2); }
                }
            }

            const float c = close[i];


            ring_close_w.push(c);


            int dummy_len = 0;
            float bull_avg = compact_and_avg_bull_f32_comp<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, c, dummy_len);
            float bear_avg = compact_and_avg_bear_f32_comp<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, c, dummy_len);
            if (!isnan(bull_avg)) { last_bull_non_na = i; bs_len_bull = 0; bs_bad_bull = false; bs_acc_bull.reset(); }
            if (!isnan(bear_avg)) { last_bear_non_na = i; bs_len_bear = 0; bs_bad_bear = false; bs_acc_bear.reset(); }


            float bull_sma = CUDART_NAN_F;
            float bear_sma = CUDART_NAN_F;

            if (isnan(bull_avg)) {
                if (last_bull_non_na < 0) {
                    bull_sma = isfinite(c) ? c : CUDART_NAN_F;
                } else {
                    if (bs_len_bull == 0) { bs_len_bull = 1; bs_bad_bull = !isfinite(c); bs_acc_bull.reset(); if (!bs_bad_bull) bs_acc_bull.add(c); bull_sma = bs_bad_bull ? CUDART_NAN_F : bs_acc_bull.total(); }
                    else if (bs_len_bull < w) { ++bs_len_bull; if (!isfinite(c)) bs_bad_bull = true; else bs_acc_bull.add(c); bull_sma = bs_bad_bull ? CUDART_NAN_F : (bs_acc_bull.total() / (float)bs_len_bull); }
                    else { bull_sma = ring_close_w.avg(); }
                }
            }

            if (isnan(bear_avg)) {
                if (last_bear_non_na < 0) {
                    bear_sma = isfinite(c) ? c : CUDART_NAN_F;
                } else {
                    if (bs_len_bear == 0) { bs_len_bear = 1; bs_bad_bear = !isfinite(c); bs_acc_bear.reset(); if (!bs_bad_bear) bs_acc_bear.add(c); bear_sma = bs_bad_bear ? CUDART_NAN_F : bs_acc_bear.total(); }
                    else if (bs_len_bear < w) { ++bs_len_bear; if (!isfinite(c)) bs_bad_bear = true; else bs_acc_bear.add(c); bear_sma = bs_bad_bear ? CUDART_NAN_F : (bs_acc_bear.total() / (float)bs_len_bear); }
                    else { bear_sma = ring_close_w.avg(); }
                }
            }

            const float xbull = isnan(bull_avg) ? bull_sma : bull_avg;
            const float xbear = isnan(bear_avg) ? bear_sma : bear_avg;


            ring_xbull_w.push(xbull);
            ring_xbear_w.push(xbear);
            const float bull_disp = ring_xbull_w.avg();
            const float bear_disp = ring_xbear_w.avg();

            int prev_os = os;
            if (!isnan(bear_disp) && c > bear_disp) { os = 1; }
            else if (!isnan(bull_disp) && c < bull_disp) { os = -1; }

            if (os != 0 && prev_os != 0) {
                if (os == 1 && prev_os != 1) { ts = bull_disp; }
                else if (os == -1 && prev_os != -1) { ts = bear_disp; }
                else if (os == 1)  { if (!isnan(ts)) ts = fmaxf(ts, bull_disp); }
                else if (os == -1) { if (!isnan(ts)) ts = fminf(ts, bear_disp); }
            } else {
                if (os == 1 && !isnan(ts)) ts = fmaxf(ts, bull_disp);
                if (os == -1 && !isnan(ts)) ts = fminf(ts, bear_disp);
            }

            if (rst) {
                if (os == 1) {
                    if (!isnan(ts) && c < ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bear_disp) && c > bear_disp) { ts = bull_disp; }
                } else if (os == -1) {
                    if (!isnan(ts) && c > ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bull_disp) && c < bull_disp) { ts = bear_disp; }
                }
            }

            const bool show = (!isnan(ts)) || (!isnan(ts_prev));
            const float ts_nz = !isnan(ts) ? ts : ts_prev;
            if (os == 1 && show) {
                U[i] = CUDART_NAN_F; L[i] = bull_disp; UT[i] = CUDART_NAN_F; LT[i] = ts_nz;
            } else if (os == -1 && show) {
                U[i] = bear_disp; L[i] = CUDART_NAN_F; UT[i] = ts_nz; LT[i] = CUDART_NAN_F;
            } else {
                U[i] = L[i] = UT[i] = LT[i] = CUDART_NAN_F;
            }
            ts_prev = ts;
        }


        int warm = first_valid_sh + 2 + (w - 1);
        if (warm > len) warm = len;
        for (int i = 0; i < warm; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
    }
}


extern "C" __global__ void fvg_trailing_stop_batch_shmem_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    const int*   __restrict__ lookbacks,
    const int*   __restrict__ smoothings,
    const int*   __restrict__ resets,
    int n_combos,
    float* __restrict__ upper_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_ts_out,
    float* __restrict__ lower_ts_out,
    int ,
    int smem_stride
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    __shared__ int first_valid_sh;
    if (threadIdx.x == 0) {
        int fv = len;
        for (int i = 0; i < len; ++i) { if (finite3(high[i], low[i], close[i])) { fv = i; break; } }
        first_valid_sh = fv;
    }
    __syncthreads();
    if (first_valid_sh >= len) {
        for (int row = tid0; row < n_combos; row += stride) {
            float* U = upper_out    + (size_t)row * len;
            float* L = lower_out    + (size_t)row * len;
            float* UT= upper_ts_out + (size_t)row * len;
            float* LT= lower_ts_out + (size_t)row * len;
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
        }
        return;
    }

    extern __shared__ float shmem[];
    const int stride_per_ring = smem_stride * blockDim.x;
    const int t_off = threadIdx.x * smem_stride;
    float* close_ring_buf = shmem + 0 * stride_per_ring + t_off;
    float* xbull_ring_buf = shmem + 1 * stride_per_ring + t_off;
    float* xbear_ring_buf = shmem + 2 * stride_per_ring + t_off;

    for (int row = tid0; row < n_combos; row += stride) {
        const int look = lookbacks[row];
        const int w    = smoothings[row];
        const bool rst = (resets[row] != 0);

        float* U  = upper_out    + (size_t)row * len;
        float* L  = lower_out    + (size_t)row * len;
        float* UT = upper_ts_out + (size_t)row * len;
        float* LT = lower_ts_out + (size_t)row * len;

        if (look <= 0 || look > FVGTS_MAX_LOOKBACK || w <= 0 || w > 64 || smem_stride < w) {
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
            continue;
        }

        float bull_buf[FVGTS_MAX_LOOKBACK]; int bull_len = 0;
        float bear_buf[FVGTS_MAX_LOOKBACK]; int bear_len = 0;

        RingAvg<64, true> ring_close_w; ring_close_w.init(w, close_ring_buf);
        RingAvg<64, true> ring_xbull_w; ring_xbull_w.init(w, xbull_ring_buf);
        RingAvg<64, true> ring_xbear_w; ring_xbear_w.init(w, xbear_ring_buf);

        NeumaierAcc bs_acc_bull; bs_acc_bull.reset(); int bs_len_bull = 0; bool bs_bad_bull = false;
        NeumaierAcc bs_acc_bear; bs_acc_bear.reset(); int bs_len_bear = 0; bool bs_bad_bear = false;

        int last_bull_non_na = -1;
        int last_bear_non_na = -1;
        int os = 0;
        float ts = CUDART_NAN_F;
        float ts_prev = CUDART_NAN_F;

        for (int i = 0; i < len; ++i) {
            if (i >= 2) {
                float hi2 = high[i-2];
                float lo2 = low [i-2];
                float cm1 = close[i-1];
                float hi  = high[i];
                float lo  = low [i];
                if (finite3(hi2, lo2, cm1) && finite3(hi, lo, cm1)) {
                    if (lo > hi2 && cm1 > hi2) { push_with_lookback<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, look, hi2); }
                    if (hi < lo2 && cm1 < lo2) { push_with_lookback<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, look, lo2); }
                }
            }

            const float c = close[i];
            ring_close_w.push(c);

            int dummy_len = 0;
            float bull_avg = compact_and_avg_bull_f32_comp<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, c, dummy_len);
            float bear_avg = compact_and_avg_bear_f32_comp<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, c, dummy_len);
            if (!isnan(bull_avg)) { last_bull_non_na = i; bs_len_bull = 0; bs_bad_bull = false; bs_acc_bull.reset(); }
            if (!isnan(bear_avg)) { last_bear_non_na = i; bs_len_bear = 0; bs_bad_bear = false; bs_acc_bear.reset(); }

            float bull_sma = CUDART_NAN_F;
            float bear_sma = CUDART_NAN_F;

            if (isnan(bull_avg)) {
                if (last_bull_non_na < 0) {
                    bull_sma = isfinite(c) ? c : CUDART_NAN_F;
                } else {
                    if (bs_len_bull == 0) { bs_len_bull = 1; bs_bad_bull = !isfinite(c); bs_acc_bull.reset(); if (!bs_bad_bull) bs_acc_bull.add(c); bull_sma = bs_bad_bull ? CUDART_NAN_F : bs_acc_bull.total(); }
                    else if (bs_len_bull < w) { ++bs_len_bull; if (!isfinite(c)) bs_bad_bull = true; else bs_acc_bull.add(c); bull_sma = bs_bad_bull ? CUDART_NAN_F : (bs_acc_bull.total() / (float)bs_len_bull); }
                    else { bull_sma = ring_close_w.avg(); }
                }
            }

            if (isnan(bear_avg)) {
                if (last_bear_non_na < 0) {
                    bear_sma = isfinite(c) ? c : CUDART_NAN_F;
                } else {
                    if (bs_len_bear == 0) { bs_len_bear = 1; bs_bad_bear = !isfinite(c); bs_acc_bear.reset(); if (!bs_bad_bear) bs_acc_bear.add(c); bear_sma = bs_bad_bear ? CUDART_NAN_F : bs_acc_bear.total(); }
                    else if (bs_len_bear < w) { ++bs_len_bear; if (!isfinite(c)) bs_bad_bear = true; else bs_acc_bear.add(c); bear_sma = bs_bad_bear ? CUDART_NAN_F : (bs_acc_bear.total() / (float)bs_len_bear); }
                    else { bear_sma = ring_close_w.avg(); }
                }
            }

            const float xbull = isnan(bull_avg) ? bull_sma : bull_avg;
            const float xbear = isnan(bear_avg) ? bear_sma : bear_avg;

            ring_xbull_w.push(xbull);
            ring_xbear_w.push(xbear);
            const float bull_disp = ring_xbull_w.avg();
            const float bear_disp = ring_xbear_w.avg();

            int prev_os = os;
            if (!isnan(bear_disp) && c > bear_disp) { os = 1; }
            else if (!isnan(bull_disp) && c < bull_disp) { os = -1; }

            if (os != 0 && prev_os != 0) {
                if (os == 1 && prev_os != 1) { ts = bull_disp; }
                else if (os == -1 && prev_os != -1) { ts = bear_disp; }
                else if (os == 1)  { if (!isnan(ts)) ts = fmaxf(ts, bull_disp); }
                else if (os == -1) { if (!isnan(ts)) ts = fminf(ts, bear_disp); }
            } else {
                if (os == 1 && !isnan(ts)) ts = fmaxf(ts, bull_disp);
                if (os == -1 && !isnan(ts)) ts = fminf(ts, bear_disp);
            }

            if (rst) {
                if (os == 1) {
                    if (!isnan(ts) && c < ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bear_disp) && c > bear_disp) { ts = bull_disp; }
                } else if (os == -1) {
                    if (!isnan(ts) && c > ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bull_disp) && c < bull_disp) { ts = bear_disp; }
                }
            }

            const bool show = (!isnan(ts)) || (!isnan(ts_prev));
            const float ts_nz = !isnan(ts) ? ts : ts_prev;
            if (os == 1 && show) {
                U[i] = CUDART_NAN_F; L[i] = bull_disp; UT[i] = CUDART_NAN_F; LT[i] = ts_nz;
            } else if (os == -1 && show) {
                U[i] = bear_disp; L[i] = CUDART_NAN_F; UT[i] = ts_nz; LT[i] = CUDART_NAN_F;
            } else {
                U[i] = L[i] = UT[i] = LT[i] = CUDART_NAN_F;
            }
            ts_prev = ts;
        }

        int warm = first_valid_sh + 2 + (w - 1);
        if (warm > len) warm = len;
        for (int i = 0; i < warm; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
    }
}


extern "C" __global__ void fvg_trailing_stop_batch_small_shmem_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    const int*   __restrict__ lookbacks,
    const int*   __restrict__ smoothings,
    const int*   __restrict__ resets,
    int n_combos,
    float* __restrict__ upper_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_ts_out,
    float* __restrict__ lower_ts_out,
    int ,
    int smem_stride
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    __shared__ int first_valid_sh;
    if (threadIdx.x == 0) {
        int fv = len;
        for (int i = 0; i < len; ++i) { if (finite3(high[i], low[i], close[i])) { fv = i; break; } }
        first_valid_sh = fv;
    }
    __syncthreads();
    if (first_valid_sh >= len) {
        for (int row = tid0; row < n_combos; row += stride) {
            float* U = upper_out    + (size_t)row * len;
            float* L = lower_out    + (size_t)row * len;
            float* UT= upper_ts_out + (size_t)row * len;
            float* LT= lower_ts_out + (size_t)row * len;
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
        }
        return;
    }

    extern __shared__ float shmem[];
    const int stride_per_ring = smem_stride * blockDim.x;
    const int t_off = threadIdx.x * smem_stride;
    float* close_ring_buf = shmem + 0 * stride_per_ring + t_off;
    float* xbull_ring_buf = shmem + 1 * stride_per_ring + t_off;
    float* xbear_ring_buf = shmem + 2 * stride_per_ring + t_off;

    for (int row = tid0; row < n_combos; row += stride) {
        const int look = lookbacks[row];
        const int w    = smoothings[row];
        const bool rst = (resets[row] != 0);

        float* U  = upper_out    + (size_t)row * len;
        float* L  = lower_out    + (size_t)row * len;
        float* UT = upper_ts_out + (size_t)row * len;
        float* LT = lower_ts_out + (size_t)row * len;

        if (look <= 0 || look > 32 || w <= 0 || w > 64 || smem_stride < w) {
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
            continue;
        }

        float bull_buf[32]; int bull_len = 0;
        float bear_buf[32]; int bear_len = 0;

        RingAvg<64, true> ring_close_w; ring_close_w.init(w, close_ring_buf);
        RingAvg<64, true> ring_xbull_w; ring_xbull_w.init(w, xbull_ring_buf);
        RingAvg<64, true> ring_xbear_w; ring_xbear_w.init(w, xbear_ring_buf);

        NeumaierAcc bs_acc_bull; bs_acc_bull.reset(); int bs_len_bull = 0; bool bs_bad_bull = false;
        NeumaierAcc bs_acc_bear; bs_acc_bear.reset(); int bs_len_bear = 0; bool bs_bad_bear = false;

        int last_bull_non_na = -1;
        int last_bear_non_na = -1;
        int os = 0;
        float ts = CUDART_NAN_F;
        float ts_prev = CUDART_NAN_F;

        for (int i = 0; i < len; ++i) {
            if (i >= 2) {
                float hi2 = high[i-2];
                float lo2 = low [i-2];
                float cm1 = close[i-1];
                float hi  = high[i];
                float lo  = low [i];
                if (finite3(hi2, lo2, cm1) && finite3(hi, lo, cm1)) {
                    if (lo > hi2 && cm1 > hi2) { push_with_lookback<32>(bull_buf, bull_len, look, hi2); }
                    if (hi < lo2 && cm1 < lo2) { push_with_lookback<32>(bear_buf, bear_len, look, lo2); }
                }
            }

            const float c = close[i];
            ring_close_w.push(c);

            int dummy_len = 0;
            float bull_avg = compact_and_avg_bull_f32_comp<32>(bull_buf, bull_len, c, dummy_len);
            float bear_avg = compact_and_avg_bear_f32_comp<32>(bear_buf, bear_len, c, dummy_len);
            if (!isnan(bull_avg)) { last_bull_non_na = i; bs_len_bull = 0; bs_bad_bull = false; bs_acc_bull.reset(); }
            if (!isnan(bear_avg)) { last_bear_non_na = i; bs_len_bear = 0; bs_bad_bear = false; bs_acc_bear.reset(); }

            float bull_sma = CUDART_NAN_F;
            float bear_sma = CUDART_NAN_F;

            if (isnan(bull_avg)) {
                if (last_bull_non_na < 0) {
                    bull_sma = isfinite(c) ? c : CUDART_NAN_F;
                } else {
                    if (bs_len_bull == 0) { bs_len_bull = 1; bs_bad_bull = !isfinite(c); bs_acc_bull.reset(); if (!bs_bad_bull) bs_acc_bull.add(c); bull_sma = bs_bad_bull ? CUDART_NAN_F : bs_acc_bull.total(); }
                    else if (bs_len_bull < w) { ++bs_len_bull; if (!isfinite(c)) bs_bad_bull = true; else bs_acc_bull.add(c); bull_sma = bs_bad_bull ? CUDART_NAN_F : (bs_acc_bull.total() / (float)bs_len_bull); }
                    else { bull_sma = ring_close_w.avg(); }
                }
            }

            if (isnan(bear_avg)) {
                if (last_bear_non_na < 0) {
                    bear_sma = isfinite(c) ? c : CUDART_NAN_F;
                } else {
                    if (bs_len_bear == 0) { bs_len_bear = 1; bs_bad_bear = !isfinite(c); bs_acc_bear.reset(); if (!bs_bad_bear) bs_acc_bear.add(c); bear_sma = bs_bad_bear ? CUDART_NAN_F : bs_acc_bear.total(); }
                    else if (bs_len_bear < w) { ++bs_len_bear; if (!isfinite(c)) bs_bad_bear = true; else bs_acc_bear.add(c); bear_sma = bs_bad_bear ? CUDART_NAN_F : (bs_acc_bear.total() / (float)bs_len_bear); }
                    else { bear_sma = ring_close_w.avg(); }
                }
            }

            const float xbull = isnan(bull_avg) ? bull_sma : bull_avg;
            const float xbear = isnan(bear_avg) ? bear_sma : bear_avg;

            ring_xbull_w.push(xbull);
            ring_xbear_w.push(xbear);
            const float bull_disp = ring_xbull_w.avg();
            const float bear_disp = ring_xbear_w.avg();

            int prev_os = os;
            if (!isnan(bear_disp) && c > bear_disp) { os = 1; }
            else if (!isnan(bull_disp) && c < bull_disp) { os = -1; }

            if (os != 0 && prev_os != 0) {
                if (os == 1 && prev_os != 1) { ts = bull_disp; }
                else if (os == -1 && prev_os != -1) { ts = bear_disp; }
                else if (os == 1)  { if (!isnan(ts)) ts = fmaxf(ts, bull_disp); }
                else if (os == -1) { if (!isnan(ts)) ts = fminf(ts, bear_disp); }
            } else {
                if (os == 1 && !isnan(ts)) ts = fmaxf(ts, bull_disp);
                if (os == -1 && !isnan(ts)) ts = fminf(ts, bear_disp);
            }

            if (rst) {
                if (os == 1) {
                    if (!isnan(ts) && c < ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bear_disp) && c > bear_disp) { ts = bull_disp; }
                } else if (os == -1) {
                    if (!isnan(ts) && c > ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bull_disp) && c < bull_disp) { ts = bear_disp; }
                }
            }

            const bool show = (!isnan(ts)) || (!isnan(ts_prev));
            const float ts_nz = !isnan(ts) ? ts : ts_prev;
            if (os == 1 && show) {
                U[i] = CUDART_NAN_F; L[i] = bull_disp; UT[i] = CUDART_NAN_F; LT[i] = ts_nz;
            } else if (os == -1 && show) {
                U[i] = bear_disp; L[i] = CUDART_NAN_F; UT[i] = ts_nz; LT[i] = CUDART_NAN_F;
            } else {
                U[i] = L[i] = UT[i] = LT[i] = CUDART_NAN_F;
            }
            ts_prev = ts;
        }

        int warm = first_valid_sh + 2 + (w - 1);
        if (warm > len) warm = len;
        for (int i = 0; i < warm; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
    }
}


extern "C" __global__ void fvg_trailing_stop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int look,
    int w,
    int reset_on_cross,
    float* __restrict__ upper_tm_out,
    float* __restrict__ lower_tm_out,
    float* __restrict__ upper_ts_tm_out,
    float* __restrict__ lower_ts_tm_out
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;


    for (int t = 0; t < rows; ++t) {
        int idx = t * cols + s;
        upper_tm_out[idx] = CUDART_NAN_F;
        lower_tm_out[idx] = CUDART_NAN_F;
        upper_ts_tm_out[idx] = CUDART_NAN_F;
        lower_ts_tm_out[idx] = CUDART_NAN_F;
    }

    if (look <= 0 || look > FVGTS_MAX_LOOKBACK || w <= 0 || w > FVGTS_MAX_SMOOTH) return;


    int first_valid = rows;
    for (int t = 0; t < rows; ++t) {
        int idx = t * cols + s;
        if (finite3(high_tm[idx], low_tm[idx], close_tm[idx])) { first_valid = t; break; }
    }
    if (first_valid >= rows) return;

    float bull_buf[FVGTS_MAX_LOOKBACK]; int bull_len = 0;
    float bear_buf[FVGTS_MAX_LOOKBACK]; int bear_len = 0;

    double bull_ring_vals[FVGTS_MAX_SMOOTH]; bool bull_ring_nan[FVGTS_MAX_SMOOTH];
    double bear_ring_vals[FVGTS_MAX_SMOOTH]; bool bear_ring_nan[FVGTS_MAX_SMOOTH];
    int bull_ring_count = 0, bear_ring_count = 0;
    int bull_ring_idx = 0,   bear_ring_idx = 0;
    int bull_nan_cnt = 0,    bear_nan_cnt = 0;
    double bull_sum = 0.0,   bear_sum = 0.0;
    int last_bull_non_na = -1;
    int last_bear_non_na = -1;
    int os = 0; float ts = CUDART_NAN_F; float ts_prev = CUDART_NAN_F;
    const bool rst = (reset_on_cross != 0);

    for (int t = 0; t < rows; ++t) {
        int idx = t * cols + s;

        if (t >= 2) {
            int im2 = (t-2) * cols + s;
            int im1 = (t-1) * cols + s;
            float hi2 = high_tm[im2]; float lo2 = low_tm[im2]; float cm1 = close_tm[im1];
            float hi = high_tm[idx]; float lo = low_tm[idx];
            if (finite3(hi2, lo2, cm1) && finite3(hi, lo, cm1)) {
                if (lo > hi2 && cm1 > hi2) {
                    push_with_lookback<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, look, hi2);
                }
                if (hi < lo2 && cm1 < lo2) {
                    push_with_lookback<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, look, lo2);
                }
            }
        }

        float c = close_tm[idx];
        int dummy=0;
        float bull_avg = compact_and_avg_bull<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, c, dummy);
        float bear_avg = compact_and_avg_bear<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, c, dummy);
        if (!isnan(bull_avg)) last_bull_non_na = t;
        if (!isnan(bear_avg)) last_bear_non_na = t;

        int bull_bs = (!isnan(bull_avg)) ? 1 : ((last_bull_non_na >= 0) ? min(max(t - last_bull_non_na, 1), w) : 1);
        int bear_bs = (!isnan(bear_avg)) ? 1 : ((last_bear_non_na >= 0) ? min(max(t - last_bear_non_na, 1), w) : 1);
        float bull_sma = isnan(bull_avg) ? sma_last_bs_over_close(close_tm + s, t, bull_bs) : CUDART_NAN_F;


        if (isnan(bull_avg)) {

            double ss = 0.0; bool bad=false; int start = t + 1 - bull_bs;
            if (start >= 0) {
                for (int j = start; j <= t; ++j) { float v = close_tm[j * cols + s]; if (isnan(v) || isinf(v)) { bad=true; break; } ss += (double)v; }
                bull_sma = bad ? CUDART_NAN_F : (float)(ss / (double)bull_bs);
            } else { bull_sma = CUDART_NAN_F; }
        }

        float bear_sma = CUDART_NAN_F;
        if (isnan(bear_avg)) {
            double ss = 0.0; bool bad=false; int start = t + 1 - bear_bs;
            if (start >= 0) {
                for (int j = start; j <= t; ++j) { float v = close_tm[j * cols + s]; if (isnan(v) || isinf(v)) { bad=true; break; } ss += (double)v; }
                bear_sma = bad ? CUDART_NAN_F : (float)(ss / (double)bear_bs);
            }
        }

        const float xbull = isnan(bull_avg) ? bull_sma : bull_avg;
        const float xbear = isnan(bear_avg) ? bear_sma : bear_avg;


        if (bull_ring_count < w) {
            const bool is_nan = isnan(xbull);
            bull_ring_nan[bull_ring_count] = is_nan;
            bull_ring_vals[bull_ring_count] = is_nan ? 0.0 : (double)xbull;
            if (is_nan) { bull_nan_cnt += 1; } else { bull_sum += (double)xbull; }
            bull_ring_count += 1;
        } else {
            int idx = bull_ring_idx;
            if (bull_ring_nan[idx]) { bull_nan_cnt -= 1; } else { bull_sum -= bull_ring_vals[idx]; }
            const bool is_nan = isnan(xbull);
            bull_ring_nan[idx] = is_nan;
            if (is_nan) { bull_ring_vals[idx] = 0.0; bull_nan_cnt += 1; }
            else { bull_ring_vals[idx] = (double)xbull; bull_sum += (double)xbull; }
            bull_ring_idx = (idx + 1 == w) ? 0 : (idx + 1);
        }


        if (bear_ring_count < w) {
            const bool is_nan = isnan(xbear);
            bear_ring_nan[bear_ring_count] = is_nan;
            bear_ring_vals[bear_ring_count] = is_nan ? 0.0 : (double)xbear;
            if (is_nan) { bear_nan_cnt += 1; } else { bear_sum += (double)xbear; }
            bear_ring_count += 1;
        } else {
            int idx = bear_ring_idx;
            if (bear_ring_nan[idx]) { bear_nan_cnt -= 1; } else { bear_sum -= bear_ring_vals[idx]; }
            const bool is_nan = isnan(xbear);
            bear_ring_nan[idx] = is_nan;
            if (is_nan) { bear_ring_vals[idx] = 0.0; bear_nan_cnt += 1; }
            else { bear_ring_vals[idx] = (double)xbear; bear_sum += (double)xbear; }
            bear_ring_idx = (idx + 1 == w) ? 0 : (idx + 1);
        }

        const float bull_disp = (bull_ring_count >= w && bull_nan_cnt == 0) ? (float)(bull_sum / (double)w) : CUDART_NAN_F;
        const float bear_disp = (bear_ring_count >= w && bear_nan_cnt == 0) ? (float)(bear_sum / (double)w) : CUDART_NAN_F;

        int prev_os = os;
        if (!isnan(bear_disp) && c > bear_disp) { os = 1; }
        else if (!isnan(bull_disp) && c < bull_disp) { os = -1; }

        if (os != 0 && prev_os != 0) {
            if (os == 1 && prev_os != 1) { ts = bull_disp; }
            else if (os == -1 && prev_os != -1) { ts = bear_disp; }
            else if (os == 1) { if (!isnan(ts)) ts = fmaxf(ts, bull_disp); }
            else if (os == -1) { if (!isnan(ts)) ts = fminf(ts, bear_disp); }
        } else {
            if (os == 1 && !isnan(ts)) ts = fmaxf(ts, bull_disp);
            if (os == -1 && !isnan(ts)) ts = fminf(ts, bear_disp);
        }

        if (rst) {
            if (os == 1) {
                if (!isnan(ts) && c < ts) { ts = CUDART_NAN_F; }
                else if (isnan(ts) && !isnan(bear_disp) && c > bear_disp) { ts = bull_disp; }
            } else if (os == -1) {
                if (!isnan(ts) && c > ts) { ts = CUDART_NAN_F; }
                else if (isnan(ts) && !isnan(bull_disp) && c < bull_disp) { ts = bear_disp; }
            }
        }

        bool show = (!isnan(ts)) || (!isnan(ts_prev));
        float ts_nz = !isnan(ts) ? ts : ts_prev;
        if (os == 1 && show) {
            upper_tm_out[idx] = CUDART_NAN_F; lower_tm_out[idx] = bull_disp; upper_ts_tm_out[idx] = CUDART_NAN_F; lower_ts_tm_out[idx] = ts_nz;
        } else if (os == -1 && show) {
            upper_tm_out[idx] = bear_disp; lower_tm_out[idx] = CUDART_NAN_F; upper_ts_tm_out[idx] = ts_nz; lower_ts_tm_out[idx] = CUDART_NAN_F;
        } else {

        }
        ts_prev = ts;
    }


    int warm = first_valid + 2 + (w - 1);
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) {
        int idx = t * cols + s;
        upper_tm_out[idx] = lower_tm_out[idx] = upper_ts_tm_out[idx] = lower_ts_tm_out[idx] = CUDART_NAN_F;
    }
}


// ===========================================================================
// NEOETHOS exact f64 shared authority
//
// CPU reference: fvg_ts_scalar (src/indicators/fvg_trailing_stop.rs:152-466).
// At the batch defaults (lookback 5, smoothing 9, reset_on_cross false)
// fvg_ts_compute_into routes to fvg_ts_scalar_default_5_9; that
// function is fvg_ts_scalar with the two windows constant-folded and an
// ALL_VALID short-circuit on a PREDICATE, not on any arithmetic, so one body
// reproduces both.
//
// The canonical full route emits [upper, lower, upper_ts, lower_ts] in exactly
// that order. The preserved primary ABI emits upper only and interprets its
// period vector as the admitted smoothing_length, with lookback 5 and reset
// false. Both entry points call the same row below.
//
// FIRST-VALID: Ignored, and that is the contract rather than a shrug. The
// batch calls fvg_trailing_stop_with_kernel, which allocates with
// alloc_uninit_f64 and applies NO warmup prefix at all -- the prefix in
// fvg_trailing_stop_into_slices is on a different entry point the
// batch does not take. The loop runs from bar 0.
//
// SHAPE: one thread per combo walking bars ASCENDING. Two unmitigated-gap
// buffers are compacted in place at every bar, two runtime-sized displacement
// rings carry a running sum with a NaN count, and the trailing stop is a
// ratchet whose value reads its own previous value. Nothing here is
// bar-parallel.
//
// NaN SEMANTICS: the CPU ratchet is bull_disp.max(t) / bear_disp.min(t)
// (:719-727), which is f64::max -- it RETURNS THE NON-NaN OPERAND. bull_disp
// is NaN whenever the ring is short or carries a NaN, so an if-chain here
// would let the NaN survive and poison every later bar of the carry. fmax and
// fmin are used instead.
// ===========================================================================

#define FVG_TS_NEO_LOOKBACK 5
#define FVG_TS_NEO_MAX_SMOOTHING 200

static __forceinline__ __device__ double fvg_ts_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

static __forceinline__ __device__
void fvg_trailing_stop_row_f64(const double* __restrict__ high,
                               const double* __restrict__ low,
                               const double* __restrict__ close,
                               int n,
                               int lookback,
                               int w,
                               bool reset_on_cross,
                               double* __restrict__ bull_buf,
                               double* __restrict__ bear_buf,
                               double* __restrict__ bull_ring_vals,
                               double* __restrict__ bear_ring_vals,
                               int* __restrict__ bull_ring_nan,
                               int* __restrict__ bear_ring_nan,
                               double* __restrict__ upper_row,
                               double* __restrict__ lower_row,
                               double* __restrict__ upper_ts_row,
                               double* __restrict__ lower_ts_row) {
    if (n <= 0) return;
    const double nn = fvg_ts_neo_qnan();

    // fvg_ts_prepare, :895-917. first_valid_ohlc_status returns usize::MAX
    // when every bar of the triple is NaN, which is AllValuesNaN.
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(high[i]) && !isnan(low[i]) && !isnan(close[i])) { first = i; break; }
    }
    const int need = 2 + (w > 0 ? w - 1 : 0);
    if (lookback <= 0 || w <= 0 || first < 0 || (n - first) < need) {
        for (int i = 0; i < n; ++i) {
            if (upper_row != nullptr) upper_row[i] = nn;
            if (lower_row != nullptr) lower_row[i] = nn;
            if (upper_ts_row != nullptr) upper_ts_row[i] = nn;
            if (lower_ts_row != nullptr) lower_ts_row[i] = nn;
        }
        return;
    }

    int bull_len = 0, bear_len = 0;

    int last_bull_non_na = -1;
    int last_bear_non_na = -1;

    double bull_sum = 0.0, bear_sum = 0.0;
    int bull_nan_cnt = 0, bear_nan_cnt = 0;
    int bull_ring_count = 0, bear_ring_count = 0;
    int bull_ring_idx = 0, bear_ring_idx = 0;

    int os = 0;          // 0 == None, 1 / -1 as in the CPU Option<i8>
    bool ts_some = false;
    double ts_val = 0.0;
    bool ts_prev_some = false;
    double ts_prev_val = 0.0;

    for (int i = 0; i < n; ++i) {
        // :219-241 -- record an unmitigated gap.
        if (i >= 2 && !isnan(high[i - 2]) && !isnan(low[i - 2]) && !isnan(close[i - 1])) {
            if (low[i] > high[i - 2] && close[i - 1] > high[i - 2]) {
                if (bull_len < lookback) {
                    bull_buf[bull_len] = high[i - 2];
                    bull_len += 1;
                } else {
                    for (int k = 1; k < lookback; ++k) bull_buf[k - 1] = bull_buf[k];
                    bull_buf[lookback - 1] = high[i - 2];
                }
            }
            if (high[i] < low[i - 2] && close[i - 1] < low[i - 2]) {
                if (bear_len < lookback) {
                    bear_buf[bear_len] = low[i - 2];
                    bear_len += 1;
                } else {
                    for (int k = 1; k < lookback; ++k) bear_buf[k - 1] = bear_buf[k];
                    bear_buf[lookback - 1] = low[i - 2];
                }
            }
        }

        const double c = close[i];

        // :245-266 -- compact in place, summing the survivors in order.
        int new_bull_len = 0;
        double bull_acc = 0.0;
        for (int k = 0; k < bull_len; ++k) {
            const double v = bull_buf[k];
            if (c >= v) {
                bull_buf[new_bull_len] = v;
                new_bull_len += 1;
                bull_acc += v;
            }
        }
        bull_len = new_bull_len;

        int new_bear_len = 0;
        double bear_acc = 0.0;
        for (int k = 0; k < bear_len; ++k) {
            const double v = bear_buf[k];
            if (c <= v) {
                bear_buf[new_bear_len] = v;
                new_bear_len += 1;
                bear_acc += v;
            }
        }
        bear_len = new_bear_len;

        const double bull_avg = (bull_len > 0) ? (bull_acc / (double)bull_len) : nn;
        const double bear_avg = (bear_len > 0) ? (bear_acc / (double)bear_len) : nn;

        if (!isnan(bull_avg)) last_bull_non_na = i;
        if (!isnan(bear_avg)) last_bear_non_na = i;

        // :285-308 -- the fallback SMA length.
        int bull_bs;
        if (isnan(bull_avg)) {
            if (last_bull_non_na >= 0) {
                int d = i - last_bull_non_na;
                if (d < 1) d = 1;
                if (d > w) d = w;
                bull_bs = d;
            } else {
                bull_bs = 1;
            }
        } else {
            bull_bs = 1;
        }
        int bear_bs;
        if (isnan(bear_avg)) {
            if (last_bear_non_na >= 0) {
                int d = i - last_bear_non_na;
                if (d < 1) d = 1;
                if (d > w) d = w;
                bear_bs = d;
            } else {
                bear_bs = 1;
            }
        } else {
            bear_bs = 1;
        }

        double bull_sma = nn;
        if (isnan(bull_avg) && (i + 1) >= bull_bs) {
            double s = 0.0;
            const int start = i + 1 - bull_bs;
            for (int j = start; j <= i; ++j) s += close[j];
            bull_sma = s / (double)bull_bs;
        }
        double bear_sma = nn;
        if (isnan(bear_avg) && (i + 1) >= bear_bs) {
            double s = 0.0;
            const int start = i + 1 - bear_bs;
            for (int j = start; j <= i; ++j) s += close[j];
            bear_sma = s / (double)bear_bs;
        }

        const double x_bull = !isnan(bull_avg) ? bull_avg : bull_sma;
        const double x_bear = !isnan(bear_avg) ? bear_avg : bear_sma;

        // :332-392 -- the displacement rings, with an explicit NaN count so a
        // NaN never enters the running sum.
        if (bull_ring_count < w) {
            const bool is_nan = isnan(x_bull);
            bull_ring_nan[bull_ring_count] = is_nan;
            bull_ring_vals[bull_ring_count] = is_nan ? 0.0 : x_bull;
            if (is_nan) bull_nan_cnt += 1; else bull_sum += x_bull;
            bull_ring_count += 1;
        } else {
            const int idx = bull_ring_idx;
            if (bull_ring_nan[idx]) bull_nan_cnt -= 1;
            else bull_sum -= bull_ring_vals[idx];
            const bool is_nan = isnan(x_bull);
            bull_ring_nan[idx] = is_nan;
            if (is_nan) { bull_ring_vals[idx] = 0.0; bull_nan_cnt += 1; }
            else { bull_ring_vals[idx] = x_bull; bull_sum += x_bull; }
            bull_ring_idx = (idx + 1 == w) ? 0 : (idx + 1);
        }

        if (bear_ring_count < w) {
            const bool is_nan = isnan(x_bear);
            bear_ring_nan[bear_ring_count] = is_nan;
            bear_ring_vals[bear_ring_count] = is_nan ? 0.0 : x_bear;
            if (is_nan) bear_nan_cnt += 1; else bear_sum += x_bear;
            bear_ring_count += 1;
        } else {
            const int idx = bear_ring_idx;
            if (bear_ring_nan[idx]) bear_nan_cnt -= 1;
            else bear_sum -= bear_ring_vals[idx];
            const bool is_nan = isnan(x_bear);
            bear_ring_nan[idx] = is_nan;
            if (is_nan) { bear_ring_vals[idx] = 0.0; bear_nan_cnt += 1; }
            else { bear_ring_vals[idx] = x_bear; bear_sum += x_bear; }
            bear_ring_idx = (idx + 1 == w) ? 0 : (idx + 1);
        }

        const double bull_disp =
            (bull_ring_count >= w && bull_nan_cnt == 0) ? (bull_sum / (double)w) : nn;
        const double bear_disp =
            (bear_ring_count >= w && bear_nan_cnt == 0) ? (bear_sum / (double)w) : nn;

        // :407-419 -- the sticky trend flag.
        const int prev_os = os;
        int next_os;
        if (!isnan(bear_disp) && c > bear_disp) next_os = 1;
        else if (!isnan(bull_disp) && c < bull_disp) next_os = -1;
        else next_os = os;
        os = next_os;

        // :421-446 -- the ratchet. fmax / fmin, never an if-chain: see the
        // header.
        if (os != 0 && prev_os != 0) {
            if (os == 1 && prev_os != 1) {
                ts_some = true; ts_val = bull_disp;
            } else if (os == -1 && prev_os != -1) {
                ts_some = true; ts_val = bear_disp;
            } else if (os == 1) {
                if (ts_some) ts_val = fmax(bull_disp, ts_val);
            } else if (os == -1) {
                if (ts_some) ts_val = fmin(bear_disp, ts_val);
            }
        } else {
            if (os == 1 && ts_some) ts_val = fmax(bull_disp, ts_val);
            if (os == -1 && ts_some) ts_val = fmin(bear_disp, ts_val);
        }

        // :448-466 -- unreachable at the batch default, kept faithful.
        if (reset_on_cross) {
            if (os == 1) {
                if (ts_some) {
                    if (c < ts_val) ts_some = false;
                } else if (!isnan(bear_disp) && c > bear_disp) {
                    ts_some = true; ts_val = bull_disp;
                }
            } else if (os == -1) {
                if (ts_some) {
                    if (c > ts_val) ts_some = false;
                } else if (!isnan(bull_disp) && c < bull_disp) {
                    ts_some = true; ts_val = bear_disp;
                }
            }
        }

        // :444-465 -- all four canonical outputs in registry order.
        const bool show = ts_some || ts_prev_some;
        const double ts_nz = ts_some ? ts_val : ts_prev_val;
        if (os == 1 && show) {
            if (upper_row != nullptr) upper_row[i] = nn;
            if (lower_row != nullptr) lower_row[i] = bull_disp;
            if (upper_ts_row != nullptr) upper_ts_row[i] = nn;
            if (lower_ts_row != nullptr) lower_ts_row[i] = ts_nz;
        } else if (os == -1 && show) {
            if (upper_row != nullptr) upper_row[i] = bear_disp;
            if (lower_row != nullptr) lower_row[i] = nn;
            if (upper_ts_row != nullptr) upper_ts_row[i] = ts_nz;
            if (lower_ts_row != nullptr) lower_ts_row[i] = nn;
        } else {
            if (upper_row != nullptr) upper_row[i] = nn;
            if (lower_row != nullptr) lower_row[i] = nn;
            if (upper_ts_row != nullptr) upper_ts_row[i] = nn;
            if (lower_ts_row != nullptr) lower_ts_row[i] = nn;
        }

        ts_prev_some = ts_some;
        ts_prev_val = ts_val;
    }
}

extern "C" __global__
void fvg_trailing_stop_outputs_f64(const double* __restrict__ high,
                                   const double* __restrict__ low,
                                   const double* __restrict__ close,
                                   int n,
                                   const int* __restrict__ lookbacks,
                                   const int* __restrict__ smoothing_lengths,
                                   const int* __restrict__ reset_on_cross,
                                   int n_combos,
                                   int max_lookback,
                                   int max_smoothing,
                                   double* __restrict__ bull_gap_scratch,
                                   double* __restrict__ bear_gap_scratch,
                                   double* __restrict__ bull_ring_scratch,
                                   double* __restrict__ bear_ring_scratch,
                                   int* __restrict__ bull_ring_nan_scratch,
                                   int* __restrict__ bear_ring_nan_scratch,
                                   double* __restrict__ upper_out,
                                   double* __restrict__ lower_out,
                                   double* __restrict__ upper_ts_out,
                                   double* __restrict__ lower_ts_out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const int lookback = lookbacks[combo];
    const int smoothing_length = smoothing_lengths[combo];
    double* __restrict__ upper_row = upper_out + (size_t)combo * (size_t)n;
    double* __restrict__ lower_row = lower_out + (size_t)combo * (size_t)n;
    double* __restrict__ upper_ts_row = upper_ts_out + (size_t)combo * (size_t)n;
    double* __restrict__ lower_ts_row = lower_ts_out + (size_t)combo * (size_t)n;
    if (lookback <= 0 || lookback > max_lookback ||
        smoothing_length <= 0 || smoothing_length > max_smoothing) {
        const double nn = fvg_ts_neo_qnan();
        for (int i = 0; i < n; ++i) {
            upper_row[i] = nn;
            lower_row[i] = nn;
            upper_ts_row[i] = nn;
            lower_ts_row[i] = nn;
        }
        return;
    }

    fvg_trailing_stop_row_f64(
        high,
        low,
        close,
        n,
        lookback,
        smoothing_length,
        reset_on_cross[combo] != 0,
        bull_gap_scratch + (size_t)combo * (size_t)max_lookback,
        bear_gap_scratch + (size_t)combo * (size_t)max_lookback,
        bull_ring_scratch + (size_t)combo * (size_t)max_smoothing,
        bear_ring_scratch + (size_t)combo * (size_t)max_smoothing,
        bull_ring_nan_scratch + (size_t)combo * (size_t)max_smoothing,
        bear_ring_nan_scratch + (size_t)combo * (size_t)max_smoothing,
        upper_row,
        lower_row,
        upper_ts_row,
        lower_ts_row);
}

// Preserved generic primary ABI. Canonical production bypasses this entry
// point for the four-output route, but direct f64 callers retain the exact
// symbol/signature and now receive truthful smoothing_length rows.
extern "C" __global__
void fvg_trailing_stop_neo_batch_f64(const double* __restrict__ high,
                                     const double* __restrict__ low,
                                     const double* __restrict__ close,
                                     int n,
                                     const int* __restrict__ periods,
                                     int n_combos,
                                     int first_valid,
                                     double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;
    (void)first_valid;

    double* __restrict__ upper_row = out + (size_t)combo * (size_t)n;
    const int smoothing_length = periods[combo];
    if (smoothing_length <= 0 || smoothing_length > FVG_TS_NEO_MAX_SMOOTHING) {
        const double nn = fvg_ts_neo_qnan();
        for (int i = 0; i < n; ++i) upper_row[i] = nn;
        return;
    }

    double bull_buf[FVG_TS_NEO_LOOKBACK];
    double bear_buf[FVG_TS_NEO_LOOKBACK];
    double bull_ring_vals[FVG_TS_NEO_MAX_SMOOTHING];
    double bear_ring_vals[FVG_TS_NEO_MAX_SMOOTHING];
    int bull_ring_nan[FVG_TS_NEO_MAX_SMOOTHING];
    int bear_ring_nan[FVG_TS_NEO_MAX_SMOOTHING];
    fvg_trailing_stop_row_f64(
        high,
        low,
        close,
        n,
        FVG_TS_NEO_LOOKBACK,
        smoothing_length,
        false,
        bull_buf,
        bear_buf,
        bull_ring_vals,
        bear_ring_vals,
        bull_ring_nan,
        bear_ring_nan,
        upper_row,
        nullptr,
        nullptr,
        nullptr);
}
