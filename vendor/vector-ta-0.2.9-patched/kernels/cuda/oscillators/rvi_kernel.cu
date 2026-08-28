#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef RVI_BLOCK_X
#define RVI_BLOCK_X 256
#endif


static __device__ __forceinline__ float smooth_sma_push(
    float x,
    float* __restrict__ ring,
    int* __restrict__ head,
    int* __restrict__ count,
    int ma_len,
    double* __restrict__ sum,
    double inv_m)
{
    if (!isfinite(x)) {
        *sum = 0.0;
        *count = 0;
        *head = 0;
        return NAN;
    }
    if (*count < ma_len) {
        ring[*head] = x;
        *sum += (double)x;
        *head = (*head + 1 == ma_len) ? 0 : *head + 1;
        (*count)++;
        if (*count == ma_len) {
            return (float)((*sum) * inv_m);
        } else {
            return NAN;
        }
    } else {
        const float old = ring[*head];
        ring[*head] = x;
        *head = (*head + 1 == ma_len) ? 0 : *head + 1;
        *sum += (double)x - (double)old;
        return (float)((*sum) * inv_m);
    }
}

static __device__ __forceinline__ float smooth_ema_push(
    float x,
    bool* __restrict__ started,
    double* __restrict__ seed_sum,
    int* __restrict__ seed_cnt,
    int ma_len,
    double inv_m,
    double alpha,
    double one_m_alpha,
    double* __restrict__ prev)
{
    if (!isfinite(x)) {

        *started = false;
        *seed_sum = 0.0;
        *seed_cnt = 0;
        return NAN;
    }
    if (!*started) {
        *seed_sum += (double)x;
        (*seed_cnt)++;
        if (*seed_cnt == ma_len) {
            *prev = (*seed_sum) * inv_m;
            *started = true;
            return (float)(*prev);
        }
        return NAN;
    } else {
        *prev = fma(one_m_alpha, *prev, alpha * (double)x);
        return (float)(*prev);
    }
}


static __device__ __forceinline__ double smooth_ema_push_d(
    double x,
    bool* __restrict__ started,
    double* __restrict__ seed_sum,
    int* __restrict__ seed_cnt,
    int ma_len,
    double inv_m,
    double alpha,
    double one_m_alpha,
    double* __restrict__ prev)
{
    if (!isfinite(x)) {
        *started = false; *seed_sum = 0.0; *seed_cnt = 0; return NAN;
    }
    if (!*started) {
        *seed_sum += x;
        (*seed_cnt)++;
        if (*seed_cnt == ma_len) {
            *prev = (*seed_sum) * inv_m;
            *started = true;
            return *prev;
        }
        return NAN;
    } else {
        *prev = fma(one_m_alpha, *prev, alpha * x);
        return *prev;
    }
}


static __device__ __forceinline__ void kahan_add(float x, float &sum, float &comp) {
    float t = sum + x;
    if (fabsf(sum) >= fabsf(x)) comp += (sum - t) + x;
    else                        comp += (x   - t) + sum;
    sum = t;
}

static __device__ __forceinline__ void kahan_add_diff(float x_add_minus_y, float &sum, float &comp) {
    kahan_add(x_add_minus_y, sum, comp);
}


extern "C" __global__
void rvi_segprefix_f32(const float* __restrict__ prices,
                       int len,
                       float* __restrict__ pref,
                       float* __restrict__ pref2,
                       int*   __restrict__ runlen)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float s = 0.0f, q = 0.0f, cs = 0.0f, cq = 0.0f;
    int   r = 0;

    float prev = NAN;
    for (int i = 0; i < len; ++i) {
        const float x = prices[i];
        if (!isfinite(x)) {
            s = q = cs = cq = 0.0f;
            r = 0; prev = NAN;
            pref[i] = 0.0f; pref2[i] = 0.0f; runlen[i] = 0;
            continue;
        }
        if (isfinite(prev)) {
            kahan_add(x, s, cs);
            const float xx = x * x;
            kahan_add(xx, q, cq);
            r += 1;
        } else {
            s = x; q = x * x; cs = cq = 0.0f; r = 1;
        }
        pref[i]  = s + cs;
        pref2[i] = q + cq;
        runlen[i]= r;
        prev = x;
    }
}


static __device__ __forceinline__ float smooth_sma_push_f32(
    float x,
    float* __restrict__ ring,
    int* __restrict__ head,
    int* __restrict__ count,
    int ma_len,
    float* __restrict__ sum,
    float* __restrict__ comp,
    float inv_m)
{
    if (!isfinite(x)) {
        *sum = 0.0f; *comp = 0.0f; *count = 0; *head = 0; return NAN;
    }
    if (*count < ma_len) {
        ring[*head] = x;
        kahan_add(x, *sum, *comp);
        *head = (*head + 1 == ma_len) ? 0 : *head + 1;
        (*count)++;
        return (*count == ma_len) ? (*sum) * inv_m : NAN;
    } else {
        const float old = ring[*head];
        ring[*head] = x;
        *head = (*head + 1 == ma_len) ? 0 : *head + 1;
        kahan_add_diff(x - old, *sum, *comp);
        return (*sum) * inv_m;
    }
}

static __device__ __forceinline__ float smooth_ema_push_f32(
    float x,
    bool* __restrict__ started,
    float* __restrict__ seed_sum,
    float* __restrict__ seed_comp,
    int* __restrict__ seed_cnt,
    int ma_len,
    float inv_m,
    float alpha,
    float one_m_alpha,
    float* __restrict__ prev)
{
    if (!isfinite(x)) {
        *started = false; *seed_sum = 0.0f; *seed_comp = 0.0f; *seed_cnt = 0; return NAN;
    }
    if (!*started) {
        kahan_add(x, *seed_sum, *seed_comp);
        (*seed_cnt)++;
        if (*seed_cnt == ma_len) {
            *prev = (*seed_sum) * inv_m;
            *started = true;
            return *prev;
        }
        return NAN;
    } else {
        *prev = fmaf(one_m_alpha, *prev, alpha * x);
        return *prev;
    }
}


extern "C" __global__
void rvi_batch_stddev_from_prefix_f32(const float* __restrict__ prices,
                                      const float* __restrict__ pref,
                                      const float* __restrict__ pref2,
                                      const int*   __restrict__ runlen,
                                      const int* __restrict__ periods,
                                      const int* __restrict__ ma_lens,
                                      const int* __restrict__ matypes,
                                      int series_len,
                                      int first_valid,
                                      int n_combos,
                                      int max_ma_len,
                                      const int* __restrict__ row_ids,
                                      float* __restrict__ out)
{
    const int row = blockIdx.x;
    if (row >= n_combos) return;
    if (threadIdx.x != 0) return;

    const int period = periods[row];
    const int ma_len = ma_lens[row];
    const int matype = matypes[row];
    if (period <= 0 || ma_len <= 0) return;


    extern __shared__ unsigned char shraw[];
    float* up_ring = reinterpret_cast<float*>(shraw);
    float* dn_ring = up_ring + max_ma_len;

    const int global_row = row_ids ? row_ids[row] : row;
    const int base = global_row * series_len;
    const int warm = first_valid + (period - 1) + (ma_len - 1);


    for (int i = 0; i < ((warm < series_len) ? warm : series_len); ++i) out[base + i] = NAN;
    if (warm >= series_len) return;


    const double inv_m_d = 1.0 / (double)ma_len;
    const double alpha_d  = 2.0 / ((double)ma_len + 1.0);
    const double one_m_alpha_d = 1.0 - alpha_d;

    bool  up_started = false, dn_started = false;
    double up_seed_sum = 0.0, dn_seed_sum = 0.0;
    int   up_seed_cnt = 0,    dn_seed_cnt = 0;

    int   up_h = 0, dn_h = 0, up_cnt = 0, dn_cnt = 0;
    double up_sum = 0.0, dn_sum = 0.0;
    double up_prev = 0.0, dn_prev = 0.0;

    float prev = prices[0];

    for (int i = 0; i < series_len; ++i) {
        const float x = prices[i];

        double dd;
        if (i == 0 || !isfinite(x) || !isfinite(prev)) dd = NAN; else dd = (double)x - (double)prev;
        prev = x;

        float dev;
        if (i + 1 < period) {
            dev = NAN;
        } else if (runlen[i] < period) {
            dev = NAN;
        } else {

            double s = (double)(pref[i]  - ((i == period - 1) ? 0.f : pref[i - period]));
            double q = (double)(pref2[i] - ((i == period - 1) ? 0.f : pref2[i - period]));
            const double invP = 1.0 / (double)period;
            const double mean = s * invP;
            const double var  = fmax(0.0, fma(-mean, mean, q * invP));
            dev = (float)sqrt(var);
        }

        float up_i, dn_i;
        if (!isfinite(dd) || !isfinite(dev))      { up_i = NAN;  dn_i = NAN; }
        else if (dd > 0.0)                       { up_i = dev;  dn_i = 0.0f; }
        else if (dd < 0.0)                       { up_i = 0.0f; dn_i = dev;  }
        else                                     { up_i = 0.0f; dn_i = 0.0f; }

        float up_s, dn_s;
        if (matype == 0) {
            up_s = smooth_sma_push(up_i, up_ring, &up_h, &up_cnt, ma_len, &up_sum, inv_m_d);
            dn_s = smooth_sma_push(dn_i, dn_ring, &dn_h, &dn_cnt, ma_len, &dn_sum, inv_m_d);
        } else {
            up_s = smooth_ema_push(up_i, &up_started, &up_seed_sum, &up_seed_cnt, ma_len, inv_m_d, alpha_d, one_m_alpha_d, &up_prev);
            dn_s = smooth_ema_push(dn_i, &dn_started, &dn_seed_sum, &dn_seed_cnt, ma_len, inv_m_d, alpha_d, one_m_alpha_d, &dn_prev);
        }

        if (i >= warm) {
            if (!isfinite(up_s) || !isfinite(dn_s)) {
                out[base + i] = NAN;
            } else {
                const double denom_d = (double)up_s + (double)dn_s;
                out[base + i] = (fabs(denom_d) <= 1e-15) ? NAN : (100.0f * (up_s / (float)denom_d));
            }
        }
    }
}


extern "C" __global__
void scatter_rows_f32(const float* __restrict__ src,
                      int src_rows,
                      int len,
                      const int* __restrict__ row_ids,
                      float* __restrict__ dst) {
    const int row = blockIdx.x;
    if (row >= src_rows) return;
    const int dst_row = row_ids[row];
    const int src_base = row * len;
    const int dst_base = dst_row * len;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        dst[dst_base + i] = src[src_base + i];
    }
}


static __device__ __forceinline__ void rvi_compute_series(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int period,
    int ma_len,
    int matype,
    int devtype,
    int max_period,
    int max_ma_len,
    float* __restrict__ out)
{
    if (len <= 0 || first_valid >= len || period <= 0 || ma_len <= 0) return;

    extern __shared__ unsigned char shraw[];
    float* up_ring = reinterpret_cast<float*>(shraw);
    float* dn_ring = up_ring + max_ma_len;
    float* dev_ring = dn_ring + max_ma_len;
    unsigned char* vflag = reinterpret_cast<unsigned char*>(dev_ring + max_period);

    const int warm = first_valid + (period - 1) + (ma_len - 1);


    double sum = 0.0, sumsq = 0.0;
    int head = 0, filled = 0;
    double ring_sum = 0.0;


    const double inv_m = 1.0 / (double)ma_len;
    const double alpha = 2.0 / ((double)ma_len + 1.0);
    const double one_m_alpha = 1.0 - alpha;
    bool up_started = false, dn_started = false;
    double up_seed_sum = 0.0, dn_seed_sum = 0.0;
    int up_seed_cnt = 0, dn_seed_cnt = 0;
    int up_h = 0, dn_h = 0, up_cnt = 0, dn_cnt = 0;
    double up_sum = 0.0, dn_sum = 0.0;
    double up_prev = 0.0, dn_prev = 0.0;

    float prev = prices[0];


    if (devtype != 0) {

        filled = 0;
        head = 0;
        ring_sum = 0.0;
    }

    for (int i = 0; i < len; ++i) {
        const float x = prices[i];
        float d;
        if (i == 0 || !isfinite(x) || !isfinite(prev)) d = NAN; else d = x - prev;
        prev = x;

        float dev;
        if (i + 1 < period) {
            dev = NAN;
        } else if (devtype == 0) {

            if (i == period - 1) {
                sum = 0.0; sumsq = 0.0;
                bool ok = true;
                for (int k = 0; k < period; ++k) {
                    const float v = prices[k];
                    if (!isfinite(v)) { ok = false; break; }
                    sum += (double)v;
                    sumsq += (double)v * (double)v;
                }
                if (ok) {
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                } else {
                    dev = NAN;
                }
            } else {
                const float leaving = prices[i - period];
                if (!isfinite(leaving) || !isfinite(x)) {
                    sum = 0.0; sumsq = 0.0;
                    bool ok = true;
                    for (int k = i - period + 1; k <= i; ++k) {
                        const float v = prices[k];
                        if (!isfinite(v)) { ok = false; break; }
                        sum += (double)v;
                        sumsq += (double)v * (double)v;
                    }
                    if (ok) {
                        const double mean = sum / (double)period;
                        const double mean_sq = sumsq / (double)period;
                        dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                    } else {
                        dev = NAN;
                    }
                } else {
                    sum += (double)x - (double)leaving;
                    sumsq += (double)x * (double)x - (double)leaving * (double)leaving;
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                }
            }
        } else {
            if (!isfinite(x)) {

                filled = 0;
                head = 0;
                ring_sum = 0.0;
                dev = NAN;
            } else if (filled < period) {
                dev_ring[head] = x;
                ring_sum += (double)x;
                head = (head + 1 == period) ? 0 : head + 1;
                filled += 1;
                dev = (filled == period) ? 0.0f : NAN;
                if (filled == period) {

                    const double mean = ring_sum / (double)period;
                    double abs_sum = 0.0;
                    for (int k = 0; k < period; ++k) {
                        abs_sum += fabs((double)dev_ring[k] - mean);
                    }
                    dev = (float)(abs_sum / (double)period);
                }
            } else {

                const float old = dev_ring[head];
                dev_ring[head] = x;
                head = (head + 1 == period) ? 0 : head + 1;
                ring_sum += (double)x - (double)old;
                const double mean = ring_sum / (double)period;
                double abs_sum = 0.0;
                for (int k = 0; k < period; ++k) {
                    abs_sum += fabs((double)dev_ring[k] - mean);
                }
                dev = (float)(abs_sum / (double)period);
            }
        }

        float up_i, dn_i;
        if (!isfinite(d) || !isfinite(dev)) {
            up_i = NAN; dn_i = NAN;
        } else if (d > 0.0f) {
            up_i = dev; dn_i = 0.0f;
        } else if (d < 0.0f) {
            up_i = 0.0f; dn_i = dev;
        } else { up_i = 0.0f; dn_i = 0.0f; }

        double up_sd, dn_sd; float up_s, dn_s;
        if (matype == 0) {
            up_s = smooth_sma_push(up_i, up_ring, &up_h, &up_cnt, ma_len, &up_sum, inv_m);
            dn_s = smooth_sma_push(dn_i, dn_ring, &dn_h, &dn_cnt, ma_len, &dn_sum, inv_m);
            up_sd = (double)up_s; dn_sd = (double)dn_s;
        } else {
            up_sd = smooth_ema_push_d((double)up_i, &up_started, &up_seed_sum, &up_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &up_prev);
            dn_sd = smooth_ema_push_d((double)dn_i, &dn_started, &dn_seed_sum, &dn_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &dn_prev);
            up_s = (float)up_sd; dn_s = (float)dn_sd;
        }

        if (i >= warm) {
            if (!isfinite(up_s) || !isfinite(dn_s)) {
                out[i] = NAN;
            } else {
                const double denom_d = up_sd + dn_sd;
                out[i] = (fabs(denom_d) <= 1e-15) ? NAN : (float)(100.0 * (up_sd / denom_d));
            }
        }
    }
}


extern "C" __global__
void rvi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   const int* __restrict__ ma_lens,
                   const int* __restrict__ matypes,
                   const int* __restrict__ devtypes,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   int max_period,
                   int max_ma_len,
                   float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= n_combos) return;

    const int period = periods[row];
    const int ma_len = ma_lens[row];
    const int matype = matypes[row];
    const int devtype = devtypes[row];
    if (period <= 0 || ma_len <= 0) return;

    const int base = row * series_len;


    int warm = first_valid + (period - 1) + (ma_len - 1);
    if (warm > series_len) warm = series_len;
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    rvi_compute_series(prices, series_len, first_valid, period, ma_len, matype, devtype, max_period, max_ma_len, out + base);
}


extern "C" __global__
void rvi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   int ma_len,
                                   int matype,
                                   int devtype,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    if (period <= 0 || ma_len <= 0) return;
    const int first = first_valids[s];


    int warm = first + (period - 1) + (ma_len - 1);
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;


    const bool use_sma = (matype == 0) && (ma_len <= 1024);

    const double inv_m = 1.0 / (double)ma_len;
    const double alpha = 2.0 / ((double)ma_len + 1.0);
    const double one_m_alpha = 1.0 - alpha;
    bool up_started = false, dn_started = false;
    double up_seed_sum = 0.0, dn_seed_sum = 0.0;
    int up_seed_cnt = 0, dn_seed_cnt = 0;
    int up_h = 0, dn_h = 0, up_cnt = 0, dn_cnt = 0;
    double up_sum = 0.0, dn_sum = 0.0;
    double up_prev = 0.0, dn_prev = 0.0;


    double sum = 0.0, sumsq = 0.0;
    int valid = 0, head = 0, filled = 0;
    double ring_sum = 0.0;

    float prev = prices_tm[0 * cols + s];


    float up_ring_local[ (1024) ];
    float dn_ring_local[ (1024) ];
    float* up_ring = up_ring_local;
    float* dn_ring = dn_ring_local;

    const bool mad_local_ok = (period <= 2048);
    float dev_ring_local[ (2048) ];


    if (devtype != 0) {
        filled = 0; head = 0; ring_sum = 0.0;
    }

    for (int i = 0; i < rows; ++i) {
        const float x = prices_tm[i * cols + s];
        float d;
        if (i == 0 || !isfinite(x) || !isfinite(prev)) d = NAN; else d = x - prev;
        prev = x;

        float dev;
        if (i + 1 < period) {
            dev = NAN;
        } else if (devtype == 0) {

            if (i == period - 1) {
                sum = 0.0; sumsq = 0.0; bool ok = true;
                for (int k = 0; k < period; ++k) {
                    const float v = prices_tm[k * cols + s];
                    if (!isfinite(v)) { ok = false; break; }
                    sum += (double)v; sumsq += (double)v * (double)v;
                }
                if (ok) {
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                } else { dev = NAN; }
            } else {
                const float leaving = prices_tm[(i - period) * cols + s];
                if (!isfinite(leaving) || !isfinite(x)) {
                    sum = 0.0; sumsq = 0.0; bool ok = true;
                    for (int k = i - period + 1; k <= i; ++k) {
                        const float v = prices_tm[k * cols + s];
                        if (!isfinite(v)) { ok = false; break; }
                        sum += (double)v; sumsq += (double)v * (double)v;
                    }
                    if (ok) {
                        const double mean = sum / (double)period;
                        const double mean_sq = sumsq / (double)period;
                        dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                    } else { dev = NAN; }
                } else {
                    sum += (double)x - (double)leaving;
                    sumsq += (double)x * (double)x - (double)leaving * (double)leaving;
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                }
            }
        } else {
            if (!isfinite(x)) { filled = 0; head = 0; ring_sum = 0.0; dev = NAN; }
            else if (filled < period) {
                if (mad_local_ok) dev_ring_local[head] = x;
                ring_sum += (double)x; head = (head + 1 == period) ? 0 : head + 1; filled++;
                if (filled == period) {
                    const double mean = ring_sum / (double)period; double abs_sum = 0.0;
                    if (mad_local_ok) { for (int k = 0; k < period; ++k) abs_sum += fabs((double)dev_ring_local[k] - mean); }
                    else { for (int k = i - period + 1; k <= i; ++k) { float v = prices_tm[k * cols + s]; abs_sum += fabs((double)v - mean); } }
                    dev = (float)(abs_sum / (double)period);
                } else dev = NAN;
            } else {
                const float old = mad_local_ok ? dev_ring_local[head] : prices_tm[(i - period) * cols + s];
                if (mad_local_ok) dev_ring_local[head] = x;
                head = (head + 1 == period) ? 0 : head + 1; ring_sum += (double)x - (double)old;
                const double mean = ring_sum / (double)period; double abs_sum = 0.0;
                if (mad_local_ok) { for (int k = 0; k < period; ++k) abs_sum += fabs((double)dev_ring_local[k] - mean); }
                else { for (int k = i - period + 1; k <= i; ++k) { float v = prices_tm[k * cols + s]; abs_sum += fabs((double)v - mean); } }
                dev = (float)(abs_sum / (double)period);
            }
        }

        float up_i, dn_i;
        if (!isfinite(d) || !isfinite(dev)) { up_i = NAN; dn_i = NAN; }
        else if (d > 0.0f) { up_i = dev; dn_i = 0.0f; }
        else if (d < 0.0f) { up_i = 0.0f; dn_i = dev; }
        else { up_i = 0.0f; dn_i = 0.0f; }

        float up_s, dn_s;
        if (use_sma) {
            up_s = smooth_sma_push(up_i, up_ring, &up_h, &up_cnt, ma_len, &up_sum, inv_m);
            dn_s = smooth_sma_push(dn_i, dn_ring, &dn_h, &dn_cnt, ma_len, &dn_sum, inv_m);
        } else {
            up_s = smooth_ema_push(up_i, &up_started, &up_seed_sum, &up_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &up_prev);
            dn_s = smooth_ema_push(dn_i, &dn_started, &dn_seed_sum, &dn_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &dn_prev);
        }

        if (i >= warm) {
            if (!isfinite(up_s) || !isfinite(dn_s)) out_tm[i * cols + s] = NAN;
            else {
                const double denom_d = (double)up_s + (double)dn_s;
                out_tm[i * cols + s] = (fabs(denom_d) <= 1e-15) ? NAN : (100.0f * (up_s / (float)denom_d));
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 6
 *
 * ORACLE: `rvi_scalar` (src/indicators/rvi.rs:400), the `devtype == 0`
 * (standard deviation) and `matype == 1` (EMA) arm -- :603-719. Those are the
 * batch defaults (cpu_batch.rs:16610-16611), and `rvi_with_kernel` maps
 * `Kernel::Auto` onto `Kernel::Scalar` outright (:272), so there is exactly
 * one CPU answer to match.
 *
 * PERIOD-SWEPT. `compute_rvi_batch` reads a parameter literally named
 * `period`, default 10 (cpu_batch.rs:16608). `ma_len` (14), `matype` (1) and
 * `devtype` (0) are NOT swept and are pinned to those defaults here.
 *
 * SINGLE OUTPUT: `output_id == "value"` is the only column this indicator
 * offers (cpu_batch.rs:16626).
 *
 * WARMUP: `first + (period - 1) + (ma_len - 1)` (:415). Note the CPU's own
 * walk starts at index 0, NOT at `first`: `prev` is seeded from `data[0]`
 * (:471) and the variance window is seeded from `data[0..period]` (:459).
 * `first` enters only through the warmup gate. Starting the kernel at
 * `first` instead would seed a different variance and shift every value.
 *
 * THE EPSILON IS ALREADY f64. `denom.abs() < f64::EPSILON` (:710) uses
 * 2.220446049250313e-16, the DOUBLE epsilon -- it is carried across
 * unchanged, and that is correct precisely because it was not an f32
 * constant to begin with. The f32 lane in this file is not the reference for
 * it.
 *
 * NaN IS LOAD-BEARING, NOT AN ACCIDENT. Four separate resets hang off it: a
 * non-finite bar blanks `d`, blanks `dev`, and RESTARTS both EMA seeds
 * (`up_started = false`, :675). fmax/fmin would erase exactly the behaviour
 * being reproduced, so the comparisons here are the CPU's `is_nan` tests, not
 * fmax.
 *
 * THE VARIANCE WINDOW IS ROLLED, AND RE-SEEDED ON A HOLE. `sum += x - leaving;
 * sumsq += x*x - leaving*leaving` (:648) is the steady state; when either end
 * of the window or an accumulator is NaN the CPU RESCANS the whole window
 * from scratch (:628-645). Both branches are reproduced, in that order --
 * a kernel that only rolled would carry a poisoned accumulator forever.
 *
 * ONE ROUNDING, NOT THREE: `alpha.mul_add(up_i, one_m_alpha * up_prev)`
 * (:685) is `fma(alpha, up_i, one_m_alpha * up_prev)`.
 *
 * NO PER-THREAD ARRAY: the variance window is read straight out of the
 * resident input and the EMA carries two scalars, so there is no compile-time
 * bound and no `max_period`.
 *
 * SEQUENTIAL, one thread per combo column.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* f64::EPSILON -- rvi.rs:710. NOT an f32 constant. */
#define RVI_NEO_F64_EPS 2.2204460492503131e-16

#define RVI_NEO_MA_LEN 14   /* cpu_batch.rs:16609 */

extern "C" __global__
void rvi_neo_batch_f64(const double* __restrict__ data,
                       int series_len,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int n = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    if (n == 0) return;
    if (first_valid < 0 || first_valid >= n) return;

    const int period = periods[combo];
    const int ma_len = RVI_NEO_MA_LEN;
    if (period <= 0 || ma_len <= 0 || period > n || ma_len > n) return;   /* :253 */

    const int max_needed = (period - 1) + (ma_len - 1);
    if (n - first_valid <= max_needed) return;                            /* :260 */

    const int warmup = first_valid + max_needed;                          /* :415 */

    const double inv_p = 1.0 / (double)period;
    const double inv_m = 1.0 / (double)ma_len;
    const double alpha = 2.0 / ((double)ma_len + 1.0);
    const double one_m_alpha = 1.0 - alpha;

    /* Seed of the variance window, :459-468. Walks from index 0. */
    double sum = 0.0, sumsq = 0.0;
    {
        const int lim = min(period, n);
        for (int i = 0; i < lim; ++i) {
            const double x = data[i];
            if (isnan(x)) { sum = NEO_F64_NAN; sumsq = NEO_F64_NAN; break; }
            sum   += x;
            sumsq += x * x;
        }
    }

    double up_prev = 0.0, dn_prev = 0.0;
    bool   up_started = false, dn_started = false;
    double up_seed_sum = 0.0, dn_seed_sum = 0.0;
    int    up_seed_cnt = 0,   dn_seed_cnt = 0;

    double prev = data[0];                                                /* :471 */

    for (int i = 0; i < n; ++i) {
        const double x = data[i];
        const double d = (i == 0 || isnan(x) || isnan(prev))
                             ? NEO_F64_NAN
                             : (x - prev);
        prev = x;

        double dev;
        if (i + 1 < period) {
            dev = NEO_F64_NAN;
        } else if (i == period - 1) {
            if (isnan(sum)) {
                dev = NEO_F64_NAN;
            } else {
                const double mean    = sum   * inv_p;
                const double mean_sq = sumsq * inv_p;
                dev = sqrt(mean_sq - mean * mean);
            }
        } else {
            const double leaving = data[i - period];
            if (isnan(leaving) || isnan(x) || isnan(sum) || isnan(sumsq)) {
                /* Full rescan, :628-645. */
                sum = 0.0; sumsq = 0.0;
                const int start = i + 1 - period;
                bool bad = false;
                for (int k = start; k <= i; ++k) {
                    const double v = data[k];
                    if (isnan(v)) { bad = true; break; }
                    sum   += v;
                    sumsq += v * v;
                }
                if (bad) {
                    sum = NEO_F64_NAN; sumsq = NEO_F64_NAN;
                    dev = NEO_F64_NAN;
                } else {
                    const double mean    = sum   * inv_p;
                    const double mean_sq = sumsq * inv_p;
                    dev = sqrt(mean_sq - mean * mean);
                }
            } else {
                sum   += x - leaving;
                sumsq += x * x - leaving * leaving;
                const double mean    = sum   * inv_p;
                const double mean_sq = sumsq * inv_p;
                dev = sqrt(mean_sq - mean * mean);
            }
        }

        double up_i, dn_i;
        if (isnan(d) || isnan(dev)) { up_i = NEO_F64_NAN; dn_i = NEO_F64_NAN; }
        else if (d > 0.0)           { up_i = dev;         dn_i = 0.0; }
        else if (d < 0.0)           { up_i = 0.0;         dn_i = dev; }
        else                        { up_i = 0.0;         dn_i = 0.0; }

        double up_s;
        if (isnan(up_i)) {
            up_started = false; up_seed_sum = 0.0; up_seed_cnt = 0;
            up_s = NEO_F64_NAN;
        } else if (!up_started) {
            up_seed_sum += up_i;
            up_seed_cnt += 1;
            if (up_seed_cnt == ma_len) {
                up_prev = up_seed_sum * inv_m;
                up_started = true;
                up_s = up_prev;
            } else {
                up_s = NEO_F64_NAN;
            }
        } else {
            up_prev = fma(alpha, up_i, one_m_alpha * up_prev);            /* :685 */
            up_s = up_prev;
        }

        double dn_s;
        if (isnan(dn_i)) {
            dn_started = false; dn_seed_sum = 0.0; dn_seed_cnt = 0;
            dn_s = NEO_F64_NAN;
        } else if (!dn_started) {
            dn_seed_sum += dn_i;
            dn_seed_cnt += 1;
            if (dn_seed_cnt == ma_len) {
                dn_prev = dn_seed_sum * inv_m;
                dn_started = true;
                dn_s = dn_prev;
            } else {
                dn_s = NEO_F64_NAN;
            }
        } else {
            dn_prev = fma(alpha, dn_i, one_m_alpha * dn_prev);
            dn_s = dn_prev;
        }

        if (i >= warmup) {
            if (isnan(up_s) || isnan(dn_s)) {
                o[i] = NEO_F64_NAN;
            } else {
                const double denom = up_s + dn_s;
                o[i] = (fabs(denom) < RVI_NEO_F64_EPS)
                           ? NEO_F64_NAN
                           : (100.0 * (up_s / denom));                    /* :713 */
            }
        }
    }
}
