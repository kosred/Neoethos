#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace { __device__ inline bool is_finitef(float x) { return !isnan(x) && !isinf(x); } }


#ifndef CCI_RING_MAX
#define CCI_RING_MAX 128
#endif

/* Public f32 kernels are the explicit pre-v9 VectorTA legacy lane. */
#define CCI_CYCLE_F32_LEGACY_SEMANTIC_VERSION 8


__device__ inline void scan_minmax_ring(const float* __restrict__ ring,
                                        int L, int have, int start,
                                        float &mn, float &mx)
{
    mn = CUDART_INF_F;
    mx = -CUDART_INF_F;
    int idx = start;
    #pragma unroll
    for (int t = 0; t < CCI_RING_MAX; ++t) {
        if (t >= have) break;
        float v = ring[idx];
        if (is_finitef(v)) {
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        idx++;
        if (idx == L) idx = 0;
    }
}


extern "C" __global__ void cci_cycle_batch_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ lengths,
    const float* __restrict__ factors,
    float* __restrict__ out
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid; row < n_combos; row += stride) {
        const int   L      = lengths[row];
        const float factor = factors[row];
        float* row_out     = out + static_cast<size_t>(row) * len;


        if (L <= 0 || L > len) {
            for (int i = 0; i < len; ++i) row_out[i] = CUDART_NAN_F;
            continue;
        }
        const int needed = L * 2;
        if (len - first_valid < needed) {
            for (int i = 0; i < len; ++i) row_out[i] = CUDART_NAN_F;
            continue;
        }
        if (L > CCI_RING_MAX) {

            for (int i = 0; i < len; ++i) row_out[i] = CUDART_NAN_F;
            continue;
        }

        const float invL   = 1.0f / (float)L;
        const int   half   = (L + 1) / 2;
        const float alpha_s = 2.0f / (half + 1.0f);
        const float beta_s  = 1.0f - alpha_s;
        const float alpha_l = 2.0f / (L + 1.0f);
        const float beta_l  = 1.0f - alpha_l;
        const int   smma_p  = max(1, (int)rintf(sqrtf((float)L)));


        const int i0 = first_valid;
        const int i1 = first_valid + L;
        float sum = 0.0f;
        for (int i = i0; i < i1; ++i) sum += prices[i];
        float sma = sum * invL;

        float sum_abs = 0.0f;
        for (int i = i0; i < i1; ++i) sum_abs += fabsf(prices[i] - sma);

        const int out_start = first_valid + L - 1;


        for (int i = 0; i < out_start; ++i) row_out[i] = CUDART_NAN_F;

        float denom = 0.015f * (sum_abs * invL);
        float cci   = (denom == 0.0f) ? 0.0f : ((prices[out_start] - sma) / denom);


        float ema_s = cci;
        float ema_l = cci;


        float smma        = CUDART_NAN_F;
        float smma_sum    = 0.0f;
        int   smma_count  = 0;
        bool  smma_inited = false;


        float prev_f1  = CUDART_NAN_F;
        float prev_pf  = CUDART_NAN_F;
        float prev_out = CUDART_NAN_F;


        float ccis_ring[CCI_RING_MAX]; int ccis_valid = 0;
        float  pf_ring[CCI_RING_MAX];  int  pf_valid  = 0;

        for (int i = out_start; i < len; ++i) {

            const float entering = prices[i];
            const float exiting  = prices[i - L];
            sum = sum - exiting + entering;
            sma = sum * invL;


            float sabs = 0.0f;
            const int wstart = i + 1 - L;
            #pragma unroll
            for (int k = 0; k < CCI_RING_MAX; ++k) {
                if (k >= L) break;
                float v = prices[wstart + k];
                sabs += fabsf(v - sma);
            }
            float denom2 = 0.015f * (sabs * invL);
            float cci2   = (denom2 == 0.0f) ? 0.0f : ((entering - sma) / denom2);


            ema_s = fmaf(beta_s, ema_s, alpha_s * cci2);
            ema_l = fmaf(beta_l, ema_l, alpha_l * cci2);
            const float de = ema_s + ema_s - ema_l;


            if (!smma_inited) {
                if (is_finitef(de)) {
                    smma_sum += de;
                    if (++smma_count >= smma_p) {
                        smma = smma_sum / (float)smma_p;
                        smma_inited = true;
                    }
                }
            } else {
                smma = (smma * (smma_p - 1) + de) / (float)smma_p;
            }


            const int pos = i % L;
            ccis_ring[pos] = smma;
            if (ccis_valid < L) ccis_valid++;


            float pf = CUDART_NAN_F;
            {
                const int have  = ccis_valid;
                int start = (i - have + 1) % L; if (start < 0) start += L;
                float mn1, mx1;
                scan_minmax_ring(ccis_ring, L, have, start, mn1, mx1);
                if (is_finitef(mn1) && is_finitef(mx1)) {
                    const float range = mx1 - mn1;
                    float cur_f1 = 50.0f;
                    if (range > 0.0f && is_finitef(smma))
                        cur_f1 = ((smma - mn1) / range) * 100.0f;
                    else
                        cur_f1 = isnan(prev_f1) ? 50.0f : prev_f1;

                    pf      = (isnan(prev_pf) || factor == 0.0f)
                            ? cur_f1
                            : fmaf((cur_f1 - prev_pf), factor, prev_pf);
                    prev_f1 = cur_f1;
                    prev_pf = pf;
                }
            }


            pf_ring[pos] = pf; if (pf_valid < L) pf_valid++;


            float out_i = CUDART_NAN_F;
            {
                const int have  = pf_valid;
                int start = (i - have + 1) % L; if (start < 0) start += L;
                float mn2, mx2;
                scan_minmax_ring(pf_ring, L, have, start, mn2, mx2);
                if (is_finitef(mn2) && is_finitef(mx2)) {
                    const float range = mx2 - mn2;
                    if (range > 0.0f && is_finitef(pf)) {
                        const float f2 = ((pf - mn2) / range) * 100.0f;
                        out_i = (isnan(prev_out) || factor == 0.0f)
                              ? f2
                              : fmaf((f2 - prev_out), factor, prev_out);
                    } else {
                        out_i = isnan(prev_out) ? 50.0f : prev_out;
                    }
                    prev_out = out_i;
                }
            }

            row_out[i] = out_i;
        }
    }
}


extern "C" __global__ void cci_cycle_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    int length,
    float factor,
    float* __restrict__ out_tm
) {
    const int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= rows) return;

    const int L = length;
    float* out_row = out_tm + (size_t)rid * cols;

    if (L <= 0 || L > cols || L > CCI_RING_MAX) {
        for (int i = 0; i < cols; ++i) out_row[i] = CUDART_NAN_F;
        return;
    }

    const float invL   = 1.0f / (float)L;
    const int   half   = (L + 1) / 2;
    const float alpha_s = 2.0f / (half + 1.0f);
    const float beta_s  = 1.0f - alpha_s;
    const float alpha_l = 2.0f / (L + 1.0f);
    const float beta_l  = 1.0f - alpha_l;
    const int   smma_p  = max(1, (int)rintf(sqrtf((float)L)));

    int first_valid = first_valids[rid];
    if (first_valid < 0) first_valid = 0;
    if (cols - first_valid < L * 2) {
        for (int i = 0; i < cols; ++i) out_row[i] = CUDART_NAN_F;
        return;
    }

    const float* prices = data_tm + (size_t)rid * cols;


    const int i0 = first_valid;
    const int i1 = first_valid + L;
    float sum = 0.0f;
    for (int i = i0; i < i1; ++i) sum += prices[i];
    float sma = sum * invL;

    float sum_abs = 0.0f;
    for (int i = i0; i < i1; ++i) sum_abs += fabsf(prices[i] - sma);

    const int out_start = first_valid + L - 1;
    for (int i = 0; i < out_start; ++i) out_row[i] = CUDART_NAN_F;

    float denom = 0.015f * (sum_abs * invL);
    float cci   = (denom == 0.0f) ? 0.0f : ((prices[out_start] - sma) / denom);

    float ema_s = cci, ema_l = cci;
    float smma = CUDART_NAN_F, smma_sum = 0.0f; int smma_count = 0; bool smma_inited = false;
    float prev_f1 = CUDART_NAN_F, prev_pf = CUDART_NAN_F, prev_out = CUDART_NAN_F;

    float ccis_ring[CCI_RING_MAX]; int ccis_valid = 0;
    float  pf_ring[CCI_RING_MAX];  int  pf_valid  = 0;

    for (int i = out_start; i < cols; ++i) {

        const float entering = prices[i];
        const float exiting  = prices[i - L];
        sum = sum - exiting + entering;
        sma = sum * invL;

        float sabs = 0.0f;
        const int wstart = i + 1 - L;
        #pragma unroll
        for (int k = 0; k < CCI_RING_MAX; ++k) {
            if (k >= L) break;
            sabs += fabsf(prices[wstart + k] - sma);
        }
        denom = 0.015f * (sabs * invL);
        cci   = (denom == 0.0f) ? 0.0f : ((entering - sma) / denom);


        ema_s = fmaf(beta_s, ema_s, alpha_s * cci);
        ema_l = fmaf(beta_l, ema_l, alpha_l * cci);
        const float de = ema_s + ema_s - ema_l;

        if (!smma_inited) {
            if (is_finitef(de)) { smma_sum += de; if (++smma_count >= smma_p) { smma = smma_sum / (float)smma_p; smma_inited = true; } }
        } else { smma = (smma * (smma_p - 1) + de) / (float)smma_p; }


        const int pos = i % L; ccis_ring[pos] = smma; if (ccis_valid < L) ccis_valid++;


        float pf = CUDART_NAN_F;
        {
            const int have  = ccis_valid;
            int start = (i - have + 1) % L; if (start < 0) start += L;
            float mn1, mx1; scan_minmax_ring(ccis_ring, L, have, start, mn1, mx1);
            if (is_finitef(mn1) && is_finitef(mx1)) {
                const float range = mx1 - mn1;
                float cur_f1 = 50.0f;
                if (range > 0.0f && is_finitef(smma)) cur_f1 = ((smma - mn1) / range) * 100.0f; else cur_f1 = isnan(prev_f1) ? 50.0f : prev_f1;
                pf = (isnan(prev_pf) || factor == 0.0f) ? cur_f1 : fmaf((cur_f1 - prev_pf), factor, prev_pf);
                prev_f1 = cur_f1; prev_pf = pf;
            }
        }


        pf_ring[pos] = pf; if (pf_valid < L) pf_valid++;


        float out_i = CUDART_NAN_F;
        {
            const int have  = pf_valid; float mn2, mx2; int start = (i - have + 1) % L; if (start < 0) start += L;
            scan_minmax_ring(pf_ring, L, have, start, mn2, mx2);
            if (is_finitef(mn2) && is_finitef(mx2)) {
                const float range = mx2 - mn2;
                if (range > 0.0f && is_finitef(pf)) {
                    const float f2 = ((pf - mn2) / range) * 100.0f;
                    out_i = (isnan(prev_out) || factor == 0.0f) ? f2 : fmaf((f2 - prev_out), factor, prev_out);
                } else {
                    out_i = isnan(prev_out) ? 50.0f : prev_out;
                }
                prev_out = out_i;
            }
        }
        out_row[i] = out_i;
    }
}

/* ===========================================================================
 * Classic semantic-v9 f64 lane.
 *
 * Creator-aligned local-current-resolution formula:
 *   CCI(length) -> 2*EMA(floor(length/2))-EMA(length)
 *   -> RMA(round(sqrt(length))) -> stochastic/factor -> stochastic/factor.
 * EMA and RMA are SMA seeded. Startup and flat stochastic ranges carry zero;
 * factor zero freezes the seeded value. Every non-finite close emits NaN and
 * resets all state, so the next finite segment restarts from zero. One thread
 * owns one requested length and keeps only bounded O(length) state.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_CCICYC_MAX_LENGTH 200
#define NEO_CCICYC_FACTOR 0.5
#define NEO_CCICYC_CLASSIC_SEMANTIC_VERSION 9

extern "C" __global__
void cci_cycle_neo_batch_f64(const double* __restrict__ data,
                             int series_len,
                             const int* __restrict__ periods,
                             int n_combos,
                             int first_valid,
                             double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    const int length = periods[combo];
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)first_valid;

    if (len <= 0 || length < 2 || length > len ||
        length > NEO_CCICYC_MAX_LENGTH) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const double factor = NEO_CCICYC_FACTOR;
    const int half = length / 2;
    int rma_length = (int)round(sqrt((double)length));
    if (rma_length < 1) rma_length = 1;
    const double ema_short_alpha = 2.0 / ((double)half + 1.0);
    const double ema_long_alpha = 2.0 / ((double)length + 1.0);
    const double rma_alpha = 1.0 / (double)rma_length;

    double close_ring[NEO_CCICYC_MAX_LENGTH];
    double ccis_ring[NEO_CCICYC_MAX_LENGTH];
    double pf_ring[NEO_CCICYC_MAX_LENGTH];
    for (int k = 0; k < length; ++k) {
        close_ring[k] = NEO_F64_NAN;
        ccis_ring[k] = NEO_F64_NAN;
        pf_ring[k] = NEO_F64_NAN;
    }

    int segment_bars = 0;
    int close_count = 0;
    double ema_short_seed = 0.0;
    double ema_long_seed = 0.0;
    int ema_short_count = 0;
    int ema_long_count = 0;
    double ema_short = NEO_F64_NAN;
    double ema_long = NEO_F64_NAN;
    bool ema_short_inited = false;
    bool ema_long_inited = false;
    double rma_seed = 0.0;
    int rma_count = 0;
    double rma = NEO_F64_NAN;
    bool rma_inited = false;
    double previous_f1 = 0.0;
    double previous_pf = 0.0;
    double previous_f2 = 0.0;
    double previous_pff = 0.0;

    for (int i = 0; i < len; ++i) {
        const double close = data[i];
        if (!isfinite(close)) {
            o[i] = NEO_F64_NAN;
            segment_bars = 0;
            close_count = 0;
            ema_short_seed = 0.0;
            ema_long_seed = 0.0;
            ema_short_count = 0;
            ema_long_count = 0;
            ema_short = NEO_F64_NAN;
            ema_long = NEO_F64_NAN;
            ema_short_inited = false;
            ema_long_inited = false;
            rma_seed = 0.0;
            rma_count = 0;
            rma = NEO_F64_NAN;
            rma_inited = false;
            previous_f1 = 0.0;
            previous_pf = 0.0;
            previous_f2 = 0.0;
            previous_pff = 0.0;
            for (int k = 0; k < length; ++k) {
                close_ring[k] = NEO_F64_NAN;
                ccis_ring[k] = NEO_F64_NAN;
                pf_ring[k] = NEO_F64_NAN;
            }
            continue;
        }

        const int slot = segment_bars % length;
        close_ring[slot] = close;
        if (close_count < length) ++close_count;

        double cci = NEO_F64_NAN;
        if (close_count == length) {
            const int oldest = (slot + 1) % length;
            double sum = 0.0;
            for (int k = 0; k < length; ++k) {
                sum += close_ring[(oldest + k) % length];
            }
            const double mean = sum / (double)length;
            double deviation_sum = 0.0;
            for (int k = 0; k < length; ++k) {
                deviation_sum += fabs(close_ring[(oldest + k) % length] - mean);
            }
            const double deviation = deviation_sum / (double)length;
            if (deviation > 0.0 && isfinite(deviation)) {
                const double candidate = (close - mean) / (0.015 * deviation);
                if (isfinite(candidate)) cci = candidate;
            }
        }

        if (isfinite(cci)) {
            if (!ema_short_inited) {
                ema_short_seed += cci;
                if (++ema_short_count == half) {
                    ema_short = ema_short_seed / (double)half;
                    ema_short_inited = true;
                }
            } else {
                ema_short += ema_short_alpha * (cci - ema_short);
            }

            if (!ema_long_inited) {
                ema_long_seed += cci;
                if (++ema_long_count == length) {
                    ema_long = ema_long_seed / (double)length;
                    ema_long_inited = true;
                }
            } else {
                ema_long += ema_long_alpha * (cci - ema_long);
            }
        }

        double de = NEO_F64_NAN;
        if (ema_short_inited && ema_long_inited) {
            de = ema_short + ema_short - ema_long;
        }

        if (isfinite(de)) {
            if (!rma_inited) {
                rma_seed += de;
                if (++rma_count == rma_length) {
                    rma = rma_seed / (double)rma_length;
                    rma_inited = true;
                }
            } else {
                rma += rma_alpha * (de - rma);
            }
        }

        const double ccis = rma_inited ? rma : NEO_F64_NAN;
        ccis_ring[slot] = ccis;
        double low = INFINITY;
        double high = -INFINITY;
        for (int k = 0; k < length; ++k) {
            const double value = ccis_ring[k];
            if (isfinite(value)) {
                if (value < low) low = value;
                if (value > high) high = value;
            }
        }
        double f1 = previous_f1;
        if (isfinite(ccis) && isfinite(low) && high > low) {
            f1 = (ccis - low) / (high - low) * 100.0;
        }
        const double pf = previous_pf + factor * (f1 - previous_pf);
        previous_f1 = f1;
        previous_pf = pf;
        pf_ring[slot] = pf;

        low = INFINITY;
        high = -INFINITY;
        for (int k = 0; k < length; ++k) {
            const double value = pf_ring[k];
            if (isfinite(value)) {
                if (value < low) low = value;
                if (value > high) high = value;
            }
        }
        double f2 = previous_f2;
        if (isfinite(low) && high > low) {
            f2 = (pf - low) / (high - low) * 100.0;
        }
        const double pff = previous_pff + factor * (f2 - previous_pff);
        previous_f2 = f2;
        previous_pff = pff;
        o[i] = pff;
        ++segment_bars;
    }
}
