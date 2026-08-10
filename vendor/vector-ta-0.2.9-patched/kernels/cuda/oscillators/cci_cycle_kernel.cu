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
 * S4 f64 LANE — cci_cycle
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/cci_cycle.rs, a FIVE-STAGE pipeline:
 *   `cci_cycle_prepare`             (:398) — first_valid, Err branches
 *   stage 1  `cci_into_slice`       (cci.rs:277) -> `cci_scalar` (cci.rs:317)
 *   stage 2  `cci_cycle_double_ema_in_place`      (:444)
 *   stage 3  the explicit NaN blank                (:508)
 *   stage 4  `smma_into_slice` (smma.rs:1816) -> `smma_scalar` (smma.rs:211)
 *   stage 5  `naive_pf_and_normalize_scalar`       (:744)
 * plus `fmadd` (:46) == `f64::mul_add`, and `cci_cycle_is_finite_fast` (:60),
 * which tests the EXPONENT FIELD and therefore rejects infinities too.
 *
 * PERIOD-INVARIANT, AND FOR A HARD REASON. `compute_cci_cycle_batch`
 * (cpu_batch.rs:3454) reads `length` (10) and `factor` (0.5) and never
 * `period`. Pinning `length` at 10 is not only faithful to that, it is also
 * the only value for which this kernel is the reference at all:
 * `cci_cycle_compute_from_parts:526` routes `length <= 16` to
 * `naive_pf_and_normalize_scalar` and anything larger to
 * `fused_pf_and_normalize_scalar`, which is a DIFFERENT function. A swept
 * length would silently cross that boundary at 17.
 *
 * WHY THIS IS ONE FUSED ASCENDING PASS AND NOT FIVE. Every stage is either a
 * forward recurrence or a window of at most `length` bars:
 *   cci        — rolling sum of `length` closes plus a `length`-term absolute
 *                deviation, both anchored at bar i;
 *   double EMA — two carried means;
 *   smma       — one carried value;
 *   stage 5    — a `length`-wide min/max over the smma output, then a second
 *                `length`-wide min/max over stage 5's own output.
 * So the whole pipeline runs in O(length) per-thread state instead of the
 * reference's four O(n) intermediate vectors, which a per-thread kernel cannot
 * afford. The ORDER of every accumulation is unchanged, which is the part that
 * has to be right.
 *
 * THE ONE PLACE THE FUSION IS SUBTLE. The reference reuses `work` for two
 * different things: stage 2 leaves the double-EMA'd CCI there, stage 4 reads
 * it, and stage 5 then OVERWRITES it with the `pf` series. In a fused pass
 * bar i must be consumed by smma BEFORE stage 5 writes pf[i] over it. That
 * ordering is preserved below and is the reason the smma step appears above
 * the pf step inside the loop body.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. FIVE STACKED STAGES IN f32. cci already divides by `0.015 * mean|dev|`;
 *     the double EMA then differences two nearly equal means
 *     (`mean_s + mean_s - mean_l`), which in f32 is pure cancellation; the
 *     stochastic normalisation then divides by `mx - mn`, a difference of two
 *     window extremes. Three cancellations in series. There is no tolerance
 *     that makes the f32 answer mean anything.
 *  2. `rintf` x2 -> the `smma_p` derivation. The CPU is
 *     `((length as f64).sqrt().round() as usize).max(1)` (:511) — `f64::round`
 *     is HALF-AWAY-FROM-ZERO, which is `round()` in CUDA, NOT `rint()`, which
 *     is round-half-to-EVEN. At length 10 both give 3, but the two disagree at
 *     every half-integer and the f32 file used `rintf`. Written as `round`.
 *  3. `fabsf` x4, `fmaxf` x1, `fminf` x1, `sqrtf` x2 -> the f64 forms.
 *  4. `__int_as_float(0x7f...)` x19 -> `__longlong_as_double(0x7ff8...)`.
 *  5. `0.015f` and `100.0f` -> `0.015` and `100.0`.
 *  6. THE MIN/MAX SCANS ARE COMPARISON CHAINS THAT SKIP NaN EXPLICITLY
 *     (:777-784), NOT `fmax`/`fmin`. They start from ±infinity and only
 *     consider `!v.is_nan()` values, then test `mn.is_finite()` to decide
 *     whether the window had ANY finite member. `fmax` from -inf would give
 *     the same value here, but the `is_finite` guard is what distinguishes an
 *     empty window from a real one and it is reproduced literally.
 *  7. THE STAGE-2 WARM-UP MEANS ARE INCREMENTAL, NOT SUMS.
 *     `mean = ((count - 1) * mean + x) / count` (:473) — three roundings, and
 *     a different number from `sum / count`. Copied as written.
 *
 * ONE THREAD PER COLUMN. Carried: the cci rolling sum, two EMA means and their
 * counts, the smma value, two `length`-wide rings and `out[i-1]`.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_CCICYC_LENGTH 10
#define NEO_CCICYC_FACTOR 0.5
#define NEO_CCICYC_RING   NEO_CCICYC_LENGTH

/* cci_cycle.rs:60 — tests the EXPONENT FIELD, so it rejects inf as well. */
__device__ __forceinline__ bool neo_ccicyc_finite_fast(double x) {
    const unsigned long long EXP_MASK = 0x7ff0000000000000ULL;
    return (__double_as_longlong(x) & (long long)EXP_MASK) != (long long)EXP_MASK;
}

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
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;   /* period-invariant — see the header. */

    const int length = NEO_CCICYC_LENGTH;
    const double factor = NEO_CCICYC_FACTOR;

    /* cci_cycle_prepare:417-433 */
    if (len <= 0 || length > len || first_valid < 0 || first_valid >= len ||
        (len - first_valid) < 2 * length) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int first = first_valid;
    const double inv_p = 1.0 / (double)length;

    /* stage 4's period: `sqrt(length).round().max(1)`. `round` is
     * half-away-from-zero, matching `f64::round`. */
    int smma_p = (int)round(sqrt((double)length));
    if (smma_p < 1) smma_p = 1;

    /* ---- stage 2 state ------------------------------------------------- */
    const int de_start = first + length - 1;         /* cci's first output    */
    const int half = (length + 1) / 2;
    const double alpha_s = 2.0 / ((double)half + 1.0);
    const double beta_s  = 1.0 - alpha_s;
    const double alpha_l = 2.0 / ((double)length + 1.0);
    const double beta_l  = 1.0 - alpha_l;
    const int warm_s = de_start + half   < len ? de_start + half   : len;
    const int warm_l = de_start + length < len ? de_start + length : len;
    double mean_s = 0.0, mean_l = 0.0;
    int count_s = 1, count_l = 1;

    /* ---- stage 1 state ------------------------------------------------- */
    double sum = 0.0;

    /* ---- stage 4 state ------------------------------------------------- */
    const int smma_first = de_start;                 /* first non-NaN of work */
    const int smma_warm  = smma_first + smma_p - 1;
    const double smma_pf   = (double)smma_p;
    const double smma_pm1  = smma_pf - 1.0;
    const double smma_invp = 1.0 / smma_pf;
    double smma_seed_sum = 0.0;
    double smma_prev = 0.0;

    /* ---- stage 5 state ------------------------------------------------- */
    const int stoch_warm = first + length - 1;
    double ring_ccis[NEO_CCICYC_RING];
    double ring_pf[NEO_CCICYC_RING];
    for (int k = 0; k < length; ++k) { ring_ccis[k] = NEO_F64_NAN; ring_pf[k] = NEO_F64_NAN; }
    double prev_f1 = NEO_F64_NAN;
    double prev_pf = NEO_F64_NAN;
    double prev_out = NEO_F64_NAN;

    for (int i = 0; i < len; ++i) {

        /* ---------------- stage 1: cci(length) over `data` --------------- */
        double cci_i = NEO_F64_NAN;
        if (i == de_start) {
            /* cci.rs:333 — the seed sum, ascending over [first, first+length) */
            sum = 0.0;
            for (int k = first; k < first + length; ++k) sum += data[k];
            const double sma = sum * inv_p;
            double sum_abs = 0.0;
            for (int k = first; k < first + length; ++k) sum_abs += fabs(data[k] - sma);
            const double denom = 0.015 * (sum_abs * inv_p);
            cci_i = (denom == 0.0) ? 0.0 : ((data[i] - sma) / denom);
        } else if (i > de_start) {
            /* cci.rs:352-372 */
            sum = sum - data[i - length] + data[i];
            const double sma = sum * inv_p;
            double sabs = 0.0;
            for (int k = i + 1 - length; k <= i; ++k) sabs += fabs(data[k] - sma);
            const double denom = 0.015 * (sabs * inv_p);
            cci_i = (denom == 0.0) ? 0.0 : ((data[i] - sma) / denom);
        }

        /* ---------------- stage 2: the double EMA, in place -------------- */
        double work_i;
        if (i < de_start) {
            work_i = NEO_F64_NAN;                     /* cci's NaN prefix     */
        } else if (i == de_start) {
            mean_s = cci_i;
            mean_l = cci_i;
            work_i = mean_s;                          /* :459 writes the seed */
        } else {
            const double x = cci_i;
            if (i < warm_s) {
                if (neo_ccicyc_finite_fast(x)) {
                    count_s += 1;
                    const double vc = (double)count_s;
                    mean_s = ((vc - 1.0) * mean_s + x) / vc;
                }
            } else if (neo_ccicyc_finite_fast(x)) {
                mean_s = fma(beta_s, mean_s, alpha_s * x);
            }

            if (i < warm_l) {
                if (neo_ccicyc_finite_fast(x)) {
                    count_l += 1;
                    const double vc = (double)count_l;
                    mean_l = ((vc - 1.0) * mean_l + x) / vc;
                }
            } else if (neo_ccicyc_finite_fast(x)) {
                mean_l = fma(beta_l, mean_l, alpha_l * x);
            }

            work_i = mean_s + mean_s - mean_l;
        }

        /* stage 3 (:508) blanks [0, de_start); `work_i` is already NaN there. */

        /* ---------------- stage 4: smma(smma_p) over `work` -------------- */
        double ccis_i = NEO_F64_NAN;
        if (i >= smma_first) {
            if (smma_p == 1) {
                ccis_i = work_i;                      /* smma.rs:215-222      */
            } else if (i < smma_warm) {
                smma_seed_sum += work_i;
            } else if (i == smma_warm) {
                smma_seed_sum += work_i;
                smma_prev = smma_seed_sum * smma_invp;
                ccis_i = smma_prev;
            } else {
                smma_prev = fma(smma_prev, smma_pm1, work_i) * smma_invp;
                ccis_i = smma_prev;
            }
        }

        const int slot = i % NEO_CCICYC_RING;
        ring_ccis[slot] = ccis_i;

        /* ---------------- stage 5, loop 1: stochastic + pf --------------- */
        double pf_i;
        if (i < stoch_warm) {
            pf_i = NEO_F64_NAN;                       /* :758-760             */
        } else if (isnan(ccis_i)) {
            pf_i = NEO_F64_NAN;
            prev_f1 = NEO_F64_NAN;                    /* :769                 */
        } else {
            double mn = INFINITY;
            double mx = -INFINITY;
            for (int k = i + 1 - length; k <= i; ++k) {
                const double v = ring_ccis[((k % NEO_CCICYC_RING) + NEO_CCICYC_RING)
                                            % NEO_CCICYC_RING];
                if (!isnan(v)) {
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            }

            double cur_f1;
            if (isfinite(mn)) {
                const double range = mx - mn;
                if (range > 0.0) {
                    cur_f1 = ((ccis_i - mn) / range) * 100.0;
                } else if (isnan(prev_f1)) {
                    cur_f1 = 50.0;
                } else {
                    cur_f1 = prev_f1;
                }
            } else {
                cur_f1 = NEO_F64_NAN;
            }

            if (isnan(cur_f1)) {
                pf_i = NEO_F64_NAN;
            } else if (isnan(prev_pf) || factor == 0.0) {
                pf_i = cur_f1;
            } else {
                pf_i = fma(cur_f1 - prev_pf, factor, prev_pf);   /* :805 */
            }

            prev_f1 = cur_f1;
            prev_pf = pf_i;
        }

        ring_pf[slot] = pf_i;

        /* ---------------- stage 5, loop 2: normalise -------------------- */
        double out_i;
        if (isnan(pf_i)) {
            out_i = NEO_F64_NAN;                      /* :815-817             */
        } else {
            const int start = (i >= length - 1) ? (i - (length - 1)) : 0;
            double mn = INFINITY, mx = -INFINITY;
            for (int k = start; k <= i; ++k) {
                const double v = ring_pf[((k % NEO_CCICYC_RING) + NEO_CCICYC_RING)
                                          % NEO_CCICYC_RING];
                if (!isnan(v)) {
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            }
            if (!isfinite(mn)) {
                out_i = NEO_F64_NAN;
            } else {
                const double range = mx - mn;
                if (range > 0.0) {
                    const double f2 = ((pf_i - mn) / range) * 100.0;
                    const double prev = (i > 0) ? prev_out : NEO_F64_NAN;
                    if (isnan(prev) || factor == 0.0) out_i = f2;
                    else                              out_i = fma(f2 - prev, factor, prev);
                } else {
                    out_i = (i > 0) ? prev_out : 50.0;
                }
            }
        }

        o[i] = out_i;
        prev_out = out_i;
    }
}
