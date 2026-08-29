#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


#ifndef AT_USE_F64_SUM

#define AT_USE_F64_SUM 0
#endif

#ifndef AT_BLOCK_SIZE
#define AT_BLOCK_SIZE 256
#endif


__device__ __forceinline__ void kahan_add(float& sum, float& c, float x) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

namespace {
__device__ inline bool is_finite(float x) { return !isnan(x) && !isinf(x); }
}

extern "C" __global__ void alphatrend_build_tr_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int first_valid,
    float* __restrict__ tr_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    if (i < first_valid) {
        tr_out[i] = CUDART_NAN_F;
        return;
    }
    if (i == first_valid) {
        tr_out[i] = high[i] - low[i];
        return;
    }
    const float hl = high[i] - low[i];
    const float hc = fabsf(high[i] - close[i - 1]);
    const float lc = fabsf(low[i] - close[i - 1]);
    tr_out[i] = fmaxf(hl, fmaxf(hc, lc));
}

extern "C" __global__ void alphatrend_build_hlc3_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    float* __restrict__ hlc3_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    hlc3_out[i] = (high[i] + low[i] + close[i]) * (1.0f / 3.0f);
}

extern "C" __global__ void alphatrend_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ tr,
    const float* __restrict__ momentum_table,
    const int*   __restrict__ mrow_for_combo,
    const float* __restrict__ coeffs,
    const int*   __restrict__ periods,
    int len,
    int first_valid,
    int n_combos,
    int n_mrows,
    float* __restrict__ k1_out,
    float* __restrict__ k2_out)
{
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid0; row < n_combos; row += stride) {
        const int period = periods[row];
        const float coeff = coeffs[row];
        float* __restrict__ k1_row = k1_out + (size_t)row * len;
        float* __restrict__ k2_row = k2_out + (size_t)row * len;


        if (period <= 0 || period > len) {
            for (int i = 0; i < len; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }
            continue;
        }

        const int warm = first_valid + period - 1;
        if (warm >= len) {
            for (int i = 0; i < len; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }
            continue;
        }

        const int mrow = mrow_for_combo[row];
        if (mrow < 0 || mrow >= n_mrows) {
            for (int i = 0; i < len; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }
            continue;
        }
        const float* __restrict__ mom = momentum_table + (size_t)mrow * len;


        for (int i = 0; i < warm; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }


        const float p_inv = 1.0f / (float)period;
#if AT_USE_F64_SUM
        double s = 0.0;
        for (int j = first_valid; j <= warm; ++j) s += (double)tr[j];
#else
        float s = 0.0f, c = 0.0f;
        for (int j = first_valid; j <= warm; ++j) kahan_add(s, c, tr[j]);
#endif


        float prev_alpha = CUDART_NAN_F;
        float prev1 = CUDART_NAN_F;
        float prev2 = CUDART_NAN_F;

        #pragma unroll 1
        for (int i = warm; i < len; ++i) {

#if AT_USE_F64_SUM
            const float a = (float)(s * (double)p_inv);
#else
            const float a = s * p_inv;
#endif
            const float up = fmaf(-coeff, a, low[i]);
            const float dn = fmaf( coeff, a, high[i]);

            const float m = mom[i];
            const bool m_ge_50 = is_finite(m) ? (m >= 50.0f) : false;


            const float up_clamped = fmaxf(up, prev_alpha);
            const float dn_clamped = fminf(dn, prev_alpha);
            const float cur = m_ge_50 ? up_clamped : dn_clamped;

            k1_row[i] = cur;
            k2_row[i] = prev2;

            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;

            const int nxt = i + 1;
            if (nxt < len) {
#if AT_USE_F64_SUM
                s += (double)tr[nxt] - (double)tr[nxt - period];
#else
                kahan_add(s, c, tr[nxt]);
                kahan_add(s, c, -tr[nxt - period]);
#endif
            }
        }
    }
}


extern "C" __global__ void alphatrend_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ tr_tm,
    const float* __restrict__ momentum_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float coeff,
    int period,
    float* __restrict__ k1_tm_out,
    float* __restrict__ k2_tm_out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;
    const int fv = first_valids[s];

    if (period <= 0 || fv >= series_len) {
        for (int t = 0; t < series_len; ++t) {
            const int idx = t * num_series + s;
            k1_tm_out[idx] = CUDART_NAN_F;
            k2_tm_out[idx] = CUDART_NAN_F;
        }
        return;
    }

    const int warm = fv + period - 1;
    const float p_inv = 1.0f / (float)period;

    if (warm >= series_len) {
        for (int t = 0; t < series_len; ++t) {
            const int idx = t * num_series + s;
            k1_tm_out[idx] = CUDART_NAN_F;
            k2_tm_out[idx] = CUDART_NAN_F;
        }
        return;
    }


    for (int t = 0; t < warm; ++t) {
        const int idx = t * num_series + s;
        k1_tm_out[idx] = CUDART_NAN_F;
        k2_tm_out[idx] = CUDART_NAN_F;
    }


#if AT_USE_F64_SUM
    double ssum = 0.0;
    for (int t = fv; t <= warm; ++t) ssum += (double)tr_tm[t * num_series + s];
#else
    float ssum = 0.0f, csum = 0.0f;
    for (int t = fv; t <= warm; ++t) kahan_add(ssum, csum, tr_tm[t * num_series + s]);
#endif

    float prev_alpha = CUDART_NAN_F, prev1 = CUDART_NAN_F, prev2 = CUDART_NAN_F;

    #pragma unroll 1
    for (int t = warm; t < series_len; ++t) {
        const int idx = t * num_series + s;

#if AT_USE_F64_SUM
        const float a = (float)(ssum * (double)p_inv);
#else
        const float a = ssum * p_inv;
#endif
        const float up = fmaf(-coeff, a, low_tm[idx]);
        const float dn = fmaf( coeff, a, high_tm[idx]);
        const float m  = momentum_tm[idx];
        const bool m_ge_50 = is_finite(m) ? (m >= 50.0f) : false;

        const float up_clamped = fmaxf(up, prev_alpha);
        const float dn_clamped = fminf(dn, prev_alpha);
        const float cur = m_ge_50 ? up_clamped : dn_clamped;

        k1_tm_out[idx] = cur;
        k2_tm_out[idx] = prev2;

        prev2 = prev1;
        prev1 = cur;
        prev_alpha = cur;

        const int nxt = t + 1;
        if (nxt < series_len) {
#if AT_USE_F64_SUM
            ssum += (double)tr_tm[nxt * num_series + s] - (double)tr_tm[(nxt - period) * num_series + s];
#else
            kahan_add(ssum, csum,  tr_tm[nxt * num_series + s]);
            kahan_add(ssum, csum, -tr_tm[(nxt - period) * num_series + s]);
#endif
        }
    }
}


extern "C" __global__ void atr_table_from_tr_f32(
    const float* __restrict__ tr,
    int len,
    int first_valid,
    const int* __restrict__ periods_unique,
    int n_u,
    float* __restrict__ atr_table
){
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_u) return;

    const int period = periods_unique[u];
    float* __restrict__ out = atr_table + (size_t)u * len;

    if (period <= 0 || period > len) {
        for (int i=0;i<len;++i) out[i] = CUDART_NAN_F;
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i=0;i<warm;++i) out[i] = CUDART_NAN_F;

#if AT_USE_F64_SUM
    double s = 0.0;
    for (int j = first_valid; j <= warm; ++j) s += (double)tr[j];
#else
    float s = 0.0f, c = 0.0f;
    for (int j = first_valid; j <= warm; ++j) kahan_add(s, c, tr[j]);
#endif

    const float p_inv = 1.0f / (float)period;

    #pragma unroll 1
    for (int i = warm; i < len; ++i) {
#if AT_USE_F64_SUM
        out[i] = (float)(s * (double)p_inv);
#else
        out[i] = s * p_inv;
#endif
        const int nxt = i + 1;
        if (nxt < len) {
#if AT_USE_F64_SUM
            s += (double)tr[nxt] - (double)tr[nxt - period];
#else
            kahan_add(s, c, tr[nxt]);
            kahan_add(s, c, -tr[nxt - period]);
#endif
        }
    }
}

extern "C" __global__ void momentum_to_mask_bits(
    const float* __restrict__ momentum_table,
    int len, int n_mrows,
    unsigned* __restrict__ mask_bits
){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_mrows) return;

    const float* __restrict__ mrow = momentum_table + (size_t)row * len;
    const int n_words = (len + 31) >> 5;
    unsigned* __restrict__ out = mask_bits + (size_t)row * n_words;

    for (int w = 0; w < n_words; ++w) {
        unsigned word = 0u;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            const int i = (w << 5) + b;
            if (i >= len) break;
            const float m = mrow[i];
            const unsigned bit = (is_finite(m) && m >= 50.0f) ? 1u : 0u;
            word |= (bit << b);
        }
        out[w] = word;
    }
}

__device__ __forceinline__ bool mask_test(const unsigned* __restrict__ row, int i){
    const unsigned w = row[i >> 5];
    return (w >> (i & 31)) & 1u;
}

extern "C" __global__ void alphatrend_batch_from_precomputed_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ atr_table,
    const unsigned* __restrict__ mask_bits,
    const int* __restrict__ period_row_for_combo,
    const int* __restrict__ mrow_for_combo,
    const float* __restrict__ coeffs,
    const int*   __restrict__ periods,
    int len,
    int first_valid,
    int n_combos,
    int n_pr, int n_mrows,
    float* __restrict__ k1_out, float* __restrict__ k2_out)
{
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid0; row < n_combos; row += stride) {
        const int period = periods[row];
        float* __restrict__ k1_row = k1_out + (size_t)row * len;
        float* __restrict__ k2_row = k2_out + (size_t)row * len;

        if (period <= 0 || period > len) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const int warm = first_valid + period - 1;
        if (warm >= len) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const int pr = period_row_for_combo[row];
        if (pr < 0 || pr >= n_pr) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const int mrow = mrow_for_combo[row];
        if (mrow < 0 || mrow >= n_mrows) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const float* __restrict__ arow = atr_table + (size_t)pr * len;
        const int n_words = (len + 31) >> 5;
        const unsigned* __restrict__ mask_row = mask_bits + (size_t)mrow * n_words;
        const float coeff = coeffs[row];

        for (int i=0;i<warm;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }

        float prev_alpha = CUDART_NAN_F, prev1 = CUDART_NAN_F, prev2 = CUDART_NAN_F;
        int word_idx = warm >> 5;
        unsigned mask_word = mask_row[word_idx];
        unsigned bit = 1u << (warm & 31);

        #pragma unroll 1
        for (int i = warm; i < len; ++i){
            const float a = arow[i];
            const float up = fmaf(-coeff, a, low[i]);
            const float dn = fmaf( coeff, a, high[i]);

            const bool m_ge_50 = (mask_word & bit) != 0u;
            const float cur = m_ge_50 ? fmaxf(up, prev_alpha) : fminf(dn, prev_alpha);

            k1_row[i] = cur;
            k2_row[i] = prev2;

            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;

            bit <<= 1;
            if (bit == 0u && (i + 1) < len) {
                ++word_idx;
                mask_word = mask_row[word_idx];
                bit = 1u;
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * Every other entry point in this file is f32. This section adds the f64 lane
 * entry points beside them; the f32 wrappers are untouched, and listing this
 * file in build.rs opts the WHOLE translation unit out of `--use_fast_math`.
 *
 * CPU REFERENCE: src/indicators/alphatrend.rs `alphatrend_scalar` (:578-690),
 *   with `alphatrend_prepare` (:448-528) for the warmup, and
 *   src/indicators/mfi.rs `mfi_scalar` (:235) / `mfi_prepare` (:144) for the
 *   momentum series.
 *   Batch dispatcher accepts the canonical registry identities `k1` and `k2`
 *   only. Both entry points below share the same per-row state authority; the
 *   source-stable primary ABI emits k1 and the full ABI emits k1 plus k2.
 *
 * INPUT: (high, low, close, volume) -- F64InputKind::Hlcv. The CPU batch calls
 *   `extract_ohlcv_full_input` (:13973) but `alphatrend_scalar` binds open as
 *   `_open` (:579) and NEVER reads it, so a four-pointer shape is faithful.
 *
 * FIRST-VALID = CLOSE ALONE: `alphatrend_prepare` scans
 *   `close.iter().position(|x| !x.is_nan())` (:493) -- high, low and volume are
 *   never scanned. That is exactly `F64FirstValidRule::HlcCloseOnly`, the rule
 *   `adxr` already declares. Adopting the Hlc triple's index would shift the
 *   whole series on any frame where high or low starts later than close, and
 *   `first` sets BOTH the NaN prefix and the seed window.
 *
 * PERIOD-SWEPT: `period` is the swept parameter (cpu_batch.rs:13998) and it is
 *   BOTH the true-range window AND the MFI period. `coeff` is pinned at the
 *   CPU default 1.0 (:13997) and `no_volume` at false (:13999), which selects
 *   the MFI branch (:610-630) rather than the RSI branch.
 *
 * TWO INDEPENDENT FIRST-VALID SCANS, DELIBERATELY: alphatrend's own `first` is
 *   close-only, but the MFI it consumes runs `mfi_prepare` on ITS OWN inputs
 *   and scans `!typical_price[i].is_nan() && !volume[i].is_nan()` (mfi.rs:164).
 *   `typical_price` is hlc3, so the MFI index depends on high and low as well
 *   and can sit LATER than alphatrend's. Both are derived here; collapsing
 *   them to one would move the MFI warmup and therefore every `m_check`.
 *
 * NOTE ON `is_nan` vs `is_finite`: both scans use `!is_nan`, so an INFINITE
 *   bar is ACCEPTED by the CPU. `isnan` is used here for that reason;
 *   `isfinite` would skip a bar the CPU counts and shift the whole series.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. `prev_alpha` is a RATCHET --
 *   the level can only move in the direction the momentum check selects -- and
 *   the true-range sum is a sliding window maintained by subtract-then-add.
 *   Neither is bar-parallel.
 *
 * ARITHMETIC taken verbatim:
 *   * the true range is `hl.max(hc).max(lc)` (:604) -- f64::max, hence fmax,
 *     which returns the non-NaN operand. An if-chain would let a NaN survive
 *     into the sliding sum and poison every later bar.
 *   * the sliding sum is seeded by a plain forward accumulation over
 *     `first..=warmup` (:635-638) and then maintained as
 *     `sum += tr[i+1] - tr[i+1-period]` (:679) -- the DIFFERENCE is formed
 *     first, then added, which is what the CPU writes.
 *   * `a = sum / period` (:645), then `up_t = low - a*coeff` and
 *     `down_t = high + a*coeff` (:647-648) -- the product is formed before the
 *     subtraction, TWO roundings, not a fused multiply-add.
 *   * `m_check` is `momentum_values[i] >= 50.0` (:649). A NaN momentum makes
 *     that comparison FALSE, which selects `down_t`; the NaN is never
 *     propagated into the output, and that is the CPU's behaviour.
 *   * the MFI ratio guard is `total < 1e-14` (mfi.rs:291) -- an f64-sized
 *     tolerance already present in the CPU source, carried across unchanged.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:13997-13999 */
#define NEO_AT_COEFF 1.0
/* Both rings are a function of the SWEPT period, so the kernel carries a bound
 * and the host reports it through `F64Kernel::max_period`. */
#define NEO_AT_MAX_PERIOD 512
/* mfi.rs:291 -- already f64-sized. */
#define NEO_AT_MFI_TOL 1e-14

__device__ __forceinline__
void alphatrend_row_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        const double* __restrict__ close,
                        const double* __restrict__ volume,
                        int n,
                        int period,
                        int first_valid,
                        double* __restrict__ out_k1,
                        double* __restrict__ out_k2)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) {
        out_k1[i] = NEO_F64_NAN;
        if (out_k2 != nullptr) out_k2[i] = NEO_F64_NAN;
    }

    if (period <= 0 || period > n || period > NEO_AT_MAX_PERIOD) return;

    /* alphatrend_prepare (:493) -- CLOSE ALONE, `!is_nan`. The caller's index
     * is honoured when it is in range and re-derived otherwise, so a caller
     * that resolved `HlcCloseOnly` and this kernel cannot disagree. */
    int first = first_valid;
    if (first < 0 || first >= n || isnan(close[first])) {
        first = -1;
        for (int i = 0; i < n; ++i) { if (!isnan(close[i])) { first = i; break; } }
    }
    if (first < 0) return;                 /* AllValuesNaN */
    if (n - first < period) return;        /* NotEnoughValidData */

    /* mfi_prepare (:164) -- hlc3 AND volume, `!is_nan`, its OWN scan. */
    int mfi_first = -1;
    for (int i = 0; i < n; ++i) {
        const double tp = (high[i] + low[i] + close[i]) / 3.0;
        if (!isnan(tp) && !isnan(volume[i])) { mfi_first = i; break; }
    }
    /* mfi_with_kernel returning Err makes alphatrend_scalar return Err, which
     * the batch dispatcher turns into a failed request -- an all-NaN row is
     * the honest device answer, not a silently different series. */
    if (mfi_first < 0) return;
    if (n - mfi_first < period) return;

    const int warmup = first + period - 1;
    if (warmup >= n) return;

    const double pf    = (double)period;
    const double coeff = NEO_AT_COEFF;

    /* True-range ring. Depth period + 1: at bar i the CPU adds tr[i+1] and
     * subtracts tr[i+1-period], a span of `period`, so one extra slot keeps
     * the subtrahend alive across the store. */
    double tr_ring[NEO_AT_MAX_PERIOD + 2];
    const int tr_cap = period + 1;
    for (int k = 0; k < tr_cap; ++k) tr_ring[k] = 0.0;

    /* MFI rings (mfi.rs:247 -- one allocation split in two). */
    double pos_ring[NEO_AT_MAX_PERIOD];
    double neg_ring[NEO_AT_MAX_PERIOD];
    for (int k = 0; k < period; ++k) { pos_ring[k] = 0.0; neg_ring[k] = 0.0; }
    double pos_sum = 0.0, neg_sum = 0.0;
    double mfi_prev = (high[mfi_first] + low[mfi_first] + close[mfi_first]) / 3.0;
    int    mfi_ring = 0;
    const int mfi_seed_end = mfi_first + period;   /* exclusive */
    const int mfi_idx0     = mfi_seed_end - 1;

    double tr_sum = 0.0;
    double prev_alpha = NEO_F64_NAN;
    double prev1 = NEO_F64_NAN;
    double prev2 = NEO_F64_NAN;

    for (int i = 0; i < n; ++i) {
        /* ---- true range (:600-606) ---- */
        if (i >= first) {
            double tr;
            if (i == first) {
                tr = high[i] - low[i];
            } else {
                const double hl = high[i] - low[i];
                const double hc = fabs(high[i] - close[i - 1]);
                const double lc = fabs(low[i]  - close[i - 1]);
                tr = fmax(fmax(hl, hc), lc);
            }
            tr_ring[i % tr_cap] = tr;
            if (i <= warmup) {
                /* the seed accumulation (:635-638) */
                tr_sum += tr;
            } else {
                /* the CPU's end-of-iteration update, applied at the head of
                 * the bar it takes effect on (:679) */
                tr_sum += tr - tr_ring[(i - period) % tr_cap];
            }
        }

        /* ---- MFI (mfi.rs:258-300) ---- */
        double mfi_value = NEO_F64_NAN;
        if (i > mfi_first) {
            const double tp_i = (high[i] + low[i] + close[i]) / 3.0;
            const double flow = tp_i * volume[i];
            const double diff = tp_i - mfi_prev;
            mfi_prev = tp_i;

            if (i >= mfi_seed_end) {
                pos_sum -= pos_ring[mfi_ring];
                neg_sum -= neg_ring[mfi_ring];
            }

            const double gt = (diff > 0.0) ? 1.0 : 0.0;
            const double lt = (diff < 0.0) ? 1.0 : 0.0;
            const double pos_new = flow * gt;
            const double neg_new = flow * lt;

            pos_ring[mfi_ring] = pos_new;
            neg_ring[mfi_ring] = neg_new;
            pos_sum += pos_new;
            neg_sum += neg_new;
            mfi_ring += 1; if (mfi_ring == period) mfi_ring = 0;
        }
        if (i >= mfi_idx0) {
            const double total = pos_sum + neg_sum;
            mfi_value = (total < NEO_AT_MFI_TOL) ? 0.0 : (100.0 * (pos_sum / total));
        }

        /* ---- the alphatrend ratchet (:640-680) ---- */
        if (i >= warmup) {
            const double a = tr_sum / pf;
            const double up_t   = low[i]  - a * coeff;
            const double down_t = high[i] + a * coeff;
            /* NaN >= 50.0 is FALSE -- see header. */
            const bool m_check = (mfi_value >= 50.0);

            double cur;
            if (i == warmup) {
                cur = m_check ? up_t : down_t;
            } else if (m_check) {
                cur = (up_t < prev_alpha) ? prev_alpha : up_t;
            } else {
                cur = (down_t > prev_alpha) ? prev_alpha : down_t;
            }

            out_k1[i] = cur;
            if (out_k2 != nullptr && i >= warmup + 2) out_k2[i] = prev2;
            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;
        }
    }
}

/* Preserve the primary entry point's exact ABI for every existing generic
 * f64 dispatcher consumer. It shares the full state machine and requests k1
 * only; no auxiliary allocation or host discard is introduced. */
extern "C" __global__
void alphatrend_neo_batch_f64(const double* __restrict__ high,
                              const double* __restrict__ low,
                              const double* __restrict__ close,
                              const double* __restrict__ volume,
                              int n,
                              const int* __restrict__ periods,
                              int n_combos,
                              int first_valid,
                              double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    double* __restrict__ out_k1 = out + (size_t)combo * (size_t)n;
    alphatrend_row_f64(high, low, close, volume, n, periods[combo], first_valid,
                       out_k1, nullptr);
}

/* Canonical full-output ABI: one parameter-row thread, one launch, exact k1
 * plus the CPU's two-bar-lagged k2. */
extern "C" __global__
void alphatrend_outputs_f64(const double* __restrict__ high,
                            const double* __restrict__ low,
                            const double* __restrict__ close,
                            const double* __restrict__ volume,
                            int n,
                            const int* __restrict__ periods,
                            int n_combos,
                            int first_valid,
                            double* __restrict__ out_k1,
                            double* __restrict__ out_k2)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    const size_t offset = (size_t)combo * (size_t)n;
    alphatrend_row_f64(high, low, close, volume, n, periods[combo], first_valid,
                       out_k1 + offset, out_k2 + offset);
}
