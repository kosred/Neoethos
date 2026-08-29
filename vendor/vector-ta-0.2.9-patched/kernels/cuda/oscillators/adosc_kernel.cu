#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// External formula/edge authority (audit oracle only; never a runtime backend):
// https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADOSC.c


struct KahanF32 {
    float sum;
    float c;
};

__device__ __forceinline__ void kahan_add(KahanF32& s, float x) {

    float y = x - s.c;
    float t = s.sum + y;
    s.c = (t - s.sum) - y;
    s.sum = t;
}


__device__ __forceinline__ float mfm_from_hlc(float h, float l, float c) {
    const float hl = h - l;
    if (!(hl > 0.0f)) return 0.0f;
    const float num = (c - l) - (h - c);
    return num / hl;
}


extern "C" __global__ void adosc_adl_f32(const float* __restrict__ high,
                                         const float* __restrict__ low,
                                         const float* __restrict__ close,
                                         const float* __restrict__ volume,
                                         int series_len,
                                         float* __restrict__ adl_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;


    const float mfm0 = mfm_from_hlc(high[0], low[0], close[0]);
    KahanF32 acc { mfm0 * volume[0], 0.0f };
    adl_out[0] = acc.sum;


    for (int i = 1; i < series_len; ++i) {
        const float mfv = mfm_from_hlc(high[i], low[i], close[i]) * volume[i];
        kahan_add(acc, mfv);
        adl_out[i] = acc.sum;
    }
}


extern "C" __global__ void adosc_batch_from_adl_f32(const float* __restrict__ adl,
                                                    const int*   __restrict__ short_periods,
                                                    const int*   __restrict__ long_periods,
                                                    int series_len,
                                                    int n_combos,
                                                    float* __restrict__ out)
{
    if (series_len <= 0) return;


    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    const int sp = short_periods[combo];
    const int lp = long_periods[combo];
    if (sp <= 0 || lp <= 0 || sp >= lp) {

        return;
    }

    const float a_s = 2.0f / (float)(sp + 1);
    const float a_l = 2.0f / (float)(lp + 1);
    const float oms = 1.0f - a_s;
    const float oml = 1.0f - a_l;

    float* out_row = out + (size_t)combo * (size_t)series_len;


    if (lane == 0) out_row[0] = 0.0f;
    float s_ema = adl[0];
    float l_ema = adl[0];

    const unsigned mask = 0xffffffffu;


    for (int t0 = 1; t0 < series_len; t0 += 32) {
        const int t = t0 + (int)lane;


        float As = 1.0f;
        float Bs = 0.0f;
        float Al = 1.0f;
        float Bl = 0.0f;
        if (t < series_len) {
            const float x = adl[t];
            As = oms;
            Bs = a_s * x;
            Al = oml;
            Bl = a_l * x;
        }


        for (int offset = 1; offset < 32; offset <<= 1) {
            const float As_prev = __shfl_up_sync(mask, As, offset);
            const float Bs_prev = __shfl_up_sync(mask, Bs, offset);
            const float Al_prev = __shfl_up_sync(mask, Al, offset);
            const float Bl_prev = __shfl_up_sync(mask, Bl, offset);
            if (lane >= (unsigned)offset) {
                const float As_cur = As;
                const float Bs_cur = Bs;
                const float Al_cur = Al;
                const float Bl_cur = Bl;
                As = As_cur * As_prev;
                Bs = __fmaf_rn(As_cur, Bs_prev, Bs_cur);
                Al = Al_cur * Al_prev;
                Bl = __fmaf_rn(Al_cur, Bl_prev, Bl_cur);
            }
        }

        const float ys = __fmaf_rn(As, s_ema, Bs);
        const float yl = __fmaf_rn(Al, l_ema, Bl);

        if (t < series_len) {
            out_row[t] = ys - yl;
        }


        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        s_ema = __shfl_sync(mask, ys, last_lane);
        l_ema = __shfl_sync(mask, yl, last_lane);
    }
}


extern "C" __global__ void adosc_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{
    if (short_p <= 0 || long_p <= 0 || short_p >= long_p) return;
    if (rows <= 0 || cols <= 0) return;

    const float a_s = 2.0f / (float)(short_p + 1);
    const float a_l = 2.0f / (float)(long_p + 1);
    const float oms = 1.0f - a_s;
    const float oml = 1.0f - a_l;

    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;


    for (int s = tid; s < cols; s += totalThreads) {
        int idx0 =  0 * cols + s;
        const float mfm0 = mfm_from_hlc(high_tm[idx0], low_tm[idx0], close_tm[idx0]);
        KahanF32 acc { mfm0 * volume_tm[idx0], 0.0f };

        float s_ema = acc.sum;
        float l_ema = acc.sum;
        out_tm[idx0] = 0.0f;

        for (int t = 1; t < rows; ++t) {
            const int idx = t * cols + s;
            const float mfv = mfm_from_hlc(high_tm[idx], low_tm[idx], close_tm[idx]) * volume_tm[idx];
            kahan_add(acc, mfv);
            const float x = acc.sum;
            s_ema = fmaf(a_s, x, oms * s_ema);
            l_ema = fmaf(a_l, x, oml * l_ema);
            out_tm[idx] = s_ema - l_ema;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — adosc
 * ---------------------------------------------------------------------------
 * Creator source oracle (the exact vendored release commit):
 * https://raw.githubusercontent.com/VectorAlpha-dev/VectorTA/e6197777837a18b88e43ce5c163b2e0023f73a2a/src/indicators/adosc.rs
 *
 * Column: output_id "value" — compute_adosc_batch calls expect_value_output
 *   (cpu_batch.rs:2666) and returns out.values.
 *
 * PARAMETER ROUTING: NeoEthos' generic f64 ABI carries the sweep's long-period
 *   anchor in periods[combo]. The CPU plan scales VectorTA's default 3:10 tuple
 *   with positive integer half-up rounding. Reconstructing that exact tuple is
 *   required here: 7/21/50/100/200 map to short 2/6/15/30/60 respectively.
 *
 * FIRST-VALID IGNORED: adosc_prepare returns `first = 0` OUTRIGHT (:331) and
 *   the scalar walks every bar from index 0 with no reset and no warmup
 *   prefix — the accumulation-distribution line is a CUMULATIVE SUM, so
 *   starting anywhere else would give a different series from bar one.
 *
 * Input: (high, low, close, volume) — extract_hlcv_input (cpu_batch.rs:2667)
 *   — F64InputKind::Hlcv.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `sum_ad` is a running sum from
 *   index 0 and both EMAs are seeded FROM IT at bar 0, so nothing here is
 *   bar-parallel.
 *
 * ARITHMETIC taken verbatim:
 *   * the money-flow multiplier is ((c - l) - (h - c)) / hl with the exact
 *     guard hl > 0.0 and the exact substitute 0.0 (:451-455) — no epsilon,
 *     because introducing one would change the answer.
 *   * both EMAs are written as alpha * sum + (1 - alpha) * prev —
 *     TWO products and one add, NOT a fused prev + alpha*(sum - prev).
 *   * both 2/(period+1) alphas and their complements are formed once, exactly
 *     as the CPU forms them, rather than being folded into literals.
 *
 * This file previously contained ZERO double-pointer entry points — every
 * kernel in it was f32. This is the file's f64 lane.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void adosc_neo_batch_f64(const double* __restrict__ high,
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
    (void)first_valid; /* adosc_prepare returns 0 outright — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    const int long_period = periods[combo];
    const long long scaled_short =
        (3LL * (long long)long_period + 5LL) / 10LL;
    const int short_period = (scaled_short < 1LL) ? 1 : (int)scaled_short;

    /* Mirror the CPU request validation before any row arithmetic. */
    if (long_period <= 0 || short_period >= long_period || long_period > n) {
        for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const double alpha_short = 2.0 / ((double)short_period + 1.0);
    const double alpha_long = 2.0 / ((double)long_period + 1.0);
    const double one_minus_alpha_short = 1.0 - alpha_short;
    const double one_minus_alpha_long = 1.0 - alpha_long;

    const double h0 = high[0], l0 = low[0], c0 = close[0], v0 = volume[0];
    const double hl0 = h0 - l0;
    const double mfm0 = (hl0 > 0.0) ? (((c0 - l0) - (h0 - c0)) / hl0) : 0.0;
    const double mfv0 = mfm0 * v0;

    double sum_ad    = mfv0;
    double short_ema = sum_ad;
    double long_ema  = sum_ad;
    o[0] = short_ema - long_ema;

    for (int i = 1; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i], v = volume[i];
        const double hl  = h - l;
        const double mfm = (hl > 0.0) ? (((c - l) - (h - c)) / hl) : 0.0;
        const double mfv = mfm * v;

        sum_ad   += mfv;
        short_ema = alpha_short * sum_ad + one_minus_alpha_short * short_ema;
        long_ema  = alpha_long * sum_ad + one_minus_alpha_long * long_ema;
        o[i] = short_ema - long_ema;
    }
}
