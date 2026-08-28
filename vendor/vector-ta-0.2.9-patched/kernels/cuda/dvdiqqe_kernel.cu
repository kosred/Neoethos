#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>


static __forceinline__ __device__ float dvdiqqe_tick_volume(
    const float o, const float c, const float tick, float &tickrng_prev)
{
    const float rng = c - o;
    const float arng = fabsf(rng);
    const float tickrng = (arng < tick) ? tickrng_prev : rng;
    tickrng_prev = tickrng;
    const float tv = (tick > 0.0f) ? fabsf(tickrng) / tick : 0.0f;
    return tv > 0.0f ? tv : 0.0f;
}


static __forceinline__ __device__ float select_volume(
    const float vol_opt, const float tick_vol, const int use_tick_only)
{
    if (use_tick_only) return tick_vol;
    if (isfinite(vol_opt)) return vol_opt;
    return tick_vol;
}


static __forceinline__ __device__ void kahan_add(float &sum, float &comp, const float x)
{
    float y = x - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}


static __forceinline__ __device__ float ema_update(const float y, const float x, const float a)
{
    return __fmaf_rn(a, (x - y), y);
}


static __forceinline__ __device__ float qnan_f32() { return nanf(""); }

extern "C" __global__
void dvdiqqe_batch_f32(
    const float* __restrict__ open,
    const float* __restrict__ close,
    const float* __restrict__ volume,
    const int   has_volume,
    const int*  __restrict__ periods,
    const int*  __restrict__ smoothings,
    const float* __restrict__ fast_mults,
    const float* __restrict__ slow_mults,
    const int   n_combos,
    const int   series_len,
    const int   first_valid,
    const float tick_size,
    const int   center_dynamic,
    float* __restrict__ out_dvdi,
    float* __restrict__ out_fast,
    float* __restrict__ out_slow,
    float* __restrict__ out_center)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0 || first_valid >= series_len) return;

    const int   period    = periods[combo];
    const int   smoothing = smoothings[combo];
    const float fast_mult = fast_mults[combo];
    const float slow_mult = slow_mults[combo];
    if (period <= 0 || smoothing <= 0) return;

    const int   wper = period * 2 - 1;
    const int   warm = first_valid + wper;

    float* dvdi_row = out_dvdi  + combo * series_len;
    float* fast_row = out_fast  + combo * series_len;
    float* slow_row = out_slow  + combo * series_len;
    float* cent_row = out_center+ combo * series_len;


    const float a_p = 2.0f / (float)(period + 1);
    const float a_s = 2.0f / (float)(smoothing + 1);
    const float a_r = 2.0f / (float)(wper + 1);


    float pvi_sum = 0.0f, pvi_c = 0.0f;
    float nvi_sum = 0.0f, nvi_c = 0.0f;

    float pvi_ema = 0.0f, nvi_ema = 0.0f; int pvi_cnt = 0;
    float pdiv_ema = 0.0f, ndiv_ema = 0.0f; int div_cnt = 0;

    float dvdi_prev = 0.0f; bool dvdi_inited = false;

    float rng_ema1 = 0.0f, rng_ema2 = 0.0f; int rng1_cnt = 0, rng2_cnt = 0; bool rng2_ready = false;

    float center_mean = 0.0f; int center_cnt = 0;

    float prev_vol = 0.0f;
    float prev_close = 0.0f;
    float tickrng_prev = tick_size;


    if (threadIdx.x == 0)
    {
        #pragma unroll 1
        for (int t = 0; t < series_len; ++t) {
            const float oi = open[t];
            const float ci = close[t];
            if (!isfinite(ci)) { continue; }


            const float tick_vol = dvdiqqe_tick_volume(oi, ci, tick_size, tickrng_prev);
            const float real_vol = has_volume ? volume[t] : NAN;
            const float sel_vol  = select_volume(real_vol, tick_vol,  0);

            if (t == 0) { prev_close = ci; prev_vol = sel_vol; }

            const float dpc = ci - prev_close;
            if (sel_vol > prev_vol) { kahan_add(pvi_sum, pvi_c,  dpc); }
            if (sel_vol < prev_vol) { kahan_add(nvi_sum, nvi_c, -dpc); }
            prev_close = ci; prev_vol = sel_vol;


            if (t >= first_valid) {
                const float pvi_val = pvi_sum + pvi_c;
                const float nvi_val = nvi_sum + nvi_c;
                if (pvi_cnt < period) {

                    pvi_cnt += 1;
                    const float inv = 1.0f / (float)pvi_cnt;
                    pvi_ema = __fmaf_rn((pvi_val - pvi_ema), inv, pvi_ema);
                    nvi_ema = __fmaf_rn((nvi_val - nvi_ema), inv, nvi_ema);
                } else {
                    pvi_ema = ema_update(pvi_ema, pvi_val, a_p);
                    nvi_ema = ema_update(nvi_ema, nvi_val, a_p);
                }
            }

            const float pdiv = (pvi_sum + pvi_c) - pvi_ema;
            const float ndiv = (nvi_sum + nvi_c) - nvi_ema;


            if (t >= first_valid) {
                if (div_cnt < smoothing) {
                    div_cnt += 1;
                    const float inv = 1.0f / (float)div_cnt;
                    pdiv_ema = __fmaf_rn((pdiv - pdiv_ema), inv, pdiv_ema);
                    ndiv_ema = __fmaf_rn((ndiv - ndiv_ema), inv, ndiv_ema);
                } else {
                    pdiv_ema = ema_update(pdiv_ema, pdiv, a_s);
                    ndiv_ema = ema_update(ndiv_ema, ndiv, a_s);
                }
            }

            const float dv = pdiv_ema - ndiv_ema;
            dvdi_row[t] = dv;


            if (!dvdi_inited) { dvdi_prev = dv; dvdi_inited = true; }
            const float abs_delta = fabsf(dv - dvdi_prev);
            if (t >= first_valid + 1) {
                if (rng1_cnt < wper) {
                    rng1_cnt += 1;
                    const float inv = 1.0f / (float)rng1_cnt;
                    rng_ema1 = __fmaf_rn((abs_delta - rng_ema1), inv, rng_ema1);
                } else {
                    rng_ema1 = ema_update(rng_ema1, abs_delta, a_r);
                }
                if (!rng2_ready) { rng2_ready = true; rng2_cnt = 0; }
                if (rng2_cnt < wper) {
                    rng2_cnt += 1;
                    const float inv2 = 1.0f / (float)rng2_cnt;
                    rng_ema2 = __fmaf_rn((rng_ema1 - rng_ema2), inv2, rng_ema2);
                } else {
                    rng_ema2 = ema_update(rng_ema2, rng_ema1, a_r);
                }
            }


            if (t == warm && rng2_cnt >= 1) {
                fast_row[t] = dv; slow_row[t] = dv;
            } else if (t > warm && rng2_cnt >= wper) {
                const float fr = rng_ema2 * fast_mult;
                const float sr = rng_ema2 * slow_mult;

                const float prev_fast = fast_row[t - 1];
                if (dv > prev_fast) {
                    const float nv = dv - fr;
                    fast_row[t] = (nv < prev_fast) ? prev_fast : nv;
                } else {
                    const float nv = dv + fr;
                    fast_row[t] = (nv > prev_fast) ? prev_fast : nv;
                }

                const float prev_slow = slow_row[t - 1];
                if (dv > prev_slow) {
                    const float nv = dv - sr;
                    slow_row[t] = (nv < prev_slow) ? prev_slow : nv;
                } else {
                    const float nv = dv + sr;
                    slow_row[t] = (nv > prev_slow) ? prev_slow : nv;
                }
            }


            if (t >= warm) {
                if (center_dynamic) {

                    center_cnt += 1;
                    const float invc = 1.0f / (float)center_cnt;
                    center_mean = __fmaf_rn((dv - center_mean), invc, center_mean);
                    cent_row[t] = center_mean;
                } else {
                    cent_row[t] = 0.0f;
                }
            }

            dvdi_prev = dv;
        }
    }

    __syncthreads();


    const float qnan = qnan_f32();
    for (int i = threadIdx.x; i < series_len && i < warm; i += blockDim.x) {
        dvdi_row[i] = qnan;
        fast_row[i] = qnan;
        slow_row[i] = qnan;
        cent_row[i] = qnan;
    }
}


extern "C" __global__
void dvdiqqe_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int   has_volume,
    const int*  __restrict__ first_valids,
    const int   period,
    const int   smoothing,
    const float fast_mult,
    const float slow_mult,
    const float tick_size,
    const int   center_dynamic,
    const int   num_series,
    const int   series_len,
    float* __restrict__ dvdi_tm,
    float* __restrict__ fast_tm,
    float* __restrict__ slow_tm,
    float* __restrict__ center_tm)
{
    if (period <= 0 || smoothing <= 0 || num_series <= 0 || series_len <= 0) return;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    const int wper = period * 2 - 1;
    const float a_p = 2.0f / (float)(period + 1);
    const float a_s = 2.0f / (float)(smoothing + 1);
    const float a_r = 2.0f / (float)(wper + 1);

    const int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep    = gridDim.x  * warps_per_block;
    const int cols     = num_series;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) {

            const float qnan = qnan_f32();
            for (int t = lane; t < series_len; t += warpSize) {
                const int idx = t * cols + s;
                dvdi_tm[idx] = qnan; fast_tm[idx] = qnan; slow_tm[idx] = qnan; center_tm[idx] = qnan;
            }
            continue;
        }

        const int warm = first_valid + wper;


        float pvi_sum = 0.0f, pvi_c = 0.0f;
        float nvi_sum = 0.0f, nvi_c = 0.0f;

        float pvi_ema = 0.0f, nvi_ema = 0.0f; int pvi_cnt = 0;
        float pdiv_ema = 0.0f, ndiv_ema = 0.0f; int div_cnt = 0;

        float dvdi_prev = 0.0f; bool dvdi_inited = false;

        float rng_ema1 = 0.0f, rng_ema2 = 0.0f; int rng1_cnt = 0, rng2_cnt = 0; bool rng2_ready = false;

        float center_mean = 0.0f; int center_cnt = 0;

        float prev_vol = 0.0f;
        float prev_close = 0.0f;
        float tickrng_prev = tick_size;

        if (lane == 0) {
            #pragma unroll 1
            for (int t = 0; t < series_len; ++t) {
                const int idx = t * cols + s;
                const float oi = open_tm[idx];
                const float ci = close_tm[idx];
                if (!isfinite(ci)) continue;

                const float tick_vol = dvdiqqe_tick_volume(oi, ci, tick_size, tickrng_prev);
                const float vol_tm   = has_volume ? volume_tm[idx] : NAN;
                const float sel_vol  = select_volume(vol_tm, tick_vol,  0);

                if (t == 0) { prev_close = ci; prev_vol = sel_vol; }

                const float dpc = ci - prev_close;
                if (sel_vol > prev_vol) { kahan_add(pvi_sum, pvi_c,  dpc); }
                if (sel_vol < prev_vol) { kahan_add(nvi_sum, nvi_c, -dpc); }
                prev_close = ci; prev_vol = sel_vol;

                if (t >= first_valid) {
                    const float pvi_val = pvi_sum + pvi_c;
                    const float nvi_val = nvi_sum + nvi_c;
                    if (pvi_cnt < period) {
                        pvi_cnt += 1;
                        const float inv = 1.0f / (float)pvi_cnt;
                        pvi_ema = __fmaf_rn((pvi_val - pvi_ema), inv, pvi_ema);
                        nvi_ema = __fmaf_rn((nvi_val - nvi_ema), inv, nvi_ema);
                    } else {
                        pvi_ema = ema_update(pvi_ema, pvi_val, a_p);
                        nvi_ema = ema_update(nvi_ema, nvi_val, a_p);
                    }
                }

                const float pdiv = (pvi_sum + pvi_c) - pvi_ema;
                const float ndiv = (nvi_sum + nvi_c) - nvi_ema;

                if (t >= first_valid) {
                    if (div_cnt < smoothing) {
                        div_cnt += 1;
                        const float inv = 1.0f / (float)div_cnt;
                        pdiv_ema = __fmaf_rn((pdiv - pdiv_ema), inv, pdiv_ema);
                        ndiv_ema = __fmaf_rn((ndiv - ndiv_ema), inv, ndiv_ema);
                    } else {
                        pdiv_ema = ema_update(pdiv_ema, pdiv, a_s);
                        ndiv_ema = ema_update(ndiv_ema, ndiv, a_s);
                    }
                }

                const float dv = pdiv_ema - ndiv_ema;
                dvdi_tm[idx] = dv;

                if (!dvdi_inited) { dvdi_prev = dv; dvdi_inited = true; }
                const float abs_delta = fabsf(dv - dvdi_prev);
                if (t >= first_valid + 1) {
                    if (rng1_cnt < wper) {
                        rng1_cnt += 1;
                        const float inv = 1.0f / (float)rng1_cnt;
                        rng_ema1 = __fmaf_rn((abs_delta - rng_ema1), inv, rng_ema1);
                    } else {
                        rng_ema1 = ema_update(rng_ema1, abs_delta, a_r);
                    }
                    if (!rng2_ready) { rng2_ready = true; rng2_cnt = 0; }
                    if (rng2_cnt < wper) {
                        rng2_cnt += 1;
                        const float inv2 = 1.0f / (float)rng2_cnt;
                        rng_ema2 = __fmaf_rn((rng_ema1 - rng_ema2), inv2, rng_ema2);
                    } else {
                        rng_ema2 = ema_update(rng_ema2, rng_ema1, a_r);
                    }
                }

                if (t == warm && rng2_cnt >= 1) {
                    fast_tm[idx] = dv; slow_tm[idx] = dv;
                } else if (t > warm && rng2_cnt >= wper) {
                    const float fr = rng_ema2 * fast_mult;
                    const float sr = rng_ema2 * slow_mult;
                    const float pf = fast_tm[(t - 1) * cols + s];
                    const float ps = slow_tm[(t - 1) * cols + s];

                    if (dv > pf) { const float nv = dv - fr; fast_tm[idx] = (nv < pf) ? pf : nv; }
                    else          { const float nv = dv + fr; fast_tm[idx] = (nv > pf) ? pf : nv; }

                    if (dv > ps) { const float nv = dv - sr; slow_tm[idx] = (nv < ps) ? ps : nv; }
                    else          { const float nv = dv + sr; slow_tm[idx] = (nv > ps) ? ps : nv; }
                }

                if (t >= warm) {
                    if (center_dynamic) {
                        center_cnt += 1;
                        const float invc = 1.0f / (float)center_cnt;
                        center_mean = __fmaf_rn((dv - center_mean), invc, center_mean);
                        center_tm[idx] = center_mean;
                    } else {
                        center_tm[idx] = 0.0f;
                    }
                }

                dvdi_prev = dv;
            }
        }


        const float qnan = qnan_f32();
        for (int t = lane; t < series_len && t < warm; t += warpSize) {
            const int idx = t * cols + s;
            dvdi_tm[idx] = qnan; fast_tm[idx] = qnan; slow_tm[idx] = qnan; center_tm[idx] = qnan;
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — dvdiqqe
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/dvdiqqe.rs
 *   `dvdiqqe_prepare`      (:333) — first_valid scans CLOSE for `is_finite`
 *                                   (NOT `!is_nan`), plus every Err branch
 *   `dvdiqqe_with_kernel`  (:847) — warmup = first + (period * 2 - 1)
 *   `dvdiqqe_compute_into` (:445) — the PVI/NVI accumulator and the four
 *                                   chained EMAs that produce the dvdi line
 * and `ema_scalar_into` (moving_averages/ema.rs:461) for every EMA stage —
 * seed at the first non-NaN, incremental mean for `period` bars, then
 * `beta.mul_add(prev, alpha * x)`.
 *
 * WHICH SERIES THIS EMITS. The canonical family is exactly `dvdi`, `fast_tl`,
 * `slow_tl`, `center_line`. The full ABI materializes all four in one launch;
 * the preserved primary ABI materializes only `dvdi` but delegates to the same
 * row authority and therefore retains no second arithmetic implementation.
 *
 * INPUT IS (open, close, volume). `dvdiqqe_compute_into` takes high and low as
 * `_high` and `_low` — it never reads them (:447-448). Declaring an Hlc or
 * Ohlc shape would hand this kernel two series it does not use and, worse,
 * would not hand it `open`, which the tick-range term needs at every bar.
 *
 * PERIOD-SWEPT. `compute_dvdiqqe_batch`
 * (cpu_batch.rs:14490-14496) reads `period` (13), `smoothing_period` (6),
 * two multipliers, `volume_type` ("default"), `center_type` ("dynamic") and
 * `tick_size` (0.01). It DOES read a parameter literally named `period` — but
 * the f64 lane's swept int is fed to `periods[combo]`, and this kernel uses it
 * as that `period`, so this one is genuinely SWEPT, not invariant.
 *
 * FIRST-VALID USES `is_finite`, NOT `!is_nan`. dvdiqqe.rs:385 is
 * `c.iter().position(|x| x.is_finite())`, so a leading +inf is SKIPPED where
 * every other indicator in this shard would accept it. Declared as its own
 * rule rather than folded into `AllInputsNonNan`.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. PVI AND NVI ARE UNBOUNDED RUNNING SUMS OF CLOSE DIFFERENCES over the
 *     WHOLE series, started at index 0 and never reset. Over 100k bars they
 *     drift to a magnitude where an f32 accumulator's ulp exceeds a single
 *     bar's `d_close` on an FX series, and the accumulator FREEZES. The f32
 *     kernel's dvdi line therefore flattens on long series — not by a
 *     tolerance, by construction.
 *  2. FOUR CHAINED EMAs, and the last stage is a DIFFERENCE of two of them
 *     (`pdiv - ndiv`). Cancellation on top of a frozen accumulator.
 *  3. `__fmaf_rn` x15 -> `fma`, matching `f64::mul_add` one-for-one on each
 *     EMA step: ONE rounding on the fused pair, a separate rounding on
 *     `alpha * x`. Two per step, not three.
 *  4. `fabsf` x4 -> `fabs`.
 *  5. `__int_as_float(0x7f...)` -> `__longlong_as_double(0x7ff8...)`.
 *  6. `tick` DEFAULTS TO 0.01 AND IS A DIVISOR (`|tickrng| / tick`). In f32
 *     0.01f is 0.00999999977648258; the quotient it produces is a different
 *     number from the f64 one at every bar, and `tick_vol` is then COMPARED
 *     against volume, so the difference is not a rounding — it flips which
 *     branch of `sel_vol > prev_vol` runs and therefore which bars contribute
 *     to PVI at all. A discrete, compounding divergence.
 *  7. `(tickrng.abs() / tick).max(0.0)` is `f64::max`, so `fmax` — a NaN
 *     quotient becomes 0.0, not NaN.
 *
 * THE INITIAL STATE IS NOT ZERO EVERYWHERE. `tickrng_prev` starts at `tick`
 * (:485), while `prev_close`, `prev_vol`, `pvi_prev` and `nvi_prev` start at
 * 0.0. So bar 0 has `d_close = close[0] - 0.0`, a full price, and it enters
 * PVI whenever `sel_vol > 0`. Copied, because it sets the level of the entire
 * PVI series.
 *
 * ONE THREAD PER COLUMN. Carried: pvi, nvi, prev_close, prev_vol, tickrng_prev,
 * six EMA states, two ratchets, and the cumulative center sum/count. Every
 * stage is fused into ONE ascending pass — each value at bar i needs only bar
 * i's input and its own prior carry, so the reference's intermediate vectors
 * collapse to scalar state without moving a single rounding.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_DVDIQQE_SMOOTHING 6
#define NEO_DVDIQQE_FAST      2.618
#define NEO_DVDIQQE_SLOW      4.236
#define NEO_DVDIQQE_TICK      0.01

/* `is_finite_fast` in ema.rs / cci_cycle.rs: tests the EXPONENT FIELD, so it
 * rejects infinities as well as NaN. A bare `!isnan` would not. */
__device__ __forceinline__ bool neo_dvdi_finite_fast(double x) {
    const long long EXP_MASK = (long long)0x7ff0000000000000ULL;
    return (__double_as_longlong(x) & EXP_MASK) != EXP_MASK;
}

/* One EMA stage, stepped one bar at a time. Mirrors `ema_scalar_into`
 * (ema.rs:461) exactly: seed at the first non-NaN, incremental mean for
 * `period` bars, then the fused step. `out` is NaN before the seed. */
struct NeoDvdiEma {
    double mean;
    double prev;
    int    first;        /* index of the seed bar, -1 until seeded          */
    long long warmup_end;/* first + period without signed-int overflow       */
    int    valid_count;
    double alpha, beta;
    int    period;
};

__device__ __forceinline__ void neo_dvdi_ema_init(NeoDvdiEma* e, int period) {
    e->mean = 0.0;
    e->prev = 0.0;
    e->first = -1;
    e->warmup_end = 0;
    e->valid_count = 0;
    e->alpha = 2.0 / ((double)period + 1.0);
    e->beta = 1.0 - e->alpha;
    e->period = period;
}

__device__ __forceinline__ double neo_dvdi_ema_step(NeoDvdiEma* e, int i, double x, int len) {
    if (e->first < 0) {
        if (isnan(x)) return NEO_F64_NAN;      /* ema_prepare's `!is_nan` scan */
        e->first = i;
        e->warmup_end = (long long)i + (long long)e->period;
        if (e->warmup_end > len) e->warmup_end = len;
        e->mean = x;
        e->valid_count = 1;
        e->prev = e->mean;
        return e->mean;
    }
    if ((long long)i < e->warmup_end) {
        if (neo_dvdi_finite_fast(x)) {
            e->valid_count += 1;
            const double vc = (double)e->valid_count;
            e->mean = ((vc - 1.0) * e->mean + x) / vc;
        }
        e->prev = e->mean;
        return e->mean;
    }
    if (neo_dvdi_finite_fast(x)) {
        e->prev = fma(e->beta, e->prev, e->alpha * x);
    }
    return e->prev;
}

/* One exact arithmetic authority for the canonical four-output route and the
 * preserved dvdi-primary ABI. Null destinations suppress materialization only;
 * all scalar state still advances in the canonical order. */
__device__ __forceinline__
static void dvdiqqe_row_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    int period,
    int smoothing_period,
    double fast_multiplier,
    double slow_multiplier,
    bool use_tick_only,
    bool dynamic_center,
    double tick_size,
    int first_valid,
    double* __restrict__ out_dvdi,
    double* __restrict__ out_fast_tl,
    double* __restrict__ out_slow_tl,
    double* __restrict__ out_center_line)
{
    for (int i = 0; i < len; ++i) {
        if (out_dvdi != nullptr) out_dvdi[i] = NEO_F64_NAN;
        if (out_fast_tl != nullptr) out_fast_tl[i] = NEO_F64_NAN;
        if (out_slow_tl != nullptr) out_slow_tl[i] = NEO_F64_NAN;
        if (out_center_line != nullptr) out_center_line[i] = NEO_F64_NAN;
    }

    /* dvdiqqe_prepare:388-425 */
    if (len <= 0 || period <= 0 || period > len ||
        smoothing_period <= 0 || smoothing_period > len ||
        !isfinite(fast_multiplier) || fast_multiplier <= 0.0 ||
        !isfinite(slow_multiplier) || slow_multiplier <= 0.0 ||
        !isfinite(tick_size) || tick_size <= 0.0 ||
        first_valid < 0 || first_valid >= len ||
        (len - first_valid) < period) {
        return;
    }

    const long long wper_wide = ((long long)period * 2LL) - 1LL;
    if (wper_wide <= 0LL || wper_wide > 0x7fffffffLL) return;
    const int wper = (int)wper_wide;
    const long long warmup = (long long)first_valid + wper_wide;

    double pvi_prev = 0.0, nvi_prev = 0.0;
    double prev_vol = 0.0, prev_close = 0.0;
    double tickrng_prev = tick_size;
    double prev_dvdi = NEO_F64_NAN;
    double fast_tl = NEO_F64_NAN;
    double slow_tl = NEO_F64_NAN;
    double center_sum = 0.0;
    double center_count = 0.0;

    NeoDvdiEma ema_p, ema_n, ema_pd, ema_nd, ema_range, ema_smooth_range;
    neo_dvdi_ema_init(&ema_p,  period);
    neo_dvdi_ema_init(&ema_n,  period);
    neo_dvdi_ema_init(&ema_pd, smoothing_period);
    neo_dvdi_ema_init(&ema_nd, smoothing_period);
    neo_dvdi_ema_init(&ema_range, wper);
    neo_dvdi_ema_init(&ema_smooth_range, wper);

    for (int i = 0; i < len; ++i) {
        const double oi = open[i];
        const double ci = close[i];

        /* :492-507 */
        const double rng = ci - oi;
        const double tickrng = (fabs(rng) < tick_size) ? tickrng_prev : rng;
        const double tick_vol = fmax(fabs(tickrng) / tick_size, 0.0);

        const double vv = volume[i];
        const double sel_vol = use_tick_only ? tick_vol : (isfinite(vv) ? vv : tick_vol);

        const double d_close = ci - prev_close;
        if (sel_vol > prev_vol) pvi_prev += d_close;
        if (sel_vol < prev_vol) nvi_prev -= d_close;

        const double pvi_i = pvi_prev;
        const double nvi_i = nvi_prev;

        prev_close = ci;
        prev_vol = sel_vol;
        tickrng_prev = tickrng;

        /* :524-542 — EMA(period) of pvi / nvi, then the difference. */
        const double ep = neo_dvdi_ema_step(&ema_p, i, pvi_i, len);
        const double en = neo_dvdi_ema_step(&ema_n, i, nvi_i, len);
        const double dp = pvi_i - ep;
        const double dn = nvi_i - en;

        /* :544-557 — EMA(smoothing_period) of each difference. */
        const double pd = neo_dvdi_ema_step(&ema_pd, i, dp, len);
        const double nd = neo_dvdi_ema_step(&ema_nd, i, dn, len);

        /* :560-576 — raw DVDI, absolute change, then the two range EMAs. */
        const double dvdi = pd - nd;
        const double range = (i == 0) ? NEO_F64_NAN : fabs(dvdi - prev_dvdi);
        const double avg_range = neo_dvdi_ema_step(&ema_range, i, range, len);
        const double smooth_range =
            neo_dvdi_ema_step(&ema_smooth_range, i, avg_range, len);
        prev_dvdi = dvdi;

        if ((long long)i < warmup) continue;

        if ((long long)i == warmup) {
            fast_tl = dvdi;
            slow_tl = dvdi;
        } else {
            const double fast_range = smooth_range * fast_multiplier;
            const double slow_range = smooth_range * slow_multiplier;
            if (dvdi > fast_tl) {
                const double next = dvdi - fast_range;
                fast_tl = (next < fast_tl) ? fast_tl : next;
            } else {
                const double next = dvdi + fast_range;
                fast_tl = (next > fast_tl) ? fast_tl : next;
            }
            if (dvdi > slow_tl) {
                const double next = dvdi - slow_range;
                slow_tl = (next < slow_tl) ? slow_tl : next;
            } else {
                const double next = dvdi + slow_range;
                slow_tl = (next > slow_tl) ? slow_tl : next;
            }
        }

        double center_line;
        if (dynamic_center) {
            if (isfinite(dvdi)) {
                center_sum += dvdi;
                center_count += 1.0;
            }
            center_line = (center_count > 0.0) ? (center_sum / center_count) : NEO_F64_NAN;
        } else {
            center_line = 0.0;
        }

        if (out_dvdi != nullptr) out_dvdi[i] = dvdi;
        if (out_fast_tl != nullptr) out_fast_tl[i] = fast_tl;
        if (out_slow_tl != nullptr) out_slow_tl[i] = slow_tl;
        if (out_center_line != nullptr) out_center_line[i] = center_line;
    }
}

extern "C" __global__
void dvdiqqe_all_outputs_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int series_len,
    const int* __restrict__ periods,
    const int* __restrict__ smoothing_periods,
    const double* __restrict__ fast_multipliers,
    const double* __restrict__ slow_multipliers,
    const int* __restrict__ use_tick_only_flags,
    const int* __restrict__ dynamic_center_flags,
    const double* __restrict__ tick_sizes,
    int n_combos,
    int first_valid,
    double* __restrict__ out_dvdi,
    double* __restrict__ out_fast_tl,
    double* __restrict__ out_slow_tl,
    double* __restrict__ out_center_line)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || series_len <= 0) return;
    const size_t base = (size_t)combo * (size_t)series_len;
    dvdiqqe_row_f64(
        open,
        close,
        volume,
        series_len,
        periods[combo],
        smoothing_periods[combo],
        fast_multipliers[combo],
        slow_multipliers[combo],
        use_tick_only_flags[combo] != 0,
        dynamic_center_flags[combo] != 0,
        tick_sizes[combo],
        first_valid,
        out_dvdi + base,
        out_fast_tl + base,
        out_slow_tl + base,
        out_center_line + base);
}

extern "C" __global__
void dvdiqqe_neo_batch_f64(const double* __restrict__ open,
                           const double* __restrict__ close,
                           const double* __restrict__ volume,
                           int series_len,
                           const int* __restrict__ periods,
                           int n_combos,
                           int first_valid,
                           double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || series_len <= 0) return;
    dvdiqqe_row_f64(
        open,
        close,
        volume,
        series_len,
        periods[combo],
        NEO_DVDIQQE_SMOOTHING,
        NEO_DVDIQQE_FAST,
        NEO_DVDIQQE_SLOW,
        false,
        true,
        NEO_DVDIQQE_TICK,
        first_valid,
        out + (size_t)combo * (size_t)series_len,
        nullptr,
        nullptr,
        nullptr);
}
