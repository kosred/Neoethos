#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


__device__ __forceinline__ float qnan_f32() {
    return __int_as_float(0x7fc00000);
}


__device__ __forceinline__ int warp_min_int(int v, unsigned mask) {
    for (int ofs = WARP_SIZE / 2; ofs > 0; ofs >>= 1) {
        int o = __shfl_down_sync(mask, v, ofs);
        v = (o < v) ? o : v;
    }
    return v;
}


extern "C" __global__ void supertrend_build_hl2_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    float* __restrict__ out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    const float h = high[idx];
    const float l = low[idx];
    out[idx] = (isnan(h) || isnan(l)) ? qnan_f32() : 0.5f * (h + l);
}


extern "C" __global__ void supertrend_batch_f32(
    const float* __restrict__ hl2,
    const float* __restrict__ close,
    const float* __restrict__ atr_rows,
    const int*   __restrict__ row_period_idx,
    const float* __restrict__ row_factors,
    const int*   __restrict__ row_warms,
    int len,
    int rows,
    float* __restrict__ out_trend,
    float* __restrict__ out_changed
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;


    const int   pidx   = row_period_idx[r];
    const int   warm   = row_warms[r];
    const float factor = row_factors[r];

    const int base_p = pidx * len;
    const int base_r = r    * len;

    const float* __restrict__ atr_row = atr_rows + base_p;
    float* __restrict__ out_tr = out_trend   + base_r;
    float* __restrict__ out_ch = out_changed + base_r;


    const unsigned mask = __activemask();
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int src  = __ffs(mask) - 1;


    const int warp_min_warm = warp_min_int(warm, mask);
    for (int t = 0; t < warp_min_warm; ++t) {
        out_tr[t] = qnan_f32();
        out_ch[t] = qnan_f32();
    }


    const int warp_p0 = __shfl_sync(mask, pidx, src);
    const int same_p  = __all_sync(mask, pidx == warp_p0);
    const int base_p0 = warp_p0 * len;


    int   upper_state = 0;
    float prev_upper  = 0.0f;
    float prev_lower  = 0.0f;
    float last_close  = 0.0f;
    bool  active      = false;


    for (int t = warp_min_warm; t < len; ++t) {

        float hl_b = 0.0f, c_b = 0.0f, a_b = 0.0f;
        if (lane == src) {
            hl_b = hl2[t];
            c_b  = close[t];
            if (same_p) a_b = atr_rows[base_p0 + t];
        }
        const float hl = __shfl_sync(mask, hl_b, src);
        const float c  = __shfl_sync(mask, c_b,  src);
        const float a  = same_p ? __shfl_sync(mask, a_b, src) : atr_row[t];


        if (t < warm) {
            out_tr[t] = qnan_f32();
            out_ch[t] = qnan_f32();
            continue;
        }

        if (!active) {

            prev_upper  = fmaf(factor,  a, hl);
            prev_lower  = fmaf(-factor, a, hl);
            last_close  = c;
            upper_state = (last_close <= prev_upper);
            out_tr[t]   = upper_state ? prev_upper : prev_lower;
            out_ch[t]   = 0.0f;
            active      = true;
            continue;
        }


        const float upper_basic = fmaf(factor,  a, hl);
        const float lower_basic = fmaf(-factor, a, hl);

        const float curr_upper = (last_close <= prev_upper) ? fminf(upper_basic, prev_upper) : upper_basic;
        const float curr_lower = (last_close >= prev_lower) ? fmaxf(lower_basic, prev_lower) : lower_basic;

        float outv, changed = 0.0f;
        if (upper_state) {
            if (c <= curr_upper) { outv = curr_upper; }
            else { outv = curr_lower; changed = 1.0f; upper_state = 0; }
        } else {
            if (c >= curr_lower) { outv = curr_lower; }
            else { outv = curr_upper; changed = 1.0f; upper_state = 1; }
        }

        out_tr[t] = outv;
        out_ch[t] = changed;

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}


extern "C" __global__ void supertrend_many_series_one_param_f32(
    const float* __restrict__ hl2_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ atr_tm,
    const int*   __restrict__ first_valids,
    int period,
    int cols,
    int rows,
    float factor,
    float* __restrict__ out_trend_tm,
    float* __restrict__ out_changed_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv   = first_valids[s];
    const int warm = fv + period - 1;


    const int stride = cols;
    const float* __restrict__ p_hl    = hl2_tm   + s;
    const float* __restrict__ p_close = close_tm + s;
    const float* __restrict__ p_atr   = atr_tm   + s;
    float* __restrict__ p_out_tr = out_trend_tm   + s;
    float* __restrict__ p_out_ch = out_changed_tm + s;


    int t = 0;
    for (; t < rows && t < warm; ++t) {
        p_out_tr[ t*stride ] = qnan_f32();
        p_out_ch[ t*stride ] = qnan_f32();
    }
    if (t >= rows) return;


    const float hl_w    = p_hl   [ t*stride ];
    const float atr_w   = p_atr  [ t*stride ];
    const float close_w = p_close[ t*stride ];

    float prev_upper = fmaf(factor,  atr_w, hl_w);
    float prev_lower = fmaf(-factor, atr_w, hl_w);
    float last_close = close_w;
    int   upper_state = (last_close <= prev_upper);

    p_out_tr[ t*stride ] = upper_state ? prev_upper : prev_lower;
    p_out_ch[ t*stride ] = 0.0f;


    for (++t; t < rows; ++t) {
        const float hl = p_hl   [ t*stride ];
        const float a  = p_atr  [ t*stride ];
        const float c  = p_close[ t*stride ];

        const float upper_basic = fmaf(factor,  a, hl);
        const float lower_basic = fmaf(-factor, a, hl);

        const float curr_upper = (last_close <= prev_upper) ? fminf(upper_basic, prev_upper) : upper_basic;
        const float curr_lower = (last_close >= prev_lower) ? fmaxf(lower_basic, prev_lower) : lower_basic;

        float outv, changed = 0.0f;
        if (upper_state) {
            if (c <= curr_upper) { outv = curr_upper; }
            else { outv = curr_lower; changed = 1.0f; upper_state = 0; }
        } else {
            if (c >= curr_lower) { outv = curr_lower; }
            else { outv = curr_upper; changed = 1.0f; upper_state = 1; }
        }

        p_out_tr[ t*stride ] = outv;
        p_out_ch[ t*stride ] = changed;

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

__device__ __forceinline__ double qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}
extern "C" __global__ void supertrend_build_hl2_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    double* __restrict__ out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    const double h = high[idx];
    const double l = low[idx];
    out[idx] = (isnan(h) || isnan(l)) ? qnan_f64() : 0.5 * (h + l);
}
extern "C" __global__ void supertrend_batch_f64(
    const double* __restrict__ hl2,
    const double* __restrict__ close,
    const double* __restrict__ atr_rows,
    const int*   __restrict__ row_period_idx,
    const double* __restrict__ row_factors,
    const int*   __restrict__ row_warms,
    int len,
    int rows,
    double* __restrict__ out_trend,
    double* __restrict__ out_changed
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;


    const int   pidx   = row_period_idx[r];
    const int   warm   = row_warms[r];
    const double factor = row_factors[r];

    const int base_p = pidx * len;
    const int base_r = r    * len;

    const double* __restrict__ atr_row = atr_rows + base_p;
    double* __restrict__ out_tr = out_trend   + base_r;
    double* __restrict__ out_ch = out_changed + base_r;


    const unsigned mask = __activemask();
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int src  = __ffs(mask) - 1;


    const int warp_min_warm = warp_min_int(warm, mask);
    for (int t = 0; t < warp_min_warm; ++t) {
        out_tr[t] = qnan_f64();
        out_ch[t] = qnan_f64();
    }


    const int warp_p0 = __shfl_sync(mask, pidx, src);
    const int same_p  = __all_sync(mask, pidx == warp_p0);
    const int base_p0 = warp_p0 * len;


    int   upper_state = 0;
    double prev_upper  = 0.0;
    double prev_lower  = 0.0;
    double last_close  = 0.0;
    bool  active      = false;


    for (int t = warp_min_warm; t < len; ++t) {

        double hl_b = 0.0, c_b = 0.0, a_b = 0.0;
        if (lane == src) {
            hl_b = hl2[t];
            c_b  = close[t];
            if (same_p) a_b = atr_rows[base_p0 + t];
        }
        const double hl = __shfl_sync(mask, hl_b, src);
        const double c  = __shfl_sync(mask, c_b,  src);
        const double a  = same_p ? __shfl_sync(mask, a_b, src) : atr_row[t];


        if (t < warm) {
            out_tr[t] = qnan_f64();
            out_ch[t] = qnan_f64();
            continue;
        }

        if (!active) {

            prev_upper  = fma(factor,  a, hl);
            prev_lower  = fma(-factor, a, hl);
            last_close  = c;
            upper_state = (last_close <= prev_upper);
            out_tr[t]   = upper_state ? prev_upper : prev_lower;
            out_ch[t]   = 0.0;
            active      = true;
            continue;
        }


        const double upper_basic = fma(factor,  a, hl);
        const double lower_basic = fma(-factor, a, hl);

        const double curr_upper = (last_close <= prev_upper) ? fmin(upper_basic, prev_upper) : upper_basic;
        const double curr_lower = (last_close >= prev_lower) ? fmax(lower_basic, prev_lower) : lower_basic;

        double outv, changed = 0.0;
        if (upper_state) {
            if (c <= curr_upper) { outv = curr_upper; }
            else { outv = curr_lower; changed = 1.0; upper_state = 0; }
        } else {
            if (c >= curr_lower) { outv = curr_lower; }
            else { outv = curr_upper; changed = 1.0; upper_state = 1; }
        }

        out_tr[t] = outv;
        out_ch[t] = changed;

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}
extern "C" __global__ void supertrend_many_series_one_param_f64(
    const double* __restrict__ hl2_tm,
    const double* __restrict__ close_tm,
    const double* __restrict__ atr_tm,
    const int*   __restrict__ first_valids,
    int period,
    int cols,
    int rows,
    double factor,
    double* __restrict__ out_trend_tm,
    double* __restrict__ out_changed_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv   = first_valids[s];
    const int warm = fv + period - 1;


    const int stride = cols;
    const double* __restrict__ p_hl    = hl2_tm   + s;
    const double* __restrict__ p_close = close_tm + s;
    const double* __restrict__ p_atr   = atr_tm   + s;
    double* __restrict__ p_out_tr = out_trend_tm   + s;
    double* __restrict__ p_out_ch = out_changed_tm + s;


    int t = 0;
    for (; t < rows && t < warm; ++t) {
        p_out_tr[ t*stride ] = qnan_f64();
        p_out_ch[ t*stride ] = qnan_f64();
    }
    if (t >= rows) return;


    const double hl_w    = p_hl   [ t*stride ];
    const double atr_w   = p_atr  [ t*stride ];
    const double close_w = p_close[ t*stride ];

    double prev_upper = fma(factor,  atr_w, hl_w);
    double prev_lower = fma(-factor, atr_w, hl_w);
    double last_close = close_w;
    int   upper_state = (last_close <= prev_upper);

    p_out_tr[ t*stride ] = upper_state ? prev_upper : prev_lower;
    p_out_ch[ t*stride ] = 0.0;


    for (++t; t < rows; ++t) {
        const double hl = p_hl   [ t*stride ];
        const double a  = p_atr  [ t*stride ];
        const double c  = p_close[ t*stride ];

        const double upper_basic = fma(factor,  a, hl);
        const double lower_basic = fma(-factor, a, hl);

        const double curr_upper = (last_close <= prev_upper) ? fmin(upper_basic, prev_upper) : upper_basic;
        const double curr_lower = (last_close >= prev_lower) ? fmax(lower_basic, prev_lower) : lower_basic;

        double outv, changed = 0.0;
        if (upper_state) {
            if (c <= curr_upper) { outv = curr_upper; }
            else { outv = curr_lower; changed = 1.0; upper_state = 0; }
        } else {
            if (c >= curr_lower) { outv = curr_lower; }
            else { outv = curr_upper; changed = 1.0; upper_state = 1; }
        }

        p_out_tr[ t*stride ] = outv;
        p_out_ch[ t*stride ] = changed;

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — supertrend
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/supertrend.rs:454 supertrend_scalar_fused, which
 *   is what supertrend_compute_direct_into (:312) selects for Kernel::Scalar
 *   and for every non-AVX build.
 *
 * Column: output_id "value" resolves to out.trend — cpu_batch.rs:6366 accepts
 *   "trend"/"value". The `changed` column is a separate output id; its flag is
 *   a by-product of the same state machine and is not written here.
 *
 * PERIOD-SWEPT, unlike most of this shard: compute_supertrend_batch reads a
 *   parameter literally named `period` (default 10, cpu_batch.rs:6348), so
 *   each row of the sweep is a DIFFERENT column and the kernel reads
 *   periods[combo]. `factor` (3.0) is the only pinned default.
 *
 * first_valid: F64FirstValidRule::AllInputsNonNan. supertrend_prepare
 *   (:239-247) scans for the first index at which high, low AND close are all
 *   NOT NaN simultaneously — the common rule, not the max-of-independent-scans
 *   rule adx and natr use. The NaN prefix runs to first_valid + period - 1
 *   exclusive (:383 warmup_end), and the first written bar is that index.
 *
 * Input: high / low / close — F64InputKind::Hlc.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. The ATR is a Wilder
 *   recurrence, and both bands plus the up/down state carry across bars: the
 *   current band clamps against the PREVIOUS one, so bar i cannot be computed
 *   without bar i-1.
 *
 * ARITHMETIC taken verbatim:
 *   * the seed ATR is a plain SUM of true ranges over [first_valid, warmup]
 *     divided ONCE by period (:479-501) — the first term is high-low with no
 *     previous close, and each later term takes the max of three by
 *     COMPARISON (`>`), not by fmax, so the tie behaviour matches.
 *   * the ATR step is (-alpha).mul_add(atr, atr) + alpha * true_range (:535)
 *     — ONE fma plus one separate product and add. It is NOT
 *     (tr - atr).mul_add(alpha, atr), which is one rounding fewer and drifts.
 *   * the bands are factor.mul_add(atr, hl2) and (-factor).mul_add(atr, hl2)
 *     (:538-539) — ONE fma each, with the negated factor precomputed exactly
 *     as the CPU precomputes it.
 *   * the clamps are min / max on f64 (:543, :547), so fmin/fmax are used:
 *     f64::min returns the non-NaN operand and an if-chain would not.
 *   * there is no epsilon anywhere in this indicator; every guard is an exact
 *     comparison and is reproduced as written.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Default from cpu_batch.rs:6349. `period` is SWEPT and is read per combo. */
#define NEO_SUPERTREND_FACTOR 3.0

extern "C" __global__
void supertrend_neo_batch_f64(const double* __restrict__ high,
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

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = periods[combo];
    if (period <= 0 || period > n) return;          /* InvalidPeriod (:213) */
    if (first_valid < 0 || first_valid >= n) return;
    if (n - first_valid < period) return;           /* NotEnoughValidData (:257) */

    const int start = first_valid + period;
    if (start > n) return;
    const int warmup = start - 1;

    const double factor     = NEO_SUPERTREND_FACTOR;
    const double neg_factor = -factor;
    const double alpha      = 1.0 / (double)period;

    /* Seed: sum of true ranges over [first_valid, warmup]. */
    double sum_tr = high[first_valid] - low[first_valid];
    if (warmup > first_valid) {
        double prev_c = close[first_valid];
        for (int i = first_valid + 1; i <= warmup; ++i) {
            const double hi = high[i];
            const double lo = low[i];
            double true_range = hi - lo;
            const double high_close = fabs(hi - prev_c);
            if (high_close > true_range) true_range = high_close;
            const double low_close = fabs(lo - prev_c);
            if (low_close > true_range) true_range = low_close;
            sum_tr += true_range;
            prev_c = close[i];
        }
    }

    double atr = sum_tr / (double)period;

    const double hw    = high[warmup];
    const double lw    = low[warmup];
    const double hl2_w = (hw + lw) * 0.5;
    double prev_upper_band = hl2_w + factor * atr;
    double prev_lower_band = hl2_w - factor * atr;

    double last_close = close[warmup];
    bool upper_state;
    if (last_close <= prev_upper_band) { o[warmup] = prev_upper_band; upper_state = true; }
    else                               { o[warmup] = prev_lower_band; upper_state = false; }

    for (int i = warmup + 1; i < n; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double prev_close = last_close;

        double true_range = hi - lo;
        const double high_close = fabs(hi - prev_close);
        if (high_close > true_range) true_range = high_close;
        const double low_close = fabs(lo - prev_close);
        if (low_close > true_range) true_range = low_close;

        atr = fma(-alpha, atr, atr) + alpha * true_range;

        const double hl2         = (hi + lo) * 0.5;
        const double upper_basic = fma(factor, atr, hl2);
        const double lower_basic = fma(neg_factor, atr, hl2);

        double curr_upper_band = upper_basic;
        if (prev_close <= prev_upper_band) curr_upper_band = fmin(curr_upper_band, prev_upper_band);
        double curr_lower_band = lower_basic;
        if (prev_close >= prev_lower_band) curr_lower_band = fmax(curr_lower_band, prev_lower_band);

        const double curr_close = close[i];
        if (upper_state) {
            if (curr_close <= curr_upper_band) {
                o[i] = curr_upper_band;
            } else {
                o[i] = curr_lower_band;
                upper_state = false;
            }
        } else {
            if (curr_close >= curr_lower_band) {
                o[i] = curr_lower_band;
            } else {
                o[i] = curr_upper_band;
                upper_state = true;
            }
        }

        prev_upper_band = curr_upper_band;
        prev_lower_band = curr_lower_band;
        last_close = curr_close;
    }
}
