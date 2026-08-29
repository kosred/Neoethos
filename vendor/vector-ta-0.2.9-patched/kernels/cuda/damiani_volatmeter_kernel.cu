#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LDG
#  if __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif

__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }
__device__ __forceinline__ bool finite_f32(float x){ return isfinite(x); }


__device__ __forceinline__ void kahan_add(float &sum, float &comp, float x){
    float y = x - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}


__device__ __forceinline__ float2 ff_two_sum(float a, float b){
    float s  = a + b;
    float bb = s - a;
    float e  = (a - (s - bb)) + (b - bb);
    return make_float2(s, e);
}

__device__ __forceinline__ float2 ff_add(float2 x, float2 y){
    float2 t = ff_two_sum(x.x, y.x);
    float e  = t.y + x.y + y.y;
    return ff_two_sum(t.x, e);
}

__device__ __forceinline__ float2 ff_neg(float2 x){ return make_float2(-x.x, -x.y); }
__device__ __forceinline__ float2 ff_sub(float2 x, float2 y){ return ff_add(x, ff_neg(y)); }

__device__ __forceinline__ float2 ff_two_prod(float a, float b){
    float p = a * b;
    float e = fmaf(a, b, -p);
    return make_float2(p, e);
}

__device__ __forceinline__ float2 ff_scale(float2 x, float s){

    return ff_two_sum(x.x * s, x.y * s);
}

__device__ __forceinline__ float2 ff_mul(float2 x, float2 y){

    float2 p  = ff_two_prod(x.x, y.x);
    float cross = x.x * y.y + x.y * y.x;
    float2 s  = ff_two_sum(p.x, cross);
    float e   = p.y + s.y + x.y * y.y;
    return ff_two_sum(s.x, e);
}

__device__ __forceinline__ float ff_to_f32(float2 x){ return x.x + x.y; }


__device__ __forceinline__ float safe_pos_den(float x){
    const float EPS = 1.1920929e-7f;
    return (finite_f32(x) && x > 0.0f) ? x : EPS;
}


__device__ __forceinline__ float std_from_ff_prefix(const float2 s_t, const float2 s_prev,
                                                    const float2 ss_t, const float2 ss_prev,
                                                    int win)
{
    const float inv_n = 1.0f / (float)win;
    const float2 sum   = ff_sub(s_t,  s_prev);
    const float2 sumsq = ff_sub(ss_t, ss_prev);
    const float2 mean   = ff_scale(sum,   inv_n);
    const float2 mean2  = ff_mul(mean, mean);
    const float2 ex2    = ff_scale(sumsq, inv_n);
    const float2 var_ff = ff_sub(ex2, mean2);
    const float var = fmaxf(ff_to_f32(var_ff), 0.0f);
    return sqrtf(var);
}

extern "C" __global__
void damiani_build_close_workspace_f32(const float* __restrict__ prices,
                                       int series_len,
                                       int first_valid,
                                       float2* __restrict__ s_prefix,
                                       float2* __restrict__ ss_prefix,
                                       float* __restrict__ tr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len) return;

    float2 acc_s = make_float2(0.f, 0.f);
    float2 acc_ss = make_float2(0.f, 0.f);
    float prev_close = nan_f32();
    bool have_prev = false;

    for (int i = 0; i < first_valid; ++i) {
        s_prefix[i] = make_float2(0.f, 0.f);
        ss_prefix[i] = make_float2(0.f, 0.f);
        tr[i] = 0.f;
    }

    for (int i = first_valid; i < series_len; ++i) {
        const float c = LDG(&prices[i]);
        const float v = finite_f32(c) ? c : 0.0f;
        acc_s = ff_add(acc_s, make_float2(v, 0.f));
        acc_ss = ff_add(acc_ss, make_float2(v * v, 0.f));
        s_prefix[i] = acc_s;
        ss_prefix[i] = acc_ss;
        tr[i] = (have_prev && finite_f32(c)) ? fabsf(c - prev_close) : 0.0f;
        if (finite_f32(c)) {
            prev_close = c;
            have_prev = true;
        }
    }
}

extern "C" __global__
void damiani_select_output_rows_f32(const float* __restrict__ packed,
                                    int series_len,
                                    int combo_count,
                                    int output_index,
                                    float* __restrict__ out)
{
    const int row = blockIdx.y;
    const int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= combo_count || t >= series_len) return;

    const int src_row = row * 2 + output_index;
    out[row * series_len + t] = packed[src_row * series_len + t];
}


extern "C" __global__
void damiani_volatmeter_batch_f32(const float* __restrict__ prices,
                                  int series_len,
                                  int first_valid,
                                  const int* __restrict__ vis_atr,
                                  const int* __restrict__ vis_std,
                                  const int* __restrict__ sed_atr,
                                  const int* __restrict__ sed_std,
                                  const float* __restrict__ threshold,
                                  int n_combos,
                                  const float2* __restrict__ s_prefix,
                                  const float2* __restrict__ ss_prefix,
                                  const float* __restrict__ tr,
                                  float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int total_threads = blockDim.x * gridDim.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    for (; row < n_combos; row += total_threads) {
        const int p_vis_atr = vis_atr[row];
        const int p_vis_std = vis_std[row];
        const int p_sed_atr = sed_atr[row];
        const int p_sed_std = sed_std[row];
        const float th = threshold[row];

        const int needed = max(max(max(p_vis_atr, p_vis_std), max(p_sed_atr, p_sed_std)), 3);

        const size_t base_vol  = ((size_t)(row * 2 + 0)) * (size_t)series_len;
        const size_t base_anti = ((size_t)(row * 2 + 1)) * (size_t)series_len;

        const int warm_end = min(series_len, first_valid + needed - 1);


        float atr_vis = NAN, atr_sed = NAN;
        float sum_vis = 0.0f, c_vis = 0.0f;
        float sum_sed = 0.0f, c_sed = 0.0f;


        float vh1 = 0.0f, vh2 = 0.0f, vh3 = 0.0f;
        const float lag_s = 0.5f;

        for (int t = first_valid; t < series_len; ++t) {
            const float tr_t = LDG(&tr[t]);
            const int k = t - first_valid;


            if (k < p_vis_atr) {
                kahan_add(sum_vis, c_vis, tr_t);
                if (k == p_vis_atr - 1) atr_vis = sum_vis / (float)p_vis_atr;
            } else if (finite_f32(atr_vis)) {
                const float alpha = 1.0f / (float)p_vis_atr;
                atr_vis = fmaf(atr_vis, (1.0f - alpha), tr_t * alpha);
            }


            if (k < p_sed_atr) {
                kahan_add(sum_sed, c_sed, tr_t);
                if (k == p_sed_atr - 1) atr_sed = sum_sed / (float)p_sed_atr;
            } else if (finite_f32(atr_sed)) {
                const float alpha = 1.0f / (float)p_sed_atr;
                atr_sed = fmaf(atr_sed, (1.0f - alpha), tr_t * alpha);
            }


            if (k >= needed) {
                const float inv_sed = 1.0f / safe_pos_den(atr_sed);
                const float base    = atr_vis * inv_sed;
                const float vol_t   = fmaf(lag_s, (vh1 - vh3), base);
                out[base_vol + (size_t)t] = vol_t;

                vh3 = vh2; vh2 = vh1; vh1 = vol_t;


                if (k >= max(p_vis_std, p_sed_std) - 1) {
                    const int prev_v = t - p_vis_std;
                    const int prev_s = t - p_sed_std;

                    const float2 S_t   = s_prefix[t];
                    const float2 SS_t  = ss_prefix[t];
                    const float2 S_pv  = (prev_v >= 0) ? s_prefix[prev_v]  : make_float2(0.f,0.f);
                    const float2 SS_pv = (prev_v >= 0) ? ss_prefix[prev_v] : make_float2(0.f,0.f);
                    const float2 S_ps  = (prev_s >= 0) ? s_prefix[prev_s]  : make_float2(0.f,0.f);
                    const float2 SS_ps = (prev_s >= 0) ? ss_prefix[prev_s] : make_float2(0.f,0.f);

                    const float std_v = std_from_ff_prefix(S_t, S_pv, SS_t, SS_pv, p_vis_std);
                    const float std_s = std_from_ff_prefix(S_t, S_ps, SS_t, SS_ps, p_sed_std);

                    const float anti_t = th - (std_v / safe_pos_den(std_s));
                    out[base_anti + (size_t)t] = anti_t;
                }
            }
        }

        for (int t = 0; t <= warm_end && t < series_len; ++t) {
            out[base_vol + (size_t)t] = nan_f32();
            out[base_anti + (size_t)t] = nan_f32();
        }
    }
}


extern "C" __global__
void damiani_volatmeter_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int num_series,
    int series_len,
    int vis_atr,
    int vis_std,
    int sed_atr,
    int sed_std,
    float threshold,
    const int* __restrict__ first_valids,
    const float2* __restrict__ s_tm,
    const float2* __restrict__ ss_tm,
    float* __restrict__ out_tm)
{
    if (num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int total_threads = blockDim.x * gridDim.x;
    int series = blockIdx.x * blockDim.x + threadIdx.x;

    for (; series < num_series; series += total_threads) {
        const int fv = max(0, first_valids[series]);
        if (fv >= series_len) continue;

        const int needed = max(max(max(vis_atr, vis_std), max(sed_atr, sed_std)), 3);
        const int warm_end = min(series_len, fv + needed - 1);

        float atr_vis = NAN, atr_sed = NAN;
        float sum_vis = 0.0f, c_vis = 0.0f;
        float sum_sed = 0.0f, c_sed = 0.0f;
        const float lag_s = 0.5f;
        float prev_close = NAN;
        bool have_prev = false;


        float vh1 = 0.0f, vh2 = 0.0f, vh3 = 0.0f;

        for (int t = fv; t < series_len; ++t) {
            const int idx = t * stride + series;
            const int k   = t - fv;
            const float h = LDG(&high_tm[idx]);
            const float l = LDG(&low_tm[idx]);
            const float c = LDG(&close_tm[idx]);

            float tr;
            if (have_prev && finite_f32(c)) {
                const float tr1 = h - l;
                const float tr2 = fabsf(h - prev_close);
                const float tr3 = fabsf(l - prev_close);
                tr = fmaxf(tr1, fmaxf(tr2, tr3));
            } else {
                tr = 0.0f;
            }
            if (finite_f32(c)) { prev_close = c; have_prev = true; }


            if (k < vis_atr) {
                kahan_add(sum_vis, c_vis, tr);
                if (k == vis_atr - 1) atr_vis = sum_vis / (float)vis_atr;
            } else if (finite_f32(atr_vis)) {
                const float alpha = 1.0f / (float)vis_atr;
                atr_vis = fmaf(atr_vis, (1.0f - alpha), tr * alpha);
            }


            if (k < sed_atr) {
                kahan_add(sum_sed, c_sed, tr);
                if (k == sed_atr - 1) atr_sed = sum_sed / (float)sed_atr;
            } else if (finite_f32(atr_sed)) {
                const float alpha = 1.0f / (float)sed_atr;
                atr_sed = fmaf(atr_sed, (1.0f - alpha), tr * alpha);
            }

            if (k >= needed - 1) {
                const size_t out_row = (size_t)t * (size_t)(2 * stride);
                const float inv_sed = 1.0f / safe_pos_den(atr_sed);
                const float base    = atr_vis * inv_sed;
                const float vol_t   = fmaf(lag_s, (vh1 - vh3), base);
                out_tm[out_row + (size_t)series] = vol_t;

                vh3 = vh2; vh2 = vh1; vh1 = vol_t;


                if (k >= max(vis_std, sed_std) - 1) {
                    const int prev_v = t - vis_std;
                    const int prev_s = t - sed_std;

                    const int pv_idx = (prev_v >= 0) ? (prev_v * stride + series) : -1;
                    const int ps_idx = (prev_s >= 0) ? (prev_s * stride + series) : -1;

                    const float2 S_t   = s_tm[idx];
                    const float2 SS_t  = ss_tm[idx];
                    const float2 S_pv  = (pv_idx >= 0) ? s_tm[pv_idx]  : make_float2(0.f,0.f);
                    const float2 SS_pv = (pv_idx >= 0) ? ss_tm[pv_idx] : make_float2(0.f,0.f);
                    const float2 S_ps  = (ps_idx >= 0) ? s_tm[ps_idx]  : make_float2(0.f,0.f);
                    const float2 SS_ps = (ps_idx >= 0) ? ss_tm[ps_idx] : make_float2(0.f,0.f);

                    const float std_v = std_from_ff_prefix(S_t, S_pv, SS_t, SS_pv, vis_std);
                    const float std_s = std_from_ff_prefix(S_t, S_ps, SS_t, SS_ps, sed_std);
                    out_tm[out_row + (size_t)(stride + series)] = threshold - (std_v / safe_pos_den(std_s));
                }
            }
        }

        for (int t = 0; t <= warm_end && t < series_len; ++t) {
            const size_t out_row = (size_t)t * (size_t)(2 * stride);
            out_tm[out_row + (size_t)series] = nan_f32();
            out_tm[out_row + (size_t)(stride + series)] = nan_f32();
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — damiani_volatmeter
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/damiani_volatmeter.rs
 *   `damiani_volatmeter_prepare`   (:297) — first_valid, needed, Err branches,
 *                                           and `Slice(s) => (s, s, s)`
 *   `damiani_volatmeter_with_kernel`(:380)— the warm prefix AND the SECOND
 *                                           NaN pass at :412-418
 *   `damiani_volatmeter_scalar`    (:491) — the two Wilder ATRs, the two
 *                                           rolling variances, the lag term
 *
 * WHICH SERIES THIS EMITS. `compute_damiani_volatmeter_batch`
 * (cpu_batch.rs:14400) maps output_id "value" -> `out.vol`. One matrix, so
 * this is VOL. `anti` is a separate output.
 *
 * INPUT IS ONE SERIES, NOT THREE. `compute_damiani_volatmeter_batch` calls
 * `from_slice(close)`, and `damiani_volatmeter_prepare:323` expands a Slice to
 * `(slice, slice, slice)`. So high == low == close, `tr1 = high - low` is
 * IDENTICALLY ZERO and the true range collapses to `|c - prev_c|`. It is
 * written out in full below anyway, because the collapse is a property of how
 * the batch happens to call it and not of the indicator.
 *
 * PERIOD-INVARIANT, AND THAT IS FAITHFUL. The batch reads `vis_atr` (13),
 * `vis_std` (20), `sed_atr` (40), `sed_std` (100) and `threshold` (1.4) —
 * cpu_batch.rs:14380-14384 — never `period`. Identical CPU columns, identical
 * rows here. The two std rings are therefore compile-time 20 and 100 slots and
 * this kernel needs no `max_period`.
 *
 * THE ATR WARM-UP TESTS THE ABSOLUTE INDEX, NOT THE OFFSET FROM first_valid.
 * damiani_volatmeter.rs:548 is `if i < vis_atr`, and `i` is an index into the
 * whole series while the loop starts at `first`. On a series whose first bar
 * is valid the two coincide; on a gapped one they do not, and the ATR seeds
 * over fewer bars than the period. That is what the reference does and it is
 * reproduced literally — noted here so it reads as a copied quirk rather than
 * a transcription slip.
 *
 * A DEFECT IN THE REFERENCE, AND WHAT THIS KERNEL DOES ABOUT IT.
 * `vol[i]` reads `vol[i-1]` and `vol[i-3]` and tests them with `is_nan()`.
 * `alloc_with_nan_prefix` (helpers.rs:103) NaN-fills only `[0, warm)` and
 * leaves the tail UNINITIALISED in release builds. When `first_valid == 0`,
 * `warm == needed - 1`, so the very first read at `i == needed` touches index
 * `needed - 1`, which is past the NaN prefix and has not been written yet:
 * the reference reads uninitialised memory. When `first_valid > 0` the same
 * index IS inside the prefix and the read is a well-defined NaN -> 0.0.
 * This kernel NaN-fills the WHOLE row before the loop, which makes every such
 * read the well-defined branch. That is the only self-consistent reading of
 * the reference, it is what the reference does for every gapped series, and
 * it is stated here rather than left to be discovered as a parity mismatch.
 *
 * THE SECOND NaN PASS IS NOT REDUNDANT. :412-418 blanks `[0, warm_end + 1)`
 * AFTER the loop has run. Bars in `[needed, warm_end]` are therefore COMPUTED,
 * READ by later bars through the `p1`/`p3` lag, and only then erased. Blanking
 * them up front would change every subsequent value.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. `const float EPS = 1.1920929e-7f` at damiani_volatmeter_kernel.cu:68 IS
 *     f32 MACHINE EPSILON. The reference uses `f64::EPSILON`, which is
 *     2.220446049250313e-16 — nine orders of magnitude smaller. Copying the
 *     f32 constant into an f64 kernel would make the zero-guard fire on
 *     denominators that are perfectly usable and would shift `vol` wherever
 *     the sedimentary ATR is small. This is exactly the hazard the brief names
 *     and it is re-derived here, not copied.
 *  2. `fabsf` x3, `fmaxf` x3, `sqrtf` x1 -> `fabs`, `fmax`, `sqrt`. The maxima
 *     ARE `f64::max` in the reference (:538, :615, :620), so `fmax` is
 *     correct here — unlike `cksp` and the `ttm_squeeze` warm-up, which use
 *     comparison chains and must not.
 *  3. `__int_as_float(0x7f...)` -> `__longlong_as_double(0x7ff8...)`.
 *  4. THE ATR STEP IS THREE ROUNDINGS, NOT ONE. :554 is
 *     `((p - 1)*atr + tr) / p` — multiply, add, divide. It is NOT the Wilder
 *     `fma` form used by `atr`/`natr` elsewhere in this crate. Writing
 *     `fma(inv_p, tr - atr, atr)` here would be one rounding where the
 *     reference has three, and would disagree from the first bar.
 *
 * ONE THREAD PER COLUMN. Carried: two ATRs, four rolling sums, two rings.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_DAMIANI_VIS_ATR   13
#define NEO_DAMIANI_VIS_STD   20
#define NEO_DAMIANI_SED_ATR   40
#define NEO_DAMIANI_SED_STD  100
#define NEO_DAMIANI_THRESHOLD 1.4
/* f64::EPSILON — NOT the f32 1.1920929e-7f the f32 kernel above uses. */
#define NEO_F64_EPSILON 2.2204460492503131e-16

__device__ __forceinline__
void damiani_volatmeter_row_f64(
    const double* __restrict__ prices,
    int len,
    int first_valid,
    int vis_atr,
    int vis_std,
    int sed_atr,
    int sed_std,
    double threshold,
    double* __restrict__ ring_vis,
    double* __restrict__ ring_sed,
    double* __restrict__ vol,
    double* __restrict__ anti)
{
    int needed = vis_atr;
    if (vis_std > needed) needed = vis_std;
    if (sed_atr > needed) needed = sed_atr;
    if (sed_std > needed) needed = sed_std;
    if (3 > needed) needed = 3;

    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        vis_atr <= 0 || vis_std <= 0 || sed_atr <= 0 || sed_std <= 0 ||
        vis_atr > len || vis_std > len || sed_atr > len || sed_std > len ||
        (len - first_valid) < needed) {
        for (int i = 0; i < len; ++i) {
            vol[i] = NEO_F64_NAN;
            if (anti) anti[i] = NEO_F64_NAN;
        }
        return;
    }

    for (int i = 0; i < len; ++i) {
        vol[i] = NEO_F64_NAN;
        if (anti) anti[i] = NEO_F64_NAN;
    }
    for (int i = 0; i < vis_std; ++i) ring_vis[i] = 0.0;
    for (int i = 0; i < sed_std; ++i) ring_sed[i] = 0.0;

    /* Slice input authority: high == low == close == prices. */
    const double* __restrict__ high = prices;
    const double* __restrict__ low = prices;
    const double* __restrict__ close = prices;
    double atr_vis_val = NEO_F64_NAN;
    double atr_sed_val = NEO_F64_NAN;
    double sum_vis = 0.0;
    double sum_sed = 0.0;
    const double vis_atr_f = (double)vis_atr;
    const double sed_atr_f = (double)sed_atr;
    double prev_close = NEO_F64_NAN;
    bool have_prev = false;
    double sum_vis_std = 0.0;
    double sum_sq_vis_std = 0.0;
    double sum_sed_std = 0.0;
    double sum_sq_sed_std = 0.0;
    int idx_vis = 0;
    int idx_sed = 0;
    int filled_vis = 0;
    int filled_sed = 0;
    const double lag_s = 0.5;

    for (int i = first_valid; i < len; ++i) {
        const double ci = close[i];
        double tr;
        if (have_prev && isfinite(ci)) {
            const double tr1 = high[i] - low[i];
            const double tr2 = fabs(high[i] - prev_close);
            const double tr3 = fabs(low[i] - prev_close);
            tr = fmax(fmax(tr1, tr2), tr3);
        } else {
            tr = 0.0;
        }
        if (isfinite(ci)) {
            prev_close = ci;
            have_prev = true;
        }

        /* The scalar authority intentionally uses the absolute row index. */
        if (i < vis_atr) {
            sum_vis += tr;
            if (i == vis_atr - 1) atr_vis_val = sum_vis / vis_atr_f;
        } else if (isfinite(atr_vis_val)) {
            atr_vis_val = ((vis_atr_f - 1.0) * atr_vis_val + tr) / vis_atr_f;
        }
        if (i < sed_atr) {
            sum_sed += tr;
            if (i == sed_atr - 1) atr_sed_val = sum_sed / sed_atr_f;
        } else if (isfinite(atr_sed_val)) {
            atr_sed_val = ((sed_atr_f - 1.0) * atr_sed_val + tr) / sed_atr_f;
        }

        const double val = isnan(ci) ? 0.0 : ci;
        const double old_v = ring_vis[idx_vis];
        ring_vis[idx_vis] = val;
        idx_vis = (idx_vis + 1) % vis_std;
        if (filled_vis < vis_std) {
            filled_vis += 1;
            sum_vis_std += val;
            sum_sq_vis_std += val * val;
        } else {
            sum_vis_std = sum_vis_std - old_v + val;
            sum_sq_vis_std = sum_sq_vis_std - (old_v * old_v) + (val * val);
        }

        const double old_s = ring_sed[idx_sed];
        ring_sed[idx_sed] = val;
        idx_sed = (idx_sed + 1) % sed_std;
        if (filled_sed < sed_std) {
            filled_sed += 1;
            sum_sed_std += val;
            sum_sq_sed_std += val * val;
        } else {
            sum_sed_std = sum_sed_std - old_s + val;
            sum_sq_sed_std = sum_sq_sed_std - (old_s * old_s) + (val * val);
        }

        if (i >= needed) {
            const double p1 = (i >= 1 && !isnan(vol[i - 1])) ? vol[i - 1] : 0.0;
            const double p3 = (i >= 3 && !isnan(vol[i - 3])) ? vol[i - 3] : 0.0;
            const double sed_safe =
                (isfinite(atr_sed_val) && atr_sed_val != 0.0)
                    ? atr_sed_val
                    : (atr_sed_val + NEO_F64_EPSILON);
            vol[i] = (atr_vis_val / sed_safe) + lag_s * (p1 - p3);

            if (anti && filled_vis == vis_std && filled_sed == sed_std) {
                const double mean_vis = sum_vis_std / (double)vis_std;
                const double mean_sq_vis = sum_sq_vis_std / (double)vis_std;
                const double var_vis = fmax(mean_sq_vis - mean_vis * mean_vis, 0.0);
                const double std_vis = sqrt(var_vis);
                const double mean_sed = sum_sed_std / (double)sed_std;
                const double mean_sq_sed = sum_sq_sed_std / (double)sed_std;
                const double var_sed = fmax(mean_sq_sed - mean_sed * mean_sed, 0.0);
                const double std_sed = sqrt(var_sed);
                const double ratio =
                    (std_sed != 0.0) ? (std_vis / std_sed)
                                     : (std_vis / (std_sed + NEO_F64_EPSILON));
                anti[i] = threshold - ratio;
            }
        }
    }

    const int warm_end = first_valid + needed - 1;
    int cut = warm_end + 1;
    if (cut > len) cut = len;
    for (int i = 0; i < cut; ++i) {
        vol[i] = NEO_F64_NAN;
        if (anti) anti[i] = NEO_F64_NAN;
    }
}

extern "C" __global__
void damiani_volatmeter_neo_batch_f64(const double* __restrict__ data,
                                      int series_len,
                                      const int* __restrict__ periods,
                                      int n_combos,
                                      int first_valid,
                                      double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;   /* period-invariant — see the header. */

    double ring_vis[NEO_DAMIANI_VIS_STD];
    double ring_sed[NEO_DAMIANI_SED_STD];
    double* __restrict__ vol = out + (size_t)combo * (size_t)series_len;
    damiani_volatmeter_row_f64(
        data,
        series_len,
        first_valid,
        NEO_DAMIANI_VIS_ATR,
        NEO_DAMIANI_VIS_STD,
        NEO_DAMIANI_SED_ATR,
        NEO_DAMIANI_SED_STD,
        NEO_DAMIANI_THRESHOLD,
        ring_vis,
        ring_sed,
        vol,
        0);
}

/* Dynamic NeoEthos production route. The preserved primary entry point above
 * remains fixed-default and period-invariant; this exact ABI carries the
 * registry's four coupled windows and both canonical outputs in one launch. */
extern "C" __global__
void damiani_volatmeter_outputs_f64(
    const double* __restrict__ prices,
    int series_len,
    const int* __restrict__ vis_atrs,
    const int* __restrict__ vis_stds,
    const int* __restrict__ sed_atrs,
    const int* __restrict__ sed_stds,
    const double* __restrict__ thresholds,
    int n_combos,
    int first_valid,
    double* __restrict__ ring_vis_scratch,
    int ring_vis_stride,
    double* __restrict__ ring_sed_scratch,
    int ring_sed_stride,
    double* __restrict__ out_vol,
    double* __restrict__ out_anti)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int vis_atr = vis_atrs[combo];
    const int vis_std = vis_stds[combo];
    const int sed_atr = sed_atrs[combo];
    const int sed_std = sed_stds[combo];
    if (vis_std > ring_vis_stride || sed_std > ring_sed_stride) return;

    double* __restrict__ ring_vis =
        ring_vis_scratch + (size_t)combo * (size_t)ring_vis_stride;
    double* __restrict__ ring_sed =
        ring_sed_scratch + (size_t)combo * (size_t)ring_sed_stride;
    double* __restrict__ vol = out_vol + (size_t)combo * (size_t)series_len;
    double* __restrict__ anti = out_anti + (size_t)combo * (size_t)series_len;
    damiani_volatmeter_row_f64(
        prices,
        series_len,
        first_valid,
        vis_atr,
        vis_std,
        sed_atr,
        sed_std,
        thresholds[combo],
        ring_vis,
        ring_sed,
        vol,
        anti);
}
