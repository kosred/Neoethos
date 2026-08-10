#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

static __forceinline__ __device__ bool finite_f(float x) { return isfinite(x); }


static __forceinline__ __device__ float alpha_from_period_iir_f(int p) {
    if (p < 1) p = 1;
    const float omega = 2.0f * CUDART_PI_F / (float)p;
    float s, c;

    sincosf(omega, &s, &c);
    return (1.0f - s) / c;
}

static __forceinline__ __device__ float lut_or_formula_alpha(
    int p, const float* __restrict__ alpha_lut, int lut_len, int lut_pmin)
{
    if (p < lut_pmin) p = lut_pmin;
    if (alpha_lut) {
        int idx = p - lut_pmin;
        if (idx < 0) idx = 0;
        if (idx >= lut_len) idx = lut_len - 1;
        return alpha_lut[idx];
    }
    return alpha_from_period_iir_f(p);
}

extern "C" __global__ void lpc_build_true_range_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    float* __restrict__ tr_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    if (i == 0) {
        tr_out[0] = high[0] - low[0];
        return;
    }

    const float hl  = high[i] - low[i];
    const float c_l = fabsf(close[i] - low[i - 1]);
    const float c_h = fabsf(close[i] - high[i - 1]);
    tr_out[i] = fmaxf(hl, fmaxf(c_l, c_h));
}

extern "C" __global__ void lpc_build_dom_cycle_f32_serial(
    const float* __restrict__ src,
    int len,
    int max_cycle_limit,
    double* __restrict__ delta_phase_ring,
    int delta_phase_ring_len,
    float* __restrict__ dom_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);
    for (int i = 0; i < len; ++i) {
        dom_out[i] = qnan;
    }
    if (len < 8 || delta_phase_ring_len <= 0) return;

    for (int i = 0; i < delta_phase_ring_len; ++i) {
        delta_phase_ring[i] = 0.0;
    }

    double in_phase_hist[4] = {0.0, 0.0, 0.0, 0.0};
    double quadrature_hist[4] = {0.0, 0.0, 0.0, 0.0};
    double real_prev = 0.0;
    double imag_prev = 0.0;
    double inst_prev = 0.0;
    double dom_prev = CUDART_NAN;

    for (int i = 7; i < len; ++i) {
        const int src_m11 = (i >= 11) ? (i - 11) : 0;
        const int src_m9  = (i >= 9) ? (i - 9) : 0;

        const double val1 = (double)src[i] - (double)src[i - 7];
        const double val1_4 = (double)src[i - 4] - (double)src[src_m11];
        const double val1_2 = (double)src[i - 2] - (double)src[src_m9];

        const double in_phase_i =
            1.25 * (val1_4 - 0.635 * val1_2) + 0.635 * in_phase_hist[(i - 3) & 3];
        const double quadrature_i =
            val1_2 - 0.338 * val1 + 0.338 * quadrature_hist[(i - 2) & 3];

        const double in_phase_prev = in_phase_hist[(i - 1) & 3];
        const double quadrature_prev = quadrature_hist[(i - 1) & 3];

        const double real_i =
            0.2 * (in_phase_i * in_phase_prev + quadrature_i * quadrature_prev) + 0.8 * real_prev;
        const double imag_i =
            0.2 * (in_phase_i * quadrature_prev - in_phase_prev * quadrature_i) + 0.8 * imag_prev;

        double delta_i = 0.0;
        if (real_i != 0.0) {
            delta_i = atan(imag_i / real_i);
        }
        delta_phase_ring[i % delta_phase_ring_len] = delta_i;

        double val2 = 0.0;
        bool found_period = false;
        double inst_i = inst_prev;
        const int limit = max_cycle_limit < i ? max_cycle_limit : i;
        for (int j = 0; j <= limit; ++j) {
            val2 += delta_phase_ring[(i - j) % delta_phase_ring_len];
            if (val2 > 2.0 * CUDART_PI && !found_period) {
                inst_i = (double)j;
                found_period = true;
                break;
            }
        }

        if (!found_period) {
            inst_i = (i > 0) ? inst_prev : 20.0;
        }

        const double dom_i = !isnan(dom_prev) ? (0.25 * inst_i + 0.75 * dom_prev) : inst_i;
        dom_out[i] = (float)dom_i;

        in_phase_hist[i & 3] = in_phase_i;
        quadrature_hist[i & 3] = quadrature_i;
        real_prev = real_i;
        imag_prev = imag_i;
        inst_prev = inst_i;
        dom_prev = dom_i;
    }
}


extern "C" __global__ void lpc_batch_f32_v2(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ src,
    int len,
    const float* __restrict__ tr_opt,
    const int*   __restrict__ fixed_periods,
    const float* __restrict__ cycle_mults,
    const float* __restrict__ tr_mults,
    int n_combos,
    int first_valid,
    int cutoff_mode,
    int max_cycle_limit,
    const float* __restrict__ dom,

    const float* __restrict__ alpha_lut,
    int alpha_lut_len,
    int alpha_lut_pmin,

    int out_time_major,
    float* __restrict__ out_filter,
    float* __restrict__ out_high,
    float* __restrict__ out_low
){
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);

    for (int combo = tid; combo < n_combos; combo += stride) {

        auto store_triplet = [&](int i, float f, float hi, float lo) {
            size_t idx = out_time_major ? (size_t)i * (size_t)n_combos + (size_t)combo
                                        : (size_t)combo * (size_t)len    + (size_t)i;
            out_filter[idx] = f;
            out_high[idx]   = hi;
            out_low[idx]    = lo;
        };


        if (first_valid > 0) {
            const int upto = first_valid < len ? first_valid : len;
            for (int i = 0; i < upto; ++i) store_triplet(i, qnan, qnan, qnan);
            if (first_valid >= len) continue;
        }


        const float tm       = tr_mults[combo];
        const int   p_fixed  = fixed_periods[combo];
        const float cm       = cycle_mults[combo];
        const bool  adaptive = (cutoff_mode != 0) && (dom != nullptr);


        const int i0 = first_valid;
        float s_prev = src[i0];
        float f_prev = s_prev;


        float tr_prev = tr_opt ? tr_opt[i0] : (high[i0] - low[i0]);
        float ftr_prev = tr_prev;


        int last_p = adaptive ? 0 : p_fixed;
        float alpha = lut_or_formula_alpha(p_fixed, alpha_lut, alpha_lut_len, alpha_lut_pmin);


        store_triplet(i0, f_prev, f_prev + tm * tr_prev, f_prev - tm * tr_prev);


        #pragma unroll 1
        for (int i = i0 + 1; i < len; ++i) {

            int p_i = p_fixed;
            if (adaptive) {
                float base = dom[i];
                if (!finite_f(base)) {
                    p_i = p_fixed;
                } else {
                    float pd = nearbyintf(base * cm);
                    if (pd < 3.0f) pd = 3.0f;
                    if (max_cycle_limit > 0 && pd > (float)max_cycle_limit) pd = (float)max_cycle_limit;
                    p_i = (int)pd;
                }
            }
            if (p_i != last_p) {
                alpha  = lut_or_formula_alpha(p_i, alpha_lut, alpha_lut_len, alpha_lut_pmin);
                last_p = p_i;
            }
            const float one_m_a = 1.0f - alpha;
            const float w = 0.5f * one_m_a;


            const float s_i = src[i];

            const float f_i = fmaf(alpha, f_prev, w * (s_i + s_prev));
            s_prev = s_i;
            f_prev = f_i;


            float tr_i;
            if (tr_opt) {
                tr_i = tr_opt[i];
            } else {
                const float hl  = high[i] - low[i];
                const float c_l = fabsf(close[i] - low[i - 1]);
                const float c_h = fabsf(close[i] - high[i - 1]);
                tr_i = fmaxf(hl, fmaxf(c_l, c_h));
            }
            const float ftr_i = fmaf(alpha, ftr_prev, w * (tr_i + tr_prev));
            tr_prev  = tr_i;
            ftr_prev = ftr_i;


            const float hi = f_i + tm * ftr_i;
            const float lo = f_i - tm * ftr_i;
            store_triplet(i, f_i, hi, lo);
        }
    }
}


extern "C" __global__ void lpc_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ src,
    int len,
    const float* __restrict__ tr_opt,
    const int* __restrict__ fixed_periods,
    const float* __restrict__ cycle_mults,
    const float* __restrict__ tr_mults,
    int n_combos,
    int first_valid,
    int cutoff_mode,
    int max_cycle_limit,
    const float* __restrict__ dom,
    float* __restrict__ out_filter,
    float* __restrict__ out_high,
    float* __restrict__ out_low)
{
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int combo = row0; combo < n_combos; combo += stride) {
        float* f_row  = out_filter + (size_t)combo * (size_t)len;
        float* hi_row = out_high   + (size_t)combo * (size_t)len;
        float* lo_row = out_low    + (size_t)combo * (size_t)len;

        const float tm_f = tr_mults[combo];
        const double tm = (double)tm_f;
        const int p_fixed = fixed_periods[combo];
        const float cm_f = cycle_mults[combo];
        const double cm = (double)cm_f;


        const uint32_t qnan_bits = 0x7fc00000u;
        const float qnan = __int_as_float(qnan_bits);
        const int warm = first_valid < len ? first_valid : len;
        for (int i = 0; i < warm; ++i) {
            f_row[i]  = qnan;
            hi_row[i] = qnan;
            lo_row[i] = qnan;
        }
        if (first_valid >= len) continue;


        const int i0 = first_valid;
        const double s0 = (double)src[i0];
        f_row[i0] = (float)s0;

        double tr_prev = (double)(tr_opt ? tr_opt[i0] : (high[i0] - low[i0]));
        double ftr_prev = tr_prev;
        hi_row[i0] = (float)(s0 + tm * tr_prev);
        lo_row[i0] = (float)(s0 - tm * tr_prev);


        int last_p = (cutoff_mode == 0 ? p_fixed : 0);

        auto alpha_from_period_iir = [](int p)->double {
            if (p < 1) p = 1;
            const double omega = 2.0 * CUDART_PI / (double)p;
            double s = sin(omega), c = cos(omega);
            return (1.0 - s) / c;
        };
        double alpha = (cutoff_mode == 0 ? alpha_from_period_iir(p_fixed) : 0.0);

        for (int i = i0 + 1; i < len; ++i) {

            int p_i = p_fixed;
            if (cutoff_mode != 0 && dom != nullptr) {
                double base = (double)dom[i];
                if (!isfinite(base)) {
                    p_i = p_fixed;
                } else {
                    double pd = nearbyint(base * cm);
                    if (pd < 3.0) pd = 3.0;
                    if (max_cycle_limit > 0 && pd > (double)max_cycle_limit) pd = (double)max_cycle_limit;
                    p_i = (int)pd;
                }
            }

            if (p_i != last_p) {
                last_p = p_i;
                alpha = alpha_from_period_iir(p_i);
            }
            const double one_m_a = 1.0 - alpha;


            const double s_im1 = (double)src[i - 1];
            const double s_i   = (double)src[i];
            const double prev_f = (double)f_row[i - 1];
            const double f_i = fma(alpha, prev_f, 0.5 * one_m_a * (s_i + s_im1));
            f_row[i] = (float)f_i;


            double tr_i;
            if (tr_opt) {
                tr_i = (double)tr_opt[i];
            } else {
                const double hl  = (double)(high[i] - low[i]);
                const double c_l = fabs((double)close[i] - (double)low[i - 1]);
                const double c_h = fabs((double)close[i] - (double)high[i - 1]);
                tr_i = fmax(hl, fmax(c_l, c_h));
            }
            const double ftr_i = fma(alpha, ftr_prev, 0.5 * one_m_a * (tr_i + tr_prev));
            tr_prev = tr_i;
            ftr_prev = ftr_i;

            hi_row[i] = (float)(f_i + tm * ftr_i);
            lo_row[i] = (float)(f_i - tm * ftr_i);
        }
    }
}


extern "C" __global__ void lpc_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ src_tm,
    int cols,
    int rows,
    int fixed_period,
    float cycle_mult,
    float tr_mult,
    int cutoff_mode,
    int max_cycle_limit,
    const int* __restrict__ first_valids,
    float* __restrict__ out_filter_tm,
    float* __restrict__ out_high_tm,
    float* __restrict__ out_low_tm
) {
    const int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s0 >= cols) return;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);

    const int first = first_valids[s0];
    for (int t = 0; t < (first < rows ? first : rows); ++t) {
        const size_t idx = (size_t)t * (size_t)cols + (size_t)s0;
        out_filter_tm[idx] = qnan;
        out_high_tm[idx]   = qnan;
        out_low_tm[idx]    = qnan;
    }
    if (first >= rows) return;

    const float tm = tr_mult;
    float alpha = alpha_from_period_iir_f(fixed_period);

    auto AT = [&](const float* a, int t) -> float { return a[(size_t)t * (size_t)cols + (size_t)s0]; };
    auto W  = [&](float* a, int t, float v)       { a[(size_t)t * (size_t)cols + (size_t)s0] = v;  };


    float s_prev = AT(src_tm, first);
    float f_prev = s_prev;
    float tr_prev = AT(high_tm, first) - AT(low_tm, first);
    float ftr_prev = tr_prev;

    W(out_filter_tm, first, f_prev);
    W(out_high_tm,   first, f_prev + tm * tr_prev);
    W(out_low_tm,    first, f_prev - tm * tr_prev);


    #pragma unroll 1
    for (int t = first + 1; t < rows; ++t) {
        const float one_m_a = 1.0f - alpha;
        const float w = 0.5f * one_m_a;

        const float s_i = AT(src_tm, t);
        const float f_i = fmaf(alpha, f_prev, w * (s_i + s_prev));
        s_prev = s_i;
        f_prev = f_i;

        const float hl  = AT(high_tm, t) - AT(low_tm, t);
        const float c_l = fabsf(AT(close_tm, t) - AT(low_tm, t - 1));
        const float c_h = fabsf(AT(close_tm, t) - AT(high_tm, t - 1));
        const float tr_i = fmaxf(hl, fmaxf(c_l, c_h));

        const float ftr_i = fmaf(alpha, ftr_prev, w * (tr_i + tr_prev));
        tr_prev = tr_i;
        ftr_prev = ftr_i;

        W(out_filter_tm, t, f_i);
        W(out_high_tm,   t, f_i + tm * ftr_i);
        W(out_low_tm,    t, f_i - tm * ftr_i);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3   (lpc)
 *
 * CPU reference: `lpc_scalar` (src/indicators/lpc.rs:523) plus the two helpers
 *   it calls, `dom_cycle` (:415) and the per-bar `alpha_from_period` (:564).
 *
 * Column: output_id "value" -> `out.filter` (cpu_batch.rs:15271-15273). The
 *   `high_band`/`low_band` columns need `tr_mult` and the smoothed true range;
 *   neither touches `filter`, so neither is computed here.
 *
 * PERIOD-INVARIANT: `compute_lpc_batch` reads `cutoff_type` ("adaptive"),
 *   `fixed_period` (20), `max_cycle_limit` (60), `cycle_mult` (1.0) and
 *   `tr_mult` (1.0) (cpu_batch.rs:15248-15252) and NEVER `period`.
 *
 * Input: high / low / close, with `src` bound to CLOSE
 *   (cpu_batch.rs:15218-15221 -> `(high, low, close, close)`) ->
 *   F64InputKind::Hlc.
 *
 * FIRST-VALID: `lpc_prepare` :1285-1290 is the first index at which src, high,
 *   low AND close are all non-NaN -- and since src IS close, that is exactly
 *   F64FirstValidRule::AllInputsNonNan over the Hlc triple.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending, and this one is a recurrence
 *   stacked on a recurrence. `dom_cycle` carries in_phase (which reads its own
 *   value THREE bars back), quadrature (two back), and real/imag (one back);
 *   the filter then carries `out_filter[i-1]` through a per-bar alpha that the
 *   dominant cycle itself selects. Neither level has a scan form that keeps the
 *   rounding.
 *
 * The two loops are FUSED into one ascending pass. That is not an optimisation
 *   -- `dom_cycle` writes a whole array on the CPU and the filter reads
 *   `dc[i]` at the same i, so a single pass is the only shape that avoids a
 *   second `n`-long device buffer per row.
 *
 * The CPU's filter loop is unrolled by two (:616-662 plus the `if i < len`
 *   tail at :665). The unrolling changes no arithmetic and no order: bar i+1
 *   consumes bar i's `f_i` and `tr_prev` exactly as a plain sequential loop
 *   would. This kernel writes the plain loop.
 *
 * `alpha_cache` (:592-607) is not reproduced, and that is a decision, not an
 *   omission: the cache stores `alpha_from_period(p)` for p in 3..=max, and
 *   `alpha_for_period` (:569) returns exactly that or recomputes it. The two
 *   branches are bit-identical by construction, so computing it every bar is
 *   the same number with no table.
 *
 * Roundings, counted against the CPU lines:
 *   :436  in_phase = 1.25 * (val1_4 - 0.635 * val1_2) + 0.635 * in_phase[i-3]
 *   :438  quadrature = val1_2 - 0.338 * val1 + 0.338 * quadrature[i-2]
 *   :440  real = 0.2 * (ip*ip1 + q*q1) + 0.8 * real[i-1]
 *   :446  delta_phase = (imag / real).atan()
 *   :621  let f_i = alpha.mul_add(prev_f, 0.5 * one_m_a * (s_i + s_im1))
 *   Exactly ONE mul_add on this column (:621) -- everything in `dom_cycle` is
 *   plain adds and multiplies, so nothing there is fused here either.
 *
 * `atan`, `sin`, `cos`, never the `f`-suffixed forms: this column is f64 end
 *   to end. The f32 entry points above keep `atanf`/`sincosf`; they are a
 *   different translation unit's worth of callers and are untouched.
 *
 * NaN semantics: `hl.max(c_low1).max(c_hi1)` (:534, :592) is `f64::max` and
 *   returns the NON-NaN operand -- `fmax` is used for the true range even
 *   though `filter` does not consume it, because the same expression appears
 *   in the band columns this file will serve next. The `filter` recursion
 *   itself has no max/min.
 *
 * Epsilon: none. The only tolerance-shaped comparisons on this path are
 *   `real_part[i] != 0.0` (:445) and `val2 > tau` (:456), both exact, so
 *   there is no f32-sized constant to re-derive.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#ifndef NEO_F64_PI
#define NEO_F64_PI 3.14159265358979323846
#endif

/* Defaults from cpu_batch.rs:15249-15251. `max_cycle_limit` bounds the
 * delta-phase ring, and it is a CPU DEFAULT rather than a swept parameter
 * (this indicator is period-invariant), so no caller-supplied number reaches
 * it. */
#define NEO_LPC_FIXED_PERIOD    20
#define NEO_LPC_MAX_CYCLE_LIMIT 60
#define NEO_LPC_CYCLE_MULT      1.0
#define NEO_LPC_DP_RING         (NEO_LPC_MAX_CYCLE_LIMIT + 1)

/* :564-568 -- omega = 2*pi/p, alpha = (1 - sin omega) / cos omega. */
__device__ __forceinline__ double neo_lpc_alpha_from_period(int p)
{
    const double omega = 2.0 * NEO_F64_PI / (double)p;
    double s, c;
    sincos(omega, &s, &c);
    return (1.0 - s) / c;
}

extern "C" __global__
void lpc_neo_batch_f64(const double* __restrict__ high,
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

    const double* __restrict__ src = close;   /* cpu_batch.rs:15221 */

    const int    fixed_period    = NEO_LPC_FIXED_PERIOD;
    const int    max_cycle_limit = NEO_LPC_MAX_CYCLE_LIMIT;
    const double cycle_mult      = NEO_LPC_CYCLE_MULT;

    if (fixed_period == 0 || fixed_period > n) return;   /* :1277 InvalidPeriod */

    const int first = first_valid;
    if (first < 0 || first >= n) return;
    if (n - first < 2) return;                           /* :1293 NotEnoughValidData */

    /* ---- dom_cycle state (:415-456). `len < 8` returns an all-NaN cycle. -- */
    const bool dc_live = (n >= 8);
    double ip1 = 0.0, ip2 = 0.0, ip3 = 0.0;   /* in_phase at i-1, i-2, i-3 */
    double q1  = 0.0, q2  = 0.0;              /* quadrature at i-1, i-2     */
    double real_prev = 0.0, imag_prev = 0.0;
    double inst_prev = 0.0;                   /* inst_per is a zeroed vec    */
    double dom_prev  = NEO_F64_NAN;           /* dom_cycles is a NaN vec     */
    double dp_ring[NEO_LPC_DP_RING];
    for (int k = 0; k < NEO_LPC_DP_RING; ++k) dp_ring[k] = 0.0;
    const double tau = 2.0 * NEO_F64_PI;

    /* ---- filter state (:558-561) ---------------------------------------- */
    double filt_prev = 0.0;
    bool   filt_live = false;
    int    last_p    = 0;      /* dc is Some(..) here, so :608-613 seeds 0 / 0.0 */
    double alpha     = 0.0;

    for (int i = 0; i < n; ++i) {
        /* ---------- dominant cycle for bar i ---------- */
        double dom_i = NEO_F64_NAN;
        if (dc_live && i >= 7) {
            const double val1   = src[i] - src[i - 7];
            const double val1_4 = src[i - 4] - src[(i >= 11) ? (i - 11) : 0];
            const double val1_2 = src[i - 2] - src[(i >= 9)  ? (i - 9)  : 0];

            const double ip_i = 1.25 * (val1_4 - 0.635 * val1_2) + 0.635 * ip3;
            const double q_i  = val1_2 - 0.338 * val1 + 0.338 * q2;

            const double real_i = 0.2 * (ip_i * ip1 + q_i * q1) + 0.8 * real_prev;
            const double imag_i = 0.2 * (ip_i * q1 - ip1 * q_i) + 0.8 * imag_prev;

            const double dp_i = (real_i != 0.0) ? atan(imag_i / real_i) : 0.0;
            dp_ring[i % NEO_LPC_DP_RING] = dp_i;

            /* :450-460 -- walk backwards until the accumulated phase passes
             * a full turn. The ring is exactly `max_cycle_limit + 1` deep, so
             * every index this loop can reach is still resident. */
            double val2 = 0.0;
            bool   found = false;
            const int limit = (max_cycle_limit < i) ? max_cycle_limit : i;
            double inst_i = 0.0;
            for (int j = 0; j <= limit; ++j) {
                val2 += dp_ring[(i - j) % NEO_LPC_DP_RING];
                if (val2 > tau) { inst_i = (double)j; found = true; break; }
            }
            if (!found) inst_i = (i > 0) ? inst_prev : 20.0;   /* :463-465 */

            /* :467-471 */
            if (i > 0 && !isnan(dom_prev)) dom_i = 0.25 * inst_i + 0.75 * dom_prev;
            else                           dom_i = inst_i;

            ip3 = ip2; ip2 = ip1; ip1 = ip_i;
            q2  = q1;  q1  = q_i;
            real_prev = real_i; imag_prev = imag_i;
            inst_prev = inst_i;
            dom_prev  = dom_i;
        } else if (dc_live) {
            /* i < 7: dom_cycles[i] stays NaN and the four state series stay 0. */
            dp_ring[i % NEO_LPC_DP_RING] = 0.0;
        }

        /* ---------- filter for bar i ---------- */
        if (i < first) continue;

        if (i == first) {
            filt_prev = src[first];      /* :557 */
            filt_live = true;
            o[i] = filt_prev;
            continue;
        }
        if (!filt_live) continue;

        /* :578-588 `per_bar_period`: round(dc[i] * cycle_mult), floored at 3;
         * a NaN dominant cycle falls back to `fixed_period`. */
        int p_i;
        if (isnan(dom_i)) {
            p_i = fixed_period;
        } else {
            const double scaled = round(dom_i * cycle_mult);
            p_i = (int)fmax(scaled, 3.0);
        }
        if (p_i != last_p) { last_p = p_i; alpha = neo_lpc_alpha_from_period(last_p); }

        const double one_m_a = 1.0 - alpha;
        const double s_im1   = src[i - 1];
        const double s_i     = src[i];
        /* :621 -- ONE fma, matching `alpha.mul_add(prev_f, ..)`. */
        const double f_i     = fma(alpha, filt_prev, 0.5 * one_m_a * (s_i + s_im1));
        filt_prev = f_i;
        o[i] = f_i;
    }
}
