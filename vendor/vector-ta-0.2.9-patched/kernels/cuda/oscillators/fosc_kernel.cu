#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }


extern "C" __global__ void fosc_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;


    const int warm_end = (warm < len) ? warm : len;
    for (int t = 0; t < warm_end; ++t) {
        out[row_off + t] = f32_nan();
    }
    if (warm >= len) return;


    const double p   = (double)period;
    const double p1  = p + 1.0;
    const double inv_p = 1.0 / p;
    const double sx  = 0.5 * p * p1;
    const double sx2 = (p * p1 * (2.0 * p + 1.0)) / 6.0;
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);


    double sum_y = 0.0;
    double sum_xy = 0.0;
    double w = 1.0;
    for (int k = 0; k < period - 1; ++k, w += 1.0f) {
        const double d = (double)data[first_valid + k];
        sum_y += d;
        sum_xy = fma(d, w, sum_xy);
    }


    double tsf_prev = 0.0;

    for (int t = warm; t < len; ++t) {
        const float cur = data[t];
        const double y_plus = sum_y + (double)cur;
        const double xy_plus = sum_xy + (double)cur * p;


        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;


        float out_val;
        if ((cur == cur) && cur != 0.0f) {

            const double cd = (double)cur;
            const double ov = 100.0 * ((cd - tsf_prev) / cd);
            out_val = (float)ov;
        } else {
            out_val = f32_nan();
        }
        out[row_off + t] = out_val;


        tsf_prev = b * p1 + a;


        const int old_idx = t + 1 - period;
        const float oldv = data[old_idx];


        sum_xy = xy_plus - y_plus;


        sum_y = y_plus - oldv;
    }
}


extern "C" __global__ void fosc_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;


    const int warm_end = (warm < rows) ? warm : rows;
    for (int t = 0; t < warm_end; ++t) {
        out_tm[t * cols + s] = f32_nan();
    }
    if (warm >= rows) return;


    const double p   = (double)period;
    const double p1  = p + 1.0;
    const double inv_p = 1.0 / p;
    const double sx  = 0.5 * p * p1;
    const double sx2 = (p * p1 * (2.0 * p + 1.0)) / 6.0;
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);


    double sum_y = 0.0;
    double sum_xy = 0.0;
    double w = 1.0;
    for (int k = 0; k < period - 1; ++k, w += 1.0f) {
        const double d = (double)data_tm[(fv + k) * cols + s];
        sum_y += d;
        sum_xy = fma(d, w, sum_xy);
    }

    double tsf_prev = 0.0;
    for (int t = warm; t < rows; ++t) {
        const float cur = data_tm[t * cols + s];
        const double y_plus = sum_y + (double)cur;
        const double xy_plus = sum_xy + (double)cur * p;

        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;

        float out_val;
        if ((cur == cur) && cur != 0.0f) {
            const double cd = (double)cur;
            const double ov = 100.0 * ((cd - tsf_prev) / cd);
            out_val = (float)ov;
        } else {
            out_val = f32_nan();
        }
        out_tm[t * cols + s] = out_val;

        tsf_prev = b * p1 + a;

        const int old_idx = t + 1 - period;
        const float oldv = data_tm[old_idx * cols + s];

        sum_xy = xy_plus - y_plus;

        sum_y = y_plus - oldv;
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

__device__ __forceinline__ double f64_nan() { return __longlong_as_double(0x7fffffffffffffffULL); }
extern "C" __global__ void fosc_batch_f64(
    const double* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;


    const int warm_end = (warm < len) ? warm : len;
    for (int t = 0; t < warm_end; ++t) {
        out[row_off + t] = f64_nan();
    }
    if (warm >= len) return;


    const double p   = (double)period;
    const double p1  = p + 1.0;
    const double inv_p = 1.0 / p;
    const double sx  = 0.5 * p * p1;
    const double sx2 = (p * p1 * (2.0 * p + 1.0)) / 6.0;
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);


    double sum_y = 0.0;
    double sum_xy = 0.0;
    double w = 1.0;
    for (int k = 0; k < period - 1; ++k, w += 1.0) {
        const double d = (double)data[first_valid + k];
        sum_y += d;
        sum_xy = fma(d, w, sum_xy);
    }


    double tsf_prev = 0.0;

    for (int t = warm; t < len; ++t) {
        const double cur = data[t];
        const double y_plus = sum_y + (double)cur;
        const double xy_plus = sum_xy + (double)cur * p;


        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;


        double out_val;
        if ((cur == cur) && cur != 0.0) {

            const double cd = (double)cur;
            const double ov = 100.0 * ((cd - tsf_prev) / cd);
            out_val = (double)ov;
        } else {
            out_val = f64_nan();
        }
        out[row_off + t] = out_val;


        tsf_prev = b * p1 + a;


        const int old_idx = t + 1 - period;
        const double oldv = data[old_idx];


        sum_xy = xy_plus - y_plus;


        sum_y = y_plus - oldv;
    }
}
extern "C" __global__ void fosc_many_series_one_param_time_major_f64(
    const double* __restrict__ data_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    double* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;


    const int warm_end = (warm < rows) ? warm : rows;
    for (int t = 0; t < warm_end; ++t) {
        out_tm[t * cols + s] = f64_nan();
    }
    if (warm >= rows) return;


    const double p   = (double)period;
    const double p1  = p + 1.0;
    const double inv_p = 1.0 / p;
    const double sx  = 0.5 * p * p1;
    const double sx2 = (p * p1 * (2.0 * p + 1.0)) / 6.0;
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);


    double sum_y = 0.0;
    double sum_xy = 0.0;
    double w = 1.0;
    for (int k = 0; k < period - 1; ++k, w += 1.0) {
        const double d = (double)data_tm[(fv + k) * cols + s];
        sum_y += d;
        sum_xy = fma(d, w, sum_xy);
    }

    double tsf_prev = 0.0;
    for (int t = warm; t < rows; ++t) {
        const double cur = data_tm[t * cols + s];
        const double y_plus = sum_y + (double)cur;
        const double xy_plus = sum_xy + (double)cur * p;

        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;

        double out_val;
        if ((cur == cur) && cur != 0.0) {
            const double cd = (double)cur;
            const double ov = 100.0 * ((cd - tsf_prev) / cd);
            out_val = (double)ov;
        } else {
            out_val = f64_nan();
        }
        out_tm[t * cols + s] = out_val;

        tsf_prev = b * p1 + a;

        const int old_idx = t + 1 - period;
        const double oldv = data_tm[old_idx * cols + s];

        sum_xy = xy_plus - y_plus;

        sum_y = y_plus - oldv;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — fosc
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/fosc.rs:309 `fosc_core` (reached through
 *   `fosc_scalar`, :388). PERIOD-SWEPT: `compute_fosc_batch` (cpu_batch.rs:3117)
 *   reads a parameter literally named `period` (default 5), so the swept int
 *   IS this indicator's parameter.
 *
 * Column: `expect_value_output` then the single series (cpu_batch.rs:3113).
 * Warmup: `alloc_with_nan_prefix(len, first + period - 1)` (:212).
 *
 * Shape: ONE THREAD PER COLUMN. `y` and `xy` are rolling sums updated by
 *   subtract-then-add across bars, and `tsf_prev` is the PREVIOUS bar forecast,
 *   so the value at bar i depends on bar i-1. Not bar-parallel.
 *
 * Two details taken verbatim:
 *   * The seed loop unrolls by four and adds `d0 + d1 + d2 + d3` as ONE
 *     grouped term before folding it into `y` (:346). That is a different
 *     association from four separate `y +=`, and on a five-bar window it is
 *     the difference between matching the CPU and not. The tail loop (:353)
 *     then adds one at a time. Both shapes are reproduced.
 *   * `denom.abs() < f64::EPSILON` (:323) — the guard is DOUBLE epsilon,
 *     2.220446049250313e-16. An f32-sized 1.19e-7 here would zero `bd` for
 *     denominators the CPU accepts.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#ifndef NEO_F64_EPSILON
#define NEO_F64_EPSILON 2.2204460492503131e-16
#endif

extern "C" __global__
void fosc_neo_batch_f64(const double* __restrict__ data,
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
    if (period <= 0 || n < period) return;
    if (first_valid < 0 || first_valid >= n) return;

    const int begin = first_valid + period - 1;
    if (begin >= n) return;

    const double p     = (double)period;
    const double x     = 0.5 * p * (p + 1.0);
    const double x2    = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0;
    const double denom = p * x2 - x * x;
    const double bd    = (fabs(denom) < NEO_F64_EPSILON) ? 0.0 : (1.0 / denom);
    const double inv_p = 1.0 / p;
    const double p_bd  = p * bd;
    const double x_bd  = x * bd;
    const double tsf_coeff = 0.5 * (p + 1.0);

    double y = 0.0, xy = 0.0;
    const int limit = period - 1;
    int k = 0;
    while (k + 4 <= limit) {
        const double d0 = data[first_valid + k + 0];
        const double d1 = data[first_valid + k + 1];
        const double d2 = data[first_valid + k + 2];
        const double d3 = data[first_valid + k + 3];
        y  += d0 + d1 + d2 + d3;
        xy += d0 * (double)(k + 1);
        xy += d1 * (double)(k + 2);
        xy += d2 * (double)(k + 3);
        xy += d3 * (double)(k + 4);
        k += 4;
    }
    while (k < limit) {
        const double d = data[first_valid + k];
        y  += d;
        xy += d * (double)(k + 1);
        ++k;
    }

    double tsf_prev = 0.0;
    for (int i = begin; i < n; ++i) {
        const double newv = data[i];
        const double y_plus  = y + newv;
        const double xy_plus = xy + newv * p;

        o[i] = (newv != 0.0) ? (100.0 * (newv - tsf_prev) / newv) : NEO_F64_NAN;

        const double b = xy_plus * p_bd - y_plus * x_bd;
        tsf_prev = y_plus * inv_p + b * tsf_coeff;

        const double oldv = data[i + 1 - period];
        xy = xy_plus - y_plus;
        y  = y_plus - oldv;
    }
}
