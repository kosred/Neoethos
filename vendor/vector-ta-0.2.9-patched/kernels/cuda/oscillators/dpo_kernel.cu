#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

extern "C" __global__ void dpo_build_prefix_ds_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    float2* __restrict__ prefix_sum_ds)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_sum_ds[0] = make_float2(0.0f, 0.0f);

    float hi = 0.0f;
    float lo = 0.0f;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float v = data[i];
            const float y = v - lo;
            const float t = hi + y;
            lo = (t - hi) - y;
            hi = t;
        }
        prefix_sum_ds[i + 1] = make_float2(hi, lo);
    }
}


extern "C" __global__ void dpo_batch_f32(
    const float*  __restrict__ data,
    const float2* __restrict__ prefix_sum_ds,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const int back = period / 2 + 1;
    const int warm = max(first_valid + period - 1, back);
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();

    const float inv_p = 1.0f / (float)period;

    const float* __restrict__ price_base = data - back;
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = t + 1;
            const int wl = wr - period;


            const float2 r = prefix_sum_ds[wr];
            const float2 l = prefix_sum_ds[wl];
            const float sum_hi = r.x - l.x;
            const float sum_lo = r.y - l.y;

            const float price = price_base[t];

            float tmp = fmaf(-inv_p, sum_hi, price);
            out_val    = fmaf(-inv_p, sum_lo, tmp);
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void dpo_many_series_one_param_time_major_f32(
    const float*  __restrict__ data_tm,
    const float2* __restrict__ prefix_sum_tm_ds,
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int back = period / 2 + 1;
    const int warm = max(fv + period - 1, back);

    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();
    const float inv_p = 1.0f / (float)period;

    for (int t = tx; t < rows; t += stride) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = (t * cols + s) + 1;
            const int wl = (t >= period) ? ((t - period) * cols + s) + 1 : 0;

            const float2 r = prefix_sum_tm_ds[wr];
            const float2 l = prefix_sum_tm_ds[wl];
            const float sum_hi = r.x - l.x;
            const float sum_lo = r.y - l.y;

            const float price = data_tm[(t - back) * cols + s];
            float tmp = fmaf(-inv_p, sum_hi, price);
            out_val    = fmaf(-inv_p, sum_lo, tmp);
        }
        out_tm[t * cols + s] = out_val;
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
extern "C" __global__ void dpo_build_prefix_ds_f64(
    const double* __restrict__ data,
    int len,
    int first_valid,
    double2* __restrict__ prefix_sum_ds)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_sum_ds[0] = make_double2(0.0, 0.0);

    double hi = 0.0;
    double lo = 0.0;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const double v = data[i];
            const double y = v - lo;
            const double t = hi + y;
            lo = (t - hi) - y;
            hi = t;
        }
        prefix_sum_ds[i + 1] = make_double2(hi, lo);
    }
}
extern "C" __global__ void dpo_batch_f64(
    const double*  __restrict__ data,
    const double2* __restrict__ prefix_sum_ds,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const int back = period / 2 + 1;
    const int warm = max(first_valid + period - 1, back);
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const double nan = f64_nan();

    const double inv_p = 1.0 / (double)period;

    const double* __restrict__ price_base = data - back;
    while (t < len) {
        double out_val = nan;
        if (t >= warm) {
            const int wr = t + 1;
            const int wl = wr - period;


            const double2 r = prefix_sum_ds[wr];
            const double2 l = prefix_sum_ds[wl];
            const double sum_hi = r.x - l.x;
            const double sum_lo = r.y - l.y;

            const double price = price_base[t];

            double tmp = fma(-inv_p, sum_hi, price);
            out_val    = fma(-inv_p, sum_lo, tmp);
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}
extern "C" __global__ void dpo_many_series_one_param_time_major_f64(
    const double*  __restrict__ data_tm,
    const double2* __restrict__ prefix_sum_tm_ds,
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    double* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int back = period / 2 + 1;
    const int warm = max(fv + period - 1, back);

    const int stride = gridDim.x * blockDim.x;
    const double nan = f64_nan();
    const double inv_p = 1.0 / (double)period;

    for (int t = tx; t < rows; t += stride) {
        double out_val = nan;
        if (t >= warm) {
            const int wr = (t * cols + s) + 1;
            const int wl = (t >= period) ? ((t - period) * cols + s) + 1 : 0;

            const double2 r = prefix_sum_tm_ds[wr];
            const double2 l = prefix_sum_tm_ds[wl];
            const double sum_hi = r.x - l.x;
            const double sum_lo = r.y - l.y;

            const double price = data_tm[(t - back) * cols + s];
            double tmp = fma(-inv_p, sum_hi, price);
            out_val    = fma(-inv_p, sum_lo, tmp);
        }
        out_tm[t * cols + s] = out_val;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - dpo (detrended price oscillator)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/dpo.rs:316 `dpo_scalar`, entered from
 *             `dpo_into_slice` (:183) whose `first` is at :190.
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:3650 `expect_value_output`).
 *
 * PERIOD-SWEPT: `compute_dpo_batch` reads `period` (default 5), so
 * `periods[combo]` is honoured and every row differs.
 *
 * FIRST-VALID: `!is_nan` on the single source series - `AllInputsNonNan`.
 *
 * WARMUP: the batch matrix arrives PRE-FILLED WITH NaN
 * (`collect_f64_into_rows`, cpu_batch.rs:2544) and `dpo_scalar` writes only
 * from `max(first + period - 1, back)` onward, where `back = period/2 + 1`.
 * That maximum matters: for a long period the rolling sum is ready before
 * `back` bars of lag exist, and the CPU then ADVANCES THE SUM WITHOUT WRITING
 * (:349-357). Skipping that catch-up would leave the sum one window behind
 * for every later bar.
 *
 * ACCUMULATION ORDER: the CPU has three shapes - a generic 4-wide unrolled
 * loop (:355), a `period == 5` special case with a written-out seed (:399),
 * and a wasm SIMD path. The unrolled bodies compute `s1 = (sum + a1) - r1`,
 * `s2 = (s1 + a2) - r2`, ... which is EXACTLY the scalar `sum = (sum + add) -
 * sub` chain; and the `period == 5` seed `(((d0+d1)+d2)+d3)+d4` is the same
 * left-association a `sum = 0.0; sum += d[k]` loop produces, because
 * `0.0 + d0` is exact. ONE kernel therefore serves both, and `scale = 0.2` in
 * the special case is the same binary64 value as `1.0 / 5.0`.
 *
 * ROUNDING: `sum.mul_add(-scale, p)` is ONE fused rounding - written as
 * `fma(sum, -scale, p)`, not as `p - sum * scale`, which would be two.
 * `-fmad=false` in build.rs stops the compiler contracting anything else.
 *
 * SEQUENTIAL, one thread per combo column: the rolling sum is a running
 * accumulator whose add/subtract order is load-bearing.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void dpo_neo_batch_f64(const double* __restrict__ data,
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
    const int period = periods[combo];

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len) return;
    if (len - first_valid < period) return;      /* dpo.rs:202 NotEnoughValidData */

    const int    back  = period / 2 + 1;
    const double scale = 1.0 / (double)period;

    double sum = 0.0;
    for (int k = 0; k < period; ++k) sum += data[first_valid + k];

    int i = first_valid + period - 1;

    /* Catch-up: advance the window without emitting while `i < back`. */
    while (i < back && i + 1 < len) {
        const int next = i + 1;
        sum = (sum + data[next]) - data[next - period];
        i = next;
    }

    for (; i < len; ++i) {
        if (i >= back) o[i] = fma(sum, -scale, data[i - back]);
        if (i + 1 < len) {
            const int next = i + 1;
            sum = (sum + data[next]) - data[next - period];
        }
    }
}
