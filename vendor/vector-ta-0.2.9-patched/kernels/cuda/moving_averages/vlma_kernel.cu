#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float fmaf_safe(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

extern "C" __global__ void vlma_build_prefixes_f32(
    const float* __restrict__ data,
    int len,
    double* __restrict__ prefix_sum,
    double* __restrict__ prefix_sum_sq,
    int* __restrict__ prefix_nan
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_sum[0] = 0.0;
    prefix_sum_sq[0] = 0.0;
    prefix_nan[0] = 0;

    double sum = 0.0;
    double sum_sq = 0.0;
    int nan_count = 0;
    for (int t = 0; t < len; ++t) {
        const float v = data[t];
        if (isnan(v)) {
            ++nan_count;
        } else {
            const double dv = static_cast<double>(v);
            sum += dv;
            sum_sq += dv * dv;
        }
        prefix_sum[t + 1] = sum;
        prefix_sum_sq[t + 1] = sum_sq;
        prefix_nan[t + 1] = nan_count;
    }
}


extern "C" __global__ void vlma_batch_sma_std_prefix_f32(
    const float*  __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    const int*    __restrict__ min_periods,
    const int*    __restrict__ max_periods,
    int len,
    int first_valid,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || len <= 0) return;

    const int min_p = max(1, min_periods[combo]);
    const int max_p = max(min_p, max_periods[combo]);
    if (first_valid < 0 || first_valid >= len) return;

    const int base = combo * len;


    for (int i = threadIdx.x; i < first_valid; i += blockDim.x) {
        out[base + i] = NAN;
    }

    if (threadIdx.x != 0) return;

    const float x0 = data[first_valid];
    out[base + first_valid] = x0;

    const int warm_end = min(len, first_valid + max_p - 1);
    int last_p = max_p;
    float last_val = x0;


    for (int i = first_valid + 1; i < warm_end; ++i) {
        const float x = data[i];
        if (isfinite(x)) {
            const float sc = 2.0f / (float)(last_p + 1);
            last_val = fmaf_safe(x - last_val, sc, last_val);
        }
        out[base + i] = NAN;
    }

    if (warm_end >= len) return;


    for (int i = warm_end; i < len; ++i) {
        const float x = data[i];
        if (!isfinite(x)) {
            out[base + i] = NAN;
            continue;
        }


        const int t1 = i + 1;
        const int t0 = max(0, t1 - max_p);
        const int nan_cnt = prefix_nan[t1] - prefix_nan[t0];

        float sc = 2.0f / (float)(last_p + 1);
        if (nan_cnt == 0) {
            const double sum  = prefix_sum[t1]    - prefix_sum[t0];
            const double sum2 = prefix_sum_sq[t1] - prefix_sum_sq[t0];
            const double inv  = 1.0 / (double)max_p;
            const double m    = sum * inv;
            double var        = (sum2 * inv) - m * m;
            if (var < 0.0) var = 0.0;
            const double dv   = sqrt(var);


            const double d175 = dv * 1.75;
            const double d025 = dv * 0.25;
            const double a = m - d175;
            const double b = m - d025;
            const double c = m + d025;
            const double d = m + d175;

            const int inc_fast = (x < a) || (x > d);
            const int inc_slow = (x >= b) && (x <= c);
            const int delta = inc_slow - inc_fast;
            int p_next = last_p + delta;
            if (p_next < min_p) p_next = min_p;
            if (p_next > max_p) p_next = max_p;
            sc = 2.0f / (float)(p_next + 1);
            last_p = p_next;
        }

        last_val = fmaf_safe(x - last_val, sc, last_val);
        out[base + i] = last_val;
    }
}


extern "C" __global__ void vlma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int min_period,
    int max_period,
    int cols,
    int rows,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x;
    if (s >= cols || rows <= 0) return;

    int min_p = max(1, min_period);
    int max_p = max(min_p, max_period);

    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= rows) return;


    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * cols + s] = NAN;
    }
    if (threadIdx.x != 0) return;


    const float x0 = prices_tm[first_valid * cols + s];
    out_tm[first_valid * cols + s] = x0;

    const int warm_end = min(rows, first_valid + max_p - 1);
    int last_p = max_p;
    float last_val = x0;


    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * cols + s];
        if (isfinite(x)) {
            const float sc = 2.0f / (float)(last_p + 1);
            last_val = fmaf_safe(x - last_val, sc, last_val);
        }
        out_tm[t * cols + s] = NAN;
    }
    if (warm_end >= rows) return;


    double sum = 0.0, sumsq = 0.0;
    int nan_cnt = 0;
    for (int k = 0; k < max_p; ++k) {
        const float v = prices_tm[(first_valid + k) * cols + s];
        if (isfinite(v)) {
            const double dv = (double)v;
            sum += dv;
            sumsq += dv * dv;
        } else {
            ++nan_cnt;
        }
    }
    const double inv_n = 1.0 / (double)max_p;


    for (int t = warm_end; t < rows; ++t) {
        const float x = prices_tm[t * cols + s];
        if (!isfinite(x)) {
            out_tm[t * cols + s] = NAN;
        } else {
            float sc = 2.0f / (float)(last_p + 1);
            if (nan_cnt == 0) {
                const double m  = sum * inv_n;
                double var      = (sumsq * inv_n) - m * m;
                if (var < 0.0) var = 0.0;
                const double dv = sqrt(var);

                const double d175 = dv * 1.75;
                const double d025 = dv * 0.25;
                const double a = m - d175;
                const double b = m - d025;
                const double c = m + d025;
                const double d = m + d175;

                const int inc_fast = (x < a) || (x > d);
                const int inc_slow = (x >= b) && (x <= c);
                int p_next = last_p + (inc_slow - inc_fast);
                if (p_next < min_p) p_next = min_p;
                if (p_next > max_p) p_next = max_p;
                sc = 2.0f / (float)(p_next + 1);
                last_p = p_next;
            }

            last_val = fmaf_safe(x - last_val, sc, last_val);
            out_tm[t * cols + s] = last_val;
        }


        if (t + 1 < rows) {
            const int out_idx = t + 1 - max_p;
            const float leaving = prices_tm[out_idx * cols + s];
            if (isfinite(leaving)) {
                const double dl = (double)leaving;
                sum   -= dl;
                sumsq -= dl * dl;
            } else {
                nan_cnt = max(0, nan_cnt - 1);
            }
            const float enter = prices_tm[(t + 1) * cols + s];
            if (isfinite(enter)) {
                const double de = (double)enter;
                sum   += de;
                sumsq += de * de;
            } else {
                ++nan_cnt;
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 6
 *
 * ORACLE: `vlma_scalar_sma_stddev_into` (src/indicators/vlma.rs:451). That is
 * the arm `vlma_compute_into` (:398) takes when `matype == "sma"` and
 * `devtype == 0`, which are exactly the batch defaults
 * (cpu_batch.rs:15736-15737); `vlma_prepare` maps `Kernel::Auto` onto
 * `Kernel::Scalar` (:381), so there is one CPU answer.
 *
 * NOT `vlma_scalar_into` (:580). That generic arm calls out to `ma()` and
 * `deviation()` and is a DIFFERENT accumulation. It is unreachable at the
 * defaults, and transcribing it would have produced a plausible series that
 * matched nothing.
 *
 * PERIOD-INVARIANT. `compute_vlma_batch` reads `min_period` (5),
 * `max_period` (50), `matype` and `devtype` -- NEVER `period`
 * (cpu_batch.rs:15734-15737). Five swept periods give five identical CPU
 * columns, so the kernel writes five identical rows.
 *
 * SINGLE OUTPUT: "value" is the only column (cpu_batch.rs:15753).
 *
 * THE WARMUP HAS A HOLE IN IT, DELIBERATELY. `alloc_with_nan_prefix` blanks
 * [0, first + max_period - 1), and then the compute writes `out[first] = x0`
 * (:466) back INSIDE that prefix -- `vlma_into_slice` re-blanks the prefix but
 * explicitly skips `i == first` (:334). So the emitted series is: NaN before
 * `first`, the raw price AT `first`, NaN from `first+1` to `warm_end`, values
 * from `warm_end` on. A kernel that blanked the whole prefix would drop one
 * real bar.
 *
 * THE EMA RUNS THROUGH THE WARMUP. Between `first+1` and `warm_end` the CPU
 * still advances `last_val` (:484-491) without emitting it. Skipping that loop
 * would start the emitted series from a different seed.
 *
 * ONE ROUNDING: `fast_ema_update` is `(x - last).mul_add(sc, last)` (:74) --
 * ONE rounding, so `fma(x - last, sc, last)`, not `last + sc*(x-last)`.
 *
 * THE ADAPTIVE PERIOD IS AN INTEGER STATE MACHINE. `delta = inc_slow -
 * inc_fast` where both are 0/1 from four band comparisons (:536-538), then
 * clamped to [min_period, max_period]. Reproduced as ints; the smoothing
 * constant `2 / (p + 1)` is formed from that int exactly as `sc_lut` was
 * (:474), so no table is needed.
 *
 * THE VARIANCE WINDOW IS ROLLED with an explicit `nan_count`, and the roll
 * happens AFTER the emit, for the NEXT bar (:555-572). Rolling first would
 * use tomorrow's window today.
 *
 * `var < 0.0 ? 0.0 : sqrt(var)` is the CPU's `if`, not an fmax -- NaN must
 * reach `sqrt` the way it does on the CPU.
 *
 * NO PER-THREAD ARRAY: the `sc_lut` is replaced by the closed form and the
 * window is read from the resident input, so no `max_period` bound.
 *
 * SEQUENTIAL, one thread per combo column.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define VLMA_NEO_MIN_PERIOD 5    /* cpu_batch.rs:15734 */
#define VLMA_NEO_MAX_PERIOD 50   /* :15735 */

/* fast_ema_update, vlma.rs:74 -- ONE rounding. */
__device__ __forceinline__ double vlma_neo_ema_update(double last, double x, double sc)
{
    return fma(x - last, sc, last);
}

extern "C" __global__
void vlma_neo_batch_f64(const double* __restrict__ data,
                        int series_len,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;                       /* PERIOD-INVARIANT -- see header. */

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;

    if (len == 0) return;
    if (first_valid < 0 || first_valid >= len) return;

    const int min_p_raw = VLMA_NEO_MIN_PERIOD;
    const int max_p     = VLMA_NEO_MAX_PERIOD;
    if (min_p_raw > max_p) return;                       /* :349 */
    if (max_p <= 0 || max_p > len) return;               /* :356 */
    if (len - first_valid < max_p) return;               /* :369 */

    const int min_pi = (min_p_raw == 0) ? 1 : min_p_raw; /* :468 */
    const int max_pi = (max_p > min_pi) ? max_p : min_pi;

    const int warm_end = first_valid + max_p - 1;        /* :464 */

    const double x0 = data[first_valid];
    o[first_valid] = x0;                                 /* :466 -- inside the prefix */

    int    last_p   = max_pi;
    double last_val = x0;

    /* Advance the EMA through the warmup without emitting -- :484-491. */
    for (int i = first_valid + 1; i < len && i < warm_end; ++i) {
        const double x = data[i];
        if (!isnan(x)) {
            const double sc = 2.0 / ((double)last_p + 1.0);   /* sc_lut[last_p], :474 */
            last_val = vlma_neo_ema_update(last_val, x, sc);
        }
    }

    if (warm_end >= len) return;                          /* :493 */

    /* Seed the variance window -- :497-508. */
    double sum = 0.0, sumsq = 0.0;
    int    nan_count = 0;
    for (int k = 0; k < max_p; ++k) {
        const double v = data[first_valid + k];
        if (isfinite(v)) { sum += v; sumsq += v * v; }
        else             { nan_count += 1; }
    }
    const double inv_n = 1.0 / (double)max_p;

    for (int i = warm_end; i < len; ++i) {
        const double x = data[i];

        if (isnan(x)) {
            o[i] = NEO_F64_NAN;
        } else {
            double m, dv;
            if (nan_count == 0) {
                m = sum * inv_n;
                const double var = (sumsq * inv_n) - m * m;
                dv = (var < 0.0) ? 0.0 : sqrt(var);       /* the CPU's `if`, not fmax */
            } else {
                m = NEO_F64_NAN;
                dv = NEO_F64_NAN;
            }

            const int prev_p = (last_p == 0) ? max_pi : last_p;
            int next_p = prev_p;
            if (isfinite(m) && isfinite(dv)) {
                const double d175 = dv * 1.75;
                const double d025 = dv * 0.25;
                const double a = m - d175;
                const double b = m - d025;
                const double c = m + d025;
                const double d = m + d175;
                const int inc_fast = ((x < a) ? 1 : 0) | ((x > d) ? 1 : 0);
                const int inc_slow = ((x >= b) ? 1 : 0) & ((x <= c) ? 1 : 0);
                const int delta = inc_slow - inc_fast;
                const int p_tmp = prev_p + delta;
                next_p = (p_tmp < min_pi) ? min_pi : ((p_tmp > max_pi) ? max_pi : p_tmp);
            }

            const double sc = 2.0 / ((double)next_p + 1.0);
            last_val = vlma_neo_ema_update(last_val, x, sc);
            last_p = next_p;
            o[i] = last_val;
        }

        /* Roll AFTER the emit, for the next bar -- :555-572. */
        const int next = i + 1;
        if (next < len) {
            const int out_idx = next - max_p;
            const double v_out = data[out_idx];
            if (isfinite(v_out)) { sum -= v_out; sumsq -= v_out * v_out; }
            else if (nan_count > 0) { nan_count -= 1; }   /* saturating_sub */
            const double v_in = data[next];
            if (isfinite(v_in)) { sum += v_in; sumsq += v_in * v_in; }
            else                { nan_count += 1; }
        }
    }
}
