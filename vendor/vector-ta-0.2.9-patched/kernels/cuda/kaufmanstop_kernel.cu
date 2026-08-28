#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

__global__ void kaufmanstop_build_range_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    float* __restrict__ out_range
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        const float h = high[i];
        const float l = low[i];
        out_range[i] = (isnan(h) || isnan(l)) ? CUDART_NAN_F : (h - l);
    }
}


__global__ void kaufmanstop_axpy_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_row,
    int len,
    float signed_mult,
    int warm,
    int base_is_low,
    float* __restrict__ out_row
) {
    const float* __restrict__ base = base_is_low ? low : high;


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        float out;
        if (i < warm) {
            out = CUDART_NAN_F;
        } else {

            out = fmaf(ma_row[i], signed_mult, base[i]);
        }
        out_row[i] = out;
    }
}


__global__ void kaufmanstop_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ ma_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    float signed_mult,
    int base_is_low,
    int period,
    float* __restrict__ out_tm
){
    const float* __restrict__ base_tm = base_is_low ? low_tm : high_tm;


    if (gridDim.y == 1 && blockDim.y == 1) {

        const int total = rows * cols;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
            const int s = i % cols;
            const int t = i / cols;
            const int warm = first_valids[s] + period - 1;
            float out;
            if (t < warm) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_tm[i], signed_mult, base_tm[i]);
            }
            out_tm[i] = out;
        }
    } else {

        int s = blockIdx.x * blockDim.x + threadIdx.x;
        int t0 = blockIdx.y * blockDim.y + threadIdx.y;
        int t_stride = blockDim.y * gridDim.y;

        if (s >= cols) return;
        const int warm = first_valids[s] + period - 1;

        for (int t = t0; t < rows; t += t_stride) {
            const int idx = t * cols + s;
            float out;
            if (t < warm) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_tm[idx], signed_mult, base_tm[idx]);
            }
            out_tm[idx] = out;
        }
    }
}


__global__ void kaufmanstop_one_series_many_params_time_major_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_pm,
    const int*   __restrict__ warm_ps,
    const float* __restrict__ signed_mults,
    int rows,
    int params,
    int base_is_low,
    float* __restrict__ out_pm
) {
    extern __shared__ float s_base[];
    const float* __restrict__ base = base_is_low ? low : high;


    int p  = blockIdx.y * blockDim.y + threadIdx.y;
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t_stride = blockDim.x * gridDim.x;

    for (int t = t0; t < rows; t += t_stride) {

        if (threadIdx.y == 0) {
            s_base[threadIdx.x] = base[t];
        }
        __syncthreads();

        if (p < params) {
            const int idx = p * rows + t;
            float out;
            if (t < warm_ps[p]) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_pm[idx], signed_mults[p], s_base[threadIdx.x]);
            }
            out_pm[idx] = out;
        }
        __syncthreads();
    }
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

extern "C" {

__global__ void kaufmanstop_build_range_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    double* __restrict__ out_range
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        const double h = high[i];
        const double l = low[i];
        out_range[i] = (isnan(h) || isnan(l)) ? CUDART_NAN : (h - l);
    }
}


__global__ void kaufmanstop_axpy_row_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ ma_row,
    int len,
    double signed_mult,
    int warm,
    int base_is_low,
    double* __restrict__ out_row
) {
    const double* __restrict__ base = base_is_low ? low : high;


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        double out;
        if (i < warm) {
            out = CUDART_NAN;
        } else {

            out = fma(ma_row[i], signed_mult, base[i]);
        }
        out_row[i] = out;
    }
}


__global__ void kaufmanstop_many_series_one_param_time_major_f64(
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    const double* __restrict__ ma_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    double signed_mult,
    int base_is_low,
    int period,
    double* __restrict__ out_tm
){
    const double* __restrict__ base_tm = base_is_low ? low_tm : high_tm;


    if (gridDim.y == 1 && blockDim.y == 1) {

        const int total = rows * cols;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
            const int s = i % cols;
            const int t = i / cols;
            const int warm = first_valids[s] + period - 1;
            double out;
            if (t < warm) {
                out = CUDART_NAN;
            } else {
                out = fma(ma_tm[i], signed_mult, base_tm[i]);
            }
            out_tm[i] = out;
        }
    } else {

        int s = blockIdx.x * blockDim.x + threadIdx.x;
        int t0 = blockIdx.y * blockDim.y + threadIdx.y;
        int t_stride = blockDim.y * gridDim.y;

        if (s >= cols) return;
        const int warm = first_valids[s] + period - 1;

        for (int t = t0; t < rows; t += t_stride) {
            const int idx = t * cols + s;
            double out;
            if (t < warm) {
                out = CUDART_NAN;
            } else {
                out = fma(ma_tm[idx], signed_mult, base_tm[idx]);
            }
            out_tm[idx] = out;
        }
    }
}


__global__ void kaufmanstop_one_series_many_params_time_major_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ ma_pm,
    const int*   __restrict__ warm_ps,
    const double* __restrict__ signed_mults,
    int rows,
    int params,
    int base_is_low,
    double* __restrict__ out_pm
) {
    extern __shared__ double s_base_f64[];
    const double* __restrict__ base = base_is_low ? low : high;


    int p  = blockIdx.y * blockDim.y + threadIdx.y;
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t_stride = blockDim.x * gridDim.x;

    for (int t = t0; t < rows; t += t_stride) {

        if (threadIdx.y == 0) {
            s_base_f64[threadIdx.x] = base[t];
        }
        __syncthreads();

        if (p < params) {
            const int idx = p * rows + t;
            double out;
            if (t < warm_ps[p]) {
                out = CUDART_NAN;
            } else {
                out = fma(ma_pm[idx], signed_mults[p], s_base_f64[threadIdx.x]);
            }
            out_pm[idx] = out;
        }
        __syncthreads();
    }
}

}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                               kaufmanstop
 * ---------------------------------------------------------------------------
 * CPU reference: `kaufmanstop_scalar_classic_sma`,
 * src/indicators/kaufmanstop.rs:2093, reached from
 * `kaufmanstop_compute_prepared_into` (:345) for the default `ma_type = "sma"`
 * (cpu_batch.rs:15181), `direction = "long"` (:15180) and `mult = 2.0`
 * (:15179). `period` IS the swept parameter (:15178, default 22).
 *
 * WHY THE NaN-AWARE FORM AND NOT THE FAST ONE. The crate has two: a fast path
 * (:2160) that keeps a plain running sum, and this one, which keeps a running
 * sum AND a running count of valid bars. The fast path is not a different
 * answer -- the moment it meets a NaN it THROWS AWAY its work and re-runs this
 * function from the top (:2179-2188, :2208-2216). On NaN-free data the two are
 * bit-identical: the seed loop visits the same bars in the same ascending order
 * (:2107-2113 vs :2175-2191), `valid_count` never leaves `period`, and
 * `sum / valid_count as f64` is `sum / period_f`. So this single form is the
 * exact CPU answer in BOTH cases, which is why the kernel carries one loop
 * rather than two.
 *
 * SEQUENTIAL, one thread per column. The sliding sum is updated as
 * `sum -= old_range; sum += new_range` (:2133, :2139) -- two separate roundings
 * in that order, carried across every bar. A fresh window sum per bar would be
 * a different number, and a fused `sum += (new - old)` would be a third.
 * Reproduced literally, statement for statement.
 *
 * NO PER-THREAD RING, so no `max_period` bound: the leaving bar is read
 * straight out of the resident high/low series at `i - period`, exactly as the
 * CPU reads it. NEVER-OOM by construction.
 *
 * NaN: `high[i] - sma * mult` is written even when `high[i]` is NaN (:2150),
 * and `sma` becomes NaN when the window empties (:2146). Both reproduced.
 *
 * `first_valid` is the lane's `AllInputsNonNan` over (high, low), which is the
 * CPU's `position(|(&h, &l)| !h.is_nan() && !l.is_nan())` (:307-311).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:15179 -- `mult` default. */
#define NEO_KAUFMANSTOP_MULT 2.0

extern "C" __global__
void kaufmanstop_neo_batch_f64(const double* __restrict__ high,
                               const double* __restrict__ low,
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
    if (first_valid < 0 || first_valid >= n) return;
    /* kaufmanstop.rs:300-305 -- period 0 or longer than the data is an error. */
    if (period <= 0 || period > n) return;
    /* kaufmanstop.rs:313-318 -- NotEnoughValidData leaves the row NaN. */
    if (n - first_valid < period) return;

    const double mult      = NEO_KAUFMANSTOP_MULT;
    const int    start_idx = first_valid + period - 1;   /* :2102 */

    /* :2105-2113 -- seed sum over the first `period` bars, ascending, skipping
     * bars where either series is NaN. */
    double sum = 0.0;
    int    valid_count = 0;
    for (int k = 0; k < period; ++k) {
        const int idx = first_valid + k;
        if (!isnan(high[idx]) && !isnan(low[idx])) {
            sum += high[idx] - low[idx];
            valid_count += 1;
        }
    }

    /* :2115-2117 -- an entirely-NaN seed window is an error on the CPU. */
    if (valid_count == 0) return;

    double sma = sum / (double)valid_count;

    /* direction == "long" (cpu_batch.rs:15180), so :2122. */
    o[start_idx] = low[start_idx] - sma * mult;

    for (int i = start_idx + 1; i < n; ++i) {
        const int old_idx = i - period;

        if (!isnan(high[old_idx]) && !isnan(low[old_idx])) {
            sum -= high[old_idx] - low[old_idx];      /* :2132-2134 */
            valid_count -= 1;
        }
        if (!isnan(high[i]) && !isnan(low[i])) {
            sum += high[i] - low[i];                  /* :2138-2140 */
            valid_count += 1;
        }

        sma = (valid_count > 0) ? (sum / (double)valid_count)   /* :2143-2147 */
                                : NEO_F64_NAN;

        o[i] = low[i] - sma * mult;                   /* :2150 */
    }
}
