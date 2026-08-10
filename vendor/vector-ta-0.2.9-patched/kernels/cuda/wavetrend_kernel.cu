#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


namespace {

__device__ __forceinline__ bool is_finite_f(float x) {
    return !(isnan(x) || isinf(x));
}


__device__ __forceinline__ float ema_update(float state, float x, float alpha) {
    return fmaf(alpha, x - state, state);
}


struct KahanSumF {
    float s;
    float c;
    __device__ KahanSumF() : s(0.0f), c(0.0f) {}
    __device__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ void sub(float x) { add(-x); }
};

}

extern "C" __global__ void wavetrend_batch_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ channel_lengths,
    const int* __restrict__ average_lengths,
    const int* __restrict__ ma_lengths,
    const float* __restrict__ factors,
    float* __restrict__ wt1_out,
    float* __restrict__ wt2_out,
    float* __restrict__ wt_diff_out
){
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride  = blockDim.x * gridDim.x;

    for (int row = tid; row < n_combos; row += stride) {
        const int ch  = channel_lengths[row];
        const int avg = average_lengths[row];
        const int ma  = ma_lengths[row];
        const float factor = factors[row];

        float* __restrict__ wt1_row  = wt1_out     + (size_t)row * (size_t)len;
        float* __restrict__ wt2_row  = wt2_out     + (size_t)row * (size_t)len;
        float* __restrict__ diff_row = wt_diff_out + (size_t)row * (size_t)len;


        if (len <= 0 || ch <= 0 || avg <= 0 || ma <= 0) {
            for (int i = 0; i < len; ++i) {
                wt1_row[i]  = CUDART_NAN_F;
                wt2_row[i]  = CUDART_NAN_F;
                diff_row[i] = CUDART_NAN_F;
            }
            continue;
        }

        const float alpha_ch  = 2.0f / (float(ch) + 1.0f);
        const float alpha_avg = 2.0f / (float(avg) + 1.0f);
        const float inv_ma    = 1.0f / (float)ma;


        int warmup = first_valid + (ch - 1) + (avg - 1) + (ma - 1);
        if (warmup < 0)       warmup = 0;
        if (warmup > len)     warmup = len;


        int prefill = first_valid;
        if (prefill < 0) prefill = 0;
        if (prefill > len) prefill = len;
        for (int i = 0; i < prefill; ++i) {
            wt1_row[i]  = CUDART_NAN_F;
            wt2_row[i]  = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }


        bool esa_init = false, de_init = false, wt1_init = false;
        float esa = 0.0f, de = 0.0f, wt1_state = 0.0f;


        KahanSumF acc;
        int window_count = 0;

        int start = first_valid > 0 ? first_valid : 0;
        for (int i = start; i < len; ++i) {
            const float price = prices[i];
            const bool price_ok = is_finite_f(price);


            if (!esa_init) {
                if (price_ok) {
                    esa = price;
                    esa_init = true;
                }
            } else if (price_ok) {
                esa = ema_update(esa, price, alpha_ch);
            }


            if (esa_init && price_ok) {
                const float absdiff = fabsf(price - esa);
                if (!de_init) {
                    de = absdiff;
                    de_init = true;
                } else {
                    de = ema_update(de, absdiff, alpha_ch);
                }
            }


            float wt1_val = CUDART_NAN_F;
            if (esa_init && de_init && price_ok) {
                const float denom = factor * de;
                if (denom != 0.0f && is_finite_f(denom)) {
                    const float ci = (price - esa) / denom;
                    if (!wt1_init) {
                        if (is_finite_f(ci)) {
                            wt1_state = ci;
                            wt1_init = true;
                        }
                    } else if (is_finite_f(ci)) {
                        wt1_state = ema_update(wt1_state, ci, alpha_avg);
                    }
                }
            }
            if (wt1_init) wt1_val = wt1_state;


            wt1_row[i] = wt1_val;


            if (is_finite_f(wt1_val)) { acc.add(wt1_val); ++window_count; }

            if (i >= ma) {
                const float old = wt1_row[i - ma];
                if (is_finite_f(old)) { acc.sub(old); --window_count; }
            }


            float wt2_val = CUDART_NAN_F;
            if (window_count >= ma) {
                wt2_val = acc.s * inv_ma;
            }
            wt2_row[i] = wt2_val;


            if (i >= warmup && is_finite_f(wt2_val) && is_finite_f(wt1_val)) {
                diff_row[i] = wt2_val - wt1_val;
            } else {
                diff_row[i] = CUDART_NAN_F;
            }
        }


        for (int i = 0; i < warmup; ++i) {
            wt1_row[i]  = CUDART_NAN_F;
            wt2_row[i]  = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }
    }
}


extern "C" __global__ void wavetrend_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int channel_length,
    int average_length,
    int ma_length,
    float factor,
    const int* __restrict__ first_valids,
    float* __restrict__ wt1_tm,
    float* __restrict__ wt2_tm,
    float* __restrict__ wt_diff_tm
){
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (rows <= 0 || cols <= 0 || channel_length <= 0 || average_length <= 0 || ma_length <= 0) return;

    const float alpha_ch  = 2.0f / (float(channel_length) + 1.0f);
    const float alpha_avg = 2.0f / (float(average_length) + 1.0f);
    const float inv_ma    = 1.0f / (float)ma_length;

    for (int series = tid; series < cols; series += stride) {
        float* __restrict__ wt1_col  = wt1_tm     + series;
        float* __restrict__ wt2_col  = wt2_tm     + series;
        float* __restrict__ diff_col = wt_diff_tm + series;

        const int first_valid = first_valids[series];
        int warmup = first_valid + (channel_length - 1) + (average_length - 1) + (ma_length - 1);
        if (warmup < 0) warmup = 0;
        if (warmup > rows) warmup = rows;


        int pre = first_valid;
        if (pre < 0) pre = 0;
        if (pre > rows) pre = rows;
        for (int t = 0; t < pre; ++t) {
            const int idx = t * cols;
            wt1_col[idx]  = CUDART_NAN_F;
            wt2_col[idx]  = CUDART_NAN_F;
            diff_col[idx] = CUDART_NAN_F;
        }


        bool esa_init = false, de_init = false, wt1_init = false;
        double esa = 0.0, de = 0.0, wt1_state = 0.0;


        double sum_wt1 = 0.0;
        int window_count = 0;

        int start = first_valid > 0 ? first_valid : 0;
        for (int t = start; t < rows; ++t) {
            const int idx = t * cols;
            const double price = static_cast<double>(prices_tm[idx + series]);
            const bool price_ok = isfinite(price);


            if (!esa_init) {
                if (price_ok) { esa = price; esa_init = true; }
            } else if (price_ok) {
                const double alpha_ch_d = static_cast<double>(alpha_ch);
                const double beta_ch_d  = 1.0 - alpha_ch_d;
                esa = fma(alpha_ch_d, price, beta_ch_d * esa);
            }


            if (esa_init && price_ok) {
                const double absdiff = fabs(price - esa);
                if (!de_init) { de = absdiff; de_init = isfinite(de); }
                else if (isfinite(absdiff)) {
                    const double alpha_ch_d = static_cast<double>(alpha_ch);
                    const double beta_ch_d  = 1.0 - alpha_ch_d;
                    de = fma(alpha_ch_d, absdiff, beta_ch_d * de);
                }
            }


            float wt1_val = CUDART_NAN_F;
            if (esa_init && de_init && price_ok) {
                const double denom = static_cast<double>(factor) * de;
                if (denom != 0.0 && isfinite(denom)) {
                    const double ci = (price - esa) / denom;
                    if (!wt1_init) {
                        if (isfinite(ci)) { wt1_state = ci; wt1_init = true; }
                    } else if (isfinite(ci)) {
                        const double alpha_avg_d = static_cast<double>(alpha_avg);
                        const double beta_avg_d  = 1.0 - alpha_avg_d;
                        wt1_state = fma(alpha_avg_d, ci, beta_avg_d * wt1_state);
                    }
                }
            }
            if (wt1_init) wt1_val = static_cast<float>(wt1_state);
            wt1_col[idx] = wt1_val;


            if (isfinite(static_cast<double>(wt1_val))) { sum_wt1 += wt1_state; ++window_count; }
            if (t >= ma_length) {
                const float old = wt1_col[(t - ma_length) * cols];
                if (isfinite(static_cast<double>(old))) { sum_wt1 -= static_cast<double>(old); --window_count; }
            }

            float wt2_val = CUDART_NAN_F;
            if (window_count >= ma_length) wt2_val = static_cast<float>(sum_wt1 * inv_ma);
            wt2_col[idx] = wt2_val;


            if (t >= warmup && isfinite(static_cast<double>(wt1_val)) && isfinite(static_cast<double>(wt2_val))) {
                diff_col[idx] = wt2_val - wt1_val;
            } else {
                diff_col[idx] = CUDART_NAN_F;
            }
        }


        for (int t = 0; t < rows && t < warmup; ++t) {
            const int idx = t * cols;
            wt1_col[idx]  = CUDART_NAN_F;
            wt2_col[idx]  = CUDART_NAN_F;
            diff_col[idx] = CUDART_NAN_F;
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — wavetrend
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/wavetrend.rs
 *   `wavetrend_with_kernel`     (:269) — first_valid, `needed`, Err branches
 *   `wavetrend_kernel_dispatch` (:336) — warmup = first + ch + avg + ma - 3
 *   `wavetrend_compute_into`    (:668-765) — THE SCALAR BRANCH, which is what
 *                                            this kernel mirrors
 *
 * SOURCE IS hlc3, NOT close. `compute_wavetrend_batch` (cpu_batch.rs:6490)
 * calls `extract_slice_input("wavetrend", req.data, "hlc3")`. Handing this
 * kernel `close` computes a different indicator that passes every shape check,
 * which is why the spec declares `F64InputKind::Hlc3Slice`.
 *
 * WHICH SERIES THIS EMITS. cpu_batch.rs:6492 maps "value" -> `Wt1`.
 *
 * PERIOD-INVARIANT, AND THAT IS FAITHFUL. The batch reads `channel_length` (9),
 * `average_length` (12), `ma_length` (3) and `factor` (0.015) —
 * cpu_batch.rs:6516-6519 — never `period`. Identical CPU columns, identical
 * rows here. `ma_length` fixed at 3 makes the SMA ring a compile-time 3 slots,
 * so no `max_period` is needed.
 *
 * ------------------------------------------------------------------------
 * A CRATE SELF-INCONSISTENCY, NAMED RATHER THAN PAPERED OVER
 * ------------------------------------------------------------------------
 * `wavetrend` has TWO CPU implementations that are not the same function:
 *
 *   * the SCALAR branch (:668-765) seeds each EMA at the FIRST FINITE INPUT
 *     (`esa_state = x`, `de_state = abs_diff`, `wt1_state = ci`) and steps it
 *     with `alpha*x + beta*state` — three roundings;
 *   * the AVX branch (:768-843) calls `wavetrend_core_computation` (:1007),
 *     which runs `ema_compute_into` over the whole slice — a different seed
 *     (an SMA of the first `channel_len` bars) and a different step.
 *
 * `wavetrend_with_kernel:312` resolves `Kernel::Auto` through
 * `detect_best_kernel()`, so on an x86 host with AVX the DEFAULT CPU answer is
 * the second one. This is the same class of defect the crate already records
 * for `vwap` in `cuda_f64::WITHHELD_PENDING_CPU_SELF_CONSISTENCY`, but with a
 * larger mechanism: the two disagree in the SEED, not in the last place, so
 * they differ visibly over the whole warm-up and asymptotically thereafter.
 *
 * THIS KERNEL MIRRORS THE SCALAR BRANCH, for the same reason the Inventory
 * settled `wilders_scalar` / `vwap_scalar` as the oracles: `Kernel::Scalar` is
 * the path with the explicit, readable definition, it is the one
 * `Kernel::ScalarBatch` routes to, and it is host-independent — the AVX answer
 * is not even stable across machines, since `detect_best_kernel` picks AVX2 or
 * AVX-512 by CPU. A device kernel cannot be parity-checked against an answer
 * that changes with the host. The remaining work is to make
 * `wavetrend_core_computation` seed the way the scalar branch does so the
 * crate agrees with itself; that is a CPU-side edit, it is NOT this shard's
 * territory, and it is reported rather than silently absorbed.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. THREE STACKED EMAs AND A DIVISION BY A SMOOTHED ABSOLUTE DEVIATION.
 *     `ci = (x - esa) / (0.015 * de)`: the numerator is a difference of nearly
 *     equal quantities and the denominator is 1.5% of a small number. In f32
 *     the numerator keeps ~2 significant digits and the quotient is noise
 *     multiplied by ~67. This is the worst-conditioned expression in the
 *     shard.
 *  2. `fabsf` x1 -> `fabs`.
 *  3. `__int_as_float(0x7f...)` x21 NaN patterns -> `__longlong_as_double`.
 *  4. `0.015f` is NOT `0.015`. It is the literal that divides the numerator.
 *  5. THE EMA STEPS ARE `alpha*x + beta*state` — THREE roundings — NOT
 *     `fma(alpha, x, beta*state)` and NOT the Wilder single-rounding form. The
 *     reference writes it out at :703, :711 and :722 and it is copied as
 *     written.
 *
 * THE SMA RING COUNTS ONLY FINITE ENTRIES. `sma_count` is incremented only for
 * a finite `wt1_i` and `wt2` is emitted only when the ring is FULL of finite
 * values (:749). A plain `sum / ma_len` would emit a value during the warm-up
 * that the reference leaves NaN.
 *
 * ONE THREAD PER COLUMN. Carried: esa, de, wt1 states plus the 3-slot ring.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_WT_CHANNEL 9
#define NEO_WT_AVERAGE 12
#define NEO_WT_MALEN   3
#define NEO_WT_FACTOR  0.015

extern "C" __global__
void wavetrend_neo_batch_f64(const double* __restrict__ data,
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
    (void)periods;   /* period-invariant — see the header. */

    const int channel_len = NEO_WT_CHANNEL;
    const int average_len = NEO_WT_AVERAGE;
    const int ma_len      = NEO_WT_MALEN;
    const double factor   = NEO_WT_FACTOR;

    int needed = channel_len;
    if (average_len > needed) needed = average_len;
    if (ma_len > needed) needed = ma_len;

    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        channel_len > len || average_len > len || ma_len > len ||
        (len - first_valid) < needed) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warmup = first_valid + channel_len - 1 + average_len - 1 + ma_len - 1;
    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (warmup >= len) return;

    const double alpha_ch  = 2.0 / ((double)channel_len + 1.0);
    const double beta_ch   = 1.0 - alpha_ch;
    const double alpha_avg = 2.0 / ((double)average_len + 1.0);
    const double beta_avg  = 1.0 - alpha_avg;

    double esa_state = NEO_F64_NAN;
    double de_state  = NEO_F64_NAN;
    double wt1_state = NEO_F64_NAN;
    bool esa_seeded = false, de_seeded = false, wt1_seeded = false;

    double ring_vals[NEO_WT_MALEN];
    unsigned char ring_mask[NEO_WT_MALEN];
    for (int k = 0; k < ma_len; ++k) { ring_vals[k] = NEO_F64_NAN; ring_mask[k] = 0; }
    int head = 0;
    double sma_sum = 0.0;
    int sma_count = 0;

    for (int idx = first_valid; idx < len; ++idx) {
        const double x = data[idx];

        double wt1_i = NEO_F64_NAN;

        if (isfinite(x)) {
            if (!esa_seeded) { esa_state = x; esa_seeded = true; }
            else             { esa_state = alpha_ch * x + beta_ch * esa_state; }

            const double abs_diff = fabs(x - esa_state);
            if (!de_seeded) { de_state = abs_diff; de_seeded = true; }
            else            { de_state = alpha_ch * abs_diff + beta_ch * de_state; }

            const double den = factor * de_state;
            if (den != 0.0 && isfinite(den) && isfinite(esa_state)) {
                const double ci = (x - esa_state) / den;
                if (isfinite(ci)) {
                    if (!wt1_seeded) { wt1_state = ci; wt1_seeded = true; }
                    else             { wt1_state = alpha_avg * ci + beta_avg * wt1_state; }
                    wt1_i = wt1_state;
                }
            }
        }

        /* The ring is maintained even though only wt1 is emitted: `head`,
         * `sma_sum` and `sma_count` are pure carried state for wt2, and the
         * loop is kept a line-for-line mirror so the wt2 entry point this file
         * may gain reads the same state machine. */
        if (ring_mask[head] != 0) { sma_sum -= ring_vals[head]; sma_count -= 1; }
        if (isfinite(wt1_i)) {
            ring_vals[head] = wt1_i;
            ring_mask[head] = 1;
            sma_sum += wt1_i;
            sma_count += 1;
        } else {
            ring_vals[head] = NEO_F64_NAN;
            ring_mask[head] = 0;
        }
        head += 1; if (head == ma_len) head = 0;

        if (idx >= warmup) o[idx] = wt1_i;
    }
}
