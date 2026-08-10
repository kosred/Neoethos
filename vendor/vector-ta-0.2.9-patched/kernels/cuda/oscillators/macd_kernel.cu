#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {

    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }


__device__ __forceinline__ void warp_inclusive_scan_affine(float &A, float &B, unsigned lane, unsigned mask) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float A_prev = __shfl_up_sync(mask, A, offset);
        const float B_prev = __shfl_up_sync(mask, B, offset);
        if (lane >= static_cast<unsigned>(offset)) {
            const float A_cur = A;
            const float B_cur = B;
            A = A_cur * A_prev;
            B = __fmaf_rn(A_cur, B_prev, B_cur);
        }
    }
}


extern "C" __global__
void macd_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ fasts,
                    const int*   __restrict__ slows,
                    const int*   __restrict__ signals,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ macd_out,
                    float* __restrict__ signal_out,
                    float* __restrict__ hist_out) {
    if (series_len <= 0) return;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int combo = blockIdx.x * warps_per_block + warp_id;
    if (combo >= n_combos) return;
    const unsigned mask = 0xffffffffu;

    const int fast   = fasts[combo];
    const int slow   = slows[combo];
    const int signal = signals[combo];
    if (fast <= 0 || slow <= 0 || signal <= 0) return;

    int fv = first_valid;
    if (fv >= series_len) return;
    fv = imax(fv, 0);

    const size_t row_base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);
    const int macd_warmup   = fv + slow - 1;
    const int signal_warmup = fv + slow + signal - 2;


    const float nanv = NAN;
    const int macd_nan_end   = imin(macd_warmup, series_len);
    const int signal_nan_end = imin(signal_warmup, series_len);
    for (int i = lane; i < macd_nan_end; i += 32) {
        macd_out[row_base + static_cast<size_t>(i)] = nanv;
    }
    for (int i = lane; i < signal_nan_end; i += 32) {
        const size_t idx = row_base + static_cast<size_t>(i);
        signal_out[idx] = nanv;
        hist_out[idx]   = nanv;
    }

    if (macd_warmup >= series_len) return;


    float fast_prev = 0.0f;
    float slow_prev = 0.0f;
    float se_prev   = 0.0f;
    int   have_seed = 0;

    const float af    = 2.0f / (static_cast<float>(fast)   + 1.0f);
    const float aslow = 2.0f / (static_cast<float>(slow)   + 1.0f);
    const float asig  = 2.0f / (static_cast<float>(signal) + 1.0f);
    const float bf    = 1.0f - af;
    const float bslow = 1.0f - aslow;
    const float bsig  = 1.0f - asig;

    if (lane == 0) {

        float fsum = 0.0f, fc = 0.0f;
        const int fcap = imin(fast, series_len - fv);
        for (int i = 0; i < fcap; ++i) {
            kahan_add(prices[fv + i], fsum, fc);
        }
        float fast_ema = fsum / static_cast<float>(fast);

        float ssum = 0.0f, sc = 0.0f;
        const int scap = imin(slow, series_len - fv);
        for (int i = 0; i < scap; ++i) {
            kahan_add(prices[fv + i], ssum, sc);
        }
        float slow_ema = ssum / static_cast<float>(slow);


        const int mwu = imin(macd_warmup, series_len - 1);
        for (int t = fv + fast; t <= mwu; ++t) {
            const float x = prices[t];
            if (isfinite(x)) {

                fast_ema = fmaf(x - fast_ema, af, fast_ema);
            }
        }


        float m0 = fast_ema - slow_ema;
        macd_out[row_base + static_cast<size_t>(macd_warmup)] = m0;


        if (signal_warmup < series_len) {
            if (signal == 1) {
                se_prev = m0;
                signal_out[row_base + static_cast<size_t>(signal_warmup)] = se_prev;
                hist_out[row_base + static_cast<size_t>(signal_warmup)]   = 0.0f;
                have_seed = 1;
            } else {
                float sig_acc = m0;
                float sig_c = 0.0f;


                for (int k = macd_warmup + 1; k <= signal_warmup; ++k) {
                    const float x = prices[k];
                    if (isfinite(x)) {
                        fast_ema = fmaf(x - fast_ema, af,    fast_ema);
                        slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
                    }
                    const float m = fast_ema - slow_ema;
                    macd_out[row_base + static_cast<size_t>(k)] = m;
                    kahan_add(m, sig_acc, sig_c);
                }

                se_prev = sig_acc / static_cast<float>(signal);
                signal_out[row_base + static_cast<size_t>(signal_warmup)] = se_prev;
                const float m_seed = macd_out[row_base + static_cast<size_t>(signal_warmup)];
                hist_out[row_base + static_cast<size_t>(signal_warmup)] = m_seed - se_prev;
                have_seed = 1;
            }
        } else {
            have_seed = 0;
        }

        fast_prev = fast_ema;
        slow_prev = slow_ema;
    }


    fast_prev = __shfl_sync(mask, fast_prev, 0);
    slow_prev = __shfl_sync(mask, slow_prev, 0);
    se_prev   = __shfl_sync(mask, se_prev,   0);
    have_seed = __shfl_sync(mask, have_seed, 0);

    int t0 = have_seed ? (signal_warmup + 1) : (macd_warmup + 1);


    for (; t0 < series_len; t0 += 32) {
        const int t = t0 + lane;


        float Af = 1.0f;
        float Bf = 0.0f;

        float As = 1.0f;
        float Bs = 0.0f;

        float x = 0.0f;
        int x_finite = 0;
        if (t < series_len) {
            x = prices[t];
            x_finite = isfinite(x) ? 1 : 0;
        }
        if (x_finite) {
            Af = bf;    Bf = af * x;
            As = bslow; Bs = aslow * x;
        }

        warp_inclusive_scan_affine(Af, Bf, lane, mask);
        warp_inclusive_scan_affine(As, Bs, lane, mask);

        const float fast_y = __fmaf_rn(Af, fast_prev, Bf);
        const float slow_y = __fmaf_rn(As, slow_prev, Bs);

        float m = 0.0f;
        if (t < series_len) {
            m = fast_y - slow_y;
            macd_out[row_base + static_cast<size_t>(t)] = m;
        }

        float sig_y = nanv;
        if (have_seed) {
            float Ase = 1.0f;
            float Bse = 0.0f;
            if (t < series_len) {
                Ase = bsig;
                Bse = asig * m;
            }
            warp_inclusive_scan_affine(Ase, Bse, lane, mask);
            sig_y = __fmaf_rn(Ase, se_prev, Bse);

            if (t < series_len) {
                const size_t idx = row_base + static_cast<size_t>(t);
                signal_out[idx] = sig_y;
                hist_out[idx]   = m - sig_y;
            }
        }

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        fast_prev = __shfl_sync(mask, fast_y, last_lane);
        slow_prev = __shfl_sync(mask, slow_y, last_lane);
        if (have_seed) {
            se_prev = __shfl_sync(mask, sig_y, last_lane);
        }
    }
}


extern "C" __global__
void macd_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int cols,
                                    int rows,
                                    int fast,
                                    int slow,
                                    int signal,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ macd_tm,
                                    float* __restrict__ signal_tm,
                                    float* __restrict__ hist_tm) {
    if (rows <= 0 || cols <= 0 || fast <= 0 || slow <= 0 || signal <= 0) return;

    const int stride = cols;

    for (int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
         series_idx < cols;
         series_idx += blockDim.x * gridDim.x) {

        int fv = first_valids[series_idx];
        if (fv < 0) fv = 0;
        if (fv >= rows) continue;

        const int macd_warmup   = fv + slow - 1;
        const int signal_warmup = fv + slow + signal - 2;


        const int macd_nan_end   = imin(macd_warmup, rows);
        const int signal_nan_end = imin(signal_warmup, rows);
        for (int t = 0; t < macd_nan_end; ++t) {
            macd_tm[t * stride + series_idx] = NAN;
        }
        for (int t = 0; t < signal_nan_end; ++t) {
            signal_tm[t * stride + series_idx] = NAN;
            hist_tm  [t * stride + series_idx] = NAN;
        }

        if (macd_warmup >= rows) continue;


        float fsum = 0.f, fc = 0.f;
        const int fcap = imin(fast, rows - fv);
        for (int i = 0; i < fcap; ++i) {
            kahan_add(prices_tm[(fv + i) * stride + series_idx], fsum, fc);
        }
        float fast_ema = fsum / (float)fast;

        float ssum = 0.f, sc = 0.f;
        const int scap = imin(slow, rows - fv);
        for (int i = 0; i < scap; ++i) {
            kahan_add(prices_tm[(fv + i) * stride + series_idx], ssum, sc);
        }
        float slow_ema = ssum / (float)slow;

        const float af    = 2.0f / (fast   + 1.0f);
        const float aslow = 2.0f / (slow   + 1.0f);
        const float asig  = 2.0f / (signal + 1.0f);


        const int mwu = imin(macd_warmup, rows - 1);
        for (int t = fv + fast; t <= mwu; ++t) {
            const float x = prices_tm[t * stride + series_idx];
            if (isfinite(x)) {
                fast_ema = fmaf(x - fast_ema, af, fast_ema);
            }
        }


        macd_tm[macd_warmup * stride + series_idx] = fast_ema - slow_ema;


        bool  have_seed = false;
        float se        = 0.0f;

        float sig_acc = (signal > 1) ? macd_tm[macd_warmup * stride + series_idx] : 0.0f;
        float sig_c   = 0.0f;

        if (signal == 1 && signal_warmup < rows) {
            se = macd_tm[signal_warmup * stride + series_idx];
            have_seed = true;
            signal_tm[signal_warmup * stride + series_idx] = se;
            hist_tm  [signal_warmup * stride + series_idx] = macd_tm[signal_warmup * stride + series_idx] - se;
        }


        for (int t = macd_warmup + 1; t < rows; ++t) {
            const float x = prices_tm[t * stride + series_idx];
            if (isfinite(x)) {
                fast_ema = fmaf(x - fast_ema, af,    fast_ema);
                slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
            }
            const float m = fast_ema - slow_ema;
            macd_tm[t * stride + series_idx] = m;

            if (!have_seed) {
                if (signal > 1 && t <= signal_warmup) {
                    kahan_add(m, sig_acc, sig_c);
                    if (t == signal_warmup) {
                        se = sig_acc / (float)signal;
                        have_seed = true;
                        signal_tm[t * stride + series_idx] = se;
                        hist_tm  [t * stride + series_idx] = m - se;
                    }
                }
            } else {
                se = fmaf(m - se, asig, se);
                if (t >= signal_warmup) {
                    signal_tm[t * stride + series_idx] = se;
                    hist_tm  [t * stride + series_idx] = m - se;
                }
            }
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — macd
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/macd.rs
 *   `macd_prepare`                      (:576) — first_valid, both warmups,
 *                                                every Err branch
 *   `macd_with_kernel`                  (:975) — ma_type "ema" takes the
 *                                                classic path
 *   `macd_compute_into_classic_ema`     (:642) — dispatches on fast <= slow
 *   `macd_compute_into_classic_ema_fast`(:761) — the path the defaults take
 *
 * WHICH SERIES THIS EMITS. `compute_macd_batch` (cpu_batch.rs:5463) maps
 * output_id "value" -> `out.macd`. The lane carries one matrix, so this is the
 * MACD line, not the signal and not the histogram.
 *
 * PERIOD-INVARIANT, AND THAT IS FAITHFUL. `compute_macd_batch` reads
 * `fast_period` (12), `slow_period` (26), `signal_period` (9) and `ma_type`
 * ("ema") — cpu_batch.rs:5444-5447. It NEVER reads `period`. A caller sweeping
 * `[7,21,50,100,200]` therefore gets five identical CPU columns, so this
 * kernel emits five identical rows and `F64Kernel::is_period_invariant`
 * reports it. This is the same situation as `tsi`, declared the same way,
 * rather than inventing a mapping from the swept int onto one of the three
 * periods — which would compute something the CPU never computes.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. THREE COUPLED EMA RECURRENCES OVER THE WHOLE SERIES. macd is the
 *     difference of two exponential averages, and the difference of two nearly
 *     equal f32 numbers keeps only the digits they DISAGREE on. On a 1.1-level
 *     FX close, fast_ema and slow_ema agree to ~5 digits, so the f32 macd line
 *     carries ~2 significant digits. The signal EMA then smooths that noise
 *     and the histogram differences it again.
 *  2. `__fmaf_rn` x4 -> `fma`. The CPU uses `f64::mul_add` on the EMA step
 *     (`x.mul_add(af, omf * fast_ema)`): ONE rounding on the fused pair and a
 *     separate rounding on `omf * ema`. Two roundings per step, reproduced
 *     exactly — not `af*x + omf*ema` (three) and not `fma(af, x, fma(omf, ema, 0))`.
 *  3. `__int_as_float` NaN patterns -> `__longlong_as_double(0x7ff8...)`.
 *
 * THE SEEDS ARE SIMPLE MEANS, NOT EMAs, AND THEIR WINDOWS DIFFER. `fast_ema`
 * seeds as the mean of `data[first .. first+fast]` and is then stepped forward
 * to `macd_warmup`; `slow_ema` seeds as the mean of `data[first .. first+slow]`
 * and is NOT stepped. Getting that asymmetry wrong shifts the whole line.
 *
 * ONE THREAD PER COLUMN, ascending. Three carried scalars.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_MACD_FAST   12
#define NEO_MACD_SLOW   26
#define NEO_MACD_SIGNAL 9

extern "C" __global__
void macd_neo_batch_f64(const double* __restrict__ data,
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
    // `periods[combo]` is deliberately unread — see PERIOD-INVARIANT above.
    (void)periods;

    const int fast   = NEO_MACD_FAST;
    const int slow   = NEO_MACD_SLOW;
    const int signal = NEO_MACD_SIGNAL;

    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        fast > len || slow > len || signal > len ||
        (len - first_valid) < slow) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int macd_warmup = first_valid + slow - 1;
    for (int i = 0; i < len && i < macd_warmup; ++i) o[i] = NEO_F64_NAN;
    if (macd_warmup >= len) return;

    const double af    = 2.0 / ((double)fast + 1.0);
    const double aslow = 2.0 / ((double)slow + 1.0);
    const double omf   = 1.0 - af;
    const double oms   = 1.0 - aslow;

    double fsum = 0.0;
    for (int k = 0; k < fast; ++k) fsum += data[first_valid + k];
    double ssum = 0.0;
    for (int k = 0; k < slow; ++k) ssum += data[first_valid + k];

    double fast_ema = fsum / (double)fast;
    double slow_ema = ssum / (double)slow;

    // macd.rs:804-809 — the FAST ema alone is stepped up to macd_warmup.
    for (int t = first_valid + fast; t <= macd_warmup; ++t) {
        fast_ema = fma(data[t], af, omf * fast_ema);
    }

    double m = fast_ema - slow_ema;
    o[macd_warmup] = m;

    for (int i = macd_warmup + 1; i < len; ++i) {
        const double x = data[i];
        fast_ema = fma(x, af,    omf * fast_ema);
        slow_ema = fma(x, aslow, oms * slow_ema);
        m = fast_ema - slow_ema;
        o[i] = m;
    }
}
