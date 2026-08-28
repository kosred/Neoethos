#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void vidya_batch_f32(const float* __restrict__ prices,
                     const int*   __restrict__ short_periods,
                     const int*   __restrict__ long_periods,
                     const float* __restrict__ alphas,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int sp = short_periods[combo];
    const int lp = long_periods[combo];
    const float alpha = alphas[combo];
    const int base = combo * series_len;


    bool invalid = (sp < 2) || (lp < sp) || (lp < 2) || (alpha < 0.0f) || (alpha > 1.0f) ||
                   (first_valid < 0) || (first_valid >= series_len) ||
                   (lp > (series_len - first_valid));

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }

    const int warm_end = first_valid + lp;
    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;
    const int warmup_prefix = idx_m2;


    for (int i = threadIdx.x; i < warmup_prefix; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }

    if (threadIdx.x != 0) return;


    double long_sum = 0.0;
    double long_sum2 = 0.0;
    double short_sum = 0.0;
    double short_sum2 = 0.0;

    const int short_head = warm_end - sp;
    for (int i = first_valid; i < short_head; ++i) {
        const double x = static_cast<double>(prices[i]);
        long_sum += x;
        long_sum2 += x * x;
    }
    for (int i = short_head; i < warm_end; ++i) {
        const double x = static_cast<double>(prices[i]);
        long_sum += x;
        long_sum2 += x * x;
        short_sum += x;
        short_sum2 += x * x;
    }


    float val = prices[idx_m2];
    out[base + idx_m2] = val;

    if (idx_m1 < series_len) {
        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - short_mean * short_mean;
        const double long_var  = long_sum2 * long_inv - long_mean * long_mean;
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);

        const float x = prices[idx_m1];
        val = fmaf(x - val, static_cast<float>(k), val);
        out[base + idx_m1] = val;
    }


    for (int t = warm_end; t < series_len; ++t) {
        const double x_new = static_cast<double>(prices[t]);
        const double x_new2 = x_new * x_new;


        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;


        const double x_long_out = static_cast<double>(prices[t - lp]);
        const double x_short_out = static_cast<double>(prices[t - sp]);
        long_sum -= x_long_out;
        long_sum2 -= x_long_out * x_long_out;
        short_sum -= x_short_out;
        short_sum2 -= x_short_out * x_short_out;

        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - short_mean * short_mean;
        const double long_var  = long_sum2 * long_inv - long_mean * long_mean;
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);

        const float x = prices[t];
        val = fmaf(x - val, static_cast<float>(k), val);
        out[base + t] = val;
    }
}


extern "C" __global__ __launch_bounds__(32)
void vidya_batch_prefix_f32(const float* __restrict__ prices,
                            const double* __restrict__ prefix_sum,
                            const double* __restrict__ prefix_sum2,
                            const int*   __restrict__ short_periods,
                            const int*   __restrict__ long_periods,
                            const float* __restrict__ alphas,
                            int series_len,
                            int first_valid,
                            int n_combos,
                            float* __restrict__ out) {
    constexpr int WARP = 32;

    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int sp = short_periods[combo];
    const int lp = long_periods[combo];
    const float alpha = alphas[combo];
    const int base = combo * series_len;

    const bool invalid =
        (sp < 2) || (lp < sp) || (lp < 2) || (alpha < 0.0f) || (alpha > 1.0f) ||
        (first_valid < 0) || (first_valid >= series_len) ||
        (lp > (series_len - first_valid));

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }

    const int warm_end = first_valid + lp;
    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;
    const int warmup_prefix = idx_m2;

    for (int i = threadIdx.x; i < warmup_prefix; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }

    if (threadIdx.x == 0) {
        out[base + idx_m2] = prices[idx_m2];
    }

    const int lane = threadIdx.x;
    if (lane >= WARP) return;

    float prev = prices[idx_m2];
    const double sp_inv = 1.0 / static_cast<double>(sp);
    const double lp_inv = 1.0 / static_cast<double>(lp);

    int chunk_start = idx_m1;
    for (; (chunk_start + (WARP - 1)) < series_len; chunk_start += WARP) {
        const int t = chunk_start + lane;
        const int tp1 = t + 1;

        const double long_sum = prefix_sum[tp1] - prefix_sum[tp1 - lp];
        const double long_sum2 = prefix_sum2[tp1] - prefix_sum2[tp1 - lp];
        const double short_sum = prefix_sum[tp1] - prefix_sum[tp1 - sp];
        const double short_sum2 = prefix_sum2[tp1] - prefix_sum2[tp1 - sp];

        const double short_mean = short_sum * sp_inv;
        const double long_mean  = long_sum * lp_inv;
        double short_var = fma(-short_mean, short_mean, short_sum2 * sp_inv);
        double long_var  = fma(-long_mean,  long_mean,  long_sum2  * lp_inv);
        short_var = fmax(0.0, short_var);
        long_var  = fmax(0.0, long_var);

        float k = 0.0f;
        if (long_var > 0.0 && short_var > 0.0) {
            const float ratio = static_cast<float>(short_var / long_var);
            k = alpha * sqrtf(ratio);
        }

        float a = 1.0f - k;
        float b = k * prices[t];

        const unsigned m = 0xFFFFFFFFu;
        #pragma unroll
        for (int off = 1; off < WARP; off <<= 1) {
            const float a_up = __shfl_up_sync(m, a, off);
            const float b_up = __shfl_up_sync(m, b, off);
            if (lane >= off) {
                b = fmaf(a, b_up, b);
                a = a * a_up;
            }
        }

        const float x = fmaf(a, prev, b);
        out[base + t] = x;

        prev = __shfl_sync(m, x, WARP - 1);
    }

    if (lane == 0) {
        float val = prev;
        for (int t = chunk_start; t < series_len; ++t) {
            const int tp1 = t + 1;
            const double long_sum = prefix_sum[tp1] - prefix_sum[tp1 - lp];
            const double long_sum2 = prefix_sum2[tp1] - prefix_sum2[tp1 - lp];
            const double short_sum = prefix_sum[tp1] - prefix_sum[tp1 - sp];
            const double short_sum2 = prefix_sum2[tp1] - prefix_sum2[tp1 - sp];

            const double short_mean = short_sum * sp_inv;
            const double long_mean  = long_sum * lp_inv;
            double short_var = fma(-short_mean, short_mean, short_sum2 * sp_inv);
            double long_var  = fma(-long_mean,  long_mean,  long_sum2  * lp_inv);
            short_var = fmax(0.0, short_var);
            long_var  = fmax(0.0, long_var);

            float k = 0.0f;
            if (long_var > 0.0 && short_var > 0.0) {
                const float ratio = static_cast<float>(short_var / long_var);
                k = alpha * sqrtf(ratio);
            }

            const float x = prices[t];
            val = fmaf(x - val, k, val);
            out[base + t] = val;
        }
    }
}

extern "C" __global__
void vidya_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
                                     int short_period,
                                     int long_period,
                                     float alpha,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) return;

    const int sp = short_period;
    const int lp = long_period;
    int first_valid = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const bool invalid = (sp < 2) || (lp < sp) || (lp < 2) || (alpha < 0.0f) || (alpha > 1.0f) ||
                         (lp > (series_len - first_valid));
    const int stride = num_series;

    if (invalid) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + series_idx] = CUDART_NAN_F;
        }
        return;
    }

    const int warm_end = first_valid + lp;
    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;


    for (int t = threadIdx.x; t < idx_m2; t += blockDim.x) {
        out_tm[t * stride + series_idx] = CUDART_NAN_F;
    }

    if (threadIdx.x != 0) return;


    double long_sum = 0.0;
    double long_sum2 = 0.0;
    double short_sum = 0.0;
    double short_sum2 = 0.0;
    const int short_head = warm_end - sp;
    for (int i = first_valid; i < short_head; ++i) {
        const double x = static_cast<double>(prices_tm[i * stride + series_idx]);
        long_sum += x;
        long_sum2 += x * x;
    }
    for (int i = short_head; i < warm_end; ++i) {
        const double x = static_cast<double>(prices_tm[i * stride + series_idx]);
        long_sum += x;
        long_sum2 += x * x;
        short_sum += x;
        short_sum2 += x * x;
    }

    float val = prices_tm[idx_m2 * stride + series_idx];
    out_tm[idx_m2 * stride + series_idx] = val;

    if (idx_m1 < series_len) {
        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - (short_mean * short_mean);
        const double long_var  = long_sum2 * long_inv - (long_mean * long_mean);
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);
        const float x = prices_tm[idx_m1 * stride + series_idx];
        val = fmaf(x - val, static_cast<float>(k), val);
        out_tm[idx_m1 * stride + series_idx] = val;
    }

    for (int t = warm_end; t < series_len; ++t) {
        const double x_new = static_cast<double>(prices_tm[t * stride + series_idx]);
        const double x_new2 = x_new * x_new;
        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;
        const double x_long_out = static_cast<double>(prices_tm[(t - lp) * stride + series_idx]);
        const double x_short_out = static_cast<double>(prices_tm[(t - sp) * stride + series_idx]);
        long_sum -= x_long_out;
        long_sum2 -= x_long_out * x_long_out;
        short_sum -= x_short_out;
        short_sum2 -= x_short_out * x_short_out;

        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - short_mean * short_mean;
        const double long_var  = long_sum2 * long_inv - long_mean * long_mean;
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);
        const float x = prices_tm[t * stride + series_idx];
        val = fmaf(x - val, static_cast<float>(k), val);
        out_tm[t * stride + series_idx] = val;
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/vidya.rs
//   * vidya_with_kernel (:366) — warmup prefix = first + long_period - 2.
//   * vidya_scalar (:494) — the arithmetic reproduced below.
//
// PERIOD-INVARIANT. compute_vidya_batch (cpu_batch.rs:15699) reads
// short_period (default 2), long_period (default 5) and alpha (default 0.2),
// and NEVER reads period. A period sweep therefore emits identical CPU rows and
// must emit identical rows here.
//
// NaN SEMANTICS. The CPU writes
//     let mut k = short_std / long_std;
//     if k.is_nan() { k = 0.0; }
// A comparison chain would NOT catch this: every comparison against NaN is
// false, so the NaN would survive into the recurrence and poison every later
// bar. isnan() is the only correct test and is what is used here.
//
// ROUNDING COUNT. Four fused steps on the CPU, all reproduced as fma():
//     long_sum2 = x.mul_add(x, long_sum2)                 -> fma(x, x, long_sum2)
//     long_sum2 = (-x_out).mul_add(x_out, long_sum2)      -> fma(-x_out, x_out, long_sum2)
//     val = (x - val).mul_add(k, val)                     -> fma(x - val, k, val)
// Note the STEADY-STATE long/short sums use `+= x_new2` (unfused) on the way in
// and a FUSED negative multiply-add on the way out — an asymmetry that is
// deliberate in the CPU source and is reproduced rather than tidied.
//
// Sequential: four accumulators and val carry across bars. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_vidya() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

#define NEF_VIDYA_SHORT 2
#define NEF_VIDYA_LONG  5
#define NEF_VIDYA_ALPHA 0.2

extern "C" __global__
void neoethos_vidya_f64(const double* __restrict__ prices,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;  // PERIOD-INVARIANT: see the header.

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const double QNAN = nef_qnan_vidya();

    const int short_period = NEF_VIDYA_SHORT;
    const int long_period = NEF_VIDYA_LONG;
    const double alpha = NEF_VIDYA_ALPHA;

    if (first_valid < 0 || first_valid + long_period > n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    {
        const int warm = (first_valid + long_period - 2) < n ? (first_valid + long_period - 2) : n;
        for (int i = 0; i < warm; ++i) row[i] = QNAN;
        for (int i = warm; i < n; ++i) row[i] = QNAN;
    }

    double long_sum = 0.0, long_sum2 = 0.0, short_sum = 0.0, short_sum2 = 0.0;

    const double sp_f = (double)short_period;
    const double lp_f = (double)long_period;
    const double short_inv = 1.0 / sp_f;
    const double long_inv = 1.0 / lp_f;

    const int warm_end = first_valid + long_period;
    const int short_head = warm_end - short_period;

    for (int i = first_valid; i < short_head; ++i) {
        const double x = prices[i];
        long_sum += x;
        long_sum2 = fma(x, x, long_sum2);
    }
    for (int i = short_head; i < warm_end; ++i) {
        const double x = prices[i];
        long_sum += x;
        long_sum2 = fma(x, x, long_sum2);
        short_sum += x;
        short_sum2 = fma(x, x, short_sum2);
    }

    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;

    double val = prices[idx_m2];
    row[idx_m2] = val;

    if (idx_m1 < n) {
        const double short_mean = short_sum * short_inv;
        const double long_mean = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - (short_mean * short_mean);
        const double long_var = long_sum2 * long_inv - (long_mean * long_mean);
        const double short_std = sqrt(short_var);
        const double long_std = sqrt(long_var);

        double k = short_std / long_std;
        if (isnan(k)) k = 0.0;
        k *= alpha;

        const double x = prices[idx_m1];
        val = fma(x - val, k, val);
        row[idx_m1] = val;
    }

    for (int t = warm_end; t < n; ++t) {
        const double x_new = prices[t];
        const double x_new2 = x_new * x_new;

        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;

        const double x_long_out = prices[t - long_period];
        const double x_short_out = prices[t - short_period];
        long_sum -= x_long_out;
        long_sum2 = fma(-x_long_out, x_long_out, long_sum2);
        short_sum -= x_short_out;
        short_sum2 = fma(-x_short_out, x_short_out, short_sum2);

        const double short_mean = short_sum * short_inv;
        const double long_mean = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - (short_mean * short_mean);
        const double long_var = long_sum2 * long_inv - (long_mean * long_mean);
        const double short_std = sqrt(short_var);
        const double long_std = sqrt(long_var);

        double k = short_std / long_std;
        if (isnan(k)) k = 0.0;
        k *= alpha;

        val = fma(x_new - val, k, val);
        row[t] = val;
    }
}


// ===========================================================================
// S1 f64 LANE  --  vidya
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/vidya.rs -- `vidya_scalar` (:494), `vidya_with_kernel` (:317)
//
// PERIOD-INVARIANT. `compute_vidya_batch` (cpu_batch.rs:15693) reads
// `short_period` (2), `long_period` (5) and `alpha` (0.2). There is no
// `period` parameter, so every row of a sweep is byte-identical.
//
// ARITHMETIC ORDER, and every `mul_add` that must stay `fma`:
//   seed sums: `long_sum2 = x.mul_add(x, long_sum2)` -- ONE rounding per bar,
//   not a multiply plus an add.
//   slide: `long_sum2 = (-x_out).mul_add(x_out, long_sum2)` -- likewise, and
//   note the negation is on the FIRST operand, which matters for the sign of
//   the rounding of the product.
//   recurrence: `val = (x - val).mul_add(k, val)` -- ONE rounding. This is the
//   same shape the brief names for `natr`: `(tr - atr).mul_add(inv_p, atr)`.
//   Written as `val + (x - val) * k` it would be two roundings and a different
//   series from the seed bar onward.
// The variance is the RAW form `sum2 * inv - mean * mean`, which is what the
// CPU computes; the numerically-better centred form would be a different
// number and is deliberately not used.
//
// `k.is_nan()` -> 0.0 is a NaN TEST, not a comparison chain: when `long_std`
// is zero the ratio is NaN (0/0) or infinite, and only the NaN case is mapped
// to zero. Reproduced with an explicit `x != x`, because an `fmax`-style
// rewrite would silently also swallow the infinite case.
//
// WARMUP: `alloc_with_nan_prefix(len, first + long_period - 2)` -- note the
// `- 2`, not the `- 1` most windowed indicators use, and the first two emitted
// bars are written by the seed block rather than by the main loop.
//
// KERNEL SELECTION CAVEAT, RECORDED RATHER THAN HIDDEN: `vidya_with_kernel`
// maps `Kernel::Auto` to `detect_best_kernel()` with Avx512 folded to Avx2
// (vidya.rs:357-362), so on an x86_64 host with `nightly-avx` the CPU answer
// is `vidya_avx2`, not `vidya_scalar`. This kernel is written against the
// SCALAR reference. If the two disagree the fix belongs in the CPU -- the same
// remedy `wilders` already received (one shared seed function) -- not in a
// tolerance here.
// ===========================================================================

#ifndef NEO_S1_QNAN_DEFINED
#define NEO_S1_QNAN_DEFINED
// The f32 kernels in this crate spell NaN `__int_as_float(0x7fc00000)`. That is
// a 32-bit pattern; widening it is a value change, not a cast. This is the f64
// quiet-NaN pattern, stated once per translation unit.
__device__ __forceinline__ double neo_s1_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}
__device__ __forceinline__ bool neo_s1_isnan(double x) { return x != x; }
#endif

extern "C" __global__ void neoethos_vidya_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    double* __restrict__ row = out + (size_t)r * (size_t)n;
    (void)periods;

    // `VidyaParams::default()` as read by cpu_batch.rs:15700-15702.
    const int short_period = 2;
    const int long_period = 5;
    const double alpha = 0.2;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (short_period < 2) ||
        (long_period < short_period) || (long_period < 2) || (long_period > n) ||
        ((n - first_valid) < long_period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warmup = first_valid + long_period - 2;
    for (int i = 0; i < warmup && i < n; ++i) row[i] = neo_s1_qnan();

    double long_sum = 0.0, long_sum2 = 0.0, short_sum = 0.0, short_sum2 = 0.0;
    const double short_inv = 1.0 / (double)short_period;
    const double long_inv  = 1.0 / (double)long_period;

    const int warm_end = first_valid + long_period;
    const int short_head = warm_end - short_period;

    for (int i = first_valid; i < short_head; ++i) {
        const double x = prices[i];
        long_sum += x;
        long_sum2 = fma(x, x, long_sum2);
    }
    for (int i = short_head; i < warm_end; ++i) {
        const double x = prices[i];
        long_sum += x;
        long_sum2 = fma(x, x, long_sum2);
        short_sum += x;
        short_sum2 = fma(x, x, short_sum2);
    }

    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;

    double val = prices[idx_m2];
    row[idx_m2] = val;

    if (idx_m1 < n) {
        const double short_mean = short_sum * short_inv;
        const double long_mean = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - (short_mean * short_mean);
        const double long_var = long_sum2 * long_inv - (long_mean * long_mean);
        const double short_std = sqrt(short_var);
        const double long_std = sqrt(long_var);

        double k = short_std / long_std;
        if (neo_s1_isnan(k)) k = 0.0;
        k *= alpha;

        const double x = prices[idx_m1];
        val = fma(x - val, k, val);
        row[idx_m1] = val;
    }

    for (int t = warm_end; t < n; ++t) {
        const double x_new = prices[t];
        const double x_new2 = x_new * x_new;

        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;

        const double x_long_out = prices[t - long_period];
        const double x_short_out = prices[t - short_period];
        long_sum -= x_long_out;
        long_sum2 = fma(-x_long_out, x_long_out, long_sum2);
        short_sum -= x_short_out;
        short_sum2 = fma(-x_short_out, x_short_out, short_sum2);

        const double short_mean = short_sum * short_inv;
        const double long_mean = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - (short_mean * short_mean);
        const double long_var = long_sum2 * long_inv - (long_mean * long_mean);
        const double short_std = sqrt(short_var);
        const double long_std = sqrt(long_var);

        double k = short_std / long_std;
        if (neo_s1_isnan(k)) k = 0.0;
        k *= alpha;

        val = fma(x_new - val, k, val);
        row[t] = val;
    }
}
