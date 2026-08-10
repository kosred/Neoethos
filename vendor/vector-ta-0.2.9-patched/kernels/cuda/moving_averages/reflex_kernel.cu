#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


constexpr double REFLEX_PI_D = 3.14159265358979323846264338327950288;
constexpr double REFLEX_SQRT2_APPROX_D = 1.414;


static __device__ __forceinline__ int wrap_inc(int idx, int len) {
    idx += 1;
    return idx - (idx == len) * len;
}

static __device__ __forceinline__ float reflex_out_if_valid(double ms, double my_sum) {
    if (ms > 0.0 && isfinite(ms)) {
        return static_cast<float>(my_sum / sqrt(ms));
    }
    return 0.0f;
}


extern "C" __global__
void reflex_batch_f32(const float* __restrict__ prices,
                      const int*   __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int ,
                      float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) { return; }

    const int period = __ldg(periods + combo);
    if (period < 2 || series_len <= 0) { return; }

    float* __restrict__ out_row = out + combo * series_len;


    const int warm = period < series_len ? period : series_len;
    for (int i = 0; i < warm; ++i) { out_row[i] = 0.0f; }


    int half_period_i = period / 2; if (half_period_i < 1) half_period_i = 1;
    const double half_p = static_cast<double>(half_period_i);
    const double a  = exp(-REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double a2 = a * a;
    const double b  = 2.0 * a * cos(REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double c  = 0.5 * (1.0 + a2 - b);


    extern __shared__ double sdata[];
    double* __restrict__ ring = sdata;
    const int ring_len = period + 1;


    if (series_len > 0) ring[0] = static_cast<double>(__ldg(prices + 0));
    if (series_len > 1) ring[1] = static_cast<double>(__ldg(prices + 1));


    double ssf_sum = ((series_len > 0) ? ring[0] : 0.0) + ((series_len > 1) ? ring[1] : 0.0);

    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;

    double ms = 0.0;


    int idx    = 2;
    int idx_im1 = 1;
    int idx_im2 = 0;
    int idx_ip  = 0;


    double dim1 = (series_len > 1) ? static_cast<double>(__ldg(prices + 1)) : 0.0;


    int i = 2;
    for (; i < series_len; ) {
    #pragma unroll 4
        for (int u = 0; u < 4; ++u) {
            if (i >= series_len) break;


            const double di = static_cast<double>(__ldg(prices + i));


            const double ssf_im1 = ring[idx_im1];
            const double ssf_im2 = ring[idx_im2];
            const double t0 = c * (di + dim1);
            const double t1 = fma(-a2, ssf_im2, t0);
            const double ssf_i = fma(b, ssf_im1, t1);


            ring[idx] = ssf_i;

            if (i < period) {
                ssf_sum += ssf_i;
            } else {
                const double ssf_old = ring[idx_ip];
                const double mean_lp = ssf_sum * inv_p;
                const double my_sum  = fma(beta, ssf_i, alpha * ssf_old) - mean_lp;

                ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
                out_row[i] = reflex_out_if_valid(ms, my_sum);


                ssf_sum += ssf_i - ssf_old;


                idx_ip = wrap_inc(idx_ip, ring_len);
            }


            idx    = wrap_inc(idx,    ring_len);
            idx_im1 = wrap_inc(idx_im1, ring_len);
            idx_im2 = wrap_inc(idx_im2, ring_len);

            dim1 = di;
            ++i;
        }
    }
}


extern "C" __global__
void reflex_batch_f32_precomp(const float* __restrict__ prices,
                              const int*   __restrict__ periods,
                              const double* __restrict__ a2s,
                              const double* __restrict__ bs,
                              const double* __restrict__ cs,
                              int series_len,
                              int n_combos,
                              int ,
                              float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) { return; }

    const int period = __ldg(periods + combo);
    if (period < 2 || series_len <= 0) { return; }

    float* __restrict__ out_row = out + combo * series_len;
    const int warm = period < series_len ? period : series_len;
    for (int i = 0; i < warm; ++i) { out_row[i] = 0.0f; }

    const double a2 = __ldg(a2s + combo);
    const double b  = __ldg(bs  + combo);
    const double c  = __ldg(cs  + combo);

    extern __shared__ double sdata[];
    double* __restrict__ ring = sdata;
    const int ring_len = period + 1;

    if (series_len > 0) ring[0] = static_cast<double>(__ldg(prices + 0));
    if (series_len > 1) ring[1] = static_cast<double>(__ldg(prices + 1));

    double ssf_sum = ((series_len > 0) ? ring[0] : 0.0) + ((series_len > 1) ? ring[1] : 0.0);

    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;
    double ms = 0.0;

    int idx = 2, idx_im1 = 1, idx_im2 = 0, idx_ip = 0;
    double dim1 = (series_len > 1) ? static_cast<double>(__ldg(prices + 1)) : 0.0;

    int i = 2;
    for (; i < series_len; ) {
    #pragma unroll 4
        for (int u = 0; u < 4; ++u) {
            if (i >= series_len) break;

            const double di = static_cast<double>(__ldg(prices + i));

            const double ssf_im1 = ring[idx_im1];
            const double ssf_im2 = ring[idx_im2];
            const double t0 = c * (di + dim1);
            const double t1 = fma(-a2, ssf_im2, t0);
            const double ssf_i = fma(b, ssf_im1, t1);

            ring[idx] = ssf_i;

            if (i < period) {
                ssf_sum += ssf_i;
            } else {
                const double ssf_old = ring[idx_ip];
                const double mean_lp = ssf_sum * inv_p;
                const double my_sum  = fma(beta, ssf_i, alpha * ssf_old) - mean_lp;

                ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
                out_row[i] = reflex_out_if_valid(ms, my_sum);

                ssf_sum += ssf_i - ssf_old;

                idx_ip = wrap_inc(idx_ip, ring_len);
            }

            idx    = wrap_inc(idx,    ring_len);
            idx_im1 = wrap_inc(idx_im1, ring_len);
            idx_im2 = wrap_inc(idx_im2, ring_len);

            dim1 = di;
            ++i;
        }
    }
}


extern "C" __global__
void reflex_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ ,
                                      float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || threadIdx.x != 0) { return; }
    if (period < 2 || series_len <= 0) { return; }


    for (int t = 0; t < series_len; ++t) { out_tm[t * num_series + series] = 0.0f; }
    const int warm = period < series_len ? period : series_len;
    for (int t = 0; t < warm; ++t) { out_tm[t * num_series + series] = 0.0f; }

    int half_period_i = period / 2; if (half_period_i < 1) half_period_i = 1;
    const double half_p = static_cast<double>(half_period_i);
    const double a  = exp(-REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double a2 = a * a;
    const double b  = 2.0 * a * cos(REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double c  = 0.5 * (1.0 + a2 - b);

    extern __shared__ double sdata[];
    double* ring = sdata;
    const int ring_len = period + 1;
    if (series_len > 0) ring[0] = static_cast<double>(prices_tm[0 * num_series + series]);
    if (series_len > 1) ring[1] = static_cast<double>(prices_tm[1 * num_series + series]);

    double ssf_sum = 0.0;
    if (period == 1) {
        ssf_sum = (series_len > 0) ? ring[0] : 0.0;
    } else {
        ssf_sum = ((series_len > 0) ? ring[0] : 0.0)
                + ((series_len > 1) ? ring[1] : 0.0);
    }
    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;
    double ms = 0.0;

    for (int t = 2; t < series_len; ++t) {
        const int idx     = t % ring_len;
        const int idx_im1 = (t - 1) % ring_len;
        const int idx_im2 = (t - 2) % ring_len;

        const double di   = static_cast<double>(prices_tm[t * num_series + series]);
        const double dim1 = static_cast<double>(prices_tm[(t - 1) * num_series + series]);
        const double ssf_im1 = ring[idx_im1];
        const double ssf_im2 = ring[idx_im2];

        const double t0 = c * (di + dim1);
        const double t1 = (-a2) * ssf_im2 + t0;
        const double ssf_t = b * ssf_im1 + t1;
        ring[idx] = ssf_t;

        if (t < period) { ssf_sum += ssf_t; continue; }

        const int idx_ip = (t - period) % ring_len;
        const double ssf_ip = ring[idx_ip];
        const double mean_lp = ssf_sum * inv_p;
        const double my_sum = beta * ssf_t + alpha * ssf_ip - mean_lp;

        ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
        out_tm[t * num_series + series] = reflex_out_if_valid(ms, my_sum);


        ssf_sum += ssf_t - ssf_ip;
    }
}


// ===========================================================================
// S2 f64 LANE — reflex
// ===========================================================================
// Reference: src/indicators/moving_averages/reflex.rs
//   `reflex_prepare`     (:332) — first_valid + the refusals
//   `reflex_with_kernel` (:190) — alloc_with_nan_prefix(len, period), then
//                                 out[..period].fill(0.0)
//   `reflex_scalar`      (:238) — the two-pole super-smoother recurrence and
//                                 the RMS normaliser
//
// THREE THINGS THIS FIXES
//  1. f32 -> f64 for the whole 2-pole IIR. Same argument as gaussian: the pole
//     radius is exp(-1.414*PI/(period/2)), which at period 200 is 0.9578, so
//     an f32 rounding is remembered for hundreds of bars.
//  2. `ms` is an exponentially-weighted MEAN SQUARE and the output divides by
//     its square root. In f32 a small `ms` loses half its significant digits
//     before `sqrt`, and the quotient is the indicator itself.
//  3. NaN was built with `__int_as_float`; here it is the f64 quiet-NaN bit
//     pattern.
//
// ROUNDING COUNT, per bar, from the CPU line by line:
//     t0     = c * (di + dim1)                  -> add, mul        (2)
//     t1     = (-a2).mul_add(ssf_im2, t0)       -> fma             (1)
//     ssf_i  = b.mul_add(ssf_im1, t1)           -> fma             (1)
//     my_sum = ssf_i.mul_add(beta, ssf_ip*alpha) - mean_lp -> mul, fma, sub (3)
//     ms     = 0.96.mul_add(ms, 0.04*(my_sum*my_sum)) -> mul, mul, fma  (3)
// Reproduced one for one below. `0.96` and `0.04` are Ehlers' constants, not
// f32-sized epsilons, so they are carried unchanged — they are exact in
// neither width and the CPU's f64 value of `0.04` is what we must match, which
// is what writing `0.04` in a double context gives.
//
// THE RING. `ssf` is `period + 1` doubles and the recursion needs the entry
// from `period` bars ago, so there is no way to drop it. It is a per-thread
// local array bounded at compile time by REFLEX_MAX_PERIOD, exactly as
// `neoethos_mfi_batch_f64` bounds its two rings, and the host refuses a larger
// period BY NAME (`F64Kernel::max_period`) rather than truncating the window.
//
// WHERE THIS IS MORE DEFINED THAN THE CPU. `alloc_with_nan_prefix(len, period)`
// leaves indices >= period UNINITIALISED, and `reflex_scalar` writes index i
// only when `ms > 0`. In practice `ms` is positive from the first bar past the
// warmup, so the CPU's uninitialised window is empty; this kernel fills the
// whole row with NaN first so that if it ever is not empty the answer is a
// loud NaN rather than whatever was in the buffer.
// ===========================================================================

// Rust's `std::f64::consts::PI`, written out rather than relying on `M_PI`
// (which MSVC hides behind _USE_MATH_DEFINES) or on `CUDART_PI` being in
// scope. Same bits either way; stated once so it cannot drift.
#define NEO_S2_PI 3.14159265358979323846264338327950288

#define REFLEX_MAX_PERIOD 512

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_reflex_batch_f64(
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
    const int period = periods[r];

    const bool declined =
        (n <= 0) ||
        (period < 2) || (period > REFLEX_MAX_PERIOD) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period > (n - first_valid));
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    // `alloc_with_nan_prefix(len, period)` then `out[..period].fill(0.0)`:
    // the first `period` bars are ZERO, not NaN. Everything after starts NaN
    // and is overwritten by the loop.
    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
    const int zend = period < n ? period : n;
    for (int i = 0; i < zend; ++i) row[i] = 0.0;

    if (n < 2) return;

    // `half_p = (period / 2).max(1) as f64` — INTEGER division first.
    const int half_i = (period / 2) > 1 ? (period / 2) : 1;
    const double half_p = (double)half_i;
    const double a = exp(-1.414 * NEO_S2_PI / half_p);
    const double a2 = a * a;
    const double b = 2.0 * a * cos(1.414 * NEO_S2_PI / half_p);
    const double c = 0.5 * (1.0 + a2 - b);

    const int ring_len = period + 1;
    double ssf[REFLEX_MAX_PERIOD + 1];
    for (int i = 0; i < ring_len; ++i) ssf[i] = 0.0;

    ssf[0] = prices[0];
    ssf[1] = prices[1];
    double ssf_sum = ssf[0] + ssf[1];

    const double inv_p = 1.0 / (double)period;
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta = 1.0 - alpha;

    double ms = 0.0;

    int idx_im2 = 0;
    int idx_im1 = 1;
    int idx = 2;

    for (int i = 2; i < n; ++i) {
        const double di = prices[i];
        const double dim1 = prices[i - 1];
        const double ssf_im1 = ssf[idx_im1];
        const double ssf_im2 = ssf[idx_im2];

        const double t0 = c * (di + dim1);
        const double t1 = fma(-a2, ssf_im2, t0);
        const double ssf_i = fma(b, ssf_im1, t1);

        ssf[idx] = ssf_i;

        if (i < period) {
            ssf_sum += ssf_i;
        } else {
            int idx_ip = idx + 1;
            if (idx_ip == ring_len) idx_ip = 0;
            const double ssf_ip = ssf[idx_ip];

            const double mean_lp = ssf_sum * inv_p;
            const double my_sum = fma(ssf_i, beta, ssf_ip * alpha) - mean_lp;

            ms = fma(0.96, ms, 0.04 * (my_sum * my_sum));
            if (ms > 0.0) {
                row[i] = my_sum / sqrt(ms);
            }

            ssf_sum += ssf_i - ssf_ip;
        }

        idx_im2 = idx_im1;
        idx_im1 = idx;
        idx += 1;
        if (idx == ring_len) idx = 0;
    }
}
