#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

 #include <cuda_runtime.h>
 #include <math.h>


 #ifndef GAUSS_BLOCK_DIM
 #define GAUSS_BLOCK_DIM 256
 #endif

 #ifndef GAUSS_USE_STREAMING_STORES
 #define GAUSS_USE_STREAMING_STORES 0
 #endif


 static __device__ __forceinline__ float qnan_f() {
     return __int_as_float(0x7fffffff);
 }

 static __device__ __forceinline__ void store_out(float* __restrict__ p, float v) {
 #if GAUSS_USE_STREAMING_STORES


     asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v));
 #else
     *p = v;
 #endif
 }

 static __device__ __forceinline__ int clampi(int x, int lo, int hi) {
     return (x < lo) ? lo : (x > hi ? hi : x);
 }


 static __device__ __forceinline__ void gaussian_run_poles1(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1)
 {
     double y_prev = 0.0;
     int idx = 0;
     int t = 0;


     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         y_prev = c1 * y_prev + c0 * x;
         store_out(out + idx, nan_f);
     }

     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         y_prev = c1 * y_prev + c0 * x;
         store_out(out + idx, static_cast<float>(y_prev));
     }
 }

 static __device__ __forceinline__ void gaussian_run_poles2(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1,
     double c2)
 {
     double p1 = 0.0;
     double p0 = 0.0;
     int idx = 0;
     int t = 0;

     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c2 * p0 + c1 * p1 + c0 * x;
         p0 = p1; p1 = y;
         store_out(out + idx, nan_f);
     }
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c2 * p0 + c1 * p1 + c0 * x;
         p0 = p1; p1 = y;
         store_out(out + idx, static_cast<float>(y));
     }
 }

 static __device__ __forceinline__ void gaussian_run_poles3(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1,
     double c2,
     double c3)
 {
     double p2 = 0.0;
     double p1 = 0.0;
     double p0 = 0.0;
     int idx = 0;
     int t = 0;

     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c3 * p0 + c2 * p1 + c1 * p2 + c0 * x;
         p0 = p1; p1 = p2; p2 = y;
         store_out(out + idx, nan_f);
     }
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c3 * p0 + c2 * p1 + c1 * p2 + c0 * x;
         p0 = p1; p1 = p2; p2 = y;
         store_out(out + idx, static_cast<float>(y));
     }
 }

 static __device__ __forceinline__ void gaussian_run_poles4(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1,
     double c2,
     double c3,
     double c4)
 {
     double p3 = 0.0;
     double p2 = 0.0;
     double p1 = 0.0;
     double p0 = 0.0;
     int idx = 0;
     int t = 0;

     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = (((c4 * p0) + (c3 * p1)) + (c2 * p2)) + (c1 * p3) + (c0 * x);
         p0 = p1; p1 = p2; p2 = p3; p3 = y;
         store_out(out + idx, nan_f);
     }
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = (((c4 * p0) + (c3 * p1)) + (c2 * p2)) + (c1 * p3) + (c0 * x);
         p0 = p1; p1 = p2; p2 = p3; p3 = y;
         store_out(out + idx, static_cast<float>(y));
     }
 }


 extern "C" __global__ void gaussian_batch_f32(
     const float* __restrict__ prices,
     const int* __restrict__ periods,
     const int* __restrict__ poles,
     const float* __restrict__ coeffs,
     int coeff_stride,
     int series_len,
     int n_combos,
     int first_valid,
     float* __restrict__ out)
 {
     const float nan_f = qnan_f();


     for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
          combo < n_combos;
          combo += gridDim.x * blockDim.x)
     {
         const int period = periods[combo];
         const int pole   = poles[combo];


         if (period < 2 || pole < 1 || pole > 4 || series_len <= 0) {


             continue;
         }

         float* out_row = out + combo * series_len;

         int start = first_valid;
         start = clampi(start, 0, series_len);

         int warm = first_valid + period;
         warm = clampi(warm, 0, series_len);


         int valid = warm > start ? warm : start;
         valid = clampi(valid, 0, series_len);

         const float* coeff = coeffs + combo * coeff_stride;
         const double c0 = static_cast<double>(coeff[0]);
         const double c1 = static_cast<double>(coeff[1]);
         const double c2 = static_cast<double>(coeff[2]);
         const double c3 = static_cast<double>(coeff[3]);
         const double c4 = static_cast<double>(coeff[4]);

         switch (pole) {
             case 1:
                 gaussian_run_poles1(prices, out_row, series_len, 1, valid, nan_f, c0, c1);
                 break;
             case 2:
                 gaussian_run_poles2(prices, out_row, series_len, 1, valid, nan_f, c0, c1, c2);
                 break;
             case 3:
                 gaussian_run_poles3(prices, out_row, series_len, 1, valid, nan_f, c0, c1, c2, c3);
                 break;
             case 4:
             default:
                 gaussian_run_poles4(prices, out_row, series_len, 1, valid, nan_f, c0, c1, c2, c3, c4);
                 break;
         }
     }
 }


 extern "C" __global__ void gaussian_many_series_one_param_f32(
     const float* __restrict__ prices_tm,
     const float* __restrict__ coeffs,
     int period,
     int poles,
     int num_series,
     int series_len,
     const int* __restrict__ first_valids,
     float* __restrict__ out_tm)
 {
     if (period < 2 || poles < 1 || poles > 4 || series_len <= 0) return;

     const double c0 = static_cast<double>(coeffs[0]);
     const double c1 = static_cast<double>(coeffs[1]);
     const double c2 = static_cast<double>(coeffs[2]);
     const double c3 = static_cast<double>(coeffs[3]);
     const double c4 = static_cast<double>(coeffs[4]);
     const float nan_f = qnan_f();


     for (int s = blockIdx.x * blockDim.x + threadIdx.x;
          s < num_series;
          s += gridDim.x * blockDim.x)
     {
         int start = first_valids[s];
         start = clampi(start, 0, series_len);


         int warm = first_valids[s] + period;
         warm = clampi(warm, 0, series_len);

         int valid = warm > start ? warm : start;
         valid = clampi(valid, 0, series_len);

         const float* price_series = prices_tm + s;
         float* out_series = out_tm + s;
         const int stride = num_series;

         switch (poles) {
             case 1:
                 gaussian_run_poles1(price_series, out_series, series_len, stride, valid, nan_f, c0, c1);
                 break;
             case 2:
                 gaussian_run_poles2(price_series, out_series, series_len, stride, valid, nan_f, c0, c1, c2);
                 break;
             case 3:
                 gaussian_run_poles3(price_series, out_series, series_len, stride, valid, nan_f, c0, c1, c2, c3);
                 break;
             case 4:
             default:
                 gaussian_run_poles4(price_series, out_series, series_len, stride, valid, nan_f, c0, c1, c2, c3, c4);
                 break;
         }
     }
 }


// ===========================================================================
// S2 f64 LANE — gaussian
// ===========================================================================
// Reference: src/indicators/moving_averages/gaussian.rs
//   `gaussian_with_kernel` (:381)  — validation + first_valid
//   `gaussian_scalar_fma`  (:335)  — beta/alpha derivation
//   `gaussian_poles{1,2,3,4}_fma` (:466,:479,:499,:524) — the recurrences
//   Default poles = 4 (`GaussianParams::get_poles`, :206 `unwrap_or(4)`), which
//   is the only value the batch lane can produce because the sweep varies
//   `period` alone. The other three pole orders are compiled in anyway so a
//   later params-carrying sweep does not need a second kernel.
//
// WHAT THE f32 KERNEL ABOVE GETS WRONG
//   * f32 throughout, in a 4-pole IIR whose poles sit at 1-alpha < 1. Error
//     injected at bar i is multiplied by (1-alpha)^k and summed over the whole
//     series; at period 200 alpha is ~0.03 so (1-alpha) is ~0.97 and the
//     effective memory is hundreds of bars. f32's 24-bit mantissa is not a
//     rounding difference here, it is an accumulated one.
//   * `__int_as_float(0x7fc00000)` for NaN — an f32 bit pattern.
//
// ARITHMETIC ORDER. The CPU writes
//     y = c4.mul_add(p0, c3.mul_add(p1, c2.mul_add(p2, c1.mul_add(p3, c0*x))))
// which is ONE multiply plus FOUR fused multiply-adds — five roundings, in
// that nesting. Reproduced literally below with `fma`, innermost first. Not
// `c0*x + c1*p3 + ...`, which would be nine.
//
// NO WARMUP. `gaussian_with_kernel` allocates with `alloc_uninit_f64(len)` —
// NOT `alloc_with_nan_prefix` — and `gaussian_scalar` writes EVERY index from
// 0, seeding the delay line with 0.0 rather than with the first price. So the
// f64 row here has no NaN prefix either, and `first_valid` is used ONLY for
// the same "not enough valid data" refusal the CPU makes.
// ===========================================================================

// Rust's `std::f64::consts::PI`, written out rather than relying on `M_PI`
// (which MSVC hides behind _USE_MATH_DEFINES) or on `CUDART_PI` being in
// scope. Same bits either way; stated once so it cannot drift.
#define NEO_S2_PI 3.14159265358979323846264338327950288

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_gaussian_batch_f64(
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
    const int poles = 4;   // GaussianParams::get_poles -> unwrap_or(4)

    // Every branch of `gaussian_with_kernel` that returns Err.
    const bool declined =
        (n <= 0) ||
        (period < 2) || (period > n) ||
        (poles < 1) || (poles > 4) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    // beta / alpha, exactly as `gaussian_scalar_fma` derives them.
    //   num = 1 - cos(2*PI/period);  den = 2^(1/poles) - 1;  beta = num/den
    //   alpha = -beta + sqrt(beta*beta + 2*beta)
    // `NEO_S2_PI` is the double constant; the f32 kernel above used a truncated
    // literal, which is a different number before any arithmetic happens.
    const double num = 1.0 - cos(2.0 * NEO_S2_PI / (double)period);
    const double den = pow(2.0, 1.0 / (double)poles) - 1.0;
    const double beta = num / den;
    const double tmp = beta * beta + 2.0 * beta;
    const double alpha = -beta + sqrt(tmp);

    const double one = 1.0 - alpha;

    if (poles == 1) {
        const double c0 = alpha;
        const double c1 = one;
        double prev = 0.0;
        for (int i = 0; i < n; ++i) {
            prev = fma(c1, prev, c0 * prices[i]);
            row[i] = prev;
        }
        return;
    }
    if (poles == 2) {
        const double a2 = alpha * alpha;
        const double c0 = a2;
        const double c1 = 2.0 * one;
        const double c2 = -(one * one);
        double prev1 = 0.0, prev0 = 0.0;
        for (int i = 0; i < n; ++i) {
            const double y = fma(c2, prev0, fma(c1, prev1, c0 * prices[i]));
            prev0 = prev1;
            prev1 = y;
            row[i] = y;
        }
        return;
    }
    if (poles == 3) {
        const double a3 = alpha * alpha * alpha;
        const double one2 = one * one;
        const double c0 = a3;
        const double c1 = 3.0 * one;
        const double c2 = -3.0 * one2;
        const double c3 = one2 * one;
        double p2 = 0.0, p1 = 0.0, p0 = 0.0;
        for (int i = 0; i < n; ++i) {
            const double y = fma(c3, p0, fma(c2, p1, fma(c1, p2, c0 * prices[i])));
            p0 = p1;
            p1 = p2;
            p2 = y;
            row[i] = y;
        }
        return;
    }

    // poles == 4, the default.
    const double a4 = alpha * alpha * alpha * alpha;
    const double one2 = one * one;
    const double one3 = one2 * one;
    const double c0 = a4;
    const double c1 = 4.0 * one;
    const double c2 = -6.0 * one2;
    const double c3 = 4.0 * one3;
    const double c4 = -(one3 * one);

    double p3 = 0.0, p2 = 0.0, p1 = 0.0, p0 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double y = fma(c4, p0, fma(c3, p1, fma(c2, p2, fma(c1, p3, c0 * prices[i]))));
        p0 = p1;
        p1 = p2;
        p2 = p3;
        p3 = y;
        row[i] = y;
    }
}
