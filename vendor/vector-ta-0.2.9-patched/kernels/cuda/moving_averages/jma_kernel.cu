#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


#ifdef JMA_INTERNAL_F64
  using JMA_T = double;
  #define JMA_FMA  fma
  #define JMA_NAN  CUDART_NAN
  __device__ __forceinline__ JMA_T cvt(float x){ return static_cast<double>(x); }
  __device__ __forceinline__ float cvt_back(JMA_T x){ return static_cast<float>(x); }
#else
  using JMA_T = float;
  #define JMA_FMA  __fmaf_rn
  #define JMA_NAN  CUDART_NAN_F
  __device__ __forceinline__ JMA_T cvt(float x){ return x; }
  __device__ __forceinline__ float cvt_back(JMA_T x){ return x; }
#endif


extern "C" __global__
void jma_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ alphas,
                   const float* __restrict__ one_minus_betas,
                   const float* __restrict__ phase_ratios,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out)
{

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* __restrict__ out_row = out + combo * series_len;

    if (series_len <= 0) return;

    int fv = first_valid;
    if (fv < 0) fv = 0;


    if (fv >= series_len) {
        const float nanv = JMA_NAN;
        for (int i = 0; i < series_len; ++i) out_row[i] = nanv;
        return;
    }


    if (fv > 0) {
        const float nanv = JMA_NAN;
        for (int i = 0; i < fv; ++i) out_row[i] = nanv;
    }


    const JMA_T alpha           = cvt(alphas[combo]);
    const JMA_T one_minus_beta  = cvt(one_minus_betas[combo]);
    const JMA_T beta            = JMA_T(1) - one_minus_beta;
    const JMA_T phase_ratio     = cvt(phase_ratios[combo]);
    const JMA_T one_minus_alpha = JMA_T(1) - alpha;
    const JMA_T alpha_sq        = alpha * alpha;
    const JMA_T oma_sq          = one_minus_alpha * one_minus_alpha;


    JMA_T e0 = cvt(prices[fv]);
    JMA_T e1 = JMA_T(0);
    JMA_T e2 = JMA_T(0);
    JMA_T j_prev = e0;

    out_row[fv] = cvt_back(j_prev);


    for (int i = fv + 1; i < series_len; ++i) {
        const JMA_T price = cvt(prices[i]);


        e0 = JMA_FMA(alpha, e0, one_minus_alpha * price);


        const JMA_T diff_price = price - e0;
        e1 = JMA_FMA(beta, e1, one_minus_beta * diff_price);


        const JMA_T diff = JMA_FMA(phase_ratio, e1, e0) - j_prev;


        e2 = JMA_FMA(alpha_sq, e2, oma_sq * diff);

        j_prev += e2;
        out_row[i] = cvt_back(j_prev);
    }
}

extern "C" __global__
void jma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   float alpha_f,
                                   float one_minus_beta_f,
                                   float phase_ratio_f,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ out_tm)
{

    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    if (series_len <= 0) return;

    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;

    const float nanv = JMA_NAN;


    if (fv >= series_len) {
        int idx = series_idx;
        for (int t = 0; t < series_len; ++t, idx += num_series) out_tm[idx] = nanv;
        return;
    }


    if (fv > 0) {
        int idx = series_idx;
        for (int t = 0; t < fv; ++t, idx += num_series) out_tm[idx] = nanv;
    }


    const JMA_T alpha           = cvt(alpha_f);
    const JMA_T one_minus_beta  = cvt(one_minus_beta_f);
    const JMA_T beta            = JMA_T(1) - one_minus_beta;
    const JMA_T phase_ratio     = cvt(phase_ratio_f);
    const JMA_T one_minus_alpha = JMA_T(1) - alpha;
    const JMA_T alpha_sq        = alpha * alpha;
    const JMA_T oma_sq          = one_minus_alpha * one_minus_alpha;


    int idx = fv * num_series + series_idx;
    JMA_T e0 = cvt(prices_tm[idx]);
    JMA_T e1 = JMA_T(0);
    JMA_T e2 = JMA_T(0);
    JMA_T j_prev = e0;

    out_tm[idx] = cvt_back(j_prev);


    for (int t = fv + 1; t < series_len; ++t) {
        idx += num_series;
        const JMA_T price = cvt(prices_tm[idx]);

        e0 = JMA_FMA(alpha, e0, one_minus_alpha * price);
        const JMA_T diff_price = price - e0;
        e1 = JMA_FMA(beta,  e1, one_minus_beta * diff_price);
        const JMA_T diff = JMA_FMA(phase_ratio, e1, e0) - j_prev;
        e2 = JMA_FMA(alpha_sq, e2, oma_sq * diff);

        j_prev += e2;
        out_tm[idx] = cvt_back(j_prev);
    }
}


// ===========================================================================
// S2 f64 LANE — jma  (Jurik moving average)
// ===========================================================================
// Reference: src/indicators/moving_averages/jma.rs
//   `jma_with_kernel` (:233) — first_valid, refusals, alloc_with_nan_prefix
//   `jma_scalar`      (:373) — the three-stage recurrence
//   Defaults: phase = 50.0 (:113), power = 2 (:117). The sweep varies `period`
//   alone, so those are the values the batch lane can produce; they are named
//   here rather than assumed.
//
// WHY THE f32 KERNEL ABOVE IS NOT SALVAGEABLE BY WIDENING ALONE
//   `e2` is a second-order feedback term multiplied by `alpha_sq` each bar and
//   ADDED into `j_prev`, which is the output. The output is therefore an
//   integrator: every rounding error ever made is retained, not damped. In f32
//   the series drifts monotonically; that is the whole reason this lane exists.
//
// ROUNDINGS PER BAR, read off the CPU:
//   e0 = one_minus_alpha.mul_add(x, alpha * e0)        -> mul + fma      (2)
//   e1 = (x - e0).mul_add(one_minus_beta, beta * e1)   -> sub + mul + fma(3)
//   d  = e0 + pr * e1 - j_prev                         -> mul + add + sub(3)
//   e2 = d.mul_add(oma_sq, alpha_sq * e2)              -> mul + fma      (2)
//   j_prev += e2                                       -> add           (1)
// Eleven, in that order. Written out below in exactly that shape.
//
// THE UNROLL IS NOT PART OF THE SPEC. `jma_scalar` unrolls by four, but the
// four bodies are identical and each depends on the previous, so the unroll
// changes no association. One loop here.
//
// WARMUP. `alloc_with_nan_prefix(len, first)` — NaN strictly BEFORE
// `first_valid`; `output[first_valid] = data[first_valid]` is a real value, not
// a warmup marker.
// ===========================================================================

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_jma_batch_f64(
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

    const double phase = 50.0;   // JmaInput::get_phase  -> unwrap_or(50.0)
    const int    power = 2;      // JmaInput::get_power  -> unwrap_or(2)

    const bool declined =
        (n <= 0) ||
        (period <= 0) || (period > n) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = neo_s2_qnan();

    // pr, from `phase`: the CPU's three-way clamp, not a min/max pair — the
    // boundary cases are `< -100` and `> 100`, so exactly -100 and exactly 100
    // take the linear branch.
    double pr;
    if (phase < -100.0)      pr = 0.5;
    else if (phase > 100.0)  pr = 2.5;
    else                     pr = phase / 100.0 + 1.5;

    const double num = 0.45 * ((double)period - 1.0);
    const double beta = num / (num + 2.0);
    const double one_minus_beta = 1.0 - beta;

    // `beta.powi(power as i32)` — repeated multiplication, NOT `pow`. For
    // power == 2 that is one multiply; `pow(beta, 2.0)` is a different
    // function and may round differently.
    double alpha = 1.0;
    for (int k = 0; k < power; ++k) alpha *= beta;

    const double one_minus_alpha = 1.0 - alpha;
    const double alpha_sq = alpha * alpha;
    const double oma_sq = one_minus_alpha * one_minus_alpha;

    double e0 = prices[first_valid];
    double e1 = 0.0;
    double e2 = 0.0;
    double j_prev = prices[first_valid];

    row[first_valid] = j_prev;

    for (int i = first_valid + 1; i < n; ++i) {
        const double x = prices[i];
        e0 = fma(one_minus_alpha, x, alpha * e0);
        e1 = fma(x - e0, one_minus_beta, beta * e1);
        const double d = e0 + pr * e1 - j_prev;
        e2 = fma(d, oma_sq, alpha_sq * e2);
        j_prev += e2;
        row[i] = j_prev;
    }
}
