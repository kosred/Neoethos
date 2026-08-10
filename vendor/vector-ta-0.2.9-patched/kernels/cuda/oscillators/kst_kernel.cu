#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


static __device__ __forceinline__ float kst_qnan() {
  return __int_as_float(0x7fffffff);
}


struct CompSum {
  float sum;
  float c;
  __device__ __forceinline__ void init() { sum = 0.f; c = 0.f; }
  __device__ __forceinline__ void add(float x) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  __device__ __forceinline__ float val() const { return sum; }
};


__device__ __forceinline__ float kst_safe_roc(float curr, float prev) {
  if (prev != 0.0f && isfinite(curr) && isfinite(prev)) {
    const float inv100_prev = 100.0f / prev;
    return __fmaf_rn(curr, inv100_prev, -100.0f);
  }
  return 0.0f;
}


extern "C" __global__
void kst_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ s1s,
                   const int*   __restrict__ s2s,
                   const int*   __restrict__ s3s,
                   const int*   __restrict__ s4s,
                   const int*   __restrict__ r1s,
                   const int*   __restrict__ r2s,
                   const int*   __restrict__ r3s,
                   const int*   __restrict__ r4s,
                   const int*   __restrict__ sigs,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out_line,
                   float* __restrict__ out_signal) {

  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float nn   = kst_qnan();

  for (int combo = tid; combo < n_combos; combo += stride) {
    const int s1  = s1s[combo];
    const int s2  = s2s[combo];
    const int s3  = s3s[combo];
    const int s4  = s4s[combo];
    const int r1  = r1s[combo];
    const int r2  = r2s[combo];
    const int r3  = r3s[combo];
    const int r4  = r4s[combo];
    const int sig = sigs[combo];

    const float inv1 = (s1 > 0) ? (1.0f / float(s1)) : 0.0f;
    const float w2   = (s2 > 0) ? (2.0f / float(s2)) : 0.0f;
    const float w3   = (s3 > 0) ? (3.0f / float(s3)) : 0.0f;
    const float w4   = (s4 > 0) ? (4.0f / float(s4)) : 0.0f;
    const float invSig = (sig > 0) ? (1.0f / float(sig)) : 0.0f;

    const int start1 = first_valid + r1;
    const int start2 = first_valid + r2;
    const int start3 = first_valid + r3;
    const int start4 = first_valid + r4;

    const int warm_line = max(max(start1 + s1 - 1, start2 + s2 - 1),
                              max(start3 + s3 - 1, start4 + s4 - 1));
    const int warm_sig  = warm_line + sig - 1;

    float* __restrict__ line_row   = out_line   + combo * series_len;
    float* __restrict__ signal_row = out_signal + combo * series_len;


    const int nan_end_line = (warm_line < series_len ? warm_line : series_len);
    for (int i = 0; i < nan_end_line; ++i) line_row[i] = nn;
    const int nan_end_sig = (warm_sig < series_len ? warm_sig : series_len);
    for (int i = 0; i < nan_end_sig; ++i) signal_row[i] = nn;

    CompSum sum1, sum2, sum3, sum4, ssum;
    sum1.init(); sum2.init(); sum3.init(); sum4.init(); ssum.init();

    for (int i = first_valid; i < series_len; ++i) {
      const float x = prices[i];

      if (i >= start1) {
        const float v = kst_safe_roc(x, prices[i - r1]);
        if (i < start1 + s1) sum1.add(v);
        else                 sum1.add(v - kst_safe_roc(prices[i - s1], prices[i - s1 - r1]));
      }
      if (i >= start2) {
        const float v = kst_safe_roc(x, prices[i - r2]);
        if (i < start2 + s2) sum2.add(v);
        else                 sum2.add(v - kst_safe_roc(prices[i - s2], prices[i - s2 - r2]));
      }
      if (i >= start3) {
        const float v = kst_safe_roc(x, prices[i - r3]);
        if (i < start3 + s3) sum3.add(v);
        else                 sum3.add(v - kst_safe_roc(prices[i - s3], prices[i - s3 - r3]));
      }
      if (i >= start4) {
        const float v = kst_safe_roc(x, prices[i - r4]);
        if (i < start4 + s4) sum4.add(v);
        else                 sum4.add(v - kst_safe_roc(prices[i - s4], prices[i - s4 - r4]));
      }

      if (i >= warm_line) {

        float k = __fmaf_rn(sum4.val(), w4,
                  __fmaf_rn(sum3.val(), w3,
                  __fmaf_rn(sum2.val(), w2, sum1.val() * inv1)));

        line_row[i] = k;


        ssum.add(k);


        if (sig > 0 && (i - sig) >= warm_line) {
          ssum.add(-line_row[i - sig]);
        }

        if (i >= warm_sig) {
          signal_row[i] = ssum.val() * invSig;
        }
      }
    }
  }
}


extern "C" __global__
void kst_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   int num_series,
                                   int series_len,
                                   int s1, int s2, int s3, int s4,
                                   int r1, int r2, int r3, int r4,
                                   int sig,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ out_line_tm,
                                   float* __restrict__ out_signal_tm) {
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float nn   = kst_qnan();

  const float inv1  = (s1 > 0) ? (1.0f / float(s1)) : 0.0f;
  const float w2    = (s2 > 0) ? (2.0f / float(s2)) : 0.0f;
  const float w3    = (s3 > 0) ? (3.0f / float(s3)) : 0.0f;
  const float w4    = (s4 > 0) ? (4.0f / float(s4)) : 0.0f;
  const float invSig = (sig > 0) ? (1.0f / float(sig)) : 0.0f;

  for (int s = tid; s < num_series; s += stride) {
    int fv = first_valids[s];
    if (fv < 0)       fv = 0;
    if (fv >= series_len) {
      for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + s;
        out_line_tm[idx] = nn;
        out_signal_tm[idx] = nn;
      }
      continue;
    }

    const int start1 = fv + r1;
    const int start2 = fv + r2;
    const int start3 = fv + r3;
    const int start4 = fv + r4;

    const int warm_line = max(max(start1 + s1 - 1, start2 + s2 - 1),
                              max(start3 + s3 - 1, start4 + s4 - 1));
    const int warm_sig  = warm_line + sig - 1;

    for (int t = 0; t < warm_line && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_line_tm[idx] = nn;
    }
    for (int t = 0; t < warm_sig && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_signal_tm[idx] = nn;
    }

    CompSum sum1, sum2, sum3, sum4, ssum;
    sum1.init(); sum2.init(); sum3.init(); sum4.init(); ssum.init();

    for (int t = fv; t < series_len; ++t) {
      const int idx  = t * num_series + s;
      const float x  = prices_tm[idx];

      if (t >= start1) {
        const float v = kst_safe_roc(x, prices_tm[(t - r1) * num_series + s]);
        if (t < start1 + s1) sum1.add(v);
        else                 sum1.add(v - kst_safe_roc(prices_tm[(t - s1) * num_series + s],
                                                      prices_tm[(t - s1 - r1) * num_series + s]));
      }
      if (t >= start2) {
        const float v = kst_safe_roc(x, prices_tm[(t - r2) * num_series + s]);
        if (t < start2 + s2) sum2.add(v);
        else                 sum2.add(v - kst_safe_roc(prices_tm[(t - s2) * num_series + s],
                                                      prices_tm[(t - s2 - r2) * num_series + s]));
      }
      if (t >= start3) {
        const float v = kst_safe_roc(x, prices_tm[(t - r3) * num_series + s]);
        if (t < start3 + s3) sum3.add(v);
        else                 sum3.add(v - kst_safe_roc(prices_tm[(t - s3) * num_series + s],
                                                      prices_tm[(t - s3 - r3) * num_series + s]));
      }
      if (t >= start4) {
        const float v = kst_safe_roc(x, prices_tm[(t - r4) * num_series + s]);
        if (t < start4 + s4) sum4.add(v);
        else                 sum4.add(v - kst_safe_roc(prices_tm[(t - s4) * num_series + s],
                                                      prices_tm[(t - s4 - r4) * num_series + s]));
      }

      if (t >= warm_line) {
        float k = __fmaf_rn(sum4.val(), w4,
                  __fmaf_rn(sum3.val(), w3,
                  __fmaf_rn(sum2.val(), w2, sum1.val() * inv1)));
        out_line_tm[idx] = k;


        ssum.add(k);
        if (sig > 0 && (t - sig) >= warm_line) {
          ssum.add(-out_line_tm[(t - sig) * num_series + s]);
        }
        if (t >= warm_sig) {
          out_signal_tm[idx] = ssum.val() * invSig;
        }
      }
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

static __device__ __forceinline__ double kst_qnan_f64() {
  return __longlong_as_double(0x7fffffffffffffffULL);
}
struct CompSum_f64 {
  double sum;
  double c;
  __device__ __forceinline__ void init_f64() { sum = 0.; c = 0.; }
  __device__ __forceinline__ void add_f64(double x) {
    double y = x - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  __device__ __forceinline__ double val_f64() const { return sum; }
};
__device__ __forceinline__ double kst_safe_roc_f64(double curr, double prev) {
  if (prev != 0.0 && isfinite(curr) && isfinite(prev)) {
    const double inv100_prev = 100.0 / prev;
    return __fma_rn(curr, inv100_prev, -100.0);
  }
  return 0.0;
}
extern "C" __global__
void kst_batch_f64(const double* __restrict__ prices,
                   const int*   __restrict__ s1s,
                   const int*   __restrict__ s2s,
                   const int*   __restrict__ s3s,
                   const int*   __restrict__ s4s,
                   const int*   __restrict__ r1s,
                   const int*   __restrict__ r2s,
                   const int*   __restrict__ r3s,
                   const int*   __restrict__ r4s,
                   const int*   __restrict__ sigs,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   double* __restrict__ out_line,
                   double* __restrict__ out_signal) {

  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const double nn   = kst_qnan_f64();

  for (int combo = tid; combo < n_combos; combo += stride) {
    const int s1  = s1s[combo];
    const int s2  = s2s[combo];
    const int s3  = s3s[combo];
    const int s4  = s4s[combo];
    const int r1  = r1s[combo];
    const int r2  = r2s[combo];
    const int r3  = r3s[combo];
    const int r4  = r4s[combo];
    const int sig = sigs[combo];

    const double inv1 = (s1 > 0) ? (1.0 / double(s1)) : 0.0;
    const double w2   = (s2 > 0) ? (2.0 / double(s2)) : 0.0;
    const double w3   = (s3 > 0) ? (3.0 / double(s3)) : 0.0;
    const double w4   = (s4 > 0) ? (4.0 / double(s4)) : 0.0;
    const double invSig = (sig > 0) ? (1.0 / double(sig)) : 0.0;

    const int start1 = first_valid + r1;
    const int start2 = first_valid + r2;
    const int start3 = first_valid + r3;
    const int start4 = first_valid + r4;

    const int warm_line = max(max(start1 + s1 - 1, start2 + s2 - 1),
                              max(start3 + s3 - 1, start4 + s4 - 1));
    const int warm_sig  = warm_line + sig - 1;

    double* __restrict__ line_row   = out_line   + combo * series_len;
    double* __restrict__ signal_row = out_signal + combo * series_len;


    const int nan_end_line = (warm_line < series_len ? warm_line : series_len);
    for (int i = 0; i < nan_end_line; ++i) line_row[i] = nn;
    const int nan_end_sig = (warm_sig < series_len ? warm_sig : series_len);
    for (int i = 0; i < nan_end_sig; ++i) signal_row[i] = nn;

    CompSum_f64 sum1, sum2, sum3, sum4, ssum;
    sum1.init_f64(); sum2.init_f64(); sum3.init_f64(); sum4.init_f64(); ssum.init_f64();

    for (int i = first_valid; i < series_len; ++i) {
      const double x = prices[i];

      if (i >= start1) {
        const double v = kst_safe_roc_f64(x, prices[i - r1]);
        if (i < start1 + s1) sum1.add_f64(v);
        else                 sum1.add_f64(v - kst_safe_roc_f64(prices[i - s1], prices[i - s1 - r1]));
      }
      if (i >= start2) {
        const double v = kst_safe_roc_f64(x, prices[i - r2]);
        if (i < start2 + s2) sum2.add_f64(v);
        else                 sum2.add_f64(v - kst_safe_roc_f64(prices[i - s2], prices[i - s2 - r2]));
      }
      if (i >= start3) {
        const double v = kst_safe_roc_f64(x, prices[i - r3]);
        if (i < start3 + s3) sum3.add_f64(v);
        else                 sum3.add_f64(v - kst_safe_roc_f64(prices[i - s3], prices[i - s3 - r3]));
      }
      if (i >= start4) {
        const double v = kst_safe_roc_f64(x, prices[i - r4]);
        if (i < start4 + s4) sum4.add_f64(v);
        else                 sum4.add_f64(v - kst_safe_roc_f64(prices[i - s4], prices[i - s4 - r4]));
      }

      if (i >= warm_line) {

        double k = __fma_rn(sum4.val_f64(), w4,
                  __fma_rn(sum3.val_f64(), w3,
                  __fma_rn(sum2.val_f64(), w2, sum1.val_f64() * inv1)));

        line_row[i] = k;


        ssum.add_f64(k);


        if (sig > 0 && (i - sig) >= warm_line) {
          ssum.add_f64(-line_row[i - sig]);
        }

        if (i >= warm_sig) {
          signal_row[i] = ssum.val_f64() * invSig;
        }
      }
    }
  }
}
extern "C" __global__
void kst_many_series_one_param_f64(const double* __restrict__ prices_tm,
                                   int num_series,
                                   int series_len,
                                   int s1, int s2, int s3, int s4,
                                   int r1, int r2, int r3, int r4,
                                   int sig,
                                   const int* __restrict__ first_valids,
                                   double* __restrict__ out_line_tm,
                                   double* __restrict__ out_signal_tm) {
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const double nn   = kst_qnan_f64();

  const double inv1  = (s1 > 0) ? (1.0 / double(s1)) : 0.0;
  const double w2    = (s2 > 0) ? (2.0 / double(s2)) : 0.0;
  const double w3    = (s3 > 0) ? (3.0 / double(s3)) : 0.0;
  const double w4    = (s4 > 0) ? (4.0 / double(s4)) : 0.0;
  const double invSig = (sig > 0) ? (1.0 / double(sig)) : 0.0;

  for (int s = tid; s < num_series; s += stride) {
    int fv = first_valids[s];
    if (fv < 0)       fv = 0;
    if (fv >= series_len) {
      for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + s;
        out_line_tm[idx] = nn;
        out_signal_tm[idx] = nn;
      }
      continue;
    }

    const int start1 = fv + r1;
    const int start2 = fv + r2;
    const int start3 = fv + r3;
    const int start4 = fv + r4;

    const int warm_line = max(max(start1 + s1 - 1, start2 + s2 - 1),
                              max(start3 + s3 - 1, start4 + s4 - 1));
    const int warm_sig  = warm_line + sig - 1;

    for (int t = 0; t < warm_line && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_line_tm[idx] = nn;
    }
    for (int t = 0; t < warm_sig && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_signal_tm[idx] = nn;
    }

    CompSum_f64 sum1, sum2, sum3, sum4, ssum;
    sum1.init_f64(); sum2.init_f64(); sum3.init_f64(); sum4.init_f64(); ssum.init_f64();

    for (int t = fv; t < series_len; ++t) {
      const int idx  = t * num_series + s;
      const double x  = prices_tm[idx];

      if (t >= start1) {
        const double v = kst_safe_roc_f64(x, prices_tm[(t - r1) * num_series + s]);
        if (t < start1 + s1) sum1.add_f64(v);
        else                 sum1.add_f64(v - kst_safe_roc_f64(prices_tm[(t - s1) * num_series + s],
                                                      prices_tm[(t - s1 - r1) * num_series + s]));
      }
      if (t >= start2) {
        const double v = kst_safe_roc_f64(x, prices_tm[(t - r2) * num_series + s]);
        if (t < start2 + s2) sum2.add_f64(v);
        else                 sum2.add_f64(v - kst_safe_roc_f64(prices_tm[(t - s2) * num_series + s],
                                                      prices_tm[(t - s2 - r2) * num_series + s]));
      }
      if (t >= start3) {
        const double v = kst_safe_roc_f64(x, prices_tm[(t - r3) * num_series + s]);
        if (t < start3 + s3) sum3.add_f64(v);
        else                 sum3.add_f64(v - kst_safe_roc_f64(prices_tm[(t - s3) * num_series + s],
                                                      prices_tm[(t - s3 - r3) * num_series + s]));
      }
      if (t >= start4) {
        const double v = kst_safe_roc_f64(x, prices_tm[(t - r4) * num_series + s]);
        if (t < start4 + s4) sum4.add_f64(v);
        else                 sum4.add_f64(v - kst_safe_roc_f64(prices_tm[(t - s4) * num_series + s],
                                                      prices_tm[(t - s4 - r4) * num_series + s]));
      }

      if (t >= warm_line) {
        double k = __fma_rn(sum4.val_f64(), w4,
                  __fma_rn(sum3.val_f64(), w3,
                  __fma_rn(sum2.val_f64(), w2, sum1.val_f64() * inv1)));
        out_line_tm[idx] = k;


        ssum.add_f64(k);
        if (sig > 0 && (t - sig) >= warm_line) {
          ssum.add_f64(-out_line_tm[(t - sig) * num_series + s]);
        }
        if (t >= warm_sig) {
          out_signal_tm[idx] = ssum.val_f64() * invSig;
        }
      }
    }
  }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * warmup / first-valid : kst_prepare, src/indicators/kst.rs:339-430.
//                            first = data.iter().position(|x| !x.is_nan())
//                            -> F64FirstValidRule::AllInputsNonNan
//   * arithmetic (dirty)   : kst_compute_into,         src/indicators/kst.rs:435
//   * arithmetic (clean)   : kst_compute_into_nonzero, src/indicators/kst.rs:649
//   * emitted column       : `line`. compute_kst_batch (cpu_batch.rs:15122)
//                            maps output_id "value" -> out.line (:15156).
//   * PERIOD-INVARIANT     : the CPU batch reads sma_period1..4, roc_period1..4
//                            and signal_period and NEVER a parameter named
//                            `period` (cpu_batch.rs:15128-15136), so a sweep of
//                            five periods produces five IDENTICAL CPU columns
//                            and this kernel emits five identical rows. The
//                            `periods` argument is read and discarded, not
//                            mapped onto one of the nine named windows --
//                            inventing that mapping would compute something the
//                            CPU never computes.
//
// WHY THE "all finite and non-zero" SCAN IS HERE AND IS NOT AN OPTIMISATION.
// The crate has TWO scalar paths and they are NOT bit-identical:
// kst_compute_into builds the fourth weight as w4 = 4.0 * (1.0 / s4) (:744 of
// the dirty path) while kst_compute_into_nonzero builds it as inv4 = 4.0 / s4
// (:744 of the clean path). Those differ by one rounding, and the CPU chooses
// between them (kst.rs:600) with exactly the predicate reproduced below.
// Picking one unconditionally would be wrong for whichever series took the
// other branch, so the kernel reproduces the branch.
//
// Everything else the two paths do is identical: safe_roc (kst.rs:558) reduces
// to ((x / p) - 1.0) * 100.0 whenever p != 0 and both are finite, which is
// precisely the clean path's precondition.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. fma() is the transliteration of Rust's mul_add -- ONE rounding,
// same as the CPU line -- and the whole translation unit is compiled with
// -prec-div=true -fmad=false (build.rs F64_LANE_SOURCES), so (x / p) is a
// correctly-rounded divide and a * b + c stays two roundings exactly as on the
// host.
// ===========================================================================

#define KST_NEO_S1 10
#define KST_NEO_S2 10
#define KST_NEO_S3 10
#define KST_NEO_S4 15
#define KST_NEO_R1 10
#define KST_NEO_R2 15
#define KST_NEO_R3 20
#define KST_NEO_R4 30
#define KST_NEO_SIG 9

static __device__ __forceinline__ double kst_neo_qnan() {
  // The exact bit pattern alloc_with_nan_prefix writes
  // (utilities/helpers.rs:116), not an arbitrary NaN.
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// safe_roc, kst.rs:558-564. Reproduced rounding for rounding: one divide, one
// subtract, one multiply -- NOT folded into an fma, because the CPU does not
// fold it either.
static __device__ __forceinline__ double kst_neo_safe_roc(double curr, double prev) {
  if (prev != 0.0 && isfinite(curr) && isfinite(prev)) {
    return ((curr / prev) - 1.0) * 100.0;
  }
  return 0.0;
}

// ring_update, kst.rs:573-581 and :773-781.
static __device__ __forceinline__ void kst_neo_ring(double* buf, int* idx, int cap,
                                                    double* sum, double v) {
  const double old = buf[*idx];
  *sum = (*sum) + (v - old);
  buf[*idx] = v;
  *idx += 1;
  if (*idx == cap) *idx = 0;
}

extern "C" __global__
void kst_neo_batch_f64(const double* __restrict__ prices,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out) {
  const int combo = blockIdx.x * blockDim.x + threadIdx.x;
  if (combo >= n_combos) return;
  (void)periods;  // PERIOD-INVARIANT -- see the header.

  if (n <= 0) return;
  double* __restrict__ row = out + (size_t)combo * (size_t)n;
  const double nn = kst_neo_qnan();

  int first = first_valid;
  if (first < 0) first = 0;

  const int s1 = KST_NEO_S1, s2 = KST_NEO_S2, s3 = KST_NEO_S3, s4 = KST_NEO_S4;
  const int r1 = KST_NEO_R1, r2 = KST_NEO_R2, r3 = KST_NEO_R3, r4 = KST_NEO_R4;
  const int sig = KST_NEO_SIG;

  const int warm1 = r1 + s1 - 1;
  const int warm2 = r2 + s2 - 1;
  const int warm3 = r3 + s3 - 1;
  const int warm4 = r4 + s4 - 1;
  int warm_line = warm1;
  if (warm2 > warm_line) warm_line = warm2;
  if (warm3 > warm_line) warm_line = warm3;
  if (warm4 > warm_line) warm_line = warm4;
  const int warm_sig = warm_line + sig - 1;

  // Every refusal kst_prepare makes (:373-412). When the CPU returns Err the
  // caller gets NO series at all, so the row is the NaN a missing column reads
  // as -- never a partially computed one.
  const int max_p = r4;  // the largest of the nine defaults
  const bool refused =
      first >= n || max_p > n || (n - first) < warm_line || warm_sig > n;
  if (refused) {
    for (int i = 0; i < n; ++i) row[i] = nn;
    return;
  }

  // kst.rs:600 -- data[first..].all(|v| v.is_finite() && *v != 0.0).
  bool clean = true;
  for (int i = first; i < n; ++i) {
    const double v = prices[i];
    if (!isfinite(v) || v == 0.0) { clean = false; break; }
  }

  const double inv1 = 1.0 / (double)s1;
  const double inv2 = 1.0 / (double)s2;
  const double inv3 = 1.0 / (double)s3;
  const double w2 = inv2 + inv2;
  const double w3 = inv3 + inv3 + inv3;
  // The one place the two CPU paths disagree.
  const double w4 = clean ? (4.0 / (double)s4) : (4.0 * (1.0 / (double)s4));

  const int start1 = first + r1;
  const int start2 = first + r2;
  const int start3 = first + r3;
  const int start4 = first + r4;
  const int start_line = first + warm_line;

  double b1[KST_NEO_S1], b2[KST_NEO_S2], b3[KST_NEO_S3], b4[KST_NEO_S4];
  for (int k = 0; k < s1; ++k) b1[k] = 0.0;
  for (int k = 0; k < s2; ++k) b2[k] = 0.0;
  for (int k = 0; k < s3; ++k) b3[k] = 0.0;
  for (int k = 0; k < s4; ++k) b4[k] = 0.0;
  int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
  double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

  const int nan_end = start_line < n ? start_line : n;
  for (int i = 0; i < nan_end; ++i) row[i] = nn;

  for (int i = first; i < n; ++i) {
    const double x = prices[i];

    if (i >= start1) {
      const double p = prices[i - r1];
      const double v = clean ? (((x / p) - 1.0) * 100.0) : kst_neo_safe_roc(x, p);
      kst_neo_ring(b1, &i1, s1, &sum1, v);
    }
    if (i >= start2) {
      const double p = prices[i - r2];
      const double v = clean ? (((x / p) - 1.0) * 100.0) : kst_neo_safe_roc(x, p);
      kst_neo_ring(b2, &i2, s2, &sum2, v);
    }
    if (i >= start3) {
      const double p = prices[i - r3];
      const double v = clean ? (((x / p) - 1.0) * 100.0) : kst_neo_safe_roc(x, p);
      kst_neo_ring(b3, &i3, s3, &sum3, v);
    }
    if (i >= start4) {
      const double p = prices[i - r4];
      const double v = clean ? (((x / p) - 1.0) * 100.0) : kst_neo_safe_roc(x, p);
      kst_neo_ring(b4, &i4, s4, &sum4, v);
    }

    if (i < start_line) continue;

    // kst.rs:607 and :806 -- sum1 OUTERMOST. The nesting is load-bearing: it is
    // three fmas whose order sets the rounding, and the kst_batch_f64 already in
    // this file nests them the other way round.
    row[i] = fma(sum1, inv1, fma(sum2, w2, fma(sum3, w3, sum4 * w4)));
  }
}
