#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  #define LRDG(ptr) __ldg(ptr)
#else
  #define LRDG(ptr) (*(ptr))
#endif


static __device__ __forceinline__ float warp_broadcast_ldg(const float* addr) {
    unsigned mask = __activemask();
    int leader   = __ffs(mask) - 1;
    int lane     = threadIdx.x & 31;
    float v = 0.0f;
    if (lane == leader) v = LRDG(addr);
    return __shfl_sync(mask, v, leader);
}

extern "C" __global__
void lrsi_build_hl2_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        int len,
                        float* __restrict__ out_prices) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < len;
         idx += blockDim.x * gridDim.x) {
        const float h = high[idx];
        const float l = low[idx];
        out_prices[idx] = 0.5f * (h + l);
    }
}


static __device__ __forceinline__
void laguerre4_step(float p, float alpha, float gamma, float mgamma,
                    float &l0, float &l1, float &l2, float &l3,
                    float &t0, float &t1, float &t2, float &t3) {

    t0 = fmaf(alpha, (p - l0), l0);
    t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
    t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
    t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));
    l0 = t0; l1 = t1; l2 = t2; l3 = t3;
}


extern "C" __global__
void lrsi_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ alphas,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {

    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x) {
        const int base = combo * series_len;


        if (first_valid < 0 || first_valid >= series_len) {

            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }

        const float alpha = alphas[combo];
        if (!(alpha > 0.0f && alpha < 1.0f)) {
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }
        const float gamma  = 1.0f - alpha;
        const float mgamma = -gamma;

        const int warm = first_valid + 3;
        if (warm >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }


        for (int t = 0; t < warm; ++t) out[base + t] = NAN;


        const float p0 = prices[first_valid];
        float l0 = p0, l1 = p0, l2 = p0, l3 = p0;


        for (int t = first_valid + 1; t < warm; ++t) {
            const float p = warp_broadcast_ldg(prices + t);
            if (isnan(p)) continue;
            const float t0 = fmaf(alpha, (p - l0), l0);
            const float t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
            const float t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
            const float t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));
            l0 = t0; l1 = t1; l2 = t2; l3 = t3;
        }


        for (int t = warm; t < series_len; ++t) {
            const float p = warp_broadcast_ldg(prices + t);
            if (isnan(p)) { out[base + t] = NAN; continue; }

            const float t0 = fmaf(alpha, (p - l0), l0);
            const float t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
            const float t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
            const float t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));

            l0 = t0; l1 = t1; l2 = t2; l3 = t3;

            const float d01 = t0 - t1;
            const float d12 = t1 - t2;
            const float d23 = t2 - t3;
            const float a01 = fabsf(d01);
            const float a12 = fabsf(d12);
            const float a23 = fabsf(d23);
            const float sum_abs = a01 + a12 + a23;
            if (sum_abs <= FLT_EPSILON) {
                out[base + t] = 0.0f;
            } else {
                const float cu = 0.5f * (d01 + a01 + d12 + a12 + d23 + a23);
                out[base + t] = cu / sum_abs;
            }
        }
    }
}


extern "C" __global__
void lrsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    float alpha,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {

    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_series;
         s += blockDim.x * gridDim.x) {
        if (!(alpha > 0.0f && alpha < 1.0f)) {

            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + s] = NAN;
            continue;
        }

        const float gamma  = 1.0f - alpha;
        const float mgamma = -gamma;

        const int first = max(0, first_valids[s]);
        const int warm  = first + 3;


        if (first >= series_len || warm >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + s] = NAN;
            continue;
        }

        const int cols = num_series;


        for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = NAN;


        const int idx0 = first * cols + s;
        float l0 = prices_tm[idx0];
        float l1 = l0, l2 = l0, l3 = l0;


        for (int t = first + 1; t < warm; ++t) {
            const float p = prices_tm[t * cols + s];
            if (isnan(p)) continue;
            float t0, t1, t2, t3;
            laguerre4_step(p, alpha, gamma, mgamma, l0, l1, l2, l3, t0, t1, t2, t3);
        }


        for (int t = warm; t < series_len; ++t) {
            const int idx = t * cols + s;
            const float p = prices_tm[idx];
            if (isnan(p)) { out_tm[idx] = NAN; continue; }

            float t0, t1, t2, t3;
            laguerre4_step(p, alpha, gamma, mgamma, l0, l1, l2, l3, t0, t1, t2, t3);

            const float d01 = t0 - t1;
            const float d12 = t1 - t2;
            const float d23 = t2 - t3;
            const float a01 = fabsf(d01);
            const float a12 = fabsf(d12);
            const float a23 = fabsf(d23);
            const float sum_abs = a01 + a12 + a23;

            if (sum_abs <= FLT_EPSILON) {
                out_tm[idx] = 0.0f;
            } else {
                const float cu = 0.5f * (d01 + a01 + d12 + a12 + d23 + a23);
                out_tm[idx]   = cu / sum_abs;
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

#include <double.h>
static __device__ __forceinline__ double warp_broadcast_ldg_f64(const double* addr) {
    unsigned mask = __activemask();
    int leader   = __ffs(mask) - 1;
    int lane     = threadIdx.x & 31;
    double v = 0.0;
    if (lane == leader) v = LRDG(addr);
    return __shfl_sync(mask, v, leader);
}
extern "C" __global__
void lrsi_build_hl2_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        int len,
                        double* __restrict__ out_prices) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < len;
         idx += blockDim.x * gridDim.x) {
        const double h = high[idx];
        const double l = low[idx];
        out_prices[idx] = 0.5 * (h + l);
    }
}
static __device__ __forceinline__
void laguerre4_step_f64(double p, double alpha, double gamma, double mgamma,
                    double &l0, double &l1, double &l2, double &l3,
                    double &t0, double &t1, double &t2, double &t3) {

    t0 = fma(alpha, (p - l0), l0);
    t1 = fma(gamma, l1, fma(mgamma, t0, l0));
    t2 = fma(gamma, l2, fma(mgamma, t1, l1));
    t3 = fma(gamma, l3, fma(mgamma, t2, l2));
    l0 = t0; l1 = t1; l2 = t2; l3 = t3;
}
extern "C" __global__
void lrsi_batch_f64(const double* __restrict__ prices,
                    const double* __restrict__ alphas,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    double* __restrict__ out) {

    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x) {
        const int base = combo * series_len;


        if (first_valid < 0 || first_valid >= series_len) {

            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }

        const double alpha = alphas[combo];
        if (!(alpha > 0.0 && alpha < 1.0)) {
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }
        const double gamma  = 1.0 - alpha;
        const double mgamma = -gamma;

        const int warm = first_valid + 3;
        if (warm >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }


        for (int t = 0; t < warm; ++t) out[base + t] = NAN;


        const double p0 = prices[first_valid];
        double l0 = p0, l1 = p0, l2 = p0, l3 = p0;


        for (int t = first_valid + 1; t < warm; ++t) {
            const double p = warp_broadcast_ldg_f64(prices + t);
            if (isnan(p)) continue;
            const double t0 = fma(alpha, (p - l0), l0);
            const double t1 = fma(gamma, l1, fma(mgamma, t0, l0));
            const double t2 = fma(gamma, l2, fma(mgamma, t1, l1));
            const double t3 = fma(gamma, l3, fma(mgamma, t2, l2));
            l0 = t0; l1 = t1; l2 = t2; l3 = t3;
        }


        for (int t = warm; t < series_len; ++t) {
            const double p = warp_broadcast_ldg_f64(prices + t);
            if (isnan(p)) { out[base + t] = NAN; continue; }

            const double t0 = fma(alpha, (p - l0), l0);
            const double t1 = fma(gamma, l1, fma(mgamma, t0, l0));
            const double t2 = fma(gamma, l2, fma(mgamma, t1, l1));
            const double t3 = fma(gamma, l3, fma(mgamma, t2, l2));

            l0 = t0; l1 = t1; l2 = t2; l3 = t3;

            const double d01 = t0 - t1;
            const double d12 = t1 - t2;
            const double d23 = t2 - t3;
            const double a01 = fabs(d01);
            const double a12 = fabs(d12);
            const double a23 = fabs(d23);
            const double sum_abs = a01 + a12 + a23;
            if (sum_abs <= DBL_EPSILON) {
                out[base + t] = 0.0;
            } else {
                const double cu = 0.5 * (d01 + a01 + d12 + a12 + d23 + a23);
                out[base + t] = cu / sum_abs;
            }
        }
    }
}
extern "C" __global__
void lrsi_many_series_one_param_f64(const double* __restrict__ prices_tm,
                                    double alpha,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    double* __restrict__ out_tm) {

    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_series;
         s += blockDim.x * gridDim.x) {
        if (!(alpha > 0.0 && alpha < 1.0)) {

            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + s] = NAN;
            continue;
        }

        const double gamma  = 1.0 - alpha;
        const double mgamma = -gamma;

        const int first = max(0, first_valids[s]);
        const int warm  = first + 3;


        if (first >= series_len || warm >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + s] = NAN;
            continue;
        }

        const int cols = num_series;


        for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = NAN;


        const int idx0 = first * cols + s;
        double l0 = prices_tm[idx0];
        double l1 = l0, l2 = l0, l3 = l0;


        for (int t = first + 1; t < warm; ++t) {
            const double p = prices_tm[t * cols + s];
            if (isnan(p)) continue;
            double t0, t1, t2, t3;
            laguerre4_step_f64(p, alpha, gamma, mgamma, l0, l1, l2, l3, t0, t1, t2, t3);
        }


        for (int t = warm; t < series_len; ++t) {
            const int idx = t * cols + s;
            const double p = prices_tm[idx];
            if (isnan(p)) { out_tm[idx] = NAN; continue; }

            double t0, t1, t2, t3;
            laguerre4_step_f64(p, alpha, gamma, mgamma, l0, l1, l2, l3, t0, t1, t2, t3);

            const double d01 = t0 - t1;
            const double d12 = t1 - t2;
            const double d23 = t2 - t3;
            const double a01 = fabs(d01);
            const double a12 = fabs(d12);
            const double a23 = fabs(d23);
            const double sum_abs = a01 + a12 + a23;

            if (sum_abs <= DBL_EPSILON) {
                out_tm[idx] = 0.0;
            } else {
                const double cu = 0.5 * (d01 + a01 + d12 + a12 + d23 + a23);
                out_tm[idx]   = cu / sum_abs;
            }
        }
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                                      lrsi
 * ---------------------------------------------------------------------------
 * CPU reference: `lrsi_scalar_hl`, src/indicators/lrsi.rs:397, reached through
 * `lrsi_with_kernel` (:165) whose `Kernel::Auto` resolves to `Kernel::Scalar`
 * (:225) -- lrsi has no batch kernel at all (:227-232 rejects every *Batch
 * variant), so the scalar path is the ONLY oracle and there is no
 * scalar-vs-AVX seed disagreement to settle here.
 *
 * PERIOD-INVARIANT. `LrsiParams` carries a single `alpha`, default 0.2
 * (cpu_batch.rs:3481). There is no period, so `periods` is unread; the lane's
 * sweep produces one identical row per requested period, which is what the CPU
 * batch does for the same combo list.
 *
 * INPUT is (high, low), NOT a precomputed price series. The other f64 entry
 * point already in this file, `lrsi_batch_f64` (:270), takes `prices` -- the
 * host having formed hl2 for it. This lane takes the two series the CPU takes
 * and forms `(high + low) * 0.5` per bar exactly where the CPU does (:416), so
 * no host-side pass exists to disagree with.
 *
 * SEQUENTIAL, one thread per column: four carried Laguerre stages
 * (l0..l3, :410-413) each written with `mul_add` on the CPU. Every one of the
 * four is reproduced as a single `fma`, in the same order and with the same
 * operands:
 *     t0 = fma(p - l0, alpha, l0)                       lrsi.rs:425
 *     t1 = fma(gamma, l1, fma(mgamma, t0, l0))          lrsi.rs:426
 *     t2 = fma(gamma, l2, fma(mgamma, t1, l1))          lrsi.rs:427
 *     t3 = fma(gamma, l3, fma(mgamma, t2, l2))          lrsi.rs:428
 * Expanding any of these into multiply-then-add would add a rounding per stage
 * per bar, and these values feed a threshold comparison.
 *
 * EPSILON: the CPU guard is `sum_abs <= f64::EPSILON` (:442), so this uses
 * `DBL_EPSILON`. The `FLT_EPSILON` at :124 and :199 of this file belongs to the
 * f32 entry points and is correct THERE; copying it into an f64 kernel would be
 * a guard sized 2^29 times too large.
 *
 * NaN: a NaN price mid-stream is written through as NaN when past the warmup
 * and the four stages are LEFT UNCHANGED (:418-423) -- the recurrence skips the
 * bar rather than poisoning. `v.min(1.0).max(0.0)` (:448) is f64::min/max,
 * which return the non-NaN operand, so they become `fmin`/`fmax`; an if-chain
 * would let a NaN survive the clamp.
 *
 * `first_valid` is the lane's `AllInputsNonNan` over (high, low), which is the
 * CPU's scan for the first bar whose `(high + low) / 2.0` is not NaN (:206-213)
 * -- (h + l) / 2 is NaN exactly when h or l is.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* LrsiParams default alpha, cpu_batch.rs:3481. */
#define NEO_LRSI_ALPHA 0.2

extern "C" __global__
void lrsi_neo_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;   /* lrsi has no period -- see the header above. */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    if (first_valid < 0 || first_valid >= n) return;
    /* lrsi.rs:216-221 -- fewer than four bars after first_valid is an error on
     * the CPU, so the row stays NaN rather than inventing a value. */
    if (n - first_valid < 4) return;

    const double alpha  = NEO_LRSI_ALPHA;
    const double gamma  = 1.0 - alpha;
    const double mgamma = -gamma;
    const int    warm   = first_valid + 3;          /* lrsi.rs:407 */

    /* lrsi.rs:409-413 -- all four stages seeded from the first hl2. */
    const double first_price = (high[first_valid] + low[first_valid]) * 0.5;
    double l0 = first_price;
    double l1 = first_price;
    double l2 = first_price;
    double l3 = first_price;

    for (int i = first_valid + 1; i < n; ++i) {
        const double p = (high[i] + low[i]) * 0.5;

        if (isnan(p)) {
            /* lrsi.rs:418-423 -- emit NaN past the warmup, carry l0..l3. */
            if (i >= warm) o[i] = NEO_F64_NAN;
            continue;
        }

        const double t0 = fma(p - l0, alpha, l0);
        const double t1 = fma(gamma, l1, fma(mgamma, t0, l0));
        const double t2 = fma(gamma, l2, fma(mgamma, t1, l1));
        const double t3 = fma(gamma, l3, fma(mgamma, t2, l2));

        if (i >= warm) {
            const double d01 = t0 - t1;
            const double d12 = t1 - t2;
            const double d23 = t2 - t3;

            const double a01 = fabs(d01);
            const double a12 = fabs(d12);
            const double a23 = fabs(d23);

            /* lrsi.rs:439-440 -- association reproduced literally. */
            const double sum_abs = a01 + a12 + a23;
            const double cu = 0.5 * (d01 + a01 + d12 + a12 + d23 + a23);

            const double v = (sum_abs <= DBL_EPSILON) ? 0.0 : (cu / sum_abs);
            o[i] = fmax(fmin(v, 1.0), 0.0);   /* lrsi.rs:448 */
        }

        l0 = t0;
        l1 = t1;
        l2 = t2;
        l3 = t3;
    }
}
