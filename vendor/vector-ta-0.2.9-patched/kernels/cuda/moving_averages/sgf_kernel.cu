#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef SGF_BLOCK_X
#define SGF_BLOCK_X 128
#endif
#ifndef SGF_SERIES_PER_BLOCK
#define SGF_SERIES_PER_BLOCK 4
#endif
#ifndef SGF_MAX_PERIOD
#define SGF_MAX_PERIOD 4096
#endif
#ifndef SGF_USE_CONST_WEIGHTS
#define SGF_USE_CONST_WEIGHTS 1
#endif

#if SGF_USE_CONST_WEIGHTS
__constant__ float c_sgf_weights[SGF_MAX_PERIOD];
#endif

extern "C" __global__
void sgf_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ weights_flat,
                   const int* __restrict__ periods,
                   const int* __restrict__ warm_indices,
                   int series_len,
                   int n_combos,
                   int max_period,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int t = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= series_len) return;

    const int warm = warm_indices[combo];
    const int out_base = combo * series_len;

    extern __shared__ float smem[];
    float* w_sh = smem;
    float* tile = smem + max_period;

    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        w_sh[k] = weights_flat[combo * max_period + k];
    }
    __syncthreads();

    const int tile_start = int(blockIdx.x * blockDim.x);
    const int load_begin = max(tile_start - (period - 1), 0);
    const int load_end = min(tile_start + int(blockDim.x) - 1, series_len - 1);
    const int load_len = max(0, load_end - load_begin + 1);
    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        tile[i] = prices[load_begin + i];
    }
    __syncthreads();

    if (t < warm) {
        out[out_base + t] = NAN;
        return;
    }

    const int start = t - period + 1;
    const int tile_off = start - load_begin;
    float acc = 0.0f;
    for (int k = 0; k < period; ++k) {
        acc = fmaf(tile[tile_off + k], w_sh[k], acc);
    }
    out[out_base + t] = acc;
}

extern "C" __global__
void sgf_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const float* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {
    const int series_block_base = int(blockIdx.y * blockDim.y);
    const int s_local = int(threadIdx.y);
    const int s = series_block_base + s_local;
    if (s >= num_series) return;

    extern __shared__ float smem[];
    float* tile = smem;
#if !SGF_USE_CONST_WEIGHTS
    float* w_sh = smem;
    tile = w_sh + period;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        w_sh[k] = weights[k];
    }
    __syncthreads();
#endif

    const int tile_t0 = int(blockIdx.x * blockDim.x);
    const int local_t = int(threadIdx.x);
    const int t = tile_t0 + local_t;
    const int warm = first_valids[s] + period - 1;

    const int in_begin = max(tile_t0 - (period - 1), 0);
    const int in_end = min(tile_t0 + int(blockDim.x) - 1, series_len - 1);
    const int load_len = max(0, in_end - in_begin + 1);
    const int tile_span = load_len * int(blockDim.y);

    int linear = local_t * int(blockDim.y) + s_local;
    for (int idx = linear; idx < tile_span; idx += int(blockDim.x * blockDim.y)) {
        int dt = idx / int(blockDim.y);
        int ss = idx % int(blockDim.y);
        int gs = series_block_base + ss;
        if (gs < num_series) {
            tile[idx] = prices_tm[(in_begin + dt) * num_series + gs];
        }
    }
    __syncthreads();

    if (t >= series_len) return;
    if (t < warm) {
        out_tm[t * num_series + s] = NAN;
        return;
    }

    const int start_t = t - period + 1;
    const int base = (start_t - in_begin) * int(blockDim.y) + s_local;
    float acc = 0.0f;
    for (int k = 0; k < period; ++k) {
#if SGF_USE_CONST_WEIGHTS
        acc = fmaf(tile[base + k * int(blockDim.y)], c_sgf_weights[k], acc);
#else
        acc = fmaf(tile[base + k * int(blockDim.y)], w_sh[k], acc);
#endif
    }
    out_tm[t * num_series + s] = acc;
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

__constant__ double c_sgf_weights_f64[SGF_MAX_PERIOD];
extern "C" __global__
void sgf_batch_f64(const double* __restrict__ prices,
                   const double* __restrict__ weights_flat,
                   const int* __restrict__ periods,
                   const int* __restrict__ warm_indices,
                   int series_len,
                   int n_combos,
                   int max_period,
                   double* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int t = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= series_len) return;

    const int warm = warm_indices[combo];
    const int out_base = combo * series_len;

    extern __shared__ double smem_f64[];
    double* w_sh = smem_f64;
    double* tile = smem_f64 + max_period;

    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        w_sh[k] = weights_flat[combo * max_period + k];
    }
    __syncthreads();

    const int tile_start = int(blockIdx.x * blockDim.x);
    const int load_begin = max(tile_start - (period - 1), 0);
    const int load_end = min(tile_start + int(blockDim.x) - 1, series_len - 1);
    const int load_len = max(0, load_end - load_begin + 1);
    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        tile[i] = prices[load_begin + i];
    }
    __syncthreads();

    if (t < warm) {
        out[out_base + t] = NAN;
        return;
    }

    const int start = t - period + 1;
    const int tile_off = start - load_begin;
    double acc = 0.0;
    for (int k = 0; k < period; ++k) {
        acc = fma(tile[tile_off + k], w_sh[k], acc);
    }
    out[out_base + t] = acc;
}
extern "C" __global__
void sgf_multi_series_one_param_f64(const double* __restrict__ prices_tm,
                                    const double* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    double* __restrict__ out_tm) {
    const int series_block_base = int(blockIdx.y * blockDim.y);
    const int s_local = int(threadIdx.y);
    const int s = series_block_base + s_local;
    if (s >= num_series) return;

    extern __shared__ double smem_f64[];
    double* tile = smem_f64;
#if !SGF_USE_CONST_WEIGHTS
    double* w_sh = smem_f64;
    tile = w_sh + period;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        w_sh[k] = weights[k];
    }
    __syncthreads();
#endif

    const int tile_t0 = int(blockIdx.x * blockDim.x);
    const int local_t = int(threadIdx.x);
    const int t = tile_t0 + local_t;
    const int warm = first_valids[s] + period - 1;

    const int in_begin = max(tile_t0 - (period - 1), 0);
    const int in_end = min(tile_t0 + int(blockDim.x) - 1, series_len - 1);
    const int load_len = max(0, in_end - in_begin + 1);
    const int tile_span = load_len * int(blockDim.y);

    int linear = local_t * int(blockDim.y) + s_local;
    for (int idx = linear; idx < tile_span; idx += int(blockDim.x * blockDim.y)) {
        int dt = idx / int(blockDim.y);
        int ss = idx % int(blockDim.y);
        int gs = series_block_base + ss;
        if (gs < num_series) {
            tile[idx] = prices_tm[(in_begin + dt) * num_series + gs];
        }
    }
    __syncthreads();

    if (t >= series_len) return;
    if (t < warm) {
        out_tm[t * num_series + s] = NAN;
        return;
    }

    const int start_t = t - period + 1;
    const int base = (start_t - in_begin) * int(blockDim.y) + s_local;
    double acc = 0.0;
    for (int k = 0; k < period; ++k) {
#if SGF_USE_CONST_WEIGHTS
        acc = fma(tile[base + k * int(blockDim.y)], c_sgf_weights_f64[k], acc);
#else
        acc = fma(tile[base + k * int(blockDim.y)], w_sh[k], acc);
#endif
    }
    out_tm[t * num_series + s] = acc;
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                                       sgf
 * ---------------------------------------------------------------------------
 * CPU reference: `sgf_compute_into`, src/indicators/moving_averages/sgf.rs:570,
 * with its window dot `sgf_dot` (:479) and the weight construction
 * `build_endpoint_sgf_weights` (:331) reached through `sgf_prepare` (:429).
 * `period` is the swept parameter; `poly_order` is fixed at its default 2
 * (SgfParams::default, :86-87), so the normal system is 3x3.
 *
 * WHY THE WEIGHTS ARE BUILT IN THE KERNEL. The entry point already in this
 * file, `sgf_multi_series_one_param_f64`, takes the weight vector as a device
 * pointer -- the HOST having solved the least-squares system for it. This lane
 * launches (series, n, periods, n_combos, first_valid, out): there is no weight
 * buffer in that signature, and there must not be one, because each row sweeps
 * a DIFFERENT period and therefore a different weight vector. So the thread
 * solves its own 3x3 system. That is nine doubles of state, not an allocation.
 *
 * NO PER-THREAD WEIGHT ARRAY, so no `max_period` bound.
 * `build_endpoint_sgf_weights` materialises `effective` weights only to
 * normalise them by their sum; the sum is accumulated in one pass here and each
 * weight is then RECOMPUTED from the same three coefficients, by the same
 * `weight += coef * power; power *= x` sequence the CPU uses (:360-368). Same
 * operations in the same order gives the same double. NEVER-OOM by
 * construction.
 *
 * `effective` is the CPU effective_period (:248): an EVEN period is reduced by
 * one. The window, however, is still `period` bars wide (:604), and `sgf_dot`
 * walks only `weights.len() == effective` of them (:485-497). For an even
 * period that means the NEWEST bar of the window is never read. That is what
 * the crate computes, so it is what this kernel computes; "fixing" it here
 * would put the GPU and the CPU on different series.
 *
 * ACCUMULATION ORDER IS LOAD-BEARING and is reproduced exactly: four
 * accumulators strided by four, the tail folded into acc0, and the final
 * (acc0 + acc1) + (acc2 + acc3) (:499). The crate hand-unrolled `sgf_dot_21`
 * (:502) is the SAME association -- acc0 takes 0,4,8,12,16 then 20; acc1 takes
 * 1,5,9,13,17; and so on -- so period 21 needs no special case here, and there
 * is one oracle rather than two.
 *
 * `first_valid` is the lane AllInputsNonNan over the close slice, which is the
 * CPU position(|x| !x.is_nan()) (:437-440).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* SgfParams::default -> poly_order 2, sgf.rs:87. order = poly_order + 1. */
#define NEO_SGF_ORDER 3

/* solve_linear_system, sgf.rs:281 -- Gauss-Jordan with partial pivoting on the
 * 3x3 normal matrix. Transcribed statement for statement, including the
 * `factor == 0.0` skip (:319-321), which changes WHICH roundings happen, and
 * the `best_abs <= 1e-15` singular guard (:292). That constant is already an
 * f64 guard in an f64 routine -- it is NOT an f32 epsilon and must not be
 * resized. */
__device__ __forceinline__
static bool neo_sgf_solve3(double a[NEO_SGF_ORDER * NEO_SGF_ORDER],
                           double b[NEO_SGF_ORDER])
{
    const int n = NEO_SGF_ORDER;
    for (int pivot = 0; pivot < n; ++pivot) {
        int    best_row = pivot;
        double best_abs = fabs(a[pivot * n + pivot]);
        for (int row = pivot + 1; row < n; ++row) {
            const double cand = fabs(a[row * n + pivot]);
            if (cand > best_abs) { best_abs = cand; best_row = row; }
        }
        if (best_abs <= 1e-15) return false;

        if (best_row != pivot) {
            for (int col = pivot; col < n; ++col) {
                const double t = a[pivot * n + col];
                a[pivot * n + col] = a[best_row * n + col];
                a[best_row * n + col] = t;
            }
            const double tb = b[pivot]; b[pivot] = b[best_row]; b[best_row] = tb;
        }

        const double pivot_val = a[pivot * n + pivot];
        for (int col = pivot; col < n; ++col) a[pivot * n + col] /= pivot_val;
        b[pivot] /= pivot_val;

        for (int row = 0; row < n; ++row) {
            if (row == pivot) continue;
            const double factor = a[row * n + pivot];
            if (factor == 0.0) continue;
            for (int col = pivot; col < n; ++col) {
                a[row * n + col] -= factor * a[pivot * n + col];
            }
            b[row] -= factor * b[pivot];
        }
    }
    return true;
}

/* One un-normalised weight, sgf.rs:358-366. x is `i - (effective - 1)`. */
__device__ __forceinline__
static double neo_sgf_weight_raw(int i, int effective, const double c[NEO_SGF_ORDER])
{
    const double x = (double)i - (double)(effective - 1);
    double power  = 1.0;
    double weight = 0.0;
    for (int k = 0; k < NEO_SGF_ORDER; ++k) {
        weight += c[k] * power;
        power  *= x;
    }
    return weight;
}

extern "C" __global__
void sgf_neo_batch_f64(const double* __restrict__ data,
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

    /* effective_period, sgf.rs:248-256. */
    const int effective =
        (period <= 1) ? period : (((period & 1) == 0) ? period - 1 : period);

    /* validate_period_and_order, sgf.rs:259-279, with len = data.len(). */
    if (period < 3 || effective < 3 || effective > n) return;
    if (NEO_SGF_ORDER - 1 >= effective) return;      /* poly_order < effective */
    /* sgf.rs:445-450 -- NotEnoughValidData leaves the row NaN. */
    if (n - first_valid < period) return;

    /* build_endpoint_sgf_weights, sgf.rs:336-354 -- the Gram matrix. */
    double gram[NEO_SGF_ORDER * NEO_SGF_ORDER];
    for (int k = 0; k < NEO_SGF_ORDER * NEO_SGF_ORDER; ++k) gram[k] = 0.0;

    for (int i = 0; i < effective; ++i) {
        const double x = (double)i - (double)(effective - 1);
        double powers[NEO_SGF_ORDER];
        powers[0] = 1.0;
        for (int k = 1; k < NEO_SGF_ORDER; ++k) powers[k] = powers[k - 1] * x;
        for (int row = 0; row < NEO_SGF_ORDER; ++row) {
            for (int col = 0; col < NEO_SGF_ORDER; ++col) {
                gram[row * NEO_SGF_ORDER + col] += powers[row] * powers[col];
            }
        }
    }

    double coeffs[NEO_SGF_ORDER];
    coeffs[0] = 1.0;                                 /* rhs[0] = 1, :353 */
    for (int k = 1; k < NEO_SGF_ORDER; ++k) coeffs[k] = 0.0;
    if (!neo_sgf_solve3(gram, coeffs)) return;       /* SingularFit -> NaN row */

    /* :356-368 -- the normalising sum, in ascending i. */
    double wsum = 0.0;
    for (int i = 0; i < effective; ++i) {
        wsum += neo_sgf_weight_raw(i, effective, coeffs);
    }
    const bool normalise = (wsum != 0.0);            /* :370 */

    /* sgf_compute_into, :578 and :603-606. */
    const int start = first_valid + period - 1;

    for (int idx = start; idx < n; ++idx) {
        const int from = idx + 1 - period;

        /* sgf_dot, :479-500 -- four accumulators strided by four, tail into
         * acc0, then (acc0 + acc1) + (acc2 + acc3). */
        double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
        int j = 0;
        while (j + 3 < effective) {
            double w0 = neo_sgf_weight_raw(j,     effective, coeffs);
            double w1 = neo_sgf_weight_raw(j + 1, effective, coeffs);
            double w2 = neo_sgf_weight_raw(j + 2, effective, coeffs);
            double w3 = neo_sgf_weight_raw(j + 3, effective, coeffs);
            if (normalise) { w0 /= wsum; w1 /= wsum; w2 /= wsum; w3 /= wsum; }
            acc0 += data[from + j]     * w0;
            acc1 += data[from + j + 1] * w1;
            acc2 += data[from + j + 2] * w2;
            acc3 += data[from + j + 3] * w3;
            j += 4;
        }
        while (j < effective) {
            double w = neo_sgf_weight_raw(j, effective, coeffs);
            if (normalise) w /= wsum;
            acc0 += data[from + j] * w;
            j += 1;
        }

        o[idx] = (acc0 + acc1) + (acc2 + acc3);
    }
}
