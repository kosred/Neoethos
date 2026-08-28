#include <cuda_runtime.h>

#ifndef QS_NAN
#define QS_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__ void qstick_build_prefix_serial_f32(
    const float* __restrict__ open,
    const float* __restrict__ close,
    int len,
    int first_valid,
    float* __restrict__ prefix_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    prefix_out[0] = 0.0f;
    double acc = 0.0;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            acc += static_cast<double>(close[i]) - static_cast<double>(open[i]);
        }
        prefix_out[i + 1] = static_cast<float>(acc);
    }
}

extern "C" __global__ void qstick_batch_prefix_f32(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (UNLIKELY(period <= 0)) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float inv_p = 1.0f / static_cast<float>(period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        if (t < warm) {
            out[row_off + t] = QS_NAN;
        } else {
            const int t1 = t + 1;
            int start = t1 - period; if (start < 0) start = 0;
            const float sum = prefix_diff[t1] - prefix_diff[start];
            out[row_off + t] = sum * inv_p;
        }
        t += stride;
    }
}

template<int TILE>
__device__ __forceinline__ void qstick_batch_prefix_tiled_impl(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int period = periods[combo];
    if (UNLIKELY(period <= 0)) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float inv_p = 1.0f / static_cast<float>(period);

    const int t0 = blockIdx.x * TILE;
    const int t = t0 + threadIdx.x;
    if (t >= len) return;

    if (t < warm) {
        out[row_off + t] = QS_NAN;
        return;
    }
    const int t1 = t + 1;
    int start = t1 - period; if (start < 0) start = 0;
    const float sum = prefix_diff[t1] - prefix_diff[start];
    out[row_off + t] = sum * inv_p;
}

extern "C" __global__ void qstick_batch_prefix_tiled_f32_tile128(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    qstick_batch_prefix_tiled_impl<128>(prefix_diff, len, first_valid, periods, n_combos, out);
}

extern "C" __global__ void qstick_batch_prefix_tiled_f32_tile256(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    qstick_batch_prefix_tiled_impl<256>(prefix_diff, len, first_valid, periods, n_combos, out);
}


extern "C" __global__ void qstick_many_series_one_param_f32(
    const float* __restrict__ prefix_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (UNLIKELY(period <= 0)) return;

    const int warm = first_valids[series] + period - 1;
    const int stride = num_series;
    const float inv_p = 1.0f / static_cast<float>(period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + series;
        if (t < warm) {
            out_tm[out_idx] = QS_NAN;
        } else {
            const int t1 = t + 1;
            int start = t1 - period; if (start < 0) start = 0;
            const int p_idx = t1 * stride + series;
            const int s_idx = start * stride + series;
            const float sum = prefix_tm[p_idx] - prefix_tm[s_idx];
            out_tm[out_idx] = sum * inv_p;
        }
        t += step;
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `qstick.rs::qstick_scalar` (l.363). Inputs are (open, close).
//   warm  = first_valid + period - 1
//   seed  = sum of (close-open) over [first_valid, first_valid+period), summed
//           in GROUPS OF FOUR — `sum += (c0-o0) + (c1-o1) + (c2-o2) + (c3-o3)`
//           is ONE add into `sum` of a value that was itself associated
//           ((a+b)+c)+d. A flat one-at-a-time loop is a DIFFERENT summation
//           tree and a different `sum`, so the group-of-four shape is
//           reproduced here literally.
//   roll  = `sum = (sum + (c_new - o_new)) - (c_old - o_old)` — note the
//           parenthesisation: add first, then subtract. Two roundings.
//   out   = sum * inv_p, with inv_p = 1.0 / period (reciprocal, not divide).
//   period == 1 is a separate branch that emits `close[i] - open[i]` from
//   `first_valid` with no warmup.
//
// f32 -> f64 audit: every pointer and local widened; `__int_as_float` NaN ->
// f64 quiet-NaN bit pattern; no `fmaf`/`__fmaf_rn`/`__fdividef` survives; no
// epsilon in this indicator; no min/max chain, so no fmax/fmin substitution is
// required. The f32 file precomputed a prefix sum on the host and read it back
// — a prefix-sum reformulation has a different rounding from the CPU's rolling
// accumulator, so this kernel keeps the rolling form.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double qstick_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void qstick_batch_f64(const double* __restrict__ open,
                      const double* __restrict__ close,
                      int n,
                      const int*   __restrict__ periods,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = qstick_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    if (period <= 0 || first_valid >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    if (period == 1) {
        for (int t = 0; t < first_valid; ++t) row[t] = nan_d;
        for (int t = first_valid; t < n; ++t) row[t] = close[t] - open[t];
        return;
    }

    const int start = first_valid;
    const long long warm_ll = static_cast<long long>(start) + static_cast<long long>(period) - 1;
    if (warm_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    const int warm = static_cast<int>(warm_ll);
    const double inv_p = 1.0 / static_cast<double>(period);

    for (int t = 0; t < warm; ++t) row[t] = nan_d;

    // Seed, in the CPU's groups of four.
    double sum = 0.0;
    const int end_init = start + period;
    const int end_unroll = start + (period & ~3);
    int k = start;
    while (k < end_unroll) {
        sum += (close[k]     - open[k])
             + (close[k + 1] - open[k + 1])
             + (close[k + 2] - open[k + 2])
             + (close[k + 3] - open[k + 3]);
        k += 4;
    }
    while (k < end_init) {
        sum += close[k] - open[k];
        ++k;
    }

    row[warm] = sum * inv_p;

    int i_new = warm + 1;
    int i_old = start;
    while (i_new < n) {
        sum = (sum + (close[i_new] - open[i_new])) - (close[i_old] - open[i_old]);
        row[i_new] = sum * inv_p;
        ++i_new;
        ++i_old;
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `qstick_with_kernel` (src/indicators/qstick.rs:206) for the
// validity rules and the warmup, `qstick_scalar` (:363) for the value.
// `Kernel::Auto` resolves to `Kernel::Scalar` (:260), so `qstick_scalar` IS
// the oracle -- not one of the AVX paths.
//
// SHAPE: one thread per column. The value is a SLIDING SUM whose accumulation
// order is load-bearing: `qstick_scalar` seeds it once and then updates it as
// `sum = (sum + new) - old` for every later bar (:418, :433). Recomputing the
// window fresh at each bar would be a different rounding, so this walks bars
// ascending and carries `sum` exactly as the host does.
//
// ROUNDING COUNT, seed loop (:400-411). Rust `a + b + c + d` associates as
// `((a+b)+c)+d`, and `sum += X` is `sum = sum + X`, so the 4-wide chunk is
//   sum = sum + ((( (c0-o0) + (c1-o1) ) + (c2-o2)) + (c3-o3))
// -- five roundings per chunk, reproduced literally below. The tail
// (:408-411) adds one term at a time. The four-way unrolled emit loop
// (:417-431) performs the SAME per-element update four times, so a plain loop
// reproduces it exactly.
//
// INPUTS: this kernel is registered with `F64InputKind::Ohlc4` and therefore
// receives (open, high, low, close). It reads OPEN and CLOSE only -- qstick's
// CPU source pair is ("open", "close") (cpu_batch.rs:3709) and high/low are
// never touched. Ohlc4 was chosen over inventing a two-pointer OpenClose shape
// because the resident OHLCV upload already carries all four and the launch
// arm for four price pointers already exists; passing (high, low) unread costs
// two pointer arguments and no work.
//
// FIRST-VALID: `F64FirstValidRule::OpenCloseNonNan`. qstick.rs:235-243 scans
// OPEN AND CLOSE simultaneously for the first index where neither is NaN. It
// does NOT look at high or low, so the Ohlc rule would shift the whole series
// on any frame where high or low starts later.
//
// WARMUP: `first + period - 1` (:252-255).
//
// f32 -> f64 audit of this file: the f32 entry points above use
// `__int_as_float(0x7fffffff)` for NaN and `__fmaf_rn`. Below: the f64
// quiet-NaN bit pattern and plain `+`/`-`/`*`, because the CPU reference uses
// no fused multiply-add here at all -- `sum * inv_p` is one multiply. No f32
// literal, no f32-suffixed math function, no fast-math intrinsic. No epsilon
// exists in this indicator and none was invented.
// ===========================================================================

// This file's original includes are `cuda_runtime.h` alone (qstick) or
// `cuda_runtime.h` + `math_constants.h` (psychological_line); the f64 lane
// below calls `isfinite`, so pull in the header that declares it rather than
// relying on a transitive include.
#include <math.h>

static __device__ __forceinline__ double neo_qstick_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void neoethos_qstick_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    (void)high;
    (void)low;

    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_qstick_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int period = periods[r];
    const int first  = first_valid;

    if (period <= 0 || period > n) return;              // :220
    if (first < 0 || first >= n) return;
    if ((n - first) < period) return;                   // :246

    if (period == 1) {                                  // :379-394
        for (int i = first; i < n; ++i) row[i] = close[i] - open[i];
        return;
    }

    const double inv_p = 1.0 / static_cast<double>(period);
    const int warm = first + period - 1;

    // Seed sum -- :396-412, 4-wide association then a one-at-a-time tail.
    double sum = 0.0;
    const int end_unroll = first + (period & ~3);
    int k = first;
    while (k < end_unroll) {
        sum = sum + ((((close[k]     - open[k])
                     + (close[k + 1] - open[k + 1]))
                     + (close[k + 2] - open[k + 2]))
                     + (close[k + 3] - open[k + 3]));
        k += 4;
    }
    const int end_init = first + period;
    while (k < end_init) {
        sum = sum + (close[k] - open[k]);
        ++k;
    }

    row[warm] = sum * inv_p;                            // :413

    int i_new = warm + 1;
    int i_old = first;
    while (i_new < n) {                                 // :415-438
        sum = (sum + (close[i_new] - open[i_new])) - (close[i_old] - open[i_old]);
        row[i_new] = sum * inv_p;
        ++i_new;
        ++i_old;
    }
}
