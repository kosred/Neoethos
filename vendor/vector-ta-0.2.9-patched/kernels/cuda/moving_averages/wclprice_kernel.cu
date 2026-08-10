#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


extern "C" __global__ void wclprice_batch_f32(const float* __restrict__ high,
                                              const float* __restrict__ low,
                                              const float* __restrict__ close,
                                              int series_len,
                                              int first_valid,
                                              float* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (series_len <= 0) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = tid; i < series_len; i += stride) {
        if (i < fv) {
            out[i] = CUDART_NAN_F; continue;
        }
        const float h = high[i];
        const float l = low[i];
        const float c = close[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            out[i] = CUDART_NAN_F; continue;
        }

        out[i] = c * 0.5f + (h + l) * 0.25f;
    }
}


extern "C" __global__ void wclprice_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= cols || cols <= 0 || rows <= 0) return;
    const int fv = max(0, first_valids[s]);
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int t0 = tid; t0 < rows; t0 += stride) {
        const int idx = t0 * cols + s;
        if (t0 < fv) { out_tm[idx] = CUDART_NAN_F; continue; }
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            out_tm[idx] = CUDART_NAN_F; continue;
        }
        out_tm[idx] = c * 0.5f + (h + l) * 0.25f;
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

extern "C" __global__ void wclprice_batch_f64(const double* __restrict__ high,
                                              const double* __restrict__ low,
                                              const double* __restrict__ close,
                                              int series_len,
                                              int first_valid,
                                              double* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (series_len <= 0) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = tid; i < series_len; i += stride) {
        if (i < fv) {
            out[i] = CUDART_NAN; continue;
        }
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            out[i] = CUDART_NAN; continue;
        }

        out[i] = c * 0.5 + (h + l) * 0.25;
    }
}
extern "C" __global__ void wclprice_many_series_one_param_time_major_f64(
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    const double* __restrict__ close_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= cols || cols <= 0 || rows <= 0) return;
    const int fv = max(0, first_valids[s]);
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int t0 = tid; t0 < rows; t0 += stride) {
        const int idx = t0 * cols + s;
        if (t0 < fv) { out_tm[idx] = CUDART_NAN; continue; }
        const double h = high_tm[idx];
        const double l = low_tm[idx];
        const double c = close_tm[idx];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            out_tm[idx] = CUDART_NAN; continue;
        }
        out_tm[idx] = c * 0.5 + (h + l) * 0.25;
    }
}
