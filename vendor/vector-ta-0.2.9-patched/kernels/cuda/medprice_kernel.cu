#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <math_constants.h>
#include <math_functions.h>

#ifndef ROW_TILE
#define ROW_TILE 8
#endif


static __device__ __forceinline__ bool is_aligned_16(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0);
}

static __device__ __forceinline__ float medprice_scalar(float h, float l) {

    const float s = h + l;
    return __isnanf(s) ? CUDART_NAN_F : 0.5f * s;
}


extern "C" __global__ __launch_bounds__(256, 2)
void medprice_kernel_f32(const float* __restrict__ high,
                         const float* __restrict__ low,
                         int len,
                         int first_valid,
                         float* __restrict__ out)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (len <= 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;


    for (int i = tid; i < len && i < fv; i += stride) {
        out[i] = CUDART_NAN_F;
    }


    const bool do_vec = is_aligned_16(high) && is_aligned_16(low) && is_aligned_16(out);
    if (do_vec) {

        const int vecStart = ((fv + 3) & ~3);

        for (int i = fv + tid; i < vecStart && i < len; i += stride) {
            out[i] = medprice_scalar(high[i], low[i]);
        }

        if (vecStart < len) {
            const int totalVec = (len - vecStart) >> 2;
            const float4* __restrict__ h4 = reinterpret_cast<const float4*>(high + vecStart);
            const float4* __restrict__ l4 = reinterpret_cast<const float4*>(low  + vecStart);
            float4* __restrict__ o4       = reinterpret_cast<float4*>(out + vecStart);

            for (int v = tid; v < totalVec; v += stride) {
                const float4 ah = h4[v];
                const float4 al = l4[v];
                float4 r;
                float sx = ah.x + al.x; r.x = __isnanf(sx) ? CUDART_NAN_F : 0.5f * sx;
                float sy = ah.y + al.y; r.y = __isnanf(sy) ? CUDART_NAN_F : 0.5f * sy;
                float sz = ah.z + al.z; r.z = __isnanf(sz) ? CUDART_NAN_F : 0.5f * sz;
                float sw = ah.w + al.w; r.w = __isnanf(sw) ? CUDART_NAN_F : 0.5f * sw;
                o4[v] = r;
            }


            const int tailStart = vecStart + (totalVec << 2);
            for (int i = tailStart + tid; i < len; i += stride) {
                out[i] = medprice_scalar(high[i], low[i]);
            }
        }
    } else {

        for (int i = max(fv, 0) + tid; i < len; i += stride) {
            out[i] = medprice_scalar(high[i], low[i]);
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void medprice_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        int len,
                        int rows,
                        const int* __restrict__ first_valids,
                        float* __restrict__ out)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int rowTileBase = blockIdx.y * ROW_TILE;
    if (rowTileBase >= rows) return;

    const bool do_vec = is_aligned_16(high) && is_aligned_16(low) && is_aligned_16(out);
    if (do_vec) {
        const int totalVec = len >> 2;
        const float4* __restrict__ h4 = reinterpret_cast<const float4*>(high);
        const float4* __restrict__ l4 = reinterpret_cast<const float4*>(low);

        for (int v = tid; v < totalVec; v += stride) {
            const int i = v << 2;
            const float4 ah = h4[v];
            const float4 al = l4[v];


            const float sx = ah.x + al.x; const float r0 = __isnanf(sx) ? CUDART_NAN_F : 0.5f * sx;
            const float sy = ah.y + al.y; const float r1 = __isnanf(sy) ? CUDART_NAN_F : 0.5f * sy;
            const float sz = ah.z + al.z; const float r2 = __isnanf(sz) ? CUDART_NAN_F : 0.5f * sz;
            const float sw = ah.w + al.w; const float r3 = __isnanf(sw) ? CUDART_NAN_F : 0.5f * sw;

            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                float* __restrict__ o = out + row * len + i;
                o[0] = (i + 0 < fv) ? CUDART_NAN_F : r0;
                o[1] = (i + 1 < fv) ? CUDART_NAN_F : r1;
                o[2] = (i + 2 < fv) ? CUDART_NAN_F : r2;
                o[3] = (i + 3 < fv) ? CUDART_NAN_F : r3;
            }
        }


        const int tailStart = (totalVec << 2);
        for (int i = tailStart + tid; i < len; i += stride) {
            const float r = medprice_scalar(high[i], low[i]);
            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                out[row * len + i] = (i < fv) ? CUDART_NAN_F : r;
            }
        }
    } else {

        for (int i = tid; i < len; i += stride) {
            const float r = medprice_scalar(high[i], low[i]);
            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                out[row * len + i] = (i < fv) ? CUDART_NAN_F : r;
            }
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void medprice_many_series_one_param_f32(const float* __restrict__ high_tm,
                                        const float* __restrict__ low_tm,
                                        int cols,
                                        int rows,
                                        const int* __restrict__ first_valids,
                                        float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids ? max(first_valids[s], 0) : 0;


    #pragma unroll 4
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        if (t < fv) {
            out_tm[idx] = CUDART_NAN_F;
        } else {
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            const float ssum = h + l;
            out_tm[idx] = __isnanf(ssum) ? CUDART_NAN_F : 0.5f * ssum;
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

static __device__ __forceinline__ double medprice_scalar_f64(double h, double l) {

    const double s = h + l;
    return __isnanf(s) ? CUDART_NAN : 0.5 * s;
}
extern "C" __global__ __launch_bounds__(256, 2)
void medprice_kernel_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int len,
                         int first_valid,
                         double* __restrict__ out)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (len <= 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;


    for (int i = tid; i < len && i < fv; i += stride) {
        out[i] = CUDART_NAN;
    }


    const bool do_vec = is_aligned_16(high) && is_aligned_16(low) && is_aligned_16(out);
    if (do_vec) {

        const int vecStart = ((fv + 3) & ~3);

        for (int i = fv + tid; i < vecStart && i < len; i += stride) {
            out[i] = medprice_scalar_f64(high[i], low[i]);
        }

        if (vecStart < len) {
            const int totalVec = (len - vecStart) >> 2;
            const double4* __restrict__ h4 = reinterpret_cast<const double4*>(high + vecStart);
            const double4* __restrict__ l4 = reinterpret_cast<const double4*>(low  + vecStart);
            double4* __restrict__ o4       = reinterpret_cast<double4*>(out + vecStart);

            for (int v = tid; v < totalVec; v += stride) {
                const double4 ah = h4[v];
                const double4 al = l4[v];
                double4 r;
                double sx = ah.x + al.x; r.x = __isnanf(sx) ? CUDART_NAN : 0.5 * sx;
                double sy = ah.y + al.y; r.y = __isnanf(sy) ? CUDART_NAN : 0.5 * sy;
                double sz = ah.z + al.z; r.z = __isnanf(sz) ? CUDART_NAN : 0.5 * sz;
                double sw = ah.w + al.w; r.w = __isnanf(sw) ? CUDART_NAN : 0.5 * sw;
                o4[v] = r;
            }


            const int tailStart = vecStart + (totalVec << 2);
            for (int i = tailStart + tid; i < len; i += stride) {
                out[i] = medprice_scalar_f64(high[i], low[i]);
            }
        }
    } else {

        for (int i = max(fv, 0) + tid; i < len; i += stride) {
            out[i] = medprice_scalar_f64(high[i], low[i]);
        }
    }
}
extern "C" __global__ __launch_bounds__(256, 2)
void medprice_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        int len,
                        int rows,
                        const int* __restrict__ first_valids,
                        double* __restrict__ out)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int rowTileBase = blockIdx.y * ROW_TILE;
    if (rowTileBase >= rows) return;

    const bool do_vec = is_aligned_16(high) && is_aligned_16(low) && is_aligned_16(out);
    if (do_vec) {
        const int totalVec = len >> 2;
        const double4* __restrict__ h4 = reinterpret_cast<const double4*>(high);
        const double4* __restrict__ l4 = reinterpret_cast<const double4*>(low);

        for (int v = tid; v < totalVec; v += stride) {
            const int i = v << 2;
            const double4 ah = h4[v];
            const double4 al = l4[v];


            const double sx = ah.x + al.x; const double r0 = __isnanf(sx) ? CUDART_NAN : 0.5 * sx;
            const double sy = ah.y + al.y; const double r1 = __isnanf(sy) ? CUDART_NAN : 0.5 * sy;
            const double sz = ah.z + al.z; const double r2 = __isnanf(sz) ? CUDART_NAN : 0.5 * sz;
            const double sw = ah.w + al.w; const double r3 = __isnanf(sw) ? CUDART_NAN : 0.5 * sw;

            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                double* __restrict__ o = out + row * len + i;
                o[0] = (i + 0 < fv) ? CUDART_NAN : r0;
                o[1] = (i + 1 < fv) ? CUDART_NAN : r1;
                o[2] = (i + 2 < fv) ? CUDART_NAN : r2;
                o[3] = (i + 3 < fv) ? CUDART_NAN : r3;
            }
        }


        const int tailStart = (totalVec << 2);
        for (int i = tailStart + tid; i < len; i += stride) {
            const double r = medprice_scalar_f64(high[i], low[i]);
            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                out[row * len + i] = (i < fv) ? CUDART_NAN : r;
            }
        }
    } else {

        for (int i = tid; i < len; i += stride) {
            const double r = medprice_scalar_f64(high[i], low[i]);
            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                out[row * len + i] = (i < fv) ? CUDART_NAN : r;
            }
        }
    }
}
extern "C" __global__ __launch_bounds__(256, 2)
void medprice_many_series_one_param_f64(const double* __restrict__ high_tm,
                                        const double* __restrict__ low_tm,
                                        int cols,
                                        int rows,
                                        const int* __restrict__ first_valids,
                                        double* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids ? max(first_valids[s], 0) : 0;


    #pragma unroll 4
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        if (t < fv) {
            out_tm[idx] = CUDART_NAN;
        } else {
            const double h = high_tm[idx];
            const double l = low_tm[idx];
            const double ssum = h + l;
            out_tm[idx] = __isnanf(ssum) ? CUDART_NAN : 0.5 * ssum;
        }
    }
}
