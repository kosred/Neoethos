#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef NVI_SCAN_BLOCK_X
#define NVI_SCAN_BLOCK_X 256
#endif

#ifndef NVI_SCAN_ITEMS_PER_THREAD
#define NVI_SCAN_ITEMS_PER_THREAD 8
#endif

#define NVI_SCAN_TILE (NVI_SCAN_BLOCK_X * NVI_SCAN_ITEMS_PER_THREAD)


struct dsfloat {
    float hi;
    float lo;
};

__device__ __forceinline__ dsfloat ds_make(float x) {
    dsfloat a; a.hi = x; a.lo = 0.0f; return a;
}


__device__ __forceinline__ void ds_renorm(dsfloat& a, float t) {
    float s = a.hi + t;
    a.lo    = t - (s - a.hi);
    a.hi    = s;
}


__device__ __forceinline__ dsfloat ds_add(dsfloat a, dsfloat b) {

    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float err = (a.hi - (s - bb)) + (b.hi - bb);

    float t = a.lo + b.lo + err;

    dsfloat r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}


__device__ __forceinline__ dsfloat ds_mul_scalar(dsfloat a, float b) {
    float p = a.hi * b;
    float e = fmaf(a.hi, b, -p);
    float t = a.lo * b + e;
    dsfloat r;
    r.hi = p + t;
    r.lo = t - (r.hi - p);
    return r;
}


__device__ __forceinline__ float ds_to_float(dsfloat a) {
    return a.hi + a.lo;
}


extern "C" __global__ void nvi_scan_blocks_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out,
    double* __restrict__ block_products)
{
    __shared__ double scan[NVI_SCAN_TILE];
    __shared__ double temp[NVI_SCAN_TILE];

    const int base = blockIdx.x * NVI_SCAN_TILE;
    const int tid = threadIdx.x;
    const float nan_f = CUDART_NAN_F;

    if (first_valid < 0) first_valid = 0;

    #pragma unroll
    for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * NVI_SCAN_BLOCK_X;
        const int idx = base + lane;
        double factor = 1.0;
        if (idx > first_valid && idx < len) {
            const float c = close[idx];
            const float c0 = close[idx - 1];
            const float v = volume[idx];
            const float v0 = volume[idx - 1];
            if (v < v0) factor = 1.0 + (double)((c - c0) / c0);
        }
        scan[lane] = factor;
    }
    __syncthreads();

    for (int offset = 1; offset < NVI_SCAN_TILE; offset <<= 1) {
        #pragma unroll
        for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * NVI_SCAN_BLOCK_X;
            temp[lane] = scan[lane] * (lane >= offset ? scan[lane - offset] : 1.0);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * NVI_SCAN_BLOCK_X;
            scan[lane] = temp[lane];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * NVI_SCAN_BLOCK_X;
        const int idx = base + lane;
        if (idx < len) {
            if (idx < first_valid) out[idx] = nan_f;
            else if (idx == first_valid) out[idx] = 1000.0f;
            else out[idx] = (float)(1000.0 * scan[lane]);
        }
    }

    if (tid == 0) {
        int remaining = len - base;
        int count = remaining > NVI_SCAN_TILE ? NVI_SCAN_TILE : remaining;
        block_products[blockIdx.x] = count > 0 ? scan[count - 1] : 1.0;
    }
}


extern "C" __global__ void nvi_scan_block_products_f64(
    double* __restrict__ block_products,
    int num_blocks)
{
    __shared__ double scan[NVI_SCAN_TILE];
    __shared__ double temp[NVI_SCAN_TILE];

    const int tid = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * NVI_SCAN_BLOCK_X;
        scan[lane] = lane < num_blocks ? block_products[lane] : 1.0;
    }
    __syncthreads();

    for (int offset = 1; offset < NVI_SCAN_TILE; offset <<= 1) {
        #pragma unroll
        for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * NVI_SCAN_BLOCK_X;
            temp[lane] = scan[lane] * (lane >= offset ? scan[lane - offset] : 1.0);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * NVI_SCAN_BLOCK_X;
            scan[lane] = temp[lane];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * NVI_SCAN_BLOCK_X;
        if (lane < num_blocks) block_products[lane] = scan[lane];
    }
}


extern "C" __global__ void nvi_apply_block_products_f32(
    float* __restrict__ out,
    int len,
    int first_valid,
    const double* __restrict__ block_products)
{
    const int base = blockIdx.x * NVI_SCAN_TILE;
    if (blockIdx.x == 0) return;

    if (first_valid < 0) first_valid = 0;
    const double factor = block_products[blockIdx.x - 1];
    const int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < NVI_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * NVI_SCAN_BLOCK_X;
        const int idx = base + lane;
        if (idx < len && idx > first_valid) out[idx] = (float)((double)out[idx] * factor);
    }
}


extern "C" __global__ void nvi_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    if (len <= 0) return;


    if (blockIdx.x != 0) return;


    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 16) return;
    const unsigned mask = 0x0000ffffu;

    const int fv = first_valid < 0 ? 0 : first_valid;


    const float nan_f = CUDART_NAN_F;
    for (int i = lane; i < fv && i < len; i += 16) out[i] = nan_f;
    if (fv >= len) return;


    if (lane == 0) out[fv] = 1000.0f;
    if (fv + 1 >= len) return;

    double nvi0 = 1000.0;

    for (int t0 = fv + 1; t0 < len; t0 += 16) {
        const int i = t0 + lane;
        double f = 1.0;
        if (i < len) {
            const float c = close[i];
            const float c0 = close[i - 1];
            const float v = volume[i];
            const float v0 = volume[i - 1];
            if (v < v0) {
                const float pct = (c - c0) / c0;
                f = 1.0 + (double)pct;
            }
        }


        double prefix = f;
        for (int offset = 1; offset < 16; offset <<= 1) {
            double other = __shfl_up_sync(mask, prefix, offset, 16);
            if (lane >= offset) prefix *= other;
        }

        double base = __shfl_sync(mask, nvi0, 0, 16);
        if (i < len) out[i] = (float)(base * prefix);

        double tile_prod = __shfl_sync(mask, prefix, 15, 16);
        if (lane == 0) nvi0 *= tile_prod;
    }
}


extern "C" __global__ void nvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    if (rows <= 0 || cols <= 0) return;
    const float nan_f = CUDART_NAN_F;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s] < 0 ? 0 : first_valids[s];


        if (fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
            continue;
        }


        for (int t = 0; t < fv; ++t) out_tm[t * cols + s] = nan_f;


        dsfloat nvi = ds_make(1000.0f);
        out_tm[fv * cols + s] = ds_to_float(nvi);
        if (fv + 1 >= rows) continue;

        float prev_close  = close_tm[fv * cols + s];
        float prev_volume = volume_tm[fv * cols + s];

        for (int t = fv + 1; t < rows; ++t) {
            const float c = close_tm[t * cols + s];
            const float v = volume_tm[t * cols + s];

            if (v < prev_volume) {
                const float pct = (c - prev_close) / prev_close;
                dsfloat prod = ds_mul_scalar(nvi, pct);
                nvi = ds_add(nvi, prod);
            }
            out_tm[t * cols + s] = ds_to_float(nvi);
            prev_close  = c;
            prev_volume = v;
        }
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/nvi.rs
//   * nvi_with_kernel (:216) — first_valid is the first index at which CLOSE and
//     VOLUME are both non-NaN, and the warmup prefix is exactly `first` (:236),
//     because out[first] is the 1000.0 seed.
//   * nvi_scalar (:327) — the arithmetic reproduced below.
//
// PERIOD-INVARIANT. compute_nvi_batch (cpu_batch.rs:3944) takes |_params|.
//
// COMPARISON SEMANTICS, deliberately NOT fmax/fmin. The CPU gate is a bare
//     if v < prev_volume { ... }
// so a NaN volume makes the comparison FALSE and nvi_val is carried forward
// unchanged. Rewriting this as fmin would change which bars update. Rule 4 is
// "match the CPU", and here the CPU is a raw comparison — the fmax/fmin
// rewrite applies where the CPU itself calls f64::max, not everywhere a
// comparison appears.
//
// Sequential: nvi_val, prev_close and prev_volume carry across bars.
// One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_nvi() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__
void neoethos_nvi_f64(const double* __restrict__ close,
                      const double* __restrict__ volume,
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
    const double QNAN = nef_qnan_nvi();

    if (first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = QNAN;
    for (int i = first_valid; i < n; ++i) row[i] = QNAN;

    double nvi_val = 1000.0;
    row[first_valid] = nvi_val;

    if (first_valid + 1 >= n) return;

    double prev_close = close[first_valid];
    double prev_volume = volume[first_valid];

    for (int i = first_valid + 1; i < n; ++i) {
        const double c = close[i];
        const double v = volume[i];

        if (v < prev_volume) {
            const double pct = (c - prev_close) / prev_close;
            nvi_val += nvi_val * pct;
        }

        row[i] = nvi_val;

        prev_close = c;
        prev_volume = v;
    }
}



// ===========================================================================
// S1 f64 LANE  --  nvi
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/nvi.rs -- `nvi_scalar` (:327), `nvi_with_kernel` (:198)
//
// PERIOD-INVARIANT. `compute_nvi_batch` (cpu_batch.rs:3937) takes
// `|_params|` -- nvi has no period parameter at all -- so every row of a sweep
// is byte-identical, as with `obv`.
//
// `nvi_with_kernel` collapses EVERY `Kernel` variant to `Kernel::Scalar`
// (nvi.rs:237-246), so `nvi_scalar` is the only CPU answer on any host and
// there is no scalar/AVX disagreement to settle for this indicator.
//
// ARITHMETIC ORDER: `pct = (c - prev_close) / prev_close`, then
// `nvi_val += nvi_val * pct` -- a multiply and an add, TWO roundings, no
// `mul_add`. Reproduced literally.
//
// WARMUP: `alloc_with_nan_prefix(len, first)` then `out[first] = 1000.0`.
// first_valid is the first index at which close AND volume are both non-NaN
// (nvi.rs:219-222) -- the common `AllInputsNonNan` rule over a (close, volume)
// pair.
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

extern "C" __global__ void neoethos_nvi_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
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

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < 2);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = neo_s1_qnan();

    double nvi_val = 1000.0;
    row[first_valid] = nvi_val;

    double prev_close = close[first_valid];
    double prev_volume = volume[first_valid];

    for (int i = first_valid + 1; i < n; ++i) {
        const double c = close[i];
        const double v = volume[i];
        if (v < prev_volume) {
            const double pct = (c - prev_close) / prev_close;
            nvi_val += nvi_val * pct;
        }
        row[i] = nvi_val;
        prev_close = c;
        prev_volume = v;
    }
}
