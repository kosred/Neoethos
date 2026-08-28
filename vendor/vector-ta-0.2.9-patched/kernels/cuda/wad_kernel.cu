#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef WAD_SCAN_BLOCK_X
#define WAD_SCAN_BLOCK_X 256
#endif

#ifndef WAD_SCAN_ITEMS_PER_THREAD
#define WAD_SCAN_ITEMS_PER_THREAD 8
#endif

#define WAD_SCAN_TILE (WAD_SCAN_BLOCK_X * WAD_SCAN_ITEMS_PER_THREAD)


struct KBNAcc32 {
  float sum;
  float c;
  __device__ __forceinline__ KBNAcc32() : sum(0.f), c(0.f) {}

  __device__ __forceinline__ void add(float x) {
    float t = sum + x;

    float e = (fabsf(sum) >= fabsf(x)) ? (sum - t) + x : (x - t) + sum;
    c += e;
    sum = t;
  }

  __device__ __forceinline__ float value() const { return sum + c; }
};


__device__ __forceinline__ float wad_step(float hi, float lo, float c, float pc) {
  const float trh = (pc > hi) ? pc : hi;
  const float trl = (pc < lo) ? pc : lo;

  float ad = 0.0f;
  if (c > pc)       ad = c - trl;
  else if (c < pc)  ad = c - trh;
  return ad;
}


extern "C" __global__ void wad_series_scan_blocks_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    float* __restrict__ out,
    double* __restrict__ block_sums) {

  __shared__ double scan[WAD_SCAN_TILE];
  __shared__ double temp[WAD_SCAN_TILE];

  const int base = blockIdx.x * WAD_SCAN_TILE;
  const int tid = threadIdx.x;

  #pragma unroll
  for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
    const int lane = tid + j * WAD_SCAN_BLOCK_X;
    const int idx = base + lane;
    double inc = 0.0;
    if (idx > 0 && idx < series_len) {
      inc = (double)wad_step(high[idx], low[idx], close[idx], close[idx - 1]);
    }
    scan[lane] = inc;
  }
  __syncthreads();

  for (int offset = 1; offset < WAD_SCAN_TILE; offset <<= 1) {
    #pragma unroll
    for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
      const int lane = tid + j * WAD_SCAN_BLOCK_X;
      temp[lane] = scan[lane] + (lane >= offset ? scan[lane - offset] : 0.0);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
      const int lane = tid + j * WAD_SCAN_BLOCK_X;
      scan[lane] = temp[lane];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
    const int lane = tid + j * WAD_SCAN_BLOCK_X;
    const int idx = base + lane;
    if (idx < series_len) out[idx] = (float)scan[lane];
  }

  if (tid == 0) {
    int remaining = series_len - base;
    int count = remaining > WAD_SCAN_TILE ? WAD_SCAN_TILE : remaining;
    block_sums[blockIdx.x] = count > 0 ? scan[count - 1] : 0.0;
  }
}


extern "C" __global__ void wad_scan_block_sums_f64(
    double* __restrict__ block_sums,
    int num_blocks) {

  __shared__ double scan[WAD_SCAN_TILE];
  __shared__ double temp[WAD_SCAN_TILE];

  const int tid = threadIdx.x;
  #pragma unroll
  for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
    const int lane = tid + j * WAD_SCAN_BLOCK_X;
    scan[lane] = lane < num_blocks ? block_sums[lane] : 0.0;
  }
  __syncthreads();

  for (int offset = 1; offset < WAD_SCAN_TILE; offset <<= 1) {
    #pragma unroll
    for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
      const int lane = tid + j * WAD_SCAN_BLOCK_X;
      temp[lane] = scan[lane] + (lane >= offset ? scan[lane - offset] : 0.0);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
      const int lane = tid + j * WAD_SCAN_BLOCK_X;
      scan[lane] = temp[lane];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
    const int lane = tid + j * WAD_SCAN_BLOCK_X;
    if (lane < num_blocks) block_sums[lane] = scan[lane];
  }
}


extern "C" __global__ void wad_add_block_offsets_f32(
    float* __restrict__ out,
    int series_len,
    const double* __restrict__ block_sums) {

  const int base = blockIdx.x * WAD_SCAN_TILE;
  if (blockIdx.x == 0) return;

  const double offset = block_sums[blockIdx.x - 1];
  const int tid = threadIdx.x;

  #pragma unroll
  for (int j = 0; j < WAD_SCAN_ITEMS_PER_THREAD; ++j) {
    const int lane = tid + j * WAD_SCAN_BLOCK_X;
    const int idx = base + lane;
    if (idx < series_len) out[idx] = (float)((double)out[idx] + offset);
  }
}


extern "C" __global__ void wad_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int n_combos,
    float* __restrict__ out) {

  if (series_len <= 0 || n_combos <= 0) return;

  const int tpb = blockDim.x * gridDim.x;
  int combo = blockIdx.x * blockDim.x + threadIdx.x;

  for (; combo < n_combos; combo += tpb) {
    float* __restrict__ out_row = out + combo * series_len;


    out_row[0] = 0.0f;
    KBNAcc32 acc;
    float pc = close[0];


    #pragma unroll 1
    for (int i = 1; i < series_len; ++i) {
      const float ad = wad_step(high[i], low[i], close[i], pc);
      acc.add(ad);
      out_row[i] = acc.value();
      pc = close[i];
    }
  }
}


extern "C" __global__ void wad_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    float* __restrict__ out_tm) {

  if (rows <= 0 || cols <= 0) return;


  const int stride_series = blockDim.x * gridDim.x;
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride_series) {
    const int stride = cols;

    out_tm[0 * stride + s] = 0.0f;

    KBNAcc32 acc;
    float pc = close_tm[0 * stride + s];

    #pragma unroll 1
    for (int t = 1; t < rows; ++t) {
      const int idx = t * stride + s;
      const float ad = wad_step(high_tm[idx], low_tm[idx], close_tm[idx], pc);
      acc.add(ad);
      out_tm[idx] = acc.value();
      pc = close_tm[idx];
    }
  }
}


extern "C" __global__ void wad_series_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int n_series,
    float* __restrict__ out) {

  if (len <= 0 || n_series <= 0) return;

  const int stride_series = blockDim.x * gridDim.x;
  for (int series = blockIdx.x * blockDim.x + threadIdx.x; series < n_series; series += stride_series) {
    const int offset = series * len;
    const float* high_row  = high  + offset;
    const float* low_row   = low   + offset;
    const float* close_row = close + offset;
    float* out_row         = out   + offset;

    out_row[0] = 0.0f;
    KBNAcc32 acc;
    float pc = close_row[0];

    #pragma unroll 1
    for (int i = 1; i < len; ++i) {
      const float ad = wad_step(high_row[i], low_row[i], close_row[i], pc);
      acc.add(ad);
      out_row[i] = acc.value();
      pc = close_row[i];
    }
  }
}


extern "C" __global__ void wad_compute_single_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    float* __restrict__ out_row) {
  if (series_len <= 0) return;
  out_row[0] = 0.0f;
  KBNAcc32 acc;
  float pc = close[0];
  #pragma unroll 1
  for (int i = 1; i < series_len; ++i) {
    const float ad = wad_step(high[i], low[i], close[i], pc);
    acc.add(ad);
    out_row[i] = acc.value();
    pc = close[i];
  }
}


extern "C" __global__ void broadcast_row_f32(
    const float* __restrict__ row,
    int series_len,
    int n_combos,
    float* __restrict__ out) {
  if (series_len <= 0 || n_combos <= 0) return;
  const int n = series_len * n_combos;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    const int j = idx % series_len;
    out[idx] = row[j];
  }
}


// ===========================================================================
// S3 f64 LANE — wad (Williams Accumulation/Distribution)
// ===========================================================================
// Reference: src/indicators/wad.rs
//   wad_with_kernel (:148) — alloc_with_nan_prefix(len, 0): NO warmup at all
//   wad_scalar (:194)      — the arithmetic
//   wad_prepare (:1416)    — the Err branches
//
// FIRST-VALID IS NOT READ. The CPU never computes a first-non-NaN index for
// wad; it starts at bar 0 with out[0] = 0.0 and accumulates. The lane still
// passes first_valid because the ABI is uniform; this kernel ignores it, which
// is why its registry row declares F64FirstValidRule::Ignored rather than
// silently adopting a prefix the reference does not have.
//
// PERIOD-INVARIANT. compute_wad_batch takes no period parameter at all.
//
// NaN SEMANTICS — THIS IS THE adx-CLASS BUG, AND IT IS LIVE HERE.
// The CPU writes pc.max(h) and pc.min(l), i.e. f64::max / f64::min, which
// RETURN THE NON-NaN OPERAND. The f32 kernels above use comparison chains; a
// comparison against NaN is false, so the NaN survives, enters acc, and every
// subsequent bar of the accumulation is NaN. fmax/fmin below have exactly
// f64::max/min semantics.
//
// ROUNDING. gt.mul_add(c - trl, lt * (c - trh)) is: one subtract, one
// subtract, one multiply, one fma — and the fma rounds ONCE. Written as
// gt*(c-trl) + lt*(c-trh) it rounds twice more. fma() below matches.
//
// One thread per column: acc is a running total across bars.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_wad_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;      // PERIOD-INVARIANT — compute_wad_batch reads no period.
    (void)first_valid;  // see the header: the CPU reference has no warmup.

    double* __restrict__ row = out + (size_t)r * (size_t)n;

    if (n <= 0) return;

    // wad_prepare :1441 — an all-NaN series in any of the three is Err, so the
    // CPU produces nothing.
    bool h_all_nan = true, l_all_nan = true, c_all_nan = true;
    for (int i = 0; i < n; ++i) {
        if (!isnan(high[i]))  { h_all_nan = false; }
        if (!isnan(low[i]))   { l_all_nan = false; }
        if (!isnan(close[i])) { c_all_nan = false; }
        if (!h_all_nan && !l_all_nan && !c_all_nan) break;
    }
    if (h_all_nan || l_all_nan || c_all_nan) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    row[0] = 0.0;
    double acc = 0.0;
    double pc = close[0];

    for (int i = 1; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double trh = fmax(pc, h);   // f64::max — returns the non-NaN side
        const double trl = fmin(pc, l);   // f64::min — likewise

        const double gt = (c > pc) ? 1.0 : 0.0;
        const double lt = (c < pc) ? 1.0 : 0.0;

        const double ad = fma(gt, c - trl, lt * (c - trh));
        acc += ad;
        row[i] = acc;
        pc = c;
    }
}
