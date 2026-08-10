#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

static __device__ __forceinline__ unsigned lane_id() {
  return threadIdx.x & (WARP_SIZE - 1);
}

static __device__ __forceinline__ int warp_reduce_max(int v) {
  const unsigned full = 0xFFFFFFFFu;
  for (int ofs = WARP_SIZE >> 1; ofs > 0; ofs >>= 1) {
    int other = __shfl_down_sync(full, v, ofs);
    v = max(v, other);
  }
  return v;
}

static __device__ __forceinline__ int warp_reduce_max(int v, unsigned mask) {
  for (int ofs = WARP_SIZE >> 1; ofs > 0; ofs >>= 1) {
    int other = __shfl_down_sync(mask, v, ofs);
    v = max(v, other);
  }
  return v;
}

static __device__ __forceinline__ int warp_reduce_min(int v, unsigned mask) {
  for (int ofs = WARP_SIZE >> 1; ofs > 0; ofs >>= 1) {
    int other = __shfl_down_sync(mask, v, ofs);
    v = min(v, other);
  }
  return v;
}

extern "C" __global__
void alligator_batch_f32(const float* __restrict__ prices,
                         const int*   __restrict__ jaw_periods,
                         const int*   __restrict__ jaw_offsets,
                         const int*   __restrict__ teeth_periods,
                         const int*   __restrict__ teeth_offsets,
                         const int*   __restrict__ lips_periods,
                         const int*   __restrict__ lips_offsets,
                         int first_valid,
                         int series_len,
                         int n_combos,
                         float* __restrict__ out_jaw,
                         float* __restrict__ out_teeth,
                         float* __restrict__ out_lips) {

  const int combo = blockIdx.x * blockDim.x + threadIdx.x;
  const float nan_f = __int_as_float(0x7fffffff);


  int pj = 0, pt = 0, pl = 0;
  int oj = 0, ot = 0, ol = 0;
  int base = 0;
  bool valid = false;
  float aj = 0.f, bj = 0.f;
  float at = 0.f, bt = 0.f;
  float al = 0.f, bl = 0.f;

  if (combo < n_combos) {
    pj = jaw_periods[combo];
    pt = teeth_periods[combo];
    pl = lips_periods[combo];
    oj = jaw_offsets[combo];
    ot = teeth_offsets[combo];
    ol = lips_offsets[combo];
    base = combo * series_len;
    valid = (pj > 0) & (pt > 0) & (pl > 0);
    if (valid) {
      aj = 1.0f / float(pj); bj = 1.0f - aj;
      at = 1.0f / float(pt); bt = 1.0f - at;
      al = 1.0f / float(pl); bl = 1.0f - al;
    }
  }


  if (combo < n_combos && valid) {
    const int warm_base_j = first_valid + pj - 1;
    const int warm_base_t = first_valid + pt - 1;
    const int warm_base_l = first_valid + pl - 1;
    const int warm_j = min(series_len, warm_base_j + oj);
    const int warm_t = min(series_len, warm_base_t + ot);
    const int warm_l = min(series_len, warm_base_l + ol);
    for (int i = 0; i < warm_j; ++i) out_jaw[base + i] = nan_f;
    for (int i = 0; i < warm_t; ++i) out_teeth[base + i] = nan_f;
    for (int i = 0; i < warm_l; ++i) out_lips[base + i] = nan_f;


  }


  float prev_j = 0.f, prev_t = 0.f, prev_l = 0.f;
  if (combo < n_combos && valid) {
    const unsigned mask = __activemask();
    const int warm_base_j = first_valid + pj - 1;
    const int warm_base_t = first_valid + pt - 1;
    const int warm_base_l = first_valid + pl - 1;
    int maxP = warp_reduce_max(max(pj, max(pt, pl)), mask);


    maxP = __shfl_sync(mask, maxP, 0);
    const int leader = 0;
    float sum_j = 0.f, sum_t = 0.f, sum_l = 0.f;
    for (int k = 0; k < maxP; ++k) {
      float v = 0.f;
      if (lane_id() == (unsigned)leader) {
        v = prices[first_valid + k];
      }
      const float pk = __shfl_sync(mask, v, leader);
      if (k < pj) sum_j += pk;
      if (k < pt) sum_t += pk;
      if (k < pl) sum_l += pk;
    }
    prev_j = sum_j * aj;
    prev_t = sum_t * at;
    prev_l = sum_l * al;


    const int tj = warm_base_j + oj;
    const int tt = warm_base_t + ot;
    const int tl = warm_base_l + ol;
    if (tj < series_len) out_jaw[base + tj] = prev_j;
    if (tt < series_len) out_teeth[base + tt] = prev_t;
    if (tl < series_len) out_lips[base + tl] = prev_l;


    const int min_base = min(min(warm_base_j, warm_base_t), warm_base_l);
    int start_i = warp_reduce_min(min_base, mask);
    start_i = __shfl_sync(mask, start_i, 0) + 1;
    for (int i = start_i; i < series_len; ++i) {
      float v2 = 0.f;
      if (lane_id() == (unsigned)leader) {
        v2 = prices[i];
      }
      const float px = __shfl_sync(mask, v2, leader);

      if (i > warm_base_j) {
        prev_j = fmaf(prev_j, bj, px * aj);
      }
      if (i > warm_base_t) {
        prev_t = fmaf(prev_t, bt, px * at);
      }
      if (i > warm_base_l) {
        prev_l = fmaf(prev_l, bl, px * al);
      }
      const int wj = i + oj;
      const int wt = i + ot;
      const int wl = i + ol;
      if (wj < series_len && i >= warm_base_j) out_jaw[base + wj] = prev_j;
      if (wt < series_len && i >= warm_base_t) out_teeth[base + wt] = prev_t;
      if (wl < series_len && i >= warm_base_l) out_lips[base + wl] = prev_l;
    }
  }
}

extern "C" __global__
void alligator_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                         int jaw_period,
                                         int jaw_offset,
                                         int teeth_period,
                                         int teeth_offset,
                                         int lips_period,
                                         int lips_offset,
                                         int num_series,
                                         int series_len,
                                         const int* __restrict__ first_valids,
                                         float* __restrict__ out_jaw_tm,
                                         float* __restrict__ out_teeth_tm,
                                         float* __restrict__ out_lips_tm) {
  const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (series_idx >= num_series) return;

  const int first = first_valids[series_idx];
  if (first < 0 || first >= series_len) return;
  if (jaw_period <= 0 || teeth_period <= 0 || lips_period <= 0) return;

  const int warm_base_j = first + jaw_period - 1;
  const int warm_base_t = first + teeth_period - 1;
  const int warm_base_l = first + lips_period - 1;
  const int warm_j = warm_base_j + jaw_offset;
  const int warm_t = warm_base_t + teeth_offset;
  const int warm_l = warm_base_l + lips_offset;

  const float aj = 1.0f / float(jaw_period);
  const float bj = 1.0f - aj;
  const float at = 1.0f / float(teeth_period);
  const float bt = 1.0f - at;
  const float al = 1.0f / float(lips_period);
  const float bl = 1.0f - al;

  const size_t stride = size_t(num_series);
  const size_t col    = size_t(series_idx);
  const float nan_f = __int_as_float(0x7fffffff);


  for (int t = 0; t < min(series_len, warm_j); ++t)
    out_jaw_tm[size_t(t) * stride + col] = nan_f;
  for (int t = 0; t < min(series_len, warm_t); ++t)
    out_teeth_tm[size_t(t) * stride + col] = nan_f;
  for (int t = 0; t < min(series_len, warm_l); ++t)
    out_lips_tm[size_t(t) * stride + col] = nan_f;

  if (warm_base_j >= series_len && warm_base_t >= series_len && warm_base_l >= series_len) return;


  size_t idx = size_t(first) * stride + col;
  float sum_j = 0.f, sum_t = 0.f, sum_l = 0.f;
  for (int k = 0; k < max(max(jaw_period, teeth_period), lips_period); ++k) {
    const float px = prices_tm[idx];
    if (k < jaw_period)  sum_j += px;
    if (k < teeth_period) sum_t += px;
    if (k < lips_period)  sum_l += px;
    idx += stride;
  }
  float prev_j = sum_j * aj;
  float prev_t = sum_t * at;
  float prev_l = sum_l * al;
  if (warm_j < series_len) out_jaw_tm[size_t(warm_j) * stride + col] = prev_j;
  if (warm_t < series_len) out_teeth_tm[size_t(warm_t) * stride + col] = prev_t;
  if (warm_l < series_len) out_lips_tm[size_t(warm_l) * stride + col] = prev_l;


  int start_i = min(min(warm_base_j, warm_base_t), warm_base_l) + 1;
  size_t t_idx = size_t(start_i) * stride + col;
  for (int i = start_i; i < series_len; ++i, t_idx += stride) {
    const float px = prices_tm[t_idx];
    if (i > warm_base_j) prev_j = fmaf(prev_j, bj, px * aj);
    if (i > warm_base_t) prev_t = fmaf(prev_t, bt, px * at);
    if (i > warm_base_l) prev_l = fmaf(prev_l, bl, px * al);
    const int tj = i + jaw_offset;
    const int tt = i + teeth_offset;
    const int tl = i + lips_offset;
    if (tj < series_len && i >= warm_base_j) out_jaw_tm[size_t(tj) * stride + col] = prev_j;
    if (tt < series_len && i >= warm_base_t) out_teeth_tm[size_t(tt) * stride + col] = prev_t;
    if (tl < series_len && i >= warm_base_l) out_lips_tm[size_t(tl) * stride + col] = prev_l;
  }
}


// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/alligator.rs
//   * `alligator_with_kernel` (:300) — first_valid = first non-NaN of the
//     SOURCE, and the source is `hl2` (cpu_batch.rs:13912
//     `extract_slice_input("alligator", req.data, "hl2")`), NOT close. Handing
//     this kernel close computes a different indicator.
//   * `alligator_scalar` (:552) — jaw/teeth/lips NaN prefixes are
//     `first + period - 1 + offset`, one per line.
//   * `alligator_smma_scalar` (:706) — the arithmetic reproduced below.
//
// PERIOD-INVARIANT. `compute_alligator_batch` (cpu_batch.rs:13938) reads
// jaw_period 13 / jaw_offset 8, teeth_period 8 / teeth_offset 5,
// lips_period 5 / lips_offset 3 and NEVER reads `period`.
//
// DEFAULT OUTPUT is JAW: cpu_batch.rs:13914 maps output_id "value" to
// `AlligatorOutputField::Jaw`. `neoethos_alligator_jaw_f64` is therefore the
// entry point the registry names; teeth and lips ship beside it so an
// output-aware caller can reach them without a second conversion pass.
//
// ROUNDING COUNT. The CPU recurrence is
//     smma = (smma * scale + x) * inv_period;
// — multiply, add, multiply: THREE roundings, and NOT a `mul_add`. Written
// literally here; `-fmad=false` keeps nvcc from fusing the first two.
//
// The seed is `sum / period` — a true divide, not a multiply by the reciprocal.
//
// Sequential: `smma` carries across bars. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_alli() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// One SMMA line of the alligator, written into `row` shifted by `offset`.
__device__ __forceinline__ void nef_alligator_line(const double* __restrict__ src,
                                                   int n,
                                                   int first_valid,
                                                   int period,
                                                   int offset,
                                                   double* __restrict__ row)
{
    const double QNAN = nef_qnan_alli();

    // alligator_scalar:553 — warm = first + period - 1 + offset.
    long long warm_ll = (long long)first_valid + (long long)period - 1 + (long long)offset;
    int warm = warm_ll > (long long)n ? n : (int)warm_ll;
    if (warm < 0) warm = 0;
    for (int i = 0; i < warm; ++i) row[i] = QNAN;
    // Bars after the warm prefix that the shifted write never reaches keep the
    // CPU's NaN too — the CPU leaves them at the prefix value only when
    // shifted_index >= len, which cannot happen below warm.
    for (int i = warm; i < n; ++i) row[i] = QNAN;

    if (period <= 0 || first_valid < 0 || first_valid >= n) return;

    const double scale = (double)(period - 1);
    const double inv_period = 1.0 / (double)period;

    double sum = 0.0;
    double val = 0.0;
    bool ready = false;

    for (int i = first_valid; i < n; ++i) {
        const double x = src[i];
        if (!ready) {
            if (i < first_valid + period) {
                sum += x;
                if (i == first_valid + period - 1) {
                    val = sum / (double)period;
                    ready = true;
                    const int shifted = i + offset;
                    if (shifted < n) row[shifted] = val;
                }
            }
        } else {
            val = (val * scale + x) * inv_period;
            const int shifted = i + offset;
            if (shifted < n) row[shifted] = val;
        }
    }
}

#define NEF_ALLI_JAW_PERIOD   13
#define NEF_ALLI_JAW_OFFSET    8
#define NEF_ALLI_TEETH_PERIOD  8
#define NEF_ALLI_TEETH_OFFSET  5
#define NEF_ALLI_LIPS_PERIOD   5
#define NEF_ALLI_LIPS_OFFSET   3

extern "C" __global__
void neoethos_alligator_jaw_f64(const double* __restrict__ hl2,
                                int n,
                                const int* __restrict__ periods,
                                int n_combos,
                                int first_valid,
                                double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;  // PERIOD-INVARIANT.
    nef_alligator_line(hl2, n, first_valid,
                       NEF_ALLI_JAW_PERIOD, NEF_ALLI_JAW_OFFSET,
                       out + (size_t)r * (size_t)n);
}

extern "C" __global__
void neoethos_alligator_teeth_f64(const double* __restrict__ hl2,
                                  int n,
                                  const int* __restrict__ periods,
                                  int n_combos,
                                  int first_valid,
                                  double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;
    nef_alligator_line(hl2, n, first_valid,
                       NEF_ALLI_TEETH_PERIOD, NEF_ALLI_TEETH_OFFSET,
                       out + (size_t)r * (size_t)n);
}

extern "C" __global__
void neoethos_alligator_lips_f64(const double* __restrict__ hl2,
                                 int n,
                                 const int* __restrict__ periods,
                                 int n_combos,
                                 int first_valid,
                                 double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;
    nef_alligator_line(hl2, n, first_valid,
                       NEF_ALLI_LIPS_PERIOD, NEF_ALLI_LIPS_OFFSET,
                       out + (size_t)r * (size_t)n);
}


// ===========================================================================
// S1 f64 LANE  --  alligator
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/alligator.rs -- `alligator_smma_one_scalar` (:414), `alligator_scalar` (:552), `alligator_with_kernel` (:300)
//
// SOURCE SERIES IS hl2, NOT close: `compute_alligator_batch`
// (cpu_batch.rs:13912) calls `extract_slice_input("alligator", req.data,
// "hl2")`. Declared `Hl2Slice` for the same reason `cci`/`mfi` declare hlc3.
//
// PERIOD-INVARIANT. The batch arm reads jaw/teeth/lips period and offset
// (13/8, 8/5, 5/3) and has no `period` parameter.
//
// PRIMARY OUTPUT: `jaw`. cpu_batch.rs:13914 maps output_id "value" to
// `AlligatorOutputField::Jaw`. Teeth and lips are separate series and this
// lane carries one matrix per launch; the three are INDEPENDENT smoothings of
// the same input, so dropping two changes no value of the first.
//
// ARITHMETIC ORDER: the SMMA update is
// `value = (value * scale + data_point) * inv_period` with
// `scale = period - 1` -- multiply, add, multiply: THREE roundings, and NOT
// `mul_add`. Note this differs from the Wilder form `(x - y).mul_add(a, y)`
// used elsewhere in this crate: same mathematics, different rounding count,
// and the CPU here spells it the three-rounding way.
// The seed is `sum / period` over the first `period` bars accumulated
// ASCENDING -- plain `+=`, no grouping.
//
// THE OFFSET IS A WRITE SHIFT, NOT A READ SHIFT: the value computed at bar i
// is stored at `i + offset`, and only when that index is inside the series.
// Bars between the seed and the shift therefore keep their NaN prefix.
//
// `alligator_avx2` and `alligator_avx512` both call `alligator_scalar`
// verbatim (alligator.rs:590-620), so there is one CPU answer on every host.
//
// WARMUP: `first + jaw_period - 1 + jaw_offset` for the jaw series.
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

extern "C" __global__ void neoethos_alligator_batch_f64(
    const double* __restrict__ hl2,
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

    // `AlligatorParams` defaults as read by cpu_batch.rs:13939-13944.
    const int jaw_period = 13;
    const int jaw_offset = 8;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (jaw_period == 0) || (jaw_period > n) || (jaw_offset > n) ||
        ((n - first_valid) < jaw_period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    // The whole row starts as NaN: the offset means the written indices are
    // not contiguous from the warmup, so every bar the CPU never assigns must
    // stay NaN rather than hold whatever the allocation contained.
    for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();

    const double scale = (double)(jaw_period - 1);
    const double inv_period = 1.0 / (double)jaw_period;

    double sum = 0.0;
    double value = 0.0;
    bool ready = false;

    for (int i = first_valid; i < n; ++i) {
        const double data_point = hl2[i];
        if (!ready) {
            if (i < first_valid + jaw_period) {
                sum += data_point;
                if (i == first_valid + jaw_period - 1) {
                    value = sum / (double)jaw_period;
                    ready = true;
                    const int shifted = i + jaw_offset;
                    if (shifted < n) row[shifted] = value;
                }
            }
        } else {
            value = (value * scale + data_point) * inv_period;
            const int shifted = i + jaw_offset;
            if (shifted < n) row[shifted] = value;
        }
    }
}
