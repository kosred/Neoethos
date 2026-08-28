#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>


__device__ __forceinline__ float mfi_elem(float h, float l, float v, bool ok) {
    if (!ok) return CUDART_NAN_F;
    if (isnan(h) || isnan(l) || isnan(v) || v == 0.0f) return CUDART_NAN_F;
    return (h - l) / v;
}


extern "C" __global__ void marketefi_kernel_f32(const float* __restrict__ high,
                                                 const float* __restrict__ low,
                                                 const float* __restrict__ volume,
                                                 int len,
                                                 int first_valid,
                                                 float* __restrict__ out) {
    if (len <= 0) return;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int first  = first_valid < 0 ? 0 : first_valid;


    constexpr int ILP = 4;

    for (int base = tid; base < len; base += stride * ILP) {
#pragma unroll
        for (int k = 0; k < ILP; ++k) {
            int i = base + k * blockDim.x;
            if (i < len) {
                const bool ok = (i >= first);
                const float h = high[i];
                const float l = low[i];
                const float v = volume[i];
                out[i] = mfi_elem(h, l, v, ok);
            }
        }
    }
}


#ifndef MKT_T_TILE
#define MKT_T_TILE 128
#endif

extern "C" __global__ void marketefi_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {

    if (num_series <= 0 || series_len <= 0) return;


    const bool legacy_1d = (gridDim.y == 1) && (gridDim.x == num_series);
    if (legacy_1d) {

        const int s = blockIdx.x;
        if (s >= num_series) return;
        const int first = first_valids ? (first_valids[s] < 0 ? 0 : first_valids[s]) : 0;
        const int stride_series = num_series;


        for (int t = threadIdx.x; t < min(first, series_len); t += blockDim.x) {
            out_tm[t * stride_series + s] = CUDART_NAN_F;
        }

        for (int t = threadIdx.x + first; t < series_len; t += blockDim.x) {
            const int idx = t * stride_series + s;
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            const float v = volume_tm[idx];
            out_tm[idx] = mfi_elem(h, l, v, true);
        }
        return;
    }


    const uintptr_t mask16 = 0xF;
    const bool aligned16 =
        (((uintptr_t)high_tm   | (uintptr_t)low_tm |
          (uintptr_t)volume_tm | (uintptr_t)out_tm |
          (uintptr_t)first_valids) & mask16) == 0;
    const bool vec_ok = aligned16 && ((num_series & 3) == 0);

    if (vec_ok) {

        const int series4 = num_series >> 2;
        const int s4 = blockIdx.y * blockDim.x + threadIdx.x;
        if (s4 >= series4) return;


        int4 fv4 = make_int4(0, 0, 0, 0);
        if (first_valids) {
            const int4* __restrict__ fv_ptr = reinterpret_cast<const int4*>(first_valids);
            fv4 = fv_ptr[s4];
            fv4.x = fv4.x < 0 ? 0 : fv4.x;
            fv4.y = fv4.y < 0 ? 0 : fv4.y;
            fv4.z = fv4.z < 0 ? 0 : fv4.z;
            fv4.w = fv4.w < 0 ? 0 : fv4.w;
        }

        const float4* __restrict__ H = reinterpret_cast<const float4*>(high_tm);
        const float4* __restrict__ L = reinterpret_cast<const float4*>(low_tm);
        const float4* __restrict__ V = reinterpret_cast<const float4*>(volume_tm);
        float4* __restrict__ O       = reinterpret_cast<float4*>(out_tm);

        const int stride4_t = series4;

        for (int t0 = blockIdx.x * MKT_T_TILE; t0 < series_len; t0 += gridDim.x * MKT_T_TILE) {
            const int t_end = min(series_len, t0 + MKT_T_TILE);

#pragma unroll 4
            for (int t = t0; t < t_end; ++t) {
                const int idx4 = t * stride4_t + s4;

                const float4 h4 = H[idx4];
                const float4 l4 = L[idx4];
                const float4 v4 = V[idx4];

                float4 out4;
                out4.x = mfi_elem(h4.x, l4.x, v4.x, t >= fv4.x);
                out4.y = mfi_elem(h4.y, l4.y, v4.y, t >= fv4.y);
                out4.z = mfi_elem(h4.z, l4.z, v4.z, t >= fv4.z);
                out4.w = mfi_elem(h4.w, l4.w, v4.w, t >= fv4.w);

                O[idx4] = out4;
            }
        }
    } else {

        const int s = blockIdx.y * blockDim.x + threadIdx.x;
        if (s >= num_series) return;

        const int first = first_valids ? (first_valids[s] < 0 ? 0 : first_valids[s]) : 0;
        const int stride_series = num_series;

        for (int t0 = blockIdx.x * MKT_T_TILE; t0 < series_len; t0 += gridDim.x * MKT_T_TILE) {
            const int t_end = min(series_len, t0 + MKT_T_TILE);

#pragma unroll 4
            for (int t = t0; t < t_end; ++t) {
                const int idx = t * stride_series + s;
                const float h = high_tm[idx];
                const float l = low_tm[idx];
                const float v = volume_tm[idx];
                out_tm[idx] = mfi_elem(h, l, v, t >= first);
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

__device__ __forceinline__ double mfi_elem_f64(double h, double l, double v, bool ok) {
    if (!ok) return CUDART_NAN;
    if (isnan(h) || isnan(l) || isnan(v) || v == 0.0) return CUDART_NAN;
    return (h - l) / v;
}
extern "C" __global__ void marketefi_kernel_f64(const double* __restrict__ high,
                                                 const double* __restrict__ low,
                                                 const double* __restrict__ volume,
                                                 int len,
                                                 int first_valid,
                                                 double* __restrict__ out) {
    if (len <= 0) return;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int first  = first_valid < 0 ? 0 : first_valid;


    constexpr int ILP = 4;

    for (int base = tid; base < len; base += stride * ILP) {
#pragma unroll
        for (int k = 0; k < ILP; ++k) {
            int i = base + k * blockDim.x;
            if (i < len) {
                const bool ok = (i >= first);
                const double h = high[i];
                const double l = low[i];
                const double v = volume[i];
                out[i] = mfi_elem_f64(h, l, v, ok);
            }
        }
    }
}
extern "C" __global__ void marketefi_many_series_one_param_f64(
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    const double* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    double* __restrict__ out_tm) {

    if (num_series <= 0 || series_len <= 0) return;


    const bool legacy_1d = (gridDim.y == 1) && (gridDim.x == num_series);
    if (legacy_1d) {

        const int s = blockIdx.x;
        if (s >= num_series) return;
        const int first = first_valids ? (first_valids[s] < 0 ? 0 : first_valids[s]) : 0;
        const int stride_series = num_series;


        for (int t = threadIdx.x; t < min(first, series_len); t += blockDim.x) {
            out_tm[t * stride_series + s] = CUDART_NAN;
        }

        for (int t = threadIdx.x + first; t < series_len; t += blockDim.x) {
            const int idx = t * stride_series + s;
            const double h = high_tm[idx];
            const double l = low_tm[idx];
            const double v = volume_tm[idx];
            out_tm[idx] = mfi_elem_f64(h, l, v, true);
        }
        return;
    }


    const uintptr_t mask16 = 0xF;
    const bool aligned16 =
        (((uintptr_t)high_tm   | (uintptr_t)low_tm |
          (uintptr_t)volume_tm | (uintptr_t)out_tm |
          (uintptr_t)first_valids) & mask16) == 0;
    const bool vec_ok = aligned16 && ((num_series & 3) == 0);

    if (vec_ok) {

        const int series4 = num_series >> 2;
        const int s4 = blockIdx.y * blockDim.x + threadIdx.x;
        if (s4 >= series4) return;


        int4 fv4 = make_int4(0, 0, 0, 0);
        if (first_valids) {
            const int4* __restrict__ fv_ptr = reinterpret_cast<const int4*>(first_valids);
            fv4 = fv_ptr[s4];
            fv4.x = fv4.x < 0 ? 0 : fv4.x;
            fv4.y = fv4.y < 0 ? 0 : fv4.y;
            fv4.z = fv4.z < 0 ? 0 : fv4.z;
            fv4.w = fv4.w < 0 ? 0 : fv4.w;
        }

        const double4* __restrict__ H = reinterpret_cast<const double4*>(high_tm);
        const double4* __restrict__ L = reinterpret_cast<const double4*>(low_tm);
        const double4* __restrict__ V = reinterpret_cast<const double4*>(volume_tm);
        double4* __restrict__ O       = reinterpret_cast<double4*>(out_tm);

        const int stride4_t = series4;

        for (int t0 = blockIdx.x * MKT_T_TILE; t0 < series_len; t0 += gridDim.x * MKT_T_TILE) {
            const int t_end = min(series_len, t0 + MKT_T_TILE);

#pragma unroll 4
            for (int t = t0; t < t_end; ++t) {
                const int idx4 = t * stride4_t + s4;

                const double4 h4 = H[idx4];
                const double4 l4 = L[idx4];
                const double4 v4 = V[idx4];

                double4 out4;
                out4.x = mfi_elem_f64(h4.x, l4.x, v4.x, t >= fv4.x);
                out4.y = mfi_elem_f64(h4.y, l4.y, v4.y, t >= fv4.y);
                out4.z = mfi_elem_f64(h4.z, l4.z, v4.z, t >= fv4.z);
                out4.w = mfi_elem_f64(h4.w, l4.w, v4.w, t >= fv4.w);

                O[idx4] = out4;
            }
        }
    } else {

        const int s = blockIdx.y * blockDim.x + threadIdx.x;
        if (s >= num_series) return;

        const int first = first_valids ? (first_valids[s] < 0 ? 0 : first_valids[s]) : 0;
        const int stride_series = num_series;

        for (int t0 = blockIdx.x * MKT_T_TILE; t0 < series_len; t0 += gridDim.x * MKT_T_TILE) {
            const int t_end = min(series_len, t0 + MKT_T_TILE);

#pragma unroll 4
            for (int t = t0; t < t_end; ++t) {
                const int idx = t * stride_series + s;
                const double h = high_tm[idx];
                const double l = low_tm[idx];
                const double v = volume_tm[idx];
                out_tm[idx] = mfi_elem_f64(h, l, v, t >= first);
            }
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// WHY A SECOND ENTRY POINT RATHER THAN REGISTERING `marketefi_kernel_f64`.
// That one (:192) is a grid-stride kernel over a SINGLE series with the
// signature `(high, low, volume, len, first_valid, out)` -- no period list, no
// row dimension. The lane launches `(inputs..., n, periods, n_combos,
// first_valid, out)` and writes `n_combos` rows. Launching the existing symbol
// through the lane would pass `periods` where it expects `len`.
//
// CPU REFERENCE
// -------------
//   src/indicators/marketefi.rs
//     :206 marketefi_first_valid       -- first index where NONE of high/low/
//                                        volume is NaN (`!is_nan`, so an
//                                        infinite bar is VALID)
//     :386 marketefi_scalar_any_valid  <- the whole per-bar body
//     :314 marketefi_into_slice        -- NaN prefix is `[..first]`
//
// SHAPE
// -----
// The value is POINTWISE, `(high - low) / volume`, so it has no accumulation
// order at all. It is nevertheless written one thread per row with a bar loop,
// because the lane's bar-parallel launch arm does not accept the
// `HighLowVolume` shape (`neoethos_f64_wrapper.rs:2172`) -- it refuses rather
// than launching a three-pointer kernel against a two-pointer argument pack.
// Registering a bar-parallel kernel here would mean widening that arm, which is
// a change to shared launch code this closer does not own.
//
// PERIOD-INVARIANT. marketefi has NO period parameter -- there is no
// `compute_marketefi_batch` in cpu_batch.rs at all and `MarketefiParams` is
// only a source selector -- so every row of a sweep is byte-identical.
//
// ARITHMETIC
// ----------
// f64 end to end, no fast-math. The `v == 0.0` guard is the CPU's own and is
// an EXACT comparison, not a tolerance; it is not widened. The second guard is
// `res.is_nan()` -- note it is NaN-only, so an INFINITE result (finite range
// over a denormal volume) is PASSED THROUGH by the CPU and is passed through
// here too.

__device__ __forceinline__ double mefi_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void marketefi_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = mefi_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);

    (void)periods;   // PERIOD-INVARIANT: marketefi has no period at all.

    for (int i = 0; i < n; ++i) o[i] = nan_d;
    if (n <= 0 || first_valid < 0 || first_valid >= n) return;

    for (int i = first_valid; i < n; ++i) {
        const double v = volume[i];
        if (v == 0.0) {
            o[i] = nan_d;
        } else {
            const double res = (high[i] - low[i]) / v;
            o[i] = isnan(res) ? nan_d : res;
        }
    }
}
