#include <cuda_runtime.h>

#ifndef __CUDACC_RTC__

static __device__ __forceinline__ float fast_nan() {
    return __int_as_float(0x7fffffff);
}
#else

static __device__ __forceinline__ float fast_nan() { return nanf(""); }
#endif


extern "C" __global__ __launch_bounds__(256, 2)
void keltner_batch_f32(
    const float* __restrict__ ma_rows,
    const float* __restrict__ atr_rows,
    const int*   __restrict__ row_period_idx,
    const float* __restrict__ row_multipliers,
    const int*   __restrict__ row_warms,
    int len,
    int rows,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    const int r = blockIdx.y;
    if (r >= rows) return;


    __shared__ int   s_pidx;
    __shared__ int   s_warm;
    __shared__ float s_mult;
    if (threadIdx.x == 0) {
        s_pidx = row_period_idx[r];
        s_warm = row_warms[r];
        s_mult = row_multipliers[r];
    }
    __syncthreads();

    const float neg_mult = -s_mult;
    const float nanv     = fast_nan();


    const size_t base_p = static_cast<size_t>(s_pidx) * static_cast<size_t>(len);
    const size_t base_r = static_cast<size_t>(r)      * static_cast<size_t>(len);

    const float* __restrict__ ma  = ma_rows  + base_p;
    const float* __restrict__ atr = atr_rows + base_p;
    float* __restrict__ outM = out_middle + base_r;
    float* __restrict__ outU = out_upper  + base_r;
    float* __restrict__ outL = out_lower  + base_r;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;


    if ((len & 3) == 0) {
        const int len4 = len >> 2;
        for (int i4 = t0; i4 < len4; i4 += stride) {
            const int t = (i4 << 2);

            if (t + 3 < s_warm) {
                const float4 n4 = make_float4(nanv, nanv, nanv, nanv);
                reinterpret_cast<float4*>(outM)[i4] = n4;
                reinterpret_cast<float4*>(outU)[i4] = n4;
                reinterpret_cast<float4*>(outL)[i4] = n4;
                continue;
            }

            const float4 mid4 = reinterpret_cast<const float4*>(ma )[i4];
            const float4 a4   = reinterpret_cast<const float4*>(atr)[i4];

            const bool v0 = (t + 0) >= s_warm;
            const bool v1 = (t + 1) >= s_warm;
            const bool v2 = (t + 2) >= s_warm;
            const bool v3 = (t + 3) >= s_warm;

            const float m0 = v0 ? mid4.x : nanv;
            const float m1 = v1 ? mid4.y : nanv;
            const float m2 = v2 ? mid4.z : nanv;
            const float m3 = v3 ? mid4.w : nanv;

            const float u0 = v0 ? fmaf(s_mult, a4.x, mid4.x) : nanv;
            const float u1 = v1 ? fmaf(s_mult, a4.y, mid4.y) : nanv;
            const float u2 = v2 ? fmaf(s_mult, a4.z, mid4.z) : nanv;
            const float u3 = v3 ? fmaf(s_mult, a4.w, mid4.w) : nanv;

            const float l0 = v0 ? fmaf(neg_mult, a4.x, mid4.x) : nanv;
            const float l1 = v1 ? fmaf(neg_mult, a4.y, mid4.y) : nanv;
            const float l2 = v2 ? fmaf(neg_mult, a4.z, mid4.z) : nanv;
            const float l3 = v3 ? fmaf(neg_mult, a4.w, mid4.w) : nanv;

            reinterpret_cast<float4*>(outM)[i4] = make_float4(m0, m1, m2, m3);
            reinterpret_cast<float4*>(outU)[i4] = make_float4(u0, u1, u2, u3);
            reinterpret_cast<float4*>(outL)[i4] = make_float4(l0, l1, l2, l3);
        }
        return;
    }


    for (int t = t0; t < len; t += stride) {
        if (t < s_warm) {
            outM[t] = nanv; outU[t] = nanv; outL[t] = nanv;
            continue;
        }
        const float mid = ma[t];
        const float a   = atr[t];
        outM[t] = mid;
        outU[t] = fmaf(s_mult,  a, mid);
        outL[t] = fmaf(neg_mult, a, mid);
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void keltner_many_series_one_param_f32(
    const float* __restrict__ ma_tm,
    const float* __restrict__ atr_tm,
    const int*   __restrict__ first_valids,
    int period,
    int cols,
    int rows,
    int elems,
    float multiplier,
    float* __restrict__ out_upper_tm,
    float* __restrict__ out_middle_tm,
    float* __restrict__ out_lower_tm
) {
    const float nanv      = fast_nan();
    const float neg_mult  = -multiplier;


    if (gridDim.y > 1) {
        const int t = blockIdx.y;
        if (t >= rows) return;

        const int s0     = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        const size_t row_off = static_cast<size_t>(t) * static_cast<size_t>(cols);
        const float* __restrict__ ma_row  = ma_tm  + row_off;
        const float* __restrict__ atr_row = atr_tm + row_off;
        float* __restrict__ outM_row = out_middle_tm + row_off;
        float* __restrict__ outU_row = out_upper_tm  + row_off;
        float* __restrict__ outL_row = out_lower_tm  + row_off;

        if ((cols & 3) == 0) {
            const int cols4 = cols >> 2;
            for (int i4 = s0; i4 < cols4; i4 += stride) {
                const int s = (i4 << 2);


                const int4 fv4 = reinterpret_cast<const int4*>(first_valids)[i4];
                const int w0 = fv4.x + period - 1;
                const int w1 = fv4.y + period - 1;
                const int w2 = fv4.z + period - 1;
                const int w3 = fv4.w + period - 1;

                const bool v0 = t >= w0;
                const bool v1 = t >= w1;
                const bool v2 = t >= w2;
                const bool v3 = t >= w3;


                if (!(v0 | v1 | v2 | v3)) {
                    const float4 n4 = make_float4(nanv, nanv, nanv, nanv);
                    reinterpret_cast<float4*>(outM_row)[i4] = n4;
                    reinterpret_cast<float4*>(outU_row)[i4] = n4;
                    reinterpret_cast<float4*>(outL_row)[i4] = n4;
                    continue;
                }

                const float4 mid4 = reinterpret_cast<const float4*>(ma_row )[i4];
                const float4 a4   = reinterpret_cast<const float4*>(atr_row)[i4];

                const float m0 = v0 ? mid4.x : nanv;
                const float m1 = v1 ? mid4.y : nanv;
                const float m2 = v2 ? mid4.z : nanv;
                const float m3 = v3 ? mid4.w : nanv;

                const float u0 = v0 ? fmaf(multiplier, a4.x, mid4.x) : nanv;
                const float u1 = v1 ? fmaf(multiplier, a4.y, mid4.y) : nanv;
                const float u2 = v2 ? fmaf(multiplier, a4.z, mid4.z) : nanv;
                const float u3 = v3 ? fmaf(multiplier, a4.w, mid4.w) : nanv;

                const float l0 = v0 ? fmaf(neg_mult, a4.x, mid4.x) : nanv;
                const float l1 = v1 ? fmaf(neg_mult, a4.y, mid4.y) : nanv;
                const float l2 = v2 ? fmaf(neg_mult, a4.z, mid4.z) : nanv;
                const float l3 = v3 ? fmaf(neg_mult, a4.w, mid4.w) : nanv;

                reinterpret_cast<float4*>(outM_row)[i4] = make_float4(m0, m1, m2, m3);
                reinterpret_cast<float4*>(outU_row)[i4] = make_float4(u0, u1, u2, u3);
                reinterpret_cast<float4*>(outL_row)[i4] = make_float4(l0, l1, l2, l3);
            }
            return;
        }


        for (int s = s0; s < cols; s += stride) {
            const int warm = first_valids[s] + period - 1;
            if (t < warm) {
                outM_row[s] = nanv; outU_row[s] = nanv; outL_row[s] = nanv;
                continue;
            }
            const float mid = ma_row[s];
            const float a   = atr_row[s];
            outM_row[s] = mid;
            outU_row[s] = fmaf(multiplier, a, mid);
            outL_row[s] = fmaf(neg_mult,  a, mid);
        }
        return;
    }


    {
        const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int idx = i0; idx < elems; idx += stride) {
            const int t = idx / cols;
            const int s = idx - t * cols;
            const int warm = first_valids[s] + period - 1;
            if (t < warm) {
                out_middle_tm[idx] = nanv;
                out_upper_tm [idx] = nanv;
                out_lower_tm [idx] = nanv;
                continue;
            }
            const float mid = ma_tm[idx];
            const float a   = atr_tm[idx];
            out_middle_tm[idx] = mid;
            out_upper_tm [idx] = fmaf(multiplier, a, mid);
            out_lower_tm [idx] = fmaf(-multiplier, a, mid);
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

static __device__ __forceinline__ double fast_nan_f64() {
    return __longlong_as_double(0x7fffffffffffffffULL);
}
static __device__ __forceinline__ double fast_nan_f64() { return nan(""); }
extern "C" __global__ __launch_bounds___f64(256, 2)
void keltner_batch_f32(
    const double* __restrict__ ma_rows,
    const double* __restrict__ atr_rows,
    const int*   __restrict__ row_period_idx,
    const double* __restrict__ row_multipliers,
    const int*   __restrict__ row_warms,
    int len,
    int rows,
    double* __restrict__ out_upper,
    double* __restrict__ out_middle,
    double* __restrict__ out_lower
) {
    const int r = blockIdx.y;
    if (r >= rows) return;


    __shared__ int   s_pidx;
    __shared__ int   s_warm;
    __shared__ double s_mult;
    if (threadIdx.x == 0) {
        s_pidx = row_period_idx[r];
        s_warm = row_warms[r];
        s_mult = row_multipliers[r];
    }
    __syncthreads();

    const double neg_mult = -s_mult;
    const double nanv     = fast_nan_f64();


    const size_t base_p = static_cast<size_t>(s_pidx) * static_cast<size_t>(len);
    const size_t base_r = static_cast<size_t>(r)      * static_cast<size_t>(len);

    const double* __restrict__ ma  = ma_rows  + base_p;
    const double* __restrict__ atr = atr_rows + base_p;
    double* __restrict__ outM = out_middle + base_r;
    double* __restrict__ outU = out_upper  + base_r;
    double* __restrict__ outL = out_lower  + base_r;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;


    if ((len & 3) == 0) {
        const int len4 = len >> 2;
        for (int i4 = t0; i4 < len4; i4 += stride) {
            const int t = (i4 << 2);

            if (t + 3 < s_warm) {
                const double4 n4 = make_double4(nanv, nanv, nanv, nanv);
                reinterpret_cast<double4*>(outM)[i4] = n4;
                reinterpret_cast<double4*>(outU)[i4] = n4;
                reinterpret_cast<double4*>(outL)[i4] = n4;
                continue;
            }

            const double4 mid4 = reinterpret_cast<const double4*>(ma )[i4];
            const double4 a4   = reinterpret_cast<const double4*>(atr)[i4];

            const bool v0 = (t + 0) >= s_warm;
            const bool v1 = (t + 1) >= s_warm;
            const bool v2 = (t + 2) >= s_warm;
            const bool v3 = (t + 3) >= s_warm;

            const double m0 = v0 ? mid4.x : nanv;
            const double m1 = v1 ? mid4.y : nanv;
            const double m2 = v2 ? mid4.z : nanv;
            const double m3 = v3 ? mid4.w : nanv;

            const double u0 = v0 ? fma(s_mult, a4.x, mid4.x) : nanv;
            const double u1 = v1 ? fma(s_mult, a4.y, mid4.y) : nanv;
            const double u2 = v2 ? fma(s_mult, a4.z, mid4.z) : nanv;
            const double u3 = v3 ? fma(s_mult, a4.w, mid4.w) : nanv;

            const double l0 = v0 ? fma(neg_mult, a4.x, mid4.x) : nanv;
            const double l1 = v1 ? fma(neg_mult, a4.y, mid4.y) : nanv;
            const double l2 = v2 ? fma(neg_mult, a4.z, mid4.z) : nanv;
            const double l3 = v3 ? fma(neg_mult, a4.w, mid4.w) : nanv;

            reinterpret_cast<double4*>(outM)[i4] = make_double4(m0, m1, m2, m3);
            reinterpret_cast<double4*>(outU)[i4] = make_double4(u0, u1, u2, u3);
            reinterpret_cast<double4*>(outL)[i4] = make_double4(l0, l1, l2, l3);
        }
        return;
    }


    for (int t = t0; t < len; t += stride) {
        if (t < s_warm) {
            outM[t] = nanv; outU[t] = nanv; outL[t] = nanv;
            continue;
        }
        const double mid = ma[t];
        const double a   = atr[t];
        outM[t] = mid;
        outU[t] = fma(s_mult,  a, mid);
        outL[t] = fma(neg_mult, a, mid);
    }
}
extern "C" __global__ __launch_bounds___f64(256, 2)
void keltner_many_series_one_param_f32(
    const double* __restrict__ ma_tm,
    const double* __restrict__ atr_tm,
    const int*   __restrict__ first_valids,
    int period,
    int cols,
    int rows,
    int elems,
    double multiplier,
    double* __restrict__ out_upper_tm,
    double* __restrict__ out_middle_tm,
    double* __restrict__ out_lower_tm
) {
    const double nanv      = fast_nan_f64();
    const double neg_mult  = -multiplier;


    if (gridDim.y > 1) {
        const int t = blockIdx.y;
        if (t >= rows) return;

        const int s0     = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        const size_t row_off = static_cast<size_t>(t) * static_cast<size_t>(cols);
        const double* __restrict__ ma_row  = ma_tm  + row_off;
        const double* __restrict__ atr_row = atr_tm + row_off;
        double* __restrict__ outM_row = out_middle_tm + row_off;
        double* __restrict__ outU_row = out_upper_tm  + row_off;
        double* __restrict__ outL_row = out_lower_tm  + row_off;

        if ((cols & 3) == 0) {
            const int cols4 = cols >> 2;
            for (int i4 = s0; i4 < cols4; i4 += stride) {
                const int s = (i4 << 2);


                const int4 fv4 = reinterpret_cast<const int4*>(first_valids)[i4];
                const int w0 = fv4.x + period - 1;
                const int w1 = fv4.y + period - 1;
                const int w2 = fv4.z + period - 1;
                const int w3 = fv4.w + period - 1;

                const bool v0 = t >= w0;
                const bool v1 = t >= w1;
                const bool v2 = t >= w2;
                const bool v3 = t >= w3;


                if (!(v0 | v1 | v2 | v3)) {
                    const double4 n4 = make_double4(nanv, nanv, nanv, nanv);
                    reinterpret_cast<double4*>(outM_row)[i4] = n4;
                    reinterpret_cast<double4*>(outU_row)[i4] = n4;
                    reinterpret_cast<double4*>(outL_row)[i4] = n4;
                    continue;
                }

                const double4 mid4 = reinterpret_cast<const double4*>(ma_row )[i4];
                const double4 a4   = reinterpret_cast<const double4*>(atr_row)[i4];

                const double m0 = v0 ? mid4.x : nanv;
                const double m1 = v1 ? mid4.y : nanv;
                const double m2 = v2 ? mid4.z : nanv;
                const double m3 = v3 ? mid4.w : nanv;

                const double u0 = v0 ? fma(multiplier, a4.x, mid4.x) : nanv;
                const double u1 = v1 ? fma(multiplier, a4.y, mid4.y) : nanv;
                const double u2 = v2 ? fma(multiplier, a4.z, mid4.z) : nanv;
                const double u3 = v3 ? fma(multiplier, a4.w, mid4.w) : nanv;

                const double l0 = v0 ? fma(neg_mult, a4.x, mid4.x) : nanv;
                const double l1 = v1 ? fma(neg_mult, a4.y, mid4.y) : nanv;
                const double l2 = v2 ? fma(neg_mult, a4.z, mid4.z) : nanv;
                const double l3 = v3 ? fma(neg_mult, a4.w, mid4.w) : nanv;

                reinterpret_cast<double4*>(outM_row)[i4] = make_double4(m0, m1, m2, m3);
                reinterpret_cast<double4*>(outU_row)[i4] = make_double4(u0, u1, u2, u3);
                reinterpret_cast<double4*>(outL_row)[i4] = make_double4(l0, l1, l2, l3);
            }
            return;
        }


        for (int s = s0; s < cols; s += stride) {
            const int warm = first_valids[s] + period - 1;
            if (t < warm) {
                outM_row[s] = nanv; outU_row[s] = nanv; outL_row[s] = nanv;
                continue;
            }
            const double mid = ma_row[s];
            const double a   = atr_row[s];
            outM_row[s] = mid;
            outU_row[s] = fma(multiplier, a, mid);
            outL_row[s] = fma(neg_mult,  a, mid);
        }
        return;
    }


    {
        const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int idx = i0; idx < elems; idx += stride) {
            const int t = idx / cols;
            const int s = idx - t * cols;
            const int warm = first_valids[s] + period - 1;
            if (t < warm) {
                out_middle_tm[idx] = nanv;
                out_upper_tm [idx] = nanv;
                out_lower_tm [idx] = nanv;
                continue;
            }
            const double mid = ma_tm[idx];
            const double a   = atr_tm[idx];
            out_middle_tm[idx] = mid;
            out_upper_tm [idx] = fma(multiplier, a, mid);
            out_lower_tm [idx] = fma(-multiplier, a, mid);
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer 6
//
// CPU reference: `keltner_scalar` (src/indicators/keltner.rs:684), reached
// from `keltner_with_kernel` (:258) on the `Kernel::Scalar` arm (:317).
// NOT `keltner_scalar_classic_ema` (:608) -- that one is a separate entry
// point this dispatch path never takes, and it differs: it materialises an
// `atr_values` vector over the WHOLE series and guards each write on
// `!atr_v.is_nan()`, where :684 carries a single scalar and writes
// unconditionally. Reading the wrong one of the two would have produced a
// kernel that agrees on most bars and parts company on the warmup.
//
// OUTPUT: `upper_band`. `compute_keltner_batch:6232` maps output_id "value"
// onto `out.upper_band`. middle and lower are the same walk with
// `mid` and `(-m).mul_add(atr, mid)`; both are one launch away once the lane
// grows an output selector.
//
// PERIOD-SWEPT. `compute_keltner_batch:6212` reads a parameter literally
// named "period" (default 20), unlike the ten period-invariant variants in
// shard 4. `multiplier` (2.0, :6213) and `ma_type` ("ema", :6214) are pinned
// at the CPU defaults because the lane sweeps `period` alone.
//
// MA TYPE: "ema". That is the CPU default and it selects the :814 branch of
// `keltner_scalar`, whose middle band is an EMA SEEDED WITH AN SMA of the
// first `period` source values from `first` -- not a bare EMA. The "sma"
// branch (:780) is a sliding sum and is a different indicator; it is not
// reachable from the batch defaults.
//
// FIRST VALID IS CLOSE ALONE, and this is the trap in this indicator.
// `keltner_with_kernel:293-296` scans CLOSE only for the first non-NaN, so
// the rule is `F64FirstValidRule::HlcCloseOnly` -- the same one `adxr` and
// `cksp` carry -- and NOT `AllInputsNonNan` over the triple. On a frame
// where high or low starts later than close the two answers differ and the
// whole series shifts.
//
// THE ATR SEED IGNORES first_valid ENTIRELY. :707-724 starts at
// `high[0] - low[0]` and sums the true range over bars 1..period from index
// ZERO, then Wilder-smooths from `period` up to `warm`. `first` only enters
// through `warm = first + period - 1`. That is a quirk of the CPU -- a
// leading NaN bar poisons the ATR seed for every subsequent bar -- and it is
// REPRODUCED here rather than corrected, because the CPU is the oracle and
// "fixing" it on the device alone would put the two out of parity.
//
// SOURCE IS CLOSE. `compute_keltner_batch:6215-6219` passes `close` in both
// the close slot and the source slot, so this kernel reads the close
// pointer for the EMA leg and needs no fourth series.
//
// ROUNDINGS, counted against the CPU line by line:
//   * `atr = (tr - atr).mul_add(rma_alpha, atr)` (:722, :766) -- ONE fused
//     step. Written `fma(tr - atr, rma_alpha, atr)`. The
//     `(atr * (p - 1) + tr) / p` reformulation is THREE roundings and is
//     exactly the natr bug the brief names.
//   * `ema = (xi - ema).mul_add(ema_alpha, ema)` (:769) -- ONE fused step.
//   * `upper = m.mul_add(atr, ema)` (:772) -- ONE fused step.
// -fmad=false forbids the compiler from contracting anything else, so these
// three explicit `fma` calls are the only fused operations in the kernel.
//
// NaN: `hl.max(hc).max(lc)` (:765) is `f64::max`, which RETURNS THE NON-NaN
// OPERAND. An if-chain does not -- a comparison against NaN is false, so the
// NaN survives and poisons every later bar of the Wilder recursion. This is
// the adx-class bug named in the brief. Written here as
// `fmax(fmax(hl, hc), lc)`, which has CUDA's identical non-NaN-preferring
// semantics, in the CPU's exact left-to-right nesting.
//
// f32 -> f64 audit of this section: no f32 literal, no f32-suffixed math
// function (the f32 lane above uses fmaxf and fabsf), no fast-math
// intrinsic. The quiet NaN is built from the f64 bit pattern, not from
// `__int_as_float`.
// ===========================================================================

// `get_f64_param("keltner", params, "multiplier", 2.0)` -- cpu_batch.rs:6213.
#define KELTNER_F64_MULTIPLIER 2.0

static __device__ __forceinline__ double keltner_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void keltner_batch_f64(const double* __restrict__ high,
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

    const double nan_d = keltner_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);

    const int period = periods[r];

    // `keltner_with_kernel` (:287-303) errors -- and `collect_f64` turns the
    // error into an all-NaN column -- on a zero or oversized period and when
    // the valid tail is shorter than the period.
    if (n <= 0 || first_valid < 0 || first_valid >= n ||
        period <= 0 || period > n || n - first_valid < period) {
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }

    const int warm = first_valid + period - 1;                         // :310
    // `keltner_scalar:697-699` returns without writing anything when the
    // warmup is past the end; the prefix allocation (:311) has already made
    // the whole column NaN.
    for (int i = 0; i < n && i < warm; ++i) row[i] = nan_d;
    if (warm >= n) return;

    const double pf = static_cast<double>(period);                     // :702
    const double rma_alpha = 1.0 / pf;                                 // :703

    // ------------------------------------------------------------------
    // ATR seed. :707-724. Starts at index 0, NOT at first_valid.
    // ------------------------------------------------------------------
    double atr = high[0] - low[0];                                     // :707
    for (int i = 1; i < period; ++i) {                                 // :709-718
        const double hi = high[i];
        const double lo = low[i];
        const double pc = close[i - 1];
        const double hl = hi - lo;
        const double hc = fabs(hi - pc);
        const double lc = fabs(lo - pc);
        atr += fmax(fmax(hl, hc), lc);                                 // :716
    }
    atr /= pf;                                                         // :719

    for (int k = period; k <= warm; ++k) {                             // :721-731
        const double hi = high[k];
        const double lo = low[k];
        const double pc = close[k - 1];
        const double hl = hi - lo;
        const double hc = fabs(hi - pc);
        const double lc = fabs(lo - pc);
        const double tr = fmax(fmax(hl, hc), lc);                      // :728
        atr = fma(tr - atr, rma_alpha, atr);                           // :729
    }

    const double m = KELTNER_F64_MULTIPLIER;                           // :734

    // ------------------------------------------------------------------
    // ma_type == "ema". :736-778.
    // ------------------------------------------------------------------
    double ema = 0.0;                                                  // :737
    for (int j = 0; j < period; ++j) {                                 // :740-743
        ema += close[first_valid + j];
    }
    ema /= pf;                                                         // :745

    row[warm] = fma(m, atr, ema);                                      // :748 upper

    const double ema_alpha = 2.0 / (pf + 1.0);                         // :752

    for (int i = warm + 1; i < n; ++i) {                               // :755-775
        const double hi = high[i];
        const double lo = low[i];
        const double pc = close[i - 1];
        const double hl = hi - lo;
        const double hc = fabs(hi - pc);
        const double lc = fabs(lo - pc);
        const double tr = fmax(fmax(hl, hc), lc);                      // :765
        atr = fma(tr - atr, rma_alpha, atr);                           // :766

        const double xi = close[i];                                    // :768
        ema = fma(xi - ema, ema_alpha, ema);                           // :769

        row[i] = fma(m, atr, ema);                                     // :772 upper
    }
}
