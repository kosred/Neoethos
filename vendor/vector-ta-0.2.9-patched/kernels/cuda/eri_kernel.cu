#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef ERI_TIME_TILE

#define ERI_TIME_TILE 16
#endif


__device__ __forceinline__ float eri_qnan() {

    return nanf("");
}


extern "C" __global__ void eri_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma,
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ bull,
    float* __restrict__ bear
) {
    const int stride = blockDim.x * gridDim.x;
    const int warm   = first_valid + period - 1;
    const float nanv = eri_qnan();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += stride) {
        if (i < warm) {
            if (bull) bull[i] = nanv;
            if (bear) bear[i] = nanv;
        } else {
            const float m = ma[i];
            if (bull) bull[i] = high[i] - m;
            if (bear) bear[i] = low[i]  - m;
        }
    }
}


extern "C" __global__ void eri_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    int period,
    float* __restrict__ bull_tm,
    float* __restrict__ bear_tm
) {
    const int s  = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int warm   = first_valids[s] + period - 1;
    const float nanv = eri_qnan();


    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);


        int prefix = warm - t0;
        if (prefix < 0) prefix = 0;
        if (prefix > tlimit) prefix = tlimit;
        if (prefix > 0) {
            for (int tt = 0; tt < prefix; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                if (bull_tm) bull_tm[idx] = nanv;
                if (bear_tm) bear_tm[idx] = nanv;
            }
        }


        if (prefix < tlimit) {
            for (int tt = prefix; tt < tlimit; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                const float m = ma_tm[idx];
                if (bull_tm) bull_tm[idx] = high_tm[idx] - m;
                if (bear_tm) bear_tm[idx] = low_tm[idx]  - m;
            }
        }
    }
}


extern "C" __global__ void eri_one_series_many_params_time_major_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_tm,
    int P,
    int rows,
    int first_valid,
    const int* __restrict__ periods,
    int period,
    float* __restrict__ bull_out,
    float* __restrict__ bear_out,
    int out_row_major
) {
    __shared__ float sh_high[ERI_TIME_TILE];
    __shared__ float sh_low [ERI_TIME_TILE];

    const float nanv = eri_qnan();

    const int p0      = blockIdx.x * blockDim.x + threadIdx.x;
    const int pstride = gridDim.x  * blockDim.x;


    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);


        if (threadIdx.x < tlimit) {
            sh_high[threadIdx.x] = high[t0 + threadIdx.x];
            sh_low [threadIdx.x] = low [t0 + threadIdx.x];
        }
        __syncthreads();


        for (int p = p0; p < P; p += pstride) {
            const int per   = (periods ? periods[p] : period);
            const int warm  = first_valid + per - 1;
            const int base  = t0 * P + p;


            int prefix = warm - t0;
            if (prefix < 0) prefix = 0;
            if (prefix > tlimit) prefix = tlimit;
            if (prefix > 0) {
                if (out_row_major) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int t = t0 + tt;
                        if (bull_out) bull_out[p*rows + t] = nanv;
                        if (bear_out) bear_out[p*rows + t] = nanv;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = nanv;
                        bear_out[idx] = nanv;
                    }
                } else if (bull_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bull_out[base + tt * P] = nanv;
                    }
                } else if (bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bear_out[base + tt * P] = nanv;
                    }
                }
            }


            if (prefix < tlimit) {
                if (out_row_major) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int t = t0 + tt;
                        const float m = ma_tm[base + tt * P];
                        if (bull_out) bull_out[p*rows + t] = sh_high[tt] - m;
                        if (bear_out) bear_out[p*rows + t] = sh_low [tt] - m;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        const float m = ma_tm[idx];
                        bull_out[idx] = sh_high[tt] - m;
                        bear_out[idx] = sh_low [tt] - m;
                    }
                } else if (bull_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = sh_high[tt] - ma_tm[idx];
                    }
                } else if (bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bear_out[idx] = sh_low[tt] - ma_tm[idx];
                    }
                }
            }
        }
        __syncthreads();
    }
}


extern "C" __global__ void transpose_rm_to_tm_32x32_pad_f32(
    const float* __restrict__ in,
    int R, int C,
    float* __restrict__ out
){
    __shared__ float tile[32][32+1];

    int c0 = blockIdx.x * 32 + threadIdx.x;
    int r0 = blockIdx.y * 32 + threadIdx.y;

    if (r0 < R && c0 < C) {
        tile[threadIdx.y][threadIdx.x] = in[r0 * C + c0];
    } else {
        tile[threadIdx.y][threadIdx.x] = eri_qnan();
    }
    __syncthreads();

    int r1 = blockIdx.y * 32 + threadIdx.x;
    int c1 = blockIdx.x * 32 + threadIdx.y;
    if (r1 < R && c1 < C) {
        out[c1 * R + r1] = tile[threadIdx.x][threadIdx.y];
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

__device__ __forceinline__ double eri_qnan_f64() {

    return nan("");
}
extern "C" __global__ void eri_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ ma,
    int series_len,
    int first_valid,
    int period,
    double* __restrict__ bull,
    double* __restrict__ bear
) {
    const int stride = blockDim.x * gridDim.x;
    const int warm   = first_valid + period - 1;
    const double nanv = eri_qnan_f64();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += stride) {
        if (i < warm) {
            if (bull) bull[i] = nanv;
            if (bear) bear[i] = nanv;
        } else {
            const double m = ma[i];
            if (bull) bull[i] = high[i] - m;
            if (bear) bear[i] = low[i]  - m;
        }
    }
}
extern "C" __global__ void eri_many_series_one_param_time_major_f64(
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    const double* __restrict__ ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    int period,
    double* __restrict__ bull_tm,
    double* __restrict__ bear_tm
) {
    const int s  = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int warm   = first_valids[s] + period - 1;
    const double nanv = eri_qnan_f64();


    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);


        int prefix = warm - t0;
        if (prefix < 0) prefix = 0;
        if (prefix > tlimit) prefix = tlimit;
        if (prefix > 0) {
            for (int tt = 0; tt < prefix; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                if (bull_tm) bull_tm[idx] = nanv;
                if (bear_tm) bear_tm[idx] = nanv;
            }
        }


        if (prefix < tlimit) {
            for (int tt = prefix; tt < tlimit; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                const double m = ma_tm[idx];
                if (bull_tm) bull_tm[idx] = high_tm[idx] - m;
                if (bear_tm) bear_tm[idx] = low_tm[idx]  - m;
            }
        }
    }
}
extern "C" __global__ void eri_one_series_many_params_time_major_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ ma_tm,
    int P,
    int rows,
    int first_valid,
    const int* __restrict__ periods,
    int period,
    double* __restrict__ bull_out,
    double* __restrict__ bear_out,
    int out_row_major
) {
    __shared__ double sh_high[ERI_TIME_TILE];
    __shared__ double sh_low [ERI_TIME_TILE];

    const double nanv = eri_qnan_f64();

    const int p0      = blockIdx.x * blockDim.x + threadIdx.x;
    const int pstride = gridDim.x  * blockDim.x;


    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);


        if (threadIdx.x < tlimit) {
            sh_high[threadIdx.x] = high[t0 + threadIdx.x];
            sh_low [threadIdx.x] = low [t0 + threadIdx.x];
        }
        __syncthreads();


        for (int p = p0; p < P; p += pstride) {
            const int per   = (periods ? periods[p] : period);
            const int warm  = first_valid + per - 1;
            const int base  = t0 * P + p;


            int prefix = warm - t0;
            if (prefix < 0) prefix = 0;
            if (prefix > tlimit) prefix = tlimit;
            if (prefix > 0) {
                if (out_row_major) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int t = t0 + tt;
                        if (bull_out) bull_out[p*rows + t] = nanv;
                        if (bear_out) bear_out[p*rows + t] = nanv;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = nanv;
                        bear_out[idx] = nanv;
                    }
                } else if (bull_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bull_out[base + tt * P] = nanv;
                    }
                } else if (bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bear_out[base + tt * P] = nanv;
                    }
                }
            }


            if (prefix < tlimit) {
                if (out_row_major) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int t = t0 + tt;
                        const double m = ma_tm[base + tt * P];
                        if (bull_out) bull_out[p*rows + t] = sh_high[tt] - m;
                        if (bear_out) bear_out[p*rows + t] = sh_low [tt] - m;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        const double m = ma_tm[idx];
                        bull_out[idx] = sh_high[tt] - m;
                        bear_out[idx] = sh_low [tt] - m;
                    }
                } else if (bull_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = sh_high[tt] - ma_tm[idx];
                    }
                } else if (bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bear_out[idx] = sh_low[tt] - ma_tm[idx];
                    }
                }
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void transpose_rm_to_tm_32x32_pad_f64(
    const double* __restrict__ in,
    int R, int C,
    double* __restrict__ out
){
    __shared__ double tile[32][32+1];

    int c0 = blockIdx.x * 32 + threadIdx.x;
    int r0 = blockIdx.y * 32 + threadIdx.y;

    if (r0 < R && c0 < C) {
        tile[threadIdx.y][threadIdx.x] = in[r0 * C + c0];
    } else {
        tile[threadIdx.y][threadIdx.x] = eri_qnan_f64();
    }
    __syncthreads();

    int r1 = blockIdx.y * 32 + threadIdx.x;
    int c1 = blockIdx.x * 32 + threadIdx.y;
    if (r1 < R && c1 < C) {
        out[c1 * R + r1] = tile[threadIdx.x][threadIdx.y];
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — eri (Elder Ray Index)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/eri.rs:2176 `eri_scalar_classic_ema`, which is the
 *   branch `eri_with_kernel` (:285) takes for the DEFAULT ma_type "ema"
 *   (cpu_batch.rs:14728). The generic `eri_scalar` (:357) never runs for the
 *   default parameters — it is reached only after `ma(&ma_type, ...)` for one
 *   of the other MA families — so matching it instead would be matching a
 *   configuration the sweep does not use.
 *
 *   The `period == 13` fast path (:2226) is arithmetically IDENTICAL to the
 *   general one — same left-to-right seed sum, same `alpha*x + beta*ema`
 *   recursion — so one kernel serves both and no gate is needed. Checked term
 *   by term rather than assumed, because the analogous fast path in `epma` is
 *   NOT identical and needed reproducing.
 *
 * Column: output_id "value" / "bull" -> `out.bull` = high - ma (cpu_batch.rs
 *   :14743). `bear` is the low-side twin and is not this column.
 *
 * PERIOD-SWEPT: `compute_eri_batch` (cpu_batch.rs:14727) reads a parameter
 *   literally named `period` (default 13).
 *
 * Input: high / low / close — F64InputKind::Hlc. The CPU source default is
 *   "close" (cpu_batch.rs:14690) and the EMA runs on that source, while `high`
 *   supplies the emitted column, so all three pointers are genuine inputs.
 *
 * first_valid: the CPU scans the TRIPLE simultaneously (:249-254), which is
 *   `AllInputsNonNan`, not the max-of-independent-scans rule adx and natr use.
 *
 * Recursion: `ema = alpha * src + beta * ema` (:2204) is TWO products and one
 *   add — deliberately NOT contracted into an fma here, because the CPU line
 *   has two roundings and matching the count is the whole point.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void eri_neo_batch_f64(const double* __restrict__ high,
                       const double* __restrict__ low,
                       const double* __restrict__ close,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)low;   /* the `bear` column, not this one */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = periods[combo];
    if (period <= 0 || period > n) return;
    if (first_valid < 0 || first_valid >= n) return;
    if (n - first_valid < period) return;

    const int start_idx = first_valid + period - 1;
    if (start_idx >= n) return;

    const double alpha = 2.0 / ((double)period + 1.0);
    const double beta  = 1.0 - alpha;

    double sum = 0.0;
    for (int i = 0; i < period; ++i) sum += close[first_valid + i];
    double ema = sum / (double)period;

    o[start_idx] = high[start_idx] - ema;
    for (int i = start_idx + 1; i < n; ++i) {
        ema = alpha * close[i] + beta * ema;
        o[i] = high[i] - ema;
    }
}
