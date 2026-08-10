#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[wid] = v;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) >> 5;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

static __forceinline__ __device__ bool valid_ohlc(float o, float h, float l, float c) {
    return isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
           o > 0.0f && h > 0.0f && l > 0.0f && c > 0.0f;
}

static __forceinline__ __device__ bool valid_bar(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int j
) {
    if (j <= 0) return false;
    const float o = open[j];
    const float h = high[j];
    const float l = low[j];
    const float c = close[j];
    const float pc = close[j - 1];
    return valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f;
}

static __forceinline__ __device__ void compute_terms_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int j,
    float* rs,
    float* oret,
    float* cret
) {
    const float o = open[j];
    const float h = high[j];
    const float l = low[j];
    const float c = close[j];
    const float pc = close[j - 1];
    const float a = logf(h / c);
    const float b = logf(h / o);
    const float d = logf(l / c);
    const float e = logf(l / o);
    *rs = __fmaf_rn(d, e, a * b);
    *oret = logf(o / pc);
    *cret = logf(c / o);
}

extern "C" __global__ void yang_zhang_precompute_terms_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int* __restrict__ valid_flags,
    float* __restrict__ rs_terms,
    float* __restrict__ oret_terms,
    float* __restrict__ cret_terms
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < series_len;
         j += blockDim.x * gridDim.x) {
        int valid = 0;
        float rs = 0.0f;
        float oret = 0.0f;
        float cret = 0.0f;
        if (valid_bar(open, high, low, close, j)) {
            valid = 1;
            compute_terms_f32(open, high, low, close, j, &rs, &oret, &cret);
        }
        valid_flags[j] = valid;
        rs_terms[j] = rs;
        oret_terms[j] = oret;
        cret_terms[j] = cret;
    }
}

extern "C" __global__ void yang_zhang_prefix_terms_f32(
    const int* __restrict__ valid_flags,
    const float* __restrict__ rs_terms,
    const float* __restrict__ oret_terms,
    const float* __restrict__ cret_terms,
    int series_len,
    int* __restrict__ prefix_valid,
    float* __restrict__ prefix_rs,
    float* __restrict__ prefix_o,
    float* __restrict__ prefix_oo,
    float* __restrict__ prefix_c,
    float* __restrict__ prefix_cc
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    prefix_valid[0] = 0;
    prefix_rs[0] = 0.0f;
    prefix_o[0] = 0.0f;
    prefix_oo[0] = 0.0f;
    prefix_c[0] = 0.0f;
    prefix_cc[0] = 0.0f;

    int valid_acc = 0;
    float rs_acc = 0.0f;
    float o_acc = 0.0f;
    float oo_acc = 0.0f;
    float c_acc = 0.0f;
    float cc_acc = 0.0f;
    for (int j = 0; j < series_len; ++j) {
        const float o = oret_terms[j];
        const float c = cret_terms[j];
        valid_acc += valid_flags[j];
        rs_acc += rs_terms[j];
        o_acc += o;
        oo_acc += o * o;
        c_acc += c;
        cc_acc += c * c;

        const int out = j + 1;
        prefix_valid[out] = valid_acc;
        prefix_rs[out] = rs_acc;
        prefix_o[out] = o_acc;
        prefix_oo[out] = oo_acc;
        prefix_c[out] = c_acc;
        prefix_cc[out] = cc_acc;
    }
}

extern "C" __global__ void yang_zhang_volatility_batch_prefix_f32(
    const int* __restrict__ lookbacks,
    const int* __restrict__ k_overrides,
    const float* __restrict__ k_values,
    int series_len,
    int first_valid,
    int n_combos,
    const int* __restrict__ prefix_valid,
    const float* __restrict__ prefix_rs,
    const float* __restrict__ prefix_o,
    const float* __restrict__ prefix_oo,
    const float* __restrict__ prefix_c,
    const float* __restrict__ prefix_cc,
    float* __restrict__ out_yz,
    float* __restrict__ out_rs
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    __shared__ int lookback_s;
    __shared__ int warmup_s;
    __shared__ int combo_valid_s;
    __shared__ float k_s;
    __shared__ float inv_lb_s;
    __shared__ float inv_denom_s;

    if (threadIdx.x == 0) {
        const int lookback = lookbacks[combo];
        int combo_valid = 1;
        float k = 0.0f;
        if (lookback <= 0 || lookback > series_len) {
            combo_valid = 0;
        } else if (k_overrides[combo] != 0) {
            k = k_values[combo];
            if (!isfinite(k) || k < 0.0f || k > 1.0f) {
                combo_valid = 0;
            }
        } else {
            k = (lookback <= 1)
                ? 0.0f
                : 0.34f / (1.34f + ((float)(lookback + 1) / (float)(lookback - 1)));
        }

        lookback_s = lookback;
        warmup_s = first_valid + lookback;
        combo_valid_s = combo_valid;
        k_s = k;
        inv_lb_s = combo_valid && lookback > 0 ? 1.0f / (float)lookback : 0.0f;
        inv_denom_s = combo_valid && lookback > 1 ? 1.0f / (float)(lookback - 1) : 0.0f;
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);
    const int base = combo * series_len;
    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < series_len;
         t += (int)blockDim.x * (int)gridDim.x) {
        float yz_out = nan_f;
        float rs_out = nan_f;

        if (combo_valid_s != 0 && warmup_s < series_len && t >= warmup_s) {
            const int window_start = t + 1 - lookback_s;
            const int valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
            if (valid_count == lookback_s) {
                float rs_var = (prefix_rs[t + 1] - prefix_rs[window_start]) * inv_lb_s;
                if (rs_var < 0.0f) {
                    rs_var = 0.0f;
                }

                float o_var = 0.0f;
                float c_var = 0.0f;
                if (lookback_s > 1) {
                    const float sum_o = prefix_o[t + 1] - prefix_o[window_start];
                    const float sum_oo = prefix_oo[t + 1] - prefix_oo[window_start];
                    const float sum_c = prefix_c[t + 1] - prefix_c[window_start];
                    const float sum_cc = prefix_cc[t + 1] - prefix_cc[window_start];
                    o_var = (sum_oo - sum_o * sum_o * inv_lb_s) * inv_denom_s;
                    c_var = (sum_cc - sum_c * sum_c * inv_lb_s) * inv_denom_s;
                    if (o_var < 0.0f) {
                        o_var = 0.0f;
                    }
                    if (c_var < 0.0f) {
                        c_var = 0.0f;
                    }
                }

                float yz_var = o_var + __fmaf_rn(1.0f - k_s, rs_var, k_s * c_var);
                if (yz_var < 0.0f) {
                    yz_var = 0.0f;
                }
                rs_out = sqrtf(rs_var);
                yz_out = sqrtf(yz_var);
            }
        }
        out_rs[base + t] = rs_out;
        out_yz[base + t] = yz_out;
    }
}

extern "C" __global__ void yang_zhang_volatility_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int* __restrict__ first_valids,
    int lookback,
    int k_override,
    float k_input,
    int cols,
    int rows,
    float* __restrict__ out_yz_tm,
    float* __restrict__ out_rs_tm
) {
    const int s = (int)blockIdx.x;
    if (s >= cols) {
        return;
    }

    const float nan_f = __int_as_float(0x7fffffff);
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        const int idx = t * cols + s;
        out_yz_tm[idx] = nan_f;
        out_rs_tm[idx] = nan_f;
    }
    __syncthreads();

    if (lookback <= 0 || lookback > rows) {
        return;
    }
    const int first_valid = first_valids[s];
    if (first_valid < 0 || first_valid >= rows) {
        return;
    }

    float k = 0.0f;
    if (k_override != 0) {
        k = k_input;
        if (!isfinite(k) || k < 0.0f || k > 1.0f) {
            return;
        }
    } else {
        k = (lookback <= 1)
            ? 0.0f
            : 0.34f / (1.34f + ((float)(lookback + 1) / (float)(lookback - 1)));
    }

    const int warmup = first_valid + lookback;
    if (warmup >= rows) {
        return;
    }

    const int start = warmup;
    const int win_start = start + 1 - lookback;
    const float inv_lb = 1.0f / (float)lookback;
    const float inv_denom = (lookback > 1) ? (1.0f / (float)(lookback - 1)) : 0.0f;

    float sum_rs_local = 0.0f;
    float sum_o_local = 0.0f;
    float sumsq_o_local = 0.0f;
    float sum_c_local = 0.0f;
    float sumsq_c_local = 0.0f;
    float invalid_local = 0.0f;

    for (int offset = threadIdx.x; offset < lookback; offset += blockDim.x) {
        const int j = win_start + offset;
        if (j <= 0) {
            invalid_local += 1.0f;
            continue;
        }
        const int idx = j * cols + s;
        const int prev = (j - 1) * cols + s;
        const float o = open_tm[idx];
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        const float pc = close_tm[prev];
        if (!(valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f)) {
            invalid_local += 1.0f;
            continue;
        }

        const float a = logf(h / c);
        const float b = logf(h / o);
        const float d = logf(l / c);
        const float e = logf(l / o);
        const float rs = __fmaf_rn(d, e, a * b);
        const float oret = logf(o / pc);
        const float cret = logf(c / o);
        sum_rs_local += rs;
        sum_o_local += oret;
        sumsq_o_local += oret * oret;
        sum_c_local += cret;
        sumsq_c_local += cret * cret;
    }

    const float sum_rs = block_reduce_sum(sum_rs_local);
    const float sum_o = block_reduce_sum(sum_o_local);
    const float sumsq_o = block_reduce_sum(sumsq_o_local);
    const float sum_c = block_reduce_sum(sum_c_local);
    const float sumsq_c = block_reduce_sum(sumsq_c_local);
    const int invalid_count = (int)block_reduce_sum(invalid_local);

    if (threadIdx.x == 0) {
        float rolling_rs = sum_rs;
        float rolling_o = sum_o;
        float rolling_oo = sumsq_o;
        float rolling_c = sum_c;
        float rolling_cc = sumsq_c;
        int rolling_invalid = invalid_count;

        for (int t = start; t < rows; ++t) {
            const int out_idx = t * cols + s;
            if (rolling_invalid == 0) {
                float rs_var = rolling_rs * inv_lb;
                if (rs_var < 0.0f) {
                    rs_var = 0.0f;
                }

                float o_var = 0.0f;
                float c_var = 0.0f;
                if (lookback > 1) {
                    o_var = (rolling_oo - rolling_o * rolling_o * inv_lb) * inv_denom;
                    c_var = (rolling_cc - rolling_c * rolling_c * inv_lb) * inv_denom;
                    if (o_var < 0.0f) {
                        o_var = 0.0f;
                    }
                    if (c_var < 0.0f) {
                        c_var = 0.0f;
                    }
                }

                float yz_var = o_var + __fmaf_rn(1.0f - k, rs_var, k * c_var);
                if (yz_var < 0.0f) {
                    yz_var = 0.0f;
                }
                out_rs_tm[out_idx] = sqrtf(rs_var);
                out_yz_tm[out_idx] = sqrtf(yz_var);
            }

            if (t + 1 < rows) {
                const int add_idx = t + 1;
                const int sub_idx = add_idx - lookback;

                if (add_idx > 0) {
                    const int idx = add_idx * cols + s;
                    const int prev = (add_idx - 1) * cols + s;
                    const float o = open_tm[idx];
                    const float h = high_tm[idx];
                    const float l = low_tm[idx];
                    const float c = close_tm[idx];
                    const float pc = close_tm[prev];
                    if (valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f) {
                        const float a = logf(h / c);
                        const float b = logf(h / o);
                        const float d = logf(l / c);
                        const float e = logf(l / o);
                        const float rs = __fmaf_rn(d, e, a * b);
                        const float oret = logf(o / pc);
                        const float cret = logf(c / o);
                        rolling_rs += rs;
                        rolling_o += oret;
                        rolling_oo += oret * oret;
                        rolling_c += cret;
                        rolling_cc += cret * cret;
                    } else {
                        ++rolling_invalid;
                    }
                }

                if (sub_idx > 0) {
                    const int idx = sub_idx * cols + s;
                    const int prev = (sub_idx - 1) * cols + s;
                    const float o = open_tm[idx];
                    const float h = high_tm[idx];
                    const float l = low_tm[idx];
                    const float c = close_tm[idx];
                    const float pc = close_tm[prev];
                    if (valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f) {
                        const float a = logf(h / c);
                        const float b = logf(h / o);
                        const float d = logf(l / c);
                        const float e = logf(l / o);
                        const float rs = __fmaf_rn(d, e, a * b);
                        const float oret = logf(o / pc);
                        const float cret = logf(c / o);
                        rolling_rs -= rs;
                        rolling_o -= oret;
                        rolling_oo -= oret * oret;
                        rolling_c -= cret;
                        rolling_cc -= cret * cret;
                    } else {
                        --rolling_invalid;
                    }
                }
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 6
 *
 * ORACLE: `yang_zhang_precompute_ln_diff_scalar`
 * (src/indicators/yang_zhang_volatility.rs:521) followed by
 * `yang_zhang_row_precomputed_into` (:1643) -- the pair
 * `yang_zhang_compute_into` (:1103) calls for `Kernel::Scalar`.
 *
 * PERIOD-INVARIANT. `compute_yang_zhang_volatility_batch` reads `lookback`
 * (14), `k_override` (false) and `k` (0.34) -- NEVER `period`
 * (cpu_batch.rs:8311-8313). A sweep of five periods gets five identical CPU
 * columns, so this kernel writes five identical rows and
 * `is_period_invariant` says so. `periods` is read only to be discarded.
 *
 * k. Because `k_override` defaults FALSE, the effective k is
 * `k_default(lookback)` (:403) = 0.34 / (1.34 + (lb+1)/(lb-1)), NOT the 0.34
 * the parameter carries. Using the parameter value would be a different
 * indicator that still looked plausible.
 *
 * MULTI-OUTPUT: emits YZ, which is what `output_id == "value"` resolves to
 * (cpu_batch.rs:8331). Never `rs` silently.
 *
 * FIRST-VALID: `first_valid_ohlc` (:411) -- open, high, low AND close all
 * `!is_nan` at the same index. That is `F64FirstValidRule::Ohlc4AllNonNan`,
 * the rule `bop` already declares. NOT `Ohlc4AllFinite`: an infinite bar is
 * ACCEPTED here and rejected there.
 *
 * WARMUP: `first + lookback` (:1145) -- one MORE than the window, because the
 * seed window is `[warmup + 1 - lookback, warmup]` (:1661).
 *
 * ACCUMULATION ORDER IS LOAD-BEARING and is reproduced literally: the sums are
 * ROLLED as `sum += new - old` and `sumsq += new*new - old*old` (:1702-1712),
 * not recomputed per bar. A fresh dot product would be a different double at
 * every bar after the first.
 *
 * THE ln PRECOMPUTE IS NOT HOISTED OUT OF THE THREAD. The CPU builds three
 * whole-series arrays first; the kernel recomputes `ln` per touched bar
 * instead, because each bar's three values are a pure function of that bar
 * (plus `close[i-1]` for oret) with no carried state -- identical doubles, no
 * 3 x n scratch matrix per combo.
 *
 * SEQUENTIAL, one thread per combo column.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define YZ_NEO_LOOKBACK 14   /* cpu_batch.rs:8311 */

/* yang_zhang_volatility.rs:548 -- rs_val[i], written as the four ln
   DIFFERENCES the scalar precompute forms, not as the ratio form of
   `rs_component` (:502). The two are algebraically equal and NUMERICALLY
   different; the precompute is what the batch path runs. */
__device__ __forceinline__ double yz_neo_rs_f64(double o, double h, double l, double c)
{
    const double ln_open  = log(o);
    const double ln_high  = log(h);
    const double ln_low   = log(l);
    const double ln_close = log(c);
    const double h_c = ln_high - ln_close;
    const double h_o = ln_high - ln_open;
    const double l_c = ln_low  - ln_close;
    const double l_o = ln_low  - ln_open;
    return h_c * h_o + l_c * l_o;
}

/* sample_var, :507. The `n <= 1` guard and the `v < 0.0` clamp are both the
   CPU's; the clamp is an `if`, not an fmax, and NaN must survive it exactly as
   it survives the CPU's `if v < 0.0` (false for NaN). */
__device__ __forceinline__ double yz_neo_sample_var_f64(double sum, double sumsq, int n)
{
    if (n <= 1) return 0.0;
    const double nf    = (double)n;
    const double denom = (double)(n - 1);
    double v = (sumsq - (sum * sum) / nf) / denom;
    if (v < 0.0) v = 0.0;
    return v;
}

extern "C" __global__
void yang_zhang_volatility_neo_batch_f64(const double* __restrict__ open,
                                         const double* __restrict__ high,
                                         const double* __restrict__ low,
                                         const double* __restrict__ close,
                                         int series_len,
                                         const int* __restrict__ periods,
                                         int n_combos,
                                         int first_valid,
                                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;                       /* PERIOD-INVARIANT -- see header. */

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;

    if (first_valid < 0 || first_valid >= len) return;

    const int lb = YZ_NEO_LOOKBACK;
    const int warmup = first_valid + lb;               /* :1145 */
    if (warmup >= len) return;                         /* :1655 early return  */

    /* k_default(lookback), :403-409. NOT the 0.34 parameter -- k_override is
       false by default, so the parameter is never consulted. */
    const double k = (lb <= 1)
        ? 0.0
        : 0.34 / (1.34 + ((double)(lb + 1)) / ((double)(lb - 1)));

    const double lb_f = (double)lb;
    const int    start     = warmup;
    const int    win_start = start + 1 - lb;           /* :1661 */
    if (win_start < 0) return;

    double sum_rs = 0.0, sum_o = 0.0, sumsq_o = 0.0, sum_c = 0.0, sumsq_c = 0.0;

    /* Seed, :1669-1680. */
    for (int j = win_start; j <= start; ++j) {
        sum_rs += yz_neo_rs_f64(open[j], high[j], low[j], close[j]);

        /* oret[0] is pinned to 0.0 by the precompute (:536) and only
           overwritten for i > 0 (:551), so bar 0 contributes exactly zero. */
        const double ov = (j > 0) ? (log(open[j]) - log(close[j - 1])) : 0.0;
        sum_o   += ov;
        sumsq_o += ov * ov;

        const double cv = log(close[j]) - log(open[j]);
        sum_c   += cv;
        sumsq_c += cv * cv;
    }

    /* Roll, :1682-1713. */
    for (int t = start; t < len; ++t) {
        double rs_var = sum_rs / lb_f;
        if (rs_var < 0.0) rs_var = 0.0;

        const double o_var = yz_neo_sample_var_f64(sum_o, sumsq_o, lb);
        const double c_var = yz_neo_sample_var_f64(sum_c, sumsq_c, lb);

        double yz_var = o_var + k * c_var + (1.0 - k) * rs_var;
        if (yz_var < 0.0) yz_var = 0.0;
        o[t] = sqrt(yz_var);

        if (t + 1 < len) {
            const int add_idx = t + 1;
            const int sub_idx = add_idx - lb;

            sum_rs += yz_neo_rs_f64(open[add_idx], high[add_idx], low[add_idx], close[add_idx])
                    - yz_neo_rs_f64(open[sub_idx], high[sub_idx], low[sub_idx], close[sub_idx]);

            const double ao = (add_idx > 0) ? (log(open[add_idx]) - log(close[add_idx - 1])) : 0.0;
            const double so = (sub_idx > 0) ? (log(open[sub_idx]) - log(close[sub_idx - 1])) : 0.0;
            sum_o   += ao - so;
            sumsq_o += ao * ao - so * so;

            const double ac = log(close[add_idx]) - log(open[add_idx]);
            const double sc = log(close[sub_idx]) - log(open[sub_idx]);
            sum_c   += ac - sc;
            sumsq_c += ac * ac - sc * sc;
        }
    }
}
