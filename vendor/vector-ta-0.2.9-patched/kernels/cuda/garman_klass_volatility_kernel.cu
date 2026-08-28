#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ bool valid_ohlc(float o, float h, float l, float c) {
    return isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
           o > 0.0f && h > 0.0f && l > 0.0f && c > 0.0f;
}

static __forceinline__ __device__ float gk_term(float o, float h, float l, float c) {
    const float hl = logf(h / l);
    const float co = logf(c / o);
    const float coeff = 2.0f * logf(2.0f) - 1.0f;
    return 0.5f * hl * hl - coeff * co * co;
}

extern "C" __global__ void garman_klass_precompute_terms_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int* __restrict__ valid_flags,
    float* __restrict__ terms
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < series_len;
         j += blockDim.x * gridDim.x) {
        const float o = open[j];
        const float h = high[j];
        const float l = low[j];
        const float c = close[j];
        const bool valid = valid_ohlc(o, h, l, c);
        valid_flags[j] = valid ? 1 : 0;
        terms[j] = valid ? gk_term(o, h, l, c) : 0.0f;
    }
}

extern "C" __global__ void garman_klass_prefix_terms_f32(
    const int* __restrict__ valid_flags,
    const float* __restrict__ terms,
    int series_len,
    int* __restrict__ prefix_valid,
    float* __restrict__ prefix_sum
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    prefix_valid[0] = 0;
    prefix_sum[0] = 0.0f;

    int valid_acc = 0;
    float sum_acc = 0.0f;
    for (int j = 0; j < series_len; ++j) {
        valid_acc += valid_flags[j];
        sum_acc += terms[j];
        prefix_valid[j + 1] = valid_acc;
        prefix_sum[j + 1] = sum_acc;
    }
}

extern "C" __global__ void garman_klass_volatility_batch_prefix_f32(
    const int* __restrict__ lookbacks,
    int series_len,
    int first_valid,
    int n_combos,
    const int* __restrict__ prefix_valid,
    const float* __restrict__ prefix_sum,
    float* __restrict__ out
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    __shared__ int lookback_s;
    __shared__ int warmup_s;
    __shared__ int combo_valid_s;
    __shared__ float inv_lb_s;

    if (threadIdx.x == 0) {
        const int lookback = lookbacks[combo];
        const int combo_valid = lookback > 0 && lookback <= series_len;
        lookback_s = lookback;
        warmup_s = first_valid + lookback - 1;
        combo_valid_s = combo_valid;
        inv_lb_s = combo_valid ? 1.0f / (float)lookback : 0.0f;
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);
    const int base = combo * series_len;
    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < series_len;
         t += (int)blockDim.x * (int)gridDim.x) {
        float out_v = nan_f;
        if (combo_valid_s != 0 && t >= warmup_s) {
            const int window_start = t + 1 - lookback_s;
            const int valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
            if (valid_count == lookback_s) {
                float variance = (prefix_sum[t + 1] - prefix_sum[window_start]) * inv_lb_s;
                if (variance < 0.0f) {
                    variance = 0.0f;
                }
                out_v = sqrtf(variance);
            }
        }
        out[base + t] = out_v;
    }
}

extern "C" __global__ void garman_klass_volatility_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int num_series,
    int series_len,
    int lookback,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) {
        return;
    }

    const float nan_f = __int_as_float(0x7fffffff);
    int first_valid = -1;
    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + s;
        if (valid_ohlc(open_tm[idx], high_tm[idx], low_tm[idx], close_tm[idx])) {
            first_valid = t;
            break;
        }
    }

    if (first_valid < 0) {
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + s] = nan_f;
        }
        return;
    }

    const int warmup = first_valid + lookback - 1;
    for (int t = 0; t < series_len; ++t) {
        float out_v = nan_f;
        if (t >= warmup) {
            bool valid = true;
            float sum = 0.0f;
            for (int j = t + 1 - lookback; j <= t; ++j) {
                const int idx = j * num_series + s;
                const float o = open_tm[idx];
                const float h = high_tm[idx];
                const float l = low_tm[idx];
                const float c = close_tm[idx];
                if (!valid_ohlc(o, h, l, c)) {
                    valid = false;
                    break;
                }
                sum += gk_term(o, h, l, c);
            }
            if (valid) {
                float variance = sum / (float)lookback;
                if (variance < 0.0f) {
                    variance = 0.0f;
                }
                out_v = sqrtf(variance);
            }
        }
        out_tm[t * num_series + s] = out_v;
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

static __forceinline__ __device__ bool valid_ohlc_f64(double o, double h, double l, double c) {
    return isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
           o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0;
}
static __forceinline__ __device__ double gk_term_f64(double o, double h, double l, double c) {
    const double hl = log(h / l);
    const double co = log(c / o);
    const double coeff = 2.0 * log(2.0) - 1.0;
    return 0.5 * hl * hl - coeff * co * co;
}
extern "C" __global__ void garman_klass_precompute_terms_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int series_len,
    int* __restrict__ valid_flags,
    double* __restrict__ terms
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < series_len;
         j += blockDim.x * gridDim.x) {
        const double o = open[j];
        const double h = high[j];
        const double l = low[j];
        const double c = close[j];
        const bool valid = valid_ohlc_f64(o, h, l, c);
        valid_flags[j] = valid ? 1 : 0;
        terms[j] = valid ? gk_term_f64(o, h, l, c) : 0.0;
    }
}
extern "C" __global__ void garman_klass_prefix_terms_f64(
    const int* __restrict__ valid_flags,
    const double* __restrict__ terms,
    int series_len,
    int* __restrict__ prefix_valid,
    double* __restrict__ prefix_sum
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    prefix_valid[0] = 0;
    prefix_sum[0] = 0.0;

    int valid_acc = 0;
    double sum_acc = 0.0;
    for (int j = 0; j < series_len; ++j) {
        valid_acc += valid_flags[j];
        sum_acc += terms[j];
        prefix_valid[j + 1] = valid_acc;
        prefix_sum[j + 1] = sum_acc;
    }
}
extern "C" __global__ void garman_klass_volatility_batch_prefix_f64(
    const int* __restrict__ lookbacks,
    int series_len,
    int first_valid,
    int n_combos,
    const int* __restrict__ prefix_valid,
    const double* __restrict__ prefix_sum,
    double* __restrict__ out
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    __shared__ int lookback_s;
    __shared__ int warmup_s;
    __shared__ int combo_valid_s;
    __shared__ double inv_lb_s;

    if (threadIdx.x == 0) {
        const int lookback = lookbacks[combo];
        const int combo_valid = lookback > 0 && lookback <= series_len;
        lookback_s = lookback;
        warmup_s = first_valid + lookback - 1;
        combo_valid_s = combo_valid;
        inv_lb_s = combo_valid ? 1.0 / (double)lookback : 0.0;
    }
    __syncthreads();

    const double nan_f = __longlong_as_double(0x7fffffffffffffffULL);
    const int base = combo * series_len;
    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < series_len;
         t += (int)blockDim.x * (int)gridDim.x) {
        double out_v = nan_f;
        if (combo_valid_s != 0 && t >= warmup_s) {
            const int window_start = t + 1 - lookback_s;
            const int valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
            if (valid_count == lookback_s) {
                double variance = (prefix_sum[t + 1] - prefix_sum[window_start]) * inv_lb_s;
                if (variance < 0.0) {
                    variance = 0.0;
                }
                out_v = sqrt(variance);
            }
        }
        out[base + t] = out_v;
    }
}
extern "C" __global__ void garman_klass_volatility_many_series_one_param_f64(
    const double* __restrict__ open_tm,
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    const double* __restrict__ close_tm,
    int num_series,
    int series_len,
    int lookback,
    double* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) {
        return;
    }

    const double nan_f = __longlong_as_double(0x7fffffffffffffffULL);
    int first_valid = -1;
    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + s;
        if (valid_ohlc_f64(open_tm[idx], high_tm[idx], low_tm[idx], close_tm[idx])) {
            first_valid = t;
            break;
        }
    }

    if (first_valid < 0) {
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + s] = nan_f;
        }
        return;
    }

    const int warmup = first_valid + lookback - 1;
    for (int t = 0; t < series_len; ++t) {
        double out_v = nan_f;
        if (t >= warmup) {
            bool valid = true;
            double sum = 0.0;
            for (int j = t + 1 - lookback; j <= t; ++j) {
                const int idx = j * num_series + s;
                const double o = open_tm[idx];
                const double h = high_tm[idx];
                const double l = low_tm[idx];
                const double c = close_tm[idx];
                if (!valid_ohlc_f64(o, h, l, c)) {
                    valid = false;
                    break;
                }
                sum += gk_term_f64(o, h, l, c);
            }
            if (valid) {
                double variance = sum / (double)lookback;
                if (variance < 0.0) {
                    variance = 0.0;
                }
                out_v = sqrt(variance);
            }
        }
        out_tm[t * num_series + s] = out_v;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — garman_klass_volatility
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/garman_klass_volatility.rs:409
 *   `gk_row_from_prefix`, fed by `build_prefix_terms` (:370) and `gk_term`
 *   (:315). That PREFIX form — not the ring in `GarmanKlassVolatilityStream`
 *   (:251) — is what the batch dispatcher runs, and the two are not the same
 *   arithmetic: the prefix subtracts two running sums accumulated from index 0,
 *   the ring adds and removes terms. Reproduced as the prefix.
 *
 * Column: output_id "value" (cpu_batch.rs:8373).
 *
 * WINDOW-ANCHORED: the NeoEthos scalar ABI carries the exact `lookback` that
 * the CPU authority request reads.  Every combo therefore consumes
 * `periods[combo]`; the registry default 14 is only the base request, not a
 * compiled device constant.
 *
 * Input: open / high / low / close — F64InputKind::Ohlc4. `open` is a genuine
 *   input (gk_term takes ln(close/open)), so an Hlc triple would compute a
 *   different number while passing every length check.
 *
 * A bar counts only when all four prices are finite AND strictly positive
 *   (:303) — the term takes two logarithms and a non-positive price has none.
 *   A window with one such bar emits NaN, it does not average the survivors.
 *
 * GK_COEFF = 2*ln(2) - 1 (:31), spelled out to full double precision rather
 *   than as an f32-era literal.  The output row temporarily stores the global
 *   prefix sum.  A descending conversion reads prefix[t+1] and prefix[start]
 *   before either lower cell is overwritten, so arbitrary lookbacks need no
 *   fixed-size per-thread ring and retain the CPU accumulation/subtraction
 *   order exactly.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

__device__ __forceinline__ bool gk_neo_valid_bar(double op,
                                                  double hi,
                                                  double lo,
                                                  double cl) {
    return isfinite(op) && isfinite(hi) && isfinite(lo) && isfinite(cl)
        && op > 0.0 && hi > 0.0 && lo > 0.0 && cl > 0.0;
}

extern "C" __global__
void garman_klass_volatility_neo_batch_f64(const double* __restrict__ open,
                                           const double* __restrict__ high,
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

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int LB = periods[combo];
    if (LB <= 0 || LB > n) return;

    /* `first` is DERIVED here, not taken from the caller. `validity_summary`
     * (:346) scans for the first bar at which all FOUR prices are finite AND
     * strictly positive, and no rule in `F64FirstValidRule` expresses that:
     * `AllInputsNonNan` would accept a leading zero or infinite price that the
     * CPU skips, and would shift the whole series by however many such bars
     * lead the frame. Rather than invent a rule variant that every consumer
     * would then have to grow a field for, the row derives its own start and
     * the table declares `F64FirstValidRule::Ignored`. */
    (void)first_valid;
    int first = n;
    for (int i = 0; i < n; ++i) {
        const double op = open[i], hi = high[i], lo = low[i], cl = close[i];
        if (gk_neo_valid_bar(op, hi, lo, cl)) { first = i; break; }
    }
    if (first >= n || first > n - LB) return;

    const int warmup = first + (LB - 1);
    const double inv_lb = 1.0 / (double)LB;
    const double GK_COEFF = 2.0 * 0.69314718055994530942 - 1.0;

    /* Pass 1: CPU build_prefix_terms' global prefix_sum, accumulated from 0.
     * o[i] is prefix_sum[i + 1].
     */
    double ps_running = 0.0;
    for (int i = 0; i < n; ++i) {
        const double op = open[i], hi = high[i], lo = low[i], cl = close[i];
        const bool ok = gk_neo_valid_bar(op, hi, lo, cl);
        if (ok) {
            const double hl = log(hi / lo);
            const double co = log(cl / op);
            ps_running += 0.5 * hl * hl - GK_COEFF * co * co;
        }
        o[i] = ps_running;
    }

    /* Pass 2 descends so both prefix cells are still scratch. Validity rolls
     * backwards in O(N): the next window removes its current right edge and
     * adds the preceding left edge. This is equivalent to the CPU's validity
     * prefix subtraction without requiring a second output-sized scratch.
     */
    int invalid_count = 0;
    for (int j = n - LB; j < n; ++j) {
        if (!gk_neo_valid_bar(open[j], high[j], low[j], close[j])) {
            invalid_count += 1;
        }
    }
    for (int i = n - 1; i >= warmup; --i) {
        const int ws = i + 1 - LB;
        if (invalid_count == 0) {
            const double prefix_sum_end = o[i];
            const double prefix_sum_ws = ws == 0 ? 0.0 : o[ws - 1];
            double variance = (prefix_sum_end - prefix_sum_ws) * inv_lb;
            if (variance < 0.0) variance = 0.0;
            o[i] = sqrt(variance);
        } else {
            o[i] = NEO_F64_NAN;
        }

        if (i > warmup) {
            if (!gk_neo_valid_bar(open[i], high[i], low[i], close[i])) {
                invalid_count -= 1;
            }
            const int entering = ws - 1;
            if (!gk_neo_valid_bar(
                    open[entering], high[entering], low[entering], close[entering])) {
                invalid_count += 1;
            }
        }
    }
    for (int i = 0; i < warmup; ++i) o[i] = NEO_F64_NAN;
}
