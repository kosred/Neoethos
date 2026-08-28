#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LDG


#  if __CUDA_ARCH__ >= 700
#    define LDG(p) (*(p))
#  elif __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif

__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ void kbn_acc(float &sum, float &c, float x) {

    float t = sum + x;
    if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
    else                        c += (x   - t) + sum;
    sum = t;
}

__device__ __forceinline__ void warp_reduce_kbn(float &sum, float &c, unsigned mask) {

    for (int offset = 16; offset > 0; offset >>= 1) {
        float s2 = __shfl_down_sync(mask, sum, offset);
        float c2 = __shfl_down_sync(mask, c,   offset);

        kbn_acc(sum, c, s2);
        kbn_acc(sum, c, c2);
    }
}

template<int MaxWarps>
__device__ __forceinline__ float block_reduce_kbn(float sum, float c, float *smem_sum, float *smem_comp) {

    const unsigned mask = __activemask();
    warp_reduce_kbn(sum, c, mask);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) {
        smem_sum[warp]  = sum;
        smem_comp[warp] = c;
    }
    __syncthreads();

    float out_sum = 0.0f, out_comp = 0.0f;
    if (warp == 0) {

        const int num_warps = (blockDim.x + 31) >> 5;
        float v_sum = (lane < num_warps) ? smem_sum[lane]  : 0.0f;
        float v_comp= (lane < num_warps) ? smem_comp[lane] : 0.0f;
        warp_reduce_kbn(v_sum, v_comp, mask);
        if (lane == 0) { out_sum = v_sum; out_comp = v_comp; }
    }
    __syncthreads();

    if (threadIdx.x == 0) { smem_sum[0] = out_sum + out_comp; }
    __syncthreads();
    return smem_sum[0];
}


extern "C" __global__
void nadaraya_watson_envelope_batch_f32(const float* __restrict__ data,
                                        const float* __restrict__ weights_flat,
                                        const int*   __restrict__ lookbacks,
                                        const float* __restrict__ multipliers,
                                        int series_len,
                                        int n_combos,
                                        int first_valid,
                                        int max_lookback,
                                        float* __restrict__ out_upper,
                                        float* __restrict__ out_lower)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;


    const int MAE_LEN = 499;
    const int TILE_T  = 64;
    const int L       = lookbacks[combo];
    const float mult  = multipliers[combo];

    if (L <= 0) return;

    const int warm_out   = first_valid + L - 1;
    const int warm_total = warm_out + MAE_LEN - 1;

    const int base  = combo * series_len;
    const int wbase = combo * max_lookback;


    const int prefix = (warm_total < series_len) ? warm_total : series_len;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < prefix; i += blockDim.x) {
        out_upper[base + i] = qnan_f32();
        out_lower[base + i] = qnan_f32();
    }
    __syncthreads();
    if (warm_total >= series_len) return;


    extern __shared__ float s[];
    float *s_w    = s;
    float *s_x    = s_w + max_lookback;
    float *s_mask = s_x + (max_lookback + TILE_T - 1);

    __shared__ float smem_sum[32], smem_comp[32];
    __shared__ float s_ring[MAE_LEN];
    __shared__ int   s_nan_win_count;


    for (int k = threadIdx.x; k < L; k += blockDim.x) {
        s_w[k] = LDG(&weights_flat[wbase + k]);
    }
    __syncthreads();


    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < MAE_LEN; ++i) s_ring[i] = qnan_f32();
    }
    __syncthreads();


    int   mae_head   = 0;
    int   mae_filled = 0;
    float mae_sum    = 0.0f;
    int   mae_nan_ct = 0;


    for (int t0 = warm_out; t0 < series_len; t0 += TILE_T)
    {
        const int tile_T = min(TILE_T, series_len - t0);
        const int tile_x_start = t0 - (L - 1);
        const int tile_x_end   = t0 + tile_T - 1;
        const int tile_span    = tile_x_end - tile_x_start + 1;


        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            float xi = LDG(&data[tile_x_start + i]);
            s_x[i]   = xi;

            s_mask[i]= isnan(xi) ? 1.0f : 0.0f;
        }
        __syncthreads();


        if (threadIdx.x == 0) {
            int nc = 0;

            for (int i = 0; i < L; ++i) nc += (s_mask[i] > 0.0f);
            s_nan_win_count = nc;
        }
        __syncthreads();


        for (int u = 0; u < tile_T; ++u) {
            const int t        = t0 + u;
            const int x_off    = (L - 1) + u;
            const bool window_ok = (s_nan_win_count == 0);

            float y = qnan_f32();

            if (window_ok) {

                float sum = 0.0f, comp = 0.0f;

                for (int k = threadIdx.x; k < L; k += blockDim.x) {

                    float prod = s_w[k] * s_x[x_off - k];
                    kbn_acc(sum, comp, prod);
                }

                y = block_reduce_kbn<32>(sum, comp, smem_sum, smem_comp);
            }


            if (threadIdx.x == 0) {
                const float x_t = s_x[x_off];
                const bool y_isnan = isnan(y);
                const bool x_isnan = isnan(x_t);

                float resid = (!x_isnan && !y_isnan) ? fabsf(x_t - y) : qnan_f32();


                if (mae_filled == MAE_LEN) {
                    float old = s_ring[mae_head];
                    if (isnan(old)) { if (mae_nan_ct > 0) mae_nan_ct -= 1; }
                    else            { mae_sum -= old; }
                } else {
                    mae_filled += 1;
                }


                s_ring[mae_head] = resid;
                if (isnan(resid)) mae_nan_ct += 1; else mae_sum += resid;
                mae_head += 1; if (mae_head == MAE_LEN) mae_head = 0;


                if (t >= warm_total) {
                    float upper = qnan_f32();
                    float lower = qnan_f32();
                    if (mae_nan_ct == 0 && !y_isnan) {
                        float mae = (mae_sum / (float)MAE_LEN) * mult;
                        upper = y + mae;
                        lower = y - mae;
                    }
                    out_upper[base + t] = upper;
                    out_lower[base + t] = lower;
                }
            }
            __syncthreads();


            if (u + 1 < tile_T) {
                if (threadIdx.x == 0) {


                    int addv = (s_mask[x_off + 1] > 0.0f) ? 1 : 0;
                    int dropv= (s_mask[x_off - (L - 1)] > 0.0f) ? 1 : 0;
                    s_nan_win_count += addv - dropv;
                }
                __syncthreads();
            }
        }
    }
}


extern "C" __global__
void nadaraya_watson_envelope_many_series_one_param_f32(const float* __restrict__ data_tm,
                                                        const float* __restrict__ weights,
                                                        int lookback,
                                                        float multiplier,
                                                        int num_series,
                                                        int series_len,
                                                        const int* __restrict__ first_valids,
                                                        float* __restrict__ out_upper_tm,
                                                        float* __restrict__ out_lower_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int L = lookback;
    const int MAE_LEN = 499;

    const int warm_out   = first_valids[series] + L - 1;
    const int warm_total = warm_out + MAE_LEN - 1;


    for (int t = threadIdx.x; t < min(warm_total, series_len); t += blockDim.x) {
        const int idx = t * num_series + series;
        out_upper_tm[idx] = qnan_f32();
        out_lower_tm[idx] = qnan_f32();
    }
    __syncthreads();
    if (warm_total >= series_len) return;


    __shared__ float ring[MAE_LEN];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < MAE_LEN; ++i) ring[i] = qnan_f32();
    }
    __syncthreads();

    int head = 0, filled = 0, nan_count = 0;
    float sum = 0.0f;


    if (threadIdx.x == 0) {
        for (int t = warm_out; t < series_len; ++t) {
            bool any_nan = false;

            float acc = 0.0f, c = 0.0f;
            #pragma unroll 1
            for (int k = 0; k < L; ++k) {
                int idx = (t - k) * num_series + series;
                float x = LDG(&data_tm[idx]);
                if (isnan(x)) { any_nan = true; break; }
                float wk = LDG(&weights[k]);
                float prod = x * wk;
                float tmp = acc + prod;
                if (fabsf(acc) >= fabsf(prod)) c += (acc - tmp) + prod;
                else                            c += (prod - tmp) + acc;
                acc = tmp;
            }
            float y = any_nan ? qnan_f32() : (acc + c);

            const int idx_t = t * num_series + series;
            float x_t = LDG(&data_tm[idx_t]);
            float resid = (!isnan(x_t) && !isnan(y)) ? fabsf(x_t - y) : qnan_f32();

            if (filled == MAE_LEN) {
                float old = ring[head];
                if (isnan(old)) { if (nan_count > 0) nan_count -= 1; } else { sum -= old; }
            } else {
                filled += 1;
            }
            ring[head] = resid;
            if (isnan(resid)) nan_count += 1; else sum += resid;
            head += 1; if (head == MAE_LEN) head = 0;

            if (t >= warm_total) {
                float upper = qnan_f32();
                float lower = qnan_f32();
                if (nan_count == 0 && !isnan(y)) {
                    float mae = (sum / (float)MAE_LEN) * multiplier;
                    upper = y + mae;
                    lower = y - mae;
                }
                out_upper_tm[idx_t] = upper;
                out_lower_tm[idx_t] = lower;
            }
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer 6
//
// CPU reference: `nwe_compute_scalar_prepared`
// (src/indicators/nadaraya_watson_envelope.rs:468) and the two walks it
// selects between, `nwe_compute_scalar_no_nan` (:510) and
// `nwe_compute_scalar_nan_checked` (:566). Reached from
// `nadaraya_watson_envelope_with_kernel` (:659) -> `..._envelope` (:648) on
// every non-AVX build, which is what `compute_nadaraya_watson_envelope_batch`
// (dispatch/cpu_batch.rs:15607) takes.
//
// OUTPUT: `upper`. `compute_..._batch:15636` maps output_id "value" onto
// `out.upper`; `lower` is `y - mae` in the same walk and is one launch away
// once the lane grows an output selector.
//
// PERIOD-INVARIANT. The CPU batch reads `bandwidth` (8.0), `multiplier`
// (3.0) and `lookback` (500) and NEVER `period` (:15619-15621). A caller
// sweeping [7,21,50,100,200] gets five identical CPU columns, so this kernel
// emits five identical rows. The three named parameters are pinned at the
// CPU defaults.
//
// SEQUENTIAL, ONE THREAD PER COLUMN -- NOT bar-parallel, and the brief's
// sketch ("no carried state, so this one is genuinely bar-parallel") is
// WRONG about this indicator. The kernel regression `y` alone would be
// bar-parallel, but the band is `y +/- mae` where `mae` is a sliding sum of
// the last 499 ABSOLUTE RESIDUALS (`rbuf`/`rsum`, :523-525). That sum is
// carried across bars and is accumulation-order dependent -- `rsum -= old;
// rsum += resid` at :545-548 is not the same number as a fresh sum of the
// window. Reproducing it is the only faithful shape.
//
// THE CPU CHOOSES BETWEEN TWO WALKS AND SO DOES THIS KERNEL. :497-506 scans
// `data[first..]` for any NaN, where `first = warm_out + 1 - lookback`, and
// takes the cheap walk only if there is none. The two are NOT equivalent on
// clean data plus one late hole: the no-NaN walk lets a NaN residual poison
// `rsum` permanently, while the checked walk counts NaNs in the ring
// (`r_nan_cnt`) and emits NaN until the hole has left the 499-bar window.
// Picking one walk unconditionally would have been wrong in both directions,
// so the scan is reproduced.
//
// WARMUP IS TWO-STAGE AND THE SECOND STAGE IS 499 BARS LONG.
// `warm_out = first + lookback - 1` (:402) is where the regression starts
// being computed; `warm_total = warm_out + MAE_LEN - 1` (:403) is where the
// BAND starts being emitted, because the residual ring must be full first.
// The walk therefore begins at `warm_out` and writes nothing until
// `warm_total` -- the residuals between the two indices exist only to fill
// the ring. A kernel that started writing at `warm_out` would emit 498 bars
// the CPU leaves NaN.
//
// NaN: there is no `f64::max` in this reference at all, so no comparison
// chain here needs converting to fmax. The NaN handling that does exist is
// explicit `is_nan` bookkeeping and is reproduced as `isnan`.
//
// EPSILON: none. This indicator has no tolerance constant on either side --
// `den` is a sum of `exp(...)` terms which is strictly positive for any
// valid bandwidth, and `nwe_prepare:383` has already refused a bandwidth
// that is zero, negative or NaN.
//
// f32 -> f64 audit of this section: no f32 literal, no f32-suffixed math
// function (the f32 lane above uses expf, fabsf and __fmaf_rn), no fast-math
// intrinsic. The quiet NaN is `__longlong_as_double(0x7ff8000000000000ULL)`,
// which is the same bit pattern the CPU writes with
// `f64::from_bits(0x7ff8_0000_0000_0000)` (:481).
// ===========================================================================

// `MAE_LEN` -- :401, :521, :581. A fixed 499, not a parameter.
#define NWE_F64_MAE_LEN 499

// The CPU defaults, cpu_batch.rs:15619-15621. `lookback` also sizes the
// per-thread weight array, so it is the bound this kernel declares: an
// oversized lookback is REFUSED BY NAME in the wrapper rather than truncated
// or moved to the host.
#define NWE_F64_LOOKBACK 500
#define NWE_F64_BANDWIDTH 8.0
#define NWE_F64_MULTIPLIER 3.0

static __device__ __forceinline__ double nwe_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void nadaraya_watson_envelope_batch_f64(const double* __restrict__ prices,
                                        int n,
                                        const int* __restrict__ periods,
                                        int n_combos,
                                        int first_valid,
                                        double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    // PERIOD-INVARIANT: read so the parameter is not silently dropped from the
    // signature, but the CPU batch never consults it.
    (void)periods;

    const double nan_d = nwe_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);

    const int lookback = NWE_F64_LOOKBACK;

    if (n <= 0 || first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }

    const int warm_out   = first_valid + lookback - 1;                  // :402
    const int warm_total = warm_out + NWE_F64_MAE_LEN - 1;              // :403

    // `nwe_prepare:405` errors -- and `collect_f64` turns the error into an
    // all-NaN column -- when the series ends at or before `warm_out`.
    if (n <= warm_out) {
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }

    // `alloc_with_nan_prefix(len, warm_total)` -- :651.
    for (int i = 0; i < n && i < warm_total; ++i) row[i] = nan_d;
    if (warm_total >= n) return;                                        // :493-495

    // ------------------------------------------------------------------
    // Weights. :412-418. Built in the CPU's ascending-k order so `den`
    // accumulates identically.
    // ------------------------------------------------------------------
    double w[NWE_F64_LOOKBACK];
    double den = 0.0;
    const double bw = NWE_F64_BANDWIDTH;
    for (int k = 0; k < lookback; ++k) {
        const double kf = static_cast<double>(k);
        const double wk = exp(-(kf) * (kf) / (2.0 * bw * bw));          // :415
        w[k] = wk;
        den += wk;                                                      // :417
    }

    const double mult = NWE_F64_MULTIPLIER;
    const double scale = mult / static_cast<double>(NWE_F64_MAE_LEN);   // :526, :585

    // ------------------------------------------------------------------
    // Which walk. :497-506.
    // ------------------------------------------------------------------
    const int scan_from = warm_out + 1 - lookback;
    bool any_nan_in_tail = false;
    for (int i = scan_from; i < n; ++i) {
        if (isnan(prices[i])) { any_nan_in_tail = true; break; }
    }

    double rbuf[NWE_F64_MAE_LEN];
    int rhead = 0;
    double rsum = 0.0;

    if (!any_nan_in_tail) {
        // -------------------------------------------------------------
        // `nwe_compute_scalar_no_nan` -- :510-563.
        // -------------------------------------------------------------
        for (int k = 0; k < NWE_F64_MAE_LEN; ++k) rbuf[k] = 0.0;        // :523

        for (int t = warm_out; t < n; ++t) {                            // :531
            double num = 0.0;
            for (int k = 0; k < lookback; ++k) {                        // :535-538
                num += prices[t - k] * w[k];
            }

            const double y = num / den;                                 // :541
            const double resid = fabs(prices[t] - y);                   // :542

            const double old = rbuf[rhead];                             // :544
            rsum -= old;
            rbuf[rhead] = resid;
            rsum += resid;                                              // :548

            ++rhead;
            if (rhead == NWE_F64_MAE_LEN) rhead = 0;                    // :550-553

            if (t >= warm_total) {                                      // :555
                const double mae = rsum * scale;                        // :556
                row[t] = y + mae;                                       // :557 upper
            }
        }
        return;
    }

    // -----------------------------------------------------------------
    // `nwe_compute_scalar_nan_checked` -- :566-644.
    // -----------------------------------------------------------------
    for (int k = 0; k < NWE_F64_MAE_LEN; ++k) rbuf[k] = nan_d;          // :582
    int r_nan_cnt = NWE_F64_MAE_LEN;                                    // :584

    for (int t = warm_out; t < n; ++t) {                                // :590
        double num = 0.0;
        bool any_nan = false;
        for (int k = 0; k < lookback; ++k) {                            // :595-602
            const double x = prices[t - k];
            if (isnan(x)) { any_nan = true; break; }
            num += x * w[k];
        }

        const double y = any_nan ? nan_d : (num / den);                 // :605
        const double xt = prices[t];                                    // :606
        const double resid = (!isnan(xt) && !isnan(y))
                             ? fabs(xt - y) : nan_d;                    // :607-611

        const double old = rbuf[rhead];                                 // :613
        if (isnan(old)) {
            // `saturating_sub(1)` -- :615. Cannot underflow here because the
            // ring starts full of NaN and every NaN removed was counted, but
            // the clamp is reproduced rather than assumed away.
            r_nan_cnt = (r_nan_cnt > 0) ? (r_nan_cnt - 1) : 0;
        } else {
            rsum -= old;                                                // :617
        }

        rbuf[rhead] = resid;                                            // :621
        if (isnan(resid)) {
            ++r_nan_cnt;                                                // :623
        } else {
            rsum += resid;                                              // :625
        }

        ++rhead;
        if (rhead == NWE_F64_MAE_LEN) rhead = 0;                        // :628-631

        if (t >= warm_total) {                                          // :633
            if (!isnan(y) && r_nan_cnt == 0) {
                const double mae = rsum * scale;                        // :635
                row[t] = y + mae;                                       // :636 upper
            } else {
                row[t] = nan_d;                                         // :639
            }
        }
    }
}
