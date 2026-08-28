#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

static __forceinline__ __device__ float tr_at(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int t,
    int first_valid)
{
    const float hi = high[t];
    const float lo = low[t];
    if (t == first_valid) {
        return hi - lo;
    }
    const float pc = close[t - 1];
    float tr = hi - lo;
    float hc = fabsf(hi - pc);
    if (hc > tr) tr = hc;
    float lc = fabsf(lo - pc);
    if (lc > tr) tr = lc;
    return tr;
}

extern "C" __global__
void chande_batch_f32(const float* __restrict__ high,
                      const float* __restrict__ low,
                      const float* __restrict__ close,
                      const int* __restrict__ periods,
                      const float* __restrict__ mults,
                      const int* __restrict__ dirs,
                      const float* __restrict__ alphas,
                      const int* __restrict__ warm_indices,
                      int series_len,
                      int first_valid,
                      int n_combos,
                      float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float mult   = mults[combo];
    const int   dir    = dirs[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;


    double sum_tr = 0.0;
    for (int t = first_valid; t < first_valid + period; ++t) {
        sum_tr += (double)tr_at(high, low, close, t, first_valid);
    }
    double atr = sum_tr / (double)period;


    {
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        const int wstart = warm + 1 - period;
        for (int t = wstart; t <= warm; ++t) {
            const float v = (dir != 0) ? high[t] : low[t];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + warm] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }


    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr_at(high, low, close, t, first_valid);
        atr = fma((double)tri - atr, (double)alpha, atr);
        const int wstart = t + 1 - period;
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        for (int k = wstart; k <= t; ++k) {
            const float v = (dir != 0) ? high[k] : low[k];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + t] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }
}


extern "C" __global__
void chande_batch_from_tr_f32(const float* __restrict__ high,
                              const float* __restrict__ low,
                              const float* __restrict__ tr,
                              const int* __restrict__ periods,
                              const float* __restrict__ mults,
                              const int* __restrict__ dirs,
                              const float* __restrict__ alphas,
                              const int* __restrict__ warm_indices,
                              int series_len,
                              int first_valid,
                              int n_combos,
                              float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float mult   = mults[combo];
    const int   dir    = dirs[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) { out[base + idx] = NAN; }
    __syncthreads();
    if (threadIdx.x != 0) return;


    double sum_tr = 0.0;
    for (int t = first_valid; t < first_valid + period; ++t) { sum_tr += (double)tr[t]; }
    double atr = sum_tr / (double)period;

    {
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        const int wstart = warm + 1 - period;
        for (int t = wstart; t <= warm; ++t) {
            const float v = (dir != 0) ? high[t] : low[t];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + warm] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr[t];
        atr = fma((double)tri - atr, (double)alpha, atr);
        const int wstart = t + 1 - period;
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        for (int k = wstart; k <= t; ++k) {
            const float v = (dir != 0) ? high[k] : low[k];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + t] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }
}


extern "C" __global__
void chande_many_series_one_param_f32(const float* __restrict__ high_tm,
                                      const float* __restrict__ low_tm,
                                      const float* __restrict__ close_tm,
                                      const int* __restrict__ first_valids,
                                      int period,
                                      float mult,
                                      int dir,
                                      float alpha,
                                      int num_series,
                                      int series_len,
                                      float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];

        for (int t = lane; t < series_len; t += warpSize) {
            out_tm[t * stride + s] = NAN;
        }
        if (first_valid < 0 || first_valid >= series_len) continue;
        const int warm = first_valid + period - 1;
        if (warm >= series_len) continue;

        if (lane == 0) {

            double sum_tr = 0.0;
            for (int t = first_valid; t < first_valid + period; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                float tri;
                if (t == first_valid) {
                    tri = hi - lo;
                } else {
                    const float pc = close_tm[(t - 1) * stride + s];
                    float tr = hi - lo;
                    float hc = fabsf(hi - pc);
                    if (hc > tr) tr = hc;
                    float lc = fabsf(lo - pc);
                    if (lc > tr) tr = lc;
                    tri = tr;
                }
                sum_tr += (double)tri;
            }
            double atr = sum_tr / (double)period;

            {
                float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
                const int wstart = warm + 1 - period;
                for (int t = wstart; t <= warm; ++t) {
                    const float v = (dir != 0) ? high_tm[t * stride + s] : low_tm[t * stride + s];
                    if (dir != 0) { if (v > extrema) extrema = v; }
                    else          { if (v < extrema) extrema = v; }
                }
                out_tm[warm * stride + s] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
            }

            for (int t = warm + 1; t < series_len; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                float tr = hi - lo;
                float hc = fabsf(hi - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(lo - pc);
                if (lc > tr) tr = lc;
                atr = fma((double)tr - atr, (double)alpha, atr);
                const int wstart = t + 1 - period;
                float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
                for (int k = wstart; k <= t; ++k) {
                    const float v = (dir != 0) ? high_tm[k * stride + s] : low_tm[k * stride + s];
                    if (dir != 0) { if (v > extrema) extrema = v; }
                    else          { if (v < extrema) extrema = v; }
                }
                out_tm[t * stride + s] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
            }
        }
    }
}


static __forceinline__ __device__ float tr_from_hlpc(
    float hi, float lo, float pc, int t, int first_valid)
{
    if (t == first_valid) return hi - lo;
    float tr  = hi - lo;
    float hc  = fabsf(hi - pc);
    float lc  = fabsf(lo - pc);
    if (hc > tr) tr = hc;
    if (lc > tr) tr = lc;
    return tr;
}


static __forceinline__ __device__ void dq_push_monotone(
    int* __restrict__ idx_buf,
    float* __restrict__ val_buf,
    unsigned int mask,
    int& head, int& tail,
    int idx_new, float val_new, bool keep_max)
{

    while (head != tail) {
        unsigned int last = (static_cast<unsigned int>(tail - 1)) & mask;
        float back_val = val_buf[last];
        if (keep_max ? (back_val >= val_new) : (back_val <= val_new)) break;
        tail = static_cast<int>(last);
    }
    val_buf[tail] = val_new;
    idx_buf[tail] = idx_new;
    tail = static_cast<int>((static_cast<unsigned int>(tail) + 1u) & mask);
}


static __forceinline__ __device__ void dq_pop_expired(
    const int* __restrict__ idx_buf,
    unsigned int mask,
    int& head, int tail, int window_start)
{
    while (head != tail) {
        if (idx_buf[head] >= window_start) break;
        head = static_cast<int>((static_cast<unsigned int>(head) + 1u) & mask);
    }
}


static __forceinline__ __device__ float dq_front_value(
    const float* __restrict__ val_buf, unsigned int mask, int head)
{
    return val_buf[head & mask];
}


extern "C" __global__
void chande_one_series_many_params_f32(const float* __restrict__ high,
                                       const float* __restrict__ low,
                                       const float* __restrict__ close,
                                       const int*   __restrict__ periods,
                                       const float* __restrict__ mults,
                                       const int*   __restrict__ dirs,
                                       const float* __restrict__ alphas,
                                       int first_valid,
                                       int series_len,
                                       int n_combos,
                                       int queue_cap,
                                       int*   __restrict__ dq_idx,
                                       float* __restrict__ dq_val,
                                       float* __restrict__ out)
{
    const int lane            = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

    const unsigned full_mask = 0xFFFFFFFFu;
    const unsigned int qmask = static_cast<unsigned int>(queue_cap - 1);

    for (int w = warp_idx; w < (n_combos + 31) / 32; w += total_warps) {
        const int combo = (w << 5) + lane;
        if (combo >= n_combos) continue;

        const int   period = periods[combo];
        const float mult   = mults[combo];
        const int   dir    = dirs[combo];
        const float alpha  = alphas[combo];

        const int warm = first_valid + period - 1;
        const int base = combo * series_len;

        if (period <= 0 || warm >= series_len || first_valid >= series_len) {

            for (int t0 = 0; t0 < series_len; ++t0) {
                out[base + t0] = NAN;
            }
            continue;
        }


        for (int t0 = 0; t0 < warm; ++t0) {
            out[base + t0] = NAN;
        }


        int*   ring_idx = dq_idx + combo * queue_cap;
        float* ring_val = dq_val + combo * queue_cap;
        int head = 0, tail = 0;


        float seed_sum = 0.0f, c = 0.0f;
        float atr = 0.0f;
        bool  atr_seeded = false;


        float prev_close_b = 0.0f;
        for (int t = 0; t < series_len; ++t) {

            float hi = 0.0f, lo = 0.0f, pc = 0.0f;
            if (lane == 0) {
                hi = high[t];
                lo = low[t];
                if (t > 0) pc = close[t - 1];
            }
            hi = __shfl_sync(full_mask, hi, 0);
            lo = __shfl_sync(full_mask, lo, 0);
            if (t > 0) prev_close_b = __shfl_sync(full_mask, pc, 0);


            if (t >= first_valid) {
                const float v = (dir != 0) ? hi : lo;
                dq_push_monotone(ring_idx, ring_val, qmask, head, tail, t, v, (dir != 0));

                const int wstart = t + 1 - period;
                dq_pop_expired(ring_idx, qmask, head, tail, wstart);
            }


            if (t >= first_valid && !atr_seeded) {
                const float tri = tr_from_hlpc(hi, lo, prev_close_b, t, first_valid);

                const float y = tri - c;
                const float tmp = seed_sum + y;
                c = (tmp - seed_sum) - y;
                seed_sum = tmp;

                if (t == warm) {
                    atr = seed_sum / static_cast<float>(period);
                    atr_seeded = true;


                    const float ext = dq_front_value(ring_val, qmask, head);
                    out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
                }
            } else if (atr_seeded && t > warm) {

                const float tri = tr_from_hlpc(hi, lo, prev_close_b, t, first_valid);
                atr = __fmaf_rn(alpha, (tri - atr), atr);


                const float ext = dq_front_value(ring_val, qmask, head);
                out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
            }
        }
    }
}


extern "C" __global__
void chande_one_series_many_params_from_tr_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const float* __restrict__ tr,
                                               const int*   __restrict__ periods,
                                               const float* __restrict__ mults,
                                               const int*   __restrict__ dirs,
                                               const float* __restrict__ alphas,
                                               int first_valid,
                                               int series_len,
                                               int n_combos,
                                               int queue_cap,
                                               int*   __restrict__ dq_idx,
                                               float* __restrict__ dq_val,
                                               float* __restrict__ out)
{
    const int lane            = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;
    const unsigned full_mask = 0xFFFFFFFFu;
    const unsigned int qmask = static_cast<unsigned int>(queue_cap - 1);

    for (int w = warp_idx; w < (n_combos + 31) / 32; w += total_warps) {
        const int combo = (w << 5) + lane;
        if (combo >= n_combos) continue;

        const int   period = periods[combo];
        const float mult   = mults[combo];
        const int   dir    = dirs[combo];
        const float alpha  = alphas[combo];

        const int warm = first_valid + period - 1;
        const int base = combo * series_len;

        if (period <= 0 || warm >= series_len || first_valid >= series_len) {
            for (int t0 = 0; t0 < series_len; ++t0) out[base + t0] = NAN;
            continue;
        }
        for (int t0 = 0; t0 < warm; ++t0) out[base + t0] = NAN;

        int*   ring_idx = dq_idx + combo * queue_cap;
        float* ring_val = dq_val + combo * queue_cap;
        int head = 0, tail = 0;

        float seed_sum = 0.0f, c = 0.0f;
        float atr = 0.0f;
        bool  atr_seeded = false;

        for (int t = 0; t < series_len; ++t) {
            float hi = 0.0f, lo = 0.0f, tri = 0.0f;
            if (lane == 0) {
                hi  = high[t];
                lo  = low[t];
                tri = tr[t];
            }
            hi  = __shfl_sync(full_mask, hi, 0);
            lo  = __shfl_sync(full_mask, lo, 0);
            tri = __shfl_sync(full_mask, tri, 0);

            if (t >= first_valid) {
                const float v = (dir != 0) ? hi : lo;
                dq_push_monotone(ring_idx, ring_val, qmask, head, tail, t, v, (dir != 0));
                const int wstart = t + 1 - period;
                dq_pop_expired(ring_idx, qmask, head, tail, wstart);
            }

            if (t >= first_valid && !atr_seeded) {
                const float y = tri - c;
                const float tmp = seed_sum + y;
                c = (tmp - seed_sum) - y;
                seed_sum = tmp;

                if (t == warm) {
                    atr = seed_sum / static_cast<float>(period);
                    atr_seeded = true;
                    const float ext = dq_front_value(ring_val, qmask, head);
                    out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
                }
            } else if (atr_seeded && t > warm) {
                atr = __fmaf_rn(alpha, (tri - atr), atr);
                const float ext = dq_front_value(ring_val, qmask, head);
                out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
            }
        }
    }
}

// ===========================================================================
// S3 f64 LANE — chande (Chandelier Exit)
// ===========================================================================
// Reference: src/indicators/chande.rs
//   first_valid3 (:242)          — the first-valid rule
//   chande_with_kernel (:252)    — Err branches, warmup = first + period - 1
//   chande_scalar (:442)         — the general path
//   chande_scalar_default_long (:575) — the period==22 && mult==3 && long path
// Batch defaults: period 22, mult 3.0, direction "long" (get_direction :116).
//
// ONE KERNEL SERVES BOTH CPU PATHS. chande_scalar dispatches to
// chande_scalar_default_long for exactly the default triple. That fast path is
// the SAME arithmetic — same tr, same rma recurrence, same mul_add structure —
// with the VecDeque replaced by a fixed 32-slot ring. Nothing numeric differs,
// so no special case is written here and none is needed.
//
// FIRST-VALID IS THE SIMULTANEOUS RULE. first_valid3 scans high, low and close
// TOGETHER — !h.is_nan() && !l.is_nan() && !c.is_nan() at the same i — which is
// F64FirstValidRule::AllInputsNonNan, NOT the max-of-independent-firsts rule
// adx/natr use. Feeding this kernel adx's index would move the seed bar.
//
// THE MONOTONE DEQUE, REPRODUCED WITHOUT AN ARRAY
//
// The CPU keeps a deque of candidate indices for the rolling max of `high`
// (rolling min of `low` for direction "short"). A device thread cannot hold a
// period-length array. It does not have to: the deque contents are a PURE
// FUNCTION of the window, and its front can be recovered by one backward scan.
//
// An index j is popped from the back the first time a later k arrives with
// high[j] <= high[k] (:497). So j is still in the deque at bar i exactly when
//     for all k in (j, i]:  NOT (high[j] <= high[k])
// and the front is the SMALLEST such j inside the window. Front-pruning
// (:487-493) only ever removes indices below window_start, and window_start is
// non-decreasing, so no index at or above it was pruned earlier.
//
// NaN MATTERS HERE AND IS WHY THIS IS NOT JUST "the rolling max".
//   * If high[j] is NaN, `high[j] <= high[k]` is false for every k, so j is
//     NEVER popped and can reach the front — the CPU then emits a NaN-derived
//     value. A plain fmax scan would skip it and emit a number instead.
//   * If high[k] is NaN, it never pops anything either — which is exactly what
//     an fmax running maximum does, since fmax returns the non-NaN operand.
// The scan below encodes both: j survives iff isnan(high[j]) OR
// high[j] > (fmax over k in (j,i]). That is the deque, exactly, in O(period).
//
// ROUNDING.
//   tr        = fmax(fmax(hl, hc), lc)          — f64::max semantics, not a chain
//   rma       = fma(alpha, tr - rma, rma)       — ONE fma (CPU :514)
//   out long  = fma(-rma, mult, max_h)          — ONE fma (CPU :512)
//   out short = fma( rma, mult, min_l)          — ONE fma (CPU :562)
//
// The f32 file above carries FLT_MAX as its sentinel; there is no sentinel here
// because there is no separate max accumulator to seed.
//
// One thread per column: rma is a Wilder recurrence.
// ===========================================================================

#define NEO_S3_CHANDE_MULT      3.0
#define NEO_S3_CHANDE_DIR_LONG  1

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// The deque front for the rolling MAX of `v` over [window_start, i].
// See the header: smallest j with (isnan(v[j]) || v[j] > max of v over (j,i]).
__device__ __forceinline__ int neo_s3_dq_front_max(
    const double* __restrict__ v, int window_start, int i)
{
    int front = i;
    double run = -INFINITY;   // fmax identity; NaNs are skipped exactly as
                              // f64::max would skip them
    for (int j = i - 1; j >= window_start; --j) {
        run = fmax(run, v[j + 1]);
        const double vj = v[j];
        if (isnan(vj) || vj > run) front = j;
    }
    return front;
}

// The same, for the rolling MIN of `v`: popped when v[back] >= lo (:547).
__device__ __forceinline__ int neo_s3_dq_front_min(
    const double* __restrict__ v, int window_start, int i)
{
    int front = i;
    double run = INFINITY;
    for (int j = i - 1; j >= window_start; --j) {
        run = fmin(run, v[j + 1]);
        const double vj = v[j];
        if (isnan(vj) || vj < run) front = j;
    }
    return front;
}

extern "C" __global__ void neoethos_chande_batch_f64(
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

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int period = periods[r];
    const double mult = NEO_S3_CHANDE_MULT;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warmup = first_valid + period - 1;
    for (int i = 0; i < warmup && i < n; ++i) row[i] = neo_s3_qnan();

    const double alpha = 1.0 / (double)period;
    double sum_tr = 0.0;
    double rma = 0.0;
    double prev_close = close[first_valid];

    for (int i = first_valid; i < n; ++i) {
        const double hi = high[i];
        const double lo = low[i];

        double tr;
        if (i == first_valid) {
            tr = hi - lo;
        } else {
            const double hl = hi - lo;
            const double hc = fabs(hi - prev_close);
            const double lc = fabs(lo - prev_close);
            tr = fmax(fmax(hl, hc), lc);
        }

        if (i < warmup) {
            sum_tr += tr;
        } else {
            const int window_start = (i >= warmup) ? (i + 1 - period) : first_valid;
            const int ws = window_start < first_valid ? first_valid : window_start;

            if (i == warmup) {
                sum_tr += tr;
                rma = sum_tr / (double)period;
            } else {
                rma = fma(alpha, tr - rma, rma);
            }

#if NEO_S3_CHANDE_DIR_LONG
            const int f = neo_s3_dq_front_max(high, ws, i);
            row[i] = fma(-rma, mult, high[f]);
#else
            const int f = neo_s3_dq_front_min(low, ws, i);
            row[i] = fma(rma, mult, low[f]);
#endif
        }

        prev_close = close[i];
    }
}
