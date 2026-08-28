#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


#if __CUDA_ARCH__ >= 350
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif


__device__ __forceinline__ float ftz_f32(float x) {
    return (fabsf(x) < FLT_MIN) ? 0.0f : x;
}


struct Deque {
    int*   idx;
    float* val;
    int    cap;
    int    head;
    int    tail;
};

__device__ __forceinline__ void dq_init(Deque* d, int* idx_buf, float* val_buf, int cap) {
    d->idx = idx_buf; d->val = val_buf; d->cap = cap; d->head = 0; d->tail = 0;
}
__device__ __forceinline__ bool dq_empty(const Deque* d) { return d->head == d->tail; }
__device__ __forceinline__ int dq_dec(const Deque* d, int x) { return (x == 0 ? d->cap - 1 : x - 1); }
__device__ __forceinline__ int dq_inc(const Deque* d, int x) { return (x + 1 == d->cap ? 0 : x + 1); }


__device__ __forceinline__ void dq_expire(Deque* d, int start_idx) {
    if (!dq_empty(d) && d->idx[d->head] < start_idx) { d->head = dq_inc(d, d->head); }
}


__device__ __forceinline__ void dq_push_max(Deque* d, int idx, float v) {
    int t = d->tail;
    if (!dq_empty(d)) {
        int pos = dq_dec(d, t);
        while (pos != d->head && d->val[pos] < v) { t = pos; pos = dq_dec(d, pos); }
        if (pos == d->head && d->val[pos] < v) { t = d->head; d->head = dq_inc(d, d->head); }
    }
    d->idx[t] = idx; d->val[t] = v; d->tail = dq_inc(d, t);
}


__device__ __forceinline__ void dq_push_min(Deque* d, int idx, float v) {
    int t = d->tail;
    if (!dq_empty(d)) {
        int pos = dq_dec(d, t);
        while (pos != d->head && d->val[pos] > v) { t = pos; pos = dq_dec(d, pos); }
        if (pos == d->head && d->val[pos] > v) { t = d->head; d->head = dq_inc(d, d->head); }
    }
    d->idx[t] = idx; d->val[t] = v; d->tail = dq_inc(d, t);
}


struct Kahan {
    float s, c;
    __device__ __forceinline__ void reset() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s; }
};

extern "C" __global__
void srsi_build_rsi_f32(const float* __restrict__ prices,
                        int series_len,
                        int first_valid,
                        int period,
                        float* __restrict__ out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 0; i < series_len; ++i) out[i] = NAN;
    if (period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int warm = first_valid + period;
    if (warm >= series_len) return;

    double avg_gain = 0.0;
    double avg_loss = 0.0;
    double prev = (double)LDG(&prices[first_valid]);
    for (int i = first_valid + 1; i <= warm; ++i) {
        const double cur = (double)LDG(&prices[i]);
        const double ch = cur - prev;
        prev = cur;
        if (!isfinite(ch)) return;
        if (ch > 0.0) avg_gain += ch;
        else avg_loss += -ch;
    }

    const double inv_p = 1.0 / (double)period;
    avg_gain *= inv_p;
    avg_loss *= inv_p;
    double denom = avg_gain + avg_loss;
    out[warm] = (denom == 0.0) ? 50.0f : (float)(100.0 * avg_gain / denom);

    const double beta = 1.0 - inv_p;
    prev = LDG(&prices[warm]);
    for (int i = warm + 1; i < series_len; ++i) {
        const double cur = (double)LDG(&prices[i]);
        const double ch = cur - prev;
        prev = cur;
        if (!isfinite(ch)) return;
        const double gain = (ch > 0.0) ? ch : 0.0;
        const double loss = (ch < 0.0) ? -ch : 0.0;
        avg_gain = fma(avg_gain, beta, inv_p * gain);
        avg_loss = fma(avg_loss, beta, inv_p * loss);
        denom = avg_gain + avg_loss;
        out[i] = (denom == 0.0) ? 50.0f : (float)(100.0 * avg_gain / denom);
    }
}


extern "C" __global__
void srsi_fk_batch_f32(const float* __restrict__ rsi,
                       const int*   __restrict__ stoch_periods,
                       const int*   __restrict__ k_periods,
                       const int*   __restrict__ d_periods,
                       int series_len,
                       int first_valid,
                       int rsi_period,
                       int n_combos,
                       float* __restrict__ out_k,
                       float* __restrict__ out_d)
{
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];

    const int row_off = combo * series_len;
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len ||
        rsi_period <= 0 || sp <= 0) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    if (rsi_warmup >= series_len || stoch_warmup >= series_len) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    while (t < series_len) {
        float fk = NAN;
        if (t >= stoch_warmup) {
            const float rv = ftz_f32(LDG(&rsi[t]));
            const int start = t + 1 - sp;
            float hi = -1e30f;
            float lo =  1e30f;
            for (int i = start; i <= t; ++i) {
                const float v = ftz_f32(LDG(&rsi[i]));
                hi = fmaxf(hi, v);
                lo = fminf(lo, v);
            }
            const float denom = hi - lo;
            fk = (denom >= FLT_MIN) ? ((rv - lo) * 100.0f) / denom : 50.0f;
        }
        out_d[row_off + t] = fk;
        t += stride;
    }
}

extern "C" __global__
void srsi_sma_k_batch_f32(const float* __restrict__ rsi,
                          const int*   __restrict__ stoch_periods,
                          const int*   __restrict__ k_periods,
                          const int*   __restrict__ d_periods,
                          int series_len,
                          int first_valid,
                          int rsi_period,
                          int n_combos,
                          float* __restrict__ out_k,
                          float* __restrict__ out_d)
{
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];
    const int kp = k_periods[combo];

    const int row_off = combo * series_len;
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len ||
        rsi_period <= 0 || sp <= 0 || kp <= 0) {
        while (t < series_len) { out_k[row_off + t] = NAN; t += stride; }
        return;
    }

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    const int k_warmup     = stoch_warmup + kp - 1;
    if (rsi_warmup >= series_len || stoch_warmup >= series_len || k_warmup >= series_len) {
        while (t < series_len) { out_k[row_off + t] = NAN; t += stride; }
        return;
    }

    while (t < series_len) {
        float slow_k = NAN;
        if (t >= k_warmup) {
            const int start = t + 1 - kp;
            float sum = 0.0f;
            for (int i = start; i <= t; ++i) {
                sum += out_d[row_off + i];
            }
            slow_k = sum * (1.0f / (float)kp);
        }
        out_k[row_off + t] = slow_k;
        t += stride;
    }
}

extern "C" __global__
void srsi_sma_d_batch_f32(const float* __restrict__ rsi,
                          const int*   __restrict__ stoch_periods,
                          const int*   __restrict__ k_periods,
                          const int*   __restrict__ d_periods,
                          int series_len,
                          int first_valid,
                          int rsi_period,
                          int n_combos,
                          float* __restrict__ out_k,
                          float* __restrict__ out_d)
{
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];
    const int kp = k_periods[combo];
    const int dp = d_periods[combo];

    const int row_off = combo * series_len;
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len ||
        rsi_period <= 0 || sp <= 0 || kp <= 0 || dp <= 0) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    const int k_warmup     = stoch_warmup + kp - 1;
    const int d_warmup     = k_warmup + dp - 1;
    if (rsi_warmup >= series_len || stoch_warmup >= series_len ||
        k_warmup >= series_len || d_warmup >= series_len) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    while (t < series_len) {
        float slow_d = NAN;
        if (t >= d_warmup) {
            const int start = t + 1 - dp;
            float sum = 0.0f;
            for (int i = start; i <= t; ++i) {
                sum += out_k[row_off + i];
            }
            slow_d = sum * (1.0f / (float)dp);
        }
        out_d[row_off + t] = slow_d;
        t += stride;
    }
}


extern "C" __global__
void srsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int cols,
                                    int rows,
                                    int rsi_period,
                                    int stoch_period,
                                    int k_period,
                                    int d_period,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ k_out_tm,
                                    float* __restrict__ d_out_tm) {
    const int s = blockIdx.x;
    if (s >= cols) return;
    if (rsi_period <= 0 || stoch_period <= 0 || k_period <= 0 || d_period <= 0) return;

    const int stride = cols;
    int first = first_valids[s]; if (first < 0) first = 0; if (first >= rows) return;
    const int rsi_warmup   = first + rsi_period;
    const int stoch_warmup = rsi_warmup + stoch_period - 1;
    const int k_warmup     = stoch_warmup + k_period - 1;
    const int d_warmup     = k_warmup + d_period - 1;


    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        if (t < k_warmup) k_out_tm[t * stride + s] = NAN;
        if (t < d_warmup) d_out_tm[t * stride + s] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;


    float avg_gain = 0.0f, avg_loss = 0.0f;
    float prev = LDG(&prices_tm[first * stride + s]);
    for (int i = first + 1; i <= first + rsi_period && i < rows; ++i) {
        float cur = LDG(&prices_tm[i * stride + s]);
        const float ch = cur - prev; prev = cur;
        if (ch > 0.0f) avg_gain += ch; else avg_loss += -ch;
    }
    avg_gain /= (float)rsi_period; avg_loss /= (float)rsi_period;
    const float alpha = 1.0f / (float)rsi_period;


    extern __shared__ unsigned char smem2[];
    int*   max_idx = (int*)smem2;
    float* rsi_ring = (float*)(max_idx + stoch_period);
    int*   min_idx = (int*)(rsi_ring + stoch_period);
    float* min_val = (float*)(min_idx + stoch_period);
    float* ring_k  = (float*)(min_val + stoch_period);
    float* ring_d  = (float*)(ring_k + k_period);


    int rpos = 0; int rcnt = 0;
    float rsi = 50.0f;
    if (rsi_warmup < rows) {
        rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
    }
    rsi = ftz_f32(rsi);
    if (stoch_period > 1) {
        rsi_ring[rpos] = rsi;
        rpos = (rpos + 1 == stoch_period ? 0 : rpos + 1);
        if (rcnt < stoch_period) ++rcnt;

        for (int t = rsi_warmup + 1; t < rsi_warmup + stoch_period - 1 && t < rows; ++t) {
            float x = LDG(&prices_tm[t * stride + s]);
            const float prevp = LDG(&prices_tm[(t - 1) * stride + s]);
            const float ch = x - prevp;
            const float gain = (ch > 0.0f ? ch : 0.0f);
            const float loss = (ch < 0.0f ? -ch : 0.0f);
            avg_gain = fmaf(gain - avg_gain, alpha, avg_gain);
            avg_loss = fmaf(loss - avg_loss, alpha, avg_loss);
            rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
            rsi = ftz_f32(rsi);
            rsi_ring[rpos] = rsi;
            rpos = (rpos + 1 == stoch_period ? 0 : rpos + 1);
            if (rcnt < stoch_period) ++rcnt;
        }
    }

    float sum_k = 0.0f, sum_d = 0.0f; int head_k = 0, head_d = 0, cnt_k = 0, cnt_d = 0;
    const float inv_k = 1.0f / (float)k_period;
    const float inv_d = 1.0f / (float)d_period;

    for (int t = stoch_warmup; t < rows; ++t) {
        const float x = LDG(&prices_tm[t * stride + s]);
        const float prevp = LDG(&prices_tm[(t - 1) * stride + s]);
        const float ch = x - prevp;
        const float gain = (ch > 0.0f ? ch : 0.0f);
        const float loss = (ch < 0.0f ? -ch : 0.0f);
        avg_gain = fmaf(gain - avg_gain, alpha, avg_gain);
        avg_loss = fmaf(loss - avg_loss, alpha, avg_loss);
        rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
        rsi = ftz_f32(rsi);

        rsi_ring[rpos] = rsi; rpos = (rpos + 1 == stoch_period ? 0 : rpos + 1); if (rcnt < stoch_period) ++rcnt;
        float hi = rsi, lo = rsi;
        int cnt = rcnt < stoch_period ? rcnt : stoch_period;
        for (int j = 0; j < cnt - 1; ++j) {
            float v = rsi_ring[(rpos + j) % stoch_period];
            hi = fmaxf(hi, v);
            lo = fminf(lo, v);
        }

        const float denom = hi - lo;

        float fk = (isfinite(hi) && isfinite(lo) && denom >= FLT_MIN) ? ((rsi - lo) * 100.0f) / denom : 50.0f;

        if (cnt_k < k_period) { sum_k += fk; ring_k[head_k] = fk; ++cnt_k; if (++head_k == k_period) head_k = 0; }
        else                   { sum_k += fk - ring_k[head_k]; ring_k[head_k] = fk; if (++head_k == k_period) head_k = 0; }
        if (t >= k_warmup) {
            const float slow_k = sum_k * inv_k; k_out_tm[t * stride + s] = slow_k;
            if (cnt_d < d_period) { sum_d += slow_k; ring_d[head_d] = slow_k; ++cnt_d; if (++head_d == d_period) head_d = 0; }
            else                   { sum_d += slow_k - ring_d[head_d]; ring_d[head_d] = slow_k; if (++head_d == d_period) head_d = 0; }
            if (t >= d_warmup) d_out_tm[t * stride + s] = sum_d * inv_d;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 6
 *
 * ORACLE: `srsi_scalar` (src/indicators/srsi.rs:335). SINGLE oracle by
 * construction -- `srsi_avx2` (:554), `srsi_avx512_short` (:585) and
 * `srsi_avx512_long` (:596) all delegate straight back to `srsi_scalar`, so
 * unlike wilders/vwap there is no seed-order disagreement to settle here.
 *
 * PERIOD-INVARIANT. `compute_srsi_batch` reads `rsi_period` (14),
 * `stoch_period` (14), `k` (3) and `d` (3) -- NEVER `period`
 * (cpu_batch.rs:6308-6311). Five swept periods give five identical CPU
 * columns, so the kernel writes five identical rows.
 *
 * MULTI-OUTPUT: emits K, which is what `output_id == "value"` resolves to
 * (cpu_batch.rs:6329). Never `d` silently.
 *
 * THE FLT_MIN THAT WAS HERE IS GONE. The f32 lane in this file guarded the
 * stochastic denominator with `FLT_MIN` (~1.18e-38). That constant is sized
 * for f32 and copying it into an f64 kernel is the exact bug the brief names.
 * It is not RE-SIZED, it is REMOVED: the CPU has no epsilon at all -- it tests
 * `hi > lo` (:511) and substitutes 50.0 otherwise. An epsilon of any magnitude
 * would answer differently from the CPU on a flat window.
 *
 * WARMUPS, four of them, each one exactly the CPU's (:364-367):
 *   rsi_warmup   = first + rsi_period
 *   stoch_warmup = rsi_warmup + stoch_period - 1
 *   k_warmup     = stoch_warmup + k_period - 1
 *   d_warmup     = k_warmup + d_period - 1
 * `n <= d_warmup` is NotEnoughValidData (:370) -- the whole column stays NaN
 * even though only the D series needs that many bars, because the CPU errors
 * out before writing anything.
 *
 * THE SLIDING EXTREME IS THE CPU'S BLOCK DECOMPOSITION, NOT A DEQUE. The CPU
 * builds per-block prefix and suffix max/min arrays (:441-490) and combines
 * max(suff[t+1-sp], pref[t]). Its comparison is the ternary form, NOT `fmax`,
 * so a NaN entering the running accumulator STICKS while a NaN arriving as the
 * new value is ignored, and which of the two happens depends on the scan
 * direction. A monotone deque would agree on clean data and disagree the
 * moment a hole appears. The kernel therefore re-folds both block scans per
 * bar in the CPU's directions: O(stoch_period) per bar, 14 operations at the
 * default, and ZERO global scratch -- the alternative was two m-wide arrays
 * per combo, which at 100k bars is 1.6 MB of local memory per thread.
 *
 * WHY A RING OF EXACTLY `stoch_period` IS SUFFICIENT. Because the block size
 * equals the window length, the suffix block always ENDS at or before the
 * current bar: if t and t+1-sp share a block then that block ends at t; if
 * they do not, the suffix block ends exactly where t's block begins. So the
 * two folds together read precisely the sp bars [t+1-sp, t] and never a
 * future one.
 *
 * NaN. Rule 4 of the brief says use fmax/fmin where the CPU uses `f64::max`.
 * The CPU here does NOT use `f64::max`; it uses a ternary chain, and
 * reproducing that chain is what matches it. fmax would DISAGREE.
 *
 * ONE DELIBERATE DEPARTURE, RECORDED. When a bar is non-finite the CPU
 * SKIPS the RSI store (:415-427) and leaves that slot at whatever
 * `alloc_with_nan_prefix` left there -- which, in release, is UNINITIALIZED
 * memory (helpers.rs:110-118). There is no value to match. The kernel carries
 * the previous RSI forward, which is the only defined behaviour available;
 * the CPU's is a crate defect and is reported as one.
 *
 * SEQUENTIAL, one thread per combo column. Three fixed rings, 20 doubles.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define SRSI_NEO_RSI_PERIOD   14   /* cpu_batch.rs:6308 */
#define SRSI_NEO_STOCH_PERIOD 14   /* :6309 */
#define SRSI_NEO_K            3    /* :6310 */
#define SRSI_NEO_D            3    /* :6311 */

extern "C" __global__
void srsi_neo_batch_f64(const double* __restrict__ data,
                        int series_len,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;                       /* PERIOD-INVARIANT -- see header. */

    const int n = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    if (first_valid < 0 || first_valid >= n) return;

    const int rp_i  = SRSI_NEO_RSI_PERIOD;
    const int sp    = SRSI_NEO_STOCH_PERIOD;
    const int kp    = SRSI_NEO_K;
    const int dp    = SRSI_NEO_D;
    const int first = first_valid;

    int max_need = rp_i;
    if (sp > max_need) max_need = sp;
    if (kp > max_need) max_need = kp;
    if (dp > max_need) max_need = dp;
    if (n - first < max_need) return;                    /* :357 */

    const int rsi_warmup   = first + rp_i;               /* :364 */
    const int stoch_warmup = rsi_warmup + sp - 1;        /* :365 */
    const int k_warmup     = stoch_warmup + kp - 1;      /* :366 */
    const int d_warmup     = k_warmup + dp - 1;          /* :367 */
    if (n <= d_warmup) return;                           /* :370 */

    const int base = rsi_warmup;

    /* ---- RSI seed, :383-400 -------------------------------------------- */
    double avg_gain = 0.0, avg_loss = 0.0;
    double prev = data[first];
    const int end_init = min(first + rp_i, n - 1);
    for (int i = first + 1; i <= end_init; ++i) {
        const double cur = data[i];
        if (isfinite(cur) && isfinite(prev)) {
            const double ch = cur - prev;
            if (ch > 0.0) avg_gain += ch; else avg_loss += -ch;
        }
        prev = cur;
    }

    const double rpf = (double)rp_i;
    avg_gain /= rpf;
    avg_loss /= rpf;
    const double alpha = 1.0 / rpf;

    double rsi_ring[SRSI_NEO_STOCH_PERIOD];
    for (int j = 0; j < sp; ++j) rsi_ring[j] = NEO_F64_NAN;

    double rsi_v = (avg_loss == 0.0)
        ? 100.0
        : (100.0 - 100.0 / (1.0 + avg_gain / avg_loss));     /* :403 */
    rsi_ring[0] = rsi_v;

    double sum_k = 0.0, sum_d = 0.0;
    double fk_ring[SRSI_NEO_K];
    double sk_ring[SRSI_NEO_D];
    for (int j = 0; j < kp; ++j) fk_ring[j] = 0.0;
    for (int j = 0; j < dp; ++j) sk_ring[j] = 0.0;
    int fk_pos = 0, sk_pos = 0;

    const int i0 = stoch_warmup;
    prev = data[rsi_warmup];

    for (int i = base; i < n; ++i) {
        if (i > base) {
            const double cur = data[i];
            if (isfinite(cur) && isfinite(prev)) {
                const double ch   = cur - prev;
                const double gain = (ch > 0.0) ?  ch : 0.0;
                const double loss = (ch < 0.0) ? -ch : 0.0;
                avg_gain = fma(gain - avg_gain, alpha, avg_gain);   /* :417 */
                avg_loss = fma(loss - avg_loss, alpha, avg_loss);   /* :418 */
                rsi_v = (avg_loss == 0.0)
                    ? 100.0
                    : (100.0 - 100.0 / (1.0 + avg_gain / avg_loss));
            }
            prev = cur;
            rsi_ring[(i - base) % SRSI_NEO_STOCH_PERIOD] = rsi_v;
        }

        if (i < i0) continue;

        const int t       = i - base;
        const int t_start = t + 1 - sp;

        const int pref_start   = (t / sp) * sp;
        const int b_suff       = t_start / sp;
        const int block_end_ex = (b_suff + 1) * sp;

        /* pref over [pref_start .. t], ASCENDING -- :455-464 */
        double pmx = rsi_ring[pref_start % SRSI_NEO_STOCH_PERIOD];
        double pmn = pmx;
        for (int j = pref_start + 1; j <= t; ++j) {
            const double v = rsi_ring[j % SRSI_NEO_STOCH_PERIOD];
            pmx = (v > pmx) ? v : pmx;
            pmn = (v < pmn) ? v : pmn;
        }

        /* suff over [t_start .. block_end_ex-1], DESCENDING -- :477-487 */
        const int last = block_end_ex - 1;
        double smx = rsi_ring[last % SRSI_NEO_STOCH_PERIOD];
        double smn = smx;
        for (int j = last - 1; j >= t_start; --j) {
            const double v = rsi_ring[j % SRSI_NEO_STOCH_PERIOD];
            smx = (v > smx) ? v : smx;
            smn = (v < smn) ? v : smn;
        }

        const double hi = (smx > pmx) ? smx : pmx;         /* :508 */
        const double lo = (smn < pmn) ? smn : pmn;         /* :509 */
        const double x  = rsi_ring[t % SRSI_NEO_STOCH_PERIOD];

        /* No epsilon -- :511-515. */
        const double fk = (hi > lo) ? (((x - lo) * 100.0) / (hi - lo)) : 50.0;

        sum_k += fk;
        if (i >= i0 + kp) sum_k -= fk_ring[fk_pos];
        fk_ring[fk_pos] = fk;
        fk_pos += 1; if (fk_pos == kp) fk_pos = 0;

        if (i >= k_warmup) {
            const double sk = sum_k / (double)kp;
            o[i] = sk;                                     /* :532 -- K */

            sum_d += sk;
            if (i >= k_warmup + dp) sum_d -= sk_ring[sk_pos];
            sk_ring[sk_pos] = sk;
            sk_pos += 1; if (sk_pos == dp) sk_pos = 0;
        }
    }
}
