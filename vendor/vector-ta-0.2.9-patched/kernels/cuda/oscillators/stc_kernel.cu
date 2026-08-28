#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#ifndef STC_BLOCK_X
#define STC_BLOCK_X 1
#endif

#ifndef STC_SMALL_K
#define STC_SMALL_K 16
#endif


#define STC_RANGE_EPS 2.2204460492503131e-16f


#define STC_BATCH_SMEM_BYTES(max_k) ((size_t)(max_k) * (2*sizeof(float) + 4*sizeof(int)))


static __device__ __forceinline__ float ema_update_f32(float prev, float a, float x) {

    return __fmaf_rn(a, (x - prev), prev);
}


static __device__ __forceinline__ float div_rn_f32(float num, float den) {
    return __fdiv_rn(num, den);
}


struct KahanF32 {
    float s;
    float c;
    __device__ __forceinline__ void reset() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ __forceinline__ float result() const { return s + c; }
};


struct IndexDeque {
    int*  buf;
    int   head;
    int   len;
    int   cap;
    float* ring;
    bool  is_min;

    __device__ __forceinline__ void init(int* storage, int capacity, float* ring_ptr, bool as_min) {
        buf = storage; cap = capacity; ring = ring_ptr; is_min = as_min; head = 0; len = 0;
    }
    __device__ __forceinline__ void reset() { head = 0; len = 0; }
    __device__ __forceinline__ void push(int idx, float v) {

        while (len > 0) {
            int last = head + len - 1; if (last >= cap) last -= cap;
            float backv = ring[ buf[last] % cap ];
            if (is_min ? (backv >= v) : (backv <= v)) { len--; }
            else break;
        }

        int tail_pos = head + len; if (tail_pos >= cap) tail_pos -= cap;
        buf[tail_pos] = idx;
        if (len < cap) len++;
    }
    __device__ __forceinline__ void pop_expired(int min_idx_allowed) {
        while (len > 0 && buf[head] < min_idx_allowed) {
            head++; if (head == cap) head = 0; len--;
        }
    }
    __device__ __forceinline__ bool empty() const { return len == 0; }
    __device__ __forceinline__ float front_val() const { return ring[ buf[head] % cap ]; }
};


static __device__ __forceinline__ void stc_compute_series_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int fast,
    int slow,
    int k,
    int d,
    int max_k,
    float* __restrict__ out)
{
    if (len <= 0 || first_valid >= len) return;


    extern __shared__ unsigned char shmem[];
    float* macd_ring = reinterpret_cast<float*>(shmem);
    float* d_ring    = macd_ring + max_k;
    int* macd_min_idx = reinterpret_cast<int*>(d_ring + max_k);
    int* macd_max_idx = macd_min_idx + max_k;
    int* d_min_idx    = macd_max_idx + max_k;
    int* d_max_idx    = d_min_idx + max_k;


    const int warm = first_valid + max(max(fast, slow), max(k, d)) - 1;

    for (int i = 0; i < min(warm, len); ++i) out[i] = NAN;
    if (warm >= len) return;


    const float fast_a = div_rn_f32(2.0f, (float)(fast + 1));
    const float slow_a = div_rn_f32(2.0f, (float)(slow + 1));
    const float d_a    = div_rn_f32(2.0f, (float)(d + 1));


    KahanF32 fast_acc; fast_acc.reset();
    KahanF32 slow_acc; slow_acc.reset();
    bool fast_seed_nan = false, slow_seed_nan = false;
    const int f_end = min(fast, len - first_valid);
    const int s_end = min(slow, len - first_valid);
    for (int i = 0; i < f_end; ++i) { float v = prices[first_valid + i]; if (!isfinite(v)) { fast_seed_nan = true; break; } fast_acc.add(v); }
    for (int i = 0; i < s_end; ++i) { float v = prices[first_valid + i]; if (!isfinite(v)) { slow_seed_nan = true; break; } slow_acc.add(v); }
    float fast_ema = (f_end == fast && !fast_seed_nan) ? div_rn_f32(fast_acc.result(), (float)fast) : NAN;
    float slow_ema = (s_end == slow && !slow_seed_nan) ? div_rn_f32(slow_acc.result(), (float)slow) : NAN;


    IndexDeque macd_min, macd_max, d_min, d_max;
    macd_min.init(macd_min_idx, k, macd_ring, true);
    macd_max.init(macd_max_idx, k, macd_ring, false);
    d_min.init(d_min_idx, k, d_ring, true);
    d_max.init(d_max_idx, k, d_ring, false);


    int macd_run = 0, d_run = 0;


    KahanF32 d_seed_acc; d_seed_acc.reset(); int d_seed_cnt = 0; float d_ema = NAN;
    KahanF32 final_seed_acc; final_seed_acc.reset(); int final_seed_cnt = 0; float final_ema = NAN;

    const int fast_thr = fast > 0 ? (fast - 1) : 0;
    const int slow_thr = slow > 0 ? (slow - 1) : 0;


    for (int i = 0; i < len; ++i) {
        const float x = prices[i];


        if (i >= first_valid) {
            const int rel = i - first_valid;
            if (rel >= fast_thr) {
                if (rel != fast_thr) { if (isfinite(x) && isfinite(fast_ema)) fast_ema = ema_update_f32(fast_ema, fast_a, x); else fast_ema = NAN; }
            }
            if (rel >= slow_thr) {
                if (rel != slow_thr) { if (isfinite(x) && isfinite(slow_ema)) slow_ema = ema_update_f32(slow_ema, slow_a, x); else slow_ema = NAN; }
            }
        }


        float macd; unsigned char macd_is_valid;
        if (i >= first_valid + slow_thr && isfinite(fast_ema) && isfinite(slow_ema)) { macd = fast_ema - slow_ema; macd_is_valid = 1u; }
        else { macd = NAN; macd_is_valid = 0u; }


        float stok = NAN;
        if (macd_is_valid) {

            macd_ring[i % k] = macd;
            macd_run += 1;
            if (k <= STC_SMALL_K) {

                float mn = macd_ring[(i - (macd_run-1)) % k];
                float mx = mn;
                int start = i - min(macd_run, k) + 1;
                for (int j = 0; j < min(macd_run, k); ++j) { float v = macd_ring[(start + j) % k]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                if (macd_run >= k) {
                    const float range = mx - mn;
                    stok = (fabsf(range) > STC_RANGE_EPS) ? ((macd - mn) * div_rn_f32(100.0f, range)) : 50.0f;
                } else { stok = 50.0f; }
            } else {

                macd_min.push(i, macd); macd_max.push(i, macd);
                const int left = i - k + 1;
                macd_min.pop_expired(left); macd_max.pop_expired(left);
                if (macd_run >= k && !macd_min.empty() && !macd_max.empty()) {
                    const float mn = macd_min.front_val();
                    const float mx = macd_max.front_val();
                    const float range = mx - mn;
                    stok = (fabsf(range) > STC_RANGE_EPS) ? ((macd - mn) * div_rn_f32(100.0f, range)) : 50.0f;
                } else { stok = 50.0f; }
            }
        } else {
            macd_run = 0; macd_min.reset(); macd_max.reset(); stok = NAN;
        }


        float d_val = NAN;
        if (isfinite(stok)) {
            if (d_seed_cnt < d) {
                d_seed_acc.add(stok);
                d_seed_cnt += 1;
                const float sum = d_seed_acc.result();
                if (d_seed_cnt == d) { d_ema = div_rn_f32(sum, (float)d); d_val = d_ema; }
                else { d_val = div_rn_f32(sum, (float)d_seed_cnt); }
            } else {
                d_ema = ema_update_f32(d_ema, d_a, stok);
                d_val = d_ema;
            }
        } else {
            if (d_seed_cnt == 0) d_val = NAN;
            else if (d_seed_cnt < d) d_val = div_rn_f32(d_seed_acc.result(), (float)d_seed_cnt);
            else d_val = d_ema;
        }


        float kd = NAN;
        if (isfinite(d_val)) {
            d_ring[i % k] = d_val; d_run += 1;
            if (k <= STC_SMALL_K) {
                float mn = d_ring[(i - (d_run-1)) % k]; float mx = mn;
                int start = i - min(d_run, k) + 1;
                for (int j = 0; j < min(d_run, k); ++j) { float v = d_ring[(start + j) % k]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                if (d_run >= k) { const float range = mx - mn; kd = (fabsf(range) > STC_RANGE_EPS) ? ((d_val - mn) * div_rn_f32(100.0f, range)) : 50.0f; } else { kd = 50.0f; }
            } else {
                d_min.push(i, d_val); d_max.push(i, d_val);
                const int left = i - k + 1;
                d_min.pop_expired(left); d_max.pop_expired(left);
                if (d_run >= k && !d_min.empty() && !d_max.empty()) {
                    const float mn = d_min.front_val(); const float mx = d_max.front_val();
                    const float range = mx - mn; kd = (fabsf(range) > STC_RANGE_EPS) ? ((d_val - mn) * div_rn_f32(100.0f, range)) : 50.0f;
                } else { kd = 50.0f; }
            }
        } else { d_min.reset(); d_max.reset(); }


        float out_i = NAN;
        if (isfinite(kd)) {
            if (final_seed_cnt < d) {
                final_seed_acc.add(kd);
                final_seed_cnt += 1;
                const float sum = final_seed_acc.result();
                if (final_seed_cnt == d) { final_ema = div_rn_f32(sum, (float)d); out_i = final_ema; }
                else { out_i = div_rn_f32(sum, (float)final_seed_cnt); }
            } else {
                final_ema = ema_update_f32(final_ema, d_a, kd);
                out_i = final_ema;
            }
        } else {
            if (final_seed_cnt == 0) out_i = NAN;
            else if (final_seed_cnt < d) out_i = div_rn_f32(final_seed_acc.result(), (float)final_seed_cnt);
            else out_i = final_ema;
        }

        if (i >= warm) out[i] = out_i;
    }
}


extern "C" __global__ __launch_bounds__(1)
void stc_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ fasts,
                   const int* __restrict__ slows,
                   const int* __restrict__ ks,
                   const int* __restrict__ ds,
                   int series_len,
                   int first_valid,
                   int n_rows,
                   int max_k,
                   float* __restrict__ out)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int fast = fasts[row];
    const int slow = slows[row];
    const int kk   = ks[row];
    const int dd   = ds[row];
    if (fast <= 0 || slow <= 0 || kk <= 0 || dd <= 0) return;

    const int base = row * series_len;


    if (threadIdx.x != 0) return;
    stc_compute_series_f32(prices, series_len, first_valid, fast, slow, kk, dd, max_k, out + base);
}


extern "C" __global__
void stc_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int fast,
                                   int slow,
                                   int k,
                                   int d,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    const int first = first_valids[s];


    int warm = first + max(max(fast, slow), max(k, d)) - 1;
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = NAN;
    if (warm >= rows) return;


    const float fast_a = div_rn_f32(2.0f, (float)(fast + 1));
    const float slow_a = div_rn_f32(2.0f, (float)(slow + 1));
    const float d_a    = div_rn_f32(2.0f, (float)(d + 1));


    KahanF32 fast_acc; fast_acc.reset();
    KahanF32 slow_acc; slow_acc.reset();
    const int f_end = min(fast, rows - first);
    const int s_end = min(slow, rows - first);
    for (int i = 0; i < f_end; ++i) fast_acc.add(prices_tm[(first + i) * cols + s]);
    for (int i = 0; i < s_end; ++i) slow_acc.add(prices_tm[(first + i) * cols + s]);
    float fast_ema = (f_end == fast) ? div_rn_f32(fast_acc.result(), (float)fast) : NAN;
    float slow_ema = (s_end == slow) ? div_rn_f32(slow_acc.result(), (float)slow) : NAN;


    const int KMAX = 2048;
    const int kk = (k <= KMAX) ? k : KMAX;
    float macd_ring[KMAX];
    float d_ring[KMAX];
    for (int i = 0; i < kk; ++i) { macd_ring[i] = NAN; d_ring[i] = NAN; }


    KahanF32 d_seed_acc; d_seed_acc.reset(); int d_seed_cnt = 0; float d_ema = NAN;
    KahanF32 final_seed_acc; final_seed_acc.reset(); int final_seed_cnt = 0; float final_ema = NAN;
    const int fast_thr = fast > 0 ? (fast - 1) : 0;
    const int slow_thr = slow > 0 ? (slow - 1) : 0;
    int macd_run = 0, d_run = 0;

    for (int i = 0; i < rows; ++i) {
        const float x = prices_tm[i * cols + s];


        if (i >= first) {
            const int rel = i - first;
            if (rel >= fast_thr) {
                if (rel != fast_thr) { if (isfinite(x) && isfinite(fast_ema)) fast_ema = ema_update_f32(fast_ema, fast_a, x); else fast_ema = NAN; }
            }
            if (rel >= slow_thr) {
                if (rel != slow_thr) { if (isfinite(x) && isfinite(slow_ema)) slow_ema = ema_update_f32(slow_ema, slow_a, x); else slow_ema = NAN; }
            }
        }

        float macd; unsigned char macd_is_valid;
        if (i >= first + slow_thr && isfinite(fast_ema) && isfinite(slow_ema)) { macd = fast_ema - slow_ema; macd_is_valid = 1u; }
        else { macd = NAN; macd_is_valid = 0u; }

        float stok = NAN;
        if (macd_is_valid) {
            macd_ring[i % kk] = macd; macd_run += 1;
            if (macd_run >= k) {
                float mn = macd_ring[(i - (k-1)) % kk], mx = mn;
                for (int j = 1; j < k; ++j) { float v = macd_ring[(i - (k-1) + j) % kk]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                const float range = mx - mn; stok = (fabsf(range) > STC_RANGE_EPS) ? ((macd - mn) * div_rn_f32(100.0f, range)) : 50.0f;
            } else { stok = 50.0f; }
        } else { macd_run = 0; }

        float d_val = NAN;
        if (isfinite(stok)) {
            if (d_seed_cnt < d) {
                d_seed_acc.add(stok);
                d_seed_cnt += 1;
                const float sum = d_seed_acc.result();
                if (d_seed_cnt == d) { d_ema = div_rn_f32(sum, (float)d); d_val = d_ema; }
                else { d_val = div_rn_f32(sum, (float)d_seed_cnt); }
            } else {
                d_ema = ema_update_f32(d_ema, d_a, stok);
                d_val = d_ema;
            }
        } else {
            if (d_seed_cnt == 0) d_val = NAN;
            else if (d_seed_cnt < d) d_val = div_rn_f32(d_seed_acc.result(), (float)d_seed_cnt);
            else d_val = d_ema;
        }

        float kd = NAN;
        if (isfinite(d_val)) {
            d_ring[i % kk] = d_val; d_run += 1;
            if (d_run >= k) {
                float mn = d_ring[(i - (k-1)) % kk], mx = mn;
                for (int j = 1; j < k; ++j) { float v = d_ring[(i - (k-1) + j) % kk]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                const float range = mx - mn; kd = (fabsf(range) > STC_RANGE_EPS) ? ((d_val - mn) * div_rn_f32(100.0f, range)) : 50.0f;
            } else { kd = 50.0f; }
        } else { d_run = 0; }

        float out_i = NAN;
        if (isfinite(kd)) {
            if (final_seed_cnt < d) {
                final_seed_acc.add(kd);
                final_seed_cnt += 1;
                const float sum = final_seed_acc.result();
                if (final_seed_cnt == d) { final_ema = div_rn_f32(sum, (float)d); out_i = final_ema; }
                else { out_i = div_rn_f32(sum, (float)final_seed_cnt); }
            } else {
                final_ema = ema_update_f32(final_ema, d_a, kd);
                out_i = final_ema;
            }
        } else {
            if (final_seed_cnt == 0) out_i = NAN;
            else if (final_seed_cnt < d) out_i = div_rn_f32(final_seed_acc.result(), (float)final_seed_cnt);
            else out_i = final_ema;
        }

        if (i >= warm) out_tm[i * cols + s] = out_i;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 6
 *
 * ORACLE: `stc_scalar` (src/indicators/stc.rs:479) with the default
 * `fast_ma_type == slow_ma_type == "ema"` (:132, :136). That function then
 * picks between TWO implementations on a property of the DATA, not of the
 * parameters (:491):
 *
 *   all of data[first..] finite -> `stc_scalar_classic_ema_finite` (:561)
 *   otherwise                   -> `stc_scalar_classic_ema`        (:773)
 *
 * BOTH ARE TRANSCRIBED HERE, and the kernel runs the same scan to choose
 * between them. They are NOT the same doubles: the finite path counts ring
 * occupancy with `macd_count` and only stores VALID entries, while the
 * non-finite path keeps a per-slot validity ring, requires `macd_valid_sum ==
 * k`, and -- the part that actually changes numbers -- gates the MACD on
 * `slow_init_cnt >= slow` ALONE (:857) instead of on both EMA counters
 * (:639), and CARRIES the last d/final EMA forward through a hole instead of
 * emitting NaN. Implementing only the finite path would have been correct on
 * clean data and silently wrong on the first gapped symbol.
 *
 * PERIOD-INVARIANT. `compute_stc_batch` reads `fast_period` (23),
 * `slow_period` (50), `k_period` (10) and `d_period` (3) -- NEVER `period`
 * (cpu_batch.rs:16571-16574). Five swept periods give five identical CPU
 * columns, so the kernel writes five identical rows.
 *
 * SINGLE OUTPUT: "value" is the only column (cpu_batch.rs:16591).
 *
 * NO WARMUP PREFIX BEYOND `first`. `stc_with_kernel` allocates
 * `alloc_with_nan_prefix(len, first)` (:311) -- the ONLY blanked region is
 * before `first`. Every bar from `first` on is written by the walk, most of
 * them 50.0 or NaN. A kernel that blanked out to `first + slow + k + d` would
 * erase bars the CPU emits.
 *
 * ONE ROUNDING: the file's local `fma` helper is `(x - prev).mul_add(a, prev)`
 * (:571) -- ONE rounding -- so `fma(x - prev, a, prev)` here, not
 * `prev + a*(x-prev)`.
 *
 * THE EPSILON IS ALREADY f64. `EPS = f64::EPSILON` (:576) is 2.22e-16, the
 * DOUBLE epsilon. Carried across unchanged because it was never an f32
 * constant.
 *
 * THE EXTREMES ARE TERNARY CHAINS, NOT fmin/fmax. `if v < mn { mn = v }`
 * (:664) keeps `mn` when `v` is NaN and keeps NaN once `mn` is NaN. fmin
 * would return the non-NaN operand and disagree.
 *
 * TWO PER-THREAD RINGS OF `k_period` DOUBLES (10 at the default) plus two
 * `k_period` byte rings. Sized at a compile-time 512 so an oversized k is
 * refused by name rather than truncated -- but k is not swept, so this is a
 * guard, not a live limit.
 *
 * SEQUENTIAL, one thread per combo column.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* f64::EPSILON -- stc.rs:576. NOT an f32 constant. */
#define STC_NEO_F64_EPS 2.2204460492503131e-16

#define STC_NEO_FAST 23   /* cpu_batch.rs:16571 */
#define STC_NEO_SLOW 50   /* :16572 */
#define STC_NEO_K    10   /* :16573 */
#define STC_NEO_D    3    /* :16574 */

#define STC_NEO_MAX_K 512

/* stc.rs:571 -- ONE rounding. */
__device__ __forceinline__ double stc_neo_ema_step(double prev, double a, double x)
{
    return fma(x - prev, a, prev);
}

/* The ternary min/max fold the CPU performs over a k-slot ring (:659-671). */
__device__ __forceinline__ double stc_neo_stoch(const double* __restrict__ ring,
                                                int k, double v)
{
    double mn = ring[0];
    double mx = mn;
    for (int j = 1; j < k; ++j) {
        const double x = ring[j];
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    const double range = mx - mn;
    if (fabs(range) > STC_NEO_F64_EPS) return (v - mn) * (100.0 / range);
    return 50.0;
}

extern "C" __global__
void stc_neo_batch_f64(const double* __restrict__ data,
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

    if (len == 0) return;
    if (first_valid < 0 || first_valid >= len) return;

    const int fast = STC_NEO_FAST;
    const int slow = STC_NEO_SLOW;
    const int k    = STC_NEO_D > STC_NEO_K ? STC_NEO_D : STC_NEO_K;  /* k ring width */
    const int kk   = STC_NEO_K;
    const int d    = STC_NEO_D;
    (void)k;
    if (kk <= 0 || kk > STC_NEO_MAX_K) return;

    int needed = fast;
    if (slow > needed) needed = slow;
    if (kk   > needed) needed = kk;
    if (d    > needed) needed = d;
    if (len - first_valid < needed) return;               /* :294 */

    const int first = first_valid;
    const int n     = len - first;

    const double fast_a   = 2.0 / ((double)fast + 1.0);
    const double slow_a   = 2.0 / ((double)slow + 1.0);
    const double d_a      = 2.0 / ((double)d + 1.0);
    const double fast_inv = 1.0 / (double)fast;
    const double slow_inv = 1.0 / (double)slow;
    const double d_inv    = 1.0 / (double)d;

    /* stc.rs:491 -- the CPU picks its implementation on this scan. */
    bool all_finite = true;
    for (int i = first; i < len; ++i) {
        if (!isfinite(data[i])) { all_finite = false; break; }
    }

    double fast_sum = 0.0, slow_sum = 0.0;
    int    fast_init_cnt = 0, slow_init_cnt = 0;
    double fast_ema = NEO_F64_NAN, slow_ema = NEO_F64_NAN;

    double macd_ring[STC_NEO_MAX_K];
    double d_ring[STC_NEO_MAX_K];
    unsigned char macd_valid_ring[STC_NEO_MAX_K];
    unsigned char d_valid_ring[STC_NEO_MAX_K];
    for (int j = 0; j < kk; ++j) {
        macd_ring[j] = NEO_F64_NAN;
        d_ring[j]    = NEO_F64_NAN;
        macd_valid_ring[j] = 0;
        d_valid_ring[j]    = 0;
    }
    int macd_count = 0, macd_vpos = 0, macd_valid_sum = 0;
    int d_count = 0, d_vpos = 0, d_valid_sum = 0;

    double d_ema = NEO_F64_NAN, d_sum = 0.0;
    int    d_init_cnt = 0;
    double final_ema = NEO_F64_NAN, final_sum = 0.0;
    int    final_init_cnt = 0;

    for (int i = 0; i < n; ++i) {
        const double x = data[first + i];
        const bool   x_is_finite = isfinite(x);

        /* --------------------------------------------------------------
         * The two paths differ from here. `all_finite` never changes
         * inside the loop, so the branch is uniform across the whole walk
         * and costs nothing in divergence.
         * -------------------------------------------------------------- */
        if (all_finite || x_is_finite) {
            if (fast_init_cnt < fast) {
                fast_init_cnt += 1;
                fast_sum += x;
                if (fast_init_cnt == fast) fast_ema = fast_sum * fast_inv;
            } else {
                fast_ema = stc_neo_ema_step(fast_ema, fast_a, x);
            }
            if (slow_init_cnt < slow) {
                slow_init_cnt += 1;
                slow_sum += x;
                if (slow_init_cnt == slow) slow_ema = slow_sum * slow_inv;
            } else {
                slow_ema = stc_neo_ema_step(slow_ema, slow_a, x);
            }
        }

        double stok;
        double macd;
        if (all_finite) {
            /* stc_scalar_classic_ema_finite, :639-683 */
            const bool macd_is_valid = (fast_init_cnt >= fast) && (slow_init_cnt >= slow);
            macd = macd_is_valid ? (fast_ema - slow_ema) : NEO_F64_NAN;

            if (macd_is_valid) {
                macd_ring[macd_vpos] = macd;
                macd_vpos += 1; if (macd_vpos == kk) macd_vpos = 0;
                if (macd_count < kk) macd_count += 1;
            }

            if (!macd_is_valid)          stok = NEO_F64_NAN;
            else if (macd_count == kk)   stok = stc_neo_stoch(macd_ring, kk, macd);
            else                         stok = 50.0;
        } else {
            /* stc_scalar_classic_ema, :856-903 */
            macd = (slow_init_cnt >= slow) ? (fast_ema - slow_ema) : NEO_F64_NAN;

            if (i >= kk) macd_valid_sum -= (int)macd_valid_ring[macd_vpos];
            const unsigned char mv = isnan(macd) ? 0 : 1;
            macd_valid_ring[macd_vpos] = mv;
            macd_valid_sum += (int)mv;
            if (mv) macd_ring[macd_vpos] = macd;
            macd_vpos += 1; if (macd_vpos == kk) macd_vpos = 0;

            if (macd_valid_sum == kk && mv) stok = stc_neo_stoch(macd_ring, kk, macd);
            else if (mv)                    stok = 50.0;
            else                            stok = NEO_F64_NAN;
        }

        /* ---- d_val. The finite path emits NaN when stok is NaN (:700);
           the non-finite path CARRIES the running mean/EMA (:920-926). --- */
        double d_val;
        if (!isnan(stok)) {
            if (d_init_cnt < d) {
                d_sum += stok;
                d_init_cnt += 1;
                if (d_init_cnt == d) { d_ema = d_sum * d_inv; d_val = d_ema; }
                else                 { d_val = d_sum / (double)d_init_cnt; }
            } else {
                d_ema = stc_neo_ema_step(d_ema, d_a, stok);
                d_val = d_ema;
            }
        } else if (all_finite) {
            d_val = NEO_F64_NAN;
        } else {
            if      (d_init_cnt == 0) d_val = NEO_F64_NAN;
            else if (d_init_cnt <  d) d_val = d_sum / (double)d_init_cnt;
            else                      d_val = d_ema;
        }

        double kd;
        if (all_finite) {
            const bool d_is_valid = !isnan(d_val);
            if (d_is_valid) {
                d_ring[d_vpos] = d_val;
                d_vpos += 1; if (d_vpos == kk) d_vpos = 0;
                if (d_count < kk) d_count += 1;
            }
            if (!d_is_valid)         kd = NEO_F64_NAN;
            else if (d_count == kk)  kd = stc_neo_stoch(d_ring, kk, d_val);
            else                     kd = 50.0;
        } else {
            if (i >= kk) d_valid_sum -= (int)d_valid_ring[d_vpos];
            const unsigned char dv = isnan(d_val) ? 0 : 1;
            d_valid_ring[d_vpos] = dv;
            d_valid_sum += (int)dv;
            if (dv) d_ring[d_vpos] = d_val;
            d_vpos += 1; if (d_vpos == kk) d_vpos = 0;

            if (d_valid_sum == kk && dv) kd = stc_neo_stoch(d_ring, kk, d_val);
            else if (dv)                 kd = 50.0;
            else                         kd = NEO_F64_NAN;
        }

        /* ---- final EMA. IDENTICAL in both paths (:742-762, :967-988). --- */
        double dst;
        if (!isnan(kd)) {
            if (final_init_cnt < d) {
                final_sum += kd;
                final_init_cnt += 1;
                if (final_init_cnt == d) { final_ema = final_sum * d_inv; dst = final_ema; }
                else                     { dst = final_sum / (double)final_init_cnt; }
            } else {
                final_ema = stc_neo_ema_step(final_ema, d_a, kd);
                dst = final_ema;
            }
        } else if (final_init_cnt == 0) {
            dst = NEO_F64_NAN;
        } else if (final_init_cnt < d) {
            dst = final_sum / (double)final_init_cnt;
        } else {
            dst = final_ema;
        }

        o[first + i] = dst;
    }
}
