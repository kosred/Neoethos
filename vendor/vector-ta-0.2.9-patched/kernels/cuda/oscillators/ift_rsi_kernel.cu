#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __device__ __forceinline__ float f32_qnan() { return __int_as_float(0x7fffffff); }
static __device__ __forceinline__ int   imax(int a,int b){ return a>b? a:b; }
static __device__ __forceinline__ int   imin(int a,int b){ return a<b? a:b; }


struct KahanF32 {
    float sum;
    float c;
    __device__ __forceinline__ void init(float s = 0.f){ sum=s; c=0.f; }
    __device__ __forceinline__ void add(float x){
        float y = x - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};


static __device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = __activemask();

    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v,  8);
    v += __shfl_down_sync(mask, v,  4);
    v += __shfl_down_sync(mask, v,  2);
    v += __shfl_down_sync(mask, v,  1);
    return v;
}


extern "C" __global__ void ift_rsi_batch_f32(
    const float* __restrict__ data,
    int series_len,
    int n_combos,
    int first_valid,
    const int* __restrict__ rsi_periods,
    const int* __restrict__ wma_periods,
    float* __restrict__ out_values)
{

    for (int combo = blockIdx.x; combo < n_combos; combo += gridDim.x) {

        const int rp = rsi_periods[combo];
        const int wp = wma_periods[combo];
        const int base = combo * series_len;


        if (UNLIKELY(rp <= 0 || wp <= 0 || rp > series_len || wp > series_len)) {
            for (int t = threadIdx.x; t < series_len; t += blockDim.x) out_values[base + t] = f32_qnan();
            continue;
        }
        if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
            for (int t = threadIdx.x; t < series_len; t += blockDim.x) out_values[base + t] = f32_qnan();
            continue;
        }

        const int tail = series_len - first_valid;
        const int need = imax(rp, wp);
        if (UNLIKELY(tail < need)) {
            for (int t = threadIdx.x; t < series_len; t += blockDim.x) out_values[base + t] = f32_qnan();
            continue;
        }

        const int warm = first_valid + rp + wp - 1;
        for (int t = threadIdx.x; t < imin(warm, series_len); t += blockDim.x) out_values[base + t] = f32_qnan();


        extern __shared__ float shmem[];
        float* ring = shmem;

        if (UNLIKELY(wp <= 0)) continue;


        const int lane = threadIdx.x & 31;
        const int seed_start = first_valid + 1;
        const int seed_end   = seed_start + rp - 1;

        float gain_seed = 0.f, loss_seed = 0.f;

        if (blockDim.x >= 32) {

            float gain_part = 0.f, loss_part = 0.f;
            for (int i = seed_start + lane; i <= seed_end; i += 32) {
                float cur  = data[i];
                float prev = data[i - 1];
                float d = cur - prev;
                if (d > 0.f) gain_part += d; else loss_part += -d;
            }

            gain_seed = warp_sum(gain_part);
            loss_seed = warp_sum(loss_part);
        } else {

            if (threadIdx.x == 0) {
                float g = 0.f, l = 0.f;
                for (int i = seed_start; i <= seed_end; ++i) {
                    float d = data[i] - data[i - 1];
                    if (d > 0.f) g += d; else l += -d;
                }
                gain_seed = g; loss_seed = l;
            }
        }


        if (lane == 0) {
            const float rp_rcp = 1.0f / (float)rp;
            float avg_gain = gain_seed * rp_rcp;
            float avg_loss = loss_seed * rp_rcp;
            const float alpha = rp_rcp;
            const float beta  = 1.0f - alpha;


            const float wp_f = (float)wp;
            const float denom_rcp = 2.0f / (wp_f * (wp_f + 1.0f));
            int head = 0, filled = 0;
            float S1 = 0.0f;
            float S2 = 0.0f;


            float prev = data[first_valid + rp];


            for (int i = rp; i < tail; ++i) {
                if (i > rp) {
                    const int abs_idx = first_valid + i;
                    float curr = data[abs_idx];
                    float d = curr - prev;
                    prev = curr;
                    float g = (d > 0.f) ? d : 0.f;
                    float l = (d > 0.f) ? 0.f : -d;

                    avg_gain = __fmaf_rn(alpha, g, beta * avg_gain);
                    avg_loss = __fmaf_rn(alpha, l, beta * avg_loss);
                }


                float rs  = (avg_loss != 0.f) ? (avg_gain / avg_loss) : 100.f;
                float rsi = 100.f - 100.f / (1.f + rs);
                float x   = 0.1f * (rsi - 50.f);

                if (filled < wp) {
                    S1 += x;
                    S2 += (float)(filled + 1) * x;
                    ring[head] = x;
                    head = (head + 1 == wp) ? 0 : head + 1;
                    filled += 1;
                    if (filled == wp) {
                        float wma = S2 * denom_rcp;
                        const int abs_t = first_valid + i;
                        out_values[base + abs_t] = tanhf(wma);
                    }
                } else {
                    float x_old = ring[head];
                    ring[head]  = x;
                    head = (head + 1 == wp) ? 0 : head + 1;

                    float S1_prev = S1;
                    S1 = (S1 + x) - x_old;
                    S2 = (S2 - S1_prev) + (wp_f * x);

                    float wma = S2 * denom_rcp;
                    const int abs_t = first_valid + i;
                    out_values[base + abs_t] = tanhf(wma);
                }
            }
        }
    }
}


extern "C" __global__ void ift_rsi_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int rsi_period,
    int wma_period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int rp = rsi_period;
    const int wp = wma_period;

    if (UNLIKELY(rp <= 0 || wp <= 0 || rp > series_len || wp > series_len)) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }
    int first = first_valids ? first_valids[series] : 0;
    if (first < 0) first = 0;
    if (UNLIKELY(first >= series_len)) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }
    const int tail = series_len - first;
    if (UNLIKELY(tail < imax(rp, wp))) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }

    const int warm = first + rp + wp - 1;
    for (int r = 0; r < imin(warm, series_len); ++r) out_tm[r * num_series + series] = f32_qnan();


    float gain_part = 0.f, loss_part = 0.f;
    const int seed_start = first + 1;
    const int seed_end   = seed_start + rp - 1;
    for (int i = seed_start; i <= seed_end; ++i) {
        const float cur  = data_tm[i * num_series + series];
        const float prev = data_tm[(i - 1) * num_series + series];
        const float d = cur - prev;
        if (d > 0.f) gain_part += d; else loss_part += -d;
    }
    const float rp_rcp = 1.0f / (float)rp;
    float avg_gain = gain_part * rp_rcp;
    float avg_loss = loss_part * rp_rcp;
    const float alpha = rp_rcp;
    const float beta  = 1.0f - alpha;


    const float wp_f = (float)wp;
    const float denom_rcp = 2.0f / (wp_f * (wp_f + 1.0f));
    int head = 0, filled = 0;
    KahanF32 S1; S1.init(0.f);
    KahanF32 S2; S2.init(0.f);

    extern __shared__ float shbuf[];
    float* ring = shbuf + threadIdx.x * wp;


    float prev = data_tm[(first + rp) * num_series + series];

    for (int r = first + rp; r < series_len; ++r) {
        if (r > first + rp) {
            const float curr = data_tm[r * num_series + series];
            const float d = curr - prev;
            prev = curr;
            const float g = (d > 0.f) ? d : 0.f;
            const float l = (d > 0.f) ? 0.f : -d;
            avg_gain = __fmaf_rn(alpha, g, beta * avg_gain);
            avg_loss = __fmaf_rn(alpha, l, beta * avg_loss);
        }

        const float rs  = (avg_loss != 0.f) ? (avg_gain / avg_loss) : 100.f;
        const float rsi = 100.f - 100.f / (1.f + rs);
        const float x   = 0.1f * (rsi - 50.f);

        if (filled < wp) {
            S1.add(x);
            S2.add((float)(filled + 1) * x);
            ring[head] = x;
            head = (head + 1 == wp) ? 0 : head + 1;
            filled += 1;
            if (filled == wp) {
                const float wma = S2.sum * denom_rcp;
                out_tm[r * num_series + series] = tanhf(wma);
            }
        } else {
            const float x_old = ring[head];
            ring[head] = x;
            head = (head + 1 == wp) ? 0 : head + 1;

            const float S1_prev = S1.sum;
            S1.add(x);
            S1.add(-x_old);

            S2.add(-S1_prev);
            S2.add(wp_f * x);

            const float wma = S2.sum * denom_rcp;
            out_tm[r * num_series + series] = tanhf(wma);
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — ift_rsi (inverse Fisher transform of RSI)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ift_rsi.rs
 *   `ift_rsi_with_kernel`        (:233)  — first_valid, warmup, and the
 *                                          default-params branch
 *   `is_default_ift_rsi_params`  (:1298) — (5, 9) takes the specialised path
 *   `ift_rsi_scalar_default_5_9` (:1303) — THE path this kernel mirrors
 *
 * PERIOD-INVARIANT, AND THAT IS FAITHFUL. `compute_ift_rsi_batch`
 * (cpu_batch.rs:3142) reads `rsi_period` (5) and `wma_period` (9); it never
 * reads `period`. A period sweep therefore produces identical CPU columns and
 * identical rows here. Declared through `is_period_invariant`.
 *
 * THE SPECIALISED PATH IS NOT THE GENERIC PATH WITH CONSTANTS FOLDED IN, AND
 * THIS IS THE SUBTLE PART. `ift_rsi_scalar_classic` seeds with
 * `avg_gain /= rp_f` — a DIVISION by 5.0. `ift_rsi_scalar_default_5_9` seeds
 * with `avg_gain *= ALPHA` where `ALPHA = 0.2` — a MULTIPLICATION by the
 * nearest double to 1/5, which is not exactly 1/5. The two seeds differ in the
 * last place, that difference enters a Wilder recursion, and the recursion
 * carries it for the rest of the series. Since the batch defaults ARE (5, 9),
 * the specialised path is the reference and `* 0.2` is written below. Using
 * `/ 5.0` "because it is cleaner" would be a wrong kernel that looks right.
 *
 * WHAT THE f32 KERNELS ABOVE GET WRONG, AND IS FIXED HERE
 *
 *  1. `tanhf` x4 -> `tanh`. The inverse Fisher transform saturates: for
 *     |wma| > ~4 the f32 tanh returns exactly 1.0f and the indicator becomes a
 *     flat line, losing the ordering information the search actually uses.
 *  2. `__fmaf_rn` x4 -> `fma`, matching `f64::mul_add` one-for-one. The two
 *     Wilder steps and both WMA updates are single-rounding on the CPU.
 *  3. `__int_as_float(0x7f...)` -> `__longlong_as_double(0x7ff8...)`.
 *  4. `0.1f`, `50.0f`, `100.0f`, `0.2f` — every literal is the f64 form here.
 *     `0.1f` and `0.1` are DIFFERENT NUMBERS, and this one multiplies a
 *     quantity in [-50, 50] before a tanh.
 *
 * THE WMA IS A TWO-ACCUMULATOR ROLLING WEIGHTED SUM, NOT A DOT PRODUCT.
 * `num` holds the weighted sum and `sum` the plain sum; the update is
 * `num = fma(9.0, x, num) - sum_old` and then `sum = sum_old + x - x_old`, in
 * that order, reading the OLD sum for the num update. Reversing the two lines
 * gives a plausible series that is wrong at every bar after the ninth.
 *
 * ONE THREAD PER COLUMN. Carried: avg_gain, avg_loss, num, sum, a 9-slot ring.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_IFTRSI_RP 5
#define NEO_IFTRSI_WP 9

extern "C" __global__
void ift_rsi_neo_batch_f64(const double* __restrict__ data,
                           int series_len,
                           const int* __restrict__ periods,
                           int n_combos,
                           int first_valid,
                           double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;   /* period-invariant — see the header. */

    const int RP = NEO_IFTRSI_RP;
    const int WP = NEO_IFTRSI_WP;
    const double ALPHA = 0.2;
    const double BETA = 1.0 - ALPHA;
    const double DENOM_RCP = 1.0 / 45.0;   /* 0.5 * 9 * 10 is exactly 45. */
    const double WP_F = 9.0;

    /* Every Err branch of `ift_rsi_with_kernel` (:249-262) plus the early
     * return at :1314 — the CPU emits an all-NaN series in each case. */
    if (len <= 0 || first_valid < 0 || first_valid >= len ||
        RP > len || WP > len ||
        (len - first_valid) < (RP > WP ? RP : WP)) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int n = len - first_valid;
    const int warmup = first_valid + RP + WP - 1;   /* == first_valid + 13 */
    for (int i = 0; i < len && i < warmup; ++i) o[i] = NEO_F64_NAN;
    if ((RP + WP - 1) >= n) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    double avg_gain = 0.0;
    double avg_loss = 0.0;
    for (int s = 1; s <= RP; ++s) {
        const double d = data[first_valid + s] - data[first_valid + s - 1];
        if (d > 0.0) avg_gain += d; else avg_loss -= d;
    }
    /* :1337-1338 — MULTIPLY by 0.2, do not divide by 5. See the header. */
    avg_gain *= ALPHA;
    avg_loss *= ALPHA;

    double buf[NEO_IFTRSI_WP];
    for (int k = 0; k < WP; ++k) buf[k] = 0.0;
    int head = 0;
    int filled = 0;
    double sum = 0.0;
    double num = 0.0;

    for (int i = RP; i < n; ++i) {
        if (i > RP) {
            const double d = data[first_valid + i] - data[first_valid + i - 1];
            const double gain = (d > 0.0) ? d : 0.0;
            const double loss = (d < 0.0) ? -d : 0.0;
            avg_gain = fma(avg_gain, BETA, ALPHA * gain);
            avg_loss = fma(avg_loss, BETA, ALPHA * loss);
        }

        const double rs = (avg_loss != 0.0) ? (avg_gain / avg_loss) : 100.0;
        const double rsi = 100.0 - 100.0 / (1.0 + rs);
        const double xv = 0.1 * (rsi - 50.0);

        if (filled < WP) {
            sum += xv;
            num = fma((double)filled + 1.0, xv, num);
            buf[head] = xv;
            head += 1; if (head == WP) head = 0;
            filled += 1;

            if (filled == WP) {
                o[first_valid + i] = tanh(num * DENOM_RCP);
            }
        } else {
            const double x_old = buf[head];
            buf[head] = xv;
            head += 1; if (head == WP) head = 0;

            const double sum_t = sum;
            num = fma(WP_F, xv, num) - sum_t;   /* reads the OLD sum */
            sum = sum_t + xv - x_old;

            o[first_valid + i] = tanh(num * DENOM_RCP);
        }
    }
}
