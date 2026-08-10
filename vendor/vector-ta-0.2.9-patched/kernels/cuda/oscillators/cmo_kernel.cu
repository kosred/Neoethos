#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CMO_NAN
#define CMO_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


#ifndef CMO_BLOCK_SIZE
#define CMO_BLOCK_SIZE 256
#endif
#ifndef CMO_TILE
#define CMO_TILE 256
#endif


struct KBN32 {
    float s;
    float c;
    __device__ inline void init() { s = 0.0f; c = 0.0f; }
    __device__ inline void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ inline float result() const { return s + c; }
};


__device__ inline float cmo_from_avgs(float avg_g, float avg_l) {
    float denom = avg_g + avg_l;
    if (denom == 0.0f) return 0.0f;
    float numer = avg_g - avg_l;
    return 100.0f * (numer / denom);
}


extern "C" __global__ void cmo_batch_f32(
    const float*  __restrict__ prices,
    const int*    __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * (size_t)series_len;


    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = CMO_NAN;
        return;
    }
    const int fv   = first_valid;
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= period)) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = CMO_NAN;
        return;
    }

    const int warm = fv + period;


    for (int i = (int)lane; i < warm; i += 32) out_row[i] = CMO_NAN;


    const float beta  = 1.0f / (float)period;
    const float alpha = 1.0f - beta;


    float avg_g = 0.0f;
    float avg_l = 0.0f;
    if (lane == 0) {
        float prev = prices[fv];
        KBN32 sum_g, sum_l;
        sum_g.init();
        sum_l.init();
        for (int i = fv + 1; i <= warm; ++i) {
            float curr = prices[i];
            float diff = curr - prev;
            prev = curr;
            float g = fmaxf(diff, 0.0f);
            float l = fmaxf(-diff, 0.0f);
            sum_g.add(g);
            sum_l.add(l);
        }
        avg_g = sum_g.result() * beta;
        avg_l = sum_l.result() * beta;
        out_row[warm] = cmo_from_avgs(avg_g, avg_l);
    }

    const unsigned mask = 0xffffffffu;
    avg_g = __shfl_sync(mask, avg_g, 0);
    avg_l = __shfl_sync(mask, avg_l, 0);


    for (int t0 = warm + 1; t0 < series_len; t0 += 32) {
        const int t = t0 + (int)lane;

        float A  = 1.0f;
        float Bg = 0.0f;
        float Bl = 0.0f;
        if (t < series_len) {
            const float p1 = prices[t];
            const float p0 = prices[t - 1];
            const float diff = p1 - p0;
            const float g = fmaxf(diff, 0.0f);
            const float l = fmaxf(-diff, 0.0f);
            A  = alpha;
            Bg = beta * g;
            Bl = beta * l;
        }


        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev  = __shfl_up_sync(mask, A, offset);
            const float Bg_prev = __shfl_up_sync(mask, Bg, offset);
            const float Bl_prev = __shfl_up_sync(mask, Bl, offset);
            if (lane >= (unsigned)offset) {
                const float A_cur  = A;
                const float Bg_cur = Bg;
                const float Bl_cur = Bl;
                A  = A_cur * A_prev;
                Bg = __fmaf_rn(A_cur, Bg_prev, Bg_cur);
                Bl = __fmaf_rn(A_cur, Bl_prev, Bl_cur);
            }
        }

        const float yg = __fmaf_rn(A, avg_g, Bg);
        const float yl = __fmaf_rn(A, avg_l, Bl);

        if (t < series_len) {
            out_row[t] = cmo_from_avgs(yg, yl);
        }


        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        avg_g = __shfl_sync(mask, yg, last_lane);
        avg_l = __shfl_sync(mask, yl, last_lane);
    }
}


extern "C" __global__ void cmo_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(period <= 0 || period > series_len || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= period)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }

    const int warm = fv + period;
    const float beta  = 1.0f / (float)period;
    const float alpha = 1.0f - beta;


    {
        float* o = out_tm + series;
        for (int r = 0; r < warm; ++r, o += num_series) *o = CMO_NAN;
    }


    float prev = *(prices_tm + (size_t)fv * num_series + series);
    KBN32 sum_g, sum_l; sum_g.init(); sum_l.init();

    for (int r = fv + 1; r <= warm; ++r) {
        float curr = *(prices_tm + (size_t)r * num_series + series);
        float diff = curr - prev; prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        sum_g.add(g);
        sum_l.add(l);
    }
    float avg_g = sum_g.result() * beta;
    float avg_l = sum_l.result() * beta;

    *(out_tm + (size_t)warm * num_series + series) = cmo_from_avgs(avg_g, avg_l);


    for (int r = warm + 1; r < series_len; ++r) {
        float curr = *(prices_tm + (size_t)r * num_series + series);
        float diff = curr - prev; prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        avg_g = __fmaf_rn(alpha, avg_g, beta * g);
        avg_l = __fmaf_rn(alpha, avg_l, beta * l);
        *(out_tm + (size_t)r * num_series + series) = cmo_from_avgs(avg_g, avg_l);
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — cmo
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/cmo.rs:298 `cmo_scalar`.
 *
 * ROUNDING COUNT — the f32 kernel above is wrong here, not merely imprecise.
 * It carries `__fmaf_rn(alpha, avg_g, beta * g)`: ONE rounding for the whole
 * update. The CPU carries FOUR separate operations (cmo.rs:333-338):
 *     avg_gain *= period_m1;   avg_gain += gain;   avg_gain *= inv_period;
 * i.e. three roundings and a different value. Reproduced literally below.
 *
 * gain/loss are NOT branches on the CPU: `0.5*(diff+|diff|)` and
 * `0.5*(|diff|-diff)` (cmo.rs:316-317). A branch would agree except at -0.0
 * and at NaN, where the branch keeps the NaN and the arithmetic propagates it
 * the way the CPU does.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void cmo_neo_batch_f64(const double* __restrict__ prices,
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
    const int period = periods[combo];

    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len ||
        (len - first_valid) < period) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    // The CPU writes out[i] only from i == init_end = first + period onward.
    const int init_end = first_valid + period;
    for (int i = 0; i < init_end && i < len; ++i) o[i] = NEO_F64_NAN;

    double avg_gain = 0.0;
    double avg_loss = 0.0;
    double prev_price = prices[first_valid];

    const double period_m1  = (double)(period - 1);
    const double inv_period = 1.0 / (double)period;

    for (int i = first_valid + 1; i < len; ++i) {
        const double curr = prices[i];
        const double diff = curr - prev_price;
        prev_price = curr;

        const double abs_diff = fabs(diff);
        const double gain = 0.5 * (diff + abs_diff);
        const double loss = 0.5 * (abs_diff - diff);

        if (i <= init_end) {
            avg_gain += gain;
            avg_loss += loss;
            if (i == init_end) {
                avg_gain *= inv_period;
                avg_loss *= inv_period;
                const double sum_gl = avg_gain + avg_loss;
                o[i] = (sum_gl != 0.0) ? 100.0 * ((avg_gain - avg_loss) / sum_gl) : 0.0;
            }
        } else {
            avg_gain *= period_m1;      // three roundings, exactly as cmo.rs
            avg_loss *= period_m1;
            avg_gain += gain;
            avg_loss += loss;
            avg_gain *= inv_period;
            avg_loss *= inv_period;
            const double sum_gl = avg_gain + avg_loss;
            o[i] = (sum_gl != 0.0) ? 100.0 * ((avg_gain - avg_loss) / sum_gl) : 0.0;
        }
    }
}
