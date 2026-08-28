#include <cuda_runtime.h>
#include <math.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef LRA_NAN_F
#define LRA_NAN_F (__int_as_float(0x7fffffff))
#endif


static __device__ __forceinline__ int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

static __device__ __forceinline__ float kRad2Deg() {
    return 57.2957795130823208767981548141051703f;
}


struct df32 {
    float hi;
    float lo;
};

static __device__ __forceinline__ df32 df32_make(float x) {
    df32 r; r.hi = x; r.lo = 0.0f; return r;
}


static __device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


static __device__ __forceinline__ df32 df32_add_f(df32 a, float b) {
    float s, e; two_sum(a.hi, b, s, e);
    e += a.lo;
    float s2, e2; two_sum(s, e, s2, e2);
    return {s2, e2};
}


static __device__ __forceinline__ df32 df32_sub_f(df32 a, float b) {
    return df32_add_f(a, -b);
}


static __device__ __forceinline__ df32 df32_add(df32 a, df32 b) {
    float s, e; two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    float s2, e2; two_sum(s, e, s2, e2);
    return {s2, e2};
}


static __device__ __forceinline__ df32 df32_sub(df32 a, df32 b) {
    return df32_add(a, { -b.hi, -b.lo });
}


static __device__ __forceinline__ df32 df32_add_prod(df32 acc, float a, float b) {
    float p = a * b;
    float err = fmaf(a, b, -p);
    acc = df32_add_f(acc, p);
    acc = df32_add_f(acc, err);
    return acc;
}


static __device__ __forceinline__ df32 df32_sub_prod(df32 acc, float a, float b) {
    float p = a * b;
    float err = fmaf(a, b, -p);
    acc = df32_sub_f(acc, p);
    acc = df32_sub_f(acc, err);
    return acc;
}


static __device__ __forceinline__ df32 df32_mul_scalar(df32 a, float s) {
    float p = a.hi * s;
    float err = fmaf(a.hi, s, -p);
    err += a.lo * s;
    float s2, e2; two_sum(p, err, s2, e2);
    return {s2, e2};
}


static __device__ __forceinline__ df32 df32_from_float2(const float2 v) {
    df32 r; r.hi = v.x; r.lo = v.y; return r;
}
static __device__ __forceinline__ float2 float2_from_df32(const df32 v) {
    return make_float2(v.hi, v.lo);
}


static __device__ __forceinline__ float df32_to_float(df32 a) {
    return a.hi + a.lo;
}

extern "C" __global__ void linearreg_angle_build_prefixes_f32(
    const float* __restrict__ prices,
    int len,
    float2* __restrict__ prefix_sum2,
    float2* __restrict__ prefix_kd2,
    int* __restrict__ prefix_nan)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_sum2[0] = make_float2(0.0f, 0.0f);
    prefix_kd2[0] = make_float2(0.0f, 0.0f);
    prefix_nan[0] = 0;

    df32 sum = df32_make(0.0f);
    df32 kd = df32_make(0.0f);
    int nan_count = 0;

    for (int t = 0; t < len; ++t) {
        const float v = prices[t];
        if (isnan(v)) {
            ++nan_count;
        } else {
            sum = df32_add_f(sum, v);
            kd = df32_add_prod(kd, static_cast<float>(t), v);
        }
        prefix_sum2[t + 1] = float2_from_df32(sum);
        prefix_kd2[t + 1] = float2_from_df32(kd);
        prefix_nan[t + 1] = nan_count;
    }
}


extern "C" __global__ void linearreg_angle_batch_f32(
    const float*   __restrict__ prices,
    const float2*  __restrict__ prefix_sum2,
    const float2*  __restrict__ prefix_kd2,
    const int*     __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*     __restrict__ periods,
    const float*   __restrict__ sum_x,
    const float*   __restrict__ inv_div,
    int n_combos,
    float*         __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period < 2 || period > len) return;

    const int warm = first_valid + period - 1;
    const float sx_f   = sum_x[combo];
    const float invd_f = inv_div[combo];
    const float rad2deg = kRad2Deg();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int row_off = combo * len;

    while (t < len) {
        float outv = LRA_NAN_F;
        if (t >= warm) {
            const int start = t + 1 - period;
            const int nan_cnt = prefix_nan[t + 1] - prefix_nan[start];
            if (nan_cnt == 0) {
                df32 sum_y  = df32_sub(df32_from_float2(prefix_sum2[t + 1]),
                                       df32_from_float2(prefix_sum2[start]));
                df32 sum_kd = df32_sub(df32_from_float2(prefix_kd2[t + 1]),
                                       df32_from_float2(prefix_kd2[start]));


                df32 sum_xy = df32_sub(df32_mul_scalar(sum_y, (float)t), sum_kd);

                df32 num = df32_sub(df32_mul_scalar(sum_xy, (float)period),
                                    df32_mul_scalar(sum_y, sx_f));
                const float slope = df32_to_float(num) * invd_f;
                outv = atanf(slope) * rad2deg;
            }
        }
        out[row_off + t] = outv;
        t += stride;
    }
}


extern "C" __global__ void linearreg_angle_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float sum_x_f,
    float inv_div_f,
    float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;
    const float p_f = (float)period;
    const float sx_f = sum_x_f;
    const float invd_f = inv_div_f;
    const float rad2deg = kRad2Deg();

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride) {
        if (period < 2 || period > rows) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }
        const int fv = first_valids[s];
        if (fv < 0 || fv >= rows) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }
        const int tail = rows - fv;
        if (tail < period) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }

        const int warm = fv + period - 1;
        for (int r = 0; r < warm; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;


        df32 y_sum = df32_make(0.0f);
        df32 sum_kd = df32_make(0.0f);
        int nan_count = 0;

        for (int k = 0; k < period; ++k) {
            const int r0 = warm - period + 1 + k;
            const float v = prices_tm[tm_idx(r0, cols, s)];
            if (isnan(v)) {
                nan_count++;
            } else {
                y_sum  = df32_add_f(y_sum, v);
                sum_kd = df32_add_prod(sum_kd, (float)r0, v);
            }
        }


        {
            float outv = LRA_NAN_F;
            if (nan_count == 0) {
                df32 sum_xy = df32_sub(df32_mul_scalar(y_sum, (float)warm), sum_kd);
                df32 num = df32_sub(df32_mul_scalar(sum_xy, p_f),
                                    df32_mul_scalar(y_sum, sx_f));
                const float slope = df32_to_float(num) * invd_f;
                outv = atanf(slope) * rad2deg;
            }
            out_tm[tm_idx(warm, cols, s)] = outv;


            if (nan_count == 0) {
                const int leave0_idx = warm - period + 1;
                const float leave0 = prices_tm[tm_idx(leave0_idx, cols, s)];
                y_sum  = df32_sub_f(y_sum, leave0);
                sum_kd = df32_sub_prod(sum_kd, (float)leave0_idx, leave0);
            }
        }


        float next_enter = (warm + 1 < rows) ? prices_tm[tm_idx(warm + 1, cols, s)] : LRA_NAN_F;


        for (int r = warm + 1; r < rows; ++r) {
            const float enter = next_enter;
            if (r + 1 < rows) next_enter = prices_tm[tm_idx(r + 1, cols, s)];
            const float leave = prices_tm[tm_idx(r - period + 1, cols, s)];

            const bool enter_nan = isnan(enter);
            const bool leave_nan = isnan(leave);
            const int prev_nan_count = nan_count;
            if (enter_nan) nan_count++;
            if (leave_nan) nan_count--;

            float outv = LRA_NAN_F;

            if (nan_count == 0) {
                if (prev_nan_count == 0) {

                    y_sum  = df32_add_f(y_sum, enter);
                    sum_kd = df32_add_prod(sum_kd, (float)r, enter);

                    df32 sum_xy = df32_sub(df32_mul_scalar(y_sum, (float)r), sum_kd);
                    df32 num = df32_sub(df32_mul_scalar(sum_xy, p_f),
                                        df32_mul_scalar(y_sum, sx_f));
                    const double slope_d = (double)df32_to_float(num) * (double)invd_f;
                    outv = atanf((float)slope_d) * rad2deg;


                    y_sum  = df32_sub_f(y_sum, leave);
                    sum_kd = df32_sub_prod(sum_kd, (float)(r - period + 1), leave);
                } else {

                    y_sum  = df32_make(0.0f);
                    sum_kd = df32_make(0.0f);
                    for (int k = 0; k < period; ++k) {
                        const int r0 = r - period + 1 + k;
                        const float v = prices_tm[tm_idx(r0, cols, s)];

                        y_sum  = df32_add_f(y_sum, v);
                        sum_kd = df32_add_prod(sum_kd, (float)r0, v);
                    }
                    df32 sum_xy = df32_sub(df32_mul_scalar(y_sum, (float)r), sum_kd);
                    df32 num = df32_sub(df32_mul_scalar(sum_xy, p_f),
                                        df32_mul_scalar(y_sum, sx_f));
                    const double slope_d = (double)df32_to_float(num) * (double)invd_f;
                    outv = atanf((float)slope_d) * rad2deg;


                    y_sum  = df32_sub_f(y_sum, leave);
                    sum_kd = df32_sub_prod(sum_kd, (float)(r - period + 1), leave);
                }
            } else {

                outv = LRA_NAN_F;
            }

            out_tm[tm_idx(r, cols, s)] = outv;
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — linearreg_angle
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/linearreg_angle.rs:287 `linearreg_angle_scalar`.
 *
 * THE 4-WIDE SEED IS LOAD-BEARING. The window sums are accumulated in groups
 * of four (linearreg_angle.rs:310-321):
 *      sum_y  += y0 + y1 + y2 + y3;
 *      sum_kd += jf*y0 + (jf+1)*y1 + (jf+2)*y2 + (jf+3)*y3;
 * with a scalar tail. That association is a DIFFERENT number from four
 * separate `+=` steps, and it is the reference. Reproduced exactly, including
 * the tail. (This is the same class of thing that makes `wilders` and `smma`
 * disagree about their seeds elsewhere in this crate.)
 *
 * `atanf` x4 in the f32 kernel above is the single-precision arctangent —
 * about 2 decimal digits short of what an angle threshold needs. `atan` here.
 *
 * The CPU has two otherwise-identical loops selected by whether ANY value from
 * `first` onward is NaN; the NaN loop rebuilds the window sums from scratch
 * whenever a NaN enters or leaves. Both are reproduced, selected by the same
 * predicate, because the rebuild changes the association of the sums.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void linearreg_angle_neo_batch_f64(const double* __restrict__ data,
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

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len) return;

    const double p = (double)period;
    // The CPU forms these in usize and casts once, so the products are exact
    // integers before they reach f64. Computed the same way here (the operands
    // are far below 2^53 for any real period).
    const double sum_x     = (double)((long long)period * (long long)(period - 1)) * 0.5;
    const double sum_x_sqr = (double)((long long)period * (long long)(period - 1) *
                                      (long long)(2 * period - 1)) / 6.0;
    const double divisor = sum_x * sum_x - p * sum_x_sqr;
    const double inv_div = 1.0 / divisor;
    const double rad2deg = 180.0 / 3.14159265358979323846;

    int i = first_valid + period - 1;
    if (i >= len) return;

    int start = i + 1 - period;
    double sum_y = 0.0, sum_kd = 0.0;

    bool has_nan = false;
    for (int t = first_valid; t < len; ++t) { if (isnan(data[t])) { has_nan = true; break; } }

    {   // 4-wide seed, then scalar tail — linearreg_angle.rs:308-327
        int j = start;
        const int end = i + 1;
        while (j + 3 < end) {
            const double y0 = data[j], y1 = data[j + 1], y2 = data[j + 2], y3 = data[j + 3];
            sum_y += y0 + y1 + y2 + y3;
            const double jf = (double)j;
            sum_kd += jf * y0 + (jf + 1.0) * y1 + (jf + 2.0) * y2 + (jf + 3.0) * y3;
            j += 4;
        }
        while (j < end) {
            const double y = data[j];
            sum_y  += y;
            sum_kd += (double)j * y;
            j += 1;
        }
    }

    for (;;) {
        const double i_f = (double)i;
        const double sum_xy = i_f * sum_y - sum_kd;
        const double num = fma(p, sum_xy, -sum_x * sum_y);     // p.mul_add(sum_xy, -sum_x*sum_y)
        const double slope = num * inv_div;
        o[i] = atan(slope) * rad2deg;

        i += 1;
        if (i >= len) break;

        const double enter = data[i];
        const double leave = data[start];
        start += 1;

        if (has_nan && (isnan(enter) || isnan(leave))) {
            sum_y = 0.0; sum_kd = 0.0;
            const int ws = i + 1 - period;
            int jj = ws;
            const int ee = i + 1;
            while (jj + 3 < ee) {
                const double y0 = data[jj], y1 = data[jj + 1], y2 = data[jj + 2], y3 = data[jj + 3];
                sum_y += y0 + y1 + y2 + y3;
                const double jf = (double)jj;
                sum_kd += jf * y0 + (jf + 1.0) * y1 + (jf + 2.0) * y2 + (jf + 3.0) * y3;
                jj += 4;
            }
            while (jj < ee) {
                const double y = data[jj];
                sum_y  += y;
                sum_kd += (double)jj * y;
                jj += 1;
            }
        } else {
            sum_y  += enter - leave;
            sum_kd += (double)i * enter - (double)(i - period) * leave;
        }
    }
}
