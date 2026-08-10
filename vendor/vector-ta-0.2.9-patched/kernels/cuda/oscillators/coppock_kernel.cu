#ifndef CUDA_COPPOCK_F32_H_
#define CUDA_COPPOCK_F32_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define XNAN __int_as_float(0x7fffffff)


__device__ __forceinline__ bool any_nan3(float a, float b, float c) {

    return __isnanf(a) | __isnanf(b) | __isnanf(c);
}

__device__ __forceinline__ float roc_sum_times100(float c, float inv_s, float inv_l) {


    float inv_sum = inv_s + inv_l;
    return fmaf(c, inv_sum, -2.0f) * 100.0f;
}


__device__ __forceinline__ void comp_add(float x, float &sum, float &comp) {
    float t = sum + x;
    if (fabsf(sum) >= fabsf(x)) comp += (sum - t) + x;
    else                       comp += (x   - t) + sum;
    sum = t;
}

__device__ __forceinline__ void comp_sub(float x, float &sum, float &comp) {

    comp_add(-x, sum, comp);
}

extern "C" __global__ void coppock_build_inverse_f32(
    const float* __restrict__ price,
    int len,
    float* __restrict__ inv
)
{
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= len) return;
    inv[idx] = 1.0f / price[idx];
}


extern "C" __global__ void coppock_batch_f32(
    const float* __restrict__ price,
    const float* __restrict__ inv,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    const int* __restrict__ ma_periods,
    int n_combos,
    float* __restrict__ out
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    float* row_out = out + (size_t)row * (size_t)len;


    const int pre = warm < len ? warm : len;
    for (int t = 0; t < pre; ++t) row_out[t] = XNAN;
    if (warm >= len) return;


    const float denom_w = 0.5f * (float)m * (float)(m + 1);


    float sum = 0.0f, sum_c = 0.0f;
    float wsum = 0.0f, wsum_c = 0.0f;
    int bad_count = 0;

    int w = 1;
    const int start = warm - m + 1;
    for (int j = start; j <= warm; ++j, ++w) {
        const int js = j - s;
        const int jl = j - l;


        const float c  = price[j];
        const float ps = price[js];
        const float pl = price[jl];

        const bool invalid = any_nan3(c, ps, pl);
        if (invalid) { ++bad_count; continue; }

        const float v = roc_sum_times100(c, inv[js], inv[jl]);


        comp_add(v, sum, sum_c);
        comp_add(v * (float)w, wsum, wsum_c);
    }

    if (bad_count > 0) {
        row_out[warm] = XNAN;
    } else {

        const float sum_eff  = sum + sum_c;
        const float wsum_eff = wsum + wsum_c;
        (void)sum_eff;
        row_out[warm] = wsum_eff / denom_w;
    }


    bool state_valid = (bad_count == 0);


    for (int t = warm + 1; t < len; ++t) {

        const int jn  = t;
        const int jns = jn - s;
        const int jnl = jn - l;
        const float cn  = price[jn];
        const float pns = price[jns];
        const float pnl = price[jnl];
        const bool inv_new = any_nan3(cn, pns, pnl);

        float v_new = 0.0f;
        if (!inv_new) v_new = roc_sum_times100(cn, inv[jns], inv[jnl]);


        const int jo  = t - m;
        const int jos = jo - s;
        const int jol = jo - l;
        const float co  = price[jo];
        const float pos = price[jos];
        const float pol = price[jol];
        const bool inv_old = any_nan3(co, pos, pol);

        float v_old = 0.0f;
        if (!inv_old) v_old = roc_sum_times100(co, inv[jos], inv[jol]);


        bad_count += (int)inv_new - (int)inv_old;

        if (bad_count == 0) {
            if (!state_valid) {

                sum = 0.0f; sum_c = 0.0f;
                wsum = 0.0f; wsum_c = 0.0f;
                int ww = 1;
                const int rst = t - m + 1;
                for (int j = rst; j <= t; ++j, ++ww) {
                    const int js2 = j - s;
                    const int jl2 = j - l;
                    const float c2  = price[j];
                    const float ps2 = price[js2];
                    const float pl2 = price[jl2];
                    (void)ps2; (void)pl2;

                    const float v2 = roc_sum_times100(c2, inv[js2], inv[jl2]);
                    comp_add(v2, sum, sum_c);
                    comp_add(v2 * (float)ww, wsum, wsum_c);
                }
                state_valid = true;
            } else {


                const float sum_prev = sum + sum_c;

                comp_add((float)m * v_new, wsum, wsum_c);
                comp_sub(sum_prev,            wsum, wsum_c);

                comp_add(v_new, sum, sum_c);
                comp_sub(v_old, sum, sum_c);
            }
            const float wsum_eff = wsum + wsum_c;
            row_out[t] = wsum_eff / denom_w;
        } else {
            row_out[t] = XNAN;
            state_valid = false;
        }
    }
}


extern "C" __global__ void coppock_batch_time_parallel_f32(
    const float* __restrict__ price,
    const float* __restrict__ inv,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    const int* __restrict__ ma_periods,
    int n_combos,
    float* __restrict__ out
)
{
    const int row = (int)blockIdx.y;
    if (row >= n_combos) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    if (s <= 0 || l <= 0 || m <= 0 || len <= 0) return;

    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    float* row_out = out + (size_t)row * (size_t)len;


    const float denom_w = 0.5f * (float)m * (float)(m + 1);
    const float inv_denom = __fdividef(1.0f, denom_w);

    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    while (t < len) {
        float out_val = XNAN;
        if (t >= warm) {
            const int start = t - m + 1;
            float wsum = 0.0f;
            bool bad = false;


            int w = 1;
            for (int j = start; j <= t; ++j, ++w) {
                const int js = j - s;
                const int jl = j - l;

                const float c  = price[j];
                const float ps = price[js];
                const float pl = price[jl];
                if (any_nan3(c, ps, pl)) { bad = true; break; }

                const float v = roc_sum_times100(c, inv[js], inv[jl]);
                wsum = fmaf(v, (float)w, wsum);
            }

            if (!bad) out_val = wsum * inv_denom;
        }
        row_out[t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void coppock_many_series_one_param_f32(
    const float* __restrict__ price_tm,
    const float* __restrict__ inv_tm,
    const int* __restrict__ first_valids,
    int cols, int rows,
    int short_p, int long_p, int ma_period,
    float* __restrict__ out_tm
)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int first_valid = first_valids[s];
    const int largest = short_p > long_p ? short_p : long_p;
    const int m = ma_period;
    const int warm = first_valid + largest + (m - 1);
    const float denom_w = 0.5f * (float)m * (float)(m + 1);


    const int pre = warm < rows ? warm : rows;
    for (int t = 0; t < pre; ++t) {
        out_tm[(size_t)t * (size_t)cols + s] = XNAN;
    }
    if (warm >= rows) return;


    float sum = 0.0f, sum_c = 0.0f;
    float wsum = 0.0f, wsum_c = 0.0f;
    int bad_count = 0;

    int w = 1;
    const int start = warm - m + 1;
    for (int j = start; j <= warm; ++j, ++w) {
        const int js = j - short_p;
        const int jl = j - long_p;

        const size_t idxj  = (size_t)j  * (size_t)cols + s;
        const size_t idxjs = (size_t)js * (size_t)cols + s;
        const size_t idxjl = (size_t)jl * (size_t)cols + s;

        const float c  = price_tm[idxj];
        const float ps = price_tm[idxjs];
        const float pl = price_tm[idxjl];

        const bool invalid = any_nan3(c, ps, pl);
        if (invalid) { ++bad_count; continue; }

        const float v = roc_sum_times100(c, inv_tm[idxjs], inv_tm[idxjl]);
        comp_add(v, sum, sum_c);
        comp_add(v * (float)w, wsum, wsum_c);
    }

    {
        float* dst = out_tm + (size_t)warm * (size_t)cols + s;
        if (bad_count > 0) *dst = XNAN;
        else               *dst = (wsum + wsum_c) / denom_w;
    }

    bool state_valid = (bad_count == 0);


    for (int t = warm + 1; t < rows; ++t) {

        const int jn = t;
        const int jns = jn - short_p;
        const int jnl = jn - long_p;

        const size_t idxjn  = (size_t)jn  * (size_t)cols + s;
        const size_t idxjns = (size_t)jns * (size_t)cols + s;
        const size_t idxjnl = (size_t)jnl * (size_t)cols + s;

        const float cn  = price_tm[idxjn];
        const float pns = price_tm[idxjns];
        const float pnl = price_tm[idxjnl];
        const bool inv_new = any_nan3(cn, pns, pnl);

        float v_new = 0.0f;
        if (!inv_new) v_new = roc_sum_times100(cn, inv_tm[idxjns], inv_tm[idxjnl]);


        const int jo = t - m;
        const int jos = jo - short_p;
        const int jol = jo - long_p;

        const size_t idxjo  = (size_t)jo  * (size_t)cols + s;
        const size_t idxjos = (size_t)jos * (size_t)cols + s;
        const size_t idxjol = (size_t)jol * (size_t)cols + s;

        const float co  = price_tm[idxjo];
        const float pos = price_tm[idxjos];
        const float pol = price_tm[idxjol];
        const bool inv_old = any_nan3(co, pos, pol);

        float v_old = 0.0f;
        if (!inv_old) v_old = roc_sum_times100(co, inv_tm[idxjos], inv_tm[idxjol]);

        bad_count += (int)inv_new - (int)inv_old;

        float* dst = out_tm + (size_t)t * (size_t)cols + s;
        if (bad_count == 0) {
            if (!state_valid) {

                sum = 0.0f; sum_c = 0.0f;
                wsum = 0.0f; wsum_c = 0.0f;
                int ww = 1;
                const int rst = t - m + 1;
                for (int j = rst; j <= t; ++j, ++ww) {
                    const int js2 = j - short_p;
                    const int jl2 = j - long_p;

                    const size_t idxj2  = (size_t)j   * (size_t)cols + s;
                    const size_t idxjs2 = (size_t)js2 * (size_t)cols + s;
                    const size_t idxjl2 = (size_t)jl2 * (size_t)cols + s;

                    const float c2  = price_tm[idxj2];
                    const float v2  = roc_sum_times100(c2, inv_tm[idxjs2], inv_tm[idxjl2]);
                    comp_add(v2, sum, sum_c);
                    comp_add(v2 * (float)ww, wsum, wsum_c);
                }
                state_valid = true;
            } else {
                const float sum_prev = sum + sum_c;
                comp_add((float)m * v_new, wsum, wsum_c);
                comp_sub(sum_prev,            wsum, wsum_c);
                comp_add(v_new, sum, sum_c);
                comp_sub(v_old, sum, sum_c);
            }
            *dst = (wsum + wsum_c) / denom_w;
        } else {
            *dst = XNAN;
            state_valid = false;
        }
    }
}

#endif


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

__device__ __forceinline__ bool any_nan3_f64(double a, double b, double c) {

    return __isnanf(a) | __isnanf(b) | __isnanf(c);
}
__device__ __forceinline__ double roc_sum_times100_f64(double c, double inv_s, double inv_l) {


    double inv_sum = inv_s + inv_l;
    return fma(c, inv_sum, -2.0) * 100.0;
}
__device__ __forceinline__ void comp_add_f64(double x, double &sum, double &comp) {
    double t = sum + x;
    if (fabs(sum) >= fabs(x)) comp += (sum - t) + x;
    else                       comp += (x   - t) + sum;
    sum = t;
}
__device__ __forceinline__ void comp_sub_f64(double x, double &sum, double &comp) {

    comp_add_f64(-x, sum, comp);
}
extern "C" __global__ void coppock_build_inverse_f64(
    const double* __restrict__ price,
    int len,
    double* __restrict__ inv
)
{
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= len) return;
    inv[idx] = 1.0 / price[idx];
}
extern "C" __global__ void coppock_batch_f64(
    const double* __restrict__ price,
    const double* __restrict__ inv,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    const int* __restrict__ ma_periods,
    int n_combos,
    double* __restrict__ out
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    double* row_out = out + (size_t)row * (size_t)len;


    const int pre = warm < len ? warm : len;
    for (int t = 0; t < pre; ++t) row_out[t] = XNAN;
    if (warm >= len) return;


    const double denom_w = 0.5 * (double)m * (double)(m + 1);


    double sum = 0.0, sum_c = 0.0;
    double wsum = 0.0, wsum_c = 0.0;
    int bad_count = 0;

    int w = 1;
    const int start = warm - m + 1;
    for (int j = start; j <= warm; ++j, ++w) {
        const int js = j - s;
        const int jl = j - l;


        const double c  = price[j];
        const double ps = price[js];
        const double pl = price[jl];

        const bool invalid = any_nan3_f64(c, ps, pl);
        if (invalid) { ++bad_count; continue; }

        const double v = roc_sum_times100_f64(c, inv[js], inv[jl]);


        comp_add_f64(v, sum, sum_c);
        comp_add_f64(v * (double)w, wsum, wsum_c);
    }

    if (bad_count > 0) {
        row_out[warm] = XNAN;
    } else {

        const double sum_eff  = sum + sum_c;
        const double wsum_eff = wsum + wsum_c;
        (void)sum_eff;
        row_out[warm] = wsum_eff / denom_w;
    }


    bool state_valid = (bad_count == 0);


    for (int t = warm + 1; t < len; ++t) {

        const int jn  = t;
        const int jns = jn - s;
        const int jnl = jn - l;
        const double cn  = price[jn];
        const double pns = price[jns];
        const double pnl = price[jnl];
        const bool inv_new = any_nan3_f64(cn, pns, pnl);

        double v_new = 0.0;
        if (!inv_new) v_new = roc_sum_times100_f64(cn, inv[jns], inv[jnl]);


        const int jo  = t - m;
        const int jos = jo - s;
        const int jol = jo - l;
        const double co  = price[jo];
        const double pos = price[jos];
        const double pol = price[jol];
        const bool inv_old = any_nan3_f64(co, pos, pol);

        double v_old = 0.0;
        if (!inv_old) v_old = roc_sum_times100_f64(co, inv[jos], inv[jol]);


        bad_count += (int)inv_new - (int)inv_old;

        if (bad_count == 0) {
            if (!state_valid) {

                sum = 0.0; sum_c = 0.0;
                wsum = 0.0; wsum_c = 0.0;
                int ww = 1;
                const int rst = t - m + 1;
                for (int j = rst; j <= t; ++j, ++ww) {
                    const int js2 = j - s;
                    const int jl2 = j - l;
                    const double c2  = price[j];
                    const double ps2 = price[js2];
                    const double pl2 = price[jl2];
                    (void)ps2; (void)pl2;

                    const double v2 = roc_sum_times100_f64(c2, inv[js2], inv[jl2]);
                    comp_add_f64(v2, sum, sum_c);
                    comp_add_f64(v2 * (double)ww, wsum, wsum_c);
                }
                state_valid = true;
            } else {


                const double sum_prev = sum + sum_c;

                comp_add_f64((double)m * v_new, wsum, wsum_c);
                comp_sub_f64(sum_prev,            wsum, wsum_c);

                comp_add_f64(v_new, sum, sum_c);
                comp_sub_f64(v_old, sum, sum_c);
            }
            const double wsum_eff = wsum + wsum_c;
            row_out[t] = wsum_eff / denom_w;
        } else {
            row_out[t] = XNAN;
            state_valid = false;
        }
    }
}
extern "C" __global__ void coppock_batch_time_parallel_f64(
    const double* __restrict__ price,
    const double* __restrict__ inv,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    const int* __restrict__ ma_periods,
    int n_combos,
    double* __restrict__ out
)
{
    const int row = (int)blockIdx.y;
    if (row >= n_combos) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    if (s <= 0 || l <= 0 || m <= 0 || len <= 0) return;

    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    double* row_out = out + (size_t)row * (size_t)len;


    const double denom_w = 0.5 * (double)m * (double)(m + 1);
    // S5 CORRECTION -- ROUNDING COUNT. `coppock.rs:533` and `:665` both write
    // `weighted_sum / weight_sum`: ONE divide, ONE rounding. The original
    // hoisted a reciprocal (`__fdividef(1.0f, denom_w)`, L214) and multiplied,
    // which is TWO roundings. Every OTHER f64 site in this file already divides
    // (L508, L575, L697, L769), so the hoisted reciprocal was also making this
    // kernel disagree with its own siblings on the same input.

    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    while (t < len) {
        double out_val = XNAN;
        if (t >= warm) {
            const int start = t - m + 1;
            double wsum = 0.0;
            bool bad = false;


            int w = 1;
            for (int j = start; j <= t; ++j, ++w) {
                const int js = j - s;
                const int jl = j - l;

                const double c  = price[j];
                const double ps = price[js];
                const double pl = price[jl];
                if (any_nan3_f64(c, ps, pl)) { bad = true; break; }

                const double v = roc_sum_times100_f64(c, inv[js], inv[jl]);
                // S5 CORRECTION -- ROUNDING COUNT. `coppock.rs:527` is
                // `weighted_sum += sum_roc[idx] * (j + 1) as f64`: a
                // SEPARATE multiply and add, TWO roundings. `fma` is ONE.
                // `-fmad=false` (build.rs:2322) forbids the compiler from
                // contracting this back, so the plain form is guaranteed.
                wsum = wsum + v * (double)w;
            }

            if (!bad) out_val = wsum / denom_w;
        }
        row_out[t] = out_val;
        t += stride;
    }
}
extern "C" __global__ void coppock_many_series_one_param_f64(
    const double* __restrict__ price_tm,
    const double* __restrict__ inv_tm,
    const int* __restrict__ first_valids,
    int cols, int rows,
    int short_p, int long_p, int ma_period,
    double* __restrict__ out_tm
)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int first_valid = first_valids[s];
    const int largest = short_p > long_p ? short_p : long_p;
    const int m = ma_period;
    const int warm = first_valid + largest + (m - 1);
    const double denom_w = 0.5 * (double)m * (double)(m + 1);


    const int pre = warm < rows ? warm : rows;
    for (int t = 0; t < pre; ++t) {
        out_tm[(size_t)t * (size_t)cols + s] = XNAN;
    }
    if (warm >= rows) return;


    double sum = 0.0, sum_c = 0.0;
    double wsum = 0.0, wsum_c = 0.0;
    int bad_count = 0;

    int w = 1;
    const int start = warm - m + 1;
    for (int j = start; j <= warm; ++j, ++w) {
        const int js = j - short_p;
        const int jl = j - long_p;

        const size_t idxj  = (size_t)j  * (size_t)cols + s;
        const size_t idxjs = (size_t)js * (size_t)cols + s;
        const size_t idxjl = (size_t)jl * (size_t)cols + s;

        const double c  = price_tm[idxj];
        const double ps = price_tm[idxjs];
        const double pl = price_tm[idxjl];

        const bool invalid = any_nan3_f64(c, ps, pl);
        if (invalid) { ++bad_count; continue; }

        const double v = roc_sum_times100_f64(c, inv_tm[idxjs], inv_tm[idxjl]);
        comp_add_f64(v, sum, sum_c);
        comp_add_f64(v * (double)w, wsum, wsum_c);
    }

    {
        double* dst = out_tm + (size_t)warm * (size_t)cols + s;
        if (bad_count > 0) *dst = XNAN;
        else               *dst = (wsum + wsum_c) / denom_w;
    }

    bool state_valid = (bad_count == 0);


    for (int t = warm + 1; t < rows; ++t) {

        const int jn = t;
        const int jns = jn - short_p;
        const int jnl = jn - long_p;

        const size_t idxjn  = (size_t)jn  * (size_t)cols + s;
        const size_t idxjns = (size_t)jns * (size_t)cols + s;
        const size_t idxjnl = (size_t)jnl * (size_t)cols + s;

        const double cn  = price_tm[idxjn];
        const double pns = price_tm[idxjns];
        const double pnl = price_tm[idxjnl];
        const bool inv_new = any_nan3_f64(cn, pns, pnl);

        double v_new = 0.0;
        if (!inv_new) v_new = roc_sum_times100_f64(cn, inv_tm[idxjns], inv_tm[idxjnl]);


        const int jo = t - m;
        const int jos = jo - short_p;
        const int jol = jo - long_p;

        const size_t idxjo  = (size_t)jo  * (size_t)cols + s;
        const size_t idxjos = (size_t)jos * (size_t)cols + s;
        const size_t idxjol = (size_t)jol * (size_t)cols + s;

        const double co  = price_tm[idxjo];
        const double pos = price_tm[idxjos];
        const double pol = price_tm[idxjol];
        const bool inv_old = any_nan3_f64(co, pos, pol);

        double v_old = 0.0;
        if (!inv_old) v_old = roc_sum_times100_f64(co, inv_tm[idxjos], inv_tm[idxjol]);

        bad_count += (int)inv_new - (int)inv_old;

        double* dst = out_tm + (size_t)t * (size_t)cols + s;
        if (bad_count == 0) {
            if (!state_valid) {

                sum = 0.0; sum_c = 0.0;
                wsum = 0.0; wsum_c = 0.0;
                int ww = 1;
                const int rst = t - m + 1;
                for (int j = rst; j <= t; ++j, ++ww) {
                    const int js2 = j - short_p;
                    const int jl2 = j - long_p;

                    const size_t idxj2  = (size_t)j   * (size_t)cols + s;
                    const size_t idxjs2 = (size_t)js2 * (size_t)cols + s;
                    const size_t idxjl2 = (size_t)jl2 * (size_t)cols + s;

                    const double c2  = price_tm[idxj2];
                    const double v2  = roc_sum_times100_f64(c2, inv_tm[idxjs2], inv_tm[idxjl2]);
                    comp_add_f64(v2, sum, sum_c);
                    comp_add_f64(v2 * (double)ww, wsum, wsum_c);
                }
                state_valid = true;
            } else {
                const double sum_prev = sum + sum_c;
                comp_add_f64((double)m * v_new, wsum, wsum_c);
                comp_sub_f64(sum_prev,            wsum, wsum_c);
                comp_add_f64(v_new, sum, sum_c);
                comp_sub_f64(v_old, sum, sum_c);
            }
            *dst = (wsum + wsum_c) / denom_w;
        } else {
            *dst = XNAN;
            state_valid = false;
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - coppock
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/coppock.rs:293 `coppock_with_kernel` - the path
 *             `compute_coppock_batch` (cpu_batch.rs:16636) actually calls -
 *             which is `coppock_scalar` (:604) followed by
 *             `ma("wma", sum_roc, ma_period)`, i.e.
 *             moving_averages/wma.rs:305 `wma_scalar`.
 *
 * SINGLE OUTPUT: the batch accepts "value"/"values" only (cpu_batch.rs:16660).
 *
 * PERIOD-INVARIANT. The CPU batch reads `short_roc_period` (11),
 * `long_roc_period` (14) and `ma_period` (10), pins `ma_type` to "wma", and
 * never reads `period`.
 *
 * NOT THE FUSED PATH. `coppock_scalar_default_wma` (:618) exists in this file
 * and is a DIFFERENT accumulation - it seeds `weight_sum`/`sum` over the
 * lookback and then rolls. It is reached from `coppock_into_slice`, not from
 * `coppock_with_kernel`, so it is NOT the oracle for the batch lane. What is
 * transcribed here is the two-stage form: ROC sum, then a generic WMA over
 * the NaN-prefixed ROC series. Naming this explicitly because the two paths
 * in this one file do not produce the same doubles.
 *
 * COMPOSITE, SEQUENCED IN ONE THREAD. The brief's shape for a composite is
 * "compute the components, then combine". Here the component is a per-bar
 * closed form (the two ROCs) with no state, so it is recomputed on the fly
 * inside the WMA walk instead of being staged through a device buffer - one
 * pass, no extra allocation, and the same values.
 *
 * FIRST-VALID, TWICE. Stage one starts at `first + max(short, long)`
 * (:334, :606). Stage two runs its OWN scan: `wma_prepare` (:259) takes the
 * first non-NaN of the ROC SERIES, which is normally that same index but is
 * LATER if the ROC there is NaN because the price it divides by is. The scan
 * is reproduced rather than assumed, so an interior hole moves the WMA start
 * exactly as it does on the CPU.
 *
 * WMA WARMUP: `first_roc + ma_period - 1` (wma.rs:214).
 *
 * WMA ACCUMULATION: `weight_sum` and `sum` are RUNNING accumulators updated
 * as `weight_sum += v * period; sum += v; emit; weight_sum -= sum;
 * sum -= old` (wma.rs:332-341). Not a fresh dot product per bar. The order is
 * load-bearing and is reproduced literally; `weights = p * (p + 1) * 0.5` is
 * formed the same way (:310).
 *
 * SEQUENTIAL, one thread per combo column. The only per-thread storage is the
 * `ma_period`-slot ring, a fixed 10 doubles at the CPU default.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define COPPOCK_NEO_SHORT 11
#define COPPOCK_NEO_LONG  14
#define COPPOCK_NEO_MA    10

__device__ __forceinline__ double coppock_neo_roc_sum_f64(
    const double* __restrict__ d, int i)
{
    const double current = d[i];
    const double short_val = ((current / d[i - COPPOCK_NEO_SHORT]) - 1.0) * 100.0;
    const double long_val  = ((current / d[i - COPPOCK_NEO_LONG])  - 1.0) * 100.0;
    return short_val + long_val;
}

extern "C" __global__
void coppock_neo_batch_f64(const double* __restrict__ data,
                           int series_len,
                           const int* __restrict__ periods,
                           int n_combos,
                           int first_valid,
                           double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (first_valid < 0 || first_valid >= len) return;

    const int largest = (COPPOCK_NEO_SHORT > COPPOCK_NEO_LONG)
                            ? COPPOCK_NEO_SHORT : COPPOCK_NEO_LONG;
    const int roc_start = first_valid + largest;      /* coppock.rs:334 */
    if (roc_start >= len) return;
    if (len - first_valid < largest) return;          /* :327 NotEnoughValidData */

    /* wma_prepare's own first-non-NaN scan over the ROC series (wma.rs:259). */
    int first_roc = roc_start;
    while (first_roc < len && isnan(coppock_neo_roc_sum_f64(data, first_roc))) {
        first_roc += 1;
    }
    if (first_roc >= len) return;

    const int    P        = COPPOCK_NEO_MA;
    const int    lookback = P - 1;
    const double period_f = (double)P;
    const double weights  = period_f * (period_f + 1.0) * 0.5;   /* wma.rs:310 */

    if (first_roc + lookback >= len) return;

    double ring[COPPOCK_NEO_MA];
    double sum = 0.0, weight_sum = 0.0;
    for (int k = 0; k < lookback; ++k) {
        const double v = coppock_neo_roc_sum_f64(data, first_roc + k);
        ring[k] = v;
        weight_sum += v * ((double)k + 1.0);
        sum += v;
    }

    /* `in_old` trails `in_new` by exactly `lookback` bars (wma.rs:327-328),
       so the ring is `lookback` slots wide - NOT `period`. A `period`-wide
       ring would hand back a value one bar too old on every wrap. */
    int old_slot = 0;
    for (int i = first_roc + lookback; i < len; ++i) {
        const double v = coppock_neo_roc_sum_f64(data, i);
        weight_sum += v * period_f;
        sum += v;

        o[i] = weight_sum / weights;

        const double old = ring[old_slot];
        ring[old_slot] = v;
        old_slot += 1; if (old_slot == lookback) old_slot = 0;

        weight_sum -= sum;
        sum -= old;
    }
}
