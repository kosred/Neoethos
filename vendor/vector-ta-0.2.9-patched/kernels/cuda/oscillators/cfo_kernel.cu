#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

extern "C" __global__ void cfo_build_prefixes_serial_f64(
    const float* __restrict__ data,
    int len,
    int first_valid,
    double* __restrict__ prefix_sum,
    double* __restrict__ prefix_weighted)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    prefix_sum[0] = 0.0;
    prefix_weighted[0] = 0.0;

    double acc_s = 0.0;
    double acc_w = 0.0;
    double weight = 0.0;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const double v = (double)data[i];
            weight += 1.0;
            acc_s += v;
            acc_w += v * weight;
        }
        prefix_sum[i + 1] = acc_s;
        prefix_weighted[i + 1] = acc_w;
    }
}


extern "C" __global__ void cfo_batch_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_weighted,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ scalars,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float scalar = scalars[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;


    const double n = (double)period;
    const double sx = (double)(period * (period + 1)) * 0.5;
    const double sx2 = (double)(period * (period + 1) * (2 * period + 1)) / 6.0;
    const double inv_denom = 1.0 / (n * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (n - 1.0);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const float nanf = f32_nan();
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {

            const int idx = t - first_valid;
            const int r1 = idx + 1;
            const int l1_minus1 = r1 - period;


            const double sum_y = prefix_sum[first_valid + r1] - prefix_sum[first_valid + l1_minus1];
            const double sum_xy_raw = prefix_weighted[first_valid + r1] - prefix_weighted[first_valid + l1_minus1];
            const double sum_xy = sum_xy_raw - ((double)l1_minus1) * sum_y;

            const double b = (-sx) * sum_y + n * sum_xy;
            const double b_scaled = b * inv_denom;
            const double f = b_scaled * half_nm1 + sum_y / n;
            const float cur = data[t];
            if (!isnan(cur) && cur != 0.0f) {

                out_val = scalar * (1.0f - (float)(f / (double)cur));
            } else {
                out_val = nanf;
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void cfo_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    const double* __restrict__ prefix_sum_tm,
    const double* __restrict__ prefix_weighted_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;


    const double n = (double)period;
    const double sx = (double)(period * (period + 1)) * 0.5;
    const double sx2 = (double)(period * (period + 1) * (2 * period + 1)) / 6.0;
    const double inv_denom = 1.0 / (n * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (n - 1.0);


    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float nanf = f32_nan();

        int t = 0;
        for (; t < fv && t < rows; ++t) {
            out_tm[t * cols + s] = nanf;
        }
        if (t >= rows) return;


        double sum_y = 0.0;
        double sum_xy = 0.0;
        int warm_needed = period - 1;
        int k = 0;
        for (; k < warm_needed && t < rows; ++k, ++t) {
            float v = data_tm[t * cols + s];
            double vd = (double)v;
            double w = (double)(k + 1);
            sum_y += vd;
            sum_xy += vd * w;
            out_tm[t * cols + s] = nanf;
        }
        if (t >= rows) return;


        {
            float v = data_tm[t * cols + s];
            double vd = (double)v;
            sum_y += vd;
            sum_xy += vd * n;
            double b = (-sx) * sum_y + n * sum_xy;
            double f = (b * inv_denom) * half_nm1 + sum_y / n;
            out_tm[t * cols + s] = (!isnan(v) && v != 0.0f)
                ? (float)(scalar * (1.0 - f / (double)v))
                : nanf;
            ++t;
        }


        for (; t < rows; ++t) {
            float v_new = data_tm[t * cols + s];
            float v_old = data_tm[(t - period) * cols + s];
            double vd_new = (double)v_new;
            double vd_old = (double)v_old;
            double new_sum_xy = (n * vd_new) + (sum_xy - sum_y);
            double new_sum_y = sum_y - vd_old + vd_new;
            sum_xy = new_sum_xy;
            sum_y = new_sum_y;
            double b = (-sx) * sum_y + n * sum_xy;
            double f = (b * inv_denom) * half_nm1 + sum_y / n;
            out_tm[t * cols + s] = (!isnan(v_new) && v_new != 0.0f)
                ? (float)(scalar * (1.0 - f / (double)v_new))
                : nanf;
        }
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `cfo.rs::cfo_scalar` (l.367). Defaults: period 14, scalar
// 100.0 (`cfo.rs:112`, `cfo.rs:116`). The registry sweeps `period`; `scalar`
// takes the CPU default.
//
// The rounding structure is copied line for line, because every one of these
// is a fused multiply-add on the CPU and an unfused pair here would round
// twice:
//   sum_xy = v.mul_add(w, sum_xy)                -> fma(v, w, sum_xy)
//   sum_xy = v.mul_add(n, sum_xy)                -> fma(v, n, sum_xy)
//   b      = (-sx).mul_add(sum_y, n*sum_xy) * inv_denom
//                                                -> fma(-sx, sum_y, n*sum_xy) * inv_denom
//   f      = b.mul_add(half_nm1, sum_y * inv_n)  -> fma(b, half_nm1, sum_y*inv_n)
// and the emit is `(v - f) * (scalar / v)` — NOT `(v-f)*scalar/v`, which is a
// different rounding.
//
// sx and sx2 are integer triangular / square-pyramidal numbers on the CPU
// (`(period*(period+1))/2`, `(period*(period+1)*(2*period+1))/6`) evaluated in
// `usize` and only then cast to f64. They are computed here in `long long` for
// the same reason: forming them in double would round for periods past 2^53/6.
//
// f32 -> f64 audit: pointers/locals widened; `__int_as_float` NaN -> f64
// quiet-NaN pattern; no fast-math intrinsic survives; no epsilon (`v != 0.0` is
// an exact test and stays exact); the only comparison is `isfinite(v) && v != 0`
// which cannot let a NaN through the true branch.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double cfo_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void cfo_batch_f64(const double* __restrict__ prices,
                   int n,
                   const int*   __restrict__ periods,
                   int n_combos,
                   int first_valid,
                   double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = cfo_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    if (period <= 0 || first_valid >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const int pre = period - 1;
    const long long start_ll = static_cast<long long>(first_valid) + static_cast<long long>(pre);
    if (start_ll >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const double scalar = 100.0;                 // cfo.rs:116 default
    const double nn = static_cast<double>(period);
    const double inv_n = 1.0 / nn;
    const long long pll = static_cast<long long>(period);
    const double sx  = static_cast<double>((pll * (pll + 1)) / 2);
    const double sx2 = static_cast<double>((pll * (pll + 1) * (2 * pll + 1)) / 6);
    const double inv_denom = 1.0 / (nn * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (nn - 1.0);

    const int start = first_valid;
    for (int t = 0; t < start + pre; ++t) row[t] = nan_d;

    double sum_y = 0.0;
    double sum_xy = 0.0;
    for (int k = 0; k < pre; ++k) {
        const double v = prices[start + k];
        const double w = static_cast<double>(k) + 1.0;
        sum_y += v;
        sum_xy = fma(v, w, sum_xy);
    }

    for (int i = start + pre; i < n; ++i) {
        const double v = prices[i];
        sum_xy = fma(v, nn, sum_xy);
        sum_y += v;
        const double b = fma(-sx, sum_y, nn * sum_xy) * inv_denom;
        const double f = fma(b, half_nm1, sum_y * inv_n);
        row[i] = (isfinite(v) && v != 0.0) ? (v - f) * (scalar / v) : nan_d;
        sum_xy -= sum_y;
        sum_y -= prices[i - pre];
    }
}
