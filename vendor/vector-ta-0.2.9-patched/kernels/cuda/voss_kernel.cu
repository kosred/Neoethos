#include <cuda_runtime.h>
#include <math.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ float voss_s1_f32(float g1) {
    const float inv_g1 = 1.0f / g1;
    const float t = fmaxf(inv_g1 * inv_g1 - 1.0f, 0.0f);
    const float root = sqrtf(t);
    return inv_g1 - root;
}


struct dsfloat { float hi, lo; };

__device__ __forceinline__ dsfloat ds_from_float(float x) { return {x, 0.0f}; }
__device__ __forceinline__ float  ds_to_float(const dsfloat &a) { return a.hi + a.lo; }


__device__ __forceinline__ dsfloat ds_add(dsfloat a, dsfloat b) {
    float s  = a.hi + b.hi;
    float z  = s - a.hi;
    float e  = (a.hi - (s - z)) + (b.hi - z) + a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return {hi, lo};
}

__device__ __forceinline__ dsfloat ds_sub(dsfloat a, dsfloat b) {
    b.hi = -b.hi; b.lo = -b.lo;
    return ds_add(a, b);
}


__device__ __forceinline__ dsfloat two_prod_fma(float a, float b) {
    float p = a * b;
    float e = fmaf(a, b, -p);
    return {p, e};
}


__device__ __forceinline__ dsfloat ds_mul_scalar(dsfloat a, float s) {
    float p  = a.hi * s;
    float e  = fmaf(a.hi, s, -p) + a.lo * s;
    float hi = p + e;
    float lo = e - (hi - p);
    return {hi, lo};
}


__device__ __forceinline__ dsfloat ds_fma_scalar(dsfloat a, float s, dsfloat c) {
    return ds_add(ds_mul_scalar(a, s), c);
}


extern "C" __global__ void voss_cast_f32_to_f64(
    const float* __restrict__ input,
    int len,
    double* __restrict__ output)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        output[idx] = (double)input[idx];
    }
}


extern "C" __global__ void voss_batch_f32(
    const double* __restrict__ prices,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const int* __restrict__ predicts,
    const double* __restrict__ bandwidths,
    int nrows,
    float* __restrict__ out_voss,
    float* __restrict__ out_filt)
{
    const int row = blockIdx.y;
    if (row >= nrows) return;

    if (threadIdx.x != 0) return;

    const int p  = periods[row];
    const int q  = predicts[row];
    const float bw = (float)bandwidths[row];
    if (p <= 0 || q < 0) return;

    const int order     = 3 * q;
    const int min_index = max(max(p, 5), order);
    const int start     = first_valid + min_index;
    const int row_off   = row * len;


    const int warm_end = (start < len ? start : len);
    for (int t = 0; t < warm_end; ++t) {
        out_voss[row_off + t] = f32_nan();
        out_filt[row_off + t] = f32_nan();
    }


    if (start - 2 >= 0 && start - 2 < len) out_filt[row_off + (start - 2)] = 0.0f;
    if (start - 1 >= 0 && start - 1 < len) out_filt[row_off + (start - 1)] = 0.0f;

    if (start >= len) return;


    const float TWO_PI = 6.2831853071795864769f;
    const float w0 = TWO_PI / (float)p;
    const float f1 = cosf(w0);
    const float g1 = cosf(bw * w0);
    const float s1 = voss_s1_f32(g1);
    const float c1 = 0.5f * (1.0f - s1);
    const float c2 = f1 * (1.0f + s1);
    const float c3 = -s1;
    const float scale = 0.5f * (3.0f + (float)order);


    dsfloat prev_f1 = ds_from_float(0.0f);
    dsfloat prev_f2 = ds_from_float(0.0f);


    float x_im2 = (float)prices[start - 2];
    float x_im1 = (float)prices[start - 1];

    if (order == 0) {
        for (int i = start; i < len; ++i) {
            const float xi   = (float)prices[i];
            const float diff = xi - x_im2;


            const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
            const dsfloat f = ds_fma_scalar(prev_f1, c2, t);

            const float f_out = ds_to_float(f);
            out_filt[row_off + i] = f_out;
            out_voss[row_off + i] = scale * f_out;


            prev_f2 = prev_f1;
            prev_f1 = f;
            x_im2 = x_im1;
            x_im1 = xi;
        }
        return;
    }


    dsfloat a_sum = ds_from_float(0.0f);
    dsfloat d_sum = ds_from_float(0.0f);
    const float inv_m = 1.0f / (float)order;

    for (int i = start; i < len; ++i) {
        const float xi   = (float)prices[i];
        const float diff = xi - x_im2;


        const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
        const dsfloat f = ds_fma_scalar(prev_f1, c2, t);
        const float   f_out = ds_to_float(f);
        out_filt[row_off + i] = f_out;

        prev_f2 = prev_f1;
        prev_f1 = f;


        const float sumc = ds_to_float(d_sum) * inv_m;
        const float vi   = scale * f_out - sumc;
        out_voss[row_off + i] = vi;

        const float v_new_nz = isnan(vi) ? 0.0f : vi;


        const int j_old = i - order;
        float v_old = 0.0f;
        if (j_old >= start) {
            const float vv = out_voss[row_off + j_old];
            v_old = isnan(vv) ? 0.0f : vv;
        }

        const dsfloat a_prev = a_sum;

        a_sum = ds_add(ds_sub(a_prev, ds_from_float(v_old)), ds_from_float(v_new_nz));

        d_sum = ds_add(ds_sub(d_sum, a_prev), ds_from_float((float)order * v_new_nz));


        x_im2 = x_im1;
        x_im1 = xi;
    }
}


extern "C" __global__ void voss_many_series_one_param_time_major_f32(
    const double* __restrict__ data_tm,
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    int predict,
    double bandwidth,
    float* __restrict__ out_voss_tm,
    float* __restrict__ out_filt_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;
    if (threadIdx.x != 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {

        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            out_voss_tm[idx] = f32_nan();
            out_filt_tm[idx] = f32_nan();
        }
        return;
    }

    const int order     = 3 * predict;
    const int min_index = max(max(period, 5), order);
    const int start     = fv + min_index;


    const int warm_end = (start < rows ? start : rows);
    for (int t = 0; t < warm_end; ++t) {
        const int idx = t * cols + s;
        out_voss_tm[idx] = f32_nan();
        out_filt_tm[idx] = f32_nan();
    }
    if (start - 2 >= 0 && start - 2 < rows) out_filt_tm[(start - 2) * cols + s] = 0.0f;
    if (start - 1 >= 0 && start - 1 < rows) out_filt_tm[(start - 1) * cols + s] = 0.0f;

    if (start >= rows) return;


    const float TWO_PI = 6.2831853071795864769f;
    const float w0 = TWO_PI / (float)period;
    const float f1 = cosf(w0);
    const float g1 = cosf((float)bandwidth * w0);
    const float s1 = voss_s1_f32(g1);
    const float c1 = 0.5f * (1.0f - s1);
    const float c2 = f1 * (1.0f + s1);
    const float c3 = -s1;
    const float scale = 0.5f * (3.0f + (float)order);

    dsfloat prev_f1 = ds_from_float(0.0f);
    dsfloat prev_f2 = ds_from_float(0.0f);


    float x_im2 = (float)data_tm[(start - 2) * cols + s];
    float x_im1 = (float)data_tm[(start - 1) * cols + s];

    if (order == 0) {
        for (int i = start; i < rows; ++i) {
            const int   idx  = i * cols + s;
            const float xi   = (float)data_tm[idx];
            const float diff = xi - x_im2;

            const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
            const dsfloat f = ds_fma_scalar(prev_f1, c2, t);
            const float   f_out = ds_to_float(f);

            out_filt_tm[idx] = f_out;
            out_voss_tm[idx] = scale * f_out;

            prev_f2 = prev_f1;
            prev_f1 = f;

            x_im2 = x_im1;
            x_im1 = xi;
        }
        return;
    }

    dsfloat a_sum = ds_from_float(0.0f);
    dsfloat d_sum = ds_from_float(0.0f);
    const float inv_m = 1.0f / (float)order;

    for (int i = start; i < rows; ++i) {
        const int   idx  = i * cols + s;
        const float xi   = (float)data_tm[idx];
        const float diff = xi - x_im2;

        const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
        const dsfloat f = ds_fma_scalar(prev_f1, c2, t);
        const float   f_out = ds_to_float(f);

        out_filt_tm[idx] = f_out;

        prev_f2 = prev_f1;
        prev_f1 = f;

        const float sumc = ds_to_float(d_sum) * inv_m;
        const float vi   = scale * f_out - sumc;
        out_voss_tm[idx] = vi;

        const float v_new_nz = isnan(vi) ? 0.0f : vi;

        const int j_old = i - order;
        float v_old = 0.0f;
        if (j_old >= start) {
            const float vv = out_voss_tm[j_old * cols + s];
            v_old = isnan(vv) ? 0.0f : vv;
        }

        const dsfloat a_prev = a_sum;
        a_sum = ds_add(ds_sub(a_prev, ds_from_float(v_old)), ds_from_float(v_new_nz));
        d_sum = ds_add(ds_sub(d_sum, a_prev), ds_from_float((float)order * v_new_nz));

        x_im2 = x_im1;
        x_im1 = xi;
    }
}


// ===========================================================================
// f64 LANE  --  shard S6
//
// CPU reference: `voss_row_scalar` (src/indicators/voss.rs:1329), which
// `voss_scalar` (:430) forwards to unchanged.
//
// OUTPUT: `voss`, which is `OUTPUTS_VOSS[0]` (registry.rs:2533-2541 ->
// [voss, filt]). `filt` is the band-pass stage the same loop produces.
//
// first_valid: `data.iter().position(|x| !x.is_nan())` (:252-254) over the
// single source series (default "close", :119) ->
// `F64FirstValidRule::AllInputsNonNan`.
//
// warm: `start = first + min_index` where
// `min_index = period.max(5).max(order)` and `order = 3 * predict` (:1340-1342).
// The swept `periods` value is `period`; `predict` keeps its default of 3
// (:126-128), so `order = 9`, and `bandwidth` keeps 0.25 (:130-132).
//
// ONE THREAD PER COLUMN, ascending bars. Two carried filter lags (prev_f1,
// prev_f2), two carried running sums (a_sum, d_sum) and a ring of the last
// `order` voss values. Not bar-parallel and not scan-reformulable.
//
// THE RING IS BOUNDED BY `order`, WHICH THE LANE DOES NOT SWEEP. `periods`
// carries `period`, never `predict`, so `order` is the constant 9 for every
// row of every sweep. The kernel still declares `VOSS_MAX_ORDER` and refuses
// a larger order by emitting the all-NaN column rather than overrunning the
// per-thread array -- silently truncating the ring would compute a different
// indicator, which is the exact failure this lane exists to remove.
//
// FOUR FUSED MULTIPLY-ADDS PER BAR, MATCHING THE CPU ONE FOR ONE:
//   t     = c3.mul_add(prev_f2, c1 * diff)          (:1386)
//   f     = c2.mul_add(prev_f1, t)                  (:1387)
//   vi    = scale.mul_add(f, -sumc)                 (:1393)
//   d_sum = ord_f.mul_add(v_new_nz, d_sum - a_prev) (:1401)
// Written with `fma`, so four roundings, not eight. `a_sum = a_prev - v_old +
// v_new_nz` (:1400) is deliberately NOT fused -- the CPU leaves it as two
// plain operations and fusing it would drop a rounding the host performs.
//
// NaN HANDLING IS THE CPU'S, AND IT IS NOT fmax. :1396 writes
// `let v_new_nz = if vi.is_nan() { 0.0 } else { vi }` -- a NaN output is
// substituted by ZERO before it enters the ring, so the running sums stay
// finite while the emitted value stays NaN. Reproduced with `isnan`, because
// `fmax(vi, 0.0)` would also clamp every genuinely negative voss value to
// zero.
//
// f32 -> f64 audit: the f32 lane above uses `cosf` x4, `fmaxf`, `sqrtf` and
// `__int_as_float`. Below: `cos`, `sqrt`, `fma`, and the f64 quiet-NaN bit
// pattern. PI is the full-precision f64 constant, not an f32-width decimal.
// No fast-math intrinsic: `1.0 / g1` and `1.0 / (g1 * g1)` are true divisions,
// and `s1 = 1/g1 - sqrt(1/(g1*g1) - 1)` is a cancellation-sensitive
// expression that `__frcp_rn`/`rsqrt` would visibly damage -- s1 feeds all
// three filter coefficients, so its error is amplified by the recursion at
// every subsequent bar. This indicator has no epsilon.
// ===========================================================================

#define VOSS_MAX_ORDER 64

static __device__ __forceinline__ double voss_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void voss_batch_f64(const double* __restrict__ data,
                    int n,
                    const int* __restrict__ periods,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = voss_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const int first  = (first_valid < 0) ? 0 : first_valid;

    for (int t = 0; t < n; ++t) row[t] = nan_d;

    const int predict   = 3;                 // voss.rs:126-128, not swept
    const double bandwidth = 0.25;           // voss.rs:130-132, not swept
    const int order     = 3 * predict;       // :1340
    if (order > VOSS_MAX_ORDER) return;

    int min_index = period;                  // :1341 -- period.max(5).max(order)
    if (5 > min_index)     min_index = 5;
    if (order > min_index) min_index = order;

    // `voss_prepare` rejects period == 0, period > len and len - first < min_index.
    if (period <= 0 || period > n || first >= n || (n - first) < min_index) return;

    const int start = first + min_index;     // :1342
    if (start >= n) return;                  // :1343-1345
    if (start < 2) return;                   // filt[start-2] would be out of range

    const double PI_F64 = 3.14159265358979323846264338327950288;
    const double w0 = 2.0 * PI_F64 / static_cast<double>(period);   // :1347
    const double f1 = cos(w0);                                      // :1348
    const double g1 = cos(bandwidth * w0);                          // :1349
    const double s1 = 1.0 / g1 - sqrt(1.0 / (g1 * g1) - 1.0);       // :1350
    const double c1 = 0.5 * (1.0 - s1);                             // :1351
    const double c2 = f1 * (1.0 + s1);                              // :1352
    const double c3 = -s1;                                          // :1353

    double prev_f1 = 0.0, prev_f2 = 0.0;                            // :1360-1361
    const double scale = 0.5 * static_cast<double>(3 + order);      // :1362

    if (order == 0) {                                               // :1364-1375
        for (int i = start; i < n; ++i) {
            const double diff = data[i] - data[i - 2];
            const double t = fma(c3, prev_f2, c1 * diff);
            const double f = fma(c2, prev_f1, t);
            prev_f2 = prev_f1;
            prev_f1 = f;
            row[i] = scale * f;
        }
        return;
    }

    const double ord_f     = static_cast<double>(order);            // :1377
    const double inv_order = 1.0 / ord_f;                           // :1378
    double a_sum = 0.0, d_sum = 0.0;                                // :1379-1380
    double ring[VOSS_MAX_ORDER];
    for (int k = 0; k < order; ++k) ring[k] = 0.0;                  // :1381
    int rpos = 0;

    for (int i = start; i < n; ++i) {                               // :1384-1406
        const double diff = data[i] - data[i - 2];
        const double t = fma(c3, prev_f2, c1 * diff);
        const double f = fma(c2, prev_f1, t);
        prev_f2 = prev_f1;
        prev_f1 = f;

        const double sumc = d_sum * inv_order;
        const double vi   = fma(scale, f, -sumc);
        row[i] = vi;

        const double v_new_nz = isnan(vi) ? 0.0 : vi;
        const double v_old    = ring[rpos];

        const double a_prev = a_sum;
        a_sum = a_prev - v_old + v_new_nz;
        d_sum = fma(ord_f, v_new_nz, d_sum - a_prev);

        ring[rpos] = v_new_nz;
        rpos += 1;
        if (rpos == order) rpos = 0;
    }
}
