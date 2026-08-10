#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

extern "C" __global__
void di_build_up_dn_tr_f32(const float* __restrict__ high,
                           const float* __restrict__ low,
                           const float* __restrict__ close,
                           int len,
                           int first_valid,
                           float* __restrict__ up,
                           float* __restrict__ dn,
                           float* __restrict__ tr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0) return;

    for (int i = 0; i < len; ++i) {
        up[i] = 0.0f;
        dn[i] = 0.0f;
        tr[i] = 0.0f;
    }

    if (first_valid < 0 || first_valid >= len) return;

    float prev_h = high[first_valid];
    float prev_l = low[first_valid];
    float prev_c = close[first_valid];
    for (int i = first_valid + 1; i < len; ++i) {
        const float ch = high[i];
        const float cl = low[i];
        const float dp = ch - prev_h;
        const float dm = prev_l - cl;
        if (dp > dm && dp > 0.0f) up[i] = dp;
        if (dm > dp && dm > 0.0f) dn[i] = dm;
        float t = ch - cl;
        const float t2 = fabsf(ch - prev_c);
        if (t2 > t) t = t2;
        const float t3 = fabsf(cl - prev_c);
        if (t3 > t) t = t3;
        tr[i] = t;
        prev_h = ch;
        prev_l = cl;
        prev_c = close[i];
    }
}


struct ds {
    float hi;
    float lo;
};

static __forceinline__ __device__ ds ds_make(float x) {
    ds r; r.hi = x; r.lo = 0.0f; return r;
}

static __forceinline__ __device__ float ds_value(const ds& v) {
    return v.hi + v.lo;
}


static __forceinline__ __device__ void twoProductFMA(float a, float b, float &x, float &y) {
    x = a * b;
    y = fmaf(a, b, -x);
}


static __forceinline__ __device__ void twoSum(float a, float b, float &s, float &e) {
    s = a + b;
    float z = s - a;
    e = (a - (s - z)) + (b - z);
}


static __forceinline__ __device__ void ds_rma_update(ds &s, float keep, float inc) {

    float p_hi, p_err;
    twoProductFMA(s.hi, keep, p_hi, p_err);
    float t_lo = s.lo * keep;


    float sh, e_sum;
    twoSum(p_hi, inc, sh, e_sum);


    float slo = e_sum + (p_err + t_lo);
    float new_hi = sh + slo;
    s.lo = slo - (new_hi - sh);
    s.hi = new_hi;
}


extern "C" __global__
void di_batch_from_precomputed_f32(const float* __restrict__ up,
                                   const float* __restrict__ dn,
                                   const float* __restrict__ tr,
                                   const int* __restrict__ periods,
                                   const int* __restrict__ warm_indices,
                                   int series_len,
                                   int first_valid,
                                   int n_combos,
                                   float* __restrict__ plus_out,
                                   float* __restrict__ minus_out)
{
    if (blockDim.x == 1) {
        for (int combo = blockIdx.x; combo < n_combos; combo += gridDim.x) {
            const int period = periods[combo];
            const int warm   = warm_indices[combo];
            if (period <= 0 || warm < 0 || warm >= series_len) continue;

            const float invp = 1.0f / (float)period;
            const float keep = 1.0f - invp;
            const int base = combo * series_len;

            const int start = first_valid + 1;
            const int stop  = first_valid + period;
            if (stop > series_len) {
                for (int i = 0; i < series_len; ++i) {
                    plus_out[base + i] = NAN;
                    minus_out[base + i] = NAN;
                }
                continue;
            }

            for (int i = 0; i < warm; ++i) {
                plus_out[base + i] = NAN;
                minus_out[base + i] = NAN;
            }

            float sp = 0.0f, sm = 0.0f, st = 0.0f;
            for (int t = start; t < stop; ++t) {
                sp += up[t];
                sm += dn[t];
                st += tr[t];
            }

            ds cur_p = ds_make(sp);
            ds cur_m = ds_make(sm);
            ds cur_t = ds_make(st);

            float denom = ds_value(cur_t);
            float scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
            plus_out[base + warm] = ds_value(cur_p) * scale;
            minus_out[base + warm] = ds_value(cur_m) * scale;

            for (int t = warm + 1; t < series_len; ++t) {
                ds_rma_update(cur_p, keep, up[t]);
                ds_rma_update(cur_m, keep, dn[t]);
                ds_rma_update(cur_t, keep, tr[t]);

                denom = ds_value(cur_t);
                scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
                plus_out[base + t] = ds_value(cur_p) * scale;
                minus_out[base + t] = ds_value(cur_m) * scale;
            }
        }
        return;
    }

    for (int combo = blockIdx.x; combo < n_combos; combo += gridDim.x) {
        const int period = periods[combo];
        const int warm   = warm_indices[combo];
        if (period <= 0 || warm < 0 || warm >= series_len) continue;

        const float invp = 1.0f / (float)period;
        const float keep = 1.0f - invp;

        const int base = combo * series_len;


        const int start = first_valid + 1;
        const int stop  = first_valid + period;
        if (stop > series_len) {

            for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
                plus_out [base + i] = NAN;
                minus_out[base + i] = NAN;
            }
            __syncthreads();
            continue;
        }


        for (int i = threadIdx.x; i < warm; i += blockDim.x) {
            plus_out [base + i] = NAN;
            minus_out[base + i] = NAN;
        }
        __syncthreads();


        float lp = 0.0f, lm = 0.0f, lt = 0.0f;
        for (int t = start + threadIdx.x; t < stop; t += blockDim.x) {
            lp += up[t];
            lm += dn[t];
            lt += tr[t];
        }
        float sp = block_reduce_sum(lp);
        float sm = block_reduce_sum(lm);
        float st = block_reduce_sum(lt);

        if (threadIdx.x == 0) {

            ds cur_p = ds_make(sp);
            ds cur_m = ds_make(sm);
            ds cur_t = ds_make(st);

            float denom = ds_value(cur_t);
            float scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
            plus_out [base + warm] = ds_value(cur_p) * scale;
            minus_out[base + warm] = ds_value(cur_m) * scale;


            for (int t = warm + 1; t < series_len; ++t) {
                ds_rma_update(cur_p, keep, up[t]);
                ds_rma_update(cur_m, keep, dn[t]);
                ds_rma_update(cur_t, keep, tr[t]);

                denom = ds_value(cur_t);
                scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
                plus_out [base + t] = ds_value(cur_p) * scale;
                minus_out[base + t] = ds_value(cur_m) * scale;
            }
        }
        __syncthreads();
    }
}


extern "C" __global__
void di_many_series_one_param_f32(const float* __restrict__ high_tm,
                                  const float* __restrict__ low_tm,
                                  const float* __restrict__ close_tm,
                                  const int* __restrict__ first_valids,
                                  int period,
                                  int num_series,
                                  int series_len,
                                  float* __restrict__ plus_tm,
                                  float* __restrict__ minus_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    const float invp = 1.0f / (float)period;
    const float keep = 1.0f - invp;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) {

            for (int t = lane; t < series_len; t += warpSize) {
                plus_tm [t * stride + s] = NAN;
                minus_tm[t * stride + s] = NAN;
            }
            continue;
        }

        const int start = first_valid + 1;
        const int stop  = first_valid + period;
        if (stop > series_len) {
            for (int t = lane; t < series_len; t += warpSize) {
                plus_tm [t * stride + s] = NAN;
                minus_tm[t * stride + s] = NAN;
            }
            continue;
        }
        const int warm = stop - 1;


        for (int t = lane; t < warm; t += warpSize) {
            plus_tm [t * stride + s] = NAN;
            minus_tm[t * stride + s] = NAN;
        }


        float lp = 0.0f, lm = 0.0f, lt = 0.0f;
        for (int t = start + lane; t < stop; t += warpSize) {
            const float ch = high_tm[t * stride + s];
            const float cl = low_tm [t * stride + s];
            const float ph = high_tm[(t - 1) * stride + s];
            const float pl = low_tm [(t - 1) * stride + s];
            const float pc = close_tm[(t - 1) * stride + s];

            const float dp = ch - ph;
            const float dm = pl - cl;
            if (dp > dm && dp > 0.0f) lp += dp;
            if (dm > dp && dm > 0.0f) lm += dm;

            float tr = ch - cl;
            float hc = fabsf(ch - pc);
            if (hc > tr) tr = hc;
            float lc = fabsf(cl - pc);
            if (lc > tr) tr = lc;
            lt += tr;
        }

        lp = warp_reduce_sum(lp);
        lm = warp_reduce_sum(lm);
        lt = warp_reduce_sum(lt);

        if (lane == 0) {
            ds cur_p = ds_make(lp);
            ds cur_m = ds_make(lm);
            ds cur_t = ds_make(lt);

            float denom = ds_value(cur_t);
            float scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
            plus_tm [warm * stride + s] = ds_value(cur_p) * scale;
            minus_tm[warm * stride + s] = ds_value(cur_m) * scale;


            int t  = warm + 1;
            const float* h_ptr  = high_tm  + t * stride + s;
            const float* l_ptr  = low_tm   + t * stride + s;
            const float* ph_ptr = high_tm  + (t - 1) * stride + s;
            const float* pl_ptr = low_tm   + (t - 1) * stride + s;
            const float* pc_ptr = close_tm + (t - 1) * stride + s;

            for (; t < series_len; ++t) {
                const float ch = *h_ptr;  const float cl = *l_ptr;
                const float ph = *ph_ptr; const float pl = *pl_ptr;
                const float pc = *pc_ptr;

                const float dp = ch - ph;
                const float dm = pl - cl;
                const float inc_p = (dp > dm && dp > 0.0f) ? dp : 0.0f;
                const float inc_m = (dm > dp && dm > 0.0f) ? dm : 0.0f;

                float tr = ch - cl;
                float hc = fabsf(ch - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(cl - pc);
                if (lc > tr) tr = lc;

                ds_rma_update(cur_p, keep, inc_p);
                ds_rma_update(cur_m, keep, inc_m);
                ds_rma_update(cur_t, keep, tr);

                denom = ds_value(cur_t);
                scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
                plus_tm [t * stride + s] = ds_value(cur_p) * scale;
                minus_tm[t * stride + s] = ds_value(cur_m) * scale;

                h_ptr  += stride;  l_ptr  += stride;
                ph_ptr += stride;  pl_ptr += stride;  pc_ptr += stride;
            }
        }
    }
}


// ===========================================================================
// S3 f64 LANE — di (Directional Indicator, +DI)
// ===========================================================================
// Reference: src/indicators/di.rs
//   di_prepare (:207)              — first_valid + the Err branches
//   di_selected_with_kernel (:273) — alloc_with_nan_prefix(n, first + period - 1)
//   di_scalar_into (:406)          — the arithmetic
// Batch default period 14, input high/low/close.
//
// WHICH OUTPUT. `di` is multi-output (plus / minus) and the lane emits ONE
// matrix. compute_di_batch (cpu_batch.rs) maps output_id "value" — the id the
// f64 lane requests — to PLUS: `if output_id == "plus" || output_id == "value"
// { 1u8 }`. So this kernel is +DI, matching the column the CPU emits for the
// same request. -DI is a second entry point's job, not a silent alternative.
//
// FIRST-VALID IS THE SIMULTANEOUS RULE (:225):
//   (0..n).find(|i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
// — one index at which all three are non-NaN, i.e. AllInputsNonNan. This is NOT
// adx's max-of-independent-firsts rule even though the two indicators share a
// Wilder core; using adx's index here shifts the seed window.
//
// NaN SEMANTICS — THE CPU USES AN IF-CHAIN HERE, ON PURPOSE, AND SO DOES THIS.
// The true range is built as
//     let mut tr = ch - cl;
//     if tr2 > tr { tr = tr2; }
//     if tr3 > tr { tr = tr3; }        (:449-457, :489-497)
// which is NOT f64::max: if `tr` is NaN both comparisons are false and the NaN
// SURVIVES. Rewriting this as fmax(fmax(...)) — the correction the adx kernel
// needed — would be wrong HERE, because it would return the non-NaN operand and
// silently repair a bar the reference leaves NaN. Matching the reference means
// matching its NaN behaviour in BOTH directions. The chain is transcribed
// literally.
//
// The +DM / -DM selection is likewise a comparison, not a max:
//   dp > dm && dp > 0.0 → dp else 0.0 ; with NaN inputs both are false and the
//   increment is 0.0, exactly as on the CPU.
//
// ROUNDING.
//   cur_plus = cur_plus.mul_add(keep, inc_p)  → ONE fma (:486)
//   cur_tr   = cur_tr.mul_add(keep, tr)       → ONE fma (:498)
//   keep     = 1.0 - period.recip()           — recip() then subtract, in that
//                                               order (:421-422)
//
// ZERO GUARD, NOT AN EPSILON. scale is 0.0 when cur_tr == 0.0 EXACTLY (:471,
// :500). No tolerance appears in the reference and none is invented here.
//
// One thread per column: the three Wilder accumulators are carried.
// ===========================================================================

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_di_batch_f64(
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

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();

    const double pf = (double)period;
    const double invp = 1.0 / pf;
    const double keep = 1.0 - invp;

    double prev_h = high[first_valid];
    double prev_l = low[first_valid];
    double prev_c = close[first_valid];

    const int start = first_valid + 1;
    const int stop  = first_valid + period;

    double plus_dm_sum = 0.0;
    double tr_sum = 0.0;

    for (int i = start; i < stop; ++i) {
        const double ch = high[i];
        const double cl = low[i];
        const double cc = close[i];

        const double dp = ch - prev_h;
        const double dm = prev_l - cl;
        if (dp > dm && dp > 0.0) plus_dm_sum += dp;

        double tr = ch - cl;
        const double tr2 = fabs(ch - prev_c);
        const double tr3 = fabs(cl - prev_c);
        if (tr2 > tr) tr = tr2;
        if (tr3 > tr) tr = tr3;
        tr_sum += tr;

        prev_h = ch;
        prev_l = cl;
        prev_c = cc;
    }

    double cur_plus = plus_dm_sum;
    double cur_tr = tr_sum;

    int idx = stop - 1;
    double scale = (cur_tr == 0.0) ? 0.0 : (100.0 / cur_tr);
    row[idx] = cur_plus * scale;
    idx += 1;

    for (; idx < n; ++idx) {
        const double ch = high[idx];
        const double cl = low[idx];
        const double cc = close[idx];

        const double dp = ch - prev_h;
        const double dm = prev_l - cl;
        const double inc_p = (dp > dm && dp > 0.0) ? dp : 0.0;

        cur_plus = fma(cur_plus, keep, inc_p);

        double tr = ch - cl;
        const double tr2 = fabs(ch - prev_c);
        const double tr3 = fabs(cl - prev_c);
        if (tr2 > tr) tr = tr2;
        if (tr3 > tr) tr = tr3;
        cur_tr = fma(cur_tr, keep, tr);

        scale = (cur_tr == 0.0) ? 0.0 : (100.0 / cur_tr);
        row[idx] = cur_plus * scale;

        prev_h = ch;
        prev_l = cl;
        prev_c = cc;
    }
}
