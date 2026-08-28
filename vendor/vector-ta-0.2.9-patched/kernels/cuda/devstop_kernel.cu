#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }
__device__ __forceinline__ int wrap_inc(int x, int cap) { int nx = x + 1; return (nx == cap) ? 0 : nx; }


struct kahan_t { float s, c; };
__device__ __forceinline__ void kahan_add(kahan_t &K, float x) {
    float y = x - K.c;
    float t = K.s + y;
    K.c = (t - K.s) - y;
    K.s = t;
}
__device__ __forceinline__ void kahan_sub(kahan_t &K, float x) { kahan_add(K, -x); }


struct ds_t { float hi, lo; };
__device__ __forceinline__ ds_t ds_from2(float2 v) { ds_t r; r.hi = v.x; r.lo = v.y; return r; }
__device__ __forceinline__ float ds_to_f(ds_t a) { return a.hi + a.lo; }
__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b; float bb = s - a; e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b; e = b - (s - a);
}
__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b; e = __fmaf_rn(a, b, -p);
}
__device__ __forceinline__ ds_t ds_add(ds_t x, ds_t y) {
    float s1, e1; two_sum(x.hi, y.hi, s1, e1);
    float s2, e2; two_sum(x.lo, y.lo, s2, e2);
    float s3, e3; two_sum(s1, s2, s3, e3);
    float e = e1 + e2 + e3; float hi, lo; quick_two_sum(s3, e, hi, lo); ds_t r{hi, lo}; return r;
}
__device__ __forceinline__ ds_t ds_sub(ds_t x, ds_t y) { ds_t r{ x.hi - y.hi, x.lo - y.lo };
    float s, e; two_sum(x.hi, -y.hi, s, e); float t, f; two_sum(x.lo, -y.lo, t, f); float u, g; two_sum(s, t, u, g);
    float hi, lo; quick_two_sum(u, e + f + g, hi, lo); return ds_t{hi, lo}; }
__device__ __forceinline__ ds_t ds_scale(ds_t x, float b) {
    float p, e; two_prod(x.hi, b, p, e); float s, t; two_sum(p, x.lo * b, s, t); float hi, lo; quick_two_sum(s, e + t, hi, lo); return ds_t{hi, lo};
}
__device__ __forceinline__ ds_t ds_mul(ds_t x, ds_t y) {
    float p, e; two_prod(x.hi, y.hi, p, e);
    float c1 = __fmaf_rn(x.hi, y.lo, 0.0f); float c2 = __fmaf_rn(x.lo, y.hi, 0.0f);
    float s, t; two_sum(p, c1 + c2, s, t); float err = e + t + __fmaf_rn(x.lo, y.lo, 0.0f);
    float hi, lo; quick_two_sum(s, err, hi, lo); return ds_t{hi, lo};
}

extern "C" __global__ void devstop_build_range_prefixes_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    float2* __restrict__ p1,
    float2* __restrict__ p2,
    int* __restrict__ pc
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    ds_t s1{0.0f, 0.0f};
    ds_t s2{0.0f, 0.0f};
    int accc = 0;
    float prev_h = (first_valid >= 0 && first_valid < len) ? high[first_valid] : qnan32();
    float prev_l = (first_valid >= 0 && first_valid < len) ? low[first_valid] : qnan32();

    if (len >= 0) {
        p1[0] = make_float2(0.0f, 0.0f);
        p2[0] = make_float2(0.0f, 0.0f);
        pc[0] = 0;
    }

    for (int i = 0; i < len; ++i) {
        if (i >= first_valid + 1) {
            const float h = high[i];
            const float l = low[i];
            if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
                const float hi2 = (h > prev_h) ? h : prev_h;
                const float lo2 = (l < prev_l) ? l : prev_l;
                const float r = hi2 - lo2;
                s1 = ds_add(s1, ds_t{r, 0.0f});
                s2 = ds_add(s2, ds_t{r * r, 0.0f});
                accc += 1;
            }
            prev_h = h;
            prev_l = l;
        }
        p1[i + 1] = make_float2(s1.hi, s1.lo);
        p2[i + 1] = make_float2(s2.hi, s2.lo);
        pc[i + 1] = accc;
    }
}


extern "C" __global__ void devstop_batch_grouped_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float2* __restrict__ p1,
    const float2* __restrict__ p2,
    const int* __restrict__ pc,
    int len,
    int first_valid,
    int period,
    const float* __restrict__ mults,
    int n_combos,
    int is_long,
    int out_row_base,
    float* __restrict__ out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || period <= 0) return;

    const int warm = first_valid + 2 * period - 1;
    const int row = out_row_base + combo;
    const int row_off = row * len;


    const int warm_clamp = (warm < len) ? warm : len;
    for (int t = threadIdx.x; t < warm_clamp; t += blockDim.x) { out[row_off + t] = qnan32(); }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (warm >= len) return;

    extern __shared__ unsigned char smem[];
    float* base_ring = reinterpret_cast<float*>(smem);
    int* dq_idx = reinterpret_cast<int*>(base_ring + period);
    for (int i = 0; i < period; ++i) { base_ring[i] = qnan32(); dq_idx[i] = 0; }

    int dq_head = 0, dq_len = 0;
    const int cap = period;


    auto dq_back_at = [&](int len_) { int pos = dq_head + len_ - 1; if (pos >= cap) pos -= cap; return dq_idx[pos]; };
    auto dq_push_back = [&](int value) { int pos = dq_head + dq_len; if (pos >= cap) pos -= cap; dq_idx[pos] = value; dq_len += 1; };
    auto dq_pop_back = [&]() { dq_len -= 1; };
    auto dq_pop_front = [&]() { dq_head = wrap_inc(dq_head, cap); dq_len -= 1; };
    auto dq_front = [&]() { return dq_idx[dq_head]; };

    const float mult = mults[combo];
    const int start_base = first_valid + period;
    const int start_final = start_base + period - 1;

    int slot = start_base % period;
    for (int i = start_base; i < len; ++i) {

        const int t1 = i + 1;
        int a = t1 - period; if (a < 0) a = 0;
        const int cnt = pc[t1] - pc[a];
        float base = qnan32();
        if (cnt > 0) {

            ds_t S1 = ds_sub(ds_from2(p1[t1]), ds_from2(p1[a]));
            ds_t S2 = ds_sub(ds_from2(p2[t1]), ds_from2(p2[a]));
            const float inv = 1.0f / (float)cnt;
            ds_t mean_ds = ds_scale(S1, inv);
            ds_t m2_ds   = ds_scale(S2, inv);
            ds_t var_ds  = ds_sub(m2_ds, ds_mul(mean_ds, mean_ds));
            const float mean = ds_to_f(mean_ds);
            float var = ds_to_f(var_ds);
            if (var < 0.0f) var = 0.0f;
            const float sigma = sqrtf(var);
            const float h = high[i];
            const float l = low[i];
            if (is_long) {
                if (!isnan(h)) {
                    base = h - mean - mult * sigma;
                }
            } else {
                if (!isnan(l)) {
                    base = l + mean + mult * sigma;
                }
            }
        }


        if (isnan(base)) { base = is_long ? -INFINITY : INFINITY; }
        base_ring[slot] = base;
        {

            const int cut = i + 1 - period;
            while (dq_len > 0 && dq_front() < cut) dq_pop_front();

            if (is_long) {

                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj <= base) dq_pop_back(); else break;
                }
            } else {

                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj >= base) dq_pop_back(); else break;
                }
            }
            dq_push_back(i);
        }


        const int cut = i + 1 - period;
        while (dq_len > 0 && dq_front() < cut) { dq_pop_front(); }

        if (i >= start_final) {
            float out_val = qnan32();
            if (dq_len > 0) {
                int j = dq_front();
                out_val = base_ring[j % period];
            }
            out[row_off + i] = out_val;
        }
        slot = wrap_inc(slot, cap);
    }
}


extern "C" __global__ void devstop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float mult,
    int is_long,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= cols || period <= 0) return;


    const int fv = first_valids[s];
    const int start_base  = fv + period;
    const int start_final = start_base + period - 1;


    const int warm_clamp = (start_final < rows) ? start_final : rows;
    for (int t = threadIdx.x; t < warm_clamp; t += blockDim.x) { out_tm[t * cols + s] = qnan32(); }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (start_base >= rows) return;

    extern __shared__ unsigned char smem_uc[];
    float* r_ring = reinterpret_cast<float*>(smem_uc);
    float* base_ring = r_ring + period;
    int* dq_idx = reinterpret_cast<int*>(base_ring + period);
    for (int i = 0; i < period; ++i) { r_ring[i] = qnan32(); base_ring[i] = qnan32(); dq_idx[i] = 0; }

    int r_pos = 0; int r_inserted = 0; int cnt = 0;
    kahan_t S1{0.0f, 0.0f}, S2{0.0f, 0.0f};
    float prev_h = high_tm[fv * cols + s];
    float prev_l = low_tm [fv * cols + s];


    for (int k = fv + 1; k < min(start_base, rows); ++k) {
        const float h = high_tm[k * cols + s];
        const float l = low_tm [k * cols + s];
        float r = qnan32();
        if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
            const float hi2 = (h > prev_h) ? h : prev_h;
            const float lo2 = (l < prev_l) ? l : prev_l;
            r = hi2 - lo2;
        }
        r_ring[r_pos] = r; r_pos = (r_pos + 1) % period; r_inserted += 1;
        if (!isnan(r)) { kahan_add(S1, r); kahan_add(S2, __fmaf_rn(r, r, 0.0f)); cnt += 1; }
        prev_h = h; prev_l = l;
    }
    r_pos = (period - 1) % period;

    int dq_head = 0, dq_len = 0; const int cap = period;
    auto dq_back_at = [&](int len_) { int pos = dq_head + len_ - 1; if (pos >= cap) pos -= cap; return dq_idx[pos]; };
    auto dq_push_back = [&](int value) { int pos = dq_head + dq_len; if (pos >= cap) pos -= cap; dq_idx[pos] = value; dq_len += 1; };
    auto dq_pop_back = [&]() { dq_len -= 1; };
    auto dq_pop_front = [&]() { dq_head = wrap_inc(dq_head, cap); dq_len -= 1; };
    auto dq_front = [&]() { return dq_idx[dq_head]; };

    for (int i = start_base; i < rows; ++i) {
        const float h = high_tm[i * cols + s];
        const float l = low_tm [i * cols + s];

        float r_new = qnan32();
        if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
            const float hi2 = (h > prev_h) ? h : prev_h;
            const float lo2 = (l < prev_l) ? l : prev_l;
            r_new = hi2 - lo2;
        }
        prev_h = h; prev_l = l;

        const bool had_full = (r_inserted >= period);
        const float old = had_full ? r_ring[r_pos] : qnan32();
        if (had_full && !isnan(old)) { kahan_sub(S1, old); kahan_sub(S2, __fmaf_rn(old, old, 0.0f)); cnt -= 1; }
        r_ring[r_pos] = r_new; r_pos = (r_pos + 1) % period; r_inserted += 1;
        if (!isnan(r_new)) { kahan_add(S1, r_new); kahan_add(S2, __fmaf_rn(r_new, r_new, 0.0f)); cnt += 1; }

        float base = qnan32();
        if (cnt > 0) {
            const float inv = 1.0f / (float)cnt;
            const float mean = S1.s * inv;
            float var = __fmaf_rn(-mean, mean, S2.s * inv);
            if (var < 0.0f) var = 0.0f;
            const float sigma = sqrtf(var);
            if (is_long) {
                if (!isnan(h)) base = h - mean - mult * sigma;
            } else {
                if (!isnan(l)) base = l + mean + mult * sigma;
            }
        }

        const int slot = i % period;
        if (isnan(base)) { base = is_long ? -INFINITY : INFINITY; }
        base_ring[slot] = base;
        {

            const int cut = i + 1 - period;
            while (dq_len > 0 && dq_front() < cut) dq_pop_front();

            if (is_long) {
                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj <= base) dq_pop_back(); else break;
                }
            } else {
                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj >= base) dq_pop_back(); else break;
                }
            }
            dq_push_back(i);
        }

        if (i >= start_final) {
            float out_val = qnan32();
            if (dq_len > 0) { int j = dq_front(); out_val = base_ring[j % period]; }
            out_tm[i * cols + s] = out_val;
        }
    }
}


// ===========================================================================
// S2 f64 LANE — devstop
// ===========================================================================
// Reference: src/indicators/devstop.rs
//   `devstop_prepare`               (:289) — first = MIN of high's and low's
//                                             first non-NaN
//   `devstop_with_kernel`           (:509) -> `devstop_into_slice` (:357)
//   `devstop_scalar_classic_sma`    (:2270) -> `devstop_scalar_classic_fused
//                                             ::<false>` (:2030) — the branch
//                                             the batch route reaches
//   Batch route: `cpu_batch.rs:15047` sweeps `period`; mult = 0.0,
//   devtype = 0, direction = "long", ma_type = "sma" — so `EMA == false` and
//   the `mult * sigma` term is multiplied by ZERO at the default. The sigma
//   arithmetic is still reproduced exactly, because a caller that passes a
//   non-zero `mult` must get the CPU's number and not a simplification that
//   was only correct at the default.
//
// FIRST-VALID IS THE **MIN** OF TWO SCANS, NOT THE MAX AND NOT A JOINT SCAN.
//   `first = fh.min(fl)` (:320). Every other high/low indicator in this lane
//   takes the first index where BOTH are non-NaN. devstop deliberately starts
//   at the EARLIER of the two and lets the NaN flow into the range term, where
//   the `r.is_finite()` guard drops it from the running moments. Using the
//   common rule would start the series late and change every moment window.
//   The registry declares `AllInputsNonNan` because that is what the host
//   computes; the kernel derives devstop's own `first` from the arrays, as
//   recorded here.
//
// WARMUP IS TWO PERIODS. `devstop_warmup = first + 2*period - 1` (:285):
// `period` bars to fill the range ring and another `period` for the rolling
// extremum of the base line. Half of that would be the obvious wrong answer.
//
// THE VARIANCE IS THE TEXTBOOK UNSTABLE FORM, AND THAT IS THE SPEC.
//   `var = max0((sum2 - (sum*sum) * inv) * inv)` — sum of squares minus the
//   square of the sum. In f32 this cancels catastrophically: for a range
//   series with mean 1e-3 and spread 1e-5, `sum2` and `sum*sum/n` agree to six
//   digits, which is every digit f32 has, and `sqrt` of the difference is
//   noise. This is the single strongest reason devstop must be f64, and it is
//   why the f32 kernel's `sqrtf` x2 could not be fixed by widening the sqrt
//   alone.
//   `max0` is `if x < 0.0 { 0.0 } else { x }` — an IF, so a NaN variance stays
//   NaN. `fmax(x, 0.0)` would return 0.0 and then `sqrt(0)` = 0, turning "no
//   answer" into "zero volatility".
//
// THE RING IS INSERTED AT A POSITION THAT IS **NOT** RESET TO 0.
//   After the init loop the CPU sets `r_ins_pos = (period - 1) % period`
//   (:2107), not 0. The main loop then evicts `r_ring[r_ins_pos]` before
//   writing. Starting at 0 would evict the wrong element for the first
//   `period` bars and the moments would be wrong for the whole series.
// ===========================================================================

#define DEVSTOP_MAX_PERIOD 512

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_devstop_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
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
    const double mult = 0.0;      // DevStopParams::get_mult   -> unwrap_or(0.0)
    const bool is_long = true;    // direction "long"
    // devtype 0 + ma_type "sma" -> devstop_scalar_classic_fused::<false>

    const bool declined =
        (n <= 0) ||
        (period <= 0) || (period > n) || (period > DEVSTOP_MAX_PERIOD) ||
        (first_valid < 0) || (first_valid >= n);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();

    // `first = fh.min(fl)` — the MIN of two independent scans.
    int fh = -1, fl = -1;
    for (int i = 0; i < n; ++i) { if (!isnan(high[i])) { fh = i; break; } }
    for (int i = 0; i < n; ++i) { if (!isnan(low[i]))  { fl = i; break; } }
    if (fh < 0 || fl < 0) return;
    const int first = (fh < fl) ? fh : fl;
    if ((n - first) < period) return;

    const int start_base = first + period;
    const int start_final = start_base + period - 1;
    if (start_base >= n) return;

    double r_ring[DEVSTOP_MAX_PERIOD];
    for (int k = 0; k < period; ++k) r_ring[k] = neo_s2_qnan();
    int r_ins_pos = 0;
    int r_inserted = 0;

    double sum = 0.0;
    double sum2 = 0.0;
    int cnt = 0;

    double prev_h = high[first];
    double prev_l = low[first];
    const int end_init = (start_base < n) ? start_base : n;

    for (int k = first + 1; k < end_init; ++k) {
        const double h = high[k];
        const double l = low[k];
        double rr;
        if (isnan(h) || isnan(l) || isnan(prev_h) || isnan(prev_l)) {
            rr = neo_s2_qnan();
        } else {
            const double hi2 = (h > prev_h) ? h : prev_h;
            const double lo2 = (l < prev_l) ? l : prev_l;
            rr = hi2 - lo2;
        }
        r_ring[r_ins_pos] = rr;
        r_ins_pos += 1;
        r_inserted += 1;
        if (isfinite(rr)) {
            sum += rr;
            sum2 = fma(rr, rr, sum2);
            cnt += 1;
        }
        prev_h = h;
        prev_l = l;
    }
    r_ins_pos = (period - 1) % period;

    double base_ring[DEVSTOP_MAX_PERIOD];
    for (int k = 0; k < period; ++k) base_ring[k] = neo_s2_qnan();

    const int cap = period;
    int dq_buf[DEVSTOP_MAX_PERIOD];
    int dq_head = 0;
    int dq_len = 0;

    for (int i = start_base; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];

        double r_new;
        if (isnan(h) || isnan(l) || isnan(prev_h) || isnan(prev_l)) {
            r_new = neo_s2_qnan();
        } else {
            const double hi2 = (h > prev_h) ? h : prev_h;
            const double lo2 = (l < prev_l) ? l : prev_l;
            r_new = hi2 - lo2;
        }
        prev_h = h;
        prev_l = l;

        const bool had_full = (r_inserted >= period);
        const double old = had_full ? r_ring[r_ins_pos] : neo_s2_qnan();
        if (had_full && isfinite(old)) {
            sum -= old;
            sum2 -= old * old;
            cnt -= 1;
        }

        r_ring[r_ins_pos] = r_new;
        r_ins_pos = (r_ins_pos + 1) % period;
        r_inserted += 1;
        if (isfinite(r_new)) {
            sum += r_new;
            sum2 = fma(r_new, r_new, sum2);
            cnt += 1;
        }

        double avtr, sigma;
        if (cnt == 0) {
            avtr = neo_s2_qnan();
            sigma = neo_s2_qnan();
        } else {
            const double inv = 1.0 / (double)cnt;
            const double mean = sum * inv;
            double var = (sum2 - (sum * sum) * inv) * inv;
            if (var < 0.0) var = 0.0;      // max0 — an if, so NaN survives
            avtr = mean;
            sigma = sqrt(var);
        }

        double base;
        if (is_long) {
            base = (isnan(h) || isnan(avtr) || isnan(sigma))
                 ? neo_s2_qnan()
                 : (h - avtr - mult * sigma);
        } else {
            base = (isnan(l) || isnan(avtr) || isnan(sigma))
                 ? neo_s2_qnan()
                 : (l + avtr + mult * sigma);
        }

        const int bslot = i % period;
        base_ring[bslot] = base;

        if (is_long) {
            while (dq_len > 0) {
                const int j = dq_buf[(dq_head + dq_len - 1) % cap];
                const double bj = base_ring[j % period];
                if (isnan(bj) || bj <= base) dq_len -= 1;
                else break;
            }
        } else {
            while (dq_len > 0) {
                const int j = dq_buf[(dq_head + dq_len - 1) % cap];
                const double bj = base_ring[j % period];
                if (isnan(bj) || bj >= base) dq_len -= 1;
                else break;
            }
        }
        dq_buf[(dq_head + dq_len) % cap] = i;
        dq_len += 1;

        const int cut = i + 1 - period;
        while (dq_len > 0 && dq_buf[dq_head] < cut) {
            dq_head = (dq_head + 1) % cap;
            dq_len -= 1;
        }

        if (i >= start_final) {
            row[i] = (dq_len > 0) ? base_ring[dq_buf[dq_head] % period] : neo_s2_qnan();
        }
    }
}
