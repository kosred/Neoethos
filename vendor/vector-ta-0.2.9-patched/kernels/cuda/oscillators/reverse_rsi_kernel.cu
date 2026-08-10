#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

#ifndef RRSI_NAN
#define RRSI_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }


static __device__ __forceinline__ bool is_finite_bits(float x) {
    return ((__float_as_uint(x) & 0x7f800000u) != 0x7f800000u);
}


static __device__ __forceinline__ void kahan_add(float x, float& sum, float& c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

extern "C" __global__ void reverse_rsi_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ lengths,
    const float* __restrict__ levels,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {
#if __CUDA_ARCH__ >= 800


    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos);

    int n = 1;
    float L = 50.0f;
    float* out_row = nullptr;
    if (active) {
        n = lengths[combo];
        L = levels[combo];
        out_row = out + (size_t)combo * (size_t)series_len;
    }

    bool valid = active;
    if (valid) {
        valid = (n > 0) && (L > 0.0f && L < 100.0f) && is_finite_bits(L) &&
                (first_valid >= 0) && (first_valid < series_len);
    }

    int ema_len = 1;
    int warm_end = 0;
    int warm_idx = 0;
    float alpha = 0.0f;
    float neg_scale = 0.0f;
    float n_minus_1 = 0.0f;
    float rs_coeff = 0.0f;
    if (valid) {
        ema_len = (2 * n) - 1;
        const int tail = series_len - first_valid;
        if (tail <= ema_len) {
            valid = false;
        } else {
            warm_end = first_valid + ema_len;
            warm_idx = warm_end - 1;


            alpha = 1.0f / float(n);
            const float inv = 100.0f - L;
            const float rs_target = L / inv;
            neg_scale = inv / L;
            n_minus_1 = float(n - 1);
            rs_coeff = n_minus_1 * rs_target;
        }
    }


    constexpr int TILE = 256;


    extern __shared__ float smem[];
    float* prices_buf = smem;
    float* up_buf     = smem + 2 * TILE;
    float* dn_buf     = smem + 3 * TILE;

    const int num_tiles = (series_len + TILE - 1) / TILE;


    float sum_up = 0.f, c_up = 0.f;
    float sum_dn = 0.f, c_dn = 0.f;
    float up_ema = 0.f, dn_ema = 0.f;


    __shared__ float    prev_carry;
    __shared__ unsigned prev_carry_is_finite;
    if (threadIdx.x == 0) { prev_carry = 0.0f; prev_carry_is_finite = 1u; }
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        const int start = t * TILE;
        const int len   = min(TILE, series_len - start);
        float* p = prices_buf;


        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            p[i] = prices[start + i];
        }
        __syncthreads();


        if (threadIdx.x == 0) {
            float    prev = prev_carry;
            unsigned pfin = prev_carry_is_finite;
            for (int j = 0; j < len; ++j) {
                const int r = start + j;
                const float cur = p[j];
                const unsigned cfin = is_finite_bits(cur);
                float d = 0.0f;
                if (r >= first_valid) {

                    const float    prev_used = (r == first_valid) ? 0.0f : prev;
                    const unsigned p_used_ok = (r == first_valid) ? 1u   : pfin;
                    d = (cfin & p_used_ok) ? (cur - prev_used) : 0.0f;
                    up_buf[j] = d > 0.0f ? d : 0.0f;
                    dn_buf[j] = d < 0.0f ? -d : 0.0f;
                } else {
                    up_buf[j] = 0.0f; dn_buf[j] = 0.0f;
                }
                prev = cur; pfin = cfin;
            }
            if (len > 0) {
                prev_carry = p[len - 1];
                prev_carry_is_finite = is_finite_bits(prev_carry);
            }
        }
        __syncthreads();

        if (active) {
            if (!valid) {

                for (int j = 0; j < len; ++j) {
                    out_row[start + j] = RRSI_NAN;
                }
            } else {

                for (int j = 0; j < len; ++j) {
                    const int r = start + j;

                    if (r < warm_end) {


                        kahan_add(up_buf[j], sum_up, c_up);
                        kahan_add(dn_buf[j], sum_dn, c_dn);

                        if (r == warm_idx) {
                            up_ema = sum_up / float(ema_len);
                            dn_ema = sum_dn / float(ema_len);

                            const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
                            const float m = (x >= 0.0f) ? 1.0f : 0.0f;
                            const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
                            const float v = p[j] + x * scale;
                            out_row[r] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
                        } else {
                            out_row[r] = RRSI_NAN;
                        }
                    } else {

                        up_ema = fmaf(alpha, (up_buf[j] - up_ema), up_ema);
                        dn_ema = fmaf(alpha, (dn_buf[j] - dn_ema), dn_ema);

                        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
                        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
                        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
                        const float v = p[j] + x * scale;
                        out_row[r] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
                    }
                }
            }
        }

        __syncthreads();
    }

#else
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;


    const int n = lengths[combo];
    const float L = levels[combo];
    float* out_row = out + (size_t)combo * (size_t)series_len;


    if (UNLIKELY(n <= 0 || !(L > 0.0f && L < 100.0f) || !is_finite_bits(L))) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }

    const int ema_len = (2 * n) - 1;
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail <= ema_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }


    const int warm_end = first_valid + ema_len;
    const int warm_idx = warm_end - 1;


    const float alpha = 1.0f / float(n);
    const float inv   = 100.0f - L;
    const float rs_target = L / inv;
    const float neg_scale = inv / L;
    const float n_minus_1 = float(n - 1);
    const float rs_coeff  = n_minus_1 * rs_target;


    for (int i = 0; i < warm_idx; ++i) out_row[i] = RRSI_NAN;

    float sum_up = 0.f, c_up = 0.f;
    float sum_dn = 0.f, c_dn = 0.f;

    float prev = 0.0f;
    for (int r = first_valid; r < warm_end; ++r) {
        const float cur = prices[r];
        const unsigned ok = (is_finite_bits(cur) & is_finite_bits(prev));
        const float d = ok ? (cur - prev) : 0.0f;
        kahan_add((d > 0.f) ? d : 0.f, sum_up, c_up);
        kahan_add((d < 0.f) ? -d : 0.f, sum_dn, c_dn);
        prev = cur;
    }
    float up_ema = sum_up / float(ema_len);
    float dn_ema = sum_dn / float(ema_len);


    {
        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
        const float v = prices[warm_idx] + x * scale;
        out_row[warm_idx] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
    }

    float prevd = prices[warm_idx];
    for (int r = warm_end; r < series_len; ++r) {
        const float cur = prices[r];
        const unsigned ok = (is_finite_bits(cur) & is_finite_bits(prevd));
        const float d = ok ? (cur - prevd) : 0.0f;
        const float up = (d > 0.f) ? d : 0.0f;
        const float dn = (d < 0.f) ? -d : 0.0f;
        up_ema = fmaf(alpha, (up - up_ema), up_ema);
        dn_ema = fmaf(alpha, (dn - dn_ema), dn_ema);
        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
        const float v = cur + x * scale;
        out_row[r] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
        prevd = cur;
    }
#endif
}


extern "C" __global__ void reverse_rsi_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int rsi_length,
    float rsi_level,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(rsi_length <= 0 || !(rsi_level > 0.0f && rsi_level < 100.0f) || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = RRSI_NAN;
        return;
    }
    const int ema_len = (2 * rsi_length) - 1;
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= ema_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = RRSI_NAN;
        return;
    }

    const int warm_end = fv + ema_len;
    const int warm_idx = warm_end - 1;


    {
        float* o = out_tm + series;
        for (int r = 0; r < warm_idx; ++r, o += num_series) *o = RRSI_NAN;
    }


    const float nf = static_cast<float>(rsi_length);
    const float n_minus_1 = nf - 1.0f;
    const float inv = 100.0f - rsi_level;
    const float rs_target = rsi_level / inv;
    const float rs_coeff = n_minus_1 * rs_target;
    const float neg_scale = inv / rsi_level;
    const float alpha = 1.0f / nf;


    float sum_up = 0.0f, c_up = 0.0f;
    float sum_dn = 0.0f, c_dn = 0.0f;
    float prev = 0.0f;
    for (int r = fv; r < warm_end; ++r) {
        const float cf = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        const unsigned ok = (is_finite_bits(cf) & is_finite_bits(prev));
        const float d = ok ? (cf - prev) : 0.0f;
        const float up = d > 0.0f ? d : 0.0f;
        const float dn = d < 0.0f ? -d : 0.0f;
        kahan_add(up, sum_up, c_up);
        kahan_add(dn, sum_dn, c_dn);
        prev = cf;
    }
    float up_ema = sum_up / static_cast<float>(ema_len);
    float dn_ema = sum_dn / static_cast<float>(ema_len);


    {
        const float base = *(prices_tm + static_cast<size_t>(warm_idx) * num_series + series);
        const float x0 = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m0 = (x0 >= 0.0f) ? 1.0f : 0.0f;
        const float scale0 = fmaf(m0, (1.0f - neg_scale), neg_scale);
        const float v0 = base + x0 * scale0;
        *(out_tm + static_cast<size_t>(warm_idx) * num_series + series) =
            (is_finite_bits(v0) || x0 >= 0.0f) ? v0 : 0.0f;
    }


    float prevd = *(prices_tm + static_cast<size_t>(warm_idx) * num_series + series);
    for (int r = warm_end; r < series_len; ++r) {
        const float cf = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        const unsigned ok = (is_finite_bits(cf) & is_finite_bits(prevd));
        const float d = ok ? (cf - prevd) : 0.0f;
        const float up = d > 0.0f ? d : 0.0f;
        const float dn = d < 0.0f ? -d : 0.0f;
        up_ema = fmaf(alpha, (up - up_ema), up_ema);
        dn_ema = fmaf(alpha, (dn - dn_ema), dn_ema);
        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
        const float v = cf + x * scale;
        *(out_tm + static_cast<size_t>(r) * num_series + series) =
            (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
        prevd = cf;
    }
}


// ===========================================================================
// S3 f64 LANE — reverse_rsi
// ===========================================================================
// Reference: src/indicators/reverse_rsi.rs
//   reverse_rsi_prepare (:346)                    — first_valid + Err branches
//   reverse_rsi_compute_into_scalar_safe (:387)   — the arithmetic
// Batch defaults: rsi_length 14 (the SWEPT parameter), rsi_level 50.0, source
// close. ema_len = 2*rsi_length - 1, and the warmup is first + ema_len — NOT
// first + rsi_length - 1.
//
// THE all_finite FLAG IS COMPUTED ONCE AND USED AS A PREDICATE (:408, :415).
// When the tail is fully finite the difference is taken unconditionally; when
// it is not, each bar re-tests cur and prev and substitutes 0.0 for the
// difference. The two are the same on clean data and different on the first bar
// after a gap, so the scan is reproduced rather than assumed away.
//
// prev SEEDS TO 0.0, NOT TO data[first] (:412). So the very first difference is
// data[first] - 0.0 — the whole price, not a return. That looks like a bug and
// is not this kernel's to fix: it sets sum_up for the seed window and therefore
// every subsequent EMA value.
//
// NaN SEMANTICS. d.max(0.0) and (-d).max(0.0) are f64::max, which RETURNS THE
// NON-NaN OPERAND — a NaN d yields 0.0, not NaN. fmax below has exactly that
// behaviour; an if-chain would let the NaN into the EMA and poison every
// remaining bar. This is the adx-class bug, and it is why fmax is correct here
// and the if-chain was correct in di.
//
// THE OUTPUT GUARD IS NOT A CLAMP (:434, :454):
//     out = if v.is_finite() || x >= 0.0 { v } else { 0.0 }
// A non-finite v is kept when x >= 0 and replaced by 0.0 only when x < 0.
//
// ROUNDING.
//   x      = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema)  — ONE fma
//   up_ema = beta.mul_add(up_ema, alpha * up)               — product then fma
//   scale  = neg_scale + m * (1.0 - neg_scale)              — a BRANCHLESS
//            lerp on the 0/1 mask m, not an if. Written the same way so the
//            rounding matches even where the two agree algebraically.
//
// One thread per column.
// ===========================================================================

#define NEO_S3_RRSI_LEVEL 50.0

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_reverse_rsi_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int rsi_length = periods[r];
    const double rsi_level = NEO_S3_RRSI_LEVEL;

    const long long ema_len_ll = 2LL * (long long)rsi_length - 1LL;
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (rsi_length == 0) || (rsi_length > n) ||
        !(0.0 < rsi_level && rsi_level < 100.0) ||
        (ema_len_ll < 1) ||
        ((long long)(n - first_valid) < ema_len_ll);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int ema_len = (int)ema_len_ll;
    const int warm_end = first_valid + ema_len;
    const int warm_idx = warm_end - 1;

    for (int i = 0; i < warm_idx && i < n; ++i) row[i] = neo_s3_qnan();

    const double l = rsi_level;
    const double inv = 100.0 - l;
    const double n_minus_1 = (double)(rsi_length - 1);
    const double rs_target = l / inv;
    const double neg_scale = inv / l;
    const double rs_coeff = n_minus_1 * rs_target;

    const double alpha = 2.0 / ((double)ema_len + 1.0);
    const double beta = 1.0 - alpha;

    bool all_finite = true;
    for (int i = first_valid; i < n; ++i) {
        if (!isfinite(data[i])) { all_finite = false; break; }
    }

    double sum_up = 0.0, sum_dn = 0.0;
    double prev = 0.0;                    // reverse_rsi.rs:412 — seeds to ZERO
    for (int i = first_valid; i < warm_end; ++i) {
        const double cur = data[i];
        const double d = (all_finite || (isfinite(cur) && isfinite(prev)))
            ? (cur - prev) : 0.0;
        sum_up += fmax(d, 0.0);           // f64::max — NaN yields the 0.0 side
        sum_dn += fmax(-d, 0.0);
        prev = cur;
    }

    double up_ema = sum_up / (double)ema_len;
    double dn_ema = sum_dn / (double)ema_len;

    const double base = data[warm_idx];
    const double x0 = fma(rs_coeff, dn_ema, -n_minus_1 * up_ema);
    const double m0 = (x0 >= 0.0) ? 1.0 : 0.0;
    const double scale0 = neg_scale + m0 * (1.0 - neg_scale);
    const double v0 = base + x0 * scale0;
    row[warm_idx] = (isfinite(v0) || x0 >= 0.0) ? v0 : 0.0;

    prev = base;
    for (int i = warm_end; i < n; ++i) {
        const double cur = data[i];
        const double d = (all_finite || (isfinite(cur) && isfinite(prev)))
            ? (cur - prev) : 0.0;
        const double up = fmax(d, 0.0);
        const double dn = fmax(-d, 0.0);

        up_ema = fma(beta, up_ema, alpha * up);
        dn_ema = fma(beta, dn_ema, alpha * dn);

        const double x = fma(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const double m = (x >= 0.0) ? 1.0 : 0.0;
        const double scale = neg_scale + m * (1.0 - neg_scale);
        const double v = cur + x * scale;
        row[i] = (isfinite(v) || x >= 0.0) ? v : 0.0;
        prev = cur;
    }
}
