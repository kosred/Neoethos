#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef EMD_NAN
#define EMD_NAN (__int_as_float(0x7fffffff))
#endif


struct KahanF {
    float sum;
    float c;
    __device__ __forceinline__ void init() { sum = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float t = sum + x;
        if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
        else                        c += (x - t) + sum;
        sum = t;
    }
    __device__ __forceinline__ void sub(float x) { add(-x); }
    __device__ __forceinline__ float value() const { return sum + c; }
};


__device__ __forceinline__ float clampcos(float x) {
    const float eps = 1e-6f;
    return fmaxf(fminf(x, 1.0f - eps), -1.0f + eps);
}


extern "C" __global__ void emd_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ periods,
    const float* __restrict__ deltas,
    const float* __restrict__ fractions,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ ub_out,
    float* __restrict__ mb_out,
    float* __restrict__ lb_out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ ub_row = ub_out + base;
    float* __restrict__ mb_row = mb_out + base;
    float* __restrict__ lb_row = lb_out + base;


    const int fv = first_valid;
    const int per_up_low = 50;
    const int period = periods[combo];
    if (period <= 0 || fv < 0 || fv >= series_len) return;
    const int per_mid    = 2 * period;
    const int warm_ul = min(series_len, fv + per_up_low - 1);
    const int warm_mid = min(series_len, fv + per_mid - 1);
    for (int i0 = 0; i0 < warm_ul; ++i0) {
        ub_row[i0] = EMD_NAN;
        lb_row[i0] = EMD_NAN;
    }


    const float delta     = deltas[combo];
    const float fraction  = fractions[combo];


    const float beta  = cospif(2.0f / (float)period);
    const float cos4  = clampcos(cospif(4.0f * delta / (float)period));
    const float gamma = 1.0f / cos4;
    const float alpha = gamma - sqrtf(fmaxf(gamma * gamma - 1.0f, 0.0f));
    const float half_one_minus_alpha      = 0.5f * (1.0f - alpha);
    const float beta_times_one_plus_alpha = beta  * (1.0f + alpha);

    const float   inv_up_low = 1.0f / (float)per_up_low;
    const float   inv_mid    = 1.0f / (float)per_mid;


    extern __shared__ float smem[];
    float* __restrict__ ring_sp_all = smem;
    float* __restrict__ ring_sv_all = smem + (size_t)blockDim.x * per_up_low;
    const int ring_base = threadIdx.x * per_up_low;

    KahanF sum_up;  sum_up.init();
    KahanF sum_low; sum_low.init();
    KahanF sum_mid; sum_mid.init();

    float bp_prev1 = 0.0f, bp_prev2 = 0.0f;
    float peak_prev = 0.0f, valley_prev = 0.0f;
    float price_prev1 = 0.0f, price_prev2 = 0.0f;

    int i = fv;
    if (i < series_len) {
        const float p0 = prices[i];
        bp_prev1 = p0; bp_prev2 = p0; peak_prev = p0; valley_prev = p0;
        price_prev1 = p0; price_prev2 = p0;
    }
    int count = 0;
    int idx50 = 0;

    for (; i < series_len; ++i) {
        const float price = prices[i];

        const float bp_curr = (count >= 2)
            ? fmaf(half_one_minus_alpha, (price - price_prev2),
                   fmaf(beta_times_one_plus_alpha, bp_prev1, (-alpha) * bp_prev2))
            : price;

        float peak_curr = peak_prev;
        float valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr   = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const float sp = peak_curr   * fraction;
        const float sv = valley_curr * fraction;


        if (count + 1 > per_up_low) {
            sum_up.sub(ring_sp_all[ring_base + idx50]);
            sum_low.sub(ring_sv_all[ring_base + idx50]);
        }
        ring_sp_all[ring_base + idx50] = sp;
        ring_sv_all[ring_base + idx50] = sv;
        sum_up.add(sp);
        sum_low.add(sv);
        idx50++; if (idx50 == per_up_low) idx50 = 0;


        sum_mid.add(bp_curr);
        if (count + 1 > per_mid) {
            sum_mid.sub(mb_row[i - per_mid]);
        }


        mb_row[i] = bp_curr;


        if (count + 1 >= per_up_low) {
            ub_row[i] = sum_up.value()  * inv_up_low;
            lb_row[i] = sum_low.value() * inv_up_low;
        }
        if (count + 1 >= per_mid) {
            mb_row[i] = sum_mid.value() * inv_mid;
        }

        bp_prev2 = bp_prev1; bp_prev1 = bp_curr;
        price_prev2 = price_prev1; price_prev1 = price;
        peak_prev = peak_curr; valley_prev = valley_curr;
        ++count;
    }

    for (int i0 = 0; i0 < warm_mid; ++i0) {
        mb_row[i0] = EMD_NAN;
    }
}


extern "C" __global__ void emd_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int period,
    float delta,
    float fraction,
    const int* __restrict__ first_valids,
    float* __restrict__ ub_tm,
    float* __restrict__ mb_tm,
    float* __restrict__ lb_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    float* __restrict__ ub_col = ub_tm + series;
    float* __restrict__ mb_col = mb_tm + series;
    float* __restrict__ lb_col = lb_tm + series;

    const int fv = first_valids[series];
    if (period <= 0 || fv < 0 || fv >= rows) return;
    const int per_up_low = 50;
    const int per_mid = 2 * period;
    const int warm_ul = min(rows, fv + per_up_low - 1);
    const int warm_mid = min(rows, fv + per_mid - 1);
    for (int t = 0; t < warm_ul; ++t) {
        ub_col[(size_t)t * cols] = EMD_NAN;
        lb_col[(size_t)t * cols] = EMD_NAN;
    }


    const float beta  = cospif(2.0f / (float)period);
    const float cos4  = clampcos(cospif(4.0f * delta / (float)period));
    const float gamma = 1.0f / cos4;
    const float alpha = gamma - sqrtf(fmaxf(gamma * gamma - 1.0f, 0.0f));
    const float half_one_minus_alpha      = 0.5f * (1.0f - alpha);
    const float beta_times_one_plus_alpha = beta  * (1.0f + alpha);

    const float   inv_up_low = 1.0f / (float)per_up_low;
    const float   inv_mid    = 1.0f / (float)per_mid;

    extern __shared__ float smem[];
    float* __restrict__ ring_sp_all = smem;
    float* __restrict__ ring_sv_all = smem + (size_t)blockDim.x * per_up_low;
    const int ring_base = threadIdx.x * per_up_low;

    KahanF sum_up;  sum_up.init();
    KahanF sum_low; sum_low.init();
    KahanF sum_mid; sum_mid.init();

    float bp_prev1 = 0.0f, bp_prev2 = 0.0f;
    float peak_prev = 0.0f, valley_prev = 0.0f;
    float price_prev1 = 0.0f, price_prev2 = 0.0f;

    int t = fv;
    if (t < rows) {
        const float p0 = prices_tm[(size_t)t * cols + series];
        bp_prev1 = p0; bp_prev2 = p0; peak_prev = p0; valley_prev = p0;
        price_prev1 = p0; price_prev2 = p0;
    }

    int idx_ul = 0, idx_mid = 0;
    int count = 0;
    int idx50 = 0;
    for (; t < rows; ++t) {
        const float price = prices_tm[(size_t)t * cols + series];

        const float bp_curr = (count >= 2)
            ? fmaf(half_one_minus_alpha, (price - price_prev2),
                   fmaf(beta_times_one_plus_alpha, bp_prev1, (-alpha) * bp_prev2))
            : price;

        float peak_curr = peak_prev;
        float valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr   = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const float sp = peak_curr   * fraction;
        const float sv = valley_curr * fraction;

        if (count + 1 > per_up_low) {
            sum_up.sub(ring_sp_all[ring_base + idx50]);
            sum_low.sub(ring_sv_all[ring_base + idx50]);
        }
        ring_sp_all[ring_base + idx50] = sp;
        ring_sv_all[ring_base + idx50] = sv;
        sum_up.add(sp);
        sum_low.add(sv);
        idx50++; if (idx50 == per_up_low) idx50 = 0;

        sum_mid.add(bp_curr);
        if (count + 1 > per_mid) {
            sum_mid.sub(mb_col[(size_t)(t - per_mid) * cols]);
        }


        mb_col[(size_t)t * cols] = bp_curr;

        if (count + 1 >= per_up_low) {
            ub_col[(size_t)t * cols] = sum_up.value()  * inv_up_low;
            lb_col[(size_t)t * cols] = sum_low.value() * inv_up_low;
        }
        if (count + 1 >= per_mid) {
            mb_col[(size_t)t * cols] = sum_mid.value() * inv_mid;
        }

        bp_prev2 = bp_prev1; bp_prev1 = bp_curr;
        price_prev2 = price_prev1; price_prev1 = price;
        peak_prev = peak_curr; valley_prev = valley_curr;
        ++count;
    }

    for (int t0 = 0; t0 < warm_mid; ++t0) {
        mb_col[(size_t)t0 * cols] = EMD_NAN;
    }
}

// ===========================================================================
// f64 LANE  --  closer 6
//
// CPU reference: `emd_scalar_into` (src/indicators/emd.rs:518), reached from
// `emd_with_kernel` (:404) -> `emd_compute_into` (:370). The batch entry
// `compute_emd_batch` (dispatch/cpu_batch.rs:14526) builds the input with
// `EmdInput::from_slices`, so `emd_price_source` (:362) returns None -- it
// only fires for the `Candles` arm -- and the HIGH/LOW path is the one that
// runs. That is why this kernel takes (high, low) and forms
// `price = (h + l) * 0.5` itself rather than taking an hl2 series.
//
// OUTPUTS: canonical `upperband`, `middleband`, and `lowerband`. The retired
// unversioned `value`/`upper`/`middle`/`lower` spellings are not production
// identities. The resident entry below materializes all three registry
// receipts in one launch; the preserved primary ABI remains upper-only.
//
// PERIOD-SWEPT, with delta and fraction PINNED at the CPU defaults 0.5 and
// 0.1 (`get_f64_param("emd", params, "delta", 0.5)` /
// `get_f64_param("emd", params, "fraction", 0.1)`, cpu_batch.rs:14533-14534).
// The f64 lane sweeps `period` alone, so those two are compile-time constants
// here for the same reason cksp's p/x/q are: inventing a mapping from the
// swept int onto them would compute something the CPU never computes.
//
// SEQUENTIAL, ONE THREAD PER COLUMN. Five carried scalars (bp_prev1,
// bp_prev2, peak_prev, valley_prev, price_prev1/price_prev2) plus three
// sliding sums, and the bandpass is a 2-pole IIR. No scan reformulation is
// bit-faithful.
//
// THE f32 LANE ABOVE USES KAHAN SUMMATION AND THE CPU DOES NOT. `KahanF`
// (:14-26) compensates every add; `emd_scalar_into` writes the bare
// `sum_up += sp - old_sp` (:610). Kahan is MORE accurate and therefore WRONG
// here -- it produces a different number from the oracle. This kernel
// reproduces the bare running sum.
//
// THE f32 EPSILON IS DELETED, NOT RESCALED. `clampcos` (:28-31) clamps the
// cosine argument away from +/-1 by `1e-6f`. The CPU has NO such clamp: it
// writes `gamma = 1.0 / (two_pi * 2.0 * delta / period).cos()` (:542)
// straight and lets an infinite gamma flow. Rescaling `1e-6f` to an
// f64-sized epsilon would have kept a guard the oracle does not have; the
// correct f64 answer is to have no guard at all.
//
// NaN: the peak/valley updates are COMPARISONS in the CPU
// (`bp_prev1 > bp_curr && bp_prev1 > bp_prev2`, :597-602) and a Rust `>`
// against NaN is false exactly as a CUDA `>` is, so the carried peak
// survives unchanged on both sides. These are NOT `f64::max` sites and
// converting them to fmax would change the answer. There is no `f64::max`
// anywhere in `emd_scalar_into`.
//
// f32 -> f64 audit of this section: no f32 literal, no f32-suffixed math
// function (the f32 lane above uses fabsf x2, fmaxf, fminf, cosf, sqrtf), no
// fast-math intrinsic. `EMD_NAN` above is `__int_as_float(0x7fffffff)`, an
// f32 bit pattern; this section builds its quiet NaN with
// `__longlong_as_double(0x7ff8000000000000ULL)`.
// ===========================================================================

// The bp ring is `2 * period` long (`per_mid`, :536). 512 is the bound
// devstop / sama / nama already carry (S2_RING_MAX_PERIOD), so the ring is
// 1024 doubles. An oversized period is REFUSED BY NAME in the wrapper rather
// than truncated here or moved to the host.
#define EMD_F64_MAX_PERIOD 512

// `get_f64_param("emd", params, "delta", 0.5)` -- cpu_batch.rs:14533.
#define EMD_F64_DELTA 0.5
// `get_f64_param("emd", params, "fraction", 0.1)` -- cpu_batch.rs:14534.
#define EMD_F64_FRACTION 0.1
#define EMD_F64_COEFFICIENTS_PER_ROW 6

static __device__ __forceinline__ double emd_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// One complete f64 EMD state authority. The production entry receives the
// scalar CPU's exact immutable coefficient bits; the compatibility primary
// below constructs the same six-slot shape locally without changing its ABI.
static __device__ __forceinline__
void emd_row_f64(const double* __restrict__ high,
                 const double* __restrict__ low,
                 int n,
                 int period,
                 int first_valid,
                 const double* __restrict__ coefficients,
                 double* __restrict__ sp_ring,
                 double* __restrict__ sv_ring,
                 double* __restrict__ bp_ring,
                 double* __restrict__ upper_row,
                 double* __restrict__ middle_row,
                 double* __restrict__ lower_row)
{
    const double nan_d = emd_qnan_f64();
    const int per_up_low = 50;
    if (period <= 0 || period > n || period > EMD_F64_MAX_PERIOD ||
        first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) {
            if (upper_row != nullptr) upper_row[i] = nan_d;
            if (middle_row != nullptr) middle_row[i] = nan_d;
            if (lower_row != nullptr) lower_row[i] = nan_d;
        }
        return;
    }
    const int per_mid = 2 * period;
    const int needed = (per_mid > per_up_low) ? per_mid : per_up_low;
    if (n - first_valid < needed) {
        for (int i = 0; i < n; ++i) {
            if (upper_row != nullptr) upper_row[i] = nan_d;
            if (middle_row != nullptr) middle_row[i] = nan_d;
            if (lower_row != nullptr) lower_row[i] = nan_d;
        }
        return;
    }

    const int warm_ul = first_valid + per_up_low - 1;
    const int warm_mid = first_valid + per_mid - 1;
    for (int i = 0; i < n && i < warm_ul; ++i) {
        if (upper_row != nullptr) upper_row[i] = nan_d;
        if (lower_row != nullptr) lower_row[i] = nan_d;
    }
    for (int i = 0; i < n && i < warm_mid; ++i) {
        if (middle_row != nullptr) middle_row[i] = nan_d;
    }

    const double inv_up_low = coefficients[0];
    const double inv_mid = coefficients[1];
    const double alpha = coefficients[2];
    const double half_one_minus_alpha = coefficients[3];
    const double beta_times_one_plus_alpha = coefficients[4];
    const double fraction = coefficients[5];

    for (int k = 0; k < per_up_low; ++k) {
        sp_ring[k] = 0.0;
        sv_ring[k] = 0.0;
    }
    for (int k = 0; k < per_mid; ++k) bp_ring[k] = 0.0;

    int idx_ul = 0;
    int idx_mid = 0;
    double sum_up = 0.0;
    double sum_low = 0.0;
    double sum_mb = 0.0;
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    double peak_prev = 0.0;
    double valley_prev = 0.0;
    double price_prev1 = 0.0;
    double price_prev2 = 0.0;

    {
        const double p0 = (high[first_valid] + low[first_valid]) * 0.5;
        bp_prev1 = p0;
        bp_prev2 = p0;
        peak_prev = p0;
        valley_prev = p0;
        price_prev1 = p0;
        price_prev2 = p0;
    }

    int count = 0;
    for (int i = first_valid; i < n; ++i) {
        const double price = (high[i] + low[i]) * 0.5;
        double bp_curr;
        if (count >= 2) {
            bp_curr = half_one_minus_alpha * (price - price_prev2)
                    + beta_times_one_plus_alpha * bp_prev1
                    - alpha * bp_prev2;
        } else {
            bp_curr = price;
        }

        double peak_curr = peak_prev;
        double valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const double sp = peak_curr * fraction;
        const double sv = valley_curr * fraction;
        const double old_sp = sp_ring[idx_ul];
        const double old_sv = sv_ring[idx_ul];
        const double old_bp = bp_ring[idx_mid];
        sp_ring[idx_ul] = sp;
        sv_ring[idx_ul] = sv;
        bp_ring[idx_mid] = bp_curr;
        sum_up = sum_up + (sp - old_sp);
        sum_low = sum_low + (sv - old_sv);
        sum_mb = sum_mb + (bp_curr - old_bp);

        ++idx_ul;
        if (idx_ul == per_up_low) idx_ul = 0;
        ++idx_mid;
        if (idx_mid == per_mid) idx_mid = 0;

        const int filled = count + 1;
        if (filled >= per_up_low) {
            if (upper_row != nullptr) upper_row[i] = sum_up * inv_up_low;
            if (lower_row != nullptr) lower_row[i] = sum_low * inv_up_low;
        }
        if (filled >= per_mid && middle_row != nullptr) {
            middle_row[i] = sum_mb * inv_mid;
        }

        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
        price_prev2 = price_prev1;
        price_prev1 = price;
        ++count;
    }
}

// Canonical production ABI: one resident launch emits all three named output
// matrices from runtime-sized three-ring scratch. No price or output crosses
// the host boundary.
extern "C" __global__
void emd_outputs_f64(const double* __restrict__ high,
                     const double* __restrict__ low,
                     int n,
                     const int* __restrict__ periods,
                     const double* __restrict__ coefficients,
                     int n_combos,
                     int first_valid,
                     int bp_stride,
                     double* __restrict__ sp_rings,
                     double* __restrict__ sv_rings,
                     double* __restrict__ bp_rings,
                     double* __restrict__ upper_out,
                     double* __restrict__ middle_out,
                     double* __restrict__ lower_out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    const size_t output_base = static_cast<size_t>(r) * static_cast<size_t>(n);
    emd_row_f64(
        high,
        low,
        n,
        periods[r],
        first_valid,
        coefficients + static_cast<size_t>(r) * EMD_F64_COEFFICIENTS_PER_ROW,
        sp_rings + static_cast<size_t>(r) * 50ULL,
        sv_rings + static_cast<size_t>(r) * 50ULL,
        bp_rings + static_cast<size_t>(r) * static_cast<size_t>(bp_stride),
        upper_out + output_base,
        middle_out + output_base,
        lower_out + output_base);
}

extern "C" __global__
void emd_batch_f64(const double* __restrict__ high,
                   const double* __restrict__ low,
                   int n,
                   const int* __restrict__ periods,
                   int n_combos,
                   int first_valid,
                   double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    const int period = periods[r];
    if (period <= 0 || period > n || period > EMD_F64_MAX_PERIOD ||
        first_valid < 0 || first_valid >= n) {
        const double nan_d = emd_qnan_f64();
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }
    const int per_mid = 2 * period;
    const double two_pi = 6.283185307179586476925286766559;
    const double beta = cos(two_pi / static_cast<double>(period));
    const double gamma = 1.0 / cos(two_pi * 2.0 * EMD_F64_DELTA
                                   / static_cast<double>(period));
    const double alpha = gamma - sqrt(gamma * gamma - 1.0);
    const double coefficients[EMD_F64_COEFFICIENTS_PER_ROW] = {
        1.0 / 50.0,
        1.0 / static_cast<double>(per_mid),
        alpha,
        0.5 * (1.0 - alpha),
        beta * (1.0 + alpha),
        EMD_F64_FRACTION,
    };
    double sp_ring[50];
    double sv_ring[50];
    double bp_ring[2 * EMD_F64_MAX_PERIOD];
    emd_row_f64(high, low, n, period, first_valid, coefficients,
                sp_ring, sv_ring, bp_ring, row, nullptr, nullptr);
}
