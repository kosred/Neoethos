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
// OUTPUT: `upperband`. `compute_emd_batch:14554` maps output_id "value" onto
// `out.upperband`, and `OUTPUTS_UPPER_MIDDLE_LOWER_BAND` (registry.rs:12159)
// lists upper first. middleband and lowerband are the same walk with the
// other two running sums; they are one launch away once the lane grows an
// output selector.
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

static __device__ __forceinline__ double emd_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
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

    const double nan_d = emd_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);

    const int period = periods[r];
    const int per_up_low = 50;

    // `emd_prepare` (:331-352) errors -- and `collect_f64` turns the error
    // into an all-NaN column -- when the period is out of range or the valid
    // tail is shorter than max(2 * period, 50).
    if (period <= 0 || period > n || period > EMD_F64_MAX_PERIOD ||
        first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }
    const int per_mid = 2 * period;
    const int needed = (per_mid > per_up_low) ? per_mid : per_up_low;   // :340
    if (n - first_valid < needed) {
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }

    // `alloc_with_nan_prefix(len, first + 50 - 1)` -- :407, :410.
    const int warm = first_valid + per_up_low - 1;
    for (int i = 0; i < n && i < warm; ++i) row[i] = nan_d;

    const double inv_up_low = 1.0 / static_cast<double>(per_up_low);     // :537

    const double two_pi = 6.283185307179586476925286766559;              // :540
    const double beta  = cos(two_pi / static_cast<double>(period));      // :541
    const double gamma = 1.0 / cos(two_pi * 2.0 * EMD_F64_DELTA
                                   / static_cast<double>(period));       // :542
    const double alpha = gamma - sqrt(gamma * gamma - 1.0);              // :543
    const double half_one_minus_alpha      = 0.5 * (1.0 - alpha);        // :544
    const double beta_times_one_plus_alpha = beta * (1.0 + alpha);       // :545

    double sp_ring[50];
    double bp_ring[2 * EMD_F64_MAX_PERIOD];
    for (int k = 0; k < per_up_low; ++k) sp_ring[k] = 0.0;               // :547-548
    for (int k = 0; k < per_mid; ++k) bp_ring[k] = 0.0;                  // :549

    int idx_ul = 0;
    int idx_mid = 0;

    double sum_up = 0.0;                                                 // :554
    // sum_low and sum_mb drive the two columns this lane does not emit. The
    // rings they read still have to advance identically, so bp_ring is kept
    // and only the sums feeding `upperband` are carried.

    double bp_prev1 = 0.0, bp_prev2 = 0.0;
    double peak_prev = 0.0, valley_prev = 0.0;
    double price_prev1 = 0.0, price_prev2 = 0.0;

    // :572-580 -- the seed reads bar `first` and primes every carried scalar
    // with it BEFORE the walk, which then reads bar `first` again.
    {
        const double p0 = (high[first_valid] + low[first_valid]) * 0.5;
        bp_prev1 = p0; bp_prev2 = p0;
        peak_prev = p0; valley_prev = p0;
        price_prev1 = p0; price_prev2 = p0;
    }

    int count = 0;
    for (int i = first_valid; i < n; ++i) {
        const double price = (high[i] + low[i]) * 0.5;                   // :585

        // :587-591. Left to right, three separate roundings, exactly as the
        // Rust expression `a * b + c * d - e * f` evaluates. -fmad=false in
        // F64_LANE_SOURCES forbids the compiler from contracting any of them.
        double bp_curr;
        if (count >= 2) {
            bp_curr = half_one_minus_alpha * (price - price_prev2)
                    + beta_times_one_plus_alpha * bp_prev1
                    - alpha * bp_prev2;
        } else {
            bp_curr = price;
        }

        double peak_curr = peak_prev;                                    // :594
        double valley_curr = valley_prev;                                // :595
        if (count >= 2) {                                                // :596-603
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const double sp = peak_curr * EMD_F64_FRACTION;                  // :605

        const double old_sp = sp_ring[idx_ul];                           // :608
        sp_ring[idx_ul] = sp;
        bp_ring[idx_mid] = bp_curr;

        sum_up = sum_up + (sp - old_sp);                                 // :610

        ++idx_ul; if (idx_ul == per_up_low) idx_ul = 0;                  // :615-618
        ++idx_mid; if (idx_mid == per_mid) idx_mid = 0;                  // :619-622

        const int filled = count + 1;                                    // :624
        if (filled >= per_up_low) {
            row[i] = sum_up * inv_up_low;                                // :626
        }

        bp_prev2 = bp_prev1;                                             // :634
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
        price_prev2 = price_prev1;
        price_prev1 = price;

        ++count;
    }
}
