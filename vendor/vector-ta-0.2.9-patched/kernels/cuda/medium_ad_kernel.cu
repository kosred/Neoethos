#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef MEDIUM_AD_MAX_PERIOD
#define MEDIUM_AD_MAX_PERIOD 512
#endif


__device__ __forceinline__ float fabsf_fast(float x) {
    return fabsf(x);
}


__device__ __forceinline__ void two_sum_f32(float a, float b, float &s, float &err) {
    s = a + b;
    float z = s - a;
    err = (a - (s - z)) + (b - z);
}


__device__ __forceinline__ float avg2_compensated(float a, float b) {
    float s, e;
    two_sum_f32(a, b, s, e);
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(0.5f, e, 0.5f * s);
#else
    return 0.5f * (s + e);
#endif
}


__device__ __forceinline__ float median3f(float a, float b, float c) {
    float ab = fminf(a, b), AB = fmaxf(a, b);
    float bc = fminf(AB, c), BC = fmaxf(AB, c);
    (void)BC;
    return fmaxf(ab, bc);
}


__device__ __forceinline__ float nth_element_inplace(float* a, int n, int k) {
    int left = 0, right = n - 1;
    while (left < right) {
        const int mid = (left + right) >> 1;
        const float pivot = median3f(a[left], a[mid], a[right]);

        int lt = left, i = left, gt = right;
        while (i <= gt) {
            const float v = a[i];
            if (v < pivot) {
                float tmp = a[lt]; a[lt] = a[i]; a[i] = tmp;
                ++lt; ++i;
            } else if (v > pivot) {
                float tmp = a[i]; a[i] = a[gt]; a[gt] = tmp;
                --gt;
            } else {
                ++i;
            }
        }
        if (k < lt) {
            right = lt - 1;
        } else if (k > gt) {
            left = gt + 1;
        } else {
            return a[k];
        }
    }
    return a[k];
}


__device__ __forceinline__ float median_from_window(const float* __restrict__ orig, int n, float* __restrict__ scratch) {

    for (int i = 0; i < n; ++i) scratch[i] = orig[i];

    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace(scratch, n, k);
    } else {
        const int k = n >> 1;
        const float upper = nth_element_inplace(scratch, n, k);

        float lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) {
            lower = fmaxf(lower, scratch[i]);
        }
        return avg2_compensated(lower, upper);
    }
}


__device__ __forceinline__ float mad_from_window(const float* __restrict__ orig, int n, float* __restrict__ scratch) {

    const float med = median_from_window(orig, n, scratch);


    for (int i = 0; i < n; ++i) {
        scratch[i] = fabsf_fast(orig[i] - med);
    }


    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace(scratch, n, k);
    } else {
        const int k = n >> 1;
        const float upper = nth_element_inplace(scratch, n, k);
        float lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) lower = fmaxf(lower, scratch[i]);
        return avg2_compensated(lower, upper);
    }
}


__device__ __forceinline__ float mad_period_2(float x0, float x1) {

    return 0.5f * fabsf_fast(x1 - x0);
}


extern "C" __global__ void medium_ad_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = nanf("");


    float orig[MEDIUM_AD_MAX_PERIOD];
    float scratch[MEDIUM_AD_MAX_PERIOD];


    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f;

        if (t >= warm) {
            if (period == 1) {
                const float v = data[t];
                out_val = isfinite(v) ? 0.0f : nan_f;
            } else if (period == 2) {
                const float x0 = data[t - 1];
                const float x1 = data[t];
                out_val = (isfinite(x0) && isfinite(x1)) ? mad_period_2(x0, x1) : nan_f;
            } else {
                const int start = t + 1 - period;
                bool has_nan = false;


                #pragma unroll 1
                for (int k = 0; k < period; ++k) {
                    const float v = data[start + k];
                    if (!isfinite(v)) has_nan = true;
                    orig[k] = v;
                }

                if (!has_nan) {
                    out_val = mad_from_window(orig, period, scratch);
                }
            }
        }

        out[row_off + t] = out_val;
        t += stride;
    }
}


extern "C" __global__ void medium_ad_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) {
        const float nan_f = nanf("");
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    const int warm = first_valid + period - 1;
    const float nan_f = nanf("");


    int prefill = warm < rows ? warm : rows;
    for (int t = 0; t < prefill; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (warm >= rows) return;

    float orig[MEDIUM_AD_MAX_PERIOD];
    float scratch[MEDIUM_AD_MAX_PERIOD];

    for (int t = warm; t < rows; ++t) {
        if (period == 1) {
            const float v = data_tm[t * cols + s];
            out_tm[t * cols + s] = isfinite(v) ? 0.0f : nan_f;
            continue;
        }
        if (period == 2) {
            const float x0 = data_tm[(t - 1) * cols + s];
            const float x1 = data_tm[t * cols + s];
            out_tm[t * cols + s] = (isfinite(x0) && isfinite(x1)) ? mad_period_2(x0, x1) : nan_f;
            continue;
        }

        const int start = t + 1 - period;
        bool has_nan = false;
        for (int k = 0; k < period; ++k) {
            const float v = data_tm[(start + k) * cols + s];
            if (!isfinite(v)) has_nan = true;
            orig[k] = v;
        }

        if (has_nan) {
            out_tm[t * cols + s] = nan_f;
        } else {
            out_tm[t * cols + s] = mad_from_window(orig, period, scratch);
        }
    }
}


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

__device__ __forceinline__ double fabsf_fast_f64(double x) {
    return fabs(x);
}
__device__ __forceinline__ void two_sum_f64(double a, double b, double &s, double &err) {
    s = a + b;
    double z = s - a;
    err = (a - (s - z)) + (b - z);
}
__device__ __forceinline__ double avg2_compensated_f64(double a, double b) {
    double s, e;
    two_sum_f64(a, b, s, e);
#if defined(__CUDA_ARCH__)
    return __fma_rn(0.5, e, 0.5 * s);
#else
    return 0.5 * (s + e);
#endif
}
__device__ __forceinline__ double median3(double a, double b, double c) {
    double ab = fmin(a, b), AB = fmax(a, b);
    double bc = fmin(AB, c), BC = fmax(AB, c);
    (void)BC;
    return fmax(ab, bc);
}
__device__ __forceinline__ double nth_element_inplace_f64(double* a, int n, int k) {
    int left = 0, right = n - 1;
    while (left < right) {
        const int mid = (left + right) >> 1;
        const double pivot = median3(a[left], a[mid], a[right]);

        int lt = left, i = left, gt = right;
        while (i <= gt) {
            const double v = a[i];
            if (v < pivot) {
                double tmp = a[lt]; a[lt] = a[i]; a[i] = tmp;
                ++lt; ++i;
            } else if (v > pivot) {
                double tmp = a[i]; a[i] = a[gt]; a[gt] = tmp;
                --gt;
            } else {
                ++i;
            }
        }
        if (k < lt) {
            right = lt - 1;
        } else if (k > gt) {
            left = gt + 1;
        } else {
            return a[k];
        }
    }
    return a[k];
}
__device__ __forceinline__ double median_from_window_f64(const double* __restrict__ orig, int n, double* __restrict__ scratch) {

    for (int i = 0; i < n; ++i) scratch[i] = orig[i];

    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace_f64(scratch, n, k);
    } else {
        const int k = n >> 1;
        const double upper = nth_element_inplace_f64(scratch, n, k);

        double lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) {
            lower = fmax(lower, scratch[i]);
        }
        return avg2_compensated_f64(lower, upper);
    }
}
__device__ __forceinline__ double mad_from_window_f64(const double* __restrict__ orig, int n, double* __restrict__ scratch) {

    const double med = median_from_window_f64(orig, n, scratch);


    for (int i = 0; i < n; ++i) {
        scratch[i] = fabsf_fast_f64(orig[i] - med);
    }


    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace_f64(scratch, n, k);
    } else {
        const int k = n >> 1;
        const double upper = nth_element_inplace_f64(scratch, n, k);
        double lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) lower = fmax(lower, scratch[i]);
        return avg2_compensated_f64(lower, upper);
    }
}
__device__ __forceinline__ double mad_period_2_f64(double x0, double x1) {

    return 0.5 * fabsf_fast_f64(x1 - x0);
}
extern "C" __global__ void medium_ad_batch_f64(
    const double* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const double nan_f = nan("");


    double orig[MEDIUM_AD_MAX_PERIOD];
    double scratch[MEDIUM_AD_MAX_PERIOD];


    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        double out_val = nan_f;

        if (t >= warm) {
            if (period == 1) {
                const double v = data[t];
                out_val = isfinite(v) ? 0.0 : nan_f;
            } else if (period == 2) {
                const double x0 = data[t - 1];
                const double x1 = data[t];
                out_val = (isfinite(x0) && isfinite(x1)) ? mad_period_2_f64(x0, x1) : nan_f;
            } else {
                const int start = t + 1 - period;
                bool has_nan = false;


                #pragma unroll 1
                for (int k = 0; k < period; ++k) {
                    const double v = data[start + k];
                    if (!isfinite(v)) has_nan = true;
                    orig[k] = v;
                }

                if (!has_nan) {
                    out_val = mad_from_window_f64(orig, period, scratch);
                }
            }
        }

        out[row_off + t] = out_val;
        t += stride;
    }
}
extern "C" __global__ void medium_ad_many_series_one_param_f64(
    const double* __restrict__ data_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) {
        const double nan_f = nan("");
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    const int warm = first_valid + period - 1;
    const double nan_f = nan("");


    int prefill = warm < rows ? warm : rows;
    for (int t = 0; t < prefill; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (warm >= rows) return;

    double orig[MEDIUM_AD_MAX_PERIOD];
    double scratch[MEDIUM_AD_MAX_PERIOD];

    for (int t = warm; t < rows; ++t) {
        if (period == 1) {
            const double v = data_tm[t * cols + s];
            out_tm[t * cols + s] = isfinite(v) ? 0.0 : nan_f;
            continue;
        }
        if (period == 2) {
            const double x0 = data_tm[(t - 1) * cols + s];
            const double x1 = data_tm[t * cols + s];
            out_tm[t * cols + s] = (isfinite(x0) && isfinite(x1)) ? mad_period_2_f64(x0, x1) : nan_f;
            continue;
        }

        const int start = t + 1 - period;
        bool has_nan = false;
        for (int k = 0; k < period; ++k) {
            const double v = data_tm[(start + k) * cols + s];
            if (!isfinite(v)) has_nan = true;
            orig[k] = v;
        }

        if (has_nan) {
            out_tm[t * cols + s] = nan_f;
        } else {
            out_tm[t * cols + s] = mad_from_window_f64(orig, period, scratch);
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// WHY A SECOND ENTRY POINT RATHER THAN REGISTERING `medium_ad_batch_f64`.
// Four measured divergences from the CPU, each of which moves a value:
//   1. that kernel tests `isfinite(v)`; the CPU tests `v != v`
//      (medium_ad.rs:436, :344, :404). An INFINITE bar is VALID to the CPU and
//      is rejected by `isfinite`, so the two disagree on real data with an inf.
//   2. even-period median: the CPU is `0.5 * (lo_max + buf[mid])` (:391) --
//      one add, one multiply. That kernel calls `avg2_compensated_f64`, a
//      two-sum plus an fma, which is a DIFFERENT (better, and therefore wrong
//      for parity) rounding.
//   3. `period == 5` takes `medium_ad_period5` (:334) on the CPU -- a 5-element
//      sorting network, not the generic quickselect.
//   4. `period == 2` is not a CPU special case at all; that kernel invents one.
// Its argument order is also not the lane ABI. So the lane gets its own entry
// point and the existing one is left to the callers that already launch it.
//
// CPU REFERENCE
// -------------
//   src/indicators/medium_ad.rs
//     :302 medium_ad_abs      -- bit-mask abs, i.e. `fabs`
//     :307 medium_ad_median5  -- the 5-element sorting network
//     :334 medium_ad_period5
//     :362 medium_ad_scalar   <- the whole specification
//     :371 median_from        -- select_nth then, for even n, 0.5*(lo_max+kth)
//     :186 prepare            -- `first` is the first `!is_nan`;
//                                `period > len` and `len - first < period`
//                                are Errs, i.e. an all-NaN row
//     :218 NaN prefix is `[..first + period - 1]`
//   dispatch: cpu_batch.rs:3594, param `period` (default 5).
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW. An ORDER STATISTIC: the median selects an
// element of the window, so unlike a sum it has no accumulation order to
// preserve, and any correct selection reproduces the CPU bit for bit. The two
// places where arithmetic DOES happen -- `0.5*(lo+hi)` for an even window and
// `|x - med|` -- are written exactly as the CPU writes them.
//
// PERIOD-SWEPT: the CPU batch reads a parameter literally named `period`.
//
// The window buffer is a per-thread local array bounded by
// MEDIUM_AD_MAX_PERIOD, so the lane REFUSES a larger period by name
// (`F64Kernel::max_period`) rather than truncating the window.
//
// ARITHMETIC
// ----------
// f64 end to end, no fast-math, no f32-suffixed function. No epsilon exists in
// this indicator and none was invented.

__device__ __forceinline__ double medad_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// :307 medium_ad_median5 -- the exact comparison/swap sequence.
__device__ __forceinline__ double medad_neo_median5(double a, double b, double c,
                                                    double d, double e) {
    double t;
    if (b < a) { t = a; a = b; b = t; }
    if (d < c) { t = c; c = d; d = t; }
    if (c < a) { t = a; a = c; c = t; t = b; b = d; d = t; }
    if (e < b) { t = b; b = e; e = t; }
    if (c < b) { t = b; b = c; c = t; }
    if (e < d) { t = d; d = e; e = t; }
    if (d < c) { t = c; c = d; d = t; }
    (void)a; (void)e;
    return c;
}

// `select_nth_unstable_by` with the CPU comparator (`<` Less, `>` Greater, else
// Equal). Hoare/Bentley-McIlroy three-way partition. The POST-CONDITION is what
// matters and it is identical to the Rust one: `buf[k]` holds the k-th smallest
// and everything before index k is <= it. NaNs cannot reach here -- the caller
// rejects a window containing one -- so the comparator has no unordered case to
// resolve.
__device__ __forceinline__ double medad_neo_select_nth(double* a, int n, int k) {
    int left = 0, right = n - 1;
    while (left < right) {
        const int mid = (left + right) >> 1;
        // median-of-three pivot, purely a performance choice; it cannot change
        // which element ends up at index k.
        double x = a[left], y = a[mid], z = a[right];
        double lo = fmin(x, y), hi = fmax(x, y);
        double pivot = fmax(lo, fmin(hi, z));

        int lt = left, i = left, gt = right;
        while (i <= gt) {
            const double v = a[i];
            if (v < pivot) {
                double t = a[lt]; a[lt] = a[i]; a[i] = t; ++lt; ++i;
            } else if (v > pivot) {
                double t = a[i]; a[i] = a[gt]; a[gt] = t; --gt;
            } else {
                ++i;
            }
        }
        if (k < lt) right = lt - 1;
        else if (k > gt) left = gt + 1;
        else return a[k];
    }
    return a[k];
}

// :371 median_from -- odd n is the k-th element; even n is
// `0.5 * (max of the left partition + the k-th element)`.
__device__ __forceinline__ double medad_neo_median_from(double* buf, int n, int mid) {
    const double kth = medad_neo_select_nth(buf, n, mid);
    if (n & 1) return kth;
    double lo_max = -INFINITY;
    for (int i = 0; i < mid; ++i) {
        if (buf[i] > lo_max) lo_max = buf[i];
    }
    return 0.5 * (lo_max + kth);
}

extern "C" __global__ void medium_ad_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = medad_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) o[i] = nan_d;

    const int period = periods[row];
    if (n <= 0 || first_valid < 0 || first_valid >= n) return;
    if (period <= 0 || period > n) return;                    // :205 InvalidPeriod
    if (period > MEDIUM_AD_MAX_PERIOD) return;                // refused upstream by name
    if (n - first_valid < period) return;                     // :212 NotEnoughValidData

    if (period == 1) {                                        // :400
        for (int i = first_valid; i < n; ++i) {
            const double v = data[i];
            o[i] = isnan(v) ? nan_d : 0.0;
        }
        return;
    }

    if (period == 5) {                                        // :396 -> :334
        const int warm = first_valid + 4;
        for (int i = warm; i < n; ++i) {
            const double a0 = data[i - 4];
            const double a1 = data[i - 3];
            const double a2 = data[i - 2];
            const double a3 = data[i - 1];
            const double a4 = data[i];
            if (isnan(a0) || isnan(a1) || isnan(a2) || isnan(a3) || isnan(a4)) {
                o[i] = nan_d;
                continue;
            }
            const double med = medad_neo_median5(a0, a1, a2, a3, a4);
            o[i] = medad_neo_median5(fabs(a0 - med), fabs(a1 - med), fabs(a2 - med),
                                     fabs(a3 - med), fabs(a4 - med));
        }
        return;
    }

    double buf[MEDIUM_AD_MAX_PERIOD];
    const int mid = period >> 1;
    const int warm = first_valid + period - 1;

    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - period;
        bool has_nan = false;
        for (int k = 0; k < period; ++k) {
            const double v = data[start + k];
            buf[k] = v;
            has_nan = has_nan || (v != v);          // the CPU test, verbatim
        }
        if (has_nan) { o[i] = nan_d; continue; }

        const double med = medad_neo_median_from(buf, period, mid);
        // The CPU rewrites the PERMUTED buffer in place (:453-468). Permutation
        // is irrelevant to a median, and doing the same thing keeps the two
        // implementations line-for-line comparable.
        for (int k = 0; k < period; ++k) buf[k] = fabs(buf[k] - med);
        o[i] = medad_neo_median_from(buf, period, mid);
    }
}
