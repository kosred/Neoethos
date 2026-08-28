#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STOCHF_QNAN
#define STOCHF_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float stoch_from_tables(
    int t,
    int fast_k,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum
) {
    const int start = t - fast_k + 1;


    if (nan_psum[t + 1] - nan_psum[start]) return STOCHF_QNAN;

    const int k           = log2_tbl[fast_k];
    const int offset      = 1 << k;
    const int level_base  = level_offsets[k];
    const int idx_a       = level_base + start;
    const int idx_b       = level_base + (t + 1 - offset);

    const float h = fmaxf(st_max[idx_a], st_max[idx_b]);
    const float l = fminf(st_min[idx_a], st_min[idx_b]);
    const float c = close[t];


    if (!(h == h) || !(l == l) || !(c == c)) return STOCHF_QNAN;

    const float den = h - l;
    if (den == 0.0f) {

        return (c == h) ? 100.0f : 0.0f;
    }
    return 100.0f * ((c - l) / den);
}

extern "C" __global__ void stochf_batch_f32(

    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    const int*   __restrict__ fastk_arr,
    const int*   __restrict__ fastd_arr,
    const int*   __restrict__ matype_arr,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,

    float* __restrict__ out_k,
    float* __restrict__ out_d
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int fk = fastk_arr[combo];
    const int fd = fastd_arr[combo];
    const int mt = matype_arr[combo];

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    if (UNLIKELY(level_count <= 0 || fk <= 0 || fd <= 0))       return;

    const int base   = combo * series_len;
    const int k_warm = first_valid + fk - 1;
    const int d_warm = k_warm + fd - 1;

    if (UNLIKELY(k_warm >= series_len)) {

        for (int t = threadIdx.x; t < min(series_len, d_warm); t += blockDim.x)
            out_d[base + t] = STOCHF_QNAN;
        return;
    }


    for (int t = threadIdx.x; t < k_warm; t += blockDim.x) out_k[base + t] = STOCHF_QNAN;
    for (int t = threadIdx.x; t < min(series_len, d_warm); t += blockDim.x) out_d[base + t] = STOCHF_QNAN;

    __syncthreads();


    for (int t = k_warm + threadIdx.x; t < series_len; t += blockDim.x) {
        out_k[base + t] = stoch_from_tables(t, fk, close, log2_tbl, level_offsets, st_max, st_min, nan_psum);
    }

    __syncthreads();


    if (mt == 0) {
        if (fd == 1) {

            for (int t = k_warm + threadIdx.x; t < series_len; t += blockDim.x)
                out_d[base + t] = out_k[base + t];
        } else {
            for (int t = d_warm + threadIdx.x; t < series_len; t += blockDim.x) {
                float sum = 0.0f;
                bool ok = true;
                const int start = t - fd + 1;
                for (int j = start; j <= t; ++j) {
                    const float kv = out_k[base + j];
                    if (UNLIKELY(!(kv == kv))) { ok = false; break; }
                    sum += kv;
                }
                out_d[base + t] = ok ? (sum / (float)fd) : STOCHF_QNAN;
            }
        }
    } else {

    }
}


extern "C" __global__ void stochf_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int fast_k,
    int fast_d,
    int matype,
    float* __restrict__ k_out_tm,
    float* __restrict__ d_out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];


    for (int t = 0; t < series_len; ++t) {
        *(k_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
        *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
    }
    if (UNLIKELY(fv < 0 || fv >= series_len || fast_k <= 0 || fast_d <= 0)) return;

    const int k_warm = fv + fast_k - 1;
    const int d_warm = k_warm + fast_d - 1;
    if (UNLIKELY(k_warm >= series_len)) return;

    auto load_tm = [num_series, series](const float* base, int t)->float {
        return *(base + (size_t)t * num_series + series);
    };

    auto stoch_naive = [&](int t)->float {
        const int start = t - fast_k + 1;


        float h = load_tm(high_tm, start);
        float l = load_tm(low_tm,  start);
        if (!(h == h) || !(l == l)) return STOCHF_QNAN;

        for (int i = start + 1; i <= t; ++i) {
            const float hi = load_tm(high_tm, i);
            const float lo = load_tm(low_tm,  i);
            if (!(hi == hi) || !(lo == lo)) return STOCHF_QNAN;
            h = fmaxf(h, hi);
            l = fminf(l, lo);
        }
        const float c = load_tm(close_tm, t);
        if (!(c == c)) return STOCHF_QNAN;

        const float den = h - l;
        if (den == 0.0f) return (c == h) ? 100.0f : 0.0f;
        return 100.0f * ((c - l) / den);
    };


    for (int t = k_warm; t < series_len; ++t) {
        float kv = stoch_naive(t);
        *(k_out_tm + (size_t)t * num_series + series) = kv;
    }


    if (matype == 0) {
        float sum = 0.0f, comp = 0.0f; int consec = 0;
        auto kahan_add = [](float &s, float x, float &c){ float y=x-c; float t=s+y; c=(t-s)-y; s=t; };

        for (int t = k_warm; t < series_len; ++t) {
            const float kv = *(k_out_tm + (size_t)t * num_series + series);
            if (kv == kv) {
                kahan_add(sum, kv, comp); ++consec;
                if (consec < fast_d) {
                    *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
                } else if (consec == fast_d) {
                    *(d_out_tm + (size_t)t * num_series + series) = sum / (float)fast_d;
                } else {
                    const float oldk = *(k_out_tm + (size_t)(t - fast_d) * num_series + series);
                    kahan_add(sum, -oldk, comp);
                    *(d_out_tm + (size_t)t * num_series + series) = sum / (float)fast_d;
                }
            } else {
                *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
                sum = 0.0f; comp = 0.0f; consec = 0;
            }
        }
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/stochf.rs
//   * stochf_with_kernel (:386) — first_valid is the first index at which HIGH,
//     LOW and CLOSE are ALL non-NaN simultaneously;
//     k_warmup = first + fastk_period - 1,
//     d_warmup = first + fastk_period + fastd_period - 2 (:397-398).
//   * stochf_scalar (:564) — the general path.
//   * stochf_scalar_default_5_3_sma (:466) — the 5/3/SMA path. Verified line by
//     line (:530-556 vs :662-690) to be the SAME expressions in the SAME order,
//     fully unrolled, so ONE implementation is bit-identical to both and the
//     branch is deliberately not reproduced.
//
// PERIOD-INVARIANT. compute_stochf_batch (cpu_batch.rs:5622) reads
// fastk_period (default 5), fastd_period (default 3) and fastd_matype
// (default 0 = SMA) and NEVER reads `period`.
//
// DEFAULT OUTPUT is FASTK. neoethos_stochf_fastd_f64 ships beside it.
//
// ROUNDING COUNT. The k line is ONE fused multiply-add on the CPU:
//     let inv = 100.0 / denom;
//     c.mul_add(inv, (-ll) * inv)
// — the (-ll)*inv multiply rounds, then a single fma. Reproduced as
//     fma(c, inv, (-ll) * inv)
// NOT as (c - ll) * inv, which is algebraically equal and numerically different.
//
// The denom == 0.0 branch and the c == hh test are EXACT comparisons on the
// CPU. There is no epsilon here to re-derive for f64; adding one would change
// which bars emit 100.0.
//
// The hh/ll window scan is a raw comparison chain on the CPU, where a NaN
// updates neither bound — reproduced as raw comparisons rather than fmax/fmin,
// because rule 4 is "match the CPU" and here the CPU is not calling f64::max.
//
// Sequential: the fastd SMA carries a running sum and needs the k value from
// fastd_period bars back. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_stochf() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

#define NEF_STOCHF_FASTK 5
#define NEF_STOCHF_FASTD 3
#define NEF_STOCHF_MAX_D 64

__device__ __forceinline__ void nef_stochf_body(const double* __restrict__ high,
                                                const double* __restrict__ low,
                                                const double* __restrict__ close,
                                                int n,
                                                int first_valid,
                                                bool want_fastd,
                                                double* __restrict__ row)
{
    const double QNAN = nef_qnan_stochf();
    for (int i = 0; i < n; ++i) row[i] = QNAN;

    const int fastk = NEF_STOCHF_FASTK;
    const int fastd = NEF_STOCHF_FASTD;
    if (first_valid < 0 || first_valid >= n) return;
    if (n - first_valid < fastk) return;

    const int k_start = first_valid + fastk - 1;
    if (k_start >= n) return;

    // Ring of the last `fastd` k values, so the steady-state d update can
    // subtract k[i - fastd] without a second pass over the row.
    double kring[NEF_STOCHF_MAX_D];
    int kpos = 0;

    double d_sum = 0.0;
    int d_cnt = 0;

    for (int i = k_start; i < n; ++i) {
        const int start = i + 1 - fastk;

        double hh = -INFINITY;
        double ll =  INFINITY;
        for (int j = start; j <= i; ++j) {
            const double h = high[j];
            const double l = low[j];
            if (h > hh) hh = h;
            if (l < ll) ll = l;
        }

        const double c = close[i];
        const double denom = hh - ll;
        double kv;
        if (denom == 0.0) {
            kv = (c == hh) ? 100.0 : 0.0;
        } else {
            const double inv = 100.0 / denom;
            kv = fma(c, inv, (-ll) * inv);
        }

        double dv = QNAN;
        const double k_out = kring[kpos];   // k[i - fastd], valid once d_cnt == fastd
        if (isnan(kv)) {
            dv = QNAN;
        } else if (d_cnt < fastd) {
            d_sum += kv;
            ++d_cnt;
            dv = (d_cnt == fastd) ? (d_sum / (double)fastd) : QNAN;
        } else {
            d_sum += kv - k_out;
            dv = d_sum / (double)fastd;
        }

        kring[kpos] = kv;
        ++kpos;
        if (kpos == fastd) kpos = 0;

        row[i] = want_fastd ? dv : kv;
    }
}

extern "C" __global__
void neoethos_stochf_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         const double* __restrict__ close,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;  // PERIOD-INVARIANT.
    nef_stochf_body(high, low, close, n, first_valid, false, out + (size_t)r * (size_t)n);
}

extern "C" __global__
void neoethos_stochf_fastd_f64(const double* __restrict__ high,
                               const double* __restrict__ low,
                               const double* __restrict__ close,
                               int n,
                               const int* __restrict__ periods,
                               int n_combos,
                               int first_valid,
                               double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    (void)periods;
    nef_stochf_body(high, low, close, n, first_valid, true, out + (size_t)r * (size_t)n);
}


// ===========================================================================
// S1 f64 LANE  --  stochf
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/stochf.rs -- `stochf_scalar` (:564), `stochf_scalar_default_5_3_sma` (:464), `stochf_with_kernel` (:362)
//
// PERIOD-INVARIANT. `compute_stochf_batch` (cpu_batch.rs:5616) reads
// `fastk_period` (5), `fastd_period` (3) and `fastd_matype` (0). There is no
// `period` parameter, so every row of a sweep is byte-identical.
//
// `stochf_with_kernel` collapses Auto AND both AVX variants to `Kernel::Scalar`
// (stochf.rs:402-407), so `stochf_scalar` is the only CPU answer on any host.
//
// ONE BODY SERVES BOTH CPU PATHS. `stochf_scalar` branches at (5, 3, 0) to
// `stochf_scalar_default_5_3_sma`, which is the `fastk_period <= 16` branch
// with the five window comparisons unrolled and `fastd_period` folded to the
// literal 3. Same order, same comparisons, same `d_sum / 3.0`; the two are
// bit-identical and only one body is written. Since (5, 3, 0) IS the default
// and this indicator has no swept period, that branch is the only one this
// lane can reach -- so the general form below is written for it directly.
//
// ARITHMETIC ORDER: `kv = c.mul_add(inv, (-ll) * inv)` where
// `inv = 100.0 / denom` -- ONE fused rounding plus one multiply. NOT
// `100.0 * (c - ll) / denom`, which is the textbook form and a different
// number. Reproduced with `fma`.
//
// THE denom == 0 BRANCH is an exact equality on zero, not an epsilon: when the
// window is flat the CPU emits 100.0 if the close equals the high and 0.0
// otherwise. An epsilon here would change WHICH bars take that branch.
//
// PRIMARY OUTPUT: `k`. `compute_stochf_batch` maps "value" to `out.k`
// (cpu_batch.rs:5642). `d` is a second series; this lane carries one matrix.
// The `d`-series bookkeeping is therefore dropped, and dropping it changes no
// `k` value -- `d` never feeds back into `k`.
//
// WARMUP: k's prefix is `first_valid + fastk_period - 1`; the separate,
// LONGER d warmup (`first + fastk + fastd - 2`) belongs to the series that is
// not emitted.
// ===========================================================================

#ifndef NEO_S1_QNAN_DEFINED
#define NEO_S1_QNAN_DEFINED
// The f32 kernels in this crate spell NaN `__int_as_float(0x7fc00000)`. That is
// a 32-bit pattern; widening it is a value change, not a cast. This is the f64
// quiet-NaN pattern, stated once per translation unit.
__device__ __forceinline__ double neo_s1_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}
__device__ __forceinline__ bool neo_s1_isnan(double x) { return x != x; }
#endif

extern "C" __global__ void neoethos_stochf_batch_f64(
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
    (void)periods;

    // `StochfParams::default()`.
    const int fastk_period = 5;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (fastk_period > n) ||
        ((n - first_valid) < fastk_period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int k_start = first_valid + fastk_period - 1;
    for (int i = 0; i < k_start && i < n; ++i) row[i] = neo_s1_qnan();
    if (k_start >= n) return;

    for (int i = k_start; i < n; ++i) {
        const int start = i + 1 - fastk_period;

        double hh = -INFINITY;
        double ll =  INFINITY;
        for (int j = start; j <= i; ++j) {
            const double h = high[j];
            const double l = low[j];
            if (h > hh) hh = h;
            if (l < ll) ll = l;
        }

        const double c = close[i];
        const double denom = hh - ll;
        double kv;
        if (denom == 0.0) {
            kv = (c == hh) ? 100.0 : 0.0;
        } else {
            const double inv = 100.0 / denom;
            kv = fma(c, inv, (-ll) * inv);
        }
        row[i] = kv;
    }
}
