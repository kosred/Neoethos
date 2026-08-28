#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STOCH_NAN
#define STOCH_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif


#ifndef STOCH_EPS
#define STOCH_EPS (1e-12f)
#endif


static __device__ __forceinline__
float stoch_k_from_chl(float c, float h, float l) {
    if (!(isfinite(c) && isfinite(h) && isfinite(l))) return STOCH_NAN;
    const float denom = h - l;
    return (fabsf(denom) < STOCH_EPS) ? 50.0f : (c - l) * (100.0f / denom);
}

extern "C" __global__ __launch_bounds__(256, 2)
void stoch_k_raw_from_hhll_f32(const float* __restrict__ close,
                               const float* __restrict__ hh,
                               const float* __restrict__ ll,
                               int series_len,
                               int first_valid,
                               int fastk_period,
                               float* __restrict__ out) {
    if (UNLIKELY(series_len <= 0 || fastk_period <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;

    const int warm = first_valid + fastk_period - 1;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (UNLIKELY(warm >= series_len)) {

        for (int t = tid; t < series_len; t += stride) out[t] = STOCH_NAN;
        return;
    }


    for (int t = tid; t < series_len; t += stride) {
        if (t < warm) {
            out[t] = STOCH_NAN;
        } else {
            const float c = close[t];
            const float h = hh[t];
            const float l = ll[t];
            out[t] = stoch_k_from_chl(c, h, l);
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void stoch_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const float* __restrict__ close_tm,
                                     const int*   __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     int fastk_period,
                                     float* __restrict__ k_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;


    if (UNLIKELY(fastk_period <= 0 || fastk_period > series_len)) {
        float* out_col = k_tm + s;
        for (int row = 0; row < series_len; ++row, out_col += num_series) *out_col = STOCH_NAN;
        return;
    }

    const int first_valid = first_valids[s];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* out_col = k_tm + s;
        for (int row = 0; row < series_len; ++row, out_col += num_series) *out_col = STOCH_NAN;
        return;
    }

    const int S = num_series;
    const int warm = first_valid + fastk_period - 1;


    {
        float* out_col = k_tm + s;
        const int limit = (warm < series_len) ? warm : series_len;
        for (int row = 0; row < limit; ++row, out_col += S) *out_col = STOCH_NAN;
        if (warm >= series_len) return;
    }


    if (fastk_period == 1) {
        const float* hptr = high_tm  + ((size_t)first_valid) * S + s;
        const float* lptr = low_tm   + ((size_t)first_valid) * S + s;
        const float* cptr = close_tm + ((size_t)first_valid) * S + s;
        float*       optr = k_tm     + ((size_t)first_valid) * S + s;
        for (int row = first_valid; row < series_len; ++row) {
            const float h = *hptr; const float l = *lptr; const float c = *cptr;
            *optr = stoch_k_from_chl(c, h, l);
            hptr += S; lptr += S; cptr += S; optr += S;
        }
        return;
    }


    for (int row = warm; row < series_len; ++row) {
        const int start = row - fastk_period + 1;

        const float* hptr = high_tm + ((size_t)start) * S + s;
        const float* lptr = low_tm  + ((size_t)start) * S + s;

        float hmax = -CUDART_INF_F;
        float lmin =  CUDART_INF_F;
        bool any_nan = false;

        int k = 0;

        for (; k + 3 < fastk_period; k += 4) {
            const float h0 = hptr[0];  const float l0 = lptr[0];
            const float h1 = hptr[S];  const float l1 = lptr[S];
            const float h2 = hptr[S*2];const float l2 = lptr[S*2];
            const float h3 = hptr[S*3];const float l3 = lptr[S*3];

            any_nan |= !(isfinite(h0) && isfinite(l0));
            any_nan |= !(isfinite(h1) && isfinite(l1));
            any_nan |= !(isfinite(h2) && isfinite(l2));
            any_nan |= !(isfinite(h3) && isfinite(l3));

            hmax = fmaxf(hmax, fmaxf(fmaxf(h0, h1), fmaxf(h2, h3)));
            lmin = fminf(lmin, fminf(fminf(l0, l1), fminf(l2, l3)));

            hptr += S * 4; lptr += S * 4;
        }

        for (; k < fastk_period; ++k) {
            const float hv = *hptr; const float lv = *lptr;
            any_nan |= !(isfinite(hv) && isfinite(lv));
            hmax = fmaxf(hmax, hv);
            lmin = fminf(lmin, lv);
            hptr += S; lptr += S;
        }

        float* outp = k_tm + ((size_t)row) * S + s;
        const float c = close_tm[((size_t)row) * S + s];

        if (any_nan || !isfinite(c) || !isfinite(hmax) || !isfinite(lmin)) {
            *outp = STOCH_NAN;
        } else {
            const float denom = hmax - lmin;
            *outp = (fabsf(denom) < STOCH_EPS) ? 50.0f : (c - lmin) * (100.0f / denom);
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void stoch_one_series_many_params_f32(const float* __restrict__ high,
                                      const float* __restrict__ low,
                                      const float* __restrict__ close,
                                      const int*   __restrict__ fastk_periods,
                                      const int*   __restrict__ first_valids,
                                      int series_len,
                                      int num_params,
                                      float* __restrict__ k_tm) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_params) return;

    const int fastk = fastk_periods[p];
    const int first_valid = first_valids[p];

    if (UNLIKELY(series_len <= 0 || fastk <= 0 || fastk > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int t = 0; t < series_len; ++t) k_tm[((size_t)t) * num_params + p] = STOCH_NAN;
        return;
    }

    const int warm = first_valid + fastk - 1;


    for (int t = 0; t < warm; ++t) k_tm[((size_t)t) * num_params + p] = STOCH_NAN;


    if (fastk == 1) {
        for (int t = first_valid; t < series_len; ++t) {
            const float h = high[t];
            const float l = low[t];
            const float c = close[t];
            k_tm[((size_t)t) * num_params + p] = stoch_k_from_chl(c, h, l);
        }
        return;
    }


    for (int t = warm; t < series_len; ++t) {
        const int start = t - fastk + 1;

        float hmax = -CUDART_INF_F;
        float lmin =  CUDART_INF_F;
        bool any_nan = false;

        int k = 0;

        for (; k + 3 < fastk; k += 4) {
            const float h0 = high[start + k + 0]; const float l0 = low[start + k + 0];
            const float h1 = high[start + k + 1]; const float l1 = low[start + k + 1];
            const float h2 = high[start + k + 2]; const float l2 = low[start + k + 2];
            const float h3 = high[start + k + 3]; const float l3 = low[start + k + 3];

            any_nan |= !(isfinite(h0) && isfinite(l0));
            any_nan |= !(isfinite(h1) && isfinite(l1));
            any_nan |= !(isfinite(h2) && isfinite(l2));
            any_nan |= !(isfinite(h3) && isfinite(l3));

            hmax = fmaxf(hmax, fmaxf(fmaxf(h0, h1), fmaxf(h2, h3)));
            lmin = fminf(lmin, fminf(fminf(l0, l1), fminf(l2, l3)));
        }
        for (; k < fastk; ++k) {
            const float hv = high[start + k];
            const float lv = low[start + k];
            any_nan |= !(isfinite(hv) && isfinite(lv));
            hmax = fmaxf(hmax, hv);
            lmin = fminf(lmin, lv);
        }

        const float c = close[t];
        float* outp = &k_tm[((size_t)t) * num_params + p];
        if (any_nan || !isfinite(c) || !isfinite(hmax) || !isfinite(lmin)) {
            *outp = STOCH_NAN;
        } else {
            const float denom = hmax - lmin;
            *outp = (fabsf(denom) < STOCH_EPS) ? 50.0f : (c - lmin) * (100.0f / denom);
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void pack_row_broadcast_rowmajor_f32(const float* __restrict__ src,
                                     int len,
                                     const int* __restrict__ rows_idx,
                                     int nrows,
                                     float* __restrict__ dst,
                                     int row_stride)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = t; i < len; i += stride) {
        const float v = src[i];
        #pragma unroll 4
        for (int j = 0; j < nrows; ++j) {
            const int row = rows_idx[j];
            dst[(size_t)row * (size_t)row_stride + (size_t)i] = v;
        }
    }
}


extern "C" __global__
void transpose_tm_to_rm_f32(const float* __restrict__ in_tm,
                            int rows,
                            int cols,
                            float* __restrict__ out_rm) {


    __shared__ float tile[32][33];

    const int x0 = blockIdx.x * 32 + threadIdx.x;
    const int y0 = blockIdx.y * 32 + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int y = y0 + j;
        if (x0 < cols && y < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in_tm[(size_t)y * (size_t)cols + (size_t)x0];
        }
    }

    __syncthreads();

    const int x1 = blockIdx.y * 32 + threadIdx.x;
    const int y1 = blockIdx.x * 32 + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int y = y1 + j;
        if (x1 < rows && y < cols) {
            out_rm[(size_t)y * (size_t)rows + (size_t)x1] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer 6
//
// CPU reference: `stoch_classic_sma_into_single_pass`
// (src/indicators/stoch.rs:724), reached from `stoch_with_kernel` (:261) on
// the branch both MA types are "sma" (:319-322), which is what
// `compute_stoch_batch` (dispatch/cpu_batch.rs:5573) always produces --
// `get_enum_param("stoch", params, "slowk_ma_type", "sma")` (:5583-5584).
//
// OUTPUT: `k`. `compute_stoch_batch:5603` maps output_id "value" onto
// `out.k`; "d" is the second running SMA in the same walk and is one launch
// away once the lane grows an output selector.
//
// PERIOD-INVARIANT. The CPU batch reads `fastk_period` (14), `slowk_period`
// (3) and `slowd_period` (3) and NEVER `period` (:5580-5582). A caller
// sweeping [7,21,50,100,200] gets five identical CPU columns, so this kernel
// emits five identical rows and `is_period_invariant` says so. Inventing a
// mapping from the swept int onto one of the three named periods would
// compute something the CPU never computes.
//
// SEQUENTIAL, ONE THREAD PER COLUMN, not the two-stage
// "bar-parallel-window then per-column smoothing" the brief sketched. The
// reason is the extreme tracker: the CPU does NOT rescan the window at every
// bar. It carries `maxi`/`max` and only rescans when the argmax falls out of
// the trailing edge (:778-793), and the rescan uses `>=` so it lands on the
// LAST tie rather than the first. A fresh per-bar window max computes the
// same NUMBER for finite data but a different `maxi`, and `maxi` decides
// whether the NEXT bar rescans -- so the two structures part company on the
// bar after a tie. Reproducing the tracker is the only faithful shape, and
// it is inherently serial in `i`.
//
// THE EPSILON IS THE CPU'S OWN AND IS ALREADY f64. `EPS: f64 =
// f64::EPSILON` (:770) is 2.220446049250313e-16, not an f32-sized constant
// copied forward, so it is reproduced verbatim as DBL_EPSILON. The guard
// `denom.abs() < EPS -> 50.0` (:819-820) is the CPU's, including the fact
// that it compares the ABSOLUTE difference and so also fires for a max
// BELOW the min.
//
// ROUNDINGS: `(c - min).mul_add(SCALE / denom, 0.0)` (:822) is ONE fused
// multiply-add over `(100.0 / denom)`, itself one rounding. Written here as
// `fma(c - min, SCALE / denom, 0.0)`. An `(c - min) * (SCALE / denom)`
// reformulation would be two roundings and is what a naive port produces.
// -fmad=false forbids the compiler from contracting anything else into an
// fma, so the explicit call is the only fused step.
//
// NaN: there is no `f64::max` in this reference. The extreme tracker is a
// comparison chain (`v >= max`, `bar_h >= max`), and a Rust `>=` against NaN
// is false exactly as a CUDA `>=` is -- a NaN high therefore leaves `max`
// unchanged on both sides. Converting these to fmax would CHANGE the answer,
// so they are kept as comparisons.
//
// f32 -> f64 audit of this section: no f32 literal, no f32-suffixed math
// function, no fast-math intrinsic. The quiet NaN is built from the f64 bit
// pattern, not `__int_as_float`.
// ===========================================================================

// `k_buf` is `slowk_period` long and `d_buf` is `slowd_period` long. The CPU
// keeps both on the stack up to 64 and spills to a Vec beyond (:750-760); a
// kernel has no Vec, so 64 is the bound and an oversized period is REFUSED
// BY NAME in the wrapper rather than truncated. With the periods pinned at
// the CPU defaults 3 and 3 the bound is never reached, but it is declared
// rather than assumed.
#define STOCH_F64_MA_MAX 64

// The CPU defaults, cpu_batch.rs:5580-5582.
#define STOCH_F64_FASTK 14
#define STOCH_F64_SLOWK 3
#define STOCH_F64_SLOWD 3

static __device__ __forceinline__ double stoch_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void stoch_batch_f64(const double* __restrict__ high,
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

    // PERIOD-INVARIANT: `periods[r]` is read so the parameter is not silently
    // dropped from the signature, but the CPU batch never consults it.
    (void)periods;

    const double nan_d = stoch_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);

    const int fastk_period = STOCH_F64_FASTK;
    const int slowk_period = STOCH_F64_SLOWK;
    const int slowd_period = STOCH_F64_SLOWD;

    // `stoch_with_kernel` (:284-303) errors -- and `collect_f64` turns the
    // error into an all-NaN column -- on a zero or oversized period, and
    // (:301) when the valid tail is shorter than fastk_period.
    if (n <= 0 || first_valid < 0 || first_valid >= n) return;
    if (fastk_period > n || slowk_period > n || slowd_period > n ||
        slowk_period > STOCH_F64_MA_MAX || slowd_period > STOCH_F64_MA_MAX ||
        n - first_valid < fastk_period) {
        for (int i = 0; i < n; ++i) row[i] = nan_d;
        return;
    }

    const int k_first_valid = first_valid + fastk_period - 1;                  // :737
    const int k_warm = k_first_valid + slowk_period - 1;                       // :738

    // `prefill_nan_prefix(out_k, k_warm)` -- :741. Everything from `k_warm`
    // on is written by the walk, and bars before `first_valid` are covered by
    // the same prefill.
    for (int i = 0; i < n && i < k_warm; ++i) row[i] = nan_d;

    int trail = first_valid;                                                   // :744
    int maxi = first_valid;
    int mini = first_valid;
    double maxv = high[first_valid];
    double minv = low[first_valid];

    double k_buf[STOCH_F64_MA_MAX];
    for (int j = 0; j < slowk_period; ++j) k_buf[j] = 0.0;                     // :750
    int k_pos = 0;
    double k_sum = 0.0;
    int k_count = 0;

    const double SCALE = 100.0;                                                // :769
    // f64::EPSILON -- :770. The CPU's own constant, already f64.
    const double EPSD = 2.2204460492503130808472633361816e-16;

    for (int i = first_valid; i < n; ++i) {
        if (i >= first_valid + fastk_period) ++trail;                          // :773-775

        const double bar_h = high[i];                                          // :777
        if (maxi < trail) {                                                    // :778-789
            maxi = trail;
            maxv = high[maxi];
            int j = trail;
            while (j < i) {
                ++j;
                const double v = high[j];
                if (v >= maxv) { maxv = v; maxi = j; }
            }
        } else if (bar_h >= maxv) {                                            // :790-793
            maxi = i; maxv = bar_h;
        }

        const double bar_l = low[i];                                           // :795
        if (mini < trail) {                                                    // :796-807
            mini = trail;
            minv = low[mini];
            int j = trail;
            while (j < i) {
                ++j;
                const double v = low[j];
                if (v <= minv) { minv = v; mini = j; }
            }
        } else if (bar_l <= minv) {                                            // :808-811
            mini = i; minv = bar_l;
        }

        if (i < k_first_valid) continue;                                       // :813-815

        const double c = close[i];                                             // :817
        const double denom = maxv - minv;                                      // :818
        double k_raw;
        if (fabs(denom) < EPSD) {                                              // :819-820
            k_raw = 50.0;
        } else {
            k_raw = fma(c - minv, SCALE / denom, 0.0);                         // :822
        }

        if (k_count >= slowk_period) k_sum -= k_buf[k_pos];                    // :825-827
        k_buf[k_pos] = k_raw;
        k_sum += k_raw;
        ++k_count;
        ++k_pos;
        if (k_pos == slowk_period) k_pos = 0;                                  // :832-834

        if (i >= k_warm) {                                                     // :836
            row[i] = k_sum / static_cast<double>(slowk_period);                // :837-838
        }
    }
}
