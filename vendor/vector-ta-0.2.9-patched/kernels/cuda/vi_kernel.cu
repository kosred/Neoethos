#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef VI_NAN
#define VI_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


extern "C" __global__ void vi_build_prefix_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int first_valid,
    float* __restrict__ out_tr,
    float* __restrict__ out_vp,
    float* __restrict__ out_vm
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    for (int i = 0; i < first_valid; ++i) {
        out_tr[i] = 0.0f;
        out_vp[i] = 0.0f;
        out_vm[i] = 0.0f;
    }

    double acc_tr = (double)(high[first_valid] - low[first_valid]);
    double acc_vp = 0.0;
    double acc_vm = 0.0;
    out_tr[first_valid] = (float)acc_tr;
    out_vp[first_valid] = 0.0f;
    out_vm[first_valid] = 0.0f;

    float prev_h = high[first_valid];
    float prev_l = low[first_valid];
    float prev_c = close[first_valid];
    for (int i = first_valid + 1; i < series_len; ++i) {
        const float hi = high[i];
        const float lo = low[i];
        const float hl = hi - lo;
        const float hc = fabsf(hi - prev_c);
        const float lc = fabsf(lo - prev_c);
        const float tr_i = fmaxf(hl, fmaxf(hc, lc));
        const float vp_i = fabsf(hi - prev_l);
        const float vm_i = fabsf(lo - prev_h);
        acc_tr += (double)tr_i;
        acc_vp += (double)vp_i;
        acc_vm += (double)vm_i;
        out_tr[i] = (float)acc_tr;
        out_vp[i] = (float)acc_vp;
        out_vm[i] = (float)acc_vm;
        prev_h = hi;
        prev_l = lo;
        prev_c = close[i];
    }
}

extern "C" __global__ void vi_batch_f32(
    const float* __restrict__ pfx_tr,
    const float* __restrict__ pfx_vp,
    const float* __restrict__ pfx_vm,
    const int*   __restrict__ periods,
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_plus,
    float* __restrict__ out_minus
) {

    if (gridDim.y > 1) {
        const int t   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        const int row = (int)blockIdx.y;
        if (t >= series_len || row >= n_rows) {
            return;
        }
        const size_t out_idx = (size_t)row * (size_t)series_len + (size_t)t;

        const int period = periods[row];
        if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
            out_plus[out_idx] = VI_NAN;
            out_minus[out_idx] = VI_NAN;
            return;
        }

        const int tail = series_len - first_valid;
        if (UNLIKELY(tail < period)) {
            out_plus[out_idx] = VI_NAN;
            out_minus[out_idx] = VI_NAN;
            return;
        }

        const int warm = first_valid + period - 1;
        if (t < warm) {
            out_plus[out_idx] = VI_NAN;
            out_minus[out_idx] = VI_NAN;
            return;
        }

        const int prev = t - period;
        const float tr_prev = (prev >= 0) ? pfx_tr[prev] : 0.0f;
        const float vp_prev = (prev >= 0) ? pfx_vp[prev] : 0.0f;
        const float vm_prev = (prev >= 0) ? pfx_vm[prev] : 0.0f;

        const float tr_sum = pfx_tr[t] - tr_prev;
        const float inv    = 1.0f / tr_sum;
        out_plus[out_idx]  = (pfx_vp[t] - vp_prev) * inv;
        out_minus[out_idx] = (pfx_vm[t] - vm_prev) * inv;
        return;
    }


    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t total = (size_t)n_rows * (size_t)series_len;
    if (tid >= total) {
        return;
    }
    const int row = (int)(tid / (size_t)series_len);
    const int t   = (int)(tid - (size_t)row * (size_t)series_len);

    const int period = periods[row];
    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        out_plus[tid] = VI_NAN;
        out_minus[tid] = VI_NAN;
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        out_plus[tid] = VI_NAN;
        out_minus[tid] = VI_NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    if (t < warm) {
        out_plus[tid] = VI_NAN;
        out_minus[tid] = VI_NAN;
        return;
    }

    const int prev = t - period;
    const float tr_prev = (prev >= 0) ? pfx_tr[prev] : 0.0f;
    const float vp_prev = (prev >= 0) ? pfx_vp[prev] : 0.0f;
    const float vm_prev = (prev >= 0) ? pfx_vm[prev] : 0.0f;

    const float tr_sum = pfx_tr[t] - tr_prev;
    const float inv    = 1.0f / tr_sum;
    out_plus[tid]  = (pfx_vp[t] - vp_prev) * inv;
    out_minus[tid] = (pfx_vm[t] - vm_prev) * inv;
}


extern "C" __global__ void vi_many_series_one_param_f32(
    const float* __restrict__ pfx_tr_tm,
    const float* __restrict__ pfx_vp_tm,
    const float* __restrict__ pfx_vm_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm
) {
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t total = (size_t)num_series * (size_t)series_len;
    if (tid >= total) {
        return;
    }

    const int series = (int)(tid % (size_t)num_series);
    const int row    = (int)(tid / (size_t)num_series);
    const size_t idx = (size_t)row * (size_t)num_series + (size_t)series;

    const int first = first_valids[series];
    if (UNLIKELY(period <= 0 || period > series_len || first < 0 || first >= series_len)) {
        plus_tm[idx] = VI_NAN;
        minus_tm[idx] = VI_NAN;
        return;
    }

    const int tail = series_len - first;
    if (UNLIKELY(tail < period)) {
        plus_tm[idx] = VI_NAN;
        minus_tm[idx] = VI_NAN;
        return;
    }

    const int warm = first + period - 1;
    if (row < warm) {
        plus_tm[idx] = VI_NAN;
        minus_tm[idx] = VI_NAN;
        return;
    }

    const int prev_row = row - period;
    if (prev_row >= 0) {
        const size_t idx_prev = (size_t)prev_row * (size_t)num_series + (size_t)series;
        const float tr_sum = pfx_tr_tm[idx] - pfx_tr_tm[idx_prev];
        const float inv    = 1.0f / tr_sum;
        plus_tm[idx]  = (pfx_vp_tm[idx] - pfx_vp_tm[idx_prev]) * inv;
        minus_tm[idx] = (pfx_vm_tm[idx] - pfx_vm_tm[idx_prev]) * inv;
    } else {
        const float tr_sum = pfx_tr_tm[idx];
        const float inv    = 1.0f / tr_sum;
        plus_tm[idx]  = pfx_vp_tm[idx] * inv;
        minus_tm[idx] = pfx_vm_tm[idx] * inv;
    }
}


// ===========================================================================
// f64 LANE  --  shard S6
//
// CPU reference: `vi_scalar` (src/indicators/vi.rs:290). `vi_avx2` (:400)
// delegates to it verbatim, and `vi_avx512` (:415) splits by period into
// short/long variants of the same recurrence, so the scalar path is the
// crate's single answer for the accumulation ORDER.
//
// OUTPUT: `plus` (VI+), which is `OUTPUTS_PLUS_MINUS[0]` (registry.rs:1064).
// `minus` shares every accumulator and differs only in the numerator; it is
// one output-selector away.
//
// first_valid: high, low AND close all non-NaN at the SAME index
// (vi.rs:213-215) -> `F64FirstValidRule::AllInputsNonNan` over Hlc. It is the
// simultaneous rule, NOT adx's max-of-independent-firsts.
//
// warm = first + period - 1 (:304). The seed accumulates from bar first+1 up
// to and including warm; bar `first` contributes tr = high-low, vp = 0, vm = 0
// (:326-332).
//
// THE RING IS NOT NEEDED ON THE DEVICE. The CPU keeps three `Vec<f64>` of
// length `period` (:312-320) purely to subtract the bar leaving the window.
// Slot `r` advances one per bar and wraps at `period`, so the value it holds
// at bar i was written at bar i-period -- and tr/vp/vm are pure functions of
// (high, low, close) at bars i-period and i-period-1. Recomputing them is
// BIT-IDENTICAL to reading them back and removes the unbounded per-thread
// array, which is why this kernel declares no `max_period`. The one special
// case is bar `first` itself, whose seeded triple is (high-low, 0, 0) rather
// than the general formula; `vi_terms_f64` carries that.
//
// ONE THREAD PER COLUMN, ascending bars: sum_tr / sum_vp / sum_vm are carried
// across bars by `+=`, so any reformulation that recomputes a window sum would
// land on different rounding.
//
// NaN SEMANTICS ARE THE CPU'S COMPARISON CHAIN, DELIBERATELY NOT fmax.
// :349-352 writes `let mut tr = if hl > hc { hl } else { hc }; if lc > tr { tr = lc; }`.
// With hl = NaN that yields hc, not NaN -- which is NOT what `f64::max` would
// give and NOT what fmax would give. Matching the CPU here means keeping the
// comparisons exactly as they are; substituting fmax would change the answer
// on any gapped bar.
//
// f32 -> f64 audit: the f32 lane above uses `fabsf` x4, `fmaxf` x2 and
// `__int_as_float` x1. Below: `fabs`, no fmax (see the paragraph above), and
// the f64 quiet-NaN bit pattern. Every literal is an f64 literal. No
// fast-math intrinsic. This indicator has no epsilon; the division
// `sum_vp / sum_tr` is left unguarded exactly as the CPU leaves it (:367,
// :383), so a zero true range produces the same infinity the host produces.
// ===========================================================================

static __device__ __forceinline__ double vi_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// The (tr, vp, vm) triple the CPU stores in its ring for bar `j`.
// vi.rs:326-332 for j == first, :346-355 for every later bar.
static __device__ __forceinline__ void vi_terms_f64(
    const double* __restrict__ high, const double* __restrict__ low,
    const double* __restrict__ close, int j, int first,
    double* tr, double* vp, double* vm)
{
    const double hi = high[j];
    const double lo = low[j];

    if (j <= first) {
        *tr = hi - lo;
        *vp = 0.0;
        *vm = 0.0;
        return;
    }

    const double prev_h = high[j - 1];
    const double prev_l = low[j - 1];
    const double prev_c = close[j - 1];

    const double hl = hi - lo;
    const double hc = fabs(hi - prev_c);
    const double lc = fabs(lo - prev_c);

    double t = (hl > hc) ? hl : hc;      // :349
    if (lc > t) { t = lc; }              // :350-352

    *tr = t;
    *vp = fabs(hi - prev_l);             // :354
    *vm = fabs(lo - prev_h);             // :355
}

extern "C" __global__
void vi_batch_f64(const double* __restrict__ high,
                  const double* __restrict__ low,
                  const double* __restrict__ close,
                  int n,
                  const int* __restrict__ periods,
                  int n_combos,
                  int first_valid,
                  double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = vi_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const int first  = (first_valid < 0) ? 0 : first_valid;

    // `vi_prepare` rejects period == 0, period > len and len - first < period.
    if (period <= 0 || period > n || first >= n || (n - first) < period) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }

    const int warm = first + period - 1;
    for (int t = 0; t < n; ++t) row[t] = nan_d;

    // Seed with bar `first` (:326-332).
    double sum_tr = high[first] - low[first];
    double sum_vp = 0.0;
    double sum_vm = 0.0;

    if (period == 1) {                    // :334-337
        row[warm] = 0.0;
    }

    for (int i = first + 1; i < n; ++i) {
        double tr_new, vp_new, vm_new;
        vi_terms_f64(high, low, close, i, first, &tr_new, &vp_new, &vm_new);

        if (i <= warm) {                  // :357-368
            sum_tr += tr_new;
            sum_vp += vp_new;
            sum_vm += vm_new;
            if (i == warm) {
                row[i] = sum_vp / sum_tr;
            }
        } else {                          // :369-385
            double tr_old, vp_old, vm_old;
            vi_terms_f64(high, low, close, i - period, first, &tr_old, &vp_old, &vm_old);

            sum_tr += tr_new - tr_old;
            sum_vp += vp_new - vp_old;
            sum_vm += vm_new - vm_old;

            row[i] = sum_vp / sum_tr;
        }
    }
}
