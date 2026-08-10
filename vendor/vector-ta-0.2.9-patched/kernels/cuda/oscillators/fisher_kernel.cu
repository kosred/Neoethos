#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

extern "C" __global__ void fisher_build_hl2_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    float* __restrict__ hl)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        hl[idx] = 0.5f * (high[idx] + low[idx]);
    }
}


__device__ __forceinline__ int rb_dec(int x, int cap) { return (x == 0) ? (cap - 1) : (x - 1); }
__device__ __forceinline__ int rb_inc(int x, int cap) { return (x + 1 == cap) ? 0 : (x + 1); }


extern "C" __global__ void fisher_batch_f32(const float* __restrict__ hl,
                                             const int*   __restrict__ periods,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out_fisher,
                                             float* __restrict__ out_signal) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;


    auto fill_all_nan = [&](int len){
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            out_fisher[base + i] = NAN;
            out_signal[base + i] = NAN;
        }
    };

    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan(series_len);
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        fill_all_nan(series_len);
        return;
    }

    const int warm = first_valid + period - 1;


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_fisher[base + i] = NAN;
        out_signal[base + i] = NAN;
    }
    __syncthreads();


    if (threadIdx.x != 0) return;


    extern __shared__ int s[];
    int* dq_min = s;
    int* dq_max = s + period;

    int hmin = 0, tmin = 0;
    int hmax = 0, tmax = 0;
    int cmin = 0, cmax = 0;


    float prev_fish = 0.0f;
    float val1 = 0.0f;


    for (int i = first_valid; i < series_len; ++i) {
        const float xi = hl[i];


        if (i >= warm) {
            const int window_start = i - period + 1;
            while (cmin > 0 && dq_min[hmin] < window_start) { hmin = rb_inc(hmin, period); --cmin; }
            while (cmax > 0 && dq_max[hmax] < window_start) { hmax = rb_inc(hmax, period); --cmax; }
        }


        while (cmin > 0) {
            int last = rb_dec(tmin, period);
            if (xi <= hl[dq_min[last]]) {
                tmin = last;
                --cmin;
            } else {
                break;
            }
        }
        dq_min[tmin] = i;
        tmin = rb_inc(tmin, period);
        ++cmin;


        while (cmax > 0) {
            int last = rb_dec(tmax, period);
            if (xi >= hl[dq_max[last]]) {
                tmax = last;
                --cmax;
            } else {
                break;
            }
        }
        dq_max[tmax] = i;
        tmax = rb_inc(tmax, period);
        ++cmax;


        if (i >= warm) {
            const float minv  = hl[dq_min[hmin]];
            const float maxv  = hl[dq_max[hmax]];
            const float range = fmaxf(maxv - minv, 1.0e-3f);
            const float norm  = (xi - minv) / range - 0.5f;


            val1 = fmaf(0.67f, val1, 0.66f * norm);
            val1 = clampf(val1, -0.999f, 0.999f);

            out_signal[base + i] = prev_fish;

            const float fish = atanhf(val1) + 0.5f * prev_fish;
            out_fisher[base + i] = fish;
            prev_fish = fish;
        }
    }
}


extern "C" __global__ void fisher_many_series_one_param_f32(
    const float* __restrict__ hl_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ fisher_tm,
    float* __restrict__ signal_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (UNLIKELY(first_valid >= series_len || (series_len - first_valid) < period)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int r = 0; r < warm; ++r) {
        const int idx = r * num_series + series;
        fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
    }

    float prev_fish = 0.0f;
    float val1 = 0.0f;
    for (int r = warm; r < series_len; ++r) {
        const int start = r + 1 - period;
        float minv = CUDART_INF_F;
        float maxv = -CUDART_INF_F;

        for (int k = 0; k < period; ++k) {
            const int idx = (start + k) * num_series + series;
            const float x = hl_tm[idx];
            minv = fminf(minv, x);
            maxv = fmaxf(maxv, x);
        }
        const float range = fmaxf(maxv - minv, 1.0e-3f);
        const float x = hl_tm[r * num_series + series];
        const float norm = (x - minv) / range - 0.5f;

        val1 = fmaf(0.67f, val1, 0.66f * norm);
        val1 = clampf(val1, -0.999f, 0.999f);

        const int idxo = r * num_series + series;
        signal_tm[idxo] = prev_fish;
        const float fish = atanhf(val1) + 0.5f * prev_fish;
        fisher_tm[idxo] = fish;
        prev_fish = fish;
    }
}

// =============================================================================
// NeoEthos f64 lane — added in place, f64 end to end.
//
// CPU reference: src/indicators/fisher.rs
//   * fisher_with_kernel (:237) — first_valid is the first index at which HIGH
//     and LOW are both non-NaN (close is never scanned), warmup prefix is
//     first + period - 1 (:258).
//   * fisher_scalar_into (:300) — the general path reproduced below.
//   * fisher_scalar_period9_into (:352) — the period == 9 path is the SAME
//     expression fully unrolled in the same order, so one implementation is
//     bit-identical to both and the branch is not reproduced.
//
// DEFAULT OUTPUT is the FISHER line: cpu_batch.rs:14777 maps output_id "value"
// to out.fisher. neoethos_fisher_signal_f64 ships beside it for the signal
// line, which is simply the PREVIOUS bar's fisher value.
//
// NaN SEMANTICS. `(max_val - min_val).max(0.001)` is f64::max, which returns
// the NON-NaN operand. fmax() has exactly that semantics; an if-chain does not,
// because a comparison against NaN is false and the NaN would survive into the
// division and then into the recurrence. This is the adx-class bug and it is
// avoided here by construction.
//
// The min/max window scan, by contrast, IS a raw comparison chain on the CPU
// (fisher_update_min_max, :348), where a NaN midpoint updates neither bound.
// Reproduced as raw comparisons for that reason — matching the CPU, not
// applying fmax/fmin blindly.
//
// SEEDS. f64::MAX = 1.7976931348623157e308 and f64::MIN = -1.7976931348623157e308
// (Rust's f64::MIN is the most NEGATIVE finite double, NOT the smallest
// positive one — the C analogue of f64::MIN is -DBL_MAX, not DBL_MIN). Getting
// this wrong is exactly the f32-epsilon class of bug the brief names.
//
// ROUNDING COUNT. Two fused steps on the CPU:
//     val1 = 0.67f64.mul_add(val1, 0.66 * (...))   -> fma(0.67, val1, 0.66 * (...))
//     new  = 0.5f64.mul_add(ln(...), 0.5*prev)     -> fma(0.5, log(...), 0.5*prev)
// log() is the correctly-rounded double natural log; logf would be the f32 one.
//
// Sequential: val1 and prev_fish carry across bars. One thread per column.
// =============================================================================

__device__ __forceinline__ double nef_qnan_fisher() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// Shared body. `want_signal` selects which of the two CPU outputs is written.
__device__ __forceinline__ void nef_fisher_body(const double* __restrict__ high,
                                                const double* __restrict__ low,
                                                int n,
                                                int period,
                                                int first_valid,
                                                bool want_signal,
                                                double* __restrict__ row)
{
    const double QNAN = nef_qnan_fisher();

    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        for (int i = 0; i < n; ++i) row[i] = QNAN;
        return;
    }

    const long long w = (long long)first_valid + (long long)period - 1;
    const int warm = w > (long long)n ? n : (int)w;
    for (int i = 0; i < n; ++i) row[i] = QNAN;
    if (warm >= n) return;

    // Rust f64::MAX / f64::MIN — the largest and the most NEGATIVE finite f64.
    const double F64_MAX =  1.7976931348623157e308;
    const double F64_MIN = -1.7976931348623157e308;

    double prev_fish = 0.0;
    double val1 = 0.0;

    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - period;

        double min_val = F64_MAX;
        double max_val = F64_MIN;
        for (int j = start; j <= i; ++j) {
            const double midpoint = 0.5 * (high[j] + low[j]);
            if (midpoint > max_val) max_val = midpoint;
            if (midpoint < min_val) min_val = midpoint;
        }

        // f64::max semantics: returns the non-NaN operand.
        const double range = fmax(max_val - min_val, 0.001);
        const double hl = 0.5 * (high[i] + low[i]);
        val1 = fma(0.67, val1, 0.66 * ((hl - min_val) / range - 0.5));
        if (val1 > 0.99) {
            val1 = 0.999;
        } else if (val1 < -0.99) {
            val1 = -0.999;
        }
        const double signal_here = prev_fish;
        const double new_fish = fma(0.5, log((1.0 + val1) / (1.0 - val1)), 0.5 * prev_fish);
        row[i] = want_signal ? signal_here : new_fish;
        prev_fish = new_fish;
    }
}

extern "C" __global__
void neoethos_fisher_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    nef_fisher_body(high, low, n, periods[r], first_valid, false,
                    out + (size_t)r * (size_t)n);
}

extern "C" __global__
void neoethos_fisher_signal_f64(const double* __restrict__ high,
                                const double* __restrict__ low,
                                int n,
                                const int* __restrict__ periods,
                                int n_combos,
                                int first_valid,
                                double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos || n <= 0) return;
    nef_fisher_body(high, low, n, periods[r], first_valid, true,
                    out + (size_t)r * (size_t)n);
}


// ===========================================================================
// S1 f64 LANE  --  fisher
// ===========================================================================
// Written by shard S1 of the f64 conversion, INTO THE FILE THIS INDICATOR
// ALREADY SHIPS IN, beside the f32 entry points that this crate's own f32
// wrappers still call. Listing this file in `F64_LANE_SOURCES` (build.rs) opts
// the WHOLE translation unit out of `--use_fast_math`, which is the only way
// the opt-out can be correct: the f32 and f64 entry points share one
// translation unit and nvcc has no per-entry flag.
//
// CPU reference: src/indicators/fisher.rs -- `fisher_scalar_into` (:300), `fisher_scalar_period9_into` (:361), `fisher_with_kernel` (:215)
//
// INPUT SHAPE: high and low ONLY. `compute_fisher_batch`
// (cpu_batch.rs:14760) calls `extract_high_low_input`, and
// `fisher_with_kernel` scans high and low for first_valid (fisher.rs:239-243).
// Close is never read, so this declares `HighLow` -- handing it an Ohlc ref
// would adopt close's first-valid and shift the whole series.
//
// PERIOD-BASED: `period` (default 9) is the swept parameter.
//
// ONE BODY SERVES BOTH CPU PATHS. `fisher_scalar_into` branches at period == 9
// to `fisher_scalar_period9_into`, which is the SAME loop with the nine
// midpoint updates written out. The order of the min/max updates is identical
// (ascending j, `>` for max and `<` for min, so ties keep the EARLIER value),
// and min/max are exact operations anyway -- there is no reassociation and
// therefore no second body here. Checked, not assumed, because period 9 is the
// default and would be the only branch this lane ever took.
//
// ARITHMETIC ORDER, and the two `mul_add`s that must stay `fma`:
//   `val1 = 0.67f64.mul_add(val1, 0.66 * ((hl - min)/range - 0.5))` -- ONE
//   rounding on the fused part. Writing `0.67*val1 + ...` would be two.
//   `new_fish = 0.5f64.mul_add(ln((1+val1)/(1-val1)), 0.5 * prev_fish)` -- same.
// Both are reproduced with `fma`, and the file is compiled with `-fmad=false`
// so the compiler contracts nothing else by accident.
//
// SENTINELS AND CONSTANTS, RE-DERIVED FOR f64:
//   `f64::MAX` = 1.7976931348623157e308 and `f64::MIN` = -1.7976931348623157e308
//   (the most NEGATIVE finite double, not the smallest positive one -- the f32
//   habit of writing `FLT_MIN` for a min-sentinel is exactly the bug this rule
//   exists to catch). Spelled as DBL_MAX / -DBL_MAX here.
//   `range.max(0.001)` -- 0.001 is a floor on the price range from the
//   published Fisher Transform, not a machine epsilon, so it does NOT scale
//   with precision and is carried over unchanged. It is `fmax`, matching
//   `f64::max`'s non-NaN-preferring semantics, not an if-chain.
//   The clamps 0.99 -> 0.999 are the published saturation and likewise fixed.
//
// PRIMARY OUTPUT: `fisher`. `compute_fisher_batch` maps output_id "value" to
// `out.fisher` (cpu_batch.rs:14777); `signal` is a second series and this lane
// carries one matrix per launch, so `signal` is not emitted.
//
// WARMUP: `alloc_with_nan_prefix(len, first + period - 1)`.
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

extern "C" __global__ void neoethos_fisher_batch_f64(
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

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s1_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s1_qnan();
    if (warm >= n) return;

    double prev_fish = 0.0;
    double val1 = 0.0;

    for (int i = warm; i < n; ++i) {
        const int start = i + 1 - period;

        double min_val =  1.7976931348623157e308;   // f64::MAX
        double max_val = -1.7976931348623157e308;   // f64::MIN
        for (int j = start; j <= i; ++j) {
            const double midpoint = 0.5 * (high[j] + low[j]);
            if (midpoint > max_val) max_val = midpoint;
            if (midpoint < min_val) min_val = midpoint;
        }

        const double range = fmax(max_val - min_val, 0.001);
        const double hl = 0.5 * (high[i] + low[i]);
        val1 = fma(0.67, val1, 0.66 * ((hl - min_val) / range - 0.5));
        if (val1 > 0.99) {
            val1 = 0.999;
        } else if (val1 < -0.99) {
            val1 = -0.999;
        }
        // `signal_out[i] = prev_fish` happens here in the CPU; the signal
        // series is not the emitted output, so only the state is carried.
        const double new_fish = fma(0.5, log((1.0 + val1) / (1.0 - val1)), 0.5 * prev_fish);
        row[i] = new_fish;
        prev_fish = new_fish;
    }
}
