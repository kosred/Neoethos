#include <cuda_runtime.h>
#include <math.h>

#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif


__device__ __forceinline__ float qnan() {
    return __int_as_float(0x7fc00000);
}


template <typename T>
__device__ __forceinline__ T ro_load(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}


__device__ __forceinline__ void fill_nan_prefix(float* ptr, int len) {
    const float nanv = qnan();
    for (int i = 0; i < len; ++i) ptr[i] = nanv;
}


__device__ __forceinline__ void dm_step(float ch, float cl, float& prev_h, float& prev_l,
                                        float& plus_val, float& minus_val)
{
    const float dp = ch - prev_h;
    const float dm = prev_l - cl;
    prev_h = ch;
    prev_l = cl;

    const float ap = (dp > 0.0f) ? dp : 0.0f;
    const float am = (dm > 0.0f) ? dm : 0.0f;


    const bool take_p = (ap > am);
    plus_val  = take_p ? ap : 0.0f;
    minus_val = take_p ? 0.0f : am;
}


struct CompSum {
    float s;
    float c;
    __device__ __forceinline__ void init() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {

        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};


struct CompEMA {
    float s;
    float c;
    __device__ __forceinline__ void init(float s0) { s = s0; c = 0.0f; }
    __device__ __forceinline__ void update(float one_minus_rp, float x) {

        float prod = s * one_minus_rp;
        float perr = __fmaf_rn(s, one_minus_rp, -prod);

        float y = (x + perr) - c;
        float t = prod + y;
        c = (t - prod) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};


extern "C" __global__
void dm_batch_f32(const float* __restrict__ high,
                  const float* __restrict__ low,
                  const int*   __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  float* __restrict__ plus_out,
                  float* __restrict__ minus_out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* plus_row  = plus_out  + combo * series_len;
    float* minus_row = minus_out + combo * series_len;

    const int p = periods[combo];
    if (p <= 0) {

        fill_nan_prefix(plus_row, series_len);
        fill_nan_prefix(minus_row, series_len);
        return;
    }
    if (first_valid < 0 || first_valid + p - 1 >= series_len) {
        fill_nan_prefix(plus_row, series_len);
        fill_nan_prefix(minus_row, series_len);
        return;
    }

    const int i0 = first_valid;
    const int warm_end = i0 + p - 1;


    if (warm_end > 0) {
        fill_nan_prefix(plus_row,  warm_end);
        fill_nan_prefix(minus_row, warm_end);
    }


    float prev_h = ro_load(high + i0);
    float prev_l = ro_load(low  + i0);


    CompSum wplus, wminus; wplus.init(); wminus.init();
    for (int i = i0 + 1; i <= warm_end; ++i) {
        const float ch = ro_load(high + i);
        const float cl = ro_load(low  + i);
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);
        if (pv != 0.0f) wplus.add(pv);
        if (mv != 0.0f) wminus.add(mv);
    }


    plus_row [warm_end] = wplus.value();
    minus_row[warm_end] = wminus.value();


    if (warm_end + 1 >= series_len) return;

    const float rp = 1.0f / (float)p;
    const float one_minus_rp = 1.0f - rp;


    CompEMA splus, sminus;
    splus.init(plus_row [warm_end]);
    sminus.init(minus_row[warm_end]);

    for (int i = warm_end + 1; i < series_len; ++i) {
        const float ch = ro_load(high + i);
        const float cl = ro_load(low  + i);

        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);

        splus.update(one_minus_rp, pv);
        sminus.update(one_minus_rp, mv);

        plus_row [i] = splus.value();
        minus_row[i] = sminus.value();
    }
}


extern "C" __global__
void dm_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (period <= 0 || fv < 0 || fv + period - 1 >= rows) {

        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            plus_tm [idx] = qnan();
            minus_tm[idx] = qnan();
        }
        return;
    }


    auto at = [&](int t) { return t * cols + s; };

    const int i0 = fv;
    const int warm_end = i0 + period - 1;


    for (int t = 0; t < warm_end; ++t) {
        const int idx = at(t);
        plus_tm [idx] = qnan();
        minus_tm[idx] = qnan();
    }

    float prev_h = ro_load(high_tm + at(i0));
    float prev_l = ro_load(low_tm  + at(i0));


    CompSum wplus, wminus; wplus.init(); wminus.init();
    for (int t = i0 + 1; t <= warm_end; ++t) {
        const float ch = ro_load(high_tm + at(t));
        const float cl = ro_load(low_tm  + at(t));
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);
        if (pv != 0.0f) wplus.add(pv);
        if (mv != 0.0f) wminus.add(mv);
    }

    plus_tm [at(warm_end)] = wplus.value();
    minus_tm[at(warm_end)] = wminus.value();

    if (warm_end + 1 >= rows) return;

    const float rp = 1.0f / (float)period;
    const float one_minus_rp = 1.0f - rp;

    CompEMA splus, sminus;
    splus.init(plus_tm [at(warm_end)]);
    sminus.init(minus_tm[at(warm_end)]);

    for (int t = warm_end + 1; t < rows; ++t) {
        const float ch = ro_load(high_tm + at(t));
        const float cl = ro_load(low_tm  + at(t));
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);

        splus.update(one_minus_rp, pv);
        sminus.update(one_minus_rp, mv);

        plus_tm [at(t)] = splus.value();
        minus_tm[at(t)] = sminus.value();
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

__device__ __forceinline__ double qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// One operation-for-operation f64 authority serves the row-major pair ABI,
// the time-major public ABI, and the preserved plus-only primary ABI. The
// input/output strides are elements, not bytes. `minus_out == nullptr` is the
// primary route and does not change the plus arithmetic.
__device__ __forceinline__ void dm_row_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int input_stride,
    int len,
    int period,
    int first_valid,
    double* __restrict__ plus_out,
    double* __restrict__ minus_out,
    int output_stride)
{
    const double nan = qnan_f64();
    for (int i = 0; i < len; ++i) {
        plus_out[i * output_stride] = nan;
        if (minus_out != nullptr) minus_out[i * output_stride] = nan;
    }
    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len) return;
    if (len - first_valid < period) return;

    const int end_init = first_valid + period - 1;
    double sum_plus = 0.0;
    double sum_minus = 0.0;
    double prev_high = ro_load(high + first_valid * input_stride);
    double prev_low = ro_load(low + first_valid * input_stride);

    for (int i = first_valid + 1; i <= end_init; ++i) {
        const double hi = ro_load(high + i * input_stride);
        const double lo = ro_load(low + i * input_stride);
        const double diff_p = hi - prev_high;
        const double diff_m = prev_low - lo;
        prev_high = hi;
        prev_low = lo;
        if (diff_p > 0.0 && diff_p > diff_m) {
            sum_plus += diff_p;
        } else if (diff_m > 0.0 && diff_m > diff_p) {
            sum_minus += diff_m;
        }
    }

    plus_out[end_init * output_stride] = sum_plus;
    if (minus_out != nullptr) minus_out[end_init * output_stride] = sum_minus;
    if (end_init + 1 >= len) return;

    const double inv_p = 1.0 / (double)period;
    for (int i = end_init + 1; i < len; ++i) {
        const double hi = ro_load(high + i * input_stride);
        const double lo = ro_load(low + i * input_stride);
        const double diff_p = hi - prev_high;
        const double diff_m = prev_low - lo;
        prev_high = hi;
        prev_low = lo;

        double plus_value = 0.0;
        double minus_value = 0.0;
        if (diff_p > 0.0 && diff_p > diff_m) {
            plus_value = diff_p;
        } else if (diff_m > 0.0 && diff_m > diff_p) {
            minus_value = diff_m;
        }
        // Keep the shipped scalar CPU branch's three-rounding Wilder update.
        sum_plus = sum_plus - (sum_plus * inv_p) + plus_value;
        sum_minus = sum_minus - (sum_minus * inv_p) + minus_value;
        plus_out[i * output_stride] = sum_plus;
        if (minus_out != nullptr) minus_out[i * output_stride] = sum_minus;
    }
}
extern "C" __global__
void dm_batch_f64(const double* __restrict__ high,
                  const double* __restrict__ low,
                  const int*   __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  double* __restrict__ plus_out,
                  double* __restrict__ minus_out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    dm_row_f64(
        high,
        low,
        1,
        series_len,
        periods[combo],
        first_valid,
        plus_out + combo * series_len,
        minus_out + combo * series_len,
        1);
}
extern "C" __global__
void dm_many_series_one_param_time_major_f64(
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    double* __restrict__ plus_tm,
    double* __restrict__ minus_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    dm_row_f64(
        high_tm + s,
        low_tm + s,
        cols,
        rows,
        period,
        first_valids[s],
        plus_tm + s,
        minus_tm + s,
        cols);
}


/* ===========================================================================
 * NEOETHOS f64 LANE - dm (directional movement)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/dm.rs `dm_compute_into_scalar` for the canonical
 *             plus/minus pair and `dm_compute_selected_scalar::<true>` for the
 *             preserved PLUS primary, both prepared by `dm_prepare`.
 *
 * COLUMNS: canonical `plus` then `minus`; no unversioned `value` alias.
 *
 * PERIOD-SWEPT: `compute_dm_batch` reads `period` (default 14).
 *
 * FIRST-VALID: `!is_nan` on high and low SIMULTANEOUSLY (:191-194) - the
 * common two-series rule, `AllInputsNonNan` over `F64InputKind::HighLow`.
 * NOT `donchian`s max-of-independent-scans rule: this one zips the two
 * iterators and takes the first index where both are non-NaN.
 *
 * WARMUP: the batch matrix arrives pre-filled with NaN and the CPU writes its
 * first value at `end_init = first + period - 1` (:355). That value is the
 * RAW SUM of the seed window, not an average - `out[end_init] = sum` - which
 * is why no `inv_p` appears before the smoothing loop.
 *
 * SEED ORDER: the seed loop runs `i = first + 1 ..= end_init` adding one
 * qualifying `diff_p` at a time from 0.0, ascending. That is `period - 1`
 * additions, not `period`, because the first bar only supplies `prev_high` /
 * `prev_low`. Reproduced literally.
 *
 * THE ROUNDING THE CRATE DISAGREES WITH ITSELF ABOUT. dm.rs:381-389 carries
 * TWO forms of the Wilder step behind a `cfg(target_feature = "fma")`:
 *
 *     fma:      sum = (-inv_p).mul_add(sum, sum + val)   -- TWO roundings
 *     default:  sum = sum - (sum * inv_p) + val          -- THREE roundings
 *
 * `target_feature = "fma"` is NOT set on a stock `cargo build` for
 * x86_64-unknown-linux-gnu or x86_64-pc-windows-msvc - it requires an
 * explicit `-C target-feature=+fma` or a `target-cpu` that implies it - so
 * the shipped scalar path, the one `Kernel::ScalarBatch` runs and the one
 * `hpc_ta` takes, is the THREE-rounding form. This kernel reproduces THAT,
 * left-associated exactly as written: `(sum - (sum * inv_p)) + val`.
 * Choosing the fma form here would make the device disagree with the default
 * CPU build while agreeing with a build nobody ships.
 *
 * SELECTION: `diff_p > 0.0 && diff_p > diff_m` - two strict comparisons, so a
 * tie between the two directional moves contributes NOTHING, and a NaN
 * `diff_p` makes both comparisons false and also contributes nothing. That is
 * the CPU behaviour and it is why no `fmax` appears here: there is no
 * `f64::max` in the reference to mistranslate.
 *
 * SEQUENTIAL, one thread per combo column: a Wilder recurrence.
 * =========================================================================== */

extern "C" __global__
void dm_neo_batch_f64(const double* __restrict__ high,
                      const double* __restrict__ low,
                      int series_len,
                      const int* __restrict__ periods,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    dm_row_f64(
        high,
        low,
        1,
        series_len,
        periods[combo],
        first_valid,
        out + (size_t)combo * (size_t)series_len,
        nullptr,
        1);
}
