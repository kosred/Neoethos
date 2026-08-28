#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void rolling_z_score_trend_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookbacks,
    int n_combos,
    double* __restrict__ out_zscore,
    double* __restrict__ out_momentum
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int lookback = lookbacks[combo_idx];
    double* row_zscore = out_zscore + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_momentum =
        out_momentum + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_zscore[t] = CUDART_NAN;
        row_momentum[t] = CUDART_NAN;
    }

    if (lookback <= 0) {
        return;
    }

    bool has_smoothed = false;
    double smoothed = CUDART_NAN;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!isfinite(value)) {
            has_smoothed = false;
            smoothed = CUDART_NAN;
            continue;
        }

        double sum = 0.0;
        double sumsq = 0.0;
        int count = 0;

        for (int i = t; i >= 0 && count < lookback; --i) {
            double v = data[i];
            if (!isfinite(v)) {
                break;
            }
            sum += v;
            sumsq += v * v;
            count += 1;
        }

        if (count < lookback) {
            continue;
        }

        double n = static_cast<double>(lookback);
        double mean = sum / n;
        double variance = sumsq / n - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        double stddev = sqrt(variance);
        double raw_zscore = stddev > DBL_EPSILON ? (value - mean) / stddev : 0.0;

        if (!has_smoothed) {
            smoothed = raw_zscore;
            has_smoothed = true;
            row_zscore[t] = smoothed;
            row_momentum[t] = CUDART_NAN;
            continue;
        }

        double prev_smoothed = smoothed;
        smoothed = 0.5 * raw_zscore + 0.5 * prev_smoothed;
        row_zscore[t] = smoothed;
        row_momentum[t] = smoothed - prev_smoothed;
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `rolling_z_score_trend_with_kernel`
// (src/indicators/rolling_z_score_trend.rs:475) -> `validate_common` (:257)
// for the validity rules, `compute_row_all_finite` (:286) for the clean path
// and `RollingZScoreTrendStream::update` (:391) for the path taken when any
// value is non-finite. This kernel emits the ZSCORE series.
//
// WHICH OUTPUT, AND WHY IT IS NAMED HERE. `compute_rolling_z_score_trend_batch`
// (cpu_batch.rs:8020) accepts `output_id` "zscore" or "momentum" and REJECTS
// "value" (:8053). So unlike most rows in this lane there is no CPU column
// under the default output id, and a parity check must ask the CPU for
// "zscore" explicitly. Stated here rather than discovered later.
//
// WHY ONE KERNEL SERVES BOTH CPU PATHS. `compute_row` (:441) picks the
// all-finite path when `longest_valid_run(data) == data.len()` and the stream
// otherwise. The two bodies are the same arithmetic; the stream additionally
// RESETS (`:382-389`) on a non-finite value. With no non-finite value the
// reset never fires, so a single walk that resets on non-finite is faithful to
// both.
//
// NO `first_valid`. This indicator starts at index 0 -- both paths iterate the
// whole series -- so the row is registered with `F64FirstValidRule::Ignored`
// and the argument is not read. That is the CPU's behaviour, not a shortcut:
// `compute_row_all_finite` runs `for i in 0..data.len()`.
//
// THE RING NEEDS NO PER-THREAD ARRAY. The CPU keeps `window[lookback]` purely
// to know which value leaves the window; at the step where `count == lookback`
// the value at `window[head]` is always `data[i - lookback]` counting from the
// start of the current finite run, and `count == lookback` guarantees
// `i - lookback` is inside that run. Reading it straight out of global memory
// is the same double, so `sum` and `sumsq` follow the host update exactly:
//   sum   = sum   + (v - old)                                        (:322)
//   sumsq = sumsq + ((v*v) - (old*old))                              (:323)
//
// NaN SEMANTICS. `variance` is `(sumsq/n - mean*mean).max(0.0)` (:331). Rust's
// `f64::max` returns the NON-NaN operand, so `fmax` is used below and NOT an
// if-chain: a comparison against NaN is false, which would let a NaN survive
// into `sqrt` and poison every later bar of the carried `smoothed`.
//
// EPSILON. `stddev > f64::EPSILON` (:333) is ALREADY an f64 epsilon --
// 2.220446049250313e-16 -- and is carried across unchanged rather than
// re-sized. It is a guard against dividing by a collapsed standard deviation,
// not an f32 tolerance that needed re-deriving.
//
// WARMUP: `zscore_warmup_prefix = lookback - 1` (:276). The first emitted bar
// is the one at which `count` first reaches `lookback`.
//
// f32 -> f64 audit of this file: it has NO f32 entry point -- the only other
// `__global__` here, `rolling_z_score_trend_batch_f64` (:6), is already double
// in and double out, and the single `float` token in the file is the
// `#include <float.h>` on line 2 that supplies `DBL_EPSILON`. So the inventory
// flag on this file is a header include, not an f32 code path. Below: the f64
// quiet-NaN bit pattern, `fmax`, `sqrt`, and plain arithmetic. No f32 literal,
// no f32-suffixed math function, no fast-math intrinsic.
// ===========================================================================

static __device__ __forceinline__ double neo_rzst_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

#define NEO_RZST_EPS 2.2204460492503131e-16

extern "C" __global__ void neoethos_rolling_z_score_trend_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    (void)first_valid;   // :475 iterates from index 0; see the header.

    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_rzst_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int lookback = periods[r];
    if (lookback <= 0 || lookback > n) return;      // validate_params :247

    // validate_common :262-272 -- the longest run of finite values must reach
    // `lookback`, otherwise the CPU returns Err and produces no column.
    {
        int best = 0, cur = 0;
        for (int i = 0; i < n; ++i) {
            if (isfinite(data[i])) { ++cur; if (cur > best) best = cur; }
            else cur = 0;
        }
        if (best < lookback) return;
    }

    const double nn = static_cast<double>(lookback);

    int    count       = 0;
    int    run_start   = 0;
    double sum         = 0.0;
    double sumsq       = 0.0;
    double smoothed    = 0.0;
    bool   has_smooth  = false;

    for (int i = 0; i < n; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {                          // reset, :382-389
            count = 0; run_start = i + 1;
            sum = 0.0; sumsq = 0.0;
            smoothed = 0.0; has_smooth = false;
            continue;
        }

        if (count < lookback) {                      // :306-314
            ++count;
            sum   += v;
            sumsq += v * v;
        } else {                                     // :315-324
            const double old = data[i - lookback];
            (void)run_start;
            sum   += v - old;
            sumsq += v * v - old * old;
        }

        if (count < lookback) continue;              // :326-328

        const double mean     = sum / nn;                              // :330
        const double variance = fmax(sumsq / nn - mean * mean, 0.0);   // :331
        const double stddev   = sqrt(variance);                        // :332
        const double raw      = (stddev > NEO_RZST_EPS)
                                    ? (v - mean) / stddev
                                    : 0.0;                             // :333-337

        if (!has_smooth) {                            // :339-342
            smoothed   = raw;
            has_smooth = true;
        } else {                                      // :344-346
            const double prev = smoothed;
            smoothed = 0.5 * raw + 0.5 * prev;
        }
        row[i] = smoothed;
    }
}
