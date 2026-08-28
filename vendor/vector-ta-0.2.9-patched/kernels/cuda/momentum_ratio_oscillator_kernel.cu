#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double mro_safe_ratio(double num, double den) {
    if (isfinite(num) && isfinite(den) && den != 0.0) {
        return num / den;
    }
    return CUDART_NAN;
}

extern "C" __global__ void momentum_ratio_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out_line,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row_line = out_line + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_line[t] = CUDART_NAN;
        row_signal[t] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    double alpha = 2.0 / static_cast<double>(period);
    bool has_ema = false;
    double ema_prev = 0.0;
    double emaa_prev = 0.0;
    double emab_prev = 0.0;
    double val_prev = CUDART_NAN;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!isfinite(value)) {
            has_ema = false;
            ema_prev = 0.0;
            emaa_prev = 0.0;
            emab_prev = 0.0;
            val_prev = CUDART_NAN;
            continue;
        }

        double prev_ema_nz = has_ema ? ema_prev : 0.0;
        double ema = prev_ema_nz + alpha * (value - prev_ema_nz);
        double ratioa = has_ema ? mro_safe_ratio(ema, ema_prev) : CUDART_NAN;
        double emaa_input = isfinite(ratioa) && ratioa < 1.0 ? ratioa : 0.0;
        double emab_input = isfinite(ratioa) && ratioa > 1.0 ? ratioa : 0.0;
        double emaa = emaa_prev + alpha * (emaa_input - emaa_prev);
        double emab = emab_prev + alpha * (emab_input - emab_prev);
        double ratiob = mro_safe_ratio(ratioa, ratioa + emab);

        double val = CUDART_NAN;
        double denom = ratioa + ratiob * emaa;
        if (isfinite(ratioa) && isfinite(ratiob) && isfinite(emaa) && isfinite(denom) &&
            denom != 0.0) {
            val = 2.0 * ratioa / denom - 1.0;
        }

        row_line[t] = val;
        row_signal[t] = val_prev;

        has_ema = true;
        ema_prev = ema;
        emaa_prev = emaa;
        emab_prev = emab;
        val_prev = val;
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// CPU REFERENCE
// -------------
//   src/indicators/momentum_ratio_oscillator.rs
//     :242 first_valid_with_second  -- `is_finite`, and FEWER THAN TWO finite
//                                     values is an Err (all-NaN row)
//     :283 safe_ratio
//     :292 momentum_ratio_oscillator_compute_into  <- the whole specification
//     :406 with_kernel -- `alloc_uninit_f64`, NO NaN prefix, because the walk
//          below writes EVERY index starting at 0
//   dispatch: cpu_batch.rs:11663, param `period` (default 50); `output_id`
//   "value" resolves to `out.line` (:11713).
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW walking bars ascending. Three EMAs feed each
// other through `ratioa`, and a non-finite bar RESETS all of them, so the state
// at bar i depends on every bar before it.
//
// PERIOD-SWEPT: `alpha = 2.0 / period` (:299).
//
// MULTI-OUTPUT: the kernel emits `line`, which is the column the CPU batch
// produces for `output_id == "value"` (:11713). `signal` is `line` delayed one
// bar and is NOT emitted here; a caller that wants it must ask for it by name
// through the CPU path, exactly as before.
//
// FIRST-VALID IGNORED: `compute_into` (:307) loops `for i in 0..data.len()` and
// never consults an index. Declaring a warmup here would blank bars the CPU
// fills.
//
// ARITHMETIC
// ----------
// f64 end to end, no fast-math, no f32-suffixed function. `safe_ratio` is
// reproduced with its three exact guards (`is_finite` on both operands and
// `den != 0.0`) rather than being folded into an `isfinite(num/den)` test,
// which would accept a 0/0 that the CPU rejects. The `val` guard chain is the
// CPU's five conditions in the CPU's order. No epsilon exists here and none was
// invented.

__device__ __forceinline__ double mro_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// :283 safe_ratio
__device__ __forceinline__ double mro_neo_safe_ratio(double num, double den) {
    if (isfinite(num) && isfinite(den) && den != 0.0) return num / den;
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void momentum_ratio_oscillator_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = mro_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) o[i] = nan_d;

    (void)first_valid;   // FIRST-VALID IGNORED -- see the header

    const int period = periods[row];
    if (n <= 0) return;
    if (period <= 0 || period > n) return;          // :390 InvalidPeriod

    // :242 first_valid_with_second -- at least TWO finite bars, or Err.
    int finite_seen = 0;
    for (int i = 0; i < n && finite_seen < 2; ++i) {
        if (isfinite(data[i])) finite_seen += 1;
    }
    if (finite_seen < 2) return;

    const double alpha = 2.0 / static_cast<double>(period);

    double ema_prev = 0.0, emaa_prev = 0.0, emab_prev = 0.0;
    bool has_ema = false, has_emaa = false, has_emab = false;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            o[i] = nan_d;
            ema_prev = 0.0; emaa_prev = 0.0; emab_prev = 0.0;
            has_ema = false; has_emaa = false; has_emab = false;
            continue;
        }

        const double prev_ema_nz = has_ema ? ema_prev : 0.0;
        const double ema = prev_ema_nz + alpha * (value - prev_ema_nz);
        const double ratioa = has_ema ? mro_neo_safe_ratio(ema, ema_prev) : nan_d;

        const double prev_emaa_nz = has_emaa ? emaa_prev : 0.0;
        const double prev_emab_nz = has_emab ? emab_prev : 0.0;
        const double emaa_input = (isfinite(ratioa) && ratioa < 1.0) ? ratioa : 0.0;
        const double emab_input = (isfinite(ratioa) && ratioa > 1.0) ? ratioa : 0.0;
        const double emaa = prev_emaa_nz + alpha * (emaa_input - prev_emaa_nz);
        const double emab = prev_emab_nz + alpha * (emab_input - prev_emab_nz);
        const double ratiob = mro_neo_safe_ratio(ratioa, ratioa + emab);

        const double denom = ratioa + ratiob * emaa;
        double val;
        if (isfinite(ratioa) && isfinite(ratiob) && isfinite(emaa)
            && isfinite(denom) && denom != 0.0) {
            val = 2.0 * ratioa / denom - 1.0;
        } else {
            val = nan_d;
        }

        o[i] = val;

        ema_prev = ema;
        emaa_prev = emaa;
        emab_prev = emab;
        has_ema = true; has_emaa = true; has_emab = true;
    }
}
