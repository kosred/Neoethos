// elastic_volume_weighted_moving_average (EVWMA) — CUDA f64 kernel.
// PRODUCTION/SEARCH AUTHORITY:
// evwma_rolling_volume_close_length_key_default30_chronological_rn_f64_v1
//
// WHAT THIS REPLACES
// ------------------
// NOTHING. There was no `.cu` for this indicator, no wrapper, and no
// `F64_KERNELS` row, so the f64 lane answered `CudaF64KernelMissing`.
//
// CPU REFERENCE — src/indicators/moving_averages/elastic_volume_weighted_moving_average.rs
// -----------------------------------------------------------------------------------------
//   :308 find_first_valid          — `price.is_finite() && volume.is_finite()`
//   :320 prepare                   — validation
//   :357 compute_absolute_into     — the `use_volume_sum == false` branch
//   :382 compute_volume_sum_into   — THE BRANCH THIS KERNEL IMPLEMENTS
//   :431 compute_into              — the selector
//   :452 elastic_volume_weighted_moving_average — the entry the brief names
//
// WHICH BRANCH, AND WHY — READ THIS BEFORE CHANGING IT
// ----------------------------------------------------
// EVWMA has two mutually exclusive modes and `use_volume_sum` selects between
// them. The DIRECT API default is `false` (:31), which takes
// `compute_absolute_into` — and that branch NEVER READS `length`. It is
// period-INVARIANT: every row of a period sweep would be the same series.
//
// The f64 lane is a PERIOD SWEEP. The route that sweeps a period through this
// indicator is the generic moving-average dispatcher, and it is explicit:
//
//   ma.rs:1105-1113   length: Some(period), use_volume_sum: Some(true)
//   registry.rs:608   "Direct API defaults to absolute-volume mode; generic MA
//                      period-based routes use the length parameter with
//                      volume-sum mode."
//
// So the oracle for a request that carries `periods[]` is the volume-sum
// branch with `length = period`, and that is what is written here. The
// absolute-volume branch is not reachable through this ABI at all — it has no
// period to sweep — so implementing it would add a mode this lane can never
// select.
//
// SHAPE — ONE THREAD PER COLUMN, BARS ASCENDING
// ---------------------------------------------
// `prev` carries the previous OUTPUT into the next bar, and `rolling_sum`
// carries the running volume window. Two carried scalars over a serial
// recurrence: one thread per column, bars in ascending order, no scan
// reformulation.
//
// THE RING IS NOT EMULATED, AND THAT CHANGES NOTHING
// --------------------------------------------------
// The CPU keeps a `length`-slot ring of `volume_value` (:383). At the bar where
// it is read, `ring[head]` is by construction the value inserted `length` steps
// earlier — step k being bar `first_valid + k` — so it is exactly
//
//   volume_value(i - length) = volumes[i-length].is_finite() ? volumes[i-length] : 0.0
//
// which this kernel reads straight out of the volume series. Same value, same
// order of the `rolling_sum` updates, so the same last bit — and no
// `length`-sized local array, which matters because `length` may be up to
// MAX_LENGTH = 4096 (:32).
//
// ARITHMETIC
// ----------
// f64 end to end; no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. The file is listed in `F64_LANE_SOURCES` so the translation unit
// is never compiled with `--use_fast_math`.
//
// `((rolling_sum - volume) * base + volume * price) / rolling_sum` (:424) is
// FOUR roundings before the divide — subtract, two multiplies, add — and it is
// written that way, NOT as an fma. The CPU does not fuse it and neither does
// this kernel; `-fmad=false` guarantees nvcc will not fuse it either.

#include <cmath>
#include <cstdint>

// elastic_volume_weighted_moving_average.rs:32
#define EVWMA_MAX_LENGTH 4096

__device__ __forceinline__ double evwma_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// The `if volume.is_finite() { volume } else { 0.0 }` of :385 and :399.
__device__ __forceinline__ double evwma_volume_value(double volume) {
    return isfinite(volume) ? volume : 0.0;
}

extern "C" __global__ void elastic_volume_weighted_moving_average_neo_batch_f64(
    const double* __restrict__ prices,
    const double* __restrict__ volumes,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int length = periods[r];

    // prepare (:320) — EmptyInputData, validate_length (:286) and the
    // AllValuesNaN that `find_first_valid` (:308) raises when no bar has both
    // a finite price and a finite volume. The caller's first_valid uses that
    // same rule (F64FirstValidRule::PriceVolumeFinite).
    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (length <= 0) || (length > EVWMA_MAX_LENGTH);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = evwma_qnan();
        return;
    }

    // `alloc_with_nan_prefix(prices.len(), first_valid)` (:462).
    for (int i = 0; i < first_valid && i < n; ++i) row[i] = evwma_qnan();

    double rolling_sum = 0.0;
    double prev = evwma_qnan();
    int count = 0;

    for (int index = first_valid; index < n; ++index) {
        const double price = prices[index];
        const double volume = volumes[index];
        const double volume_value = evwma_volume_value(volume);

        if (count < length) {
            // :396-399 — the warm branch adds without subtracting.
            rolling_sum += volume_value;
            count += 1;
        } else {
            // :401 — `rolling_sum += volume_value - ring[head]`. `ring[head]`
            // is the volume_value inserted `length` steps ago; see the header.
            const double old = evwma_volume_value(volumes[index - length]);
            rolling_sum += volume_value - old;
        }

        // :412-419. Note this is the RAW volume being tested, not
        // `volume_value` — a non-finite volume contributes 0.0 to the window
        // AND kills the bar, and it resets `prev` so the next bar re-seeds
        // `base` from its own price.
        if (!isfinite(price) || !isfinite(volume) || !isfinite(rolling_sum) ||
            rolling_sum == 0.0) {
            row[index] = evwma_qnan();
            prev = evwma_qnan();
            continue;
        }

        const double base = isfinite(prev) ? prev : price;
        const double value =
            ((rolling_sum - volume) * base + volume * price) / rolling_sum;
        row[index] = value;
        prev = value;
    }
}
