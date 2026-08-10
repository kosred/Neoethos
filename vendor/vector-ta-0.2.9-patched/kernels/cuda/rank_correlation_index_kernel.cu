#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double rci_compute_window(const double* data, int start, int length) {
    double len_f = static_cast<double>(length);
    double denom = len_f * static_cast<double>(length * length - 1);
    double sum = 0.0;

    for (int c = 0; c < length; ++c) {
        double p = data[start + c];
        double o = 1.0;
        double s = 0.0;
        for (int j = 0; j < length; ++j) {
            double other = data[start + j];
            if (p < other) {
                o += 1.0;
            } else if (p == other) {
                s += 1.0;
            }
        }
        double ord = o + (s - 1.0) * 0.5;
        double time_rank = static_cast<double>(length - c);
        double diff = time_rank - ord;
        sum += diff * diff;
    }

    return (1.0 - 6.0 * sum / denom) * 100.0;
}

extern "C" __global__ void rank_correlation_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (length < 2) {
        return;
    }

    int run_start = 0;
    for (int t = 0; t < len; ++t) {
        if (!isfinite(data[t])) {
            run_start = t + 1;
            continue;
        }
        if (t - run_start + 1 < length) {
            continue;
        }
        row[t] = rci_compute_window(data, t + 1 - length, length);
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `rank_correlation_index_with_kernel`
// (src/indicators/rank_correlation_index.rs:377) -> `rank_correlation_index_
// prepare` (:224) for the validity rules and `compute_window_rci` (:251) for
// the value. `compute_window_rci_12` (:278) is the length==12 specialisation;
// it is NOT a second answer -- its ranks are the same exact half-integers, its
// denominator is the same 1716, and its `sum` runs over `c` in the same order,
// so one implementation serves both.
//
// WHY ONE KERNEL SERVES BOTH CPU PATHS. `rank_correlation_index_compute_into`
// (:355) picks the window path when `is_fast_path_clean` (:219) holds and the
// stream otherwise; `RankCorrelationIndexStream::compute_from_ring` (:458) is
// `compute_window_rci` over the same `length` values in the same time order,
// and the stream emits exactly when `length` finite values have arrived since
// its last reset. So "the last `length` values are all finite" is the single
// emit condition for both, and the window computation is shared.
//
// This is the ORDER STATISTIC shape: ranks come from counting comparisons
// (`o` strictly-greater, `s` ties) rather than from a sort, which is what the
// CPU does and which needs no shared memory and no per-thread array.
//
// WARMUP: `first + length - 1`.
//
// f32 -> f64 audit: this file's `rci_compute_window` was already f64 and is
// reused verbatim rather than duplicated. No f32 literal, no f32-suffixed math
// function, no fast-math intrinsic, and no epsilon -- the CPU compares two
// doubles with `<` and `==`, and a tolerance would merge ranks the host keeps
// distinct.
// ===========================================================================

static __device__ __forceinline__ double neo_rci_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void neoethos_rank_correlation_index_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_rci_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int length = periods[r];
    const int first  = first_valid;

    // prepare :235 (length < 2 || length > data_len) and :240 (valid < length).
    if (length < 2 || length > n) return;
    if (first < 0 || first >= n) return;
    if ((n - first) < length) return;

    int run = 0;
    for (int i = first; i < n; ++i) {
        if (!isfinite(data[i])) { run = 0; continue; }
        ++run;
        if (run < length) continue;
        row[i] = rci_compute_window(data, i + 1 - length, length);
    }
}
