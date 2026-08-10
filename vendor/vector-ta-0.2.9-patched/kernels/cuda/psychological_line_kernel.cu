#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void psychological_line_batch_f32(
    const float* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    float* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    float* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    float scale = 100.0f / static_cast<float>(length);

    for (int t = 0; t < len; ++t) {
        if (t < length) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t - length;
        bool valid = true;
        int rising = 0;

        for (int i = start; i <= t; ++i) {
            if (!isfinite(data[i])) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        for (int i = start + 1; i <= t; ++i) {
            rising += static_cast<int>(data[i] > data[i - 1]);
        }

        row[t] = static_cast<float>(rising) * scale;
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

extern "C" __global__ void psychological_line_batch_f64(
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

    if (length <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN;
        }
        return;
    }

    double scale = 100.0 / static_cast<double>(length);

    for (int t = 0; t < len; ++t) {
        if (t < length) {
            row[t] = CUDART_NAN;
            continue;
        }

        int start = t - length;
        bool valid = true;
        int rising = 0;

        for (int i = start; i <= t; ++i) {
            if (!isfinite(data[i])) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN;
            continue;
        }

        for (int i = start + 1; i <= t; ++i) {
            rising += static_cast<int>(data[i] > data[i - 1]);
        }

        row[t] = static_cast<double>(rising) * scale;
    }
}

// ===========================================================================
// f64 LANE  --  closer 4
//
// CPU reference: `psychological_line_with_kernel`
// (src/indicators/psychological_line.rs:311) -> `psychological_line_prepare`
// (:222) for the validity rules, `psychological_line_compute_fast_checked`
// (:249) for the clean-data path and `PsychologicalLineStream::update` (:395)
// for the path the CPU takes when any value after `first` is non-finite.
//
// WHY ONE KERNEL SERVES BOTH CPU PATHS.
// The fast path keeps a rolling integer `count` of up-closes over the last
// `length` COMPARISONS and multiplies it by `100.0 / length`. The stream path
// keeps the same integer count but RESETS on every non-finite value. Both emit
// exactly when `length` comparisons are available since the last reset, and
// both multiply an EXACT INTEGER by the same scale -- so recomputing the count
// by scanning the window reproduces either path bit for bit. There is no
// floating-point accumulation to preserve the order of: `count` is an integer
// and `count * scale` is a single rounding in both.
//
// WARMUP: `first + length` -- one bar later than the `first + length - 1` most
// windowed indicators use, because the value at bar i is built from
// COMPARISONS (i vs i-1), so `length` of them need `length + 1` bars.
//
// f32 -> f64 audit of this file: the f32 entry point above builds its NaN with
// `__int_as_float(0x7fffffff)`; the f64 lane uses the f64 quiet-NaN bit
// pattern via `__longlong_as_double`. No f32 literal, no f32-suffixed math
// function, no fast-math intrinsic. No epsilon exists in this indicator on the
// CPU and none was invented -- the only comparison is `current > previous`,
// which is exactly what the host does.
// ===========================================================================

// This file's original includes are `cuda_runtime.h` alone (qstick) or
// `cuda_runtime.h` + `math_constants.h` (psychological_line); the f64 lane
// below calls `isfinite`, so pull in the header that declares it rather than
// relying on a transitive include.
#include <math.h>

static __device__ __forceinline__ double neo_psyl_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void neoethos_psychological_line_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_combos) return;

    const double nan_d = neo_psyl_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(r) * static_cast<size_t>(n);
    if (n <= 0) return;

    for (int i = 0; i < n; ++i) row[i] = nan_d;

    const int length = periods[r];
    const int first  = first_valid;

    // psychological_line_prepare :233 (length == 0 || length > data_len) and
    // :238 (valid <= length -> NotEnoughValidData). The CPU returns Err, which
    // collect_f64 turns into "no column"; the device emits an all-NaN row.
    if (length <= 0 || length > n) return;
    if (first < 0 || first >= n) return;
    if ((n - first) <= length) return;

    const double scale = 100.0 / static_cast<double>(length);

    // `run` = consecutive finite values ending at i, counted from `first`
    // (the stream is only ever fed data[first..], :286). `run - 1` is the
    // number of comparisons available since the last reset.
    int run = 0;
    for (int i = first; i < n; ++i) {
        const double v = data[i];
        if (!isfinite(v)) { run = 0; continue; }   // stream reset, :429-431
        ++run;
        if (run - 1 < length) continue;            // :410-412 / :261-268

        // Count the up-closes over the last `length` comparisons. Integer, so
        // no accumulation order to preserve.
        int count = 0;
        for (int k = 0; k < length; ++k) {
            const int j = i - k;
            count += (data[j] > data[j - 1]) ? 1 : 0;
        }
        row[i] = static_cast<double>(count) * scale;  // :268 / :424
    }
}
