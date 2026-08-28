#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline double compute_phase_angle(double real, double imaginary) {
    double phase_angle = 0.0;
    if (fabs(real) > 0.001) {
        phase_angle = atan(imaginary / real) * 180.0 / CUDART_PI;
    } else if (imaginary > 0.0) {
        phase_angle = 90.0;
    } else if (imaginary < 0.0) {
        phase_angle = -90.0;
    }
    if (real < 0.0) {
        phase_angle += 180.0;
    }
    phase_angle += 90.0;
    if (phase_angle < 0.0) {
        phase_angle += 360.0;
    }
    if (phase_angle > 360.0) {
        phase_angle -= 360.0;
    }
    return phase_angle;
}
}

extern "C" __global__ void l1_ehlers_phasor_batch_f64(
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
    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length <= 0 || length > len) {
        return;
    }

    double angle = 2.0 * CUDART_PI / static_cast<double>(length);
    for (int end = length - 1; end < len; ++end) {
        bool valid = true;
        double real = 0.0;
        double imaginary = 0.0;
        for (int j = 0; j < length; ++j) {
            double value = data[end - j];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            double theta = angle * static_cast<double>(j);
            real += cos(theta) * value;
            imaginary += sin(theta) * value;
        }
        if (!valid) {
            continue;
        }
        row[end] = compute_phase_angle(real, imaginary);
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// WHY A SECOND ENTRY POINT RATHER THAN REGISTERING THE ONE ABOVE.
// `l1_ehlers_phasor_batch_f64` (:29) recomputes the FULL length-wide dot
// product at every bar. The CPU does that ONCE, at `warm`, and then advances
// the phasor by a ROTATION plus an in/out term:
//   l1_ehlers_phasor.rs:472-478
//     real = cos_angle*prev_real - sin_angle*prev_imaginary + value - removed;
//     imaginary = sin_angle*prev_real + cos_angle*prev_imaginary;
// Those are two different accumulations of the same mathematical quantity and
// they differ in the last bits, which then compound bar over bar because the
// recurrence feeds itself. The lane needs the CPU order, so it gets its own
// entry point and the brute-force one is left to the f32 wrappers that call it.
//
// CPU REFERENCE
// -------------
//   src/indicators/l1_ehlers_phasor.rs
//     :235 resolve_params      -- cos_angle/sin_angle and the weight tables
//     :294 compute_phase_angle
//     :418 compute_l1_ehlers_phasor_into   <- clean-then-core selection
//     :441 compute_l1_ehlers_phasor_clean  <- the fast path
//     :369 Core::update                    <- the fallback path
//     :276 validate_input                  <- first_valid and the warmup guard
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW walking bars ascending. The phasor rotation is a
// two-state linear recurrence; there is no bar-parallel form that keeps the CPU
// rounding.
//
// PERIOD-INVARIANT. `compute_l1_ehlers_phasor_batch` (cpu_batch.rs:10210) reads
// ONLY `domestic_cycle_length` (default 15, l1_ehlers_phasor.rs:30) and never
// `period`, so a sweep of five periods produces five identical CPU columns and
// this kernel emits five identical rows.
//
// ARITHMETIC
// ----------
// f64 end to end. No f32 literal, no f32-suffixed function, no fast-math
// intrinsic; the file is listed in `F64_LANE_SOURCES` so the whole translation
// unit compiles `-fmad=false -prec-div=true -prec-sqrt=true` and never with
// `--use_fast_math`. No epsilon is introduced: the only tolerance in the
// indicator is the CPU own `real.abs() > 0.001`, carried over unchanged because
// it is a magnitude threshold on a price-scale quantity, not a float-width
// guard.

#define L1_NEO_DEFAULT_CYCLE 15
#define L1_NEO_PI 3.14159265358979323846

__device__ __forceinline__ double l1_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `compute_phase_angle` (:294), with the CPU `else { 0.0 }` arm written out.
// `f64::to_degrees` is `self * (180.0 / PI)` -- ONE multiply by a constant, not
// a divide, so it is spelled that way here.
__device__ __forceinline__ double l1_neo_phase_angle(double real, double imaginary) {
    double phase_angle;
    if (fabs(real) > 0.001) {
        phase_angle = atan(imaginary / real) * (180.0 / L1_NEO_PI);
    } else if (imaginary > 0.0) {
        phase_angle = 90.0;
    } else if (imaginary < 0.0) {
        phase_angle = -90.0;
    } else {
        phase_angle = 0.0;
    }
    if (real < 0.0) {
        phase_angle += 180.0;
    }
    phase_angle += 90.0;
    if (phase_angle < 0.0) {
        phase_angle += 360.0;
    }
    if (phase_angle > 360.0) {
        phase_angle -= 360.0;
    }
    return phase_angle;
}

extern "C" __global__ void l1_ehlers_phasor_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = l1_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);

    // PERIOD-INVARIANT: `periods` is deliberately unread, exactly as the CPU
    // batch never reads `period`.
    (void)periods;

    const int len = L1_NEO_DEFAULT_CYCLE;
    for (int i = 0; i < n; ++i) o[i] = nan_d;

    if (n <= 0 || first_valid < 0 || first_valid >= n) return;
    // validate_input :283 -- `valid < domestic_cycle_length` is an Err, which
    // the batch turns into an all-NaN row.
    if (n - first_valid < len) return;

    // cos_angle / sin_angle (:245) and the weight tables (:254-259). The CPU
    // builds `theta = angle * j` then `theta.cos()` / `theta.sin()`; the same
    // expression is evaluated here, in the same order, per j.
    const double angle = 2.0 * L1_NEO_PI / static_cast<double>(len);
    const double cos_angle = cos(angle);
    const double sin_angle = sin(angle);

    const int warm = first_valid + len - 1;
    if (warm >= n) return;

    // ---- clean path (:441). Bails on the FIRST non-finite, exactly like the
    // CPU `return false`.
    bool clean = true;
    double real = 0.0;
    double imaginary = 0.0;
    for (int j = 0; j < len; ++j) {
        const double value = data[warm - j];
        if (!isfinite(value)) { clean = false; break; }
        const double theta = angle * static_cast<double>(j);
        real += cos(theta) * value;
        imaginary += sin(theta) * value;
    }
    if (clean) {
        o[warm] = l1_neo_phase_angle(real, imaginary);
        for (int i = warm + 1; i < n; ++i) {
            const double value = data[i];
            const double removed = data[i - len];
            if (!isfinite(value) || !isfinite(removed)) { clean = false; break; }
            const double prev_real = real;
            const double prev_imaginary = imaginary;
            real = cos_angle * prev_real - sin_angle * prev_imaginary + value - removed;
            imaginary = sin_angle * prev_real + cos_angle * prev_imaginary;
            o[i] = l1_neo_phase_angle(real, imaginary);
        }
    }
    if (clean) return;

    // ---- core path (:418 falls through to :369). The CPU RESTARTS from index
    // 0 with a fresh Core and overwrites everything the clean attempt wrote, so
    // the row is refilled here before the walk.
    for (int i = 0; i < n; ++i) o[i] = nan_d;

    double ring[L1_NEO_DEFAULT_CYCLE];
    for (int j = 0; j < len; ++j) ring[j] = nan_d;
    int idx = 0, count = 0, invalid_count = 0;
    bool phasor_valid = false;
    real = nan_d;
    imaginary = nan_d;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        const bool full_before = (count == len);
        const double removed = full_before ? ring[idx] : nan_d;
        if (full_before && !isfinite(removed) && invalid_count > 0) invalid_count -= 1;

        ring[idx] = value;
        if (!isfinite(value)) invalid_count += 1;
        if (!full_before) count += 1;

        double output;
        if (count < len || invalid_count > 0) {
            phasor_valid = false;
            output = nan_d;
        } else {
            if (phasor_valid && full_before && isfinite(value) && isfinite(removed)) {
                const double prev_real = real;
                const double prev_imaginary = imaginary;
                real = cos_angle * prev_real - sin_angle * prev_imaginary + value - removed;
                imaginary = sin_angle * prev_real + cos_angle * prev_imaginary;
            } else {
                // recompute_window (:353) with current_idx == idx, and
                // ring_get_lag (:346) `(current_idx + len - (lag % len)) % len`.
                double rc = 0.0, ic = 0.0;
                for (int j = 0; j < len; ++j) {
                    int k = idx + len - (j % len);
                    while (k >= len) k -= len;
                    const double v = ring[k];
                    const double theta = angle * static_cast<double>(j);
                    rc += cos(theta) * v;
                    ic += sin(theta) * v;
                }
                real = rc;
                imaginary = ic;
                phasor_valid = true;
            }
            output = l1_neo_phase_angle(real, imaginary);
        }
        o[i] = output;

        idx += 1;
        if (idx == len) idx = 0;
    }
}
