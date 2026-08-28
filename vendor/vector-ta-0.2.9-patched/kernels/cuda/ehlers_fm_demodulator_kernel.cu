#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
struct Coefficients {
    double c1;
    double c2;
    double c3;
};

__device__ inline Coefficients coefficients(int period) {
    double period_f = static_cast<double>(period);
    double a1 = exp(-1.414 * CUDART_PI / period_f);
    double b1 = 2.0 * a1 * cos(1.414 * CUDART_PI / period_f);
    Coefficients out;
    out.c2 = b1;
    out.c3 = -(a1 * a1);
    out.c1 = 1.0 - out.c2 - out.c3;
    return out;
}
}

extern "C" __global__ void ehlers_fm_demodulator_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    Coefficients coeffs = coefficients(period);
    int warmup_bars = period > 3 ? period - 3 : 0;
    int valid_count = 0;
    double prev_hl = 0.0;
    double ss1 = 0.0;
    double ss2 = 0.0;

    for (int i = 0; i < len; ++i) {
        double open_value = open[i];
        double close_value = close[i];
        if (isnan(open_value) || isnan(close_value)) {
            valid_count = 0;
            prev_hl = 0.0;
            ss1 = 0.0;
            ss2 = 0.0;
            continue;
        }

        double derivative = close_value - open_value;
        double hl = fmin(fmax(10.0 * derivative, -1.0), 1.0);
        double value = valid_count < 3
            ? derivative
            : coeffs.c1 * (hl + prev_hl) * 0.5 + coeffs.c2 * ss1 + coeffs.c3 * ss2;

        prev_hl = hl;
        ss2 = ss1;
        ss1 = value;
        valid_count += 1;

        if (valid_count > warmup_bars) {
            row[i] = value;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — ehlers_fm_demodulator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ehlers_fm_demodulator.rs:318 `compute_scalar_into`.
 *
 * Column: `expect_value_output` then `out.values` (cpu_batch.rs:3760, :3825).
 *
 * PERIOD-SWEPT: `compute_ehlers_fm_demodulator_batch` (cpu_batch.rs:3811) reads
 *   a parameter literally named `period` (default 30), so the swept int IS this
 *   indicator parameter.
 *
 * Input: OPEN and CLOSE. High and low are never read — the batch destructures
 *   them only to length-check (cpu_batch.rs:3765-3777). Served by
 *   F64InputKind::Ohlc4, which the resident OHLCV upload already carries, so no
 *   new upload shape is needed; the kernel simply ignores the two it does not
 *   use. first-valid is `OpenCloseNonNan` (batch_prepare, :566) — the first
 *   index at which OPEN and CLOSE are both non-NaN, which is a different bar
 *   from the OHLC quadruple whenever high or low has the earlier hole.
 *
 * A hole INSIDE the series does not merely emit NaN: it resets `prev_hl`,
 *   `ss1`, `ss2` and `valid_count` (:331-334), so the two-pole filter restarts
 *   its three-bar seeding. Emitting NaN and carrying the state would produce a
 *   different series for every bar after the first gap.
 *
 * The warmup is `period - 3` VALID bars (saturating, :320) and the test is
 *   `valid_count > warmup_bars` AFTER the increment — an off-by-one that is
 *   reproduced literally rather than normalised.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void ehlers_fm_demodulator_neo_batch_f64(const double* __restrict__ open,
                                         const double* __restrict__ high,
                                         const double* __restrict__ low,
                                         const double* __restrict__ close,
                                         int n,
                                         const int* __restrict__ periods,
                                         int n_combos,
                                         int first_valid,
                                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)high; (void)low;   /* length-checked by the CPU, never read */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = periods[combo];
    if (period <= 0) return;
    const int fv = (first_valid < 0) ? 0 : first_valid;
    if (fv >= n) return;

    /* coefficients (:235) */
    const double NEO_PI = 3.14159265358979323846;
    const double period_f = (double)period;
    const double a1 = exp(-1.414 * NEO_PI / period_f);
    const double b1 = 2.0 * a1 * cos(1.414 * NEO_PI / period_f);
    const double c2 = b1;
    const double c3 = -(a1 * a1);
    const double c1 = 1.0 - c2 - c3;

    /* warmup_bars = period.saturating_sub(3) (:320) */
    const int warmup_bars = (period > 3) ? (period - 3) : 0;

    double prev_hl = 0.0, ss1 = 0.0, ss2 = 0.0;
    long long valid_count = 0;

    for (int i = fv; i < n; ++i) {
        const double op = open[i], cl = close[i];
        if (isnan(op) || isnan(cl)) {
            o[i] = NEO_F64_NAN;
            prev_hl = 0.0; ss1 = 0.0; ss2 = 0.0; valid_count = 0;
            continue;
        }

        const double derivative = cl - op;
        /* (10.0 * derivative).clamp(-1.0, 1.0) — Rust `clamp` panics on a NaN
         * bound, never on a NaN value, and returns the value unchanged only if
         * it compares inside; a NaN derivative is impossible here because both
         * prices passed the isnan gate above. */
        double hl = 10.0 * derivative;
        if (hl < -1.0) hl = -1.0;
        else if (hl > 1.0) hl = 1.0;

        const double value = (valid_count < 3)
            ? derivative
            : (c1 * (hl + prev_hl) * 0.5 + c2 * ss1 + c3 * ss2);

        prev_hl = hl;
        ss2 = ss1;
        ss1 = value;
        ++valid_count;

        o[i] = (valid_count > (long long)warmup_bars) ? value : NEO_F64_NAN;
    }
}
