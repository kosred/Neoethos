#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef FORCE_INLINE
#define FORCE_INLINE __forceinline__ __device__
#endif

// NeoEthos exposes only the reviewed f64 CUDA implementation. The superseded
// f32 packed/many-series kernels and their wrapper were removed after exact
// CPU/GPU parity and resident-output tests passed on real hardware.

static inline __device__ double f_nan_f64() { return CUDART_NAN; }
FORCE_INLINE void pivot_compute_levels_core(
    const int mode, const double h, const double l, const double c,
    const double previous_open, const double current_open,
    double &r4, double &r3, double &r2, double &r1, double &pp, double &s1, double &s2, double &s3, double &s4)
{
    const double d = h - l;


    r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan_f64();

    switch (mode) {

        case 0: {
            pp = (h + l + c) / 3.0;
            const double t2 = 2.0 * pp;
            const double t3 = 3.0 * pp;
            r1 = t2 - l;
            r2 = pp + d;
            r3 = t2 + h - 2.0 * l;
            r4 = t3 + h - 3.0 * l;
            s1 = t2 - h;
            s2 = pp - d;
            s3 = t2 - 2.0 * h + l;
            s4 = t3 - 3.0 * h + l;
            break;
        }

        case 1: {
            pp = (h + l + c) / 3.0;
            r1 = pp + 0.382 * d;
            r2 = pp + 0.618 * d;
            r3 = pp + d;
            s1 = pp - 0.382 * d;
            s2 = pp - 0.618 * d;
            s3 = pp - d;
            break;
        }

        case 2: {
            const double x = (c < previous_open)
                ? (h + 2.0 * l + c)
                : ((c > previous_open) ? (2.0 * h + l + c) : (h + l + 2.0 * c));
            pp = x / 4.0;
            r1 = x / 2.0 - l;
            s1 = x / 2.0 - h;
            break;
        }

        case 3: {
            pp = (h + l + c) / 3.0;
            const double scaled_range = 1.1 * d;
            const double c1 = scaled_range / 12.0;
            const double c2 = scaled_range / 6.0;
            const double c3 = scaled_range / 4.0;
            const double c4 = scaled_range / 2.0;
            r1 = c + c1;
            r2 = c + c2;
            r3 = c + c3;
            r4 = c + c4;
            s1 = c - c1;
            s2 = c - c2;
            s3 = c - c3;
            s4 = c - c4;
            break;
        }

        case 4: {
            pp = (h + l + 2.0 * current_open) / 4.0;
            const double t2p = 2.0 * pp;
            r1 = t2p - l;
            r2 = pp + d;
            r3 = h + 2.0 * (pp - l);
            r4 = r3 + d;
            s1 = t2p - h;
            s2 = pp - d;
            s3 = l - 2.0 * (h - pp);
            s4 = s3 - d;
            break;
        }
        default: {  break; }
    }
}

// One thread computes one (formula row, output bar) cell. Every output stays
// in its own resident row-major matrix; the host reads none of them between
// formulas or levels.
extern "C" __global__
void pivot_outputs_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ open,
    int n,
    const int* __restrict__ modes,
    int rows,
    double* __restrict__ out_r4,
    double* __restrict__ out_r3,
    double* __restrict__ out_r2,
    double* __restrict__ out_r1,
    double* __restrict__ out_pp,
    double* __restrict__ out_s1,
    double* __restrict__ out_s2,
    double* __restrict__ out_s3,
    double* __restrict__ out_s4)
{
    const int work = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * n;
    if (work >= total || n <= 0) return;

    const int row = work / n;
    const int index = work - row * n;
    double r4 = f_nan_f64(), r3 = f_nan_f64(), r2 = f_nan_f64();
    double r1 = f_nan_f64(), pp = f_nan_f64(), s1 = f_nan_f64();
    double s2 = f_nan_f64(), s3 = f_nan_f64(), s4 = f_nan_f64();

    if (index > 0) {
        const int previous = index - 1;
        const int mode = modes[row];
        const double h = high[previous];
        const double l = low[previous];
        const double c = close[previous];
        const double previous_open = open[previous];
        const double current_open = open[index];
        bool valid = false;
        if (mode == 0 || mode == 1 || mode == 3) {
            valid = isfinite(h) && isfinite(l) && isfinite(c);
        } else if (mode == 2) {
            valid = isfinite(h) && isfinite(l) && isfinite(c) && isfinite(previous_open);
        } else if (mode == 4) {
            valid = isfinite(h) && isfinite(l) && isfinite(current_open);
        }
        if (valid) {
            pivot_compute_levels_core(
                mode, h, l, c, previous_open, current_open,
                r4, r3, r2, r1, pp, s1, s2, s3, s4);
        }
    }

    out_r4[work] = r4;
    out_r3[work] = r3;
    out_r2[work] = r2;
    out_r1[work] = r1;
    out_pp[work] = pp;
    out_s1[work] = s1;
    out_s2[work] = s2;
    out_s3[work] = s3;
    out_s4[work] = s4;
}
/* Primary-only compatibility entry point. The period list is intentionally
 * ignored because Pivot has a formula selector, not a lookback period. This
 * entry point exposes the reviewed default formula's `pp`; the explicit
 * `pivot_outputs_f64` route above owns all five formulas and nine levels.
 *
 * One CUDA work item computes one (duplicate row, output bar) cell. Output bar
 * `t` is sourced from period `t-1`, so bar zero and the bar after any invalid
 * source period are undefined. No same-bar shortcut is permitted. */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void pivot_neo_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         const double* __restrict__ close,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int combo = (int)blockIdx.y;
    if (i >= n || combo >= n_combos || n <= 0) return;
    (void)periods;

    double value = NEO_F64_NAN;
    if (first_valid >= 0 && i > first_valid) {
        const int previous = i - 1;
        const double h = high[previous];
        const double l = low[previous];
        const double c = close[previous];
        if (isfinite(h) && isfinite(l) && isfinite(c)) {
            value = (h + l + c) / 3.0;
        }
    }
    out[(size_t)combo * (size_t)n + (size_t)i] = value;
}
