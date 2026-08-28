#include <cmath>
#include <cstdint>

/* FreeBSD msun k_cos/k_sin and the small-argument s_cos reduction.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft. Permission to use, copy, modify, and
 * distribute this software is freely granted, provided this notice is
 * preserved.
 *
 * Adaptive Bandpass only calls cos with finite positive arguments below
 * 2*pi/3 because length >= 6 and 0 < delta < 1. The Rust scalar carries the
 * same bounded routine. Keeping the exact constants, reduction and
 * parenthesisation avoids a host-libm versus CUDA-libdevice ULP split inside
 * the recursive state.
 */
static __device__ __forceinline__ double abto_ms_k_cos(double x, double y) {
    const double c1 = 0x1.555555555554cp-5;
    const double c2 = -0x1.6c16c16c15177p-10;
    const double c3 = 0x1.a01a019cb1590p-16;
    const double c4 = -0x1.27e4f809c52adp-22;
    const double c5 = 0x1.1ee9ebdb4b1c4p-29;
    const double c6 = -0x1.8fae9be8838d4p-37;
    const double z = x * x;
    const double w2 = z * z;
    const double r = z * (c1 + z * (c2 + z * c3))
        + w2 * w2 * (c4 + z * (c5 + z * c6));
    const double hz = 0.5 * z;
    const double w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + (z * r - x * y));
}

static __device__ __forceinline__ double abto_ms_k_sin(double x, double y) {
    const double s1 = -0x1.5555555555549p-3;
    const double s2 = 0x1.111111110f8a6p-7;
    const double s3 = -0x1.a01a019c161d5p-13;
    const double s4 = 0x1.71de357b1fe7dp-19;
    const double s5 = -0x1.ae5e68a2b9cebp-26;
    const double s6 = 0x1.5d93a5acfd57cp-33;
    const double z = x * x;
    const double w = z * z;
    const double r = s2 + z * (s3 + z * s4) + z * w * (s5 + z * s6);
    const double v = z * x;
    return x - ((z * (0.5 * y - v * r) - y) - v * s1);
}

static __device__ __forceinline__ void abto_reduce_pio2_near_half_pi(
    double x,
    unsigned int high,
    double* y0_out,
    double* y1_out) {
    const double inv_pio2 = 0x1.45f306dc9c883p-1;
    const double to_int = 0x1.8p+52;
    const double pio2_1 = 0x1.921fb54400000p+0;
    const double pio2_1t = 0x1.0b4611a626331p-34;
    const double pio2_2 = 0x1.0b4611a600000p-34;
    const double pio2_2t = 0x1.3198a2e037073p-69;
    const double pio2_3 = 0x1.3198a2e000000p-69;
    const double pio2_3t = 0x1.b839a252049c1p-104;

    const double tmp = x * inv_pio2 + to_int;
    const double f_n = tmp - to_int;
    double r = x - f_n * pio2_1;
    double w = f_n * pio2_1t;
    double y0 = r - w;
    const int ex = static_cast<int>(high >> 20);
    int ey = static_cast<int>(
        (static_cast<unsigned long long>(__double_as_longlong(y0)) >> 52) & 0x7ffULL);
    if (ex - ey > 16) {
        const double t = r;
        w = f_n * pio2_2;
        r = t - w;
        w = f_n * pio2_2t - ((t - r) - w);
        y0 = r - w;
        ey = static_cast<int>(
            (static_cast<unsigned long long>(__double_as_longlong(y0)) >> 52) & 0x7ffULL);
        if (ex - ey > 49) {
            const double t2 = r;
            w = f_n * pio2_3;
            r = t2 - w;
            w = f_n * pio2_3t - ((t2 - r) - w);
            y0 = r - w;
        }
    }
    *y0_out = y0;
    *y1_out = (r - y0) - w;
}

static __device__ __forceinline__ double abto_deterministic_cos(double x) {
    const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(x));
    const unsigned int high = static_cast<unsigned int>((bits >> 32) & 0x7fffffffULL);
    if (high <= 0x3fe921fbU) {
        return abto_ms_k_cos(x, 0.0);
    }
    double y0;
    double y1;
    if ((high & 0x000fffffU) == 0x000921fbU) {
        abto_reduce_pio2_near_half_pi(x, high, &y0, &y1);
    } else {
        const double pio2_1 = 0x1.921fb54400000p+0;
        const double pio2_1t = 0x1.0b4611a626331p-34;
        const double z = x - pio2_1;
        y0 = z - pio2_1t;
        y1 = (z - y0) - pio2_1t;
    }
    return -abto_ms_k_sin(y0, y1);
}

static __device__ inline double abto_median3(double x, double y, double z) {
    double min_xy = x < y ? x : y;
    double min_v = min_xy < z ? min_xy : z;
    double max_xy = x > y ? x : y;
    double max_v = max_xy > z ? max_xy : z;
    return (x + y + z) - min_v - max_v;
}

extern "C" __global__ void adaptive_bandpass_trigger_oscillator_batch_f64(
    const double* data,
    int len,
    const double* deltas,
    const double* alphas,
    int rows,
    double* out_in_phase,
    double* out_lead
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double delta = deltas[row];
    double alpha = alphas[row];
    if (!(isfinite(delta) && isfinite(alpha)) || delta <= 0.0 || delta >= 1.0 || alpha <= 0.0
        || alpha >= 1.0) {
        return;
    }

    const double pi = 3.14159265358979323846;
    const double float_tol = 1e-12;
    const double nan = NAN;
    const int in_phase_warmup = 11;
    const int lead_warmup = 12;

    double price[4] = {0.0, 0.0, 0.0, 0.0};
    double smooth_hist[2] = {0.0, 0.0};
    double c_hist[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dp_hist[4] = {0.0, 0.0, 0.0, 0.0};
    double q1_prev = 0.0;
    double i1_prev = 0.0;
    double ip_prev = 0.0;
    double p_prev = 0.0;
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    int valid_count = 0;

    double* row_in_phase = out_in_phase + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lead = out_lead + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            for (int j = 0; j < 4; ++j) {
                price[j] = 0.0;
            }
            smooth_hist[0] = 0.0;
            smooth_hist[1] = 0.0;
            for (int j = 0; j < 6; ++j) {
                c_hist[j] = 0.0;
            }
            for (int j = 0; j < 4; ++j) {
                dp_hist[j] = 0.0;
            }
            q1_prev = 0.0;
            i1_prev = 0.0;
            ip_prev = 0.0;
            p_prev = 0.0;
            bp_prev1 = 0.0;
            bp_prev2 = 0.0;
            valid_count = 0;
            row_in_phase[i] = nan;
            row_lead[i] = nan;
            continue;
        }

        price[3] = price[2];
        price[2] = price[1];
        price[1] = price[0];
        price[0] = value;

        int index = valid_count;
        valid_count += 1;

        double smooth =
            index >= 3 ? (price[0] + 2.0 * price[1] + 2.0 * price[2] + price[3]) / 6.0 : 0.0;

        double c = 0.0;
        if (index < 2) {
            c = 0.0;
        } else if (index < 7) {
            c = (price[0] - 2.0 * price[1] + price[2]) * 0.25;
        } else {
            double smooth_gain = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
            c = smooth_gain * (smooth - 2.0 * smooth_hist[0] + smooth_hist[1])
                + 2.0 * (1.0 - alpha) * c_hist[0]
                - (1.0 - alpha) * (1.0 - alpha) * c_hist[1];
        }

        double q1 =
            index >= 6
                ? (0.0962 * c + 0.5769 * c_hist[1] - 0.5769 * c_hist[3] - 0.0962 * c_hist[5])
                      * (0.5 + 0.08 * ip_prev)
                : 0.0;
        double i1 = index >= 3 ? c_hist[2] : 0.0;

        double dp_raw = 0.0;
        if (fabs(q1) > float_tol && fabs(q1_prev) > float_tol) {
            double denominator = 1.0 + (i1 * i1_prev) / (q1 * q1_prev);
            if (fabs(denominator) > float_tol) {
                dp_raw = ((i1 / q1) - (i1_prev / q1_prev)) / denominator;
            }
        }
        double dp = fmin(fmax(dp_raw, 0.1), 1.1);

        double md = 0.0;
        if (index >= 10) {
            md = abto_median3(dp, dp_hist[0], abto_median3(dp_hist[1], dp_hist[2], dp_hist[3]));
        }
        double dc = fabs(md) <= float_tol ? 15.0 : (2.0 * pi) / md + 0.5;
        double ip = 0.33 * dc + 0.67 * ip_prev;
        double p = 0.15 * ip + 0.85 * p_prev;

        double in_phase = nan;
        double lead = nan;
        if (index >= in_phase_warmup) {
            double length = fmax(p, 6.0);
            double beta = abto_deterministic_cos(2.0 * pi / length);
            double cos_angle = abto_deterministic_cos(4.0 * pi * delta / length);
            double denom = fabs(cos_angle) < float_tol
                               ? (cos_angle < 0.0 ? -float_tol : float_tol)
                               : cos_angle;
            double gamma = 1.0 / denom;
            double root = gamma * gamma - 1.0;
            if (root < 0.0) {
                root = 0.0;
            }
            double alpha_bp = gamma - sqrt(root);

            in_phase = 0.5 * (1.0 - alpha_bp) * (price[0] - price[2])
                + beta * (1.0 + alpha_bp) * bp_prev1 - alpha_bp * bp_prev2;
            if (index >= lead_warmup) {
                double quadrature = (in_phase - bp_prev1) * length / (2.0 * pi);
                lead = 0.5 * in_phase + 0.866 * quadrature;
            }
        }

        smooth_hist[1] = smooth_hist[0];
        smooth_hist[0] = smooth;

        c_hist[5] = c_hist[4];
        c_hist[4] = c_hist[3];
        c_hist[3] = c_hist[2];
        c_hist[2] = c_hist[1];
        c_hist[1] = c_hist[0];
        c_hist[0] = c;

        dp_hist[3] = dp_hist[2];
        dp_hist[2] = dp_hist[1];
        dp_hist[1] = dp_hist[0];
        dp_hist[0] = dp;

        q1_prev = q1;
        i1_prev = i1;
        ip_prev = ip;
        p_prev = p;

        if (isfinite(in_phase)) {
            bp_prev2 = bp_prev1;
            bp_prev1 = in_phase;
            row_in_phase[i] = in_phase;
            row_lead[i] = lead;
        } else {
            row_in_phase[i] = nan;
            row_lead[i] = nan;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE - adaptive_bandpass_trigger_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/adaptive_bandpass_trigger_oscillator.rs:490
 *             `..._output_row_from_slice`, driving `Stream::update` (:283).
 *
 * COLUMN: `in_phase`. This indicator has NO "value" output - its registry row
 * declares `OUTPUTS_IN_PHASE_LEAD` (registry.rs:2523) and the CPU batch
 * matches only `"in_phase"` and `"lead"` (cpu_batch.rs:8617). The lane emits
 * the FIRST declared output, `in_phase`, exactly as shard 3 does for
 * `di -> plus` and `kdj -> k`. Never `lead` silently.
 *
 * PERIOD-INVARIANT. The CPU batch reads `delta` (0.1) and `alpha` (0.07) and
 * never `period`, so a sweep yields identical columns and identical rows.
 *
 * FIRST-VALID IGNORED. The row builds a fresh stream and walks from index 0.
 * A non-finite bar calls `reset` (:286), which returns EVERY carried scalar -
 * the four-deep price ring, the six-deep `c` ring, the four-deep `dp` ring,
 * both bandpass lags and `valid_count` - to construction, so the 11-bar
 * warmup restarts after every hole. `valid_count`, not the bar index, is what
 * gates the warmup: bars are counted only while finite.
 *
 * EPSILON: FLOAT_TOL = 1e-12 (:37), an f64-sized guard on three DIFFERENT
 * denominators (q1, the dp denominator, and cos_angle). Carried across
 * unchanged - it is not an f32 machine epsilon and re-deriving it would move
 * the branch points relative to the CPU.
 *
 * TRANSCENDENTAL PARITY: both beta and cos_angle use the bounded msun cosine
 * above, exactly matching the scalar's bounded msun implementation. Calling
 * CUDA device `cos` here is forbidden: its valid 1-ULP difference from host
 * libm is carried recursively and becomes a larger output mismatch.
 *
 * NaN SEMANTICS: `p.max(6.0)` and `(gamma*gamma - 1.0).max(0.0)` are
 * `f64::max`, which returns the NON-NaN operand. `fmax` matches; an if-chain
 * would let a NaN survive into `beta` and then into every later bar through
 * `bp_prev1`. `median3` (:215) is likewise written with `f64::min`/`f64::max`
 * and is reproduced with `fmin`/`fmax`.
 *
 * `dp_raw.clamp(0.1, 1.1)` is the Rust `f64::clamp`: `< min -> min`,
 * `> max -> max`, otherwise SELF - so a NaN falls through both comparisons
 * and stays NaN. Written as that same pair of comparisons rather than as
 * `fmin(fmax(..))`, which would return the bound instead.
 *
 * SEQUENTIAL, one thread per combo column. Every ring is a fixed-size local
 * array (4 + 2 + 6 + 4 doubles), so there is no dynamic allocation and no
 * period bound to refuse.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

__device__ __forceinline__ double abpto_neo_median3_f64(double x, double y, double z)
{
    /* (x + y + z) - min - max, exactly as
       adaptive_bandpass_trigger_oscillator.rs:215. fmin/fmax so a NaN operand
       is dropped the way f64::min / f64::max drop it, not propagated. */
    return (x + y + z) - fmin(x, fmin(y, z)) - fmax(x, fmax(y, z));
}

extern "C" __global__
void adaptive_bandpass_trigger_oscillator_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods; (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    const double PI_F64    = 3.14159265358979311599796346854418516159057617187500;
    const double DELTA     = 0.1;    /* adaptive_bandpass_trigger_oscillator.rs:32 */
    const double ALPHA     = 0.07;   /* :33 */
    const double FLOAT_TOL = 1e-12;  /* :37 */
    const int IN_PHASE_WARMUP = 11;  /* MIN_VALID_SAMPLES - 1, :35 */

    double price[4]       = {0.0, 0.0, 0.0, 0.0};
    double smooth_hist[2] = {0.0, 0.0};
    double c_hist[6]      = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dp_hist[4]     = {0.0, 0.0, 0.0, 0.0};
    double q1_prev = 0.0, i1_prev = 0.0, ip_prev = 0.0, p_prev = 0.0;
    double bp_prev1 = 0.0, bp_prev2 = 0.0;
    int    valid_count = 0;

    for (int i = 0; i < len; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) { price[k] = 0.0; dp_hist[k] = 0.0; }
            smooth_hist[0] = 0.0; smooth_hist[1] = 0.0;
            #pragma unroll
            for (int k = 0; k < 6; ++k) c_hist[k] = 0.0;
            q1_prev = 0.0; i1_prev = 0.0; ip_prev = 0.0; p_prev = 0.0;
            bp_prev1 = 0.0; bp_prev2 = 0.0;
            valid_count = 0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        price[3] = price[2]; price[2] = price[1]; price[1] = price[0];
        price[0] = v;

        const int index = valid_count;
        valid_count += 1;

        const double smooth =
            (index >= 3)
                ? (price[0] + 2.0 * price[1] + 2.0 * price[2] + price[3]) / 6.0
                : 0.0;

        double c;
        if (index < 2) {
            c = 0.0;
        } else if (index < 7) {
            c = (price[0] - 2.0 * price[1] + price[2]) * 0.25;
        } else {
            const double smooth_gain = (1.0 - 0.5 * ALPHA) * (1.0 - 0.5 * ALPHA);
            c = smooth_gain * (smooth - 2.0 * smooth_hist[0] + smooth_hist[1])
                + 2.0 * (1.0 - ALPHA) * c_hist[0]
                - (1.0 - ALPHA) * (1.0 - ALPHA) * c_hist[1];
        }

        const double q1 =
            (index >= 6)
                ? (0.0962 * c + 0.5769 * c_hist[1]
                   - 0.5769 * c_hist[3] - 0.0962 * c_hist[5])
                      * (0.5 + 0.08 * ip_prev)
                : 0.0;
        const double i1 = (index >= 3) ? c_hist[2] : 0.0;

        double dp_raw = 0.0;
        if (fabs(q1) > FLOAT_TOL && fabs(q1_prev) > FLOAT_TOL) {
            const double denominator = 1.0 + (i1 * i1_prev) / (q1 * q1_prev);
            dp_raw = (fabs(denominator) > FLOAT_TOL)
                         ? ((i1 / q1) - (i1_prev / q1_prev)) / denominator
                         : 0.0;
        }
        double dp = dp_raw;
        if (dp < 0.1)      dp = 0.1;
        else if (dp > 1.1) dp = 1.1;

        const double md =
            (index >= 10)
                ? abpto_neo_median3_f64(
                      dp, dp_hist[0],
                      abpto_neo_median3_f64(dp_hist[1], dp_hist[2], dp_hist[3]))
                : 0.0;
        const double dc = (fabs(md) <= FLOAT_TOL) ? 15.0
                                                  : (2.0 * PI_F64) / md + 0.5;

        const double ip = 0.33 * dc + 0.67 * ip_prev;
        const double p  = 0.15 * ip + 0.85 * p_prev;

        double in_phase = NEO_F64_NAN;
        if (index >= IN_PHASE_WARMUP) {
            const double length    = fmax(p, 6.0);
            const double beta = abto_deterministic_cos(2.0 * PI_F64 / length);
            const double cos_angle =
                abto_deterministic_cos(4.0 * PI_F64 * DELTA / length);
            double denom;
            if (fabs(cos_angle) < FLOAT_TOL) {
                denom = signbit(cos_angle) ? -FLOAT_TOL : FLOAT_TOL;
            } else {
                denom = cos_angle;
            }
            const double gamma    = 1.0 / denom;
            const double alpha_bp = gamma - sqrt(fmax(gamma * gamma - 1.0, 0.0));

            in_phase = 0.5 * (1.0 - alpha_bp) * (price[0] - price[2])
                       + beta * (1.0 + alpha_bp) * bp_prev1
                       - alpha_bp * bp_prev2;
            /* `lead` (index >= LEAD_WARMUP = 12) is the OTHER output and is
               not emitted by this lane; named here so the omission is visible
               rather than forgotten. */
        }

        smooth_hist[1] = smooth_hist[0];
        smooth_hist[0] = smooth;

        c_hist[5] = c_hist[4]; c_hist[4] = c_hist[3]; c_hist[3] = c_hist[2];
        c_hist[2] = c_hist[1]; c_hist[1] = c_hist[0]; c_hist[0] = c;

        dp_hist[3] = dp_hist[2]; dp_hist[2] = dp_hist[1];
        dp_hist[1] = dp_hist[0]; dp_hist[0] = dp;

        q1_prev = q1; i1_prev = i1; ip_prev = ip; p_prev = p;

        if (isfinite(in_phase)) {
            bp_prev2 = bp_prev1;
            bp_prev1 = in_phase;
            o[i] = in_phase;
        } else {
            /* `update` returns None and the row writes NaN. The bandpass lags
               are NOT advanced in that case - the CPU only shifts them inside
               the is_finite arm (:404). */
            o[i] = NEO_F64_NAN;
        }
    }
}
