#include <cmath>
#include <cstdint>

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
            double beta = cos(2.0 * pi / length);
            double cos_angle = cos(4.0 * pi * delta / length);
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
            const double beta      = cos(2.0 * PI_F64 / length);
            const double cos_angle = cos(4.0 * PI_F64 * DELTA / length);
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
