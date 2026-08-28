#include <cuda_runtime.h>
#include <math.h>

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3   (corrected_moving_average)
 *
 * WRITTEN FROM SCRATCH. Before this file there was no `.cu` for this indicator
 * anywhere under `kernels/cuda`, no wrapper, and no F64_KERNELS row -- a card
 * present and no kernel, which this workflow exists to close.
 *
 * CPU reference: `corrected_moving_average_into_slice`
 *   (src/indicators/moving_averages/corrected_moving_average.rs:410) driving
 *   `CorrectedMovingAverageStream::update` (:365), with `RollingStats::update`
 *   (:235) and `gain_factor` (:300). The kernel selector is accepted and then
 *   never consulted (:415-424 only VALIDATES it), so there is one CPU path and
 *   therefore one oracle -- no scalar-vs-AVX seed question here.
 *
 * Column: the id resolves through the MA dispatcher --
 *   `ma.rs:263` ("corrected_moving_average" | "cma") reads a parameter
 *   literally named `period`, so this kernel IS period-swept and reads
 *   `periods[combo]`.
 *
 * Input: ONE price series, CPU source `close` -> F64InputKind::CloseSlice.
 *
 * FIRST-VALID IGNORED. `corrected_moving_average_into_slice` fills the WHOLE
 *   destination with NaN (:432) and then walks EVERY bar from index 0 (:436),
 *   and `RollingStats::update` RESETS the window on any non-finite value
 *   (:236-239) while `update` (:369-374) simultaneously drops `prev_cma`. So
 *   there is no start index to adopt: a caller's first-valid would skip bars
 *   the CPU processes, and the mid-series reset is what reproduces the warmup.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `prev_cma` at bar i is a
 *   function of its own value at bar i-1 through a gain the bar itself
 *   computes, and `gain_factor` is a fixed-point iteration -- neither has a
 *   parallel-scan form.
 *
 * NO RING BUFFER, and the reason is a property of the CPU code rather than an
 *   optimisation: the window is a VecDeque that is CLEARED on every non-finite
 *   bar and otherwise pushed once per bar, so whenever it is full its contents
 *   are the CONTIGUOUS run `data[i-period+1 ..= i]` and the value popped is
 *   exactly `data[i - period]`. That index is a direct read.
 *
 * Roundings, counted against the CPU lines:
 *   :242  self.sum += value; self.sumsq += value * value;   -- plain, NO fma
 *   :246  self.sum -= oldest; self.sumsq -= oldest * oldest; -- plain
 *   :254  let mean = self.sum / denom;
 *   :255  let variance = (self.sumsq / denom - mean * mean).max(0.0);
 *   :382  let v2 = (prev - sma).powi(2);          -- ONE multiply
 *   :386  v2 / (variance + v2)
 *   :389  prev + k * (sma - prev)                 -- plain, NO fma
 *   :310  k = v3 * k_prev * (2.0 - k_prev);       -- plain
 *   Not one `mul_add` appears on this path, so not one `fma` appears here.
 *   Fusing :389 into `fma(k, sma - prev, prev)` would REMOVE a rounding the
 *   reference performs.
 *
 * NaN semantics: `.max(0.0)` at :255 is `f64::max`, which returns the NON-NaN
 *   operand -- `fmax` is used here for exactly that reason. `variance` feeds
 *   the gain, the gain feeds `prev_cma`, and `prev_cma` is carried, so a NaN
 *   admitted by an if-chain would poison every later bar of the row.
 *
 * Epsilons: `f64::EPSILON` at :384 and :386 and :304 is 2.220446049250313e-16.
 *   It is spelled out below as `DBL_EPSILON`-valued and NOT copied from any
 *   f32 constant -- an `FLT_EPSILON`-sized 1.1920929e-7 here would declare a
 *   variance "zero" eight orders of magnitude too early and pin the gain at
 *   1.0, which is a different indicator that still returns plausible numbers.
 *   The `1e-5` at :313 is a CONVERGENCE tolerance on a dimensionless gain in
 *   [0,1], not a machine epsilon, and is carried across unchanged.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* f64::EPSILON -- corrected_moving_average.rs:304, :384, :386. */
#define NEO_CMA_F64_EPSILON 2.220446049250313e-16
/* The fixed-point convergence tolerance and iteration cap -- :309, :313. */
#define NEO_CMA_GAIN_TOL    1e-5
#define NEO_CMA_GAIN_ITERS  64

/* `gain_factor` -- :300-322. */
__device__ __forceinline__ double neo_cma_gain_factor(double v3)
{
    if (!isfinite(v3)) return 0.0;
    if (fabs(v3 - 1.0) <= NEO_CMA_F64_EPSILON) return 1.0;

    double k_prev = 1.0;
    double k      = 1.0;
    for (int it = 0; it < NEO_CMA_GAIN_ITERS; ++it) {
        k = v3 * k_prev * (2.0 - k_prev);
        const double err = k_prev - k;
        k_prev = k;
        if (err <= NEO_CMA_GAIN_TOL) break;
    }
    if (!isfinite(k)) return 0.0;
    /* `k.clamp(0.0, 1.0)` -- :318. */
    if (k < 0.0) return 0.0;
    if (k > 1.0) return 1.0;
    return k;
}

extern "C" __global__
void corrected_moving_average_neo_batch_f64(const double* __restrict__ prices,
                                            int n,
                                            const int* __restrict__ periods,
                                            int n_combos,
                                            int first_valid,
                                            double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)first_valid; /* the mid-series reset reproduces it -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = periods[combo];

    /* `validate_input` -- :273-292. */
    if (period <= 0 || period > n) return;               /* InvalidPeriod  */
    {
        bool any_finite = false;
        int  cur = 0, best = 0;
        for (int i = 0; i < n; ++i) {
            if (isfinite(prices[i])) { any_finite = true; cur += 1; if (cur > best) best = cur; }
            else                     { cur = 0; }
        }
        if (!any_finite) return;                         /* AllValuesNaN       */
        if (best < period) return;                       /* NotEnoughValidData */
    }

    const double denom = (double)period;

    double sum      = 0.0;
    double sumsq    = 0.0;
    int    count    = 0;
    double prev_cma = 0.0;
    bool   have_prev = false;

    for (int i = 0; i < n; ++i) {
        const double value = prices[i];

        /* `RollingStats::update` :236-239 -- a non-finite bar CLEARS the
         * window, and `update` :369-374 then drops `prev_cma`. */
        if (!isfinite(value)) {
            sum = 0.0; sumsq = 0.0; count = 0;
            have_prev = false;
            continue;                                    /* the bar stays NaN */
        }

        sum   += value;
        sumsq += value * value;
        count += 1;
        if (count > period) {
            /* The window is the contiguous run since the last reset, so the
             * value leaving the front is exactly `prices[i - period]`. */
            const double oldest = prices[i - period];
            sum   -= oldest;
            sumsq -= oldest * oldest;
            count -= 1;
        }
        if (count < period) continue;                    /* :251 -- None */

        const double mean     = sum / denom;
        /* `.max(0.0)` is `f64::max`: a NaN variance becomes 0.0. */
        const double variance = fmax(sumsq / denom - mean * mean, 0.0);

        double cma;
        if (!have_prev) {
            cma = mean;                                  /* :378 */
        } else {
            const double d  = prev_cma - mean;
            const double v2 = d * d;                     /* :382 `powi(2)` */
            const double v3 = (variance <= NEO_CMA_F64_EPSILON || v2 <= NEO_CMA_F64_EPSILON)
                                ? 1.0
                                : (v2 / (variance + v2));
            const double k  = neo_cma_gain_factor(v3);
            cma = prev_cma + k * (mean - prev_cma);       /* :389 -- NO fma */
        }
        prev_cma  = cma;
        have_prev = true;
        o[i]      = cma;
    }
}
