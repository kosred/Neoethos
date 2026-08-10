#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void ewma_volatility_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    const double* __restrict__ alphas,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double alpha = alphas[combo_idx];
    double beta = 1.0 - alpha;
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (period <= 0 || !isfinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
        return;
    }

    int valid_count = 0;
    int seed_idx = -1;
    double seed_sum = 0.0;

    for (int i = 1; i < len; ++i) {
        double prev = data[i - 1];
        double curr = data[i];
        if (!isfinite(prev) || !isfinite(curr) || prev <= 0.0 || curr <= 0.0) {
            continue;
        }

        double ret = log(curr / prev);
        double sq = ret * ret;
        if (valid_count < period) {
            seed_sum += sq;
        }
        valid_count += 1;

        if (valid_count == period) {
            seed_idx = i;
            break;
        }
    }

    if (seed_idx < 0) {
        return;
    }

    double ema = seed_sum / static_cast<double>(period);
    row[seed_idx] = sqrt(fmax(ema, 0.0)) * 100.0;

    for (int i = seed_idx + 1; i < len; ++i) {
        double prev = data[i - 1];
        double curr = data[i];
        if (isfinite(prev) && isfinite(curr) && prev > 0.0 && curr > 0.0) {
            double ret = log(curr / prev);
            double sq = ret * ret;
            ema = beta * ema + alpha * sq;
        }
        row[i] = sqrt(fmax(ema, 0.0)) * 100.0;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — ewma_volatility
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ewma_volatility.rs:284 `seed_ewma_single_pass`
 *   plus :305 `fill_row_single_pass`. That pair — not the `EwmaVolatilityStream`
 *   at :426 — is what `ewma_volatility_with_kernel` (:359) runs, and it is what
 *   the batch dispatcher reaches through `collect_f64`.
 *
 * Column: `expect_value_output` then `out.values` (cpu_batch.rs:10298, :10320).
 *
 * PERIOD-INVARIANT: `compute_ewma_volatility_batch` (cpu_batch.rs:10307) reads
 *   `lambda` and NEVER `period`. Pinned at the CPU default lambda = 0.94, from
 *   which `period_from_lambda` (:215) gives
 *   round(2/(1-0.94) - 1) = round(32.3333...) = 32, and `alpha_from_period`
 *   (:225) gives 2/33.
 *
 * The seed is the MEAN of the first `period` VALID squared log returns, and a
 * return is valid only when both closes are finite AND strictly positive
 * (:274) — a zero previous close is skipped, not divided by. After the seed,
 * EVERY bar is written, valid return or not: an invalid bar simply carries the
 * previous ema forward (:308-312). Emitting NaN there would be a different
 * series.
 *
 * `beta.mul_add(ema, alpha * sq)` (:311) is ONE rounding — fma(beta, ema, ...).
 * Writing it as `beta*ema + alpha*sq` would be two and would drift over a
 * 32-period recursion.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* EWMA_SCALE, ewma_volatility.rs:212. */
#define NEO_EWMA_SCALE 100.0

extern "C" __global__
void ewma_volatility_neo_batch_f64(const double* __restrict__ data,
                                   int n,
                                   const int* __restrict__ periods,
                                   int n_combos,
                                   int first_valid,
                                   double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;
    (void)first_valid;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    /* period_from_lambda(0.94) -> 32; alpha_from_period(32) -> 2/33. */
    const double lambda = 0.94;
    double raw = round(2.0 / (1.0 - lambda) - 1.0);
    if (!(raw >= 1.0)) raw = 1.0;                 /* raw.max(1.0), :220 */
    const int    period = (int)raw;
    const double alpha  = 2.0 / ((double)period + 1.0);
    const double beta   = 1.0 - alpha;

    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    /* seed_ewma_single_pass (:284) */
    int    seed_idx = -1;
    int    valid    = 0;
    double sum      = 0.0;
    double ema      = 0.0;
    for (int i = 1; i < n; ++i) {
        const double prev = data[i - 1], curr = data[i];
        if (isfinite(prev) && isfinite(curr) && prev > 0.0 && curr > 0.0) {
            const double ret = log(curr / prev);
            sum += ret * ret;
            ++valid;
            if (valid == period) { seed_idx = i; ema = sum / (double)period; break; }
        }
    }
    if (seed_idx < 0) return;   /* NotEnoughValidData -> the CPU errors; the row
                                 * stays NaN rather than inventing a value. */

    /* fill_row_single_pass (:305) */
    o[seed_idx] = sqrt(fmax(ema, 0.0)) * NEO_EWMA_SCALE;
    for (int i = seed_idx + 1; i < n; ++i) {
        const double prev = data[i - 1], curr = data[i];
        if (isfinite(prev) && isfinite(curr) && prev > 0.0 && curr > 0.0) {
            const double ret = log(curr / prev);
            ema = fma(beta, ema, alpha * (ret * ret));
        }
        o[i] = sqrt(fmax(ema, 0.0)) * NEO_EWMA_SCALE;
    }
}
