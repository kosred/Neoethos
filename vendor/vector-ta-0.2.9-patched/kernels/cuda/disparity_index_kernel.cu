#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void disparity_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ ema_periods,
    const int* __restrict__ lookback_periods,
    const int* __restrict__ smoothing_periods,
    const int* __restrict__ smoothing_flags,
    int n_combos,
    int max_lookback,
    int max_smoothing,
    double* __restrict__ disparity_buffer,
    double* __restrict__ sma_buffer,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_lookback <= 0 || max_smoothing <= 0) {
        return;
    }

    int ema_period = ema_periods[combo_idx];
    int lookback_period = lookback_periods[combo_idx];
    int smoothing_period = smoothing_periods[combo_idx];
    int smoothing_flag = smoothing_flags[combo_idx];
    double* disparity_ring =
        disparity_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_lookback);
    double* sma_ring =
        sma_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_smoothing);
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (ema_period <= 0 ||
        lookback_period <= 0 ||
        smoothing_period <= 0 ||
        lookback_period > max_lookback ||
        smoothing_period > max_smoothing ||
        (smoothing_flag != 0 && smoothing_flag != 1)) {
        return;
    }

    double ema_alpha = 2.0 / (static_cast<double>(ema_period) + 1.0);
    double ema_beta = 1.0 - ema_alpha;
    double smoothing_alpha = 2.0 / (static_cast<double>(smoothing_period) + 1.0);
    double smoothing_beta = 1.0 - smoothing_alpha;

    int ema_seed_count = 0;
    double ema_seed_sum = 0.0;
    double ema = CUDART_NAN;
    bool ema_ready = false;

    int disparity_count = 0;
    int disparity_index = 0;

    int smoothing_seed_count = 0;
    double smoothing_seed_sum = 0.0;
    double smoothed = CUDART_NAN;
    bool smoothed_ready = false;

    int sma_count = 0;
    int sma_index = 0;
    double sma_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            ema_seed_count = 0;
            ema_seed_sum = 0.0;
            ema = CUDART_NAN;
            ema_ready = false;
            disparity_count = 0;
            disparity_index = 0;
            smoothing_seed_count = 0;
            smoothing_seed_sum = 0.0;
            smoothed = CUDART_NAN;
            smoothed_ready = false;
            sma_count = 0;
            sma_index = 0;
            sma_sum = 0.0;
            continue;
        }

        if (!ema_ready) {
            ema_seed_sum += value;
            ema_seed_count += 1;
            if (ema_seed_count < ema_period) {
                continue;
            }
            ema = ema_seed_sum / static_cast<double>(ema_period);
            ema_ready = true;
        } else {
            ema = ema_beta * ema + ema_alpha * value;
        }

        double disparity = CUDART_NAN;
        if (fabs(ema) <= 2.2204460492503131e-16) {
            if (fabs(value) <= 2.2204460492503131e-16) {
                disparity = 0.0;
            } else {
                continue;
            }
        } else {
            disparity = (value - ema) / ema * 100.0;
        }

        disparity_ring[disparity_index] = disparity;
        disparity_index += 1;
        if (disparity_index == lookback_period) {
            disparity_index = 0;
        }
        if (disparity_count < lookback_period) {
            disparity_count += 1;
        }
        if (disparity_count < lookback_period) {
            continue;
        }

        double high = -CUDART_INF;
        double low = CUDART_INF;
        for (int j = 0; j < lookback_period; ++j) {
            double window_value = disparity_ring[j];
            if (window_value > high) {
                high = window_value;
            }
            if (window_value < low) {
                low = window_value;
            }
        }

        double scaled = !(high > low) ? 50.0 : (disparity - low) / (high - low) * 100.0;

        if (smoothing_flag == 0) {
            if (!smoothed_ready) {
                smoothing_seed_sum += scaled;
                smoothing_seed_count += 1;
                if (smoothing_seed_count < smoothing_period) {
                    continue;
                }
                smoothed = smoothing_seed_sum / static_cast<double>(smoothing_period);
                smoothed_ready = true;
                row[i] = smoothed;
            } else {
                smoothed = smoothing_beta * smoothed + smoothing_alpha * scaled;
                row[i] = smoothed;
            }
        } else {
            if (sma_count < smoothing_period) {
                sma_ring[sma_count] = scaled;
                sma_sum += scaled;
                sma_count += 1;
                if (sma_count < smoothing_period) {
                    continue;
                }
                row[i] = sma_sum / static_cast<double>(smoothing_period);
            } else {
                double old = sma_ring[sma_index];
                sma_ring[sma_index] = scaled;
                sma_sum += scaled - old;
                sma_index += 1;
                if (sma_index == smoothing_period) {
                    sma_index = 0;
                }
                row[i] = sma_sum / static_cast<double>(smoothing_period);
            }
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - disparity_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/disparity_index.rs:564 `compute_row`, driving
 *             `DisparityIndexStream::update` (:449), `disparity_from_price`
 *             (:366), `scaled_from_disparity_window` (:472) and
 *             `smooth_scaled` (:490).
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:6909 `expect_value_output`).
 *
 * PERIOD-INVARIANT. The CPU batch reads `ema_period` (14), `lookback_period`
 * (14), `smoothing_period` (9) and `smoothing_type` ("ema") and never
 * `period`.
 *
 * SMOOTHING KIND is pinned to EMA because that is the batch default
 * (cpu_batch.rs:6924); the SMA arm of `smooth_scaled` (:510) is a different
 * parameterisation and is not what a sweep computes.
 *
 * FIRST-VALID IGNORED. `compute_row` fills NaN and walks from index 0 with a
 * fresh stream; the entry points do not pass a first index at all
 * (`alloc_with_nan_prefix(len, 0)` then `fill(NAN)`, :603). A non-finite bar
 * calls `reset` (:441), which clears BOTH ring buffers and BOTH seeds, so the
 * whole `ema + lookback + smoothing` warmup restarts after every hole.
 *
 * EPSILON: `ema.abs() <= f64::EPSILON` and `close.abs() <= f64::EPSILON`
 * (:370-371) - f64 machine epsilon, already correct for this lane and copied
 * as the exact literal rather than as a header macro.
 *
 * ROUNDING: `ema.mul_add(ema_beta, ema_alpha * value)` and the identical
 * shape for the smoothing EMA are ONE fused rounding each over a separately
 * rounded product. Written as `fma(...)` for that count.
 *
 * SEEDS: both EMAs seed with a PLAIN ascending sum divided by the period
 * (:546, :500) - not a chunked or pairwise tree. Reproduced literally.
 *
 * NaN SEMANTICS: the lookback high/low scan uses `f64::max` / `f64::min`
 * seeded with -inf / +inf (:476-481), which drop a NaN operand. `fmax`/`fmin`
 * match. The window is filled with NaN at construction, but the scan is only
 * reached once `disparity_count == lookback_period`, i.e. once every slot
 * holds a real disparity.
 *
 * `!(high > low)` (:482) is deliberately the NEGATION of a strict comparison,
 * so a NaN pair yields 50.0 rather than a division. Written the same way.
 *
 * SEQUENTIAL, one thread per combo column. Both rings are fixed-size
 * per-thread arrays at the CPU defaults, so no dynamic allocation.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define DISP_NEO_EMA_PERIOD   14
#define DISP_NEO_LOOKBACK     14
#define DISP_NEO_SMOOTHING     9

extern "C" __global__
void disparity_index_neo_batch_f64(
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

    const int EP = DISP_NEO_EMA_PERIOD;
    const int LB = DISP_NEO_LOOKBACK;
    const int SP = DISP_NEO_SMOOTHING;

    const double ema_alpha = 2.0 / ((double)EP + 1.0);
    const double ema_beta  = 1.0 - ema_alpha;
    const double sm_alpha  = 2.0 / ((double)SP + 1.0);
    const double sm_beta   = 1.0 - sm_alpha;
    const double DBL_EPS   = 2.2204460492503130808472633361816e-16;

    int    ema_seed_count = 0;
    double ema_seed_sum = 0.0;
    double ema = NEO_F64_NAN;
    bool   ema_ready = false;

    double disp_win[DISP_NEO_LOOKBACK];
    int    disp_count = 0, disp_idx = 0;
    #pragma unroll
    for (int k = 0; k < DISP_NEO_LOOKBACK; ++k) disp_win[k] = NEO_F64_NAN;

    int    sm_seed_count = 0;
    double sm_seed_sum = 0.0;
    double smoothed = NEO_F64_NAN;
    bool   smoothed_ready = false;

    for (int i = 0; i < len; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {
            ema_seed_count = 0; ema_seed_sum = 0.0;
            ema = NEO_F64_NAN; ema_ready = false;
            disp_count = 0; disp_idx = 0;
            #pragma unroll
            for (int k = 0; k < DISP_NEO_LOOKBACK; ++k) disp_win[k] = NEO_F64_NAN;
            sm_seed_count = 0; sm_seed_sum = 0.0;
            smoothed = NEO_F64_NAN; smoothed_ready = false;
            o[i] = NEO_F64_NAN;
            continue;
        }

        if (!ema_ready) {
            ema_seed_sum += v;
            ema_seed_count += 1;
            if (ema_seed_count < EP) { o[i] = NEO_F64_NAN; continue; }
            ema = ema_seed_sum / (double)EP;
            ema_ready = true;
        } else {
            ema = fma(ema, ema_beta, ema_alpha * v);
        }

        /* disparity_from_price (:366) - returns None, i.e. NaN out, on the
           non-finite and the "ema is zero but close is not" cases. */
        double disparity;
        bool   have_disparity;
        if (!isfinite(v) || !isfinite(ema)) {
            have_disparity = false; disparity = NEO_F64_NAN;
        } else if (fabs(ema) <= DBL_EPS) {
            if (fabs(v) <= DBL_EPS) { have_disparity = true; disparity = 0.0; }
            else                    { have_disparity = false; disparity = NEO_F64_NAN; }
        } else {
            have_disparity = true;
            disparity = (v - ema) / ema * 100.0;
        }
        if (!have_disparity) { o[i] = NEO_F64_NAN; continue; }

        disp_win[disp_idx] = disparity;
        disp_idx += 1; if (disp_idx == LB) disp_idx = 0;
        if (disp_count < LB) disp_count += 1;

        if (disp_count < LB) { o[i] = NEO_F64_NAN; continue; }

        double hi = -INFINITY, lo = INFINITY;
        for (int k = 0; k < LB; ++k) {
            hi = fmax(hi, disp_win[k]);
            lo = fmin(lo, disp_win[k]);
        }
        const double scaled = (!(hi > lo)) ? 50.0
                                           : (disparity - lo) / (hi - lo) * 100.0;

        if (!smoothed_ready) {
            sm_seed_sum += scaled;
            sm_seed_count += 1;
            if (sm_seed_count < SP) { o[i] = NEO_F64_NAN; continue; }
            smoothed = sm_seed_sum / (double)SP;
            smoothed_ready = true;
        } else {
            smoothed = fma(smoothed, sm_beta, sm_alpha * scaled);
        }
        o[i] = smoothed;
    }
}
