#include <cmath>
#include <cstddef>

namespace {
struct RsiState {
    int period;
    double inv_p;
    double beta;
    bool initialized;
    double prev_price;
    int seed_count;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool seeded;

    __device__ void init(int value) {
        period = value;
        inv_p = 1.0 / static_cast<double>(value);
        beta = 1.0 - inv_p;
        reset();
    }

    __device__ void reset() {
        initialized = false;
        prev_price = NAN;
        seed_count = 0;
        sum_gain = 0.0;
        sum_loss = 0.0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        seeded = false;
    }

    __device__ double update(double value) {
        if (!initialized) {
            prev_price = value;
            initialized = true;
            return NAN;
        }

        const double delta = value - prev_price;
        prev_price = value;
        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);

        if (!seeded) {
            sum_gain += gain;
            sum_loss += loss;
            seed_count += 1;
            if (seed_count < period) {
                return NAN;
            }
            avg_gain = sum_gain * inv_p;
            avg_loss = sum_loss * inv_p;
            seeded = true;
        } else {
            avg_gain = fma(avg_gain, beta, inv_p * gain);
            avg_loss = fma(avg_loss, beta, inv_p * loss);
        }

        const double denom = avg_gain + avg_loss;
        if (denom == 0.0) {
            return 50.0;
        }
        return 100.0 * avg_gain / denom;
    }
};

__device__ inline bool pine_cross(double prev_a, double prev_b, double curr_a, double curr_b) {
    return (prev_a <= prev_b && curr_a > curr_b) || (prev_a >= prev_b && curr_a < curr_b);
}

__device__ inline bool pine_crossover(double prev_a, double prev_b, double curr_a, double curr_b) {
    return prev_a <= prev_b && curr_a > curr_b;
}

__device__ inline bool pine_crossunder(
    double prev_a,
    double prev_b,
    double curr_a,
    double curr_b
) {
    return prev_a >= prev_b && curr_a < curr_b;
}
}

extern "C" __global__ void adaptive_bounds_rsi_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ rsi_lengths,
    const double* __restrict__ alphas,
    int rows,
    double* __restrict__ out_rsi,
    double* __restrict__ out_lower_bound,
    double* __restrict__ out_lower_mid,
    double* __restrict__ out_mid,
    double* __restrict__ out_upper_mid,
    double* __restrict__ out_upper_bound,
    double* __restrict__ out_regime,
    double* __restrict__ out_regime_flip,
    double* __restrict__ out_lower_signal,
    double* __restrict__ out_upper_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int rsi_length = rsi_lengths[row];
    const double alpha = alphas[row];

    double* row_rsi = out_rsi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_bound =
        out_lower_bound + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_mid =
        out_lower_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mid = out_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_mid =
        out_upper_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_bound =
        out_upper_bound + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_regime = out_regime + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_regime_flip =
        out_regime_flip + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_signal =
        out_lower_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_signal =
        out_upper_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_rsi[i] = NAN;
        row_lower_bound[i] = NAN;
        row_lower_mid[i] = NAN;
        row_mid[i] = NAN;
        row_upper_mid[i] = NAN;
        row_upper_bound[i] = NAN;
        row_regime[i] = NAN;
        row_regime_flip[i] = NAN;
        row_lower_signal[i] = NAN;
        row_upper_signal[i] = NAN;
    }

    if (rsi_length <= 0 || !isfinite(alpha) || alpha < 0.001 || alpha > 1.0) {
        return;
    }

    RsiState rsi_state;
    rsi_state.init(rsi_length);

    double c1 = 20.0;
    double c2 = 40.0;
    double c3 = 50.0;
    double c4 = 60.0;
    double c5 = 80.0;

    bool has_prev_rsi = false;
    bool has_prev_c1 = false;
    bool has_prev_c5 = false;
    bool has_prev_regime = false;
    double prev_rsi = NAN;
    double prev_c1 = NAN;
    double prev_c5 = NAN;
    int prev_regime = 0;
    bool can_show_lower = true;
    bool can_show_upper = true;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            rsi_state.reset();
            has_prev_rsi = false;
            has_prev_c1 = false;
            has_prev_c5 = false;
            has_prev_regime = false;
            continue;
        }

        const double rsi = rsi_state.update(value);
        if (!isfinite(rsi)) {
            continue;
        }

        const double d1 = fabs(rsi - c1);
        const double d2 = fabs(rsi - c2);
        const double d3 = fabs(rsi - c3);
        const double d4 = fabs(rsi - c4);
        const double d5 = fabs(rsi - c5);
        const double min_dist = fmin(d1, fmin(d2, fmin(d3, fmin(d4, d5))));
        if (min_dist == d1) {
            c1 += (rsi - c1) * alpha;
        } else if (min_dist == d2) {
            c2 += (rsi - c2) * alpha;
        } else if (min_dist == d3) {
            c3 += (rsi - c3) * alpha;
        } else if (min_dist == d4) {
            c4 += (rsi - c4) * alpha;
        } else {
            c5 += (rsi - c5) * alpha;
        }

        int regime = 0;
        if (rsi <= c1) {
            regime = -2;
        } else if (rsi <= c2) {
            regime = -1;
        } else if (rsi <= c3) {
            regime = 0;
        } else if (rsi <= c4) {
            regime = 1;
        } else {
            regime = 2;
        }

        const bool crossed_mid = has_prev_rsi && pine_cross(prev_rsi, 50.0, rsi, 50.0);
        if (crossed_mid) {
            can_show_lower = true;
            can_show_upper = true;
        }

        const bool lower_signal = has_prev_rsi && has_prev_c1 &&
            pine_crossunder(prev_rsi, prev_c1, rsi, c1) && can_show_lower;
        const bool upper_signal = has_prev_rsi && has_prev_c5 &&
            pine_crossover(prev_rsi, prev_c5, rsi, c5) && can_show_upper;

        if (lower_signal) {
            can_show_lower = false;
        }
        if (upper_signal) {
            can_show_upper = false;
        }

        const bool regime_flip = has_prev_regime && prev_regime == 0 && regime != 0;

        row_rsi[i] = rsi;
        row_lower_bound[i] = c1;
        row_lower_mid[i] = c2;
        row_mid[i] = c3;
        row_upper_mid[i] = c4;
        row_upper_bound[i] = c5;
        row_regime[i] = static_cast<double>(regime);
        row_regime_flip[i] = regime_flip ? 1.0 : 0.0;
        row_lower_signal[i] = lower_signal ? 1.0 : 0.0;
        row_upper_signal[i] = upper_signal ? 1.0 : 0.0;

        prev_rsi = rsi;
        prev_c1 = c1;
        prev_c5 = c5;
        prev_regime = regime;
        has_prev_rsi = true;
        has_prev_c1 = true;
        has_prev_c5 = true;
        has_prev_regime = true;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE - adaptive_bounds_rsi
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/adaptive_bounds_rsi.rs:670
 *             `adaptive_bounds_rsi_output_into_slice`, driving
 *             `RsiState::update` (:301) and `rsi_value` (:331).
 *
 * COLUMN: `rsi`. `adaptive_bounds_rsi_field` (cpu_batch.rs:12552) maps
 * output_id "value" onto `OutputField::Rsi`, so the value column IS the RSI
 * series. The five adaptive bounds c1..c5, the regime and the four signal
 * flags are OTHER columns and do not enter this one at all - which is why
 * `alpha` never appears below. Emitting a bound here instead would pass every
 * length and shape check and be a different indicator.
 *
 * PERIOD-INVARIANT. The CPU batch reads `rsi_length` (14) and `alpha` (0.1)
 * and never `period`.
 *
 * FIRST-VALID IGNORED. The output row builds a fresh stream and walks from
 * index 0, filling NaN first (:679). A non-finite bar RESETS the RSI state
 * (:579) - `prev_price`, the seed sums, the seed count and both Wilder
 * averages - so the 14-bar seed restarts after every hole rather than
 * bridging it.
 *
 * ROUNDING: `avg_gain.mul_add(beta, inv_p * gain)` is ONE fused rounding on
 * the outer multiply-add over a separately rounded `inv_p * gain`. Written as
 * `fma(avg_gain, beta, inv_p * gain)` for exactly that count - two roundings,
 * not three. `-fmad=false` in build.rs forbids the COMPILER contracting a
 * separate `*` and `+`, and does not touch this explicit `fma`.
 *
 * SEED: `avg = sum * inv_p`, where `sum` was accumulated one bar at a time
 * from 0.0 in ascending order (:312). NOT a chunked or pairwise sum - the
 * association is the CPU loop and is reproduced literally.
 *
 * NaN SEMANTICS: `delta.max(0.0)` and `(-delta).max(0.0)` are `f64::max`,
 * which returns the NON-NaN operand. `fmax` matches; `delta > 0 ? delta : 0`
 * would return 0.0 for a NaN delta on one side and NaN on the other,
 * silently changing the seed.
 *
 * SEQUENTIAL, one thread per combo column: a Wilder recurrence.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void adaptive_bounds_rsi_neo_batch_f64(
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

    const int    period = 14;                    /* adaptive_bounds_rsi.rs default */
    const double inv_p  = 1.0 / (double)period;  /* RsiState::new, :274 */
    const double beta   = 1.0 - inv_p;

    bool   have_prev = false;
    double prev_price = 0.0;
    int    seed_count = 0;
    double sum_gain = 0.0, sum_loss = 0.0;
    double avg_gain = 0.0, avg_loss = 0.0;
    bool   seeded = false;

    for (int i = 0; i < len; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {
            have_prev = false; prev_price = 0.0;
            seed_count = 0;
            sum_gain = 0.0; sum_loss = 0.0;
            avg_gain = 0.0; avg_loss = 0.0;
            seeded = false;
            o[i] = NEO_F64_NAN;
            continue;
        }
        if (!have_prev) {
            have_prev = true;
            prev_price = v;
            o[i] = NEO_F64_NAN;   /* RsiState::update returns None, :302 */
            continue;
        }

        const double delta = v - prev_price;
        prev_price = v;
        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);

        double rsi;
        bool   emit;
        if (!seeded) {
            sum_gain += gain;
            sum_loss += loss;
            seed_count += 1;
            if (seed_count == period) {
                seeded = true;
                avg_gain = sum_gain * inv_p;
                avg_loss = sum_loss * inv_p;
                emit = true;
            } else {
                emit = false;
            }
        } else {
            avg_gain = fma(avg_gain, beta, inv_p * gain);
            avg_loss = fma(avg_loss, beta, inv_p * loss);
            emit = true;
        }

        if (emit) {
            const double denom = avg_gain + avg_loss;
            rsi = (denom == 0.0) ? 50.0 : 100.0 * avg_gain / denom;
            o[i] = rsi;
        } else {
            o[i] = NEO_F64_NAN;
        }
    }
}
