#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool bbpower_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close) && close != 0.0;
}

extern "C" __global__ void bull_power_vs_bear_power_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
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

    double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    double beta = 1.0 - alpha;
    int count = 0;
    double mean = CUDART_NAN;

    for (int i = 0; i < len; ++i) {
        if (!bbpower_valid_ohlc(open[i], high[i], low[i], close[i])) {
            count = 0;
            mean = CUDART_NAN;
            continue;
        }

        double value = ((high[i] + low[i]) - (2.0 * open[i])) * (100.0 / close[i]);
        count += 1;
        if (count == 1) {
            mean = value;
        } else if (count <= period) {
            double c = static_cast<double>(count);
            mean = ((c - 1.0) * mean + value) / c;
        } else {
            mean = beta * mean + alpha * value;
        }

        if (count >= period) {
            row[i] = mean;
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - bull_power_vs_bear_power
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/bull_power_vs_bear_power.rs:352
 *             `bbpower_row_from_ohlc`, value at :294.
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:8423 `expect_value_output`).
 *
 * PERIOD-SWEPT: `compute_bull_power_vs_bear_power_batch` reads `period`
 * (default 5), so `periods[combo]` is honoured and every row differs. One of
 * the few in this shard that is NOT period-invariant.
 *
 * FIRST-VALID IGNORED. The row walks from index 0 and re-derives validity per
 * bar; `bull_power_vs_bear_power_prepare`s `first` is not passed to it. An
 * invalid bar emits NaN and RESETS both `count` and `mean`, so the `period`
 * warmup restarts after every hole. Registered as
 * `F64FirstValidRule::Ignored` rather than an Ohlc4 rule the kernel would
 * then ignore.
 *
 * VALIDITY IS STRICTER THAN "ALL FOUR FINITE": `valid_ohlc_bar` (:289) also
 * requires `close != 0.0`, because the value divides by close. A zero close
 * therefore breaks the run exactly as a NaN does.
 *
 * SEED IS A RUNNING MEAN, NOT AN SMA WINDOW: for `count <= period` the CPU
 * uses `((c - 1) * mean + value) / c` (:378), an incremental average with a
 * different rounding from `sum / count`. Then it switches to the EMA
 * recurrence. Both forms are reproduced literally; substituting a windowed
 * mean for the seed would move every value in the first `period` bars and,
 * because the EMA is seeded from it, every bar after them too.
 *
 * ROUNDING: `beta.mul_add(mean, alpha * value)` is ONE fused rounding over a
 * separately rounded product - written as `fma(beta, mean, alpha * value)`.
 *
 * `100.0 / close` is computed as the CPU writes it (:295) - a reciprocal
 * scaled by 100 and THEN multiplied - not as `x * 100.0 / close`, which
 * associates differently.
 *
 * SEQUENTIAL, one thread per combo column: a running mean into an EMA.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void bull_power_vs_bear_power_neo_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int period = periods[combo];

    if (period <= 0) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const double alpha = 2.0 / ((double)period + 1.0);
    const double beta  = 1.0 - alpha;

    int    count = 0;
    double mean  = NEO_F64_NAN;

    for (int i = 0; i < len; ++i) {
        const double op = open[i], h = high[i], l = low[i], c = close[i];
        if (!(isfinite(op) && isfinite(h) && isfinite(l) && isfinite(c) && c != 0.0)) {
            o[i] = NEO_F64_NAN;
            count = 0;
            mean = NEO_F64_NAN;
            continue;
        }

        const double value = ((h + l) - (2.0 * op)) * (100.0 / c);
        count += 1;
        if (count == 1) {
            mean = value;
        } else if (count <= period) {
            const double cf = (double)count;
            mean = ((cf - 1.0) * mean + value) / cf;
        } else {
            mean = fma(beta, mean, alpha * value);
        }

        o[i] = (count >= period) ? mean : NEO_F64_NAN;
    }
}
