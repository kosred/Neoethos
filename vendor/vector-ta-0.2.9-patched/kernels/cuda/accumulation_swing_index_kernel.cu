#include <cmath>
#include <cstddef>

namespace {
__device__ double compute_increment(
    double prev_open,
    double prev_close,
    double open,
    double high,
    double low,
    double close,
    double daily_limit
) {
    const double abs_high_close = fabs(high - prev_close);
    const double abs_low_close = fabs(low - prev_close);
    const double abs_close_open = fabs(prev_close - prev_open);
    const double k = abs_high_close >= abs_low_close ? abs_high_close : abs_low_close;
    const double range = high - low;
    double r = 0.0;
    if (abs_high_close >= abs_low_close) {
        if (abs_high_close >= range) {
            r = abs_high_close - 0.5 * abs_low_close + 0.25 * abs_close_open;
        } else {
            r = range + 0.25 * abs_close_open;
        }
    } else if (abs_low_close >= range) {
        r = abs_low_close - 0.5 * abs_high_close + 0.25 * abs_close_open;
    } else {
        r = range + 0.25 * abs_close_open;
    }

    if (r != 0.0) {
        return 50.0 *
            (((close - prev_close) + 0.5 * (close - open) + 0.25 * (prev_close - prev_open)) / r) *
            k / daily_limit;
    }
    return 0.0;
}
}

extern "C" __global__ void accumulation_swing_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ daily_limits,
    int rows,
    double* __restrict__ out_values
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_out = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    const double daily_limit = daily_limits[row];
    if (!isfinite(daily_limit) || daily_limit <= 0.0) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (isfinite(open[i]) && isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    double accum = 0.0;
    row_out[first] = 0.0;
    double prev_open = open[first];
    double prev_close = close[first];

    for (int i = first + 1; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
            isfinite(prev_open) && isfinite(prev_close)) {
            const double delta = compute_increment(prev_open, prev_close, o, h, l, c, daily_limit);
            if (isfinite(delta)) {
                accum += delta;
            }
        }
        row_out[i] = accum;
        prev_open = o;
        prev_close = c;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — accumulation_swing_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/accumulation_swing_index.rs:326
 *             `compute_accumulation_swing_index_into`, increment at :286.
 *
 * SINGLE OUTPUT. `compute_accumulation_swing_index_batch` (cpu_batch.rs:10807)
 * calls `expect_value_output`, so "value" is the only column and there is no
 * choice to get wrong.
 *
 * PERIOD-INVARIANT. The only parameter is `daily_limit` (default 10_000.0);
 * `period` is never read.
 *
 * FIRST-VALID: `first_valid_ohlc` (:245) — the first index at which OPEN,
 * high, low and close are ALL `is_finite` SIMULTANEOUSLY. Not the Hlc rule:
 * open is an input here, and a frame whose open starts late would otherwise
 * seed `prev_open` from a NaN. Registered as
 * `F64FirstValidRule::Ohlc4AllFinite`.
 *
 * WARMUP: NaN strictly before `first`; `out[first] = 0.0` (:341); the running
 * sum starts at bar `first + 1`. The accumulator is NOT reset by a later hole
 * — the CPU keeps `accum`, writes it unchanged, and carries the (now non
 * finite) open/close forward as `prev_*`, which the finiteness test at :351
 * then rejects on the following bar. Reproduced exactly, including the fact
 * that a hole emits the carried `accum` rather than NaN.
 *
 * SEQUENTIAL, one thread per combo column: `accum` is a running sum whose
 * order is the CPU's.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

__device__ __forceinline__ double asi_neo_increment_f64(
    double prev_open, double prev_close,
    double open, double high, double low, double close,
    double daily_limit)
{
    const double abs_high_close = fabs(high - prev_close);
    const double abs_low_close  = fabs(low  - prev_close);
    const double abs_close_open = fabs(prev_close - prev_open);
    // `k` and `r` branch on the SAME `>=` the CPU uses (:298, :304); a tie
    // takes the high-close arm on both sides.
    const double k = (abs_high_close >= abs_low_close) ? abs_high_close
                                                       : abs_low_close;
    const double range = high - low;
    double r;
    if (abs_high_close >= abs_low_close) {
        r = (abs_high_close >= range)
                ? abs_high_close - 0.5 * abs_low_close + 0.25 * abs_close_open
                : range + 0.25 * abs_close_open;
    } else if (abs_low_close >= range) {
        r = abs_low_close - 0.5 * abs_high_close + 0.25 * abs_close_open;
    } else {
        r = range + 0.25 * abs_close_open;
    }

    if (r != 0.0) {
        return 50.0 * (((close - prev_close) + 0.5 * (close - open)
                        + 0.25 * (prev_close - prev_open)) / r)
               * k / daily_limit;
    }
    return 0.0;
}

extern "C" __global__
void accumulation_swing_index_neo_batch_f64(
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
    (void)periods;                       // period-invariant

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    const double daily_limit = 10000.0;  // accumulation_swing_index.rs default

    if (first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    for (int i = 0; i < first_valid; ++i) o[i] = NEO_F64_NAN;

    double accum = 0.0;
    o[first_valid] = 0.0;
    double prev_open  = open[first_valid];
    double prev_close = close[first_valid];

    for (int i = first_valid + 1; i < len; ++i) {
        const double oo = open[i];
        const double hh = high[i];
        const double ll = low[i];
        const double cc = close[i];
        if (isfinite(oo) && isfinite(hh) && isfinite(ll) && isfinite(cc)
            && isfinite(prev_open) && isfinite(prev_close)) {
            const double delta = asi_neo_increment_f64(prev_open, prev_close,
                                                       oo, hh, ll, cc,
                                                       daily_limit);
            if (isfinite(delta)) accum += delta;
        }
        o[i] = accum;
        prev_open  = oo;
        prev_close = cc;
    }
}
