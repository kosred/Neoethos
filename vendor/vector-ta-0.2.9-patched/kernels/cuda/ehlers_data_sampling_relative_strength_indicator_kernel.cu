#include <cmath>
#include <cstddef>

static __device__ inline double edsrsi_source_value(
    const double* open,
    const double* close,
    int idx,
    int midpoint_mode
) {
    if (midpoint_mode == 0) {
        return close[idx];
    }

    double open_value = open[idx];
    double close_value = close[idx];
    if (!isfinite(open_value) || !isfinite(close_value)) {
        return NAN;
    }
    return (open_value + close_value) * 0.5;
}

static __device__ inline int edsrsi_first_finite(
    const double* open,
    const double* close,
    int len,
    int midpoint_mode
) {
    for (int i = 0; i < len; ++i) {
        if (isfinite(edsrsi_source_value(open, close, i, midpoint_mode))) {
            return i;
        }
    }
    return -1;
}

static __device__ inline void edsrsi_compute_rsi_row(
    const double* open,
    const double* close,
    int len,
    int period,
    int midpoint_mode,
    double* out
) {
    int first = edsrsi_first_finite(open, close, len, midpoint_mode);
    if (first < 0 || period <= 0 || len - first < period) {
        return;
    }

    double inv_period = 1.0 / static_cast<double>(period);
    double beta = 1.0 - inv_period;
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    bool has_nan = false;

    int warm_last = first + period;
    if (warm_last >= len) {
        warm_last = len - 1;
    }

    double prev = edsrsi_source_value(open, close, first, midpoint_mode);
    for (int i = first + 1; i <= warm_last; ++i) {
        double curr = edsrsi_source_value(open, close, i, midpoint_mode);
        double delta = curr - prev;
        prev = curr;
        if (!isfinite(delta)) {
            has_nan = true;
            break;
        }
        if (delta > 0.0) {
            avg_gain += delta;
        } else if (delta < 0.0) {
            avg_loss -= delta;
        }
    }

    int idx0 = first + period;
    if (idx0 < len) {
        if (has_nan) {
            avg_gain = NAN;
            avg_loss = NAN;
            out[idx0] = NAN;
        } else {
            avg_gain *= inv_period;
            avg_loss *= inv_period;
            double denom = avg_gain + avg_loss;
            out[idx0] = denom == 0.0 ? 50.0 : (100.0 * avg_gain / denom);
        }
    }

    int j = idx0 + 1;
    while (j + 1 < len) {
        double curr1 = edsrsi_source_value(open, close, j, midpoint_mode);
        double prev1 = edsrsi_source_value(open, close, j - 1, midpoint_mode);
        double d1 = curr1 - prev1;
        double g1 = d1 > 0.0 ? d1 : 0.0;
        double l1 = d1 < 0.0 ? -d1 : 0.0;
        avg_gain = fma(avg_gain, beta, inv_period * g1);
        avg_loss = fma(avg_loss, beta, inv_period * l1);
        double denom1 = avg_gain + avg_loss;
        out[j] = denom1 == 0.0 ? 50.0 : (100.0 * avg_gain / denom1);

        double curr2 = edsrsi_source_value(open, close, j + 1, midpoint_mode);
        double prev2 = edsrsi_source_value(open, close, j, midpoint_mode);
        double d2 = curr2 - prev2;
        double g2 = d2 > 0.0 ? d2 : 0.0;
        double l2 = d2 < 0.0 ? -d2 : 0.0;
        avg_gain = fma(avg_gain, beta, inv_period * g2);
        avg_loss = fma(avg_loss, beta, inv_period * l2);
        double denom2 = avg_gain + avg_loss;
        out[j + 1] = denom2 == 0.0 ? 50.0 : (100.0 * avg_gain / denom2);

        j += 2;
    }

    if (j < len) {
        double curr = edsrsi_source_value(open, close, j, midpoint_mode);
        double prev_value = edsrsi_source_value(open, close, j - 1, midpoint_mode);
        double d = curr - prev_value;
        double g = d > 0.0 ? d : 0.0;
        double l = d < 0.0 ? -d : 0.0;
        avg_gain = fma(avg_gain, beta, inv_period * g);
        avg_loss = fma(avg_loss, beta, inv_period * l);
        double denom = avg_gain + avg_loss;
        out[j] = denom == 0.0 ? 50.0 : (100.0 * avg_gain / denom);
    }
}

static __device__ inline double edsrsi_classify_signal(double slo, double prev_slo_nz) {
    if (slo > 0.0) {
        return slo > prev_slo_nz ? 2.0 : 1.0;
    }
    if (slo < 0.0) {
        return slo < prev_slo_nz ? -2.0 : -1.0;
    }
    return 0.0;
}

static __device__ inline void edsrsi_fill_signal(
    const double* ds_rsi,
    int len,
    double* signal
) {
    double prev_slo = NAN;
    for (int i = 0; i < len; ++i) {
        double ds = ds_rsi[i];
        if (!isfinite(ds)) {
            prev_slo = NAN;
            continue;
        }
        double prev_ds_nz = (i > 0 && isfinite(ds_rsi[i - 1])) ? ds_rsi[i - 1] : 0.0;
        double slo = ds - prev_ds_nz;
        signal[i] = edsrsi_classify_signal(slo, isfinite(prev_slo) ? prev_slo : 0.0);
        prev_slo = slo;
    }
}

extern "C" __global__ void ehlers_data_sampling_relative_strength_indicator_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    int rows,
    double* __restrict__ out_ds_rsi,
    double* __restrict__ out_original_rsi,
    double* __restrict__ out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int period = lengths[row];
    double* row_ds_rsi = out_ds_rsi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_original_rsi =
        out_original_rsi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_ds_rsi[i] = NAN;
        row_original_rsi[i] = NAN;
        row_signal[i] = NAN;
    }

    if (period <= 0 || period > len) {
        return;
    }

    edsrsi_compute_rsi_row(open, close, len, period, 1, row_ds_rsi);
    edsrsi_compute_rsi_row(open, close, len, period, 0, row_original_rsi);
    edsrsi_fill_signal(row_ds_rsi, len, row_signal);
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference:
// src/indicators/ehlers_data_sampling_relative_strength_indicator.rs:481
// (ehlers_data_sampling_relative_strength_indicator_with_kernel).
//
// WHICH COLUMN, AND WHY IT IS NAMED HERE. The CPU batch
// accepts only ds_rsi / original_rsi / signal -- it has NO "value" alias and
// returns UnknownOutput for one. So
// a parity run must ask the CPU for ds_rsi explicitly; this kernel emits that
// column, which is the RSI of the bar MIDPOINT (open + close) / 2, and never
// the close-only original.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- the
// Wilder average is avg = avg*(1 - 1/period) + (1/period)*delta, carried bar to
// bar after a mean seed over the first period deltas. Reformulating it as a
// scan changes the rounding, and this value feeds a threshold comparison.
//
// THE TWO-AT-A-TIME LOOP IS THE CPU's, NOT AN OPTIMISATION. The reference
// steps its recurrence in pairs and then handles a trailing odd bar; the
// additions therefore associate in that order. Reproduced exactly, including
// the tail.
//
// LENGTH-SWEPT. The canonical registry names `length`; the primary ABI's
// historical `periods` array carries that exact length for each requested row.
//
// INPUT SHAPE: only OPEN and CLOSE are read. The lane row declares
// F64InputKind::Ohlc4 for the same reason Qstick does -- the resident upload
// already carries all four series and the four-pointer launch arm already
// exists, so high and low are accepted and ignored rather than forcing a new
// shape into shared launch code this change does not own.
//
// FIRST VALID IS DERIVED HERE: the CPU scans for the first bar whose MIDPOINT
// is finite (edsrsi_first_finite above / first_valid in the reference), which
// is a scan over a DERIVED series and not over any input series, so no
// F64FirstValidRule variant can express it. The lane row declares
// F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, no f32-suffixed math function, no fast-math
// intrinsic, and no epsilon at all -- the zero-denominator test is the CPU's
// exact == 0.0.
// ---------------------------------------------------------------------------

extern "C" __global__ void ehlers_data_sampling_relative_strength_indicator_neo_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row_idx >= n_combos || n <= 0) {
        return;
    }
    (void)high;
    (void)low;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int period = periods[row_idx];
    if (period <= 0 || period > n) {
        return;
    }

    // midpoint_mode = 1 is the (open + close) / 2 source, which is what the
    // ds_rsi column is built on.
    edsrsi_compute_rsi_row(open, close, n, period, 1, row);
}
