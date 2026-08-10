#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool obvo_valid_bar(double source, double volume) {
    return isfinite(source) && isfinite(volume);
}

extern "C" __global__ void on_balance_volume_oscillator_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ obv_lengths,
    const int* __restrict__ ema_lengths,
    int n_combos,
    double* __restrict__ out_line,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int obv_length = obv_lengths[combo_idx];
    int ema_length = ema_lengths[combo_idx];
    double* row_line = out_line + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_line[t] = CUDART_NAN;
        row_signal[t] = CUDART_NAN;
    }

    if (obv_length <= 0 || ema_length <= 0) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);
    bool ema_initialized = false;
    double ema_value = CUDART_NAN;
    int run_start = 0;

    for (int t = 0; t < len; ++t) {
        double src = source[t];
        double vol = volume[t];
        if (!obvo_valid_bar(src, vol)) {
            run_start = t + 1;
            ema_initialized = false;
            ema_value = CUDART_NAN;
            continue;
        }

        int run_len = t - run_start + 1;
        if (run_len < obv_length) {
            continue;
        }

        double signed_sum = 0.0;
        double volume_sum = 0.0;
        int count = 0;

        for (int j = t; j >= run_start && count < obv_length; --j) {
            double signed_volume;
            if (j == run_start) {
                signed_volume = 0.0;
            } else {
                double prev_source = source[j - 1];
                double sign =
                    source[j] > prev_source ? 1.0 : (source[j] < prev_source ? -1.0 : 0.0);
                signed_volume = volume[j] * sign;
            }
            signed_sum += signed_volume;
            volume_sum += volume[j];
            count += 1;
        }

        if (count < obv_length) {
            continue;
        }

        double line = volume_sum == 0.0 ? CUDART_NAN : signed_sum / volume_sum;
        row_line[t] = line;

        if (isfinite(line)) {
            if (ema_initialized) {
                ema_value += alpha * (line - ema_value);
            } else {
                ema_value = line;
                ema_initialized = true;
            }
            row_signal[t] = ema_value;
        }
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// CPU REFERENCE
// -------------
//   src/indicators/on_balance_volume_oscillator.rs
//     :269 is_valid_bar   -- source AND volume both `is_finite`
//     :482 prepare        -- `obv_length > data_len` is an Err (all-NaN row)
//     :523 on_balance_volume_oscillator_compute_default_20_9_into
//          <- the whole specification; `compute_into` (:623) routes to it
//             whenever obv_length == 20 && ema_length == 9, which is exactly
//             the defaults the lane's caller gets
//     :667 with_kernel -- `alloc_with_nan_prefix(len, 0)`, i.e. NO warmup
//          prefix; every index is written by the walk below
//   dispatch: cpu_batch.rs:9736 -- `extract_close_volume_input(.., "close")`,
//   params `obv_length` (20) and `ema_length` (9), never `period`. `output_id`
//   "value" resolves to `out.line` (:9768).
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW walking bars ascending. Two rolling sums with an
// add-on-entry / subtract-on-exit update, a carried `prev_source`, and an EMA
// seeded by the first finite `line` -- and a non-finite bar RESETS all of it,
// so bar i depends on every bar before it.
//
// PERIOD-INVARIANT and FIRST-VALID IGNORED, both for the reasons above.
//
// MULTI-OUTPUT: this emits `line`, the column the CPU batch produces for
// `output_id == "value"`. `signal` is not emitted; a caller that wants it asks
// for it by name through the CPU path.
//
// ARITHMETIC
// ----------
// f64 end to end, no fast-math. Two details that are easy to get wrong and are
// therefore spelled out:
//   * `sign` is `(value > prev) - (value < prev)` computed in the INTEGER
//     domain and then converted (:562). Writing it as a chain of f64 compares
//     would give the same number here, but the integer form is what the CPU
//     runs and it makes the equal case (0.0, not +0.0-vs--0.0) unambiguous.
//   * the EMA alpha is the literal `0.2` the CPU writes (:600), not
//     `2.0 / (ema_length + 1.0)` recomputed -- those are the same real number
//     for ema_length == 9 but not necessarily the same f64.
// `volume_sum == 0.0` is the CPU's exact guard and is not softened into a
// tolerance.

#define OBVOSC_NEO_OBV_LENGTH 20

__device__ __forceinline__ double obvosc_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void on_balance_volume_oscillator_neo_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = obvosc_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) o[i] = nan_d;

    (void)periods;       // PERIOD-INVARIANT
    (void)first_valid;   // FIRST-VALID IGNORED

    if (n <= 0) return;
    if (OBVOSC_NEO_OBV_LENGTH > n) return;      // :498 InvalidObvLength

    double prev_source = nan_d;
    bool has_prev_source = false;
    double signed_buffer[OBVOSC_NEO_OBV_LENGTH];
    double volume_buffer[OBVOSC_NEO_OBV_LENGTH];
    int count = 0;
    int head = 0;
    double signed_sum = 0.0;
    double volume_sum = 0.0;
    double ema_value = nan_d;
    bool ema_initialized = false;

    for (int i = 0; i < n; ++i) {
        const double value = source[i];
        const double vol = volume[i];
        if (!(isfinite(value) && isfinite(vol))) {          // :269 is_valid_bar
            prev_source = nan_d;
            has_prev_source = false;
            count = 0;
            head = 0;
            signed_sum = 0.0;
            volume_sum = 0.0;
            ema_value = nan_d;
            ema_initialized = false;
            o[i] = nan_d;
            continue;
        }

        double signed_volume;
        if (has_prev_source) {
            const int sign_i = (value > prev_source ? 1 : 0) - (value < prev_source ? 1 : 0);
            prev_source = value;
            signed_volume = vol * static_cast<double>(sign_i);
        } else {
            prev_source = value;
            has_prev_source = true;
            signed_volume = 0.0;
        }

        bool ready;
        if (count < OBVOSC_NEO_OBV_LENGTH) {
            signed_buffer[count] = signed_volume;
            volume_buffer[count] = vol;
            signed_sum += signed_volume;
            volume_sum += vol;
            count += 1;
            ready = (count == OBVOSC_NEO_OBV_LENGTH);
        } else {
            const double old_signed = signed_buffer[head];
            const double old_volume = volume_buffer[head];
            signed_buffer[head] = signed_volume;
            volume_buffer[head] = vol;
            signed_sum += signed_volume - old_signed;
            volume_sum += vol - old_volume;
            head += 1;
            if (head == OBVOSC_NEO_OBV_LENGTH) head = 0;
            ready = true;
        }

        if (ready) {
            const double line = (volume_sum == 0.0) ? nan_d : (signed_sum / volume_sum);
            if (isfinite(line)) {
                if (ema_initialized) {
                    ema_value += 0.2 * (line - ema_value);
                } else {
                    ema_value = line;
                    ema_initialized = true;
                }
            }
            o[i] = line;
        } else {
            o[i] = nan_d;
        }
    }
}
