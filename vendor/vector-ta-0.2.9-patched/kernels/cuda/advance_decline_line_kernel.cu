#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void advance_decline_line_batch_f64(
    const double* __restrict__ data,
    int len,
    double* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || len <= 0) {
        return;
    }

    bool started = false;
    double sum = 0.0;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!isfinite(value)) {
            out[t] = CUDART_NAN;
            started = false;
            sum = 0.0;
            continue;
        }

        if (!started) {
            started = true;
            sum = value;
        } else {
            sum += value;
        }

        out[t] = sum;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — advance_decline_line
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/advance_decline_line.rs:227
 *             `advance_decline_line_row`.
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:8452 `expect_value_output`).
 *
 * PERIOD-INVARIANT AND PARAMETERLESS. `AdvanceDeclineLineParams` is a unit
 * struct and the batch closure takes `|_params, row|` (cpu_batch.rs:8462), so
 * a sweep of any period list yields identical columns.
 *
 * FIRST-VALID IGNORED. `advance_decline_line_row` walks from index 0 and
 * `first_valid_value` (:216) is used only to reject an all-NaN frame. A non
 * finite bar emits NaN and RESTARTS the sum from the next finite value —
 * `started = false; sum = 0.0` — so the running total does NOT bridge a hole.
 * That restart is the whole behaviour of the indicator on gapped data and is
 * reproduced literally.
 *
 * SEQUENTIAL, one thread per combo column: `sum` is a running total.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void advance_decline_line_neo_batch_f64(
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

    bool   started = false;
    double sum     = 0.0;
    for (int i = 0; i < len; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {
            o[i] = NEO_F64_NAN;
            started = false;
            sum = 0.0;
            continue;
        }
        if (!started) { started = true; sum = v; }
        else          { sum += v; }
        o[i] = sum;
    }
}
