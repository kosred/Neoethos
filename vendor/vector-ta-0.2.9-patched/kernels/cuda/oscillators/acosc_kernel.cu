#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

/* ===========================================================================
 * NEOETHOS f64 LANE — Accelerator Oscillator
 * ---------------------------------------------------------------------------
 * Published formula:
 *   median = (high + low) / 2
 *   AO     = SMA5(median) - SMA34(median)
 *   AC     = AO - SMA5(AO)
 *   change = AC[t] - AC[t-1]
 *
 * The canonical scalar implementation owns rolling rings and therefore its
 * floating operation order is part of the result. One CUDA work item owns one
 * complete parameter row, walks bars in CPU order, and emits osc/change from
 * the same state. A non-finite high/low resets that state exactly; no host
 * formula, repair, fill, f32 conversion, fixed-window replay or fallback
 * participates.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

struct AcoscStateF64 {
    double median_fast[5];
    double median_slow[34];
    double ao_signal[5];
    double median_fast_sum;
    double median_slow_sum;
    double ao_signal_sum;
    int median_fast_index;
    int median_slow_index;
    int ao_signal_index;
    int median_count;
    int ao_count;
    double previous_ac;
    bool has_previous_ac;
};

static __device__ __forceinline__ void acosc_reset_f64(AcoscStateF64* state)
{
    for (int slot = 0; slot < 5; ++slot) {
        state->median_fast[slot] = 0.0;
        state->ao_signal[slot] = 0.0;
    }
    for (int slot = 0; slot < 34; ++slot) state->median_slow[slot] = 0.0;
    state->median_fast_sum = 0.0;
    state->median_slow_sum = 0.0;
    state->ao_signal_sum = 0.0;
    state->median_fast_index = 0;
    state->median_slow_index = 0;
    state->ao_signal_index = 0;
    state->median_count = 0;
    state->ao_count = 0;
    state->previous_ac = 0.0;
    state->has_previous_ac = false;
}

static __device__ __forceinline__ void acosc_row_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int series_len,
    double* __restrict__ out_osc,
    double* __restrict__ out_change)
{
    AcoscStateF64 state;
    acosc_reset_f64(&state);

    for (int bar = 0; bar < series_len; ++bar) {
        double ac = NEO_F64_NAN;
        double change = NEO_F64_NAN;
        const double high_value = high[bar];
        const double low_value = low[bar];

        if (!isfinite(high_value) || !isfinite(low_value)) {
            acosc_reset_f64(&state);
            if (out_osc != nullptr) out_osc[bar] = ac;
            if (out_change != nullptr) out_change[bar] = change;
            continue;
        }

        const double median = (high_value + low_value) * 0.5;
        if (!isfinite(median)) {
            acosc_reset_f64(&state);
            if (out_osc != nullptr) out_osc[bar] = ac;
            if (out_change != nullptr) out_change[bar] = change;
            continue;
        }

        if (state.median_count < 5) {
            state.median_fast[state.median_count] = median;
            state.median_fast_sum += median;
        } else {
            state.median_fast_sum += median - state.median_fast[state.median_fast_index];
            state.median_fast[state.median_fast_index] = median;
            state.median_fast_index = (state.median_fast_index + 1) % 5;
        }

        if (state.median_count < 34) {
            state.median_slow[state.median_count] = median;
            state.median_slow_sum += median;
            state.median_count += 1;
        } else {
            state.median_slow_sum += median - state.median_slow[state.median_slow_index];
            state.median_slow[state.median_slow_index] = median;
            state.median_slow_index = (state.median_slow_index + 1) % 34;
        }

        if (state.median_count < 34) {
            if (out_osc != nullptr) out_osc[bar] = ac;
            if (out_change != nullptr) out_change[bar] = change;
            continue;
        }

        const double ao = state.median_fast_sum / 5.0 - state.median_slow_sum / 34.0;
        if (!isfinite(ao)) {
            acosc_reset_f64(&state);
            if (out_osc != nullptr) out_osc[bar] = ac;
            if (out_change != nullptr) out_change[bar] = change;
            continue;
        }

        if (state.ao_count < 5) {
            state.ao_signal[state.ao_count] = ao;
            state.ao_signal_sum += ao;
            state.ao_count += 1;
            if (state.ao_count < 5) {
                if (out_osc != nullptr) out_osc[bar] = ac;
                if (out_change != nullptr) out_change[bar] = change;
                continue;
            }
        } else {
            state.ao_signal_sum += ao - state.ao_signal[state.ao_signal_index];
            state.ao_signal[state.ao_signal_index] = ao;
            state.ao_signal_index = (state.ao_signal_index + 1) % 5;
        }

        ac = ao - state.ao_signal_sum / 5.0;
        if (state.has_previous_ac) change = ac - state.previous_ac;
        state.previous_ac = ac;
        state.has_previous_ac = true;
        if (out_osc != nullptr) out_osc[bar] = ac;
        if (out_change != nullptr) out_change[bar] = change;
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void acosc_neo_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         int series_len,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;
    (void)first_valid;
    acosc_row_f64(high,
                  low,
                  series_len,
                  out + (size_t)combo * (size_t)series_len,
                  nullptr);
}

extern "C" __global__ __launch_bounds__(256, 4)
void acosc_outputs_f64(const double* __restrict__ high,
                       const double* __restrict__ low,
                       int series_len,
                       double* __restrict__ out_osc,
                       double* __restrict__ out_change)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row != 0) return;
    acosc_row_f64(high, low, series_len, out_osc, out_change);
}
