#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
constexpr int SMA_LENGTH = 5;
constexpr double EMA_ALPHA = 2.0 / 5.0;
constexpr double MULTIPLIER = 1000.0;
constexpr double EPSILON = 1e-12;

__device__ inline bool valid_breadth_pair(double advancing, double declining) {
    if (!isfinite(advancing) || !isfinite(declining)) {
        return false;
    }
    double total = advancing + declining;
    return isfinite(total) && fabs(total) > EPSILON;
}
}

extern "C" __global__ void decisionpoint_breadth_swenlin_trading_oscillator_batch_f64(
    const double* __restrict__ advancing,
    const double* __restrict__ declining,
    int len,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    bool ema_started = false;
    double ema_value = CUDART_NAN;
    double sma_values[SMA_LENGTH] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int sma_idx = 0;
    int sma_count = 0;
    double sma_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double adv = advancing[i];
        double dec = declining[i];
        if (!valid_breadth_pair(adv, dec)) {
            row[i] = CUDART_NAN;
            ema_started = false;
            ema_value = CUDART_NAN;
            sma_idx = 0;
            sma_count = 0;
            sma_sum = 0.0;
            for (int j = 0; j < SMA_LENGTH; ++j) {
                sma_values[j] = 0.0;
            }
            continue;
        }

        double breadth = ((adv - dec) / (adv + dec)) * MULTIPLIER;
        if (!ema_started) {
            ema_started = true;
            ema_value = breadth;
        } else {
            ema_value += EMA_ALPHA * (breadth - ema_value);
        }

        if (sma_count == SMA_LENGTH) {
            sma_sum -= sma_values[sma_idx];
        } else {
            sma_count += 1;
        }
        sma_values[sma_idx] = ema_value;
        sma_sum += ema_value;
        sma_idx += 1;
        if (sma_idx == SMA_LENGTH) {
            sma_idx = 0;
        }

        row[i] = sma_count < SMA_LENGTH ? CUDART_NAN : (sma_sum / static_cast<double>(SMA_LENGTH));
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — decisionpoint_breadth_swenlin_trading_oscillator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/decisionpoint_breadth_swenlin_trading_oscillator
 *             .rs:329 `..._row`, driving `Stream::update` (:194).
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:9264 `expect_value_output`).
 *
 * PERIOD-INVARIANT AND PARAMETERLESS: the params struct is a unit struct and
 * the batch closure takes `|_params, row|` (cpu_batch.rs:9276). EMA_LENGTH=4,
 * SMA_LENGTH=5 and MULTIPLIER=1000.0 are compile-time constants (:57-60).
 *
 * INPUT is (advancing, declining) — carried by the lane's HighLow shape,
 * which is two `const double*` in declaration order. Which two series they
 * are is settled by `F64InputKind::HighLow` upstream, not here.
 *
 * FIRST-VALID IGNORED. The row builds a fresh stream and walks from index 0;
 * `prepare`'s `first` is used only to reject an all-invalid frame. An invalid
 * PAIR resets the whole stream (EMA seed, SMA ring and count), so the 5-bar
 * SMA warmup restarts after every hole. Registered as
 * `F64FirstValidRule::Ignored`.
 *
 * EPSILON: the CPU's validity test is `total.abs() > EPSILON` with
 * EPSILON = 1e-12 (:62) — already an f64-sized guard on a ratio denominator,
 * NOT an f32 machine epsilon, so it is carried across unchanged. Deriving a
 * different one here would accept or reject bars the CPU does not.
 *
 * SEQUENTIAL, one thread per combo column: an EMA recurrence feeding a
 * 5-slot circular SMA whose running sum is order-dependent. The ring is five
 * doubles in registers, so there is no dynamic allocation and no period bound
 * to refuse.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void decisionpoint_breadth_swenlin_trading_oscillator_neo_batch_f64(
    const double* __restrict__ advancing,
    const double* __restrict__ declining,
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

    const int    SMA_LENGTH = 5;
    const double MULTIPLIER = 1000.0;
    const double EMA_ALPHA  = 2.0 / (4.0 + 1.0);   // EMA_LENGTH = 4
    const double EPS        = 1e-12;

    double sma_values[5];
    int    sma_idx = 0, sma_count = 0;
    double sma_sum = 0.0;
    bool   ema_started = false;
    double ema_value = 0.0;
    #pragma unroll
    for (int k = 0; k < 5; ++k) sma_values[k] = 0.0;

    for (int i = 0; i < len; ++i) {
        const double a = advancing[i];
        const double d = declining[i];
        const double total = a + d;
        const bool valid = isfinite(a) && isfinite(d)
                           && isfinite(total) && fabs(total) > EPS;
        if (!valid) {
            // `Stream::reset` (:181) — the ring, the count and the EMA seed
            // all go back to construction, not just the EMA value.
            sma_idx = 0; sma_count = 0; sma_sum = 0.0;
            ema_started = false; ema_value = 0.0;
            #pragma unroll
            for (int k = 0; k < 5; ++k) sma_values[k] = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double breadth = ((a - d) / total) * MULTIPLIER;
        if (!ema_started) { ema_started = true; ema_value = breadth; }
        else              { ema_value += EMA_ALPHA * (breadth - ema_value); }

        if (sma_count == SMA_LENGTH) sma_sum -= sma_values[sma_idx];
        else                         sma_count += 1;
        sma_values[sma_idx] = ema_value;
        sma_sum += ema_value;
        sma_idx += 1;
        if (sma_idx == SMA_LENGTH) sma_idx = 0;

        o[i] = (sma_count < SMA_LENGTH)
                   ? NEO_F64_NAN
                   : sma_sum / (double)SMA_LENGTH;
    }
}
