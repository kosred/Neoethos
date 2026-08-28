#include <cmath>
#include <cstdint>

extern "C" __global__ void normalized_resonator_batch_f64(
    const double* data,
    int len,
    const int* periods,
    const double* deltas,
    const double* lookback_mults,
    const int* signal_lengths,
    int rows,
    double* out_oscillator,
    double* out_signal,
    double* bp_history
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int period = periods[row];
    double delta = deltas[row];
    double lookback_mult = lookback_mults[row];
    int signal_length = signal_lengths[row];
    if (period < 2 || !isfinite(delta) || delta <= 0.0 || delta > 1.0 || !isfinite(lookback_mult)
        || lookback_mult <= 0.0 || signal_length <= 0) {
        return;
    }

    const double nan = NAN;
    const double pi = 3.14159265358979323846;

    double alpha = tan(pi * delta / static_cast<double>(period));
    if (!isfinite(alpha)) {
        return;
    }
    double beta = cos(2.0 * pi / static_cast<double>(period));
    double r = 1.0 / (1.0 + alpha);
    double c1 = 2.0 * r * beta;
    double c2 = -(2.0 * r - 1.0);
    double gain = alpha * r;
    double peak_lookback_raw = floor(static_cast<double>(period) * lookback_mult);
    int peak_lookback = static_cast<int>(peak_lookback_raw < 1.0 ? 1.0 : peak_lookback_raw);
    double ema_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bp = bp_history + static_cast<size_t>(row) * static_cast<size_t>(len);

    double src_prev1 = 0.0;
    double src_prev2 = 0.0;
    int src_count = 0;
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    double ema_value = 0.0;
    bool ema_seeded = false;
    int run_start = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        row_oscillator[i] = nan;
        row_signal[i] = nan;
        row_bp[i] = nan;

        if (!isfinite(value)) {
            src_prev1 = 0.0;
            src_prev2 = 0.0;
            src_count = 0;
            bp_prev1 = 0.0;
            bp_prev2 = 0.0;
            ema_value = 0.0;
            ema_seeded = false;
            run_start = i + 1;
            continue;
        }

        if (src_count >= 2) {
            double bp = gain * (value - src_prev2) + c1 * bp_prev1 + c2 * bp_prev2;
            row_bp[i] = bp;

            int peak_start = i - peak_lookback + 1;
            if (peak_start < run_start) {
                peak_start = run_start;
            }
            double peak = 0.0;
            for (int j = peak_start; j <= i; ++j) {
                double hist = row_bp[j];
                if (isfinite(hist)) {
                    double abs_hist = fabs(hist);
                    if (abs_hist > peak) {
                        peak = abs_hist;
                    }
                }
            }

            double oscillator = peak > 0.0 ? bp / peak : 0.0;
            double signal = oscillator;
            if (ema_seeded) {
                ema_value += ema_alpha * (oscillator - ema_value);
                signal = ema_value;
            } else {
                ema_value = oscillator;
                ema_seeded = true;
            }

            bp_prev2 = bp_prev1;
            bp_prev1 = bp;
            row_oscillator[i] = oscillator;
            row_signal[i] = signal;
        }

        if (src_count == 0) {
            src_prev1 = value;
            src_count = 1;
        } else if (src_count == 1) {
            src_prev2 = src_prev1;
            src_prev1 = value;
            src_count = 2;
        } else {
            src_prev2 = src_prev1;
            src_prev1 = value;
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// normalized_resonator_batch_f64 above is genuine double-in/double-out, but it
// takes 10 parameters, writes TWO output matrices and demands a caller-supplied
// bp_history matrix the width of the whole series. The f64 lane launches one
// shape -- (prices, n, periods, n_combos, first_valid, out) -- and allocates
// ONE output matrix, so that entry point cannot be reached from it.
//
// CPU REFERENCE
//   src/indicators/normalized_resonator.rs:731
//     normalized_resonator_with_kernel -> :622 normalized_resonator_row_from_slice
//     -> :653 normalized_resonator_default_row (the path every lane row takes:
//     :627-632 selects it when period, delta, lookback_mult, signal_length and
//     peak_lookback all equal their defaults, which the pins below satisfy).
//   resolve_params  :316   (alpha/beta/r/c1/c2/gain/ema_alpha, term for term)
//   RollingAbsMax100::update :450
//
// THE COLUMN THIS EMITS is oscillator. This indicator's CPU batch has NO
// "value" output at all -- compute_normalized_resonator_batch accepts only
// "oscillator" and "signal" (cpu_batch.rs) -- so the lane declares the primary
// series, exactly as it already does for range_oscillator.
//
// SOURCE IS hl2, NOT close. DEFAULT_SOURCE = "hl2" (:37) and the batch's
// get_enum_param default is "hl2". Handing this kernel close computes a
// different indicator and passes every length check on the way through, which
// is why the lane row declares F64InputKind::Hl2Slice.
//
// PERIOD-INVARIANT. The batch reads source, period, delta, lookback_mult and
// signal_length; the swept `period` name is not among the parameters the lane
// varies for this id, and all five are pinned to the CPU defaults below, so
// five swept rows are five identical CPU columns and five identical kernel
// rows.
//
// SHAPE: one thread per combo, bars ascending. A 2-POLE RESONATOR IIR -- bp[i]
// reads bp[i-1] and bp[i-2] (:687-688) -- plus a carried EMA and a carried
// 100-deep rolling absolute maximum. NOT reformulated as a matrix-power warp
// scan: that changes the accumulation order, and this series feeds a threshold
// comparison where one ULP flips a trade.
//
// THE PEAK WINDOW IS THE CPU'S OWN MONOTONIC DEQUE, not a linear rescan. The
// kernel above rescans up to peak_lookback bars of bp history for each bar,
// which is the same MAXIMUM (max is order-independent) but needs the whole
// history resident. RollingAbsMax100 (:422-483) is a 101-slot monotonic deque
// keyed by a push counter, and that is what is transliterated here -- so the
// per-thread state is bounded and the window is "the last 100 PUSHES", which is
// what the CPU means.
//
// FIRST VALID IS NOT READ: normalized_resonator_default_row writes every index
// of oscillator_out -- NaN for the first two bars of every run and after every
// gap -- so the alloc_with_nan_prefix warmup (:743) is overwritten wholesale.
// The lane row declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double tan/cos/fabs, no f32-suffixed math
// function, no fast-math intrinsic. The CPU has no epsilon in this path -- the
// peak guard is a literal peak > 0.0 (:691) -- so none is invented here. The
// file is listed in F64_LANE_SOURCES, so it is never built with
// --use_fast_math; that matters because alpha comes from tan() and the gain it
// produces multiplies every later bar of the recursion.
// ---------------------------------------------------------------------------

#define NRES_NEO_PERIOD 100
#define NRES_NEO_DELTA 0.5
#define NRES_NEO_LOOKBACK_MULT 1.0
#define NRES_NEO_SIGNAL_LENGTH 9
// RollingAbsMax100 is [_; 101] -- one more slot than the window, so a full
// deque can hold the incoming push before the stale front is dropped (:423).
#define NRES_NEO_DEQUE_CAP 101

__device__ __forceinline__ double nres_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void normalized_resonator_neo_batch_f64(
    const double* __restrict__ prices,
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
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    const double qnan = nres_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    // resolve_params (:339-353). std::f64::consts::PI is the double nearest pi;
    // the literal below rounds to that same double.
    const double pi = 3.14159265358979323846;
    const double period_f = static_cast<double>(NRES_NEO_PERIOD);
    const double alpha = tan(pi * NRES_NEO_DELTA / period_f);
    if (!isfinite(alpha)) {
        return;
    }
    const double beta = cos(2.0 * pi / period_f);
    const double r = 1.0 / (1.0 + alpha);
    const double c1 = 2.0 * r * beta;
    const double c2 = -(2.0 * r - 1.0);
    const double gain = alpha * r;
    const double ema_alpha = 2.0 / (static_cast<double>(NRES_NEO_SIGNAL_LENGTH) + 1.0);

    // RollingAbsMax100 (:422-483): values, push indices, head, len, counter.
    double dq_val[NRES_NEO_DEQUE_CAP];
    int dq_idx[NRES_NEO_DEQUE_CAP];
    int dq_head = 0;
    int dq_len = 0;
    int next_index = 0;

    double src_prev1 = 0.0;
    double src_prev2 = 0.0;
    int src_count = 0;
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    double ema_value = 0.0;
    bool ema_seeded = false;

    for (int i = 0; i < n; ++i) {
        const double value = prices[i];

        // :666-679 -- a non-finite bar clears the resonator, the EMA and the
        // peak window, and emits NaN.
        if (!isfinite(value)) {
            src_prev1 = 0.0;
            src_prev2 = 0.0;
            src_count = 0;
            bp_prev1 = 0.0;
            bp_prev2 = 0.0;
            dq_head = 0;
            dq_len = 0;
            next_index = 0;
            ema_value = 0.0;
            ema_seeded = false;
            continue;
        }

        if (src_count >= 2) {
            // :682-683 -- three products summed LEFT TO RIGHT, two roundings
            // for the adds and one each for the products. -fmad=false keeps the
            // compiler from contracting any pair into an fma, which is what the
            // CPU does (there is no mul_add here).
            const double bp = gain * (value - src_prev2) + c1 * bp_prev1 + c2 * bp_prev2;

            // RollingAbsMax100::update(bp.abs()) (:450-482).
            const double pushed = fabs(bp);
            const int index = next_index;
            next_index = next_index + 1;

            while (dq_len > 0) {
                int back = dq_head + dq_len - 1;
                if (back >= NRES_NEO_DEQUE_CAP) {
                    back -= NRES_NEO_DEQUE_CAP;
                }
                if (dq_val[back] <= pushed) {
                    dq_len -= 1;
                } else {
                    break;
                }
            }
            {
                int tail = dq_head + dq_len;
                if (tail >= NRES_NEO_DEQUE_CAP) {
                    tail -= NRES_NEO_DEQUE_CAP;
                }
                dq_idx[tail] = index;
                dq_val[tail] = pushed;
                dq_len += 1;
            }
            // saturating_sub on the CPU (:468): index + 1 - 100, floored at 0.
            const int min_index =
                (index + 1 >= NRES_NEO_PERIOD) ? (index + 1 - NRES_NEO_PERIOD) : 0;
            while (dq_len > 0 && dq_idx[dq_head] < min_index) {
                dq_head += 1;
                if (dq_head == NRES_NEO_DEQUE_CAP) {
                    dq_head = 0;
                }
                dq_len -= 1;
            }
            const double peak = (dq_len > 0) ? dq_val[dq_head] : 0.0;

            const double oscillator = (peak > 0.0) ? (bp / peak) : 0.0;
            if (ema_seeded) {
                // :693 -- ema_value += ema_alpha * (oscillator - ema_value):
                // one subtract, one multiply, one add. NOT an fma.
                ema_value = ema_value + ema_alpha * (oscillator - ema_value);
            } else {
                ema_value = oscillator;
                ema_seeded = true;
            }

            bp_prev2 = bp_prev1;
            bp_prev1 = bp;
            row[i] = oscillator;
        }

        // :712-725 -- the source history advances on every finite bar, whether
        // or not an output was produced.
        if (src_count == 0) {
            src_prev1 = value;
            src_count = 1;
        } else if (src_count == 1) {
            src_prev2 = src_prev1;
            src_prev1 = value;
            src_count = 2;
        } else {
            src_prev2 = src_prev1;
            src_prev1 = value;
        }
    }
}
