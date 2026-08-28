#include <cmath>
#include <cstddef>

namespace {
struct RsiState {
    int period;
    double inv_p;
    double beta;
    bool initialized;
    double prev;
    int deltas_seen;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool ready;

    __device__ void init(int value) {
        period = value;
        inv_p = 1.0 / static_cast<double>(value);
        beta = 1.0 - inv_p;
        reset();
    }

    __device__ void reset() {
        initialized = false;
        prev = NAN;
        deltas_seen = 0;
        sum_gain = 0.0;
        sum_loss = 0.0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        ready = false;
    }

    __device__ double update(double value) {
        if (!initialized) {
            prev = value;
            initialized = true;
            return NAN;
        }

        const double delta = value - prev;
        prev = value;
        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);

        if (!ready) {
            sum_gain += gain;
            sum_loss += loss;
            deltas_seen += 1;
            if (deltas_seen < period) {
                return NAN;
            }
            avg_gain = sum_gain * inv_p;
            avg_loss = sum_loss * inv_p;
            ready = true;
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

struct WmaState {
    int len;
    double denom;
    double* buf;
    int pos;
    int count;
    double sum;
    double weighted_sum;

    __device__ void init(int length, double* storage) {
        len = length;
        denom = static_cast<double>(length * (length + 1) / 2);
        buf = storage;
        reset();
    }

    __device__ void reset() {
        pos = 0;
        count = 0;
        sum = 0.0;
        weighted_sum = 0.0;
    }

    __device__ double update(double value) {
        if (count < len) {
            buf[count] = value;
            count += 1;
            sum += value;
            weighted_sum += static_cast<double>(count) * value;
            if (count == len) {
                return weighted_sum / denom;
            }
            return NAN;
        }

        const double old_sum = sum;
        const double old = buf[pos];
        buf[pos] = value;
        pos += 1;
        if (pos == len) {
            pos = 0;
        }
        weighted_sum = weighted_sum + static_cast<double>(len) * value - old_sum;
        sum = old_sum + value - old;
        return weighted_sum / denom;
    }
};
}

extern "C" __global__ void relative_strength_index_wave_indicator_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ length1s,
    const int* __restrict__ length2s,
    const int* __restrict__ length3s,
    const int* __restrict__ length4s,
    int rows,
    double* __restrict__ out_rsi_ma1,
    double* __restrict__ out_rsi_ma2,
    double* __restrict__ out_rsi_ma3,
    double* __restrict__ out_rsi_ma4,
    double* __restrict__ out_state
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int rsi_length = rsi_lengths[row];
    const int length1 = length1s[row];
    const int length2 = length2s[row];
    const int length3 = length3s[row];
    const int length4 = length4s[row];

    double* row_ma1 = out_rsi_ma1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma2 = out_rsi_ma2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma3 = out_rsi_ma3 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma4 = out_rsi_ma4 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_ma1[i] = NAN;
        row_ma2[i] = NAN;
        row_ma3[i] = NAN;
        row_ma4[i] = NAN;
        row_state[i] = NAN;
    }

    if (rsi_length <= 0 || length1 <= 0 || length2 <= 0 || length3 <= 0 || length4 <= 0) {
        return;
    }

    double* w1_buf = new double[length1];
    double* w2_buf = new double[length2];
    double* w3_buf = new double[length3];
    double* w4_buf = new double[length4];
    if (w1_buf == nullptr || w2_buf == nullptr || w3_buf == nullptr || w4_buf == nullptr) {
        delete[] w1_buf;
        delete[] w2_buf;
        delete[] w3_buf;
        delete[] w4_buf;
        return;
    }

    RsiState rsi_source;
    RsiState rsi_high;
    RsiState rsi_low;
    rsi_source.init(rsi_length);
    rsi_high.init(rsi_length);
    rsi_low.init(rsi_length);

    WmaState wma1;
    WmaState wma2;
    WmaState wma3;
    WmaState wma4;
    wma1.init(length1, w1_buf);
    wma2.init(length2, w2_buf);
    wma3.init(length3, w3_buf);
    wma4.init(length4, w4_buf);

    bool has_prev_slo = false;
    double prev_slo = NAN;

    for (int i = 0; i < len; ++i) {
        const double source_value = source[i];
        const double high_value = high[i];
        const double low_value = low[i];

        if (!isfinite(source_value) || !isfinite(high_value) || !isfinite(low_value)) {
            rsi_source.reset();
            rsi_high.reset();
            rsi_low.reset();
            wma1.reset();
            wma2.reset();
            wma3.reset();
            wma4.reset();
            has_prev_slo = false;
            prev_slo = NAN;
            continue;
        }

        const double custom_rsi = rsi_source.update(source_value);
        const double high_rsi = rsi_high.update(high_value);
        const double low_rsi = rsi_low.update(low_value);
        if (!isfinite(custom_rsi) || !isfinite(high_rsi) || !isfinite(low_rsi)) {
            continue;
        }

        const double hlc_rsi = (high_rsi + low_rsi + 2.0 * custom_rsi) * 0.25;
        const double rsi_ma1 = wma1.update(hlc_rsi);
        const double rsi_ma2 = wma2.update(hlc_rsi);
        const double rsi_ma3 = wma3.update(hlc_rsi);
        const double rsi_ma4 = wma4.update(hlc_rsi);

        row_ma1[i] = rsi_ma1;
        row_ma2[i] = rsi_ma2;
        row_ma3[i] = rsi_ma3;
        row_ma4[i] = rsi_ma4;

        if (isfinite(rsi_ma1) && isfinite(rsi_ma2)) {
            const double slo = rsi_ma1 - rsi_ma2;
            const double prev = has_prev_slo ? prev_slo : 0.0;
            prev_slo = slo;
            has_prev_slo = true;
            if (slo > 0.0) {
                row_state[i] = slo > prev ? 2.0 : 1.0;
            } else if (slo < 0.0) {
                row_state[i] = slo < prev ? -2.0 : -1.0;
            } else {
                row_state[i] = 0.0;
            }
        }
    }

    delete[] w1_buf;
    delete[] w2_buf;
    delete[] w3_buf;
    delete[] w4_buf;
}


// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// relative_strength_index_wave_indicator_batch_f64 above is genuine
// double-in/double-out, but it takes 15 parameters, writes FIVE output matrices
// and calls `new double[]` on the DEVICE for its four WMA rings. The f64 lane
// launches one shape -- (high, low, close, n, periods, n_combos, first_valid,
// out) -- and allocates ONE output matrix, so that entry point cannot be
// reached from it.
//
// CPU REFERENCE
//   src/indicators/relative_strength_index_wave_indicator.rs:708
//     relative_strength_index_wave_indicator_with_kernel ->
//     :641 compute_relative_strength_index_wave_indicator_into ->
//     RelativeStrengthIndexWaveCore::update :594
//   RsiCore::update :477   WmaCore::update :549
//
// THE COLUMN THIS EMITS is rsi_ma1, which is what output_id == "value" resolves
// to (cpu_batch.rs -- "rsi_ma1" || "value").
//
// PINNED CPU DEFAULTS (compute_relative_strength_index_wave_indicator_batch):
// source "close", rsi_length 14, length1 2. length2/3/4 (5, 9, 13) are NOT
// pinned because they feed rsi_ma2/3/4 and the state column, none of which is
// the emitted one -- rsi_ma1 comes from wma1 alone (:614).
//
// THE INPUT IS A TRIPLE, and the third series is the SOURCE, not merely a
// close: the CPU runs THREE independent Wilder RSIs -- one on the source, one
// on high, one on low (:601-603) -- and blends them. With source defaulting to
// "close" the lane's Hlc shape is exactly right, which is why the lane row
// declares F64InputKind::Hlc.
//
// PERIOD-INVARIANT. The batch reads source, rsi_length and length1..length4 and
// NEVER `period`, so five swept periods give five identical CPU columns and
// this kernel emits five identical rows.
//
// SHAPE: one thread per combo, bars ascending. Three Wilder recursions plus a
// rolling WMA, all carried; a bar where any of source/high/low is non-finite
// RESETS all four (:595-598), so the series restarts its warmup after every
// gap.
//
// ARITHMETIC ORDER:
//   * The Wilder step is `avg_gain.mul_add(beta, inv_p * gain)` (:494) -- ONE
//     rounding -- so `fma()` is used. The seed is `sum_gain * inv_p` (:490),
//     a multiply by the reciprocal, NOT a divide.
//   * `gain = delta.max(0.0)` and `loss = (-delta).max(0.0)` are f64::max, so
//     fmax is used: a NaN delta must not win the comparison and slip into the
//     carried average as a finite number.
//   * `hlc_rsi = (high_rsi + low_rsi + 2.0 * custom_rsi) * 0.25` (:613) -- the
//     adds are left to right and the scale is a multiply by 0.25, not a divide
//     by 4.
//   * WmaCore on a full ring (:566-567) computes
//     `weighted_sum = weighted_sum + len*value - old_sum` then
//     `sum = old_sum + value - old`, both reading the PRE-update sum, and its
//     divisor is `(len * (len + 1) / 2) as f64` -- an INTEGER division, exact.
//
// FIRST VALID IS NOT READ: compute_..._into writes every index (NaN before the
// cascade is ready and after every gap). The lane row declares
// F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double fmax/fma, no f32-suffixed math
// function, no fast-math intrinsic, and no epsilon -- the CPU's only guard on
// this path is the literal `denom == 0.0` at :501.
// ---------------------------------------------------------------------------

#define RSIW_NEO_RSI_LENGTH 14
#define RSIW_NEO_LENGTH1 2

__device__ __forceinline__ double rsiw_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// RsiCore (relative_strength_index_wave_indicator.rs:404-506), carried per
// series. Three instances: source, high, low.
struct RsiwNeoRsi {
    bool initialized;
    double prev;
    int deltas_seen;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool ready;
};

__device__ __forceinline__ void rsiw_neo_rsi_reset(RsiwNeoRsi* s) {
    s->initialized = false;
    s->prev = rsiw_neo_qnan();
    s->deltas_seen = 0;
    s->sum_gain = 0.0;
    s->sum_loss = 0.0;
    s->avg_gain = 0.0;
    s->avg_loss = 0.0;
    s->ready = false;
}

__device__ __forceinline__ double rsiw_neo_rsi_update(
    RsiwNeoRsi* s,
    double value,
    double inv_p,
    double beta
) {
    if (!s->initialized) {
        s->prev = value;
        s->initialized = true;
        return rsiw_neo_qnan();
    }
    const double delta = value - s->prev;
    s->prev = value;
    const double gain = fmax(delta, 0.0);
    const double loss = fmax(-delta, 0.0);

    if (!s->ready) {
        s->sum_gain += gain;
        s->sum_loss += loss;
        s->deltas_seen += 1;
        if (s->deltas_seen < RSIW_NEO_RSI_LENGTH) {
            return rsiw_neo_qnan();
        }
        s->avg_gain = s->sum_gain * inv_p;
        s->avg_loss = s->sum_loss * inv_p;
        s->ready = true;
    } else {
        s->avg_gain = fma(s->avg_gain, beta, inv_p * gain);
        s->avg_loss = fma(s->avg_loss, beta, inv_p * loss);
    }

    const double denom = s->avg_gain + s->avg_loss;
    if (denom == 0.0) {
        return 50.0;
    }
    return 100.0 * s->avg_gain / denom;
}

extern "C" __global__ void relative_strength_index_wave_indicator_neo_batch_f64(
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
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    const double qnan = rsiw_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    const double inv_p = 1.0 / static_cast<double>(RSIW_NEO_RSI_LENGTH);
    const double beta = 1.0 - inv_p;

    RsiwNeoRsi rsi_source;
    RsiwNeoRsi rsi_high;
    RsiwNeoRsi rsi_low;
    rsiw_neo_rsi_reset(&rsi_source);
    rsiw_neo_rsi_reset(&rsi_high);
    rsiw_neo_rsi_reset(&rsi_low);

    // WmaCore::new(length1) (:513-523). The divisor is the CPU's INTEGER
    // (len * (len + 1) / 2) cast to f64 -- exact, because the product is even.
    double wma_ring[RSIW_NEO_LENGTH1];
    int wma_pos = 0;
    int wma_count = 0;
    double wma_sum = 0.0;
    double wma_weighted_sum = 0.0;
    const double wma_denom =
        static_cast<double>(RSIW_NEO_LENGTH1 * (RSIW_NEO_LENGTH1 + 1) / 2);

    for (int i = 0; i < n; ++i) {
        const double source_value = close[i];
        const double high_value = high[i];
        const double low_value = low[i];

        if (!isfinite(source_value) || !isfinite(high_value) || !isfinite(low_value)) {
            // :595-598 -- reset() clears the three RSIs, the WMAs and prev_slo.
            rsiw_neo_rsi_reset(&rsi_source);
            rsiw_neo_rsi_reset(&rsi_high);
            rsiw_neo_rsi_reset(&rsi_low);
            wma_pos = 0;
            wma_count = 0;
            wma_sum = 0.0;
            wma_weighted_sum = 0.0;
            continue;
        }

        const double custom_rsi = rsiw_neo_rsi_update(&rsi_source, source_value, inv_p, beta);
        const double high_rsi = rsiw_neo_rsi_update(&rsi_high, high_value, inv_p, beta);
        const double low_rsi = rsiw_neo_rsi_update(&rsi_low, low_value, inv_p, beta);
        if (!isfinite(custom_rsi) || !isfinite(high_rsi) || !isfinite(low_rsi)) {
            continue;
        }

        const double hlc_rsi = (high_rsi + low_rsi + 2.0 * custom_rsi) * 0.25;

        // WmaCore::update (:549-575).
        double rsi_ma1 = qnan;
        if (wma_count < RSIW_NEO_LENGTH1) {
            wma_ring[wma_count] = hlc_rsi;
            wma_count += 1;
            wma_sum += hlc_rsi;
            wma_weighted_sum += static_cast<double>(wma_count) * hlc_rsi;
            if (wma_count == RSIW_NEO_LENGTH1) {
                rsi_ma1 = wma_weighted_sum / wma_denom;
            }
        } else {
            const double old_sum = wma_sum;
            const double old = wma_ring[wma_pos];
            wma_ring[wma_pos] = hlc_rsi;
            wma_pos += 1;
            if (wma_pos == RSIW_NEO_LENGTH1) {
                wma_pos = 0;
            }
            wma_weighted_sum =
                wma_weighted_sum + static_cast<double>(RSIW_NEO_LENGTH1) * hlc_rsi - old_sum;
            wma_sum = old_sum + hlc_rsi - old;
            rsi_ma1 = wma_weighted_sum / wma_denom;
        }

        row[i] = rsi_ma1;
    }
}
