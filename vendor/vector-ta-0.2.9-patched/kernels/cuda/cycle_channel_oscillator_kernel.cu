#include <cmath>
#include <cstddef>

namespace {
struct RmaState {
    int length;
    int count;
    double sum;
    double value;
    bool seeded;

    __device__ inline void init(int len) {
        length = len;
        count = 0;
        sum = 0.0;
        value = NAN;
        seeded = false;
    }

    __device__ inline double update(double input) {
        if (count < length) {
            sum += input;
            count += 1;
            if (count == length) {
                value = sum / static_cast<double>(length);
                seeded = true;
            }
        } else {
            value = value + (input - value) / static_cast<double>(length);
            count += 1;
        }
        return value;
    }
};

struct AtrState {
    RmaState rma;
    bool have_prev_close;
    double prev_close;

    __device__ inline void init(int len) {
        rma.init(len);
        have_prev_close = false;
        prev_close = NAN;
    }

    __device__ inline double update(double high, double low, double close) {
        const double tr = have_prev_close
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        have_prev_close = true;
        return rma.update(tr);
    }
};
}

extern "C" __global__ void cycle_channel_oscillator_batch_f64(
    const double* source,
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* short_cycle_lengths,
    const int* medium_cycle_lengths,
    const double* short_multipliers,
    const double* medium_multipliers,
    int rows,
    double* out_fast,
    double* out_slow,
    double* short_history,
    double* medium_history
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int short_cycle_length = short_cycle_lengths[row];
    const int medium_cycle_length = medium_cycle_lengths[row];
    const double short_multiplier = short_multipliers[row];
    const double medium_multiplier = medium_multipliers[row];

    double* row_fast = out_fast + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_slow = out_slow + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_history =
        short_history + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_medium_history =
        medium_history + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_fast[i] = NAN;
        row_slow[i] = NAN;
        row_short_history[i] = NAN;
        row_medium_history[i] = NAN;
    }

    if (short_cycle_length < 2 || medium_cycle_length < 2 || !isfinite(short_multiplier)
        || short_multiplier < 0.0 || !isfinite(medium_multiplier) || medium_multiplier < 0.0) {
        return;
    }

    const int short_period = short_cycle_length / 2;
    const int medium_period = medium_cycle_length / 2;
    const int short_delay = short_period / 2;
    const int medium_delay = medium_period / 2;

    RmaState short_rma;
    RmaState medium_rma;
    AtrState medium_atr;
    short_rma.init(short_period);
    medium_rma.init(medium_period);
    medium_atr.init(medium_period);

    int valid_count = 0;
    for (int i = 0; i < len; ++i) {
        const double src = source[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (!(isfinite(src) && isfinite(h) && isfinite(l) && isfinite(c))) {
            continue;
        }

        const double short_ma = short_rma.update(src);
        const double medium_ma = medium_rma.update(src);
        const double medium_atr_value = medium_atr.update(h, l, c);

        row_short_history[valid_count] = short_ma;
        row_medium_history[valid_count] = medium_ma;

        double short_center = src;
        if (valid_count + 1 > short_delay) {
            const double delayed = row_short_history[valid_count - short_delay];
            if (isfinite(delayed)) {
                short_center = delayed;
            }
        }

        double medium_center = src;
        if (valid_count + 1 > medium_delay) {
            const double delayed = row_medium_history[valid_count - medium_delay];
            if (isfinite(delayed)) {
                medium_center = delayed;
            }
        }

        const double offset = medium_multiplier * medium_atr_value;
        const double denom = 2.0 * offset;
        if (isfinite(denom) && denom != 0.0) {
            const double medium_bottom = medium_center - offset;
            row_fast[i] = (src - medium_bottom) / denom;
            row_slow[i] = (short_center - medium_bottom) / denom;
        }

        valid_count += 1;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/cycle_channel_oscillator.rs:594
// (`cycle_channel_oscillator_with_kernel`). The preserved primary entry emits
// canonical `fast`; the full entry above emits canonical `[fast, slow]`.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- three
// interlocking Wilder RMA recurrences (the short MA, the medium MA and the
// medium ATR, all `value + (input - value)/length` after a mean seed) plus a
// VALID-BAR COUNTER that indexes the delay history. The counter advances only
// on bars where source, high, low and close are ALL finite, so a bar-parallel
// form cannot know its own history index.
//
// DEFAULT-ONLY PRIMARY ABI. The canonical production planner maps its admitted
// ratio points into explicit short/medium lengths and launches the full
// two-output entry point above. This preserved primary entry pins every CPU
// default below and emits only canonical `fast`; `source` defaults to "close",
// which is why this kernel takes the HLC triple and reads CLOSE where the full
// entry point takes a separate `source` pointer.
//
// THE DELAY HISTORY IS A RING, NOT THE WHOLE SERIES. The entry point above
// keeps a `len`-long history per row because it is handed scratch; a lane
// kernel is not. The CPU only ever reads `history[valid_count - delay]`, so a
// `delay + 1` ring holds exactly what is reachable. `short_delay` is
// `(short_cycle_length / 2) / 2` = 2 and `medium_delay` = 7 at the pinned
// defaults; the ring bound below is checked, not assumed.
//
// FIRST VALID IS NOT READ: the CPU emits from bar 0 and SKIPS -- rather than
// stops at -- any bar whose four series are not all finite, so there is no
// warmup index. The lane row declares `F64FirstValidRule::Ignored`.
//
// f64 END TO END: `fmax`/`fabs` are the double overloads (and `fmax` is what
// the CPU's `f64::max` is, so a NaN true range cannot survive an if-chain --
// rule 4), every literal is a double, and there is no fast-math intrinsic.
// ---------------------------------------------------------------------------

#define NEO_CCO_SHORT_CYCLE_LENGTH 10
#define NEO_CCO_MEDIUM_CYCLE_LENGTH 30
#define NEO_CCO_SHORT_MULTIPLIER 1.0
#define NEO_CCO_MEDIUM_MULTIPLIER 3.0
#define NEO_CCO_MAX_DELAY_RING 512

extern "C" __global__ void cycle_channel_oscillator_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int short_cycle_length = NEO_CCO_SHORT_CYCLE_LENGTH;
    const int medium_cycle_length = NEO_CCO_MEDIUM_CYCLE_LENGTH;
    const double short_multiplier = NEO_CCO_SHORT_MULTIPLIER;
    const double medium_multiplier = NEO_CCO_MEDIUM_MULTIPLIER;

    if (short_cycle_length < 2 || medium_cycle_length < 2 || !isfinite(short_multiplier) ||
        short_multiplier < 0.0 || !isfinite(medium_multiplier) || medium_multiplier < 0.0) {
        return;
    }

    const int short_period = short_cycle_length / 2;
    const int medium_period = medium_cycle_length / 2;
    const int short_delay = short_period / 2;
    const int medium_delay = medium_period / 2;
    const int short_ring_len = short_delay + 1;
    const int medium_ring_len = medium_delay + 1;
    if (short_ring_len > NEO_CCO_MAX_DELAY_RING || medium_ring_len > NEO_CCO_MAX_DELAY_RING) {
        return;
    }

    double short_hist[NEO_CCO_MAX_DELAY_RING];
    double medium_hist[NEO_CCO_MAX_DELAY_RING];
    for (int j = 0; j < short_ring_len; ++j) {
        short_hist[j] = NAN;
    }
    for (int j = 0; j < medium_ring_len; ++j) {
        medium_hist[j] = NAN;
    }

    RmaState short_rma;
    RmaState medium_rma;
    AtrState medium_atr;
    short_rma.init(short_period);
    medium_rma.init(medium_period);
    medium_atr.init(medium_period);

    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
        const double src = close[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (!(isfinite(src) && isfinite(h) && isfinite(l) && isfinite(c))) {
            continue;
        }

        const double short_ma = short_rma.update(src);
        const double medium_ma = medium_rma.update(src);
        const double medium_atr_value = medium_atr.update(h, l, c);

        short_hist[valid_count % short_ring_len] = short_ma;
        medium_hist[valid_count % medium_ring_len] = medium_ma;

        double short_center = src;
        if (valid_count + 1 > short_delay) {
            const double delayed = short_hist[(valid_count - short_delay) % short_ring_len];
            if (isfinite(delayed)) {
                short_center = delayed;
            }
        }
        (void)short_center;

        double medium_center = src;
        if (valid_count + 1 > medium_delay) {
            const double delayed = medium_hist[(valid_count - medium_delay) % medium_ring_len];
            if (isfinite(delayed)) {
                medium_center = delayed;
            }
        }

        const double offset = medium_multiplier * medium_atr_value;
        const double denom = 2.0 * offset;
        if (isfinite(denom) && denom != 0.0) {
            const double medium_bottom = medium_center - offset;
            row[i] = (src - medium_bottom) / denom;
        }

        valid_count += 1;
    }
}
