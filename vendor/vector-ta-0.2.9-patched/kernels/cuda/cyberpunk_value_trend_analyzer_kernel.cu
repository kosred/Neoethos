#include <cmath>
#include <cstddef>

namespace {

constexpr int SMA13_WINDOW = 13;
constexpr int RANGE75_WINDOW = 75;
constexpr int RANGE75_QUEUE_CAP = RANGE75_WINDOW + 1;

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

struct RollingSumDevice {
    double buf[SMA13_WINDOW];
    int pos;
    int count;
    double sum;

    __device__ void reset() {
        pos = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ double push(double value, bool* ready) {
        if (count < SMA13_WINDOW) {
            buf[pos] = value;
            pos = (pos + 1) % SMA13_WINDOW;
            count += 1;
            sum += value;
            if (count == SMA13_WINDOW) {
                *ready = true;
                return sum / static_cast<double>(SMA13_WINDOW);
            }
            *ready = false;
            return NAN;
        }

        const double old = buf[pos];
        buf[pos] = value;
        pos = (pos + 1) % SMA13_WINDOW;
        sum += value - old;
        *ready = true;
        return sum / static_cast<double>(SMA13_WINDOW);
    }
};

struct MonotonicQueueDevice {
    int idx[RANGE75_QUEUE_CAP];
    double val[RANGE75_QUEUE_CAP];
    int head;
    int tail;

    __device__ void reset() {
        head = 0;
        tail = 0;
    }

    __device__ void compact() {
        if (head <= 0) {
            return;
        }
        const int size = tail - head;
        for (int i = 0; i < size; ++i) {
            idx[i] = idx[head + i];
            val[i] = val[head + i];
        }
        head = 0;
        tail = size;
    }

    __device__ void prepare_for_push() {
        if (tail >= RANGE75_QUEUE_CAP && head > 0) {
            compact();
        }
    }

    __device__ void push_min(int index, double value) {
        while (tail > head) {
            if (val[tail - 1] <= value) {
                break;
            }
            tail -= 1;
        }
        prepare_for_push();
        if (tail < RANGE75_QUEUE_CAP) {
            idx[tail] = index;
            val[tail] = value;
            tail += 1;
        }
    }

    __device__ void push_max(int index, double value) {
        while (tail > head) {
            if (val[tail - 1] >= value) {
                break;
            }
            tail -= 1;
        }
        prepare_for_push();
        if (tail < RANGE75_QUEUE_CAP) {
            idx[tail] = index;
            val[tail] = value;
            tail += 1;
        }
    }

    __device__ void prune(int min_index) {
        while (tail > head && idx[head] < min_index) {
            head += 1;
        }
        if (head == tail) {
            head = 0;
            tail = 0;
        }
    }

    __device__ double current() const {
        return tail > head ? val[head] : NAN;
    }
};

struct WeightedSmaDevice {
    double alpha;
    double value;
    bool has_value;

    __device__ void init(double next_alpha) {
        alpha = next_alpha;
        reset();
    }

    __device__ void reset() {
        value = NAN;
        has_value = false;
    }

    __device__ double update(double source) {
        if (!isfinite(source)) {
            reset();
            return NAN;
        }
        const double next = has_value ? (alpha * source + (1.0 - alpha) * value) : source;
        value = next;
        has_value = true;
        return next;
    }
};

}

extern "C" __global__ void cyberpunk_value_trend_analyzer_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ entry_levels,
    const int* __restrict__ exit_levels,
    int rows,
    double* __restrict__ out_value_trend,
    double* __restrict__ out_value_trend_lag,
    double* __restrict__ out_deviation_index,
    double* __restrict__ out_overbought_signal,
    double* __restrict__ out_buy_signal,
    double* __restrict__ out_sell_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int entry_level = entry_levels[row];
    const int exit_level = exit_levels[row];

    double* row_value_trend = out_value_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_value_trend_lag =
        out_value_trend_lag + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_deviation_index =
        out_deviation_index + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_overbought_signal =
        out_overbought_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_buy_signal = out_buy_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_sell_signal = out_sell_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value_trend[i] = NAN;
        row_value_trend_lag[i] = NAN;
        row_deviation_index[i] = NAN;
        row_overbought_signal[i] = NAN;
        row_buy_signal[i] = NAN;
        row_sell_signal[i] = NAN;
    }

    if (entry_level < 1 || entry_level > 100 || exit_level < 1 || exit_level > 100) {
        return;
    }

    RollingSumDevice sma13;
    MonotonicQueueDevice lowest75;
    MonotonicQueueDevice highest75;
    WeightedSmaDevice close_norm_sma;
    WeightedSmaDevice smooth5;
    sma13.reset();
    lowest75.reset();
    highest75.reset();
    close_norm_sma.init(1.0 / 20.0);
    smooth5.init(1.0 / 5.0);

    int valid_run = 0;
    double prev_value_trend = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!is_valid_ohlc(o, h, l, c)) {
            sma13.reset();
            lowest75.reset();
            highest75.reset();
            close_norm_sma.reset();
            smooth5.reset();
            valid_run = 0;
            prev_value_trend = NAN;
            continue;
        }

        valid_run += 1;

        bool sma_ready = false;
        const double avg13 = sma13.push(c, &sma_ready);
        lowest75.push_min(i, l);
        highest75.push_max(i, h);
        const int min_index = i >= RANGE75_WINDOW - 1 ? (i - (RANGE75_WINDOW - 1)) : 0;
        lowest75.prune(min_index);
        highest75.prune(min_index);

        if (isfinite(prev_value_trend)) {
            row_value_trend_lag[i] = prev_value_trend;
        }

        double current_value_trend = NAN;
        if (valid_run >= RANGE75_WINDOW) {
            const double range_low = lowest75.current();
            const double range_high = highest75.current();
            const double range = range_high - range_low;
            if (isfinite(range) && range > 0.0) {
                const double close_norm = (c - range_low) * 100.0 / range;
                const double close_norm_avg = close_norm_sma.update(close_norm);
                const double smooth = smooth5.update(close_norm_avg);
                if (isfinite(close_norm_avg) && isfinite(smooth)) {
                    current_value_trend = 3.0 * close_norm_avg - 2.0 * smooth;
                    row_value_trend[i] = current_value_trend;
                    row_buy_signal[i] = 0.0;
                    row_sell_signal[i] = 0.0;
                }
            } else {
                close_norm_sma.reset();
                smooth5.reset();
            }
        }

        if (sma_ready && isfinite(avg13) && avg13 != 0.0) {
            const double deviation_index = 100.0 - fabs(((c - avg13) / avg13) * 100.0);
            row_deviation_index[i] = deviation_index;
            if (isfinite(current_value_trend) && current_value_trend > deviation_index) {
                row_overbought_signal[i] = deviation_index;
            }
        }

        if (isfinite(current_value_trend) && isfinite(prev_value_trend)) {
            if (prev_value_trend <= static_cast<double>(entry_level) &&
                current_value_trend > static_cast<double>(entry_level)) {
                row_buy_signal[i] = 1.0;
            }
            if (prev_value_trend >= static_cast<double>(exit_level) &&
                current_value_trend < static_cast<double>(exit_level)) {
                row_sell_signal[i] = 1.0;
            }
        }

        prev_value_trend = current_value_trend;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/cyberpunk_value_trend_analyzer.rs:641
// (cyberpunk_value_trend_analyzer_with_kernel). The column this emits is the
// canonical primary `value_trend`; the retired unversioned `value` alias is
// deliberately not part of the production schema.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- two
// monotone deques carrying the 75-bar low and high, and two chained one-pole
// filters (alpha 1/20 then alpha 1/5) whose state is carried and RESET by the
// CPU on a non-finite bar OR on a degenerate zero range. A monotone deque is
// the classic sliding extreme and is not bar-parallel; nor is the filter pair.
//
// PERIOD-INVARIANT. compute_cyberpunk_value_trend_analyzer_batch
// (cpu_batch.rs:14657-14660) reads entry_level and exit_level and NEVER
// period, so five swept periods give five identical CPU columns and this
// kernel emits five identical rows. Both CPU defaults are pinned below --
// and neither of them enters value_trend at all: they gate the buy and sell
// signal columns only. They are still validated, because the CPU REFUSES the
// whole computation for an out-of-range level and this kernel must refuse the
// same inputs.
//
// WHAT IS DELIBERATELY ABSENT: the 13-bar rolling mean and the deviation
// index it feeds. They are consumed only by deviation_index and
// overbought_signal, never by value_trend.
//
// THE VALID-RUN COUNTER IS NOT THE BAR INDEX: the CPU counts CONSECUTIVE valid
// OHLC bars and requires 75 of them before it publishes, restarting the count
// at every invalid bar. Reproduced exactly -- using i >= 74 instead would
// publish across a gap the CPU refuses to cross.
//
// FIRST VALID IS NOT READ: there is no warmup index, only that consecutive
// run. The lane row declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double fabs, no f32-suffixed math function,
// no fast-math intrinsic, and no epsilon -- the range test is the CPU's exact
// range > 0.0.
// ---------------------------------------------------------------------------

#define NEO_CVTA_ENTRY_LEVEL 30
#define NEO_CVTA_EXIT_LEVEL 75

extern "C" __global__ void cyberpunk_value_trend_analyzer_neo_batch_f64(
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
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int entry_level = NEO_CVTA_ENTRY_LEVEL;
    const int exit_level = NEO_CVTA_EXIT_LEVEL;
    if (entry_level < 1 || entry_level > 100 || exit_level < 1 || exit_level > 100) {
        return;
    }

    MonotonicQueueDevice lowest75;
    MonotonicQueueDevice highest75;
    WeightedSmaDevice close_norm_sma;
    WeightedSmaDevice smooth5;
    lowest75.reset();
    highest75.reset();
    close_norm_sma.init(1.0 / 20.0);
    smooth5.init(1.0 / 5.0);

    int valid_run = 0;

    for (int i = 0; i < n; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!is_valid_ohlc(o, h, l, c)) {
            lowest75.reset();
            highest75.reset();
            close_norm_sma.reset();
            smooth5.reset();
            valid_run = 0;
            continue;
        }

        valid_run += 1;

        lowest75.push_min(i, l);
        highest75.push_max(i, h);
        const int min_index = i >= RANGE75_WINDOW - 1 ? (i - (RANGE75_WINDOW - 1)) : 0;
        lowest75.prune(min_index);
        highest75.prune(min_index);

        if (valid_run >= RANGE75_WINDOW) {
            const double range_low = lowest75.current();
            const double range_high = highest75.current();
            const double range = range_high - range_low;
            if (isfinite(range) && range > 0.0) {
                const double close_norm = (c - range_low) * 100.0 / range;
                const double close_norm_avg = close_norm_sma.update(close_norm);
                const double smooth = smooth5.update(close_norm_avg);
                if (isfinite(close_norm_avg) && isfinite(smooth)) {
                    row[i] = 3.0 * close_norm_avg - 2.0 * smooth;
                }
            } else {
                close_norm_sma.reset();
                smooth5.reset();
            }
        }
    }
}
