#include <cmath>
#include <cstddef>

namespace {
constexpr int MIN_DATA_LENGTH = 1;
constexpr int MIN_NORMALIZATION_LENGTH = 10;
constexpr double MIN_POSITIVE = 2.2250738585072014e-308;
constexpr double BIAS_BEARISH = 0.0;
constexpr double BIAS_BULLISH = 1.0;

struct MonoDeque {
    int* idx;
    double* vals;
    int cap;
    int head;
    int tail;
    int count;
    bool descending;

    __device__ void init(int* idx_ptr, double* vals_ptr, int capacity, bool is_descending) {
        idx = idx_ptr;
        vals = vals_ptr;
        cap = capacity;
        descending = is_descending;
        clear();
    }

    __device__ void clear() {
        head = 0;
        tail = 0;
        count = 0;
    }

    __device__ void pop_back() {
        if (count == 0) {
            return;
        }
        tail = (tail + cap - 1) % cap;
        count -= 1;
    }

    __device__ void pop_front() {
        if (count == 0) {
            return;
        }
        head = (head + 1) % cap;
        count -= 1;
    }

    __device__ int back_slot() const {
        return (tail + cap - 1) % cap;
    }

    __device__ void push(int index, double value) {
        while (count > 0) {
            const int slot = back_slot();
            const double last = vals[slot];
            const bool remove = descending ? (last <= value) : (last >= value);
            if (!remove) {
                break;
            }
            pop_back();
        }
        if (count == cap) {
            pop_front();
        }
        idx[tail] = index;
        vals[tail] = value;
        tail = (tail + 1) % cap;
        count += 1;
    }

    __device__ void expire(int min_index) {
        while (count > 0 && idx[head] < min_index) {
            pop_front();
        }
    }

    __device__ double front_value() const {
        return vals[head];
    }
};

struct RingHistory {
    double* values;
    int cap;
    int head;
    int count;

    __device__ void init(double* ptr, int capacity) {
        values = ptr;
        cap = capacity;
        clear();
    }

    __device__ void clear() {
        head = 0;
        count = 0;
    }

    __device__ void push(double value) {
        values[head] = value;
        head += 1;
        if (head == cap) {
            head = 0;
        }
        if (count < cap) {
            count += 1;
        }
    }

    __device__ bool get_from_end(int offset, double* out) const {
        if (offset <= 0 || offset > count) {
            return false;
        }
        int idx = head + cap - offset;
        idx %= cap;
        *out = values[idx];
        return true;
    }
};

struct RollingStats {
    double* ring;
    int cap;
    int head;
    int count;
    double sum;
    double sum_sq;

    __device__ void init(double* ptr, int capacity) {
        ring = ptr;
        cap = capacity;
        clear();
    }

    __device__ void clear() {
        head = 0;
        count = 0;
        sum = 0.0;
        sum_sq = 0.0;
    }

    __device__ bool push(double value, double* mean_out, double* stdev_out) {
        if (count < cap) {
            ring[head] = value;
            head += 1;
            if (head == cap) {
                head = 0;
            }
            count += 1;
            sum += value;
            sum_sq += value * value;
        } else {
            const double old = ring[head];
            ring[head] = value;
            head += 1;
            if (head == cap) {
                head = 0;
            }
            sum += value - old;
            sum_sq += value * value - old * old;
        }

        if (count < cap) {
            return false;
        }

        const double n = static_cast<double>(cap);
        const double mean = sum / n;
        const double variance = fmax(sum_sq / n - mean * mean, 0.0);
        *mean_out = mean;
        *stdev_out = sqrt(variance);
        return true;
    }
};

__device__ inline double hlc3(double high, double low, double close) {
    return (high + low + close) / 3.0;
}

__device__ inline double floor_positive(double value) {
    return value > 0.0 ? value : MIN_POSITIVE;
}
}

extern "C" __global__ void statistical_trailing_stop_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* data_lengths,
    const int* normalization_lengths,
    const int* base_level_indices,
    int rows,
    int deque_cap,
    int stats_cap,
    double* out_level,
    double* out_anchor,
    double* out_bias,
    double* out_changed,
    int* max_high_idx_storage,
    double* max_high_val_storage,
    int* min_low_idx_storage,
    double* min_low_val_storage,
    double* close_history_storage,
    double* stats_ring_storage
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int data_length = data_lengths[row];
    const int normalization_length = normalization_lengths[row];
    const int base_level_index = base_level_indices[row];
    const int row_deque_cap = data_length + 2;

    double* row_level = out_level + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_anchor = out_anchor + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bias = out_bias + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_level[i] = NAN;
        row_anchor[i] = NAN;
        row_bias[i] = NAN;
        row_changed[i] = NAN;
    }

    if (data_length < MIN_DATA_LENGTH || normalization_length < MIN_NORMALIZATION_LENGTH
        || data_length + normalization_length + 1 > len || base_level_index < 0
        || base_level_index > 3 || deque_cap < row_deque_cap || stats_cap < normalization_length) {
        return;
    }

    const size_t deque_offset = static_cast<size_t>(row) * static_cast<size_t>(deque_cap);
    const size_t stats_offset = static_cast<size_t>(row) * static_cast<size_t>(stats_cap);

    MonoDeque max_high;
    MonoDeque min_low;
    RingHistory close_history;
    RollingStats stats;

    max_high.init(
        max_high_idx_storage + deque_offset,
        max_high_val_storage + deque_offset,
        row_deque_cap,
        true
    );
    min_low.init(
        min_low_idx_storage + deque_offset,
        min_low_val_storage + deque_offset,
        row_deque_cap,
        false
    );
    close_history.init(close_history_storage + deque_offset, row_deque_cap);
    stats.init(stats_ring_storage + stats_offset, normalization_length);

    int valid_run = 0;
    double bias = BIAS_BEARISH;
    double level = NAN;
    double anchor = NAN;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            valid_run = 0;
            max_high.clear();
            min_low.clear();
            close_history.clear();
            stats.clear();
            bias = BIAS_BEARISH;
            level = NAN;
            anchor = NAN;
            continue;
        }

        valid_run += 1;
        max_high.push(i, h);
        min_low.push(i, l);
        const int window_start = i + 1 - (valid_run < data_length ? valid_run : data_length);
        max_high.expire(window_start);
        min_low.expire(window_start);
        close_history.push(c);

        if (valid_run < data_length + 2) {
            continue;
        }

        double previous_close = NAN;
        if (!close_history.get_from_end(data_length + 2, &previous_close)) {
            continue;
        }

        const double highest = max_high.front_value();
        const double lowest = min_low.front_value();
        const double tr =
            fmax(highest - lowest, fmax(fabs(highest - previous_close), fabs(lowest - previous_close)));

        double mean = NAN;
        double stdev = NAN;
        if (!stats.push(log(floor_positive(tr)), &mean, &stdev)) {
            continue;
        }

        const double delta = exp(mean + static_cast<double>(base_level_index) * stdev);
        const double current_hlc3 = hlc3(h, l, c);

        if (!isfinite(level)) {
            level =
                bias == BIAS_BEARISH ? (current_hlc3 + delta) : fmax(current_hlc3 - delta, 0.0);
        }

        if (bias == BIAS_BEARISH) {
            level = fmin(level, current_hlc3 + delta);
        } else {
            level = fmax(level, fmax(current_hlc3 - delta, 0.0));
        }

        const bool triggered =
            (bias == BIAS_BEARISH && c >= level) || (bias == BIAS_BULLISH && c <= level);
        double changed = 0.0;

        if (triggered) {
            anchor = c;
            bias = bias == BIAS_BEARISH ? BIAS_BULLISH : BIAS_BEARISH;
            level =
                bias == BIAS_BEARISH ? (current_hlc3 + delta) : fmax(current_hlc3 - delta, 0.0);
            changed = 1.0;
        }

        row_level[i] = level;
        row_anchor[i] = anchor;
        row_bias[i] = bias;
        row_changed[i] = changed;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — statistical_trailing_stop
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/statistical_trailing_stop.rs:934 compute_row,
 *   driving StatisticalTrailingStopState::update (:595) ->
 *   update_finite (:611), with MonoDeque (:374), RingHistory (:437) and
 *   RollingStats::push (:510).
 *
 *   The all_finite fast path (:965) calls update_finite DIRECTLY; the checked
 *   path calls update, which only adds the non-finite guard in front of it.
 *   Reproducing the checked form reproduces both.
 *
 * Column: output_id "value" resolves to out.level — cpu_batch.rs:4881 accepts
 *   "level"/"value". anchor, bias and changed are separate output ids; anchor
 *   and bias are still CARRIED here because the level recurrence reads bias
 *   and the trigger writes anchor.
 *
 * PERIOD-INVARIANT: compute_statistical_trailing_stop_batch reads data_length
 *   (10), normalization_length (100) and base_level ("level2" -> index 2) and
 *   NEVER period (cpu_batch.rs:4855-4864).
 *
 * FIRST-VALID IGNORED: update RESETS every accumulator on a non-finite bar
 *   (:602-605) and compute_row walks EVERY bar from index 0.
 *
 * Input: high / low / close — extract_ohlc_input (cpu_batch.rs:4847) —
 *   F64InputKind::Hlc.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Two MONOTONE DEQUES over the
 *   data_length window, a (data_length + 2)-deep close ring for the lagged
 *   close, a 100-deep (sum, sumsq) rolling-stat ring, and a latched
 *   bias/level/extreme trio all carry across bars.
 *
 * ARITHMETIC taken verbatim:
 *   * the true range is (hh - ll).max(|hh - prev_close|).max(|ll - prev_close|)
 *     (:633-635) — f64::max, hence fmax: it returns the non-NaN operand.
 *   * floor_positive (:80) replaces a non-positive range with
 *     f64::MIN_POSITIVE = 2.2250738585072014e-308 BEFORE the log. That is the
 *     smallest normal double, NOT a tolerance and NOT an f32 constant; it is
 *     carried across exactly.
 *   * RollingStats keeps sum and sum_sq incrementally as sum += value - old
 *     (:528-529) — the DIFFERENCE first, one rounding, then the add — and
 *     forms mean = sum * inv_len and
 *     variance = (sum_sq * inv_len - mean * mean).max(0.0) (:534-535). Two
 *     multiplies by a precomputed reciprocal, not divides, and an f64::max
 *     clamp before the sqrt.
 *   * delta is (mean + base_level * stdev).exp() (:637) — exp of the sum,
 *     not a product of exponentials.
 *   * every level clamp is f64::min / f64::max (:645, :651-658, :676) —
 *     fmin/fmax throughout.
 *   * there is no epsilon: every guard is an exact comparison.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:4855-4864. Each sizes a per-thread ring or
 * deque, so the bounds belong to the COMPILED kernel. */
#define NEO_STS_DATA_LENGTH           10
#define NEO_STS_NORMALIZATION_LENGTH 100
#define NEO_STS_BASE_LEVEL_INDEX       2
/* MonoDeque::new / RingHistory::new are both given data_length + 2 (:566-568). */
#define NEO_STS_DEQUE_CAP  (NEO_STS_DATA_LENGTH + 2)
#define NEO_STS_HIST_CAP   (NEO_STS_DATA_LENGTH + 2)
/* f64::MIN_POSITIVE — the smallest NORMAL double. */
#define NEO_STS_MIN_POSITIVE 2.2250738585072014e-308

#define NEO_STS_BIAS_BEARISH 0
#define NEO_STS_BIAS_BULLISH 1

extern "C" __global__
void statistical_trailing_stop_neo_batch_f64(const double* __restrict__ high,
                                             const double* __restrict__ low,
                                             const double* __restrict__ close,
                                             int n,
                                             const int* __restrict__ periods,
                                             int n_combos,
                                             int first_valid,
                                             double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* the mid-series reset reproduces it — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int data_length = NEO_STS_DATA_LENGTH;
    const int norm_length = NEO_STS_NORMALIZATION_LENGTH;
    const double base_lvl = (double)NEO_STS_BASE_LEVEL_INDEX;

    /* validate_periods refuses a window longer than the series. */
    if (data_length > n || norm_length > n) return;

    /* Two monotone deques, stored as circular arrays of at most
     * data_length + 2 live entries. */
    double mx_v[NEO_STS_DEQUE_CAP]; int mx_i[NEO_STS_DEQUE_CAP];
    double mn_v[NEO_STS_DEQUE_CAP]; int mn_i[NEO_STS_DEQUE_CAP];
    int mx_lo = 0, mx_len = 0, mn_lo = 0, mn_len = 0;

    double hist[NEO_STS_HIST_CAP];
    for (int k = 0; k < NEO_STS_HIST_CAP; ++k) hist[k] = 0.0;
    int hist_head = 0, hist_count = 0;

    double stat_ring[NEO_STS_NORMALIZATION_LENGTH];
    for (int k = 0; k < norm_length; ++k) stat_ring[k] = 0.0;
    int stat_head = 0, stat_count = 0;
    double stat_sum = 0.0, stat_sum_sq = 0.0;
    const double inv_len = 1.0 / (double)norm_length;

    int    valid_run = 0;
    int    bias      = NEO_STS_BIAS_BEARISH;
    double level     = NEO_F64_NAN;
    double extreme   = NEO_F64_NAN;
    double anchor    = NEO_F64_NAN;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            mx_lo = 0; mx_len = 0; mn_lo = 0; mn_len = 0;
            hist_head = 0; hist_count = 0;
            stat_head = 0; stat_count = 0; stat_sum = 0.0; stat_sum_sq = 0.0;
            valid_run = 0; bias = NEO_STS_BIAS_BEARISH;
            level = NEO_F64_NAN; extreme = NEO_F64_NAN; anchor = NEO_F64_NAN;
            o[i] = NEO_F64_NAN;
            continue;
        }

        ++valid_run;

        /* MonoDeque::push — descending for the highs, ascending for the lows. */
        while (mx_len > 0 && mx_v[(mx_lo + mx_len - 1) % NEO_STS_DEQUE_CAP] <= h) --mx_len;
        mx_v[(mx_lo + mx_len) % NEO_STS_DEQUE_CAP] = h;
        mx_i[(mx_lo + mx_len) % NEO_STS_DEQUE_CAP] = i;
        ++mx_len;

        while (mn_len > 0 && mn_v[(mn_lo + mn_len - 1) % NEO_STS_DEQUE_CAP] >= l) --mn_len;
        mn_v[(mn_lo + mn_len) % NEO_STS_DEQUE_CAP] = l;
        mn_i[(mn_lo + mn_len) % NEO_STS_DEQUE_CAP] = i;
        ++mn_len;

        const int span = (valid_run < data_length) ? valid_run : data_length;
        const int window_start = i + 1 - span;
        while (mx_len > 0 && mx_i[mx_lo] < window_start) { mx_lo = (mx_lo + 1) % NEO_STS_DEQUE_CAP; --mx_len; }
        while (mn_len > 0 && mn_i[mn_lo] < window_start) { mn_lo = (mn_lo + 1) % NEO_STS_DEQUE_CAP; --mn_len; }

        /* RingHistory::push (:460) */
        hist[hist_head] = c;
        ++hist_head;
        if (hist_head == NEO_STS_HIST_CAP) hist_head = 0;
        if (hist_count < NEO_STS_HIST_CAP) ++hist_count;

        if (valid_run < data_length + 2) { o[i] = NEO_F64_NAN; continue; }

        const double previous_close = hist[hist_head];   /* oldest() (:472) */
        const double highest = mx_v[mx_lo];
        const double lowest  = mn_v[mn_lo];
        const double tr = fmax(fmax(highest - lowest, fabs(highest - previous_close)),
                               fabs(lowest - previous_close));

        /* RollingStats::push (:510) over ln(floor_positive(tr)) */
        const double value    = log((tr > 0.0) ? tr : NEO_STS_MIN_POSITIVE);
        const double value_sq = value * value;
        if (stat_count < norm_length) {
            stat_ring[stat_head] = value;
            ++stat_head; if (stat_head == norm_length) stat_head = 0;
            ++stat_count;
            stat_sum    += value;
            stat_sum_sq += value_sq;
        } else {
            const double old = stat_ring[stat_head];
            stat_ring[stat_head] = value;
            ++stat_head; if (stat_head == norm_length) stat_head = 0;
            stat_sum    += value - old;
            stat_sum_sq += value_sq - old * old;
        }
        if (stat_count < norm_length) { o[i] = NEO_F64_NAN; continue; }

        const double mean     = stat_sum * inv_len;
        const double variance = fmax(stat_sum_sq * inv_len - mean * mean, 0.0);
        const double stdev    = sqrt(variance);

        const double delta = exp(mean + base_lvl * stdev);

        const double current_hlc3 = (h + l + c) / 3.0;
        if (!isfinite(level)) {
            level = (bias == NEO_STS_BIAS_BEARISH)
                ? (current_hlc3 + delta)
                : fmax(current_hlc3 - delta, 0.0);
        }

        if (bias == NEO_STS_BIAS_BEARISH) {
            if (isfinite(extreme)) extreme = fmin(extreme, l);
            level = fmin(level, current_hlc3 + delta);
        } else {
            if (isfinite(extreme)) extreme = fmax(extreme, h);
            level = fmax(level, fmax(current_hlc3 - delta, 0.0));
        }

        const bool triggered = (bias == NEO_STS_BIAS_BEARISH && c >= level) ||
                               (bias == NEO_STS_BIAS_BULLISH && c <= level);

        if (triggered) {
            anchor = c;
            bias = (bias == NEO_STS_BIAS_BEARISH) ? NEO_STS_BIAS_BULLISH : NEO_STS_BIAS_BEARISH;
            level = (bias == NEO_STS_BIAS_BEARISH)
                ? (current_hlc3 + delta)
                : fmax(current_hlc3 - delta, 0.0);
            extreme = (bias == NEO_STS_BIAS_BEARISH) ? l : h;
        }

        o[i] = level;
    }
}
