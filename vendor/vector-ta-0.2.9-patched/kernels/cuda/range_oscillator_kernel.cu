#include <cmath>
#include <cstddef>

namespace {
constexpr int ATR_FALLBACK_PERIOD = 200;
constexpr int ATR_PRIMARY_PERIOD = 2000;
constexpr double ZERO_EPS = 1e-12;

struct AtrState {
    int count;
    double sum;
    double value;
    bool seeded;
    double prev_close;
    bool have_prev;

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        seeded = false;
        prev_close = NAN;
        have_prev = false;
    }

    __device__ double update(int period, double high, double low, double close, bool* ready) {
        const double tr = have_prev
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        have_prev = true;

        if (seeded) {
            value = (value * static_cast<double>(period - 1) + tr) / static_cast<double>(period);
            *ready = true;
            return value;
        }

        count += 1;
        sum += tr;
        if (count == period) {
            value = sum / static_cast<double>(period);
            seeded = true;
            *ready = true;
            return value;
        }

        *ready = false;
        return NAN;
    }
};

__device__ inline void push_close(double* ring, int cap, int* head, int* count, double value) {
    if (*count < cap) {
        ring[(*head + *count) % cap] = value;
        *count += 1;
        return;
    }
    ring[*head] = value;
    *head += 1;
    if (*head == cap) {
        *head = 0;
    }
}

__device__ inline int back_index(int cap, int head, int count, int offset) {
    int idx = head + count - 1 - offset;
    idx %= cap;
    if (idx < 0) {
        idx += cap;
    }
    return idx;
}
}

extern "C" __global__ void range_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* lengths,
    const double* mults,
    int rows,
    int storage_cols,
    double* out_oscillator,
    double* out_ma,
    double* out_upper_band,
    double* out_lower_band,
    double* out_range_width,
    double* out_in_range,
    double* out_trend,
    double* out_break_up,
    double* out_break_down,
    double* close_storage
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_band =
        out_upper_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_band =
        out_lower_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_width =
        out_range_width + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_in_range = out_in_range + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_break_up = out_break_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_break_down =
        out_break_down + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = NAN;
        row_ma[i] = NAN;
        row_upper_band[i] = NAN;
        row_lower_band[i] = NAN;
        row_range_width[i] = NAN;
        row_in_range[i] = NAN;
        row_trend[i] = NAN;
        row_break_up[i] = NAN;
        row_break_down[i] = NAN;
    }

    if (length <= 0 || length + 1 > storage_cols || !isfinite(mult) || mult < 0.1) {
        return;
    }

    double* ring = close_storage + static_cast<size_t>(row) * static_cast<size_t>(storage_cols);
    int ring_head = 0;
    int ring_count = 0;
    double trend_state = 0.0;

    AtrState atr_fallback;
    AtrState atr_primary;
    atr_fallback.reset();
    atr_primary.reset();

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            atr_fallback.reset();
            atr_primary.reset();
            ring_head = 0;
            ring_count = 0;
            trend_state = 0.0;
            continue;
        }

        bool fallback_ready = false;
        bool primary_ready = false;
        const double atr200 = atr_fallback.update(ATR_FALLBACK_PERIOD, h, l, c, &fallback_ready);
        const double atr2000 = atr_primary.update(ATR_PRIMARY_PERIOD, h, l, c, &primary_ready);

        push_close(ring, storage_cols, &ring_head, &ring_count, c);

        const bool atr_ready = primary_ready || fallback_ready;
        if (!atr_ready || ring_count < length + 1) {
            continue;
        }

        const double atr_raw = primary_ready ? atr2000 : atr200;
        const double range_width = atr_raw * mult;

        double sum_weighted = 0.0;
        double sum_weights = 0.0;
        for (int j = 0; j < length; ++j) {
            const double curr = ring[back_index(storage_cols, ring_head, ring_count, j)];
            const double prev = ring[back_index(storage_cols, ring_head, ring_count, j + 1)];
            if (fabs(prev) <= ZERO_EPS) {
                continue;
            }
            const double weight = fabs(curr - prev) / prev;
            sum_weighted += curr * weight;
            sum_weights += weight;
        }
        if (fabs(sum_weights) <= ZERO_EPS) {
            continue;
        }

        const double ma = sum_weighted / sum_weights;
        double max_dist = 0.0;
        for (int j = 0; j < length; ++j) {
            const double value = ring[back_index(storage_cols, ring_head, ring_count, j)];
            const double dist = fabs(value - ma);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }

        if (c > ma) {
            trend_state = 1.0;
        } else if (c < ma) {
            trend_state = -1.0;
        }

        const double upper_band = ma + range_width;
        const double lower_band = ma - range_width;
        const double break_up = c > upper_band ? 1.0 : 0.0;
        const double break_down = c < lower_band ? 1.0 : 0.0;
        const double oscillator =
            fabs(range_width) <= ZERO_EPS ? NAN : (100.0 * (c - ma) / range_width);

        row_oscillator[i] = oscillator;
        row_ma[i] = ma;
        row_upper_band[i] = upper_band;
        row_lower_band[i] = lower_band;
        row_range_width[i] = range_width;
        row_in_range[i] = max_dist <= range_width ? 1.0 : 0.0;
        row_trend[i] = trend_state;
        row_break_up[i] = break_up;
        row_break_down[i] = break_down;
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                          range_oscillator
 * ---------------------------------------------------------------------------
 * CPU reference: `compute_into_slices`, src/indicators/range_oscillator.rs:974
 * (general arm, :1032-1110), with `compute_weighted_ma` (:537),
 * `compute_point` (:563) and `AtrState::update` (:276), reached through
 * `range_oscillator_with_kernel` (:333) and `prepare_input` (:489).
 *
 * `length` IS the swept parameter (cpu_batch.rs:16029, default 50); `mult` is
 * 2.0 (:16030). The lane emits the OSCILLATOR series, which is what the
 * dispatcher returns for `output_id` "value" as well as "oscillator" and "osc"
 * (:16044-16049).
 *
 * WHY THE EXISTING ENTRY POINT COULD NOT BE REUSED. `range_oscillator_batch_f64`
 * already in this file writes several output matrices and takes its parameters
 * as per-row arrays; the lane launches
 * (high, low, close, n, periods, n_combos, first_valid, out) and allocates one
 * matrix.
 *
 * THE CRATE HAS A `length == 50 && mult == 2.0` FAST PATH (:1016-1030) and the
 * sweep can hit it. It is not a second oracle: `compute_default_into_slices`
 * (:711) replaces the `VecDeque` with a fixed ring and visits the same closes
 * in the same `last - i` order, so it produces the same doubles as the general
 * arm. One transcription therefore serves both, and there is no special case
 * for period 50 here.
 *
 * SEQUENTIAL, one thread per column, three carried recurrences:
 *   * two ATR states, periods 200 and 2000 (:30-31), each a Wilder-style
 *     smoothing whose value depends on every previous bar;
 *   * `trend_state`, which is sticky across bars (:581-585);
 *   * the rolling window of the last `length + 1` VALID closes, which resets
 *     wholesale on a non-finite bar (:1041-1058).
 *
 * THE ATR RECURRENCE IS THREE ROUNDINGS, NOT ONE. The CPU writes
 * `(prev * (period as f64 - 1.0) + tr) / period as f64` (:287) -- a multiply,
 * an add and a divide. Folding it into `fma` or into `prev + (tr - prev)/period`
 * would be a different number at every bar and would walk forward through the
 * whole series. Copied literally.
 *
 * `hl.max(hc).max(lc)` (:283) is f64::max, which returns the NON-NaN operand;
 * it becomes `fmax(fmax(hl, hc), lc)`. An if-chain would let a NaN through and
 * poison the recurrence from that bar on.
 *
 * NO WINDOW ARRAY, so no `max_period` bound and NEVER-OOM by construction. The
 * deque holds the last `length + 1` valid closes and is cleared by any invalid
 * bar, so at every bar it emits at, the window is exactly
 * `close[t - length .. t]` and is read straight out of the resident series;
 * `valid_run` is the deque's length.
 *
 * ZERO_EPS is 1e-12 (:32) -- already an f64 guard in an f64 routine, NOT an f32
 * epsilon, and it must not be resized.
 *
 * FIRST-VALID IS `Ignored`, read off the CPU rather than assumed:
 * `range_oscillator_with_kernel` allocates with `alloc_with_nan_prefix(len, 0)`
 * (:335-343) -- prefix ZERO -- and `compute_into_slices` writes EVERY index
 * from 0. `prepared.first` is used only by the validation at :515-521, which
 * this kernel reproduces itself.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* range_oscillator.rs:29-32 and cpu_batch.rs:16030. */
#define NEO_RO_MULT                2.0
#define NEO_RO_ATR_FALLBACK_PERIOD 200
#define NEO_RO_ATR_PRIMARY_PERIOD  2000
#define NEO_RO_ZERO_EPS            1e-12

/* AtrState, range_oscillator.rs:246-301. `has_value` mirrors `Option<f64>`. */
struct NeoRoAtr {
    int    period;
    int    count;
    double sum;
    double value;
    bool   has_value;
    double prev_close;
    bool   has_prev_close;
};

__device__ __forceinline__
static void neo_ro_atr_init(NeoRoAtr* s, int period)
{
    s->period = period;
    s->count = 0;
    s->sum = 0.0;
    s->value = 0.0;
    s->has_value = false;
    s->prev_close = 0.0;
    s->has_prev_close = false;
}

__device__ __forceinline__
static void neo_ro_atr_reset(NeoRoAtr* s)
{
    s->count = 0;
    s->sum = 0.0;
    s->value = 0.0;
    s->has_value = false;
    s->prev_close = 0.0;
    s->has_prev_close = false;
}

/* Returns true and writes *outv when the CPU would return Some. */
__device__ __forceinline__
static bool neo_ro_atr_update(NeoRoAtr* s, double high, double low, double close,
                              double* outv)
{
    double tr;
    if (s->has_prev_close) {
        const double hl = high - low;
        const double hc = fabs(high - s->prev_close);
        const double lc = fabs(low - s->prev_close);
        tr = fmax(fmax(hl, hc), lc);          /* :281-284 */
    } else {
        tr = high - low;                      /* :285-286 */
    }
    s->prev_close = close;
    s->has_prev_close = true;

    if (s->has_value) {
        /* :287-289 -- multiply, add, divide. THREE roundings, deliberately. */
        const double next =
            (s->value * ((double)s->period - 1.0) + tr) / (double)s->period;
        s->value = next;
        *outv = next;
        return true;
    }

    s->count += 1;                            /* :292-293 */
    s->sum += tr;
    if (s->count == s->period) {
        const double seeded = s->sum / (double)s->period;   /* :295 */
        s->value = seeded;
        s->has_value = true;
        *outv = seeded;
        return true;
    }
    return false;
}

extern "C" __global__
void range_oscillator_neo_batch_f64(const double* __restrict__ high,
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
    (void)first_valid;   /* the CPU row starts at bar 0 -- see the header. */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = periods[combo];
    /* prepare_input, :505-510 -- note `length >= len` errors, not `>`. */
    if (length <= 0 || length >= n) return;

    /* :498-500 -- the first bar with all three finite; AllValuesNaN errors. */
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i])) { first = i; break; }
    }
    if (first < 0) return;

    /* :515-521 -- NotEnoughValidData leaves the row NaN. */
    {
        int valid = 0;
        for (int i = first; i < n; ++i) {
            if (isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i])) valid += 1;
        }
        const int needed = (length + 1 > NEO_RO_ATR_FALLBACK_PERIOD)
                         ? (length + 1) : NEO_RO_ATR_FALLBACK_PERIOD;
        if (valid < needed) return;
    }

    NeoRoAtr atr_fallback, atr_primary;
    neo_ro_atr_init(&atr_fallback, NEO_RO_ATR_FALLBACK_PERIOD);
    neo_ro_atr_init(&atr_primary,  NEO_RO_ATR_PRIMARY_PERIOD);

    int    valid_run   = 0;      /* the deque's length, capped at length + 1 */
    double trend_state = 0.0;    /* carried but not emitted by this column */

    for (int t = 0; t < n; ++t) {
        const double h = high[t], l = low[t], c = close[t];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            neo_ro_atr_reset(&atr_fallback);
            neo_ro_atr_reset(&atr_primary);
            valid_run   = 0;
            trend_state = 0.0;
            continue;                                  /* row already NaN */
        }

        double a200, a2000;
        const bool have200  = neo_ro_atr_update(&atr_fallback, h, l, c, &a200);
        const bool have2000 = neo_ro_atr_update(&atr_primary,  h, l, c, &a2000);

        if (valid_run < length + 1) valid_run += 1;    /* :1065-1068 */

        /* :1063 -- `atr2000.or(atr200)`. */
        double atr_raw;
        if (have2000)      atr_raw = a2000;
        else if (have200)  atr_raw = a200;
        else               continue;                   /* :1070-1084 -> NaN */

        if (valid_run < length + 1) continue;          /* :1085-1099 -> NaN */

        const double range_width = atr_raw * NEO_RO_MULT;   /* :1101 */

        /* compute_weighted_ma, :537-559 -- i ascending from the NEWEST bar. */
        double sum_weighted = 0.0;
        double sum_weights  = 0.0;
        for (int i = 0; i < length; ++i) {
            const double curr = close[t - i];
            const double prev = close[t - i - 1];
            if (fabs(prev) <= NEO_RO_ZERO_EPS) continue;
            const double delta = fabs(curr - prev);
            const double w = delta / prev;
            sum_weighted += curr * w;
            sum_weights  += w;
        }
        if (fabs(sum_weights) <= NEO_RO_ZERO_EPS) continue;  /* None -> NaN */
        const double ma = sum_weighted / sum_weights;

        /* :571-579 -- max_dist starts at 0.0 and only a STRICTLY greater
         * distance replaces it, so a NaN distance never wins. Reproduced with
         * the same strict comparison rather than with fmax. */
        double max_dist = 0.0;
        for (int i = 0; i < length; ++i) {
            const double dist = fabs(close[t - i] - ma);
            if (dist > max_dist) max_dist = dist;
        }
        (void)max_dist;   /* feeds the "in_range" column, not this one */

        if (c > ma)      trend_state = 1.0;            /* :581-585 */
        else if (c < ma) trend_state = -1.0;

        /* :591-595 -- the oscillator itself. */
        o[t] = (fabs(range_width) <= NEO_RO_ZERO_EPS)
             ? NEO_F64_NAN
             : (100.0 * (c - ma) / range_width);
    }

    (void)trend_state;
}
