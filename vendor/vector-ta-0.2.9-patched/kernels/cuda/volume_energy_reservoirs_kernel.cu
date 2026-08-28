#include <cmath>
#include <cstddef>

namespace {

constexpr int VOLUME_STDEV_LENGTH = 100;
constexpr double MOMENTUM_EMA_ALPHA = 2.0 / 6.0;
constexpr double RESERVOIR_CAP = 10.0;
constexpr double RESERVOIR_SQUEEZE_THRESHOLD = 5.0;
constexpr double STABILITY_THRESHOLD = 0.2;
constexpr double FLOAT_TOL = 1.0e-12;

__device__ inline bool finite_ohlcv(double high, double low, double close, double volume) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(volume);
}

struct ExtremumDeque {
    int* indices;
    double* values;
    int head;
    int tail;
    int cap;

    __device__ void init(int* indices_ptr, double* values_ptr, int cap_value) {
        indices = indices_ptr;
        values = values_ptr;
        cap = cap_value;
        reset();
    }

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
            indices[i] = indices[head + i];
            values[i] = values[head + i];
        }
        head = 0;
        tail = size;
    }

    __device__ void ensure_capacity() {
        if (tail >= cap && head > 0) {
            compact();
        }
    }

    __device__ void normalize_if_empty() {
        if (head == tail) {
            head = 0;
            tail = 0;
        }
    }

    __device__ double front_value(double fallback) const {
        return tail > head ? values[head] : fallback;
    }

    __device__ void push_high(int idx, double value, int length) {
        while (tail > head && values[tail - 1] <= value) {
            tail -= 1;
        }
        ensure_capacity();
        if (tail < cap) {
            indices[tail] = idx;
            values[tail] = value;
            tail += 1;
        }
        while (tail > head && indices[head] + length <= idx) {
            head += 1;
        }
        normalize_if_empty();
    }

    __device__ void push_low(int idx, double value, int length) {
        while (tail > head && values[tail - 1] >= value) {
            tail -= 1;
        }
        ensure_capacity();
        if (tail < cap) {
            indices[tail] = idx;
            values[tail] = value;
            tail += 1;
        }
        while (tail > head && indices[head] + length <= idx) {
            head += 1;
        }
        normalize_if_empty();
    }
};

}

extern "C" __global__ void volume_energy_reservoirs_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ sensitivities,
    int rows,
    int window_cap,
    int* __restrict__ high_idx_scratch,
    double* __restrict__ high_val_scratch,
    int* __restrict__ low_idx_scratch,
    double* __restrict__ low_val_scratch,
    double* __restrict__ out_momentum,
    double* __restrict__ out_reservoir,
    double* __restrict__ out_squeeze_active,
    double* __restrict__ out_squeeze_start,
    double* __restrict__ out_range_high,
    double* __restrict__ out_range_low
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const double sensitivity = sensitivities[row];

    double* row_momentum = out_momentum + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_reservoir = out_reservoir + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_squeeze_active =
        out_squeeze_active + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_squeeze_start =
        out_squeeze_start + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_high =
        out_range_high + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_low = out_range_low + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_momentum[i] = NAN;
        row_reservoir[i] = NAN;
        row_squeeze_active[i] = NAN;
        row_squeeze_start[i] = NAN;
        row_range_high[i] = NAN;
        row_range_low[i] = NAN;
    }

    if (length < 5 || !isfinite(sensitivity) || sensitivity < 0.5) {
        return;
    }

    ExtremumDeque high_window;
    ExtremumDeque low_window;
    high_window.init(
        high_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        high_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        window_cap
    );
    low_window.init(
        low_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        low_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        window_cap
    );

    double volume_ring[VOLUME_STDEV_LENGTH];
    for (int i = 0; i < VOLUME_STDEV_LENGTH; ++i) {
        volume_ring[i] = 0.0;
    }

    int segment_index = 0;
    int volume_head = 0;
    int volume_count = 0;
    double volume_sum = 0.0;
    double volume_sum_sq = 0.0;
    double reservoir = 0.0;
    double ema = 0.0;
    bool ema_ready = false;
    bool prev_squeeze_active = false;
    double current_high = NAN;
    double current_low = NAN;
    bool has_range = false;
    bool is_extending = false;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!finite_ohlcv(h, l, c, v)) {
            segment_index = 0;
            volume_head = 0;
            volume_count = 0;
            volume_sum = 0.0;
            volume_sum_sq = 0.0;
            high_window.reset();
            low_window.reset();
            reservoir = 0.0;
            ema = 0.0;
            ema_ready = false;
            prev_squeeze_active = false;
            current_high = NAN;
            current_low = NAN;
            has_range = false;
            is_extending = false;
            continue;
        }

        const int idx = segment_index;
        segment_index += 1;

        if (volume_count == VOLUME_STDEV_LENGTH) {
            const double old = volume_ring[volume_head];
            volume_sum -= old;
            volume_sum_sq -= old * old;
        } else {
            volume_count += 1;
        }
        volume_ring[volume_head] = v;
        volume_head = (volume_head + 1) % VOLUME_STDEV_LENGTH;
        volume_sum += v;
        volume_sum_sq += v * v;

        high_window.push_high(idx, h, length);
        low_window.push_low(idx, l, length);

        const double hi = high_window.front_value(h);
        const double lo = low_window.front_value(l);
        const double mid_price = 0.5 * (hi + lo);
        const double price_range = hi - lo;
        const double hl2 = 0.5 * (h + l);
        const double price_rel =
            fabs(price_range) <= FLOAT_TOL ? 0.0 : (hl2 - mid_price) / price_range;

        double norm_vol = 0.0;
        if (volume_count >= VOLUME_STDEV_LENGTH) {
            const double mean = volume_sum / static_cast<double>(VOLUME_STDEV_LENGTH);
            const double variance =
                fmax(volume_sum_sq / static_cast<double>(VOLUME_STDEV_LENGTH) - mean * mean, 0.0);
            const double stdev = sqrt(variance);
            norm_vol = fabs(stdev) <= FLOAT_TOL ? 1.0 : (v / stdev);
        }

        if (norm_vol < 1.0 && fabs(price_rel) < STABILITY_THRESHOLD) {
            reservoir += 0.5;
        } else if (norm_vol > sensitivity) {
            reservoir *= 0.7;
        } else {
            reservoir = fmax(reservoir - 0.1, 0.0);
        }
        reservoir = fmin(reservoir, RESERVOIR_CAP);

        const double momentum = price_rel * norm_vol * 20.0;
        if (!ema_ready) {
            ema = momentum;
            ema_ready = true;
        } else {
            ema += MOMENTUM_EMA_ALPHA * (momentum - ema);
        }

        const bool squeeze_active = reservoir > RESERVOIR_SQUEEZE_THRESHOLD;
        const bool squeeze_start = squeeze_active && !prev_squeeze_active;
        const bool squeeze_end = !squeeze_active && prev_squeeze_active;

        if (squeeze_start) {
            current_high = h;
            current_low = l;
            has_range = true;
            is_extending = false;
        }

        if (squeeze_active && has_range) {
            current_high = fmax(current_high, h);
            current_low = fmin(current_low, l);
        }

        bool range_visible = squeeze_active || is_extending;
        if (squeeze_end && has_range) {
            is_extending = true;
            range_visible = true;
        }
        if (is_extending && has_range) {
            range_visible = true;
            if (c > current_high || c < current_low) {
                is_extending = false;
            }
        }

        prev_squeeze_active = squeeze_active;

        row_momentum[i] = ema;
        row_reservoir[i] = reservoir;
        row_squeeze_active[i] = squeeze_active ? 1.0 : 0.0;
        row_squeeze_start[i] = squeeze_start ? 1.0 : 0.0;
        row_range_high[i] = range_visible && has_range ? current_high : NAN;
        row_range_low[i] = range_visible && has_range ? current_low : NAN;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/volume_energy_reservoirs.rs
 *   `ReservoirCoreState::update` (:389-483), reached through
 *   `volume_energy_reservoirs_selected_row_from_slices` (:641-686) and
 *   `..._output_into_slice` (:804).
 *   Batch dispatcher: cpu_batch.rs:8953 -- output "value" is an ALIAS OF
 *   "momentum" (:8961), and `momentum` is the EMA of the raw momentum, i.e.
 *   `self.ema` (:468), NOT the raw `price_rel * norm_vol * 20.0`.
 *
 * WHY A SECOND ENTRY POINT: `volume_energy_reservoirs_batch_f64` (:101) takes
 *   19 parameters and emits six series. The lane launches
 *   (high, low, close, volume, n, periods, n_combos, first_valid, out).
 *
 * INPUT: (high, low, close, volume) -- extract_ohlcv_full_input discards open
 *   (`let (_, high, low, close, volume)`, cpu_batch.rs:8957) and the CPU
 *   reference never reads it -- F64InputKind::Hlcv.
 *
 * FIRST-VALID IGNORED: the row walker (:620) tests every bar itself and calls
 *   `state.reset()` (:627) on an invalid one, restarting the volume ring, both
 *   monotone windows, the reservoir, the EMA and the squeeze latch. A global
 *   first-valid index would be wrong after the first hole.
 *
 * PERIOD-INVARIANT: the CPU batch reads `length` and `sensitivity`
 *   (cpu_batch.rs:8984-8986) and never `period`. Both are pinned at the CPU
 *   defaults (20 and 1.5), so every row of a sweep is byte-identical.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. The reservoir level, the
 *   momentum EMA, the squeeze latch and the extending-range latch all carry
 *   across bars, and the two rolling extremes are monotone deques keyed by the
 *   SEGMENT index (`segment_index`, :397), which restarts at every reset.
 *
 * ARITHMETIC taken verbatim:
 *   * the reservoir is a three-way branch (:420-426): `+= 0.5`, `*= 0.7`, or
 *     `(x - 0.1).max(0.0)` -- f64::max, hence fmax -- then `min(10.0)`, again
 *     f64::min, hence fmin.
 *   * the momentum EMA step is `ema += alpha * (momentum - ema)` (:434) --
 *     a DIFFERENCE then a scaled add, TWO roundings. NOT `alpha*x + beta*ema`,
 *     which is three. The seed is the first momentum itself (:431).
 *   * `normalized_volume` (:501) forms the variance as
 *     `(sum_sq/n - mean*mean).max(0.0)` -- the population form with the mean
 *     subtracted AFTER the division, which is what the CPU writes.
 *   * `current_high.max(high)` / `current_low.min(low)` (:449-450) are
 *     f64::max / f64::min, hence fmax / fmin, which return the non-NaN
 *     operand.
 *
 * EPSILON: `FLOAT_TOL` is 1e-12 (:39) and is ALREADY an f64-sized tolerance in
 *   the CPU source -- carried across unchanged, not rescaled from an f32
 *   value. `RESERVOIR_CAP` 10.0, `RESERVOIR_SQUEEZE_THRESHOLD` 5.0 and
 *   `STABILITY_THRESHOLD` 0.2 are MODEL CONSTANTS, not tolerances, and are
 *   exactly representable.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* volume_energy_reservoirs.rs:30-39 */
#define NEO_VER_VOLUME_STDEV_LENGTH 100
#define NEO_VER_MOMENTUM_EMA_LENGTH 5
#define NEO_VER_RESERVOIR_CAP       10.0
#define NEO_VER_SQUEEZE_THRESHOLD   5.0
#define NEO_VER_STABILITY_THRESHOLD 0.2
#define NEO_VER_FLOAT_TOL           1e-12
/* cpu_batch.rs:8984-8986 */
#define NEO_VER_LENGTH      20
#define NEO_VER_SENSITIVITY 1.5
/* Monotone deque depth is `length`. TWO slots spare, not one: the CPU pushes
 * BEFORE it evicts the front (:517-531), so the transient occupancy is
 * `length + 1`, and a [head, tail) ring needs one more slot than its maximum
 * occupancy to tell full from empty. `length + 1` would alias the two. */
#define NEO_VER_MAX_WINDOW (NEO_VER_LENGTH + 2)

extern "C" __global__
void volume_energy_reservoirs_neo_batch_f64(const double* __restrict__ high,
                                            const double* __restrict__ low,
                                            const double* __restrict__ close,
                                            const double* __restrict__ volume,
                                            int n,
                                            const int* __restrict__ periods,
                                            int n_combos,
                                            int first_valid,
                                            double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* handled in place -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    length = NEO_VER_LENGTH;
    const double sensitivity = NEO_VER_SENSITIVITY;
    const double ema_alpha = 2.0 / ((double)NEO_VER_MOMENTUM_EMA_LENGTH + 1.0);
    const double vsl = (double)NEO_VER_VOLUME_STDEV_LENGTH;

    double vol_ring[NEO_VER_VOLUME_STDEV_LENGTH];
    int    hi_idx[NEO_VER_MAX_WINDOW], lo_idx[NEO_VER_MAX_WINDOW];
    double hi_val[NEO_VER_MAX_WINDOW], lo_val[NEO_VER_MAX_WINDOW];

    int    segment_index = 0;
    int    vol_head = 0, vol_count = 0;
    double vol_sum = 0.0, vol_sum_sq = 0.0;
    int    hi_head = 0, hi_tail = 0, lo_head = 0, lo_tail = 0; /* [head, tail) */
    double reservoir = 0.0;
    double ema = 0.0;
    bool   ema_ready = false;
    bool   prev_squeeze_active = false;
    double current_high = NEO_F64_NAN, current_low = NEO_F64_NAN;
    bool   has_range = false, is_extending = false;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i], v = volume[i];

        if (!(isfinite(h) && isfinite(l) && isfinite(c) && isfinite(v))) {
            /* ReservoirCoreState::reset (:370) */
            segment_index = 0;
            vol_head = 0; vol_count = 0; vol_sum = 0.0; vol_sum_sq = 0.0;
            for (int k = 0; k < NEO_VER_VOLUME_STDEV_LENGTH; ++k) vol_ring[k] = 0.0;
            hi_head = hi_tail = lo_head = lo_tail = 0;
            reservoir = 0.0;
            ema = 0.0; ema_ready = false;
            prev_squeeze_active = false;
            current_high = NEO_F64_NAN; current_low = NEO_F64_NAN;
            has_range = false; is_extending = false;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const int idx = segment_index;
        segment_index += 1;

        /* push_volume (:486) -- the ring is OVERWRITTEN in place, and the old
         * value is subtracted only once the ring is full. */
        if (vol_count == NEO_VER_VOLUME_STDEV_LENGTH) {
            const double old = vol_ring[vol_head];
            vol_sum -= old;
            vol_sum_sq -= old * old;
        } else {
            vol_count += 1;
        }
        vol_ring[vol_head] = v;
        vol_head += 1; if (vol_head == NEO_VER_VOLUME_STDEV_LENGTH) vol_head = 0;
        vol_sum += v;
        vol_sum_sq += v * v;

        /* push_high (:516) -- pop the tail while it is <= value, push, then
         * evict the front by index age. */
        while (hi_tail != hi_head && hi_val[(hi_tail == 0 ? NEO_VER_MAX_WINDOW : hi_tail) - 1] <= h) {
            hi_tail = (hi_tail == 0) ? (NEO_VER_MAX_WINDOW - 1) : (hi_tail - 1);
        }
        hi_idx[hi_tail] = idx; hi_val[hi_tail] = h;
        hi_tail += 1; if (hi_tail == NEO_VER_MAX_WINDOW) hi_tail = 0;
        while (hi_tail != hi_head && hi_idx[hi_head] + length <= idx) {
            hi_head += 1; if (hi_head == NEO_VER_MAX_WINDOW) hi_head = 0;
        }

        /* push_low (:535) */
        while (lo_tail != lo_head && lo_val[(lo_tail == 0 ? NEO_VER_MAX_WINDOW : lo_tail) - 1] >= l) {
            lo_tail = (lo_tail == 0) ? (NEO_VER_MAX_WINDOW - 1) : (lo_tail - 1);
        }
        lo_idx[lo_tail] = idx; lo_val[lo_tail] = l;
        lo_tail += 1; if (lo_tail == NEO_VER_MAX_WINDOW) lo_tail = 0;
        while (lo_tail != lo_head && lo_idx[lo_head] + length <= idx) {
            lo_head += 1; if (lo_head == NEO_VER_MAX_WINDOW) lo_head = 0;
        }

        const double hi = (hi_tail != hi_head) ? hi_val[hi_head] : h;
        const double lo = (lo_tail != lo_head) ? lo_val[lo_head] : l;

        const double mid_price = 0.5 * (hi + lo);
        const double price_range = hi - lo;
        const double hl2 = 0.5 * (h + l);
        const double price_rel = (fabs(price_range) <= NEO_VER_FLOAT_TOL)
                               ? 0.0
                               : ((hl2 - mid_price) / price_range);

        /* normalized_volume (:501) */
        double norm_vol;
        if (vol_count < NEO_VER_VOLUME_STDEV_LENGTH) {
            norm_vol = 0.0;
        } else {
            const double mean = vol_sum / vsl;
            const double variance = fmax(vol_sum_sq / vsl - mean * mean, 0.0);
            const double stdev = sqrt(variance);
            norm_vol = (fabs(stdev) <= NEO_VER_FLOAT_TOL) ? 1.0 : (v / stdev);
        }

        if (norm_vol < 1.0 && fabs(price_rel) < NEO_VER_STABILITY_THRESHOLD) {
            reservoir += 0.5;
        } else if (norm_vol > sensitivity) {
            reservoir *= 0.7;
        } else {
            reservoir = fmax(reservoir - 0.1, 0.0);
        }
        reservoir = fmin(reservoir, NEO_VER_RESERVOIR_CAP);

        const double momentum = price_rel * norm_vol * 20.0;
        if (!ema_ready) { ema = momentum; ema_ready = true; }
        else            { ema += ema_alpha * (momentum - ema); }

        const bool squeeze_active = reservoir > NEO_VER_SQUEEZE_THRESHOLD;
        const bool squeeze_start  = squeeze_active && !prev_squeeze_active;

        if (squeeze_start) {
            current_high = h; current_low = l;
            has_range = true; is_extending = false;
        }
        if (squeeze_active && has_range) {
            current_high = fmax(current_high, h);
            current_low  = fmin(current_low, l);
        }
        /* squeeze_end and the range-visible latch are computed by the CPU even
         * though the `momentum` column never reads them -- they mutate
         * `is_extending`, which survives into later bars, so they are kept. */
        const bool squeeze_end = !squeeze_active && prev_squeeze_active;
        if (squeeze_end && has_range) {
            is_extending = true;
        }
        if (is_extending && has_range) {
            if (c > current_high || c < current_low) is_extending = false;
        }

        prev_squeeze_active = squeeze_active;

        o[i] = ema;
    }
}
