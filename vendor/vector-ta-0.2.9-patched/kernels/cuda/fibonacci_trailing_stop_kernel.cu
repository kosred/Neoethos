#include <cmath>
#include <cstddef>

namespace {
constexpr int TRIGGER_CLOSE = 0;
constexpr int TRIGGER_WICK = 1;

struct PivotPoint {
    double price;
    int dir;
};

__device__ inline bool confirmed_pivot_high_at(
    const double* data,
    int len,
    int idx,
    int left,
    int right,
    double* out
) {
    if (idx < right) {
        return false;
    }
    const int center = idx - right;
    if (center < left || center + right >= len) {
        return false;
    }

    const double candidate = data[center];
    if (!isfinite(candidate)) {
        return false;
    }

    for (int j = center - left; j <= center + right; ++j) {
        const double value = data[j];
        if (!isfinite(value) || value > candidate) {
            return false;
        }
    }

    *out = candidate;
    return true;
}

__device__ inline bool confirmed_pivot_low_at(
    const double* data,
    int len,
    int idx,
    int left,
    int right,
    double* out
) {
    if (idx < right) {
        return false;
    }
    const int center = idx - right;
    if (center < left || center + right >= len) {
        return false;
    }

    const double candidate = data[center];
    if (!isfinite(candidate)) {
        return false;
    }

    for (int j = center - left; j <= center + right; ++j) {
        const double value = data[j];
        if (!isfinite(value) || value < candidate) {
            return false;
        }
    }

    *out = candidate;
    return true;
}

__device__ inline void apply_pivot_high(
    PivotPoint* pivots,
    int* pivot_count,
    double value
) {
    if (*pivot_count > 0) {
        PivotPoint& first = pivots[0];
        if (first.dir > 0 && value > first.price) {
            first.price = value;
            return;
        }
        if (first.dir < 0 && value > first.price) {
            const int old_count = *pivot_count;
            const int new_count = old_count >= 3 ? 3 : (old_count + 1);
            for (int j = new_count - 1; j > 0; --j) {
                if (j - 1 < old_count) {
                    pivots[j] = pivots[j - 1];
                }
            }
            pivots[0] = {value, 1};
            *pivot_count = new_count;
            return;
        }
        return;
    }

    pivots[0] = {value, 1};
    *pivot_count = 1;
}

__device__ inline void apply_pivot_low(
    PivotPoint* pivots,
    int* pivot_count,
    double value
) {
    if (*pivot_count > 0) {
        PivotPoint& first = pivots[0];
        if (first.dir < 0 && value < first.price) {
            first.price = value;
            return;
        }
        if (first.dir > 0 && value < first.price) {
            const int old_count = *pivot_count;
            const int new_count = old_count >= 3 ? 3 : (old_count + 1);
            for (int j = new_count - 1; j > 0; --j) {
                if (j - 1 < old_count) {
                    pivots[j] = pivots[j - 1];
                }
            }
            pivots[0] = {value, -1};
            *pivot_count = new_count;
            return;
        }
        return;
    }

    pivots[0] = {value, -1};
    *pivot_count = 1;
}
}

extern "C" __global__ void fibonacci_trailing_stop_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* left_bars,
    const int* right_bars,
    const double* levels,
    const int* trigger_modes,
    int rows,
    double* out_trailing_stop,
    double* out_long_stop,
    double* out_short_stop,
    double* out_direction
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int left = left_bars[row];
    const int right = right_bars[row];
    const double level = levels[row];
    const int trigger = trigger_modes[row];

    double* row_trailing_stop =
        out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_stop = out_long_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_stop = out_short_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction = out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_trailing_stop[i] = NAN;
        row_long_stop[i] = NAN;
        row_short_stop[i] = NAN;
        row_direction[i] = NAN;
    }

    if (left <= 0 || right <= 0 || !isfinite(level)
        || (trigger != TRIGGER_CLOSE && trigger != TRIGGER_WICK)
        || left + right + 1 > len) {
        return;
    }

    bool state_active = false;
    int dir = 0;
    double st = NAN;
    double max_level = NAN;
    double min_level = NAN;
    PivotPoint pivots[3];
    int pivot_count = 0;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            state_active = false;
            dir = 0;
            st = NAN;
            max_level = NAN;
            min_level = NAN;
            pivot_count = 0;
            continue;
        }

        double ph_value = NAN;
        double pl_value = NAN;
        const bool has_ph = confirmed_pivot_high_at(high, len, i, left, right, &ph_value);
        const bool has_pl = confirmed_pivot_low_at(low, len, i, left, right, &pl_value);

        if (!state_active) {
            state_active = true;
            dir = 0;
            st = c;
            max_level = h;
            min_level = l;
            pivot_count = 0;
        }

        if (has_ph) {
            apply_pivot_high(pivots, &pivot_count, ph_value);
        }
        if (has_pl) {
            apply_pivot_low(pivots, &pivot_count, pl_value);
        }

        if (pivot_count >= 2) {
            const double p0 = pivots[0].price;
            const double p1 = pivots[1].price;
            double max_value = p0 > p1 ? p0 : p1;
            double min_value = p0 < p1 ? p0 : p1;
            if (pivot_count == 2) {
                st = 0.5 * (max_value + min_value);
            }
            const double dif = max_value - min_value;
            max_value += dif * level;
            min_value -= dif * level;
            max_level = max_value;
            min_level = min_value;
        }

        const double price =
            trigger == TRIGGER_CLOSE ? c : ((dir < 1) ? h : l);

        if (dir < 1) {
            if (price > st) {
                st = min_level;
                dir = 1;
            } else {
                st = fmin(st, max_level);
            }
        }

        if (dir > -1) {
            if (price < st) {
                st = max_level;
                dir = -1;
            } else {
                st = fmax(st, min_level);
            }
        }

        row_trailing_stop[i] = st;
        row_long_stop[i] = dir == 1 ? st : NAN;
        row_short_stop[i] = dir == -1 ? st : NAN;
        row_direction[i] = static_cast<double>(dir);
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/fibonacci_trailing_stop.rs:773
// (fibonacci_trailing_stop_with_kernel). The column this emits is
// trailing_stop, which is what output_id == "value" resolves to
// (dispatch/cpu_batch.rs:8828-8830).
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- a ratchet:
// the stop st is carried, tightened with fmin/fmax against the Fibonacci
// extension of the last two confirmed pivots, and FLIPPED by a direction state
// machine. Every bar reads the previous bar's stop and direction, and the CPU
// clears the whole state on a non-finite bar.
//
// PERIOD-INVARIANT. compute_fibonacci_trailing_stop_batch
// (cpu_batch.rs:8807-8810) reads left_bars, right_bars, level and trigger and
// NEVER period, so five swept periods give five identical CPU columns and this
// kernel emits five identical rows. All four CPU defaults are pinned below,
// including trigger "close".
//
// THE PIVOT STORE IS THREE ENTRIES, NOT A GROWING LIST: the CPU keeps at most
// the three most recent alternating pivots and only ever reads the first two,
// so the fixed array below is the CPU's own bound and not a truncation.
//
// PIVOTS ARE CONFIRMED, NOT LOOKAHEAD: confirmed_pivot_*_at inspects the
// window centred at i - right, so the kernel reads only bars at or before i --
// the same lag the CPU carries. This is worth stating because the helper takes
// the whole series and could be misread as peeking forward.
//
// FIRST VALID IS NOT READ: the CPU emits from bar 0 and re-arms its state at
// the first finite bar after any gap, so there is no warmup index. The lane
// row declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double fmin/fmax (which are what the CPU's
// f64::min / f64::max are, so a NaN level cannot survive), no f32-suffixed math
// function, no fast-math intrinsic, no epsilon.
// ---------------------------------------------------------------------------

#define NEO_FTS_LEFT_BARS 20
#define NEO_FTS_RIGHT_BARS 1
#define NEO_FTS_LEVEL (-0.382)
#define NEO_FTS_TRIGGER TRIGGER_CLOSE

extern "C" __global__ void fibonacci_trailing_stop_neo_batch_f64(
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

    const int left = NEO_FTS_LEFT_BARS;
    const int right = NEO_FTS_RIGHT_BARS;
    const double level = NEO_FTS_LEVEL;
    const int trigger = NEO_FTS_TRIGGER;

    if (left <= 0 || right <= 0 || !isfinite(level) ||
        (trigger != TRIGGER_CLOSE && trigger != TRIGGER_WICK) || left + right + 1 > n) {
        return;
    }

    bool state_active = false;
    int dir = 0;
    double st = NAN;
    double max_level = NAN;
    double min_level = NAN;
    PivotPoint pivots[3];
    int pivot_count = 0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            state_active = false;
            dir = 0;
            st = NAN;
            max_level = NAN;
            min_level = NAN;
            pivot_count = 0;
            continue;
        }

        double ph_value = NAN;
        double pl_value = NAN;
        const bool has_ph = confirmed_pivot_high_at(high, n, i, left, right, &ph_value);
        const bool has_pl = confirmed_pivot_low_at(low, n, i, left, right, &pl_value);

        if (!state_active) {
            state_active = true;
            dir = 0;
            st = c;
            max_level = h;
            min_level = l;
            pivot_count = 0;
        }

        if (has_ph) {
            apply_pivot_high(pivots, &pivot_count, ph_value);
        }
        if (has_pl) {
            apply_pivot_low(pivots, &pivot_count, pl_value);
        }

        if (pivot_count >= 2) {
            const double p0 = pivots[0].price;
            const double p1 = pivots[1].price;
            double max_value = p0 > p1 ? p0 : p1;
            double min_value = p0 < p1 ? p0 : p1;
            if (pivot_count == 2) {
                st = 0.5 * (max_value + min_value);
            }
            const double dif = max_value - min_value;
            max_value += dif * level;
            min_value -= dif * level;
            max_level = max_value;
            min_level = min_value;
        }

        const double price = trigger == TRIGGER_CLOSE ? c : ((dir < 1) ? h : l);

        if (dir < 1) {
            if (price > st) {
                st = min_level;
                dir = 1;
            } else {
                st = fmin(st, max_level);
            }
        }

        if (dir > -1) {
            if (price < st) {
                st = max_level;
                dir = -1;
            } else {
                st = fmax(st, min_level);
            }
        }

        row[i] = st;
    }
}
