#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_EMA = 0;
constexpr int MA_SMA = 1;
constexpr double NBDEV = 4.0;

__device__ inline int max_i(int a, int b) {
    return a > b ? a : b;
}

__device__ inline int min_i(int a, int b) {
    return a < b ? a : b;
}

__device__ inline int sqrt_period(int period) {
    const double root = floor(sqrt(static_cast<double>(period)));
    const int value = static_cast<int>(root);
    return value > 0 ? value : 1;
}

struct EmaState {
    int period;
    int count;
    double alpha;
    double beta;
    double mean;
    bool filled;

    __device__ void init(int period) {
        this->period = period;
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
        filled = false;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = filled ? mean : NAN;
            return filled;
        }
        count += 1;
        if (count == 1) {
            mean = input;
        } else if (count <= period) {
            mean += (input - mean) / static_cast<double>(count);
        } else {
            mean = alpha * input + beta * mean;
        }
        if (!filled && count >= period) {
            filled = true;
        }
        *out = filled ? mean : NAN;
        return filled;
    }
};

struct SmaState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            if (count == period) {
                *out = sum / static_cast<double>(period);
                return true;
            }
            *out = NAN;
            return false;
        }
        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum += input - old;
        *out = sum / static_cast<double>(period);
        return true;
    }
};

struct WmaState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;
    double wsum;
    double inv_norm;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        const double norm = static_cast<double>(period) * (static_cast<double>(period) + 1.0) * 0.5;
        inv_norm = 1.0 / norm;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        wsum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            wsum += static_cast<double>(count) * input;
            if (count == period) {
                *out = wsum * inv_norm;
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        const double prev_sum = sum;
        sum = prev_sum + input - old;
        wsum = static_cast<double>(period) * input + wsum - prev_sum;
        *out = wsum * inv_norm;
        return true;
    }
};

struct HmaState {
    WmaState wma_half;
    WmaState wma_full;
    WmaState wma_sqrt;

    __device__ void init(int period, double* half_storage, double* full_storage, double* sqrt_storage) {
        const int half = max_i(period / 2, 1);
        const int sqrt_len = sqrt_period(period);
        wma_half.init(half, half_storage);
        wma_full.init(period, full_storage);
        wma_sqrt.init(sqrt_len, sqrt_storage);
    }

    __device__ void reset() {
        wma_half.reset();
        wma_full.reset();
        wma_sqrt.reset();
    }

    __device__ bool update(double input, double* out) {
        double half_value = NAN;
        double full_value = NAN;
        const bool half_ready = wma_half.update(input, &half_value);
        const bool full_ready = wma_full.update(input, &full_value);
        if (half_ready && full_ready) {
            return wma_sqrt.update(2.0 * half_value - full_value, out);
        }
        *out = NAN;
        return false;
    }
};

struct StddevState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;
    double sumsq;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        sumsq = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            sumsq += input * input;
            if (count == period) {
                const double mean = sum / static_cast<double>(period);
                const double var = fmax(sumsq / static_cast<double>(period) - mean * mean, 0.0);
                *out = sqrt(var) * NBDEV;
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum += input - old;
        sumsq += input * input - old * old;
        const double mean = sum / static_cast<double>(period);
        const double var = fmax(sumsq / static_cast<double>(period) - mean * mean, 0.0);
        *out = sqrt(var) * NBDEV;
        return true;
    }
};

__device__ bool finite_window(
    const double* data,
    int start,
    int end
) {
    for (int i = start; i <= end; ++i) {
        if (!isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

__device__ bool truncated_ema_from_slice(
    const double* data,
    int end_idx,
    int history_len,
    double alpha,
    double beta,
    double* out
) {
    const int start_idx = end_idx - history_len + 1;
    if (start_idx < 0) {
        *out = NAN;
        return false;
    }
    double ema = data[start_idx];
    if (!isfinite(ema)) {
        *out = NAN;
        return false;
    }
    for (int idx = start_idx + 1; idx <= end_idx; ++idx) {
        const double value = data[idx];
        if (!isfinite(value)) {
            *out = NAN;
            return false;
        }
        ema = alpha * value + beta * ema;
    }
    *out = ema;
    return true;
}

__device__ bool probability_from_slice(
    const double* data,
    int end_idx,
    int ma_type,
    int slow_length,
    int fast_length,
    int resolution,
    int history_window_len,
    double lower,
    double upper,
    double direction,
    double* out
) {
    const int start_idx = end_idx - history_window_len + 1;
    if (start_idx < 0 || !finite_window(data, start_idx, end_idx)) {
        *out = NAN;
        return false;
    }

    const double step = (upper - lower) / static_cast<double>(resolution - 1);
    int hits = 0;

    if (ma_type == MA_EMA) {
        const double slow_alpha = 2.0 / (static_cast<double>(slow_length) + 1.0);
        const double slow_beta = 1.0 - slow_alpha;
        const double fast_alpha = 2.0 / (static_cast<double>(fast_length) + 1.0);
        const double fast_beta = 1.0 - fast_alpha;
        double slow_current = NAN;
        double fast_current = NAN;
        if (!truncated_ema_from_slice(data, end_idx, history_window_len, slow_alpha, slow_beta, &slow_current) ||
            !truncated_ema_from_slice(data, end_idx, history_window_len, fast_alpha, fast_beta, &fast_current)) {
            *out = NAN;
            return false;
        }
        for (int idx = 0; idx < resolution; ++idx) {
            const double price = lower + step * static_cast<double>(idx);
            const double slow_future = slow_alpha * price + slow_beta * slow_current;
            const double fast_future = fast_alpha * price + fast_beta * fast_current;
            const bool crossed = direction < 0.0 ? (slow_future > fast_future) : (slow_future <= fast_future);
            if (crossed) {
                hits += 1;
            }
        }
    } else {
        const int slow_needed = slow_length - 1;
        const int fast_needed = fast_length - 1;
        double slow_sum = 0.0;
        double fast_sum = 0.0;
        for (int idx = 0; idx < slow_needed; ++idx) {
            slow_sum += data[end_idx - idx];
        }
        for (int idx = 0; idx < fast_needed; ++idx) {
            fast_sum += data[end_idx - idx];
        }
        for (int idx = 0; idx < resolution; ++idx) {
            const double price = lower + step * static_cast<double>(idx);
            const double slow_future = (price + slow_sum) / static_cast<double>(slow_length);
            const double fast_future = (price + fast_sum) / static_cast<double>(fast_length);
            const bool crossed = direction < 0.0 ? (slow_future > fast_future) : (slow_future <= fast_future);
            if (crossed) {
                hits += 1;
            }
        }
    }

    *out = 100.0 * static_cast<double>(hits) / static_cast<double>(resolution);
    return true;
}
}

extern "C" __global__ void moving_average_cross_probability_batch_f64(
    const double* data,
    int len,
    const int* smoothing_windows,
    const int* slow_lengths,
    const int* fast_lengths,
    const int* resolutions,
    const int* ma_types,
    int rows,
    int max_smoothing_window,
    int max_slow_length,
    int max_fast_length,
    double* scratch,
    double* out_value,
    double* out_slow_ma,
    double* out_fast_ma,
    double* out_forecast,
    double* out_upper,
    double* out_lower,
    double* out_direction
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int smoothing_window = smoothing_windows[row];
    const int slow_length = slow_lengths[row];
    const int fast_length = fast_lengths[row];
    const int resolution = resolutions[row];
    const int ma_type = ma_types[row];

    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_slow_ma = out_slow_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_fast_ma = out_fast_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_forecast = out_forecast + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction = out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = NAN;
        row_slow_ma[i] = NAN;
        row_fast_ma[i] = NAN;
        row_forecast[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_direction[i] = NAN;
    }

    if (smoothing_window < 2 || slow_length < 2 || fast_length <= 0 || slow_length <= fast_length ||
        resolution < 2 || ma_type < MA_EMA || ma_type > MA_SMA) {
        return;
    }

    const int history_window_len = 2 * slow_length + 1;
    const int row_stride = max_smoothing_window * 4 + max_slow_length + max_fast_length;
    double* row_scratch = scratch + static_cast<size_t>(row) * static_cast<size_t>(row_stride);
    double* hma_half = row_scratch;
    double* hma_full = hma_half + max_smoothing_window;
    double* hma_sqrt = hma_full + max_smoothing_window;
    double* stddev_ring = hma_sqrt + max_smoothing_window;
    double* slow_ring = stddev_ring + max_smoothing_window;
    double* fast_ring = slow_ring + max_slow_length;

    EmaState slow_ema;
    EmaState fast_ema;
    SmaState slow_sma;
    SmaState fast_sma;
    HmaState hma;
    StddevState stddev;

    slow_ema.init(slow_length);
    fast_ema.init(fast_length);
    slow_sma.init(slow_length, slow_ring);
    fast_sma.init(fast_length, fast_ring);
    hma.init(smoothing_window, hma_half, hma_full, hma_sqrt);
    stddev.init(smoothing_window, stddev_ring);

    double previous_hma = NAN;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            slow_ema.reset();
            fast_ema.reset();
            slow_sma.reset();
            fast_sma.reset();
            hma.reset();
            stddev.reset();
            previous_hma = NAN;
            continue;
        }

        double slow_ma = NAN;
        double fast_ma = NAN;
        double current_hma = NAN;
        double current_std = NAN;

        if (ma_type == MA_EMA) {
            slow_ema.update(value, &slow_ma);
            fast_ema.update(value, &fast_ma);
        } else {
            slow_sma.update(value, &slow_ma);
            fast_sma.update(value, &fast_ma);
        }
        hma.update(value, &current_hma);
        stddev.update(value, &current_std);

        row_slow_ma[i] = slow_ma;
        row_fast_ma[i] = fast_ma;

        double direction = NAN;
        if (isfinite(slow_ma) && isfinite(fast_ma)) {
            direction = fast_ma > slow_ma ? -1.0 : 1.0;
        }
        row_direction[i] = direction;

        if (isfinite(current_hma) && isfinite(previous_hma) && isfinite(current_std)) {
            const double forecast = current_hma + (current_hma - previous_hma);
            const double upper = forecast + current_std;
            const double lower = forecast - current_std;
            row_forecast[i] = forecast;
            row_upper[i] = upper;
            row_lower[i] = lower;

            if (isfinite(direction) && i + 1 >= history_window_len) {
                double probability = NAN;
                if (probability_from_slice(
                        data,
                        i,
                        ma_type,
                        slow_length,
                        fast_length,
                        resolution,
                        history_window_len,
                        lower,
                        upper,
                        direction,
                        &probability)) {
                    row_value[i] = probability;
                }
            }
        }

        previous_hma = current_hma;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `moving_average_cross_probability_with_kernel`
// (src/indicators/moving_average_cross_probability.rs:935) ->
// `moving_average_cross_probability_compute_into` (:855), with
// `count_ema_crosses_estimated` (:547), `truncated_ema_pair_from_slice` (:499),
// `EmaStream::update` (ema.rs:592), `LinWma::update` (hma.rs:750),
// `HmaStream::update` (hma.rs:844) and `StdDevStream::update` (stddev.rs:545).
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `moving_average_cross_probability_batch_f64` (:357) is double-clean but
// declares nineteen parameters -- five `const int*` per-row parameter arrays, a
// host-allocated `double* scratch` and SEVEN output matrices. The f64 lane
// launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// and reads back ONE matrix, so the lane gets its own entry point here. Every
// ring is a fixed-size PER-THREAD array at the pinned defaults: 3 + 7 + 2 + 7 =
// 19 doubles, 152 bytes. Bounded at compile time, not allocated.
//
// WHICH COLUMN: `value`. `compute_moving_average_cross_probability_batch`
// (cpu_batch.rs:11520) maps output_id "value" onto `out.value`, the crossing
// probability itself.
//
// SHAPE: one thread per combo, bars ascending. Four carried filters (slow EMA,
// fast EMA, HMA, rolling stddev), `previous_hma`, and the pair of TRUNCATED
// EMAs that the probability is projected from -- and that pair is a recurrence
// with a sliding-window correction term (:888-897), so bar i cannot be computed
// without bar i-1.
//
// PERIOD-INVARIANT: the CPU batch reads `ma_type`, `smoothing_window`,
// `slow_length`, `fast_length` and `resolution` and NEVER `period`
// (cpu_batch.rs:11469-11496), so every swept period gives the same CPU column
// and this kernel writes identical rows. Pinned at the CPU defaults: ma_type
// `ema`, smoothing_window 7, slow_length 30, fast_length 14, resolution 50
// (cpu_batch.rs:11470-11496), hence `history_window_len = 2*30+1 = 61` (:413).
//
// WHY THIS DOES NOT REUSE THE STATE STRUCTS ABOVE IN THIS FILE
//
// `EmaState`, `WmaState`, `HmaState` and `StddevState` (:22-240) each write the
// arithmetic UNFUSED where the CPU writes `mul_add`, which under this lane's
// `-fmad=false` is one extra rounding per bar inside a recurrence:
//   * EMA seed -- CPU `(x - mean).mul_add(inv, mean)` (ema.rs:604) vs
//     `mean += (input - mean) / count`; and the CPU stores `inv = 1/n` and
//     MULTIPLIES, it does not divide.
//   * EMA tail -- CPU `beta.mul_add(mean, alpha * x)` (ema.rs:606) vs
//     `alpha * input + beta * mean`.
//   * WMA seed -- CPU `(count as f64).mul_add(value, wsum)` (hma.rs:769).
//   * WMA roll -- CPU `n.mul_add(value, wsum - prev_sum)` (hma.rs:806).
//   * HMA mix -- CPU `2.0f64.mul_add(h, -f)` (hma.rs:849) vs `2.0*h - f`.
//   * stddev -- CPU multiplies by a stored `inv_den = 1/period`
//     (stddev.rs:563) where the struct above DIVIDES, and the CPU returns a
//     hard `0.0` when `var <= 0.0` (:565) where the struct clamps with `fmax`.
//   * the probability EMAs -- the CPU carries them with a drop-scale
//     correction (:888-897); `truncated_ema_from_slice` (:257) re-runs the
//     whole 61-bar window every bar, which is the same VALUE by algebra and a
//     different one in the last place.
// So this section carries its own Neo states, written operand for operand
// against those CPU lines. The existing structs are left alone because the
// multi-output entry point above and its own wrapper still use them.
//
// `slow_drop_scale` is `beta.powi(61)`. Rust lowers `f64::powi` to compiler-rt
// `__powidf2`, which is square-and-multiply with the squaring skipped on the
// final iteration; `neo_macp_powi` below is that routine, not `pow(beta,61.0)`
// -- `pow` is correctly rounded for the whole expression and would differ.
//
// NaN SEMANTICS: every state reproduces the CPU's own NaN bookkeeping --
// `LinWma`'s `nan_count`/`dirty`/`rebuild` (hma.rs:729-800) and
// `StdDevStream`'s four-way `(old_is_nan, new_is_nan)` match
// (stddev.rs:580-604). No comparison chain stands in for an `f64::max` here,
// so rule 4 has nothing to catch: the only min/max in the CPU path is
// `var <= 0.0`, reproduced as a branch.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic, no epsilon. `sqrt`/`floor` are the double overloads. The NaN is a
// DOUBLE quiet-NaN bit pattern.
//
// FIRST VALID IS NOT READ: the CPU writes `out_value[idx]` at EVERY bar (:922),
// NaN where the filters are not ready, and never resets on a hole. The lane row
// declares `F64FirstValidRule::Ignored`.
// ---------------------------------------------------------------------------

#define NEO_MACP_SMOOTHING_WINDOW 7
#define NEO_MACP_SLOW_LENGTH 30
#define NEO_MACP_FAST_LENGTH 14
#define NEO_MACP_RESOLUTION 50
#define NEO_MACP_HISTORY_LEN (2 * NEO_MACP_SLOW_LENGTH + 1)
#define NEO_MACP_WMA_HALF (NEO_MACP_SMOOTHING_WINDOW / 2)
#define NEO_MACP_WMA_SQRT 2

__device__ inline double neo_macp_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// compiler-rt `__powidf2`, which is what Rust's `f64::powi` lowers to.
__device__ inline double neo_macp_powi(double a, int b) {
    double r = 1.0;
    while (true) {
        if (b & 1) {
            r *= a;
        }
        b /= 2;
        if (b == 0) {
            break;
        }
        a *= a;
    }
    return r;
}

// `EmaStream` -- ema.rs:592-617. Seeds with a running mean over the first
// `period` samples, then the classic recurrence. Both steps fused exactly
// where the CPU fuses them.
struct NeoMacpEma {
    int period;
    double alpha;
    double beta;
    int count;
    double mean;
    bool filled;

    __device__ void init(int p) {
        period = p;
        alpha = 2.0 / (static_cast<double>(p) + 1.0);
        beta = 1.0 - alpha;
        count = 0;
        mean = neo_macp_qnan();
        filled = false;
    }

    __device__ double update(double x) {
        if (!isfinite(x)) {
            return filled ? mean : neo_macp_qnan();
        }
        count += 1;
        if (count == 1) {
            mean = x;
        } else if (count <= period) {
            // CPU: `(x - self.mean).mul_add(inv, self.mean)` with a STORED
            // `inv = 1/count`, ema.rs:603-604.
            const double inv = 1.0 / static_cast<double>(count);
            mean = fma(x - mean, inv, mean);
        } else {
            // CPU: `self.beta.mul_add(self.mean, self.alpha * x)`, ema.rs:606.
            mean = fma(beta, mean, alpha * x);
        }
        if (!filled && count >= period) {
            filled = true;
        }
        return filled ? mean : neo_macp_qnan();
    }
};

// `LinWma` -- hma.rs:696-808, NaN bookkeeping included.
struct NeoMacpWma {
    double* buffer;
    int period;
    double inv_norm;
    int head;
    int count;
    int nan_count;
    bool filled;
    bool dirty;
    double sum;
    double wsum;

    __device__ void init(double* storage, int p) {
        buffer = storage;
        period = p;
        const double norm =
            static_cast<double>(p) * (static_cast<double>(p) + 1.0) * 0.5;
        inv_norm = 1.0 / norm;
        head = 0;
        count = 0;
        nan_count = 0;
        filled = false;
        dirty = false;
        sum = 0.0;
        wsum = 0.0;
        for (int i = 0; i < p; ++i) {
            buffer[i] = neo_macp_qnan();
        }
    }

    __device__ void rebuild() {
        sum = 0.0;
        wsum = 0.0;
        nan_count = 0;
        int idx = head;
        for (int i = 0; i < period; ++i) {
            const double v = buffer[idx];
            if (isnan(v)) {
                nan_count += 1;
            } else {
                sum += v;
                wsum = fma(static_cast<double>(i) + 1.0, v, wsum);
            }
            idx = (idx + 1 == period) ? 0 : (idx + 1);
        }
        dirty = nan_count != 0;
    }

    // Returns true when the CPU returns `Some`; `*out` carries the value.
    __device__ bool update(double value, double* out) {
        const double n = static_cast<double>(period);
        const double old = buffer[head];
        buffer[head] = value;
        head = (head + 1 == period) ? 0 : (head + 1);

        if (!filled) {
            count += 1;
            if (isnan(value)) {
                nan_count += 1;
                dirty = true;
            } else {
                sum += value;
                wsum = fma(static_cast<double>(count), value, wsum);
            }
            if (count == period) {
                filled = true;
                *out = (nan_count > 0) ? neo_macp_qnan() : (wsum * inv_norm);
                return true;
            }
            *out = neo_macp_qnan();
            return false;
        }

        if (isnan(old) && nan_count > 0) {
            nan_count -= 1;
        }
        if (isnan(value)) {
            nan_count += 1;
        }
        if (nan_count > 0) {
            dirty = true;
            *out = neo_macp_qnan();
            return true;
        }
        if (dirty) {
            rebuild();
            dirty = false;
            *out = wsum * inv_norm;
            return true;
        }

        const double prev_sum = sum;
        sum = prev_sum + value - old;
        // CPU: `n.mul_add(value, self.wsum - prev_sum)`, hma.rs:806.
        wsum = fma(n, value, wsum - prev_sum);
        *out = wsum * inv_norm;
        return true;
    }
};

// `HmaStream` -- hma.rs:844-854. Note the CPU updates FULL first, then HALF.
struct NeoMacpHma {
    NeoMacpWma wma_half;
    NeoMacpWma wma_full;
    NeoMacpWma wma_sqrt;

    __device__ void init(double* half_storage, double* full_storage, double* sqrt_storage) {
        wma_half.init(half_storage, NEO_MACP_WMA_HALF);
        wma_full.init(full_storage, NEO_MACP_SMOOTHING_WINDOW);
        wma_sqrt.init(sqrt_storage, NEO_MACP_WMA_SQRT);
    }

    __device__ double update(double value) {
        double f = neo_macp_qnan();
        double h = neo_macp_qnan();
        const bool full_ready = wma_full.update(value, &f);
        const bool half_ready = wma_half.update(value, &h);
        if (full_ready && half_ready) {
            // CPU: `2.0f64.mul_add(h, -f)`, hma.rs:849.
            const double x = fma(2.0, h, -f);
            double result = neo_macp_qnan();
            if (wma_sqrt.update(x, &result)) {
                return result;
            }
            return neo_macp_qnan();
        }
        return neo_macp_qnan();
    }
};

// `StdDevStream` -- stddev.rs:545-623, `nbdev` pinned to 4.0 by the caller
// (moving_average_cross_probability.rs:841).
struct NeoMacpStddev {
    double* buffer;
    int period;
    double inv_den;
    double nbdev;
    int head;
    int nan_count;
    bool filled;
    double sum;
    double sum_sqr;

    __device__ void init(double* storage, int p, double dev) {
        buffer = storage;
        period = p;
        inv_den = 1.0 / static_cast<double>(p);
        nbdev = dev;
        head = 0;
        nan_count = 0;
        filled = false;
        sum = 0.0;
        sum_sqr = 0.0;
        for (int i = 0; i < p; ++i) {
            buffer[i] = neo_macp_qnan();
        }
    }

    __device__ double finish() const {
        if (nan_count > 0) {
            return neo_macp_qnan();
        }
        const double mean = sum * inv_den;
        const double var = (sum_sqr * inv_den) - (mean * mean);
        return (var <= 0.0) ? 0.0 : (sqrt(var) * nbdev);
    }

    __device__ double update(double value) {
        if (!filled) {
            if (isnan(value)) {
                nan_count += 1;
            } else {
                sum += value;
                sum_sqr += value * value;
            }
            buffer[head] = value;
            const int next = head + 1;
            if (next == period) {
                head = 0;
                filled = true;
                return finish();
            }
            head = next;
            return neo_macp_qnan();
        }

        const double old = buffer[head];
        const bool new_is_nan = isnan(value);
        const bool old_is_nan = isnan(old);
        if (!old_is_nan && !new_is_nan) {
            sum += value - old;
            sum_sqr += (value * value) - (old * old);
        } else if (!old_is_nan && new_is_nan) {
            sum -= old;
            sum_sqr -= old * old;
            nan_count += 1;
        } else if (old_is_nan && !new_is_nan) {
            if (nan_count > 0) {
                nan_count -= 1;
            }
            sum += value;
            sum_sqr += value * value;
        } else {
            // (true, true): the CPU decrements then increments, a no-op.
        }

        buffer[head] = value;
        head += 1;
        if (head == period) {
            head = 0;
        }
        return finish();
    }
};

extern "C" __global__ void moving_average_cross_probability_neo_batch_f64(
    const double* __restrict__ data,
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

    const double nan_value = neo_macp_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    double half_ring[NEO_MACP_WMA_HALF];
    double full_ring[NEO_MACP_SMOOTHING_WINDOW];
    double sqrt_ring[NEO_MACP_WMA_SQRT];
    double std_ring[NEO_MACP_SMOOTHING_WINDOW];

    // `resolve_params`, :414-417.
    const double slow_alpha = 2.0 / (static_cast<double>(NEO_MACP_SLOW_LENGTH) + 1.0);
    const double slow_beta = 1.0 - 2.0 / (static_cast<double>(NEO_MACP_SLOW_LENGTH) + 1.0);
    const double fast_alpha = 2.0 / (static_cast<double>(NEO_MACP_FAST_LENGTH) + 1.0);
    const double fast_beta = 1.0 - 2.0 / (static_cast<double>(NEO_MACP_FAST_LENGTH) + 1.0);

    NeoMacpEma slow_stream;
    NeoMacpEma fast_stream;
    NeoMacpHma hma_stream;
    NeoMacpStddev stddev_stream;
    slow_stream.init(NEO_MACP_SLOW_LENGTH);
    fast_stream.init(NEO_MACP_FAST_LENGTH);
    hma_stream.init(half_ring, full_ring, sqrt_ring);
    stddev_stream.init(std_ring, NEO_MACP_SMOOTHING_WINDOW, 4.0);

    const int history_len = NEO_MACP_HISTORY_LEN;
    const double slow_drop_scale = neo_macp_powi(slow_beta, history_len);
    const double fast_drop_scale = neo_macp_powi(fast_beta, history_len);
    double slow_probability_ema = nan_value;
    double fast_probability_ema = nan_value;
    double previous_hma = nan_value;

    for (int idx = 0; idx < n; ++idx) {
        const double value = data[idx];
        const double slow_ma = slow_stream.update(value);
        const double fast_ma = fast_stream.update(value);
        const double current_hma = hma_stream.update(value);
        const double current_std = stddev_stream.update(value);

        double direction = nan_value;
        if (isfinite(slow_ma) && isfinite(fast_ma)) {
            direction = (fast_ma > slow_ma) ? -1.0 : 1.0;
        }

        double probability = nan_value;

        // Truncated-EMA pair, :884-897.
        if (idx + 1 == history_len) {
            double s = data[0];
            double f = data[0];
            for (int j = 1; j <= idx; ++j) {
                const double v = data[j];
                s = fma(slow_alpha, v, slow_beta * s);
                f = fma(fast_alpha, v, fast_beta * f);
            }
            slow_probability_ema = s;
            fast_probability_ema = f;
        } else if (idx + 1 > history_len) {
            const double dropped = data[idx - history_len];
            const double new_oldest = data[idx + 1 - history_len];
            slow_probability_ema =
                fma(slow_alpha, value, slow_beta * slow_probability_ema) +
                slow_drop_scale * (new_oldest - dropped);
            fast_probability_ema =
                fma(fast_alpha, value, fast_beta * fast_probability_ema) +
                fast_drop_scale * (new_oldest - dropped);
        }

        if (isfinite(current_hma) && isfinite(previous_hma) && isfinite(current_std)) {
            const double forecast = current_hma + (current_hma - previous_hma);
            const double upper = forecast + current_std;
            const double lower = forecast - current_std;

            if (isfinite(direction) && idx + 1 >= history_len) {
                const double step =
                    (upper - lower) / (static_cast<double>(NEO_MACP_RESOLUTION) - 1.0);
                // `count_ema_crosses_estimated`, :547-612. The predicate is
                // affine in the probe index -- slow_future - fast_future =
                // base + slope*k -- so it is monotone and the CPU estimate-
                // plus-correction search and this direct scan return the SAME
                // integer. The scan is 50 iterations and needs no search.
                int hits = 0;
                for (int k = 0; k < NEO_MACP_RESOLUTION; ++k) {
                    const double price = lower + step * static_cast<double>(k);
                    const double slow_future =
                        fma(slow_alpha, price, slow_beta * slow_probability_ema);
                    const double fast_future =
                        fma(fast_alpha, price, fast_beta * fast_probability_ema);
                    const bool crossed = (direction < 0.0)
                        ? (slow_future > fast_future)
                        : (slow_future <= fast_future);
                    if (crossed) {
                        hits += 1;
                    }
                }
                probability = 100.0 * static_cast<double>(hits) /
                              static_cast<double>(NEO_MACP_RESOLUTION);
            }
        }

        row[idx] = probability;
        previous_hma = current_hma;
    }
}
