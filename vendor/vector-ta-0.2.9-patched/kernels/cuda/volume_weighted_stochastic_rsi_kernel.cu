#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_WSMA = 0;
constexpr int MA_SMA = 1;
constexpr int MA_EMA = 2;
constexpr int MA_WMA = 3;
constexpr int MA_VWMA = 4;

__device__ inline bool is_valid_pair(double source, double volume) {
    return isfinite(source) && isfinite(volume);
}

__device__ inline double rsi_from_avgs(double avg_gain, double avg_loss) {
    if (avg_loss <= 0.0) {
        if (avg_gain <= 0.0) {
            return 50.0;
        }
        return 100.0;
    }
    if (avg_gain <= 0.0) {
        return 0.0;
    }
    const double rs = avg_gain / avg_loss;
    return 100.0 - 100.0 / (1.0 + rs);
}

struct WeightedRsiState {
    int period;
    double prev_source;
    bool has_prev_source;
    double gain_sum;
    double loss_sum;
    int count;
    double avg_gain;
    double avg_loss;
    bool initialized;

    __device__ void init(int value) {
        period = value;
        prev_source = NAN;
        has_prev_source = false;
        gain_sum = 0.0;
        loss_sum = 0.0;
        count = 0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        initialized = false;
    }

    __device__ double update(double source, double volume) {
        if (!is_valid_pair(source, volume)) {
            if (isfinite(source)) {
                prev_source = source;
                has_prev_source = true;
            } else {
                prev_source = NAN;
                has_prev_source = false;
            }
            return NAN;
        }

        if (!has_prev_source || !isfinite(prev_source)) {
            prev_source = source;
            has_prev_source = true;
            return NAN;
        }

        const double change = source - prev_source;
        const double gain = (change > 0.0 ? change : 0.0) * volume;
        const double loss = (change < 0.0 ? -change : 0.0) * volume;
        prev_source = source;

        if (!initialized) {
            gain_sum += gain;
            loss_sum += loss;
            count += 1;
            if (count == period) {
                avg_gain = gain_sum / static_cast<double>(period);
                avg_loss = loss_sum / static_cast<double>(period);
                initialized = true;
                return rsi_from_avgs(avg_gain, avg_loss);
            }
            return NAN;
        }

        const double p = static_cast<double>(period);
        avg_gain = (avg_gain * (p - 1.0) + gain) / p;
        avg_loss = (avg_loss * (p - 1.0) + loss) / p;
        return rsi_from_avgs(avg_gain, avg_loss);
    }
};

struct StochState {
    double* window;
    int period;
    int head;
    int count;
    int valid;

    __device__ void init(int value, double* storage) {
        window = storage;
        period = value;
        head = 0;
        count = 0;
        valid = 0;
        for (int i = 0; i < value; ++i) {
            window[i] = NAN;
        }
    }

    __device__ double update(double value) {
        if (count == period) {
            const double old = window[head];
            if (isfinite(old)) {
                valid -= 1;
            }
        } else {
            count += 1;
        }

        window[head] = value;
        head += 1;
        if (head == period) {
            head = 0;
        }

        if (isfinite(value)) {
            valid += 1;
        }

        if (count < period || valid < period || !isfinite(value)) {
            return NAN;
        }

        double lowest = INFINITY;
        double highest = -INFINITY;
        for (int i = 0; i < period; ++i) {
            lowest = fmin(lowest, window[i]);
            highest = fmax(highest, window[i]);
        }
        const double denom = highest - lowest;
        if (!isfinite(denom) || denom == 0.0) {
            return NAN;
        }
        return (value - lowest) / denom * 100.0;
    }
};

struct WeightedSmaState {
    int period;
    double* numerators;
    double* weights;
    int head;
    int count;
    double numerator_sum;
    double weight_sum;

    __device__ void init(int value, double* numerator_storage, double* weight_storage) {
        period = value;
        numerators = numerator_storage;
        weights = weight_storage;
        head = 0;
        count = 0;
        numerator_sum = 0.0;
        weight_sum = 0.0;
    }

    __device__ double update(double value, double weight) {
        const double numerator = value * weight;
        if (count < period) {
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            count += 1;
            numerator_sum += numerator;
            weight_sum += weight;
        } else {
            const double old_numerator = numerators[head];
            const double old_weight = weights[head];
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            numerator_sum += numerator - old_numerator;
            weight_sum += weight - old_weight;
        }

        if (count == period && weight_sum != 0.0) {
            return numerator_sum / weight_sum;
        }
        return NAN;
    }
};

struct WeightedEmaState {
    double alpha;
    double numerator;
    double denominator;
    bool initialized;

    __device__ void init(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        numerator = 0.0;
        denominator = 0.0;
        initialized = false;
    }

    __device__ double update(double value, double weight) {
        const double num = value * weight;
        if (!initialized) {
            numerator = num;
            denominator = weight;
            initialized = true;
        } else {
            const double beta = 1.0 - alpha;
            numerator = alpha * num + beta * numerator;
            denominator = alpha * weight + beta * denominator;
        }

        if (denominator != 0.0) {
            return numerator / denominator;
        }
        return NAN;
    }
};

struct WeightedWsmaState {
    int period;
    double numerator_sum;
    double denominator_sum;
    int count;
    double numerator_avg;
    double denominator_avg;
    bool initialized;

    __device__ void init(int value) {
        period = value;
        numerator_sum = 0.0;
        denominator_sum = 0.0;
        count = 0;
        numerator_avg = 0.0;
        denominator_avg = 0.0;
        initialized = false;
    }

    __device__ double update(double value, double weight) {
        const double numerator = value * weight;
        if (!initialized) {
            numerator_sum += numerator;
            denominator_sum += weight;
            count += 1;
            if (count == period) {
                numerator_avg = numerator_sum / static_cast<double>(period);
                denominator_avg = denominator_sum / static_cast<double>(period);
                initialized = true;
                if (denominator_avg != 0.0) {
                    return numerator_avg / denominator_avg;
                }
            }
            return NAN;
        }

        const double p = static_cast<double>(period);
        numerator_avg = (numerator_avg * (p - 1.0) + numerator) / p;
        denominator_avg = (denominator_avg * (p - 1.0) + weight) / p;
        if (denominator_avg != 0.0) {
            return numerator_avg / denominator_avg;
        }
        return NAN;
    }
};

struct WeightedWmaState {
    int period;
    double* numerators;
    double* weights;
    int head;
    int count;
    double numerator_plain_sum;
    double numerator_weighted_sum;
    double weight_plain_sum;
    double weight_weighted_sum;

    __device__ void init(int value, double* numerator_storage, double* weight_storage) {
        period = value;
        numerators = numerator_storage;
        weights = weight_storage;
        head = 0;
        count = 0;
        numerator_plain_sum = 0.0;
        numerator_weighted_sum = 0.0;
        weight_plain_sum = 0.0;
        weight_weighted_sum = 0.0;
    }

    __device__ double update(double value, double weight) {
        const double numerator = value * weight;
        if (count < period) {
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            count += 1;
            numerator_plain_sum += numerator;
            weight_plain_sum += weight;
            numerator_weighted_sum += numerator * static_cast<double>(count);
            weight_weighted_sum += weight * static_cast<double>(count);
        } else {
            const double old_numerator = numerators[head];
            const double old_weight = weights[head];
            const double prev_numerator_plain = numerator_plain_sum;
            const double prev_weight_plain = weight_plain_sum;
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            numerator_plain_sum = prev_numerator_plain - old_numerator + numerator;
            weight_plain_sum = prev_weight_plain - old_weight + weight;
            numerator_weighted_sum =
                numerator_weighted_sum - prev_numerator_plain + numerator * static_cast<double>(period);
            weight_weighted_sum =
                weight_weighted_sum - prev_weight_plain + weight * static_cast<double>(period);
        }

        if (count == period && weight_weighted_sum != 0.0) {
            return numerator_weighted_sum / weight_weighted_sum;
        }
        return NAN;
    }
};

struct WeightedMaState {
    int kind;
    WeightedWsmaState wsma;
    WeightedSmaState sma;
    WeightedEmaState ema;
    WeightedWmaState wma;
    WeightedSmaState vwma;

    __device__ void init(int ma_kind, int period, double* buf1, double* buf2) {
        kind = ma_kind;
        if (kind == MA_WSMA) {
            wsma.init(period);
        } else if (kind == MA_SMA) {
            sma.init(period, buf1, buf2);
        } else if (kind == MA_EMA) {
            ema.init(period);
        } else if (kind == MA_WMA) {
            wma.init(period, buf1, buf2);
        } else {
            vwma.init(period, buf1, buf2);
        }
    }

    __device__ double update(double value, double weight) {
        if (kind == MA_WSMA) {
            return wsma.update(value, weight);
        }
        if (kind == MA_SMA) {
            return sma.update(value, weight);
        }
        if (kind == MA_EMA) {
            return ema.update(value, weight);
        }
        if (kind == MA_WMA) {
            return wma.update(value, weight);
        }
        return vwma.update(value, weight);
    }
};
}

extern "C" __global__ void volume_weighted_stochastic_rsi_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ stoch_lengths,
    const int* __restrict__ k_lengths,
    const int* __restrict__ d_lengths,
    const int* __restrict__ ma_codes,
    int rows,
    double* __restrict__ out_k,
    double* __restrict__ out_d
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_k = out_k + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_d = out_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_k[i] = NAN;
        row_d[i] = NAN;
    }

    const int rsi_length = rsi_lengths[row];
    const int stoch_length = stoch_lengths[row];
    const int k_length = k_lengths[row];
    const int d_length = d_lengths[row];
    const int ma_code = ma_codes[row];
    if (rsi_length <= 0 || rsi_length > len || stoch_length <= 0 || stoch_length > len ||
        k_length <= 0 || k_length > len || d_length <= 0 || d_length > len ||
        ma_code < MA_WSMA || ma_code > MA_VWMA) {
        return;
    }

    double* stoch_window = new double[stoch_length];
    double* k_buf1 = nullptr;
    double* k_buf2 = nullptr;
    double* d_buf1 = nullptr;
    double* d_buf2 = nullptr;
    if (ma_code == MA_SMA || ma_code == MA_WMA || ma_code == MA_VWMA) {
        k_buf1 = new double[k_length];
        k_buf2 = new double[k_length];
        d_buf1 = new double[d_length];
        d_buf2 = new double[d_length];
    }
    if (stoch_window == nullptr ||
        ((ma_code == MA_SMA || ma_code == MA_WMA || ma_code == MA_VWMA) &&
         (k_buf1 == nullptr || k_buf2 == nullptr || d_buf1 == nullptr || d_buf2 == nullptr))) {
        delete[] stoch_window;
        delete[] k_buf1;
        delete[] k_buf2;
        delete[] d_buf1;
        delete[] d_buf2;
        return;
    }

    WeightedRsiState rsi_state;
    StochState stoch_state;
    WeightedMaState k_ma;
    WeightedMaState d_ma;
    rsi_state.init(rsi_length);
    stoch_state.init(stoch_length, stoch_window);
    k_ma.init(ma_code, k_length, k_buf1, k_buf2);
    d_ma.init(ma_code, d_length, d_buf1, d_buf2);

    for (int i = 0; i < len; ++i) {
        const double rsi = rsi_state.update(source[i], volume[i]);
        const double stoch = stoch_state.update(rsi);
        const double k = isfinite(stoch) ? k_ma.update(stoch, volume[i]) : NAN;
        const double d = isfinite(k) ? d_ma.update(k, volume[i]) : NAN;
        row_k[i] = k;
        row_d[i] = d;
    }

    delete[] stoch_window;
    delete[] k_buf1;
    delete[] k_buf2;
    delete[] d_buf1;
    delete[] d_buf2;
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/volume_weighted_stochastic_rsi.rs
 *   `volume_weighted_stochastic_rsi_compute_k_into` (:877-898) -- the path
 *   `..._output_into_slice` (:1068) takes for the `K` field -- built from
 *   `WeightedRsiState::update` (:481-521), `StochState::update` (:544-581)
 *   and `WeightedWsmaState::update` (:702-742).
 *   Batch dispatcher: cpu_batch.rs:12385 -- output "value" is an ALIAS OF
 *   "k" (:12392), so this kernel emits `k`, never `d`.
 *
 * WHY A SECOND ENTRY POINT: `volume_weighted_stochastic_rsi_batch_f64` (:372)
 *   takes 11 parameters and emits two series. The lane launches
 *   (close, volume, n, periods, n_combos, first_valid, out).
 *
 * INPUT: (close, volume) -- extract_close_volume_input (cpu_batch.rs:12389)
 *   with source "close" -- F64InputKind::CloseVolume.
 *
 * FIRST-VALID IGNORED: `compute_k_into` walks EVERY bar from 0; an invalid
 *   bar is handled inside `WeightedRsiState::update` (:482-489), which does
 *   NOT reset the averages -- it only drops prev_source and returns NaN, and
 *   the NaN then flows through StochState as a window entry. Reproducing that
 *   requires walking every bar, so the caller's index is not read. Note
 *   `first` is computed by `..._prepare` (:984) purely for the length check
 *   and is explicitly discarded at :1021 (`let _ = first;`).
 *
 * PERIOD-INVARIANT: the CPU batch reads NAMED parameters -- `rsi_length`,
 *   `stoch_length`, `k_length`, `d_length`, `ma_type` (cpu_batch.rs:12413-
 *   12419) -- and never `period`. All are pinned at their CPU defaults here
 *   (14 / 14 / 3 / "WSMA"), so every row of a sweep is byte-identical.
 *
 * MA TYPE: the default "WSMA" (:12419) parses to VwsrsiMaType::Wsma (:347),
 *   which is the volume-weighted WILDER average, not an SMA. Only that arm is
 *   compiled here; a caller asking for another ma_type is not on this lane.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. A Wilder recurrence on
 *   volume-weighted gains/losses feeds a rolling stochastic window which feeds
 *   a second Wilder recurrence -- three carried states in a cascade.
 *
 * ARITHMETIC taken verbatim:
 *   * the RSI seed divides the accumulated sums by `period` AFTER exactly
 *     `period` deltas (:509-510); the step is
 *     `(avg * (period - 1) + x) / period` (:518) -- multiply, add, divide.
 *   * `rsi_from_avgs` (:438) is a four-way branch on `<= 0.0`, NOT a division
 *     guarded by an epsilon: avg_loss <= 0 with avg_gain <= 0 gives 50.0.
 *   * `StochState` scans the WHOLE window with f64::min / f64::max (:572-573)
 *     -- hence fmin/fmax, which return the non-NaN operand. It refuses when
 *     `count < period || valid < period || !value.is_finite()` (:565).
 *   * the WSMA carries a numerator AND a denominator average, and divides
 *     only when the denominator average is exactly non-zero (:738).
 *
 * EPSILON: there is none anywhere on this path. Every CPU guard is an exact
 *   `== 0.0` / `<= 0.0` test, and no f32-sized tolerance is imported.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:12413-12419. */
#define NEO_VWSRSI_RSI_LENGTH   14
#define NEO_VWSRSI_STOCH_LENGTH 14
#define NEO_VWSRSI_K_LENGTH     3

extern "C" __global__
void volume_weighted_stochastic_rsi_neo_batch_f64(const double* __restrict__ close,
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

    const int rsi_len   = NEO_VWSRSI_RSI_LENGTH;
    const int stoch_len = NEO_VWSRSI_STOCH_LENGTH;
    const int k_len     = NEO_VWSRSI_K_LENGTH;
    /* ..._prepare refuses any length > len. */
    if (rsi_len > n || stoch_len > n || k_len > n) return;

    const double rsi_pf = (double)rsi_len;
    const double k_pf   = (double)k_len;

    /* WeightedRsiState (:454) */
    bool   has_prev = false;
    double prev_source = 0.0;
    double gain_sum = 0.0, loss_sum = 0.0;
    int    rsi_count = 0;
    double avg_gain = 0.0, avg_loss = 0.0;
    bool   rsi_init = false;

    /* StochState (:525) -- window is `vec![NAN; period]`. */
    double win[NEO_VWSRSI_STOCH_LENGTH];
    for (int k = 0; k < stoch_len; ++k) win[k] = NEO_F64_NAN;
    int win_head = 0, win_count = 0, win_valid = 0;

    /* WeightedWsmaState (:676) for the K smoothing. */
    double k_num_sum = 0.0, k_den_sum = 0.0;
    int    k_count = 0;
    double k_num_avg = 0.0, k_den_avg = 0.0;
    bool   k_init = false;

    for (int i = 0; i < n; ++i) {
        const double src = close[i];
        const double vol = volume[i];

        /* ---- WeightedRsiState::update (:481) ---- */
        double rsi_value;
        if (!(isfinite(src) && isfinite(vol))) {
            /* :483 -- prev_source is KEPT when source alone is finite. */
            if (isfinite(src)) { prev_source = src; has_prev = true; }
            else               { has_prev = false; }
            rsi_value = NEO_F64_NAN;
        } else if (!has_prev) {
            prev_source = src; has_prev = true;
            rsi_value = NEO_F64_NAN;
        } else {
            const double change = src - prev_source;
            const double gain = fmax(change, 0.0) * vol;
            const double loss = fmax(-change, 0.0) * vol;
            prev_source = src;

            if (!rsi_init) {
                gain_sum += gain;
                loss_sum += loss;
                rsi_count += 1;
                if (rsi_count == rsi_len) {
                    avg_gain = gain_sum / rsi_pf;
                    avg_loss = loss_sum / rsi_pf;
                    rsi_init = true;
                    /* rsi_from_avgs (:438) */
                    if (avg_loss <= 0.0)      rsi_value = (avg_gain <= 0.0) ? 50.0 : 100.0;
                    else if (avg_gain <= 0.0) rsi_value = 0.0;
                    else                      rsi_value = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
                } else {
                    rsi_value = NEO_F64_NAN;
                }
            } else {
                avg_gain = (avg_gain * (rsi_pf - 1.0) + gain) / rsi_pf;
                avg_loss = (avg_loss * (rsi_pf - 1.0) + loss) / rsi_pf;
                if (avg_loss <= 0.0)      rsi_value = (avg_gain <= 0.0) ? 50.0 : 100.0;
                else if (avg_gain <= 0.0) rsi_value = 0.0;
                else                      rsi_value = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
            }
        }

        /* ---- StochState::update (:544) ---- */
        if (win_count == stoch_len) {
            const double old = win[win_head];
            if (isfinite(old)) win_valid -= 1;
        } else {
            win_count += 1;
        }
        win[win_head] = rsi_value;
        win_head += 1; if (win_head == stoch_len) win_head = 0;
        if (isfinite(rsi_value)) win_valid += 1;

        double stoch;
        if (win_count < stoch_len || win_valid < stoch_len || !isfinite(rsi_value)) {
            stoch = NEO_F64_NAN;
        } else {
            double lowest = INFINITY, highest = -INFINITY;
            for (int k = 0; k < stoch_len; ++k) {
                lowest  = fmin(lowest,  win[k]);
                highest = fmax(highest, win[k]);
            }
            const double denom = highest - lowest;
            stoch = (!isfinite(denom) || denom == 0.0)
                  ? NEO_F64_NAN
                  : ((rsi_value - lowest) / denom * 100.0);
        }

        /* ---- WeightedWsmaState::update (:702), gated on a finite stoch ---- */
        if (!isfinite(stoch)) { o[i] = NEO_F64_NAN; continue; }

        const double numerator = stoch * vol;
        double kv;
        if (!k_init) {
            k_num_sum += numerator;
            k_den_sum += vol;
            k_count += 1;
            kv = NEO_F64_NAN;
            if (k_count == k_len) {
                k_num_avg = k_num_sum / k_pf;
                k_den_avg = k_den_sum / k_pf;
                k_init = true;
                if (k_den_avg != 0.0) kv = k_num_avg / k_den_avg;
            }
        } else {
            k_num_avg = (k_num_avg * (k_pf - 1.0) + numerator) / k_pf;
            k_den_avg = (k_den_avg * (k_pf - 1.0) + vol) / k_pf;
            kv = (k_den_avg != 0.0) ? (k_num_avg / k_den_avg) : NEO_F64_NAN;
        }
        o[i] = kv;
    }
}
