#include <cmath>
#include <cstddef>

namespace {
constexpr double EPS = 1e-12;
constexpr int MA_EMA = 0;
constexpr int MA_SMA = 1;
constexpr int MA_HMA = 2;
constexpr int MA_RMA = 3;
constexpr int MA_WMA = 4;
constexpr int MA_VWMA = 5;

__device__ inline int sqrt_period(int period) {
    const double root = floor(sqrt(static_cast<double>(period)));
    return max(static_cast<int>(root), 1);
}

struct EmaState {
    double alpha;
    double value;
    bool has_value;

    __device__ void init(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        reset();
    }

    __device__ void reset() {
        value = 0.0;
        has_value = false;
    }

    __device__ bool update(double input, double* out) {
        const double next = has_value ? alpha * input + (1.0 - alpha) * value : input;
        value = next;
        has_value = true;
        *out = next;
        return true;
    }
};

struct RmaState {
    int period;
    int count;
    double sum;
    double value;
    bool seeded;

    __device__ void init(int p) {
        period = p;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        seeded = false;
    }

    __device__ bool update(double input, double* out) {
        if (seeded) {
            value = ((value * static_cast<double>(period - 1)) + input) / static_cast<double>(period);
            *out = value;
            return true;
        }
        count += 1;
        sum += input;
        if (count == period) {
            value = sum / static_cast<double>(period);
            seeded = true;
            *out = value;
            return true;
        }
        *out = NAN;
        return false;
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

struct VwmaState {
    double* pv_ring;
    double* vol_ring;
    int period;
    int head;
    int count;
    double pv_sum;
    double vol_sum;

    __device__ void init(int p, double* pv_storage, double* vol_storage) {
        period = p;
        pv_ring = pv_storage;
        vol_ring = vol_storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        pv_sum = 0.0;
        vol_sum = 0.0;
    }

    __device__ bool update(double value, double volume, double* out) {
        const double pv = value * volume;
        if (count < period) {
            pv_ring[count] = pv;
            vol_ring[count] = volume;
            count += 1;
            pv_sum += pv;
            vol_sum += volume;
            if (count == period) {
                *out = fabs(vol_sum) <= EPS ? NAN : (pv_sum / vol_sum);
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old_pv = pv_ring[head];
        const double old_vol = vol_ring[head];
        pv_ring[head] = pv;
        vol_ring[head] = volume;
        head += 1;
        if (head == period) {
            head = 0;
        }
        pv_sum += pv - old_pv;
        vol_sum += volume - old_vol;
        *out = fabs(vol_sum) <= EPS ? NAN : (pv_sum / vol_sum);
        return true;
    }
};

struct HmaState {
    WmaState wma_half;
    WmaState wma_full;
    WmaState wma_sqrt;

    __device__ void init(int period, double* half_storage, double* full_storage, double* sqrt_storage) {
        const int half = max(period / 2, 1);
        wma_half.init(half, half_storage);
        wma_full.init(max(period, 1), full_storage);
        wma_sqrt.init(sqrt_period(period), sqrt_storage);
    }

    __device__ void reset() {
        wma_half.reset();
        wma_full.reset();
        wma_sqrt.reset();
    }

    __device__ bool update(double value, double* out) {
        double half_value = NAN;
        double full_value = NAN;
        const bool half_ready = wma_half.update(value, &half_value);
        const bool full_ready = wma_full.update(value, &full_value);
        if (half_ready && full_ready) {
            const double diff = 2.0 * half_value - full_value;
            return wma_sqrt.update(diff, out);
        }
        *out = NAN;
        return false;
    }
};
}

extern "C" __global__ void volume_weighted_relative_strength_index_batch_f64(
    const double* source,
    const double* volume,
    int len,
    const int* rsi_lengths,
    const int* range_lengths,
    const int* ma_lengths,
    const int* ma_codes,
    int rows,
    int scratch_cap,
    double* scratch,
    double* out_rsi,
    double* out_consolidation_strength,
    double* out_rsi_ma,
    double* out_bearish_tp,
    double* out_bullish_tp
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int rsi_length = rsi_lengths[row];
    const int range_length = range_lengths[row];
    const int ma_length = ma_lengths[row];
    const int ma_code = ma_codes[row];

    double* row_rsi = out_rsi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_consolidation =
        out_consolidation_strength + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_rsi_ma = out_rsi_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish = out_bearish_tp + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish = out_bullish_tp + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_rsi[i] = NAN;
        row_consolidation[i] = NAN;
        row_rsi_ma[i] = NAN;
        row_bearish[i] = NAN;
        row_bullish[i] = NAN;
    }

    if (rsi_length <= 0 || range_length <= 0 || ma_length <= 0 || scratch_cap <= 0) {
        return;
    }
    if (ma_code < MA_EMA || ma_code > MA_VWMA) {
        return;
    }
    const int max_needed_cap = max(ma_length, max(range_length, range_length * 2));
    if (max_needed_cap > scratch_cap) {
        return;
    }

    double* row_scratch =
        scratch + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap) * static_cast<size_t>(10);
    double* ma_buf_a = row_scratch + static_cast<size_t>(scratch_cap) * 0;
    double* ma_buf_b = row_scratch + static_cast<size_t>(scratch_cap) * 1;
    double* ma_buf_c = row_scratch + static_cast<size_t>(scratch_cap) * 2;
    double* range_sma_buf = row_scratch + static_cast<size_t>(scratch_cap) * 3;
    double* short_buf_a = row_scratch + static_cast<size_t>(scratch_cap) * 4;
    double* short_buf_b = row_scratch + static_cast<size_t>(scratch_cap) * 5;
    double* short_buf_c = row_scratch + static_cast<size_t>(scratch_cap) * 6;
    double* long_buf_a = row_scratch + static_cast<size_t>(scratch_cap) * 7;
    double* long_buf_b = row_scratch + static_cast<size_t>(scratch_cap) * 8;
    double* long_buf_c = row_scratch + static_cast<size_t>(scratch_cap) * 9;

    RmaState up_rma;
    RmaState down_rma;
    RmaState volume_rma;
    EmaState ma_ema;
    SmaState ma_sma;
    HmaState ma_hma;
    RmaState ma_rma;
    WmaState ma_wma;
    VwmaState ma_vwma;
    SmaState range_sma;
    HmaState range_hma_short;
    HmaState range_hma_long;

    up_rma.init(rsi_length);
    down_rma.init(rsi_length);
    volume_rma.init(rsi_length);
    ma_ema.init(ma_length);
    ma_sma.init(ma_length, ma_buf_a);
    ma_hma.init(ma_length, ma_buf_a, ma_buf_b, ma_buf_c);
    ma_rma.init(ma_length);
    ma_wma.init(ma_length, ma_buf_a);
    ma_vwma.init(ma_length, ma_buf_a, ma_buf_b);
    range_sma.init(range_length, range_sma_buf);
    range_hma_short.init(max(range_length / 2, 1), short_buf_a, short_buf_b, short_buf_c);
    range_hma_long.init(max(range_length * 2, 1), long_buf_a, long_buf_b, long_buf_c);

    bool have_prev_source = false;
    double prev_source = NAN;
    bool have_prev_rsi = false;
    double prev_rsi = NAN;
    bool have_prev_dir = false;
    int prev_dir = 0;
    int valid_rsi_count = 0;
    const double scale = static_cast<double>(max(range_length / 2, 1)) * (5.0 / 3.0);

    for (int i = 0; i < len; ++i) {
        const double src = source[i];
        const double vol = volume[i];
        if (!isfinite(src) || !isfinite(vol)) {
            have_prev_source = false;
            prev_source = NAN;
            have_prev_rsi = false;
            prev_rsi = NAN;
            have_prev_dir = false;
            prev_dir = 0;
            valid_rsi_count = 0;
            up_rma.reset();
            down_rma.reset();
            volume_rma.reset();
            ma_ema.reset();
            ma_sma.reset();
            ma_hma.reset();
            ma_rma.reset();
            ma_wma.reset();
            ma_vwma.reset();
            range_sma.reset();
            range_hma_short.reset();
            range_hma_long.reset();
            continue;
        }

        if (!have_prev_source) {
            have_prev_source = true;
            prev_source = src;
            continue;
        }

        const double delta = src - prev_source;
        prev_source = src;

        const double gain = fmax(delta, 0.0) * vol;
        const double loss = fmax(-delta, 0.0) * vol;
        double up_num = NAN;
        double down_num = NAN;
        double vol_avg = NAN;
        const bool up_ready = up_rma.update(gain, &up_num);
        const bool down_ready = down_rma.update(loss, &down_num);
        const bool vol_ready = volume_rma.update(vol, &vol_avg);
        if (!(up_ready && down_ready && vol_ready)) {
            continue;
        }
        if (fabs(vol_avg) <= EPS) {
            continue;
        }

        const double up = up_num / vol_avg;
        const double down = down_num / vol_avg;
        const double rsi = fabs(down) <= EPS
            ? 100.0
            : (fabs(up) <= EPS ? 0.0 : (100.0 - (100.0 / (1.0 + up / down))));

        double rsi_ma_value = NAN;
        bool rsi_ma_ready = false;
        switch (ma_code) {
            case MA_EMA:
                rsi_ma_ready = ma_ema.update(rsi, &rsi_ma_value);
                break;
            case MA_SMA:
                rsi_ma_ready = ma_sma.update(rsi, &rsi_ma_value);
                break;
            case MA_HMA:
                rsi_ma_ready = ma_hma.update(rsi, &rsi_ma_value);
                break;
            case MA_RMA:
                rsi_ma_ready = ma_rma.update(rsi, &rsi_ma_value);
                break;
            case MA_WMA:
                rsi_ma_ready = ma_wma.update(rsi, &rsi_ma_value);
                break;
            case MA_VWMA:
                rsi_ma_ready = ma_vwma.update(rsi, vol, &rsi_ma_value);
                break;
            default:
                rsi_ma_ready = false;
                rsi_ma_value = NAN;
                break;
        }

        const double bearish_tp =
            have_prev_rsi && prev_rsi >= 80.0 && rsi < 80.0 ? 95.0 : NAN;
        const double bullish_tp =
            have_prev_rsi && prev_rsi <= 20.0 && rsi > 20.0 ? 5.0 : NAN;

        valid_rsi_count += 1;
        const int dir = rsi > 50.0 ? 1 : -1;
        const double transition =
            have_prev_dir && (prev_dir + dir == 0) ? 1.0 : 0.0;
        const double denom = static_cast<double>(min(valid_rsi_count, range_length));
        const double x = transition / denom;

        double p = NAN;
        const bool p_ready = range_sma.update(x, &p);
        double short_value = NAN;
        const bool short_ready = p_ready ? range_hma_short.update(p, &short_value) : false;
        const double f = short_ready ? (short_value * scale) : NAN;
        double consolidation = NAN;
        if (short_ready) {
            double long_value = NAN;
            if (range_hma_long.update(f, &long_value)) {
                consolidation = fmax(long_value, 0.0);
            }
        }

        have_prev_dir = true;
        prev_dir = dir;
        have_prev_rsi = true;
        prev_rsi = rsi;

        row_rsi[i] = rsi;
        row_consolidation[i] = consolidation;
        row_rsi_ma[i] = rsi_ma_ready ? rsi_ma_value : NAN;
        row_bearish[i] = bearish_tp;
        row_bullish[i] = bullish_tp;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/volume_weighted_relative_strength_index.rs
 *   `volume_weighted_relative_strength_index_rsi_only_into` (:1144-1197),
 *   which is the path `..._output_into_slice` (:1101) takes for the `Rsi`
 *   field, and `RmaState::update` (:310-325).
 *   Batch dispatcher: cpu_batch.rs:16302 -- output "value" is an ALIAS OF
 *   "rsi" (:16309), so this kernel emits `rsi`. NOT rsi_ma, NOT the two
 *   take-profit columns.
 *
 * WHY A SECOND ENTRY POINT: `volume_weighted_relative_strength_index_batch_f64`
 *   (:263) takes 15 parameters and emits five series. The lane launches
 *   (close, volume, n, periods, n_combos, first_valid, out) and consumes one.
 *
 * INPUT: (close, volume) -- extract_close_volume_input (cpu_batch.rs:16306)
 *   with source "close" -- F64InputKind::CloseVolume.
 *
 * FIRST-VALID IGNORED: the loop walks EVERY bar from 0 and handles an invalid
 *   bar IN PLACE (:1156-1163) by RESETTING all three RMA accumulators and
 *   clearing prev_source. A global first-valid index would be wrong after the
 *   first hole, because the CPU restarts the whole seed there.
 *
 * PERIOD-INVARIANT: the CPU batch reads NAMED parameters -- `rsi_length`,
 *   `range_length`, `ma_length`, `ma_type` (cpu_batch.rs:16340-16362) -- and
 *   never `period`. The rsi column depends only on `rsi_length`, pinned here
 *   at the CPU default 14. Every row of a sweep is byte-identical, and
 *   `is_period_invariant` says so.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. Three Wilder (RMA)
 *   recurrences run in lockstep and each carries a seed sum across bars.
 *
 * ARITHMETIC taken verbatim:
 *   * the RMA seed is `sum / period` after exactly `period` UPDATES (:319) --
 *     counted in updates, not in bars, so a reset restarts the count.
 *   * the RMA step is `((prev * (period - 1)) + input) / period` (:312) --
 *     one multiply, one add, one divide, in that order. NOT the algebraically
 *     equal `prev + (input - prev)/period`, which rounds differently.
 *   * `gain = delta.max(0.0) * volume` and `loss = (-delta).max(0.0) * volume`
 *     (:1173-1174) -- f64::max, hence fmax.
 *
 * EPSILON: `EPS` is 1e-12 (:35) and it is ALREADY an f64-sized tolerance in
 *   the CPU source -- carried across unchanged, not rescaled from an f32
 *   value. It guards `vol_avg`, `down` and `up` by MAGNITUDE (`abs() <= EPS`),
 *   which is what is reproduced.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:16340-16344. */
#define NEO_VWRSI_RSI_LENGTH 14
/* volume_weighted_relative_strength_index.rs:35 -- already f64-sized. */
#define NEO_VWRSI_EPS 1e-12

extern "C" __global__
void volume_weighted_relative_strength_index_neo_batch_f64(const double* __restrict__ close,
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

    const int period = NEO_VWRSI_RSI_LENGTH;
    /* prepare_input refuses rsi_length > len. */
    if (period > n) return;
    const double pf = (double)period;

    bool   has_prev = false;
    double prev_source = 0.0;

    /* Three RmaState instances (:1149-1151). */
    int    up_count = 0,   dn_count = 0,   vol_count = 0;
    double up_sum = 0.0,   dn_sum = 0.0,   vol_sum = 0.0;
    bool   up_seeded = false, dn_seeded = false, vol_seeded = false;
    double up_val = 0.0,   dn_val = 0.0,   vol_val = 0.0;

    for (int i = 0; i < n; ++i) {
        const double src = close[i];
        const double vol = volume[i];

        if (!isfinite(src) || !isfinite(vol)) {
            has_prev = false;
            up_count = 0;  up_sum = 0.0;  up_seeded = false;
            dn_count = 0;  dn_sum = 0.0;  dn_seeded = false;
            vol_count = 0; vol_sum = 0.0; vol_seeded = false;
            o[i] = NEO_F64_NAN;
            continue;
        }

        if (!has_prev) {
            prev_source = src;
            has_prev = true;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double delta = src - prev_source;
        prev_source = src;

        const double gain = fmax(delta, 0.0) * vol;
        const double loss = fmax(-delta, 0.0) * vol;

        /* RmaState::update (:310) -- gain, then loss, then volume. */
        bool   up_ready = false; double up_num = 0.0;
        if (up_seeded) {
            up_val = ((up_val * (pf - 1.0)) + gain) / pf;
            up_num = up_val; up_ready = true;
        } else {
            up_count += 1; up_sum += gain;
            if (up_count == period) { up_val = up_sum / pf; up_seeded = true; up_num = up_val; up_ready = true; }
        }

        bool   dn_ready = false; double dn_num = 0.0;
        if (dn_seeded) {
            dn_val = ((dn_val * (pf - 1.0)) + loss) / pf;
            dn_num = dn_val; dn_ready = true;
        } else {
            dn_count += 1; dn_sum += loss;
            if (dn_count == period) { dn_val = dn_sum / pf; dn_seeded = true; dn_num = dn_val; dn_ready = true; }
        }

        bool   vol_ready = false; double vol_avg = 0.0;
        if (vol_seeded) {
            vol_val = ((vol_val * (pf - 1.0)) + vol) / pf;
            vol_avg = vol_val; vol_ready = true;
        } else {
            vol_count += 1; vol_sum += vol;
            if (vol_count == period) { vol_val = vol_sum / pf; vol_seeded = true; vol_avg = vol_val; vol_ready = true; }
        }

        if (!(up_ready && dn_ready && vol_ready)) { o[i] = NEO_F64_NAN; continue; }
        if (fabs(vol_avg) <= NEO_VWRSI_EPS)       { o[i] = NEO_F64_NAN; continue; }

        const double up   = up_num / vol_avg;
        const double down = dn_num / vol_avg;

        if (fabs(down) <= NEO_VWRSI_EPS)      o[i] = 100.0;
        else if (fabs(up) <= NEO_VWRSI_EPS)   o[i] = 0.0;
        else                                  o[i] = 100.0 - (100.0 / (1.0 + up / down));
    }
}
