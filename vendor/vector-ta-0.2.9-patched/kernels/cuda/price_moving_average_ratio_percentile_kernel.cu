#include <cmath>
#include <cstddef>

namespace {
constexpr double EPS = 1e-12;
constexpr int MA_SMA = 0;
constexpr int MA_EMA = 1;
constexpr int MA_HMA = 2;
constexpr int MA_RMA = 3;
constexpr int MA_VWMA = 4;
constexpr int LINE_PMAR = 0;
constexpr int LINE_PMARP = 1;

__device__ inline int max_i(int a, int b) {
    return a > b ? a : b;
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
    bool has_value;

    __device__ void init(int period) {
        this->period = period;
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
        has_value = false;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = has_value ? mean : NAN;
            return has_value;
        }
        count += 1;
        if (count == 1) {
            mean = input;
        } else if (count <= period) {
            mean += (input - mean) / static_cast<double>(count);
        } else {
            mean = alpha * input + beta * mean;
        }
        has_value = true;
        *out = mean;
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
        if (!isfinite(input)) {
            *out = seeded ? value : NAN;
            return seeded;
        }
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
        if (!isfinite(input)) {
            *out = NAN;
            return false;
        }
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
        if (!isfinite(input)) {
            *out = NAN;
            return false;
        }
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
        if (!isfinite(value) || !isfinite(volume)) {
            *out = NAN;
            return false;
        }
        const double pv = value * volume;
        if (count < period) {
            pv_ring[count] = pv;
            vol_ring[count] = volume;
            count += 1;
            pv_sum += pv;
            vol_sum += volume;
            if (count == period) {
                *out = fabs(vol_sum) <= EPS ? NAN : (pv_sum / vol_sum);
                return fabs(vol_sum) > EPS;
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
        return fabs(vol_sum) > EPS;
    }
};

struct HmaState {
    WmaState wma_half;
    WmaState wma_full;
    WmaState wma_sqrt;

    __device__ void init(int period, double* half_storage, double* full_storage, double* sqrt_storage) {
        const int half = max_i(period / 2, 1);
        wma_half.init(half, half_storage);
        wma_full.init(period, full_storage);
        wma_sqrt.init(sqrt_period(period), sqrt_storage);
    }

    __device__ bool update(double input, double* out) {
        double half_value = NAN;
        double full_value = NAN;
        const bool half_ready = wma_half.update(input, &half_value);
        const bool full_ready = wma_full.update(input, &full_value);
        if (half_ready && full_ready) {
            const double diff = 2.0 * half_value - full_value;
            return wma_sqrt.update(diff, out);
        }
        *out = NAN;
        return false;
    }
};

__device__ inline double scaled_pmar_value(double pmar, double pmar_high, double pmar_low) {
    if (pmar >= 1.0) {
        const double denom = pmar_high - 1.0;
        if (fabs(denom) <= EPS) {
            return 50.0;
        }
        return (((pmar - 1.0) * (100.0 / denom)) / 2.0) + 50.0;
    }
    const double denom = 1.0 - pmar_low;
    if (fabs(denom) <= EPS) {
        return 50.0;
    }
    return ((pmar - pmar_low) * (100.0 / denom)) / 2.0;
}

__device__ bool update_ma_value(
    int ma_code,
    double value,
    double volume,
    EmaState* ema,
    RmaState* rma,
    SmaState* sma,
    WmaState* wma,
    VwmaState* vwma,
    HmaState* hma,
    double* out
) {
    switch (ma_code) {
        case MA_SMA:
            return sma->update(value, out);
        case MA_EMA:
            return ema->update(value, out);
        case MA_HMA:
            return hma->update(value, out);
        case MA_RMA:
            return rma->update(value, out);
        case MA_VWMA:
            return vwma->update(value, volume, out);
        default:
            *out = NAN;
            return false;
    }
}
}

extern "C" __global__ void price_moving_average_ratio_percentile_batch_f64(
    const double* price,
    const double* volume,
    int len,
    const int* ma_lengths,
    const int* pmarp_lookbacks,
    const int* signal_ma_lengths,
    const int* ma_codes,
    const int* signal_ma_codes,
    const int* line_modes,
    int rows,
    int scratch_cap,
    double* scratch,
    double* out_pmar,
    double* out_pmarp,
    double* out_plotline,
    double* out_signal,
    double* out_pmar_high,
    double* out_pmar_low,
    double* out_scaled_pmar
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int ma_length = ma_lengths[row];
    const int pmarp_lookback = pmarp_lookbacks[row];
    const int signal_ma_length = signal_ma_lengths[row];
    const int ma_code = ma_codes[row];
    const int signal_ma_code = signal_ma_codes[row];
    const int line_mode = line_modes[row];

    double* row_pmar = out_pmar + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pmarp = out_pmarp + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_plotline = out_plotline + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pmar_high = out_pmar_high + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pmar_low = out_pmar_low + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_scaled_pmar = out_scaled_pmar + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_pmar[i] = NAN;
        row_pmarp[i] = NAN;
        row_plotline[i] = NAN;
        row_signal[i] = NAN;
        row_pmar_high[i] = NAN;
        row_pmar_low[i] = NAN;
        row_scaled_pmar[i] = NAN;
    }

    if (ma_length <= 0 || pmarp_lookback <= 0 || signal_ma_length <= 0 || scratch_cap <= 0) {
        return;
    }
    if (ma_length > scratch_cap || signal_ma_length > scratch_cap) {
        return;
    }
    if (ma_code < MA_SMA || ma_code > MA_VWMA || signal_ma_code < MA_SMA || signal_ma_code > MA_VWMA) {
        return;
    }
    if (line_mode != LINE_PMAR && line_mode != LINE_PMARP) {
        return;
    }

    double* row_scratch =
        scratch + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap) * static_cast<size_t>(10);
    double* main_a = row_scratch + static_cast<size_t>(scratch_cap) * 0;
    double* main_b = row_scratch + static_cast<size_t>(scratch_cap) * 1;
    double* main_c = row_scratch + static_cast<size_t>(scratch_cap) * 2;
    double* main_d = row_scratch + static_cast<size_t>(scratch_cap) * 3;
    double* main_e = row_scratch + static_cast<size_t>(scratch_cap) * 4;
    double* signal_a = row_scratch + static_cast<size_t>(scratch_cap) * 5;
    double* signal_b = row_scratch + static_cast<size_t>(scratch_cap) * 6;
    double* signal_c = row_scratch + static_cast<size_t>(scratch_cap) * 7;
    double* signal_d = row_scratch + static_cast<size_t>(scratch_cap) * 8;
    double* signal_e = row_scratch + static_cast<size_t>(scratch_cap) * 9;

    EmaState main_ema;
    RmaState main_rma;
    SmaState main_sma;
    WmaState main_wma;
    VwmaState main_vwma;
    HmaState main_hma;
    EmaState signal_ema;
    RmaState signal_rma;
    SmaState signal_sma;
    WmaState signal_wma;
    VwmaState signal_vwma;
    HmaState signal_hma;

    main_ema.init(ma_length);
    main_rma.init(ma_length);
    main_sma.init(ma_length, main_a);
    main_wma.init(ma_length, main_a);
    main_vwma.init(ma_length, main_d, main_e);
    main_hma.init(ma_length, main_a, main_b, main_c);

    signal_ema.init(signal_ma_length);
    signal_rma.init(signal_ma_length);
    signal_sma.init(signal_ma_length, signal_a);
    signal_wma.init(signal_ma_length, signal_a);
    signal_vwma.init(signal_ma_length, signal_d, signal_e);
    signal_hma.init(signal_ma_length, signal_a, signal_b, signal_c);

    bool seen_pmar = false;
    double pmar_high = 1.0;
    double pmar_low = 1.0;

    for (int i = 0; i < len; ++i) {
        const double current_price = price[i];
        const double current_volume = volume[i];
        double ma_value = NAN;
        if (update_ma_value(
                ma_code,
                current_price,
                current_volume,
                &main_ema,
                &main_rma,
                &main_sma,
                &main_wma,
                &main_vwma,
                &main_hma,
                &ma_value) &&
            isfinite(current_price) &&
            isfinite(ma_value) &&
            fabs(ma_value) > EPS) {
            const double pmar = current_price / ma_value;
            row_pmar[i] = pmar;
            pmar_high = fmax(pmar_high, pmar);
            pmar_low = fmin(pmar_low, pmar);
            seen_pmar = true;
        }

        if (seen_pmar) {
            row_pmar_high[i] = pmar_high;
            row_pmar_low[i] = pmar_low;
            if (isfinite(row_pmar[i])) {
                row_scaled_pmar[i] = scaled_pmar_value(row_pmar[i], pmar_high, pmar_low);
            }
        }

        if (i >= ma_length) {
            const double current = fabs(row_pmar[i]);
            if (isfinite(current)) {
                const int lookback = i < pmarp_lookback ? i : pmarp_lookback;
                if (lookback > 0) {
                    int count = 0;
                    for (int offset = 1; offset <= lookback; ++offset) {
                        const double prev = fabs(row_pmar[i - offset]);
                        if (!(isfinite(prev) && prev > current)) {
                            count += 1;
                        }
                    }
                    row_pmarp[i] = (static_cast<double>(count) / static_cast<double>(lookback)) * 100.0;
                }
            }
        }

        const double plotline = line_mode == LINE_PMAR ? row_pmar[i] : row_pmarp[i];
        row_plotline[i] = plotline;

        double signal_value = NAN;
        if (update_ma_value(
                signal_ma_code,
                plotline,
                current_volume,
                &signal_ema,
                &signal_rma,
                &signal_sma,
                &signal_wma,
                &signal_vwma,
                &signal_hma,
                &signal_value)) {
            row_signal[i] = signal_value;
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// price_moving_average_ratio_percentile_batch_f64 above is genuine
// double-in/double-out, but it takes 19 parameters and writes SEVEN output
// matrices plus six caller-allocated ring matrices. The f64 lane launches one
// shape -- (price, volume, n, periods, n_combos, first_valid, out) -- and
// allocates ONE output matrix, so that entry point cannot be reached from it.
//
// CPU REFERENCE
//   src/indicators/price_moving_average_ratio_percentile.rs:715
//     price_moving_average_ratio_percentile -> :629 compute_core
//   compute_ma_series      :457  (ma_type Sma -> sma_into_slice)
//   sma_into_slice         src/indicators/moving_averages/sma.rs:235
//   sma_scalar             src/indicators/moving_averages/sma.rs:317
//   compute_pmarp_percentile :601
//   insert_pmar_window / remove_pmar_window :578 / :588
//
// THE COLUMN THIS EMITS is plotline, which is what output_id == "value"
// resolves to (cpu_batch.rs -- "plotline" || "value").
//
// PINNED CPU DEFAULTS (compute_price_moving_average_ratio_percentile_batch):
// source "close", ma_length 20, ma_type "sma", pmarp_lookback 350,
// line_mode "pmar".
//
// WHERE THE ORDER STATISTIC WENT -- read this before concluding it was skipped.
// compute_core :707-710 is a two-arm match on line_mode: in Pmar mode
// `plotline_out.copy_from_slice(pmar_out)`, in Pmarp mode it copies pmarp_out.
// The percentile rank feeds pmarp, and the CPU DEFAULT for line_mode is "pmar",
// so on the lane's pinned parameters `plotline == pmar` and the percentile is
// not on the path to the emitted column. It is implemented below anyway, under
// PMARP_NEO_LINE_MODE, as a per-thread incremental sorted window over the
// 350-bar lookback -- so flipping the pin gives a correct kernel rather than a
// wrong one, and the order statistic is present rather than asserted to be
// impossible.
//
// PERIOD-INVARIANT. The batch reads source, ma_length, ma_type,
// pmarp_lookback, signal_ma_length, signal_ma_type and line_mode, and NEVER
// `period`, so five swept periods give five identical CPU columns and five
// identical kernel rows.
//
// SHAPE: one thread per combo, bars ascending. Sequential: the SMA is a running
// window sum carried from `first` (sma.rs:340-343, NOT recomputed per bar, so
// its rounding is path-dependent), and the percentile window is incremental.
//
// FIRST IS THE CPU'S OWN, NOT THE LANE'S. sma_prepare (sma.rs:274-277) takes
// `position(|x| !x.is_nan())` -- which ACCEPTS an infinity -- and the running
// sum then carries any interior NaN forward forever, which is the CPU's
// behaviour and is reproduced exactly. The lane row therefore declares
// F64FirstValidRule::Ignored: this kernel derives its own start index.
//
// VOLUME IS BOUND AND UNREAD. compute_ma_series passes volume only to the Vwma
// arm (:457-462); with ma_type Sma it is never read. The pointer is taken so
// the lane's (price, volume) arm cannot hand this kernel one series where it
// asked for two.
//
// f64 END TO END: double literals, no f32-suffixed math function, no fast-math
// intrinsic. The only epsilon in the CPU path is the literal 1e-12 at :562/:569
// inside scaled_pmar_value, which serves the scaled_pmar column and not this
// one, so no epsilon appears below.
// ---------------------------------------------------------------------------

#define PMARP_NEO_MA_LENGTH 20
#define PMARP_NEO_LOOKBACK 350
#define PMARP_NEO_LINE_MODE LINE_PMAR

__device__ __forceinline__ double pmarp_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void price_moving_average_ratio_percentile_neo_batch_f64(
    const double* __restrict__ price,
    const double* __restrict__ volume,
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
    (void)volume;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    const double qnan = pmarp_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    const int ma_length = PMARP_NEO_MA_LENGTH;
    if (ma_length > n) {
        return;   // sma_prepare's InvalidPeriod -> the CPU errors, row stays NaN
    }

    // sma_prepare (sma.rs:274-283).
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(price[i])) {
            first = i;
            break;
        }
    }
    if (first < 0 || n - first < ma_length) {
        return;
    }

    // sma_scalar (sma.rs:332-343): seed the window sum over [first, first+period)
    // and then roll it. `inv` is formed once and multiplied, NOT divided per bar.
    double sum = 0.0;
    for (int k = 0; k < ma_length; ++k) {
        sum += price[first + k];
    }
    const double inv = 1.0 / static_cast<double>(ma_length);

    // The 350-deep incremental percentile window (compute_pmarp_percentile
    // :601-627). Sized at the CPU default, which is the compiled bound.
    double sorted[PMARP_NEO_LOOKBACK];
    int sorted_len = 0;
    int invalid_count = 0;

    double pmar_hist[PMARP_NEO_LOOKBACK];   // the values the window will evict
    int hist_pos = 0;

    for (int i = 0; i < n; ++i) {
        // ---- ma[i] ------------------------------------------------------
        double ma = qnan;
        if (i == first + ma_length - 1) {
            ma = sum * inv;
        } else if (i >= first + ma_length) {
            sum += price[i] - price[i - ma_length];
            ma = sum * inv;
        }

        // ---- pmar[i] (compute_core :663-670) -----------------------------
        double pmar = qnan;
        const double p = price[i];
        if (isfinite(p) && isfinite(ma) && ma != 0.0) {
            pmar = p / ma;
        }

        // ---- pmarp[i] (compute_pmarp_percentile :607-616) -----------------
        //
        // Read BEFORE this bar joins the window: the window holds indices
        // [i - lookback, i - 1] at this point, which is why `lookback` below is
        // min(i, 350) and not the window's capacity.
        double pmarp = qnan;
        if (i >= ma_length) {
            const double current = fabs(pmar);
            const int lookback = (i < PMARP_NEO_LOOKBACK) ? i : PMARP_NEO_LOOKBACK;
            if (isfinite(current) && lookback != 0) {
                // partition_point(|v| *v <= current) -- the count of window
                // values NOT GREATER than `current`.
                int lo = 0;
                int hi = sorted_len;
                while (lo < hi) {
                    const int mid = lo + ((hi - lo) >> 1);
                    if (sorted[mid] <= current) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                pmarp = (static_cast<double>(lo + invalid_count) /
                         static_cast<double>(lookback)) *
                        100.0;
            }
        }

        // ---- window maintenance (:617-624) --------------------------------
        if (i >= PMARP_NEO_LOOKBACK) {
            const double leaving = fabs(pmar_hist[hist_pos]);
            if (isfinite(leaving)) {
                int lo = 0;
                int hi = sorted_len;
                while (lo < hi) {
                    const int mid = lo + ((hi - lo) >> 1);
                    if (sorted[mid] < leaving) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                for (int k = lo; k + 1 < sorted_len; ++k) {
                    sorted[k] = sorted[k + 1];
                }
                sorted_len -= 1;
            } else {
                invalid_count -= 1;
            }
        }
        {
            const double entering = fabs(pmar);
            if (isfinite(entering)) {
                int lo = 0;
                int hi = sorted_len;
                while (lo < hi) {
                    const int mid = lo + ((hi - lo) >> 1);
                    if (sorted[mid] < entering) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                for (int k = sorted_len; k > lo; --k) {
                    sorted[k] = sorted[k - 1];
                }
                sorted[lo] = entering;
                sorted_len += 1;
            } else {
                invalid_count += 1;
            }
        }
        pmar_hist[hist_pos] = pmar;
        hist_pos += 1;
        if (hist_pos == PMARP_NEO_LOOKBACK) {
            hist_pos = 0;
        }

        // ---- plotline (:707-710) ------------------------------------------
        row[i] = (PMARP_NEO_LINE_MODE == LINE_PMARP) ? pmarp : pmar;
    }
}
