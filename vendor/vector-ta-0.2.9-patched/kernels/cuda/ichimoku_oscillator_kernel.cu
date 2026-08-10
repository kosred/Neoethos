// ichimoku_oscillator — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line:  extern "C" __global__ void ichimoku_oscillator_batch_f64() {}
// plus a wrapper that resolved that empty symbol, computed all THIRTEEN output
// series on the host, and uploaded them so the caller believed the card had
// produced them.
//
// CPU REFERENCE
// -------------
//   src/indicators/ichimoku_oscillator.rs
//     :567 avg_if_finite            :576 diff_if_finite
//     :585 gaussian_value           :589 rolling_midpoint
//     :638 chebyshev_series         :657 gaussian_kernel_series
//     :682 smooth_series            :691 wma_series
//     :712 shift_back               :724 rolling_rms_window
//     :747 rms_all                  :763 normalize_value
//     :787 compute_core             <- the whole pipeline
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW. The pipeline is a chain of full-length passes
// where every stage feeds the next, and two of those stages are recurrences:
// `chebyshev_series` is a one-pole IIR (`out[i] = (1-c)*data[i] + c*out[i-1]`,
// :649) and `rolling_rms_window` carries a running sum that RESETS on a
// non-finite bar (:731). Neither can be re-derived bar-locally, so a thread
// owns a row and walks it in the reference's order.
//
// The thread's working set is fourteen `cols`-long scratch arrays, which is why
// the launch is planned in SLOTS: the host sizes the slot count from free VRAM
// and each thread loops `row = slot; row < rows; row += slots`. A wider sweep
// costs passes, never memory.
//
// TWO PLACES THE CPU IS SUBTLER THAN IT LOOKS
// -------------------------------------------
// * `rolling_midpoint` (:589) SKIPS a bar whose high or low is non-finite with
//   `continue` and does NOT clear its monotonic deques, so state survives the
//   hole. Reproduced literally.
// * `rolling_rms_window` (:724) pops the STORED square, not a recomputed one.
//   `data[i] * data[i]` is deterministic, so recomputing the departing square
//   from `data[qstart]` is bit-identical and removes the need for a ring — but
//   only because the reset makes `qstart` unambiguous, which is why the reset
//   is tracked explicitly below.
//
// ARITHMETIC
// ----------
// f64 throughout; the file is in `F64_LANE_SOURCES`, so it is compiled
// `-fmad=false -prec-div=true -prec-sqrt=true` and never with `--use_fast_math`.
// `fma()` appears once, in `chebyshev_series`, because the CPU writes
// `one_minus_c.mul_add(data[i], c * prev)` there and nowhere else in this file.

#include <cmath>
#include <cstdint>

#define ICH_NORM_ALL      0
#define ICH_NORM_WINDOW   1
#define ICH_NORM_DISABLED 2

// Fourteen `cols`-wide scratch arrays per slot.
#define ICH_A_CONV_RAW   0
#define ICH_A_BASE_RAW   1
#define ICH_A_SPANB_RAW  2
#define ICH_A_KUMO_A     3
#define ICH_A_KUMO_B     4
#define ICH_A_KUMO_C     5
#define ICH_A_CHIKOU     6
#define ICH_A_SIGNAL     7
#define ICH_A_CONV       8
#define ICH_A_BASE       9
#define ICH_A_MA        10
#define ICH_A_DEV       11
#define ICH_A_TMP_IN    12
#define ICH_A_TMP_CHEB  13
#define ICH_ARRAYS      14

__device__ __forceinline__ double ich_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// avg_if_finite (:567)
__device__ __forceinline__ double ich_avg(double a, double b) {
    return (isfinite(a) && isfinite(b)) ? 0.5 * (a + b) : ich_qnan();
}

// diff_if_finite (:576)
__device__ __forceinline__ double ich_diff(double a, double b) {
    return (isfinite(a) && isfinite(b)) ? a - b : ich_qnan();
}

// gaussian_value (:585)
__device__ __forceinline__ double ich_gaussian(double x, double bandwidth) {
    double ratio = x / bandwidth;
    return exp(-(ratio * ratio) * 0.5) / sqrt(2.0 * M_PI);
}

// rolling_midpoint (:589). `maxq`/`minq` are index rings of `length + 1`.
__device__ void ich_rolling_midpoint(
    const double* high, const double* low, int n, int length, int first,
    int* maxq, int* minq, int cap, double* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = ich_qnan();
    }
    int max_head = 0, max_len = 0, min_head = 0, min_len = 0;
    int warm = first + length - 1;

    for (int i = first; i < n; ++i) {
        if (!isfinite(high[i]) || !isfinite(low[i])) {
            continue;
        }
        // `i.saturating_add(1).saturating_sub(length).max(first)`
        int start = i + 1 - length;
        if (start < first) {
            start = first;
        }
        while (max_len > 0 && maxq[max_head] < start) {
            max_head += 1;
            if (max_head == cap) {
                max_head = 0;
            }
            max_len -= 1;
        }
        while (min_len > 0 && minq[min_head] < start) {
            min_head += 1;
            if (min_head == cap) {
                min_head = 0;
            }
            min_len -= 1;
        }
        while (max_len > 0) {
            int back = max_head + max_len - 1;
            if (back >= cap) {
                back -= cap;
            }
            if (high[maxq[back]] <= high[i]) {
                max_len -= 1;
            } else {
                break;
            }
        }
        while (min_len > 0) {
            int back = min_head + min_len - 1;
            if (back >= cap) {
                back -= cap;
            }
            if (low[minq[back]] >= low[i]) {
                min_len -= 1;
            } else {
                break;
            }
        }
        {
            int tail = max_head + max_len;
            if (tail >= cap) {
                tail -= cap;
            }
            maxq[tail] = i;
            max_len += 1;
        }
        {
            int tail = min_head + min_len;
            if (tail >= cap) {
                tail -= cap;
            }
            minq[tail] = i;
            min_len += 1;
        }
        if (i >= warm) {
            out[i] = 0.5 * (high[maxq[max_head]] + low[minq[min_head]]);
        }
    }
}

// chebyshev_series (:638) with the CPU's fixed ripple of 0.5.
__device__ void ich_chebyshev(const double* data, int n, int length, double* out) {
    double inv_len = 1.0 / static_cast<double>(length);
    double a = cosh(inv_len * acosh(1.0 / (1.0 - 0.5)));
    double b = sinh(inv_len * asinh(1.0 / 0.5));
    double c = (a - b) / (a + b);
    double one_minus_c = 1.0 - c;
    for (int i = 0; i < n; ++i) {
        out[i] = ich_qnan();
    }
    for (int i = 0; i < n; ++i) {
        if (isfinite(data[i])) {
            double prev = (i > 0 && isfinite(out[i - 1])) ? out[i - 1] : 0.0;
            // CPU: `one_minus_c.mul_add(data[i], c * prev)`
            out[i] = fma(one_minus_c, data[i], c * prev);
        }
    }
}

// gaussian_kernel_series (:657) with the CPU's fixed size=4, h=2.0, r=1.0.
__device__ void ich_gaussian_series(const double* data, int n, double* out) {
    const int size = 4;
    const double h = 2.0;
    const double r = 1.0;
    double weights[size + 1];
    for (int i = 0; i <= size; ++i) {
        weights[i] = ich_gaussian(static_cast<double>(i * i) / (h * h * r), r);
    }
    for (int i = 0; i < n; ++i) {
        out[i] = ich_qnan();
    }
    for (int i = size; i < n; ++i) {
        double sum = 0.0;
        double weight_sum = 0.0;
        bool ok = true;
        for (int j = 0; j <= size; ++j) {
            double value = data[i - j];
            if (!isfinite(value)) {
                ok = false;
                break;
            }
            sum += value * weights[j];
            weight_sum += weights[j];
        }
        if (ok && weight_sum != 0.0) {
            out[i] = sum / weight_sum;
        }
    }
}

// smooth_series (:682). `cheb` is scratch; `dst` receives the answer.
__device__ void ich_smooth(
    const double* data, int n, int length, int extra, double* cheb, double* dst) {
    if (extra) {
        ich_chebyshev(data, n, length, cheb);
        ich_gaussian_series(cheb, n, dst);
    } else {
        ich_chebyshev(data, n, length, dst);
    }
}

// wma_series (:691)
__device__ void ich_wma(const double* data, int n, int length, double* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = ich_qnan();
    }
    double denom = static_cast<double>(length * (length + 1) / 2);
    for (int i = length - 1; i < n; ++i) {
        double sum = 0.0;
        bool ok = true;
        for (int j = 0; j < length; ++j) {
            double value = data[i + 1 - length + j];
            if (!isfinite(value)) {
                ok = false;
                break;
            }
            sum += value * static_cast<double>(j + 1);
        }
        if (ok) {
            out[i] = sum / denom;
        }
    }
}

// normalize_value (:763)
__device__ __forceinline__ double ich_normalize(
    double value, double min_level, double max_level, int mode, int clamp_flag) {
    if (mode == ICH_NORM_DISABLED) {
        return value;
    }
    if (!isfinite(value) || !isfinite(min_level) || !isfinite(max_level) ||
        min_level == max_level) {
        return ich_qnan();
    }
    double scaled = (value - min_level) / (max_level - min_level);
    if (clamp_flag) {
        // `f64::clamp` is `if self < min { min } else if self > max { max }
        // else { self }` and panics on a NaN bound; both bounds are 0.0/1.0
        // here, and `scaled` is finite by the guard above.
        scaled = scaled < 0.0 ? 0.0 : (scaled > 1.0 ? 1.0 : scaled);
    }
    return (scaled - 0.5) * 200.0;
}

extern "C" __global__ void ichimoku_oscillator_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ source,
    int len,
    int first,
    const int* __restrict__ conversion_periods,
    const int* __restrict__ base_periods,
    const int* __restrict__ lagging_span_periods,
    const int* __restrict__ displacements,
    const int* __restrict__ ma_lengths,
    const int* __restrict__ smoothing_lengths,
    const int* __restrict__ window_sizes,
    const double* __restrict__ top_bands,
    const double* __restrict__ mid_bands,
    int extra_smoothing,
    int normalize_mode,
    int clamp_flag,
    int rows,
    int slots,
    int deque_cap,
    double* scratch,
    int* iscratch,
    double* __restrict__ out_signal,
    double* __restrict__ out_ma,
    double* __restrict__ out_conversion,
    double* __restrict__ out_base,
    double* __restrict__ out_chikou,
    double* __restrict__ out_current_kumo_a,
    double* __restrict__ out_current_kumo_b,
    double* __restrict__ out_future_kumo_a,
    double* __restrict__ out_future_kumo_b,
    double* __restrict__ out_max_level,
    double* __restrict__ out_high_level,
    double* __restrict__ out_low_level,
    double* __restrict__ out_min_level
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = ich_qnan();
    double* sbase = scratch + static_cast<size_t>(slot) * static_cast<size_t>(ICH_ARRAYS) *
                                  static_cast<size_t>(len);
    int* ibase = iscratch + static_cast<size_t>(slot) * 2ull * static_cast<size_t>(deque_cap);

    double* conv_raw = sbase + static_cast<size_t>(ICH_A_CONV_RAW) * len;
    double* base_raw = sbase + static_cast<size_t>(ICH_A_BASE_RAW) * len;
    double* spanb_raw = sbase + static_cast<size_t>(ICH_A_SPANB_RAW) * len;
    double* kumo_a = sbase + static_cast<size_t>(ICH_A_KUMO_A) * len;
    double* kumo_b = sbase + static_cast<size_t>(ICH_A_KUMO_B) * len;
    double* kumo_center = sbase + static_cast<size_t>(ICH_A_KUMO_C) * len;
    double* chikou = sbase + static_cast<size_t>(ICH_A_CHIKOU) * len;
    double* signal = sbase + static_cast<size_t>(ICH_A_SIGNAL) * len;
    double* conversion = sbase + static_cast<size_t>(ICH_A_CONV) * len;
    double* base_series = sbase + static_cast<size_t>(ICH_A_BASE) * len;
    double* ma = sbase + static_cast<size_t>(ICH_A_MA) * len;
    double* dev = sbase + static_cast<size_t>(ICH_A_DEV) * len;
    double* tmp_in = sbase + static_cast<size_t>(ICH_A_TMP_IN) * len;
    double* tmp_cheb = sbase + static_cast<size_t>(ICH_A_TMP_CHEB) * len;
    int* maxq = ibase;
    int* minq = ibase + deque_cap;

    for (int row = slot; row < rows; row += slots) {
        int conv_p = conversion_periods[row];
        int base_p = base_periods[row];
        int lag_p = lagging_span_periods[row];
        int displacement = displacements[row];
        int ma_length = ma_lengths[row];
        int smoothing = smoothing_lengths[row];
        int window = window_sizes[row];
        double top_band = top_bands[row];
        double mid_band = mid_bands[row];
        int shift = displacement > 0 ? displacement - 1 : 0;

        ich_rolling_midpoint(high, low, len, conv_p, first, maxq, minq, deque_cap, conv_raw);
        ich_rolling_midpoint(high, low, len, base_p, first, maxq, minq, deque_cap, base_raw);
        ich_rolling_midpoint(high, low, len, lag_p, first, maxq, minq, deque_cap, spanb_raw);

        for (int i = 0; i < len; ++i) {
            tmp_in[i] = ich_avg(conv_raw[i], base_raw[i]);
        }
        ich_smooth(tmp_in, len, smoothing, extra_smoothing, tmp_cheb, kumo_a);
        ich_smooth(spanb_raw, len, smoothing, extra_smoothing, tmp_cheb, kumo_b);

        for (int i = 0; i < len; ++i) {
            kumo_center[i] = ich_avg(kumo_a[i], kumo_b[i]);
        }

        // chikou_input[i] = source[i] - source[i - (displacement + 1)]
        int chikou_shift = displacement + 1;
        for (int i = 0; i < len; ++i) {
            double shifted = (i >= chikou_shift) ? source[i - chikou_shift] : nan_value;
            tmp_in[i] = ich_diff(source[i], shifted);
        }
        ich_smooth(tmp_in, len, smoothing, extra_smoothing, tmp_cheb, chikou);

        for (int i = 0; i < len; ++i) {
            double centre_offset = (i >= shift) ? kumo_center[i - shift] : nan_value;
            tmp_in[i] = ich_diff(source[i], centre_offset);
        }
        ich_smooth(tmp_in, len, smoothing, extra_smoothing, tmp_cheb, signal);

        for (int i = 0; i < len; ++i) {
            double centre_offset = (i >= shift) ? kumo_center[i - shift] : nan_value;
            tmp_in[i] = ich_diff(conv_raw[i], centre_offset);
        }
        ich_smooth(tmp_in, len, smoothing, extra_smoothing, tmp_cheb, conversion);

        for (int i = 0; i < len; ++i) {
            double centre_offset = (i >= shift) ? kumo_center[i - shift] : nan_value;
            tmp_in[i] = ich_diff(base_raw[i], centre_offset);
        }
        ich_smooth(tmp_in, len, smoothing, extra_smoothing, tmp_cheb, base_series);

        ich_wma(signal, len, ma_length, ma);

        // dev (:868)
        if (normalize_mode == ICH_NORM_ALL) {
            // rms_all(signal, kumo_a_offset) — the GATE is the shifted kumo_a.
            double sum_sq = 0.0;
            int count = 0;
            for (int i = 0; i < len; ++i) {
                dev[i] = nan_value;
                if (isfinite(signal[i])) {
                    sum_sq += signal[i] * signal[i];
                    count += 1;
                }
                double gate = (i >= shift) ? kumo_a[i - shift] : nan_value;
                if (isfinite(gate) && count != 0) {
                    dev[i] = sqrt(sum_sq / static_cast<double>(count));
                }
            }
        } else if (normalize_mode == ICH_NORM_WINDOW) {
            // rolling_rms_window(signal, window). The departing square is
            // recomputed from `signal[qstart]`, which is bit-identical to the
            // stored one because multiplication is deterministic.
            double sum_sq = 0.0;
            int qstart = 0;
            int qlen = 0;
            for (int i = 0; i < len; ++i) {
                dev[i] = nan_value;
                if (!isfinite(signal[i])) {
                    qstart = i + 1;
                    qlen = 0;
                    sum_sq = 0.0;
                    continue;
                }
                sum_sq += signal[i] * signal[i];
                qlen += 1;
                if (qlen > window) {
                    sum_sq -= signal[qstart] * signal[qstart];
                    qstart += 1;
                    qlen -= 1;
                }
                if (qlen == window && window > 1) {
                    dev[i] = sqrt(sum_sq / static_cast<double>(window - 1));
                }
            }
        } else {
            for (int i = 0; i < len; ++i) {
                dev[i] = 0.0;
            }
        }

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* o_signal = out_signal + row_base;
        double* o_ma = out_ma + row_base;
        double* o_conversion = out_conversion + row_base;
        double* o_base = out_base + row_base;
        double* o_chikou = out_chikou + row_base;
        double* o_cur_a = out_current_kumo_a + row_base;
        double* o_cur_b = out_current_kumo_b + row_base;
        double* o_fut_a = out_future_kumo_a + row_base;
        double* o_fut_b = out_future_kumo_b + row_base;
        double* o_max = out_max_level + row_base;
        double* o_high = out_high_level + row_base;
        double* o_low = out_low_level + row_base;
        double* o_min = out_min_level + row_base;

        for (int i = 0; i < len; ++i) {
            double d = dev[i];
            bool d_ok = isfinite(d);
            double max_level = d_ok ? d * top_band : nan_value;
            double min_level = d_ok ? -d * top_band : nan_value;
            double high_level = d_ok ? d * mid_band : nan_value;
            double low_level = d_ok ? -d * mid_band : nan_value;

            o_signal[i] =
                ich_normalize(signal[i], min_level, max_level, normalize_mode, clamp_flag);
            o_ma[i] = ich_normalize(ma[i], min_level, max_level, normalize_mode, clamp_flag);
            o_conversion[i] =
                ich_normalize(conversion[i], min_level, max_level, normalize_mode, clamp_flag);
            o_base[i] =
                ich_normalize(base_series[i], min_level, max_level, normalize_mode, clamp_flag);
            o_chikou[i] =
                ich_normalize(chikou[i], min_level, max_level, normalize_mode, clamp_flag);

            double fut_a = ich_diff(kumo_a[i], kumo_center[i]);
            double fut_b = ich_diff(kumo_b[i], kumo_center[i]);
            o_fut_a[i] = ich_normalize(fut_a, min_level, max_level, normalize_mode, clamp_flag);
            o_fut_b[i] = ich_normalize(fut_b, min_level, max_level, normalize_mode, clamp_flag);

            // `current_kumo_*` is written only from `shift` onwards, and it is
            // normalised against the levels at `i - shift`, not at `i` (:1013).
            if (i >= shift) {
                double a_off = kumo_a[i - shift];
                double b_off = kumo_b[i - shift];
                double c_off = kumo_center[i - shift];
                double cur_a = ich_diff(a_off, c_off);
                double cur_b = ich_diff(b_off, c_off);
                double d_shift = dev[i - shift];
                bool ds_ok = isfinite(d_shift);
                double max_shift = ds_ok ? d_shift * top_band : nan_value;
                double min_shift = ds_ok ? -d_shift * top_band : nan_value;
                o_cur_a[i] =
                    ich_normalize(cur_a, min_shift, max_shift, normalize_mode, clamp_flag);
                o_cur_b[i] =
                    ich_normalize(cur_b, min_shift, max_shift, normalize_mode, clamp_flag);
            } else {
                o_cur_a[i] = nan_value;
                o_cur_b[i] = nan_value;
            }

            o_max[i] = ich_normalize(max_level, min_level, max_level, normalize_mode, clamp_flag);
            o_high[i] =
                ich_normalize(high_level, min_level, max_level, normalize_mode, clamp_flag);
            o_low[i] = ich_normalize(low_level, min_level, max_level, normalize_mode, clamp_flag);
            o_min[i] = ich_normalize(min_level, min_level, max_level, normalize_mode, clamp_flag);
        }
    }
}
