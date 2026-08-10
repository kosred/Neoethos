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

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `ichimoku_oscillator_with_kernel`
// (src/indicators/ichimoku_oscillator.rs:1043) -> `compute_core` (:787), with
// `rolling_midpoint` (:589), `chebyshev_series` (:638),
// `gaussian_kernel_series` (:657), `smooth_series` (:698), `shift_back` (:710),
// `rolling_rms_window` (:723) and `normalize_value` (:753).
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `ichimoku_oscillator_batch_f64` (:278) is double-clean but declares
// thirty-six parameters and TWENTY `double*` -- thirteen output matrices plus
// scratch. The f64 lane launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// and reads back ONE matrix, so the lane gets its own entry point here.
//
// WHICH COLUMN: `signal`. `compute_ichimoku_oscillator_batch`
// (cpu_batch.rs:10906) maps output_id "signal" AND "value" onto `out.signal`,
// so this is the lane's `value` column, named rather than guessed.
//
// SHAPE -- AND WHY THE THIRTEEN FULL-LENGTH INTERMEDIATE VECTORS ARE NOT
// NEEDED. `compute_core` is written as a dozen whole-array passes, which reads
// like an algorithm that needs O(n) scratch per thread. It does not. Every
// stage on the path to `signal` is either a FINITE-SUPPORT window over the raw
// inputs or a CAUSAL filter with bounded state:
//   * the three `rolling_midpoint`s read `high`/`low` directly out of global
//     memory over a 9-, 26- and 52-bar window;
//   * `chebyshev_series` is a ONE-POLE IIR -- one carried double each;
//   * `gaussian_kernel_series` is a FIVE-TAP FIR over the Chebyshev output --
//     a five-slot ring each;
//   * `shift_back(kumo_center, displacement - 1)` is a 26-slot delay ring;
//   * `rolling_rms_window(signal, 20)` is a 20-slot ring of squares.
// So the whole chain is ONE forward pass with 3 + 3*(1 + 5) + 26 + 20 = 67
// doubles of state, 536 bytes per thread. One thread per combo, bars
// ascending. The Chebyshev poles alone make it non-bar-parallel.
//
// PERIOD-INVARIANT: the CPU batch reads `source`, `conversion_periods`,
// `base_periods`, `lagging_span_periods`, `displacement`, `ma_length`,
// `smoothing_length`, `extra_smoothing`, `normalize`, `window_size`, `clamp`,
// `top_band` and `mid_band` -- and NEVER `period` (cpu_batch.rs:10853-10875),
// so every swept period gives the same CPU column and this kernel writes
// identical rows. Pinned at the CPU defaults: source `close`, conversion 9,
// base 26, lagging span 52, displacement 26, smoothing_length 3,
// extra_smoothing true, normalize `window`, window_size 20, clamp true,
// top_band 2.0. `ma_length` and `mid_band` are read only by the `ma`,
// `high_level` and `low_level` columns and are not on this path.
//
// ROUNDING: `chebyshev_series` writes `one_minus_c.mul_add(data[i], c * prev)`
// (:651) -- ONE multiply and ONE FUSED multiply-add, TWO roundings -- so this
// kernel writes `fma(one_minus_c, x, c * prev)`. Everywhere else the CPU
// writes plain operators and so does this: the FIR accumulates
// `sum += value * weights[j]` in ASCENDING j (:673), the RMS ring adds the new
// square BEFORE subtracting the departing one (:735-741), and the midpoint is
// `0.5 * (high + low)` (:631). No `fma` is introduced where the CPU has none.
//
// EPSILON: none. The gate is `min_level == max_level` (:761), an exact
// comparison the CPU makes and this kernel makes, not a tolerance.
//
// NaN SEMANTICS: every stage the CPU guards with `is_finite` is guarded here
// the same way, and the guards are what produce the NaN prefix -- the FIR
// refuses a window containing one non-finite tap (:668-671), the RMS ring
// CLEARS on a non-finite sample (:728-732), and `normalize_value` returns NaN
// unless value, min and max are all finite and min != max (:757-763). There is
// no `f64::max` on this path, so rule 4 has nothing to catch; the rolling
// midpoint's max/min are over integer-indexed prices with the CPU's own
// `<=`/`>=` monotone-deque predicate, reproduced as an exact scan below.
//
// WHY A SCAN AND NOT A DEQUE for the rolling midpoint: the CPU's monotone
// deques (:591-629) compute the MAXIMUM high and MINIMUM low over the valid
// indices in `[max(i+1-length, first), i]`. A direct scan over that same range,
// skipping the same bars the CPU skips (`!high[j].is_finite() ||
// !low[j].is_finite()`, :596), returns the same two values exactly -- max and
// min are selections, not sums, so there is no accumulation order to preserve
// and no rounding to differ. 87 comparisons per bar for the three windows.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. `cosh`, `sinh`, `acosh`, `asinh`, `exp`, `sqrt` are the double
// overloads. The NaN is a DOUBLE quiet-NaN bit pattern.
//
// FIRST VALID: the kernel derives it with the CPU's own rule
// (`first_valid_hlcs`, :511 -- high, low, close all finite at the same index;
// `source` is `close` at the pinned default, so the fourth scan is the same
// series as the third). The lane row declares `F64FirstValidRule::Ignored`
// because the kernel does not read the caller's value.
// ---------------------------------------------------------------------------

#define NEO_ICHI_CONVERSION 9
#define NEO_ICHI_BASE 26
#define NEO_ICHI_SPAN_B 52
#define NEO_ICHI_DISPLACEMENT 26
#define NEO_ICHI_SHIFT (NEO_ICHI_DISPLACEMENT - 1)
#define NEO_ICHI_SMOOTHING 3
#define NEO_ICHI_GAUSS_SIZE 4
#define NEO_ICHI_GAUSS_TAPS (NEO_ICHI_GAUSS_SIZE + 1)
#define NEO_ICHI_WINDOW_SIZE 20
#define NEO_ICHI_TOP_BAND 2.0

__device__ inline double neo_ichi_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `avg_if_finite`, :567.
__device__ inline double neo_ichi_avg_if_finite(double a, double b) {
    return (isfinite(a) && isfinite(b)) ? (0.5 * (a + b)) : neo_ichi_qnan();
}

// `diff_if_finite`, :576.
__device__ inline double neo_ichi_diff_if_finite(double a, double b) {
    return (isfinite(a) && isfinite(b)) ? (a - b) : neo_ichi_qnan();
}

// `gaussian_value`, :585.
__device__ inline double neo_ichi_gaussian_value(double x, double bandwidth) {
    const double t = x / bandwidth;
    return exp(-(t * t) * 0.5) / sqrt(2.0 * 3.14159265358979323846);
}

// `rolling_midpoint`, :589-635, evaluated at ONE bar.
__device__ inline double neo_ichi_rolling_midpoint(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int length,
    int first,
    int i
) {
    const int warm = first + length - 1;
    if (i < warm) {
        return neo_ichi_qnan();
    }
    int start = i + 1 - length;
    if (start < first) {
        start = first;
    }
    double highest = neo_ichi_qnan();
    double lowest = neo_ichi_qnan();
    bool seen = false;
    for (int j = start; j <= i; ++j) {
        const double h = high[j];
        const double l = low[j];
        if (!isfinite(h) || !isfinite(l)) {
            continue;
        }
        if (!seen) {
            highest = h;
            lowest = l;
            seen = true;
        } else {
            if (h > highest) {
                highest = h;
            }
            if (l < lowest) {
                lowest = l;
            }
        }
    }
    if (!seen) {
        return neo_ichi_qnan();
    }
    return 0.5 * (highest + lowest);
}

// `smooth_series(data, 3, true)`, :698-704: a one-pole Chebyshev IIR followed
// by a five-tap Gaussian FIR. Both stages carried as bounded state.
struct NeoIchiSmoother {
    double one_minus_c;
    double c;
    double prev_out;      // `out[i - 1]` of the Chebyshev stage
    bool has_prev;
    double taps[NEO_ICHI_GAUSS_TAPS];
    int filled;           // how many Chebyshev outputs have been produced
    int head;             // next slot to overwrite
    const double* weights;
    double weight_sum;

    __device__ void init(const double* w, double wsum) {
        // `chebyshev_series`, :639-642, with len = 3 and ripple = 0.5.
        const double inv_len = 1.0 / static_cast<double>(NEO_ICHI_SMOOTHING);
        const double a = cosh(inv_len * acosh(1.0 / (1.0 - 0.5)));
        const double b = sinh(inv_len * asinh(1.0 / 0.5));
        c = (a - b) / (a + b);
        one_minus_c = 1.0 - c;
        prev_out = neo_ichi_qnan();
        has_prev = false;
        filled = 0;
        head = 0;
        weights = w;
        weight_sum = wsum;
        for (int i = 0; i < NEO_ICHI_GAUSS_TAPS; ++i) {
            taps[i] = neo_ichi_qnan();
        }
    }

    __device__ double update(double x) {
        // Chebyshev stage -- :644-653.
        double cheb;
        if (isfinite(x)) {
            const double prev = (has_prev && isfinite(prev_out)) ? prev_out : 0.0;
            cheb = fma(one_minus_c, x, c * prev);
        } else {
            cheb = neo_ichi_qnan();
        }
        prev_out = cheb;
        has_prev = true;

        taps[head] = cheb;
        head += 1;
        if (head == NEO_ICHI_GAUSS_TAPS) {
            head = 0;
        }
        if (filled < NEO_ICHI_GAUSS_TAPS) {
            filled += 1;
        }
        if (filled < NEO_ICHI_GAUSS_TAPS) {
            // `for i in size..data.len()` -- :661, no output before bar `size`.
            return neo_ichi_qnan();
        }

        // Gaussian stage -- :662-678. `j` counts BACKWARDS from the newest
        // sample, so tap j is `data[i - j]`.
        double sum = 0.0;
        int idx = head - 1;
        if (idx < 0) {
            idx = NEO_ICHI_GAUSS_TAPS - 1;
        }
        for (int j = 0; j < NEO_ICHI_GAUSS_TAPS; ++j) {
            const double value = taps[idx];
            if (!isfinite(value)) {
                return neo_ichi_qnan();
            }
            sum += value * weights[j];
            idx -= 1;
            if (idx < 0) {
                idx = NEO_ICHI_GAUSS_TAPS - 1;
            }
        }
        if (weight_sum == 0.0) {
            return neo_ichi_qnan();
        }
        return sum / weight_sum;
    }
};

// `normalize_value` with mode = Window and clamp = true, :753-770.
__device__ inline double neo_ichi_normalize(double value, double min_level, double max_level) {
    if (!isfinite(value) || !isfinite(min_level) || !isfinite(max_level) ||
        min_level == max_level) {
        return neo_ichi_qnan();
    }
    double scaled = (value - min_level) / (max_level - min_level);
    if (scaled < 0.0) {
        scaled = 0.0;
    } else if (scaled > 1.0) {
        scaled = 1.0;
    }
    return (scaled - 0.5) * 200.0;
}

extern "C" __global__ void ichimoku_oscillator_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
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

    const double nan_value = neo_ichi_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = nan_value;
    }

    // `first_valid_hlcs`, :511. `source` is `close` at the pinned default, so
    // the CPU's fourth scan is the same series as its third.
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    // `gaussian_kernel_series` weights, :658-660: size 4, h 2.0, r 1.0.
    double weights[NEO_ICHI_GAUSS_TAPS];
    double weight_sum = 0.0;
    for (int j = 0; j < NEO_ICHI_GAUSS_TAPS; ++j) {
        const double x = static_cast<double>(j * j) / (2.0 * 2.0 * 1.0);
        weights[j] = neo_ichi_gaussian_value(x, 1.0);
        weight_sum += weights[j];
    }

    NeoIchiSmoother smooth_kumo_a;
    NeoIchiSmoother smooth_kumo_b;
    NeoIchiSmoother smooth_signal;
    smooth_kumo_a.init(weights, weight_sum);
    smooth_kumo_b.init(weights, weight_sum);
    smooth_signal.init(weights, weight_sum);

    // `shift_back(kumo_center, displacement - 1)`, :710-721. A ring of exactly
    // `shift` slots read BEFORE it is written gives the value from `shift`
    // bars back, which is `kumo_center[i - 25]`.
    double center_delay[NEO_ICHI_SHIFT];
    for (int i = 0; i < NEO_ICHI_SHIFT; ++i) {
        center_delay[i] = nan_value;
    }
    int delay_head = 0;

    // `rolling_rms_window(signal, 20)`, :723-746.
    double rms_ring[NEO_ICHI_WINDOW_SIZE];
    int rms_head = 0;
    int rms_len = 0;
    double rms_sum_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        const double conversion_raw =
            neo_ichi_rolling_midpoint(high, low, NEO_ICHI_CONVERSION, first, i);
        const double base_raw =
            neo_ichi_rolling_midpoint(high, low, NEO_ICHI_BASE, first, i);
        const double span_b_raw =
            neo_ichi_rolling_midpoint(high, low, NEO_ICHI_SPAN_B, first, i);

        const double kumo_a = smooth_kumo_a.update(
            neo_ichi_avg_if_finite(conversion_raw, base_raw));
        const double kumo_b = smooth_kumo_b.update(span_b_raw);
        const double kumo_center = neo_ichi_avg_if_finite(kumo_a, kumo_b);

        // `shift_back` with shift = 25: the value 25 bars back, NaN before.
        const double kumo_center_offset =
            (i >= NEO_ICHI_SHIFT) ? center_delay[delay_head] : nan_value;
        center_delay[delay_head] = kumo_center;
        delay_head += 1;
        if (delay_head == NEO_ICHI_SHIFT) {
            delay_head = 0;
        }

        const double signal_input =
            neo_ichi_diff_if_finite(close[i], kumo_center_offset);
        const double signal = smooth_signal.update(signal_input);

        // `rolling_rms_window`, :726-745.
        double dev = nan_value;
        if (!isfinite(signal)) {
            rms_head = 0;
            rms_len = 0;
            rms_sum_sq = 0.0;
        } else {
            const double sq = signal * signal;
            const double departing = rms_ring[rms_head];
            rms_ring[rms_head] = sq;
            rms_head += 1;
            if (rms_head == NEO_ICHI_WINDOW_SIZE) {
                rms_head = 0;
            }
            // The CPU pushes first and pops only when the queue EXCEEDS the
            // window, so the add happens BEFORE the subtract.
            rms_sum_sq += sq;
            if (rms_len < NEO_ICHI_WINDOW_SIZE) {
                rms_len += 1;
            } else {
                rms_sum_sq -= departing;
            }
            if (rms_len == NEO_ICHI_WINDOW_SIZE && NEO_ICHI_WINDOW_SIZE > 1) {
                dev = sqrt(rms_sum_sq /
                           (static_cast<double>(NEO_ICHI_WINDOW_SIZE) - 1.0));
            }
        }

        double max_level = nan_value;
        double min_level = nan_value;
        if (isfinite(dev)) {
            max_level = dev * NEO_ICHI_TOP_BAND;
            min_level = -dev * NEO_ICHI_TOP_BAND;
        }

        row[i] = neo_ichi_normalize(signal, min_level, max_level);
    }
}
