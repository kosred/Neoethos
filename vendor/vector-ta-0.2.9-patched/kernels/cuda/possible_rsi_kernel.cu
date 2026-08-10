// possible_rsi — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line:  extern "C" __global__ void possible_rsi_batch_f64() {}
// resolved by `possible_rsi_wrapper.rs:104` purely so symbol lookup succeeded,
// after which the wrapper computed all SEVEN output series on the host and
// uploaded them. This is the file the header of
// `src/indicators/dispatch/cuda_f64.rs` names as the canonical example of the
// disguise.
//
// CPU REFERENCE
// -------------
//   src/indicators/possible_rsi.rs
//     :670 for_each_finite_segment    :692 highpass_series
//     :719 cutler_rsi_series          :764 harris_rsi_series
//     :793 ehlers_smoothed_rsi_series :807 ema_valid_series
//     :827 compute_rsi_series         :874 rolling_min_max
//     :926 rolling_mean_std           :959 fisher_transform_series
//     :998 softmax_series            :1017 regular_norm_series
//    :1034 normalize_min_max         :1064 build_nonlag_weights
//    :1090 nonlag_ma_series          :1118 percentile_nearest_rank
//    :1129 rolling_percentile_series :1164 crossover  :1179 crossunder
//    :1193 compute_possible_rsi_output   <- the whole pipeline
//   src/indicators/rsi.rs:327  rsi_compute_into_scalar   (mode Regular / Slow)
//   src/indicators/rsx.rs:305  rsx_scalar                (mode Rsx)
//
// The two cross-indicator dependencies are inlined below rather than "called",
// because a kernel has no crate to call into; each is a line-for-line
// transliteration of the named scalar function, including the RSI's odd
// two-bars-per-iteration loop, whose ORDER is part of the answer.
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW. The pipeline is nine full-length passes and
// four of them are recurrences: the highpass 2-pole IIR (:704), the Wilder RSI
// (`rsi.rs:373`), the Fisher transform's `prev_value`/`prev_fish` (:983), and
// the RSX's eighteen carried state variables (`rsx.rs:306`). None can be
// re-derived bar-locally.
//
// ARITHMETIC
// ----------
// f64 throughout; in `F64_LANE_SOURCES`, so never `--use_fast_math`. `fma()`
// appears only in the RSI recursion, where `rsi.rs:376` writes
// `avg_gain.mul_add(beta, inv_p * g1)`. Every epsilon is the CPU's own
// `f64::EPSILON` (:975, :1006, :1044, `rsx.rs:392`) or its literal `1e-10`
// guard (`rsx.rs:395`) — none is an f32 tolerance carried across.
//
// ONE DELIBERATE DIVERGENCE, DOCUMENTED
// -------------------------------------
// The CPU batch worker (:1901) is `if let Ok(out) = possible_rsi(..)` — on an
// error it leaves the row holding a NaN warmup prefix followed by UNINITIALISED
// memory from `make_uninit_matrix`. This kernel writes NaN across the whole row
// instead. Reproducing uninitialised memory is not parity, it is a bug.

#include <cmath>
#include <cfloat>
#include <cstdint>

#define PR_MODE_RSX      0
#define PR_MODE_REGULAR  1
#define PR_MODE_SLOW     2
#define PR_MODE_RAPID    3
#define PR_MODE_HARRIS   4
#define PR_MODE_CUTLER   5
#define PR_MODE_EHLERS   6

#define PR_NORM_GAUSSIAN_FISHER 0
#define PR_NORM_SOFTMAX         1
#define PR_NORM_REGULAR         2

#define PR_SIG_SLOPE        0
#define PR_SIG_DYN_MIDDLE   1
#define PR_SIG_LEVELS       2
#define PR_SIG_ZEROLINE     3

// Six `cols`-wide scratch arrays per slot.
#define PR_A_SRC   0
#define PR_A_RSI   1
#define PR_A_SCALE 2
#define PR_A_NORM  3
#define PR_A_TMPA  4
#define PR_A_TMPB  5
#define PR_ARRAYS  6

__device__ __forceinline__ double pr_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

__device__ __forceinline__ void pr_fill_nan(double* a, int n) {
    double q = pr_qnan();
    for (int i = 0; i < n; ++i) {
        a[i] = q;
    }
}

// highpass_series (:692)
__device__ void pr_highpass(const double* data, int n, int period, double* out) {
    pr_fill_nan(out, n);
    double a1 = exp(-1.414 * M_PI / static_cast<double>(period));
    double b1 = 2.0 * a1 * cos(1.414 * M_PI / static_cast<double>(period));
    double c2 = b1;
    double c3 = -a1 * a1;
    double c1 = (1.0 + c2 - c3) / 4.0;

    int start = 0;
    while (start < n) {
        while (start < n && !isfinite(data[start])) {
            start += 1;
        }
        if (start >= n) {
            break;
        }
        int end = start + 1;
        while (end < n && isfinite(data[end])) {
            end += 1;
        }
        double hp1 = 0.0, hp2 = 0.0;
        for (int i = start; i < end; ++i) {
            if (i - start < 4) {
                out[i] = 0.0;
                hp2 = hp1;
                hp1 = 0.0;
                continue;
            }
            double hp = c1 * (data[i] - 2.0 * data[i - 1] + data[i - 2]) + c2 * hp1 + c3 * hp2;
            out[i] = hp;
            hp2 = hp1;
            hp1 = hp;
        }
        start = end;
    }
}

// cutler_rsi_series (:719)
__device__ void pr_cutler_rsi(const double* data, int n, int period, double* out) {
    pr_fill_nan(out, n);
    int start = 0;
    while (start < n) {
        while (start < n && !isfinite(data[start])) {
            start += 1;
        }
        if (start >= n) {
            break;
        }
        int end = start + 1;
        while (end < n && isfinite(data[end])) {
            end += 1;
        }
        if (end - start > period) {
            double gain = 0.0, loss = 0.0;
            for (int i = start + 1; i <= start + period; ++i) {
                double diff = data[i] - data[i - 1];
                if (diff > 0.0) {
                    gain += diff;
                } else {
                    loss += -diff;
                }
            }
            out[start + period] =
                (gain + loss == 0.0) ? 50.0 : 100.0 * gain / (gain + loss);
            for (int i = start + period + 1; i < end; ++i) {
                double old_diff = data[i - period] - data[i - period - 1];
                if (old_diff > 0.0) {
                    gain -= old_diff;
                } else {
                    loss -= -old_diff;
                }
                double new_diff = data[i] - data[i - 1];
                if (new_diff > 0.0) {
                    gain += new_diff;
                } else {
                    loss += -new_diff;
                }
                out[i] = (gain + loss == 0.0) ? 50.0 : 100.0 * gain / (gain + loss);
            }
        }
        start = end;
    }
}

// harris_rsi_series (:764)
__device__ void pr_harris_rsi(const double* data, int n, int period, double* out) {
    pr_fill_nan(out, n);
    int start = 0;
    while (start < n) {
        while (start < n && !isfinite(data[start])) {
            start += 1;
        }
        if (start >= n) {
            break;
        }
        int end = start + 1;
        while (end < n && isfinite(data[end])) {
            end += 1;
        }
        if (end - start > period) {
            for (int i = start + period; i < end; ++i) {
                double current = data[i];
                double up = 0.0, down = 0.0;
                for (int j = 1; j <= period; ++j) {
                    double diff = current - data[i - j];
                    if (diff > 0.0) {
                        up += diff;
                    } else {
                        down += -diff;
                    }
                }
                out[i] = (up + down == 0.0) ? 50.0 : 100.0 * up / (up + down);
            }
        }
        start = end;
    }
}

// ema_valid_series (:807)
__device__ void pr_ema_valid(const double* data, int n, int period, double* out) {
    pr_fill_nan(out, n);
    double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    double state = 0.0;
    bool seeded = false;
    for (int i = 0; i < n; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            seeded = false;
            continue;
        }
        double next = seeded ? state + alpha * (value - state) : value;
        out[i] = next;
        state = next;
        seeded = true;
    }
}

// rsi.rs:327 `rsi_compute_into_scalar`, plus the NaN prefix `rsi_into_slice`
// writes afterwards (rsi.rs:316). Returns 0 where the CPU returns Err.
__device__ int pr_rsi_regular(const double* data, int n, int period, double* out) {
    pr_fill_nan(out, n);
    // `first` is the first NON-NaN bar -- `!x.is_nan()`, which ACCEPTS an
    // infinity. Not `is_finite`. (rsi.rs:284)
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return 0;
    }
    if (period == 0 || period > n) {
        return 0;
    }
    if (n - first < period) {
        return 0;
    }

    double inv_p = 1.0 / static_cast<double>(period);
    double beta = 1.0 - inv_p;
    double avg_gain = 0.0, avg_loss = 0.0;
    bool has_nan = false;

    int warm_last = first + period;
    int upper = n > 0 ? n - 1 : 0;
    if (warm_last > upper) {
        warm_last = upper;
    }
    for (int i = first + 1; i <= warm_last; ++i) {
        double delta = data[i] - data[i - 1];
        if (!isfinite(delta)) {
            has_nan = true;
            break;
        }
        if (delta > 0.0) {
            avg_gain += delta;
        } else if (delta < 0.0) {
            avg_loss -= delta;
        }
    }

    int idx0 = first + period;
    if (has_nan) {
        avg_gain = pr_qnan();
        avg_loss = pr_qnan();
        if (idx0 < n) {
            out[idx0] = pr_qnan();
        }
    } else {
        avg_gain *= inv_p;
        avg_loss *= inv_p;
        if (idx0 < n) {
            double denom = avg_gain + avg_loss;
            out[idx0] = (denom == 0.0) ? 50.0 : 100.0 * avg_gain / denom;
        }
    }

    // The CPU walks TWO bars per iteration then mops up the last one. The
    // arithmetic per bar is identical either way, but the transliteration is
    // kept literal so a reviewer can diff it against rsi.rs:373-412.
    int j = idx0 + 1;
    while (j + 1 < n) {
        double d1 = data[j] - data[j - 1];
        double g1 = d1 > 0.0 ? d1 : 0.0;
        double l1 = d1 < 0.0 ? -d1 : 0.0;
        avg_gain = fma(avg_gain, beta, inv_p * g1);
        avg_loss = fma(avg_loss, beta, inv_p * l1);
        double denom1 = avg_gain + avg_loss;
        out[j] = (denom1 == 0.0) ? 50.0 : 100.0 * avg_gain / denom1;

        double d2 = data[j + 1] - data[j];
        double g2 = d2 > 0.0 ? d2 : 0.0;
        double l2 = d2 < 0.0 ? -d2 : 0.0;
        avg_gain = fma(avg_gain, beta, inv_p * g2);
        avg_loss = fma(avg_loss, beta, inv_p * l2);
        double denom2 = avg_gain + avg_loss;
        out[j + 1] = (denom2 == 0.0) ? 50.0 : 100.0 * avg_gain / denom2;

        j += 2;
    }
    if (j < n) {
        double d = data[j] - data[j - 1];
        double g = d > 0.0 ? d : 0.0;
        double l = d < 0.0 ? -d : 0.0;
        avg_gain = fma(avg_gain, beta, inv_p * g);
        avg_loss = fma(avg_loss, beta, inv_p * l);
        double denom = avg_gain + avg_loss;
        out[j] = (denom == 0.0) ? 50.0 : 100.0 * avg_gain / denom;
    }

    int warmup_end = first + period;
    if (warmup_end > n) {
        warmup_end = n;
    }
    for (int i = 0; i < warmup_end; ++i) {
        out[i] = pr_qnan();
    }
    return 1;
}

// rsx.rs:305 `rsx_scalar`, plus the NaN prefix `rsx_into_slice` writes (:283).
__device__ int pr_rsx(const double* data, int n, int period, double* out) {
    pr_fill_nan(out, n);
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0 || period == 0 || period > n || n - first < period) {
        return 0;
    }

    int start = first + period - 1;
    if (start >= n) {
        return 1;
    }

    double f0 = 0.0, f8 = 0.0, f18 = 0.0, f20 = 0.0, f28 = 0.0, f30 = 0.0;
    double f38 = 0.0, f40 = 0.0, f48 = 0.0, f50 = 0.0, f58 = 0.0, f60 = 0.0;
    double f68 = 0.0, f70 = 0.0, f78 = 0.0, f80 = 0.0, f88 = 0.0, f90 = 0.0;

    f90 = 1.0;
    f0 = 0.0;
    f88 = period >= 6 ? static_cast<double>(period - 1) : 5.0;
    f8 = 100.0 * data[start];
    f18 = 3.0 / (static_cast<double>(period) + 2.0);
    f20 = 1.0 - f18;
    out[start] = pr_qnan();

    for (int i = start + 1; i < n; ++i) {
        f90 = (f88 <= f90) ? f88 + 1.0 : f90 + 1.0;

        double prev = f8;
        f8 = 100.0 * data[i];
        double v8 = f8 - prev;

        f28 = f20 * f28 + f18 * v8;
        f30 = f18 * f28 + f20 * f30;
        double v_c = f28 * 1.5 - f30 * 0.5;

        f38 = f20 * f38 + f18 * v_c;
        f40 = f18 * f38 + f20 * f40;
        double v10 = f38 * 1.5 - f40 * 0.5;

        f48 = f20 * f48 + f18 * v10;
        f50 = f18 * f48 + f20 * f50;
        double v14 = f48 * 1.5 - f50 * 0.5;

        double av = fabs(v8);
        f58 = f20 * f58 + f18 * av;
        f60 = f18 * f58 + f20 * f60;
        double v18 = f58 * 1.5 - f60 * 0.5;

        f68 = f20 * f68 + f18 * v18;
        f70 = f18 * f68 + f20 * f70;
        double v1c = f68 * 1.5 - f70 * 0.5;

        f78 = f20 * f78 + f18 * v1c;
        f80 = f18 * f78 + f20 * f80;
        double v20_ = f78 * 1.5 - f80 * 0.5;

        if (f88 >= f90 && f8 != prev) {
            f0 = 1.0;
        }
        if (fabs(f88 - f90) < DBL_EPSILON && f0 == 0.0) {
            f90 = 0.0;
        }

        if (f88 < f90 && v20_ > 1e-10) {
            double v4 = (v14 / v20_ + 1.0) * 50.0;
            if (v4 > 100.0) {
                v4 = 100.0;
            }
            if (v4 < 0.0) {
                v4 = 0.0;
            }
            out[i] = v4;
        } else {
            out[i] = 50.0;
        }
    }

    int warmup_end = first + period - 1;
    if (warmup_end > n) {
        warmup_end = n;
    }
    for (int i = 0; i < warmup_end; ++i) {
        out[i] = pr_qnan();
    }
    return 1;
}

// compute_rsi_series (:827). `tmp` is one `cols`-wide scratch array.
__device__ int pr_compute_rsi(
    const double* src, int n, int mode, int period, double* tmp, double* out) {
    switch (mode) {
        case PR_MODE_REGULAR:
            return pr_rsi_regular(src, n, period, out);
        case PR_MODE_RSX:
            return pr_rsx(src, n, period, out);
        case PR_MODE_CUTLER:
        case PR_MODE_RAPID:
            pr_cutler_rsi(src, n, period, out);
            return 1;
        case PR_MODE_SLOW: {
            if (!pr_rsi_regular(src, n, period, tmp)) {
                return 0;
            }
            pr_ema_valid(tmp, n, period, out);
            return 1;
        }
        case PR_MODE_HARRIS:
            pr_harris_rsi(src, n, period, out);
            return 1;
        case PR_MODE_EHLERS: {
            // ehlers_smoothed_rsi_series (:793)
            pr_fill_nan(tmp, n);
            int start = 0;
            while (start < n) {
                while (start < n && !isfinite(src[start])) {
                    start += 1;
                }
                if (start >= n) {
                    break;
                }
                int end = start + 1;
                while (end < n && isfinite(src[end])) {
                    end += 1;
                }
                if (end - start >= 4) {
                    for (int i = start + 3; i < end; ++i) {
                        tmp[i] = (src[i] + 2.0 * src[i - 1] + 2.0 * src[i - 2] + src[i - 3]) / 6.0;
                    }
                }
                start = end;
            }
            pr_cutler_rsi(tmp, n, period, out);
            return 1;
        }
        default:
            return 0;
    }
}

// rolling_min_max (:874). `minq`/`maxq` are index rings of `cap >= period + 1`.
__device__ void pr_rolling_min_max(
    const double* data, int n, int period, int* minq, int* maxq, int cap,
    double* mins, double* maxs) {
    pr_fill_nan(mins, n);
    pr_fill_nan(maxs, n);
    int start = 0;
    while (start < n) {
        while (start < n && !isfinite(data[start])) {
            start += 1;
        }
        if (start >= n) {
            break;
        }
        int end = start + 1;
        while (end < n && isfinite(data[end])) {
            end += 1;
        }
        if (end - start >= period) {
            int min_head = 0, min_len = 0, max_head = 0, max_len = 0;
            for (int i = start; i < end; ++i) {
                while (min_len > 0 && minq[min_head] + period <= i) {
                    min_head += 1;
                    if (min_head == cap) {
                        min_head = 0;
                    }
                    min_len -= 1;
                }
                while (max_len > 0 && maxq[max_head] + period <= i) {
                    max_head += 1;
                    if (max_head == cap) {
                        max_head = 0;
                    }
                    max_len -= 1;
                }
                while (min_len > 0) {
                    int back = min_head + min_len - 1;
                    if (back >= cap) {
                        back -= cap;
                    }
                    if (data[minq[back]] >= data[i]) {
                        min_len -= 1;
                    } else {
                        break;
                    }
                }
                while (max_len > 0) {
                    int back = max_head + max_len - 1;
                    if (back >= cap) {
                        back -= cap;
                    }
                    if (data[maxq[back]] <= data[i]) {
                        max_len -= 1;
                    } else {
                        break;
                    }
                }
                {
                    int tail = min_head + min_len;
                    if (tail >= cap) {
                        tail -= cap;
                    }
                    minq[tail] = i;
                    min_len += 1;
                }
                {
                    int tail = max_head + max_len;
                    if (tail >= cap) {
                        tail -= cap;
                    }
                    maxq[tail] = i;
                    max_len += 1;
                }
                if (i + 1 >= start + period) {
                    mins[i] = data[minq[min_head]];
                    maxs[i] = data[maxq[max_head]];
                }
            }
        }
        start = end;
    }
}

// rolling_mean_std (:926)
__device__ void pr_rolling_mean_std(
    const double* data, int n, int period, double* means, double* stds) {
    pr_fill_nan(means, n);
    pr_fill_nan(stds, n);
    int start = 0;
    while (start < n) {
        while (start < n && !isfinite(data[start])) {
            start += 1;
        }
        if (start >= n) {
            break;
        }
        int end = start + 1;
        while (end < n && isfinite(data[end])) {
            end += 1;
        }
        if (end - start >= period) {
            double sum = 0.0, sumsq = 0.0;
            for (int i = start; i < end; ++i) {
                double value = data[i];
                sum += value;
                sumsq += value * value;
                if (i >= start + period) {
                    double old = data[i - period];
                    sum -= old;
                    sumsq -= old * old;
                }
                if (i + 1 >= start + period) {
                    double mean = sum / static_cast<double>(period);
                    double var = sumsq / static_cast<double>(period) - mean * mean;
                    if (var < 0.0) {
                        var = 0.0;
                    }
                    means[i] = mean;
                    stds[i] = sqrt(var);
                }
            }
        }
        start = end;
    }
}

// percentile_nearest_rank (:1118) over an ascending buffer.
__device__ __forceinline__ double pr_percentile(const double* sorted, int n, double probability) {
    if (n == 0) {
        return pr_qnan();
    }
    double rank = ceil(probability * static_cast<double>(n));
    if (rank < 1.0) {
        rank = 1.0;
    }
    int index = static_cast<int>(rank) - 1;
    if (index > n - 1) {
        index = n - 1;
    }
    if (index < 0) {
        index = 0;
    }
    return sorted[index];
}

// rolling_percentile_series (:1129).
//
// The CPU keeps the window in a sorted Vec and does binary_search + remove +
// insert. `binary_search_by` returns an UNSPECIFIED index among equal elements,
// so the CPU is only well-defined up to "remove one element equal to the
// departing value" — which is exactly what the linear scan below does, and the
// resulting ascending sequence is identical either way.
__device__ void pr_rolling_percentile(
    const double* data, int n, int period, double probability, double* sorted, double* out) {
    pr_fill_nan(out, n);
    int start = 0;
    while (start < n) {
        while (start < n && !isfinite(data[start])) {
            start += 1;
        }
        if (start >= n) {
            break;
        }
        int end = start + 1;
        while (end < n && isfinite(data[end])) {
            end += 1;
        }
        if (end - start >= period) {
            for (int k = 0; k < period; ++k) {
                sorted[k] = data[start + k];
            }
            // Insertion sort ascending; `period` is the dynamic-zone window,
            // tens of bars, so O(n^2) here is cheaper than a heap.
            for (int i = 1; i < period; ++i) {
                double key = sorted[i];
                int j = i - 1;
                while (j >= 0 && sorted[j] > key) {
                    sorted[j + 1] = sorted[j];
                    j -= 1;
                }
                sorted[j + 1] = key;
            }
            out[start + period - 1] = pr_percentile(sorted, period, probability);

            for (int i = start + period; i < end; ++i) {
                double old = data[i - period];
                int remove_idx = period - 1;
                for (int k = 0; k < period; ++k) {
                    if (sorted[k] == old) {
                        remove_idx = k;
                        break;
                    }
                }
                for (int k = remove_idx; k + 1 < period; ++k) {
                    sorted[k] = sorted[k + 1];
                }
                double new_value = data[i];
                int insert_idx = period - 1;
                for (int k = 0; k + 1 < period; ++k) {
                    if (sorted[k] > new_value) {
                        insert_idx = k;
                        break;
                    }
                }
                for (int k = period - 1; k > insert_idx; --k) {
                    sorted[k] = sorted[k - 1];
                }
                sorted[insert_idx] = new_value;
                out[i] = pr_percentile(sorted, period, probability);
            }
        }
        start = end;
    }
}

// crossover (:1164) / crossunder (:1179)
__device__ __forceinline__ double pr_crossover(
    double a_prev, double a, double b_prev, double b) {
    return (isfinite(a_prev) && isfinite(a) && isfinite(b_prev) && isfinite(b) &&
            a_prev <= b_prev && a > b)
               ? 1.0
               : 0.0;
}

__device__ __forceinline__ double pr_crossunder(
    double a_prev, double a, double b_prev, double b) {
    return (isfinite(a_prev) && isfinite(a) && isfinite(b_prev) && isfinite(b) &&
            a_prev >= b_prev && a < b)
               ? 1.0
               : 0.0;
}

extern "C" __global__ void possible_rsi_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    const int* __restrict__ norm_periods,
    const int* __restrict__ normalization_lengths,
    const int* __restrict__ nonlag_periods,
    const int* __restrict__ dynamic_zone_periods,
    const double* __restrict__ buy_probabilities,
    const double* __restrict__ sell_probabilities,
    const int* __restrict__ highpass_periods,
    int rsi_mode,
    int normalization_mode,
    int signal_type,
    int run_highpass,
    int rows,
    int slots,
    int weights_cap,
    int sorted_cap,
    int deque_cap,
    double* scratch,
    int* iscratch,
    double* __restrict__ out_value,
    double* __restrict__ out_buy,
    double* __restrict__ out_sell,
    double* __restrict__ out_middle,
    double* __restrict__ out_state,
    double* __restrict__ out_long,
    double* __restrict__ out_short
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = pr_qnan();
    size_t doubles_per_slot = static_cast<size_t>(PR_ARRAYS) * static_cast<size_t>(len) +
                              static_cast<size_t>(weights_cap) +
                              static_cast<size_t>(sorted_cap);
    double* sbase = scratch + static_cast<size_t>(slot) * doubles_per_slot;
    double* src = sbase + static_cast<size_t>(PR_A_SRC) * len;
    double* rsi_buf = sbase + static_cast<size_t>(PR_A_RSI) * len;
    double* scaled = sbase + static_cast<size_t>(PR_A_SCALE) * len;
    double* normalized = sbase + static_cast<size_t>(PR_A_NORM) * len;
    double* tmp_a = sbase + static_cast<size_t>(PR_A_TMPA) * len;
    double* tmp_b = sbase + static_cast<size_t>(PR_A_TMPB) * len;
    double* weights = sbase + static_cast<size_t>(PR_ARRAYS) * len;
    double* sorted = weights + weights_cap;
    int* minq = iscratch + static_cast<size_t>(slot) * 2ull * static_cast<size_t>(deque_cap);
    int* maxq = minq + deque_cap;

    for (int row = slot; row < rows; row += slots) {
        int period = periods[row];
        int norm_period = norm_periods[row];
        int normalization_length = normalization_lengths[row];
        int nonlag_period = nonlag_periods[row];
        int dz_period = dynamic_zone_periods[row];
        double buy_probability = buy_probabilities[row];
        double sell_probability = sell_probabilities[row];
        int highpass_period = highpass_periods[row];

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* value = out_value + row_base;
        double* buy_level = out_buy + row_base;
        double* sell_level = out_sell + row_base;
        double* middle_level = out_middle + row_base;
        double* state = out_state + row_base;
        double* long_signal = out_long + row_base;
        double* short_signal = out_short + row_base;

        for (int i = 0; i < len; ++i) {
            value[i] = nan_value;
            buy_level[i] = nan_value;
            sell_level[i] = nan_value;
            middle_level[i] = nan_value;
            state[i] = nan_value;
            long_signal[i] = 0.0;
            short_signal[i] = 0.0;
        }

        // compute_possible_rsi_levels (:1225)
        if (run_highpass) {
            pr_highpass(data, len, highpass_period, src);
        } else {
            for (int i = 0; i < len; ++i) {
                src[i] = data[i];
            }
        }

        if (!pr_compute_rsi(src, len, rsi_mode, period, tmp_a, rsi_buf)) {
            continue;
        }

        // normalize_min_max (:1034)
        pr_rolling_min_max(rsi_buf, len, norm_period, minq, maxq, deque_cap, tmp_a, tmp_b);
        for (int i = 0; i < len; ++i) {
            scaled[i] = nan_value;
            if (isfinite(rsi_buf[i]) && isfinite(tmp_a[i]) && isfinite(tmp_b[i]) &&
                fabs(tmp_b[i] - tmp_a[i]) > DBL_EPSILON) {
                scaled[i] = 100.0 * (rsi_buf[i] - tmp_a[i]) / (tmp_b[i] - tmp_a[i]);
            }
        }

        // apply_secondary_normalization (:1051)
        if (normalization_mode == PR_NORM_GAUSSIAN_FISHER) {
            pr_rolling_min_max(scaled, len, normalization_length, minq, maxq, deque_cap, tmp_a,
                               tmp_b);
            double prev_value = 0.0, prev_fish = 0.0;
            bool seeded = false;
            for (int i = 0; i < len; ++i) {
                normalized[i] = nan_value;
                double s = scaled[i];
                double low = tmp_a[i];
                double high = tmp_b[i];
                if (!isfinite(s) || !isfinite(low) || !isfinite(high) ||
                    fabs(high - low) <= DBL_EPSILON) {
                    seeded = false;
                    prev_value = 0.0;
                    prev_fish = 0.0;
                    continue;
                }
                double v = 0.66 * ((s - low) / (high - low) - 0.5) +
                           0.67 * (seeded ? prev_value : 0.0);
                if (v > 0.99) {
                    v = 0.999;
                }
                if (v < -0.99) {
                    v = -0.999;
                }
                double fish =
                    0.5 * log((1.0 + v) / (1.0 - v)) + 0.5 * (seeded ? prev_fish : 0.0);
                normalized[i] = fish;
                prev_value = v;
                prev_fish = fish;
                seeded = true;
            }
        } else {
            pr_rolling_mean_std(scaled, len, normalization_length, tmp_a, tmp_b);
            for (int i = 0; i < len; ++i) {
                normalized[i] = nan_value;
                if (!isfinite(scaled[i]) || !isfinite(tmp_a[i]) || !isfinite(tmp_b[i]) ||
                    tmp_b[i] <= DBL_EPSILON) {
                    continue;
                }
                if (normalization_mode == PR_NORM_SOFTMAX) {
                    double z = (scaled[i] - tmp_a[i]) / tmp_b[i];
                    double e = exp(-z);
                    normalized[i] = (1.0 - e) / (1.0 + e);
                } else {
                    normalized[i] = (scaled[i] - tmp_a[i]) / (tmp_b[i] * 3.0);
                }
            }
        }

        // build_nonlag_weights (:1064) + nonlag_ma_series (:1090)
        double cycle = 4.0;
        double coeff = 3.0 * M_PI;
        double phase = static_cast<double>(nonlag_period) - 1.0;
        int wlen = static_cast<int>(static_cast<double>(nonlag_period) * cycle + phase);
        if (wlen < 0 || wlen > weights_cap) {
            continue;
        }
        double weight_sum = 0.0;
        for (int k = 0; k < wlen; ++k) {
            double t;
            if (phase > 1.0 && static_cast<double>(k) <= phase - 1.0) {
                t = static_cast<double>(k) / (phase - 1.0);
            } else {
                t = 1.0 + (static_cast<double>(k) - phase + 1.0) * (2.0 * cycle - 1.0) /
                              (cycle * static_cast<double>(nonlag_period) - 1.0);
            }
            double beta = cos(M_PI * t);
            double g = 1.0 / (coeff * t + 1.0);
            if (t <= 0.5) {
                g = 1.0;
            }
            double weight = g * beta;
            weights[k] = weight;
            weight_sum += weight;
        }

        {
            int start = 0;
            while (start < len) {
                while (start < len && !isfinite(normalized[start])) {
                    start += 1;
                }
                if (start >= len) {
                    break;
                }
                int end = start + 1;
                while (end < len && isfinite(normalized[end])) {
                    end += 1;
                }
                if (end - start >= wlen && wlen > 0) {
                    for (int i = start + wlen - 1; i < end; ++i) {
                        double sum = 0.0;
                        bool valid = true;
                        for (int k = 0; k < wlen; ++k) {
                            double v = normalized[i - k];
                            if (!isfinite(v)) {
                                valid = false;
                                break;
                            }
                            sum += weights[k] * v;
                        }
                        if (valid) {
                            value[i] = sum / weight_sum;
                        }
                    }
                }
                start = end;
            }
        }

        if (dz_period <= 0 || dz_period > sorted_cap) {
            continue;
        }
        pr_rolling_percentile(value, len, dz_period, buy_probability, sorted, buy_level);
        pr_rolling_percentile(value, len, dz_period, 1.0 - sell_probability, sorted, sell_level);
        pr_rolling_percentile(value, len, dz_period, 0.5, sorted, middle_level);

        // fill_possible_rsi_signal_outputs (:1257)
        for (int i = 0; i < len; ++i) {
            if (!isfinite(value[i])) {
                continue;
            }
            double signal_value;
            if (signal_type == PR_SIG_SLOPE) {
                if (i == 0 || !isfinite(value[i - 1])) {
                    continue;
                }
                signal_value = value[i - 1];
            } else if (signal_type == PR_SIG_DYN_MIDDLE) {
                if (!isfinite(middle_level[i])) {
                    continue;
                }
                signal_value = middle_level[i];
            } else if (signal_type == PR_SIG_LEVELS) {
                if (!isfinite(buy_level[i]) || !isfinite(sell_level[i])) {
                    continue;
                }
                signal_value = nan_value;
            } else {
                signal_value = 0.0;
            }

            if (signal_type == PR_SIG_LEVELS) {
                if (value[i] < buy_level[i]) {
                    state[i] = -1.0;
                } else if (value[i] > sell_level[i]) {
                    state[i] = 1.0;
                } else {
                    state[i] = 0.0;
                }
            } else {
                if (value[i] < signal_value) {
                    state[i] = -1.0;
                } else if (value[i] > signal_value) {
                    state[i] = 1.0;
                } else {
                    state[i] = 0.0;
                }
            }

            if (i == 0) {
                continue;
            }

            if (signal_type == PR_SIG_SLOPE) {
                double b_prev = (i > 1) ? value[i - 2] : value[i - 1];
                long_signal[i] = pr_crossover(value[i - 1], value[i], b_prev, value[i - 1]);
                short_signal[i] = pr_crossunder(value[i - 1], value[i], b_prev, value[i - 1]);
            } else if (signal_type == PR_SIG_DYN_MIDDLE) {
                long_signal[i] =
                    pr_crossover(value[i - 1], value[i], middle_level[i - 1], middle_level[i]);
                short_signal[i] =
                    pr_crossunder(value[i - 1], value[i], middle_level[i - 1], middle_level[i]);
            } else if (signal_type == PR_SIG_LEVELS) {
                long_signal[i] =
                    pr_crossover(value[i - 1], value[i], sell_level[i - 1], sell_level[i]);
                short_signal[i] =
                    pr_crossunder(value[i - 1], value[i], buy_level[i - 1], buy_level[i]);
            } else {
                long_signal[i] = pr_crossover(value[i - 1], value[i], 0.0, 0.0);
                short_signal[i] = pr_crossunder(value[i - 1], value[i], 0.0, 0.0);
            }
        }
    }
}
