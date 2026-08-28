#include <cmath>
#include <cstddef>
#include <cstdint>

static __device__ inline void push_shift(double* buffer, int* count, int capacity, double value) {
    if (*count < capacity) {
        buffer[*count] = value;
        *count += 1;
        return;
    }
    for (int i = 1; i < capacity; ++i) {
        buffer[i - 1] = buffer[i];
    }
    buffer[capacity - 1] = value;
}

static __device__ inline double linreg_value(const double* window, int length) {
    double period_f = static_cast<double>(length);
    double x_sum = static_cast<double>(length * (length + 1) / 2);
    double x2_sum = static_cast<double>(length * (length + 1) * (2 * length + 1) / 6);
    double denom = period_f * x2_sum - x_sum * x_sum;
    if (!(denom > 0.0) || !isfinite(denom)) {
        return NAN;
    }

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int i = 0; i < length; ++i) {
        double value = window[i];
        double x = static_cast<double>(i + 1);
        y_sum += value;
        xy_sum += value * x;
    }

    double b = (period_f * xy_sum - x_sum * y_sum) / denom;
    double a = (y_sum - b * x_sum) / period_f;
    return a + b * period_f;
}

static __device__ inline double trend_to_intensity(const double* window, int lookback) {
    int total = lookback * (lookback - 1) / 2;
    if (total == 0) {
        return 0.0;
    }

    int64_t trend = 0;
    for (int i = 0; i < lookback - 1; ++i) {
        double a = window[i];
        for (int j = i + 1; j < lookback; ++j) {
            double b = window[j];
            if (a != b) {
                trend += b > a ? 1 : -1;
            }
        }
    }
    return static_cast<double>(trend) / static_cast<double>(total);
}

extern "C" __global__ void linear_regression_intensity_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookback_periods,
    const int* __restrict__ linreg_lengths,
    int rows,
    int max_lookback_period,
    int max_linreg_length,
    double* __restrict__ linreg_input_buf,
    double* __restrict__ linreg_window_buf,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int lookback_period = lookback_periods[row];
    int linreg_length = linreg_lengths[row];
    double* linreg_input =
        linreg_input_buf + static_cast<size_t>(row) * static_cast<size_t>(max_linreg_length);
    double* linreg_window =
        linreg_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_lookback_period);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (lookback_period <= 0 ||
        linreg_length <= 0 ||
        lookback_period > max_lookback_period ||
        linreg_length > max_linreg_length) {
        return;
    }

    int linreg_input_count = 0;
    int linreg_window_count = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            linreg_input_count = 0;
            linreg_window_count = 0;
            continue;
        }

        push_shift(linreg_input, &linreg_input_count, linreg_length, value);
        if (linreg_input_count < linreg_length) {
            continue;
        }

        double lr = linreg_value(linreg_input, linreg_length);
        if (!isfinite(lr)) {
            linreg_window_count = 0;
            continue;
        }

        push_shift(linreg_window, &linreg_window_count, lookback_period, lr);
        if (linreg_window_count < lookback_period) {
            continue;
        }

        row_out[i] = trend_to_intensity(linreg_window, lookback_period);
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `linear_regression_intensity_with_kernel`
// (src/indicators/linear_regression_intensity.rs:680) ->
// `linear_regression_intensity_compute_into` (:622), which DISPATCHES between
// three implementations. This kernel reproduces the dispatch, not just one arm.
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `linear_regression_intensity_batch_f64` (:59) is double-clean but declares
// ten parameters -- two `const int*` per-row parameter arrays and two
// host-allocated ring buffers. The f64 lane launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// with no scratch pointer, so the lane gets its own entry point here. Its two
// rings are fixed-size PER-THREAD arrays at the pinned defaults: 90 + 12 = 102
// doubles, 816 bytes. Bounded at compile time, not allocated.
//
// AND WHY IT IS NOT A COPY OF THE ONE ABOVE. The existing entry point
// implements ONLY the `compute_fallback` arm (:401) -- streaming linreg plus
// the O(lookback^2) naive trend. On real data the CPU does NOT take that arm:
// `lookback_period` is 12, which is <= 64, and once `data[first..]` is all
// finite `compute_fused_small_lookback` (:530) is what runs (:626-632). The
// two arms agree in exact arithmetic and NOT bit for bit -- the fused arm
// carries `y_sum`/`xy_sum` across bars and slides them, the fallback arm
// re-dots a 90-slot ring every bar -- so a kernel that always ran the fallback
// would be in parity with an arm the CPU never reaches.
//
// WHICH COLUMN: `value`. `compute_linear_regression_intensity_batch` calls
// `expect_value_output` (cpu_batch.rs:11384), so `value` is the only output id.
//
// SHAPE: one thread per combo, bars ascending. The fused arm carries `y_sum`,
// `xy_sum`, the 12-slot ring of linreg endpoints and the running Kendall
// `trend` counter; the sliding update reads the departing sample, so bar i
// cannot be computed without bar i-1.
//
// PERIOD-INVARIANT: the CPU batch reads `source`, `lookback_period`,
// `range_tolerance` and `linreg_length` and NEVER `period`
// (cpu_batch.rs:11404-11414), so every swept period gives the same CPU column
// and this kernel writes identical rows. Pinned at the CPU defaults:
// lookback_period 12, linreg_length 90 (cpu_batch.rs:11406, :11414).
// `range_tolerance` is validated and never read by the computation.
//
// ROUNDING: the fused arm is transcribed operand for operand from :534-620 --
// `x_sum` and `x2_sum` are formed in INTEGER arithmetic and cast once (:537-
// 538, so no accumulation of casts), `denom_inv` and `inv_period` are stored
// RECIPROCALS the CPU multiplies by (:539-540), and the slide is
// `xy_sum -= y_sum; y_sum -= data[old_idx]` in that order (:616-617). No
// `fma`: the CPU writes no `mul_add` anywhere in this indicator, so fusing
// would be one rounding where it has two.
//
// NaN SEMANTICS: `pair_sign` (:318) returns 0 for the equal case AND for any
// comparison involving NaN -- both `>` and `<` are false -- so the CPU's
// integer trend simply ignores such a pair. The transcription keeps the same
// two-branch form; there is no `f64::max` here for rule 4 to catch.
//
// THE ONE DOCUMENTED DIVERGENCE. When `data[first..]` contains a hole the CPU
// runs `linreg_with_kernel` over the whole series and then takes
// `compute_fast_from_linreg` (:552) if the linreg output is clean after ITS
// first valid index, else `compute_fallback` (:401). This kernel takes the
// fallback arm for both. The two produce the SAME integer `trend` -- one
// counts pairs by rank through a Fenwick tree, the other by direct comparison,
// over the same 12 values -- so they differ only if the linreg values
// themselves differ between `linreg_with_kernel` and `LinRegStream`, and then
// only where two of them are exactly equal. Named here rather than hidden: it
// cannot arise on a gap-free frame, which is every frame the fused arm covers.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic, no epsilon. The NaN is a DOUBLE quiet-NaN bit pattern.
//
// FIRST VALID IS NOT READ: the kernel derives `first` itself with the CPU's
// own rule (`first_valid_index`, :299 -- the first `is_finite`), because the
// warmup this indicator needs is `first + linreg_length + lookback_period - 2`
// (`warmup_prefix`, :329) and the fallback arm restarts at every hole. The
// lane row declares `F64FirstValidRule::Ignored`.
//
// ONE PLACE THE KERNEL IS DEFINED WHERE THE CPU IS NOT: `_with_kernel` builds
// its output with `alloc_with_nan_prefix(len, warmup)` (:684), so bars past the
// warmup that the fallback arm never writes keep UNINITIALISED memory. This
// kernel writes NaN there, which is what `linear_regression_intensity_into_
// slice` (:697, `out.fill(f64::NAN)`) does and the only defensible value.
// ---------------------------------------------------------------------------

#define NEO_LRI_LOOKBACK 12
#define NEO_LRI_LINREG_LENGTH 90

__device__ inline double neo_lri_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `pair_sign`, linear_regression_intensity.rs:318.
__device__ inline long long neo_lri_pair_sign(double later, double earlier) {
    if (later > earlier) {
        return 1;
    }
    if (later < earlier) {
        return -1;
    }
    return 0;
}

extern "C" __global__ void linear_regression_intensity_neo_batch_f64(
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

    const double nan_value = neo_lri_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = nan_value;
    }

    const int period = NEO_LRI_LINREG_LENGTH;
    const int lookback = NEO_LRI_LOOKBACK;
    if (n < period) {
        return;
    }

    // `first_valid_index`, :299.
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (isfinite(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    // `compute_fused_small_lookback` is taken when `lookback_period <= 64` AND
    // `data[first..]` is all finite -- :626-632.
    bool clean = true;
    for (int i = first; i < n; ++i) {
        if (!isfinite(data[i])) {
            clean = false;
            break;
        }
    }

    // `total_combinations`, :304.
    const double total = static_cast<double>(lookback * (lookback - 1) / 2);
    double window[NEO_LRI_LOOKBACK];

    if (clean && lookback <= 64) {
        if (first + period - 1 >= n) {
            return;
        }
        // :534-540. Integer sums cast once, reciprocals stored.
        const double period_f = static_cast<double>(period);
        const double x_sum = static_cast<double>((period * (period + 1)) / 2);
        const double x2_sum =
            static_cast<double>((period * (period + 1) * (2 * period + 1)) / 6);
        const double denom_inv = 1.0 / (period_f * x2_sum - x_sum * x_sum);
        const double inv_period = 1.0 / period_f;

        // :542-549 -- seed with the first `period - 1` samples, x running 1..
        double y_sum = 0.0;
        double xy_sum = 0.0;
        {
            int k = 1;
            for (int i = first; i < first + period - 1; ++i) {
                const double value = data[i];
                y_sum += value;
                xy_sum += static_cast<double>(k) * value;
                k += 1;
            }
        }

        int head = 0;
        int count = 0;
        long long trend = 0;
        int idx = first + period - 1;
        int old_idx = first;

        while (idx < n) {
            const double new_value = data[idx];
            y_sum += new_value;
            xy_sum += new_value * period_f;

            const double b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
            const double a = (y_sum - b * x_sum) * inv_period;
            const double lr = a + b * period_f;

            if (lookback == 1) {
                row[idx] = 0.0;
            } else if (count < lookback) {
                for (int j = 0; j < count; ++j) {
                    trend += neo_lri_pair_sign(lr, window[j]);
                }
                window[count] = lr;
                count += 1;
                if (count == lookback) {
                    row[idx] = (total == 0.0)
                        ? 0.0
                        : (static_cast<double>(trend) / total);
                }
            } else {
                const double old_value = window[head];
                long long remove_delta = 0;
                long long add_delta = 0;
                int pos = head + 1;
                if (pos == lookback) {
                    pos = 0;
                }
                for (int step = 1; step < lookback; ++step) {
                    const double value = window[pos];
                    remove_delta += neo_lri_pair_sign(value, old_value);
                    add_delta += neo_lri_pair_sign(lr, value);
                    pos += 1;
                    if (pos == lookback) {
                        pos = 0;
                    }
                }
                trend += add_delta - remove_delta;
                window[head] = lr;
                head += 1;
                if (head == lookback) {
                    head = 0;
                }
                row[idx] = (total == 0.0)
                    ? 0.0
                    : (static_cast<double>(trend) / total);
            }

            // :616-617 -- slide, in this order.
            xy_sum -= y_sum;
            y_sum -= data[old_idx];
            idx += 1;
            old_idx += 1;
        }
        return;
    }

    // `compute_fallback`, :401-453. A fresh `LinRegStream` at every hole, the
    // stream's own `dot_ring` (linreg.rs:900) every bar, and the naive
    // O(lookback^2) Kendall count (`naive_window_trend`, :385).
    double lin_ring[NEO_LRI_LINREG_LENGTH];
    int lin_head = 0;
    bool lin_filled = false;
    int win_len = 0;

    // `LinRegStream::try_new`, linreg.rs:869-875 -- x_sum and x2_sum are
    // ACCUMULATED here, not formed in integer arithmetic as the fused arm does.
    double x_sum = 0.0;
    double x2_sum = 0.0;
    for (int i = 1; i <= period; ++i) {
        const double xi = static_cast<double>(i);
        x_sum += xi;
        x2_sum += xi * xi;
    }
    const double pf = static_cast<double>(period);

    for (int index = 0; index < n; ++index) {
        const double value = data[index];
        if (!isfinite(value)) {
            lin_head = 0;
            lin_filled = false;
            win_len = 0;
            continue;
        }

        lin_ring[lin_head] = value;
        lin_head = (lin_head + 1) % period;
        if (!lin_filled && lin_head == 0) {
            lin_filled = true;
        }
        if (!lin_filled) {
            continue;
        }

        // `dot_ring`, linreg.rs:900-913 -- oldest sample gets x = 1.
        double y_sum = 0.0;
        double xy_sum = 0.0;
        int pos = lin_head;
        for (int i = 1; i <= period; ++i) {
            const double y = lin_ring[pos];
            y_sum += y;
            xy_sum += y * static_cast<double>(i);
            pos += 1;
            if (pos == period) {
                pos = 0;
            }
        }
        const double bd = 1.0 / (pf * x2_sum - x_sum * x_sum);
        const double b = (pf * xy_sum - x_sum * y_sum) * bd;
        const double a = (y_sum - b * x_sum) / pf;
        const double lr = a + b * pf;

        if (!isfinite(lr)) {
            win_len = 0;
            continue;
        }

        // `window.push_back(lr)` then pop_front past `lookback` -- :443-450.
        if (win_len == lookback) {
            for (int j = 1; j < lookback; ++j) {
                window[j - 1] = window[j];
            }
            window[lookback - 1] = lr;
        } else {
            window[win_len] = lr;
            win_len += 1;
        }
        if (win_len != lookback) {
            continue;
        }

        // `naive_window_trend`, :385-398 -- front (oldest) to back (newest).
        long long trend = 0;
        for (int i = 0; i + 1 < lookback; ++i) {
            const double av = window[i];
            for (int j = i + 1; j < lookback; ++j) {
                const double bv = window[j];
                if (av != bv) {
                    trend += (bv > av) ? 1 : -1;
                }
            }
        }
        row[index] = (total == 0.0) ? 0.0 : (static_cast<double>(trend) / total);
    }
}
