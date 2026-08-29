#include <cmath>
#include <cstdint>

// One exact f64 filter authority shared by the generic primary ABI and the
// production selected-output ABI. Arithmetic and reset order mirror
// `UltimateSmootherState` plus `filter_series` in
// `src/indicators/autocorrelation_indicator.rs`.
__device__ __forceinline__ void neo_aci_filter_row_f64(
    const double* __restrict__ data,
    int n,
    int length,
    int use_test_signal,
    double* __restrict__ out
) {
    for (int i = 0; i < n; ++i) {
        out[i] = NAN;
    }
    if (length <= 0) {
        return;
    }

    const double pi_d = 3.14159265358979323846;
    const double period_f = static_cast<double>(length);
    const double a1 = exp(-1.414 * pi_d / period_f);
    const double c2 = 2.0 * a1 * cos(1.414 * pi_d / period_f);
    const double c3 = -a1 * a1;
    const double c1 = (1.0 + c2 - c3) * 0.25;

    int count = 0;
    double prev_src1 = NAN;
    double prev_src2 = NAN;
    double prev_us1 = NAN;
    double prev_us2 = NAN;
    for (int i = 0; i < n; ++i) {
        const double raw = use_test_signal != 0
            ? sin(2.0 * pi_d * static_cast<double>(i) / 30.0)
            : data[i];
        if (!isfinite(raw)) {
            count = 0;
            prev_src1 = NAN;
            prev_src2 = NAN;
            prev_us1 = NAN;
            prev_us2 = NAN;
            continue;
        }

        const double value = count >= 4
            ? (1.0 - c1) * raw + (2.0 * c1 - c2) * prev_src1
                - (c1 + c3) * prev_src2 + c2 * prev_us1 + c3 * prev_us2
            : raw;
        prev_src2 = prev_src1;
        prev_src1 = raw;
        prev_us2 = prev_us1;
        prev_us1 = value;
        count += 1;
        out[i] = value;
    }
}

// Exact selected-lag authority matching `compute_segment_correlation_lag`.
// Prefix arrays are resident per-tuple scratch; each finite segment reuses the
// row scratch from zero because one thread owns the complete tuple.
__device__ __forceinline__ void neo_aci_selected_correlation_row_f64(
    const double* __restrict__ filtered,
    int n,
    int length,
    int lag,
    double* __restrict__ out,
    double* __restrict__ prefix,
    double* __restrict__ prefix_sq
) {
    for (int i = 0; i < n; ++i) {
        out[i] = NAN;
    }
    if (length <= 0 || lag <= 0) {
        return;
    }

    int seg_start = 0;
    while (seg_start < n) {
        while (seg_start < n && !isfinite(filtered[seg_start])) {
            seg_start += 1;
        }
        if (seg_start >= n) {
            break;
        }
        int seg_end = seg_start + 1;
        while (seg_end < n && isfinite(filtered[seg_end])) {
            seg_end += 1;
        }

        const int seg_len = seg_end - seg_start;
        prefix[0] = 0.0;
        prefix_sq[0] = 0.0;
        for (int i = 0; i < seg_len; ++i) {
            const double value = filtered[seg_start + i];
            prefix[i + 1] = prefix[i] + value;
            prefix_sq[i + 1] = prefix_sq[i] + value * value;
        }

        if (length <= seg_len && lag <= seg_len - length) {
            const double length_f = static_cast<double>(length);
            const int t0 = lag + length - 1;
            double cross = 0.0;
            for (int j = 0; j < length; ++j) {
                cross += filtered[seg_start + lag + j] * filtered[seg_start + j];
            }
            for (int t = t0; t < seg_len; ++t) {
                const int x_start = t + 1 - length;
                const int y_start = x_start - lag;
                const double sx = prefix[t + 1] - prefix[x_start];
                const double sxx = prefix_sq[t + 1] - prefix_sq[x_start];
                const double sy = prefix[y_start + length] - prefix[y_start];
                const double syy = prefix_sq[y_start + length] - prefix_sq[y_start];
                const double ca1 = length_f * sxx - sx * sx;
                const double ca2 = length_f * syy - sy * sy;
                out[seg_start + t] = (ca1 > 0.0 && ca2 > 0.0)
                    ? ((length_f * cross - sx * sy) / sqrt(ca1 * ca2))
                    : 0.0;
                if (t + 1 < seg_len) {
                    cross += filtered[seg_start + t + 1] * filtered[seg_start + t + 1 - lag]
                        - filtered[seg_start + x_start] * filtered[seg_start + y_start];
                }
            }
        }
        seg_start = seg_end;
    }
}

extern "C" __global__ void autocorrelation_indicator_batch_f64(
    const double* data,
    int len,
    const int* lengths,
    int rows,
    int max_lag,
    int use_test_signal,
    double* out_filtered,
    double* out_correlations
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    if (length <= 0 || max_lag <= 0) {
        return;
    }

    double* row_filtered = out_filtered + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_correlations =
        out_correlations + static_cast<size_t>(row) * static_cast<size_t>(max_lag) * static_cast<size_t>(len);
    neo_aci_filter_row_f64(data, len, length, use_test_signal, row_filtered);
    for (int i = 0; i < max_lag * len; ++i) {
        row_correlations[i] = NAN;
    }

    int seg_start = 0;
    while (seg_start < len) {
        while (seg_start < len && !isfinite(row_filtered[seg_start])) {
            seg_start += 1;
        }
        if (seg_start >= len) {
            break;
        }

        int seg_end = seg_start + 1;
        while (seg_end < len && isfinite(row_filtered[seg_end])) {
            seg_end += 1;
        }

        int seg_len = seg_end - seg_start;
        for (int lag = 1; lag <= max_lag; ++lag) {
            if (seg_len < length + lag) {
                continue;
            }

            double* lag_row = row_correlations + static_cast<size_t>(lag - 1) * static_cast<size_t>(len);
            for (int t = lag + length - 1; t < seg_len; ++t) {
                int start_x = t + 1 - length;
                int start_y = start_x - lag;
                double sx = 0.0;
                double sy = 0.0;
                double sxx = 0.0;
                double syy = 0.0;
                double sxy = 0.0;
                for (int j = 0; j < length; ++j) {
                    double x = row_filtered[seg_start + start_x + j];
                    double y = row_filtered[seg_start + start_y + j];
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                }

                double length_f = static_cast<double>(length);
                double ca1 = length_f * sxx - sx * sx;
                double ca2 = length_f * syy - sy * sy;
                lag_row[seg_start + t] = (ca1 > 0.0 && ca2 > 0.0)
                    ? ((length_f * sxy - sx * sy) / sqrt(ca1 * ca2))
                    : 0.0;
            }
        }

        seg_start = seg_end;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/autocorrelation_indicator.rs:743
// (`autocorrelation_indicator_with_kernel`). The primary column is canonical
// `filtered`; production's named entry point emits it beside canonical
// `correlation` at one selected lag. The arithmetic is `filter_series`
// driving `UltimateSmootherState::update`.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- the
// ultimate smoother is a two-pole IIR whose state (`prev_src1`, `prev_src2`,
// `prev_us1`, `prev_us2`, `count`) is carried bar to bar, and the CPU RESETS
// all five on a non-finite bar. A bar-parallel form cannot see the reset
// history.
//
// PERIOD-SWEPT. NeoEthos maps its one-dimensional sweep to canonical `length`,
// and both entry points consume that exact value. The selected-output entry
// additionally consumes canonical `lag` and `use_test_signal` per tuple.
//
// FIRST VALID IS NOT READ: `filter_series` fills the whole output with NaN and
// then walks from index 0, emitting a value at EVERY finite bar (:592-602), so
// there is no warmup index to consult. The lane row declares
// `F64FirstValidRule::Ignored`.
//
// f64 END TO END: every constant is a double literal, `exp`/`cos`/`sin` are
// the double overloads, and there is no fast-math intrinsic. The five-term
// combination is written in the CPU's term order so the additions round the
// same way.
// ---------------------------------------------------------------------------

extern "C" __global__ void autocorrelation_indicator_neo_batch_f64(
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
    (void)first_valid;

    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    const int length = periods[combo];
    neo_aci_filter_row_f64(data, n, length, 0, row);
}

extern "C" __global__ void autocorrelation_indicator_outputs_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ lengths,
    const int* __restrict__ lags,
    const int* __restrict__ use_test_signals,
    int rows,
    double* __restrict__ out_filtered,
    double* __restrict__ out_correlation,
    double* __restrict__ scratch_prefix,
    double* __restrict__ scratch_prefix_sq
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || n <= 0) {
        return;
    }

    const int length = lengths[row];
    const int lag = lags[row];
    const int use_test_signal = use_test_signals[row];
    const size_t output_offset = static_cast<size_t>(row) * static_cast<size_t>(n);
    const size_t scratch_offset = static_cast<size_t>(row) *
        (static_cast<size_t>(n) + static_cast<size_t>(1));
    double* row_filtered = out_filtered + output_offset;
    double* row_correlation = out_correlation + output_offset;
    double* row_prefix = scratch_prefix + scratch_offset;
    double* row_prefix_sq = scratch_prefix_sq + scratch_offset;

    neo_aci_filter_row_f64(data, n, length, use_test_signal, row_filtered);
    neo_aci_selected_correlation_row_f64(
        row_filtered,
        n,
        length,
        lag,
        row_correlation,
        row_prefix,
        row_prefix_sq
    );
}
