#include <cmath>
#include <cstdint>

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

    const double nan = NAN;
    const double pi = 3.14159265358979323846;
    const double test_signal_period = 30.0;

    double* row_filtered = out_filtered + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_correlations =
        out_correlations + static_cast<size_t>(row) * static_cast<size_t>(max_lag) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_filtered[i] = nan;
    }
    for (int i = 0; i < max_lag * len; ++i) {
        row_correlations[i] = nan;
    }

    double period_f = static_cast<double>(length);
    double a1 = exp(-1.414 * pi / period_f);
    double c2 = 2.0 * a1 * cos(1.414 * pi / period_f);
    double c3 = -a1 * a1;
    double c1 = (1.0 + c2 - c3) * 0.25;

    int smoother_count = 0;
    double prev_src1 = nan;
    double prev_src2 = nan;
    double prev_us1 = nan;
    double prev_us2 = nan;

    for (int i = 0; i < len; ++i) {
        double raw = use_test_signal != 0
            ? sin(2.0 * pi * static_cast<double>(i) / test_signal_period)
            : data[i];

        if (!isfinite(raw)) {
            smoother_count = 0;
            prev_src1 = nan;
            prev_src2 = nan;
            prev_us1 = nan;
            prev_us2 = nan;
            continue;
        }

        double filtered = smoother_count >= 4
            ? (1.0 - c1) * raw + (2.0 * c1 - c2) * prev_src1 - (c1 + c3) * prev_src2
                + c2 * prev_us1 + c3 * prev_us2
            : raw;
        prev_src2 = prev_src1;
        prev_src1 = raw;
        prev_us2 = prev_us1;
        prev_us1 = filtered;
        smoother_count += 1;
        row_filtered[i] = filtered;
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
// (`autocorrelation_indicator_with_kernel`). The column this emits is
// `filtered`, which is what `output_id == "value"` resolves to
// (dispatch/cpu_batch.rs:7779). The arithmetic is `filter_series` :588-603
// driving `UltimateSmootherState::update` :328-345.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- the
// ultimate smoother is a two-pole IIR whose state (`prev_src1`, `prev_src2`,
// `prev_us1`, `prev_us2`, `count`) is carried bar to bar, and the CPU RESETS
// all five on a non-finite bar. A bar-parallel form cannot see the reset
// history.
//
// PERIOD-INVARIANT. `compute_autocorrelation_indicator_batch`
// (cpu_batch.rs:7798-7800) reads `length`, `lag` and `use_test_signal` and
// NEVER `period`, so five swept periods give five identical CPU columns and
// this kernel emits five identical rows. `length` is pinned at the CPU default
// 20; `lag` and `use_test_signal` do not enter the `filtered` column at all.
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

#define NEO_ACI_LENGTH 20

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
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int length = NEO_ACI_LENGTH;
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
        const double raw = data[i];
        if (!isfinite(raw)) {
            count = 0;
            prev_src1 = NAN;
            prev_src2 = NAN;
            prev_us1 = NAN;
            prev_us2 = NAN;
            continue;
        }

        const double value = (count >= 4)
            ? ((1.0 - c1) * raw + (2.0 * c1 - c2) * prev_src1 - (c1 + c3) * prev_src2 +
               c2 * prev_us1 + c3 * prev_us2)
            : raw;

        prev_src2 = prev_src1;
        prev_src1 = raw;
        prev_us2 = prev_us1;
        prev_us1 = value;
        count += 1;
        row[i] = value;
    }
}
