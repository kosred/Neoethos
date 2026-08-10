#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void historical_volatility_rank_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ hv_lengths,
    const int* __restrict__ rank_lengths,
    const double* __restrict__ annualization_scales,
    int n_combos,
    double* __restrict__ out_hvr,
    double* __restrict__ out_hv
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int hv_length = hv_lengths[combo_idx];
    int rank_length = rank_lengths[combo_idx];
    double annualization_scale = annualization_scales[combo_idx];
    double* row_hvr = out_hvr + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_hv = out_hv + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (hv_length <= 0 || rank_length <= 0 || !isfinite(annualization_scale) || annualization_scale <= 0.0) {
        for (int t = 0; t < len; ++t) {
            row_hvr[t] = CUDART_NAN;
            row_hv[t] = CUDART_NAN;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        row_hvr[t] = CUDART_NAN;
        row_hv[t] = CUDART_NAN;
    }

    for (int t = hv_length; t < len; ++t) {
        int start = t + 1 - hv_length;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = start; i <= t; ++i) {
            double prev = data[i - 1];
            double curr = data[i];
            if (!isfinite(prev) || !isfinite(curr) || prev <= 0.0 || curr <= 0.0) {
                valid = false;
                break;
            }
            double ret = log(curr / prev);
            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            continue;
        }

        double n = static_cast<double>(hv_length);
        double mean = sum / n;
        double variance = (sumsq / n) - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        row_hv[t] = 100.0 * sqrt(variance) * annualization_scale;
    }

    for (int t = rank_length - 1; t < len; ++t) {
        int start = t + 1 - rank_length;
        bool valid = true;
        double min_v = CUDART_INF;
        double max_v = -CUDART_INF;
        double value = row_hv[t];

        for (int i = start; i <= t; ++i) {
            double hv = row_hv[i];
            if (!isfinite(hv)) {
                valid = false;
                break;
            }
            if (hv < min_v) {
                min_v = hv;
            }
            if (hv > max_v) {
                max_v = hv;
            }
        }

        if (!valid) {
            continue;
        }

        double range = max_v - min_v;
        if (!isfinite(range) || range <= 0.0) {
            row_hvr[t] = 0.0;
        } else {
            row_hvr[t] = 100.0 * (value - min_v) / range;
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/historical_volatility_rank.rs:667
// (`historical_volatility_rank_with_kernel`). The column this emits is `hvr`,
// which is what `output_id == "value"` resolves to
// (dispatch/cpu_batch.rs:6647-6650). `annualization_scale` is
// `(annualization_days / bar_days).sqrt()` (:329).
//
// SHAPE: one thread per combo, bars ascending for the HV pass and then bars
// DESCENDING for the rank pass. The descending direction is not a rounding
// choice -- it is what lets the rank pass overwrite the HV series IN PLACE:
// `hvr[t]` reads `hv[j]` for `j <= t` only, so walking downwards never reads a
// slot already replaced. A lane kernel has one row of output and no scratch
// allocator, so the alternative would be a per-thread array as long as the
// whole series.
//
// PERIOD-INVARIANT. `compute_historical_volatility_rank_batch`
// (cpu_batch.rs:6664-6673) reads `hv_length`, `rank_length`,
// `annualization_days` and `bar_days` and NEVER `period`, so five swept
// periods give five identical CPU columns and this kernel emits five identical
// rows. All four CPU defaults are pinned below.
//
// FIRST VALID IS NOT READ: the CPU walks the whole series and decides validity
// window by window (every bar of the window must be finite AND strictly
// positive, because the return is `ln(curr/prev)`), so there is no single
// warmup index. The lane row declares `F64FirstValidRule::Ignored`.
//
// NaN CANNOT SURVIVE: the min/max scan breaks on the first non-finite HV
// before the comparison rather than comparing against it, so no `if (hv <
// min_v)` chain can let a NaN through -- the failure mode rule 4 names.
//
// f64 END TO END: double literals, double `log`/`sqrt`, no fast-math
// intrinsic, and no epsilon at all (the range guard is an exact `> 0.0`).
// ---------------------------------------------------------------------------

#define NEO_HVR_HV_LENGTH 10
#define NEO_HVR_RANK_LENGTH (52 * 7)
#define NEO_HVR_ANNUALIZATION_DAYS 365.0
#define NEO_HVR_BAR_DAYS 1.0

extern "C" __global__ void historical_volatility_rank_neo_batch_f64(
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
    for (int t = 0; t < n; ++t) {
        row[t] = CUDART_NAN;
    }

    const int hv_length = NEO_HVR_HV_LENGTH;
    const int rank_length = NEO_HVR_RANK_LENGTH;
    const double annualization_scale = sqrt(NEO_HVR_ANNUALIZATION_DAYS / NEO_HVR_BAR_DAYS);
    if (hv_length <= 0 || rank_length <= 0 || !isfinite(annualization_scale) ||
        annualization_scale <= 0.0) {
        return;
    }

    // Pass 1 -- the HV series, written into the row it will later be replaced
    // in.
    for (int t = hv_length; t < n; ++t) {
        const int start = t + 1 - hv_length;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = start; i <= t; ++i) {
            const double prev = data[i - 1];
            const double curr = data[i];
            if (!isfinite(prev) || !isfinite(curr) || prev <= 0.0 || curr <= 0.0) {
                valid = false;
                break;
            }
            const double ret = log(curr / prev);
            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            continue;
        }

        const double nf = static_cast<double>(hv_length);
        const double mean = sum / nf;
        double variance = (sumsq / nf) - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        row[t] = 100.0 * sqrt(variance) * annualization_scale;
    }

    // Pass 2 -- the rank, in place, walking DOWNWARDS.
    for (int t = n - 1; t >= rank_length - 1; --t) {
        const int start = t + 1 - rank_length;
        bool valid = true;
        double min_v = CUDART_INF;
        double max_v = -CUDART_INF;
        const double value = row[t];

        for (int i = start; i <= t; ++i) {
            const double hv = row[i];
            if (!isfinite(hv)) {
                valid = false;
                break;
            }
            if (hv < min_v) {
                min_v = hv;
            }
            if (hv > max_v) {
                max_v = hv;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN;
            continue;
        }

        const double range = max_v - min_v;
        if (!isfinite(range) || range <= 0.0) {
            row[t] = 0.0;
        } else {
            row[t] = 100.0 * (value - min_v) / range;
        }
    }

    // Every bar before the first full rank window carries an HV value that the
    // CPU never publishes as `hvr`.
    const int rank_warm = (rank_length - 1) < n ? (rank_length - 1) : n;
    for (int t = 0; t < rank_warm; ++t) {
        row[t] = CUDART_NAN;
    }
}
