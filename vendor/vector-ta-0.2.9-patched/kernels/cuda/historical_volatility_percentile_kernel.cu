#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void historical_volatility_percentile_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ annual_lengths,
    int n_combos,
    double* __restrict__ out_hvp,
    double* __restrict__ out_hvp_sma
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int annual_length = annual_lengths[combo_idx];
    double* row_hvp = out_hvp + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_hvp_sma = out_hvp_sma + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length < 2 || annual_length <= 0) {
        for (int t = 0; t < len; ++t) {
            row_hvp[t] = CUDART_NAN;
            row_hvp_sma[t] = CUDART_NAN;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        row_hvp[t] = CUDART_NAN;
        row_hvp_sma[t] = CUDART_NAN;
    }

    for (int t = length - 1; t < len; ++t) {
        int start = t + 1 - length;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int j = start; j <= t; ++j) {
            double curr = data[j];
            if (!isfinite(curr) || curr <= 0.0) {
                valid = false;
                break;
            }

            double ret;
            if (j == 0) {
                ret = 0.0;
            } else {
                double prev = data[j - 1];
                if (!isfinite(prev) || prev <= 0.0) {
                    ret = 0.0;
                } else {
                    ret = log(curr / prev);
                }
            }

            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            continue;
        }

        double n = static_cast<double>(length);
        double mean = sum / n;
        double centered = sumsq - mean * mean * n;
        if (centered < 0.0) {
            centered = 0.0;
        }
        double sample_var = centered / static_cast<double>(length - 1);
        row_hvp_sma[t] = sqrt(sample_var) * sqrt(static_cast<double>(annual_length));
    }

    for (int t = annual_length - 1; t < len; ++t) {
        int start = t + 1 - annual_length;
        bool valid = true;
        int rank = 0;
        double current_hv = row_hvp_sma[t];

        for (int j = start; j <= t; ++j) {
            double hv = row_hvp_sma[j];
            if (!isfinite(hv)) {
                valid = false;
                break;
            }
            rank += static_cast<int>(hv < current_hv);
        }

        if (!valid) {
            continue;
        }

        row_hvp[t] = static_cast<double>(rank) * (100.0 / static_cast<double>(annual_length));
    }

    for (int t = length - 1; t < len; ++t) {
        int start = t + 1 - length;
        bool valid = true;
        double sum = 0.0;

        for (int j = start; j <= t; ++j) {
            double hvp = row_hvp[j];
            if (!isfinite(hvp)) {
                valid = false;
                break;
            }
            sum += hvp;
        }

        if (!valid) {
            row_hvp_sma[t] = CUDART_NAN;
            continue;
        }

        row_hvp_sma[t] = sum / static_cast<double>(length);
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/historical_volatility_percentile.rs:470
// (`historical_volatility_percentile_with_kernel`), the `hvp` column.
//
// WHICH COLUMN, AND WHY IT IS NAMED HERE RATHER THAN DISCOVERED.
// `compute_historical_volatility_percentile_batch` (cpu_batch.rs:9681-9690)
// accepts ONLY `hvp` and `hvp_sma` -- it has no `value` alias and returns
// `UnknownOutput` for one. So a parity run must ask the CPU for `hvp`
// explicitly; this kernel emits that column and never the SMA of it.
//
// SHAPE: one thread per combo. The HV pass walks bars ascending; the
// percentile pass then walks bars DESCENDING so it can replace the HV series
// IN PLACE -- `hvp[t]` reads `hv[j]` for `j <= t` only, so walking downwards
// never reads a slot already overwritten. A lane kernel owns one output row
// and no scratch allocator, and a per-thread array as long as the series is
// not available.
//
// PERIOD-INVARIANT. The CPU batch (cpu_batch.rs:9660-9666) reads `length` and
// `annual_length` and NEVER `period`, so five swept periods give five
// identical CPU columns and this kernel emits five identical rows. Both CPU
// defaults are pinned below.
//
// FIRST VALID IS NOT READ: validity is decided window by window -- every bar
// of the window must be finite AND strictly positive, because the return is
// `ln(curr/prev)` -- so there is no single warmup index to consult. The lane
// row declares `F64FirstValidRule::Ignored`.
//
// f64 END TO END: double literals, double `log`/`sqrt`, no fast-math
// intrinsic. The centred sum-of-squares is written as `sumsq - mean*mean*n`
// exactly as the CPU does rather than as a mathematically equal regrouping,
// because the regrouping rounds differently.
// ---------------------------------------------------------------------------

#define NEO_HVP_LENGTH 20
#define NEO_HVP_ANNUAL_LENGTH 252

extern "C" __global__ void historical_volatility_percentile_neo_batch_f64(
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

    const int length = NEO_HVP_LENGTH;
    const int annual_length = NEO_HVP_ANNUAL_LENGTH;
    if (length < 2 || annual_length <= 0) {
        return;
    }

    // Pass 1 -- the annualised sample standard deviation of log returns.
    for (int t = length - 1; t < n; ++t) {
        const int start = t + 1 - length;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int j = start; j <= t; ++j) {
            const double curr = data[j];
            if (!isfinite(curr) || curr <= 0.0) {
                valid = false;
                break;
            }

            double ret;
            if (j == 0) {
                ret = 0.0;
            } else {
                const double prev = data[j - 1];
                if (!isfinite(prev) || prev <= 0.0) {
                    ret = 0.0;
                } else {
                    ret = log(curr / prev);
                }
            }

            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            continue;
        }

        const double nf = static_cast<double>(length);
        const double mean = sum / nf;
        double centered = sumsq - mean * mean * nf;
        if (centered < 0.0) {
            centered = 0.0;
        }
        const double sample_var = centered / static_cast<double>(length - 1);
        row[t] = sqrt(sample_var) * sqrt(static_cast<double>(annual_length));
    }

    // Pass 2 -- the percentile rank of the current HV inside its annual
    // window, in place, walking DOWNWARDS.
    for (int t = n - 1; t >= annual_length - 1; --t) {
        const int start = t + 1 - annual_length;
        bool valid = true;
        int rank = 0;
        const double current_hv = row[t];

        for (int j = start; j <= t; ++j) {
            const double hv = row[j];
            if (!isfinite(hv)) {
                valid = false;
                break;
            }
            rank += static_cast<int>(hv < current_hv);
        }

        if (!valid) {
            row[t] = CUDART_NAN;
            continue;
        }

        row[t] = static_cast<double>(rank) * (100.0 / static_cast<double>(annual_length));
    }

    // Every bar before the first full annual window still carries an HV value
    // the CPU never publishes as `hvp`.
    const int warm = (annual_length - 1) < n ? (annual_length - 1) : n;
    for (int t = 0; t < warm; ++t) {
        row[t] = CUDART_NAN;
    }
}
