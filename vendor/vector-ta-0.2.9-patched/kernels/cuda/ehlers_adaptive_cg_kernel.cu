#include <cmath>
#include <cstddef>

static __device__ inline double ehlers_adaptive_cg_median3(double a, double b, double c) {
    return (a + b + c) - fmin(a, fmin(b, c)) - fmax(a, fmax(b, c));
}

extern "C" __global__ void ehlers_adaptive_cg_batch_f64(
    const double* __restrict__ data,
    int len,
    const double* __restrict__ alphas,
    int rows,
    double* __restrict__ out_cg,
    double* __restrict__ out_trigger
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double alpha = alphas[row];
    double* row_out_cg = out_cg + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger = out_trigger + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_cg[i] = NAN;
        row_out_trigger[i] = NAN;
    }

    if (!isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
        return;
    }

    int first_valid = -1;
    for (int i = 0; i < len; ++i) {
        if (!isnan(data[i])) {
            first_valid = i;
            break;
        }
    }
    if (first_valid < 0 || len - first_valid < 14) {
        return;
    }

    double smooth_hist[3] = {0.0, 0.0, 0.0};
    double cycle_hist[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_hist[2] = {0.0, 0.0};
    double dp_hist[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
    double ip_hist[2] = {0.0, 0.0};
    double p_hist[2] = {0.0, 0.0};

    double alpha_half = 1.0 - 0.5 * alpha;
    double alpha_half_sq = alpha_half * alpha_half;
    double one_minus_alpha = 1.0 - alpha;
    double one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    for (int i = 0; i < len; ++i) {
        double cg_cur = NAN;
        double smooth_cur = 0.0;
        double cycle_cur = 0.0;
        double q1_cur = 0.0;
        double dp_cur = 0.1;
        double ip_cur = 0.0;
        double p_cur = 0.0;

        if (i > first_valid) {
            row_out_trigger[i] = row_out_cg[i - 1];
        }

        if (i >= first_valid) {
            double x0 = data[i];
            if (!isnan(x0)) {
                double x1 = (i >= 1) ? data[i - 1] : x0;
                double x2 = (i >= 2) ? data[i - 2] : x1;
                double x3 = (i >= 3) ? data[i - 3] : x2;

                smooth_cur = (x0 + 2.0 * x1 + 2.0 * x2 + x3) / 6.0;

                if (i < first_valid + 7) {
                    cycle_cur = (x0 - 2.0 * x1 + x2) * 0.25;
                } else {
                    double smooth_prev1 = smooth_hist[(i - 1) % 3];
                    double smooth_prev2 = smooth_hist[(i - 2) % 3];
                    double cycle_prev1 = cycle_hist[(i - 1) % 7];
                    double cycle_prev2 = cycle_hist[(i - 2) % 7];
                    cycle_cur = alpha_half_sq * (smooth_cur - 2.0 * smooth_prev1 + smooth_prev2)
                        + 2.0 * one_minus_alpha * cycle_prev1
                        - one_minus_alpha_sq * cycle_prev2;
                }

                double ip_prev = (i >= 1) ? ip_hist[(i - 1) % 2] : 0.0;
                if (i >= first_valid + 6) {
                    double cycle_m2 = cycle_hist[(i - 2) % 7];
                    double cycle_m4 = cycle_hist[(i - 4) % 7];
                    double cycle_m6 = cycle_hist[(i - 6) % 7];
                    q1_cur = (0.0962 * cycle_cur + 0.5769 * cycle_m2 - 0.5769 * cycle_m4 -
                              0.0962 * cycle_m6) *
                        (0.5 + 0.08 * ip_prev);
                }

                if (i >= first_valid + 7) {
                    double i1 = cycle_hist[(i - 3) % 7];
                    double prev_i1 = cycle_hist[(i - 4) % 7];
                    double prev_q = q1_hist[(i - 1) % 2];
                    if (fabs(q1_cur) > 1e-12 && fabs(prev_q) > 1e-12) {
                        double raw = (i1 / q1_cur - prev_i1 / prev_q) /
                            (1.0 + i1 * prev_i1 / (q1_cur * prev_q));
                        if (raw < 0.1) {
                            raw = 0.1;
                        } else if (raw > 1.1) {
                            raw = 1.1;
                        }
                        dp_cur = raw;
                    }
                }

                double md = 0.1;
                if (i >= first_valid + 4) {
                    md = ehlers_adaptive_cg_median3(
                        dp_cur,
                        dp_hist[(i - 1) % 5],
                        ehlers_adaptive_cg_median3(
                            dp_hist[(i - 2) % 5],
                            dp_hist[(i - 3) % 5],
                            dp_hist[(i - 4) % 5]
                        )
                    );
                }

                double dc = (2.0 * 3.14159265358979323846) / md + 0.5;
                if (i == first_valid) {
                    ip_cur = dc;
                    p_cur = ip_cur;
                } else {
                    double prev_ip = ip_hist[(i - 1) % 2];
                    double prev_p = p_hist[(i - 1) % 2];
                    ip_cur = 0.33 * dc + 0.67 * prev_ip;
                    p_cur = 0.15 * ip_cur + 0.85 * prev_p;
                }

                int window = static_cast<int>(llround(p_cur * 0.5));
                if (window < 1) {
                    window = 1;
                } else if (window > 100) {
                    window = 100;
                }

                if (i + 1 >= first_valid + window && i + 1 >= window) {
                    double numerator = 0.0;
                    double denominator = 0.0;
                    bool has_nan = false;
                    for (int lag = 0; lag < window; ++lag) {
                        double value = data[i - lag];
                        if (isnan(value)) {
                            has_nan = true;
                            break;
                        }
                        numerator += (static_cast<double>(lag) + 1.0) * value;
                        denominator += value;
                    }
                    if (!has_nan) {
                        if (fabs(denominator) > 1e-12) {
                            cg_cur = -numerator / denominator +
                                (static_cast<double>(window) + 1.0) * 0.5;
                        } else {
                            cg_cur = 0.0;
                        }
                    }
                }
            }
        }

        smooth_hist[i % 3] = smooth_cur;
        cycle_hist[i % 7] = cycle_cur;
        q1_hist[i % 2] = q1_cur;
        dp_hist[i % 5] = dp_cur;
        ip_hist[i % 2] = ip_cur;
        p_hist[i % 2] = p_cur;
        row_out_cg[i] = cg_cur;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/ehlers_adaptive_cg.rs:475
// (ehlers_adaptive_cg_with_kernel). The column this emits is cg, which is what
// output_id == "value" resolves to (dispatch/cpu_batch.rs:15812).
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- a cascade
// of carried rings (smooth, cycle, q1, delta-phase, in-phase, period) feeding a
// PER-BAR WINDOW LENGTH: window = round(p * 0.5), where p is itself the output
// of two chained one-pole filters. A bar-parallel form would have to know the
// window before it knows the filter state.
//
// PERIOD-INVARIANT. compute_ehlers_adaptive_cg_batch (cpu_batch.rs:15801)
// reads alpha and NEVER period, so five swept periods give five identical CPU
// columns and this kernel emits five identical rows. The CPU default 0.07 is
// pinned below.
//
// SOURCE IS hl2, NOT close: extract_slice_input("ehlers_adaptive_cg",
// req.data, "hl2") at cpu_batch.rs:15793. The lane row therefore declares
// F64InputKind::Hl2Slice; handing this kernel close would compute a different
// indicator and pass every length check on the way through.
//
// FIRST VALID IS DERIVED HERE rather than taken from the caller: the CPU pairs
// its first-non-NaN scan with a "at least 14 bars remain" requirement, and the
// warmup gates below (first_valid + 4, + 6, + 7) all hang off it. Doing the
// scan in the kernel keeps the two halves of one rule in one place. The lane
// row declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double fabs/fmin/fmax/llround, no
// f32-suffixed math function and no fast-math intrinsic. The two 1e-12 guards
// are f64-sized -- they are the crate's own guard for a ratio whose
// denominator is a filtered price, NOT an f32 epsilon carried forward.
// ---------------------------------------------------------------------------

#define NEO_EACG_ALPHA 0.07

extern "C" __global__ void ehlers_adaptive_cg_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid_in,
    double* __restrict__ out
) {
    const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row_idx >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid_in;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const double alpha = NEO_EACG_ALPHA;
    if (!isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
        return;
    }

    int first_valid = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(data[i])) {
            first_valid = i;
            break;
        }
    }
    if (first_valid < 0 || n - first_valid < 14) {
        return;
    }

    double smooth_hist[3] = {0.0, 0.0, 0.0};
    double cycle_hist[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_hist[2] = {0.0, 0.0};
    double dp_hist[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
    double ip_hist[2] = {0.0, 0.0};
    double p_hist[2] = {0.0, 0.0};

    const double alpha_half = 1.0 - 0.5 * alpha;
    const double alpha_half_sq = alpha_half * alpha_half;
    const double one_minus_alpha = 1.0 - alpha;
    const double one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    for (int i = 0; i < n; ++i) {
        double cg_cur = NAN;
        double smooth_cur = 0.0;
        double cycle_cur = 0.0;
        double q1_cur = 0.0;
        double dp_cur = 0.1;
        double ip_cur = 0.0;
        double p_cur = 0.0;

        if (i >= first_valid) {
            const double x0 = data[i];
            if (!isnan(x0)) {
                const double x1 = (i >= 1) ? data[i - 1] : x0;
                const double x2 = (i >= 2) ? data[i - 2] : x1;
                const double x3 = (i >= 3) ? data[i - 3] : x2;

                smooth_cur = (x0 + 2.0 * x1 + 2.0 * x2 + x3) / 6.0;

                if (i < first_valid + 7) {
                    cycle_cur = (x0 - 2.0 * x1 + x2) * 0.25;
                } else {
                    const double smooth_prev1 = smooth_hist[(i - 1) % 3];
                    const double smooth_prev2 = smooth_hist[(i - 2) % 3];
                    const double cycle_prev1 = cycle_hist[(i - 1) % 7];
                    const double cycle_prev2 = cycle_hist[(i - 2) % 7];
                    cycle_cur = alpha_half_sq * (smooth_cur - 2.0 * smooth_prev1 + smooth_prev2) +
                        2.0 * one_minus_alpha * cycle_prev1 -
                        one_minus_alpha_sq * cycle_prev2;
                }

                const double ip_prev = (i >= 1) ? ip_hist[(i - 1) % 2] : 0.0;
                if (i >= first_valid + 6) {
                    const double cycle_m2 = cycle_hist[(i - 2) % 7];
                    const double cycle_m4 = cycle_hist[(i - 4) % 7];
                    const double cycle_m6 = cycle_hist[(i - 6) % 7];
                    q1_cur = (0.0962 * cycle_cur + 0.5769 * cycle_m2 - 0.5769 * cycle_m4 -
                              0.0962 * cycle_m6) *
                        (0.5 + 0.08 * ip_prev);
                }

                if (i >= first_valid + 7) {
                    const double i1 = cycle_hist[(i - 3) % 7];
                    const double prev_i1 = cycle_hist[(i - 4) % 7];
                    const double prev_q = q1_hist[(i - 1) % 2];
                    if (fabs(q1_cur) > 1e-12 && fabs(prev_q) > 1e-12) {
                        double raw = (i1 / q1_cur - prev_i1 / prev_q) /
                            (1.0 + i1 * prev_i1 / (q1_cur * prev_q));
                        if (raw < 0.1) {
                            raw = 0.1;
                        } else if (raw > 1.1) {
                            raw = 1.1;
                        }
                        dp_cur = raw;
                    }
                }

                double md = 0.1;
                if (i >= first_valid + 4) {
                    md = ehlers_adaptive_cg_median3(
                        dp_cur,
                        dp_hist[(i - 1) % 5],
                        ehlers_adaptive_cg_median3(
                            dp_hist[(i - 2) % 5],
                            dp_hist[(i - 3) % 5],
                            dp_hist[(i - 4) % 5]
                        )
                    );
                }

                const double dc = (2.0 * 3.14159265358979323846) / md + 0.5;
                if (i == first_valid) {
                    ip_cur = dc;
                    p_cur = ip_cur;
                } else {
                    const double prev_ip = ip_hist[(i - 1) % 2];
                    const double prev_p = p_hist[(i - 1) % 2];
                    ip_cur = 0.33 * dc + 0.67 * prev_ip;
                    p_cur = 0.15 * ip_cur + 0.85 * prev_p;
                }

                int window = static_cast<int>(llround(p_cur * 0.5));
                if (window < 1) {
                    window = 1;
                } else if (window > 100) {
                    window = 100;
                }

                if (i + 1 >= first_valid + window && i + 1 >= window) {
                    double numerator = 0.0;
                    double denominator = 0.0;
                    bool has_nan = false;
                    for (int lag = 0; lag < window; ++lag) {
                        const double value = data[i - lag];
                        if (isnan(value)) {
                            has_nan = true;
                            break;
                        }
                        numerator += (static_cast<double>(lag) + 1.0) * value;
                        denominator += value;
                    }
                    if (!has_nan) {
                        if (fabs(denominator) > 1e-12) {
                            cg_cur = -numerator / denominator +
                                (static_cast<double>(window) + 1.0) * 0.5;
                        } else {
                            cg_cur = 0.0;
                        }
                    }
                }
            }
        }

        smooth_hist[i % 3] = smooth_cur;
        cycle_hist[i % 7] = cycle_cur;
        q1_hist[i % 2] = q1_cur;
        dp_hist[i % 5] = dp_cur;
        ip_hist[i % 2] = ip_cur;
        p_hist[i % 2] = p_cur;
        row[i] = cg_cur;
    }
}
