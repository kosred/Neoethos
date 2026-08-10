#include <cuda_runtime.h>
#include <math.h>

static __device__ inline float f32_nan() {
    return __int_as_float(0x7fffffff);
}


static __device__ inline int window_has_nan(const int* __restrict__ pref_nan, int t1, int t0) {
    return (pref_nan[t1] - pref_nan[t0]) != 0;
}

static __device__ inline double window_sum(const double* __restrict__ pref, int t1, int t0) {
    return pref[t1] - pref[t0];
}

extern "C" __global__ void macz_build_prefix_single_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    double* __restrict__ pref_close_sum,
    double* __restrict__ pref_close_sumsq,
    int* __restrict__ pref_close_nan,
    double* __restrict__ pref_vol_sum,
    double* __restrict__ pref_pv_sum,
    int* __restrict__ pref_vol_nan) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    pref_close_sum[0] = 0.0;
    pref_close_sumsq[0] = 0.0;
    pref_close_nan[0] = 0;
    if (pref_vol_sum) pref_vol_sum[0] = 0.0;
    if (pref_pv_sum) pref_pv_sum[0] = 0.0;
    if (pref_vol_nan) pref_vol_nan[0] = 0;

    double acc_close = 0.0;
    double acc_close_sq = 0.0;
    int acc_close_nan = 0;
    double acc_vol = 0.0;
    double acc_pv = 0.0;
    int acc_vol_nan = 0;

    for (int i = 0; i < len; ++i) {
        const double c = (double)close[i];
        if (isnan(c)) {
            acc_close_nan += 1;
        } else {
            acc_close += c;
            acc_close_sq += c * c;
        }
        pref_close_sum[i + 1] = acc_close;
        pref_close_sumsq[i + 1] = acc_close_sq;
        pref_close_nan[i + 1] = acc_close_nan;

        if (pref_vol_sum && pref_pv_sum && pref_vol_nan) {
            const double v = (double)volume[i];
            if (isnan(c) || isnan(v)) {
                acc_vol_nan += 1;
            } else {
                acc_vol += v;
                acc_pv += v * c;
            }
            pref_vol_sum[i + 1] = acc_vol;
            pref_pv_sum[i + 1] = acc_pv;
            pref_vol_nan[i + 1] = acc_vol_nan;
        }
    }
}


extern "C" __global__ void macz_batch_macz_tmp_f32(

    const float* __restrict__ close,
    const float* __restrict__ volume,
    const double* __restrict__ pref_close_sum,
    const double* __restrict__ pref_close_sumsq,
    const int* __restrict__ pref_close_nan,
    const double* __restrict__ pref_vol_sum,
    const double* __restrict__ pref_pv_sum,
    const int* __restrict__ pref_vol_nan,

    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ lzs,
    const int* __restrict__ lsds,
    const float* __restrict__ a_s,
    const float* __restrict__ b_s,

    int len,
    int first_valid,
    int n_rows,
    int use_sma_for_vwap,

    float* __restrict__ macz_tmp
) {
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    if (row >= n_rows || t >= len) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int lz = lzs[row];
    const int lsd = lsds[row];
    const float a = a_s[row];
    const float b = b_s[row];

    const int warm_m = first_valid + max(max(s, lz), lsd) - 1;
    const int row_off = row * len;
    if (t < warm_m) {
        macz_tmp[row_off + t] = f32_nan();
        return;
    }


    double mean_vwap = NAN;
    {
        const int t1 = t + 1;
        const int t0 = t + 1 - lz;
        if (!use_sma_for_vwap && volume != nullptr) {

            if (!window_has_nan(pref_close_nan, t1, t0) && !window_has_nan(pref_vol_nan, t1, t0)) {
                const double vol_sum = window_sum(pref_vol_sum, t1, t0);
                if (vol_sum > 0.0) {
                    const double pv_sum = window_sum(pref_pv_sum, t1, t0);
                    mean_vwap = pv_sum / vol_sum;
                }
            }
        } else {

            if (!window_has_nan(pref_close_nan, t1, t0)) {
                const double ssum = window_sum(pref_close_sum, t1, t0);
                mean_vwap = ssum / (double)lz;
            }
        }
    }


    double z = NAN;
    if (!isnan(mean_vwap)) {
        const int t1 = t + 1;
        const int t0 = t + 1 - lz;
        if (!window_has_nan(pref_close_nan, t1, t0)) {
            const double ssum = window_sum(pref_close_sum, t1, t0);
            const double ssum2 = window_sum(pref_close_sumsq, t1, t0);
            const double e = ssum / (double)lz;
            const double e2 = ssum2 / (double)lz;
            double var = fma(-2.0 * mean_vwap, e, e2) + (mean_vwap * mean_vwap);
            if (var > 0.0) {
                const double std = sqrt(var);
                const double x = (double)close[t];
                z = (x - mean_vwap) / std;
            } else {
                z = 0.0;
            }
        }
    }


    double macd = NAN;
    {
        const int t1s = t + 1;
        const int t0s = t + 1 - s;
        const int t1f = t + 1;
        const int t0f = t + 1 - f;
        if (!window_has_nan(pref_close_nan, t1s, t0s) && !window_has_nan(pref_close_nan, t1f, t0f)) {
            const double slow_mean = window_sum(pref_close_sum, t1s, t0s) / (double)s;
            const double fast_mean = window_sum(pref_close_sum, t1f, t0f) / (double)f;
            macd = fast_mean - slow_mean;
        }
    }


    double sd = NAN;
    {
        const int t1d = t + 1;
        const int t0d = t + 1 - lsd;
        if (!window_has_nan(pref_close_nan, t1d, t0d)) {
            const double mean = window_sum(pref_close_sum, t1d, t0d) / (double)lsd;
            const double s2 = window_sum(pref_close_sumsq, t1d, t0d) / (double)lsd;
            const double var = s2 - mean * mean;
            if (var > 0.0) sd = sqrt(var);
        }
    }

    float macz_raw = f32_nan();
    if (!isnan(z) && !isnan(macd) && !isnan(sd) && sd > 0.0) {
        const double val = (double)z * (double)a + ((double)macd / (double)sd) * (double)b;
        macz_raw = (float)val;
    }

    macz_tmp[row_off + t] = macz_raw;
}


extern "C" __global__ void macz_batch_hist_from_macz_f32(

    const float* __restrict__ macz_tmp,

    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    const int* __restrict__ lzs,
    const int* __restrict__ lsds,

    int len,
    int first_valid,
    int n_rows,

    float* __restrict__ out_hist
) {
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    if (row >= n_rows || t >= len) return;

    const int s = slows[row];
    const int g = sigs[row];
    const int lz = lzs[row];
    const int lsd = lsds[row];
    const int warm_m = first_valid + max(max(s, lz), lsd) - 1;
    const int warm_hist = warm_m + g - 1;

    const int row_off = row * len;
    if (t < warm_hist) {
        out_hist[row_off + t] = f32_nan();
        return;
    }


    double sum = 0.0;
    bool any_nan = false;
    const int start = t + 1 - g;
    for (int j = start; j <= t; ++j) {
        const float mv = macz_tmp[row_off + j];
        if (isnan(mv)) { any_nan = true; break; }
        sum += (double)mv;
    }
    if (any_nan) {
        out_hist[row_off + t] = f32_nan();
    } else {
        const float signal = (float)(sum / (double)g);
        out_hist[row_off + t] = macz_tmp[row_off + t] - signal;
    }
}


extern "C" __global__ void macz_batch_f32(

    const float* __restrict__ close,
    const float* __restrict__ volume,
    const double* __restrict__ pref_close_sum,
    const double* __restrict__ pref_close_sumsq,
    const int* __restrict__ pref_close_nan,
    const double* __restrict__ pref_vol_sum,
    const double* __restrict__ pref_pv_sum,
    const int* __restrict__ pref_vol_nan,

    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    const int* __restrict__ lzs,
    const int* __restrict__ lsds,
    const float* __restrict__ a_s,
    const float* __restrict__ b_s,
    const int* __restrict__ use_lag_s,
    const float* __restrict__ gammas,

    int len,
    int first_valid,
    int n_rows,
    int use_sma_for_vwap,

    float* __restrict__ macz_tmp,
    float* __restrict__ out_hist
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int g = sigs[row];
    const int lz = lzs[row];
    const int lsd = lsds[row];
    const float a = a_s[row];
    const float b = b_s[row];
    const int use_lag = use_lag_s[row] != 0;
    const double gamma = (double)gammas[row];

    const int warm_m = first_valid + max(max(s, lz), lsd) - 1;
    const int warm_hist = warm_m + g - 1;
    const int row_off = row * len;


    for (int i = 0; i < len; ++i) {
        macz_tmp[row_off + i] = f32_nan();
        out_hist[row_off + i] = f32_nan();
    }


    double l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;

    for (int t = warm_m; t < len; ++t) {

        double mean_vwap = NAN;
        if (t >= first_valid + lz - 1) {
            const int t1 = t + 1;
            const int t0 = t + 1 - lz;
            if (!use_sma_for_vwap && volume != nullptr) {

                if (!window_has_nan(pref_close_nan, t1, t0) && !window_has_nan(pref_vol_nan, t1, t0)) {
                    const double vol_sum = window_sum(pref_vol_sum, t1, t0);
                    if (vol_sum > 0.0) {
                        const double pv_sum = window_sum(pref_pv_sum, t1, t0);
                        mean_vwap = pv_sum / vol_sum;
                    }
                }
            } else {

                if (!window_has_nan(pref_close_nan, t1, t0)) {
                    const double ssum = window_sum(pref_close_sum, t1, t0);
                    mean_vwap = ssum / (double)lz;
                }
            }
        }


        double z = NAN;
        if (!isnan(mean_vwap)) {
            const int t1 = t + 1;
            const int t0 = t + 1 - lz;
            if (!window_has_nan(pref_close_nan, t1, t0)) {
                const double ssum = window_sum(pref_close_sum, t1, t0);
                const double ssum2 = window_sum(pref_close_sumsq, t1, t0);
                const double e = ssum / (double)lz;
                const double e2 = ssum2 / (double)lz;
                double var = fma(-2.0 * mean_vwap, e, e2) + (mean_vwap * mean_vwap);
                if (var > 0.0) {
                    const double std = sqrt(var);
                    const double x = (double)close[t];
                    z = (x - mean_vwap) / std;
                } else {
                    z = 0.0;
                }
            }
        }


        double macd = NAN;
        if (t >= first_valid + s - 1) {
            const int t1s = t + 1;
            const int t0s = t + 1 - s;
            const int t1f = t + 1;
            const int t0f = t + 1 - f;
            if (!window_has_nan(pref_close_nan, t1s, t0s) && !window_has_nan(pref_close_nan, t1f, t0f)) {
                const double slow_mean = window_sum(pref_close_sum, t1s, t0s) / (double)s;
                const double fast_mean = window_sum(pref_close_sum, t1f, t0f) / (double)f;
                macd = fast_mean - slow_mean;
            }
        }


        double sd = NAN;
        if (t >= first_valid + lsd - 1) {
            const int t1d = t + 1;
            const int t0d = t + 1 - lsd;
            if (!window_has_nan(pref_close_nan, t1d, t0d)) {
                const double mean = window_sum(pref_close_sum, t1d, t0d) / (double)lsd;
                const double s2 = window_sum(pref_close_sumsq, t1d, t0d) / (double)lsd;
                const double var = s2 - mean * mean;
                if (var > 0.0) sd = sqrt(var);
            }
        }

        float macz_raw = f32_nan();
        if (!isnan(z) && !isnan(macd) && !isnan(sd) && sd > 0.0) {
            const double val = (double)z * (double)a + ((double)macd / (double)sd) * (double)b;
            macz_raw = (float)val;
        }

        float macz_val = macz_raw;
        if (use_lag) {
            if (isnan(macz_raw)) {
                macz_val = f32_nan();
            } else {
                const double s_in = (double)macz_raw;
                const double one_minus_g = 1.0 - gamma;
                const double new_l0 = one_minus_g * s_in + gamma * l0;
                const double new_l1 = -gamma * new_l0 + l0 + gamma * l1;
                const double new_l2 = -gamma * new_l1 + l1 + gamma * l2;
                const double new_l3 = -gamma * new_l2 + l2 + gamma * l3;
                l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
                const double outv = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
                macz_val = (float)outv;
            }
        }

        macz_tmp[row_off + t] = macz_val;


        if (t >= warm_hist) {

            double sum = 0.0;
            bool any_nan = false;
            const int start = t + 1 - g;
            for (int j = start; j <= t; ++j) {
                const float mv = macz_tmp[row_off + j];
                if (isnan(mv)) { any_nan = true; break; }
                sum += (double)mv;
            }
            if (!any_nan) {
                const float signal = (float)(sum / (double)g);
                const float hv = macz_val - signal;
                out_hist[row_off + t] = hv;
            }
        }
    }
}

extern "C" __global__ void macz_many_series_one_param_time_major_f32(

    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const double* __restrict__ pref_close_sum_tm,
    const double* __restrict__ pref_close_sumsq_tm,
    const int* __restrict__ pref_close_nan_tm,
    const double* __restrict__ pref_vol_sum_tm,
    const double* __restrict__ pref_pv_sum_tm,
    const int* __restrict__ pref_vol_nan_tm,
    int cols,
    int rows,

    int fast,
    int slow,
    int sig,
    int lz,
    int lsd,
    float a,
    float b,
    int use_lag,
    float gamma_f,
    const int* __restrict__ first_valids,
    int use_sma_for_vwap,

    float* __restrict__ macz_tm,
    float* __restrict__ hist_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    const int off_pref = s * (rows + 1);

    const double* pcs = pref_close_sum_tm + off_pref;
    const double* pcsq = pref_close_sumsq_tm + off_pref;
    const int* pcn = pref_close_nan_tm + off_pref;
    const double* pvs = pref_vol_sum_tm ? (pref_vol_sum_tm + off_pref) : nullptr;
    const double* pps = pref_pv_sum_tm ? (pref_pv_sum_tm + off_pref) : nullptr;
    const int* pvn = pref_vol_nan_tm ? (pref_vol_nan_tm + off_pref) : nullptr;

    const int fv = first_valids[s];
    if (fv < 0) return;
    const int warm_m = fv + max(max(slow, lz), lsd) - 1;
    const int warm_hist = warm_m + sig - 1;

    auto at = [&](int t) { return t * cols + s; };
    for (int t = 0; t < rows; ++t) { macz_tm[at(t)] = f32_nan(); hist_tm[at(t)] = f32_nan(); }

    double l0=0.0,l1=0.0,l2=0.0,l3=0.0;
    const double gamma = (double)gamma_f;

    for (int t = warm_m; t < rows; ++t) {

        double mean_vwap = NAN;
        if (t >= fv + lz - 1) {
            const int t1 = t + 1;
            const int t0 = t + 1 - lz;
            if (!use_sma_for_vwap && volume_tm) {
                if (!window_has_nan(pcn, t1, t0) && !window_has_nan(pvn, t1, t0)) {
                    const double vs = window_sum(pvs, t1, t0);
                    if (vs > 0.0) {
                        const double pv = window_sum(pps, t1, t0);
                        mean_vwap = pv / vs;
                    }
                }
            } else {
                if (!window_has_nan(pcn, t1, t0)) {
                    mean_vwap = window_sum(pcs, t1, t0) / (double)lz;
                }
            }
        }


        double z = NAN;
        if (!isnan(mean_vwap)) {
            const int t1 = t + 1, t0 = t + 1 - lz;
            if (!window_has_nan(pcn, t1, t0)) {
                const double s2 = window_sum(pcsq, t1, t0) / (double)lz;
                const double s1 = window_sum(pcs, t1, t0) / (double)lz;
                const double var = fma(-2.0 * mean_vwap, s1, s2) + (mean_vwap * mean_vwap);
                if (var > 0.0) {
                    const double std = sqrt(var);
                    const double x = (double)close_tm[at(t)];
                    z = (x - mean_vwap) / std;
                } else {
                    z = 0.0;
                }
            }
        }


        double macd = NAN;
        if (t >= fv + slow - 1) {
            const int t1s = t + 1, t0s = t + 1 - slow;
            const int t1f = t + 1, t0f = t + 1 - fast;
            if (!window_has_nan(pcn, t1s, t0s) && !window_has_nan(pcn, t1f, t0f)) {
                const double slow_m = window_sum(pcs, t1s, t0s) / (double)slow;
                const double fast_m = window_sum(pcs, t1f, t0f) / (double)fast;
                macd = fast_m - slow_m;
            }
        }


        double sd = NAN;
        if (t >= fv + lsd - 1) {
            const int t1d = t + 1, t0d = t + 1 - lsd;
            if (!window_has_nan(pcn, t1d, t0d)) {
                const double mean = window_sum(pcs, t1d, t0d) / (double)lsd;
                const double s2 = window_sum(pcsq, t1d, t0d) / (double)lsd;
                const double var = s2 - mean * mean;
                if (var > 0.0) sd = sqrt(var);
            }
        }

        float macz_raw = f32_nan();
        if (!isnan(z) && !isnan(macd) && !isnan(sd) && sd > 0.0) {
            const double val = (double)z * (double)a + ((double)macd / (double)sd) * (double)b;
            macz_raw = (float)val;
        }

        float macz_val = macz_raw;
        if (use_lag) {
            if (isnan(macz_raw)) {
                macz_val = f32_nan();
            } else {
                const double s_in = (double)macz_raw;
                const double one_minus_g = 1.0 - gamma;
                const double new_l0 = one_minus_g * s_in + gamma * l0;
                const double new_l1 = -gamma * new_l0 + l0 + gamma * l1;
                const double new_l2 = -gamma * new_l1 + l1 + gamma * l2;
                const double new_l3 = -gamma * new_l2 + l2 + gamma * l3;
                l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
                const double outv = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
                macz_val = (float)outv;
            }
        }

        macz_tm[at(t)] = macz_val;

        if (t >= warm_hist) {
            double sum = 0.0; bool any_nan = false;
            const int start = t + 1 - sig;
            for (int j = start; j <= t; ++j) {
                const float mv = macz_tm[at(j)];
                if (isnan(mv)) { any_nan = true; break; }
                sum += (double)mv;
            }
            if (!any_nan) {
                const float signal = (float)(sum / (double)sig);
                hist_tm[at(t)] = macz_val - signal;
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3   (macz)
 *
 * CPU reference: `macz_scalar_classic` (src/indicators/macz.rs:806), reached
 *   because `macz_prepare` (:650) maps `Kernel::Auto` to `Kernel::Scalar`
 *   OUTRIGHT (:726-729) and `macz_compute_into_tail_only` (:764) then IGNORES
 *   the kernel argument entirely (`let _ = kernel;`). There is exactly one CPU
 *   path, so there is exactly one oracle.
 *
 * Column: output_id "value" -> `out.values` (cpu_batch.rs:15415-15417).
 *
 * PERIOD-INVARIANT: `compute_macz_batch` reads `fast_length` (12),
 *   `slow_length` (25), `signal_length` (9), `lengthz` (20), `length_stdev`
 *   (25), `a` (1.0), `b` (1.0), `use_lag` (false) and `gamma` (0.02)
 *   (cpu_batch.rs:15385-15393) and NEVER `period`.
 *
 * Input: (close, volume) -- `ensure_same_len_2("macz", close.len(),
 *   volume.len())` (cpu_batch.rs:15347) -> F64InputKind::CloseVolume, and the
 *   volume branch of the CPU IS taken (`has_volume` true), which changes the
 *   vwap term from a plain mean to a volume weighting.
 *
 * FIRST-VALID IGNORED, and this is not laziness: `macz_prepare` :678-681 scans
 *   CLOSE ALONE (`data.iter().position(|x| !x.is_nan())`). Volume is never
 *   scanned -- a NaN volume is handled INSIDE the loop by `n_vwap_nan`
 *   (:894-899), which makes the vwap NaN for that window instead of moving the
 *   series start. Declaring `AllInputsNonNan` over CloseVolume would adopt
 *   volume's first non-NaN too and SHIFT every window on any frame whose
 *   volume starts later than its close. So the kernel derives its own index
 *   and the caller's value is genuinely unused.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Eight accumulators are carried
 *   subtract-then-add (sum_fast, sum_slow, sum_lz, sum2_lz, sum_lsd, sum2_lsd,
 *   sum_pv, sum_v) plus a `signal_length`-deep ring, and the output at bar i
 *   is a difference against a mean of the previous `sig` outputs.
 *
 * Roundings, counted against the CPU lines:
 *   :896   sum_pv = x.mul_add(v, sum_pv)                      -- ONE fma
 *   :888   sum_lz = sum_lz + x, sum2_lz = sum2_lz + x * x     -- plain, NO fma
 *   :983   (-2.0 * vwap_i).mul_add(e, e2) + vwap_i * vwap_i   -- ONE fma
 *   :1011  zvwap.mul_add(a, (macd / sd_src) * b)              -- ONE fma
 *   :1000  (e2 - e * e).max(0.0).sqrt()                       -- plain
 *   Every mul_add above becomes an fma; every plain add/multiply stays plain.
 *   The count matches line for line -- writing fma(x, x, sum2_lz) for :889
 *   would REMOVE a rounding the reference performs.
 *
 * NaN semantics: `var.max(0.0)` (:984) and `(e2 - e*e).max(0.0)` (:1000) are
 *   `f64::max`, which returns the NON-NaN operand -- so a NaN variance becomes
 *   0.0 on the CPU. `fmax` is used here for exactly that reason; the `>= 0.0`
 *   if-chain that reads naturally in C would let the NaN through and poison
 *   `sd`, `zvwap` and then every bar of the signal ring.
 *
 * Epsilons: there is not one tolerance constant on this path. The only
 *   comparisons are exact (`sd > 0.0`, `sum_v > 0.0`, `sd_src > 0.0`), so
 *   there is no f32-sized epsilon to re-derive -- the f32-era constants
 *   elsewhere in this file belong to the f32 entry points and are untouched.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:15385-15393. */
#define NEO_MACZ_FAST     12
#define NEO_MACZ_SLOW     25
#define NEO_MACZ_SIG      9
#define NEO_MACZ_LZ       20
#define NEO_MACZ_LSD      25
#define NEO_MACZ_A        1.0
#define NEO_MACZ_B        1.0
#define NEO_MACZ_USE_LAG  0
#define NEO_MACZ_GAMMA    0.02

extern "C" __global__
void macz_neo_batch_f64(const double* __restrict__ close,
                        const double* __restrict__ volume,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* CLOSE-ONLY scan, derived below -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    fast  = NEO_MACZ_FAST;
    const int    slow  = NEO_MACZ_SLOW;
    const int    sig   = NEO_MACZ_SIG;
    const int    lz    = NEO_MACZ_LZ;
    const int    lsd   = NEO_MACZ_LSD;
    const double a     = NEO_MACZ_A;
    const double b     = NEO_MACZ_B;

    /* `macz_prepare` :678-681 -- first non-NaN of CLOSE alone. */
    int first = -1;
    for (int i = 0; i < n; ++i) { if (!isnan(close[i])) { first = i; break; } }
    if (first < 0) return;                       /* AllValuesNaN */

    int need = fast;
    if (slow > need) need = slow;
    if (lz   > need) need = lz;
    if (lsd  > need) need = lsd;
    if (n - first < need) return;                /* :700 NotEnoughValidData */

    const int fast_start = first + fast - 1;
    const int slow_start = first + slow - 1;
    const int lz_start   = first + lz - 1;
    const int lsd_start  = first + lsd - 1;
    const int warm_m     = first + need - 1;               /* :831 */
    const int warm_hist  = first + need + sig - 2;         /* macz_warm_len :646 */

    double sum_fast = 0.0, sum_slow = 0.0;
    int    n_fast_nan = 0, n_slow_nan = 0;
    double sum_lz = 0.0, sum2_lz = 0.0, sum_lsd = 0.0, sum2_lsd = 0.0;
    int    n_lz_nan = 0, n_lsd_nan = 0;
    double sum_pv = 0.0, sum_v = 0.0;
    int    n_vwap_nan = 0;

#if NEO_MACZ_USE_LAG
    double l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;
    const double gamma = NEO_MACZ_GAMMA;
#endif

    /* `sig_ring` (:855-862): a `signal_length`-deep ring seeded with NaN. Its
     * depth is a CPU DEFAULT, not a swept parameter, so it is sized here at
     * exactly that default and no caller-supplied number reaches it. */
    double sig_ring[NEO_MACZ_SIG];
    for (int k = 0; k < sig; ++k) sig_ring[k] = NEO_F64_NAN;
    double sig_sum   = 0.0;
    int    sig_count = 0, sig_nan = 0, sig_head = 0;

    const double inv_fast = 1.0 / (double)fast;
    const double inv_slow = 1.0 / (double)slow;
    const double inv_lz   = 1.0 / (double)lz;
    const double inv_lsd  = 1.0 / (double)lsd;
    const double inv_sig  = 1.0 / (double)sig;

    for (int i = first; i < n; ++i) {
        const double x        = close[i];
        const bool   x_is_nan = isnan(x);

        if (x_is_nan) {
            n_fast_nan += 1; n_slow_nan += 1; n_lz_nan += 1; n_lsd_nan += 1;
        } else {
            sum_fast = sum_fast + x;
            sum_slow = sum_slow + x;
            sum_lz   = sum_lz   + x;
            sum2_lz  = sum2_lz  + x * x;
            sum_lsd  = sum_lsd  + x;
            sum2_lsd = sum2_lsd + x * x;
        }

        {   /* has_volume is TRUE on this lane -- see header. */
            const double v = volume[i];
            if (x_is_nan || isnan(v)) {
                n_vwap_nan += 1;
            } else {
                sum_pv = fma(x, v, sum_pv);   /* :896 x.mul_add(v, sum_pv) */
                sum_v  = sum_v + v;
            }
        }

        if (i >= first + fast) {
            const double xo = close[i - fast];
            if (isnan(xo)) n_fast_nan -= 1; else sum_fast -= xo;
        }
        if (i >= first + slow) {
            const double xo = close[i - slow];
            if (isnan(xo)) n_slow_nan -= 1; else sum_slow -= xo;
        }
        if (i >= first + lz) {
            const double xo = close[i - lz];
            if (isnan(xo)) { n_lz_nan -= 1; }
            else { sum_lz -= xo; sum2_lz -= xo * xo; }
            const double vo = volume[i - lz];
            if (isnan(xo) || isnan(vo)) n_vwap_nan -= 1;
            else { sum_pv -= xo * vo; sum_v -= vo; }
        }
        if (i >= first + lsd) {
            const double xo = close[i - lsd];
            if (isnan(xo)) { n_lsd_nan -= 1; }
            else { sum_lsd -= xo; sum2_lsd -= xo * xo; }
        }

        const bool have_fast = (i >= fast_start) && (n_fast_nan == 0);
        const bool have_slow = (i >= slow_start) && (n_slow_nan == 0);

        const double fast_ma = have_fast ? (sum_fast * inv_fast) : NEO_F64_NAN;
        const double slow_ma = have_slow ? (sum_slow * inv_slow) : NEO_F64_NAN;
        const double macd    = (isnan(fast_ma) || isnan(slow_ma))
                                 ? NEO_F64_NAN : (fast_ma - slow_ma);

        double vwap_i = NEO_F64_NAN;
        if (i >= lz_start) {
            if (n_vwap_nan == 0 && sum_v > 0.0) vwap_i = sum_pv / sum_v;
        }

        double zvwap = NEO_F64_NAN;
        if (i >= lz_start && n_lz_nan == 0 && !isnan(vwap_i) && isfinite(x)) {
            const double e   = sum_lz  * inv_lz;
            const double e2  = sum2_lz * inv_lz;
            /* :983 -- ONE fma, then a plain add of vwap_i * vwap_i. */
            const double var = fma(-2.0 * vwap_i, e, e2) + vwap_i * vwap_i;
            /* var.max(0.0) is f64::max: a NaN var becomes 0.0. */
            const double sd  = sqrt(fmax(var, 0.0));
            zvwap = (sd > 0.0) ? ((x - vwap_i) / sd) : 0.0;
        }

        double sd_src = NEO_F64_NAN;
        if (i >= lsd_start && n_lsd_nan == 0) {
            const double e  = sum_lsd  * inv_lsd;
            const double e2 = sum2_lsd * inv_lsd;
            sd_src = sqrt(fmax(e2 - e * e, 0.0));
        }

        double macz_raw = NEO_F64_NAN;
        if (i >= warm_m && isfinite(sd_src) && sd_src > 0.0
            && isfinite(zvwap) && isfinite(macd)) {
            macz_raw = fma(zvwap, a, (macd / sd_src) * b);   /* :1011 */
        }

        double macz_val;
#if NEO_MACZ_USE_LAG
        if (isfinite(macz_raw)) {
            const double one_minus_g = 1.0 - gamma;
            const double new_l0 = fma(macz_raw, one_minus_g, gamma * l0);
            const double new_l1 = fma(-gamma, new_l0, l0 + gamma * l1);
            const double new_l2 = fma(-gamma, new_l1, l1 + gamma * l2);
            const double new_l3 = fma(-gamma, new_l2, l2 + gamma * l3);
            l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
            macz_val = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
        } else {
            macz_val = NEO_F64_NAN;
        }
#else
        macz_val = macz_raw;   /* use_lag default is FALSE (cpu_batch.rs:15392) */
#endif

        if (i >= warm_m) {
            if (sig_count == sig) {
                const double leaving = sig_ring[sig_head];
                if (isnan(leaving)) { if (sig_nan > 0) sig_nan -= 1; }
                else                { sig_sum -= leaving; }
            } else {
                sig_count += 1;
            }
            sig_ring[sig_head] = macz_val;
            if (isnan(macz_val)) sig_nan += 1; else sig_sum += macz_val;
            sig_head += 1;
            if (sig_head == sig) sig_head = 0;

            if (i >= warm_hist) {
                const double signal = (sig_count == sig && sig_nan == 0)
                                        ? (sig_sum * inv_sig) : NEO_F64_NAN;
                o[i] = (isnan(macz_val) || isnan(signal))
                         ? NEO_F64_NAN : (macz_val - signal);
            }
        }
    }
}
