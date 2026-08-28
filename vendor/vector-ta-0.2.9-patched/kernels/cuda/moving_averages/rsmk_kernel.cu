#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }


extern "C" __global__ void rsmk_momentum_f32(
    const float* __restrict__ main_in,
    const float* __restrict__ compare_in,
    int lookback,
    int first_valid,
    int len,
    float* __restrict__ mom_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || lookback <= 0) return;
    const int mom_fv = first_valid + lookback;
    for (int i = 0; i < min(mom_fv, len); ++i) { mom_out[i] = nanf; }
    if (mom_fv >= len) return;


    for (int i = mom_fv; i < len; ++i) {
        const float a_m = main_in[i];
        const float a_c = compare_in[i];
        const float b_m = main_in[i - lookback];
        const float b_c = compare_in[i - lookback];
        float outv = nanf;
        if (!isnan(a_m) && !isnan(a_c) && !isnan(b_m) && !isnan(b_c) && a_c != 0.0f && b_c != 0.0f) {
            const float lr_new = logf(a_m / a_c);
            const float lr_old = logf(b_m / b_c);
            outv = lr_new - lr_old;
        }
        mom_out[i] = outv;
    }
}


extern "C" __global__ void rsmk_apply_mom_single_row_ema_ema_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom,
    int period,
    int signal_period,
    float* __restrict__ out_indicator,
    float* __restrict__ out_signal
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0 || signal_period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    while (first < len && isnan(mom[first])) { first += 1; }
    if (first >= len) return;

    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double beta_ind = 1.0 - alpha_ind;
    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);
    const double beta_sig = 1.0 - alpha_sig;

    for (int i = 0; i < first; ++i) {
        out_indicator[i] = nanf;
        out_signal[i] = nanf;
    }

    double ind_mean = (double)mom[first] * 100.0;
    int ind_count = 1;
    out_indicator[first] = (float)ind_mean;

    const int ind_warm_end = min(len, first + period);
    for (int i = first + 1; i < ind_warm_end; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_count += 1;
            ind_mean = (((double)(ind_count - 1) * ind_mean) + src100) / (double)ind_count;
        }
        out_indicator[i] = (float)ind_mean;
    }

    double ind_val = ind_mean;
    for (int i = ind_warm_end; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_val = beta_ind * ind_val + alpha_ind * src100;
        }
        out_indicator[i] = (float)ind_val;
    }

    double sig_mean = (double)out_indicator[first];
    int sig_count = 1;
    out_signal[first] = (float)sig_mean;

    const int sig_warm_end = min(len, first + signal_period);
    for (int i = first + 1; i < sig_warm_end; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            sig_count += 1;
            sig_mean = (((double)(sig_count - 1) * sig_mean) + (double)iv) / (double)sig_count;
        }
        out_signal[i] = (float)sig_mean;
    }

    double sig_val = sig_mean;
    for (int i = sig_warm_end; i < len; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            sig_val = beta_sig * sig_val + alpha_sig * (double)iv;
        }
        out_signal[i] = (float)sig_val;
    }
}


extern "C" __global__ void rsmk_indicator_from_mom_ema_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom,
    int period,
    float* __restrict__ out_indicator
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    while (first < len && isnan(mom[first])) { first += 1; }
    if (first >= len) return;

    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double beta_ind = 1.0 - alpha_ind;

    for (int i = 0; i < first; ++i) {
        out_indicator[i] = nanf;
    }

    double ind_mean = (double)mom[first] * 100.0;
    int ind_count = 1;
    out_indicator[first] = (float)ind_mean;

    const int ind_warm_end = min(len, first + period);
    for (int i = first + 1; i < ind_warm_end; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_count += 1;
            ind_mean = (((double)(ind_count - 1) * ind_mean) + src100) / (double)ind_count;
        }
        out_indicator[i] = (float)ind_mean;
    }

    double ind_val = ind_mean;
    for (int i = ind_warm_end; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_val = beta_ind * ind_val + alpha_ind * src100;
        }
        out_indicator[i] = (float)ind_val;
    }
}


extern "C" __global__ void rsmk_signal_from_indicator_ema_f32(
    const float* __restrict__ indicator,
    int len,
    int first_valid_mom,
    int signal_period,
    float* __restrict__ out_signal
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || signal_period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    while (first < len && isnan(indicator[first])) { first += 1; }
    if (first >= len) return;

    for (int i = 0; i < first; ++i) {
        out_signal[i] = nanf;
    }

    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);
    const double beta_sig = 1.0 - alpha_sig;
    double sig_mean = (double)indicator[first];
    int sig_count = 1;
    out_signal[first] = (float)sig_mean;

    const int sig_warm_end = min(len, first + signal_period);
    for (int i = first + 1; i < sig_warm_end; ++i) {
        const float iv = indicator[i];
        if (!isnan(iv)) {
            sig_count += 1;
            sig_mean = (((double)(sig_count - 1) * sig_mean) + (double)iv) / (double)sig_count;
        }
        out_signal[i] = (float)sig_mean;
    }

    double sig_val = sig_mean;
    for (int i = sig_warm_end; i < len; ++i) {
        const float iv = indicator[i];
        if (!isnan(iv)) {
            sig_val = beta_sig * sig_val + alpha_sig * (double)iv;
        }
        out_signal[i] = (float)sig_val;
    }
}


extern "C" __global__ void rsmk_copy_group_indicator_tiled_f32(
    const float* __restrict__ group_indicator,
    const int* __restrict__ row_group_idx,
    int len,
    int n_rows,
    float* __restrict__ out_indicator
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= n_rows || t >= len) return;
    const int group = row_group_idx[row];
    out_indicator[row * len + t] = group_indicator[group * len + t];
}


extern "C" __global__ void rsmk_apply_mom_single_row_ema_ema_classic_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom,
    int period,
    int signal_period,
    float* __restrict__ out_indicator,
    float* __restrict__ out_signal
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0 || signal_period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    const int ind_warm = first + period - 1;
    const int sig_warm = ind_warm + signal_period - 1;
    if (ind_warm >= len) return;

    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double beta_ind = 1.0 - alpha_ind;
    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);
    const double beta_sig = 1.0 - alpha_sig;

    for (int i = 0; i < min(ind_warm, len); ++i) {
        out_indicator[i] = nanf;
    }
    for (int i = 0; i < min(sig_warm, len); ++i) {
        out_signal[i] = nanf;
    }

    double sum_ind = 0.0;
    int count_ind = 0;
    const int ind_seed_end = min(len, first + period);
    for (int i = first; i < ind_seed_end; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            sum_ind += (double)mv;
            count_ind += 1;
        }
    }
    if (count_ind == 0) return;

    double ema_ind = (sum_ind / (double)count_ind) * 100.0;
    out_indicator[ind_warm] = (float)ema_ind;

    for (int i = ind_warm + 1; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ema_ind = beta_ind * ema_ind + alpha_ind * src100;
        }
        out_indicator[i] = (float)ema_ind;
    }

    if (sig_warm >= len) return;

    double sum_sig = 0.0;
    int count_sig = 0;
    const int sig_seed_end = min(len, ind_warm + signal_period);
    for (int i = ind_warm; i < sig_seed_end; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            sum_sig += (double)iv;
            count_sig += 1;
        }
    }
    if (count_sig == 0) return;

    double ema_sig = sum_sig / (double)count_sig;
    out_signal[sig_warm] = (float)ema_sig;

    for (int i = sig_warm + 1; i < len; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            ema_sig = beta_sig * ema_sig + alpha_sig * (double)iv;
        }
        out_signal[i] = (float)ema_sig;
    }
}


extern "C" __global__ void rsmk_many_series_one_param_time_major_ema_ema_f32(
    const float* __restrict__ main_tm,
    const float* __restrict__ compare_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int lookback,
    int period,
    int signal_period,
    float* __restrict__ out_indicator_tm,
    float* __restrict__ out_signal_tm
) {
    const int s = blockIdx.y;
    if (s >= cols) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int stride = cols;
    const int fv = first_valids[s];
    const float nanf = qnan32();
    if (rows <= 0 || lookback <= 0 || period <= 0 || signal_period <= 0) return;


    const int mom_fv = fv + lookback;
    const int ind_warm = mom_fv + period - 1;
    const int sig_warm = ind_warm + signal_period - 1;
    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);


    for (int t = 0; t < min(ind_warm, rows); ++t) {
        out_indicator_tm[t * stride + s] = nanf;
    }
    for (int t = 0; t < min(sig_warm, rows); ++t) {
        out_signal_tm[t * stride + s] = nanf;
    }
    if (ind_warm >= rows) return;


    double sum = 0.0; int cnt = 0;
    const int init_end = min(rows, mom_fv + period);
    for (int t = mom_fv; t < init_end; ++t) {
        const int i_new = t * stride + s;
        const int i_old = (t - lookback) * stride + s;
        const float m_new = main_tm[i_new];
        const float c_new = compare_tm[i_new];
        const float m_old = main_tm[i_old];
        const float c_old = compare_tm[i_old];
        float mv = nanf;
        if (!isnan(m_new) && !isnan(c_new) && !isnan(m_old) && !isnan(c_old) && c_new != 0.0f && c_old != 0.0f) {
            const float lr_new = logf(m_new / c_new);
            const float lr_old = logf(m_old / c_old);
            mv = lr_new - lr_old;
        }
        if (!isnan(mv)) { sum += (double)mv; cnt += 1; }
    }

    if (cnt == 0) {
        for (int t = ind_warm; t < rows; ++t) { out_indicator_tm[t * stride + s] = nanf; }
        for (int t = sig_warm; t < rows; ++t) { out_signal_tm[t * stride + s] = nanf; }
        return;
    }

    double ema_ind = (sum / (double)cnt) * 100.0;
    out_indicator_tm[ind_warm * stride + s] = (float)ema_ind;


    double ema_sig = 0.0; bool sig_seeded = false;
    double acc_sig = ema_ind; int cnt_sig = 1;
    if (sig_warm == ind_warm) {
        ema_sig = (acc_sig / (double)cnt_sig);
        out_signal_tm[sig_warm * stride + s] = (float)ema_sig;
        sig_seeded = true;
    }

    for (int t = ind_warm + 1; t < rows; ++t) {
        const int i_new = t * stride + s;
        const int i_old = (t - lookback) * stride + s;
        const float m_new = main_tm[i_new];
        const float c_new = compare_tm[i_new];
        const float m_old = main_tm[i_old];
        const float c_old = compare_tm[i_old];
        float mv = nanf;
        if (!isnan(m_new) && !isnan(c_new) && !isnan(m_old) && !isnan(c_old) && c_new != 0.0f && c_old != 0.0f) {
            const float lr_new = logf(m_new / c_new);
            const float lr_old = logf(m_old / c_old);
            mv = lr_new - lr_old;
        }

        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ema_ind = ((src100 - ema_ind) * alpha_ind) + ema_ind;
        }
        out_indicator_tm[i_new] = (float)ema_ind;

        if (!sig_seeded) {
            if (t < sig_warm) { acc_sig += ema_ind; cnt_sig += 1; }
            else if (t == sig_warm) {
                ema_sig = (acc_sig / (double)cnt_sig);
                out_signal_tm[i_new] = (float)ema_sig; sig_seeded = true; continue;
            }
        } else {
            ema_sig = ((ema_ind - ema_sig) * alpha_sig) + ema_sig;
            out_signal_tm[i_new] = (float)ema_sig;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3   (rsmk)
 *
 * CPU reference: `rsmk_scalar` (src/indicators/rsmk.rs:391), the `_ =>` arm of
 *   `rsmk_with_kernel` (:380). The two AVX arms above it are gated on
 *   `nightly-avx`; the scalar arm is the default build's path and, per the
 *   brief's rule for a crate we fork, the single oracle.
 *
 * Column: output_id "value" -> `out.indicator` (cpu_batch.rs:16503-16505).
 *   The `signal` column is a SECOND ema over the same series; it is not this
 *   column, so its accumulator is deliberately absent below rather than
 *   computed and thrown away.
 *
 * PERIOD-SWEPT -- and it is the only one of this closer's ten that is.
 *   `compute_rsmk_batch` reads a parameter literally named `period`
 *   (cpu_batch.rs:16479, default 3) alongside `lookback` (90) and
 *   `signal_period` (20). So `periods[combo]` IS read here, and the rows of a
 *   sweep genuinely differ.
 *
 * Input: (main, compare) -- `compute_rsmk_batch` binds them from
 *   `IndicatorDataRef::CloseVolume { close, volume }` as `(close, volume)`
 *   (cpu_batch.rs:16445-16447) -> F64InputKind::CloseVolume. That pairing is
 *   surprising for a relative-strength indicator and it is DELIBERATE here:
 *   the kernel computes what the CPU computes, not what the name suggests.
 *
 * FIRST-VALID IGNORED. The CPU's index is the first non-NaN of the LOG-RATIO
 *   series, not of either input: `lr[i]` is NaN when main is NaN, OR compare
 *   is NaN, OR `compare == 0.0` (:322-326), and `first_valid` is
 *   `lr.iter().position(|x| !x.is_nan())` (:333). No rule in
 *   `F64FirstValidRule` expresses "and the divisor is non-zero", and a zero
 *   compare is a bar `AllInputsNonNan` would accept and then divide by. So
 *   the kernel derives its own index, as `garman_klass_volatility` already
 *   does for the same class of reason.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `ema_ind` at bar i is a
 *   function of its own value at bar i-1; a parallel scan would re-associate
 *   the recursion and this column feeds a threshold comparison.
 *
 * Roundings, counted against the CPU lines:
 *   :325   (m / c).ln()                          -- ONE divide, ONE log
 *   :475   let src100 = mv * 100.0               -- ONE multiply
 *   :477   (src100 - ema_ind).mul_add(alpha_ind, ema_ind)   -- ONE fma
 *   :453   let mut ema_ind = (sum / cnt as f64) * 100.0     -- divide then mul
 *   The seed is NOT an fma and the step IS one. Writing the step as
 *   `ema*(1-a) + src100*a` would be THREE roundings where the reference has
 *   ONE, which is the exact defect the brief names in `natr`.
 *
 * `log`, never `logf`: the f32 entry points above use `logf`/`__logf`; this
 *   column is f64 end to end and uses the double-precision `log`.
 *
 * NaN semantics: the CPU has no max/min on this column. `mom[i]` is NaN when
 *   either leg is NaN (:361-365) and the ema simply HOLDS its previous value
 *   through a NaN bar (:474-479 -- the update is inside `if !v.is_nan()` but
 *   the store is outside it). Reproduced exactly: a NaN bar must not reset
 *   and must not poison.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:16478-16482. `period` is NOT here -- it is swept. */
#define NEO_RSMK_LOOKBACK      90
#define NEO_RSMK_SIGNAL_PERIOD 20

/* `lr[i]` -- rsmk.rs:322-326. */
__device__ __forceinline__ double neo_rsmk_lr(const double* __restrict__ main_s,
                                              const double* __restrict__ cmp_s,
                                              int i)
{
    const double m = main_s[i];
    const double c = cmp_s[i];
    if (isnan(m) || isnan(c) || c == 0.0) return NEO_F64_NAN;
    return log(m / c);
}

extern "C" __global__
void rsmk_neo_batch_f64(const double* __restrict__ main_s,
                        const double* __restrict__ cmp_s,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)first_valid; /* log-ratio scan, derived below -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period        = periods[combo];
    const int lookback      = NEO_RSMK_LOOKBACK;
    const int signal_period = NEO_RSMK_SIGNAL_PERIOD;

    /* rsmk.rs:305-315 -- InvalidPeriod. */
    if (lookback == 0 || period <= 0 || signal_period == 0) return;
    if (period > n || signal_period > n || lookback >= n) return;

    /* :331-334 -- first non-NaN of the LOG-RATIO series. */
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(neo_rsmk_lr(main_s, cmp_s, i))) { first = i; break; }
    }
    if (first < 0) return;                       /* AllValuesNaN */

    const int max_ps = (period > signal_period) ? period : signal_period;
    const int needed = lookback + max_ps;        /* :337 */
    if (n - first < needed) return;              /* :338 NotEnoughValidData */

    const int mom_fv     = first + lookback;             /* :393 */
    const int ind_warmup = mom_fv + (period - 1);        /* :425 */
    if (ind_warmup >= n) return;                         /* :430 -- row stays NaN */

    /* Seed: the mean of the first `period` non-NaN momentum values, then x100
     * (:434-455). `cnt` counts only the non-NaN ones, so a hole shortens the
     * seed rather than poisoning it -- which is why this is a count, not a
     * fixed divisor. */
    double sum = 0.0;
    int    cnt = 0;
    int    init_end = mom_fv + period;
    if (init_end > n) init_end = n;
    for (int i = mom_fv; i < init_end; ++i) {
        const double a = neo_rsmk_lr(main_s, cmp_s, i);
        const double b = neo_rsmk_lr(main_s, cmp_s, i - lookback);
        const double v = (isnan(a) || isnan(b)) ? NEO_F64_NAN : (a - b);
        if (!isnan(v)) { sum += v; cnt += 1; }
    }
    if (cnt == 0) return;                        /* :495-500 -- the row is NaN */

    const double alpha_ind = 2.0 / ((double)period + 1.0);
    double ema_ind = (sum / (double)cnt) * 100.0;
    o[ind_warmup] = ema_ind;

    for (int i = ind_warmup + 1; i < n; ++i) {
        const double a  = neo_rsmk_lr(main_s, cmp_s, i);
        const double b  = neo_rsmk_lr(main_s, cmp_s, i - lookback);
        const double mv = (isnan(a) || isnan(b)) ? NEO_F64_NAN : (a - b);
        if (!isnan(mv)) {
            const double src100 = mv * 100.0;
            /* :477 -- ONE fma, matching `(src100 - ema_ind).mul_add(..)`. */
            ema_ind = fma(src100 - ema_ind, alpha_ind, ema_ind);
        }
        o[i] = ema_ind;   /* the store is OUTSIDE the NaN guard -- :480 */
    }
}
