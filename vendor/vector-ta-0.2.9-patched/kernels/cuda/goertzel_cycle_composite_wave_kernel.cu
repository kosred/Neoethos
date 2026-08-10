// goertzel_cycle_composite_wave — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line:  extern "C" __global__ void goertzel_cycle_composite_wave_batch_f64() {}
// plus a wrapper that resolved that empty symbol, computed on the host, and
// uploaded the host answer as if the card had produced it.
//
// CPU REFERENCE
// -------------
//   src/indicators/goertzel_cycle_composite_wave.rs
//     :334 sample_size_for_params      :460 hp_lambda
//     :464 zero_lag_ma                 :499 hodrick_prescott_filter
//     :568 detrend_ln_zero_lag_regression
//     :604 apply_detrend_mode          :640 bartels_prob
//     :683 apply_bartels               :713 extract_cycles
//     :796 current_wave_from_cycles    :827 compute_window_wave
//     :886 compute_row                 <- the per-row loop
//
// WHY THIS IS NOT "UNKERNELABLE"
// ------------------------------
// The Goertzel algorithm is a SECOND-ORDER RECURRENCE in the frequency domain:
// `w = coeff*x - y + detrended[i]` walking the window backwards (:764). That is
// the same shape as an EMA, and the same answer applies — ONE THREAD PER
// PARAMETER ROW walking in the reference's order. It is perfectly good CUDA;
// not every kernel has to be bar-parallel. The Hodrick-Prescott stage is a
// tridiagonal solve (:499), also serial, also fine in a thread.
//
// The heavy part is that EVERY BAR re-runs the whole window: at the defaults
// that is a 601-sample detrend plus 119 Goertzel passes of 240 steps each. The
// card wins here by running thousands of parameter rows at once, not by
// splitting one row.
//
// SCRATCH
// -------
// A row needs ~4,900 doubles. Handing every row its own slab would need 154 GB
// for a 4,096-row sweep, so the launch is planned in SLOTS: the host asks the
// card how much is free and each thread loops `row = slot; row < rows;
// row += slots`. Peak memory is a function of the hardware, never of the sweep
// width — a wider sweep runs in more passes, it does not run out of memory.
//
// ARITHMETIC
// ----------
// f64 throughout; listed in `F64_LANE_SOURCES` so never compiled with
// `--use_fast_math`. Every epsilon here is the CPU's own: `DBL_EPSILON` where
// the reference writes `f64::EPSILON` (:530, :592, :674), and the literal
// `1e-7` guard at :753 which is a magic constant in the f64 reference itself,
// not an f32 tolerance being carried across.
//
// SORT ORDER
// ----------
// `extract_cycles` sorts by amplitude DESCENDING with
// `b.partial_cmp(&a).unwrap_or(Equal)` and Rust's `sort_by`, which is STABLE.
// Reproduced with insertion sort, which is also stable, so ties keep their
// discovery order — and the order decides which cycles `use_top_cycles`
// selects, so it is part of the answer.

#include <cmath>
#include <cfloat>
#include <cstdint>

#define GZ_MODE_NONE        0
#define GZ_MODE_HP_SMOOTH   1
#define GZ_MODE_ZL_SMOOTH   2
#define GZ_MODE_HP_DETREND  3
#define GZ_MODE_ZL_DETREND  4
#define GZ_MODE_LOG_ZL_REG  5

__device__ __forceinline__ double gz_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// hp_lambda (:460)
__device__ __forceinline__ double gz_hp_lambda(int period) {
    double s = sin(M_PI / static_cast<double>(period));
    double s2 = s * s;
    return 0.0625 / (s2 * s2);
}

// zero_lag_ma (:464). `lwma1` and `out` must not alias.
__device__ void gz_zero_lag_ma(
    const double* src, int n, int smooth_per, double* lwma1, double* out) {
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        double sumw = 0.0;
        for (int k = 0; k < smooth_per; ++k) {
            int idx = i + k;
            if (idx < n) {
                double weight = static_cast<double>(smooth_per - k);
                sumw += weight;
                sum += weight * src[idx];
            }
        }
        lwma1[i] = sumw != 0.0 ? sum / sumw : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        double sumw = 0.0;
        for (int k = 0; k < smooth_per; ++k) {
            if (i >= k) {
                double weight = static_cast<double>(smooth_per - k);
                sumw += weight;
                sum += weight * lwma1[i - k];
            }
        }
        out[i] = sumw != 0.0 ? sum / sumw : 0.0;
    }
}

// hodrick_prescott_filter (:499). `a`, `b`, `c` are scratch; `out` receives the
// result and is seeded with `src` exactly as the CPU's `src.to_vec()` does.
__device__ void gz_hp_filter(
    const double* src, int per, double lambda, double* a, double* b, double* c, double* out) {
    for (int i = 0; i < per; ++i) {
        out[i] = src[i];
        a[i] = 0.0;
        b[i] = 0.0;
        c[i] = 0.0;
    }
    if (per == 0) {
        return;
    }

    a[0] = 1.0 + lambda;
    b[0] = -2.0 * lambda;
    c[0] = lambda;
    // `1..per.saturating_sub(2)` — empty when per < 3.
    int upper = per >= 2 ? per - 2 : 0;
    for (int i = 1; i < upper; ++i) {
        a[i] = 6.0 * lambda + 1.0;
        b[i] = -4.0 * lambda;
        c[i] = lambda;
    }
    // The CPU applies these four in exactly this order, and for per == 2 they
    // overlap (a[1], a[0], a[1], b[0]); order therefore matters.
    if (per > 1) {
        a[1] = 5.0 * lambda + 1.0;
        a[per - 2] = 5.0 * lambda + 1.0;
        a[per - 1] = 1.0 + lambda;
        b[per - 2] = -2.0 * lambda;
    }

    double h1 = 0.0, h2 = 0.0, h3 = 0.0, h4 = 0.0, h5 = 0.0;
    double hh1 = 0.0, hh2 = 0.0, hh3 = 0.0, hh5 = 0.0;

    for (int i = 0; i < per; ++i) {
        double z = a[i] - h4 * h1 - hh5 * h2;
        if (fabs(z) <= DBL_EPSILON) {
            break;
        }
        double hb = b[i];
        hh1 = h1;
        h1 = (hb - h4 * h2) / z;
        b[i] = h1;
        double hc = c[i];
        hh2 = h2;
        h2 = hc / z;
        c[i] = h2;
        a[i] = (src[i] - hh3 * hh5 - h3 * h4) / z;
        hh3 = h3;
        h3 = a[i];
        h4 = hb - h5 * hh1;
        hh5 = h5;
        h5 = hc;
    }
    // `hh2` is assigned but never read by the CPU either; keeping the write
    // makes the transliteration line-for-line checkable.
    (void)hh2;

    double h1b = a[per - 1];
    double h2b = 0.0;
    out[per - 1] = h1b;
    for (int i = per - 2; i >= 0; --i) {
        out[i] = a[i] - b[i] * h1b - c[i] * h2b;
        h2b = h1b;
        h1b = out[i];
    }
}

// detrend_ln_zero_lag_regression (:568). Returns 0 for the CPU's `None`.
__device__ int gz_detrend_ln_zl_regression(
    const double* src, int n, int smooth_per, double* lwma1, double* calc, double* out) {
    gz_zero_lag_ma(src, n, smooth_per, lwma1, calc);
    for (int i = 0; i < n; ++i) {
        double value = calc[i];
        if (value <= 0.0 || !isfinite(value)) {
            return 0;
        }
        calc[i] = log(value) * 100.0;
    }

    double sumy = 0.0, sumx = 0.0, sumxy = 0.0, sumx2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double value = calc[i];
        sumy += value;
        sumx += x;
        sumxy += x * value;
        sumx2 += x * x;
    }

    double bars = static_cast<double>(n);
    double denom = sumx2 * bars - sumx * sumx;
    if (fabs(denom) <= DBL_EPSILON) {
        return 0;
    }
    double slope = (sumxy * bars - sumx * sumy) / denom;
    double intercept = (sumy - sumx * slope) / bars;
    for (int i = 0; i < n; ++i) {
        out[i] = calc[i] - (intercept + slope * static_cast<double>(i));
    }
    return 1;
}

// bartels_prob (:640). `vsin`/`vcos` are scratch of at least `n` doubles.
__device__ double gz_bartels_prob(
    int n, int cycle_count, const double* values, int values_len, double* vsin, double* vcos) {
    if (n == 0 || cycle_count == 0 || values_len < n * cycle_count) {
        return 1.0;
    }

    double avg_coeff_a = 0.0;
    double avg_coeff_b = 0.0;
    double avg_ind_amplit = 0.0;

    for (int i = 0; i < n; ++i) {
        double theta = static_cast<double>(i + 1) / static_cast<double>(n) * 2.0 * M_PI;
        vsin[i] = sin(theta);
        vcos[i] = cos(theta);
    }

    for (int t = 0; t < cycle_count; ++t) {
        double coeff_a = 0.0;
        double coeff_b = 0.0;
        int base = t * n;
        for (int i = 0; i < n; ++i) {
            double value = values[base + i];
            coeff_a += vsin[i] * value;
            coeff_b += vcos[i] * value;
        }
        avg_coeff_a += coeff_a;
        avg_coeff_b += coeff_b;
        avg_ind_amplit += coeff_a * coeff_a + coeff_b * coeff_b;
    }

    double count = static_cast<double>(cycle_count);
    avg_coeff_a /= count;
    avg_coeff_b /= count;
    double avg_ampl = sqrt(avg_coeff_a * avg_coeff_a + avg_coeff_b * avg_coeff_b);
    double ind = sqrt(avg_ind_amplit / count);
    double expected_ampl = ind / sqrt(count);
    if (expected_ampl <= DBL_EPSILON) {
        return 1.0;
    }
    double a_ratio = avg_ampl / expected_ampl;
    return exp(-a_ratio * a_ratio);
}

// Stable descending insertion sort on (amplitude, cycle, phase, bartels),
// matching Rust's stable `sort_by(|a, b| b.key.partial_cmp(&a.key)
// .unwrap_or(Equal))`.
__device__ void gz_sort_desc(
    double* amp, double* phase, double* bart, int* cyc, int n, int by_bartels) {
    for (int i = 1; i < n; ++i) {
        double k_amp = amp[i], k_phase = phase[i], k_bart = bart[i];
        int k_cyc = cyc[i];
        double k_key = by_bartels ? k_bart : k_amp;
        int j = i - 1;
        while (j >= 0) {
            double j_key = by_bartels ? bart[j] : amp[j];
            // `b.partial_cmp(&a)` is Greater — i.e. `i` sorts BEFORE `j` — only
            // when `k_key > j_key`. A NaN on either side yields Equal, which
            // stops the shift and preserves the original order, exactly as the
            // CPU's `unwrap_or(Equal)` does.
            if (!(k_key > j_key)) {
                break;
            }
            amp[j + 1] = amp[j];
            phase[j + 1] = phase[j];
            bart[j + 1] = bart[j];
            cyc[j + 1] = cyc[j];
            j -= 1;
        }
        amp[j + 1] = k_amp;
        phase[j + 1] = k_phase;
        bart[j + 1] = k_bart;
        cyc[j + 1] = k_cyc;
    }
}

struct GzParams {
    int max_period;
    int start_at_cycle;
    int use_top_cycles;
    int bar_to_calculate;
    int detrend_mode;
    int dt_zl_per1;
    int dt_zl_per2;
    int dt_hp_per1;
    int dt_hp_per2;
    int dt_reg_zl_smooth_per;
    int hp_smooth_per;
    int zlma_smooth_per;
    int filter_bartels;
    int bart_no_cycles;
    int bart_smooth_per;
    int bart_sig_limit;
    int sort_bartels;
    int squared_amp;
    int use_cosine;
    int subtract_noise;
    int use_cycle_strength;
};

// apply_detrend_mode (:604). Returns 0 for the CPU's `None`.
__device__ int gz_apply_detrend(
    const double* src_rev, int n, const GzParams& p,
    double* a, double* b, double* c, double* tmp, double* processed) {
    switch (p.detrend_mode) {
        case GZ_MODE_NONE: {
            for (int i = 0; i < n; ++i) {
                processed[i] = src_rev[i];
            }
            break;
        }
        case GZ_MODE_HP_SMOOTH: {
            gz_hp_filter(src_rev, n, gz_hp_lambda(p.hp_smooth_per), a, b, c, processed);
            break;
        }
        case GZ_MODE_ZL_SMOOTH: {
            gz_zero_lag_ma(src_rev, n, p.zlma_smooth_per, a, processed);
            break;
        }
        case GZ_MODE_HP_DETREND: {
            gz_hp_filter(src_rev, n, gz_hp_lambda(p.dt_hp_per1), a, b, c, tmp);
            gz_hp_filter(src_rev, n, gz_hp_lambda(p.dt_hp_per2), a, b, c, processed);
            for (int i = 0; i < n; ++i) {
                processed[i] = tmp[i] - processed[i];
            }
            break;
        }
        case GZ_MODE_ZL_DETREND: {
            gz_zero_lag_ma(src_rev, n, p.dt_zl_per1, a, tmp);
            gz_zero_lag_ma(src_rev, n, p.dt_zl_per2, a, processed);
            for (int i = 0; i < n; ++i) {
                processed[i] = tmp[i] - processed[i];
            }
            break;
        }
        case GZ_MODE_LOG_ZL_REG: {
            if (!gz_detrend_ln_zl_regression(src_rev, n, p.dt_reg_zl_smooth_per, a, tmp,
                                             processed)) {
                return 0;
            }
            break;
        }
        default:
            return 0;
    }
    for (int i = 0; i < n; ++i) {
        if (!isfinite(processed[i])) {
            return 0;
        }
    }
    return 1;
}

// current_wave_from_cycles (:796)
__device__ double gz_current_wave(
    const double* amp, const double* phase, int count, const GzParams& p) {
    if (count == 0) {
        return 0.0;
    }
    int start = p.start_at_cycle > 0 ? p.start_at_cycle - 1 : 0;
    if (start >= count) {
        return 0.0;
    }
    int end = start + p.use_top_cycles;
    if (end > count) {
        end = count;
    }
    double out = 0.0;
    for (int i = start; i < end; ++i) {
        out += p.use_cosine ? amp[i] * cos(phase[i]) : amp[i] * sin(phase[i]);
    }
    if (p.subtract_noise && end < count) {
        double noise = 0.0;
        for (int i = end; i < count; ++i) {
            noise += p.use_cosine ? amp[i] * cos(phase[i]) : amp[i] * sin(phase[i]);
        }
        out -= noise;
    }
    return out;
}

extern "C" __global__ void goertzel_cycle_composite_wave_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ max_periods,
    const int* __restrict__ start_at_cycles,
    const int* __restrict__ use_top_cycles,
    // base_params, shared by every row of the sweep (only the three axes above
    // are swept -- see `expand_grid_checked`, :1090).
    int bar_to_calculate,
    int detrend_mode,
    int dt_zl_per1,
    int dt_zl_per2,
    int dt_hp_per1,
    int dt_hp_per2,
    int dt_reg_zl_smooth_per,
    int hp_smooth_per,
    int zlma_smooth_per,
    int filter_bartels,
    int bart_no_cycles,
    int bart_smooth_per,
    int bart_sig_limit,
    int sort_bartels,
    int squared_amp,
    int use_cosine,
    int subtract_noise,
    int use_cycle_strength,
    int rows,
    int slots,
    int sample_cap,   // widest window any row asks for
    int work_cap,     // 2*max_period + 1, widest amp/phase/mark/detrended array
    int cycle_cap,    // max_period + 2, widest cycle list
    double* scratch,
    int* iscratch,
    double* __restrict__ out
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = gz_qnan();

    size_t doubles_per_slot =
        static_cast<size_t>(6 * sample_cap) + static_cast<size_t>(4 * work_cap) +
        static_cast<size_t>(3 * cycle_cap);
    double* base = scratch + static_cast<size_t>(slot) * doubles_per_slot;
    double* src_rev = base;
    double* wa = src_rev + sample_cap;
    double* wb = wa + sample_cap;
    double* wc = wb + sample_cap;
    double* wtmp = wc + sample_cap;
    double* processed = wtmp + sample_cap;
    double* amp_work = processed + sample_cap;
    double* phase_work = amp_work + work_cap;
    double* mark_work = phase_work + work_cap;
    double* detrended = mark_work + work_cap;
    double* cyc_amp = detrended + work_cap;
    double* cyc_phase = cyc_amp + cycle_cap;
    double* cyc_bart = cyc_phase + cycle_cap;
    int* cyc_num = iscratch + static_cast<size_t>(slot) * static_cast<size_t>(cycle_cap);

    GzParams p;
    p.bar_to_calculate = bar_to_calculate;
    p.detrend_mode = detrend_mode;
    p.dt_zl_per1 = dt_zl_per1;
    p.dt_zl_per2 = dt_zl_per2;
    p.dt_hp_per1 = dt_hp_per1;
    p.dt_hp_per2 = dt_hp_per2;
    p.dt_reg_zl_smooth_per = dt_reg_zl_smooth_per;
    p.hp_smooth_per = hp_smooth_per;
    p.zlma_smooth_per = zlma_smooth_per;
    p.filter_bartels = filter_bartels;
    p.bart_no_cycles = bart_no_cycles;
    p.bart_smooth_per = bart_smooth_per;
    p.bart_sig_limit = bart_sig_limit;
    p.sort_bartels = sort_bartels;
    p.squared_amp = squared_amp;
    p.use_cosine = use_cosine;
    p.subtract_noise = subtract_noise;
    p.use_cycle_strength = use_cycle_strength;

    for (int row = slot; row < rows; row += slots) {
        p.max_period = max_periods[row];
        p.start_at_cycle = start_at_cycles[row];
        p.use_top_cycles = use_top_cycles[row];

        // sample_size_for_params (:334)
        int per = p.max_period;
        int cycle_span = 2 * per;
        int bart_span = p.bart_no_cycles * per;
        if (bart_span > cycle_span) {
            cycle_span = bart_span;
        }
        int sample_size = cycle_span + p.bar_to_calculate;

        double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);
        for (int i = 0; i < len; ++i) {
            row_out[i] = nan_value;
        }
        if (sample_size <= 0 || sample_size > len || sample_size > sample_cap) {
            continue;
        }

        int sample = 2 * per;
        int work_len = sample + 1;

        for (int end = sample_size - 1; end < len; ++end) {
            int window_start = end + 1 - sample_size;
            bool ok = true;
            for (int i = window_start; i <= end; ++i) {
                if (!isfinite(data[i])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
            // `src_rev` is the window REVERSED (:898): src_rev[0] is the newest bar.
            for (int i = 0; i < sample_size; ++i) {
                src_rev[i] = data[end - i];
            }

            if (!gz_apply_detrend(src_rev, sample_size, p, wa, wb, wc, wtmp, processed)) {
                continue;
            }

            // ---- extract_cycles (:713) --------------------------------------
            int cycle_count = 0;
            if (sample_size >= p.bar_to_calculate + sample && sample >= 2) {
                for (int i = 0; i < work_len; ++i) {
                    amp_work[i] = 0.0;
                    phase_work[i] = 0.0;
                    mark_work[i] = 0.0;
                    detrended[i] = 0.0;
                }

                double temp1 = processed[p.bar_to_calculate + sample - 1];
                double trend_slope =
                    (processed[p.bar_to_calculate] - temp1) / (static_cast<double>(sample) - 1.0);
                for (int k = sample - 1; k >= 1; --k) {
                    detrended[k] = processed[p.bar_to_calculate + k - 1] -
                                   (temp1 + trend_slope * static_cast<double>(sample - k));
                }

                for (int k = 2; k <= per; ++k) {
                    double z = 1.0 / static_cast<double>(k);
                    double coeff = 2.0 * cos(2.0 * M_PI * z);
                    double w = 0.0, x = 0.0, y = 0.0;
                    for (int i = sample; i >= 1; --i) {
                        w = coeff * x - y + detrended[i];
                        y = x;
                        x = w;
                    }
                    double real = x - y * coeff / 2.0;
                    if (fabs(real) <= DBL_EPSILON) {
                        // The CPU's own magic constant at :753 -- not an f32
                        // tolerance carried across, a guard in the f64 source.
                        real = 1e-7;
                    }
                    double imag = y * sin(2.0 * M_PI * z);
                    double amplitude = p.squared_amp ? real * real + imag * imag
                                                     : sqrt(real * real + imag * imag);
                    amp_work[k] = p.use_cycle_strength
                                      ? amplitude / static_cast<double>(k)
                                      : amplitude;
                    double phase = atan(imag / real);
                    if (real < 0.0) {
                        phase += M_PI;
                    } else if (imag < 0.0) {
                        phase += 2.0 * M_PI;
                    }
                    phase_work[k] = phase;
                }

                for (int k = 3; k < per; ++k) {
                    if (amp_work[k] > amp_work[k - 1] && amp_work[k] > amp_work[k + 1]) {
                        mark_work[k] = static_cast<double>(k) * 1e-4;
                    }
                }

                for (int i = 0; i <= per + 1; ++i) {
                    if (i < work_len && mark_work[i] > 0.0 && cycle_count < cycle_cap) {
                        cyc_num[cycle_count] =
                            static_cast<int>(round(10000.0 * mark_work[i]));
                        cyc_amp[cycle_count] = amp_work[i];
                        cyc_phase[cycle_count] = phase_work[i];
                        cyc_bart[cycle_count] = 0.0;
                        cycle_count += 1;
                    }
                }

                gz_sort_desc(cyc_amp, cyc_phase, cyc_bart, cyc_num, cycle_count, 0);

                if (p.filter_bartels) {
                    // apply_bartels (:683)
                    double sig_limit = static_cast<double>(p.bart_sig_limit);
                    for (int i = 0; i < cycle_count; ++i) {
                        int bars_taken = cyc_num[i] * p.bart_no_cycles;
                        if (bars_taken <= 0 || bars_taken > sample_size) {
                            cyc_bart[i] = 0.0;
                            continue;
                        }
                        // `wa` holds the zero-lag intermediate, `wb` the log
                        // series, `wtmp` the regression output; all three are
                        // free once `processed` is built and none aliases it.
                        // `wc` is then free for bartels_prob's sin/cos tables,
                        // which need `cyc_num[i] <= max_period` entries each and
                        // so fit in one `sample_cap`-wide buffer split in two —
                        // `sample_cap >= 2 * max_period` by construction.
                        if (gz_detrend_ln_zl_regression(processed, bars_taken, p.bart_smooth_per,
                                                        wa, wb, wtmp)) {
                            double prob = gz_bartels_prob(cyc_num[i], p.bart_no_cycles, wtmp,
                                                          bars_taken, wc, wc + per);
                            cyc_bart[i] = (1.0 - prob) * 100.0;
                        } else {
                            cyc_bart[i] = 0.0;
                        }
                    }
                    // retain(|c| c.bartels > sig_limit) -- order preserving.
                    int kept = 0;
                    for (int i = 0; i < cycle_count; ++i) {
                        if (cyc_bart[i] > sig_limit) {
                            cyc_amp[kept] = cyc_amp[i];
                            cyc_phase[kept] = cyc_phase[i];
                            cyc_bart[kept] = cyc_bart[i];
                            cyc_num[kept] = cyc_num[i];
                            kept += 1;
                        }
                    }
                    cycle_count = kept;
                    if (p.sort_bartels) {
                        gz_sort_desc(cyc_amp, cyc_phase, cyc_bart, cyc_num, cycle_count, 1);
                    }
                }
            }

            row_out[end] = gz_current_wave(cyc_amp, cyc_phase, cycle_count, p);
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `goertzel_cycle_composite_wave_with_kernel`
// (src/indicators/goertzel_cycle_composite_wave.rs:904) -> `compute_row`
// (:886) -> `compute_window_wave` (:827), with `apply_detrend_mode` (:597),
// `hodrick_prescott_filter` (:498), `hp_lambda` (:460), `extract_cycles`
// (:713) and `current_wave_from_cycles` (:791).
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `goertzel_cycle_composite_wave_batch_f64` (:396) is double-clean but declares
// thirty-one parameters -- three `const int*` per-row parameter arrays,
// eighteen scalar knobs and TWO host-allocated scratch pointers (`double*
// scratch`, `int* iscratch`). The f64 lane launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// and has no scratch to give, so the lane gets its own entry point here with
// every buffer a fixed-size PER-THREAD array.
//
// WHICH COLUMN: `value`. `compute_goertzel_cycle_composite_wave_batch`
// (cpu_batch.rs:7839) accepts `value` and `wave` and rejects everything else,
// and both name the same series.
//
// SHAPE -- AND WHY THE BRIEF'S "SERIAL RECURRENCE, ONE THREAD PER COLUMN" IS
// WRONG FOR THIS ONE. Goertzel IS a recurrence, but it runs INSIDE the window,
// not across bars: `compute_row` (:891-901) takes `data[end+1-sample ..= end]`
// for each `end` INDEPENDENTLY and calls `compute_window_wave` on it. Nothing
// is carried from bar to bar -- no accumulator, no state, no reset. So the
// correct shape is BAR-PARALLEL: one thread per (combo, bar), each thread
// running the window's own serial recurrences (`hodrick_prescott_filter`'s
// tridiagonal sweep and the Goertzel `w = coeff*x - y + detrended[i]` loop) by
// itself. Registered NOT sequential for that reason; the lane's bar-parallel
// launch arm accepts a single price series, which is what this indicator reads.
//   Choosing one-thread-per-column instead would put 200k independent windows,
// each ~50k flops, on FIVE threads. It would still be correct, and it would be
// about five orders of magnitude slower.
//
// PER-THREAD MEMORY, STATED IN BYTES because it is the interesting constraint:
// `sample_size` at the pinned defaults is `max(2*120, 5*120) + 1 = 601`
// (`sample_size_for_params`, :334-340). The Hodrick-Prescott solve needs three
// length-601 vectors, and the fast/slow difference needs the fast result kept
// while the slow one is computed, so four: 4 * 601 * 8 = 19,232 bytes. After
// `processed` exists, the three solve vectors are REUSED for `detrended`,
// `amp_work` and `phase_work` (241 doubles each -- `2*max_period+1`), so
// nothing is added. `mark_work` is not materialised at all: the CPU only reads
// it back in ascending k immediately after writing it (:766-780), so the peak
// test is folded into the cycle scan. `cyc_*` are not materialised either --
// see the selection note below.
//   That is a compile-time constant, not an allocation. It is also the reason
// this kernel does not simply hold the whole CPU intermediate set.
//
// PERIOD-INVARIANT: the CPU batch reads `max_period`, `start_at_cycle`,
// `use_top_cycles`, `bar_to_calculate`, `detrend_mode`, seven smoothing
// lengths, five Bartels knobs and four booleans -- and NEVER `period`
// (cpu_batch.rs:7851-7935), so every swept period gives the same CPU column and
// this kernel writes identical rows. Pinned at the CPU defaults: max_period
// 120, start_at_cycle 1, use_top_cycles 2, bar_to_calculate 1, detrend_mode
// `hodrick_prescott_detrending`, dt_hp_per1 20, dt_hp_per2 80,
// filter_bartels FALSE, sort_bartels FALSE, squared_amp TRUE, use_cosine TRUE,
// subtract_noise FALSE, use_cycle_strength TRUE (:30-43, cpu_batch.rs:7853-
// 7935).
//   `filter_bartels` being false is what keeps the Bartels probability path --
// the only part of this indicator that would need a second full detrend per
// cycle -- out of the kernel entirely, exactly as it is out of the CPU's path.
//
// WHY THE TOP-TWO SELECTION IS NOT A SORT. `extract_cycles` collects the peaks
// in ASCENDING k and then `cycles.sort_by(|a, b| b.amplitude.partial_cmp(...))`
// (:782-786). Rust's `sort_by` is STABLE, so equal amplitudes keep ascending k.
// `current_wave_from_cycles` then reads `cycles[start..end]` with start = 0 and
// end = 2 (:791-822). Taking the two largest amplitudes with STRICT `>`
// comparisons while scanning ascending k reproduces that prefix exactly -- a
// tie leaves the earlier k in front, which is what stability means -- and
// removes the need to materialise or sort the 122-slot cycle list. The sum is
// then formed in the CPU's own order, `0.0 + trig(first) + trig(second)`.
//
// ROUNDING: every expression is transcribed operand for operand. The CPU writes
// no `mul_add` anywhere in this indicator, so no `fma` is introduced -- the
// Goertzel step stays `coeff * x - y + detrended[i]` (:747), three roundings,
// and the detrend stays `src_rev[..] - (temp1 + trend_slope * (sample - k))`
// (:740). `hp_lambda` is `0.0625 / sin(PI/period).powi(4)`; `powi(4)` lowers to
// two squarings, written out as `s2 = s*s; s4 = s2*s2`, not `pow(s, 4.0)`.
//
// EPSILON: `f64::EPSILON` appears twice on this path -- the tridiagonal pivot
// guard `z.abs() <= f64::EPSILON` (:535) and the Goertzel `real.abs() <=
// f64::EPSILON` guard that substitutes 1e-7 (:753-755). BOTH are the CPU's own
// f64 constants and both are kept verbatim, including the 1e-7 substitute,
// which is a value the indicator's definition puts there and not a tolerance
// this port invented.
//
// NaN SEMANTICS: the window is rejected outright if ANY of its 601 bars is
// non-finite (:893) and again if any of the 601 detrended values is non-finite
// (:614-618), so nothing downstream can see a NaN. No comparison chain stands
// in for an `f64::max` here.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. `sin`, `cos`, `atan`, `sqrt`, `fabs`, `round` are the double
// overloads. The NaN is a DOUBLE quiet-NaN bit pattern.
//
// FIRST VALID IS NOT READ: `compute_row` (:887) fills the whole row with NaN
// and writes only the bars whose 601-bar window is entirely finite, which is a
// per-bar test and not a prefix. The lane row declares
// `F64FirstValidRule::Ignored`.
// ---------------------------------------------------------------------------

#define NEO_GZ_MAX_PERIOD 120
#define NEO_GZ_BAR_TO_CALCULATE 1
#define NEO_GZ_BART_NO_CYCLES 5
#define NEO_GZ_SAMPLE_SIZE 601   // max(2*120, 5*120) + 1
#define NEO_GZ_GSAMPLE 240       // 2 * max_period
#define NEO_GZ_WORK 241          // 2 * max_period + 1
#define NEO_GZ_DT_HP_PER1 20
#define NEO_GZ_DT_HP_PER2 80
#define NEO_GZ_USE_TOP_CYCLES 2
#define NEO_GZ_F64_EPSILON 2.2204460492503131e-16
#define NEO_GZ_PI 3.14159265358979323846

__device__ inline double neo_gz_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `hp_lambda`, :460. `powi(4)` is two squarings, not a `pow` call.
__device__ inline double neo_gz_hp_lambda(int period) {
    const double s = sin(NEO_GZ_PI / static_cast<double>(period));
    const double s2 = s * s;
    const double s4 = s2 * s2;
    return 0.0625 / s4;
}

// `hodrick_prescott_filter`, :498-565, over `src_rev[i] = data[end - i]`.
// The result is left in `a`; `b` and `c` are working vectors.
__device__ void neo_gz_hp_filter(
    const double* __restrict__ data,
    int end_bar,
    double lambda,
    double* __restrict__ a,
    double* __restrict__ b,
    double* __restrict__ c
) {
    const int per = NEO_GZ_SAMPLE_SIZE;

    for (int i = 0; i < per; ++i) {
        a[i] = 0.0;
        b[i] = 0.0;
        c[i] = 0.0;
    }

    a[0] = 1.0 + lambda;
    b[0] = -2.0 * lambda;
    c[0] = lambda;
    for (int i = 1; i < per - 2; ++i) {
        a[i] = 6.0 * lambda + 1.0;
        b[i] = -4.0 * lambda;
        c[i] = lambda;
    }
    if (per > 1) {
        a[1] = 5.0 * lambda + 1.0;
        a[per - 2] = 5.0 * lambda + 1.0;
        a[per - 1] = 1.0 + lambda;
        b[per - 2] = -2.0 * lambda;
    }

    double h1 = 0.0, h2 = 0.0, h3 = 0.0, h4 = 0.0, h5 = 0.0;
    double hh1 = 0.0, hh2 = 0.0, hh3 = 0.0, hh5 = 0.0;
    (void)hh1;
    (void)hh2;

    for (int i = 0; i < per; ++i) {
        const double z = a[i] - h4 * h1 - hh5 * h2;
        if (fabs(z) <= NEO_GZ_F64_EPSILON) {
            break;
        }
        const double hb = b[i];
        hh1 = h1;
        h1 = (hb - h4 * h2) / z;
        b[i] = h1;
        const double hc = c[i];
        hh2 = h2;
        h2 = hc / z;
        c[i] = h2;
        a[i] = (data[end_bar - i] - hh3 * hh5 - h3 * h4) / z;
        hh3 = h3;
        h3 = a[i];
        h4 = hb - h5 * hh1;
        hh5 = h5;
        h5 = hc;
    }

    // Back-substitution, :557-563. `output` is written into `a` in place: each
    // step reads a[i], b[i], c[i] and then overwrites a[i], so nothing needed
    // later is destroyed.
    double h1b = a[per - 1];
    double h2b = 0.0;
    a[per - 1] = h1b;
    for (int i = per - 2; i >= 0; --i) {
        const double value = a[i] - b[i] * h1b - c[i] * h2b;
        a[i] = value;
        h2b = h1b;
        h1b = value;
    }
}

extern "C" __global__ void goertzel_cycle_composite_wave_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }
    const int end = blockIdx.x * blockDim.x + threadIdx.x;
    if (end >= n) {
        return;
    }
    (void)periods;
    (void)first_valid;

    const double nan_value = neo_gz_neo_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    if (end < NEO_GZ_SAMPLE_SIZE - 1) {
        row[end] = nan_value;
        return;
    }
    // `compute_row`, :893 -- any non-finite bar in the window skips it and
    // leaves the NaN the row was filled with.
    for (int j = 0; j < NEO_GZ_SAMPLE_SIZE; ++j) {
        if (!isfinite(data[end - j])) {
            row[end] = nan_value;
            return;
        }
    }

    double a[NEO_GZ_SAMPLE_SIZE];
    double b[NEO_GZ_SAMPLE_SIZE];
    double c[NEO_GZ_SAMPLE_SIZE];
    double processed[NEO_GZ_SAMPLE_SIZE];

    // `apply_detrend_mode`, HodrickPrescottDetrending arm, :620-624.
    neo_gz_hp_filter(data, end, neo_gz_hp_lambda(NEO_GZ_DT_HP_PER1), a, b, c);
    for (int i = 0; i < NEO_GZ_SAMPLE_SIZE; ++i) {
        processed[i] = a[i];
    }
    neo_gz_hp_filter(data, end, neo_gz_hp_lambda(NEO_GZ_DT_HP_PER2), a, b, c);
    for (int i = 0; i < NEO_GZ_SAMPLE_SIZE; ++i) {
        processed[i] = processed[i] - a[i];
        if (!isfinite(processed[i])) {
            // `if out.iter().all(is_finite) { Some } else { None }`, :634-638;
            // `compute_row` then leaves the bar NaN.
            row[end] = nan_value;
            return;
        }
    }

    // The three solve vectors are free now -- reuse them.
    double* detrended = a;
    double* amp_work = b;
    double* phase_work = c;
    for (int i = 0; i < NEO_GZ_WORK; ++i) {
        detrended[i] = 0.0;
        amp_work[i] = 0.0;
        phase_work[i] = 0.0;
    }

    // `extract_cycles`, :713-789.
    const int per = NEO_GZ_MAX_PERIOD;
    const int for_bar = NEO_GZ_BAR_TO_CALCULATE;
    const int sample = NEO_GZ_GSAMPLE;

    const double temp1 = processed[for_bar + sample - 1];
    const double trend_slope =
        (processed[for_bar] - temp1) / (static_cast<double>(sample) - 1.0);
    for (int k = sample - 1; k >= 1; --k) {
        detrended[k] = processed[for_bar + k - 1] -
                       (temp1 + trend_slope * static_cast<double>(sample - k));
    }

    for (int k = 2; k <= per; ++k) {
        const double z = 1.0 / static_cast<double>(k);
        const double coeff = 2.0 * cos(2.0 * NEO_GZ_PI * z);
        double w = 0.0;
        double x = 0.0;
        double y = 0.0;
        for (int i = sample; i >= 1; --i) {
            w = coeff * x - y + detrended[i];
            y = x;
            x = w;
        }
        double real = x - y * coeff / 2.0;
        if (fabs(real) <= NEO_GZ_F64_EPSILON) {
            real = 1e-7;
        }
        const double imag = y * sin(2.0 * NEO_GZ_PI * z);
        // squared_amp = true, use_cycle_strength = true.
        const double amplitude = real * real + imag * imag;
        amp_work[k] = amplitude / static_cast<double>(k);
        double phase = atan(imag / real);
        if (real < 0.0) {
            phase += NEO_GZ_PI;
        } else if (imag < 0.0) {
            phase += 2.0 * NEO_GZ_PI;
        }
        phase_work[k] = phase;
    }

    // `mark_work` folded into the cycle scan: the CPU sets `mark[k] = k * 1e-4`
    // for a strict local maximum over k in 3..per (:766-770) and then walks
    // i in 0..=per+1 in ASCENDING order taking those (:773-780). The stable
    // descending sort by amplitude that follows means the first two encountered
    // maxima, compared with strict `>`, are exactly `cycles[0..2]`.
    int found = 0;
    double best_amp_1 = 0.0, best_phase_1 = 0.0;
    double best_amp_2 = 0.0, best_phase_2 = 0.0;
    for (int k = 3; k < per; ++k) {
        if (!(amp_work[k] > amp_work[k - 1] && amp_work[k] > amp_work[k + 1])) {
            continue;
        }
        const double amp = amp_work[k];
        const double phase = phase_work[k];
        if (found == 0) {
            best_amp_1 = amp;
            best_phase_1 = phase;
            found = 1;
        } else if (amp > best_amp_1) {
            best_amp_2 = best_amp_1;
            best_phase_2 = best_phase_1;
            best_amp_1 = amp;
            best_phase_1 = phase;
            if (found < 2) {
                found = 2;
            }
        } else if (found == 1 || amp > best_amp_2) {
            best_amp_2 = amp;
            best_phase_2 = phase;
            if (found < 2) {
                found = 2;
            }
        }
    }

    // `current_wave_from_cycles`, :791-822: start = 0, count = 2,
    // use_cosine = true, subtract_noise = false. Empty list gives 0.0.
    double value = 0.0;
    if (found >= 1) {
        value = value + best_amp_1 * cos(best_phase_1);
    }
    if (found >= 2) {
        value = value + best_amp_2 * cos(best_phase_2);
    }
    row[end] = value;
}
