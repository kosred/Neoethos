#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>


static __forceinline__ __device__ bool is_finite_f32(float x) {
    return !isnan(x) && !isinf(x);
}


static __forceinline__ __device__ void hpf_coeffs_from_period_f32(int period,
                                                                  float &c_out,
                                                                  float &oma_out,
                                                                  bool  &ok) {
    ok = false;
    if (period <= 0) return;
    float s, co;

    sincospif(2.0f / static_cast<float>(period), &s, &co);
    if (fabsf(co) < 1e-7f) return;
    const float alpha = 1.0f + ((s - 1.0f) / co);
    c_out   = 1.0f - 0.5f * alpha;
    oma_out = 1.0f - alpha;
    ok = true;
}


extern "C" __global__ __launch_bounds__(256, 2)
void bandpass_batch_from_hp_f32(
    const float* __restrict__ hp,
    int hp_rows,
    int len,
    const int*   __restrict__ hp_row_idx,
    const float* __restrict__ alphas,
    const float* __restrict__ betas,
    const int*   __restrict__ trig_periods,
    int n_combos,
    float* __restrict__ out_bp,
    float* __restrict__ out_bpn,
    float* __restrict__ out_sig,
    float* __restrict__ out_trg
) {
    const int row0   = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = row0; row < n_combos; row += stride) {
        const int hp_idx = hp_row_idx[row];
        if (hp_idx < 0 || hp_idx >= hp_rows) continue;

        const float* __restrict__ hp_row = hp + static_cast<size_t>(hp_idx) * len;
        float* __restrict__ bp_row   = out_bp  ? out_bp  + static_cast<size_t>(row) * len : nullptr;
        float* __restrict__ bpn_row  = out_bpn ? out_bpn + static_cast<size_t>(row) * len : nullptr;
        float* __restrict__ sig_row  = out_sig ? out_sig + static_cast<size_t>(row) * len : nullptr;
        float* __restrict__ trg_row  = out_trg ? out_trg + static_cast<size_t>(row) * len : nullptr;


        const float alpha = alphas[row];
        const float beta  = betas[row];
        const float a = 0.5f * (1.0f - alpha);
        const float c = beta * (1.0f + alpha);
        const float d = -alpha;


        float hc = 0.0f, homa = 0.0f; bool ok_hp;
        hpf_coeffs_from_period_f32(trig_periods[row], hc, homa, ok_hp);


        int start = 2;
        for (; start < len; ++start) {
            const float x2 = hp_row[start];
            const float x1 = hp_row[start - 1];
            const float x0 = hp_row[start - 2];
            if (is_finite_f32(x2) && is_finite_f32(x1) && is_finite_f32(x0)) break;
        }
        const int warm_bp = min(start, len);


        for (int i = 0; i < warm_bp; ++i) {
            if (bp_row)  bp_row[i]  = CUDART_NAN_F;
            if (bpn_row) bpn_row[i] = CUDART_NAN_F;
            if (trg_row) trg_row[i] = CUDART_NAN_F;
            if (sig_row) sig_row[i] = CUDART_NAN_F;
        }
        if (warm_bp >= len) continue;


        float y_im2 = hp_row[start - 2];
        float y_im1 = hp_row[start - 1];


        constexpr float K = 0.991f;
        float peak   = 0.0f;
        float prev_x = 0.0f, prev_y = 0.0f;
        bool  trig_init = false;


        #pragma unroll 4
        for (int i = start; i < len; ++i) {
            const float hi   = hp_row[i];
            const float him2 = hp_row[i - 2];


            float y = __fmaf_rn(d, y_im2, __fmaf_rn(c, y_im1, a * (hi - him2)));

            if (bp_row) bp_row[i] = y;


            peak = K * peak;
            const float av = fabsf(y);
            if (av > peak) peak = av;
            const float inv_peak = (peak > 0.0f) ? (1.0f / peak) : 0.0f;
            const float bn = y * inv_peak;
            if (bpn_row) bpn_row[i] = bn;


            float tr_val = CUDART_NAN_F;
            if (ok_hp) {
                if (!trig_init) {
                    prev_x = bn;
                    prev_y = bn;
                    trig_init = true;
                    tr_val = bn;
                } else {

                    prev_y = __fmaf_rn(homa, prev_y, hc * (bn - prev_x));
                    prev_x = bn;
                    tr_val = prev_y;
                }
            }
            if (trg_row) trg_row[i] = tr_val;


            if (sig_row) {
                float s = 0.0f;
                if (is_finite_f32(tr_val)) {
                    s = (bn < tr_val) ? 1.0f : ((bn > tr_val) ? -1.0f : 0.0f);
                }
                sig_row[i] = s;
            }


            y_im2 = y_im1;
            y_im1 = y;
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void bandpass_many_series_one_param_time_major_from_hp_f32(
    const float* __restrict__ hp_tm,
    int cols,
    int rows,
    float alpha_f,
    float beta_f,
    int trig_period,
    float* __restrict__ out_bp_tm,
    float* __restrict__ out_bpn_tm,
    float* __restrict__ out_sig_tm,
    float* __restrict__ out_trg_tm
) {
    if (cols <= 0 || rows <= 0) return;

    const float a = 0.5f * (1.0f - alpha_f);
    const float c = beta_f * (1.0f + alpha_f);
    const float d = -alpha_f;


    float hc = 0.0f, homa = 0.0f; bool ok_hp;
    hpf_coeffs_from_period_f32(trig_period, hc, homa, ok_hp);

    const int tpb = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += tpb) {
        auto at      = [&](const float* base, int t) -> float { return base[static_cast<size_t>(t) * cols + s]; };
        auto out_ref = [&](float* base, int t) -> float& { return base[static_cast<size_t>(t) * cols + s]; };


        int start = 2;
        for (; start < rows; ++start) {
            if (is_finite_f32(at(hp_tm, start)) &&
                is_finite_f32(at(hp_tm, start - 1)) &&
                is_finite_f32(at(hp_tm, start - 2))) break;
        }
        const int warm_bp = min(start, rows);


        for (int t = 0; t < warm_bp; ++t) {
            if (out_bp_tm)   out_ref(out_bp_tm,  t) = CUDART_NAN_F;
            if (out_bpn_tm)  out_ref(out_bpn_tm, t) = CUDART_NAN_F;
            if (out_trg_tm)  out_ref(out_trg_tm, t) = CUDART_NAN_F;
            if (out_sig_tm)  out_ref(out_sig_tm, t) = CUDART_NAN_F;
        }
        if (warm_bp >= rows) continue;


        float y_im2 = at(hp_tm, warm_bp - 2);
        float y_im1 = at(hp_tm, warm_bp - 1);


        constexpr float K = 0.991f;
        float peak   = 0.0f;
        float prev_x = 0.0f, prev_y = 0.0f;
        bool  trig_init = false;


        #pragma unroll 4
        for (int t = warm_bp; t < rows; ++t) {
            const float hi   = at(hp_tm, t);
            const float him2 = at(hp_tm, t - 2);

            float y = __fmaf_rn(d, y_im2, __fmaf_rn(c, y_im1, a * (hi - him2)));
            if (out_bp_tm) out_ref(out_bp_tm, t) = y;


            peak = K * peak;
            const float av = fabsf(y);
            if (av > peak) peak = av;
            const float inv_peak = (peak > 0.0f) ? (1.0f / peak) : 0.0f;
            const float bn = y * inv_peak;
            if (out_bpn_tm) out_ref(out_bpn_tm, t) = bn;


            float tr_val = CUDART_NAN_F;
            if (ok_hp) {
                if (!trig_init) { prev_x = bn; prev_y = bn; trig_init = true; tr_val = bn; }
                else { prev_y = __fmaf_rn(homa, prev_y, hc * (bn - prev_x)); prev_x = bn; tr_val = prev_y; }
            }
            if (out_trg_tm) out_ref(out_trg_tm, t) = tr_val;


            if (out_sig_tm) {
                float sgn = 0.0f;
                if (is_finite_f32(tr_val)) {
                    sgn = (bn < tr_val) ? 1.0f : ((bn > tr_val) ? -1.0f : 0.0f);
                }
                out_ref(out_sig_tm, t) = sgn;
            }

            y_im2 = y_im1;
            y_im1 = y;
        }
    }
}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: `bandpass_fill_bp` (src/indicators/bandpass.rs:303-370),
// which is the path `bandpass_output_into_slice` takes for the canonical `bp`
// field. The bp series is built from
// `highpass_scalar` (moving_averages/highpass.rs:438) and then
// `bandpass_scalar` (bandpass.rs:718).
//
// PERIOD-SWEPT: `compute_bandpass_batch` reads a parameter literally named
// `period` (default 20) and `bandwidth` (default 0.3, cpu_batch.rs:14179).
//
// SHAPE: one thread per combo walking bars ASCENDING. Both stages are 2-pole
// IIRs whose accumulation order is load-bearing, and the bandpass stage reads
// hp[i] and hp[i-2] -- so hp is produced INSIDE the same ascending loop and
// carried in three registers rather than materialised. No per-thread array,
// therefore no `max_period` and NEVER-OOM by construction.
//
// EPSILON: the f32 lane guards `fabsf(co) < 1e-7f` at :24. That constant is
// sized for f32 and is WRONG here; this kernel uses the CPU's own guard,
// `cos_val.abs() < 1e-15` (highpass.rs:334, :400), which is the f64 rule.
// ===========================================================================

static __forceinline__ __device__ double bandpass_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

static __forceinline__ __device__
void neo_bandpass_row_f64(const double* __restrict__ prices,
                          int n,
                          int period,
                          double bandwidth,
                          int first_valid,
                          double* __restrict__ row_bp,
                          double* __restrict__ row_bp_normalized,
                          double* __restrict__ row_signal,
                          double* __restrict__ row_trigger) {
    const double nn = bandpass_neo_qnan();
    for (int i = 0; i < n; ++i) {
        if (row_bp) row_bp[i] = nn;
        if (row_bp_normalized) row_bp_normalized[i] = nn;
        if (row_signal) row_signal[i] = nn;
        if (row_trigger) row_trigger[i] = nn;
    }

    // bandpass.rs:255 -- `position(|x| x.is_finite())`, which is the
    // `F64FirstValidRule::CloseFinite` this row declares.
    int first = first_valid;
    if (first < 0) first = 0;

    bool refused = false;

    // bandpass_prepare, :253-273.
    if (first >= n) refused = true;
    if (period <= 0 || period > n) refused = true;
    if (!refused && (n - first) < period) refused = true;
    if (!isfinite(bandwidth) || bandwidth <= 0.0 || bandwidth > 1.0) refused = true;

    // bandpass.rs:277-286. `f64::round` is half-away-from-zero and so is the
    // CUDA double `round`, so the two agree bit for bit.
    int hp_period = 0;
    int trig_period = 0;
    if (!refused) {
        const double hp_period_rounded = round(4.0 * (double)period / bandwidth);
        const double trig_period_rounded = round(((double)period / bandwidth) / 1.5);
        // Rust's positive float-to-usize cast saturates. Any value above this
        // row's length is rejected immediately by the downstream high-pass,
        // so compare in f64 before narrowing to the CUDA int ABI.
        if (!isfinite(hp_period_rounded) || hp_period_rounded < 2.0 || hp_period_rounded > (double)n) {
            refused = true;
        }
        if (!isfinite(trig_period_rounded) || trig_period_rounded < 2.0 || trig_period_rounded > (double)n) {
            refused = true;
        }
        if (!refused) {
            hp_period = (int)hp_period_rounded;
            trig_period = (int)trig_period_rounded;
        }
    }

    // highpass.rs:313-316 -- a SEPARATE scan, `!is_nan`, which names an
    // earlier bar than the `is_finite` scan above whenever the frame carries
    // an infinity. Derived here rather than taken from `first`.
    int first_hp = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(prices[i])) { first_hp = i; break; }
    }
    if (first_hp < 0) refused = true;

    double hp_theta = 0.0;
    if (!refused) {
        // highpass_with_kernel, :318-330: every refusal it makes.
        if (n <= 2 || hp_period <= 0 || hp_period > n) refused = true;
        else if ((n - first_hp) < hp_period) refused = true;
        else {
            hp_theta = 2.0 * M_PI * 1.0 / (double)hp_period;
            const double cos_val = cos(hp_theta);
            if (fabs(cos_val) < 1e-15) refused = true;
        }
    }

    if (refused) return;

    // bandpass.rs:317-318 -- warm_bp, and :331 bp_start = warm_bp - 2. The
    // highpass length check above guarantees first_hp + 2 <= n, so warm_bp is
    // exactly first_hp + 2 and bp_start is exactly first_hp.
    int warm_bp = first_hp + 2;
    if (warm_bp < 2) warm_bp = 2;
    if (warm_bp > n) warm_bp = n;
    int bp_start = warm_bp >= 2 ? (warm_bp - 2) : 0;

    // highpass_scalar, :447-450.
    const double hp_sin = sin(hp_theta);
    const double hp_cos = cos(hp_theta);
    const double alpha_hp = 1.0 + ((hp_sin - 1.0) / hp_cos);
    const double hp_c = 1.0 - 0.5 * alpha_hp;
    const double hp_oma = 1.0 - alpha_hp;

    // bandpass.rs:319-321 and bandpass_scalar :733-735.
    const double beta = cos(2.0 * M_PI / (double)period);
    const double gamma = cos(2.0 * M_PI * bandwidth / (double)period);
    const double alpha_bp = 1.0 / gamma - sqrt((1.0 / (gamma * gamma)) - 1.0);
    const double bp_a = 0.5 * (1.0 - alpha_bp);
    const double bp_c = beta * (1.0 + alpha_bp);
    const double bp_d = -alpha_bp;

    // One ascending pass. `hp_cur/hp_m1/hp_m2` carry the highpass series;
    // `y_m1/y_m2` carry the bandpass recursion. bp_start == first_hp, so the
    // two stages advance in lockstep and the relative index j = i - bp_start
    // is the index `bandpass_scalar` sees.
    double hp_cur = 0.0, hp_m1 = 0.0, hp_m2 = 0.0;
    double x_m1 = 0.0;
    double y_m1 = 0.0, y_m2 = 0.0;
    double peak = 0.0;

    for (int i = first_hp; i < n; ++i) {
        const double x = prices[i];
        if (i == first_hp) {
            hp_cur = x;               // highpass.rs:461 -- *dst = *src
        } else {
            hp_cur = fma(hp_oma, hp_m1, hp_c * (x - x_m1));
        }
        x_m1 = x;

        const int j = i - bp_start;
        double y;
        if (j == 0) {
            y = hp_cur;               // bandpass.rs:724 -- out[0] = hp[0]
        } else if (j == 1) {
            y = hp_cur;               // :728 -- out[1] = hp[1]
        } else {
            // :743 -- d.mul_add(y_im2, c.mul_add(y_im1, a * delta)). Two fmas
            // and one multiply, in that nesting. The f32 lane's unrolled form
            // is arithmetically the same recurrence; the unroll is not a
            // different accumulation.
            const double delta = hp_cur - hp_m2;
            y = fma(bp_d, y_m2, fma(bp_c, y_m1, bp_a * delta));
        }

        if (i >= warm_bp) {
            if (row_bp) row_bp[i] = y;
            if (row_bp_normalized) {
                peak *= 0.991;
                const double absolute = fabs(y);
                if (absolute > peak) peak = absolute;
                row_bp_normalized[i] = peak != 0.0 ? y / peak : 0.0;
            }
        }

        hp_m2 = hp_m1;
        hp_m1 = hp_cur;
        y_m2 = y_m1;
        y_m1 = y;
    }

    if (!row_bp_normalized || (!row_trigger && !row_signal) || warm_bp >= n) return;

    // `bandpass_output_into_slice` invokes highpass on the normalized suffix.
    // That helper performs its own first-!NaN scan and seeds the recurrence at
    // that exact element, so do not substitute the outer finite-price index.
    int first_trigger = -1;
    for (int i = warm_bp; i < n; ++i) {
        if (!isnan(row_bp_normalized[i])) {
            first_trigger = i;
            break;
        }
    }
    if (first_trigger < 0) return;

    const int trigger_len = n - warm_bp;
    const int trigger_first_relative = first_trigger - warm_bp;
    if (trigger_len <= 2 || trig_period <= 0 || trig_period > trigger_len) return;
    if ((trigger_len - trigger_first_relative) < trig_period) return;

    const double trigger_theta = 2.0 * M_PI / (double)trig_period;
    const double trigger_cos = cos(trigger_theta);
    if (fabs(trigger_cos) < 1e-15) return;
    const double trigger_alpha = 1.0 + ((sin(trigger_theta) - 1.0) / trigger_cos);
    const double trigger_c = 1.0 - 0.5 * trigger_alpha;
    const double trigger_oma = 1.0 - trigger_alpha;

    double trigger_x_prev = row_bp_normalized[first_trigger];
    double trigger_y_prev = trigger_x_prev;
    if (row_trigger) row_trigger[first_trigger] = trigger_y_prev;
    if (row_signal) row_signal[first_trigger] = 0.0;
    for (int i = first_trigger + 1; i < n; ++i) {
        const double current = row_bp_normalized[i];
        const double trigger_value =
            fma(trigger_oma, trigger_y_prev, trigger_c * (current - trigger_x_prev));
        if (row_trigger) row_trigger[i] = trigger_value;
        if (row_signal) {
            row_signal[i] = current < trigger_value
                ? 1.0
                : (current > trigger_value ? -1.0 : 0.0);
        }
        trigger_x_prev = current;
        trigger_y_prev = trigger_value;
    }
}

extern "C" __global__
void bandpass_neo_batch_f64(const double* __restrict__ prices,
                            int n,
                            const int* __restrict__ periods,
                            int n_combos,
                            int first_valid,
                            double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;
    double* __restrict__ row_bp = out + (size_t)combo * (size_t)n;
    neo_bandpass_row_f64(prices, n, periods[combo], 0.3, first_valid,
                         row_bp, nullptr, nullptr, nullptr);
}

extern "C" __global__
void bandpass_production_f64(const double* __restrict__ prices,
                             int n,
                             const int* __restrict__ periods,
                             const double* __restrict__ bandwidths,
                             int n_combos,
                             int first_valid,
                             double* __restrict__ out_bp,
                             double* __restrict__ out_bp_normalized,
                             double* __restrict__ out_signal,
                             double* __restrict__ out_trigger) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;
    const size_t offset = (size_t)combo * (size_t)n;
    neo_bandpass_row_f64(prices, n, periods[combo], bandwidths[combo], first_valid,
                         out_bp + offset,
                         out_bp_normalized + offset,
                         out_signal + offset,
                         out_trigger + offset);
}
