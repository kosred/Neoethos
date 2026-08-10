#include <cmath>
#include <cstddef>

static __device__ inline bool sad_valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close) && high >= low;
}

static __device__ inline bool sad_sma_update(
    double value,
    double* buffer,
    int period,
    int* count,
    int* head,
    double* sum,
    double* out
) {
    if (*count < period) {
        buffer[(*head + *count) % period] = value;
        *sum += value;
        *count += 1;
    } else {
        *sum -= buffer[*head];
        buffer[*head] = value;
        *sum += value;
        *head += 1;
        if (*head == period) {
            *head = 0;
        }
    }

    if (*count == period) {
        *out = *sum / static_cast<double>(period);
        return true;
    }
    return false;
}

extern "C" __global__ void stochastic_adaptive_d_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ k_lengths,
    const int* __restrict__ d_smoothings,
    const int* __restrict__ pre_smooths,
    const double* __restrict__ attenuations,
    int rows,
    double* __restrict__ out_standard_d,
    double* __restrict__ out_adaptive_d,
    double* __restrict__ out_difference
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int k_length = k_lengths[row];
    int d_smoothing = d_smoothings[row];
    int pre_smooth = pre_smooths[row];
    double attenuation = attenuations[row];

    double* row_standard = out_standard_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_adaptive = out_adaptive_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_difference = out_difference + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_standard[i] = NAN;
        row_adaptive[i] = NAN;
        row_difference[i] = NAN;
    }

    if (k_length <= 0 || d_smoothing <= 0 || pre_smooth <= 0 || !isfinite(attenuation) ||
        attenuation < 0.1) {
        return;
    }

    double* pre_high_buf = new double[pre_smooth];
    double* pre_low_buf = new double[pre_smooth];
    double* pre_close_buf = new double[pre_smooth];
    double* stoch_high_buf = new double[k_length];
    double* stoch_low_buf = new double[k_length];
    double* d_buf = new double[d_smoothing];
    if (pre_high_buf == nullptr || pre_low_buf == nullptr || pre_close_buf == nullptr ||
        stoch_high_buf == nullptr || stoch_low_buf == nullptr || d_buf == nullptr) {
        delete[] pre_high_buf;
        delete[] pre_low_buf;
        delete[] pre_close_buf;
        delete[] stoch_high_buf;
        delete[] stoch_low_buf;
        delete[] d_buf;
        return;
    }

    int pre_high_count = 0;
    int pre_low_count = 0;
    int pre_close_count = 0;
    int pre_high_head = 0;
    int pre_low_head = 0;
    int pre_close_head = 0;
    double pre_high_sum = 0.0;
    double pre_low_sum = 0.0;
    double pre_close_sum = 0.0;

    int stoch_count = 0;
    int stoch_head = 0;

    int d_count = 0;
    int d_head = 0;
    double d_sum = 0.0;

    double adaptive = 50.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];

        if (!sad_valid_bar(h, l, c)) {
            pre_high_count = 0;
            pre_low_count = 0;
            pre_close_count = 0;
            pre_high_head = 0;
            pre_low_head = 0;
            pre_close_head = 0;
            pre_high_sum = 0.0;
            pre_low_sum = 0.0;
            pre_close_sum = 0.0;
            stoch_count = 0;
            stoch_head = 0;
            d_count = 0;
            d_head = 0;
            d_sum = 0.0;
            adaptive = 50.0;
            continue;
        }

        double s_high = NAN;
        double s_low = NAN;
        double s_close = NAN;
        if (!sad_sma_update(
                h,
                pre_high_buf,
                pre_smooth,
                &pre_high_count,
                &pre_high_head,
                &pre_high_sum,
                &s_high
            ) ||
            !sad_sma_update(
                l,
                pre_low_buf,
                pre_smooth,
                &pre_low_count,
                &pre_low_head,
                &pre_low_sum,
                &s_low
            ) ||
            !sad_sma_update(
                c,
                pre_close_buf,
                pre_smooth,
                &pre_close_count,
                &pre_close_head,
                &pre_close_sum,
                &s_close
            )) {
            continue;
        }

        if (stoch_count < k_length) {
            stoch_high_buf[(stoch_head + stoch_count) % k_length] = s_high;
            stoch_low_buf[(stoch_head + stoch_count) % k_length] = s_low;
            stoch_count += 1;
        } else {
            stoch_high_buf[stoch_head] = s_high;
            stoch_low_buf[stoch_head] = s_low;
            stoch_head += 1;
            if (stoch_head == k_length) {
                stoch_head = 0;
            }
        }

        if (stoch_count < k_length) {
            continue;
        }

        double highest = stoch_high_buf[0];
        double lowest = stoch_low_buf[0];
        for (int j = 1; j < stoch_count; ++j) {
            if (stoch_high_buf[j] > highest) {
                highest = stoch_high_buf[j];
            }
            if (stoch_low_buf[j] < lowest) {
                lowest = stoch_low_buf[j];
            }
        }

        double range = highest - lowest;
        double stoch_raw = fabs(range) <= 1.0e-12 ? 50.0 : (s_close - lowest) * (100.0 / range);

        double stoch_d_raw = NAN;
        if (!sad_sma_update(
                stoch_raw,
                d_buf,
                d_smoothing,
                &d_count,
                &d_head,
                &d_sum,
                &stoch_d_raw
            )) {
            continue;
        }

        double standard_d = 50.0 + (stoch_d_raw - 50.0) * 0.5;
        double alpha = (fabs(standard_d - 50.0) / 100.0) / attenuation;
        double src_ama = (standard_d - 50.0) / attenuation + 50.0;
        adaptive = adaptive + alpha * (src_ama - adaptive);
        double difference = 50.0 + (standard_d - adaptive) * 2.0;

        row_standard[i] = standard_d;
        row_adaptive[i] = adaptive;
        row_difference[i] = difference;
    }

    delete[] pre_high_buf;
    delete[] pre_low_buf;
    delete[] pre_close_buf;
    delete[] stoch_high_buf;
    delete[] stoch_low_buf;
    delete[] d_buf;
}

/* ===========================================================================
 * NEOETHOS f64 LANE — stochastic_adaptive_d
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/stochastic_adaptive_d.rs:536
 *   stochastic_adaptive_d_compute_into, driving RollingSma::update (:417),
 *   RollingExtrema::update (:471), compute_stochastic_raw (:520) and
 *   compute_ama (:529).
 *
 * Column: output_id "value" resolves to out.standard_d — cpu_batch.rs:12092
 *   accepts "standard_d"/"value" and returns the STANDARD_D series.
 *
 * PERIOD-INVARIANT: compute_stochastic_adaptive_d_batch reads k_length (20),
 *   d_smoothing (9), pre_smooth (20) and attenuation (2.0) and NEVER period
 *   (cpu_batch.rs:12080-12083), so five swept periods give five identical CPU
 *   columns and this kernel emits five identical rows.
 *
 * FIRST-VALID IGNORED: the compute walks EVERY bar from index 0, writes NaN
 *   for a bar that fails valid_bar (:306 — all three finite AND high >= low)
 *   and RESETS the three pre-smoothers, the extrema window, the D SMA and the
 *   adaptive state to CENTER. The caller's first-valid index is never read.
 *
 * Input: high / low / close — F64InputKind::Hlc.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Five stateful stages run in
 *   series — three sliding-sum SMAs, a pair of MONOTONE DEQUES, a fourth SMA,
 *   and the AMA recurrence whose previous value is its own output — so no part
 *   of this is bar-parallel.
 *
 * ARITHMETIC taken verbatim:
 *   * RollingSma steady state is sum += value; sum -= old (:434-435) — an ADD
 *     THEN A SUBTRACT, in that order, two roundings. Not (sum - old) + value.
 *   * compute_stochastic_raw is (close - lowest).mul_add(100/range, 0.0)
 *     (:525) — ONE fma with a zero addend, which is NOT the same rounding as a
 *     bare product when the product is inexact. Reproduced with fma().
 *   * compute_ama (:529-534) forms alpha and src_ama as written and returns
 *     prev + alpha * (src_ama - prev) — no fma, because the CPU line has none.
 *   * EPS is 1.0e-12 (:35), already an f64-sized tolerance, so it is carried
 *     across unchanged rather than rescaled.
 *
 * NaN semantics: the deque comparisons only ever see values that passed
 *   valid_bar, so they are finite; the extrema are still taken from the deque
 *   fronts exactly as the CPU takes them.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:12080-12083 (:29-32). Each bounds a per-thread
 * ring or deque, so the bounds belong to the COMPILED kernel. */
#define NEO_SAD_K_LENGTH     20
#define NEO_SAD_D_SMOOTHING   9
#define NEO_SAD_PRE_SMOOTH   20
#define NEO_SAD_ATTENUATION  2.0
#define NEO_SAD_SCALE_100  100.0
#define NEO_SAD_CENTER      50.0
#define NEO_SAD_EPS         1.0e-12
/* One slot more than the window: RollingExtrema pushes BEFORE it prunes. */
#define NEO_SAD_DEQUE_CAP   (NEO_SAD_K_LENGTH + 1)

extern "C" __global__
void stochastic_adaptive_d_neo_batch_f64(const double* __restrict__ high,
                                         const double* __restrict__ low,
                                         const double* __restrict__ close,
                                         int n,
                                         const int* __restrict__ periods,
                                         int n_combos,
                                         int first_valid,
                                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* the mid-series reset reproduces it — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int    kl    = NEO_SAD_K_LENGTH;
    const int    dsm   = NEO_SAD_D_SMOOTHING;
    const int    pre   = NEO_SAD_PRE_SMOOTH;
    const double atten = NEO_SAD_ATTENUATION;

    /* validate_params refuses a window longer than the series. */
    if (pre > n || kl > n || dsm > n) return;

    double bh[NEO_SAD_PRE_SMOOTH], bl[NEO_SAD_PRE_SMOOTH], bc[NEO_SAD_PRE_SMOOTH];
    double sh = 0.0, sl = 0.0, sc = 0.0;
    int hh_ = 0, hl_ = 0, hc_ = 0, ch_ = 0, cl_ = 0, cc_ = 0;

    double bd[NEO_SAD_D_SMOOTHING];
    double sd = 0.0;
    int hd = 0, cd = 0;

    double mx_v[NEO_SAD_DEQUE_CAP]; int mx_i[NEO_SAD_DEQUE_CAP];
    double mn_v[NEO_SAD_DEQUE_CAP]; int mn_i[NEO_SAD_DEQUE_CAP];
    int mx_lo = 0, mx_len = 0, mn_lo = 0, mn_len = 0;
    int ex_index = 0;

    double adaptive = NEO_SAD_CENTER;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i];

        if (!(isfinite(h) && isfinite(l) && isfinite(c) && h >= l)) {
            sh = 0.0; hh_ = 0; ch_ = 0;
            sl = 0.0; hl_ = 0; cl_ = 0;
            sc = 0.0; hc_ = 0; cc_ = 0;
            sd = 0.0; hd = 0; cd = 0;
            ex_index = 0; mx_lo = 0; mx_len = 0; mn_lo = 0; mn_len = 0;
            adaptive = NEO_SAD_CENTER;
            o[i] = NEO_F64_NAN;
            continue;
        }

        /* Three pre-smoothing SMAs — RollingSma::update (:417).
         *
         * THE SHORT-CIRCUIT IS LOAD-BEARING AND IS REPRODUCED EXACTLY. The CPU
         * writes `let Some(s_high) = pre_high.update(h) else { ..; continue }`
         * (:576-580), so when the HIGH smoother is still warming, pre_low and
         * pre_close are NOT updated for that bar at all. The three stages
         * therefore warm up STAGGERED — low starts receiving values only once
         * high is ready, close only once low is ready — and the extrema window
         * later still. Updating all three unconditionally would fill them
         * `pre - 1` bars early each and shift the whole series. */
        double s_high, s_low, s_close;

        if (ch_ < pre) {
            bh[hh_] = h; sh += h; ++hh_; if (hh_ == pre) hh_ = 0; ++ch_;
            if (ch_ < pre) { o[i] = NEO_F64_NAN; continue; }
            s_high = sh / (double)pre;
        } else {
            const double old = bh[hh_]; bh[hh_] = h; sh += h; sh -= old;
            ++hh_; if (hh_ == pre) hh_ = 0; s_high = sh / (double)pre;
        }

        if (cl_ < pre) {
            bl[hl_] = l; sl += l; ++hl_; if (hl_ == pre) hl_ = 0; ++cl_;
            if (cl_ < pre) { o[i] = NEO_F64_NAN; continue; }
            s_low = sl / (double)pre;
        } else {
            const double old = bl[hl_]; bl[hl_] = l; sl += l; sl -= old;
            ++hl_; if (hl_ == pre) hl_ = 0; s_low = sl / (double)pre;
        }

        if (cc_ < pre) {
            bc[hc_] = c; sc += c; ++hc_; if (hc_ == pre) hc_ = 0; ++cc_;
            if (cc_ < pre) { o[i] = NEO_F64_NAN; continue; }
            s_close = sc / (double)pre;
        } else {
            const double old = bc[hc_]; bc[hc_] = c; sc += c; sc -= old;
            ++hc_; if (hc_ == pre) hc_ = 0; s_close = sc / (double)pre;
        }

        /* RollingExtrema::update (:471) over the smoothed high/low */
        const int idx = ex_index;
        ++ex_index;

        while (mx_len > 0 && mx_v[(mx_lo + mx_len - 1) % NEO_SAD_DEQUE_CAP] <= s_high) --mx_len;
        mx_v[(mx_lo + mx_len) % NEO_SAD_DEQUE_CAP] = s_high;
        mx_i[(mx_lo + mx_len) % NEO_SAD_DEQUE_CAP] = idx;
        ++mx_len;

        while (mn_len > 0 && mn_v[(mn_lo + mn_len - 1) % NEO_SAD_DEQUE_CAP] >= s_low) --mn_len;
        mn_v[(mn_lo + mn_len) % NEO_SAD_DEQUE_CAP] = s_low;
        mn_i[(mn_lo + mn_len) % NEO_SAD_DEQUE_CAP] = idx;
        ++mn_len;

        const int wstart = (idx + 1 >= kl) ? (idx + 1 - kl) : 0;
        while (mx_len > 0 && mx_i[mx_lo] < wstart) { mx_lo = (mx_lo + 1) % NEO_SAD_DEQUE_CAP; --mx_len; }
        while (mn_len > 0 && mn_i[mn_lo] < wstart) { mn_lo = (mn_lo + 1) % NEO_SAD_DEQUE_CAP; --mn_len; }

        if (idx + 1 < kl) { o[i] = NEO_F64_NAN; continue; }

        const double highest = (mx_len > 0) ? mx_v[mx_lo] : s_high;
        const double lowest  = (mn_len > 0) ? mn_v[mn_lo] : s_low;

        /* compute_stochastic_raw (:520) */
        const double range = highest - lowest;
        const double stoch_raw = (fabs(range) <= NEO_SAD_EPS)
            ? NEO_SAD_CENTER
            : fma(s_close - lowest, NEO_SAD_SCALE_100 / range, 0.0);

        /* the D SMA */
        double stoch_d_raw;
        if (cd < dsm) {
            bd[hd] = stoch_raw; sd += stoch_raw; ++hd; if (hd == dsm) hd = 0; ++cd;
            if (cd < dsm) { o[i] = NEO_F64_NAN; continue; }
            stoch_d_raw = sd / (double)dsm;
        } else {
            const double old = bd[hd]; bd[hd] = stoch_raw; sd += stoch_raw; sd -= old;
            ++hd; if (hd == dsm) hd = 0; stoch_d_raw = sd / (double)dsm;
        }

        const double standard_d = NEO_SAD_CENTER + (stoch_d_raw - NEO_SAD_CENTER) * 0.5;

        /* compute_ama (:529) — advance it even though the lane emits
         * standard_d, because adaptive is carried across bars and the
         * adaptive_d / difference columns share this stream. */
        const double alpha_a = (fabs(standard_d - NEO_SAD_CENTER) / NEO_SAD_SCALE_100) / atten;
        const double src_ama = (standard_d - NEO_SAD_CENTER) / atten + NEO_SAD_CENTER;
        adaptive = adaptive + alpha_a * (src_ama - adaptive);

        o[i] = standard_d;
    }
}
