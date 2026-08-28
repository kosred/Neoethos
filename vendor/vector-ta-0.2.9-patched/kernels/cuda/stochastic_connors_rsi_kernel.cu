#include <cmath>
#include <cstddef>

namespace {
constexpr double SCALE_100 = 100.0;
constexpr double EPS = 1.0e-14;

struct WilderRsiState {
    int period;
    double inv_p;
    double beta;
    bool has_prev;
    double prev;
    int seed_count;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool seeded;

    __device__ void init(int value) {
        period = value;
        inv_p = 1.0 / static_cast<double>(value);
        beta = 1.0 - inv_p;
        reset();
    }

    __device__ void reset() {
        has_prev = false;
        prev = NAN;
        seed_count = 0;
        sum_gain = 0.0;
        sum_loss = 0.0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        seeded = false;
    }

    __device__ double update(double value) {
        if (!has_prev) {
            prev = value;
            has_prev = true;
            return NAN;
        }

        const double delta = value - prev;
        prev = value;

        if (!seeded) {
            sum_gain += fmax(delta, 0.0);
            sum_loss += fmax(-delta, 0.0);
            seed_count += 1;
            if (seed_count == period) {
                seeded = true;
                avg_gain = sum_gain * inv_p;
                avg_loss = sum_loss * inv_p;
                const double denom = avg_gain + avg_loss;
                return denom == 0.0 ? 50.0 : SCALE_100 * avg_gain / denom;
            }
            return NAN;
        }

        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);
        avg_gain = fma(avg_gain, beta, inv_p * gain);
        avg_loss = fma(avg_loss, beta, inv_p * loss);
        const double denom = avg_gain + avg_loss;
        return denom == 0.0 ? 50.0 : SCALE_100 * avg_gain / denom;
    }
};

struct SmaState {
    int period;
    double* buf;
    int head;
    int len;
    double sum;

    __device__ void init(int value, double* storage) {
        period = value;
        buf = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        len = 0;
        sum = 0.0;
    }

    __device__ double update(double value) {
        if (len < period) {
            buf[head] = value;
            sum += value;
            head += 1;
            if (head == period) {
                head = 0;
            }
            len += 1;
            if (len == period) {
                return sum / static_cast<double>(period);
            }
            return NAN;
        }

        const double old = buf[head];
        buf[head] = value;
        sum += value - old;
        head += 1;
        if (head == period) {
            head = 0;
        }
        return sum / static_cast<double>(period);
    }
};
}

extern "C" __global__ void stochastic_connors_rsi_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ stoch_lengths,
    const int* __restrict__ smooth_ks,
    const int* __restrict__ smooth_ds,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ updown_lengths,
    const int* __restrict__ roc_lengths,
    int rows,
    double* __restrict__ out_k,
    double* __restrict__ out_d
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int stoch_length = stoch_lengths[row];
    const int smooth_k = smooth_ks[row];
    const int smooth_d = smooth_ds[row];
    const int rsi_length = rsi_lengths[row];
    const int updown_length = updown_lengths[row];
    const int roc_length = roc_lengths[row];

    double* row_k = out_k + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_d = out_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_k[i] = NAN;
        row_d[i] = NAN;
    }

    if (stoch_length <= 0 || stoch_length > len || smooth_k <= 0 || smooth_k > len ||
        smooth_d <= 0 || smooth_d > len || rsi_length <= 0 || rsi_length > len ||
        updown_length <= 0 || updown_length > len || roc_length <= 0 || roc_length > len) {
        return;
    }

    double* roc_window = new double[roc_length];
    double* crsi_window = new double[stoch_length];
    double* k_buf = new double[smooth_k];
    double* d_buf = new double[smooth_d];
    if (roc_window == nullptr || crsi_window == nullptr || k_buf == nullptr || d_buf == nullptr) {
        delete[] roc_window;
        delete[] crsi_window;
        delete[] k_buf;
        delete[] d_buf;
        return;
    }

    WilderRsiState src_rsi;
    WilderRsiState streak_rsi;
    src_rsi.init(rsi_length);
    streak_rsi.init(updown_length);

    SmaState k_sma;
    SmaState d_sma;
    k_sma.init(smooth_k, k_buf);
    d_sma.init(smooth_d, d_buf);

    bool has_prev_source = false;
    double prev_source = NAN;
    long long streak = 0;

    int roc_head = 0;
    int roc_count = 0;
    int crsi_head = 0;
    int crsi_count = 0;

    for (int i = 0; i < len; ++i) {
        const double source = data[i];
        if (!isfinite(source)) {
            has_prev_source = false;
            prev_source = NAN;
            streak = 0;
            src_rsi.reset();
            streak_rsi.reset();
            k_sma.reset();
            d_sma.reset();
            roc_head = 0;
            roc_count = 0;
            crsi_head = 0;
            crsi_count = 0;
            continue;
        }

        const bool had_prev = has_prev_source;
        const double prev_value = prev_source;
        if (had_prev) {
            if (source > prev_value) {
                streak = streak >= 0 ? streak + 1 : 1;
            } else if (source < prev_value) {
                streak = streak <= 0 ? streak - 1 : -1;
            } else {
                streak = 0;
            }
        } else {
            streak = 0;
        }
        prev_source = source;
        has_prev_source = true;

        const double src_value = src_rsi.update(source);
        const double streak_value = streak_rsi.update(static_cast<double>(streak));

        double percent_rank = NAN;
        if (had_prev) {
            const double roc =
                (prev_value == 0.0 || !isfinite(prev_value)) ? 0.0 : fma(source / prev_value, SCALE_100, -SCALE_100);

            if (roc_count < roc_length) {
                roc_window[roc_count] = roc;
                roc_count += 1;
            } else {
                roc_window[roc_head] = roc;
                roc_head += 1;
                if (roc_head == roc_length) {
                    roc_head = 0;
                }
            }

            if (roc_count == roc_length) {
                int count = 0;
                for (int j = 0; j < roc_count; ++j) {
                    if (roc_window[j] <= roc) {
                        count += 1;
                    }
                }
                percent_rank = SCALE_100 * static_cast<double>(count) / static_cast<double>(roc_length);
            }
        }

        if (!isfinite(src_value) || !isfinite(streak_value) || !isfinite(percent_rank)) {
            continue;
        }

        const double crsi = (src_value + streak_value + percent_rank) / 3.0;
        if (crsi_count < stoch_length) {
            crsi_window[crsi_count] = crsi;
            crsi_count += 1;
        } else {
            crsi_window[crsi_head] = crsi;
            crsi_head += 1;
            if (crsi_head == stoch_length) {
                crsi_head = 0;
            }
        }

        if (crsi_count < stoch_length) {
            continue;
        }

        double ll = crsi_window[0];
        double hh = crsi_window[0];
        for (int j = 1; j < crsi_count; ++j) {
            const double value = crsi_window[j];
            if (value < ll) {
                ll = value;
            }
            if (value > hh) {
                hh = value;
            }
        }

        const double denom = hh - ll;
        const double raw_k =
            fabs(denom) < EPS ? 0.0 : (crsi - ll) * (SCALE_100 / denom);
        const double k = k_sma.update(raw_k);
        if (!isfinite(k)) {
            continue;
        }
        const double d = d_sma.update(k);

        row_k[i] = k;
        row_d[i] = d;
    }

    delete[] roc_window;
    delete[] crsi_window;
    delete[] k_buf;
    delete[] d_buf;
}

/* ===========================================================================
 * NEOETHOS f64 LANE — stochastic_connors_rsi
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/stochastic_connors_rsi.rs:742
 *   stochastic_connors_rsi_compute_into, driving
 *   StochasticConnorsRsiState::update_reset_on_nan (:633) -> update (:513),
 *   with WilderRsiState::update (:377) and SmaState::update (:447).
 *
 * Column: output_id "value" resolves to out.k — cpu_batch.rs:12177 accepts
 *   "k"/"value" and returns the K series. The D series is a separate output id
 *   and its SMA feeds nothing back into K, so it is not advanced here.
 *
 * PERIOD-INVARIANT: compute_stochastic_connors_rsi_batch reads stoch_length
 *   (3), smooth_k (3), smooth_d (3), rsi_length (3), updown_length (2) and
 *   roc_length (100) and NEVER period (cpu_batch.rs:12135-12142). Five swept
 *   periods give five identical CPU columns, so this kernel emits five
 *   identical rows.
 *
 * FIRST-VALID IGNORED: update_reset_on_nan RESETS the whole state on a
 *   non-finite source and writes NaN for that bar (:634-637), and the compute
 *   walks EVERY bar from index 0. The caller's first-valid index is never read.
 *
 * Input: one price series, CPU default source close (cpu_batch.rs:12134) —
 *   F64InputKind::CloseSlice.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Two Wilder RSI recurrences, a
 *   streak counter, a 100-deep ROC ring for the percent-rank, two MONOTONE
 *   DEQUES over the ConnorsRSI series and a sliding-sum SMA all carry state.
 *
 * ARITHMETIC taken verbatim:
 *   * the Wilder seed is sum_gain += max(delta, 0) then avg = sum * inv_p
 *     (:391-397): a SUM then ONE multiply by the reciprocal, not a divide.
 *   * the Wilder step is avg_gain.mul_add(beta, inv_p * gain) (:407) — ONE
 *     fma whose addend is itself a product. Reproduced with fma().
 *   * the ROC is (source / prev).mul_add(100, -100) (:546) — ONE fma, not
 *     (source/prev - 1) * 100.
 *   * raw_k is (crsi - ll).mul_add(100 / denom, 0.0) (:623) — ONE fma with a
 *     zero addend, which does not round the same as a bare product.
 *   * EPS is 1.0e-14 (:32) and is already an f64-sized tolerance; it is
 *     carried across unchanged rather than rescaled.
 *   * SmaState steady state is sum += value - old (:462) — the DIFFERENCE is
 *     formed first, one rounding, then added. Not sum += value; sum -= old.
 *
 * NaN semantics: delta.max(0.0) and (-delta).max(0.0) are f64::max, which
 *   return the non-NaN operand; fmax is used for exactly that reason.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:12135-12142. Each bounds a per-thread ring or
 * deque, so the bounds belong to the COMPILED kernel. */
#define NEO_SCRSI_STOCH_LENGTH   3
#define NEO_SCRSI_SMOOTH_K       3
#define NEO_SCRSI_RSI_LENGTH     3
#define NEO_SCRSI_UPDOWN_LENGTH  2
#define NEO_SCRSI_ROC_LENGTH   100
#define NEO_SCRSI_SCALE_100  100.0
#define NEO_SCRSI_EPS        1.0e-14
#define NEO_SCRSI_DEQUE_CAP  (NEO_SCRSI_STOCH_LENGTH + 1)
/* One slot more than the window: the CPU pushes BEFORE it pops the front
 * (:549-552), so the ROC ring is momentarily roc_length + 1 deep. */
#define NEO_SCRSI_ROC_CAP    (NEO_SCRSI_ROC_LENGTH + 1)

/* WilderRsiState, one instance per component. */
struct NeoScrsiWilder {
    int    period;
    double inv_p;
    double beta;
    bool   has_prev;
    double prev;
    int    seed_count;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool   seeded;
};

static __device__ inline void neo_scrsi_wilder_init(NeoScrsiWilder* w, int period) {
    w->period   = period;
    w->inv_p    = 1.0 / (double)period;
    w->beta     = 1.0 - w->inv_p;
    w->has_prev = false;
    w->prev     = NEO_F64_NAN;
    w->seed_count = 0;
    w->sum_gain = 0.0;
    w->sum_loss = 0.0;
    w->avg_gain = 0.0;
    w->avg_loss = 0.0;
    w->seeded   = false;
}

static __device__ inline void neo_scrsi_wilder_reset(NeoScrsiWilder* w) {
    w->has_prev = false;
    w->prev     = NEO_F64_NAN;
    w->seed_count = 0;
    w->sum_gain = 0.0;
    w->sum_loss = 0.0;
    w->avg_gain = 0.0;
    w->avg_loss = 0.0;
    w->seeded   = false;
}

/* Returns true and writes *out when the CPU returns Some. */
static __device__ inline bool neo_scrsi_wilder_update(NeoScrsiWilder* w,
                                                      double value,
                                                      double* out) {
    if (!w->has_prev) { w->prev = value; w->has_prev = true; return false; }

    const double delta = value - w->prev;
    w->prev = value;

    if (!w->seeded) {
        w->sum_gain += fmax(delta, 0.0);
        w->sum_loss += fmax(-delta, 0.0);
        w->seed_count += 1;
        if (w->seed_count == w->period) {
            w->seeded   = true;
            w->avg_gain = w->sum_gain * w->inv_p;
            w->avg_loss = w->sum_loss * w->inv_p;
            const double denom = w->avg_gain + w->avg_loss;
            *out = (denom == 0.0) ? 50.0 : (NEO_SCRSI_SCALE_100 * w->avg_gain / denom);
            return true;
        }
        return false;
    }

    const double gain = fmax(delta, 0.0);
    const double loss = fmax(-delta, 0.0);
    w->avg_gain = fma(w->avg_gain, w->beta, w->inv_p * gain);
    w->avg_loss = fma(w->avg_loss, w->beta, w->inv_p * loss);
    const double denom = w->avg_gain + w->avg_loss;
    *out = (denom == 0.0) ? 50.0 : (NEO_SCRSI_SCALE_100 * w->avg_gain / denom);
    return true;
}

extern "C" __global__
void stochastic_connors_rsi_neo_batch_f64(const double* __restrict__ prices,
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

    const int stoch_length = NEO_SCRSI_STOCH_LENGTH;
    const int smooth_k     = NEO_SCRSI_SMOOTH_K;
    const int roc_length   = NEO_SCRSI_ROC_LENGTH;

    /* stochastic_connors_rsi_prepare (:679-728) refuses any window longer
     * than the series before a column is produced. */
    if (stoch_length > n || smooth_k > n || NEO_SCRSI_RSI_LENGTH > n ||
        NEO_SCRSI_UPDOWN_LENGTH > n || roc_length > n) return;

    NeoScrsiWilder src_rsi, streak_rsi;
    neo_scrsi_wilder_init(&src_rsi, NEO_SCRSI_RSI_LENGTH);
    neo_scrsi_wilder_init(&streak_rsi, NEO_SCRSI_UPDOWN_LENGTH);

    bool   has_prev_source = false;
    double prev_source = 0.0;
    long long streak = 0;

    double roc_ring[NEO_SCRSI_ROC_CAP];
    int roc_lo = 0, roc_len = 0;

    double mn_v[NEO_SCRSI_DEQUE_CAP]; int mn_i[NEO_SCRSI_DEQUE_CAP];
    double mx_v[NEO_SCRSI_DEQUE_CAP]; int mx_i[NEO_SCRSI_DEQUE_CAP];
    int mn_lo = 0, mn_len = 0, mx_lo = 0, mx_len = 0;
    int crsi_seen = 0;

    double k_buf[NEO_SCRSI_SMOOTH_K];
    int k_head = 0, k_len = 0;
    double k_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double source = prices[i];

        if (!isfinite(source)) {
            neo_scrsi_wilder_reset(&src_rsi);
            neo_scrsi_wilder_reset(&streak_rsi);
            has_prev_source = false; prev_source = 0.0; streak = 0;
            roc_lo = 0; roc_len = 0;
            crsi_seen = 0; mn_lo = 0; mn_len = 0; mx_lo = 0; mx_len = 0;
            k_head = 0; k_len = 0; k_sum = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const bool   had_prev  = has_prev_source;
        const double prev_val  = prev_source;

        if (had_prev && source > prev_val)      streak = (streak >= 0) ? (streak + 1) : 1;
        else if (had_prev && source < prev_val) streak = (streak <= 0) ? (streak - 1) : -1;
        else                                    streak = 0;

        has_prev_source = true;
        prev_source = source;

        double a, b;
        const bool have_a = neo_scrsi_wilder_update(&src_rsi, source, &a);
        const bool have_b = neo_scrsi_wilder_update(&streak_rsi, (double)streak, &b);

        bool   have_c = false;
        double c = 0.0;
        if (had_prev) {
            const double roc = (prev_val == 0.0 || !isfinite(prev_val))
                ? 0.0
                : fma(source / prev_val, NEO_SCRSI_SCALE_100, -NEO_SCRSI_SCALE_100);

            roc_ring[(roc_lo + roc_len) % NEO_SCRSI_ROC_CAP] = roc;
            ++roc_len;
            if (roc_len > roc_length) {
                roc_lo = (roc_lo + 1) % NEO_SCRSI_ROC_CAP;
                --roc_len;
            }
            if (roc_len == roc_length) {
                int count = 0;
                for (int q = 0; q < roc_len; ++q) {
                    if (roc_ring[(roc_lo + q) % NEO_SCRSI_ROC_CAP] <= roc) ++count;
                }
                c = NEO_SCRSI_SCALE_100 * (double)count / (double)roc_length;
                have_c = true;
            }
        }

        if (!(have_a && have_b && have_c)) { o[i] = NEO_F64_NAN; continue; }

        const double crsi = (a + b + c) / 3.0;

        const int idx = crsi_seen;
        ++crsi_seen;

        while (mn_len > 0 && mn_v[(mn_lo + mn_len - 1) % NEO_SCRSI_DEQUE_CAP] >= crsi) --mn_len;
        mn_v[(mn_lo + mn_len) % NEO_SCRSI_DEQUE_CAP] = crsi;
        mn_i[(mn_lo + mn_len) % NEO_SCRSI_DEQUE_CAP] = idx;
        ++mn_len;

        while (mx_len > 0 && mx_v[(mx_lo + mx_len - 1) % NEO_SCRSI_DEQUE_CAP] <= crsi) --mx_len;
        mx_v[(mx_lo + mx_len) % NEO_SCRSI_DEQUE_CAP] = crsi;
        mx_i[(mx_lo + mx_len) % NEO_SCRSI_DEQUE_CAP] = idx;
        ++mx_len;

        if (crsi_seen < stoch_length) { o[i] = NEO_F64_NAN; continue; }

        const int window_start = crsi_seen - stoch_length;
        while (mn_len > 0 && mn_i[mn_lo] < window_start) { mn_lo = (mn_lo + 1) % NEO_SCRSI_DEQUE_CAP; --mn_len; }
        while (mx_len > 0 && mx_i[mx_lo] < window_start) { mx_lo = (mx_lo + 1) % NEO_SCRSI_DEQUE_CAP; --mx_len; }

        const double ll = (mn_len > 0) ? mn_v[mn_lo] : crsi;
        const double hh = (mx_len > 0) ? mx_v[mx_lo] : crsi;
        const double denom = hh - ll;
        const double raw_k = (fabs(denom) < NEO_SCRSI_EPS)
            ? 0.0
            : fma(crsi - ll, NEO_SCRSI_SCALE_100 / denom, 0.0);

        /* SmaState::update (:447) */
        double k_value;
        if (k_len < smooth_k) {
            k_buf[k_head] = raw_k; k_sum += raw_k;
            ++k_head; if (k_head == smooth_k) k_head = 0;
            ++k_len;
            if (k_len < smooth_k) { o[i] = NEO_F64_NAN; continue; }
            k_value = k_sum / (double)smooth_k;
        } else {
            const double old = k_buf[k_head];
            k_buf[k_head] = raw_k;
            k_sum += raw_k - old;
            ++k_head; if (k_head == smooth_k) k_head = 0;
            k_value = k_sum / (double)smooth_k;
        }

        o[i] = k_value;
    }
}
