#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void stochastic_distance_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookback_lengths,
    const int* __restrict__ length1s,
    const int* __restrict__ length2s,
    const int* __restrict__ ob_levels,
    const int* __restrict__ os_levels,
    int n_combos,
    int max_lookback,
    int max_length1,
    double* __restrict__ close_buffer,
    double* __restrict__ distance_buffer,
    double* __restrict__ out_oscillator,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int lookback_length = lookback_lengths[combo_idx];
    int length1 = length1s[combo_idx];
    int length2 = length2s[combo_idx];
    double ob_level = static_cast<double>(ob_levels[combo_idx]);
    double os_level = static_cast<double>(os_levels[combo_idx]);
    double* close_ring =
        close_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length1);
    double* distance_ring =
        distance_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_lookback);
    double* row_oscillator =
        out_oscillator + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal =
        out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = CUDART_NAN;
        row_signal[i] = CUDART_NAN;
    }

    if (lookback_length <= 0 ||
        length1 <= 0 ||
        length2 <= 0 ||
        os_level >= ob_level ||
        lookback_length > max_lookback ||
        length1 > max_length1) {
        return;
    }

    int close_head = 0;
    int close_count = 0;
    int distance_head = 0;
    int distance_count = 0;
    double ema = 0.0;
    bool have_ema = false;
    double prev_sdo = 0.0;
    double alpha = 2.0 / (static_cast<double>(length2) + 1.0);
    const double tol = 1e-12;

    for (int i = 0; i < len; ++i) {
        double close = data[i];
        if (!isfinite(close)) {
            close_head = 0;
            close_count = 0;
            distance_head = 0;
            distance_count = 0;
            ema = 0.0;
            have_ema = false;
            prev_sdo = 0.0;
            continue;
        }

        bool have_lag = close_count >= length1;
        double lag_close = have_lag ? close_ring[close_head] : CUDART_NAN;

        close_ring[close_head] = close;
        close_head += 1;
        if (close_head == length1) {
            close_head = 0;
        }
        if (close_count < length1) {
            close_count += 1;
        }
        if (!have_lag) {
            continue;
        }

        double distance = fabs(close - lag_close);
        if (distance_count < lookback_length) {
            distance_ring[distance_count] = distance;
            distance_count += 1;
        } else {
            distance_ring[distance_head] = distance;
            distance_head += 1;
            if (distance_head == lookback_length) {
                distance_head = 0;
            }
        }
        if (distance_count < lookback_length) {
            continue;
        }

        double hh = -CUDART_INF;
        double ll = CUDART_INF;
        for (int j = 0; j < lookback_length; ++j) {
            double v = distance_ring[j];
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        double spread = hh - ll;
        double distance_sto = fabs(spread) > tol ? (distance - ll) / spread * 100.0 : 0.0;
        double distance_d = 0.0;
        if (close > lag_close + tol) {
            distance_d = distance_sto;
        } else if (close + tol < lag_close) {
            distance_d = -distance_sto;
        }

        if (have_ema) {
            ema = alpha * distance_d + (1.0 - alpha) * ema;
        } else {
            ema = distance_d;
            have_ema = true;
        }

        double signal = 0.0;
        if (distance_d > ema || (prev_sdo < os_level && ema > os_level)) {
            signal = 1.0;
        } else if (distance_d < ema || (prev_sdo > ob_level && ema < ob_level)) {
            signal = -1.0;
        }
        prev_sdo = ema;
        row_oscillator[i] = ema;
        row_signal[i] = signal;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — stochastic_distance
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/stochastic_distance.rs:509
 *   `stochastic_distance_row_from_slice`, driving
 *   `StochasticDistanceStream::update` (:411) -> `push_distance` (:437).
 *
 * Column: the OSCILLATOR series. `compute_stochastic_distance_batch`
 *   (cpu_batch.rs:13310) accepts ONLY "oscillator" and "signal" and REJECTS
 *   "value", so there is no output_id == "value" column to inherit; the lane
 *   emits out.oscillator — the EMA of the signed stochastic distance — and a
 *   parity run must ask the CPU for output_id = "oscillator" explicitly.
 *
 * PERIOD-INVARIANT: that batch reads lookback_length (200), length1 (12),
 *   length2 (3), ob_level (40) and os_level (-40) and NEVER period
 *   (cpu_batch.rs:13326-13330). Five swept periods give five identical CPU
 *   columns, so this kernel emits five identical rows.
 *
 * FIRST-VALID IGNORED: the row function walks EVERY bar from index 0 and the
 *   stream RESETS its whole state on a non-finite close (:412-415 -> reset,
 *   :402), writing NaN for that bar. The caller's first-valid index is never
 *   consulted.
 *
 * Input: one price series, CPU source close (extract_slice_input(.., "close"),
 *   cpu_batch.rs:13316) — F64InputKind::CloseSlice.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. Three pieces of state cross
 *   bars: a length1-deep close ring (the lag term), two MONOTONE DEQUES over
 *   the last lookback_length distances, and the EMA recurrence. The deques are
 *   reproduced as deques rather than replaced by a window rescan so the
 *   pop/push order matches the CPU exactly; both hold at most
 *   lookback_length + 1 entries transiently (one push before the front is
 *   pruned), which is what the capacity below is sized for.
 *
 * ARITHMETIC taken verbatim:
 *   * the EMA is alpha * d + beta * ema (:479) — TWO products and one add,
 *     NOT a fused ema + alpha*(d - ema). Reproduced as written.
 *   * FLOAT_TOL is 1e-12 (:37) and is ALREADY an f64-sized tolerance — it is
 *     not an f32 epsilon copied across, so it is kept unchanged. The three
 *     comparisons that use it (:467, :472, :474) are reproduced with the same
 *     operand order, because close > lag_close + tol and
 *     close - lag_close > tol are not the same test in floating point.
 *   * alpha = 2/(length2+1), beta = 1 - alpha are formed once (:352-361).
 *
 * NaN semantics: the deque comparisons are <= / >= on values that are always
 *   finite here (close and lag_close are both finite before a distance is
 *   formed).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:13326-13330. Both bounds size per-thread arrays,
 * so they belong to the COMPILED kernel. */
#define NEO_SD_LOOKBACK_LENGTH 200
#define NEO_SD_LENGTH1          12
#define NEO_SD_LENGTH2           3
#define NEO_SD_OB_LEVEL        40.0
#define NEO_SD_OS_LEVEL      (-40.0)
/* One slot more than the window: push_distance pushes BEFORE it prunes the
 * front, so the deque is momentarily lookback_length + 1 deep. */
#define NEO_SD_DEQUE_CAP       (NEO_SD_LOOKBACK_LENGTH + 1)
#define NEO_SD_FLOAT_TOL       1e-12

extern "C" __global__
void stochastic_distance_neo_batch_f64(const double* __restrict__ prices,
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

    const int    window  = NEO_SD_LOOKBACK_LENGTH;
    const int    length1 = NEO_SD_LENGTH1;
    const double alpha   = 2.0 / ((double)NEO_SD_LENGTH2 + 1.0);
    const double beta    = 1.0 - alpha;

    /* resolve_params (:340-349) refuses either window longer than the data. */
    if (window > n || length1 > n) return;

    double close_ring[NEO_SD_LENGTH1];
    int    close_head = 0, close_count = 0;

    double max_val[NEO_SD_DEQUE_CAP]; int max_idx[NEO_SD_DEQUE_CAP];
    double min_val[NEO_SD_DEQUE_CAP]; int min_idx[NEO_SD_DEQUE_CAP];
    int max_lo = 0, max_len = 0, min_lo = 0, min_len = 0;

    int    dist_index = 0;
    double ema = 0.0;
    bool   have_ema = false;

    for (int i = 0; i < n; ++i) {
        const double close = prices[i];

        if (!isfinite(close)) {
            /* reset (:402) rebuilds the stream from scratch. */
            close_head = 0; close_count = 0;
            max_lo = 0; max_len = 0; min_lo = 0; min_len = 0;
            dist_index = 0; ema = 0.0; have_ema = false;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const bool   have_lag  = (close_count >= length1);
        const double lag_close = have_lag ? close_ring[close_head] : 0.0;

        close_ring[close_head] = close;
        ++close_head;
        if (close_head == length1) close_head = 0;
        if (close_count < length1) ++close_count;

        if (!have_lag) { o[i] = NEO_F64_NAN; continue; }   /* the lag_close? */

        const double distance = fabs(close - lag_close);
        const int    idx      = dist_index;
        ++dist_index;

        /* max deque: pop_back while back value <= distance */
        while (max_len > 0 &&
               max_val[(max_lo + max_len - 1) % NEO_SD_DEQUE_CAP] <= distance) {
            --max_len;
        }
        max_val[(max_lo + max_len) % NEO_SD_DEQUE_CAP] = distance;
        max_idx[(max_lo + max_len) % NEO_SD_DEQUE_CAP] = idx;
        ++max_len;

        /* min deque: pop_back while back value >= distance */
        while (min_len > 0 &&
               min_val[(min_lo + min_len - 1) % NEO_SD_DEQUE_CAP] >= distance) {
            --min_len;
        }
        min_val[(min_lo + min_len) % NEO_SD_DEQUE_CAP] = distance;
        min_idx[(min_lo + min_len) % NEO_SD_DEQUE_CAP] = idx;
        ++min_len;

        const int cutoff = (idx >= window - 1) ? (idx - (window - 1)) : 0;
        while (max_len > 0 && max_idx[max_lo] < cutoff) {
            max_lo = (max_lo + 1) % NEO_SD_DEQUE_CAP; --max_len;
        }
        while (min_len > 0 && min_idx[min_lo] < cutoff) {
            min_lo = (min_lo + 1) % NEO_SD_DEQUE_CAP; --min_len;
        }

        if (idx + 1 < window) { o[i] = NEO_F64_NAN; continue; }

        const double hh = (max_len > 0) ? max_val[max_lo] : distance;
        const double ll = (min_len > 0) ? min_val[min_lo] : distance;
        const double spread = hh - ll;
        const double distance_sto =
            (fabs(spread) > NEO_SD_FLOAT_TOL) ? ((distance - ll) / spread * 100.0) : 0.0;

        double distance_d;
        if (close > lag_close + NEO_SD_FLOAT_TOL)        distance_d =  distance_sto;
        else if (close + NEO_SD_FLOAT_TOL < lag_close)   distance_d = -distance_sto;
        else                                             distance_d =  0.0;

        if (have_ema) {
            ema = alpha * distance_d + beta * ema;
        } else {
            ema = distance_d;
            have_ema = true;
        }

        o[i] = ema;
    }
}
