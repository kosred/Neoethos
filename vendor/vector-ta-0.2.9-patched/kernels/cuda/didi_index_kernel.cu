#include <cmath>
#include <cstddef>

extern "C" __global__ void didi_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ short_lengths,
    const int* __restrict__ medium_lengths,
    const int* __restrict__ long_lengths,
    int rows,
    double* __restrict__ out_short,
    double* __restrict__ out_long,
    double* __restrict__ out_crossover,
    double* __restrict__ out_crossunder
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int short_length = short_lengths[row];
    int medium_length = medium_lengths[row];
    int long_length = long_lengths[row];

    double* row_out_short = out_short + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_long = out_long + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_crossover =
        out_crossover + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_crossunder =
        out_crossunder + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_short[i] = NAN;
        row_out_long[i] = NAN;
        row_out_crossover[i] = NAN;
        row_out_crossunder[i] = NAN;
    }

    if (short_length <= 0 || medium_length <= 0 || long_length <= 0) {
        return;
    }

    int needed = short_length;
    if (medium_length > needed) {
        needed = medium_length;
    }
    if (long_length > needed) {
        needed = long_length;
    }

    int run_length = 0;
    bool have_prev = false;
    double prev_short = NAN;
    double prev_long = NAN;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            run_length = 0;
            have_prev = false;
            prev_short = NAN;
            prev_long = NAN;
            continue;
        }

        run_length += 1;
        if (run_length < needed) {
            have_prev = false;
            continue;
        }

        double short_sum = 0.0;
        for (int j = i + 1 - short_length; j <= i; ++j) {
            short_sum += data[j];
        }

        double medium_sum = 0.0;
        for (int j = i + 1 - medium_length; j <= i; ++j) {
            medium_sum += data[j];
        }

        double long_sum = 0.0;
        for (int j = i + 1 - long_length; j <= i; ++j) {
            long_sum += data[j];
        }

        double medium_ma = medium_sum / static_cast<double>(medium_length);
        if (!isfinite(medium_ma) || medium_ma == 0.0) {
            have_prev = false;
            prev_short = NAN;
            prev_long = NAN;
            continue;
        }

        double short_value = (short_sum / static_cast<double>(short_length)) / medium_ma;
        double long_value = (long_sum / static_cast<double>(long_length)) / medium_ma;
        if (!isfinite(short_value) || !isfinite(long_value)) {
            have_prev = false;
            prev_short = NAN;
            prev_long = NAN;
            continue;
        }

        double crossover =
            (have_prev && short_value > long_value && prev_short <= prev_long) ? 1.0 : 0.0;
        double crossunder =
            (have_prev && short_value < long_value && prev_short >= prev_long) ? 1.0 : 0.0;

        row_out_short[i] = short_value;
        row_out_long[i] = long_value;
        row_out_crossover[i] = crossover;
        row_out_crossunder[i] = crossunder;

        prev_short = short_value;
        prev_long = long_value;
        have_prev = true;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE - didi_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/didi_index.rs:480
 *             `didi_index_selected_row_from_slice`, driving
 *             `DidiIndexStream::update` (:374) over three `SmaWindow`s
 *             (:291).
 *
 * COLUMN: `short`. cpu_batch.rs:8481 maps output_id "value" onto
 * `OutputField::Short`; `long`, `crossover` and `crossunder` are other
 * columns.
 *
 * PERIOD-INVARIANT. The CPU batch reads `short_length` (3),
 * `medium_length` (8) and `long_length` (20) and never `period`.
 *
 * FIRST-VALID IGNORED. The selected-row function builds a fresh stream and
 * walks from index 0. A non-finite bar calls `reset` (:376), which clears all
 * three SMA windows, so every warmup restarts after a hole.
 *
 * THE THREE SMAs ARE FED UNCONDITIONALLY (:380-382) and only then is the
 * readiness of all three tested. Order matters: pushing into the rings before
 * the early return is what keeps them aligned, and a version that returned
 * first would be one bar behind for the rest of the series.
 *
 * `have_prev = false` IS SET ON THREE DIFFERENT EARLY EXITS (:384, :390,
 * :397) while `prev_short` / `prev_long` are left holding their old values.
 * That asymmetry is reproduced exactly - it decides whether the next bar can
 * emit a crossover, and "cleaning it up" would change the flag columns.
 *
 * SMA ROUNDING: `SmaWindow::update` (:291) accumulates during fill with
 * `sum += value` and afterwards with `sum += value - old` - a running
 * accumulator, NOT a fresh window sum. Reproduced literally, because a
 * re-summed window would give a different double after enough bars.
 *
 * The divide `short_ma / medium_ma` is guarded by `medium_ma.is_finite() &&
 * medium_ma != 0.0` (:389) - a branch, not an epsilon, so no f64 tolerance is
 * invented here.
 *
 * SEQUENTIAL, one thread per combo column. All three rings are fixed-size
 * per-thread arrays at the CPU defaults (3, 8, 20 doubles).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define DIDI_NEO_SHORT   3
#define DIDI_NEO_MEDIUM  8
#define DIDI_NEO_LONG   20

extern "C" __global__
void didi_index_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods; (void)first_valid;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;

    double s_val[DIDI_NEO_SHORT];
    double m_val[DIDI_NEO_MEDIUM];
    double l_val[DIDI_NEO_LONG];
    int    s_idx = 0, s_cnt = 0;
    int    m_idx = 0, m_cnt = 0;
    int    l_idx = 0, l_cnt = 0;
    double s_sum = 0.0, m_sum = 0.0, l_sum = 0.0;
    #pragma unroll
    for (int k = 0; k < DIDI_NEO_SHORT;  ++k) s_val[k] = 0.0;
    #pragma unroll
    for (int k = 0; k < DIDI_NEO_MEDIUM; ++k) m_val[k] = 0.0;
    #pragma unroll
    for (int k = 0; k < DIDI_NEO_LONG;   ++k) l_val[k] = 0.0;

    double prev_short = NEO_F64_NAN, prev_long = NEO_F64_NAN;
    bool   have_prev = false;

    for (int i = 0; i < len; ++i) {
        const double v = data[i];
        if (!isfinite(v)) {
            s_idx = 0; s_cnt = 0; s_sum = 0.0;
            m_idx = 0; m_cnt = 0; m_sum = 0.0;
            l_idx = 0; l_cnt = 0; l_sum = 0.0;
            prev_short = NEO_F64_NAN; prev_long = NEO_F64_NAN;
            have_prev = false;
            o[i] = NEO_F64_NAN;
            continue;
        }

        /* SmaWindow::update, three times, ALWAYS (:380-382). */
        bool   s_ok, m_ok, l_ok;
        double s_ma, m_ma, l_ma;

        if (s_cnt < DIDI_NEO_SHORT) {
            s_val[s_idx] = v; s_sum += v; s_cnt += 1;
            s_idx += 1; if (s_idx == DIDI_NEO_SHORT) s_idx = 0;
            s_ok = (s_cnt == DIDI_NEO_SHORT);
        } else {
            const double old = s_val[s_idx];
            s_val[s_idx] = v; s_sum += v - old;
            s_idx += 1; if (s_idx == DIDI_NEO_SHORT) s_idx = 0;
            s_ok = true;
        }
        s_ma = s_sum / (double)DIDI_NEO_SHORT;

        if (m_cnt < DIDI_NEO_MEDIUM) {
            m_val[m_idx] = v; m_sum += v; m_cnt += 1;
            m_idx += 1; if (m_idx == DIDI_NEO_MEDIUM) m_idx = 0;
            m_ok = (m_cnt == DIDI_NEO_MEDIUM);
        } else {
            const double old = m_val[m_idx];
            m_val[m_idx] = v; m_sum += v - old;
            m_idx += 1; if (m_idx == DIDI_NEO_MEDIUM) m_idx = 0;
            m_ok = true;
        }
        m_ma = m_sum / (double)DIDI_NEO_MEDIUM;

        if (l_cnt < DIDI_NEO_LONG) {
            l_val[l_idx] = v; l_sum += v; l_cnt += 1;
            l_idx += 1; if (l_idx == DIDI_NEO_LONG) l_idx = 0;
            l_ok = (l_cnt == DIDI_NEO_LONG);
        } else {
            const double old = l_val[l_idx];
            l_val[l_idx] = v; l_sum += v - old;
            l_idx += 1; if (l_idx == DIDI_NEO_LONG) l_idx = 0;
            l_ok = true;
        }
        l_ma = l_sum / (double)DIDI_NEO_LONG;

        if (!s_ok || !m_ok || !l_ok) {
            have_prev = false;
            o[i] = NEO_F64_NAN;          /* update returns None */
            continue;
        }
        if (!isfinite(m_ma) || m_ma == 0.0) {
            have_prev = false;
            o[i] = NEO_F64_NAN;          /* Some((NaN, NaN, NaN, NaN)) */
            continue;
        }

        const double sh = s_ma / m_ma;
        const double lo = l_ma / m_ma;
        if (!isfinite(sh) || !isfinite(lo)) {
            have_prev = false;
            o[i] = NEO_F64_NAN;
            continue;
        }

        prev_short = sh;
        prev_long  = lo;
        have_prev  = true;
        o[i] = sh;
    }
}
