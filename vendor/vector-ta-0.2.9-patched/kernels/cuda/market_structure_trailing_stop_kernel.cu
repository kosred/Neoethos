#include <cmath>
#include <cstddef>

namespace {
constexpr int RESET_ON_CHOCH = 0;
constexpr int RESET_ON_ALL = 1;

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline bool is_pivot_high(
    const double* high,
    int center,
    int length
) {
    const double pivot = high[center];
    for (int idx = center - length; idx < center; ++idx) {
        if (high[idx] > pivot) {
            return false;
        }
    }
    for (int idx = center + 1; idx <= center + length; ++idx) {
        if (high[idx] >= pivot) {
            return false;
        }
    }
    return true;
}

__device__ inline bool is_pivot_low(
    const double* low,
    int center,
    int length
) {
    const double pivot = low[center];
    for (int idx = center - length; idx < center; ++idx) {
        if (low[idx] < pivot) {
            return false;
        }
    }
    for (int idx = center + 1; idx <= center + length; ++idx) {
        if (low[idx] <= pivot) {
            return false;
        }
    }
    return true;
}
}

extern "C" __global__ void market_structure_trailing_stop_batch_f64(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* lengths,
    const double* increment_factors,
    int rows,
    int reset_mode,
    double* out_trailing_stop,
    double* out_state,
    double* out_structure
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double increment_factor = increment_factors[row];

    double* row_trailing_stop =
        out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_structure = out_structure + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_trailing_stop[i] = NAN;
        row_state[i] = NAN;
        row_structure[i] = NAN;
    }

    if (length <= 0 || !isfinite(increment_factor) || increment_factor < 0.0) {
        return;
    }

    const int needed = 2 * length + 1;
    const double incr = increment_factor / 100.0;
    int idx = 0;
    while (idx < len) {
        while (idx < len && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        const int start = idx;
        while (idx < len && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        const int end = idx;
        if (end - start < needed) {
            continue;
        }

        double ph_y = NAN;
        int ph_x = 0;
        double pl_y = NAN;
        int pl_x = 0;
        bool ph_cross = false;
        bool pl_cross = false;
        double top = NAN;
        double btm = NAN;
        double max_close = NAN;
        double min_close = NAN;
        double ts = NAN;
        int os = 0;

        for (int local = 0; local < end - start; ++local) {
            const int i = start + local;
            int ms = 0;

            if (local >= 2 * length) {
                const int center = i - length;
                if (is_pivot_high(high, center, length)) {
                    ph_y = high[center];
                    ph_x = center;
                    ph_cross = false;
                }
                if (is_pivot_low(low, center, length)) {
                    pl_y = low[center];
                    pl_x = center;
                    pl_cross = false;
                }
            }

            const double c = close[i];

            if (isfinite(ph_y) && !ph_cross && c > ph_y) {
                ms = (reset_mode == RESET_ON_ALL || (reset_mode == RESET_ON_CHOCH && os == -1))
                    ? 1
                    : 0;
                ph_cross = true;
                os = 1;
                btm = low[i];
                for (int scan = i; scan > ph_x; --scan) {
                    btm = fmin(btm, low[scan]);
                }
            }

            if (isfinite(pl_y) && !pl_cross && c < pl_y) {
                ms = (reset_mode == RESET_ON_ALL || (reset_mode == RESET_ON_CHOCH && os == 1))
                    ? -1
                    : 0;
                pl_cross = true;
                os = -1;
                top = high[i];
                for (int scan = i; scan > pl_x; --scan) {
                    top = fmax(top, high[scan]);
                }
            }

            const double prev_max = max_close;
            const double prev_min = min_close;

            if (ms == 1) {
                max_close = c;
            } else if (ms == -1) {
                min_close = c;
            } else {
                if (isfinite(max_close) && c > max_close) {
                    max_close = c;
                }
                if (isfinite(min_close) && c < min_close) {
                    min_close = c;
                }
            }

            if (ms == 1) {
                ts = btm;
            } else if (ms == -1) {
                ts = top;
            } else if (os == 1) {
                ts = (isfinite(ts) && isfinite(max_close) && isfinite(prev_max))
                    ? (ts + (max_close - prev_max) * incr)
                    : NAN;
            } else {
                ts = (isfinite(ts) && isfinite(min_close) && isfinite(prev_min))
                    ? (ts + (min_close - prev_min) * incr)
                    : NAN;
            }

            row_trailing_stop[i] = ts;
            row_state[i] = static_cast<double>(os);
            row_structure[i] = static_cast<double>(ms);
        }
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2           market_structure_trailing_stop
 * ---------------------------------------------------------------------------
 * CPU reference: `compute_run`,
 * src/indicators/market_structure_trailing_stop.rs:486, driven by
 * `compute_row` (:604) from `market_structure_trailing_stop_with_kernel`
 * (:650), with the pivot tests `is_pivot_high` (:439) and `is_pivot_low`
 * (:463).
 *
 * `length` IS the swept parameter (cpu_batch.rs:7168, default 14).
 * `increment_factor` is 100.0 (:7169-7174) so `incr = 1.0` (:502), and
 * `reset_on` is "CHoCH" (:7175-7180). The lane emits the TRAILING STOP series,
 * which is what the dispatcher returns for `output_id` "value" as well as
 * "trailing_stop" (:7197-7201).
 *
 * WHY `compute_row` AND NOT `compute_run` DIRECTLY. The CPU picks between them
 * on `clean_tail` (:670-706): a frame with no interior gap runs `compute_run`
 * once over `high[first..]`, otherwise `compute_row` splits the frame into
 * maximal runs of valid OHLC bars and runs each. The two are not different
 * answers -- on a clean tail `compute_row` finds exactly ONE run, `[first, n)`,
 * and everything before it is already NaN under either prefix rule. So this
 * kernel implements the run-splitting form alone and is exact in both cases.
 *
 * EVERY INDEX INSIDE A RUN IS RUN-LOCAL on the CPU, because `compute_run` is
 * handed SUB-SLICES (:625-635). `ph_x`, `pl_x`, the `idx >= 2 * length` gate
 * and the backward `scan > ph_x` walks are therefore all relative to the run
 * start, and this kernel carries `base` to keep them so. Using frame-absolute
 * indices would let a pivot scan read across a gap the CPU never crosses.
 *
 * SEQUENTIAL, one thread per column. Eleven carried scalars -- `ph_y`, `ph_x`,
 * `pl_y`, `pl_x`, `ph_cross`, `pl_cross`, `top`, `btm`, `max_close`,
 * `min_close`, `ts` and `os` -- and `ts` is a RECURRENCE: `ts + (max_close -
 * prev_max) * incr` (:589) reads its own previous value. Nothing here is
 * bar-parallel.
 *
 * `btm.min(low[scan])` / `top.max(high[scan])` (:546, :562) are f64::min and
 * f64::max, which return the non-NaN operand, so they become `fmin`/`fmax`.
 * Inside a run every bar is finite, so the choice cannot bite today -- it is
 * written this way so that it still cannot bite if the validity predicate ever
 * loosens.
 *
 * NO ARRAYS AT ALL, so no `max_period` bound and NEVER-OOM by construction: the
 * pivot tests and the backward scans read the resident high/low series.
 *
 * `is_valid_ohlc` is all four `is_finite` (:279-281), OPEN included -- which is
 * why the lane declares `Ohlc4` and the `Ohlc4AllFinite` first-valid rule even
 * though `compute_run` itself reads only high, low and close.
 *
 * FIRST-VALID IS `Ignored` for the kernel's own start: `compute_row` scans from
 * index 0 (:615) and finds the runs itself. The declared `Ohlc4AllFinite` rule
 * exists so the lane reports the right warmup, not so the kernel skips bars.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* increment_factor 100.0 / 100.0, market_structure_trailing_stop.rs:502. */
#define NEO_MST_INCR 1.0

/* is_pivot_high, market_structure_trailing_stop.rs:439-460. `center` and the
 * scan bounds are RUN-LOCAL; `base` maps them into the resident series. */
__device__ __forceinline__
static bool neo_mst_pivot_high(const double* __restrict__ high, int base,
                               int center, int length)
{
    const double pivot = high[base + center];
    for (int idx = center - length; idx < center; ++idx) {
        if (high[base + idx] > pivot) return false;
    }
    for (int idx = center + 1; idx <= center + length; ++idx) {
        if (high[base + idx] >= pivot) return false;
    }
    return true;
}

/* is_pivot_low, market_structure_trailing_stop.rs:463-484. */
__device__ __forceinline__
static bool neo_mst_pivot_low(const double* __restrict__ low, int base,
                              int center, int length)
{
    const double pivot = low[base + center];
    for (int idx = center - length; idx < center; ++idx) {
        if (low[base + idx] < pivot) return false;
    }
    for (int idx = center + 1; idx <= center + length; ++idx) {
        if (low[base + idx] <= pivot) return false;
    }
    return true;
}

extern "C" __global__
void market_structure_trailing_stop_neo_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
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
    (void)first_valid;   /* compute_row scans from bar 0 -- see the header. */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = periods[combo];
    /* validate_common_stats, :410-412 -- length 0 errors. */
    if (length <= 0) return;
    /* :419-430 -- the longest valid run must reach 2 * length + 1. */
    const long long needed_ll = 2LL * (long long)length + 1LL;
    if (needed_ll > (long long)n) return;
    const int needed = (int)needed_ll;
    {
        int run = 0, best = 0;
        for (int i = 0; i < n; ++i) {
            if (isfinite(open[i]) && isfinite(high[i]) && isfinite(low[i])
                && isfinite(close[i])) {
                run += 1;
                if (run > best) best = run;
            } else {
                run = 0;
            }
        }
        if (best < needed) return;
    }

    /* compute_row, :604-643 -- maximal runs of valid OHLC bars. */
    int idx0 = 0;
    while (idx0 < n) {
        while (idx0 < n
               && !(isfinite(open[idx0]) && isfinite(high[idx0])
                    && isfinite(low[idx0]) && isfinite(close[idx0]))) {
            idx0 += 1;
        }
        const int base = idx0;
        while (idx0 < n
               && isfinite(open[idx0]) && isfinite(high[idx0])
               && isfinite(low[idx0]) && isfinite(close[idx0])) {
            idx0 += 1;
        }
        const int run_len = idx0 - base;
        if (run_len <= 0) continue;

        /* compute_run, :497-500 -- a short run stays NaN. */
        if (run_len < 2 * length + 1) continue;

        const double incr = NEO_MST_INCR;
        double ph_y = NEO_F64_NAN; int ph_x = 0;
        double pl_y = NEO_F64_NAN; int pl_x = 0;
        bool   ph_cross = false, pl_cross = false;
        double top = NEO_F64_NAN, btm = NEO_F64_NAN;
        double max_close = NEO_F64_NAN, min_close = NEO_F64_NAN;
        double ts = NEO_F64_NAN;
        int    os = 0;

        for (int idx = 0; idx < run_len; ++idx) {
            int ms = 0;

            if (idx >= 2 * length) {                       /* :519-531 */
                const int center = idx - length;
                if (neo_mst_pivot_high(high, base, center, length)) {
                    ph_y = high[base + center];
                    ph_x = center;
                    ph_cross = false;
                }
                if (neo_mst_pivot_low(low, base, center, length)) {
                    pl_y = low[base + center];
                    pl_x = center;
                    pl_cross = false;
                }
            }

            const double c = close[base + idx];

            if (isfinite(ph_y) && !ph_cross && c > ph_y) {  /* :535-549 */
                /* reset_on == CHoCH: only a flip out of a bearish state is a
                 * structure event (:536-540). */
                ms = (os == -1) ? 1 : 0;
                ph_cross = true;
                os = 1;
                btm = low[base + idx];
                for (int scan = idx; scan > ph_x; --scan) {
                    btm = fmin(btm, low[base + scan]);
                }
            }

            if (isfinite(pl_y) && !pl_cross && c < pl_y) {  /* :551-565 */
                ms = (os == 1) ? -1 : 0;
                pl_cross = true;
                os = -1;
                top = high[base + idx];
                for (int scan = idx; scan > pl_x; --scan) {
                    top = fmax(top, high[base + scan]);
                }
            }

            const double prev_max = max_close;             /* :567-568 */
            const double prev_min = min_close;

            if (ms == 1) {                                 /* :570-581 */
                max_close = c;
            } else if (ms == -1) {
                min_close = c;
            } else {
                if (isfinite(max_close) && c > max_close) max_close = c;
                if (isfinite(min_close) && c < min_close) min_close = c;
            }

            if (ms == 1) {                                 /* :583-597 */
                ts = btm;
            } else if (ms == -1) {
                ts = top;
            } else if (os == 1) {
                ts = (isfinite(ts) && isfinite(max_close) && isfinite(prev_max))
                   ? (ts + (max_close - prev_max) * incr)
                   : NEO_F64_NAN;
            } else {
                ts = (isfinite(ts) && isfinite(min_close) && isfinite(prev_min))
                   ? (ts + (min_close - prev_min) * incr)
                   : NEO_F64_NAN;
            }

            o[base + idx] = ts;                            /* :599 */
        }
    }
}
