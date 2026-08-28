#include <cmath>
#include <cstddef>

namespace {

struct PivotState {
    int confirm_idx;
    double value;
};

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline void shift_left(int* deque, int count) {
    for (int i = 1; i < count; ++i) {
        deque[i - 1] = deque[i];
    }
}

__device__ inline void compute_segment_offsets_abs(
    const double* open,
    const double* close,
    int seg_start,
    int start_idx,
    int end_idx,
    double start_value,
    double end_value,
    double* up_offset,
    double* dn_offset
) {
    if (end_idx <= start_idx) {
        *up_offset = 0.0;
        *dn_offset = 0.0;
        return;
    }

    if (end_idx == start_idx + 1) {
        const int abs_idx = seg_start + end_idx;
        const double top = fmax(open[abs_idx], close[abs_idx]);
        const double bottom = fmin(open[abs_idx], close[abs_idx]);
        *up_offset = fmax(top - end_value, 0.0);
        *dn_offset = fmax(end_value - bottom, 0.0);
        return;
    }

    double max_diff_up = 0.0;
    double max_diff_dn = 0.0;
    const double denom = static_cast<double>(end_idx - start_idx - 1);
    const double span = end_value - start_value;

    for (int idx = start_idx + 1; idx <= end_idx; ++idx) {
        const double j = static_cast<double>(idx - start_idx - 1);
        const double point = start_value + (j / denom) * span;
        const int abs_idx = seg_start + idx;
        const double top = fmax(open[abs_idx], close[abs_idx]);
        const double bottom = fmin(open[abs_idx], close[abs_idx]);
        max_diff_up = fmax(max_diff_up, top - point);
        max_diff_dn = fmax(max_diff_dn, point - bottom);
    }

    *up_offset = fmax(max_diff_up, 0.0);
    *dn_offset = fmax(max_diff_dn, 0.0);
}

__device__ inline void fill_segment_abs(
    double* middle,
    double* upper,
    double* lower,
    int seg_start,
    int start_idx,
    int end_idx,
    double start_value,
    double end_value,
    double up_offset,
    double dn_offset
) {
    if (end_idx < start_idx) {
        return;
    }

    if (start_idx == end_idx) {
        const int abs_idx = seg_start + start_idx;
        middle[abs_idx] = start_value;
        upper[abs_idx] = start_value + up_offset;
        lower[abs_idx] = start_value - dn_offset;
        return;
    }

    const double denom = static_cast<double>(end_idx - start_idx);
    const double span = end_value - start_value;
    for (int idx = start_idx; idx <= end_idx; ++idx) {
        const double t = static_cast<double>(idx - start_idx) / denom;
        const double value = start_value + t * span;
        const int abs_idx = seg_start + idx;
        middle[abs_idx] = value;
        upper[abs_idx] = value + up_offset;
        lower[abs_idx] = value - dn_offset;
    }
}

__device__ void compute_run_abs(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    int seg_start,
    int seg_len,
    int length,
    bool extend,
    int* max_deque,
    int* min_deque,
    double* middle,
    double* upper,
    double* lower
) {
    if (seg_len <= length) {
        return;
    }

    int max_count = 0;
    int min_count = 0;
    int os = 0;
    bool has_last_top = false;
    bool has_last_bottom = false;
    PivotState last_top{0, NAN};
    PivotState last_bottom{0, NAN};

    for (int idx = 0; idx < seg_len; ++idx) {
        const double current_close = close[seg_start + idx];

        while (max_count > 0 && close[seg_start + max_deque[max_count - 1]] <= current_close) {
            max_count -= 1;
        }
        max_deque[max_count] = idx;
        max_count += 1;

        while (min_count > 0 && close[seg_start + min_deque[min_count - 1]] >= current_close) {
            min_count -= 1;
        }
        min_deque[min_count] = idx;
        min_count += 1;

        if (idx < length) {
            continue;
        }

        const int window_start = idx + 1 - length;
        while (max_count > 0 && max_deque[0] < window_start) {
            shift_left(max_deque, max_count);
            max_count -= 1;
        }
        while (min_count > 0 && min_deque[0] < window_start) {
            shift_left(min_deque, min_count);
            min_count -= 1;
        }

        const int candidate = idx - length;
        const double upper_close = close[seg_start + max_deque[0]];
        const double lower_close = close[seg_start + min_deque[0]];
        const int prev_os = os;
        const double candidate_close = close[seg_start + candidate];

        if (candidate_close > upper_close) {
            os = 0;
        } else if (candidate_close < lower_close) {
            os = 1;
        }

        if (os == 1 && prev_os != 1) {
            const int end_idx = candidate;
            const double end_value = low[seg_start + end_idx];
            if (has_last_top) {
                const int start_idx = last_top.confirm_idx - length;
                double up_offset = 0.0;
                double dn_offset = 0.0;
                compute_segment_offsets_abs(
                    open,
                    close,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_top.value,
                    end_value,
                    &up_offset,
                    &dn_offset
                );
                fill_segment_abs(
                    middle,
                    upper,
                    lower,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_top.value,
                    end_value,
                    up_offset,
                    dn_offset
                );
            }
            last_bottom.confirm_idx = idx;
            last_bottom.value = end_value;
            has_last_bottom = true;
        }

        if (os == 0 && prev_os != 0) {
            const int end_idx = candidate;
            const double end_value = high[seg_start + end_idx];
            if (has_last_bottom) {
                const int start_idx = last_bottom.confirm_idx - length;
                double up_offset = 0.0;
                double dn_offset = 0.0;
                compute_segment_offsets_abs(
                    open,
                    close,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_bottom.value,
                    end_value,
                    &up_offset,
                    &dn_offset
                );
                fill_segment_abs(
                    middle,
                    upper,
                    lower,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_bottom.value,
                    end_value,
                    up_offset,
                    dn_offset
                );
            }
            last_top.confirm_idx = idx;
            last_top.value = end_value;
            has_last_top = true;
        }
    }

    if (!extend) {
        return;
    }

    const int end_idx = seg_len - 1;
    const double end_value = close[seg_start + end_idx];
    if (os == 1) {
        if (has_last_bottom) {
            const int start_idx = last_bottom.confirm_idx - length;
            double up_offset = 0.0;
            double dn_offset = 0.0;
            compute_segment_offsets_abs(
                open,
                close,
                seg_start,
                start_idx,
                end_idx,
                last_bottom.value,
                end_value,
                &up_offset,
                &dn_offset
            );
            fill_segment_abs(
                middle,
                upper,
                lower,
                seg_start,
                start_idx,
                end_idx,
                last_bottom.value,
                end_value,
                up_offset,
                dn_offset
            );
        }
    } else if (has_last_top) {
        const int start_idx = last_top.confirm_idx - length;
        double up_offset = 0.0;
        double dn_offset = 0.0;
        compute_segment_offsets_abs(
            open,
            close,
            seg_start,
            start_idx,
            end_idx,
            last_top.value,
            end_value,
            &up_offset,
            &dn_offset
        );
        fill_segment_abs(
            middle,
            upper,
            lower,
            seg_start,
            start_idx,
            end_idx,
            last_top.value,
            end_value,
            up_offset,
            dn_offset
        );
    }
}

}

extern "C" __global__ void zig_zag_channels_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ extends,
    int rows,
    int scratch_cap,
    int* __restrict__ scratch_buf,
    double* __restrict__ out_middle,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const bool extend = extends[row] != 0;

    double* row_middle = out_middle + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_middle[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
    }

    if (length <= 0 || length > scratch_cap) {
        return;
    }

    int* row_scratch = scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap * 2);
    int* max_deque = row_scratch;
    int* min_deque = row_scratch + scratch_cap;

    int idx = 0;
    while (idx < len) {
        while (idx < len && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        if (idx >= len) {
            break;
        }

        const int seg_start = idx;
        idx += 1;
        while (idx < len && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        const int seg_end = idx;
        const int seg_len = seg_end - seg_start;

        if (seg_len >= length + 1) {
            compute_run_abs(
                open,
                high,
                low,
                close,
                seg_start,
                seg_len,
                length,
                extend,
                max_deque,
                min_deque,
                row_middle,
                row_upper,
                row_lower
            );
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/zig_zag_channels.rs `compute_row` (:574-611)
 *   and `compute_run` (:396-573), with `fill_segment` (:362-394).
 *   Batch dispatcher: cpu_batch.rs:7339 -- output "value" is an ALIAS OF
 *   "middle" (:7346), so this kernel emits `middle`.
 *
 * WHY A SECOND ENTRY POINT: `zig_zag_channels_batch_f64` (:310) takes 13
 *   parameters and emits three series. The lane launches
 *   (open, high, low, close, n, periods, n_combos, first_valid, out).
 *
 * INPUT: open / high / low / close -- extract_ohlc_full_input
 *   (cpu_batch.rs:7343) -- F64InputKind::Ohlc4. OPEN is a genuine input to the
 *   VALIDITY gate: `is_valid_ohlc` (:240) tests it, so a bar with a non-finite
 *   open BREAKS the run (:589) even when the other three are fine, and the two
 *   segments either side are then computed independently.
 *
 * WHY `compute_segment_offsets` IS ABSENT: it returns `up_offset` and
 *   `dn_offset` (:329-359), and `fill_segment` uses them ONLY for `upper` and
 *   `lower` (:383-384, :390-391). The `middle` column is
 *   `start_value + t * span` and reads neither. Omitting the O(segment) offset
 *   scan is exact for this column, not an approximation -- it is not a
 *   shortcut that changes a rounding.
 *
 * FIRST-VALID IGNORED: `compute_row` scans for MAXIMAL VALID RUNS (:576-609)
 *   and computes each one independently with its own local bar numbering. A
 *   single global first-valid index cannot express that: the second run starts
 *   its pivot search from scratch.
 *
 * PERIOD-INVARIANT: the CPU batch reads `length` and `extend`
 *   (cpu_batch.rs:7365-7366) and never `period`. Both are pinned at the CPU
 *   defaults (100 and true), so every row of a sweep is byte-identical.
 *
 * SHAPE: ONE THREAD PER COLUMN. A zig-zag pivot state machine with two
 *   monotone deques over close, and segments that are filled RETROACTIVELY --
 *   `fill_segment` writes bars that were already visited. One thread owns the
 *   whole row, so the back-fill is a plain store; a bar-parallel launch could
 *   not express it at all.
 *
 * ARITHMETIC taken verbatim:
 *   * the segment interpolation is `start_value + t * span` with
 *     `t = (idx - start_idx) / (end_idx - start_idx)` (:388-389) -- the
 *     division is formed FIRST and then multiplied, TWO roundings, not a
 *     single fused blend.
 *   * a one-bar segment (`start_idx == end_idx`) writes `start_value` with no
 *     arithmetic at all (:379).
 *   * the deque comparisons are `<=` on the max side and `>=` on the min side
 *     (:418, :427), which decides WHICH of two equal closes stays -- the later
 *     one -- and that choice moves the pivot index.
 *
 * EPSILON: there is none. Every CPU guard is a strict order comparison
 *   between two closes; no tolerance is imported.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:7365-7366 */
#define NEO_ZZC_LENGTH 100
#define NEO_ZZC_EXTEND 1
/* Deque depth. The CPU pushes BEFORE it evicts the front (:407-427 then
 * :439-452), so the transient occupancy is `length + 1`; a [head, tail) ring
 * needs one slot beyond its maximum occupancy to tell full from empty. */
#define NEO_ZZC_CAP (NEO_ZZC_LENGTH + 2)

/* fill_segment (:362) -- MIDDLE ONLY. See the header for why the offsets are
 * absent. Indices are LOCAL to the run; `base` maps them onto the row. */
__device__ __forceinline__ void neo_zzc_fill_segment(double* __restrict__ row,
                                                     int base,
                                                     int start_idx,
                                                     int end_idx,
                                                     double start_value,
                                                     double end_value)
{
    if (end_idx < start_idx) return;
    if (start_idx == end_idx) { row[base + start_idx] = start_value; return; }

    const double denom = (double)(end_idx - start_idx);
    const double span  = end_value - start_value;
    for (int idx = start_idx; idx <= end_idx; ++idx) {
        const double t = (double)(idx - start_idx) / denom;
        row[base + idx] = start_value + t * span;
    }
}

extern "C" __global__
void zig_zag_channels_neo_batch_f64(const double* __restrict__ open,
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
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* per-run, not global -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = NEO_ZZC_LENGTH;
    /* validate_common refuses length == 0. */
    if (length <= 0 || length > NEO_ZZC_LENGTH) return;

    int max_q[NEO_ZZC_CAP];
    int min_q[NEO_ZZC_CAP];

    int scan = 0;
    while (scan < n) {
        /* compute_row (:576) -- skip to the next valid bar. */
        while (scan < n && !(isfinite(open[scan]) && isfinite(high[scan])
                             && isfinite(low[scan]) && isfinite(close[scan]))) {
            scan += 1;
        }
        if (scan >= n) break;

        const int seg_start = scan;
        scan += 1;
        while (scan < n && isfinite(open[scan]) && isfinite(high[scan])
                        && isfinite(low[scan]) && isfinite(close[scan])) {
            scan += 1;
        }
        const int seg_end = scan;          /* exclusive */
        const int run_n   = seg_end - seg_start;

        /* compute_row (:602) requires seg_end - seg_start >= length + 1, and
         * compute_run (:404) returns immediately when n <= length. */
        if (run_n < length + 1) continue;

        const int base = seg_start;

        int max_head = 0, max_tail = 0;    /* [head, tail) */
        int min_head = 0, min_tail = 0;
        int os = 0;
        bool   has_top = false, has_bottom = false;
        int    top_confirm = 0, bottom_confirm = 0;
        double top_value = 0.0, bottom_value = 0.0;

        for (int idx = 0; idx < run_n; ++idx) {
            const double current_close = close[base + idx];

            while (max_tail != max_head) {
                const int back = (max_tail == 0 ? NEO_ZZC_CAP : max_tail) - 1;
                if (close[base + max_q[back]] <= current_close) max_tail = back;
                else break;
            }
            max_q[max_tail] = idx;
            max_tail += 1; if (max_tail == NEO_ZZC_CAP) max_tail = 0;

            while (min_tail != min_head) {
                const int back = (min_tail == 0 ? NEO_ZZC_CAP : min_tail) - 1;
                if (close[base + min_q[back]] >= current_close) min_tail = back;
                else break;
            }
            min_q[min_tail] = idx;
            min_tail += 1; if (min_tail == NEO_ZZC_CAP) min_tail = 0;

            if (idx < length) continue;

            const int window_start = idx + 1 - length;
            while (max_tail != max_head && max_q[max_head] < window_start) {
                max_head += 1; if (max_head == NEO_ZZC_CAP) max_head = 0;
            }
            while (min_tail != min_head && min_q[min_head] < window_start) {
                min_head += 1; if (min_head == NEO_ZZC_CAP) min_head = 0;
            }

            const int    candidate      = idx - length;
            const double upper_close    = close[base + max_q[max_head]];
            const double lower_close    = close[base + min_q[min_head]];
            const int    prev_os        = os;
            const double candidate_close = close[base + candidate];

            if (candidate_close > upper_close)      os = 0;
            else if (candidate_close < lower_close) os = 1;

            if (os == 1 && prev_os != 1) {
                const int    end_idx   = candidate;
                const double end_value = low[base + end_idx];
                if (has_top) {
                    neo_zzc_fill_segment(o, base, top_confirm - length, end_idx,
                                         top_value, end_value);
                }
                has_bottom = true; bottom_confirm = idx; bottom_value = end_value;
            }

            if (os == 0 && prev_os != 0) {
                const int    end_idx   = candidate;
                const double end_value = high[base + end_idx];
                if (has_bottom) {
                    neo_zzc_fill_segment(o, base, bottom_confirm - length, end_idx,
                                         bottom_value, end_value);
                }
                has_top = true; top_confirm = idx; top_value = end_value;
            }
        }

#if NEO_ZZC_EXTEND
        /* compute_run (:534-571) -- the trailing leg to the last bar. */
        {
            const int    end_idx   = run_n - 1;
            const double end_value = close[base + end_idx];
            if (os == 1) {
                if (has_bottom) {
                    neo_zzc_fill_segment(o, base, bottom_confirm - length, end_idx,
                                         bottom_value, end_value);
                }
            } else if (has_top) {
                neo_zzc_fill_segment(o, base, top_confirm - length, end_idx,
                                     top_value, end_value);
            }
        }
#endif
    }
}
