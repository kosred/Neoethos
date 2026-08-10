#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double trend_trigger_factor_value(
    double hh,
    double ll,
    double hist_hh,
    double hist_ll
) {
    double buy_power = hh - hist_ll;
    double sell_power = hist_hh - ll;
    double denom = buy_power + sell_power;
    if (isfinite(denom) && denom != 0.0) {
        return 200.0 * (buy_power - sell_power) / denom;
    }
    return CUDART_NAN;
}

extern "C" __global__ void trend_trigger_factor_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    int first_valid,
    const int* __restrict__ lengths,
    int n_combos,
    int max_length,
    int* __restrict__ maxq_idx,
    int* __restrict__ minq_idx,
    double* __restrict__ hh_history,
    double* __restrict__ ll_history,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    int* maxq = maxq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    int* minq = minq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* hh_hist =
        hh_history + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* ll_hist =
        ll_history + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length <= 0 || length > max_length) {
        return;
    }

    int warm = first_valid + length - 1;
    int maxq_head = 0;
    int maxq_size = 0;
    int minq_head = 0;
    int minq_size = 0;
    int hist_head = 0;
    int hist_size = 0;

    for (int i = first_valid; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            if (i >= warm) {
                row[i] = CUDART_NAN;
            }
            continue;
        }

        int window_start = i + 1 - length;
        if (window_start < first_valid) {
            window_start = first_valid;
        }

        while (maxq_size > 0) {
            int front_idx = maxq[maxq_head];
            if (front_idx < window_start) {
                maxq_head += 1;
                if (maxq_head == length) {
                    maxq_head = 0;
                }
                maxq_size -= 1;
            } else {
                break;
            }
        }

        while (minq_size > 0) {
            int front_idx = minq[minq_head];
            if (front_idx < window_start) {
                minq_head += 1;
                if (minq_head == length) {
                    minq_head = 0;
                }
                minq_size -= 1;
            } else {
                break;
            }
        }

        while (maxq_size > 0) {
            int back_pos = maxq_head + maxq_size - 1;
            if (back_pos >= length) {
                back_pos -= length;
            }
            int back_idx = maxq[back_pos];
            if (high[back_idx] <= h) {
                maxq_size -= 1;
            } else {
                break;
            }
        }
        int max_insert = maxq_head + maxq_size;
        if (max_insert >= length) {
            max_insert -= length;
        }
        maxq[max_insert] = i;
        maxq_size += 1;

        while (minq_size > 0) {
            int back_pos = minq_head + minq_size - 1;
            if (back_pos >= length) {
                back_pos -= length;
            }
            int back_idx = minq[back_pos];
            if (low[back_idx] >= l) {
                minq_size -= 1;
            } else {
                break;
            }
        }
        int min_insert = minq_head + minq_size;
        if (min_insert >= length) {
            min_insert -= length;
        }
        minq[min_insert] = i;
        minq_size += 1;

        if (i >= warm) {
            double hh = high[maxq[maxq_head]];
            double ll = low[minq[minq_head]];
            double hist_hh = hist_size == length ? hh_hist[hist_head] : 0.0;
            double hist_ll = hist_size == length ? ll_hist[hist_head] : 0.0;
            row[i] = trend_trigger_factor_value(hh, ll, hist_hh, hist_ll);

            int hist_insert = hist_head + hist_size;
            if (hist_insert >= length) {
                hist_insert -= length;
            }
            if (hist_size < length) {
                hh_hist[hist_insert] = hh;
                ll_hist[hist_insert] = ll;
                hist_size += 1;
            } else {
                hh_hist[hist_head] = hh;
                ll_hist[hist_head] = ll;
                hist_head += 1;
                if (hist_head == length) {
                    hist_head = 0;
                }
            }
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — trend_trigger_factor                        (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/trend_trigger_factor.rs
 *   :346 compute_trend_trigger_factor_into  <- the per-bar body
 *   :241 calc_ttf                           200*(buy-sell)/denom
 *   :252 IndexMonoQueue                     cap == window + 1
 *   :211 first_valid_high_low               BOTH finite, not merely non-NaN
 *   :418 trend_trigger_factor_with_kernel   warm = first + length - 1
 *
 * PERIOD-INVARIANT (cpu_batch.rs:10524 reads "length", default 15), so both
 * monotone deques and both history rings are the CPU fixed sizes and live in
 * per-thread arrays.
 *
 * FIRST-VALID: HighLowFinite. The CPU scan is is_finite on BOTH series at the
 * same index -- an INFINITE high is skipped, which a plain non-NaN scan would
 * accept and then feed to a subtraction.
 *
 * SEQUENTIAL, one thread per column: two monotone deques plus a length-deep
 * hh/ll history are carried across bars.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_TTF_LENGTH 15
#define NEO_TTF_QCAP   16   /* IndexMonoQueue::new(window) -> window + 1 */

extern "C" __global__
void trend_trigger_factor_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;

    if (len <= 0) return;

    const int length = NEO_TTF_LENGTH;

    // prepare(): length > len, or (len - first) < length, is an Err on the CPU,
    // i.e. no column at all -- so the device answer is a NaN column.
    if (first_valid < 0 || first_valid >= len || length > len ||
        (len - first_valid) < length) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warm = first_valid + length - 1;
    for (int i = 0; i < len && i < warm; ++i) o[i] = NEO_F64_NAN;
    if (warm >= len) return;

    int maxq[NEO_TTF_QCAP];
    int minq[NEO_TTF_QCAP];
    int max_head = 0, max_tail = 0, max_count = 0;
    int min_head = 0, min_tail = 0, min_count = 0;

    double hh_history[NEO_TTF_LENGTH];
    double ll_history[NEO_TTF_LENGTH];
    int hist_head = 0, hist_len = 0;

    for (int i = first_valid; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            if (i >= warm) o[i] = NEO_F64_NAN;
            continue;
        }

        int window_start = i + 1 - length;
        if (window_start < 0) window_start = 0;
        if (window_start < first_valid) window_start = first_valid;

        while (max_count > 0 && maxq[max_head] < window_start) {
            max_head += 1;
            if (max_head == NEO_TTF_QCAP) max_head = 0;
            max_count -= 1;
        }
        while (min_count > 0 && minq[min_head] < window_start) {
            min_head += 1;
            if (min_head == NEO_TTF_QCAP) min_head = 0;
            min_count -= 1;
        }

        while (max_count > 0) {
            const int back = (max_tail == 0) ? (NEO_TTF_QCAP - 1) : (max_tail - 1);
            if (high[maxq[back]] <= h) { max_tail = back; max_count -= 1; }
            else break;
        }
        maxq[max_tail] = i;
        max_tail += 1;
        if (max_tail == NEO_TTF_QCAP) max_tail = 0;
        max_count += 1;

        while (min_count > 0) {
            const int back = (min_tail == 0) ? (NEO_TTF_QCAP - 1) : (min_tail - 1);
            if (low[minq[back]] >= l) { min_tail = back; min_count -= 1; }
            else break;
        }
        minq[min_tail] = i;
        min_tail += 1;
        if (min_tail == NEO_TTF_QCAP) min_tail = 0;
        min_count += 1;

        if (i >= warm) {
            const double hh = high[maxq[max_head]];
            const double ll = low[minq[min_head]];
            const double hist_hh = (hist_len == length) ? hh_history[hist_head] : 0.0;
            const double hist_ll = (hist_len == length) ? ll_history[hist_head] : 0.0;

            const double buy_power  = hh - hist_ll;
            const double sell_power = hist_hh - ll;
            const double denom = buy_power + sell_power;
            o[i] = (isfinite(denom) && denom != 0.0)
                 ? (200.0 * (buy_power - sell_power) / denom)
                 : NEO_F64_NAN;

            if (hist_len < length) {
                int pos = hist_head + hist_len;
                if (pos >= length) pos -= length;
                hh_history[pos] = hh;
                ll_history[pos] = ll;
                hist_len += 1;
            } else {
                hh_history[hist_head] = hh;
                ll_history[hist_head] = ll;
                hist_head += 1;
                if (hist_head == length) hist_head = 0;
            }
        }
    }
}
