#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
constexpr double LN_2 = 0.69314718055994530942;
}

extern "C" __global__ void fractal_dimension_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length < 2 || length > len) {
        return;
    }

    double log_den = log(static_cast<double>(2 * length));
    for (int end = length - 1; end < len; ++end) {
        int start = end + 1 - length;
        bool valid = true;
        double low = 0.0;
        double high = 0.0;

        for (int i = start; i <= end; ++i) {
            double value = data[i];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            if (i == start || value < low) {
                low = value;
            }
            if (i == start || value > high) {
                high = value;
            }
        }

        if (!valid) {
            continue;
        }

        double range = high - low;
        double length_sum;
        if (!isfinite(range) || range <= 0.0) {
            length_sum = static_cast<double>(length - 1) / static_cast<double>(length);
        } else {
            double inv_n_sq = 1.0 / static_cast<double>(length * length);
            double prev = (data[start] - low) / range;
            double acc = 0.0;
            for (int i = start + 1; i <= end; ++i) {
                double cur = (data[i] - low) / range;
                double delta = cur - prev;
                acc += sqrt(delta * delta + inv_n_sq);
                prev = cur;
            }
            length_sum = acc;
        }

        if (!isfinite(length_sum) || length_sum <= 0.0) {
            continue;
        }

        row[end] = 1.0 + (log(length_sum) + LN_2) / log_den;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — fractal_dimension_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/fractal_dimension_index.rs:406 `compute_fdi_row`,
 *   with `path_length_from_window_precomputed` (:353) and
 *   `fdi_from_length_with_den` (:340).
 *
 * Column: output_id "value" (cpu_batch.rs:6771).
 *
 * PERIOD-INVARIANT: `compute_fractal_dimension_index_batch` (cpu_batch.rs:6758)
 *   reads `length` (default 30) and NEVER `period`.
 *
 * Shape: ONE THREAD PER COLUMN. The window extremes come from two MONOTONIC
 *   DEQUES over indices — an order statistic, so it selects an input value and
 *   has no accumulation order of its own — but the path length is a running sum
 *   over the window in ASCENDING bar order and that order IS load-bearing.
 *   The deques are fixed rings of `length` entries because a monotone deque
 *   never holds more than the window.
 *
 * The validity gate is a PREFIX COUNT of non-finite bars over the window
 *   (:459): one hole anywhere in the 30-bar window suppresses the value. A
 *   per-bar `isfinite` test would emit a number where the CPU emits NaN. The
 *   prefix is reproduced incrementally with a ring of the last `length + 1`
 *   counts, so the comparison is integer-exact.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* DEFAULT length for fractal_dimension_index (cpu_batch.rs:6758). The deques
 * and the prefix ring are per-thread arrays, so the bound belongs to the
 * compiled kernel. */
#define NEO_FDI_LENGTH 30

extern "C" __global__
void fractal_dimension_index_neo_batch_f64(const double* __restrict__ data,
                                           int n,
                                           const int* __restrict__ periods,
                                           int n_combos,
                                           int first_valid,
                                           double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;
    (void)first_valid;   /* compute_fdi_row walks from index 0 */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int L = NEO_FDI_LENGTH;
    if (n < L) return;

    const double inv_n_sq    = 1.0 / (double)((long long)L * (long long)L);
    const double flat_length = (double)(L - 1) / (double)L;
    const double ln_2_len    = log((double)(2 * L));
    const double LN_2        = 0.69314718055994530942;

    int minq[NEO_FDI_LENGTH]; int min_head = 0, min_len = 0;
    int maxq[NEO_FDI_LENGTH]; int max_head = 0, max_len = 0;

    /* invalid[i+1] - invalid[start], build_invalid_prefix (:326). Only the last
     * L + 1 entries are ever read, so a ring of that size is exact. */
    int inv_ring[NEO_FDI_LENGTH + 1];
    for (int i = 0; i <= L; ++i) inv_ring[i] = 0;
    int inv_running = 0;
    inv_ring[0] = 0;                       /* invalid[0] */

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        const bool ok = isfinite(value);

        if (ok) {
            while (min_len > 0) {
                const int back = minq[(min_head + min_len - 1) % NEO_FDI_LENGTH];
                if (data[back] <= value) break;
                --min_len;
            }
            minq[(min_head + min_len) % NEO_FDI_LENGTH] = i;
            ++min_len;

            while (max_len > 0) {
                const int back = maxq[(max_head + max_len - 1) % NEO_FDI_LENGTH];
                if (data[back] >= value) break;
                --max_len;
            }
            maxq[(max_head + max_len) % NEO_FDI_LENGTH] = i;
            ++max_len;
        }

        inv_running += ok ? 0 : 1;
        inv_ring[(i + 1) % (NEO_FDI_LENGTH + 1)] = inv_running;   /* invalid[i+1] */

        if (i + 1 < L) continue;

        const int start = i + 1 - L;
        while (min_len > 0 && minq[min_head] < start) {
            min_head = (min_head + 1) % NEO_FDI_LENGTH; --min_len;
        }
        while (max_len > 0 && maxq[max_head] < start) {
            max_head = (max_head + 1) % NEO_FDI_LENGTH; --max_len;
        }

        const int inv_at_start = inv_ring[start % (NEO_FDI_LENGTH + 1)];
        if (inv_running - inv_at_start != 0) continue;

        const double low  = data[minq[min_head]];
        const double high = data[maxq[max_head]];

        /* path_length_from_window_precomputed (:353) */
        double length_sum;
        const double range = high - low;
        if (!isfinite(range) || range <= 0.0) {
            length_sum = flat_length;
        } else {
            double prev = (data[start] - low) / range;
            double acc  = 0.0;
            for (int k = start + 1; k <= i; ++k) {
                const double cur   = (data[k] - low) / range;
                const double delta = cur - prev;
                acc += sqrt(delta * delta + inv_n_sq);
                prev = cur;
            }
            length_sum = acc;
        }

        o[i] = 1.0 + (log(length_sum) + LN_2) / ln_2_len;
    }
}
