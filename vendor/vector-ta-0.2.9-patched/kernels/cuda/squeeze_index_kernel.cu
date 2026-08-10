#include <cmath>
#include <cstdint>

static __device__ inline double psi_from_corr(double sum_x, double sum_x2, double weighted, int length) {
    double n = static_cast<double>(length);
    double sum_y = n * (n - 1.0) * 0.5;
    double sum_y2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    double denom_x = n * sum_x2 - sum_x * sum_x;
    double denom_y = n * sum_y2 - sum_y * sum_y;
    double denom = denom_x * denom_y;
    if (denom <= 0.0 || !isfinite(denom)) {
        return NAN;
    }
    double corr = (n * weighted - sum_x * sum_y) / sqrt(denom);
    return -50.0 * corr + 50.0;
}

extern "C" __global__ void squeeze_index_batch_f64(
    const double* data,
    int len,
    const double* convs,
    const int* lengths,
    int rows,
    int max_length,
    double* ring_vals_buf,
    int* ring_valid_buf,
    double* out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double conv = convs[row];
    int length = lengths[row];
    if (!(isfinite(conv)) || conv <= 1.0 || length <= 0 || length > max_length) {
        return;
    }

    const double nan = NAN;
    double* ring_vals = ring_vals_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* ring_valid = ring_valid_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    double max_state = 0.0;
    double min_state = 0.0;
    int head = 0;
    int filled = 0;
    int valid_count = 0;
    double sum_x = 0.0;
    double sum_x2 = 0.0;
    double weighted = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            max_state = 0.0;
            min_state = 0.0;
            head = 0;
            filled = 0;
            valid_count = 0;
            sum_x = 0.0;
            sum_x2 = 0.0;
            weighted = 0.0;
            row_out[i] = nan;
            continue;
        }

        double max_next = fmax(value, max_state - (max_state - value) / conv);
        double min_next = fmin(value, min_state + (value - min_state) / conv);
        max_state = max_next;
        min_state = min_next;

        double spread = max_next - min_next;
        bool is_valid = spread > 0.0;
        double push_value = is_valid ? log(spread) : 0.0;

        if (filled < length) {
            int pos = filled;
            ring_vals[pos] = push_value;
            ring_valid[pos] = is_valid ? 1 : 0;
            sum_x += push_value;
            sum_x2 += push_value * push_value;
            weighted += static_cast<double>(pos) * push_value;
            if (is_valid) {
                valid_count += 1;
            }
            filled += 1;
            if (filled < length) {
                row_out[i] = nan;
                continue;
            }
        } else {
            double old_value = ring_vals[head];
            int old_valid = ring_valid[head];
            double old_sum = sum_x;

            weighted = weighted - old_sum + old_value + static_cast<double>(length - 1) * push_value;
            sum_x = old_sum - old_value + push_value;
            sum_x2 = sum_x2 - old_value * old_value + push_value * push_value;
            valid_count = valid_count + (is_valid ? 1 : 0) - old_valid;

            ring_vals[head] = push_value;
            ring_valid[head] = is_valid ? 1 : 0;
            head += 1;
            if (head == length) {
                head = 0;
            }
        }

        row_out[i] = valid_count == length ? psi_from_corr(sum_x, sum_x2, weighted, length) : nan;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — squeeze_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/squeeze_index.rs:474
 *   `squeeze_index_row_from_slice`, driving `SqueezeIndexStream::update`
 *   (:304) -> `push_diff` (:324) and `psi_from_corr` (:396).
 *
 *   The CPU has TWO row paths and this kernel reproduces the GENERAL one
 *   (:485-493), not the `all_finite` fast path (:510). That is not a shortcut:
 *   the fast path bails (`return false`) the moment a spread is non-positive
 *   or a log is non-finite (:537-543) and the general path then runs anyway,
 *   and with finite data the two compute the SAME expressions in the SAME
 *   order — warm-up `sum_x += diff; sum_x2 += diff*diff; weighted += pos*diff`,
 *   steady state `weighted - old_sum + old_value + last_index_f*diff`. So the
 *   general path is the oracle for both.
 *
 * Column: output_id "value" — `compute_squeeze_index_batch` calls
 *   `expect_value_output` (cpu_batch.rs:13277) and returns `out.values`.
 *
 * PERIOD-INVARIANT: that batch reads `conv` (50.0) and `length` (20) and NEVER
 *   `period` (cpu_batch.rs:13286-13287), so five swept periods give five
 *   identical CPU columns and this kernel emits five identical rows.
 *
 * FIRST-VALID IGNORED: `squeeze_index_row_from_slice` pre-fills `length-1`
 *   NaNs and then walks EVERY bar from index 0, RESETTING the whole stream on
 *   a non-finite value (:305-308 -> `reset`, :286). The caller's first-valid
 *   index is never consulted and the mid-series reset is what reproduces it.
 *
 * Input: one price series, CPU source `close` (`extract_slice_input(..,
 *   "close")`, cpu_batch.rs:13278) — F64InputKind::CloseSlice.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `max_state`/`min_state` are a
 *   two-sided exponential envelope carried across bars, and the (sum, sumsq,
 *   weighted) triple is updated INCREMENTALLY — `weighted - old_sum +
 *   old_value + last_index_f*value` is a four-term chain whose value depends
 *   on the running `sum_x` of the previous bar, so there is no bar-parallel
 *   form that keeps the rounding.
 *
 * NaN semantics: `value.max(..)` / `value.min(..)` are `f64::max` / `f64::min`,
 *   which return the NON-NaN operand. `fmax`/`fmin` are used here for exactly
 *   that reason; an if-chain would let a NaN survive into the envelope and
 *   poison every later bar.
 *
 * The `1e-12`-class guard question does not arise: the only tolerance in the
 *   CPU path is the exact comparison `spread > 0.0` and `denom <= 0.0`, and
 *   both are reproduced as written.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:13286-13287. `length` bounds the per-thread ring,
 * so the bound belongs to the COMPILED kernel. */
#define NEO_SQUEEZE_INDEX_LENGTH 20
#define NEO_SQUEEZE_INDEX_CONV   50.0

extern "C" __global__
void squeeze_index_neo_batch_f64(const double* __restrict__ prices,
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

    const int    len      = NEO_SQUEEZE_INDEX_LENGTH;
    const double conv     = NEO_SQUEEZE_INDEX_CONV;
    const double inv_conv = 1.0 / conv;

    /* `resolve_params` (:406) refuses `length > data_len`. */
    if (len > n) return;

    const double length_f    = (double)len;
    const double sum_y       = length_f * (length_f - 1.0) * 0.5;
    const double sum_y2      = (length_f - 1.0) * length_f * (2.0 * length_f - 1.0) / 6.0;
    const double denom_y     = length_f * sum_y2 - sum_y * sum_y;
    const double last_idx_f  = (double)(len - 1);
    const int    warm        = len - 1;

    double ring_vals[NEO_SQUEEZE_INDEX_LENGTH];
    unsigned char ring_valid[NEO_SQUEEZE_INDEX_LENGTH];
    for (int k = 0; k < len; ++k) { ring_vals[k] = 0.0; ring_valid[k] = 0; }

    double max_state = 0.0, min_state = 0.0;
    double sum_x = 0.0, sum_x2 = 0.0, weighted = 0.0;
    int head = 0, filled = 0, valid_count = 0;

    for (int i = 0; i < n; ++i) {
        const double v = prices[i];

        if (!isfinite(v)) {
            /* `reset` (:286) */
            max_state = 0.0; min_state = 0.0;
            for (int k = 0; k < len; ++k) { ring_vals[k] = 0.0; ring_valid[k] = 0; }
            head = 0; filled = 0; valid_count = 0;
            sum_x = 0.0; sum_x2 = 0.0; weighted = 0.0;
            if (i >= warm) o[i] = NEO_F64_NAN;
            continue;
        }

        const double max_next = fmax(v, max_state - (max_state - v) * inv_conv);
        const double min_next = fmin(v, min_state + (v - min_state) * inv_conv);
        max_state = max_next;
        min_state = min_next;

        const double spread = max_next - min_next;
        const double diff   = (spread > 0.0) ? log(spread) : NEO_F64_NAN;

        const bool   is_valid = isfinite(diff);
        const double value    = is_valid ? diff : 0.0;

        bool emit = true;
        if (filled < len) {
            const int pos = filled;
            ring_vals[pos]  = value;
            ring_valid[pos] = is_valid ? 1 : 0;
            sum_x    += value;
            sum_x2   += value * value;
            weighted += (double)pos * value;
            if (is_valid) ++valid_count;
            ++filled;
            if (filled < len) emit = false;   /* `return None` */
        } else {
            const double old_value = ring_vals[head];
            const int    old_valid = (int)ring_valid[head];
            const double old_sum   = sum_x;

            weighted    = weighted - old_sum + old_value + last_idx_f * value;
            sum_x       = old_sum - old_value + value;
            sum_x2      = sum_x2 - old_value * old_value + value * value;
            valid_count = valid_count + (is_valid ? 1 : 0) - old_valid;

            ring_vals[head]  = value;
            ring_valid[head] = is_valid ? 1 : 0;
            ++head;
            if (head == len) head = 0;
        }

        if (!emit) {
            if (i >= warm) o[i] = NEO_F64_NAN;
            continue;
        }

        if (valid_count != len) {
            o[i] = NEO_F64_NAN;
            continue;
        }

        /* `psi_from_corr` (:396) */
        const double denom_x = length_f * sum_x2 - sum_x * sum_x;
        const double denom   = denom_x * denom_y;
        if (denom <= 0.0 || !isfinite(denom)) {
            o[i] = NEO_F64_NAN;
        } else {
            const double corr = (length_f * weighted - sum_x * sum_y) / sqrt(denom);
            o[i] = -50.0 * corr + 50.0;
        }
    }
}
