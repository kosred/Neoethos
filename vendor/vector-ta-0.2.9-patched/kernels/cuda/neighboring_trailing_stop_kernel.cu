#include <cmath>
#include <cstddef>

namespace {
__device__ inline int lower_bound_sorted(const double* sorted, int size, double value) {
    int left = 0;
    int right = size;
    while (left < right) {
        const int mid = left + ((right - left) >> 1);
        if (sorted[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ inline void insert_sorted(double* sorted, int* size, double value, int capacity) {
    if (*size >= capacity) {
        return;
    }
    const int idx = lower_bound_sorted(sorted, *size, value);
    for (int i = *size; i > idx; --i) {
        sorted[i] = sorted[i - 1];
    }
    sorted[idx] = value;
    *size += 1;
}

__device__ inline void remove_sorted_once(double* sorted, int* size, double value) {
    const int idx = lower_bound_sorted(sorted, *size, value);
    if (idx < *size && sorted[idx] == value) {
        for (int i = idx; i + 1 < *size; ++i) {
            sorted[i] = sorted[i + 1];
        }
        *size -= 1;
    }
}

__device__ inline double percentile_sorted_slice(
    const double* sorted,
    int size,
    double percentile
) {
    if (size <= 0) {
        return NAN;
    }
    if (size == 1) {
        return sorted[0];
    }

    const double idx = static_cast<double>(size - 1) * percentile / 100.0;
    const int i1 = static_cast<int>(floor(idx));
    const int i2 = static_cast<int>(ceil(idx));
    if (i1 == i2) {
        return sorted[i1];
    }
    const double v1 = sorted[i1];
    const double v2 = sorted[i2];
    return v1 + (v2 - v1) * (idx - static_cast<double>(i1));
}

__device__ inline void sma_reset(
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    *count = 0;
    *head = 0;
    *sum = 0.0;
    for (int i = 0; i < period; ++i) {
        buffer[i] = 0.0;
    }
}

__device__ inline double sma_update_ignore_nan(
    double value,
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    if (isfinite(value)) {
        if (*count < period) {
            buffer[*count] = value;
            *sum += value;
            *count += 1;
        } else {
            const double old = buffer[*head];
            buffer[*head] = value;
            *sum += value - old;
            *head += 1;
            if (*head == period) {
                *head = 0;
            }
        }
    }

    return *count == period ? (*sum / static_cast<double>(period)) : NAN;
}
}

extern "C" __global__ void neighboring_trailing_stop_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* buffer_sizes,
    const int* ks,
    const double* percentiles,
    const int* smooths,
    int rows,
    int max_buffer_size,
    int max_smooth,
    double* out_trailing_stop,
    double* out_bullish_band,
    double* out_bearish_band,
    double* out_direction,
    double* out_discovery_bull,
    double* out_discovery_bear,
    double* price_buffers,
    double* sorted_buffers,
    double* bull_sma_buffers,
    double* bear_sma_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int buffer_size = buffer_sizes[row];
    const int k = ks[row];
    const double percentile = percentiles[row];
    const int smooth = smooths[row];

    double* row_trailing_stop =
        out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_band =
        out_bullish_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_band =
        out_bearish_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction = out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_discovery_bull =
        out_discovery_bull + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_discovery_bear =
        out_discovery_bear + static_cast<size_t>(row) * static_cast<size_t>(len);

    double* row_price_buffer =
        price_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_buffer_size);
    double* row_sorted_buffer =
        sorted_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_buffer_size);
    double* row_bull_sma =
        bull_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_smooth);
    double* row_bear_sma =
        bear_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_smooth);

    for (int i = 0; i < len; ++i) {
        row_trailing_stop[i] = NAN;
        row_bullish_band[i] = NAN;
        row_bearish_band[i] = NAN;
        row_direction[i] = NAN;
        row_discovery_bull[i] = NAN;
        row_discovery_bear[i] = NAN;
    }

    if (buffer_size < 100 || buffer_size > max_buffer_size || k < 5 || !isfinite(percentile)
        || percentile < 1.0 || percentile > 99.0 || smooth <= 0 || smooth > max_smooth) {
        return;
    }

    int price_count = 0;
    int price_head = 0;
    int sorted_count = 0;
    int bull_sma_count = 0;
    int bull_sma_head = 0;
    double bull_sma_sum = 0.0;
    int bear_sma_count = 0;
    int bear_sma_head = 0;
    double bear_sma_sum = 0.0;
    int direction = 0;
    double trailing_stop = NAN;

    sma_reset(&bull_sma_count, &bull_sma_head, &bull_sma_sum, row_bull_sma, smooth);
    sma_reset(&bear_sma_count, &bear_sma_head, &bear_sma_sum, row_bear_sma, smooth);

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            price_count = 0;
            price_head = 0;
            sorted_count = 0;
            direction = 0;
            trailing_stop = NAN;
            sma_reset(&bull_sma_count, &bull_sma_head, &bull_sma_sum, row_bull_sma, smooth);
            sma_reset(&bear_sma_count, &bear_sma_head, &bear_sma_sum, row_bear_sma, smooth);
            continue;
        }

        double bear_val = NAN;
        double bull_val = NAN;
        const int size = sorted_count;

        if (size > 5) {
            const int idx = lower_bound_sorted(row_sorted_buffer, size, c);
            const int bear_start = idx > k ? (idx - k) : 0;
            if (idx > bear_start) {
                bear_val = percentile_sorted_slice(
                    row_sorted_buffer + bear_start,
                    idx - bear_start,
                    100.0 - percentile
                );
            }

            if (size > 0) {
                const int bull_end = min(idx + k, size - 1);
                if (bull_end > idx) {
                    bull_val = percentile_sorted_slice(
                        row_sorted_buffer + idx,
                        bull_end - idx + 1,
                        percentile
                    );
                }
            }
        }

        if (price_count < buffer_size) {
            const int insert_idx = (price_head + price_count) % buffer_size;
            row_price_buffer[insert_idx] = c;
            price_count += 1;
        } else {
            const double old = row_price_buffer[price_head];
            remove_sorted_once(row_sorted_buffer, &sorted_count, old);
            row_price_buffer[price_head] = c;
            price_head += 1;
            if (price_head == buffer_size) {
                price_head = 0;
            }
        }
        insert_sorted(row_sorted_buffer, &sorted_count, c, max_buffer_size);

        const double final_bull = sma_update_ignore_nan(
            bull_val,
            &bull_sma_count,
            &bull_sma_head,
            &bull_sma_sum,
            row_bull_sma,
            smooth
        );
        const double final_bear = sma_update_ignore_nan(
            bear_val,
            &bear_sma_count,
            &bear_sma_head,
            &bear_sma_sum,
            row_bear_sma,
            smooth
        );

        const bool discovery_bull = !isfinite(bull_val) && isfinite(bear_val);
        const bool discovery_bear = !isfinite(bear_val) && isfinite(bull_val);

        const int prev_direction = direction;
        if (discovery_bull) {
            direction = 1;
        } else if (discovery_bear) {
            direction = -1;
        }

        if (direction > prev_direction) {
            trailing_stop = isfinite(final_bear) ? final_bear : l;
        } else if (direction < prev_direction) {
            trailing_stop = isfinite(final_bull) ? final_bull : h;
        }

        if (direction == 1) {
            const double candidate = isfinite(final_bear) ? final_bear : trailing_stop;
            trailing_stop = isfinite(trailing_stop) ? fmax(trailing_stop, candidate) : candidate;
        } else if (direction == -1) {
            const double candidate = isfinite(final_bull) ? final_bull : trailing_stop;
            trailing_stop = isfinite(trailing_stop) ? fmin(trailing_stop, candidate) : candidate;
        }

        row_trailing_stop[i] = trailing_stop;
        row_bullish_band[i] = final_bull;
        row_bearish_band[i] = final_bear;
        row_direction[i] = static_cast<double>(direction);
        row_discovery_bull[i] = discovery_bull ? 1.0 : 0.0;
        row_discovery_bear[i] = discovery_bear ? 1.0 : 0.0;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// neighboring_trailing_stop_batch_f64 above is genuine double-in/double-out,
// but it takes 21 parameters and writes SIX output matrices plus four scratch
// matrices. The f64 lane launches exactly one shape -- (inputs, n, periods,
// n_combos, first_valid, out) -- and allocates ONE output matrix, so that
// entry point cannot be called from it. This is the lane-shaped twin.
//
// CPU REFERENCE
//   src/indicators/neighboring_trailing_stop.rs:858
//     neighboring_trailing_stop_with_kernel -> :718
//     neighboring_trailing_stop_default_row (the path every lane row takes,
//     because every parameter below is the CPU default and :605-616 selects the
//     default row when all four match).
//   Helpers transliterated: lower_bound :336, percentile_sorted_slice :368,
//   NeighborSma5::update :672.
//
// THE COLUMN THIS EMITS is trailing_stop, which is what output_id == "value"
// resolves to (cpu_batch.rs:9043 -- "trailing_stop" | "value").
//
// PERIOD-INVARIANT. compute_neighboring_trailing_stop_batch
// (cpu_batch.rs:9019-9025) reads buffer_size, k, percentile and smooth and
// NEVER period, so five swept periods give five identical CPU columns and this
// kernel emits five identical rows. All four CPU defaults are pinned below.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- the stop is
// a ratchet carried across bars, the 200-deep price window and its sorted twin
// are carried, the two 5-deep smoothing rings are carried, and a non-finite bar
// CLEARS all of it (:745-758).
//
// TWO FIDELITY FIXES relative to the 21-parameter kernel above, both found by
// reading the CPU line by line:
//
//  1. discovery_bull is bull_val.is_nan() && bear_val.is_finite() (:800). The
//     kernel above writes !isfinite(bull_val), which is ALSO true for an
//     infinity. percentile_sorted_slice can return an infinity from an infinite
//     price, so the two disagree on real data. This one uses isnan.
//
//  2. NeighborSma5::update (:678-681) does sum += value; sum -= old; -- TWO
//     roundings. The kernel above writes *sum += value - old -- ONE. Same value
//     in exact arithmetic, a different double. This one keeps the CPU's two.
//
// NaN SEMANTICS: the CPU's stop.max(candidate) / stop.min(candidate) are
// f64::max / f64::min, which return the NON-NaN operand. fmax/fmin are exactly
// that, and are what is used below -- an if-chain would let a NaN survive into
// the carried stop and poison every later bar.
//
// FIRST VALID IS NOT READ: neighboring_trailing_stop_with_kernel allocates with
// alloc_with_nan_prefix(len, 0) (:872-877) and the row writes every index, so
// there is no warmup prefix to agree about. The lane row declares
// F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, fmax/fmin/isfinite/isnan, no f32-suffixed
// math function, no fast-math intrinsic. The file is listed in
// F64_LANE_SOURCES, so it is never compiled with --use_fast_math. The only
// tolerance in the CPU reference is FLOAT_TOL = 1e-12 (:35), used by the
// batch-combo comparison and NOT by the row -- so no epsilon appears here.
// ---------------------------------------------------------------------------

#define NTS_NEO_BUFFER 200
#define NTS_NEO_K 50
#define NTS_NEO_PCT 90.0
#define NTS_NEO_SMOOTH 5

// The exact quiet NaN the CPU writes: f64::from_bits(0x7ff8_0000_0000_0000).
__device__ __forceinline__ double nts_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// NeighborSma5::update (neighboring_trailing_stop.rs:672-706), rounding for
// rounding: on a full ring sum += value THEN sum -= old, two roundings, in that
// order.
__device__ __forceinline__ double nts_neo_sma5(
    double value,
    double* ring,
    int* head,
    int* count,
    double* sum
) {
    if (isfinite(value)) {
        if (*count == NTS_NEO_SMOOTH) {
            const double old = ring[*head];
            ring[*head] = value;
            *sum += value;
            *sum -= old;
        } else {
            *count += 1;
            ring[*head] = value;
            *sum += value;
        }
        *head += 1;
        if (*head == NTS_NEO_SMOOTH) {
            *head = 0;
        }
    }
    return (*count == NTS_NEO_SMOOTH)
        ? (*sum / static_cast<double>(NTS_NEO_SMOOTH))
        : nts_neo_qnan();
}

extern "C" __global__ void neighboring_trailing_stop_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row_idx >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    const double qnan = nts_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    // The CPU's own bounds, not a truncation: price_ring is
    // [f64; DEFAULT_BUFFER_SIZE] (:733) and the two smoothing rings are
    // [f64; DEFAULT_SMOOTH] (:660). sorted never exceeds the price window.
    double price_ring[NTS_NEO_BUFFER];
    double sorted[NTS_NEO_BUFFER];
    double bull_ring[NTS_NEO_SMOOTH];
    double bear_ring[NTS_NEO_SMOOTH];

    int price_head = 0;
    int price_count = 0;
    int sorted_count = 0;
    int bull_head = 0;
    int bull_count = 0;
    double bull_sum = 0.0;
    int bear_head = 0;
    int bear_count = 0;
    double bear_sum = 0.0;
    int direction = 0;
    double stop = qnan;

    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            // :745-758 -- clear everything and emit NaN. row[i] already holds
            // the CPU's quiet NaN from the prefill.
            price_head = 0;
            price_count = 0;
            sorted_count = 0;
            bull_head = 0;
            bull_count = 0;
            bull_sum = 0.0;
            bear_head = 0;
            bear_count = 0;
            bear_sum = 0.0;
            direction = 0;
            stop = qnan;
            continue;
        }

        // :764-778 -- the neighbourhood percentiles are read from the window as
        // it stood BEFORE this bar's close joins it.
        double bear_val = qnan;
        double bull_val = qnan;
        const int size = sorted_count;
        if (size > 5) {
            const int idx = lower_bound_sorted(sorted, size, c);
            const int bear_start = idx > NTS_NEO_K ? (idx - NTS_NEO_K) : 0;
            if (idx > bear_start) {
                bear_val = percentile_sorted_slice(
                    sorted + bear_start,
                    idx - bear_start,
                    100.0 - NTS_NEO_PCT
                );
            }
            const int bull_end =
                (idx + NTS_NEO_K) < (size - 1) ? (idx + NTS_NEO_K) : (size - 1);
            if (bull_end > idx) {
                bull_val = percentile_sorted_slice(
                    sorted + idx,
                    bull_end - idx + 1,
                    NTS_NEO_PCT
                );
            }
        }

        // :780-796 -- evict then insert, always writing at price_head and
        // always advancing it, exactly as the CPU does.
        if (price_count == NTS_NEO_BUFFER) {
            remove_sorted_once(sorted, &sorted_count, price_ring[price_head]);
        } else {
            price_count += 1;
        }
        price_ring[price_head] = c;
        price_head += 1;
        if (price_head == NTS_NEO_BUFFER) {
            price_head = 0;
        }
        insert_sorted(sorted, &sorted_count, c, NTS_NEO_BUFFER);

        const double final_bull =
            nts_neo_sma5(bull_val, bull_ring, &bull_head, &bull_count, &bull_sum);
        const double final_bear =
            nts_neo_sma5(bear_val, bear_ring, &bear_head, &bear_count, &bear_sum);

        // :800-801 -- is_nan(), NOT !is_finite(). An infinite percentile is NOT
        // a discovery for the CPU.
        const bool discovery_bull = isnan(bull_val) && isfinite(bear_val);
        const bool discovery_bear = isnan(bear_val) && isfinite(bull_val);

        const int prev_direction = direction;
        if (discovery_bull) {
            direction = 1;
        } else if (discovery_bear) {
            direction = -1;
        }

        if (direction > prev_direction) {
            stop = isfinite(final_bear) ? final_bear : l;
        } else if (direction < prev_direction) {
            stop = isfinite(final_bull) ? final_bull : h;
        }

        if (direction == 1) {
            const double candidate = isfinite(final_bear) ? final_bear : stop;
            stop = isfinite(stop) ? fmax(stop, candidate) : candidate;
        } else if (direction == -1) {
            const double candidate = isfinite(final_bull) ? final_bull : stop;
            stop = isfinite(stop) ? fmin(stop, candidate) : candidate;
        }

        row[i] = stop;
    }
}
