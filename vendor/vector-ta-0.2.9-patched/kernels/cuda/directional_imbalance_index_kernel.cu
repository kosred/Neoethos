#include <cmath>
#include <cstdint>

// One operation authority for the dynamic six-output ABI and the preserved
// primary-up ABI. Each CUDA thread owns one complete parameter row and its
// four runtime rings; non-finite high/low pairs reset the complete state at
// the same bar as DirectionalImbalanceIndexStream::update.
__device__ __forceinline__ void directional_imbalance_index_row_f64(
    const double* high,
    const double* low,
    int len,
    int length,
    int period,
    double* row_high_ring,
    double* row_low_ring,
    double* row_up_hits,
    double* row_down_hits,
    double* row_out_up,
    double* row_out_down,
    double* row_out_bulls,
    double* row_out_bears,
    double* row_out_upper,
    double* row_out_lower
) {
    const int window_cap = length + 1;
    int price_head = 0;
    int price_count = 0;
    int hit_head = 0;
    int hit_count = 0;
    double up_sum = 0.0;
    double down_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            price_head = 0;
            price_count = 0;
            hit_head = 0;
            hit_count = 0;
            up_sum = 0.0;
            down_sum = 0.0;
            if (row_out_up != nullptr) {
                row_out_up[i] = NAN;
            }
            if (row_out_down != nullptr) {
                row_out_down[i] = NAN;
            }
            if (row_out_bulls != nullptr) {
                row_out_bulls[i] = NAN;
            }
            if (row_out_bears != nullptr) {
                row_out_bears[i] = NAN;
            }
            if (row_out_upper != nullptr) {
                row_out_upper[i] = NAN;
            }
            if (row_out_lower != nullptr) {
                row_out_lower[i] = NAN;
            }
            continue;
        }

        row_high_ring[price_head] = h;
        row_low_ring[price_head] = l;
        price_head += 1;
        if (price_head == window_cap) {
            price_head = 0;
        }
        if (price_count < window_cap) {
            price_count += 1;
        }

        double upper = -INFINITY;
        double lower = INFINITY;
        const bool window_full = price_count == window_cap;
        for (int j = 0; j < price_count; ++j) {
            int ring_index = window_full ? price_head + j : j;
            if (ring_index >= window_cap) {
                ring_index -= window_cap;
            }
            const double window_high = row_high_ring[ring_index];
            const double window_low = row_low_ring[ring_index];
            // The scalar monotone deques pop equal extrema, so the newest
            // equal value owns the output bit pattern. Chronological scan plus
            // inclusive comparison preserves that signed-zero behavior.
            if (window_high >= upper) {
                upper = window_high;
            }
            if (window_low <= lower) {
                lower = window_low;
            }
        }

        const double up_hit = h == upper ? 1.0 : 0.0;
        const double down_hit = l == lower ? 1.0 : 0.0;
        if (hit_count == period) {
            up_sum -= row_up_hits[hit_head];
            down_sum -= row_down_hits[hit_head];
        } else {
            hit_count += 1;
        }
        row_up_hits[hit_head] = up_hit;
        row_down_hits[hit_head] = down_hit;
        up_sum += up_hit;
        down_sum += down_hit;
        hit_head += 1;
        if (hit_head == period) {
            hit_head = 0;
        }

        const double total = up_sum + down_sum;
        if (row_out_up != nullptr) {
            row_out_up[i] = up_sum;
        }
        if (row_out_down != nullptr) {
            row_out_down[i] = down_sum;
        }
        if (row_out_bulls != nullptr) {
            row_out_bulls[i] = total > 0.0 ? (up_sum / total) * 100.0 : NAN;
        }
        if (row_out_bears != nullptr) {
            row_out_bears[i] = total > 0.0 ? (down_sum / total) * 100.0 : NAN;
        }
        if (row_out_upper != nullptr) {
            row_out_upper[i] = upper;
        }
        if (row_out_lower != nullptr) {
            row_out_lower[i] = lower;
        }
    }
}

extern "C" __global__ void directional_imbalance_index_batch_f64(
    const double* high,
    const double* low,
    int len,
    const int* lengths,
    const int* periods,
    int rows,
    int max_window,
    int max_period,
    double* high_ring,
    double* low_ring,
    double* up_hits_ring,
    double* down_hits_ring,
    double* out_up,
    double* out_down,
    double* out_bulls,
    double* out_bears,
    double* out_upper,
    double* out_lower
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int period = periods[row];
    const int window_cap = length + 1;
    if (length <= 0 || period <= 0 || window_cap > max_window || period > max_period) {
        return;
    }

    const size_t row_index = static_cast<size_t>(row);
    const size_t output_offset = row_index * static_cast<size_t>(len);
    directional_imbalance_index_row_f64(
        high,
        low,
        len,
        length,
        period,
        high_ring + row_index * static_cast<size_t>(max_window),
        low_ring + row_index * static_cast<size_t>(max_window),
        up_hits_ring + row_index * static_cast<size_t>(max_period),
        down_hits_ring + row_index * static_cast<size_t>(max_period),
        out_up + output_offset,
        out_down + output_offset,
        out_bulls + output_offset,
        out_bears + output_offset,
        out_upper + output_offset,
        out_lower + output_offset
    );
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE -- preserved primary `up` ABI
//
// The generic primary route remains length=10 with one dynamic period per row.
// It now delegates to the exact same state authority as the canonical six-
// output route. The fixed local bounds remain fail-closed at the host ABI.
// ---------------------------------------------------------------------------

#define NEO_DII_LENGTH 10
#define NEO_DII_MAX_WINDOW 64
#define NEO_DII_MAX_PERIOD 512

extern "C" __global__ void directional_imbalance_index_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) {
        return;
    }
    (void)first_valid;

    const int length = NEO_DII_LENGTH;
    const int period = periods[combo];
    const int window_cap = length + 1;
    if (length <= 0 || period <= 0 || window_cap > NEO_DII_MAX_WINDOW ||
        period > NEO_DII_MAX_PERIOD) {
        return;
    }

    double high_ring[NEO_DII_MAX_WINDOW];
    double low_ring[NEO_DII_MAX_WINDOW];
    double up_hits[NEO_DII_MAX_PERIOD];
    double down_hits[NEO_DII_MAX_PERIOD];
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    directional_imbalance_index_row_f64(
        high,
        low,
        n,
        length,
        period,
        high_ring,
        low_ring,
        up_hits,
        down_hits,
        row,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
}
