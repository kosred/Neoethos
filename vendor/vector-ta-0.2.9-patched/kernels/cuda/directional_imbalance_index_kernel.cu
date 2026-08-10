#include <cmath>
#include <cstdint>

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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int period = periods[row];
    if (length <= 0 || period <= 0) {
        return;
    }

    const double nan = NAN;
    const double inf = 1.7976931348623157e308;
    int window_cap = length + 1;
    double* row_high_ring = high_ring + static_cast<size_t>(row) * static_cast<size_t>(max_window);
    double* row_low_ring = low_ring + static_cast<size_t>(row) * static_cast<size_t>(max_window);
    double* row_up_hits = up_hits_ring + static_cast<size_t>(row) * static_cast<size_t>(max_period);
    double* row_down_hits =
        down_hits_ring + static_cast<size_t>(row) * static_cast<size_t>(max_period);
    double* row_out_up = out_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_down = out_down + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bulls = out_bulls + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bears = out_bears + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);

    int price_head = 0;
    int price_count = 0;
    int hit_head = 0;
    int hit_count = 0;
    double up_sum = 0.0;
    double down_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            price_head = 0;
            price_count = 0;
            hit_head = 0;
            hit_count = 0;
            up_sum = 0.0;
            down_sum = 0.0;
            row_out_up[i] = nan;
            row_out_down[i] = nan;
            row_out_bulls[i] = nan;
            row_out_bears[i] = nan;
            row_out_upper[i] = nan;
            row_out_lower[i] = nan;
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

        double upper = -inf;
        double lower = inf;
        for (int j = 0; j < price_count; ++j) {
            double window_high = row_high_ring[j];
            double window_low = row_low_ring[j];
            if (window_high > upper) {
                upper = window_high;
            }
            if (window_low < lower) {
                lower = window_low;
            }
        }

        double up_hit = (h == upper) ? 1.0 : 0.0;
        double down_hit = (l == lower) ? 1.0 : 0.0;

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

        double total = up_sum + down_sum;
        row_out_up[i] = up_sum;
        row_out_down[i] = down_sum;
        row_out_bulls[i] = total > 0.0 ? (up_sum / total) * 100.0 : nan;
        row_out_bears[i] = total > 0.0 ? (down_sum / total) * 100.0 : nan;
        row_out_upper[i] = upper;
        row_out_lower[i] = lower;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/directional_imbalance_index.rs:647
// (`directional_imbalance_index_with_kernel`). The column this emits is `up`,
// which is what `output_id == "value"` resolves to
// (dispatch/cpu_batch.rs:7412-7413).
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- the value
// is a sliding sum of hit flags maintained with subtract-then-add
// (`up_sum -= ring[head]; up_sum += up_hit`), and the CPU RESETS the whole
// state (both rings, both sums, both counts) on any bar whose high or low is
// not finite. Neither the carried sum nor the reset history can be recovered
// bar-parallel.
//
// PERIOD-SWEPT, unlike most of this closer's kernels:
// `compute_directional_imbalance_index_batch` (cpu_batch.rs:7436-7437) reads a
// parameter literally named `period` (default 70) ALONGSIDE `length`
// (default 10). `periods[combo]` therefore binds to `period`, and `length`
// -- which the sweep does not touch -- is pinned at its CPU default.
//
// THE TWO RINGS ARE PER-THREAD, so their lengths are a property of THIS
// COMPILED KERNEL: an oversized period is refused BY NAME by the host
// (`F64Kernel::max_period` -> `DII_MAX_PERIOD`) rather than truncating the
// hit window into a different indicator that still returns plausible numbers.
//
// FIRST VALID IS NOT READ: the CPU has no warmup index at all -- it emits from
// bar 0 and restarts the state at every non-finite bar. The lane row declares
// `F64FirstValidRule::Ignored`.
//
// f64 END TO END: double literals throughout, no f32-suffixed math function,
// no fast-math intrinsic. `up_hit` / `down_hit` are compared with `==` exactly
// as the CPU does, because the hit test asks whether THIS bar's high IS the
// window extreme, not whether it is within a tolerance of it -- a tolerance
// here would count neighbouring bars as hits and change the indicator.
// ---------------------------------------------------------------------------

#define NEO_DII_LENGTH 10
// `window_cap` is `length + 1` and `length` is pinned at the CPU default 10,
// so 64 can never be approached; it is checked below rather than assumed.
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

    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int length = NEO_DII_LENGTH;
    const int period = periods[combo];
    if (length <= 0 || period <= 0) {
        return;
    }
    const int window_cap = length + 1;
    if (window_cap > NEO_DII_MAX_WINDOW || period > NEO_DII_MAX_PERIOD) {
        return;
    }

    double high_ring[NEO_DII_MAX_WINDOW];
    double low_ring[NEO_DII_MAX_WINDOW];
    double up_hits[NEO_DII_MAX_PERIOD];
    double down_hits[NEO_DII_MAX_PERIOD];

    int price_head = 0;
    int price_count = 0;
    int hit_head = 0;
    int hit_count = 0;
    double up_sum = 0.0;
    double down_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            price_head = 0;
            price_count = 0;
            hit_head = 0;
            hit_count = 0;
            up_sum = 0.0;
            down_sum = 0.0;
            row[i] = NAN;
            continue;
        }

        high_ring[price_head] = h;
        low_ring[price_head] = l;
        price_head += 1;
        if (price_head == window_cap) {
            price_head = 0;
        }
        if (price_count < window_cap) {
            price_count += 1;
        }

        // Every value in the ring is finite by construction (a non-finite bar
        // clears it above), so this comparison chain cannot meet a NaN and the
        // rule-4 hazard does not arise.
        double upper = -INFINITY;
        double lower = INFINITY;
        for (int j = 0; j < price_count; ++j) {
            const double wh = high_ring[j];
            const double wl = low_ring[j];
            if (wh > upper) {
                upper = wh;
            }
            if (wl < lower) {
                lower = wl;
            }
        }

        const double up_hit = (h == upper) ? 1.0 : 0.0;
        const double down_hit = (l == lower) ? 1.0 : 0.0;

        if (hit_count == period) {
            up_sum -= up_hits[hit_head];
            down_sum -= down_hits[hit_head];
        } else {
            hit_count += 1;
        }
        up_hits[hit_head] = up_hit;
        down_hits[hit_head] = down_hit;
        up_sum += up_hit;
        down_sum += down_hit;
        hit_head += 1;
        if (hit_head == period) {
            hit_head = 0;
        }

        row[i] = up_sum;
    }
}
