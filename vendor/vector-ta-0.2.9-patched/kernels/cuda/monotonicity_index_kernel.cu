#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline void copy_ordered_window(
    double* dst,
    const double* ring,
    int len,
    int head
) {
    if (head == 0) {
        for (int i = 0; i < len; ++i) {
            dst[i] = ring[i];
        }
        return;
    }

    int tail = len - head;
    for (int i = 0; i < tail; ++i) {
        dst[i] = ring[head + i];
    }
    for (int i = 0; i < head; ++i) {
        dst[tail + i] = ring[i];
    }
}

__device__ inline void pava_fit(
    const double* data,
    int len,
    bool non_decreasing,
    double* pool_vals,
    int* pool_weights,
    double* mse,
    int* pools,
    double* start_value,
    double* end_value
) {
    int pool_count = 0;
    for (int i = 0; i < len; ++i) {
        double current_pool = data[i];
        int current_weight = 1;
        while (pool_count > 0) {
            double prev_pool = pool_vals[pool_count - 1];
            bool violation = non_decreasing ? (prev_pool > current_pool) : (prev_pool < current_pool);
            if (!violation) {
                break;
            }
            int prev_weight = pool_weights[pool_count - 1];
            double last_pool = pool_vals[pool_count - 1];
            pool_count -= 1;
            int combined_weight = prev_weight + current_weight;
            current_pool =
                (last_pool * static_cast<double>(prev_weight) +
                 current_pool * static_cast<double>(current_weight)) /
                static_cast<double>(combined_weight);
            current_weight = combined_weight;
        }
        pool_vals[pool_count] = current_pool;
        pool_weights[pool_count] = current_weight;
        pool_count += 1;
    }

    double total_error = 0.0;
    int idx = 0;
    for (int pool = 0; pool < pool_count; ++pool) {
        double pool_value = pool_vals[pool];
        int pool_weight = pool_weights[pool];
        for (int j = 0; j < pool_weight; ++j) {
            double delta = data[idx] - pool_value;
            total_error += delta * delta;
            idx += 1;
        }
    }

    *mse = total_error / static_cast<double>(len);
    *pools = pool_count;
    *start_value = pool_count > 0 ? pool_vals[0] : 0.0;
    *end_value = pool_count > 0 ? pool_vals[pool_count - 1] : 0.0;
}

extern "C" __global__ void monotonicity_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ index_smooths,
    const int* __restrict__ mode_flags,
    int n_combos,
    int max_length,
    int max_index_smooth,
    double* __restrict__ window_ring,
    double* __restrict__ window_copy,
    double* __restrict__ inc_pool_vals,
    int* __restrict__ inc_pool_weights,
    double* __restrict__ dec_pool_vals,
    int* __restrict__ dec_pool_weights,
    double* __restrict__ sma_buf,
    double* __restrict__ out_index,
    double* __restrict__ out_cumulative_mean,
    double* __restrict__ out_upper_bound
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0 || max_index_smooth <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int index_smooth = index_smooths[combo_idx];
    int mode_flag = mode_flags[combo_idx];

    double* row_index = out_index + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_cumulative =
        out_cumulative_mean + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_upper =
        out_upper_bound + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* ring =
        window_ring + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* ordered =
        window_copy + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* inc_vals =
        inc_pool_vals + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    int* inc_weights =
        inc_pool_weights + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* dec_vals =
        dec_pool_vals + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    int* dec_weights =
        dec_pool_weights + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* sma =
        sma_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_index_smooth);

    for (int i = 0; i < len; ++i) {
        row_index[i] = CUDART_NAN;
        row_cumulative[i] = CUDART_NAN;
        row_upper[i] = CUDART_NAN;
    }

    if (length < 2 || length > max_length || index_smooth <= 0 || index_smooth > max_index_smooth) {
        return;
    }

    int window_next = 0;
    int window_len = 0;
    int sma_next = 0;
    int sma_len = 0;
    double sma_sum = 0.0;
    double cumulative_sum = 0.0;
    int cumulative_count = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            window_next = 0;
            window_len = 0;
            sma_next = 0;
            sma_len = 0;
            sma_sum = 0.0;
            cumulative_sum = 0.0;
            cumulative_count = 0;
            continue;
        }

        ring[window_next] = value;
        window_next += 1;
        if (window_next == length) {
            window_next = 0;
        }
        if (window_len < length) {
            window_len += 1;
        }
        if (window_len < length) {
            continue;
        }

        copy_ordered_window(ordered, ring, length, window_next);

        double inc_mse = 0.0;
        double dec_mse = 0.0;
        int inc_pools = 0;
        int dec_pools = 0;
        double inc_start = 0.0;
        double inc_end = 0.0;
        double dec_start = 0.0;
        double dec_end = 0.0;
        pava_fit(
            ordered,
            length,
            true,
            inc_vals,
            inc_weights,
            &inc_mse,
            &inc_pools,
            &inc_start,
            &inc_end
        );
        pava_fit(
            ordered,
            length,
            false,
            dec_vals,
            dec_weights,
            &dec_mse,
            &dec_pools,
            &dec_start,
            &dec_end
        );

        bool use_inc = inc_mse < dec_mse;
        double raw_index = 0.0;
        if (mode_flag == 0) {
            double start_value = use_inc ? inc_start : dec_start;
            double end_value = use_inc ? inc_end : dec_end;
            double price_path = 0.0;
            for (int j = 1; j < length; ++j) {
                price_path += fabs(ordered[j] - ordered[j - 1]);
            }
            if (price_path > 0.0) {
                raw_index = fabs(end_value - start_value) / price_path * 100.0;
            }
        } else {
            int pools = use_inc ? inc_pools : dec_pools;
            raw_index =
                (static_cast<double>(pools > 0 ? pools - 1 : 0) / static_cast<double>(length - 1)) *
                100.0;
        }

        if (sma_len == index_smooth) {
            sma_sum -= sma[sma_next];
        } else {
            sma_len += 1;
        }
        sma[sma_next] = raw_index;
        sma_sum += raw_index;
        sma_next += 1;
        if (sma_next == index_smooth) {
            sma_next = 0;
        }
        if (sma_len < index_smooth) {
            continue;
        }

        double smoothed = sma_sum / static_cast<double>(index_smooth);
        cumulative_sum += smoothed;
        cumulative_count += 1;
        double cumulative_mean = cumulative_sum / static_cast<double>(cumulative_count);
        row_index[i] = smoothed;
        row_cumulative[i] = cumulative_mean;
        row_upper[i] = cumulative_mean * 2.0;
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic  : MonotonicityIndexStream::update,
//                   src/indicators/monotonicity_index.rs:498-516, driven by
//                   monotonicity_index_row_from_slice (:639-670) -- the only
//                   compute body this indicator has.
//   * isotonic fit: PavaScratch::fit, :293-344 (pool-adjacent-violators).
//   * raw index   : compute_raw_index, :578-605.
//   * rings       : RollingWindow (:354-405) and RollingSma (:414-452).
//   * refusals    : monotonicity_index_prepare, :608-637.
//   * emitted col : `index`. compute_monotonicity_index_batch
//                   (cpu_batch.rs:9410) accepts ONLY "index",
//                   "cumulative_mean" and "upper_bound" -- there is no "value"
//                   arm (:9450-9457) -- so a parity run must ask the CPU for
//                   "index" explicitly.
//   * PERIOD-INVARIANT: the batch reads `source`, `length` (20), `mode`
//                   ("efficiency") and `index_smooth` (5) and never `period`
//                   (cpu_batch.rs:9420-9423).
//   * FIRST-VALID IGNORED: the stream walks from index 0 and RESETS on any
//                   non-finite bar (:499-502), and row_from_slice writes NaN
//                   into EVERY slot the stream does not fill (:664-668), so no
//                   first-valid index reaches the output at all.
//
// MODE: `efficiency` is the CPU default (cpu_batch.rs:9422), so the
// `complexity` branch of compute_raw_index is unreachable from this lane and is
// not implemented. Implementing it would be a second indicator, not a second
// code path.
//
// ORDER STATISTIC / ISOTONIC REGRESSION ON THE CARD: PAVA is a stack machine
// over a 20-bar window, which is a per-thread local array of 20 doubles and 20
// ints -- the "order statistic -> per-column selection" shape from the brief.
// It is O(length) amortised per bar, so the whole thread body stays O(n *
// length) exactly as the host does.
//
// SEQUENTIAL, one thread per column: the price ring, the 5-deep SMA ring and
// the cumulative (sum, count) pair all carry across bars.
// ===========================================================================

#define MONI_NEO_LENGTH 20
#define MONI_NEO_SMOOTH 5

static __device__ __forceinline__ double moni_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// PavaScratch::fit, :293-344. `non_decreasing` picks the direction; the pools
// are a stack, popped and merged while the monotonicity is violated.
struct Moni_Fit { double mse; int pools; double start_value; double end_value; };

static __device__ __forceinline__ Moni_Fit moni_neo_fit(const double* data, int len,
                                                        bool non_decreasing) {
  double pool_vals[MONI_NEO_LENGTH];
  int pool_w[MONI_NEO_LENGTH];
  int top = 0;

  for (int k = 0; k < len; ++k) {
    double current_pool = data[k];
    int current_weight = 1;
    while (top > 0) {
      const double prev_pool = pool_vals[top - 1];
      const bool violation = non_decreasing ? (prev_pool > current_pool)
                                            : (prev_pool < current_pool);
      if (!violation) break;
      const int prev_weight = pool_w[top - 1];
      const double last_pool = pool_vals[top - 1];
      top -= 1;
      const int combined = prev_weight + current_weight;
      current_pool = (last_pool * (double)prev_weight + current_pool * (double)current_weight)
                     / (double)combined;
      current_weight = combined;
    }
    pool_vals[top] = current_pool;
    pool_w[top] = current_weight;
    top += 1;
  }

  double total_error = 0.0;
  int idx = 0;
  for (int p = 0; p < top; ++p) {
    for (int r = 0; r < pool_w[p]; ++r) {
      const double delta = data[idx] - pool_vals[p];
      total_error += delta * delta;
      idx += 1;
    }
  }

  Moni_Fit f;
  f.mse = total_error / (double)len;
  f.pools = top;
  f.start_value = (top > 0) ? pool_vals[0] : 0.0;
  f.end_value = (top > 0) ? pool_vals[top - 1] : 0.0;
  return f;
}

// compute_raw_index, :578-605, Efficiency branch.
static __device__ __forceinline__ double moni_neo_raw(const double* data, int len) {
  const Moni_Fit inc = moni_neo_fit(data, len, true);
  const Moni_Fit dec = moni_neo_fit(data, len, false);
  const Moni_Fit best = (inc.mse < dec.mse) ? inc : dec;

  double price_path = 0.0;
  for (int k = 1; k < len; ++k) price_path += fabs(data[k] - data[k - 1]);
  if (price_path > 0.0) {
    return fabs(best.end_value - best.start_value) / price_path * 100.0;
  }
  return 0.0;
}

extern "C" __global__
void monotonicity_index_neo_batch_f64(const double* __restrict__ prices,
                                      int n,
                                      const int* __restrict__ periods,
                                      int n_combos,
                                      int first_valid,
                                      double* __restrict__ out) {
  const int combo = blockIdx.x * blockDim.x + threadIdx.x;
  if (combo >= n_combos) return;
  (void)periods;      // PERIOD-INVARIANT -- see the header.
  (void)first_valid;  // FIRST-VALID IGNORED -- see the header.

  if (n <= 0) return;
  double* __restrict__ row = out + (size_t)combo * (size_t)n;
  const double nn = moni_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int length = MONI_NEO_LENGTH;
  const int smooth = MONI_NEO_SMOOTH;
  const int needed_valid = length + smooth - 1;  // resolve_params, :561-564

  // monotonicity_index_prepare, :608-637.
  int first = n;
  for (int i = 0; i < n; ++i) { if (isfinite(prices[i])) { first = i; break; } }
  if (first >= n) return;
  int best_run = 0, run = 0;
  for (int i = 0; i < n; ++i) {
    if (isfinite(prices[i])) { run += 1; if (run > best_run) best_run = run; }
    else run = 0;
  }
  if (best_run < needed_valid) return;

  double win[MONI_NEO_LENGTH];
  double ordered[MONI_NEO_LENGTH];
  double sma_buf[MONI_NEO_SMOOTH];
  int win_next = 0, win_len = 0;
  int sma_next = 0, sma_len = 0;
  double sma_sum = 0.0;
  double cum_sum = 0.0;
  int cum_count = 0;
  for (int k = 0; k < length; ++k) win[k] = 0.0;
  for (int k = 0; k < smooth; ++k) sma_buf[k] = 0.0;

  for (int i = 0; i < n; ++i) {
    const double value = prices[i];
    if (!isfinite(value)) {
      // reset(), :484-490.
      win_next = 0; win_len = 0;
      sma_next = 0; sma_len = 0; sma_sum = 0.0;
      cum_sum = 0.0; cum_count = 0;
      continue;
    }

    // RollingWindow::push, :380-390.
    win[win_next] = value;
    win_next += 1;
    if (win_next == length) win_next = 0;
    if (win_len < length) win_len += 1;

    if (win_len < length) continue;

    // copy_to_vec, :393-405 -- CHRONOLOGICAL order: [next..] then [..next].
    int m = 0;
    for (int k = win_next; k < length; ++k) ordered[m++] = win[k];
    for (int k = 0; k < win_next; ++k) ordered[m++] = win[k];

    const double raw_index = moni_neo_raw(ordered, length);

    // RollingSma::update, :434-452.
    if (sma_len == smooth) sma_sum -= sma_buf[sma_next];
    else sma_len += 1;
    sma_buf[sma_next] = raw_index;
    sma_sum += raw_index;
    sma_next += 1;
    if (sma_next == smooth) sma_next = 0;
    if (sma_len < smooth) continue;

    const double smoothed = sma_sum / (double)smooth;
    // The CPU updates the cumulative pair here (:512-514) and this lane emits
    // `index`, so the mean is state the row does not carry.
    cum_sum += smoothed;
    cum_count += 1;
    row[i] = smoothed;
  }
}
