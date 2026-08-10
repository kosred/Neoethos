#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool market_meanness_valid_bar(double open, double close, int mode_flag) {
    if (mode_flag == 0) {
        return isfinite(close);
    }
    return isfinite(open) && isfinite(close);
}

__device__ inline double market_meanness_source_value(double open, double close, int mode_flag) {
    return mode_flag == 0 ? close : close - open;
}

__device__ inline void ordered_window_from_ring_device(
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

__device__ inline void insertion_sort_device(double* data, int len) {
    for (int i = 1; i < len; ++i) {
        double value = data[i];
        int j = i - 1;
        while (j >= 0 && data[j] > value) {
            data[j + 1] = data[j];
            --j;
        }
        data[j + 1] = value;
    }
}

__device__ inline double median_from_sorted_device(const double* data, int len) {
    int mid = len / 2;
    if ((len & 1) == 1) {
        return data[mid];
    }
    return 0.5 * (data[mid - 1] + data[mid]);
}

extern "C" __global__ void market_meanness_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ mode_flags,
    int n_combos,
    int max_length,
    double* __restrict__ source_ring,
    double* __restrict__ window_buf,
    double* __restrict__ median_buf,
    double* __restrict__ smoothing_buf,
    double* __restrict__ out_mmi,
    double* __restrict__ out_mmi_smoothed
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int mode_flag = mode_flags[combo_idx];
    double* row_mmi = out_mmi + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_smoothed =
        out_mmi_smoothed + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* source =
        source_ring + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* window =
        window_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* median =
        median_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* smooth =
        smoothing_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row_mmi[i] = CUDART_NAN;
        row_smoothed[i] = CUDART_NAN;
    }

    if (length < 6 || length > max_length) {
        return;
    }

    int source_count = 0;
    int source_head = 0;
    int smooth_count = 0;
    int smooth_head = 0;
    double smooth_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double open_value = open[i];
        double close_value = close[i];
        if (!market_meanness_valid_bar(open_value, close_value, mode_flag)) {
            source_count = 0;
            source_head = 0;
            smooth_count = 0;
            smooth_head = 0;
            smooth_sum = 0.0;
            continue;
        }

        double value = market_meanness_source_value(open_value, close_value, mode_flag);
        if (source_count < length) {
            source[source_count] = value;
            source_count += 1;
            if (source_count < length) {
                continue;
            }
        } else {
            source[source_head] = value;
            source_head += 1;
            if (source_head == length) {
                source_head = 0;
            }
        }

        ordered_window_from_ring_device(window, source, length, source_head);
        for (int j = 0; j < length; ++j) {
            median[j] = window[j];
        }
        insertion_sort_device(median, length);
        double median_value = median_from_sorted_device(median, length);

        int count = 0;
        for (int j = 1; j < length; ++j) {
            double prev = window[j - 1];
            double curr = window[j];
            if ((curr > median_value && curr > prev) || (curr < median_value && curr < prev)) {
                count += 1;
            }
        }

        double mmi = static_cast<double>(count) * (100.0 / static_cast<double>(length - 1));
        row_mmi[i] = mmi;

        if (smooth_count < length) {
            smooth[smooth_count] = mmi;
            smooth_sum += mmi;
            smooth_count += 1;
            if (smooth_count == length) {
                row_smoothed[i] = smooth_sum / static_cast<double>(length);
            }
            continue;
        }

        double old = smooth[smooth_head];
        smooth[smooth_head] = mmi;
        smooth_sum += mmi - old;
        smooth_head += 1;
        if (smooth_head == length) {
            smooth_head = 0;
        }
        row_smoothed[i] = smooth_sum / static_cast<double>(length);
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * clean path  : market_meanness_index_compute_clean_sorted,
//                   src/indicators/market_meanness_index.rs:622-672 -- an
//                   INCREMENTALLY SORTED window plus count_meanness_from_ring
//                   (:594-620).
//   * dirty path  : MarketMeannessIndexStream::update_reset_on_nan / ::update,
//                   :456-505.
//   * refusals    : market_meanness_index_prepare, :518-556.
//   * emitted col : `mmi`. compute_market_meanness_index_batch
//                   (cpu_batch.rs:11586) maps output_id "value" -> out.mmi
//                   (:11718 region, the "mmi" || "value" arm).
//   * PERIOD-INVARIANT: the batch reads `length` (300) and `source_mode`
//                   ("Price") and never `period` (cpu_batch.rs:11597-11599).
//
// THE TWO PATHS ARE THE SAME ARITHMETIC, which is why ONE kernel serves both.
// The clean path keeps a sorted array and reads the median off it; the stream
// copies the window and calls median_from (:336-351), which is
// select_nth_unstable + (lower + upper) * 0.5 for an even length. An order
// statistic is EXACT -- both name the same two doubles -- and the meanness
// count walks the same chronological window in the same direction. The only
// real difference is the stream's RESET on an invalid bar, which this kernel
// implements, so it is a superset of the clean path rather than a choice
// between them.
//
// WHY CloseSlice AND NOT Ohlc4. `source_mode` defaults to "Price"
// (cpu_batch.rs:11599), and in Price mode both is_valid_bar (:273-279) and
// source_value (:281-287) read CLOSE ALONE -- open is length-checked by the
// dispatcher and never dereferenced. Registering Ohlc4 would claim this kernel
// reads three series it never touches.
//
// FIRST-VALID IGNORED: the stream walks from index 0 and resets on any
// non-finite close, and this kernel starts the row as all-NaN, so no index is
// consulted. NOTE that the CPU's DIRTY path leaves every unwritten slot past
// its NaN prefix UNINITIALISED in release (alloc_with_nan_prefix(len, warmup)
// at :685 followed by writes only on Some, :568-572). This kernel writes NaN
// there instead of reproducing uninitialised memory.
//
// ORDER STATISTIC ON THE CARD: a 300-deep sorted array in a per-thread local
// array, inserted and removed by shift. That is O(length) per bar, exactly the
// host's cost, and it is the "per-column selection" shape the brief names.
//
// -0.0 vs 0.0: the CPU orders with total_cmp, which separates them; this kernel
// orders with `<`, which does not. The two agree on every median it can
// produce, because (-0.0 + 0.0) * 0.5, (0.0 + 0.0) * 0.5 and (-0.0 + -0.0)*0.5
// are all zero and every `>`/`<` comparison against zero is sign-blind.
// ===========================================================================

#define MMI_NEO_LENGTH 300

static __device__ __forceinline__ double mmi_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void market_meanness_index_neo_batch_f64(const double* __restrict__ prices,
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
  const double nn = mmi_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int length = MMI_NEO_LENGTH;

  // prepare, :518-556. minimum_length() is 6 (:268-270).
  if (length < 6 || length > n) return;
  int first = -1, valid = 0;
  for (int i = 0; i < n; ++i) {
    if (isfinite(prices[i])) { if (first < 0) first = i; valid += 1; }
  }
  if (first < 0) return;
  if (valid < length) return;

  double ring[MMI_NEO_LENGTH];
  double sorted[MMI_NEO_LENGTH];
  int count = 0, head = 0;
  const double scale = 100.0 / (double)(length - 1);
  const int mid = length / 2;
  const bool odd = (length & 1) == 1;

  for (int i = 0; i < n; ++i) {
    const double close = prices[i];
    // is_valid_bar in Price mode, :274-275.
    if (!isfinite(close)) {
      count = 0; head = 0;           // reset(), :446-453
      continue;
    }
    const double value = close;      // source_value in Price mode, :282-283

    if (count < length) {
      ring[count] = value;
      // sorted_insert, :580-586 -- lower bound, then shift right.
      int pos = 0;
      while (pos < count && sorted[pos] < value) pos += 1;
      for (int k = count; k > pos; --k) sorted[k] = sorted[k - 1];
      sorted[pos] = value;
      count += 1;
      if (count < length) continue;
    } else {
      const double old = ring[head];
      // sorted_remove, :588-592. The outgoing value was inserted by this same
      // loop, so an element equal to it is always present and the CPU's
      // binary_search always returns Ok -- there is no Err branch to mirror.
      int pos = 0;
      while (pos < length && sorted[pos] < old) pos += 1;
      for (int k = pos; k + 1 < length; ++k) sorted[k] = sorted[k + 1];
      // Re-insert into the now (length - 1)-long prefix.
      int ip = 0;
      while (ip < length - 1 && sorted[ip] < value) ip += 1;
      for (int k = length - 1; k > ip; --k) sorted[k] = sorted[k - 1];
      sorted[ip] = value;

      ring[head] = value;
      head += 1;
      if (head == length) head = 0;
    }

    const double median = odd ? sorted[mid] : ((sorted[mid - 1] + sorted[mid]) * 0.5);

    // count_meanness_from_ring, :594-620 -- chronological walk starting at the
    // OLDEST slot, prev seeded from it.
    int meanness = 0;
    double prev = ring[head];
    for (int k = head + 1; k < length; ++k) {
      const double curr = ring[k];
      if ((curr > median && curr > prev) || (curr < median && curr < prev)) meanness += 1;
      prev = curr;
    }
    for (int k = 0; k < head; ++k) {
      const double curr = ring[k];
      if ((curr > median && curr > prev) || (curr < median && curr < prev)) meanness += 1;
      prev = curr;
    }

    row[i] = (double)meanness * scale;
  }
}
