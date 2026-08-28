#include <cmath>
#include <cstddef>

extern "C" __global__ void price_density_market_noise_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ eval_periods,
    int rows,
    int max_length,
    int max_eval_period,
    double* __restrict__ high_ring_buf,
    double* __restrict__ low_ring_buf,
    double* __restrict__ tr_ring_buf,
    double* __restrict__ pd_ring_buf,
    double* __restrict__ out_price_density,
    double* __restrict__ out_price_density_percent
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    int eval_period = eval_periods[row];
    double* high_ring =
        high_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* low_ring =
        low_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* tr_ring =
        tr_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* pd_ring =
        pd_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_eval_period);
    double* row_out_price_density =
        out_price_density + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_price_density_percent =
        out_price_density_percent + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_price_density[i] = NAN;
        row_out_price_density_percent[i] = NAN;
    }

    if (length <= 0 ||
        eval_period <= 0 ||
        length > max_length ||
        eval_period > max_eval_period) {
        return;
    }

    int window_head = 0;
    int window_count = 0;
    int pd_head = 0;
    int pd_count = 0;
    bool have_prev_close = false;
    double prev_close = NAN;
    double tr_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            window_head = 0;
            window_count = 0;
            pd_head = 0;
            pd_count = 0;
            have_prev_close = false;
            prev_close = NAN;
            tr_sum = 0.0;
            continue;
        }

        double tr = h - l;
        if (have_prev_close) {
            double high_close = fabs(h - prev_close);
            double low_close = fabs(l - prev_close);
            if (high_close > tr) {
                tr = high_close;
            }
            if (low_close > tr) {
                tr = low_close;
            }
        }
        prev_close = c;
        have_prev_close = true;

        if (window_count < length) {
            high_ring[window_count] = h;
            low_ring[window_count] = l;
            tr_ring[window_count] = tr;
            tr_sum += tr;
            window_count += 1;
        } else {
            tr_sum -= tr_ring[window_head];
            high_ring[window_head] = h;
            low_ring[window_head] = l;
            tr_ring[window_head] = tr;
            tr_sum += tr;
            window_head += 1;
            if (window_head == length) {
                window_head = 0;
            }
        }

        if (window_count < length) {
            continue;
        }

        double highest = -INFINITY;
        double lowest = INFINITY;
        for (int j = 0; j < length; ++j) {
            int idx = (window_head + j) % length;
            double high_value = high_ring[idx];
            double low_value = low_ring[idx];
            if (high_value > highest) {
                highest = high_value;
            }
            if (low_value < lowest) {
                lowest = low_value;
            }
        }

        double denom = highest - lowest;
        double price_density = denom > 0.0 ? tr_sum / denom : NAN;
        row_out_price_density[i] = price_density;

        if (pd_count < eval_period) {
            pd_ring[pd_count] = price_density;
            pd_count += 1;
        } else {
            pd_ring[pd_head] = price_density;
            pd_head += 1;
            if (pd_head == eval_period) {
                pd_head = 0;
            }
        }

        if (pd_count < eval_period || !isfinite(price_density)) {
            continue;
        }

        bool invalid = false;
        int rank = 0;
        for (int j = 0; j < eval_period; ++j) {
            int idx = (pd_head + j) % eval_period;
            double value = pd_ring[idx];
            if (!isfinite(value)) {
                invalid = true;
                break;
            }
            if (value <= price_density) {
                rank += 1;
            }
        }

        if (!invalid) {
            row_out_price_density_percent[i] =
                (static_cast<double>(rank) / static_cast<double>(eval_period)) * 100.0;
        }
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic  : PriceDensityMarketNoiseStream::update_reset_on_nan /
//                   ::update, src/indicators/price_density_market_noise.rs:
//                   369-478, driven by price_density_market_noise_compute_into
//                   (:527-552). This is the only compute body the indicator
//                   has -- no scalar/AVX split.
//   * refusals    : price_density_market_noise_prepare, :489-525.
//   * emitted col : `price_density`. compute_price_density_market_noise_batch
//                   (cpu_batch.rs:11836) maps output_id "value" ->
//                   out.price_density (:11852).
//   * PERIOD-INVARIANT: the batch reads `length` (14) and `eval_period` (200)
//                   and never `period` (cpu_batch.rs:11846-11848).
//   * FIRST-VALID IGNORED, and the CPU says so itself:
//                   price_density_market_noise_with_kernel:565 is literally
//                   `let _ = first;` and the NaN prefix is
//                   alloc_with_nan_prefix(len, 0). The stream walks from index
//                   0 and RESETS on any invalid bar, so there is no index to
//                   honour.
//
// WHY eval_period NEVER APPEARS BELOW. `eval_period` drives ONLY the second
// output, `price_density_percent` -- a percentile rank over a 200-deep sorted
// window (:437-468). This lane emits `price_density`, which the CPU computes
// before that block and never revisits, so the sorted window and its
// order-statistic are dead work here and are omitted rather than computed and
// discarded.
//
// NaN: true_range (:270-277) is `(h-l).max((h-prev).abs()).max((l-prev).abs())`
// -- f64::max, which returns the NON-NaN operand. fmax() is its exact twin. An
// if-chain would let a NaN survive into tr_sum and poison every later bar.
//
// SEQUENTIAL, one thread per column: the true-range ring sum and two monotone
// deques all carry across bars.
// ===========================================================================

#define PDMN_NEO_LENGTH 14
#define PDMN_NEO_DQ_CAP (PDMN_NEO_LENGTH + 1)

static __device__ __forceinline__ double pdmn_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// valid_bar, :242-244.
static __device__ __forceinline__ bool pdmn_neo_valid(double h, double l, double c) {
  return isfinite(h) && isfinite(l) && isfinite(c);
}

extern "C" __global__
void price_density_market_noise_neo_batch_f64(const double* __restrict__ high,
                                              const double* __restrict__ low,
                                              const double* __restrict__ close,
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
  const double nn = pdmn_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int length = PDMN_NEO_LENGTH;

  // prepare, :489-525. Every branch is an Err, and an Err is no column at all.
  if (length > n) return;
  int first = -1;
  for (int i = 0; i < n; ++i) {
    if (pdmn_neo_valid(high[i], low[i], close[i])) { first = i; break; }
  }
  if (first < 0) return;
  int valid = 0;
  for (int i = first; i < n; ++i) if (pdmn_neo_valid(high[i], low[i], close[i])) valid += 1;
  if (valid < length) return;

  double tr_win[PDMN_NEO_LENGTH];
  int hi_idx[PDMN_NEO_DQ_CAP], lo_idx[PDMN_NEO_DQ_CAP];
  double hi_val[PDMN_NEO_DQ_CAP], lo_val[PDMN_NEO_DQ_CAP];
  int index = 0, tr_head = 0, tr_len = 0;
  int hi_head = 0, hi_len = 0, lo_head = 0, lo_len = 0;
  double tr_sum = 0.0;
  bool has_prev_close = false;
  double prev_close = 0.0;
  for (int k = 0; k < length; ++k) tr_win[k] = 0.0;

  for (int i = 0; i < n; ++i) {
    const double h = high[i], l = low[i], c = close[i];

    if (!pdmn_neo_valid(h, l, c)) {
      // reset(), :353-367.
      index = 0; has_prev_close = false;
      for (int k = 0; k < length; ++k) tr_win[k] = 0.0;
      tr_head = 0; tr_len = 0; tr_sum = 0.0;
      hi_head = 0; hi_len = 0; lo_head = 0; lo_len = 0;
      continue;
    }

    const int cur = index;
    index += 1;

    // true_range, :270-277. fmax, not an if-chain -- see the header.
    double tr;
    if (has_prev_close) {
      tr = fmax(fmax(h - l, fabs(h - prev_close)), fabs(l - prev_close));
    } else {
      tr = h - l;
    }
    has_prev_close = true;
    prev_close = c;

    // :379-391 -- the fill phase writes at tr_len and leaves tr_head at 0.
    if (tr_len < length) {
      tr_win[tr_len] = tr;
      tr_sum += tr;
      tr_len += 1;
    } else {
      const double old = tr_win[tr_head];
      tr_win[tr_head] = tr;
      tr_sum += tr - old;
      tr_head += 1;
      if (tr_head == length) tr_head = 0;
    }

    while (hi_len > 0) {
      const int back = (hi_head + hi_len - 1) % PDMN_NEO_DQ_CAP;
      if (hi_val[back] <= h) hi_len -= 1; else break;
    }
    { const int s = (hi_head + hi_len) % PDMN_NEO_DQ_CAP; hi_idx[s] = cur; hi_val[s] = h; hi_len += 1; }

    while (lo_len > 0) {
      const int back = (lo_head + lo_len - 1) % PDMN_NEO_DQ_CAP;
      if (lo_val[back] >= l) lo_len -= 1; else break;
    }
    { const int s = (lo_head + lo_len) % PDMN_NEO_DQ_CAP; lo_idx[s] = cur; lo_val[s] = l; lo_len += 1; }

    const int expire_before = (cur + 1 > length) ? (cur + 1 - length) : 0;
    while (hi_len > 0 && hi_idx[hi_head] < expire_before) { hi_head = (hi_head + 1) % PDMN_NEO_DQ_CAP; hi_len -= 1; }
    while (lo_len > 0 && lo_idx[lo_head] < expire_before) { lo_head = (lo_head + 1) % PDMN_NEO_DQ_CAP; lo_len -= 1; }

    if (tr_len < length) continue;

    const double hi = (hi_len > 0) ? hi_val[hi_head] : h;
    const double lo = (lo_len > 0) ? lo_val[lo_head] : l;
    const double denom = hi - lo;
    row[i] = (denom > 0.0) ? (tr_sum / denom) : nn;
  }
}
