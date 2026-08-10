#include <cmath>
#include <cstdint>

static __device__ inline void push_shift(double* buf, int* count, int cap, double value) {
    if (*count < cap) {
        buf[*count] = value;
        *count += 1;
        return;
    }
    for (int i = 1; i < cap; ++i) {
        buf[i - 1] = buf[i];
    }
    buf[cap - 1] = value;
}

static __device__ inline double linreg_slope(const double* window, int n) {
    if (n <= 1) {
        return 0.0;
    }
    double nf = static_cast<double>(n);
    double sum_x = static_cast<double>(n * (n - 1) / 2);
    double sum_x2 = static_cast<double>((n - 1) * n * (2 * n - 1) / 6);
    double denom = nf * sum_x2 - sum_x * sum_x;
    if (fabs(denom) <= 2.2204460492503131e-16) {
        return 0.0;
    }
    double sum_y = 0.0;
    double sum_xy = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double value = window[i];
        sum_y += value;
        sum_xy += x * value;
    }
    return (nf * sum_xy - sum_x * sum_y) / denom;
}

static __device__ inline double wma_value(const double* window, int period) {
    if (period <= 1) {
        return window[period - 1];
    }
    double weighted_sum = 0.0;
    double denom = static_cast<double>(period * (period + 1) / 2);
    for (int i = 0; i < period; ++i) {
        weighted_sum += window[i] * static_cast<double>(i + 1);
    }
    return weighted_sum / denom;
}

extern "C" __global__ void projection_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* source,
    int len,
    const int* lengths,
    const int* smooth_lengths,
    int rows,
    int max_length,
    int max_smooth_length,
    double* high_window_buf,
    double* low_window_buf,
    double* high_slopes_buf,
    double* low_slopes_buf,
    double* pbo_window_buf,
    double* signal_window_buf,
    double* out_pbo,
    double* out_signal
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int smooth_length = smooth_lengths[row];
    if (length <= 0 || smooth_length <= 0) {
        return;
    }

    const double nan = NAN;
    const double inf = 1.7976931348623157e308;
    double* high_window = high_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* low_window = low_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* high_slopes = high_slopes_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* low_slopes = low_slopes_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* pbo_window = pbo_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* signal_window =
        signal_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_out_pbo = out_pbo + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int window_count = 0;
    int slope_count = 0;
    int pbo_count = 0;
    int signal_count = 0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double s = source[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(s)) {
            window_count = 0;
            slope_count = 0;
            pbo_count = 0;
            signal_count = 0;
            row_out_pbo[i] = nan;
            row_out_signal[i] = nan;
            continue;
        }

        if (window_count < length) {
            high_window[window_count] = h;
            low_window[window_count] = l;
            window_count += 1;
        } else {
            for (int j = 1; j < length; ++j) {
                high_window[j - 1] = high_window[j];
                low_window[j - 1] = low_window[j];
            }
            high_window[length - 1] = h;
            low_window[length - 1] = l;
        }

        double high_slope = nan;
        double low_slope = nan;
        if (window_count == length) {
            high_slope = linreg_slope(high_window, length);
            low_slope = linreg_slope(low_window, length);
        }

        if (slope_count < length) {
            high_slopes[slope_count] = high_slope;
            low_slopes[slope_count] = low_slope;
            slope_count += 1;
        } else {
            for (int j = 1; j < length; ++j) {
                high_slopes[j - 1] = high_slopes[j];
                low_slopes[j - 1] = low_slopes[j];
            }
            high_slopes[length - 1] = high_slope;
            low_slopes[length - 1] = low_slope;
        }

        bool slopes_ready = window_count == length && slope_count == length;
        if (slopes_ready) {
            for (int j = 0; j < length; ++j) {
                if (!isfinite(high_slopes[j]) || !isfinite(low_slopes[j])) {
                    slopes_ready = false;
                    break;
                }
            }
        }
        if (!slopes_ready) {
            row_out_pbo[i] = nan;
            row_out_signal[i] = nan;
            continue;
        }

        double upper = -inf;
        double lower = inf;
        int last = length - 1;
        for (int age = 0; age < length; ++age) {
            int idx = last - age;
            double projected_high = high_window[idx] + high_slopes[idx] * static_cast<double>(age);
            double projected_low = low_window[idx] + low_slopes[idx] * static_cast<double>(age);
            if (projected_high > upper) {
                upper = projected_high;
            }
            if (projected_low < lower) {
                lower = projected_low;
            }
        }

        double range = upper - lower;
        double raw = fabs(range) <= 2.2204460492503131e-16 ? 0.0 : (100.0 * (s - lower) / range);

        push_shift(pbo_window, &pbo_count, smooth_length, raw);
        if (pbo_count < smooth_length) {
            row_out_pbo[i] = nan;
            row_out_signal[i] = nan;
            continue;
        }
        double pbo = smooth_length == 1 ? raw : wma_value(pbo_window, smooth_length);

        push_shift(signal_window, &signal_count, smooth_length, pbo);
        row_out_pbo[i] = pbo;
        row_out_signal[i] =
            signal_count < smooth_length ? nan
                                         : (smooth_length == 1 ? pbo
                                                               : wma_value(signal_window, smooth_length));
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic  : compute_row_default_14_4,
//                   src/indicators/projection_oscillator.rs:695-790. That is
//                   the body compute_row (:562-575) selects for the defaults --
//                   length 14, smooth_length 4 -- and this lane is
//                   PERIOD-INVARIANT, so it is the ONLY body reachable here.
//   * WMA(4)      : Wma4State::update, :629-676.
//   * slope       : linreg_slope_14, :681-690.
//   * refusals    : validate_common, :529-560.
//   * warmup      : pbo_warmup_prefix = 2*length + smooth_length - 3 = 29
//                   (:499-505).
//   * emitted col : `pbo`. compute_projection_oscillator_batch
//                   (cpu_batch.rs:7111) maps output_id "value" -> out.pbo
//                   (:7152).
//   * PERIOD-INVARIANT: the batch reads `length` (14) and `smooth_length` (4)
//                   and never `period` (cpu_batch.rs:7134-7136).
//   * FIRST-VALID IGNORED: the CPU never computes one. It walks from index 0
//                   and RESETS every accumulator on any invalid triple
//                   (:713-728), and the NaN prefix comes from the fixed
//                   warmup, not from a scan.
//
// SOURCE: ProjectionOscillatorInput::from_slices(high, low, close, ..)
// (cpu_batch.rs:7137-7141) -- the third series IS close, so the Hlc triple
// carries everything and there is no fourth pointer.
//
// EPSILON: `range.abs() <= f64::EPSILON` (:777) is already an f64-width
// constant on the host, so it is carried across as DBL_EPSILON. Re-deriving it
// at f64 width IS DBL_EPSILON.
//
// The `>`/`<` comparisons in the projection max/min (:768-775) are
// transliterated rather than turned into fmax/fmin because the CPU has ALREADY
// refused to reach that block unless all 28 window and slope values are finite
// (:757-762) -- there is no NaN for an fmax to protect against, and using one
// would silently accept a state the CPU rejects.
//
// SEQUENTIAL, one thread per column: two shifting 14-windows, two shifting
// slope windows and two WMA(4) states all carry across bars.
// ===========================================================================

#include <float.h>

static __device__ __forceinline__ double pjo_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// is_valid_triple, :438-440.
static __device__ __forceinline__ bool pjo_neo_valid(double h, double l, double s) {
  return isfinite(h) && isfinite(l) && isfinite(s);
}

// linreg_slope_14, :681-690. Accumulation order is ascending index, exactly as
// the CPU iterator runs.
static __device__ __forceinline__ double pjo_neo_slope14(const double* v) {
  double sum_y = 0.0, sum_xy = 0.0;
  for (int idx = 0; idx < 14; ++idx) {
    const double x = (double)idx;
    sum_y += v[idx];
    sum_xy += x * v[idx];
  }
  return (14.0 * sum_xy - 91.0 * sum_y) / 3185.0;
}

// Wma4State, :629-676. `have` is the Option the CPU returns.
struct Pjo_Wma4 {
  double values[4];
  int pos;
  int len;
  double sum;
  double weighted_sum;
};

static __device__ __forceinline__ void pjo_wma4_reset(Pjo_Wma4* w) {
  w->pos = 0; w->len = 0; w->sum = 0.0; w->weighted_sum = 0.0;
}

static __device__ __forceinline__ bool pjo_wma4_update(Pjo_Wma4* w, double value, double* outv) {
  if (!isfinite(value)) { pjo_wma4_reset(w); return false; }
  if (w->len < 4) {
    const int weight = w->len + 1;
    w->values[w->len] = value;
    w->len += 1;
    w->sum += value;
    w->weighted_sum += value * (double)weight;
    if (w->len == 4) { *outv = w->weighted_sum / 10.0; return true; }
    return false;
  }
  const double oldest = w->values[w->pos];
  const double old_sum = w->sum;
  w->values[w->pos] = value;
  w->pos = (w->pos + 1) & 3;
  w->sum = old_sum - oldest + value;
  w->weighted_sum = w->weighted_sum - old_sum + 4.0 * value;
  *outv = w->weighted_sum / 10.0;
  return true;
}

extern "C" __global__
void projection_oscillator_neo_batch_f64(const double* __restrict__ high,
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
  const double nn = pjo_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  // validate_common, :529-560. needed = 2*14 + 2*4 - 3 = 33 CONSECUTIVE valid
  // bars (signal_needed_bars, :485-496) -- the signal warmup, not the pbo one,
  // and the CPU really does gate BOTH outputs on it.
  const int needed = 2 * 14 + 2 * 4 - 3;
  int best = 0, cur = 0;
  for (int i = 0; i < n; ++i) {
    if (pjo_neo_valid(high[i], low[i], close[i])) { cur += 1; if (cur > best) best = cur; }
    else cur = 0;
  }
  if (best == 0 || best < needed) return;

  double high_window[14], low_window[14];
  double high_slopes[14], low_slopes[14];
  int high_len = 0, low_len = 0, high_slope_len = 0, low_slope_len = 0;
  int high_slope_finite = 0, low_slope_finite = 0;
  for (int k = 0; k < 14; ++k) {
    high_window[k] = 0.0; low_window[k] = 0.0;
    high_slopes[k] = nn;  low_slopes[k] = nn;
  }
  Pjo_Wma4 pbo_wma, signal_wma;
  pjo_wma4_reset(&pbo_wma); pjo_wma4_reset(&signal_wma);
  for (int k = 0; k < 4; ++k) { pbo_wma.values[k] = 0.0; signal_wma.values[k] = 0.0; }

  for (int i = 0; i < n; ++i) {
    const double h = high[i], l = low[i], s = close[i];
    if (!pjo_neo_valid(h, l, s)) {
      high_len = 0; low_len = 0;
      high_slope_len = 0; low_slope_len = 0;
      high_slope_finite = 0; low_slope_finite = 0;
      pjo_wma4_reset(&pbo_wma); pjo_wma4_reset(&signal_wma);
      continue;
    }

    // push_fixed_14, :649-657 -- a SHIFTING window, slot 0 oldest.
    if (high_len < 14) { high_window[high_len] = h; high_len += 1; }
    else { for (int k = 0; k < 13; ++k) high_window[k] = high_window[k + 1]; high_window[13] = h; }
    if (low_len < 14) { low_window[low_len] = l; low_len += 1; }
    else { for (int k = 0; k < 13; ++k) low_window[k] = low_window[k + 1]; low_window[13] = l; }

    const double high_slope = (high_len == 14) ? pjo_neo_slope14(high_window) : nn;
    const double low_slope = (low_len == 14) ? pjo_neo_slope14(low_window) : nn;

    // push_slope_fixed_14, :660-678.
    if (high_slope_len < 14) { high_slopes[high_slope_len] = high_slope; high_slope_len += 1; }
    else {
      if (isfinite(high_slopes[0])) high_slope_finite -= 1;
      for (int k = 0; k < 13; ++k) high_slopes[k] = high_slopes[k + 1];
      high_slopes[13] = high_slope;
    }
    if (isfinite(high_slope)) high_slope_finite += 1;

    if (low_slope_len < 14) { low_slopes[low_slope_len] = low_slope; low_slope_len += 1; }
    else {
      if (isfinite(low_slopes[0])) low_slope_finite -= 1;
      for (int k = 0; k < 13; ++k) low_slopes[k] = low_slopes[k + 1];
      low_slopes[13] = low_slope;
    }
    if (isfinite(low_slope)) low_slope_finite += 1;

    if (high_len != 14 || low_len != 14 || high_slope_len != 14 || low_slope_len != 14) continue;
    if (high_slope_finite != 14 || low_slope_finite != 14) continue;

    double upper = -INFINITY, lower = INFINITY;
    for (int age = 0; age < 14; ++age) {
      const int idx = 13 - age;
      const double age_f = (double)age;
      const double projected_high = high_window[idx] + high_slopes[idx] * age_f;
      const double projected_low = low_window[idx] + low_slopes[idx] * age_f;
      if (projected_high > upper) upper = projected_high;
      if (projected_low < lower) lower = projected_low;
    }

    const double range = upper - lower;
    const double raw = (fabs(range) <= DBL_EPSILON) ? 0.0 : (100.0 * (s - lower) / range);

    double pbo = 0.0;
    if (pjo_wma4_update(&pbo_wma, raw, &pbo)) {
      row[i] = pbo;
      double sig = 0.0;
      // The CPU still feeds the signal WMA here (:784-787); this lane emits pbo,
      // so the value is computed for state and discarded.
      (void)pjo_wma4_update(&signal_wma, pbo, &sig);
    }
  }
}
