#include <cmath>
#include <cstddef>

extern "C" __global__ void premier_rsi_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ stoch_lengths,
    const int* __restrict__ smooth_lengths,
    int rows,
    double* __restrict__ out_values
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int rsi_length = rsi_lengths[row];
    int stoch_length = stoch_lengths[row];
    int smooth_length = smooth_lengths[row];
    double* row_values = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_values[i] = NAN;
    }

    if (rsi_length <= 0 || stoch_length <= 0 || smooth_length <= 0) {
        return;
    }

    int ema_length = static_cast<int>(floor(sqrt(static_cast<double>(smooth_length)) + 0.5));
    if (ema_length < 1) {
        ema_length = 1;
    }
    double ema_alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);

    double* stoch_window = new double[stoch_length];
    if (stoch_window == nullptr) {
        return;
    }

    bool has_prev = false;
    double prev = NAN;
    int seed_count = 0;
    double sum_gain = 0.0;
    double sum_loss = 0.0;
    bool seeded = false;
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    int stoch_count = 0;
    int stoch_head = 0;

    bool has_ema1 = false;
    bool has_ema2 = false;
    double ema1 = NAN;
    double ema2 = NAN;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            has_prev = false;
            prev = NAN;
            seed_count = 0;
            sum_gain = 0.0;
            sum_loss = 0.0;
            seeded = false;
            avg_gain = 0.0;
            avg_loss = 0.0;
            stoch_count = 0;
            stoch_head = 0;
            has_ema1 = false;
            has_ema2 = false;
            ema1 = NAN;
            ema2 = NAN;
            continue;
        }

        if (!has_prev) {
            prev = value;
            has_prev = true;
            continue;
        }

        double delta = value - prev;
        prev = value;

        bool rsi_ready = false;
        double rsi = NAN;
        if (!seeded) {
            double gain = delta > 0.0 ? delta : 0.0;
            double loss = delta < 0.0 ? -delta : 0.0;
            sum_gain += gain;
            sum_loss += loss;
            seed_count += 1;
            if (seed_count == rsi_length) {
                seeded = true;
                avg_gain = sum_gain / static_cast<double>(rsi_length);
                avg_loss = sum_loss / static_cast<double>(rsi_length);
                double denom = avg_gain + avg_loss;
                rsi = denom == 0.0 ? 50.0 : 100.0 * avg_gain / denom;
                rsi_ready = true;
            }
        } else {
            double gain = delta > 0.0 ? delta : 0.0;
            double loss = delta < 0.0 ? -delta : 0.0;
            double inv_p = 1.0 / static_cast<double>(rsi_length);
            double beta = 1.0 - inv_p;
            avg_gain = avg_gain * beta + inv_p * gain;
            avg_loss = avg_loss * beta + inv_p * loss;
            double denom = avg_gain + avg_loss;
            rsi = denom == 0.0 ? 50.0 : 100.0 * avg_gain / denom;
            rsi_ready = true;
        }

        if (!rsi_ready) {
            continue;
        }

        if (stoch_count < stoch_length) {
            stoch_window[(stoch_head + stoch_count) % stoch_length] = rsi;
            stoch_count += 1;
        } else {
            stoch_window[stoch_head] = rsi;
            stoch_head += 1;
            if (stoch_head == stoch_length) {
                stoch_head = 0;
            }
        }

        if (stoch_count < stoch_length) {
            continue;
        }

        double highest = stoch_window[0];
        double lowest = stoch_window[0];
        for (int j = 1; j < stoch_count; ++j) {
            double sample = stoch_window[j];
            if (sample > highest) {
                highest = sample;
            }
            if (sample < lowest) {
                lowest = sample;
            }
        }

        double denom = highest - lowest;
        double sk = fabs(denom) <= 1.0e-12 ? 50.0 : (rsi - lowest) * (100.0 / denom);
        double nsk = 0.1 * (sk - 50.0);

        ema1 = has_ema1 ? ema_alpha * nsk + (1.0 - ema_alpha) * ema1 : nsk;
        ema2 = has_ema2 ? ema_alpha * ema1 + (1.0 - ema_alpha) * ema2 : ema1;
        has_ema1 = true;
        has_ema2 = true;

        row_values[i] = tanh(ema2 * 0.5);
    }

    delete[] stoch_window;
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic  : PremierRsiOscillatorStream::update,
//                   src/indicators/premier_rsi_oscillator.rs:287-364, driven by
//                   row_from_slice_prefilled (:538-546) -- the ONLY compute path
//                   this indicator has; there is no separate scalar/AVX body.
//   * inner RSI   : RsiStream::update, src/indicators/rsi.rs:1064-1119. Wilder
//                   smoothing, seeded from a plain sum over the first `period`
//                   deltas.
//   * refusals    : prepare, :502-530.
//   * warmup      : first + rsi_length + stoch_length - 1 (:556).
//   * emitted col : the single `values` series. compute_premier_rsi_oscillator_
//                   batch (cpu_batch.rs:8628) calls expect_value_output, so
//                   output_id "value" is the only one accepted (:8632).
//   * PERIOD-INVARIANT: the batch reads rsi_length (14), stoch_length (8) and
//                   smooth_length (25) and never `period`
//                   (cpu_batch.rs:8641-8646).
//   * FIRST-VALID IGNORED: the CPU consults `first` ONLY to size the NaN prefix
//                   (:556). The stream itself walks from index 0 and RESETS on
//                   any non-finite bar (:288-291), and this kernel starts the
//                   row as all-NaN and writes only the bars the stream emits --
//                   which reproduces the prefix without needing the index. So
//                   the row declares Ignored rather than claiming a rule it
//                   never reads.
//
// EPSILON: FLOAT_TOL = 1e-12 (:36) is already an f64-width guard on a
// difference of two RSI values (0..100). Carried across unchanged BECAUSE it
// was authored at f64 width, not because it looked familiar.
//
// NaN: `delta.max(0.0)` (rsi.rs:1079) is f64::max, which returns the NON-NaN
// operand. fmax() is its exact CUDA twin; an `a > b ? a : b` chain would let a
// NaN survive and poison every later bar of the Wilder recurrence.
//
// SEQUENTIAL, one thread per column: a Wilder recurrence, two monotone deques
// and two EMAs all carry across bars.
// ===========================================================================

#define PRO_NEO_RSI_LEN 14
#define PRO_NEO_STOCH_LEN 8
#define PRO_NEO_SMOOTH_LEN 25
#define PRO_NEO_DQ_CAP (PRO_NEO_STOCH_LEN + 1)
#define PRO_NEO_FLOAT_TOL 1e-12

static __device__ __forceinline__ double pro_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void premier_rsi_oscillator_neo_batch_f64(const double* __restrict__ prices,
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
  const double nn = pro_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int rsi_len = PRO_NEO_RSI_LEN;
  const int stoch_len = PRO_NEO_STOCH_LEN;
  // resolve_params, :493-500: ema_length = round(sqrt(smooth_length)).max(1).
  const double ema_len = fmax(round(sqrt((double)PRO_NEO_SMOOTH_LEN)), 1.0);
  const double alpha = 2.0 / (ema_len + 1.0);
  const double one_minus_alpha = 1.0 - alpha;

  // prepare, :502-530. Every branch is an Err, and an Err is no column at all.
  const int needed = rsi_len + stoch_len - 1;
  int first = n;
  for (int i = 0; i < n; ++i) { if (isfinite(prices[i])) { first = i; break; } }
  if (first >= n) return;
  int valid = 0;
  for (int i = 0; i < n; ++i) if (isfinite(prices[i])) valid += 1;
  if (valid < needed) return;

  // --- RsiStream state (rsi.rs:1016-1032) -------------------------------
  const double inv_p = 1.0 / (double)rsi_len;
  const double beta = 1.0 - inv_p;
  bool has_prev = false;
  double prev = nn;
  int seed_count = 0;
  double sum_gain = 0.0, sum_loss = 0.0;
  bool poisoned = false;
  double avg_gain = 0.0, avg_loss = 0.0;
  bool seeded = false;

  // --- premier state (:250-259) ------------------------------------------
  int rsi_index = 0;
  int max_idx[PRO_NEO_DQ_CAP], min_idx[PRO_NEO_DQ_CAP];
  double max_val[PRO_NEO_DQ_CAP], min_val[PRO_NEO_DQ_CAP];
  int max_head = 0, max_len = 0, min_head = 0, min_len = 0;
  bool have_ema1 = false, have_ema2 = false;
  double ema1 = 0.0, ema2 = 0.0;

  for (int i = 0; i < n; ++i) {
    const double value = prices[i];

    if (!isfinite(value)) {
      // reset() -- :281-284 rebuilds the WHOLE stream, inner RSI included.
      has_prev = false; prev = nn;
      seed_count = 0; sum_gain = 0.0; sum_loss = 0.0; poisoned = false;
      avg_gain = 0.0; avg_loss = 0.0; seeded = false;
      rsi_index = 0; max_head = 0; max_len = 0; min_head = 0; min_len = 0;
      have_ema1 = false; have_ema2 = false;
      continue;
    }

    // ---- inner RSI ------------------------------------------------------
    double rsi = 0.0;
    bool have_rsi = false;
    if (!has_prev) {
      prev = value;
      has_prev = true;
    } else {
      const double delta = value - prev;
      prev = value;
      if (!seeded) {
        if (!isfinite(delta)) poisoned = true;
        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);
        sum_gain += gain;
        sum_loss += loss;
        seed_count += 1;
        if (seed_count == rsi_len) {
          seeded = true;
          if (poisoned) {
            avg_gain = nn; avg_loss = nn;
            rsi = nn; have_rsi = true;
          } else {
            avg_gain = sum_gain * inv_p;
            avg_loss = sum_loss * inv_p;
            const double denom = avg_gain + avg_loss;
            rsi = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
            have_rsi = true;
          }
        }
      } else {
        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);
        avg_gain = fma(avg_gain, beta, inv_p * gain);
        avg_loss = fma(avg_loss, beta, inv_p * loss);
        const double denom = avg_gain + avg_loss;
        rsi = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
        have_rsi = true;
      }
    }
    if (!have_rsi) continue;  // `?` at :293 -- None propagates, no reset.

    if (!isfinite(rsi)) {
      // :294-297 -- a non-finite RSI resets the whole stream.
      has_prev = false; prev = nn;
      seed_count = 0; sum_gain = 0.0; sum_loss = 0.0; poisoned = false;
      avg_gain = 0.0; avg_loss = 0.0; seeded = false;
      rsi_index = 0; max_head = 0; max_len = 0; min_head = 0; min_len = 0;
      have_ema1 = false; have_ema2 = false;
      continue;
    }

    // ---- monotone deques (:300-335) --------------------------------------
    while (max_len > 0) {
      const int back = (max_head + max_len - 1) % PRO_NEO_DQ_CAP;
      if (max_val[back] <= rsi) max_len -= 1; else break;
    }
    {
      const int slot = (max_head + max_len) % PRO_NEO_DQ_CAP;
      max_idx[slot] = rsi_index; max_val[slot] = rsi; max_len += 1;
    }
    while (min_len > 0) {
      const int back = (min_head + min_len - 1) % PRO_NEO_DQ_CAP;
      if (min_val[back] >= rsi) min_len -= 1; else break;
    }
    {
      const int slot = (min_head + min_len) % PRO_NEO_DQ_CAP;
      min_idx[slot] = rsi_index; min_val[slot] = rsi; min_len += 1;
    }

    const int window_start = (rsi_index + 1 > stoch_len) ? (rsi_index + 1 - stoch_len) : 0;
    while (max_len > 0 && max_idx[max_head] < window_start) {
      max_head = (max_head + 1) % PRO_NEO_DQ_CAP; max_len -= 1;
    }
    while (min_len > 0 && min_idx[min_head] < window_start) {
      min_head = (min_head + 1) % PRO_NEO_DQ_CAP; min_len -= 1;
    }

    rsi_index += 1;
    if (rsi_index < stoch_len) continue;

    const double highest = (max_len > 0) ? max_val[max_head] : rsi;
    const double lowest = (min_len > 0) ? min_val[min_head] : rsi;
    const double denom = highest - lowest;
    // :345-350 -- mul_add, ONE rounding, and the CPU really does add 0.0.
    const double sk = (fabs(denom) <= PRO_NEO_FLOAT_TOL)
                          ? 50.0
                          : fma(rsi - lowest, 100.0 / denom, 0.0);
    const double nsk = 0.1 * (sk - 50.0);

    // :353-356 and :359-362 -- TWO roundings each (a*b + c*d), NOT an fma. The
    // CPU writes `alpha * x + (1.0 - alpha) * prev`, so folding it would change
    // the result.
    ema1 = have_ema1 ? (alpha * nsk + one_minus_alpha * ema1) : nsk;
    have_ema1 = true;
    ema2 = have_ema2 ? (alpha * ema1 + one_minus_alpha * ema2) : ema1;
    have_ema2 = true;

    row[i] = tanh(ema2 * 0.5);
  }
}
