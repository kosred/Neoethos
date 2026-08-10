#include <cmath>
#include <cstddef>

extern "C" __global__ void qqe_weighted_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ factors,
    const int* __restrict__ smooths,
    const double* __restrict__ weights,
    int rows,
    double* __restrict__ out_rsi,
    double* __restrict__ out_trailing_stop
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const double factor = factors[row];
    const int smooth = smooths[row];
    const double weight = weights[row];

    double* row_rsi = out_rsi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ts = out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_rsi[i] = NAN;
        row_ts[i] = NAN;
    }

    if (length <= 0 || length > len || smooth <= 0 || !isfinite(factor) || factor < 0.0 ||
        !isfinite(weight)) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (isfinite(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0 || first + 1 >= len) {
        return;
    }

    const double ratio_alpha = 2.0 / (static_cast<double>(smooth) + 1.0);

    int num_count = 0;
    int den_count = 0;
    int diff_count = 0;
    double num_sum = 0.0;
    double den_sum = 0.0;
    double diff_sum = 0.0;
    double num_value = NAN;
    double den_value = NAN;
    double diff_value = NAN;
    bool num_seeded = false;
    bool den_seeded = false;
    bool diff_seeded = false;

    bool ratio_seeded = false;
    double ratio_value = NAN;

    bool has_prev_src = true;
    double prev_src = data[first];
    bool has_prev_rsi = false;
    bool has_prev_ts = false;
    double prev_rsi = NAN;
    double prev_ts = NAN;

    for (int i = first + 1; i < len; ++i) {
        const double current = data[i];
        if (!isfinite(current)) {
            has_prev_src = false;
            continue;
        }

        if (!has_prev_src) {
            prev_src = current;
            has_prev_src = true;
            continue;
        }

        const double delta = current - prev_src;
        prev_src = current;

        const double scale = (has_prev_rsi && has_prev_ts &&
                              delta * (prev_rsi - prev_ts) > 0.0)
            ? weight
            : 1.0;
        const double weighted_delta = delta * scale;

        bool num_ready = false;
        if (num_seeded) {
            num_value =
                (num_value * (static_cast<double>(length) - 1.0) + weighted_delta) /
                static_cast<double>(length);
            num_ready = true;
        } else {
            num_sum += weighted_delta;
            num_count += 1;
            if (num_count == length) {
                num_value = num_sum / static_cast<double>(length);
                num_seeded = true;
                num_ready = true;
            }
        }

        const double abs_delta = fabs(weighted_delta);
        bool den_ready = false;
        if (den_seeded) {
            den_value =
                (den_value * (static_cast<double>(length) - 1.0) + abs_delta) /
                static_cast<double>(length);
            den_ready = true;
        } else {
            den_sum += abs_delta;
            den_count += 1;
            if (den_count == length) {
                den_value = den_sum / static_cast<double>(length);
                den_seeded = true;
                den_ready = true;
            }
        }

        if (!num_ready || !den_ready || den_value == 0.0) {
            continue;
        }

        const double ratio_input = num_value / den_value;
        ratio_value = ratio_seeded
            ? ratio_alpha * ratio_input + (1.0 - ratio_alpha) * ratio_value
            : ratio_input;
        ratio_seeded = true;

        const double rsi = 50.0 * ratio_value + 50.0;
        row_rsi[i] = rsi;

        bool diff_ready = false;
        if (has_prev_rsi) {
            const double diff_input = fabs(rsi - prev_rsi);
            if (diff_seeded) {
                diff_value =
                    (diff_value * (static_cast<double>(length) - 1.0) + diff_input) /
                    static_cast<double>(length);
                diff_ready = true;
            } else {
                diff_sum += diff_input;
                diff_count += 1;
                if (diff_count == length) {
                    diff_value = diff_sum / static_cast<double>(length);
                    diff_seeded = true;
                    diff_ready = true;
                }
            }
        }

        double trailing_stop = rsi;
        if (diff_ready) {
            const bool crossover =
                has_prev_ts && rsi > prev_ts && prev_rsi <= prev_ts;
            const bool crossunder =
                has_prev_ts && rsi < prev_ts && prev_rsi >= prev_ts;
            if (crossover) {
                trailing_stop = rsi - diff_value * factor;
            } else if (crossunder) {
                trailing_stop = rsi + diff_value * factor;
            } else if (has_prev_ts) {
                if (rsi > prev_ts) {
                    trailing_stop = fmax(rsi - diff_value * factor, prev_ts);
                } else {
                    trailing_stop = fmin(rsi + diff_value * factor, prev_ts);
                }
            }
        }

        row_ts[i] = trailing_stop;
        prev_rsi = rsi;
        prev_ts = trailing_stop;
        has_prev_rsi = true;
        has_prev_ts = true;
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic  : compute_into_slices,
//                   src/indicators/qqe_weighted_oscillator.rs:486-594 -- the
//                   only compute body this indicator has.
//   * RMA / EMA   : RmaState::update (:322-341) and EmaState::update (:358-369).
//   * refusals    : prepare_input, :417-484.
//   * warmup      : first + length (:481).
//   * emitted col : `rsi`. compute_qqe_weighted_oscillator_batch
//                   (cpu_batch.rs:15909) maps output_id "value" -> out.rsi
//                   (:15932).
//   * PERIOD-INVARIANT: the batch reads `length` (14), `factor` (4.236),
//                   `smooth` (5) and `weight` (2.0) and never `period`
//                   (cpu_batch.rs:15917-15920).
//   * FIRST-VALID  : data.iter().position(|v| v.is_finite()) (:428-431) -- ONE
//                   price series scanned with is_finite, which is exactly
//                   F64FirstValidRule::CloseFinite. The value is LOAD-BEARING
//                   here: the loop starts at first + 1 and seeds prev_src from
//                   data[first] (:512-516), so a different index shifts the
//                   whole series rather than perturbing it.
//
// NaN: `.max(last_ts)` / `.min(last_ts)` (:570, :572) are f64::max / f64::min,
// which return the NON-NaN operand. fmax/fmin are their exact twins; an
// if-chain would let a NaN survive into prev_ts and poison every later bar.
//
// The `>`/`<`/`>=`/`<=` comparisons in the crossover tests are transliterated
// as comparisons because the CPU writes them as comparisons -- a NaN there
// makes the guard false on BOTH sides, which is the behaviour being copied.
//
// SEQUENTIAL, one thread per column: three Wilder RMAs, one EMA and the
// carried (prev_rsi, prev_ts) pair all cross bars.
// ===========================================================================

#define QWO_NEO_LENGTH 14
#define QWO_NEO_SMOOTH 5
#define QWO_NEO_FACTOR 4.236
#define QWO_NEO_WEIGHT 2.0

static __device__ __forceinline__ double qwo_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// RmaState, :322-341. Returns false while the seed window is still filling.
struct Qwo_Rma { int period; int count; double sum; double value; bool has; };

static __device__ __forceinline__ bool qwo_rma_update(Qwo_Rma* s, double input, double* outv) {
  if (!isfinite(input)) return false;
  if (s->has) {
    const double next = (s->value * ((double)s->period - 1.0) + input) / (double)s->period;
    s->value = next;
    *outv = next;
    return true;
  }
  s->count += 1;
  s->sum += input;
  if (s->count == s->period) {
    const double seeded = s->sum / (double)s->period;
    s->value = seeded; s->has = true;
    *outv = seeded;
    return true;
  }
  return false;
}

extern "C" __global__
void qqe_weighted_oscillator_neo_batch_f64(const double* __restrict__ prices,
                                           int n,
                                           const int* __restrict__ periods,
                                           int n_combos,
                                           int first_valid,
                                           double* __restrict__ out) {
  const int combo = blockIdx.x * blockDim.x + threadIdx.x;
  if (combo >= n_combos) return;
  (void)periods;  // PERIOD-INVARIANT -- see the header.

  if (n <= 0) return;
  double* __restrict__ row = out + (size_t)combo * (size_t)n;
  const double nn = qwo_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int length = QWO_NEO_LENGTH;
  const int smooth = QWO_NEO_SMOOTH;
  const double factor = QWO_NEO_FACTOR;
  const double weight = QWO_NEO_WEIGHT;

  int first = first_valid;
  if (first < 0) first = 0;

  // prepare_input, :417-484. Every branch is an Err, and an Err is no column.
  if (first >= n) return;
  if (length > n) return;
  int valid = 0;
  for (int i = first; i < n; ++i) if (isfinite(prices[i])) valid += 1;
  if (valid < length + 1) return;

  Qwo_Rma num_state = { length, 0, 0.0, 0.0, false };
  Qwo_Rma den_state = { length, 0, 0.0, 0.0, false };
  Qwo_Rma diff_state = { length, 0, 0.0, 0.0, false };
  const double alpha = 2.0 / ((double)smooth + 1.0);
  double ema_val = 0.0; bool ema_has = false;

  bool has_prev_src = true;
  double prev_src = prices[first];
  bool has_prev_rsi = false, has_prev_ts = false;
  double prev_rsi = 0.0, prev_ts = 0.0;

  for (int i = first + 1; i < n; ++i) {
    const double current = prices[i];
    if (!isfinite(current)) { has_prev_src = false; continue; }
    if (!has_prev_src) { prev_src = current; has_prev_src = true; continue; }

    const double delta = current - prev_src;
    double w = 1.0;
    if (has_prev_rsi && has_prev_ts && (delta * (prev_rsi - prev_ts)) > 0.0) w = weight;
    const double weighted_delta = delta * w;

    double num = 0.0, den = 0.0;
    const bool have_num = qwo_rma_update(&num_state, weighted_delta, &num);
    const bool have_den = qwo_rma_update(&den_state, fabs(weighted_delta), &den);

    if (have_num && have_den && den != 0.0) {
      // EmaState::update, :358-369. TWO roundings (a*b + c*d), NOT an fma --
      // the CPU writes it that way.
      const double ratio = num / den;
      if (isfinite(ratio)) {
        const double smoothed = ema_has ? (alpha * ratio + (1.0 - alpha) * ema_val) : ratio;
        ema_val = smoothed; ema_has = true;

        const double rsi = 50.0 * smoothed + 50.0;
        row[i] = rsi;

        double diff = 0.0;
        bool have_diff = false;
        if (has_prev_rsi) {
          have_diff = qwo_rma_update(&diff_state, fabs(rsi - prev_rsi), &diff);
        }

        double trailing_stop;
        if (have_diff) {
          const bool crossover =
              has_prev_rsi && has_prev_ts && (rsi > prev_ts) && (prev_rsi <= prev_ts);
          const bool crossunder =
              has_prev_rsi && has_prev_ts && (rsi < prev_ts) && (prev_rsi >= prev_ts);
          if (crossover) {
            trailing_stop = rsi - diff * factor;
          } else if (crossunder) {
            trailing_stop = rsi + diff * factor;
          } else if (has_prev_ts) {
            trailing_stop = (rsi > prev_ts) ? fmax(rsi - diff * factor, prev_ts)
                                            : fmin(rsi + diff * factor, prev_ts);
          } else {
            trailing_stop = rsi;
          }
        } else {
          trailing_stop = rsi;
        }

        prev_rsi = rsi; has_prev_rsi = true;
        prev_ts = trailing_stop; has_prev_ts = true;
      }
    }

    prev_src = current;
  }
}
