#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool pgo_valid_bar(double high, double low, double close, double source) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(source) && high >= low;
}

extern "C" __global__ void pretty_good_oscillator_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ source,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (length <= 0) {
        return;
    }

    double alpha = 1.0 / static_cast<double>(length);
    double prev_close = CUDART_NAN;
    double atr = CUDART_NAN;
    double warm_sum_tr = 0.0;
    int valid_seen = 0;
    bool atr_seeded = false;

    for (int t = 0; t < len; ++t) {
        if (!pgo_valid_bar(high[t], low[t], close[t], source[t])) {
            continue;
        }

        double tr;
        if (isnan(prev_close)) {
            tr = high[t] - low[t];
        } else {
            double up = high[t] > prev_close ? high[t] : prev_close;
            double dn = low[t] < prev_close ? low[t] : prev_close;
            tr = up - dn;
        }
        prev_close = close[t];
        valid_seen += 1;

        if (!atr_seeded) {
            warm_sum_tr += tr;
            if (valid_seen < length) {
                continue;
            }
            atr = warm_sum_tr * alpha;
            atr_seeded = true;
        } else {
            atr = atr + alpha * (tr - atr);
        }

        double sma_sum = 0.0;
        int count = 0;
        for (int j = t; j >= 0 && count < length; --j) {
            if (pgo_valid_bar(high[j], low[j], close[j], source[j])) {
                sma_sum += source[j];
                count += 1;
            }
        }

        if (count < length) {
            continue;
        }

        double sma = sma_sum * alpha;
        row[t] = atr != 0.0 ? (source[t] - sma) / atr : CUDART_NAN;
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * driver      : pretty_good_oscillator_into_slice,
//                   src/indicators/pretty_good_oscillator.rs:402-456. It picks
//                   between TWO bodies and BOTH are implemented below, because
//                   they are not the same arithmetic:
//       - clean   : pgo_compute_fast (:290-330) -- a SLIDING source sum plus a
//                   Wilder ATR seeded from a plain sum;
//       - dirty   : SmaStream::update (sma.rs:598-622) + AtrStream::update
//                   (atr.rs:788-826), which SKIP invalid bars entirely instead
//                   of resetting, so the two paths disagree on any frame with a
//                   hole.
//   * predicate   : is_fast_path_clean, :274-288.
//   * refusals    : pgo_prepare, :330-376.
//   * emitted col : the single `values` series -- expect_value_output
//                   (cpu_batch.rs:11730).
//   * PERIOD-INVARIANT: the batch reads `source` and `length` (14) and never
//                   `period` (cpu_batch.rs:11743-11744).
//
// FIRST-VALID IGNORED, and that is a decision rather than a shortcut. The CPU
// rule (is_valid_bar, :244-246) is "high, low, close and source all finite AND
// high >= low" -- an ORDERING condition no variant of F64FirstValidRule
// expresses, and the `high >= low` half genuinely names a different bar on a
// frame with a crossed quote. Rather than add a variant every consumer would
// have to grow a field for, the kernel derives its own index, exactly as
// `garman_klass_volatility` already does in this lane, and declares the
// caller's value unused.
//
// SOURCE: the CPU default source is `close` (cpu_batch.rs:11743), so the Hlc
// triple carries everything this kernel reads and there is no fourth series.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. `fma()` is Rust's `mul_add` -- ONE rounding, matching :319 and
// atr.rs:824. The plain `*`/`+` elsewhere stay two roundings, which is what
// -fmad=false guarantees.
// ===========================================================================

#define PGO_NEO_LENGTH 14

static __device__ __forceinline__ double pgo_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// is_valid_bar, :244-246. source == close under the CPU default.
static __device__ __forceinline__ bool pgo_neo_valid(double h, double l, double c) {
  return isfinite(h) && isfinite(l) && isfinite(c) && h >= l;
}

// true_range, :254-271. Note the CPU uses `>`/`<` comparisons here, NOT
// f64::max/min, and it is safe to transliterate them because this branch only
// runs on bars is_valid_bar already accepted -- no NaN can reach it.
static __device__ __forceinline__ double pgo_neo_tr(const double* __restrict__ high,
                                                    const double* __restrict__ low,
                                                    const double* __restrict__ close,
                                                    int first, int i) {
  if (i == first) return high[i] - low[i];
  const double pc = close[i - 1];
  const double up = (high[i] > pc) ? high[i] : pc;
  const double dn = (low[i] < pc) ? low[i] : pc;
  return up - dn;
}

extern "C" __global__
void pretty_good_oscillator_neo_batch_f64(const double* __restrict__ high,
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
  const double nn = pgo_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int length = PGO_NEO_LENGTH;
  if (length > n) return;  // :353 InvalidLength

  int first = -1;
  for (int i = 0; i < n; ++i) {
    if (pgo_neo_valid(high[i], low[i], close[i])) { first = i; break; }
  }
  if (first < 0) return;              // :360 AllValuesNaN
  if (n - first < length) return;     // :362 NotEnoughValidData

  bool clean = true;
  for (int i = first; i < n; ++i) {
    if (!pgo_neo_valid(high[i], low[i], close[i])) { clean = false; break; }
  }

  const int warmup = first + length - 1;
  const double inv = 1.0 / (double)length;

  if (clean) {
    // pgo_compute_fast, :290-330.
    const double alpha = 1.0 / (double)length;
    double sum_source = 0.0, sum_tr = 0.0;
    for (int i = first; i <= warmup; ++i) {
      sum_source += close[i];
      sum_tr += pgo_neo_tr(high, low, close, first, i);
    }
    double sma = sum_source * inv;
    double atr = sum_tr * inv;
    row[warmup] = (atr != 0.0) ? ((close[warmup] - sma) / atr) : nn;

    for (int i = warmup + 1; i < n; ++i) {
      sum_source += close[i] - close[i - length];
      sma = sum_source * inv;
      const double tr = pgo_neo_tr(high, low, close, first, i);
      atr = fma(alpha, tr - atr, atr);   // :319, ONE rounding
      row[i] = (atr != 0.0) ? ((close[i] - sma) / atr) : nn;
    }
    return;
  }

  // The stream path, :429-454. SmaStream keeps a ring; AtrStream keeps a Wilder
  // rma seeded from a plain sum. NEITHER resets on an invalid bar -- the CPU
  // simply does not feed it, which is why the two paths diverge.
  double sbuf[PGO_NEO_LENGTH];
  for (int k = 0; k < length; ++k) sbuf[k] = 0.0;
  int shead = 0, scount = 0;
  double ssum = 0.0;

  double prev_close = nn;
  double rma = nn, warm_sum = 0.0;
  int warm_count = 0;
  bool seeded = false;

  for (int i = 0; i < n; ++i) {
    if (!pgo_neo_valid(high[i], low[i], close[i])) { row[i] = nn; continue; }

    // SmaStream::update, sma.rs:598-622.
    bool have_sma = false;
    double sma = 0.0;
    if (length == 1) {
      ssum = close[i]; sbuf[0] = close[i]; scount = 1;
      sma = close[i]; have_sma = true;
    } else if (scount < length) {
      ssum += close[i];
      sbuf[shead] = close[i];
      shead += 1; if (shead == length) shead = 0;
      scount += 1;
      if (scount == length) { sma = ssum * inv; have_sma = true; }
    } else {
      const double old = sbuf[shead];
      ssum += close[i] - old;
      sbuf[shead] = close[i];
      shead += 1; if (shead == length) shead = 0;
      sma = ssum * inv; have_sma = true;
    }

    // AtrStream::update, atr.rs:788-826.
    double tr;
    if (isnan(prev_close)) {
      tr = high[i] - low[i];
    } else {
      const double up = (high[i] > prev_close) ? high[i] : prev_close;
      const double dn = (low[i] < prev_close) ? low[i] : prev_close;
      tr = up - dn;
    }
    prev_close = close[i];

    bool have_atr = false;
    double atr = 0.0;
    if (!seeded) {
      warm_sum += tr;
      warm_count += 1;
      if (warm_count == length) {
        rma = warm_sum * (1.0 / (double)length);
        seeded = true;
        atr = rma; have_atr = true;
      }
    } else {
      rma = fma(1.0 / (double)length, tr - rma, rma);   // atr.rs:824
      atr = rma; have_atr = true;
    }

    row[i] = (have_sma && have_atr && atr != 0.0) ? ((close[i] - sma) / atr) : nn;
  }
}
