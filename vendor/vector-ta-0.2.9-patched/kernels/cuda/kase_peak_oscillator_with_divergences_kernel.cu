// kase_peak_oscillator_with_divergences — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line: extern "C" __global__ void
//           kase_peak_oscillator_with_divergences_batch_f64() {}
// plus the now-deleted wrapper that resolved the empty symbol, computed all
// eleven public-library series on the host, and uploaded them.
//
// CPU REFERENCE
// -------------
//   src/indicators/kase_peak_oscillator_with_divergences.rs
//     :440 RollingSma        :495 RollingStd
//     :853 main_warmup       :869 in_range
//     :874 is_pivot_low      :895 is_pivot_high
//     :594 Stream::from_resolved  (the `roots` table)
//     :633 Stream::update    <- the per-bar body
//    :1121 row_from_slices_resolved <- the per-row loop
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW walking bars ascending. Six ring-buffer
// accumulators carry state (a 9-bar log-return deviation, a 30-bar mean of it,
// two 3-bar means, a 50-bar mean and a 50-bar deviation), plus a divergence
// detector that needs the FULL oscillator history because a pivot found now is
// compared against a pivot up to `range_upper` bars back.
//
// The history arrays are indexed by the STREAM's own counter, not by bar index:
// `reset()` (:625) clears them on an invalid bar, so a hole restarts the
// numbering. `hist_len` below is that counter, and every history read goes
// through it.
//
// ARITHMETIC
// ----------
// f64 throughout; in `F64_LANE_SOURCES`, so never `--use_fast_math`. `fmax`/
// `fmin` where the CPU writes `f64::max`/`f64::min` (:770-772, :548) — those
// return the non-NaN operand and a comparison chain does not, which matters at
// :548 where `variance.max(0.0)` is the guard against a negative variance from
// catastrophic cancellation. No `mul_add` appears in the reference, so no
// `fma()` appears here.

#include <cmath>
#include <cstdint>

// The six accumulator windows are FIXED in the CPU reference (:602-607) — they
// are not parameters. Sizes named rather than spelled inline so the scratch
// layout below can be checked against them.
#define KPO_CC_DEV_N   9
#define KPO_AVG_N     30
#define KPO_X1_N       3
#define KPO_XS_N       3
#define KPO_XP_N      50
#define KPO_RING_TOTAL (KPO_CC_DEV_N + KPO_AVG_N + KPO_X1_N + KPO_XS_N + KPO_XP_N + KPO_XP_N)

__device__ __forceinline__ double kpo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// RollingSma (:440) / RollingStd (:495) share this state; `sumsq` is unused by
// the SMA form.
struct KpoRoll {
    double* values;
    int period;
    int idx;
    int count;
    double sum;
    double sumsq;
};

__device__ __forceinline__ void kpo_roll_init(KpoRoll* r, double* buf, int period) {
    r->values = buf;
    r->period = period < 1 ? 1 : period;
    r->idx = 0;
    r->count = 0;
    r->sum = 0.0;
    r->sumsq = 0.0;
    for (int i = 0; i < r->period; ++i) {
        r->values[i] = 0.0;
    }
}

__device__ __forceinline__ void kpo_roll_reset(KpoRoll* r) {
    // The CPU's `reset` (:465, :516) zeroes the counters but NOT the buffer,
    // which is correct because `count` gates every read of it.
    r->idx = 0;
    r->count = 0;
    r->sum = 0.0;
    r->sumsq = 0.0;
}

__device__ __forceinline__ int kpo_sma_update(KpoRoll* r, double value, double* out) {
    if (!isfinite(value)) {
        kpo_roll_reset(r);
        return 0;
    }
    if (r->count < r->period) {
        r->values[r->idx] = value;
        r->sum += value;
        r->count += 1;
    } else {
        double old = r->values[r->idx];
        r->values[r->idx] = value;
        r->sum += value - old;
    }
    r->idx += 1;
    if (r->idx == r->period) {
        r->idx = 0;
    }
    if (r->count == r->period) {
        *out = r->sum / static_cast<double>(r->period);
        return 1;
    }
    return 0;
}

__device__ __forceinline__ int kpo_std_update(KpoRoll* r, double value, double* out) {
    if (!isfinite(value)) {
        kpo_roll_reset(r);
        return 0;
    }
    if (r->count < r->period) {
        r->values[r->idx] = value;
        r->sum += value;
        r->sumsq += value * value;
        r->count += 1;
    } else {
        double old = r->values[r->idx];
        r->values[r->idx] = value;
        r->sum += value - old;
        r->sumsq += value * value - old * old;
    }
    r->idx += 1;
    if (r->idx == r->period) {
        r->idx = 0;
    }
    if (r->count == r->period) {
        double n = static_cast<double>(r->period);
        double mean = r->sum / n;
        double variance = (r->sumsq / n) - mean * mean;
        *out = sqrt(fmax(variance, 0.0));
        return 1;
    }
    return 0;
}

// is_pivot_low (:874) / is_pivot_high (:895) over the stream's own history.
__device__ __forceinline__ bool kpo_is_pivot_low(
    const double* values, int pivot_idx, int lb_l, int lb_r) {
    double pivot = values[pivot_idx];
    if (!isfinite(pivot)) {
        return false;
    }
    for (int i = pivot_idx - lb_l; i <= pivot_idx + lb_r; ++i) {
        if (i != pivot_idx) {
            double v = values[i];
            if (!isfinite(v) || v < pivot) {
                return false;
            }
        }
    }
    return true;
}

__device__ __forceinline__ bool kpo_is_pivot_high(
    const double* values, int pivot_idx, int lb_l, int lb_r) {
    double pivot = values[pivot_idx];
    if (!isfinite(pivot)) {
        return false;
    }
    for (int i = pivot_idx - lb_l; i <= pivot_idx + lb_r; ++i) {
        if (i != pivot_idx) {
            double v = values[i];
            if (!isfinite(v) || v > pivot) {
                return false;
            }
        }
    }
    return true;
}

extern "C" __global__ void kase_peak_oscillator_with_divergences_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ deviations,
    const int* __restrict__ short_cycles,
    const int* __restrict__ long_cycles,
    const double* __restrict__ sensitivities,
    int all_peaks_mode,
    int lb_r,
    int lb_l,
    int range_upper,
    int range_lower,
    int plot_bull,
    int plot_hidden_bull,
    int plot_bear,
    int plot_hidden_bear,
    int rows,
    int slots,
    int long_cycle_cap,
    double* scratch,
    double* __restrict__ out_oscillator,
    double* __restrict__ out_max_peak,
    double* __restrict__ out_min_peak,
    double* __restrict__ out_market_extreme,
    double* __restrict__ out_regular_bullish,
    double* __restrict__ out_hidden_bullish,
    double* __restrict__ out_regular_bearish,
    double* __restrict__ out_hidden_bearish,
    double* __restrict__ out_go_long,
    double* __restrict__ out_go_short
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = kpo_qnan();
    size_t per_slot = 3ull * static_cast<size_t>(len) +
                      static_cast<size_t>(long_cycle_cap) +
                      static_cast<size_t>(KPO_RING_TOTAL);
    double* base = scratch + static_cast<size_t>(slot) * per_slot;
    double* osc_history = base;
    double* high_history = osc_history + len;
    double* low_history = high_history + len;
    double* roots = low_history + len;
    double* rings = roots + long_cycle_cap;

    for (int row = slot; row < rows; row += slots) {
        double deviation = deviations[row];
        int short_cycle = short_cycles[row];
        int long_cycle = long_cycles[row];
        double sensitivity = sensitivities[row];

        // Stream::from_resolved (:594)
        for (int k = 0; k < long_cycle; ++k) {
            roots[k] = 1.0;
        }
        for (int k = short_cycle; k < long_cycle; ++k) {
            roots[k] = sqrt(static_cast<double>(k));
        }

        double* ring = rings;
        KpoRoll cc_dev, avg, x1_sma, xs_sma, xp_abs_sma, xp_abs_std;
        kpo_roll_init(&cc_dev, ring, KPO_CC_DEV_N);
        ring += KPO_CC_DEV_N;
        kpo_roll_init(&avg, ring, KPO_AVG_N);
        ring += KPO_AVG_N;
        kpo_roll_init(&x1_sma, ring, KPO_X1_N);
        ring += KPO_X1_N;
        kpo_roll_init(&xs_sma, ring, KPO_XS_N);
        ring += KPO_XS_N;
        kpo_roll_init(&xp_abs_sma, ring, KPO_XP_N);
        ring += KPO_XP_N;
        kpo_roll_init(&xp_abs_std, ring, KPO_XP_N);

        double prev_close = 0.0;
        bool has_prev_close = false;
        int hist_len = 0;
        double prev_osc_1 = 0.0, prev_osc_2 = 0.0;
        bool has_osc_1 = false, has_osc_2 = false;
        int last_pivot_low = -1;
        int last_pivot_high = -1;

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* o_osc = out_oscillator + row_base;
        double* o_maxp = out_max_peak + row_base;
        double* o_minp = out_min_peak + row_base;
        double* o_ext = out_market_extreme + row_base;
        double* o_rbull = out_regular_bullish + row_base;
        double* o_hbull = out_hidden_bullish + row_base;
        double* o_rbear = out_regular_bearish + row_base;
        double* o_hbear = out_hidden_bearish + row_base;
        double* o_long = out_go_long + row_base;
        double* o_short = out_go_short + row_base;

        for (int i = 0; i < len; ++i) {
            double h = high[i];
            double l = low[i];
            double c = close[i];

            // Every early return in `Stream::update` produces NaN across all
            // ten mathematically distinct production outputs. The scalar
            // library's histogram display alias is not allocated here.
            o_osc[i] = nan_value;
            o_maxp[i] = nan_value;
            o_minp[i] = nan_value;
            o_ext[i] = nan_value;
            o_rbull[i] = nan_value;
            o_hbull[i] = nan_value;
            o_rbear[i] = nan_value;
            o_hbear[i] = nan_value;
            o_long[i] = nan_value;
            o_short[i] = nan_value;

            if (!isfinite(h) || !isfinite(l) || !isfinite(c) || h <= 0.0 || l <= 0.0 ||
                c <= 0.0) {
                // reset (:625)
                has_prev_close = false;
                kpo_roll_reset(&cc_dev);
                kpo_roll_reset(&avg);
                kpo_roll_reset(&x1_sma);
                kpo_roll_reset(&xs_sma);
                kpo_roll_reset(&xp_abs_sma);
                kpo_roll_reset(&xp_abs_std);
                hist_len = 0;
                has_osc_1 = false;
                has_osc_2 = false;
                last_pivot_low = -1;
                last_pivot_high = -1;
                continue;
            }

            high_history[hist_len] = h;
            low_history[hist_len] = l;

            if (!has_prev_close || !(prev_close > 0.0)) {
                prev_close = c;
                has_prev_close = true;
                osc_history[hist_len] = nan_value;
                hist_len += 1;
                continue;
            }
            double cc_log = log(c / prev_close);
            prev_close = c;

            double cc_dev_value;
            if (!kpo_std_update(&cc_dev, cc_log, &cc_dev_value) || !isfinite(cc_dev_value)) {
                osc_history[hist_len] = nan_value;
                hist_len += 1;
                continue;
            }

            double avg_value;
            if (!kpo_sma_update(&avg, cc_dev_value, &avg_value) || !isfinite(avg_value) ||
                !(avg_value > 0.0)) {
                osc_history[hist_len] = nan_value;
                hist_len += 1;
                continue;
            }

            // `high_history.len()` at this point is `hist_len + 1` on the CPU,
            // because the push happened before the early returns.
            int current_hist_len = hist_len + 1;
            if (current_hist_len < long_cycle) {
                osc_history[hist_len] = nan_value;
                hist_len += 1;
                continue;
            }

            double max1 = 0.0;
            double maxs = 0.0;
            for (int k = short_cycle; k < long_cycle; ++k) {
                double past_low = low_history[current_hist_len - 1 - k];
                double past_high = high_history[current_hist_len - 1 - k];
                double root = roots[k];
                double v1 = log(h / past_low) / root;
                double vs = log(past_high / l) / root;
                if (isfinite(v1) && v1 > max1) {
                    max1 = v1;
                }
                if (isfinite(vs) && vs > maxs) {
                    maxs = vs;
                }
            }

            double x1_avg, xs_avg;
            int have_x1 = kpo_sma_update(&x1_sma, max1 / avg_value, &x1_avg);
            int have_xs = kpo_sma_update(&xs_sma, maxs / avg_value, &xs_avg);
            if (!have_x1 || !have_xs || !isfinite(x1_avg) || !isfinite(xs_avg)) {
                osc_history[hist_len] = nan_value;
                hist_len += 1;
                continue;
            }

            double oscillator = sensitivity * (x1_avg - xs_avg);
            if (!isfinite(oscillator)) {
                has_prev_close = false;
                kpo_roll_reset(&cc_dev);
                kpo_roll_reset(&avg);
                kpo_roll_reset(&x1_sma);
                kpo_roll_reset(&xs_sma);
                kpo_roll_reset(&xp_abs_sma);
                kpo_roll_reset(&xp_abs_std);
                hist_len = 0;
                has_osc_1 = false;
                has_osc_2 = false;
                last_pivot_low = -1;
                last_pivot_high = -1;
                continue;
            }
            osc_history[hist_len] = oscillator;
            hist_len += 1;

            double xp_abs = fabs(oscillator);
            double abs_avg, abs_std;
            int have_avg = kpo_sma_update(&xp_abs_sma, xp_abs, &abs_avg);
            int have_std = kpo_std_update(&xp_abs_std, xp_abs, &abs_std);

            double max_peak_value = nan_value;
            double min_peak_value = nan_value;
            double market_extreme = 0.0;
            double go_long = 0.0;
            double go_short = 0.0;

            if (have_avg && have_std) {
                double tmp_val = abs_avg + deviation * abs_std;
                double max_val = fmax(tmp_val, 90.0);
                double min_val = fmin(tmp_val, 90.0);
                if (oscillator > 0.0) {
                    max_peak_value = max_val;
                    min_peak_value = min_val;
                } else {
                    max_peak_value = -max_val;
                    min_peak_value = -min_val;
                }

                if (has_osc_1 && has_osc_2) {
                    double prev1 = prev_osc_1;
                    double prev2 = prev_osc_2;
                    if (all_peaks_mode) {
                        if (prev1 > 0.0 && prev1 > oscillator && prev1 >= prev2) {
                            market_extreme = oscillator;
                        }
                        if (prev1 < 0.0 && prev1 < oscillator && prev1 <= prev2) {
                            market_extreme = oscillator;
                        }
                    } else {
                        if (prev1 > 0.0 && prev1 > oscillator && prev1 >= prev2 &&
                            prev1 >= max_val) {
                            market_extreme = oscillator;
                        }
                        if (prev1 < 0.0 && prev1 < oscillator && prev1 <= prev2 &&
                            prev1 <= -max_val) {
                            market_extreme = oscillator;
                        }
                    }
                }
            }

            if (market_extreme < 0.0) {
                go_long = 1.0;
            } else if (market_extreme > 0.0) {
                go_short = 1.0;
            }

            prev_osc_2 = prev_osc_1;
            has_osc_2 = has_osc_1;
            prev_osc_1 = oscillator;
            has_osc_1 = true;

            double regular_bullish = nan_value;
            double hidden_bullish = nan_value;
            double regular_bearish = nan_value;
            double hidden_bearish = nan_value;

            int current_idx = hist_len - 1;
            if (current_idx >= lb_r) {
                int pivot_idx = current_idx - lb_r;
                if (pivot_idx >= lb_l) {
                    if (kpo_is_pivot_low(osc_history, pivot_idx, lb_l, lb_r)) {
                        if (last_pivot_low >= 0) {
                            int bars = pivot_idx - last_pivot_low;
                            if (range_lower <= bars && bars <= range_upper) {
                                double osc_now = osc_history[pivot_idx];
                                double osc_prev = osc_history[last_pivot_low];
                                double low_now = low_history[pivot_idx];
                                double low_prev = low_history[last_pivot_low];
                                if (plot_bull && low_now < low_prev && osc_now > osc_prev) {
                                    regular_bullish = osc_now;
                                }
                                if (plot_hidden_bull && low_now > low_prev &&
                                    osc_now < osc_prev) {
                                    hidden_bullish = osc_now;
                                }
                            }
                        }
                        last_pivot_low = pivot_idx;
                    }

                    if (kpo_is_pivot_high(osc_history, pivot_idx, lb_l, lb_r)) {
                        if (last_pivot_high >= 0) {
                            int bars = pivot_idx - last_pivot_high;
                            if (range_lower <= bars && bars <= range_upper) {
                                double osc_now = osc_history[pivot_idx];
                                double osc_prev = osc_history[last_pivot_high];
                                double high_now = high_history[pivot_idx];
                                double high_prev = high_history[last_pivot_high];
                                if (plot_bear && high_now > high_prev && osc_now < osc_prev) {
                                    regular_bearish = osc_now;
                                }
                                if (plot_hidden_bear && high_now < high_prev &&
                                    osc_now > osc_prev) {
                                    hidden_bearish = osc_now;
                                }
                            }
                        }
                        last_pivot_high = pivot_idx;
                    }
                }
            }

            o_osc[i] = oscillator;
            o_maxp[i] = max_peak_value;
            o_minp[i] = min_peak_value;
            o_ext[i] = market_extreme;
            o_rbull[i] = regular_bullish;
            o_hbull[i] = hidden_bullish;
            o_rbear[i] = regular_bearish;
            o_hbear[i] = hidden_bearish;
            o_long[i] = go_long;
            o_short[i] = go_short;
        }
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// The entry point above, `kase_peak_oscillator_with_divergences_batch_f64`, is
// a full all-double port of
// src/indicators/kase_peak_oscillator_with_divergences.rs written by this same
// workflow. Its production ABI takes four parameter arrays, eight divergence
// flags, a caller-allocated `scratch` pointer and ten mathematically distinct
// output matrices. The scalar library's
// `histogram` remains a display alias of `oscillator` and is deliberately not
// part of this production CUDA ABI. The f64 lane launches exactly one shape,
//     (high, low, close, n, periods, n_combos, first_valid, out)
// so a variant pointing at that symbol would read the stack. This entry point
// is the lane-shaped one, and it reuses the device helpers above -- kpo_qnan,
// kpo_roll_init/reset, kpo_sma_update, kpo_std_update -- so there is ONE
// implementation of the rolling arithmetic in this file, not two.
//
// CPU reference:
//   * arithmetic  : KasePeakOscillatorWithDivergencesStream::update,
//                   src/indicators/kase_peak_oscillator_with_divergences.rs
//                   (:594 onwards for the resolved stream, :625 for the reset).
//   * emitted col : `oscillator`.
//                   compute_kase_peak_oscillator_with_divergences_batch
//                   (cpu_batch.rs:13685) maps output_id "value" ->
//                   out.oscillator.
//   * PERIOD-INVARIANT: the batch reads deviations (2.0), short_cycle (8),
//                   long_cycle (65), sensitivity (40.0), all_peaks_mode, lb_r,
//                   lb_l, range_upper and range_lower -- never `period`
//                   (cpu_batch.rs:13695-13740).
//   * FIRST-VALID IGNORED: the stream walks from index 0 and RESETS on any bar
//                   whose high, low or close is non-finite OR non-positive.
//                   That validity rule -- finite AND strictly positive on
//                   THREE series -- is not expressible by any
//                   F64FirstValidRule variant, so the row declares Ignored
//                   rather than claiming a rule it never reads; the same
//                   choice `garman_klass_volatility` already makes in this
//                   lane.
//
// WHAT THIS ENTRY POINT DROPS, AND WHY IT NEEDS NO `scratch`. The batch kernel
// allocates 3 * len doubles per slot for `osc_history`, `high_history` and
// `low_history`, because the DIVERGENCE outputs scan the whole oscillator
// history for pivots. This lane emits `oscillator` alone, and the oscillator
// only ever reads back `long_cycle - 1` bars of high/low history
// (`current_hist_len - 1 - k`, k < long_cycle). So the two histories collapse
// into per-thread RINGS of `long_cycle` doubles and `osc_history` disappears
// entirely -- which is what turns an unbounded per-slot allocation into a
// COMPILE-TIME bounded kernel with no scratch pointer at all.
//
// The compaction semantics are preserved exactly: `hist_len` advances only on
// bars the CPU accepts and resets to 0 with the stream, so the ring holds the
// same compacted sequence the full array would.
//
// f64 END TO END: `log()` and `sqrt()`, never `logf`/`sqrtf`/`__logf`. The file
// is in build.rs::F64_LANE_SOURCES, so the two logarithms in the inner
// long_cycle loop are compiled without --use_fast_math.
// ===========================================================================

#define KPO_NEO_SHORT_CYCLE 8
#define KPO_NEO_LONG_CYCLE 65
#define KPO_NEO_SENSITIVITY 40.0

extern "C" __global__ void kase_peak_oscillator_with_divergences_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out) {
  const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (combo >= n_combos) return;
  (void)periods;      // PERIOD-INVARIANT -- see the header.
  (void)first_valid;  // FIRST-VALID IGNORED -- see the header.

  if (n <= 0) return;
  double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
  const double nan_value = kpo_qnan();

  const int short_cycle = KPO_NEO_SHORT_CYCLE;
  const int long_cycle = KPO_NEO_LONG_CYCLE;
  const double sensitivity = KPO_NEO_SENSITIVITY;

  // Stream::from_resolved (:594) -- the root table, built once.
  double roots[KPO_NEO_LONG_CYCLE];
  for (int k = 0; k < long_cycle; ++k) roots[k] = 1.0;
  for (int k = short_cycle; k < long_cycle; ++k) roots[k] = sqrt(static_cast<double>(k));

  double ring_store[KPO_RING_TOTAL];
  double* ring = ring_store;
  KpoRoll cc_dev, avg, x1_sma, xs_sma;
  kpo_roll_init(&cc_dev, ring, KPO_CC_DEV_N);   ring += KPO_CC_DEV_N;
  kpo_roll_init(&avg, ring, KPO_AVG_N);         ring += KPO_AVG_N;
  kpo_roll_init(&x1_sma, ring, KPO_X1_N);       ring += KPO_X1_N;
  kpo_roll_init(&xs_sma, ring, KPO_XS_N);

  // The two histories the OSCILLATOR needs, as rings rather than n-long arrays.
  double high_hist[KPO_NEO_LONG_CYCLE];
  double low_hist[KPO_NEO_LONG_CYCLE];
  for (int k = 0; k < long_cycle; ++k) { high_hist[k] = 0.0; low_hist[k] = 0.0; }

  double prev_close = 0.0;
  bool has_prev_close = false;
  int hist_len = 0;

  for (int i = 0; i < n; ++i) {
    const double h = high[i], l = low[i], c = close[i];

    // Every early return in Stream::update produces NaN (:1161).
    row[i] = nan_value;

    if (!isfinite(h) || !isfinite(l) || !isfinite(c) || h <= 0.0 || l <= 0.0 || c <= 0.0) {
      // reset (:625)
      has_prev_close = false;
      kpo_roll_reset(&cc_dev);
      kpo_roll_reset(&avg);
      kpo_roll_reset(&x1_sma);
      kpo_roll_reset(&xs_sma);
      hist_len = 0;
      continue;
    }

    high_hist[hist_len % long_cycle] = h;
    low_hist[hist_len % long_cycle] = l;

    if (!has_prev_close || !(prev_close > 0.0)) {
      prev_close = c;
      has_prev_close = true;
      hist_len += 1;
      continue;
    }
    const double cc_log = log(c / prev_close);
    prev_close = c;

    double cc_dev_value;
    if (!kpo_std_update(&cc_dev, cc_log, &cc_dev_value) || !isfinite(cc_dev_value)) {
      hist_len += 1;
      continue;
    }

    double avg_value;
    if (!kpo_sma_update(&avg, cc_dev_value, &avg_value) || !isfinite(avg_value) ||
        !(avg_value > 0.0)) {
      hist_len += 1;
      continue;
    }

    const int current_hist_len = hist_len + 1;
    if (current_hist_len < long_cycle) {
      hist_len += 1;
      continue;
    }

    double max1 = 0.0, maxs = 0.0;
    for (int k = short_cycle; k < long_cycle; ++k) {
      const int slot = (current_hist_len - 1 - k) % long_cycle;
      const double past_low = low_hist[slot];
      const double past_high = high_hist[slot];
      const double root = roots[k];
      const double v1 = log(h / past_low) / root;
      const double vs = log(past_high / l) / root;
      if (isfinite(v1) && v1 > max1) max1 = v1;
      if (isfinite(vs) && vs > maxs) maxs = vs;
    }

    double x1_avg, xs_avg;
    const int have_x1 = kpo_sma_update(&x1_sma, max1 / avg_value, &x1_avg);
    const int have_xs = kpo_sma_update(&xs_sma, maxs / avg_value, &xs_avg);
    if (!have_x1 || !have_xs || !isfinite(x1_avg) || !isfinite(xs_avg)) {
      hist_len += 1;
      continue;
    }

    const double oscillator = sensitivity * (x1_avg - xs_avg);
    if (!isfinite(oscillator)) {
      has_prev_close = false;
      kpo_roll_reset(&cc_dev);
      kpo_roll_reset(&avg);
      kpo_roll_reset(&x1_sma);
      kpo_roll_reset(&xs_sma);
      hist_len = 0;
      continue;
    }
    hist_len += 1;
    row[i] = oscillator;
  }
}
