// market_structure_confluence — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line: extern "C" __global__ void market_structure_confluence_batch_f64() {}
// plus a wrapper that resolved the empty symbol, computed all SIXTEEN output
// series on the host, and uploaded them.
//
// CPU REFERENCE
// -------------
//   src/indicators/market_structure_confluence.rs
//     :331 AtrState      :389 RollingSma      :442 WmaState
//     :501 PivotDetector :561 MarketStructureConfluenceCore
//     :617 Core::update  <- the per-bar body
//    :1144 compute_into_slices <- the per-row loop
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW with a small local state: a Wilder ATR, a
// rolling mean of it, a weighted moving average of close, two pivot detectors
// over `2 * swing_size + 1` bars, and four booleans of market structure. Every
// one of those carries across bars.
//
// TWO THINGS THAT LOOK LIKE DETAILS AND ARE NOT
// ---------------------------------------------
// * `AtrState::update` (:360) seeds with a SIMPLE mean of the first `period`
//   true ranges and then runs `((prev * (period - 1)) + tr) / period` — THREE
//   roundings, not a `mul_add`. Written that way here on purpose: collapsing it
//   to one `fma` is a different number, and the band it feeds is compared
//   against `high`/`low` to raise an arrow.
// * `Core::update` (:617) computes `basis` and `svol` FIRST but returns `None`
//   at the END (:721), AFTER the pivot and break bookkeeping has already run
//   and `self.index` has advanced. So a warmup bar still mutates state. The
//   ordering below is the reference's, not a tidier one.
//
// ARITHMETIC
// ----------
// f64 throughout; in `F64_LANE_SOURCES`, never `--use_fast_math`. `fmax` where
// the CPU writes `f64::max` in the true-range (:365) — `hl.max(hc).max(lc)`
// returns the non-NaN operand and an if-chain does not.

#include <cmath>
#include <cstdint>

#define MSC_BOS_CANDLE_CLOSE 0
#define MSC_BOS_WICKS        1

__device__ __forceinline__ double msc_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void market_structure_confluence_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ swing_sizes,
    const int* __restrict__ bos_confirmations,
    const int* __restrict__ basis_lengths,
    const int* __restrict__ atr_lengths,
    const int* __restrict__ atr_smooths,
    const double* __restrict__ vol_mults,
    int rows,
    int slots,
    int basis_cap,
    int smooth_cap,
    int pivot_cap,
    double* scratch,
    int* iscratch,
    double* __restrict__ out_basis,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower,
    double* __restrict__ out_direction,
    double* __restrict__ out_bull_arrow,
    double* __restrict__ out_bear_arrow,
    double* __restrict__ out_bull_change,
    double* __restrict__ out_bear_change,
    double* __restrict__ out_hh,
    double* __restrict__ out_lh,
    double* __restrict__ out_hl,
    double* __restrict__ out_ll,
    double* __restrict__ out_bull_bos,
    double* __restrict__ out_bull_choch,
    double* __restrict__ out_bear_bos,
    double* __restrict__ out_bear_choch
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = msc_qnan();
    size_t doubles_per_slot = static_cast<size_t>(basis_cap) +
                              static_cast<size_t>(smooth_cap) +
                              2ull * static_cast<size_t>(pivot_cap);
    double* base = scratch + static_cast<size_t>(slot) * doubles_per_slot;
    double* wma_buf = base;
    double* sma_buf = wma_buf + basis_cap;
    double* piv_high_buf = sma_buf + smooth_cap;
    double* piv_low_buf = piv_high_buf + pivot_cap;
    int* piv_high_idx = iscratch + static_cast<size_t>(slot) * 2ull * static_cast<size_t>(pivot_cap);
    int* piv_low_idx = piv_high_idx + pivot_cap;

    for (int row = slot; row < rows; row += slots) {
        int swing_size = swing_sizes[row];
        int bos = bos_confirmations[row];
        int basis_length = basis_lengths[row];
        int atr_length = atr_lengths[row];
        int atr_smooth = atr_smooths[row];
        double vol_mult = vol_mults[row];

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* o_basis = out_basis + row_base;
        double* o_upper = out_upper + row_base;
        double* o_lower = out_lower + row_base;
        double* o_dir = out_direction + row_base;
        double* o_bull_arrow = out_bull_arrow + row_base;
        double* o_bear_arrow = out_bear_arrow + row_base;
        double* o_bull_change = out_bull_change + row_base;
        double* o_bear_change = out_bear_change + row_base;
        double* o_hh = out_hh + row_base;
        double* o_lh = out_lh + row_base;
        double* o_hl = out_hl + row_base;
        double* o_ll = out_ll + row_base;
        double* o_bull_bos = out_bull_bos + row_base;
        double* o_bull_choch = out_bull_choch + row_base;
        double* o_bear_bos = out_bear_bos + row_base;
        double* o_bear_choch = out_bear_choch + row_base;

        // The CPU pre-fills NaN and then only writes bars where `update`
        // returns `Some` (:1207). Filling the WHOLE row is that behaviour with
        // the warmup-only fast path (:1163) folded in — the fast path leaves
        // post-warmup misses uninitialised, which is a bug, not a contract.
        for (int i = 0; i < len; ++i) {
            o_basis[i] = nan_value;
            o_upper[i] = nan_value;
            o_lower[i] = nan_value;
            o_dir[i] = nan_value;
            o_bull_arrow[i] = nan_value;
            o_bear_arrow[i] = nan_value;
            o_bull_change[i] = nan_value;
            o_bear_change[i] = nan_value;
            o_hh[i] = nan_value;
            o_lh[i] = nan_value;
            o_hl[i] = nan_value;
            o_ll[i] = nan_value;
            o_bull_bos[i] = nan_value;
            o_bull_choch[i] = nan_value;
            o_bear_bos[i] = nan_value;
            o_bear_choch[i] = nan_value;
        }

        // WmaState (:442)
        int wma_pos = 0, wma_len = 0;
        double wma_sum = 0.0, wma_weighted = 0.0;
        double wma_divisor =
            static_cast<double>(basis_length) * (static_cast<double>(basis_length) + 1.0) * 0.5;
        // AtrState (:331)
        int atr_count = 0;
        double atr_sum = 0.0, atr_value = 0.0, atr_prev_close = 0.0;
        bool atr_has_value = false, atr_has_prev_close = false;
        // RollingSma (:389)
        int sma_head = 0, sma_len = 0;
        double sma_sum = 0.0;
        // PivotDetector (:501) — a FIFO of `2*swing + 1` (value, index) pairs.
        int needed = 2 * swing_size + 1;
        int ph_head = 0, ph_len = 0, pl_head = 0, pl_len = 0;

        int index = 0;
        double prev_high = 0.0, prev_low = 0.0;
        bool has_prev_high = false, has_prev_low = false;
        bool high_active = false, low_active = false;
        int prev_break_dir = 0;

        for (int i = 0; i < len; ++i) {
            double h = high[i];
            double l = low[i];
            double c = close[i];

            // --- WmaState::update (:475) ------------------------------------
            bool have_basis = false;
            double basis = 0.0;
            if (wma_len < basis_length) {
                wma_buf[wma_pos] = c;
                wma_pos = (wma_pos + 1) % basis_length;
                wma_len += 1;
                wma_sum += c;
                wma_weighted += static_cast<double>(wma_len) * c;
                if (wma_len == basis_length) {
                    basis = wma_weighted / wma_divisor;
                    have_basis = true;
                }
            } else {
                double old = wma_buf[wma_pos];
                double old_sum = wma_sum;
                wma_buf[wma_pos] = c;
                wma_pos = (wma_pos + 1) % basis_length;
                wma_weighted = wma_weighted - old_sum + static_cast<double>(basis_length) * c;
                wma_sum = old_sum - old + c;
                basis = wma_weighted / wma_divisor;
                have_basis = true;
            }

            // --- AtrState::update (:360) then RollingSma (:412) -------------
            bool have_svol = false;
            double svol = 0.0;
            {
                double tr;
                if (atr_has_prev_close) {
                    double hl = h - l;
                    double hc = fabs(h - atr_prev_close);
                    double lc = fabs(l - atr_prev_close);
                    tr = fmax(fmax(hl, hc), lc);
                } else {
                    tr = h - l;
                }
                atr_prev_close = c;
                atr_has_prev_close = true;

                bool have_atr = false;
                double atr_out = 0.0;
                if (atr_has_value) {
                    // THREE roundings, exactly as the CPU writes it (:373).
                    double next =
                        ((atr_value * (static_cast<double>(atr_length) - 1.0)) + tr) /
                        static_cast<double>(atr_length);
                    atr_value = next;
                    atr_out = next;
                    have_atr = true;
                } else {
                    atr_count += 1;
                    atr_sum += tr;
                    if (atr_count == atr_length) {
                        double seeded = atr_sum / static_cast<double>(atr_length);
                        atr_value = seeded;
                        atr_has_value = true;
                        atr_out = seeded;
                        have_atr = true;
                    }
                }

                if (have_atr) {
                    if (sma_len < atr_smooth) {
                        sma_buf[sma_len] = atr_out;
                        sma_len += 1;
                        sma_sum += atr_out;
                        if (sma_len == atr_smooth) {
                            svol = sma_sum / static_cast<double>(atr_smooth);
                            have_svol = true;
                        }
                    } else {
                        double old = sma_buf[sma_head];
                        sma_buf[sma_head] = atr_out;
                        sma_sum += atr_out - old;
                        sma_head += 1;
                        if (sma_head == atr_smooth) {
                            sma_head = 0;
                        }
                        svol = sma_sum / static_cast<double>(atr_smooth);
                        have_svol = true;
                    }
                }
            }

            double hh = 0.0, lh = 0.0, hl_flag = 0.0, ll = 0.0;

            // --- PivotDetector::update, high side (:527) --------------------
            {
                int tail = ph_head + ph_len;
                if (tail >= pivot_cap) {
                    tail -= pivot_cap;
                }
                piv_high_buf[tail] = h;
                piv_high_idx[tail] = index;
                ph_len += 1;
                if (ph_len >= needed) {
                    int centre = ph_head + swing_size;
                    if (centre >= pivot_cap) {
                        centre -= pivot_cap;
                    }
                    double centre_value = piv_high_buf[centre];
                    int centre_index = piv_high_idx[centre];
                    bool ok = isfinite(centre_value);
                    if (ok) {
                        for (int k = 0; k < ph_len; ++k) {
                            if (k == swing_size) {
                                continue;
                            }
                            int pos = ph_head + k;
                            if (pos >= pivot_cap) {
                                pos -= pivot_cap;
                            }
                            double other = piv_high_buf[pos];
                            if (!isfinite(other) || other > centre_value) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    ph_head += 1;
                    if (ph_head == pivot_cap) {
                        ph_head = 0;
                    }
                    ph_len -= 1;
                    if (ok) {
                        bool is_hh = has_prev_high ? (centre_value >= prev_high) : true;
                        if (is_hh) {
                            hh = 1.0;
                        } else {
                            lh = 1.0;
                        }
                        prev_high = centre_value;
                        has_prev_high = true;
                        high_active = true;
                        (void)centre_index;
                    }
                }
            }

            // --- PivotDetector::update, low side ----------------------------
            {
                int tail = pl_head + pl_len;
                if (tail >= pivot_cap) {
                    tail -= pivot_cap;
                }
                piv_low_buf[tail] = l;
                piv_low_idx[tail] = index;
                pl_len += 1;
                if (pl_len >= needed) {
                    int centre = pl_head + swing_size;
                    if (centre >= pivot_cap) {
                        centre -= pivot_cap;
                    }
                    double centre_value = piv_low_buf[centre];
                    int centre_index = piv_low_idx[centre];
                    bool ok = isfinite(centre_value);
                    if (ok) {
                        for (int k = 0; k < pl_len; ++k) {
                            if (k == swing_size) {
                                continue;
                            }
                            int pos = pl_head + k;
                            if (pos >= pivot_cap) {
                                pos -= pivot_cap;
                            }
                            double other = piv_low_buf[pos];
                            if (!isfinite(other) || other < centre_value) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    pl_head += 1;
                    if (pl_head == pivot_cap) {
                        pl_head = 0;
                    }
                    pl_len -= 1;
                    if (ok) {
                        bool is_hl = has_prev_low ? (centre_value >= prev_low) : true;
                        if (is_hl) {
                            hl_flag = 1.0;
                        } else {
                            ll = 1.0;
                        }
                        prev_low = centre_value;
                        has_prev_low = true;
                        low_active = true;
                        (void)centre_index;
                    }
                }
            }

            double high_src = (bos == MSC_BOS_CANDLE_CLOSE) ? c : h;
            double low_src = (bos == MSC_BOS_CANDLE_CLOSE) ? c : l;

            bool high_broken = false, low_broken = false;
            if (high_active && has_prev_high && high_src > prev_high) {
                high_broken = true;
                high_active = false;
            }
            if (low_active && has_prev_low && low_src < prev_low) {
                low_broken = true;
                low_active = false;
            }

            double bullish_change = 0.0, bearish_change = 0.0;
            double bullish_bos = 0.0, bullish_choch = 0.0;
            double bearish_bos = 0.0, bearish_choch = 0.0;

            if (high_broken) {
                int last = prev_break_dir;
                if (last == -1) {
                    bullish_choch = 1.0;
                } else {
                    bullish_bos = 1.0;
                }
                if (last == -1 || last == 0) {
                    bullish_change = 1.0;
                }
                prev_break_dir = 1;
            }
            if (low_broken) {
                int last = prev_break_dir;
                if (last == 1) {
                    bearish_choch = 1.0;
                } else {
                    bearish_bos = 1.0;
                }
                if (last == 1 || last == 0) {
                    bearish_change = 1.0;
                }
                prev_break_dir = -1;
            }

            index += 1;

            // The `None` return is here, AFTER all the state above has been
            // mutated (:721).
            if (!have_basis || !have_svol) {
                continue;
            }

            double upper_band = basis + vol_mult * svol;
            double lower_band = basis - vol_mult * svol;

            o_basis[i] = basis;
            o_upper[i] = upper_band;
            o_lower[i] = lower_band;
            o_dir[i] = static_cast<double>(prev_break_dir);
            o_bull_arrow[i] =
                (prev_break_dir == 1 && l < lower_band && h > lower_band) ? 1.0 : 0.0;
            o_bear_arrow[i] =
                (prev_break_dir == -1 && l < upper_band && h > upper_band) ? 1.0 : 0.0;
            o_bull_change[i] = bullish_change;
            o_bear_change[i] = bearish_change;
            o_hh[i] = hh;
            o_lh[i] = lh;
            o_hl[i] = hl_flag;
            o_ll[i] = ll;
            o_bull_bos[i] = bullish_bos;
            o_bull_choch[i] = bullish_choch;
            o_bear_bos[i] = bearish_bos;
            o_bear_choch[i] = bearish_choch;
        }
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// The entry point above, `market_structure_confluence_batch_f64`, is a full
// all-double port of src/indicators/market_structure_confluence.rs written by
// this same workflow -- but its ABI is the CRATE's batch ABI: six parameter
// arrays, a double `scratch` AND an int `iscratch` pointer, and SIXTEEN output
// matrices. The f64 lane launches exactly one shape,
//     (high, low, close, n, periods, n_combos, first_valid, out)
// so a variant pointing at that symbol would read the stack. This is the
// lane-shaped entry point.
//
// CPU reference:
//   * arithmetic  : WmaState::update (market_structure_confluence.rs:475),
//                   AtrState::update (:360) and RollingSma::update (:412) --
//                   transliterated here from the already-ported bodies above,
//                   line for line and rounding for rounding.
//   * gate        : the `None` return at :721, reproduced above at :419 --
//                   basis is emitted only once BOTH the basis WMA and the
//                   smoothed ATR exist.
//   * emitted col : `basis`. compute_market_structure_confluence_batch
//                   (cpu_batch.rs:16090) accepts "basis", "upper_band",
//                   "lower_band", ... and there is NO "value" arm
//                   (:16145 onwards), so a parity run must ask the CPU for
//                   "basis" explicitly.
//   * PERIOD-INVARIANT: the batch reads swing_size (10), bos_confirmation,
//                   basis_length (100), atr_length (14), atr_smooth (21) and
//                   vol_mult (2.0) -- never `period`
//                   (cpu_batch.rs:16101-16115).
//   * FIRST-VALID IGNORED: the CPU stream walks from index 0 and consults no
//                   first-valid index at all.
//
// WHAT THIS ENTRY POINT DROPS, AND WHY IT NEEDS NO `scratch`. `basis` is the
// weighted moving average of close over `basis_length`, gated on the smoothed
// ATR being ready. It does NOT depend on the pivot detector, the break-of-
// structure state machine, or the four structure flags -- those feed the OTHER
// fifteen outputs. Dropping them removes the two `pivot_cap` FIFOs and the
// whole int scratch, leaving 100 + 21 doubles of per-thread state, bounded at
// COMPILE TIME. That is what turns a scratch-pointer kernel into a lane kernel.
//
// NaN: the true range uses fmax(fmax(hl, hc), lc) (:213) because the CPU uses
// f64::max, which returns the NON-NaN operand; an if-chain would let a NaN into
// the Wilder recurrence and poison every later bar.
// ===========================================================================

#define MSC_NEO_BASIS_LENGTH 100
#define MSC_NEO_ATR_LENGTH 14
#define MSC_NEO_ATR_SMOOTH 21

extern "C" __global__ void market_structure_confluence_neo_batch_f64(
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
  const double nan_value = msc_qnan();
  for (int i = 0; i < n; ++i) row[i] = nan_value;

  const int basis_length = MSC_NEO_BASIS_LENGTH;
  const int atr_length = MSC_NEO_ATR_LENGTH;
  const int atr_smooth = MSC_NEO_ATR_SMOOTH;

  double wma_buf[MSC_NEO_BASIS_LENGTH];
  double sma_buf[MSC_NEO_ATR_SMOOTH];
  for (int k = 0; k < basis_length; ++k) wma_buf[k] = 0.0;
  for (int k = 0; k < atr_smooth; ++k) sma_buf[k] = 0.0;

  int wma_pos = 0, wma_len = 0;
  double wma_sum = 0.0, wma_weighted = 0.0;
  const double wma_divisor =
      static_cast<double>(basis_length) * (static_cast<double>(basis_length) + 1.0) * 0.5;

  int atr_count = 0;
  double atr_sum = 0.0, atr_value = 0.0, atr_prev_close = 0.0;
  bool atr_has_value = false, atr_has_prev_close = false;

  int sma_head = 0, sma_len = 0;
  double sma_sum = 0.0;

  for (int i = 0; i < n; ++i) {
    const double h = high[i], l = low[i], c = close[i];

    // WmaState::update (:475)
    bool have_basis = false;
    double basis = 0.0;
    if (wma_len < basis_length) {
      wma_buf[wma_pos] = c;
      wma_pos = (wma_pos + 1) % basis_length;
      wma_len += 1;
      wma_sum += c;
      wma_weighted += static_cast<double>(wma_len) * c;
      if (wma_len == basis_length) { basis = wma_weighted / wma_divisor; have_basis = true; }
    } else {
      const double old = wma_buf[wma_pos];
      const double old_sum = wma_sum;
      wma_buf[wma_pos] = c;
      wma_pos = (wma_pos + 1) % basis_length;
      wma_weighted = wma_weighted - old_sum + static_cast<double>(basis_length) * c;
      wma_sum = old_sum - old + c;
      basis = wma_weighted / wma_divisor;
      have_basis = true;
    }

    // AtrState::update (:360) then RollingSma::update (:412)
    bool have_svol = false;
    {
      double tr;
      if (atr_has_prev_close) {
        const double hl = h - l;
        const double hc = fabs(h - atr_prev_close);
        const double lc = fabs(l - atr_prev_close);
        tr = fmax(fmax(hl, hc), lc);
      } else {
        tr = h - l;
      }
      atr_prev_close = c;
      atr_has_prev_close = true;

      bool have_atr = false;
      double atr_out = 0.0;
      if (atr_has_value) {
        // THREE roundings, exactly as the CPU writes it (:373).
        const double next =
            ((atr_value * (static_cast<double>(atr_length) - 1.0)) + tr) /
            static_cast<double>(atr_length);
        atr_value = next; atr_out = next; have_atr = true;
      } else {
        atr_count += 1;
        atr_sum += tr;
        if (atr_count == atr_length) {
          const double seeded = atr_sum / static_cast<double>(atr_length);
          atr_value = seeded; atr_has_value = true; atr_out = seeded; have_atr = true;
        }
      }

      if (have_atr) {
        if (sma_len < atr_smooth) {
          // The fill branch writes at sma_len and leaves sma_head at 0 (:243).
          sma_buf[sma_len] = atr_out;
          sma_len += 1;
          sma_sum += atr_out;
          if (sma_len == atr_smooth) have_svol = true;
        } else {
          const double old = sma_buf[sma_head];
          sma_buf[sma_head] = atr_out;
          sma_sum += atr_out - old;
          sma_head += 1;
          if (sma_head == atr_smooth) sma_head = 0;
          have_svol = true;
        }
      }
    }

    if (!have_basis || !have_svol) continue;
    row[i] = basis;
  }
}
