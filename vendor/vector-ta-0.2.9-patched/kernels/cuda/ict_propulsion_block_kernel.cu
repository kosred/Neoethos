// ict_propulsion_block — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line:  extern "C" __global__ void ict_propulsion_block_batch_f64() {}
// plus a wrapper that resolved the empty symbol, computed all TWELVE output
// series on the host, and uploaded them as if the card had produced them.
//
// CPU REFERENCE
// -------------
//   src/indicators/ict_propulsion_block.rs
//     :363 valid_bar                :408 push_front_limited
//     :421 select_bullish_seed      :453 select_bearish_seed
//     :485 maybe_insert_bullish_order_block
//     :524 maybe_insert_bearish_order_block
//     :563 insert_bullish_propulsion :589 insert_bearish_propulsion
//     :615 write_snapshot           :653 ict_propulsion_block_row_scalar
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW with a small local state — the brief's fourth
// shape, and the natural one here. The state is: a swing side, two swing
// levels, two breach records, two monotonic deques over `swing_length` bars,
// and AT MOST TWO order blocks per side (`push_front_limited` truncates to 2,
// :408). All of that lives in registers and two `swing_length + 1` index rings,
// so the per-row scratch is tiny and the slot planner will almost always give
// every row its own.
//
// ARITHMETIC
// ----------
// No arithmetic to speak of — this is a state machine over comparisons, and the
// values it emits are copies of input bars. What matters is the ORDER of the
// state transitions, which is transliterated line for line. `fmin`/`fmax` are
// used where the CPU writes `f64::min`/`f64::max` (:775, :818): those return the
// non-NaN operand, and the bars reaching that code are already known finite, but
// using the same primitive keeps the two readable side by side.
//
// The file is listed in `F64_LANE_SOURCES`, so it is never compiled with
// `--use_fast_math`.

#include <cmath>
#include <cstdint>

#define ICT_MITIGATION_CLOSE 0
#define ICT_MITIGATION_WICK  1

__device__ __forceinline__ double ict_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

struct IctBlock {
    int start_index;
    int end_index;
    int confirmed_index;
    double open;
    double high;
    double low;
    double close;
    bool is_propulsion;
    bool is_active;
    bool is_mitigated;
};

struct IctSeed {
    int index;
    double open;
    double high;
    double low;
    double close;
};

__device__ __forceinline__ IctBlock ict_block_new(IctSeed seed, int confirmed, bool propulsion) {
    IctBlock b;
    b.start_index = seed.index;
    b.end_index = confirmed;
    b.confirmed_index = confirmed;
    b.open = seed.open;
    b.high = seed.high;
    b.low = seed.low;
    b.close = seed.close;
    b.is_propulsion = propulsion;
    b.is_active = true;
    b.is_mitigated = false;
    return b;
}

// push_front_limited (:408): insert at the front, keep at most two.
__device__ __forceinline__ void ict_push_front(IctBlock* blocks, int* count, IctBlock block) {
    if (*count >= 2) {
        blocks[1] = blocks[0];
    } else if (*count == 1) {
        blocks[1] = blocks[0];
        *count = 2;
    } else {
        *count = 1;
    }
    blocks[0] = block;
    if (*count > 2) {
        *count = 2;
    }
}

extern "C" __global__ void ict_propulsion_block_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ swing_lengths,
    // Per row, not global: `expand_grid_ict_propulsion_block` (:1381) crosses
    // swing_length with Close AND Wick, so one sweep can carry both.
    const int* __restrict__ mitigation_prices,
    int rows,
    int slots,
    int deque_cap,
    int* iscratch,
    double* __restrict__ out_bull_high,
    double* __restrict__ out_bull_low,
    double* __restrict__ out_bull_kind,
    double* __restrict__ out_bull_active,
    double* __restrict__ out_bull_mitigated,
    double* __restrict__ out_bull_new,
    double* __restrict__ out_bear_high,
    double* __restrict__ out_bear_low,
    double* __restrict__ out_bear_kind,
    double* __restrict__ out_bear_active,
    double* __restrict__ out_bear_mitigated,
    double* __restrict__ out_bear_new
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = ict_qnan();
    int* maxq = iscratch + static_cast<size_t>(slot) * 2ull * static_cast<size_t>(deque_cap);
    int* minq = maxq + deque_cap;

    for (int row = slot; row < rows; row += slots) {
        int swing_length = swing_lengths[row];
        int mitigation_price = mitigation_prices[row];

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* bh = out_bull_high + row_base;
        double* bl = out_bull_low + row_base;
        double* bk = out_bull_kind + row_base;
        double* ba = out_bull_active + row_base;
        double* bm = out_bull_mitigated + row_base;
        double* bn = out_bull_new + row_base;
        double* sh = out_bear_high + row_base;
        double* sl = out_bear_low + row_base;
        double* sk = out_bear_kind + row_base;
        double* sa = out_bear_active + row_base;
        double* sm = out_bear_mitigated + row_base;
        double* sn = out_bear_new + row_base;

        int swing_os = 0;
        double swing_high_value = nan_value;
        int swing_high_index = 0;
        bool swing_high_cross = false;
        double swing_low_value = nan_value;
        int swing_low_index = 0;
        bool swing_low_cross = false;

        double bull_breach_value = nan_value;
        int bull_breach_index_state = 0;
        bool bull_breach_cross = false;
        double bear_breach_value = nan_value;
        int bear_breach_index_state = 0;
        bool bear_breach_cross = false;

        double bull_breach_low_prev = nan_value;
        double bull_breach_high_prev = nan_value;
        int bull_breach_index_prev = 0;
        double bear_breach_low_prev = nan_value;
        double bear_breach_high_prev = nan_value;
        int bear_breach_index_prev = 0;

        IctBlock bull_blocks[2];
        IctBlock bear_blocks[2];
        int bull_count = 0;
        int bear_count = 0;

        int max_head = 0, max_len = 0, min_head = 0, min_len = 0;
        int cap = deque_cap;

        for (int i = 0; i < len; ++i) {
            double o = open[i];
            double h = high[i];
            double l = low[i];
            double c = close[i];

            if (!(isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) && h >= l)) {
                bh[i] = nan_value;
                bl[i] = nan_value;
                bk[i] = nan_value;
                ba[i] = nan_value;
                bm[i] = nan_value;
                bn[i] = nan_value;
                sh[i] = nan_value;
                sl[i] = nan_value;
                sk[i] = nan_value;
                sa[i] = nan_value;
                sm[i] = nan_value;
                sn[i] = nan_value;
                swing_os = 0;
                swing_high_value = nan_value;
                swing_high_cross = false;
                swing_low_value = nan_value;
                swing_low_cross = false;
                bull_breach_value = nan_value;
                bull_breach_cross = false;
                bear_breach_value = nan_value;
                bear_breach_cross = false;
                bull_breach_low_prev = nan_value;
                bull_breach_high_prev = nan_value;
                bear_breach_low_prev = nan_value;
                bear_breach_high_prev = nan_value;
                bull_count = 0;
                bear_count = 0;
                max_head = 0;
                max_len = 0;
                min_head = 0;
                min_len = 0;
                continue;
            }

            while (max_len > 0) {
                int back = max_head + max_len - 1;
                if (back >= cap) {
                    back -= cap;
                }
                if (high[maxq[back]] <= h) {
                    max_len -= 1;
                } else {
                    break;
                }
            }
            {
                int tail = max_head + max_len;
                if (tail >= cap) {
                    tail -= cap;
                }
                maxq[tail] = i;
                max_len += 1;
            }
            while (min_len > 0) {
                int back = min_head + min_len - 1;
                if (back >= cap) {
                    back -= cap;
                }
                if (low[minq[back]] >= l) {
                    min_len -= 1;
                } else {
                    break;
                }
            }
            {
                int tail = min_head + min_len;
                if (tail >= cap) {
                    tail -= cap;
                }
                minq[tail] = i;
                min_len += 1;
            }

            int window_start = i - (swing_length - 1);
            if (window_start < 0) {
                window_start = 0;
            }
            while (max_len > 0 && maxq[max_head] < window_start) {
                max_head += 1;
                if (max_head == cap) {
                    max_head = 0;
                }
                max_len -= 1;
            }
            while (min_len > 0 && minq[min_head] < window_start) {
                min_head += 1;
                if (min_head == cap) {
                    min_head = 0;
                }
                min_len -= 1;
            }

            if (i >= swing_length) {
                int candidate = i - swing_length;
                double upper = high[maxq[max_head]];
                double lower = low[minq[min_head]];
                int next_os = swing_os;
                if (high[candidate] > upper) {
                    next_os = 0;
                } else if (low[candidate] < lower) {
                    next_os = 1;
                }
                if (next_os == 0 && swing_os != 0) {
                    swing_high_value = high[candidate];
                    swing_high_index = candidate;
                    swing_high_cross = false;
                }
                if (next_os == 1 && swing_os != 1) {
                    swing_low_value = low[candidate];
                    swing_low_index = candidate;
                    swing_low_cross = false;
                }
                swing_os = next_os;
            }

            // ---- bullish breach tracking (:757) ---------------------------
            double breach_low = l;
            double breach_high = h;
            int breach_index = i;
            if (bull_count > 0) {
                const IctBlock& cur = bull_blocks[0];
                bool condition = l <= cur.high && l > cur.low && i > cur.confirmed_index &&
                                 !cur.is_mitigated && cur.is_active && !cur.is_propulsion &&
                                 o > cur.high;
                if (condition) {
                    double prev_low = isfinite(bull_breach_low_prev) ? bull_breach_low_prev : l;
                    breach_low = fmin(l, prev_low);
                    if (breach_low == l || !isfinite(bull_breach_high_prev)) {
                        breach_high = h;
                        breach_index = i;
                    } else {
                        breach_high = bull_breach_high_prev;
                        breach_index = bull_breach_index_prev;
                    }
                    bull_breach_value = breach_high;
                    bull_breach_index_state = breach_index;
                    bull_breach_cross = false;
                }
            }
            bull_breach_low_prev = breach_low;
            bull_breach_high_prev = breach_high;
            bull_breach_index_prev = breach_index;

            // ---- bearish breach tracking (:800) ---------------------------
            double bear_breach_low = l;
            double bear_breach_high = h;
            int bear_breach_idx = i;
            if (bear_count > 0) {
                const IctBlock& cur = bear_blocks[0];
                bool condition = h >= cur.low && h < cur.high && i > cur.confirmed_index &&
                                 !cur.is_mitigated && cur.is_active && !cur.is_propulsion &&
                                 o < cur.low;
                if (condition) {
                    double prev_high =
                        isfinite(bear_breach_high_prev) ? bear_breach_high_prev : h;
                    bear_breach_high = fmax(h, prev_high);
                    if (bear_breach_high == h || !isfinite(bear_breach_low_prev)) {
                        bear_breach_low = l;
                        bear_breach_idx = i;
                    } else {
                        bear_breach_low = bear_breach_low_prev;
                        bear_breach_idx = bear_breach_index_prev;
                    }
                    bear_breach_value = bear_breach_low;
                    bear_breach_index_state = bear_breach_idx;
                    bear_breach_cross = false;
                }
            }
            bear_breach_low_prev = bear_breach_low;
            bear_breach_high_prev = bear_breach_high;
            bear_breach_index_prev = bear_breach_idx;

            double bullish_new = 0.0;
            double bearish_new = 0.0;

            // ---- bullish order block on a swing-high break (:846) ---------
            if (isfinite(swing_high_value) && !swing_high_cross && c > swing_high_value &&
                i > swing_high_index) {
                swing_high_cross = true;
                // select_bullish_seed (:421)
                IctSeed seed;
                seed.index = i - 1;
                seed.open = open[i - 1];
                seed.high = high[i - 1];
                seed.low = low[i - 1];
                seed.close = close[i - 1];
                int diff = i > swing_high_index ? i - swing_high_index : 0;
                for (int offset = 1; offset < diff; ++offset) {
                    int idx = i - offset;
                    if (open[idx] > close[idx] && low[idx] <= seed.low) {
                        seed.index = idx;
                        seed.open = open[idx];
                        seed.high = high[idx];
                        seed.low = low[idx];
                        seed.close = close[idx];
                    }
                }

                // maybe_insert_bullish_order_block (:485)
                bool inserted;
                if (bull_count == 0) {
                    ict_push_front(bull_blocks, &bull_count, ict_block_new(seed, i, false));
                    inserted = true;
                } else {
                    if (bull_blocks[0].is_mitigated && bull_blocks[0].is_propulsion &&
                        bull_count > 1 && !bull_blocks[1].is_propulsion) {
                        bull_blocks[1].is_mitigated = true;
                    }
                    IctBlock recent = bull_blocks[0];
                    bool allow = recent.is_mitigated ||
                                 (!recent.is_mitigated && seed.high > recent.high &&
                                  seed.index > recent.start_index);
                    if (!allow) {
                        inserted = false;
                    } else {
                        ict_push_front(bull_blocks, &bull_count, ict_block_new(seed, i, false));
                        if (bull_count > 1) {
                            bull_blocks[1].is_active = false;
                            if (seed.index <= bull_blocks[1].end_index &&
                                bull_blocks[0].low <= bull_blocks[1].high &&
                                bull_blocks[0].high > bull_blocks[1].high) {
                                bull_blocks[0].is_propulsion = true;
                            }
                        }
                        inserted = true;
                    }
                }
                if (inserted) {
                    bullish_new = 1.0;
                }
            }

            // ---- bullish propulsion block (:860) --------------------------
            if (bull_count > 0) {
                const IctBlock& recent = bull_blocks[0];
                bool create_pb = isfinite(bull_breach_value) && c > bull_breach_value &&
                                 !bull_breach_cross && !recent.is_mitigated &&
                                 bull_breach_index_state > recent.confirmed_index;
                if (create_pb) {
                    bull_breach_cross = true;
                    // insert_bullish_propulsion (:563)
                    bull_blocks[0].is_active = false;
                    bull_blocks[0].end_index = i;
                    IctSeed seed;
                    seed.index = bull_breach_index_state;
                    seed.open = open[bull_breach_index_state];
                    seed.high = bull_breach_value;
                    seed.low = low[bull_breach_index_state];
                    seed.close = close[bull_breach_index_state];
                    ict_push_front(bull_blocks, &bull_count, ict_block_new(seed, i, true));
                    bullish_new = 1.0;
                }
            }

            for (int b = 0; b < bull_count; ++b) {
                if (bull_blocks[b].is_active && !bull_blocks[b].is_mitigated) {
                    bool mitigated = (mitigation_price == ICT_MITIGATION_CLOSE)
                                         ? (c < bull_blocks[b].low)
                                         : (l < bull_blocks[b].low);
                    if (mitigated) {
                        bull_blocks[b].is_mitigated = true;
                    }
                    bull_blocks[b].end_index = i;
                }
            }

            // ---- bearish order block on a swing-low break (:895) ----------
            if (isfinite(swing_low_value) && !swing_low_cross && c < swing_low_value &&
                i > swing_low_index) {
                swing_low_cross = true;
                IctSeed seed;
                seed.index = i - 1;
                seed.open = open[i - 1];
                seed.high = high[i - 1];
                seed.low = low[i - 1];
                seed.close = close[i - 1];
                int diff = i > swing_low_index ? i - swing_low_index : 0;
                for (int offset = 1; offset < diff; ++offset) {
                    int idx = i - offset;
                    if (open[idx] < close[idx] && high[idx] >= seed.high) {
                        seed.index = idx;
                        seed.open = open[idx];
                        seed.high = high[idx];
                        seed.low = low[idx];
                        seed.close = close[idx];
                    }
                }

                bool inserted;
                if (bear_count == 0) {
                    ict_push_front(bear_blocks, &bear_count, ict_block_new(seed, i, false));
                    inserted = true;
                } else {
                    if (bear_blocks[0].is_mitigated && bear_blocks[0].is_propulsion &&
                        bear_count > 1 && !bear_blocks[1].is_propulsion) {
                        bear_blocks[1].is_mitigated = true;
                    }
                    IctBlock recent = bear_blocks[0];
                    bool allow = recent.is_mitigated ||
                                 (!recent.is_mitigated && seed.low < recent.low &&
                                  seed.index > recent.start_index);
                    if (!allow) {
                        inserted = false;
                    } else {
                        ict_push_front(bear_blocks, &bear_count, ict_block_new(seed, i, false));
                        if (bear_count > 1) {
                            bear_blocks[1].is_active = false;
                            if (seed.index <= bear_blocks[1].end_index &&
                                bear_blocks[0].high >= bear_blocks[1].low &&
                                bear_blocks[0].low < bear_blocks[1].low) {
                                bear_blocks[0].is_propulsion = true;
                            }
                        }
                        inserted = true;
                    }
                }
                if (inserted) {
                    bearish_new = 1.0;
                }
            }

            if (bear_count > 0) {
                const IctBlock& recent = bear_blocks[0];
                bool create_pb = isfinite(bear_breach_value) && c < bear_breach_value &&
                                 !bear_breach_cross && !recent.is_mitigated &&
                                 bear_breach_index_state > recent.confirmed_index;
                if (create_pb) {
                    bear_breach_cross = true;
                    bear_blocks[0].is_active = false;
                    bear_blocks[0].end_index = i;
                    IctSeed seed;
                    seed.index = bear_breach_index_state;
                    seed.open = open[bear_breach_index_state];
                    seed.high = high[bear_breach_index_state];
                    seed.low = bear_breach_value;
                    seed.close = close[bear_breach_index_state];
                    ict_push_front(bear_blocks, &bear_count, ict_block_new(seed, i, true));
                    bearish_new = 1.0;
                }
            }

            for (int b = 0; b < bear_count; ++b) {
                if (bear_blocks[b].is_active && !bear_blocks[b].is_mitigated) {
                    bool mitigated = (mitigation_price == ICT_MITIGATION_CLOSE)
                                         ? (c > bear_blocks[b].high)
                                         : (h > bear_blocks[b].high);
                    if (mitigated) {
                        bear_blocks[b].is_mitigated = true;
                    }
                    bear_blocks[b].end_index = i;
                }
            }

            // write_snapshot (:615)
            if (bull_count > 0) {
                bh[i] = bull_blocks[0].high;
                bl[i] = bull_blocks[0].low;
                bk[i] = bull_blocks[0].is_propulsion ? 2.0 : 1.0;
                ba[i] = bull_blocks[0].is_active ? 1.0 : 0.0;
                bm[i] = bull_blocks[0].is_mitigated ? 1.0 : 0.0;
            } else {
                bh[i] = nan_value;
                bl[i] = nan_value;
                bk[i] = 0.0;
                ba[i] = 0.0;
                bm[i] = 0.0;
            }
            bn[i] = bullish_new;

            if (bear_count > 0) {
                sh[i] = bear_blocks[0].high;
                sl[i] = bear_blocks[0].low;
                sk[i] = bear_blocks[0].is_propulsion ? 2.0 : 1.0;
                sa[i] = bear_blocks[0].is_active ? 1.0 : 0.0;
                sm[i] = bear_blocks[0].is_mitigated ? 1.0 : 0.0;
            } else {
                sh[i] = nan_value;
                sl[i] = nan_value;
                sk[i] = 0.0;
                sa[i] = 0.0;
                sm[i] = 0.0;
            }
            sn[i] = bearish_new;
        }
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/ict_propulsion_block.rs:977. The column this
// emits is `bullish_high`.
//
// WHICH COLUMN, AND WHY IT IS NAMED HERE. `compute_ict_propulsion_block_batch`
// (cpu_batch.rs:12880-12905) accepts twelve output ids -- bullish_high,
// bullish_low, bullish_kind, bullish_active, bullish_mitigated, bullish_new
// and their bearish twins -- and has NO `value` alias, returning
// `UnknownOutput` for one. So a parity run must ask the CPU for
// `bullish_high` explicitly; this kernel emits that column and never a
// different one silently.
//
// SHAPE: one thread per combo, bars ascending -- a per-column STATE MACHINE,
// which is one of the four shapes the brief names. Everything it carries is a
// register or a tiny ring: the swing oscillator and its two pending pivots,
// two breach records, at most TWO order blocks per side
// (`push_front_limited`, :408, truncates to two -- that is the CPU's own
// bound, not a truncation invented here), and two monotone deques over
// `swing_length + 1` indices. The CPU RESETS all of it on any bar that is not
// four-way finite with `high >= low`, so a bar-parallel form cannot know which
// segment it is in.
//
// PERIOD-INVARIANT. `compute_ict_propulsion_block_batch`
// (cpu_batch.rs:12916-12918) reads `swing_length` and `mitigation_price` and
// NEVER `period`, so five swept periods give five identical CPU columns and
// this kernel emits five identical rows. Both CPU defaults are pinned below --
// `swing_length` 3 and `mitigation_price` "close".
//
// THE TWO DEQUES ARE PER-THREAD, so their capacity is a property of THIS
// COMPILED KERNEL: `swing_length + 1` indices each. At the pinned 3 that is
// four slots; the bound below is checked rather than assumed.
//
// FIRST VALID IS NOT READ: the CPU emits from bar 0 and restarts the whole
// machine at every invalid bar, so a global warmup index would be wrong after
// the first hole. The lane row declares `F64FirstValidRule::Ignored`.
//
// f64 END TO END: no arithmetic beyond comparisons and copies of input bars;
// `fmin`/`fmax` are used exactly where the CPU writes `f64::min`/`f64::max`
// (:775, :818), no f32-suffixed math function, no fast-math intrinsic, no
// epsilon. The NaN it writes is `ict_qnan()`, a DOUBLE quiet-NaN bit pattern
// -- not `__int_as_float`, which is the f32 pattern this conversion is
// removing everywhere else.
// ---------------------------------------------------------------------------

#define NEO_ICT_SWING_LENGTH 3
#define NEO_ICT_MITIGATION_PRICE ICT_MITIGATION_CLOSE
#define NEO_ICT_MAX_DEQUE_CAP 512

extern "C" __global__ void ict_propulsion_block_neo_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
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
    (void)periods;
    (void)first_valid;

    const double nan_value = ict_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = nan_value;
    }

    const int swing_length = NEO_ICT_SWING_LENGTH;
    const int mitigation_price = NEO_ICT_MITIGATION_PRICE;
    if (swing_length <= 0) {
        return;
    }
    const int cap = swing_length + 1;
    if (cap > NEO_ICT_MAX_DEQUE_CAP) {
        return;
    }

    int maxq[NEO_ICT_MAX_DEQUE_CAP];
    int minq[NEO_ICT_MAX_DEQUE_CAP];

    int swing_os = 0;
    double swing_high_value = nan_value;
    int swing_high_index = 0;
    bool swing_high_cross = false;
    double swing_low_value = nan_value;
    int swing_low_index = 0;
    bool swing_low_cross = false;

    double bull_breach_value = nan_value;
    int bull_breach_index_state = 0;
    bool bull_breach_cross = false;
    double bear_breach_value = nan_value;
    int bear_breach_index_state = 0;
    bool bear_breach_cross = false;

    double bull_breach_low_prev = nan_value;
    double bull_breach_high_prev = nan_value;
    int bull_breach_index_prev = 0;
    double bear_breach_low_prev = nan_value;
    double bear_breach_high_prev = nan_value;
    int bear_breach_index_prev = 0;

    IctBlock bull_blocks[2];
    IctBlock bear_blocks[2];
    int bull_count = 0;
    int bear_count = 0;

    int max_head = 0, max_len = 0, min_head = 0, min_len = 0;

    for (int i = 0; i < n; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!(isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) && h >= l)) {
            row[i] = nan_value;
            swing_os = 0;
            swing_high_value = nan_value;
            swing_high_cross = false;
            swing_low_value = nan_value;
            swing_low_cross = false;
            bull_breach_value = nan_value;
            bull_breach_cross = false;
            bear_breach_value = nan_value;
            bear_breach_cross = false;
            bull_breach_low_prev = nan_value;
            bull_breach_high_prev = nan_value;
            bear_breach_low_prev = nan_value;
            bear_breach_high_prev = nan_value;
            bull_count = 0;
            bear_count = 0;
            max_head = 0;
            max_len = 0;
            min_head = 0;
            min_len = 0;
            continue;
        }

        while (max_len > 0) {
            int back = max_head + max_len - 1;
            if (back >= cap) {
                back -= cap;
            }
            if (high[maxq[back]] <= h) {
                max_len -= 1;
            } else {
                break;
            }
        }
        {
            int tail = max_head + max_len;
            if (tail >= cap) {
                tail -= cap;
            }
            maxq[tail] = i;
            max_len += 1;
        }
        while (min_len > 0) {
            int back = min_head + min_len - 1;
            if (back >= cap) {
                back -= cap;
            }
            if (low[minq[back]] >= l) {
                min_len -= 1;
            } else {
                break;
            }
        }
        {
            int tail = min_head + min_len;
            if (tail >= cap) {
                tail -= cap;
            }
            minq[tail] = i;
            min_len += 1;
        }

        int window_start = i - (swing_length - 1);
        if (window_start < 0) {
            window_start = 0;
        }
        while (max_len > 0 && maxq[max_head] < window_start) {
            max_head += 1;
            if (max_head == cap) {
                max_head = 0;
            }
            max_len -= 1;
        }
        while (min_len > 0 && minq[min_head] < window_start) {
            min_head += 1;
            if (min_head == cap) {
                min_head = 0;
            }
            min_len -= 1;
        }

        if (i >= swing_length) {
            const int candidate = i - swing_length;
            const double upper = high[maxq[max_head]];
            const double lower = low[minq[min_head]];
            int next_os = swing_os;
            if (high[candidate] > upper) {
                next_os = 0;
            } else if (low[candidate] < lower) {
                next_os = 1;
            }
            if (next_os == 0 && swing_os != 0) {
                swing_high_value = high[candidate];
                swing_high_index = candidate;
                swing_high_cross = false;
            }
            if (next_os == 1 && swing_os != 1) {
                swing_low_value = low[candidate];
                swing_low_index = candidate;
                swing_low_cross = false;
            }
            swing_os = next_os;
        }

        // ---- bullish breach tracking (:757) -------------------------------
        double breach_low = l;
        double breach_high = h;
        int breach_index = i;
        if (bull_count > 0) {
            const IctBlock& cur = bull_blocks[0];
            const bool condition = l <= cur.high && l > cur.low && i > cur.confirmed_index &&
                !cur.is_mitigated && cur.is_active && !cur.is_propulsion && o > cur.high;
            if (condition) {
                const double prev_low =
                    isfinite(bull_breach_low_prev) ? bull_breach_low_prev : l;
                breach_low = fmin(l, prev_low);
                if (breach_low == l || !isfinite(bull_breach_high_prev)) {
                    breach_high = h;
                    breach_index = i;
                } else {
                    breach_high = bull_breach_high_prev;
                    breach_index = bull_breach_index_prev;
                }
                bull_breach_value = breach_high;
                bull_breach_index_state = breach_index;
                bull_breach_cross = false;
            }
        }
        bull_breach_low_prev = breach_low;
        bull_breach_high_prev = breach_high;
        bull_breach_index_prev = breach_index;

        // ---- bearish breach tracking (:800) -------------------------------
        double bear_breach_low = l;
        double bear_breach_high = h;
        int bear_breach_idx = i;
        if (bear_count > 0) {
            const IctBlock& cur = bear_blocks[0];
            const bool condition = h >= cur.low && h < cur.high && i > cur.confirmed_index &&
                !cur.is_mitigated && cur.is_active && !cur.is_propulsion && o < cur.low;
            if (condition) {
                const double prev_high =
                    isfinite(bear_breach_high_prev) ? bear_breach_high_prev : h;
                bear_breach_high = fmax(h, prev_high);
                if (bear_breach_high == h || !isfinite(bear_breach_low_prev)) {
                    bear_breach_low = l;
                    bear_breach_idx = i;
                } else {
                    bear_breach_low = bear_breach_low_prev;
                    bear_breach_idx = bear_breach_index_prev;
                }
                bear_breach_value = bear_breach_low;
                bear_breach_index_state = bear_breach_idx;
                bear_breach_cross = false;
            }
        }
        bear_breach_low_prev = bear_breach_low;
        bear_breach_high_prev = bear_breach_high;
        bear_breach_index_prev = bear_breach_idx;

        // ---- bullish order block on a swing-high break (:846) -------------
        if (isfinite(swing_high_value) && !swing_high_cross && c > swing_high_value &&
            i > swing_high_index) {
            swing_high_cross = true;
            IctSeed seed;
            seed.index = i - 1;
            seed.open = open[i - 1];
            seed.high = high[i - 1];
            seed.low = low[i - 1];
            seed.close = close[i - 1];
            const int diff = i > swing_high_index ? i - swing_high_index : 0;
            for (int offset = 1; offset < diff; ++offset) {
                const int idx = i - offset;
                if (open[idx] > close[idx] && low[idx] <= seed.low) {
                    seed.index = idx;
                    seed.open = open[idx];
                    seed.high = high[idx];
                    seed.low = low[idx];
                    seed.close = close[idx];
                }
            }

            if (bull_count == 0) {
                ict_push_front(bull_blocks, &bull_count, ict_block_new(seed, i, false));
            } else {
                if (bull_blocks[0].is_mitigated && bull_blocks[0].is_propulsion &&
                    bull_count > 1 && !bull_blocks[1].is_propulsion) {
                    bull_blocks[1].is_mitigated = true;
                }
                const IctBlock recent = bull_blocks[0];
                const bool allow = recent.is_mitigated ||
                    (!recent.is_mitigated && seed.high > recent.high &&
                     seed.index > recent.start_index);
                if (allow) {
                    ict_push_front(bull_blocks, &bull_count, ict_block_new(seed, i, false));
                    if (bull_count > 1) {
                        bull_blocks[1].is_active = false;
                        if (seed.index <= bull_blocks[1].end_index &&
                            bull_blocks[0].low <= bull_blocks[1].high &&
                            bull_blocks[0].high > bull_blocks[1].high) {
                            bull_blocks[0].is_propulsion = true;
                        }
                    }
                }
            }
        }

        // ---- bullish propulsion block (:860) ------------------------------
        if (bull_count > 0) {
            const IctBlock& recent = bull_blocks[0];
            const bool create_pb = isfinite(bull_breach_value) && c > bull_breach_value &&
                !bull_breach_cross && !recent.is_mitigated &&
                bull_breach_index_state > recent.confirmed_index;
            if (create_pb) {
                bull_breach_cross = true;
                bull_blocks[0].is_active = false;
                bull_blocks[0].end_index = i;
                IctSeed seed;
                seed.index = bull_breach_index_state;
                seed.open = open[bull_breach_index_state];
                seed.high = bull_breach_value;
                seed.low = low[bull_breach_index_state];
                seed.close = close[bull_breach_index_state];
                ict_push_front(bull_blocks, &bull_count, ict_block_new(seed, i, true));
            }
        }

        for (int b = 0; b < bull_count; ++b) {
            if (bull_blocks[b].is_active && !bull_blocks[b].is_mitigated) {
                const bool mitigated = (mitigation_price == ICT_MITIGATION_CLOSE)
                    ? (c < bull_blocks[b].low)
                    : (l < bull_blocks[b].low);
                if (mitigated) {
                    bull_blocks[b].is_mitigated = true;
                }
                bull_blocks[b].end_index = i;
            }
        }

        // ---- bearish order block on a swing-low break (:895) --------------
        if (isfinite(swing_low_value) && !swing_low_cross && c < swing_low_value &&
            i > swing_low_index) {
            swing_low_cross = true;
            IctSeed seed;
            seed.index = i - 1;
            seed.open = open[i - 1];
            seed.high = high[i - 1];
            seed.low = low[i - 1];
            seed.close = close[i - 1];
            const int diff = i > swing_low_index ? i - swing_low_index : 0;
            for (int offset = 1; offset < diff; ++offset) {
                const int idx = i - offset;
                if (open[idx] < close[idx] && high[idx] >= seed.high) {
                    seed.index = idx;
                    seed.open = open[idx];
                    seed.high = high[idx];
                    seed.low = low[idx];
                    seed.close = close[idx];
                }
            }

            if (bear_count == 0) {
                ict_push_front(bear_blocks, &bear_count, ict_block_new(seed, i, false));
            } else {
                if (bear_blocks[0].is_mitigated && bear_blocks[0].is_propulsion &&
                    bear_count > 1 && !bear_blocks[1].is_propulsion) {
                    bear_blocks[1].is_mitigated = true;
                }
                const IctBlock recent = bear_blocks[0];
                const bool allow = recent.is_mitigated ||
                    (!recent.is_mitigated && seed.low < recent.low &&
                     seed.index > recent.start_index);
                if (allow) {
                    ict_push_front(bear_blocks, &bear_count, ict_block_new(seed, i, false));
                    if (bear_count > 1) {
                        bear_blocks[1].is_active = false;
                        if (seed.index <= bear_blocks[1].end_index &&
                            bear_blocks[0].high >= bear_blocks[1].low &&
                            bear_blocks[0].low < bear_blocks[1].low) {
                            bear_blocks[0].is_propulsion = true;
                        }
                    }
                }
            }
        }

        if (bear_count > 0) {
            const IctBlock& recent = bear_blocks[0];
            const bool create_pb = isfinite(bear_breach_value) && c < bear_breach_value &&
                !bear_breach_cross && !recent.is_mitigated &&
                bear_breach_index_state > recent.confirmed_index;
            if (create_pb) {
                bear_breach_cross = true;
                bear_blocks[0].is_active = false;
                bear_blocks[0].end_index = i;
                IctSeed seed;
                seed.index = bear_breach_index_state;
                seed.open = open[bear_breach_index_state];
                seed.high = high[bear_breach_index_state];
                seed.low = bear_breach_value;
                seed.close = close[bear_breach_index_state];
                ict_push_front(bear_blocks, &bear_count, ict_block_new(seed, i, true));
            }
        }

        for (int b = 0; b < bear_count; ++b) {
            if (bear_blocks[b].is_active && !bear_blocks[b].is_mitigated) {
                const bool mitigated = (mitigation_price == ICT_MITIGATION_CLOSE)
                    ? (c > bear_blocks[b].high)
                    : (h > bear_blocks[b].high);
                if (mitigated) {
                    bear_blocks[b].is_mitigated = true;
                }
                bear_blocks[b].end_index = i;
            }
        }

        // write_snapshot (:615) -- the bullish_high column.
        row[i] = bull_count > 0 ? bull_blocks[0].high : nan_value;
    }
}
