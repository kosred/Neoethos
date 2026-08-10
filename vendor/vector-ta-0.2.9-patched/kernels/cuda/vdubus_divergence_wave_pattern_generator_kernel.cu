// vdubus_divergence_wave_pattern_generator — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line: extern "C" __global__ void
//           vdubus_divergence_wave_pattern_generator_batch_f64() {}
// plus a wrapper that resolved the empty symbol, computed all TWELVE output
// series on the host, and uploaded them.
//
// CPU REFERENCE
// -------------
//   src/indicators/vdubus_divergence_wave_pattern_generator.rs
//     :779  EmaState          :833  MacdState
//     :873  PivotDetector     :952  push_front_cap
//     :960  MomentumState (+ its six pattern predicates)
//    :1065  harmonic_family_code   :1091 standard_family_from_filters
//    :1119  StructureEngine   :1251  State::update  <- the per-bar body
//    :1715  compute_row       <- the per-row loop
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW with a small local state — three EMAs, four
// pivot detectors over fixed windows, and two ten-deep pivot stacks. Every
// piece carries across bars, and the pattern predicates read the stacks at
// indices 0..3, so the ORDER of `push_front_cap` (:952 — insert at the front,
// drop the eleventh) is part of the answer.
//
// ARITHMETIC
// ----------
// f64 throughout; in `F64_LANE_SOURCES`, never `--use_fast_math`. `fma()`
// appears once, in `EmaState::update` (:823), because that is the one place the
// reference writes `mul_add`. `err_tol` is the CPU's own parameter, not an
// epsilon invented here.

#include <cmath>
#include <cstdint>

#define VD_FAMILY_NONE            0.0
#define VD_FAMILY_RETRACEMENT     1.0
#define VD_FAMILY_GARTLEY         2.0
#define VD_FAMILY_BAT             3.0
#define VD_FAMILY_BUTTERFLY       4.0
#define VD_FAMILY_CRAB            5.0
#define VD_FAMILY_DEEP            6.0
#define VD_FAMILY_HEAD_SHOULDERS  7.0

// `push_front_cap` (:952) keeps at most ten entries.
#define VD_STACK 10

__device__ __forceinline__ double vd_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// EmaState (:779)
struct VdEma {
    int length;
    double alpha;
    int count;
    double sum;
    double value;
    bool started;
};

__device__ __forceinline__ void vd_ema_init(VdEma* e, int length) {
    e->length = length;
    e->alpha = 2.0 / (static_cast<double>(length) + 1.0);
    e->count = 0;
    e->sum = 0.0;
    e->value = vd_qnan();
    e->started = false;
}

__device__ __forceinline__ void vd_ema_reset(VdEma* e) {
    e->count = 0;
    e->sum = 0.0;
    e->value = vd_qnan();
    e->started = false;
}

__device__ __forceinline__ int vd_ema_update(VdEma* e, double value, double* out) {
    if (!isfinite(value)) {
        vd_ema_reset(e);
        return 0;
    }
    if (!e->started) {
        e->sum += value;
        e->count += 1;
        if (e->count == e->length) {
            e->value = e->sum / static_cast<double>(e->length);
            e->started = true;
            *out = e->value;
            return 1;
        }
        return 0;
    }
    // CPU: `self.alpha.mul_add(value, (1.0 - self.alpha) * self.value)`
    e->value = fma(e->alpha, value, (1.0 - e->alpha) * e->value);
    *out = e->value;
    return 1;
}

// PivotDetector (:873). `window` is a ring of `2 * span + 1` doubles.
struct VdPivot {
    double* window;
    int span;
    int size;
    int head;
    int count;
    bool is_high;
};

__device__ __forceinline__ void vd_pivot_init(VdPivot* p, double* buf, int span, bool is_high) {
    p->window = buf;
    p->span = span;
    p->size = span * 2 + 1;
    p->head = 0;
    p->count = 0;
    p->is_high = is_high;
    double q = vd_qnan();
    for (int i = 0; i < p->size; ++i) {
        p->window[i] = q;
    }
}

__device__ __forceinline__ void vd_pivot_reset(VdPivot* p) {
    p->head = 0;
    p->count = 0;
    double q = vd_qnan();
    for (int i = 0; i < p->size; ++i) {
        p->window[i] = q;
    }
}

__device__ __forceinline__ int vd_pivot_update(VdPivot* p, double value, double* out) {
    if (!isfinite(value)) {
        vd_pivot_reset(p);
        return 0;
    }
    p->window[p->head] = value;
    p->head += 1;
    if (p->head == p->size) {
        p->head = 0;
    }
    if (p->count < p->size) {
        p->count += 1;
    }
    if (p->count < p->size) {
        return 0;
    }

    int start = p->head;
    int centre_idx = (start + p->span) % p->size;
    double centre = p->window[centre_idx];
    if (!isfinite(centre)) {
        return 0;
    }
    for (int j = 0; j < p->size; ++j) {
        if (j == p->span) {
            continue;
        }
        double current = p->window[(start + j) % p->size];
        if (!isfinite(current)) {
            return 0;
        }
        if (p->is_high) {
            if (centre <= current) {
                return 0;
            }
        } else {
            if (centre >= current) {
                return 0;
            }
        }
    }
    *out = centre;
    return 1;
}

// push_front_cap (:952)
__device__ __forceinline__ void vd_push_front(double* stack, int* len, double value) {
    int n = *len;
    if (n > VD_STACK - 1) {
        n = VD_STACK - 1;
    }
    for (int i = n; i > 0; --i) {
        stack[i] = stack[i - 1];
    }
    stack[0] = value;
    if (*len < VD_STACK) {
        *len += 1;
    }
}

// harmonic_family_code (:1065)
__device__ __forceinline__ double vd_harmonic_family(
    double xb_ratio, double xd_ratio, double err_tol) {
    if (fabs(xb_ratio - 0.618) < err_tol && fabs(xd_ratio - 0.786) < err_tol) {
        return VD_FAMILY_GARTLEY;
    }
    if (xb_ratio >= 0.382 - err_tol && xb_ratio <= 0.5 + err_tol &&
        fabs(xd_ratio - 0.886) < err_tol) {
        return VD_FAMILY_BAT;
    }
    if (fabs(xb_ratio - 0.786) < err_tol && xd_ratio >= 1.27 - err_tol &&
        xd_ratio <= 1.618 + err_tol) {
        return VD_FAMILY_BUTTERFLY;
    }
    if (xb_ratio >= 0.382 - err_tol && xb_ratio <= 0.618 + err_tol &&
        fabs(xd_ratio - 1.618) < err_tol) {
        return VD_FAMILY_CRAB;
    }
    if (xd_ratio > 1.0) {
        return VD_FAMILY_DEEP;
    }
    return VD_FAMILY_RETRACEMENT;
}

struct VdShow {
    bool standard, climax, rounded, predator;
    bool gartley, bat, butterfly, crab, deep, hs;
};

// standard_family_from_filters (:1091)
__device__ __forceinline__ double vd_standard_family(
    const VdShow& show, double family, bool is_hs) {
    if (is_hs) {
        return show.hs ? VD_FAMILY_HEAD_SHOULDERS : VD_FAMILY_NONE;
    }
    int code = static_cast<int>(family);
    switch (code) {
        case 1:
            return VD_FAMILY_RETRACEMENT;
        case 2:
            return show.gartley ? VD_FAMILY_GARTLEY : VD_FAMILY_NONE;
        case 3:
            return show.bat ? VD_FAMILY_BAT : VD_FAMILY_NONE;
        case 4:
            return show.butterfly ? VD_FAMILY_BUTTERFLY : VD_FAMILY_NONE;
        case 5:
            return show.crab ? VD_FAMILY_CRAB : VD_FAMILY_NONE;
        case 6:
            return show.deep ? VD_FAMILY_DEEP : VD_FAMILY_NONE;
        default:
            return VD_FAMILY_NONE;
    }
}

struct VdSignals {
    double standard, climax, rounded, predator;
};

extern "C" __global__ void vdubus_divergence_wave_pattern_generator_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ fast_depths,
    const int* __restrict__ slow_depths,
    const int* __restrict__ fast_lengths,
    const int* __restrict__ slow_lengths,
    const int* __restrict__ signal_lengths,
    const int* __restrict__ lookbacks,
    const double* __restrict__ err_tols,
    int show_standard,
    int show_climax,
    int show_rounded,
    int show_predator,
    int show_gartley,
    int show_bat,
    int show_butterfly,
    int show_crab,
    int show_deep,
    int show_hs,
    int rows,
    int slots,
    int window_cap,
    double* scratch,
    double* __restrict__ out_fast_standard,
    double* __restrict__ out_fast_climax,
    double* __restrict__ out_fast_rounded,
    double* __restrict__ out_fast_predator,
    double* __restrict__ out_slow_standard,
    double* __restrict__ out_slow_climax,
    double* __restrict__ out_slow_rounded,
    double* __restrict__ out_slow_predator,
    double* __restrict__ out_opposing_force,
    double* __restrict__ out_macd,
    double* __restrict__ out_signal,
    double* __restrict__ out_hist
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = vd_qnan();
    // Six pivot windows (fast high/low, slow high/low, momentum high/low), each
    // at most `window_cap` wide.
    size_t per_slot = 6ull * static_cast<size_t>(window_cap);
    double* base = scratch + static_cast<size_t>(slot) * per_slot;

    VdShow show;
    show.standard = show_standard != 0;
    show.climax = show_climax != 0;
    show.rounded = show_rounded != 0;
    show.predator = show_predator != 0;
    show.gartley = show_gartley != 0;
    show.bat = show_bat != 0;
    show.butterfly = show_butterfly != 0;
    show.crab = show_crab != 0;
    show.deep = show_deep != 0;
    show.hs = show_hs != 0;

    for (int row = slot; row < rows; row += slots) {
        int fast_depth = fast_depths[row];
        int slow_depth = slow_depths[row];
        int lookback = lookbacks[row];
        double err_tol = err_tols[row];

        VdEma ema_fast, ema_slow, ema_signal;
        vd_ema_init(&ema_fast, fast_lengths[row]);
        vd_ema_init(&ema_slow, slow_lengths[row]);
        vd_ema_init(&ema_signal, signal_lengths[row]);

        VdPivot fast_high, fast_low, slow_high, slow_low, mom_high, mom_low;
        vd_pivot_init(&fast_high, base + 0 * window_cap, fast_depth, true);
        vd_pivot_init(&fast_low, base + 1 * window_cap, fast_depth, false);
        vd_pivot_init(&slow_high, base + 2 * window_cap, slow_depth, true);
        vd_pivot_init(&slow_low, base + 3 * window_cap, slow_depth, false);
        vd_pivot_init(&mom_high, base + 4 * window_cap, lookback, true);
        vd_pivot_init(&mom_low, base + 5 * window_cap, lookback, false);

        double wave_highs[VD_STACK];
        double wave_lows[VD_STACK];
        double fast_pivots[VD_STACK];
        double slow_pivots[VD_STACK];
        int wave_high_len = 0, wave_low_len = 0;
        int fast_pivot_len = 0, slow_pivot_len = 0;

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* o_fs = out_fast_standard + row_base;
        double* o_fc = out_fast_climax + row_base;
        double* o_fr = out_fast_rounded + row_base;
        double* o_fp = out_fast_predator + row_base;
        double* o_ss = out_slow_standard + row_base;
        double* o_sc = out_slow_climax + row_base;
        double* o_sr = out_slow_rounded + row_base;
        double* o_sp = out_slow_predator + row_base;
        double* o_of = out_opposing_force + row_base;
        double* o_macd = out_macd + row_base;
        double* o_signal = out_signal + row_base;
        double* o_hist = out_hist + row_base;

        for (int i = 0; i < len; ++i) {
            double h = high[i];
            double l = low[i];
            double c = close[i];

            o_fs[i] = nan_value;
            o_fc[i] = nan_value;
            o_fr[i] = nan_value;
            o_fp[i] = nan_value;
            o_ss[i] = nan_value;
            o_sc[i] = nan_value;
            o_sr[i] = nan_value;
            o_sp[i] = nan_value;
            o_of[i] = nan_value;
            o_macd[i] = nan_value;
            o_signal[i] = nan_value;
            o_hist[i] = nan_value;

            if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
                // State::reset (:1276)
                vd_ema_reset(&ema_fast);
                vd_ema_reset(&ema_slow);
                vd_ema_reset(&ema_signal);
                vd_pivot_reset(&mom_high);
                vd_pivot_reset(&mom_low);
                wave_high_len = 0;
                wave_low_len = 0;
                vd_pivot_reset(&fast_high);
                vd_pivot_reset(&fast_low);
                fast_pivot_len = 0;
                vd_pivot_reset(&slow_high);
                vd_pivot_reset(&slow_low);
                slow_pivot_len = 0;
                continue;
            }

            // MacdState::update (:864)
            bool have_macd = false;
            double macd = 0.0, signal = 0.0, hist = 0.0;
            {
                double fast_value, slow_value, signal_value;
                if (vd_ema_update(&ema_fast, c, &fast_value) &&
                    vd_ema_update(&ema_slow, c, &slow_value)) {
                    double m = fast_value - slow_value;
                    if (vd_ema_update(&ema_signal, m, &signal_value)) {
                        macd = m;
                        signal = signal_value;
                        hist = m - signal_value;
                        have_macd = true;
                    }
                } else {
                    // The CPU's `?` on `fast` short-circuits BEFORE `slow` is
                    // updated. `vd_ema_update` above relies on `&&` doing the
                    // same, so an unstarted fast EMA leaves slow untouched —
                    // which is what `MacdState::update` does.
                }
            }

            // MomentumState::update (:1000)
            if (have_macd) {
                double pivot;
                if (vd_pivot_update(&mom_high, hist, &pivot)) {
                    vd_push_front(wave_highs, &wave_high_len, pivot);
                }
                if (vd_pivot_update(&mom_low, hist, &pivot)) {
                    vd_push_front(wave_lows, &wave_low_len, pivot);
                }
            }

            // The six momentum predicates (:1021-1062), evaluated once so both
            // engines see the same snapshot — as they do on the CPU, where both
            // `update` calls borrow the SAME `&self.momentum`.
            bool standard_bearish = wave_high_len >= 3 && wave_highs[1] < wave_highs[2] &&
                                    wave_highs[0] <= wave_highs[1];
            bool standard_bullish = wave_low_len >= 3 && wave_lows[1] > wave_lows[2] &&
                                    wave_lows[0] >= wave_lows[1];
            bool climax_bearish = wave_high_len >= 3 && wave_highs[1] >= wave_highs[2] &&
                                  wave_highs[0] < wave_highs[1];
            bool climax_bullish = wave_low_len >= 3 && wave_lows[1] <= wave_lows[2] &&
                                  wave_lows[0] > wave_lows[1];
            bool rounded_bearish = wave_high_len >= 4 && wave_highs[3] > wave_highs[2] &&
                                   wave_highs[2] > wave_highs[1] &&
                                   wave_highs[1] > wave_highs[0];
            bool rounded_bullish = wave_low_len >= 4 && wave_lows[3] < wave_lows[2] &&
                                   wave_lows[2] < wave_lows[1] && wave_lows[1] < wave_lows[0];
            bool bearish_predator = wave_high_len >= 2 && wave_highs[0] > wave_highs[1];
            bool bullish_predator = wave_low_len >= 2 && wave_lows[0] < wave_lows[1];

            // StructureEngine::update (:1235), twice.
            VdSignals fast_sig = {0.0, 0.0, 0.0, 0.0};
            VdSignals slow_sig = {0.0, 0.0, 0.0, 0.0};

            for (int engine = 0; engine < 2; ++engine) {
                VdPivot* hi = engine == 0 ? &fast_high : &slow_high;
                VdPivot* lo = engine == 0 ? &fast_low : &slow_low;
                double* pivots = engine == 0 ? fast_pivots : slow_pivots;
                int* pivot_len = engine == 0 ? &fast_pivot_len : &slow_pivot_len;
                VdSignals* out_sig = engine == 0 ? &fast_sig : &slow_sig;

                double pivot;
                if (vd_pivot_update(hi, h, &pivot)) {
                    vd_push_front(pivots, pivot_len, pivot);
                    // evaluate_bearish (:1155)
                    VdSignals s = {0.0, 0.0, 0.0, 0.0};
                    if (*pivot_len >= 5) {
                        double y_d = pivots[0];
                        double y_b = pivots[2];
                        double y_a = pivots[3];
                        double y_x = pivots[4];
                        double xa_len = fabs(y_a - y_x);
                        double ab_len = fabs(y_b - y_a);
                        double xb_ratio = xa_len != 0.0 ? ab_len / xa_len : 0.0;
                        double xd_ratio = xa_len != 0.0 ? fabs(y_d - y_x) / xa_len : 0.0;
                        double raw_family = vd_harmonic_family(xb_ratio, xd_ratio, err_tol);
                        bool is_hs = show.hs && y_b > y_x && y_b > y_d;
                        if (show.standard && standard_bearish) {
                            double family = vd_standard_family(show, raw_family, is_hs);
                            if (family != VD_FAMILY_NONE) {
                                s.standard = -family;
                            }
                        }
                        if (show.climax && climax_bearish) {
                            s.climax = -1.0;
                        }
                        if (show.rounded && rounded_bearish) {
                            s.rounded = -1.0;
                        }
                        if (show.predator && !standard_bearish && y_d < y_x && bearish_predator) {
                            s.predator = -1.0;
                        }
                    }
                    *out_sig = s;
                }
                if (vd_pivot_update(lo, l, &pivot)) {
                    vd_push_front(pivots, pivot_len, pivot);
                    // evaluate_bullish (:1194)
                    VdSignals s = {0.0, 0.0, 0.0, 0.0};
                    if (*pivot_len >= 5) {
                        double y_d = pivots[0];
                        double y_b = pivots[2];
                        double y_a = pivots[3];
                        double y_x = pivots[4];
                        double xa_len = fabs(y_a - y_x);
                        double ab_len = fabs(y_b - y_a);
                        double xb_ratio = xa_len != 0.0 ? ab_len / xa_len : 0.0;
                        double xd_ratio = xa_len != 0.0 ? fabs(y_d - y_x) / xa_len : 0.0;
                        double raw_family = vd_harmonic_family(xb_ratio, xd_ratio, err_tol);
                        bool is_inverse_hs = show.hs && y_b < y_x && y_b < y_d;
                        if (show.standard && standard_bullish) {
                            double family =
                                vd_standard_family(show, raw_family, is_inverse_hs);
                            if (family != VD_FAMILY_NONE) {
                                s.standard = family;
                            }
                        }
                        if (show.climax && climax_bullish) {
                            s.climax = 1.0;
                        }
                        if (show.rounded && rounded_bullish) {
                            s.rounded = 1.0;
                        }
                        if (show.predator && !standard_bullish && y_d > y_x && bullish_predator) {
                            s.predator = 1.0;
                        }
                    }
                    *out_sig = s;
                }
            }

            // `let (macd, signal, hist) = macd_out?;` (:1310) — the engines have
            // ALREADY advanced by the time this returns None.
            if (!have_macd) {
                continue;
            }

            // opposing_force (:1010)
            double bull = fabs(wave_low_len > 0 ? wave_lows[0] : 0.0);
            double bear = fabs(wave_high_len > 0 ? wave_highs[0] : 0.0);
            double opposing = bull > bear ? 1.0 : (bear > bull ? -1.0 : 0.0);

            o_fs[i] = fast_sig.standard;
            o_fc[i] = fast_sig.climax;
            o_fr[i] = fast_sig.rounded;
            o_fp[i] = fast_sig.predator;
            o_ss[i] = slow_sig.standard;
            o_sc[i] = slow_sig.climax;
            o_sr[i] = slow_sig.rounded;
            o_sp[i] = slow_sig.predator;
            o_of[i] = opposing;
            o_macd[i] = macd;
            o_signal[i] = signal;
            o_hist[i] = hist;
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/vdubus_divergence_wave_pattern_generator.rs
 *   `VdubusDivergenceWavePatternGeneratorState::update` (:1280-1317), reached
 *   from `compute_row` (:1715-1885), built out of `EmaState::update` (:809),
 *   `MacdState::update` (:857), `PivotDetector::update` (:896-945),
 *   `MomentumState` (:952-1030) and `StructureEngine` (:1096-1246).
 *
 * WHICH COLUMN: `fast_standard`, output index 0 of
 *   `OUTPUTS_VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR` (registry.rs:1240).
 *   THE CPU BATCH CANNOT ANSWER "value" AT ALL, and that is a defect in the
 *   dispatcher rather than a choice made here: `compute_..._batch`
 *   (cpu_batch.rs:5095) calls `expect_value_output` (:5099), which admits ONLY
 *   the literal "value" (:17248), and then matches `output_id` against twelve
 *   arms (:5240-5252) NONE of which is "value" -- so every request falls
 *   through to `UnknownOutput`. A parity run must therefore ask the CPU for
 *   "fast_standard" explicitly; this kernel emits that column, which is what
 *   "value" resolves to under the registry's ordering.
 *
 * WHY A SECOND ENTRY POINT: `vdubus_divergence_wave_pattern_generator_batch_f64`
 *   (:252) takes 37 parameters -- the widest signature in the crate -- and
 *   emits twelve series. The lane launches
 *   (high, low, close, n, periods, n_combos, first_valid, out).
 *
 * INPUT: high / low / close -- extract_ohlc_input (cpu_batch.rs:5100) --
 *   F64InputKind::Hlc.
 *
 * WHY THE SLOW ENGINE IS ABSENT: `slow_engine` (:1300) feeds ONLY the four
 *   `slow_*` outputs; it shares no state with `fast_engine`, with the MACD or
 *   with the momentum tracker, and `MomentumState` is passed to it by
 *   SHARED REFERENCE (`&self.momentum`), so it cannot mutate anything the
 *   fast column reads. Omitting it is exact for this column, not an
 *   approximation. `opposing_force` (:1010) is absent for the same reason.
 *
 * FIRST-VALID IGNORED: `compute_row` walks EVERY bar from 0 and `update`
 *   (:1286) RESETS the whole machine -- both EMAs, the signal EMA, both pivot
 *   ring buffers and every wave list -- on any non-finite bar. A global
 *   first-valid index would be wrong after the first hole.
 *
 * PERIOD-INVARIANT: the CPU batch reads sixteen NAMED parameters
 *   (cpu_batch.rs:5109-5209) -- `fast_depth`, `slow_depth`, `fast_length`,
 *   `slow_length`, `signal_length`, `lookback`, `err_tol` and nine `show_*`
 *   booleans -- and never `period`. All are pinned at the CPU defaults, so
 *   every row of a sweep is byte-identical.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. Three EMA recurrences, two
 *   fixed-span pivot ring buffers and two front-pushed pivot lists all carry
 *   across bars.
 *
 * THE SHORT-CIRCUIT THAT IS EASY TO MISS: `MacdState::update` (:857) is
 *     let fast = self.fast.update(close)?;
 *     let slow = self.slow.update(close)?;
 *   -- Rust `?` RETURNS EARLY, so on every bar before the FAST ema is seeded
 *   the SLOW ema is never updated at all. Its warm-up therefore does not start
 *   at bar 0; it starts at the bar the fast ema first emits. Writing the two
 *   updates unconditionally would seed the slow ema 21 bars early and shift
 *   the whole MACD. The same applies to the signal ema.
 *
 * ARITHMETIC taken verbatim:
 *   * the EMA seed is `sum / length` after exactly `length` FINITE updates
 *     (:818); the step is `alpha.mul_add(value, (1 - alpha) * prev)` (:825) --
 *     a fused multiply-add whose addend is itself a product, TWO roundings.
 *     `fma` is used for exactly that shape.
 *   * the pivot test is STRICT on both sides -- `center <= current` rejects
 *     for a high, `center >= current` rejects for a low (:934, :939) -- so a
 *     plateau is never a pivot.
 *   * `harmonic_family_code` (:1046) compares `|ratio - k| < err_tol` with
 *     err_tol 0.15, a MODEL TOLERANCE from the CPU parameter list, not a
 *     floating-point epsilon: it is not rescaled and must not be.
 *   * `xb_ratio` and `xd_ratio` fall back to 0.0 when `xa_len` is EXACTLY
 *     zero (:1155-1160) -- an exact test, not a tolerance.
 *
 * EPSILON: there is no floating-point epsilon on this path. `err_tol` is a
 *   harmonic-pattern tolerance in ratio units and is carried across unchanged.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:5109-5209 */
#define NEO_VD_FAST_DEPTH    9
#define NEO_VD_FAST_LENGTH   21
#define NEO_VD_SLOW_LENGTH   34
#define NEO_VD_SIGNAL_LENGTH 5
#define NEO_VD_LOOKBACK      3
#define NEO_VD_ERR_TOL       0.15
#define NEO_VD_SHOW_STANDARD  1
#define NEO_VD_SHOW_CLIMAX    1
#define NEO_VD_SHOW_ROUNDED   1
#define NEO_VD_SHOW_PREDATOR  1
#define NEO_VD_SHOW_GARTLEY   0
#define NEO_VD_SHOW_BAT       0
#define NEO_VD_SHOW_BUTTERFLY 0
#define NEO_VD_SHOW_CRAB      0
#define NEO_VD_SHOW_DEEP      0
#define NEO_VD_SHOW_HS        1

/* :118-125 */
#define NEO_VD_FAMILY_NONE           0.0
#define NEO_VD_FAMILY_RETRACEMENT    1.0
#define NEO_VD_FAMILY_GARTLEY        2.0
#define NEO_VD_FAMILY_BAT            3.0
#define NEO_VD_FAMILY_BUTTERFLY      4.0
#define NEO_VD_FAMILY_CRAB           5.0
#define NEO_VD_FAMILY_DEEP           6.0
#define NEO_VD_FAMILY_HEAD_SHOULDERS 7.0

#define NEO_VD_MOM_SPAN   NEO_VD_LOOKBACK
#define NEO_VD_MOM_WIN    (2 * NEO_VD_MOM_SPAN + 1)
#define NEO_VD_FAST_SPAN  NEO_VD_FAST_DEPTH
#define NEO_VD_FAST_WIN   (2 * NEO_VD_FAST_SPAN + 1)
/* push_front_cap (:945) keeps at most ten entries. */
#define NEO_VD_LIST_CAP   10

/* PivotDetector::update (:896). Returns true and writes the centre when the
 * middle of the ring is a strict extremum of the whole span. */
__device__ __forceinline__ bool neo_vd_pivot(double* __restrict__ win,
                                             int wlen,
                                             int span,
                                             int* head,
                                             int* count,
                                             bool is_high,
                                             double value,
                                             double* centre_out)
{
    if (!isfinite(value)) {
        *head = 0; *count = 0;
        for (int k = 0; k < wlen; ++k) win[k] = NEO_F64_NAN;
        return false;
    }

    win[*head] = value;
    *head += 1; if (*head == wlen) *head = 0;
    if (*count < wlen) *count += 1;
    if (*count < wlen) return false;

    const int start = *head;
    const double centre = win[(start + span) % wlen];
    if (!isfinite(centre)) return false;

    for (int j = 0; j < wlen; ++j) {
        if (j == span) continue;
        const double cur = win[(start + j) % wlen];
        if (!isfinite(cur)) return false;
        if (is_high) { if (centre <= cur) return false; }
        else         { if (centre >= cur) return false; }
    }
    *centre_out = centre;
    return true;
}

/* push_front_cap (:945) -- insert at the front, drop the eleventh. */
__device__ __forceinline__ void neo_vd_push_front(double* __restrict__ arr, int* cnt, double v)
{
    const int lim = (*cnt < NEO_VD_LIST_CAP) ? *cnt : (NEO_VD_LIST_CAP - 1);
    for (int k = lim; k >= 1; --k) arr[k] = arr[k - 1];
    arr[0] = v;
    if (*cnt < NEO_VD_LIST_CAP) *cnt += 1;
}

/* harmonic_family_code (:1046) */
__device__ __forceinline__ double neo_vd_family_code(double xb, double xd, double tol)
{
    if (fabs(xb - 0.618) < tol && fabs(xd - 0.786) < tol) return NEO_VD_FAMILY_GARTLEY;
    if (xb >= 0.382 - tol && xb <= 0.5 + tol && fabs(xd - 0.886) < tol) return NEO_VD_FAMILY_BAT;
    if (fabs(xb - 0.786) < tol && xd >= 1.27 - tol && xd <= 1.618 + tol) return NEO_VD_FAMILY_BUTTERFLY;
    if (xb >= 0.382 - tol && xb <= 0.618 + tol && fabs(xd - 1.618) < tol) return NEO_VD_FAMILY_CRAB;
    if (xd > 1.0) return NEO_VD_FAMILY_DEEP;
    return NEO_VD_FAMILY_RETRACEMENT;
}

/* standard_family_from_filters (:1071) with the default show_* flags. */
__device__ __forceinline__ double neo_vd_family_filter(double family, bool is_hs)
{
    if (is_hs) {
#if NEO_VD_SHOW_HS
        return NEO_VD_FAMILY_HEAD_SHOULDERS;
#else
        return NEO_VD_FAMILY_NONE;
#endif
    }
    const int f = (int)family;
    if (f == 1) return NEO_VD_FAMILY_RETRACEMENT;
#if NEO_VD_SHOW_GARTLEY
    if (f == 2) return NEO_VD_FAMILY_GARTLEY;
#endif
#if NEO_VD_SHOW_BAT
    if (f == 3) return NEO_VD_FAMILY_BAT;
#endif
#if NEO_VD_SHOW_BUTTERFLY
    if (f == 4) return NEO_VD_FAMILY_BUTTERFLY;
#endif
#if NEO_VD_SHOW_CRAB
    if (f == 5) return NEO_VD_FAMILY_CRAB;
#endif
#if NEO_VD_SHOW_DEEP
    if (f == 6) return NEO_VD_FAMILY_DEEP;
#endif
    return NEO_VD_FAMILY_NONE;
}

extern "C" __global__
void vdubus_divergence_wave_pattern_generator_neo_batch_f64(const double* __restrict__ high,
                                                            const double* __restrict__ low,
                                                            const double* __restrict__ close,
                                                            int n,
                                                            const int* __restrict__ periods,
                                                            int n_combos,
                                                            int first_valid,
                                                            double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* handled in place -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const double err_tol = NEO_VD_ERR_TOL;

    /* Three EmaState instances (:779-786). */
    const double a_fast = 2.0 / ((double)NEO_VD_FAST_LENGTH   + 1.0);
    const double a_slow = 2.0 / ((double)NEO_VD_SLOW_LENGTH   + 1.0);
    const double a_sig  = 2.0 / ((double)NEO_VD_SIGNAL_LENGTH + 1.0);
    int    c_fast = 0, c_slow = 0, c_sig = 0;
    double s_fast = 0.0, s_slow = 0.0, s_sig = 0.0;   /* seed sums */
    double v_fast = 0.0, v_slow = 0.0, v_sig = 0.0;   /* levels */
    bool   b_fast = false, b_slow = false, b_sig = false;

    /* MomentumState (:958) */
    double mom_hi_win[NEO_VD_MOM_WIN], mom_lo_win[NEO_VD_MOM_WIN];
    for (int k = 0; k < NEO_VD_MOM_WIN; ++k) { mom_hi_win[k] = NEO_F64_NAN; mom_lo_win[k] = NEO_F64_NAN; }
    int mom_hi_head = 0, mom_hi_count = 0, mom_lo_head = 0, mom_lo_count = 0;
    double wave_highs[NEO_VD_LIST_CAP], wave_lows[NEO_VD_LIST_CAP];
    int wave_highs_n = 0, wave_lows_n = 0;

    /* StructureEngine, FAST only (:1103) */
    double f_hi_win[NEO_VD_FAST_WIN], f_lo_win[NEO_VD_FAST_WIN];
    for (int k = 0; k < NEO_VD_FAST_WIN; ++k) { f_hi_win[k] = NEO_F64_NAN; f_lo_win[k] = NEO_F64_NAN; }
    int f_hi_head = 0, f_hi_count = 0, f_lo_head = 0, f_lo_count = 0;
    double pivots[NEO_VD_LIST_CAP];
    int pivots_n = 0;

    for (int i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], c = close[i];

        if (!(isfinite(h) && isfinite(l) && isfinite(c))) {
            /* State::reset (:1272) -- everything at once. */
            c_fast = c_slow = c_sig = 0;
            s_fast = s_slow = s_sig = 0.0;
            v_fast = v_slow = v_sig = 0.0;
            b_fast = b_slow = b_sig = false;
            mom_hi_head = mom_hi_count = mom_lo_head = mom_lo_count = 0;
            for (int k = 0; k < NEO_VD_MOM_WIN; ++k) { mom_hi_win[k] = NEO_F64_NAN; mom_lo_win[k] = NEO_F64_NAN; }
            wave_highs_n = 0; wave_lows_n = 0;
            f_hi_head = f_hi_count = f_lo_head = f_lo_count = 0;
            for (int k = 0; k < NEO_VD_FAST_WIN; ++k) { f_hi_win[k] = NEO_F64_NAN; f_lo_win[k] = NEO_F64_NAN; }
            pivots_n = 0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        /* ---- MacdState::update (:857). The `?` SHORT-CIRCUITS -- see the
         * header. `close` is finite here, so the EmaState non-finite arm
         * cannot fire; the guard that matters is the seeding count. ---- */
        bool   macd_ready = false;
        double hist = 0.0;

        bool fast_ok = false;
        if (!b_fast) {
            s_fast += c; c_fast += 1;
            if (c_fast == NEO_VD_FAST_LENGTH) { v_fast = s_fast / (double)NEO_VD_FAST_LENGTH; b_fast = true; fast_ok = true; }
        } else {
            v_fast = fma(a_fast, c, (1.0 - a_fast) * v_fast);
            fast_ok = true;
        }

        if (fast_ok) {
            bool slow_ok = false;
            if (!b_slow) {
                s_slow += c; c_slow += 1;
                if (c_slow == NEO_VD_SLOW_LENGTH) { v_slow = s_slow / (double)NEO_VD_SLOW_LENGTH; b_slow = true; slow_ok = true; }
            } else {
                v_slow = fma(a_slow, c, (1.0 - a_slow) * v_slow);
                slow_ok = true;
            }

            if (slow_ok) {
                const double macd = v_fast - v_slow;
                bool sig_ok = false;
                if (!b_sig) {
                    s_sig += macd; c_sig += 1;
                    if (c_sig == NEO_VD_SIGNAL_LENGTH) { v_sig = s_sig / (double)NEO_VD_SIGNAL_LENGTH; b_sig = true; sig_ok = true; }
                } else {
                    v_sig = fma(a_sig, macd, (1.0 - a_sig) * v_sig);
                    sig_ok = true;
                }
                if (sig_ok) { macd_ready = true; hist = macd - v_sig; }
            }
        }

        /* ---- MomentumState::update (:1000), only when the MACD emitted ---- */
        if (macd_ready) {
            double piv;
            if (neo_vd_pivot(mom_hi_win, NEO_VD_MOM_WIN, NEO_VD_MOM_SPAN,
                             &mom_hi_head, &mom_hi_count, true, hist, &piv)) {
                neo_vd_push_front(wave_highs, &wave_highs_n, piv);
            }
            if (neo_vd_pivot(mom_lo_win, NEO_VD_MOM_WIN, NEO_VD_MOM_SPAN,
                             &mom_lo_head, &mom_lo_count, false, hist, &piv)) {
                neo_vd_push_front(wave_lows, &wave_lows_n, piv);
            }
        }

        /* ---- StructureEngine::update (:1229), fast engine ---- */
        double fast_standard = 0.0;   /* EngineSignals::default (:1090) */
        {
            double piv;
            const bool hi_fired = neo_vd_pivot(f_hi_win, NEO_VD_FAST_WIN, NEO_VD_FAST_SPAN,
                                               &f_hi_head, &f_hi_count, true, h, &piv);
            if (hi_fired) {
                neo_vd_push_front(pivots, &pivots_n, piv);
                /* evaluate_bearish (:1113) */
                fast_standard = 0.0;
                if (pivots_n >= 5) {
                    const double y_d = pivots[0], y_b = pivots[2], y_a = pivots[3], y_x = pivots[4];
                    const double xa_len = fabs(y_a - y_x);
                    const double ab_len = fabs(y_b - y_a);
                    const double xb = (xa_len != 0.0) ? (ab_len / xa_len) : 0.0;
                    const double xd = (xa_len != 0.0) ? (fabs(y_d - y_x) / xa_len) : 0.0;
                    const double raw = neo_vd_family_code(xb, xd, err_tol);
                    const bool is_hs = (NEO_VD_SHOW_HS != 0) && (y_b > y_x) && (y_b > y_d);
                    /* standard_bearish (:1015) */
                    const bool std_mom = (wave_highs_n >= 3)
                                      && (wave_highs[1] < wave_highs[2])
                                      && (wave_highs[0] <= wave_highs[1]);
#if NEO_VD_SHOW_STANDARD
                    if (std_mom) {
                        const double fam = neo_vd_family_filter(raw, is_hs);
                        if (fam != NEO_VD_FAMILY_NONE) fast_standard = -fam;
                    }
#endif
                }
            }

            double piv2;
            const bool lo_fired = neo_vd_pivot(f_lo_win, NEO_VD_FAST_WIN, NEO_VD_FAST_SPAN,
                                               &f_lo_head, &f_lo_count, false, l, &piv2);
            if (lo_fired) {
                neo_vd_push_front(pivots, &pivots_n, piv2);
                /* evaluate_bullish (:1170) -- OVERWRITES the bearish result
                 * when both detectors fire on the same bar (:1234-1240). */
                fast_standard = 0.0;
                if (pivots_n >= 5) {
                    const double y_d = pivots[0], y_b = pivots[2], y_a = pivots[3], y_x = pivots[4];
                    const double xa_len = fabs(y_a - y_x);
                    const double ab_len = fabs(y_b - y_a);
                    const double xb = (xa_len != 0.0) ? (ab_len / xa_len) : 0.0;
                    const double xd = (xa_len != 0.0) ? (fabs(y_d - y_x) / xa_len) : 0.0;
                    const double raw = neo_vd_family_code(xb, xd, err_tol);
                    const bool is_inv_hs = (NEO_VD_SHOW_HS != 0) && (y_b < y_x) && (y_b < y_d);
                    /* standard_bullish (:1023) */
                    const bool std_mom = (wave_lows_n >= 3)
                                      && (wave_lows[1] > wave_lows[2])
                                      && (wave_lows[0] >= wave_lows[1]);
#if NEO_VD_SHOW_STANDARD
                    if (std_mom) {
                        const double fam = neo_vd_family_filter(raw, is_inv_hs);
                        if (fam != NEO_VD_FAMILY_NONE) fast_standard = fam;
                    }
#endif
                }
            }
        }

        /* `let (macd, signal, hist) = macd_out?;` (:1303) -- the whole row is
         * None, and therefore NaN, on a bar where the MACD has not emitted,
         * EVEN THOUGH the engines above already advanced their state. */
        o[i] = macd_ready ? fast_standard : NEO_F64_NAN;
    }
}
