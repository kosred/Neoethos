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
