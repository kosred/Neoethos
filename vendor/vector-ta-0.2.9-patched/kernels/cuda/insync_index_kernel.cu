// insync_index — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// This file used to be exactly one line:
//
//     extern "C" __global__ void insync_index_batch_f64() {}
//
// and `src/cuda/insync_index_wrapper.rs` called `get_function` on that empty
// symbol purely so resolution succeeded, then computed the whole indicator on
// the host through `Kernel::ScalarBatch` and uploaded the host answer with
// `DeviceBuffer::from_slice`, so the caller received a device pointer and could
// not tell the card had done nothing.
//
// CPU REFERENCE (the specification this was written against)
// ----------------------------------------------------------
//   src/indicators/insync_index.rs
//     :409  valid_bar
//     :442  RollingSmaState          :499  RollingVarianceState
//     :557  RollingCciState          :619  EmaState
//     :650  WilderRsiState           :736  RocState
//     :782  DpoState                 :841  MfiState
//     :933  StochState              :1024  EmoSignalState
//    :1082  MacdSignalState         :1124  RocSignalState
//    :1229  InsyncIndexStream::update_reset_on_nan   <- the per-bar body
//    :1304  insync_index_compute_into                <- the per-row loop
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW, walking bars in ascending order. The indicator
// is nine interlocking recurrences (a Wilder RSI, two EMAs, six ring-buffer
// means, two monotonic deques), and every one of them carries state from bar to
// bar. A bar-parallel decomposition would have to re-derive that state and
// would not reproduce the CPU's accumulation order, which is the thing that has
// to match: the score is built from threshold comparisons (`cci > 100.0`,
// `rsi > 70.0`, `position < 0.05`) where one ULP flips a ±5 contribution and
// therefore flips the whole indicator by 10.
//
// Threads loop `row = slot; row < rows; row += slots` so the scratch a row needs
// is bounded by the CARD, not by how wide a sweep the operator asked for.
//
// ARITHMETIC
// ----------
// * f64 end to end. No f32 literal, no f32-suffixed math function, no fast-math
//   intrinsic. The file is listed in `F64_LANE_SOURCES` in build.rs, so it is
//   compiled `-fmad=false -prec-div=true -prec-sqrt=true` and NEVER with
//   `--use_fast_math`.
// * `fma()` appears exactly where the CPU reference writes `f64::mul_add`
//   (`EmaState::update` :641, `RollingVarianceState::update` :541) and nowhere
//   else, so the fused steps stay fused and every other step stays unfused.
// * `fmax`/`fmin`, never a comparison chain, wherever the CPU writes `f64::max`
//   / `f64::min`. `f64::max` returns the NON-NaN operand; `a > b ? a : b`
//   returns `b` when either is NaN. Inside a recurrence that difference does
//   not perturb a value, it poisons every later bar.
// * No epsilon is carried over from anywhere. The only tolerances here are the
//   CPU's own exact comparisons against 0.0.

#include <cmath>
#include <cstdint>

#define INSYNC_DPO_DELAY 10

// ---------------------------------------------------------------------------
// Scratch layout
// ---------------------------------------------------------------------------
//
// The host lays every ring buffer out at a fixed stride `seg`, the maximum
// length any row asks for. A per-row packed layout would save memory and cost
// correctness: each row has different periods, so the offsets would differ per
// row and a single off-by-one would silently read another row's history.
//
// 16 double segments, 4 int segments, in this order.
#define SEG_BB          0
#define SEG_CCI         1
#define SEG_MFI_POS     2
#define SEG_MFI_NEG     3
#define SEG_ROC         4
#define SEG_EMO_SMA     5
#define SEG_EMO_AVG     6
#define SEG_MACD_TREND  7
#define SEG_DPO_CLOSE   8
#define SEG_DPO_SMA     9
#define SEG_DPO_HIST   10
#define SEG_ROC_SMA    11
#define SEG_STOCH_K    12
#define SEG_STOCH_D    13
#define SEG_STOCH_HI   14
#define SEG_STOCH_LO   15
#define SEG_DOUBLE_N   16

#define ISEG_DPO_HIST_OK 0
#define ISEG_DPO_DELAY   1
#define ISEG_STOCH_HI_IX 2
#define ISEG_STOCH_LO_IX 3
#define ISEG_INT_N       4

__device__ __forceinline__ double insync_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// --- RollingSmaState (:442) ------------------------------------------------
struct InsyncSma {
    double* buf;
    int period;
    int head;
    int len;
    double sum;
};

__device__ __forceinline__ void insync_sma_init(InsyncSma* s, double* buf, int period) {
    s->buf = buf;
    s->period = period < 1 ? 1 : period;
    s->head = 0;
    s->len = 0;
    s->sum = 0.0;
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
}

__device__ __forceinline__ void insync_sma_reset(InsyncSma* s) {
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
    s->head = 0;
    s->len = 0;
    s->sum = 0.0;
}

// Returns 1 and writes *out when the CPU returns Some, 0 when it returns None.
__device__ __forceinline__ int insync_sma_update(InsyncSma* s, double value, double* out) {
    if (s->period == 1) {
        s->buf[0] = value;
        s->len = 1;
        s->sum = value;
        *out = value;
        return 1;
    }
    if (s->len < s->period) {
        s->buf[s->len] = value;
        s->sum += value;
        s->len += 1;
        if (s->len == s->period) {
            *out = s->sum / static_cast<double>(s->period);
            return 1;
        }
        return 0;
    }
    double old = s->buf[s->head];
    s->buf[s->head] = value;
    s->sum += value - old;
    s->head += 1;
    if (s->head == s->period) {
        s->head = 0;
    }
    *out = s->sum / static_cast<double>(s->period);
    return 1;
}

// --- RollingVarianceState (:499) -------------------------------------------
struct InsyncVar {
    double* buf;
    int period;
    int head;
    int len;
    double sum;
    double sumsq;
};

__device__ __forceinline__ void insync_var_init(InsyncVar* s, double* buf, int period) {
    s->buf = buf;
    s->period = period < 1 ? 1 : period;
    s->head = 0;
    s->len = 0;
    s->sum = 0.0;
    s->sumsq = 0.0;
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
}

__device__ __forceinline__ void insync_var_reset(InsyncVar* s) {
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
    s->head = 0;
    s->len = 0;
    s->sum = 0.0;
    s->sumsq = 0.0;
}

__device__ __forceinline__ int insync_var_update(
    InsyncVar* s, double value, double* mean_out, double* sd_out) {
    if (s->len < s->period) {
        s->buf[s->len] = value;
        s->sum += value;
        s->sumsq += value * value;
        s->len += 1;
        if (s->len < s->period) {
            return 0;
        }
    } else {
        double old = s->buf[s->head];
        s->buf[s->head] = value;
        s->sum += value - old;
        // CPU: `self.sumsq += value.mul_add(value, -(old * old));` (:541)
        s->sumsq += fma(value, value, -(old * old));
        s->head += 1;
        if (s->head == s->period) {
            s->head = 0;
        }
    }
    double n = static_cast<double>(s->period);
    double mean = s->sum / n;
    // CPU: `(self.sumsq / n - mean * mean).max(0.0)` — f64::max, so a NaN here
    // becomes 0.0 rather than propagating. fmax has the same rule; `>` does not.
    double variance = fmax(s->sumsq / n - mean * mean, 0.0);
    *mean_out = mean;
    *sd_out = sqrt(variance);
    return 1;
}

// --- RollingCciState (:557) ------------------------------------------------
struct InsyncCci {
    double* buf;
    int period;
    int head;
    int len;
    double sum;
};

__device__ __forceinline__ void insync_cci_init(InsyncCci* s, double* buf, int period) {
    s->buf = buf;
    s->period = period < 1 ? 1 : period;
    s->head = 0;
    s->len = 0;
    s->sum = 0.0;
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
}

__device__ __forceinline__ void insync_cci_reset(InsyncCci* s) {
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
    s->head = 0;
    s->len = 0;
    s->sum = 0.0;
}

__device__ __forceinline__ int insync_cci_update(InsyncCci* s, double value, double* out) {
    if (s->len < s->period) {
        s->buf[s->len] = value;
        s->sum += value;
        s->len += 1;
        if (s->len < s->period) {
            return 0;
        }
    } else {
        double old = s->buf[s->head];
        s->buf[s->head] = value;
        s->sum += value - old;
        s->head += 1;
        if (s->head == s->period) {
            s->head = 0;
        }
    }
    double n = static_cast<double>(s->period);
    double mean = s->sum / n;
    // CPU sums |x - mean| in BUFFER ORDER (`self.buf.iter().take(period)`,
    // :601), not in time order. Summation order changes the last bits, and the
    // result feeds `cci > 100.0`, so buffer order it is.
    double mad = 0.0;
    for (int i = 0; i < s->period; ++i) {
        mad += fabs(s->buf[i] - mean);
    }
    mad /= n;
    if (mad == 0.0 || !isfinite(mad)) {
        return 0;
    }
    *out = (value - mean) / (0.015 * mad);
    return 1;
}

// --- EmaState (:619) -------------------------------------------------------
struct InsyncEma {
    double alpha;
    double value;
    int has;
};

__device__ __forceinline__ void insync_ema_init(InsyncEma* s, int period) {
    s->alpha = 2.0 / (static_cast<double>(period) + 1.0);
    s->value = 0.0;
    s->has = 0;
}

__device__ __forceinline__ void insync_ema_reset(InsyncEma* s) {
    s->value = 0.0;
    s->has = 0;
}

__device__ __forceinline__ double insync_ema_update(InsyncEma* s, double value) {
    double next;
    if (s->has) {
        // CPU: `self.alpha.mul_add(value, (1.0 - self.alpha) * prev)` (:641)
        next = fma(s->alpha, value, (1.0 - s->alpha) * s->value);
    } else {
        next = value;
    }
    s->value = next;
    s->has = 1;
    return next;
}

// --- WilderRsiState (:650) -------------------------------------------------
struct InsyncRsi {
    int period;
    double prev;
    int has_prev;
    double gains;
    double losses;
    int count;
    double avg_gain;
    double avg_loss;
    int has_avg;
};

__device__ __forceinline__ void insync_rsi_init(InsyncRsi* s, int period) {
    s->period = period < 1 ? 1 : period;
    s->prev = 0.0;
    s->has_prev = 0;
    s->gains = 0.0;
    s->losses = 0.0;
    s->count = 0;
    s->avg_gain = 0.0;
    s->avg_loss = 0.0;
    s->has_avg = 0;
}

__device__ __forceinline__ void insync_rsi_reset(InsyncRsi* s) {
    s->prev = 0.0;
    s->has_prev = 0;
    s->gains = 0.0;
    s->losses = 0.0;
    s->count = 0;
    s->avg_gain = 0.0;
    s->avg_loss = 0.0;
    s->has_avg = 0;
}

__device__ __forceinline__ double insync_rsi_from_avgs(double avg_gain, double avg_loss) {
    if (avg_gain == 0.0 && avg_loss == 0.0) {
        return 50.0;
    }
    if (avg_loss == 0.0) {
        return 100.0;
    }
    if (avg_gain == 0.0) {
        return 0.0;
    }
    double rs = avg_gain / avg_loss;
    return 100.0 - 100.0 / (1.0 + rs);
}

__device__ __forceinline__ int insync_rsi_update(InsyncRsi* s, double value, double* out) {
    if (!s->has_prev) {
        s->prev = value;
        s->has_prev = 1;
        return 0;
    }
    double change = value - s->prev;
    // CPU: `change.max(0.0)` / `(-change).max(0.0)` — f64::max again.
    double gain = fmax(change, 0.0);
    double loss = fmax(-change, 0.0);
    s->prev = value;

    if (!s->has_avg) {
        s->gains += gain;
        s->losses += loss;
        s->count += 1;
        if (s->count < s->period) {
            return 0;
        }
        double n = static_cast<double>(s->period);
        s->avg_gain = s->gains / n;
        s->avg_loss = s->losses / n;
        s->has_avg = 1;
        *out = insync_rsi_from_avgs(s->avg_gain, s->avg_loss);
        return 1;
    }

    double n = static_cast<double>(s->period);
    // CPU: `((avg * (period - 1.0)) + gain) / period` (:727) — three roundings,
    // NOT a mul_add. Reproduced literally.
    double avg_gain = ((s->avg_gain * (n - 1.0)) + gain) / n;
    double avg_loss = ((s->avg_loss * (n - 1.0)) + loss) / n;
    s->avg_gain = avg_gain;
    s->avg_loss = avg_loss;
    *out = insync_rsi_from_avgs(avg_gain, avg_loss);
    return 1;
}

// --- RocState (:736) -------------------------------------------------------
struct InsyncRoc {
    double* buf;
    int period;
    int head;
    int len;
};

__device__ __forceinline__ void insync_roc_init(InsyncRoc* s, double* buf, int period) {
    s->buf = buf;
    s->period = period < 1 ? 1 : period;
    s->head = 0;
    s->len = 0;
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
}

__device__ __forceinline__ void insync_roc_reset(InsyncRoc* s) {
    for (int i = 0; i < s->period; ++i) {
        s->buf[i] = 0.0;
    }
    s->head = 0;
    s->len = 0;
}

__device__ __forceinline__ int insync_roc_update(InsyncRoc* s, double value, double* out) {
    if (s->len < s->period) {
        s->buf[s->len] = value;
        s->len += 1;
        return 0;
    }
    double prev = s->buf[s->head];
    s->buf[s->head] = value;
    s->head += 1;
    if (s->head == s->period) {
        s->head = 0;
    }
    if (prev == 0.0 || !isfinite(prev)) {
        return 0;
    }
    *out = 100.0 * (value - prev) / prev;
    return 1;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The per-row body, lifted OUT of `insync_index_batch_f64` so the f64 lane can
// run the same code without the multi-row parameter arrays and the host
// scratch pointers. Nothing in it changed except the two parameters that were
// read straight out of a per-row array (`fast_length[row]`,
// `slow_length[row]`) and are now scalars. One implementation, two callers --
// the alternative was a second copy of 400 lines, which is the failure this
// lane exists to remove.
// ---------------------------------------------------------------------------
__device__ void insync_index_row_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    int p_bb,
    int p_cci,
    int p_mfi,
    int p_roc,
    int p_emo,
    int p_sma,
    int p_dpo,
    int p_rsi,
    int p_st,
    int p_std,
    int p_stk,
    int p_fast,
    int p_slow,
    double emo_div,
    double bb_mult,
    double* dbase,
    int* ibase,
    int seg,
    double* row_out
) {
    const double nan_value = insync_qnan();
        int barsback = p_dpo / 2 + 1;
        int hist_cap = barsback + 2;

        InsyncVar bb;
        insync_var_init(&bb, dbase + SEG_BB * seg, p_bb);
        InsyncCci cci;
        insync_cci_init(&cci, dbase + SEG_CCI * seg, p_cci);
        InsyncEma macd_fast, macd_slow;
        insync_ema_init(&macd_fast, p_fast);
        insync_ema_init(&macd_slow, p_slow);
        InsyncSma macd_trend;
        insync_sma_init(&macd_trend, dbase + SEG_MACD_TREND * seg, p_sma);
        InsyncSma emo_sma, emo_avg_sma;
        insync_sma_init(&emo_sma, dbase + SEG_EMO_SMA * seg, p_emo);
        insync_sma_init(&emo_avg_sma, dbase + SEG_EMO_AVG * seg, p_sma);
        InsyncSma dpo_close_sma, dpo_sma;
        insync_sma_init(&dpo_close_sma, dbase + SEG_DPO_CLOSE * seg, p_dpo);
        insync_sma_init(&dpo_sma, dbase + SEG_DPO_SMA * seg, p_sma);
        InsyncRoc roc;
        insync_roc_init(&roc, dbase + SEG_ROC * seg, p_roc);
        InsyncSma roc_sma;
        insync_sma_init(&roc_sma, dbase + SEG_ROC_SMA * seg, p_sma);
        InsyncRsi rsi;
        insync_rsi_init(&rsi, p_rsi);
        InsyncSma stoch_k_sma, stoch_d_sma;
        insync_sma_init(&stoch_k_sma, dbase + SEG_STOCH_K * seg, p_stk);
        insync_sma_init(&stoch_d_sma, dbase + SEG_STOCH_D * seg, p_std);

        // MfiState (:841)
        double* mfi_pos = dbase + SEG_MFI_POS * seg;
        double* mfi_neg = dbase + SEG_MFI_NEG * seg;
        int mfi_head = 0, mfi_len = 0, mfi_has_prev = 0;
        double mfi_prev_tp = 0.0, mfi_pos_sum = 0.0, mfi_neg_sum = 0.0;

        // DpoState deques (:782)
        double* dpo_hist = dbase + SEG_DPO_HIST * seg;
        int* dpo_hist_ok = ibase + ISEG_DPO_HIST_OK * seg;
        int dpo_hist_head = 0, dpo_hist_len = 0;
        int* dpo_delay = ibase + ISEG_DPO_DELAY * seg;
        int dpo_delay_head = 0, dpo_delay_len = 0;

        // EmoSignalState prev_hl2 (:1024)
        double emo_prev_hl2 = 0.0;
        int emo_has_prev = 0;

        // StochState monotonic deques (:933)
        double* st_hi = dbase + SEG_STOCH_HI * seg;
        double* st_lo = dbase + SEG_STOCH_LO * seg;
        int* st_hi_ix = ibase + ISEG_STOCH_HI_IX * seg;
        int* st_lo_ix = ibase + ISEG_STOCH_LO_IX * seg;
        int st_hi_head = 0, st_hi_len = 0, st_lo_head = 0, st_lo_len = 0;
        int st_index = 0;

        // The monotonic deques hold at most `st_len` entries after expiry, and
        // momentarily `st_len + 1` between the push and the expiry sweep — so
        // the ring needs one more slot than the window is wide. `seg` is sized
        // for that on the host.
        int st_len = p_st < 1 ? 1 : p_st;
        int cap_hi = st_len + 1;

        for (int i = 0; i < len; ++i) {
            double h = high[i];
            double l = low[i];
            double c = close[i];
            double v = volume[i];

            // valid_bar (:409)
            bool ok = isfinite(h) && isfinite(l) && isfinite(c) && isfinite(v) && v > 0.0 &&
                      h >= l;
            if (!ok) {
                insync_var_reset(&bb);
                insync_cci_reset(&cci);
                insync_ema_reset(&macd_fast);
                insync_ema_reset(&macd_slow);
                insync_sma_reset(&macd_trend);
                insync_sma_reset(&emo_sma);
                insync_sma_reset(&emo_avg_sma);
                insync_sma_reset(&dpo_close_sma);
                insync_sma_reset(&dpo_sma);
                insync_roc_reset(&roc);
                insync_sma_reset(&roc_sma);
                insync_rsi_reset(&rsi);
                insync_sma_reset(&stoch_k_sma);
                insync_sma_reset(&stoch_d_sma);
                mfi_head = 0;
                mfi_len = 0;
                mfi_has_prev = 0;
                mfi_pos_sum = 0.0;
                mfi_neg_sum = 0.0;
                for (int k = 0; k < (p_mfi < 1 ? 1 : p_mfi); ++k) {
                    mfi_pos[k] = 0.0;
                    mfi_neg[k] = 0.0;
                }
                dpo_hist_head = 0;
                dpo_hist_len = 0;
                dpo_delay_head = 0;
                dpo_delay_len = 0;
                emo_has_prev = 0;
                st_hi_head = 0;
                st_hi_len = 0;
                st_lo_head = 0;
                st_lo_len = 0;
                st_index = 0;
                row_out[i] = nan_value;
                continue;
            }

            double score = 50.0;

            // Bollinger position (:1239)
            double mean, sd;
            if (insync_var_update(&bb, c, &mean, &sd)) {
                double lower = mean - bb_mult * sd;
                double upper = mean + bb_mult * sd;
                double denom = upper - lower;
                if (denom > 0.0) {
                    double position = (c - lower) / denom;
                    if (position < 0.05) {
                        score -= 5.0;
                    } else if (position > 0.95) {
                        score += 5.0;
                    }
                }
            }

            // CCI (:1252)
            double cci_value;
            if (insync_cci_update(&cci, c, &cci_value)) {
                if (cci_value > 100.0) {
                    score += 5.0;
                } else if (cci_value < -100.0) {
                    score -= 5.0;
                }
            }

            // EmoSignalState::update (:1058)
            {
                double hl2 = 0.5 * (h + l);
                int component = 0;
                if (!emo_has_prev) {
                    emo_prev_hl2 = hl2;
                    emo_has_prev = 1;
                } else {
                    double prev_hl2 = emo_prev_hl2;
                    emo_prev_hl2 = hl2;
                    double raw = emo_div * (hl2 - prev_hl2) * (h - l) / v;
                    double emo_value;
                    if (insync_sma_update(&emo_sma, raw, &emo_value)) {
                        double emo_avg;
                        if (insync_sma_update(&emo_avg_sma, emo_value, &emo_avg)) {
                            double diff = emo_value - emo_avg;
                            if (diff < 0.0 && emo_avg < 0.0) {
                                component = -5;
                            } else if (diff > 0.0 && emo_avg > 0.0) {
                                component = 5;
                            }
                        }
                    }
                }
                score += static_cast<double>(component);
            }

            // MacdSignalState::update (:1108)
            {
                double macd = insync_ema_update(&macd_fast, c) - insync_ema_update(&macd_slow, c);
                int component = 0;
                double macd_avg;
                if (insync_sma_update(&macd_trend, macd, &macd_avg)) {
                    double diff = macd - macd_avg;
                    if (diff < 0.0 && macd_avg < 0.0) {
                        component = -5;
                    } else if (diff > 0.0 && macd_avg > 0.0) {
                        component = 5;
                    }
                }
                score += static_cast<double>(component);
            }

            // MFI (:1265)
            {
                double typical = (h + l + c) / 3.0;
                int have = 0;
                double mfi_value = 0.0;
                if (!mfi_has_prev) {
                    mfi_prev_tp = typical;
                    mfi_has_prev = 1;
                } else {
                    double pos = 0.0;
                    double neg = 0.0;
                    if (typical > mfi_prev_tp) {
                        pos = v * typical;
                    } else if (typical < mfi_prev_tp) {
                        neg = v * typical;
                    }
                    mfi_prev_tp = typical;

                    int period = p_mfi < 1 ? 1 : p_mfi;
                    int ready = 1;
                    if (mfi_len < period) {
                        mfi_pos[mfi_len] = pos;
                        mfi_neg[mfi_len] = neg;
                        mfi_pos_sum += pos;
                        mfi_neg_sum += neg;
                        mfi_len += 1;
                        if (mfi_len < period) {
                            ready = 0;
                        }
                    } else {
                        double old_pos = mfi_pos[mfi_head];
                        double old_neg = mfi_neg[mfi_head];
                        mfi_pos[mfi_head] = pos;
                        mfi_neg[mfi_head] = neg;
                        mfi_pos_sum += pos - old_pos;
                        mfi_neg_sum += neg - old_neg;
                        mfi_head += 1;
                        if (mfi_head == period) {
                            mfi_head = 0;
                        }
                    }
                    if (ready) {
                        have = 1;
                        if (mfi_pos_sum == 0.0 && mfi_neg_sum == 0.0) {
                            mfi_value = 50.0;
                        } else if (mfi_neg_sum == 0.0) {
                            mfi_value = 100.0;
                        } else if (mfi_pos_sum == 0.0) {
                            mfi_value = 0.0;
                        } else {
                            double rs = mfi_pos_sum / mfi_neg_sum;
                            mfi_value = 100.0 - 100.0 / (1.0 + rs);
                        }
                    }
                }
                if (have) {
                    if (mfi_value > 80.0) {
                        score += 5.0;
                    } else if (mfi_value < 20.0) {
                        score -= 5.0;
                    }
                }
            }

            // DpoState::update (:815)
            {
                double sma_now;
                int sma_ok = insync_sma_update(&dpo_close_sma, c, &sma_now);
                int tail = dpo_hist_head + dpo_hist_len;
                if (tail >= hist_cap) {
                    tail -= hist_cap;
                }
                dpo_hist[tail] = sma_ok ? sma_now : 0.0;
                dpo_hist_ok[tail] = sma_ok;
                dpo_hist_len += 1;

                int component = 0;
                if (dpo_hist_len > barsback) {
                    double past_sma = dpo_hist[dpo_hist_head];
                    int past_ok = dpo_hist_ok[dpo_hist_head];
                    dpo_hist_head += 1;
                    if (dpo_hist_head == hist_cap) {
                        dpo_hist_head = 0;
                    }
                    dpo_hist_len -= 1;
                    if (past_ok) {
                        double dpo = c - past_sma;
                        double avg;
                        if (insync_sma_update(&dpo_sma, dpo, &avg)) {
                            double diff = dpo - avg;
                            if (diff < 0.0 && avg < 0.0) {
                                component = -5;
                            } else if (diff > 0.0 && avg > 0.0) {
                                component = 5;
                            }
                        }
                    }
                }

                int dtail = dpo_delay_head + dpo_delay_len;
                if (dtail >= INSYNC_DPO_DELAY + 2) {
                    dtail -= INSYNC_DPO_DELAY + 2;
                }
                dpo_delay[dtail] = component;
                dpo_delay_len += 1;
                int emitted = 0;
                if (dpo_delay_len > INSYNC_DPO_DELAY) {
                    emitted = dpo_delay[dpo_delay_head];
                    dpo_delay_head += 1;
                    if (dpo_delay_head == INSYNC_DPO_DELAY + 2) {
                        dpo_delay_head = 0;
                    }
                    dpo_delay_len -= 1;
                }
                score += static_cast<double>(emitted);
            }

            // RocSignalState::update (:1150)
            {
                int component = 0;
                double roc_value;
                if (insync_roc_update(&roc, c, &roc_value)) {
                    double roc_avg;
                    if (insync_sma_update(&roc_sma, roc_value, &roc_avg)) {
                        double diff = roc_value - roc_avg;
                        if (diff < 0.0 && roc_avg < 0.0) {
                            component = -5;
                        } else if (diff > 0.0 && roc_avg > 0.0) {
                            component = 5;
                        }
                    }
                }
                score += static_cast<double>(component);
            }

            // RSI (:1274)
            double rsi_value;
            if (insync_rsi_update(&rsi, c, &rsi_value)) {
                if (rsi_value > 70.0) {
                    score += 5.0;
                } else if (rsi_value < 30.0) {
                    score -= 5.0;
                }
            }

            // StochState::update (:975)
            {
                int idx = st_index;
                st_index += 1;

                while (st_hi_len > 0) {
                    int back = st_hi_head + st_hi_len - 1;
                    if (back >= cap_hi) {
                        back -= cap_hi;
                    }
                    if (st_hi[back] <= h) {
                        st_hi_len -= 1;
                    } else {
                        break;
                    }
                }
                {
                    int tail = st_hi_head + st_hi_len;
                    if (tail >= cap_hi) {
                        tail -= cap_hi;
                    }
                    st_hi[tail] = h;
                    st_hi_ix[tail] = idx;
                    st_hi_len += 1;
                }

                while (st_lo_len > 0) {
                    int back = st_lo_head + st_lo_len - 1;
                    if (back >= cap_hi) {
                        back -= cap_hi;
                    }
                    if (st_lo[back] >= l) {
                        st_lo_len -= 1;
                    } else {
                        break;
                    }
                }
                {
                    int tail = st_lo_head + st_lo_len;
                    if (tail >= cap_hi) {
                        tail -= cap_hi;
                    }
                    st_lo[tail] = l;
                    st_lo_ix[tail] = idx;
                    st_lo_len += 1;
                }

                // CPU: `idx.saturating_add(1).saturating_sub(self.length)`
                int expire_before = idx + 1 - st_len;
                if (expire_before < 0) {
                    expire_before = 0;
                }
                while (st_hi_len > 0 && st_hi_ix[st_hi_head] < expire_before) {
                    st_hi_head += 1;
                    if (st_hi_head == cap_hi) {
                        st_hi_head = 0;
                    }
                    st_hi_len -= 1;
                }
                while (st_lo_len > 0 && st_lo_ix[st_lo_head] < expire_before) {
                    st_lo_head += 1;
                    if (st_lo_head == cap_hi) {
                        st_lo_head = 0;
                    }
                    st_lo_len -= 1;
                }

                if (idx + 1 >= st_len) {
                    double highest = st_hi_len > 0 ? st_hi[st_hi_head] : h;
                    double lowest = st_lo_len > 0 ? st_lo[st_lo_head] : l;
                    double denom = highest - lowest;
                    if (denom > 0.0 && isfinite(denom)) {
                        double fast = 100.0 * (c - lowest) / denom;
                        double k_value;
                        if (insync_sma_update(&stoch_k_sma, fast, &k_value)) {
                            if (k_value > 80.0) {
                                score += 5.0;
                            } else if (k_value < 20.0) {
                                score -= 5.0;
                            }
                            double d_value;
                            if (insync_sma_update(&stoch_d_sma, k_value, &d_value)) {
                                if (d_value > 80.0) {
                                    score += 5.0;
                                } else if (d_value < 20.0) {
                                    score -= 5.0;
                                }
                            }
                        }
                    }
                }
            }

            row_out[i] = score;
        }
}

extern "C" __global__ void insync_index_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ emo_divisor,
    const int* __restrict__ emo_length,
    const int* __restrict__ fast_length,
    const int* __restrict__ slow_length,
    const int* __restrict__ mfi_length,
    const int* __restrict__ bb_length,
    const double* __restrict__ bb_multiplier,
    const int* __restrict__ cci_length,
    const int* __restrict__ dpo_length,
    const int* __restrict__ roc_length,
    const int* __restrict__ rsi_length,
    const int* __restrict__ stoch_length,
    const int* __restrict__ stoch_d_length,
    const int* __restrict__ stoch_k_length,
    const int* __restrict__ sma_length,
    int rows,
    int slots,
    int seg,
    double* scratch,
    int* iscratch,
    double* out
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    double* dbase = scratch + static_cast<size_t>(slot) * static_cast<size_t>(SEG_DOUBLE_N) *
                                  static_cast<size_t>(seg);
    int* ibase = iscratch + static_cast<size_t>(slot) * static_cast<size_t>(ISEG_INT_N) *
                                static_cast<size_t>(seg);

    for (int row = slot; row < rows; row += slots) {
        double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

        int p_bb = bb_length[row];
        int p_cci = cci_length[row];
        int p_mfi = mfi_length[row];
        int p_roc = roc_length[row];
        int p_emo = emo_length[row];
        int p_sma = sma_length[row];
        int p_dpo = dpo_length[row];
        int p_rsi = rsi_length[row];
        int p_st = stoch_length[row];
        int p_std = stoch_d_length[row];
        int p_stk = stoch_k_length[row];
        double emo_div = static_cast<double>(emo_divisor[row]);
        double bb_mult = bb_multiplier[row];
        insync_index_row_f64(
            high, low, close, volume, len,
            p_bb, p_cci, p_mfi, p_roc, p_emo, p_sma, p_dpo, p_rsi, p_st, p_std,
            p_stk, fast_length[row], slow_length[row], emo_div, bb_mult,
            dbase, ibase, seg, row_out);
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `insync_index_with_kernel`
// (src/indicators/insync_index.rs:1322) -> `insync_index_compute_into` (:1304)
// -> `InsyncIndexStream::update_reset_on_nan`.
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `insync_index_batch_f64` (now :909) is double-clean AND already single-output
// -- it writes ONE `row_out[i] = score` -- but its ABI is twenty-six parameters:
// fifteen `const int*` / `const double*` per-row parameter arrays plus `slots`,
// `seg` and two host-allocated scratch pointers. The f64 lane launches ONE
// shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// with no scratch to give, so the lane gets its own entry point here.
//
// AND IT IS NOT A COPY. The 400-line per-row body was LIFTED into
// `insync_index_row_f64` (:462) and both entry points now call it. The only
// change to that body was turning `fast_length[row]` / `slow_length[row]` into
// scalar parameters. Duplicating it would have put two implementations of one
// indicator in one file, which is the failure this lane exists to remove.
//
// WHICH COLUMN: the single `value` series. `insync_index` has no `compute_*_
// batch` arm in `cpu_batch.rs` at all -- the CPU reference is the scalar
// `insync_index_with_kernel`, which returns one `values` vector.
//
// SHAPE: one thread per combo, bars ascending. Ten sub-indicators (Bollinger
// %b, CCI, MACD, EMV, MFI, DPO, ROC, RSI, stochastic K and D) each with their
// own carried state, and the CPU RESETS EVERY ONE of them at any bar that is
// not `valid_bar` (insync_index.rs:409 -- four-way finite, volume > 0 and
// high >= low). A bar-parallel form cannot know which segment it is in.
//
// SCRATCH IS PER-THREAD AND COMPILE-TIME BOUNDED. The lifted body lays every
// ring out at a stride `seg`; the host computes that stride as the widest ring
// any swept row asks for (insync_index_wrapper.rs:206-256). At the pinned
// defaults the widest is 20 (`bb_length` and `mfi_length`), and this kernel
// declares 24 -- above the widest and with headroom, so the same layout code
// runs unchanged. That is 16 * 24 = 384 doubles and 4 * 24 = 96 ints per
// thread: 3,456 bytes. Bounded at compile time, not allocated.
//
// PERIOD-INVARIANT: `insync_index` has no batch arm, so there is no `period`
// axis to read at all; the fifteen knobs below are the CPU defaults
// (insync_index.rs:28-42) and every swept period gives the same column. This
// kernel writes identical rows.
//
// ROUNDING, NaN SEMANTICS, EPSILONS: unchanged from the lifted body, which was
// written against the CPU reference and is f64 throughout. No f32 literal, no
// f32-suffixed math function, no fast-math intrinsic is introduced here; the
// NaN is `insync_qnan()`, the file's own DOUBLE quiet-NaN.
//
// FIRST VALID IS NOT READ: the CPU restarts all ten sub-indicators at every
// invalid bar, so one global warmup index would be wrong after the first hole.
// The lane row declares `F64FirstValidRule::Ignored`.
// ---------------------------------------------------------------------------

#define NEO_INSYNC_SEG 24
#define NEO_INSYNC_EMO_DIVISOR 10000
#define NEO_INSYNC_EMO_LENGTH 14
#define NEO_INSYNC_FAST_LENGTH 12
#define NEO_INSYNC_SLOW_LENGTH 26
#define NEO_INSYNC_MFI_LENGTH 20
#define NEO_INSYNC_BB_LENGTH 20
#define NEO_INSYNC_BB_MULTIPLIER 2.0
#define NEO_INSYNC_CCI_LENGTH 14
#define NEO_INSYNC_DPO_LENGTH 18
#define NEO_INSYNC_ROC_LENGTH 10
#define NEO_INSYNC_RSI_LENGTH 14
#define NEO_INSYNC_STOCH_LENGTH 14
#define NEO_INSYNC_STOCH_D_LENGTH 3
#define NEO_INSYNC_STOCH_K_LENGTH 1
#define NEO_INSYNC_SMA_LENGTH 10

extern "C" __global__ void insync_index_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
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

    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    const double nan_value = insync_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = nan_value;
    }

    double dbase[SEG_DOUBLE_N * NEO_INSYNC_SEG];
    int ibase[ISEG_INT_N * NEO_INSYNC_SEG];

    insync_index_row_f64(
        high, low, close, volume, n,
        NEO_INSYNC_BB_LENGTH,
        NEO_INSYNC_CCI_LENGTH,
        NEO_INSYNC_MFI_LENGTH,
        NEO_INSYNC_ROC_LENGTH,
        NEO_INSYNC_EMO_LENGTH,
        NEO_INSYNC_SMA_LENGTH,
        NEO_INSYNC_DPO_LENGTH,
        NEO_INSYNC_RSI_LENGTH,
        NEO_INSYNC_STOCH_LENGTH,
        NEO_INSYNC_STOCH_D_LENGTH,
        NEO_INSYNC_STOCH_K_LENGTH,
        NEO_INSYNC_FAST_LENGTH,
        NEO_INSYNC_SLOW_LENGTH,
        static_cast<double>(NEO_INSYNC_EMO_DIVISOR),
        NEO_INSYNC_BB_MULTIPLIER,
        dbase,
        ibase,
        NEO_INSYNC_SEG,
        row);
}
