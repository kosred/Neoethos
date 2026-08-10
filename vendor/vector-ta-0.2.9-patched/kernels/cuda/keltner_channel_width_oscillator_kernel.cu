#include <cmath>
#include <cstddef>

namespace {
constexpr int BANDS_STYLE_ATR = 0;
constexpr int BANDS_STYLE_TR = 1;
constexpr int BANDS_STYLE_RANGE = 2;

__device__ inline bool is_valid_bar(double high, double low, double close, double source) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(source) && high >= low;
}

__device__ inline void reset_rolling_sma(
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    *count = 0;
    *head = 0;
    *sum = 0.0;
    for (int i = 0; i < period; ++i) {
        buffer[i] = 0.0;
    }
}

__device__ inline double update_rolling_sma(
    double value,
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    if (isfinite(value)) {
        if (*count < period) {
            buffer[*count] = value;
            *sum += value;
            *count += 1;
        } else {
            const double old = buffer[*head];
            buffer[*head] = value;
            *sum += value - old;
            *head += 1;
            if (*head == period) {
                *head = 0;
            }
        }
    }

    return *count == period ? (*sum / static_cast<double>(period)) : NAN;
}

__device__ inline void reset_seeded_avg(
    int* count,
    double* sum,
    double* value,
    bool* seeded
) {
    *count = 0;
    *sum = 0.0;
    *value = NAN;
    *seeded = false;
}

__device__ inline double update_seeded_ema(
    double input,
    int period,
    double alpha,
    int* count,
    double* sum,
    double* value,
    bool* seeded
) {
    if (!*seeded) {
        *sum += input;
        *count += 1;
        if (*count == period) {
            *value = *sum / static_cast<double>(period);
            *seeded = true;
            return *value;
        }
        return NAN;
    }

    *value = fma(alpha, input - *value, *value);
    return *value;
}

__device__ inline double update_seeded_rma(
    double input,
    int period,
    double alpha,
    int* count,
    double* sum,
    double* value,
    bool* seeded
) {
    if (!*seeded) {
        *sum += input;
        *count += 1;
        if (*count == period) {
            *value = *sum / static_cast<double>(period);
            *seeded = true;
            return *value;
        }
        return NAN;
    }

    *value = fma(alpha, input - *value, *value);
    return *value;
}
}

extern "C" __global__ void keltner_channel_width_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    const double* source,
    int len,
    const int* lengths,
    const double* multipliers,
    const int* use_exponentials,
    const int* bands_styles,
    const int* atr_lengths,
    int rows,
    int max_length,
    double* out_kbw,
    double* out_kbw_sma,
    double* center_sma_buffers,
    double* width_sma_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double multiplier = multipliers[row];
    const int use_exponential = use_exponentials[row];
    const int bands_style = bands_styles[row];
    const int atr_length = atr_lengths[row];

    double* row_kbw = out_kbw + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_kbw_sma = out_kbw_sma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_center_sma =
        center_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_width_sma =
        width_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row_kbw[i] = NAN;
        row_kbw_sma[i] = NAN;
    }

    if (length <= 0 || length > max_length || atr_length <= 0 || !isfinite(multiplier)
        || multiplier < 0.0 || (use_exponential != 0 && use_exponential != 1)
        || bands_style < BANDS_STYLE_ATR || bands_style > BANDS_STYLE_RANGE) {
        return;
    }

    int center_sma_count = 0;
    int center_sma_head = 0;
    double center_sma_sum = 0.0;
    int center_ema_count = 0;
    double center_ema_sum = 0.0;
    double center_ema_value = NAN;
    bool center_ema_seeded = false;

    int width_sma_count = 0;
    int width_sma_head = 0;
    double width_sma_sum = 0.0;

    int atr_rma_count = 0;
    double atr_rma_sum = 0.0;
    double atr_rma_value = NAN;
    bool atr_rma_seeded = false;

    int range_rma_count = 0;
    double range_rma_sum = 0.0;
    double range_rma_value = NAN;
    bool range_rma_seeded = false;

    double prev_close = NAN;
    const double center_ema_alpha = 2.0 / (static_cast<double>(length) + 1.0);
    const double atr_rma_alpha = 1.0 / static_cast<double>(atr_length);
    const double range_rma_alpha = 1.0 / static_cast<double>(length);

    reset_rolling_sma(
        &center_sma_count,
        &center_sma_head,
        &center_sma_sum,
        row_center_sma,
        length
    );
    reset_rolling_sma(
        &width_sma_count,
        &width_sma_head,
        &width_sma_sum,
        row_width_sma,
        length
    );

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double s = source[i];

        if (!is_valid_bar(h, l, c, s)) {
            reset_rolling_sma(
                &center_sma_count,
                &center_sma_head,
                &center_sma_sum,
                row_center_sma,
                length
            );
            reset_rolling_sma(
                &width_sma_count,
                &width_sma_head,
                &width_sma_sum,
                row_width_sma,
                length
            );
            reset_seeded_avg(
                &center_ema_count,
                &center_ema_sum,
                &center_ema_value,
                &center_ema_seeded
            );
            reset_seeded_avg(
                &atr_rma_count,
                &atr_rma_sum,
                &atr_rma_value,
                &atr_rma_seeded
            );
            reset_seeded_avg(
                &range_rma_count,
                &range_rma_sum,
                &range_rma_value,
                &range_rma_seeded
            );
            prev_close = NAN;
            continue;
        }

        const double middle =
            use_exponential != 0
                ? update_seeded_ema(
                      s,
                      length,
                      center_ema_alpha,
                      &center_ema_count,
                      &center_ema_sum,
                      &center_ema_value,
                      &center_ema_seeded
                  )
                : update_rolling_sma(
                      s,
                      &center_sma_count,
                      &center_sma_head,
                      &center_sma_sum,
                      row_center_sma,
                      length
                  );

        const double tr = isfinite(prev_close)
                              ? fmax(h - l, fmax(fabs(h - prev_close), fabs(l - prev_close)))
                              : (h - l);
        prev_close = c;

        double range = NAN;
        if (bands_style == BANDS_STYLE_ATR) {
            range = update_seeded_rma(
                tr,
                atr_length,
                atr_rma_alpha,
                &atr_rma_count,
                &atr_rma_sum,
                &atr_rma_value,
                &atr_rma_seeded
            );
        } else if (bands_style == BANDS_STYLE_TR) {
            range = tr;
        } else {
            range = update_seeded_rma(
                h - l,
                length,
                range_rma_alpha,
                &range_rma_count,
                &range_rma_sum,
                &range_rma_value,
                &range_rma_seeded
            );
        }

        if (!isfinite(middle) || !isfinite(range)) {
            continue;
        }

        if (middle == 0.0) {
            row_kbw[i] = NAN;
            row_kbw_sma[i] = NAN;
            continue;
        }

        const double kbw = (2.0 * multiplier * range) / middle;
        const double kbw_sma = update_rolling_sma(
            kbw,
            &width_sma_count,
            &width_sma_head,
            &width_sma_sum,
            row_width_sma,
            length
        );
        row_kbw[i] = kbw;
        row_kbw_sma[i] = kbw_sma;
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic  : keltner_channel_width_oscillator_default_ema_atr_into,
//                   src/indicators/keltner_channel_width_oscillator.rs:872-1008.
//                   keltner_channel_width_oscillator_compute_into (:819-869)
//                   selects that body when use_exponential is true,
//                   bands_style is AverageTrueRange, length is 20 and
//                   atr_length is 10 -- which is EXACTLY the CPU default set
//                   (cpu_batch.rs:13077-13092), and this lane is
//                   PERIOD-INVARIANT, so it is the only body reachable here.
//   * refusals    : keltner_channel_width_oscillator_prepare, :744-815.
//   * warmup      : kbw_warmup = first + max(length, atr_length) - 1 (:422-430).
//   * emitted col : `kbw`. compute_keltner_channel_width_oscillator_batch
//                   (cpu_batch.rs:13065) maps output_id "value" -> out.kbw.
//   * PERIOD-INVARIANT: the batch reads source, length (20), multiplier (2.0),
//                   use_exponential (true), bands_style ("Average True Range")
//                   and atr_length (10) -- never `period`
//                   (cpu_batch.rs:13077-13092).
//
// SOURCE: the CPU default source is `close` (cpu_batch.rs:13080), so the Hlc
// triple carries every series this kernel reads. `is_valid_bar` (:400-402)
// tests high, low, close AND source, and with source == close that is three
// finiteness tests plus `high >= low`.
//
// FIRST-VALID IGNORED: that `high >= low` half is an ORDERING condition no
// F64FirstValidRule variant expresses, and it names a different bar on any
// frame with a crossed quote. Rather than add a variant every consumer would
// have to grow a field for, the kernel derives its own index -- exactly as
// `garman_klass_volatility` already does in this lane -- and the row starts
// all-NaN, so the CPU prefix is reproduced without reading the caller's value.
//
// f64 END TO END: `fma()` for the two `mul_add` lines (:934 and :969), plain
// `*`/`+` everywhere the CPU writes them separately. The `>`/`<` in the true
// range (:945-955) are transliterated as comparisons because the CPU writes
// them as comparisons and the branch is unreachable for a non-finite bar --
// is_valid_bar has already rejected it.
// ===========================================================================

#define KCWO_NEO_LENGTH 20
#define KCWO_NEO_ATR_LENGTH 10
#define KCWO_NEO_MULTIPLIER 2.0

static __device__ __forceinline__ double kcwo_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// is_valid_bar, :400-402, with source == close.
static __device__ __forceinline__ bool kcwo_neo_valid(double h, double l, double c) {
  return isfinite(h) && isfinite(l) && isfinite(c) && h >= l;
}

extern "C" __global__
void keltner_channel_width_oscillator_neo_batch_f64(const double* __restrict__ high,
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
  const double nn = kcwo_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int length = KCWO_NEO_LENGTH;
  const int atr_length = KCWO_NEO_ATR_LENGTH;
  const double multiplier = KCWO_NEO_MULTIPLIER;

  // prepare, :744-815.
  if (length > n) return;
  if (atr_length > n) return;
  int first = -1;
  for (int i = 0; i < n; ++i) {
    if (kcwo_neo_valid(high[i], low[i], close[i])) { first = i; break; }
  }
  if (first < 0) return;
  // width_needed_bars for AverageTrueRange = max(length, atr_length), :410-420.
  const int needed = (length > atr_length) ? length : atr_length;
  if (n - first < needed) return;

  // :874-877 -- the constants the default path hard-codes.
  const double ema_alpha = 2.0 / 21.0;
  const double rma_alpha = 0.1;
  const double width_scale = 2.0 * multiplier;

  int ema_count = 0;
  double ema_sum = 0.0, ema_value = nn;
  bool ema_seeded = false;

  int atr_count = 0;
  double atr_sum = 0.0, atr_value = nn;
  bool atr_seeded = false;

  double prev_close = nn;
  bool has_prev_close = false;

  for (int i = 0; i < n; ++i) {
    const double h = high[i], l = low[i], c = close[i];
    const double src = c;  // source == close

    if (!kcwo_neo_valid(h, l, c)) {
      // :909-926 -- the reset.
      ema_count = 0; ema_sum = 0.0; ema_value = nn; ema_seeded = false;
      atr_count = 0; atr_sum = 0.0; atr_value = nn; atr_seeded = false;
      prev_close = nn; has_prev_close = false;
      continue;
    }

    bool have_middle = false;
    double middle = 0.0;
    if (!ema_seeded) {
      ema_sum += src;
      ema_count += 1;
      if (ema_count == length) {
        ema_value = ema_sum / (double)length;
        ema_seeded = true;
        middle = ema_value; have_middle = true;
      }
    } else {
      ema_value = fma(ema_alpha, src - ema_value, ema_value);  // :934
      middle = ema_value; have_middle = true;
    }

    double tr;
    if (has_prev_close) {
      const double up = (h > prev_close) ? h : prev_close;
      const double dn = (l < prev_close) ? l : prev_close;
      tr = up - dn;
    } else {
      tr = h - l;
    }
    prev_close = c;
    has_prev_close = true;

    bool have_range = false;
    double range = 0.0;
    if (!atr_seeded) {
      atr_sum += tr;
      atr_count += 1;
      if (atr_count == atr_length) {
        atr_value = atr_sum / (double)atr_length;
        atr_seeded = true;
        range = atr_value; have_range = true;
      }
    } else {
      atr_value = fma(rma_alpha, tr - atr_value, atr_value);  // :969
      range = atr_value; have_range = true;
    }

    if (have_middle && have_range) {
      // :974-980. The CPU also folds kbw into a 20-deep SMA for the second
      // output; this lane emits kbw, so that ring is dead work here.
      row[i] = (isfinite(middle) && isfinite(range) && middle != 0.0)
                   ? ((width_scale * range) / middle)
                   : nn;
    }
  }
}
