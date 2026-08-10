#include <cmath>
#include <cstddef>

extern "C" __global__ void normalized_volume_true_range_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const double* __restrict__ outlier_ranges,
    const int* __restrict__ atr_lengths,
    const int* __restrict__ volume_lengths,
    const int* __restrict__ styles,
    int rows,
    double* __restrict__ out_normalized_volume,
    double* __restrict__ out_normalized_true_range,
    double* __restrict__ out_baseline,
    double* __restrict__ out_atr,
    double* __restrict__ out_average_volume
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double outlier_range = outlier_ranges[row];
    int atr_length = atr_lengths[row];
    int volume_length = volume_lengths[row];
    int style = styles[row];

    double* row_out_nv =
        out_normalized_volume + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_ntr =
        out_normalized_true_range + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_baseline =
        out_baseline + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_atr = out_atr + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_avg_vol =
        out_average_volume + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_nv[i] = NAN;
        row_out_ntr[i] = NAN;
        row_out_baseline[i] = NAN;
        row_out_atr[i] = NAN;
        row_out_avg_vol[i] = NAN;
    }

    if (!isfinite(outlier_range) || outlier_range < 0.5 || atr_length < 2 || volume_length < 2 ||
        style < 0 || style > 2) {
        return;
    }

    double* atr_ring = new double[static_cast<size_t>(atr_length)];
    double* volume_ring = new double[static_cast<size_t>(volume_length)];
    if (atr_ring == nullptr || volume_ring == nullptr) {
        if (atr_ring != nullptr) {
            delete[] atr_ring;
        }
        if (volume_ring != nullptr) {
            delete[] volume_ring;
        }
        return;
    }

    double abs_sum = 0.0;
    double volume_sum = 0.0;
    int count = 0;
    double abs_variance_sum = 0.0;
    int abs_qualifying_count = 0;
    double abs_positive_deviation = NAN;
    double volume_variance_sum = 0.0;
    int volume_qualifying_count = 0;
    double volume_positive_deviation = NAN;
    double prev_close = NAN;
    bool have_prev_close = false;
    bool atr_ready = false;
    double atr_first_value = NAN;
    int atr_head = 0;
    double atr_sum = 0.0;
    bool average_volume_ready = false;
    double average_volume_first_value = NAN;
    int average_volume_head = 0;
    double average_volume_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        bool valid = false;
        if (style == 0) {
            valid = isfinite(open[i]) && isfinite(close[i]) && isfinite(volume[i]);
        } else if (style == 1) {
            valid = isfinite(high[i]) && isfinite(low[i]) && isfinite(volume[i]);
        } else {
            valid = isfinite(close[i]) && isfinite(volume[i]);
        }

        if (!valid) {
            if (isfinite(close[i])) {
                prev_close = close[i];
                have_prev_close = true;
            }
            continue;
        }

        double start = 0.0;
        double finish = 0.0;
        if (style == 0) {
            start = open[i];
            finish = close[i];
        } else if (style == 1) {
            start = low[i];
            finish = high[i];
        } else {
            start = have_prev_close ? prev_close : close[i];
            finish = close[i];
        }

        prev_close = close[i];
        have_prev_close = true;

        double denom = fmin(start, finish);
        if (!isfinite(denom) || denom <= 0.0) {
            continue;
        }

        double abs_percent = fabs(finish - start) / denom;
        if (!isfinite(abs_percent)) {
            continue;
        }

        count += 1;
        abs_sum += abs_percent;
        volume_sum += volume[i];

        double count_f64 = static_cast<double>(count);
        double avg_abs_percent = abs_sum / count_f64;
        double avg_volume = volume_sum / count_f64;

        if (abs_percent > avg_abs_percent) {
            double delta = abs_percent - avg_abs_percent;
            abs_variance_sum += delta * delta;
            abs_qualifying_count += 1;
            if (abs_qualifying_count >= 2) {
                abs_positive_deviation =
                    sqrt(abs_variance_sum / static_cast<double>(abs_qualifying_count - 1));
            }
        }

        if (volume[i] > avg_volume) {
            double delta = volume[i] - avg_volume;
            volume_variance_sum += delta * delta;
            volume_qualifying_count += 1;
            if (volume_qualifying_count >= 2) {
                volume_positive_deviation =
                    sqrt(volume_variance_sum / static_cast<double>(volume_qualifying_count - 1));
            }
        }

        double abs_percent_max = isfinite(abs_positive_deviation)
            ? avg_abs_percent + abs_positive_deviation * outlier_range
            : NAN;
        double normalized_avg_percent =
            (isfinite(abs_percent_max) && abs_percent_max > 0.0) ? avg_abs_percent / abs_percent_max
                                                                 : NAN;
        double scale_factor =
            (isfinite(normalized_avg_percent) && normalized_avg_percent > 0.0 &&
             normalized_avg_percent < 1.0 && isfinite(volume_positive_deviation) &&
             volume_positive_deviation > 0.0)
            ? avg_volume * (1.0 - normalized_avg_percent) /
                (normalized_avg_percent * volume_positive_deviation)
            : NAN;
        double max_volume = (isfinite(scale_factor) && isfinite(volume_positive_deviation))
            ? avg_volume + volume_positive_deviation * scale_factor
            : NAN;
        double normalized_abs_percent =
            (isfinite(abs_percent_max) && abs_percent_max > 0.0)
            ? fmin(abs_percent, abs_percent_max) / abs_percent_max
            : NAN;
        double normalized_volume_ratio =
            (isfinite(max_volume) && max_volume > 0.0) ? fmin(volume[i], max_volume) / max_volume
                                                       : NAN;
        double normalized_avg_volume_ratio =
            (isfinite(max_volume) && max_volume > 0.0) ? avg_volume / max_volume : NAN;

        double nv = normalized_volume_ratio * 100.0;
        double ntr = normalized_abs_percent * 100.0;
        double baseline = normalized_avg_volume_ratio * 100.0;
        double atr_value = NAN;
        if (!atr_ready) {
            if (isfinite(ntr)) {
                atr_first_value = ntr;
                atr_ready = true;
                for (int j = 0; j < atr_length; ++j) {
                    atr_ring[j] = ntr;
                }
                atr_sum = ntr * static_cast<double>(atr_length);
                atr_head = (atr_length > 0) ? (1 % atr_length) : 0;
                atr_value = ntr;
            }
        } else {
            double sanitized = isfinite(ntr) ? ntr : atr_first_value;
            double old = atr_ring[atr_head];
            atr_ring[atr_head] = sanitized;
            atr_sum += sanitized - old;
            atr_head += 1;
            if (atr_head == atr_length) {
                atr_head = 0;
            }
            atr_value = atr_sum / static_cast<double>(atr_length);
        }

        double average_volume_value = NAN;
        if (!average_volume_ready) {
            if (isfinite(nv)) {
                average_volume_first_value = nv;
                average_volume_ready = true;
                for (int j = 0; j < volume_length; ++j) {
                    volume_ring[j] = nv;
                }
                average_volume_sum = nv * static_cast<double>(volume_length);
                average_volume_head = (volume_length > 0) ? (1 % volume_length) : 0;
                average_volume_value = nv;
            }
        } else {
            double sanitized = isfinite(nv) ? nv : average_volume_first_value;
            double old = volume_ring[average_volume_head];
            volume_ring[average_volume_head] = sanitized;
            average_volume_sum += sanitized - old;
            average_volume_head += 1;
            if (average_volume_head == volume_length) {
                average_volume_head = 0;
            }
            average_volume_value = average_volume_sum / static_cast<double>(volume_length);
        }

        if (!(isfinite(nv) && isfinite(ntr) && isfinite(baseline) && isfinite(atr_value) &&
              isfinite(average_volume_value))) {
            continue;
        }

        row_out_nv[i] = nv;
        row_out_ntr[i] = ntr;
        row_out_baseline[i] = baseline;
        row_out_atr[i] = atr_value;
        row_out_avg_vol[i] = average_volume_value;
    }

    delete[] atr_ring;
    delete[] volume_ring;
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 2, round 3
//
// WHY A SECOND ENTRY POINT
//
// normalized_volume_true_range_batch_f64 above is genuine double-in/double-out,
// but it takes 16 parameters and writes FIVE output matrices -- and it calls
// `new double[]` on the DEVICE for its two rings, which is a dynamic allocation
// inside a kernel. The f64 lane launches one shape -- (open, high, low, close,
// volume, n, periods, n_combos, first_valid, out) -- and allocates ONE output
// matrix, so that entry point cannot be reached from it. This twin's rings are
// fixed-size per-thread arrays at the CPU's own defaults, so nothing is
// allocated at run time.
//
// CPU REFERENCE
//   src/indicators/normalized_volume_true_range.rs:791
//     normalized_volume_true_range_with_kernel -> :741 compute_into_slices
//     -> NormalizedVolumeTrueRangeCore::update :482
//   PositiveDeviationState::update :368
//   FilledSmaState::update         :404
//
// THE COLUMN THIS EMITS is normalized_volume, which is what output_id ==
// "value" resolves to (cpu_batch.rs -- "normalized_volume" || "value").
//
// PERIOD-INVARIANT. compute_normalized_volume_true_range_batch reads
// true_range_style, outlier_range, atr_length and volume_length and NEVER
// period, so five swept periods give five identical CPU columns and this kernel
// emits five identical rows. All four CPU defaults are pinned below --
// style Body (NormalizedVolumeTrueRangeStyle::default(), and the batch's own
// None arm supplies Body explicitly), outlier_range 5.0, atr_length 14,
// volume_length 14.
//
// OPEN IS AN INPUT, which is why the lane row declares F64InputKind::Ohlcv5 and
// not Hlcv: in Body style the bar's range is close - open (:511), so a
// four-pointer shape would compute a different indicator while passing every
// length check.
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential on three
// counts: the running mean is over ALL bars seen so far (abs_sum / count, :531
// -- not a window), the positive-deviation state accumulates a variance sum
// that is never evicted (:371), and the two smoothing rings are seeded from the
// first finite value and then rolled (:404-429).
//
// WILDER-SHAPED? NO -- and the difference is measurable. FilledSmaState is a
// FILLED SIMPLE mean, not a Wilder recursion: on the first finite input it
// fills the whole ring with that value and sets sum = value * len (:410-411),
// then rolls with sum += sanitized - old (:424) -- ONE rounding for the
// difference, which is what the CPU writes and therefore what is written here.
// Expressing it as a Wilder (tr - atr).mul_add(inv_p, atr) would be a DIFFERENT
// indicator, not a rounding improvement.
//
// NaN SEMANTICS: denom is start.min(finish) -- f64::min, which returns the
// non-NaN operand -- so fmin is used, not a comparison chain. Same for
// abs_percent.min(abs_percent_max) and volume.min(max_volume).
//
// FIRST VALID IS NOT READ: normalized_volume_true_range_with_kernel allocates
// vec![0.0; len] and compute_into_slices writes EVERY index (NaN where the core
// returns None), so there is no warmup prefix to agree about. The lane row
// declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double sqrt/fabs/fmin, no f32-suffixed math
// function, no fast-math intrinsic. The CPU has no epsilon in this path -- the
// guards are literal > 0.0 and < 1.0 comparisons -- so none is invented here.
// The file is listed in F64_LANE_SOURCES.
// ---------------------------------------------------------------------------

#define NVTR_NEO_OUTLIER_RANGE 5.0
#define NVTR_NEO_ATR_LENGTH 14
#define NVTR_NEO_VOLUME_LENGTH 14

__device__ __forceinline__ double nvtr_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void normalized_volume_true_range_neo_batch_f64(
    const double* __restrict__ open,
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
    const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row_idx >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;
    // Style Body reads open, close and volume only. high and low are bound so
    // the lane's five-pointer arm cannot hand this kernel the wrong series, and
    // are deliberately unread -- exactly as the CPU's Body arm leaves them.
    (void)high;
    (void)low;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    const double qnan = nvtr_neo_qnan();
    for (int i = 0; i < n; ++i) {
        row[i] = qnan;
    }

    double atr_ring[NVTR_NEO_ATR_LENGTH];
    double volume_ring[NVTR_NEO_VOLUME_LENGTH];

    double abs_sum = 0.0;
    double volume_sum = 0.0;
    int count = 0;

    double abs_variance_sum = 0.0;
    int abs_qualifying_count = 0;
    double abs_positive_deviation = qnan;

    double volume_variance_sum = 0.0;
    int volume_qualifying_count = 0;
    double volume_positive_deviation = qnan;

    double prev_close = qnan;
    bool have_prev_close = false;

    bool atr_ready = false;
    double atr_first_value = qnan;
    int atr_head = 0;
    double atr_sum = 0.0;

    bool average_volume_ready = false;
    double average_volume_first_value = qnan;
    int average_volume_head = 0;
    double average_volume_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double o = open[i];
        const double c = close[i];
        const double v = volume[i];

        // :493-503 -- Body validity, and prev_close still advances on a bar
        // that is rejected but has a finite close.
        if (!isfinite(o) || !isfinite(c) || !isfinite(v)) {
            if (isfinite(c)) {
                prev_close = c;
                have_prev_close = true;
            }
            continue;
        }
        (void)prev_close;
        (void)have_prev_close;

        const double start = o;
        const double finish = c;
        prev_close = c;
        have_prev_close = true;

        // :516 -- f64::min, so fmin: a NaN operand must not win.
        const double denom = fmin(start, finish);
        if (!isfinite(denom) || denom <= 0.0) {
            continue;
        }

        const double abs_percent = fabs(finish - start) / denom;
        if (!isfinite(abs_percent)) {
            continue;
        }

        count += 1;
        abs_sum += abs_percent;
        volume_sum += v;

        const double count_f64 = static_cast<double>(count);
        const double avg_abs_percent = abs_sum / count_f64;
        const double avg_volume = volume_sum / count_f64;

        // PositiveDeviationState::update (:368-378): the returned value is
        // STICKY -- once two qualifying samples exist it is refreshed, and it is
        // returned unchanged on a bar that does not qualify.
        if (abs_percent > avg_abs_percent) {
            const double delta = abs_percent - avg_abs_percent;
            abs_variance_sum += delta * delta;
            abs_qualifying_count += 1;
        }
        if (abs_qualifying_count >= 2) {
            abs_positive_deviation =
                sqrt(abs_variance_sum / static_cast<double>(abs_qualifying_count - 1));
        }

        if (v > avg_volume) {
            const double delta = v - avg_volume;
            volume_variance_sum += delta * delta;
            volume_qualifying_count += 1;
        }
        if (volume_qualifying_count >= 2) {
            volume_positive_deviation =
                sqrt(volume_variance_sum / static_cast<double>(volume_qualifying_count - 1));
        }

        const double abs_percent_max = isfinite(abs_positive_deviation)
            ? (avg_abs_percent + abs_positive_deviation * NVTR_NEO_OUTLIER_RANGE)
            : qnan;
        const double normalized_avg_percent =
            (isfinite(abs_percent_max) && abs_percent_max > 0.0)
            ? (avg_abs_percent / abs_percent_max)
            : qnan;
        const double scale_factor =
            (isfinite(normalized_avg_percent) && normalized_avg_percent > 0.0 &&
             normalized_avg_percent < 1.0 && isfinite(volume_positive_deviation) &&
             volume_positive_deviation > 0.0)
            ? (avg_volume * (1.0 - normalized_avg_percent) /
               (normalized_avg_percent * volume_positive_deviation))
            : qnan;
        const double max_volume =
            (isfinite(scale_factor) && isfinite(volume_positive_deviation))
            ? (avg_volume + volume_positive_deviation * scale_factor)
            : qnan;
        const double normalized_abs_percent =
            (isfinite(abs_percent_max) && abs_percent_max > 0.0)
            ? (fmin(abs_percent, abs_percent_max) / abs_percent_max)
            : qnan;
        const double normalized_volume_ratio = (isfinite(max_volume) && max_volume > 0.0)
            ? (fmin(v, max_volume) / max_volume)
            : qnan;
        const double normalized_avg_volume_ratio = (isfinite(max_volume) && max_volume > 0.0)
            ? (avg_volume / max_volume)
            : qnan;

        const double nv = normalized_volume_ratio * 100.0;
        const double ntr = normalized_abs_percent * 100.0;
        const double baseline = normalized_avg_volume_ratio * 100.0;

        // FilledSmaState::update (:404-429) -- seed by filling the ring, then
        // roll with sum += sanitized - old (ONE rounding, the CPU's).
        double atr_value = qnan;
        if (!atr_ready) {
            if (isfinite(ntr)) {
                atr_first_value = ntr;
                atr_ready = true;
                for (int j = 0; j < NVTR_NEO_ATR_LENGTH; ++j) {
                    atr_ring[j] = ntr;
                }
                atr_sum = ntr * static_cast<double>(NVTR_NEO_ATR_LENGTH);
                atr_head = 1 % NVTR_NEO_ATR_LENGTH;
                atr_value = ntr;
            }
        } else {
            const double sanitized = isfinite(ntr) ? ntr : atr_first_value;
            const double old = atr_ring[atr_head];
            atr_ring[atr_head] = sanitized;
            atr_sum += sanitized - old;
            atr_head += 1;
            if (atr_head == NVTR_NEO_ATR_LENGTH) {
                atr_head = 0;
            }
            atr_value = atr_sum / static_cast<double>(NVTR_NEO_ATR_LENGTH);
        }

        double average_volume_value = qnan;
        if (!average_volume_ready) {
            if (isfinite(nv)) {
                average_volume_first_value = nv;
                average_volume_ready = true;
                for (int j = 0; j < NVTR_NEO_VOLUME_LENGTH; ++j) {
                    volume_ring[j] = nv;
                }
                average_volume_sum = nv * static_cast<double>(NVTR_NEO_VOLUME_LENGTH);
                average_volume_head = 1 % NVTR_NEO_VOLUME_LENGTH;
                average_volume_value = nv;
            }
        } else {
            const double sanitized = isfinite(nv) ? nv : average_volume_first_value;
            const double old = volume_ring[average_volume_head];
            volume_ring[average_volume_head] = sanitized;
            average_volume_sum += sanitized - old;
            average_volume_head += 1;
            if (average_volume_head == NVTR_NEO_VOLUME_LENGTH) {
                average_volume_head = 0;
            }
            average_volume_value =
                average_volume_sum / static_cast<double>(NVTR_NEO_VOLUME_LENGTH);
        }

        // :600-607 -- ALL FIVE columns must be finite or the CPU returns None
        // and the bar is NaN, including the four this kernel does not emit.
        if (!(isfinite(nv) && isfinite(ntr) && isfinite(baseline) && isfinite(atr_value) &&
              isfinite(average_volume_value))) {
            continue;
        }

        row[i] = nv;
    }
}
