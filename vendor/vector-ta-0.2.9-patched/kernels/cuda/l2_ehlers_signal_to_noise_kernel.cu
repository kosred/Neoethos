#include <cmath>
#include <cstdint>

static __device__ inline double ring_get(const double* buf, int center, int off, int size) {
    int idx = center + size - (off % size);
    if (idx >= size) {
        idx -= size;
    }
    return buf[idx];
}

extern "C" __global__ void l2_ehlers_signal_to_noise_batch_f64(
    const double* source,
    const double* high,
    const double* low,
    int len,
    const int* smooth_periods,
    int rows,
    double* out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int smooth_period = smooth_periods[row];
    if (smooth_period <= 0) {
        return;
    }

    const double nan = NAN;
    const double ln10 = 2.3025850929940459;
    const double two_pi = 6.2831853071795865;
    const double period_mult = 0.075 * static_cast<double>(smooth_period) + 0.54;

    double source_ring[4] = {0.0, 0.0, 0.0, 0.0};
    double smooth_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double detrender_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double i1_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int source_idx = 0;
    int smooth_idx = 0;
    int detrender_idx = 0;
    int q1_idx = 0;
    int i1_idx = 0;
    int valid_count = 0;

    double range_1 = 0.0;
    double i2 = 0.0;
    double q2 = 0.0;
    double re = 0.0;
    double im = 0.0;
    double period = 0.0;
    double snr = 0.0;

    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        double src = source[i];
        double hi = high[i];
        double lo = low[i];
        if (!(isfinite(src) && isfinite(hi) && isfinite(lo))) {
            row_out[i] = nan;
            continue;
        }

        range_1 = 0.1 * (hi - lo) + 0.9 * range_1;
        source_ring[source_idx] = src;

        double smooth = 0.0;
        double detrender = 0.0;
        double i1 = 0.0;
        double q1 = 0.0;

        if (valid_count > 5) {
            double x0 = ring_get(source_ring, source_idx, 0, 4);
            double x1 = ring_get(source_ring, source_idx, 1, 4);
            double x2 = ring_get(source_ring, source_idx, 2, 4);
            double x3 = ring_get(source_ring, source_idx, 3, 4);
            smooth = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;

            smooth_ring[smooth_idx] = smooth;
            double s0 = ring_get(smooth_ring, smooth_idx, 0, 7);
            double s2 = ring_get(smooth_ring, smooth_idx, 2, 7);
            double s4 = ring_get(smooth_ring, smooth_idx, 4, 7);
            double s6 = ring_get(smooth_ring, smooth_idx, 6, 7);
            detrender = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) * period_mult;

            detrender_ring[detrender_idx] = detrender;
            i1 = ring_get(detrender_ring, detrender_idx, 3, 7);
            i1_ring[i1_idx] = i1;

            double d0 = ring_get(detrender_ring, detrender_idx, 0, 7);
            double d2 = ring_get(detrender_ring, detrender_idx, 2, 7);
            double d4 = ring_get(detrender_ring, detrender_idx, 4, 7);
            double d6 = ring_get(detrender_ring, detrender_idx, 6, 7);
            q1 = (0.0962 * d0 + 0.5769 * d2 - 0.5769 * d4 - 0.0962 * d6) * period_mult;
            q1_ring[q1_idx] = q1;

            double i0 = ring_get(i1_ring, i1_idx, 0, 7);
            double i2_hist = ring_get(i1_ring, i1_idx, 2, 7);
            double i4 = ring_get(i1_ring, i1_idx, 4, 7);
            double i6 = ring_get(i1_ring, i1_idx, 6, 7);
            double ji = (0.0962 * i0 + 0.5769 * i2_hist - 0.5769 * i4 - 0.0962 * i6) * period_mult;

            double q0 = ring_get(q1_ring, q1_idx, 0, 7);
            double q2_hist = ring_get(q1_ring, q1_idx, 2, 7);
            double q4 = ring_get(q1_ring, q1_idx, 4, 7);
            double q6 = ring_get(q1_ring, q1_idx, 6, 7);
            double jq = (0.0962 * q0 + 0.5769 * q2_hist - 0.5769 * q4 - 0.0962 * q6) * period_mult;

            double prev_i2 = i2;
            double prev_q2 = q2;
            double prev_re = re;
            double prev_im = im;
            double prev_period = period;
            double prev_snr = snr;

            i2 = 0.2 * (i1 - jq) + 0.8 * prev_i2;
            q2 = 0.2 * (q1 + ji) + 0.8 * prev_q2;

            double re_raw = i2 * prev_i2 + q2 * prev_q2;
            double im_raw = i2 * prev_q2 - q2 * prev_i2;
            re = 0.2 * re_raw + 0.8 * prev_re;
            im = 0.2 * im_raw + 0.8 * prev_im;

            double next_period = prev_period;
            if (re != 0.0 && im != 0.0) {
                double angle = atan2(im, re);
                if (angle != 0.0) {
                    next_period = two_pi / fabs(angle);
                }
            }
            if (prev_period != 0.0) {
                double upper = 1.5 * prev_period;
                double lower = 0.67 * prev_period;
                if (next_period > upper) {
                    next_period = upper;
                }
                if (next_period < lower) {
                    next_period = lower;
                }
            }
            if (next_period < 6.0) {
                next_period = 6.0;
            }
            if (next_period > 50.0) {
                next_period = 50.0;
            }
            period = 0.2 * next_period + 0.8 * prev_period;

            double power = i1 * i1 + q1 * q1;
            double noise = range_1 * range_1;
            if (power > 0.0 && noise > 0.0) {
                double snr_raw = 10.0 * log(power / noise) / ln10 + 6.0;
                snr = 0.25 * snr_raw + 0.75 * prev_snr;
            } else {
                snr = prev_snr;
            }
        } else {
            smooth_ring[smooth_idx] = smooth;
            detrender_ring[detrender_idx] = detrender;
            i1_ring[i1_idx] = i1;
            q1_ring[q1_idx] = q1;
        }

        valid_count += 1;
        source_idx = (source_idx + 1) % 4;
        smooth_idx = (smooth_idx + 1) % 7;
        detrender_idx = (detrender_idx + 1) % 7;
        i1_idx = (i1_idx + 1) % 7;
        q1_idx = (q1_idx + 1) % 7;

        row_out[i] = valid_count <= 6 ? nan : snr;
    }
}

// ===========================================================================
// f64 LANE  --  closer C3
// ===========================================================================
//
// CPU REFERENCE
// -------------
//   src/indicators/l2_ehlers_signal_to_noise.rs
//     :28  DEFAULT_SMOOTH_PERIOD = 10   :29 MIN_WARMUP_BARS = 6
//     :263 first_valid_triple           <- source/high/low all `is_finite`
//     :296 ring_get
//     :328 SignalToNoiseCore::new       <- period_mult = 0.075*sp + 0.54
//     :368 Core::update_clean           <- the per-bar body
//     :490 compute_..._into             <- clean-then-core selection
//     :516 compute_..._clean
//   dispatch: cpu_batch.rs:9870 -- reads `source` (default "hl2") and
//   `smooth_period` (default 10) and NEVER `period`.
//
// INPUT SHAPE, AND WHY (high, low) IS THE FAITHFUL ONE
// ---------------------------------------------------
// The CPU source defaults to `hl2`, and `Candles::compute_hl2`
// (src/utilities/data_loader.rs:168) is exactly `(h + l) / 2.0`. So the kernel
// takes high and low and forms the source itself with that same expression --
// one add, one divide by 2.0, which is exact in binary floating point. Passing
// a precomputed close series instead would compute a DIFFERENT indicator, and
// passing an Hlc triple would adopt close first-valid: `first_valid_triple`
// (:263) scans source/high/low, and with source == hl2 that is exactly "high
// and low both finite", i.e. `F64FirstValidRule::HighLowFinite`. Note the CPU
// uses `is_finite`, not `!is_nan`: an INFINITE high is skipped by the CPU and
// would be accepted by `AllInputsNonNan`.
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW walking bars ascending. Five interlocking IIR
// recurrences (range_1, i2, q2, re, im, period, snr) plus four rings; nothing
// here is bar-parallel.
//
// PERIOD-INVARIANT -- `smooth_period`, never `period`.
//
// ARITHMETIC
// ----------
// f64 end to end. `atan2`, `log`, `fabs` -- never the f32-suffixed forms. The
// CPU writes `snr_raw = 10.0 * (power/noise).ln() / LN_10 + 6.0`; the divide by
// `LN_10` is kept as a DIVIDE and not folded into a multiply by 1/LN_10,
// because that would be a different rounding. `period.clamp(6.0, 50.0)` is
// `fmin(fmax(period, 6.0), 50.0)`: `f64::clamp` propagates a NaN input, and so
// does that pair, whereas an if-chain would silently keep the NaN and then feed
// it into `self.period` for every later bar.

#define L2_NEO_DEFAULT_SMOOTH_PERIOD 10
#define L2_NEO_MIN_WARMUP_BARS 6
#define L2_NEO_PI 3.14159265358979323846
#define L2_NEO_LN10 2.30258509299404568402

__device__ __forceinline__ double l2_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `ring_get` (:296): idx = center + N - (off % N), wrapped once.
__device__ __forceinline__ double l2_neo_ring_get(const double* buf, int n, int center, int off) {
    int idx = center + n - (off % n);
    if (idx >= n) idx -= n;
    return buf[idx];
}

// One bar of `Core::update_clean` (:368). `state` layout is spelled out by the
// struct at :328 and every field is carried by reference so the recurrence is
// visible rather than hidden behind a helper.
struct L2NeoCore {
    double period_mult;
    double source_ring[4];
    int    source_idx;
    double smooth_ring[7];
    int    smooth_idx;
    double detrender_ring[7];
    int    detrender_idx;
    double q1_ring[7];
    int    q1_idx;
    double i1_ring[7];
    int    i1_idx;
    double range_1;
    double i2;
    double q2;
    double re;
    double im;
    double period;
    double snr;
    int    valid_count;
};

__device__ __forceinline__ void l2_neo_core_init(L2NeoCore* s, int smooth_period) {
    s->period_mult = 0.075 * static_cast<double>(smooth_period) + 0.54;
    for (int i = 0; i < 4; ++i) s->source_ring[i] = 0.0;
    s->source_idx = 0;
    for (int i = 0; i < 7; ++i) { s->smooth_ring[i] = 0.0; s->detrender_ring[i] = 0.0;
                                  s->q1_ring[i] = 0.0; s->i1_ring[i] = 0.0; }
    s->smooth_idx = 0; s->detrender_idx = 0; s->q1_idx = 0; s->i1_idx = 0;
    s->range_1 = 0.0; s->i2 = 0.0; s->q2 = 0.0; s->re = 0.0; s->im = 0.0;
    s->period = 0.0; s->snr = 0.0; s->valid_count = 0;
}

__device__ __forceinline__ double l2_neo_update_clean(L2NeoCore* s,
                                                      double source,
                                                      double high,
                                                      double low) {
    s->range_1 = 0.1 * (high - low) + 0.9 * s->range_1;
    s->source_ring[s->source_idx] = source;

    double smooth = 0.0;
    double detrender = 0.0;
    double i1 = 0.0;
    double q1 = 0.0;

    if (s->valid_count > 5) {
        const double x0 = l2_neo_ring_get(s->source_ring, 4, s->source_idx, 0);
        const double x1 = l2_neo_ring_get(s->source_ring, 4, s->source_idx, 1);
        const double x2 = l2_neo_ring_get(s->source_ring, 4, s->source_idx, 2);
        const double x3 = l2_neo_ring_get(s->source_ring, 4, s->source_idx, 3);
        smooth = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;

        s->smooth_ring[s->smooth_idx] = smooth;
        const double s0 = l2_neo_ring_get(s->smooth_ring, 7, s->smooth_idx, 0);
        const double s2 = l2_neo_ring_get(s->smooth_ring, 7, s->smooth_idx, 2);
        const double s4 = l2_neo_ring_get(s->smooth_ring, 7, s->smooth_idx, 4);
        const double s6 = l2_neo_ring_get(s->smooth_ring, 7, s->smooth_idx, 6);
        detrender = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) * s->period_mult;

        s->detrender_ring[s->detrender_idx] = detrender;
        i1 = l2_neo_ring_get(s->detrender_ring, 7, s->detrender_idx, 3);
        s->i1_ring[s->i1_idx] = i1;

        const double d0 = l2_neo_ring_get(s->detrender_ring, 7, s->detrender_idx, 0);
        const double d2 = l2_neo_ring_get(s->detrender_ring, 7, s->detrender_idx, 2);
        const double d4 = l2_neo_ring_get(s->detrender_ring, 7, s->detrender_idx, 4);
        const double d6 = l2_neo_ring_get(s->detrender_ring, 7, s->detrender_idx, 6);
        q1 = (0.0962 * d0 + 0.5769 * d2 - 0.5769 * d4 - 0.0962 * d6) * s->period_mult;
        s->q1_ring[s->q1_idx] = q1;

        const double i0 = l2_neo_ring_get(s->i1_ring, 7, s->i1_idx, 0);
        const double ii2 = l2_neo_ring_get(s->i1_ring, 7, s->i1_idx, 2);
        const double i4 = l2_neo_ring_get(s->i1_ring, 7, s->i1_idx, 4);
        const double i6 = l2_neo_ring_get(s->i1_ring, 7, s->i1_idx, 6);
        const double ji = (0.0962 * i0 + 0.5769 * ii2 - 0.5769 * i4 - 0.0962 * i6) * s->period_mult;

        const double q0 = l2_neo_ring_get(s->q1_ring, 7, s->q1_idx, 0);
        const double q2_hist = l2_neo_ring_get(s->q1_ring, 7, s->q1_idx, 2);
        const double q4 = l2_neo_ring_get(s->q1_ring, 7, s->q1_idx, 4);
        const double q6 = l2_neo_ring_get(s->q1_ring, 7, s->q1_idx, 6);
        const double jq = (0.0962 * q0 + 0.5769 * q2_hist - 0.5769 * q4 - 0.0962 * q6) * s->period_mult;

        const double prev_i2 = s->i2;
        const double prev_q2 = s->q2;
        const double prev_re = s->re;
        const double prev_im = s->im;
        const double prev_period = s->period;
        const double prev_snr = s->snr;

        s->i2 = 0.2 * (i1 - jq) + 0.8 * prev_i2;
        s->q2 = 0.2 * (q1 + ji) + 0.8 * prev_q2;

        const double re_raw = s->i2 * prev_i2 + s->q2 * prev_q2;
        const double im_raw = s->i2 * prev_q2 - s->q2 * prev_i2;
        s->re = 0.2 * re_raw + 0.8 * prev_re;
        s->im = 0.2 * im_raw + 0.8 * prev_im;

        double period = prev_period;
        if (s->re != 0.0 && s->im != 0.0) {
            const double a = atan2(s->im, s->re);
            if (a != 0.0) {
                period = (2.0 * L2_NEO_PI) / fabs(a);
            }
        }
        if (prev_period != 0.0) {
            const double upper = 1.5 * prev_period;
            const double lower = 0.67 * prev_period;
            if (period > upper) period = upper;
            if (period < lower) period = lower;
        }
        // `f64::clamp` -- fmin/fmax, never a comparison chain, so a NaN cannot
        // survive into `s->period` and poison every later bar.
        period = fmin(fmax(period, 6.0), 50.0);
        s->period = 0.2 * period + 0.8 * prev_period;

        const double power = i1 * i1 + q1 * q1;
        const double noise = s->range_1 * s->range_1;
        if (power > 0.0 && noise > 0.0) {
            const double snr_raw = 10.0 * log(power / noise) / L2_NEO_LN10 + 6.0;
            s->snr = 0.25 * snr_raw + 0.75 * prev_snr;
        } else {
            s->snr = prev_snr;
        }
    } else {
        s->smooth_ring[s->smooth_idx] = smooth;
        s->detrender_ring[s->detrender_idx] = detrender;
        s->i1_ring[s->i1_idx] = i1;
        s->q1_ring[s->q1_idx] = q1;
    }

    s->valid_count += 1;
    s->source_idx += 1; if (s->source_idx == 4) s->source_idx = 0;
    s->smooth_idx += 1; if (s->smooth_idx == 7) s->smooth_idx = 0;
    s->detrender_idx += 1; if (s->detrender_idx == 7) s->detrender_idx = 0;
    s->i1_idx += 1; if (s->i1_idx == 7) s->i1_idx = 0;
    s->q1_idx += 1; if (s->q1_idx == 7) s->q1_idx = 0;

    if (s->valid_count <= L2_NEO_MIN_WARMUP_BARS) return l2_neo_qnan();
    return s->snr;
}

extern "C" __global__ void l2_ehlers_signal_to_noise_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos) return;

    const double nan_d = l2_neo_qnan();
    double* __restrict__ o = out + static_cast<size_t>(row) * static_cast<size_t>(n);

    // PERIOD-INVARIANT (cpu_batch.rs:9894-9896 reads `source` and
    // `smooth_period` only).
    (void)periods;

    for (int i = 0; i < n; ++i) o[i] = nan_d;
    if (n <= 0 || first_valid < 0 || first_valid >= n) return;
    // validate_input :281 -- `valid < MIN_WARMUP_BARS + 1` is an Err.
    if (n - first_valid < L2_NEO_MIN_WARMUP_BARS + 1) return;

    L2NeoCore s;
    l2_neo_core_init(&s, L2_NEO_DEFAULT_SMOOTH_PERIOD);

    // ---- clean path (:516): walk from `first`, bail on the first non-finite.
    bool clean = true;
    for (int i = first_valid; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double src = (h + l) / 2.0;              // Candles::compute_hl2
        if (!(isfinite(src) && isfinite(h) && isfinite(l))) { clean = false; break; }
        o[i] = l2_neo_update_clean(&s, src, h, l);
    }
    if (clean) return;

    // ---- core path (:500): the CPU RESTARTS from index 0 with a fresh Core,
    // and `Core::update` (:352) returns NaN for a non-finite bar WITHOUT
    // advancing any state.
    for (int i = 0; i < n; ++i) o[i] = nan_d;
    l2_neo_core_init(&s, L2_NEO_DEFAULT_SMOOTH_PERIOD);
    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double src = (h + l) / 2.0;
        if (!(isfinite(src) && isfinite(h) && isfinite(l))) { o[i] = nan_d; continue; }
        o[i] = l2_neo_update_clean(&s, src, h, l);
    }
}
