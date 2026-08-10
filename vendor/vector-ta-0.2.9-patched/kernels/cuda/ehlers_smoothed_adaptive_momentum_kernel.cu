#include <cmath>
#include <cstdint>

static __device__ inline double esam_nz(double value) {
    return isfinite(value) ? value : 0.0;
}

static __device__ inline double esam_median3(double a, double b, double c) {
    if (!(isfinite(a) && isfinite(b) && isfinite(c))) {
        return NAN;
    }
    double min_ab = a < b ? a : b;
    double min_v = min_ab < c ? min_ab : c;
    double max_ab = a > b ? a : b;
    double max_v = max_ab > c ? max_ab : c;
    return (a + b + c) - min_v - max_v;
}

static __device__ inline double esam_ring_get(const double* buf, int center, int off, int size) {
    int idx = center + size - (off % size);
    if (idx >= size) {
        idx -= size;
    }
    return buf[idx];
}

extern "C" __global__ void ehlers_smoothed_adaptive_momentum_batch_f64(
    const double* data,
    int len,
    const double* alphas,
    const double* cutoffs,
    int rows,
    double* out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double alpha = alphas[row];
    double cutoff = cutoffs[row];
    if (!(isfinite(alpha) && isfinite(cutoff)) || alpha < 0.0 || cutoff <= 0.0) {
        return;
    }

    const int src_ring_size = 76;
    const int smooth_ring_size = 3;
    const int c_ring_size = 7;
    const int dp_ring_size = 5;
    const double nan = NAN;
    const double pi = 3.14159265358979323846;

    double coef_c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    double one_minus_alpha = 1.0 - alpha;
    double a1 = exp(-pi / cutoff);
    double b1 = 2.0 * a1 * cos(1.738 * pi / cutoff);
    double c1 = a1 * a1;
    double coef2 = b1 + c1;
    double coef3 = -(c1 + b1 * c1);
    double coef4 = c1 * c1;
    double coef1 = 1.0 - coef2 - coef3 - coef4;

    double src_ring[src_ring_size];
    double smooth_ring[smooth_ring_size];
    double c_ring[c_ring_size];
    double dp_ring[dp_ring_size];
    double f3_hist[3];

    for (int i = 0; i < src_ring_size; ++i) {
        src_ring[i] = nan;
    }
    for (int i = 0; i < smooth_ring_size; ++i) {
        smooth_ring[i] = nan;
    }
    for (int i = 0; i < c_ring_size; ++i) {
        c_ring[i] = nan;
    }
    for (int i = 0; i < dp_ring_size; ++i) {
        dp_ring[i] = nan;
    }
    for (int i = 0; i < 3; ++i) {
        f3_hist[i] = nan;
    }

    int src_idx = 0;
    int smooth_idx = 0;
    int c_idx = 0;
    int dp_idx = 0;
    int valid_count = 0;

    double prev_ip = nan;
    double prev_p = nan;
    double prev_q1 = nan;
    double prev_i1 = nan;

    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        double source = data[i];
        if (!isfinite(source)) {
            row_out[i] = nan;
            continue;
        }

        src_ring[src_idx] = source;
        double src0 = esam_ring_get(src_ring, src_idx, 0, src_ring_size);
        double src1 = esam_ring_get(src_ring, src_idx, 1, src_ring_size);
        double src2 = esam_ring_get(src_ring, src_idx, 2, src_ring_size);
        double src3 = esam_ring_get(src_ring, src_idx, 3, src_ring_size);

        double smooth =
            (isfinite(src0) && isfinite(src1) && isfinite(src2) && isfinite(src3))
                ? (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0
                : nan;
        smooth_ring[smooth_idx] = smooth;

        double smooth1 = esam_nz(esam_ring_get(smooth_ring, smooth_idx, 1, smooth_ring_size));
        double smooth2 = esam_nz(esam_ring_get(smooth_ring, smooth_idx, 2, smooth_ring_size));
        double c_prev1 = esam_nz(esam_ring_get(c_ring, c_idx, 1, c_ring_size));
        double c_prev2 = esam_nz(esam_ring_get(c_ring, c_idx, 2, c_ring_size));
        double c_main = isfinite(smooth)
                            ? coef_c * (smooth - 2.0 * smooth1 + smooth2)
                                  + 2.0 * one_minus_alpha * c_prev1
                                  - one_minus_alpha * one_minus_alpha * c_prev2
                            : nan;
        double c_fallback =
            (isfinite(src0) && isfinite(src1) && isfinite(src2)) ? (src0 - 2.0 * src1 + src2) / 4.0
                                                                  : nan;
        double c = isfinite(c_main) ? c_main : c_fallback;
        c_ring[c_idx] = c;

        double q1 = nan;
        if (isfinite(c)) {
            double factor = 0.5 + 0.08 * esam_nz(prev_ip);
            q1 = (0.0962 * c + 0.5769 * esam_nz(esam_ring_get(c_ring, c_idx, 2, c_ring_size))
                  - 0.5769 * esam_nz(esam_ring_get(c_ring, c_idx, 4, c_ring_size))
                  - 0.0962 * esam_nz(esam_ring_get(c_ring, c_idx, 6, c_ring_size)))
                 * factor;
        }
        double i1 = esam_nz(esam_ring_get(c_ring, c_idx, 3, c_ring_size));

        double dp_raw = 0.0;
        if (isfinite(q1) && isfinite(prev_q1) && q1 != 0.0 && prev_q1 != 0.0) {
            double prev_i1_nz = esam_nz(prev_i1);
            double prev_q1_nz = esam_nz(prev_q1);
            double numer = (i1 / q1) - (prev_i1_nz / prev_q1_nz);
            double denom = 1.0 + i1 * prev_i1_nz / (q1 * prev_q1_nz);
            dp_raw = numer / denom;
        }
        double dp = dp_raw < 0.1 ? 0.1 : (dp_raw > 1.1 ? 1.1 : dp_raw);
        dp_ring[dp_idx] = dp;

        double md_inner = esam_median3(
            esam_ring_get(dp_ring, dp_idx, 2, dp_ring_size),
            esam_ring_get(dp_ring, dp_idx, 3, dp_ring_size),
            esam_ring_get(dp_ring, dp_idx, 4, dp_ring_size));
        double md = esam_median3(dp, esam_ring_get(dp_ring, dp_idx, 1, dp_ring_size), md_inner);
        double dc = md == 0.0 ? 15.0 : ((2.0 * pi / md) + 0.5);
        double ip = 0.33 * dc + 0.67 * esam_nz(prev_ip);
        double p = 0.15 * ip + 0.85 * esam_nz(prev_p);

        double pr = isfinite(p) ? round(fabs(p - 1.0)) : nan;
        double v1 = 0.0;
        if (isfinite(pr)) {
            int lookback = static_cast<int>(pr);
            if (lookback >= 1 && lookback <= 75) {
                double past = esam_ring_get(src_ring, src_idx, lookback, src_ring_size);
                v1 = isfinite(past) ? (source - past) : nan;
            } else {
                v1 = 0.0;
            }
        } else {
            v1 = 0.0;
        }

        double raw_f3 =
            isfinite(v1)
                ? coef1 * v1 + coef2 * esam_nz(f3_hist[0]) + coef3 * esam_nz(f3_hist[1])
                      + coef4 * esam_nz(f3_hist[2])
                : nan;
        double f3 = isfinite(raw_f3) ? raw_f3 : v1;

        prev_q1 = q1;
        prev_i1 = i1;
        prev_ip = ip;
        prev_p = p;
        f3_hist[2] = f3_hist[1];
        f3_hist[1] = f3_hist[0];
        f3_hist[0] = f3;

        valid_count += 1;
        src_idx = (src_idx + 1) % src_ring_size;
        smooth_idx = (smooth_idx + 1) % smooth_ring_size;
        c_idx = (c_idx + 1) % c_ring_size;
        dp_idx = (dp_idx + 1) % dp_ring_size;

        row_out[i] = valid_count <= 75 ? nan : f3;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — ehlers_smoothed_adaptive_momentum
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ehlers_smoothed_adaptive_momentum.rs:530
 *   (`compute_esam_into`) driving `EsamCore::update` (:390). Walks every bar
 *   from 0; a non-finite source returns NaN WITHOUT touching the state, so
 *   `first_valid` is not read.
 *
 * Column: the batch calls `expect_value_output` and returns `out.values`
 *   (cpu_batch.rs:10246, :10289) — the single f3 series.
 *
 * PERIOD-INVARIANT: `compute_ehlers_smoothed_adaptive_momentum_batch`
 *   (cpu_batch.rs:10267-10269) reads `source`, `alpha` and `cutoff` and NEVER
 *   `period`. Pinned at the CPU defaults alpha = 0.07, cutoff = 8.0 (:32-33).
 *
 * Source: hl2 (:31) — F64InputKind::Hl2Slice.
 *
 * Shape: ONE THREAD PER COLUMN walking bars ascending. Four rings (76/3/7/5)
 *   plus four scalars carry across every bar and the dominant-cycle estimate
 *   feeds back into its own smoothing, so there is no bar-parallel form that
 *   preserves the accumulation order.
 *
 * Every ring starts at NaN (:372-378), and `nz()` (:252) maps a non-finite
 *   read to 0.0 at the point of use. Those are two different things and both
 *   are reproduced here: `c_main` is gated on `smooth` being finite while its
 *   history terms are nz-ed, so a zero-initialised ring would take a branch
 *   the CPU does not take.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* MAX_ADAPTIVE_LOOKBACK, ehlers_smoothed_adaptive_momentum.rs:34. */
#define NEO_ESAM_MAX_LOOKBACK 75
#define NEO_ESAM_SRC_RING (NEO_ESAM_MAX_LOOKBACK + 1)

__device__ __forceinline__ double neo_esam_nz(double v)
{
    return isfinite(v) ? v : 0.0;
}

/* median3 (:261): NaN in any slot -> NaN out, otherwise sum minus min minus
 * max. fmin/fmax rather than a comparison chain, so a NaN cannot survive the
 * reduction and be subtracted from a finite sum. */
__device__ __forceinline__ double neo_esam_median3(double a, double b, double c)
{
    if (!(isfinite(a) && isfinite(b) && isfinite(c))) return NEO_F64_NAN;
    return (a + b + c) - fmin(a, fmin(b, c)) - fmax(a, fmax(b, c));
}

/* ring_get (:340): the value `off` slots back from `center`, modulo N. */
__device__ __forceinline__ double neo_esam_ring_get(const double* buf, int n, int center, int off)
{
    int idx = center + n - (off % n);
    if (idx >= n) idx -= n;
    return buf[idx];
}

extern "C" __global__
void ehlers_smoothed_adaptive_momentum_neo_batch_f64(const double* __restrict__ data,
                                                     int n,
                                                     const int* __restrict__ periods,
                                                     int n_combos,
                                                     int first_valid,
                                                     double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;
    (void)first_valid;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    /* resolve_params (:290): alpha 0.07, cutoff 8.0. */
    const double alpha           = 0.07;
    const double cutoff          = 8.0;
    const double one_minus_alpha = 1.0 - alpha;
    const double coef_c          = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    const double coef_prev1      = 2.0 * one_minus_alpha;
    const double coef_prev2      = one_minus_alpha * one_minus_alpha;

    const double NEO_PI = 3.14159265358979323846;
    const double a1 = exp(-NEO_PI / cutoff);
    const double b1 = 2.0 * a1 * cos(1.738 * NEO_PI / cutoff);
    const double c1 = a1 * a1;
    const double coef2 = b1 + c1;
    const double coef3 = -(c1 + b1 * c1);
    const double coef4 = c1 * c1;
    const double coef1 = 1.0 - coef2 - coef3 - coef4;

    double src_ring[NEO_ESAM_SRC_RING];
    for (int i = 0; i < NEO_ESAM_SRC_RING; ++i) src_ring[i] = NEO_F64_NAN;
    double smooth_ring[3] = { NEO_F64_NAN, NEO_F64_NAN, NEO_F64_NAN };
    double c_ring[7];
    for (int i = 0; i < 7; ++i) c_ring[i] = NEO_F64_NAN;
    double dp_ring[5];
    for (int i = 0; i < 5; ++i) dp_ring[i] = NEO_F64_NAN;
    double f3_hist[3] = { NEO_F64_NAN, NEO_F64_NAN, NEO_F64_NAN };

    int src_idx = 0, smooth_idx = 0, c_idx = 0, dp_idx = 0;
    double prev_ip = NEO_F64_NAN, prev_p = NEO_F64_NAN;
    double prev_q1 = NEO_F64_NAN, prev_i1 = NEO_F64_NAN;
    long long valid_count = 0;

    for (int bar = 0; bar < n; ++bar) {
        const double source = data[bar];
        if (!isfinite(source)) { o[bar] = NEO_F64_NAN; continue; }

        src_ring[src_idx] = source;
        const double src0 = neo_esam_ring_get(src_ring, NEO_ESAM_SRC_RING, src_idx, 0);
        const double src1 = neo_esam_ring_get(src_ring, NEO_ESAM_SRC_RING, src_idx, 1);
        const double src2 = neo_esam_ring_get(src_ring, NEO_ESAM_SRC_RING, src_idx, 2);
        const double src3 = neo_esam_ring_get(src_ring, NEO_ESAM_SRC_RING, src_idx, 3);

        const double smooth =
            (isfinite(src0) && isfinite(src1) && isfinite(src2) && isfinite(src3))
                ? ((src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0)
                : NEO_F64_NAN;
        smooth_ring[smooth_idx] = smooth;

        const double smooth1 = neo_esam_nz(neo_esam_ring_get(smooth_ring, 3, smooth_idx, 1));
        const double smooth2 = neo_esam_nz(neo_esam_ring_get(smooth_ring, 3, smooth_idx, 2));
        const double c_prev1 = neo_esam_nz(neo_esam_ring_get(c_ring, 7, c_idx, 1));
        const double c_prev2 = neo_esam_nz(neo_esam_ring_get(c_ring, 7, c_idx, 2));

        const double c_main = isfinite(smooth)
            ? (coef_c * (smooth - 2.0 * smooth1 + smooth2)
               + coef_prev1 * c_prev1
               - coef_prev2 * c_prev2)
            : NEO_F64_NAN;
        const double c_fallback =
            (isfinite(src0) && isfinite(src1) && isfinite(src2))
                ? ((src0 - 2.0 * src1 + src2) / 4.0)
                : NEO_F64_NAN;
        const double c = isfinite(c_main) ? c_main : c_fallback;
        c_ring[c_idx] = c;

        double q1 = NEO_F64_NAN;
        if (isfinite(c)) {
            const double factor = 0.5 + 0.08 * neo_esam_nz(prev_ip);
            q1 = (0.0962 * c
                  + 0.5769 * neo_esam_nz(neo_esam_ring_get(c_ring, 7, c_idx, 2))
                  - 0.5769 * neo_esam_nz(neo_esam_ring_get(c_ring, 7, c_idx, 4))
                  - 0.0962 * neo_esam_nz(neo_esam_ring_get(c_ring, 7, c_idx, 6)))
                 * factor;
        }
        const double i1 = neo_esam_nz(neo_esam_ring_get(c_ring, 7, c_idx, 3));

        double dp_raw = 0.0;
        if (isfinite(q1) && isfinite(prev_q1) && q1 != 0.0 && prev_q1 != 0.0) {
            const double pi1 = neo_esam_nz(prev_i1);
            const double pq1 = neo_esam_nz(prev_q1);
            const double numer = (i1 / q1) - (pi1 / pq1);
            const double denom = 1.0 + i1 * pi1 / (q1 * pq1);
            dp_raw = numer / denom;
        }
        /* Explicit if-chain, NOT a clamp helper: the CPU (:453) tests `< 0.1`
         * first, so a NaN dp_raw falls through both tests and is kept. */
        const double dp = (dp_raw < 0.1) ? 0.1 : ((dp_raw > 1.1) ? 1.1 : dp_raw);
        dp_ring[dp_idx] = dp;

        const double md_inner = neo_esam_median3(
            neo_esam_ring_get(dp_ring, 5, dp_idx, 2),
            neo_esam_ring_get(dp_ring, 5, dp_idx, 3),
            neo_esam_ring_get(dp_ring, 5, dp_idx, 4));
        const double md = neo_esam_median3(dp, neo_esam_ring_get(dp_ring, 5, dp_idx, 1), md_inner);
        const double dc = (md == 0.0) ? 15.0 : ((2.0 * NEO_PI / md) + 0.5);
        const double ip = 0.33 * dc + 0.67 * neo_esam_nz(prev_ip);
        const double p  = 0.15 * ip + 0.85 * neo_esam_nz(prev_p);

        const double pr = isfinite(p) ? round(fabs(p - 1.0)) : NEO_F64_NAN;
        double v1 = 0.0;
        if (isfinite(pr)) {
            const long long lookback = (long long)pr;
            if (lookback >= 1 && lookback <= NEO_ESAM_MAX_LOOKBACK) {
                const double past = neo_esam_ring_get(src_ring, NEO_ESAM_SRC_RING,
                                                      src_idx, (int)lookback);
                v1 = isfinite(past) ? (source - past) : NEO_F64_NAN;
            } else {
                v1 = 0.0;
            }
        }

        const double raw_f3 = isfinite(v1)
            ? (coef1 * v1
               + coef2 * neo_esam_nz(f3_hist[0])
               + coef3 * neo_esam_nz(f3_hist[1])
               + coef4 * neo_esam_nz(f3_hist[2]))
            : NEO_F64_NAN;
        const double f3 = isfinite(raw_f3) ? raw_f3 : v1;

        prev_q1 = q1;
        prev_i1 = i1;
        prev_ip = ip;
        prev_p  = p;
        f3_hist[2] = f3_hist[1];
        f3_hist[1] = f3_hist[0];
        f3_hist[0] = f3;

        ++valid_count;
        src_idx    = (src_idx + 1) % NEO_ESAM_SRC_RING;
        smooth_idx = (smooth_idx + 1) % 3;
        c_idx      = (c_idx + 1) % 7;
        dp_idx     = (dp_idx + 1) % 5;

        o[bar] = (valid_count <= NEO_ESAM_MAX_LOOKBACK) ? NEO_F64_NAN : f3;
    }
}
