#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline double ring_get4(const double* buf, int center, int off) {
    int idx = center + 4 - (off % 4);
    if (idx >= 4) {
        idx -= 4;
    }
    return buf[idx];
}

__device__ inline double ring_get3(const double* buf, int center, int off) {
    int idx = center + 3 - (off % 3);
    if (idx >= 3) {
        idx -= 3;
    }
    return buf[idx];
}
}

extern "C" __global__ void ehlers_simple_cycle_indicator_batch_f64(
    const double* __restrict__ data,
    int len,
    const double* __restrict__ alphas,
    int n_combos,
    double* __restrict__ out_cycle,
    double* __restrict__ out_trigger
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    double alpha = alphas[combo_idx];
    double* cycle_row = out_cycle + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* trigger_row = out_trigger + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        cycle_row[i] = CUDART_NAN;
        trigger_row[i] = CUDART_NAN;
    }

    if (!isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
        return;
    }

    double coef_cycle = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    double one_minus_alpha = 1.0 - alpha;
    double coef_prev1 = 2.0 * one_minus_alpha;
    double coef_prev2 = one_minus_alpha * one_minus_alpha;

    double src_ring[4] = {CUDART_NAN, CUDART_NAN, CUDART_NAN, CUDART_NAN};
    double smooth_ring[3] = {CUDART_NAN, CUDART_NAN, CUDART_NAN};
    double cycle_hist[2] = {CUDART_NAN, CUDART_NAN};
    int src_idx = 0;
    int smooth_idx = 0;
    int valid_count = 0;

    for (int i = 0; i < len; ++i) {
        double source = data[i];
        if (!isfinite(source)) {
            continue;
        }

        src_ring[src_idx] = source;
        double src0 = ring_get4(src_ring, src_idx, 0);
        double src1 = ring_get4(src_ring, src_idx, 1);
        double src2 = ring_get4(src_ring, src_idx, 2);
        double src3 = ring_get4(src_ring, src_idx, 3);

        double smooth = CUDART_NAN;
        if (isfinite(src0) && isfinite(src1) && isfinite(src2) && isfinite(src3)) {
            smooth = (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0;
        }
        smooth_ring[smooth_idx] = smooth;

        double smooth1 = ring_get3(smooth_ring, smooth_idx, 1);
        double smooth2 = ring_get3(smooth_ring, smooth_idx, 2);
        double prev_cycle1 = cycle_hist[0];
        double prev_cycle2 = cycle_hist[1];

        double cycle_main = CUDART_NAN;
        if (isfinite(smooth) && isfinite(smooth1) && isfinite(smooth2) && isfinite(prev_cycle1) &&
            isfinite(prev_cycle2)) {
            cycle_main = coef_cycle * (smooth - 2.0 * smooth1 + smooth2) +
                coef_prev1 * prev_cycle1 - coef_prev2 * prev_cycle2;
        }

        double cycle_fallback = CUDART_NAN;
        if (isfinite(src0) && isfinite(src1) && isfinite(src2)) {
            cycle_fallback = (src0 - 2.0 * src1 + src2) / 4.0;
        }

        double cycle = valid_count < 7 ? cycle_fallback : cycle_main;
        double trigger = prev_cycle1;

        cycle_hist[1] = cycle_hist[0];
        cycle_hist[0] = cycle;
        valid_count += 1;
        src_idx = (src_idx + 1) % 4;
        smooth_idx = (smooth_idx + 1) % 3;

        cycle_row[i] = cycle;
        trigger_row[i] = trigger;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — ehlers_simple_cycle_indicator
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ehlers_simple_cycle_indicator.rs:355
 *   (`compute_esci_into`) driving `EsciCore::update` (:289). The driver walks
 *   `0..data.len()` and `update` returns (NaN, NaN) on a non-finite source
 *   WITHOUT advancing any index or ring — so `first_valid` is not read, and a
 *   hole leaves the state exactly where it was rather than resetting it.
 *
 * Column: output_id "value" / "cycle" -> `out.cycle` (cpu_batch.rs:10196).
 *
 * PERIOD-INVARIANT: `compute_ehlers_simple_cycle_indicator_batch`
 *   (cpu_batch.rs:10179-10180) reads `source` and `alpha` and NEVER `period`.
 *   `alpha` is pinned at the CPU default 0.07 (:28).
 *
 * Source: hl2 (:27) — served by F64InputKind::Hl2Slice, the same declaration
 *   `kurtosis` and `alligator` already use. Feeding close would compute a
 *   different indicator and pass every shape check.
 *
 * Rings start at NaN, not 0 (:278-281). That is load-bearing: the first six
 *   bars take the `cycle_fallback` branch precisely because `smooth1`/`smooth2`
 *   /`prev_cycle` are still NaN, and a zero-initialised ring would silently
 *   take the main branch and emit a number.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void ehlers_simple_cycle_indicator_neo_batch_f64(const double* __restrict__ data,
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

    /* resolve_params (:236): alpha = 0.07 (DEFAULT_ALPHA, :28). */
    const double alpha           = 0.07;
    const double one_minus_alpha = 1.0 - alpha;
    const double coef_cycle      = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    const double coef_prev1      = 2.0 * one_minus_alpha;
    const double coef_prev2      = one_minus_alpha * one_minus_alpha;

    double src_ring[4]    = { NEO_F64_NAN, NEO_F64_NAN, NEO_F64_NAN, NEO_F64_NAN };
    double smooth_ring[3] = { NEO_F64_NAN, NEO_F64_NAN, NEO_F64_NAN };
    double cycle_hist[2]  = { NEO_F64_NAN, NEO_F64_NAN };
    int src_idx = 0, smooth_idx = 0;
    long long valid_count = 0;

    for (int i = 0; i < n; ++i) {
        const double source = data[i];
        if (!isfinite(source)) { o[i] = NEO_F64_NAN; continue; }

        src_ring[src_idx] = source;
        const double src0 = source;
        const double src1 = src_ring[(src_idx + 3) & 3];
        const double src2 = src_ring[(src_idx + 2) & 3];
        const double src3 = src_ring[(src_idx + 1) & 3];

        const double smooth =
            (isfinite(src0) && isfinite(src1) && isfinite(src2) && isfinite(src3))
                ? ((src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0)
                : NEO_F64_NAN;
        smooth_ring[smooth_idx] = smooth;

        const double smooth1 = smooth_ring[(smooth_idx == 0) ? 2 : (smooth_idx - 1)];
        const double smooth2 = smooth_ring[(smooth_idx >= 2) ? (smooth_idx - 2)
                                                             : (smooth_idx + 1)];
        const double prev_cycle1 = cycle_hist[0];
        const double prev_cycle2 = cycle_hist[1];

        const double cycle_main =
            (isfinite(smooth) && isfinite(smooth1) && isfinite(smooth2)
             && isfinite(prev_cycle1) && isfinite(prev_cycle2))
                ? (coef_cycle * (smooth - 2.0 * smooth1 + smooth2)
                   + coef_prev1 * prev_cycle1
                   - coef_prev2 * prev_cycle2)
                : NEO_F64_NAN;

        const double cycle_fallback =
            (isfinite(src0) && isfinite(src1) && isfinite(src2))
                ? ((src0 - 2.0 * src1 + src2) / 4.0)
                : NEO_F64_NAN;

        const double cycle = (valid_count < 7) ? cycle_fallback : cycle_main;

        cycle_hist[1] = cycle_hist[0];
        cycle_hist[0] = cycle;
        ++valid_count;
        src_idx    = (src_idx + 1) & 3;
        smooth_idx = (smooth_idx == 2) ? 0 : (smooth_idx + 1);

        o[i] = cycle;
    }
}
