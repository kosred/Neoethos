#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void gopalakrishnan_range_index_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    float* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    float* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length <= 1) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    double log_length = log(static_cast<double>(length));

    for (int t = 0; t < len; ++t) {
        if (t + 1 < length) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t + 1 - length;
        bool valid = true;
        float highest = -CUDART_INF_F;
        float lowest = CUDART_INF_F;

        for (int i = start; i <= t; ++i) {
            float hi = high[i];
            float lo = low[i];
            if (!isfinite(hi) || !isfinite(lo)) {
                valid = false;
                break;
            }
            if (hi > highest) {
                highest = hi;
            }
            if (lo < lowest) {
                lowest = lo;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        double range = static_cast<double>(highest) - static_cast<double>(lowest);
        if (!(range > 0.0) || !isfinite(range)) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        row[t] = static_cast<float>(log(range) / log_length);
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

extern "C" __global__ void gopalakrishnan_range_index_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length <= 1) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN;
        }
        return;
    }

    double log_length = log(static_cast<double>(length));

    for (int t = 0; t < len; ++t) {
        if (t + 1 < length) {
            row[t] = CUDART_NAN;
            continue;
        }

        int start = t + 1 - length;
        bool valid = true;
        double highest = -CUDART_INF;
        double lowest = CUDART_INF;

        for (int i = start; i <= t; ++i) {
            double hi = high[i];
            double lo = low[i];
            if (!isfinite(hi) || !isfinite(lo)) {
                valid = false;
                break;
            }
            if (hi > highest) {
                highest = hi;
            }
            if (lo < lowest) {
                lowest = lo;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN;
            continue;
        }

        double range = static_cast<double>(highest) - static_cast<double>(lowest);
        if (!(range > 0.0) || !isfinite(range)) {
            row[t] = CUDART_NAN;
            continue;
        }

        row[t] = static_cast<double>(log(range) / log_length);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — gopalakrishnan_range_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/gopalakrishnan_range_index.rs:386 (the scalar row
 *   driven by `gopalakrishnan_range_index_prepare`, :662) with `gapo_value`
 *   (:342).
 *
 * Column: `expect_value_output` then the single series (cpu_batch.rs:13840).
 *
 * PERIOD-INVARIANT: `compute_gopalakrishnan_range_index_batch`
 *   (cpu_batch.rs:13849) reads `length` (default 5) and NEVER `period`.
 *
 * Input: (high, low) with NO close — F64InputKind::HighLow. The CPU reference
 *   takes `extract_high_low_input` (cpu_batch.rs:13841) and its first-valid
 *   scan covers high and low only, so an Hlc triple would adopt close's warmup
 *   as well as read a series the indicator never uses.
 *
 * first_valid IS read here: the row starts at `first` and its warmup is
 *   `first + length - 1` (:392). That is the ONE place in this shard where the
 *   caller-supplied index changes the answer rather than being reproduced by a
 *   reset.
 *
 * Shape: ONE THREAD PER COLUMN, two monotonic deques of at most `length`
 *   indices, plus an integer prefix count of valid bars so that a window
 *   containing a hole emits NaN rather than a max over the surviving bars.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* DEFAULT length (cpu_batch.rs:13849). Per-thread rings, so the bound is a
 * property of the compiled kernel. */
#define NEO_GAPO_LENGTH 5

extern "C" __global__
void gopalakrishnan_range_index_neo_batch_f64(const double* __restrict__ high,
                                              const double* __restrict__ low,
                                              int n,
                                              const int* __restrict__ periods,
                                              int n_combos,
                                              int first_valid,
                                              double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int L = NEO_GAPO_LENGTH;
    const int first = (first_valid < 0) ? 0 : first_valid;
    if (first >= n) return;

    const double log_length = log((double)L);
    const int warmup = first + (L - 1);

    int hq[NEO_GAPO_LENGTH]; int h_head = 0, h_len = 0;
    int lq[NEO_GAPO_LENGTH]; int l_head = 0, l_len = 0;

    /* prefix_valid over the last L + 1 bars, integer-exact. */
    int pv_ring[NEO_GAPO_LENGTH + 1];
    for (int i = 0; i <= L; ++i) pv_ring[i] = 0;
    int pv_running = 0;
    pv_ring[first % (NEO_GAPO_LENGTH + 1)] = 0;   /* prefix_valid[first] = 0 */

    for (int i = first; i < n; ++i) {
        const double h = high[i], l = low[i];
        const bool ok = isfinite(h) && isfinite(l);

        if (ok) {
            while (h_len > 0) {
                const int back = hq[(h_head + h_len - 1) % NEO_GAPO_LENGTH];
                if (high[back] > h) break;
                --h_len;
            }
            hq[(h_head + h_len) % NEO_GAPO_LENGTH] = i;
            ++h_len;

            while (l_len > 0) {
                const int back = lq[(l_head + l_len - 1) % NEO_GAPO_LENGTH];
                if (low[back] < l) break;
                --l_len;
            }
            lq[(l_head + l_len) % NEO_GAPO_LENGTH] = i;
            ++l_len;
        }

        pv_running += ok ? 1 : 0;
        pv_ring[(i + 1) % (NEO_GAPO_LENGTH + 1)] = pv_running;   /* prefix_valid[i+1] */

        if (i < warmup) continue;

        const int start = i + 1 - L;
        while (h_len > 0 && hq[h_head] < start) { h_head = (h_head + 1) % NEO_GAPO_LENGTH; --h_len; }
        while (l_len > 0 && lq[l_head] < start) { l_head = (l_head + 1) % NEO_GAPO_LENGTH; --l_len; }

        const int pv_at_start = pv_ring[start % (NEO_GAPO_LENGTH + 1)];
        if (pv_running - pv_at_start != L) continue;

        const double highest = (h_len > 0) ? high[hq[h_head]] : -INFINITY;
        const double lowest  = (l_len > 0) ? low[lq[l_head]]  :  INFINITY;
        const double range   = highest - lowest;
        o[i] = (isfinite(range) && range > 0.0) ? (log(range) / log_length) : NEO_F64_NAN;
    }
}
