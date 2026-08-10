#include <cuda_runtime.h>
#include <math.h>

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 5, round 3
 *                        (ehlers_undersampled_double_moving_average)
 *
 * WRITTEN FROM SCRATCH. Before this file there was no `.cu` for this indicator
 * anywhere under `kernels/cuda`, no wrapper, and no F64_KERNELS row.
 *
 * CPU reference: `compute_eudma_into`
 *   (src/indicators/moving_averages/ehlers_undersampled_double_moving_average.rs:512)
 *   driving `EudmaCore::update` (:412) and `HannFilterState::update` (:328),
 *   entered from `ehlers_undersampled_double_moving_average_with_kernel`
 *   (:566). The two AVX arms (:544, :554) are `#[target_feature]` wrappers
 *   that call `compute_eudma_into` UNCHANGED, so all three kernels are the
 *   same arithmetic and there is one oracle.
 *
 * Column: `ma_batch.rs:1789-1798` returns `out.fast_values`, so "value" is the
 *   FAST filter. The slow filter is a second Hann bank over the SAME sampled
 *   series and does not feed it, so it is deliberately not computed.
 *
 * PERIOD-INVARIANT: the MA-batch route reads `fast_length` (6), `slow_length`
 *   (12) and `sample_length` (5) (ma_batch.rs:1699-1788) and NEVER a parameter
 *   named `period`.
 *
 * Input: ONE price series -> F64InputKind::CloseSlice.
 *
 * FIRST-VALID: `eudma_prepare` :489-491 is
 *   `data.iter().position(|value| !value.is_nan())` over the single series --
 *   F64FirstValidRule::AllInputsNonNan.
 *
 * The warmup is NOT a skip. `compute_eudma_into` :515-517 runs the core over
 *   `data[..first]` and THROWS THE RESULTS AWAY -- the decimation phase and the
 *   filter ring are already loaded by the time the first output bar arrives.
 *   A kernel that started its state machine at `first` would emit a different
 *   series from bar `first` onwards, so this kernel walks from index 0 and
 *   only begins WRITING at `first`.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. `sample_countdown` and
 *   `last_sample` are a two-element state machine carried across bars, and the
 *   Hann ring's head position depends on every bar before it.
 *
 * Roundings, counted against the CPU lines:
 *   :308  1.0 - (2.0 * PI * i as f64 / (length + 1) as f64).cos()  -- weights
 *   :310  norm += weight;                                          -- plain
 *   :397  acc += self.weights[offset] * value;                     -- plain
 *   :404  acc / self.norm
 *   Not one `mul_add` on this path, so not one `fma` here. Writing
 *   `acc = fma(w, v, acc)` would REMOVE a rounding the reference performs.
 *
 * The weights are rebuilt per row rather than tabled, and the accumulation
 *   runs offset 0..len over the ring walked BACKWARDS from the newest slot,
 *   which is the CPU's order (:391-399). Summing forwards would re-associate
 *   `acc`.
 *
 * NaN semantics: the CPU replaces a non-finite ring entry with 0.0 (:393-396)
 *   rather than letting it propagate, and the not-full branch multiplies the
 *   weight by an explicit 0.0 for offsets beyond `count` (:381-388). Both are
 *   reproduced literally, including the `acc += w * 0.0` term -- dropping it
 *   would be arithmetically equal only until `acc` is an infinity.
 *
 * Epsilon: none on this path. The only tolerance-shaped comparison is
 *   `self.norm == 0.0` (:401), which is exact.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#ifndef NEO_F64_PI
#define NEO_F64_PI 3.14159265358979323846
#endif

/* Defaults from ehlers_undersampled_double_moving_average.rs:34-36. The fast
 * length bounds the per-thread ring, and it is a CPU DEFAULT rather than a
 * swept parameter (this indicator is period-invariant), so no caller-supplied
 * number reaches it. */
#define NEO_EUDMA_FAST_LENGTH   6
#define NEO_EUDMA_SAMPLE_LENGTH 5

extern "C" __global__
void ehlers_undersampled_double_moving_average_neo_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods; /* period-invariant -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int len           = NEO_EUDMA_FAST_LENGTH;
    const int sample_length = NEO_EUDMA_SAMPLE_LENGTH;

    /* `validate_params` :463-476 -- 1..=MAX_LENGTH. */
    if (len <= 0 || sample_length <= 0) return;

    const int first = first_valid;
    if (first < 0 || first >= n) return;

    /* `HannFilterState::new` :306-311. */
    double weights[NEO_EUDMA_FAST_LENGTH];
    double norm = 0.0;
    for (int i = 1; i <= len; ++i) {
        const double w = 1.0 - cos(2.0 * NEO_F64_PI * (double)i / (double)(len + 1));
        weights[i - 1] = w;
        norm += w;
    }

    double ring[NEO_EUDMA_FAST_LENGTH];
    for (int k = 0; k < len; ++k) ring[k] = 0.0;
    int head  = 0;
    int count = 0;

    /* `EudmaCore` :383-389. */
    int    sample_countdown = 0;
    double last_sample      = NEO_F64_NAN;

    for (int i = 0; i < n; ++i) {
        const double value = prices[i];

        /* `EudmaCore::update` :413-424. */
        double sampled;
        if (sample_countdown == 0) {
            sample_countdown = sample_length - 1;
            sampled = value;
        } else if (isfinite(last_sample)) {
            sample_countdown -= 1;
            sampled = last_sample;
        } else {
            sample_countdown -= 1;
            sampled = 0.0;
        }
        last_sample = sampled;

        /* `HannFilterState::update` :328-405. */
        const bool full = (count == len);
        ring[head] = sampled;
        head += 1;
        if (head == len) head = 0;
        if (!full) count += 1;

        double acc = 0.0;
        int    idx = (head == 0) ? (len - 1) : (head - 1);
        if (full) {
            for (int offset = 0; offset < len; ++offset) {
                const double current = ring[idx];
                const double v = isfinite(current) ? current : 0.0;
                acc += weights[offset] * v;
                idx = (idx == 0) ? (len - 1) : (idx - 1);
            }
        } else {
            for (int offset = 0; offset < len; ++offset) {
                double v;
                if (offset < count) {
                    const double current = ring[idx];
                    v = isfinite(current) ? current : 0.0;
                } else {
                    v = 0.0;
                }
                acc += weights[offset] * v;
                idx = (idx == 0) ? (len - 1) : (idx - 1);
            }
        }

        const double fast = (norm == 0.0) ? 0.0 : (acc / norm);

        /* :515-521 -- the warmup bars advance the state and are NOT written. */
        if (i >= first) o[i] = fast;
    }
}
