#include <cmath>
#include <cstddef>

extern "C" __global__ void emd_trend_batch_f64(
    const double* __restrict__ src,
    int len,
    const double* __restrict__ mults,
    const double* __restrict__ averages,
    const double* __restrict__ deviations,
    int rows,
    double* __restrict__ out_direction,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const double mult = mults[row];
    const double* row_avg = averages + static_cast<size_t>(row) * static_cast<size_t>(len);
    const double* row_dev = deviations + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction =
        out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);

    double direction = 0.0;
    for (int i = 0; i < len; ++i) {
        const double avg = row_avg[i];
        const double dev = row_dev[i];
        if (isfinite(avg) && isfinite(dev)) {
            row_upper[i] = avg + dev * mult;
            row_lower[i] = avg - dev * mult;
        } else {
            row_upper[i] = NAN;
            row_lower[i] = NAN;
        }

        if (i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(row_upper[i]) &&
            isfinite(row_upper[i - 1]) && src[i] > row_upper[i] &&
            src[i - 1] <= row_upper[i - 1]) {
            direction = 1.0;
        } else if (
            i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(row_lower[i]) &&
            isfinite(row_lower[i - 1]) && src[i] < row_lower[i] &&
            src[i - 1] >= row_lower[i - 1]
        ) {
            direction = -1.0;
        }
        row_direction[i] = direction;
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — emd_trend
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/emd_trend.rs `compute_from_source_into`, plus the
 * exact scalar SMA and EMA authorities it calls. Canonical production fixes
 * `source=close`, `avg_type=SMA`, and carries exact `(length, mult)` tuples.
 *
 * The old `emd_trend_batch_f64` above is a compatibility ABI over host-built
 * average/deviation matrices. It is deliberately not a production route. The
 * dynamic entry below borrows resident close, owns the whole price-dependent
 * state, and emits direction/average/upper/lower in one launch.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* DEFAULT_LENGTH, emd_trend.rs. */
#define NEO_EMD_TREND_LENGTH 28

static __device__ __forceinline__
void emd_trend_compensated_add_f64(double* sum, double* correction, double value)
{
    const double adjusted = value - *correction;
    const double next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

// One complete close/SMA EMD Trend row. Output pointers may be null only for
// compatibility primaries; canonical production always supplies all four.
static __device__ __forceinline__
void emd_trend_row_f64(const double* __restrict__ src,
                       int n,
                       int length,
                       double mult,
                       int first_valid,
                       double* __restrict__ sma_ring,
                       double* __restrict__ direction_out,
                       double* __restrict__ average_out,
                       double* __restrict__ upper_out,
                       double* __restrict__ lower_out)
{
    for (int i = 0; i < n; ++i) {
        if (direction_out != nullptr) direction_out[i] = 0.0;
        if (average_out != nullptr) average_out[i] = NEO_F64_NAN;
        if (upper_out != nullptr) upper_out[i] = NEO_F64_NAN;
        if (lower_out != nullptr) lower_out[i] = NEO_F64_NAN;
    }
    if (length <= 0 || first_valid < 0 || first_valid >= n ||
        length > n - first_valid || sma_ring == nullptr) {
        return;
    }

    double sma_sum = 0.0;
    double sma_correction = 0.0;
    for (int offset = 0; offset < length; ++offset) {
        const double value = src[first_valid + offset];
        sma_ring[offset] = value;
        emd_trend_compensated_add_f64(&sma_sum, &sma_correction, value);
    }

    const double inv_length = 1.0 / (double)length;
    const double alpha = 2.0 / ((double)length + 1.0);
    const double beta = 1.0 - alpha;
    const int average_first = first_valid + length - 1;

    bool deviation_started = false;
    int deviation_warmup_end = 0;
    int deviation_valid_count = 0;
    double deviation_mean = NEO_F64_NAN;
    double deviation_previous = NEO_F64_NAN;
    double direction = 0.0;
    double previous_upper = NEO_F64_NAN;
    double previous_lower = NEO_F64_NAN;

    for (int i = average_first; i < n; ++i) {
        if (length > 1 && i > average_first) {
            const int slot = (i - first_valid) % length;
            emd_trend_compensated_add_f64(
                &sma_sum, &sma_correction, -sma_ring[slot]);
            sma_ring[slot] = src[i];
            emd_trend_compensated_add_f64(&sma_sum, &sma_correction, src[i]);
        }

        // The scalar SMA authority copies length-one inputs directly instead
        // of advancing its compensated recurrence.
        const double average = length == 1 ? src[i] : sma_sum * inv_length;
        if (average_out != nullptr) average_out[i] = average;
        const double abs_deviation = isfinite(average) && isfinite(src[i])
            ? fabs(src[i] - average)
            : NEO_F64_NAN;

        double deviation = NEO_F64_NAN;
        if (!deviation_started) {
            if (!isnan(abs_deviation)) {
                deviation_started = true;
                deviation_valid_count = 1;
                deviation_mean = abs_deviation;
                deviation_previous = deviation_mean;
                deviation_warmup_end = length < n - i ? i + length : n;
                deviation = deviation_mean;
            }
        } else if (i < deviation_warmup_end) {
            if (isfinite(abs_deviation)) {
                ++deviation_valid_count;
                const double valid_count = (double)deviation_valid_count;
                deviation_mean =
                    ((valid_count - 1.0) * deviation_mean + abs_deviation) / valid_count;
                deviation_previous = deviation_mean;
            }
            deviation = deviation_mean;
        } else {
            if (isfinite(abs_deviation)) {
                deviation_previous = fma(beta, deviation_previous, alpha * abs_deviation);
            }
            deviation = deviation_previous;
        }

        double upper = NEO_F64_NAN;
        double lower = NEO_F64_NAN;
        if (isfinite(average) && isfinite(deviation)) {
            upper = average + deviation * mult;
            lower = average - deviation * mult;
        }
        if (upper_out != nullptr) upper_out[i] = upper;
        if (lower_out != nullptr) lower_out[i] = lower;

        if (i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(upper) &&
            isfinite(previous_upper) && src[i] > upper && src[i - 1] <= previous_upper) {
            direction = 1.0;
        } else if (i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(lower) &&
                   isfinite(previous_lower) && src[i] < lower &&
                   src[i - 1] >= previous_lower) {
            direction = -1.0;
        }
        if (direction_out != nullptr) direction_out[i] = direction;
        previous_upper = upper;
        previous_lower = lower;
    }
}

// Canonical production ABI. One shared-session launch emits every registered
// output for every exact `(length, mult)` tuple from resident close.
extern "C" __global__
void emd_trend_outputs_f64(const double* __restrict__ prices,
                           int n,
                           const int* __restrict__ lengths,
                           const double* __restrict__ mults,
                           int n_combos,
                           int first_valid,
                           int sma_stride,
                           double* __restrict__ sma_rings,
                           double* __restrict__ direction_out,
                           double* __restrict__ average_out,
                           double* __restrict__ upper_out,
                           double* __restrict__ lower_out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0 || sma_stride <= 0) return;
    const size_t base = (size_t)combo * (size_t)n;
    double* ring = sma_rings + (size_t)combo * (size_t)sma_stride;
    emd_trend_row_f64(prices,
                      n,
                      lengths[combo],
                      mults[combo],
                      first_valid,
                      ring,
                      direction_out + base,
                      average_out + base,
                      upper_out + base,
                      lower_out + base);
}

// Preserved generic-primary ABI. It remains fixed at the historical canonical
// default and is compatibility-only; typed production never enters it.
extern "C" __global__
void emd_trend_neo_batch_f64(const double* __restrict__ prices,
                             int n,
                             const int* __restrict__ periods,
                             int n_combos,
                             int first_valid,
                             double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;
    double ring[NEO_EMD_TREND_LENGTH];
    double* __restrict__ average = out + (size_t)combo * (size_t)n;
    emd_trend_row_f64(prices,
                      n,
                      NEO_EMD_TREND_LENGTH,
                      1.0,
                      first_valid,
                      ring,
                      nullptr,
                      average,
                      nullptr,
                      nullptr);
}
