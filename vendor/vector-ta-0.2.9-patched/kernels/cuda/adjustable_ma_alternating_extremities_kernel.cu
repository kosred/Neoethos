#include <cmath>
#include <cstddef>

namespace {

constexpr double TWO_PI = 6.28318530717958647692528676655900577;
constexpr double WEIGHT_SUM_EPS = 1e-12;

__device__ inline bool valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline bool pine_cross(double prev_a, double prev_b, double curr_a, double curr_b) {
    if (!(isfinite(prev_a) && isfinite(prev_b) && isfinite(curr_a) && isfinite(curr_b))) {
        return false;
    }
    return (curr_a > curr_b && prev_a <= prev_b) || (curr_a < curr_b && prev_a >= prev_b);
}

__device__ inline double raw_weight(int index, int length, double alpha, double beta) {
    const double denom = static_cast<double>(length - 1);
    const double x = static_cast<double>(index) / denom;
    return sin(TWO_PI * pow(x, alpha)) * (1.0 - pow(x, beta));
}

}

extern "C" __global__ void adjustable_ma_alternating_extremities_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ mults,
    const double* __restrict__ alphas,
    const double* __restrict__ betas,
    int rows,
    double* __restrict__ out_ma,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower,
    double* __restrict__ out_extremity,
    double* __restrict__ out_state,
    double* __restrict__ out_changed,
    double* __restrict__ out_smoothed_open,
    double* __restrict__ out_smoothed_high,
    double* __restrict__ out_smoothed_low,
    double* __restrict__ out_smoothed_close
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];
    const double alpha = alphas[row];
    const double beta = betas[row];

    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_extremity = out_extremity + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_open =
        out_smoothed_open + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_high =
        out_smoothed_high + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_low =
        out_smoothed_low + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_close =
        out_smoothed_close + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_ma[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_extremity[i] = NAN;
        row_state[i] = NAN;
        row_changed[i] = NAN;
        row_smoothed_open[i] = NAN;
        row_smoothed_high[i] = NAN;
        row_smoothed_low[i] = NAN;
        row_smoothed_close[i] = NAN;
    }

    if (length < 2 || length > len || !isfinite(mult) || mult < 1.0 || !isfinite(alpha) ||
        alpha < 0.0 || !isfinite(beta) || beta < 0.0) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (valid_bar(high[i], low[i], close[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    const int needed = (length * 2) - 1;
    if (len - first < needed) {
        return;
    }

    double weight_sum = 0.0;
    for (int j = 0; j < length; ++j) {
        weight_sum += raw_weight(j, length, alpha, beta);
    }
    if (!isfinite(weight_sum) || fabs(weight_sum) <= WEIGHT_SUM_EPS) {
        return;
    }
    const double inv_weight_sum = 1.0 / weight_sum;

    const int ma_start = first + length - 1;
    for (int i = ma_start; i < len; ++i) {
        double ma_acc = 0.0;
        double high_acc = 0.0;
        double low_acc = 0.0;
        for (int j = 0; j < length; ++j) {
            const double w = raw_weight(j, length, alpha, beta) * inv_weight_sum;
            ma_acc += close[i - j] * w;
            high_acc += high[i - j] * w;
            low_acc += low[i - j] * w;
        }
        row_ma[i] = ma_acc;
        row_smoothed_close[i] = ma_acc;
        row_smoothed_high[i] = high_acc;
        row_smoothed_low[i] = low_acc;
    }

    const int open_start = ma_start + 2;
    for (int i = open_start; i < len; ++i) {
        row_smoothed_open[i] = 0.5 * (row_ma[i - 1] + row_ma[i - 2]);
    }

    const int band_start = first + (length * 2) - 2;
    double rolling = 0.0;
    for (int i = ma_start; i <= band_start; ++i) {
        rolling += fabs(close[i] - row_ma[i]);
    }
    const double first_dev = (rolling / static_cast<double>(length)) * mult;
    row_upper[band_start] = row_ma[band_start] + first_dev;
    row_lower[band_start] = row_ma[band_start] - first_dev;

    for (int i = band_start + 1; i < len; ++i) {
        rolling += fabs(close[i] - row_ma[i]);
        rolling -= fabs(close[i - length] - row_ma[i - length]);
        const double dev = (rolling / static_cast<double>(length)) * mult;
        row_upper[i] = row_ma[i] + dev;
        row_lower[i] = row_ma[i] - dev;
    }

    row_state[band_start] = 0.0;
    row_changed[band_start] = 0.0;
    row_extremity[band_start] = row_lower[band_start];

    for (int i = band_start + 1; i < len; ++i) {
        const double prev_state = row_state[i - 1];
        const bool cross_high = pine_cross(high[i - 1], row_upper[i - 1], high[i], row_upper[i]);
        const bool cross_low = pine_cross(low[i - 1], row_lower[i - 1], low[i], row_lower[i]);
        const double next_state = cross_high ? 1.0 : (cross_low ? 0.0 : prev_state);
        row_state[i] = next_state;
        row_changed[i] = fabs(next_state - prev_state) > 0.0 ? 1.0 : 0.0;
        row_extremity[i] = next_state >= 0.5 ? row_upper[i] : row_lower[i];
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/adjustable_ma_alternating_extremities.rs:318
// (`adjustable_ma_alternating_extremities_with_kernel`), whose `ma` column is
// what `output_id == "value"` resolves to
// (dispatch/cpu_batch.rs:6385-6386). The prepare that sets the warmup is
// :600-650; the weight build is `build_weights` :655; the convolution is
// `weighted_filter_into` :722-737.
//
// SHAPE: one thread per combo, bars ascending. The convolution itself is
// bar-parallel, but the lane launches one thread per column and the bar loop
// is the thread body.
//
// PERIOD-INVARIANT. `compute_adjustable_ma_alternating_extremities_batch`
// (cpu_batch.rs:6417-6430) reads `length`, `mult`, `alpha` and `beta` and
// NEVER `period`, so a sweep of five periods produces five identical CPU
// columns and this kernel emits five identical rows. The four CPU defaults are
// pinned below.
//
// FIRST VALID IS DERIVED HERE, not taken from the caller: the CPU scans for
// the first bar at which high, low AND close are all `is_finite` (:600-602),
// which is stricter than `AllInputsNonNan` (an INFINITE high is skipped by the
// CPU and accepted by that rule). The lane row therefore declares
// `F64FirstValidRule::Ignored`, the same contract
// `garman_klass_volatility_neo_batch_f64` carries.
//
// f64 END TO END: no float literal, no f32-suffixed math function, no
// fast-math intrinsic. `WEIGHT_SUM_EPS` above is 1e-12, which is an f64-sized
// guard on a sum of O(length) terms and is NOT an f32 epsilon copied forward.
// ---------------------------------------------------------------------------

#define NEO_AMAE_LENGTH 50
#define NEO_AMAE_MULT 2.0
#define NEO_AMAE_ALPHA 1.0
#define NEO_AMAE_BETA 0.5
// The weight vector is materialised in a per-thread array because the CPU
// builds it ONCE and then multiplies by it (`build_weights` normalises with
// `*weight *= inv_sum`, one rounding), so recomputing the raw weight inside the
// bar loop would be the same bits but O(length) transcendentals per bar. The
// bound is a property of this COMPILED kernel; `length` is pinned at the CPU
// default 50, so it can never be approached, and it is checked rather than
// assumed.
#define NEO_AMAE_MAX_LENGTH 512

extern "C" __global__ void adjustable_ma_alternating_extremities_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
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
    // Period-invariant, and first-valid derived below. Both are read so the
    // signature stays the lane ABI and neither is silently ignored.
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int length = NEO_AMAE_LENGTH;
    const double mult = NEO_AMAE_MULT;
    const double alpha = NEO_AMAE_ALPHA;
    const double beta = NEO_AMAE_BETA;

    if (length < 2 || length > n || length > NEO_AMAE_MAX_LENGTH) {
        return;
    }
    if (!isfinite(mult) || mult < 1.0 || !isfinite(alpha) || alpha < 0.0 ||
        !isfinite(beta) || beta < 0.0) {
        return;
    }

    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (valid_bar(high[i], low[i], close[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    const int needed = (length * 2) - 1;
    if (n - first < needed) {
        return;
    }

    double w[NEO_AMAE_MAX_LENGTH];
    double weight_sum = 0.0;
    for (int j = 0; j < length; ++j) {
        w[j] = raw_weight(j, length, alpha, beta);
        weight_sum += w[j];
    }
    if (!isfinite(weight_sum) || fabs(weight_sum) <= WEIGHT_SUM_EPS) {
        return;
    }
    const double inv_weight_sum = 1.0 / weight_sum;
    for (int j = 0; j < length; ++j) {
        w[j] *= inv_weight_sum;
    }

    const int ma_start = first + length - 1;
    for (int i = ma_start; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < length; ++j) {
            acc += close[i - j] * w[j];
        }
        row[i] = acc;
    }
}
