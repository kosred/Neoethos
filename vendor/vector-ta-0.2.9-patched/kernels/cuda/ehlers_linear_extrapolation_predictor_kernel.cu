#include <cmath>
#include <cstddef>

namespace {
constexpr int HISTORY_LENGTH = 10;
constexpr int MAX_BARS_FORWARD = 10;
constexpr double PI_CONST = 3.14159265358979323846;
constexpr double FLOAT_TOL = 1e-12;

constexpr int SIGNAL_MODE_PREDICT_FILTER_CROSSES = 0;
constexpr int SIGNAL_MODE_PREDICT_MIDDLE_CROSSES = 1;
constexpr int SIGNAL_MODE_FILTER_MIDDLE_CROSSES = 2;

__device__ inline double signum_with_tol(double value) {
    if (value > FLOAT_TOL) {
        return 1.0;
    }
    if (value < -FLOAT_TOL) {
        return -1.0;
    }
    return 0.0;
}

__device__ inline double signal_state(int signal_mode, double prediction, double filter) {
    double lhs = prediction;
    double rhs = filter;
    if (signal_mode == SIGNAL_MODE_PREDICT_MIDDLE_CROSSES) {
        rhs = 0.0;
    } else if (signal_mode == SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        lhs = filter;
        rhs = 0.0;
    }
    return signum_with_tol(lhs - rhs);
}

__device__ inline double go_long(
    int signal_mode,
    double prev_prediction,
    double prev_filter,
    double prediction,
    double filter
) {
    double prev_lhs = prev_prediction;
    double prev_rhs = prev_filter;
    double lhs = prediction;
    double rhs = filter;
    if (signal_mode == SIGNAL_MODE_PREDICT_MIDDLE_CROSSES) {
        prev_rhs = 0.0;
        rhs = 0.0;
    } else if (signal_mode == SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        prev_lhs = prev_filter;
        prev_rhs = 0.0;
        lhs = filter;
        rhs = 0.0;
    }
    return (prev_lhs <= prev_rhs && lhs > rhs) ? 1.0 : 0.0;
}

__device__ inline double go_short(
    int signal_mode,
    double prev_prediction,
    double prev_filter,
    double prediction,
    double filter
) {
    double prev_lhs = prev_prediction;
    double prev_rhs = prev_filter;
    double lhs = prediction;
    double rhs = filter;
    if (signal_mode == SIGNAL_MODE_PREDICT_MIDDLE_CROSSES) {
        prev_rhs = 0.0;
        rhs = 0.0;
    } else if (signal_mode == SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        prev_lhs = prev_filter;
        prev_rhs = 0.0;
        lhs = filter;
        rhs = 0.0;
    }
    return (prev_lhs >= prev_rhs && lhs < rhs) ? 1.0 : 0.0;
}

__device__ inline void write_nan(double* output, int index) {
    if (output != nullptr) {
        output[index] = NAN;
    }
}
} // namespace

// One complete f64 row authority for all preserved and production ABIs.
// `hann_weights == nullptr` is retained only for the public legacy ABIs, which
// historically derive their own device coefficients. NeoEthos production
// always supplies the immutable scalar-CPU coefficient/Hann payload.
static __device__ __forceinline__ void ehlers_linear_extrapolation_predictor_row_f64(
    const double* __restrict__ data,
    int len,
    int high_pass_length,
    int low_pass_length,
    double gain,
    int bars_forward,
    int signal_mode,
    double hp_c1,
    double hp_c2,
    double hp_c3,
    const double* __restrict__ hann_weights,
    double hann_weight_sum,
    double* __restrict__ hp_history,
    int hp_history_capacity,
    double* __restrict__ out_prediction,
    double* __restrict__ out_filter,
    double* __restrict__ out_state,
    double* __restrict__ out_go_long,
    double* __restrict__ out_go_short
) {
    for (int i = 0; i < len; ++i) {
        write_nan(out_prediction, i);
        write_nan(out_filter, i);
        write_nan(out_state, i);
        write_nan(out_go_long, i);
        write_nan(out_go_short, i);
    }
    if (len <= 0 || high_pass_length <= 0 || low_pass_length <= 0 ||
        low_pass_length > hp_history_capacity || !isfinite(gain) || bars_forward < 0 ||
        bars_forward > MAX_BARS_FORWARD ||
        signal_mode < SIGNAL_MODE_PREDICT_FILTER_CROSSES ||
        signal_mode > SIGNAL_MODE_FILTER_MIDDLE_CROSSES ||
        !isfinite(hann_weight_sum) || hann_weight_sum == 0.0) {
        return;
    }

    const double pix2 = hann_weights == nullptr
        ? 2.0 * PI_CONST / static_cast<double>(low_pass_length + 1)
        : 0.0;
    int source_count = 0;
    double prev_source_1 = 0.0;
    double prev_source_2 = 0.0;
    double hp_prev_1 = 0.0;
    double hp_prev_2 = 0.0;
    int hp_count = 0;
    double filter_history[HISTORY_LENGTH];
    int filter_count = 0;
    double prev_prediction = 0.0;
    double prev_filter = 0.0;
    bool has_prev_signal = false;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            source_count = 0;
            prev_source_1 = 0.0;
            prev_source_2 = 0.0;
            hp_prev_1 = 0.0;
            hp_prev_2 = 0.0;
            hp_count = 0;
            filter_count = 0;
            prev_prediction = 0.0;
            prev_filter = 0.0;
            has_prev_signal = false;
            continue;
        }

        source_count += 1;
        const double hp = source_count <= 4
            ? 0.0
            : hp_c1 * (value - 2.0 * prev_source_1 + prev_source_2) +
                  hp_c2 * hp_prev_1 + hp_c3 * hp_prev_2;
        prev_source_2 = prev_source_1;
        prev_source_1 = value;
        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp;

        if (hp_count < low_pass_length) {
            for (int j = hp_count; j > 0; --j) {
                hp_history[j] = hp_history[j - 1];
            }
            hp_history[0] = hp;
            hp_count += 1;
        } else {
            for (int j = low_pass_length - 1; j > 0; --j) {
                hp_history[j] = hp_history[j - 1];
            }
            hp_history[0] = hp;
        }
        if (source_count < 4 + low_pass_length - 1 || hp_count < low_pass_length) {
            continue;
        }

        double filter = 0.0;
        for (int count = 1; count <= low_pass_length; ++count) {
            const double coefficient = hann_weights != nullptr
                ? hann_weights[count - 1]
                : 1.0 - cos(static_cast<double>(count) * pix2);
            filter += coefficient * hp_history[count - 1];
        }
        filter /= hann_weight_sum;

        if (filter_count < HISTORY_LENGTH) {
            filter_history[filter_count] = filter;
            filter_count += 1;
        } else {
            for (int j = 0; j < HISTORY_LENGTH - 1; ++j) {
                filter_history[j] = filter_history[j + 1];
            }
            filter_history[HISTORY_LENGTH - 1] = filter;
        }
        if (filter_count < HISTORY_LENGTH) {
            continue;
        }

        const double current = filter_history[HISTORY_LENGTH - 1];
        const double previous = filter_history[HISTORY_LENGTH - 2];
        const double prediction = bars_forward == 0
            ? current * gain
            : (current + static_cast<double>(bars_forward) * (current - previous)) * gain;
        const double state = signal_state(signal_mode, prediction, current);
        const double go_long_value = has_prev_signal
            ? go_long(signal_mode, prev_prediction, prev_filter, prediction, current)
            : 0.0;
        const double go_short_value = has_prev_signal
            ? go_short(signal_mode, prev_prediction, prev_filter, prediction, current)
            : 0.0;
        prev_prediction = prediction;
        prev_filter = current;
        has_prev_signal = true;

        if (out_prediction != nullptr) {
            out_prediction[i] = prediction;
        }
        if (out_filter != nullptr) {
            out_filter[i] = current;
        }
        if (out_state != nullptr) {
            out_state[i] = state;
        }
        if (out_go_long != nullptr) {
            out_go_long[i] = go_long_value;
        }
        if (out_go_short != nullptr) {
            out_go_short[i] = go_short_value;
        }
    }
}

// Preserved standalone public full-output ABI. Its device-side coefficient
// derivation is compatibility-only; NeoEthos production never calls it.
extern "C" __global__ void ehlers_linear_extrapolation_predictor_batch_f64(
    const double* data,
    int len,
    const int* high_pass_lengths,
    const int* low_pass_lengths,
    const double* gains,
    const int* bars_forwards,
    const int* signal_modes,
    int rows,
    int max_low_pass_length,
    double* out_prediction,
    double* out_filter,
    double* out_state,
    double* out_go_long,
    double* out_go_short,
    double* hp_history
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    const int high_pass_length = high_pass_lengths[row];
    const int low_pass_length = low_pass_lengths[row];
    const double angle = 1.414 * PI_CONST / static_cast<double>(high_pass_length);
    const double a1 = exp(-angle);
    const double hp_c2 = 2.0 * a1 * cos(angle);
    const double hp_c3 = -a1 * a1;
    const double hp_c1 = (1.0 + hp_c2 - hp_c3) * 0.25;
    const double pix2 = 2.0 * PI_CONST / static_cast<double>(low_pass_length + 1);
    double hann_weight_sum = 0.0;
    for (int count = 1; count <= low_pass_length; ++count) {
        hann_weight_sum += 1.0 - cos(static_cast<double>(count) * pix2);
    }
    const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(len);
    ehlers_linear_extrapolation_predictor_row_f64(
        data,
        len,
        high_pass_length,
        low_pass_length,
        gains[row],
        bars_forwards[row],
        signal_modes[row],
        hp_c1,
        hp_c2,
        hp_c3,
        nullptr,
        hann_weight_sum,
        hp_history + static_cast<size_t>(row) * static_cast<size_t>(max_low_pass_length),
        max_low_pass_length,
        out_prediction + offset,
        out_filter + offset,
        out_state + offset,
        out_go_long + offset,
        out_go_short + offset
    );
}

// NeoEthos production ABI. The resident wrapper supplies only parameter-owned
// exact CPU bits; prices are borrowed from the frame and all five outputs are
// produced by one sequential thread per canonical tuple.
extern "C" __global__ void ehlers_linear_extrapolation_predictor_outputs_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ parameter_rows,
    const double* __restrict__ coefficient_rows,
    const int* __restrict__ hann_offsets,
    const double* __restrict__ hann_weights,
    int rows,
    int max_low_pass_length,
    double* __restrict__ hp_history,
    double* __restrict__ out_prediction,
    double* __restrict__ out_filter,
    double* __restrict__ out_state,
    double* __restrict__ out_go_long,
    double* __restrict__ out_go_short
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }
    const int parameter_base = row * 4;
    const int coefficient_base = row * 5;
    const int high_pass_length = parameter_rows[parameter_base];
    const int low_pass_length = parameter_rows[parameter_base + 1];
    const int bars_forward = parameter_rows[parameter_base + 2];
    const int signal_mode = parameter_rows[parameter_base + 3];
    const size_t output_offset = static_cast<size_t>(row) * static_cast<size_t>(len);
    ehlers_linear_extrapolation_predictor_row_f64(
        data,
        len,
        high_pass_length,
        low_pass_length,
        coefficient_rows[coefficient_base],
        bars_forward,
        signal_mode,
        coefficient_rows[coefficient_base + 1],
        coefficient_rows[coefficient_base + 2],
        coefficient_rows[coefficient_base + 3],
        hann_weights + hann_offsets[row],
        coefficient_rows[coefficient_base + 4],
        hp_history + static_cast<size_t>(row) * static_cast<size_t>(max_low_pass_length),
        max_low_pass_length,
        out_prediction + output_offset,
        out_filter + output_offset,
        out_state + output_offset,
        out_go_long + output_offset,
        out_go_short + output_offset
    );
}

// Preserved generic primary ABI. It remains fixed at the historical defaults
// and is compatibility-only; the typed production planner bypasses it even
// for the canonical `prediction` receipt.
#define NEO_ELEP_HIGH_PASS_LENGTH 125
#define NEO_ELEP_LOW_PASS_LENGTH 12
#define NEO_ELEP_GAIN 0.7
#define NEO_ELEP_BARS_FORWARD 5
#define NEO_ELEP_SIGNAL_MODE SIGNAL_MODE_PREDICT_FILTER_CROSSES
#define NEO_ELEP_MAX_LOW_PASS_LENGTH 512

extern "C" __global__ void ehlers_linear_extrapolation_predictor_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;
    const double angle = 1.414 * PI_CONST / static_cast<double>(NEO_ELEP_HIGH_PASS_LENGTH);
    const double a1 = exp(-angle);
    const double hp_c2 = 2.0 * a1 * cos(angle);
    const double hp_c3 = -a1 * a1;
    const double hp_c1 = (1.0 + hp_c2 - hp_c3) * 0.25;
    const double pix2 = 2.0 * PI_CONST / static_cast<double>(NEO_ELEP_LOW_PASS_LENGTH + 1);
    double hann_weight_sum = 0.0;
    for (int count = 1; count <= NEO_ELEP_LOW_PASS_LENGTH; ++count) {
        hann_weight_sum += 1.0 - cos(static_cast<double>(count) * pix2);
    }
    double hp_history[NEO_ELEP_MAX_LOW_PASS_LENGTH];
    const size_t output_offset = static_cast<size_t>(row) * static_cast<size_t>(n);
    ehlers_linear_extrapolation_predictor_row_f64(
        data,
        n,
        NEO_ELEP_HIGH_PASS_LENGTH,
        NEO_ELEP_LOW_PASS_LENGTH,
        NEO_ELEP_GAIN,
        NEO_ELEP_BARS_FORWARD,
        NEO_ELEP_SIGNAL_MODE,
        hp_c1,
        hp_c2,
        hp_c3,
        nullptr,
        hann_weight_sum,
        hp_history,
        NEO_ELEP_MAX_LOW_PASS_LENGTH,
        out + output_offset,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
}
