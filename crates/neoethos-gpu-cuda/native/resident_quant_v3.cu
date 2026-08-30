#include "resident_quant_v3_abi.cuh"
#include "resident_exact_log_v3.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint64_t kFeatureColumnCountV3 = 63ULL;
constexpr std::uint64_t kTradingSessionsPerYearV3 = 252ULL;
constexpr std::int64_t kUtcDayMillisV3 = 86400000LL;
constexpr std::int64_t kAsianSessionMillisV3 = 28800000LL;
constexpr unsigned char kValidV3 = 0U;
constexpr unsigned char kWarmupV3 = 1U;
constexpr unsigned char kZeroDenominatorV3 = 5U;
constexpr unsigned char kComputeFailureV3 = 8U;
constexpr double kValueFloorV3 = 1.0e-10;
constexpr double kValidityEpsilonV3 = 1.0e-12;

__device__ __forceinline__ double from_bits_v3(unsigned long long bits) {
  return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ double canonical_nan_v3() {
  return from_bits_v3(0x7ff8000000000000ULL);
}

__device__ __forceinline__ double add_rn_v3(double left, double right) {
  return __dadd_rn(left, right);
}

__device__ __forceinline__ double sub_rn_v3(double left, double right) {
  return __dsub_rn(left, right);
}

__device__ __forceinline__ double mul_rn_v3(double left, double right) {
  return __dmul_rn(left, right);
}

__device__ __forceinline__ double div_rn_v3(double left, double right) {
  return __ddiv_rn(left, right);
}

__device__ __forceinline__ double sqrt_rn_v3(double value) {
  return __dsqrt_rn(value);
}

__device__ __forceinline__ double abs_v3(double value) {
  const unsigned long long bits =
      static_cast<unsigned long long>(__double_as_longlong(value));
  return from_bits_v3(bits & 0x7fffffffffffffffULL);
}

__device__ __forceinline__ double min_v3(double left, double right) {
  return left < right ? left : right;
}

__device__ __forceinline__ double max_v3(double left, double right) {
  return left > right ? left : right;
}

__device__ __forceinline__ double clamp_v3(double value, double low,
                                            double high) {
  return min_v3(max_v3(value, low), high);
}

__device__ __forceinline__ double signum_v3(double value) {
  return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

__device__ __forceinline__ bool quant_log_positive_f64_v3(double value,
                                                           double* output) {
  return neoethos_exact_math_v3::exact_log_positive_f64_v3(value, output);
}

__device__ __forceinline__ std::size_t feature_index_v3(std::size_t rows,
                                                         int slot,
                                                         std::size_t row) {
  return static_cast<std::size_t>(slot) * rows + row;
}

__device__ __forceinline__ void set_invalid_v3(
    const NeoResidentQuantLaunchV3& launch, int slot, std::size_t row,
    unsigned char reason) {
  const std::size_t index =
      feature_index_v3(static_cast<std::size_t>(launch.row_count), slot, row);
  launch.feature_values[index] = canonical_nan_v3();
  launch.feature_validity_u8[index] = reason;
}

__device__ __forceinline__ void set_valid_v3(
    const NeoResidentQuantLaunchV3& launch, int slot, std::size_t row,
    double value) {
  if (!isfinite(value)) {
    set_invalid_v3(launch, slot, row, kComputeFailureV3);
    return;
  }
  const std::size_t index =
      feature_index_v3(static_cast<std::size_t>(launch.row_count), slot, row);
  launch.feature_values[index] = value;
  launch.feature_validity_u8[index] = kValidV3;
}

__device__ __forceinline__ double value_v3(
    const NeoResidentQuantLaunchV3& launch, int slot, std::size_t row) {
  return launch.feature_values[
      feature_index_v3(static_cast<std::size_t>(launch.row_count), slot, row)];
}

__device__ void compute_base_and_log_v3(
    const NeoResidentQuantLaunchV3& launch) {
  const std::size_t rows = static_cast<std::size_t>(launch.row_count);
  for (int slot = 0; slot < static_cast<int>(kFeatureColumnCountV3); ++slot) {
    for (std::size_t row = 0; row < rows; ++row) {
      set_invalid_v3(launch, slot, row, kWarmupV3);
    }
  }

  for (std::size_t row = 0; row < rows; ++row) {
    set_valid_v3(launch, 0, row, launch.close[row]);
  }

  const int lags[7] = {1, 2, 3, 5, 8, 13, 21};
  for (int lag_index = 0; lag_index < 7; ++lag_index) {
    const std::size_t lag = static_cast<std::size_t>(lags[lag_index]);
    const int slot = lag_index + 1;
    for (std::size_t row = lag; row < rows; ++row) {
      double result = 0.0;
      const double previous = launch.close[row - lag];
      if (abs_v3(previous) > kValueFloorV3) {
        result = div_rn_v3(sub_rn_v3(launch.close[row], previous), previous);
      }
      set_valid_v3(launch, slot, row, result);
    }
  }

  for (std::size_t row = 1; row < rows; ++row) {
    double result = 0.0;
    if (launch.close[row - 1U] > kValueFloorV3 &&
        launch.close[row] > kValueFloorV3) {
      if (!quant_log_positive_f64_v3(
              div_rn_v3(launch.close[row], launch.close[row - 1U]),
              &result)) {
        set_invalid_v3(launch, 8, row, kComputeFailureV3);
        continue;
      }
    }
    set_valid_v3(launch, 8, row, result);
  }

  for (std::size_t row = 0; row < rows; ++row) {
    const double range = sub_rn_v3(launch.high[row], launch.low[row]);
    if (range <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 9, row, kZeroDenominatorV3);
      continue;
    }
    double result = 0.0;
    if (range > kValueFloorV3 &&
        !quant_log_positive_f64_v3(range, &result)) {
      set_invalid_v3(launch, 9, row, kComputeFailureV3);
      continue;
    }
    set_valid_v3(launch, 9, row, result);
  }
}

__device__ void compute_log_affected_v3(
    const NeoResidentQuantLaunchV3& launch) {
  const std::size_t rows = static_cast<std::size_t>(launch.row_count);
  const double annualization_scale =
      sqrt_rn_v3(static_cast<double>(launch.annualization_periods_per_year));
  double log_two = 0.0;
  double log_one_hundred = 0.0;
  if (!quant_log_positive_f64_v3(2.0, &log_two) ||
      !quant_log_positive_f64_v3(100.0, &log_one_hundred)) {
    return;
  }

  const int realized_windows[4] = {5, 10, 20, 50};
  for (int window_index = 0; window_index < 4; ++window_index) {
    const std::size_t window =
        static_cast<std::size_t>(realized_windows[window_index]);
    const int slot = 10 + window_index;
    for (std::size_t row = window; row < rows; ++row) {
      double sum_squared = 0.0;
      for (std::size_t index = row - window + 1U; index <= row; ++index) {
        const double value = value_v3(launch, 8, index);
        sum_squared = add_rn_v3(sum_squared, mul_rn_v3(value, value));
      }
      const double variance =
          div_rn_v3(sum_squared, static_cast<double>(window));
      set_valid_v3(launch, slot, row,
                   mul_rn_v3(sqrt_rn_v3(variance), annualization_scale));
    }
  }

  const int volatility_windows[2] = {10, 20};
  for (int window_index = 0; window_index < 2; ++window_index) {
    const std::size_t window =
        static_cast<std::size_t>(volatility_windows[window_index]);
    for (std::size_t row = window; row < rows; ++row) {
      double sum = 0.0;
      bool failed = false;
      for (std::size_t index = row - window + 1U; index <= row; ++index) {
        if (abs_v3(launch.open[index]) <= kValueFloorV3) continue;
        double up = 0.0;
        double down = 0.0;
        double close = 0.0;
        if (!quant_log_positive_f64_v3(
                div_rn_v3(launch.high[index], launch.open[index]), &up) ||
            !quant_log_positive_f64_v3(
                div_rn_v3(launch.low[index], launch.open[index]), &down) ||
            !quant_log_positive_f64_v3(
                div_rn_v3(launch.close[index], launch.open[index]), &close)) {
          failed = true;
          break;
        }
        const double range = sub_rn_v3(up, down);
        const double first = mul_rn_v3(0.5, mul_rn_v3(range, range));
        const double second = mul_rn_v3(
            sub_rn_v3(log_two, 1.0), mul_rn_v3(close, close));
        sum = add_rn_v3(sum, sub_rn_v3(first, second));
      }
      const int slot = 14 + window_index;
      if (failed) {
        set_invalid_v3(launch, slot, row, kComputeFailureV3);
        continue;
      }
      const double mean = div_rn_v3(sum, static_cast<double>(window));
      set_valid_v3(
          launch, slot, row,
          mul_rn_v3(sqrt_rn_v3(abs_v3(mean)), annualization_scale));
    }
  }

  for (int window_index = 0; window_index < 2; ++window_index) {
    const std::size_t window =
        static_cast<std::size_t>(volatility_windows[window_index]);
    for (std::size_t row = window; row < rows; ++row) {
      double sum = 0.0;
      bool failed = false;
      for (std::size_t index = row - window + 1U; index <= row; ++index) {
        if (launch.low[index] <= kValueFloorV3) continue;
        double high_low = 0.0;
        if (!quant_log_positive_f64_v3(
                div_rn_v3(launch.high[index], launch.low[index]),
                &high_low)) {
          failed = true;
          break;
        }
        sum = add_rn_v3(sum, mul_rn_v3(high_low, high_low));
      }
      const int slot = 16 + window_index;
      if (failed) {
        set_invalid_v3(launch, slot, row, kComputeFailureV3);
        continue;
      }
      const double factor_denominator = mul_rn_v3(
          mul_rn_v3(4.0, static_cast<double>(window)), log_two);
      const double factor = div_rn_v3(1.0, factor_denominator);
      set_valid_v3(
          launch, slot, row,
          mul_rn_v3(sqrt_rn_v3(mul_rn_v3(factor, sum)),
                    annualization_scale));
    }
  }

  for (std::size_t row = 20U; row < rows; ++row) {
    double short_squared = 0.0;
    double long_squared = 0.0;
    for (std::size_t index = row - 4U; index <= row; ++index) {
      const double value = value_v3(launch, 8, index);
      short_squared = add_rn_v3(short_squared, mul_rn_v3(value, value));
    }
    for (std::size_t index = row - 19U; index <= row; ++index) {
      const double value = value_v3(launch, 8, index);
      long_squared = add_rn_v3(long_squared, mul_rn_v3(value, value));
    }
    if (long_squared <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 18, row, kZeroDenominatorV3);
      continue;
    }
    const double short_value = sqrt_rn_v3(div_rn_v3(short_squared, 5.0));
    const double long_value = sqrt_rn_v3(div_rn_v3(long_squared, 20.0));
    set_valid_v3(launch, 18, row,
                 long_value > kValueFloorV3
                     ? div_rn_v3(short_value, long_value)
                     : 1.0);
  }

  for (std::size_t row = 100U; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t index = row - 99U; index <= row; ++index) {
      sum = add_rn_v3(sum, value_v3(launch, 8, index));
    }
    const double mean = div_rn_v3(sum, 100.0);
    double running = 0.0;
    double minimum = from_bits_v3(0x7ff0000000000000ULL);
    double maximum = -from_bits_v3(0x7ff0000000000000ULL);
    double variance_sum = 0.0;
    for (std::size_t index = row - 99U; index <= row; ++index) {
      const double deviation = sub_rn_v3(value_v3(launch, 8, index), mean);
      running = add_rn_v3(running, deviation);
      minimum = min_v3(minimum, running);
      maximum = max_v3(maximum, running);
      variance_sum =
          add_rn_v3(variance_sum, mul_rn_v3(deviation, deviation));
    }
    const double spread = sub_rn_v3(maximum, minimum);
    const double deviation = sqrt_rn_v3(div_rn_v3(variance_sum, 99.0));
    if (deviation <= kValidityEpsilonV3 ||
        spread <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 19, row, kZeroDenominatorV3);
      continue;
    }
    double logarithm = 0.0;
    if (!quant_log_positive_f64_v3(div_rn_v3(spread, deviation),
                                   &logarithm)) {
      set_invalid_v3(launch, 19, row, kComputeFailureV3);
      continue;
    }
    set_valid_v3(launch, 19, row,
                 clamp_v3(div_rn_v3(logarithm, log_one_hundred), 0.0, 1.0));
  }

  const int autocorrelation_lags[3] = {1, 5, 10};
  for (int lag_index = 0; lag_index < 3; ++lag_index) {
    const std::size_t lag =
        static_cast<std::size_t>(autocorrelation_lags[lag_index]);
    const int slot = 20 + lag_index;
    for (std::size_t row = 50U + lag; row < rows; ++row) {
      double sum = 0.0;
      for (std::size_t offset = 0; offset < 50U; ++offset) {
        sum = add_rn_v3(sum, value_v3(launch, 8, row - 49U + offset));
      }
      const double mean = div_rn_v3(sum, 50.0);
      double numerator = 0.0;
      double denominator = 0.0;
      for (std::size_t offset = lag; offset < 50U; ++offset) {
        const double current =
            sub_rn_v3(value_v3(launch, 8, row - 49U + offset), mean);
        const double lagged = sub_rn_v3(
            value_v3(launch, 8, row - 49U + offset - lag), mean);
        numerator = add_rn_v3(numerator, mul_rn_v3(current, lagged));
        denominator =
            add_rn_v3(denominator, mul_rn_v3(current, current));
      }
      if (denominator <= kValidityEpsilonV3) {
        set_invalid_v3(launch, slot, row, kZeroDenominatorV3);
      } else {
        set_valid_v3(
            launch, slot, row,
            clamp_v3(div_rn_v3(numerator, denominator), -1.0, 1.0));
      }
    }
  }

  for (std::size_t row = 30U; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t index = row - 29U; index <= row; ++index) {
      sum = add_rn_v3(sum, value_v3(launch, 8, index));
    }
    const double mean = div_rn_v3(sum, 30.0);
    double second = 0.0;
    double third = 0.0;
    double fourth = 0.0;
    for (std::size_t index = row - 29U; index <= row; ++index) {
      const double deviation = sub_rn_v3(value_v3(launch, 8, index), mean);
      const double squared = mul_rn_v3(deviation, deviation);
      second = add_rn_v3(second, squared);
      third = add_rn_v3(third, mul_rn_v3(squared, deviation));
      fourth =
          add_rn_v3(fourth, mul_rn_v3(mul_rn_v3(squared, deviation), deviation));
    }
    second = div_rn_v3(second, 30.0);
    third = div_rn_v3(third, 30.0);
    fourth = div_rn_v3(fourth, 30.0);
    const double standard_deviation = sqrt_rn_v3(second);
    if (standard_deviation <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 25, row, kZeroDenominatorV3);
      set_invalid_v3(launch, 26, row, kZeroDenominatorV3);
      continue;
    }
    const double standard_squared =
        mul_rn_v3(standard_deviation, standard_deviation);
    const double standard_cubed =
        mul_rn_v3(standard_squared, standard_deviation);
    const double standard_fourth =
        mul_rn_v3(standard_squared, standard_squared);
    set_valid_v3(launch, 25, row,
                 clamp_v3(div_rn_v3(third, standard_cubed), -10.0, 10.0));
    set_valid_v3(
        launch, 26, row,
        clamp_v3(sub_rn_v3(div_rn_v3(fourth, standard_fourth), 3.0),
                 -10.0, 50.0));
  }

  for (std::size_t row = 30U; row < rows; ++row) {
    double maximum = -from_bits_v3(0x7ff0000000000000ULL);
    double minimum = from_bits_v3(0x7ff0000000000000ULL);
    for (std::size_t index = row - 30U; index <= row; ++index) {
      maximum = max_v3(maximum, launch.close[index]);
      minimum = min_v3(minimum, launch.close[index]);
    }
    const double range = sub_rn_v3(maximum, minimum);
    if (range <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 57, row, kZeroDenominatorV3);
      continue;
    }
    double result = 1.5;
    if (range > kValueFloorV3) {
      std::size_t sign_changes = 0U;
      for (std::size_t index = row - 28U; index <= row; ++index) {
        const double current =
            sub_rn_v3(launch.close[index], launch.close[index - 1U]);
        const double previous = sub_rn_v3(launch.close[index - 1U],
                                          launch.close[index - 2U]);
        if (mul_rn_v3(current, previous) < 0.0) ++sign_changes;
      }
      double log_points = 0.0;
      double log_ratio = 0.0;
      const double points = 31.0;
      const double denominator = add_rn_v3(
          points, mul_rn_v3(0.4, static_cast<double>(sign_changes)));
      if (!quant_log_positive_f64_v3(points, &log_points) ||
          !quant_log_positive_f64_v3(div_rn_v3(points, denominator),
                                     &log_ratio)) {
        set_invalid_v3(launch, 57, row, kComputeFailureV3);
        continue;
      }
      result = clamp_v3(
          div_rn_v3(log_points, add_rn_v3(log_points, log_ratio)), 1.0,
          2.0);
    }
    set_valid_v3(launch, 57, row, result);
  }
}

__device__ void compute_bitwise_preserved_v3(
    const NeoResidentQuantLaunchV3& launch) {
  const std::size_t rows = static_cast<std::size_t>(launch.row_count);

  const int efficiency_windows[2] = {10, 20};
  for (int window_index = 0; window_index < 2; ++window_index) {
    const std::size_t window =
        static_cast<std::size_t>(efficiency_windows[window_index]);
    const int slot = 23 + window_index;
    for (std::size_t row = window; row < rows; ++row) {
      const double direction =
          abs_v3(sub_rn_v3(launch.close[row], launch.close[row - window]));
      double volatility = 0.0;
      for (std::size_t index = row - window + 1U; index <= row; ++index) {
        volatility = add_rn_v3(
            volatility,
            abs_v3(sub_rn_v3(launch.close[index], launch.close[index - 1U])));
      }
      if (volatility <= kValidityEpsilonV3) {
        set_invalid_v3(launch, slot, row, kZeroDenominatorV3);
      } else {
        set_valid_v3(launch, slot, row,
                     volatility > kValueFloorV3
                         ? div_rn_v3(direction, volatility)
                         : 0.0);
      }
    }
  }

  for (std::size_t row = 20U; row < rows; ++row) {
    double sum_delta_volume = 0.0;
    double value_volume_squared = 0.0;
    double validity_volume_squared = 0.0;
    for (std::size_t index = row - 19U; index <= row; ++index) {
      const double delta =
          sub_rn_v3(launch.close[index], launch.close[index - 1U]);
      const double signed_volume =
          mul_rn_v3(signum_v3(delta), launch.volume[index]);
      sum_delta_volume =
          add_rn_v3(sum_delta_volume, mul_rn_v3(delta, signed_volume));
      // Rust f64::signum has magnitude one for both signed zeroes, so the
      // frozen v2 value denominator includes flat-close volume. The explicit
      // f64 validity replay instead assigns zero direction to flat deltas.
      value_volume_squared = add_rn_v3(
          value_volume_squared,
          mul_rn_v3(launch.volume[index], launch.volume[index]));
      validity_volume_squared =
          add_rn_v3(validity_volume_squared,
                    mul_rn_v3(signed_volume, signed_volume));
    }
    if (validity_volume_squared <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 27, row, kZeroDenominatorV3);
    } else {
      set_valid_v3(launch, 27, row,
                   abs_v3(value_volume_squared) > kValueFloorV3
                       ? div_rn_v3(sum_delta_volume, value_volume_squared)
                       : 0.0);
    }
  }

  for (std::size_t row = 500U; row < rows; ++row) {
    double buy_volume = 0.0;
    double sell_volume = 0.0;
    double total_volume = 0.0;
    for (std::size_t index = row - 500U; index < row; ++index) {
      const double midpoint =
          div_rn_v3(add_rn_v3(launch.high[index], launch.low[index]), 2.0);
      const double volume = abs_v3(launch.volume[index]);
      if (launch.close[index] > midpoint) {
        buy_volume = add_rn_v3(buy_volume, volume);
      } else {
        sell_volume = add_rn_v3(sell_volume, volume);
      }
      total_volume = add_rn_v3(total_volume, volume);
    }
    if (total_volume <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 28, row, kZeroDenominatorV3);
    } else {
      set_valid_v3(
          launch, 28, row,
          total_volume > kValueFloorV3
              ? div_rn_v3(abs_v3(sub_rn_v3(buy_volume, sell_volume)),
                          total_volume)
              : 0.0);
    }
  }

  for (std::size_t row = 20U; row < rows; ++row) {
    double sum = 0.0;
    std::size_t count = 0U;
    bool every_volume_is_zero = true;
    for (std::size_t index = row - 19U; index <= row; ++index) {
      if (launch.volume[index] > kValidityEpsilonV3) {
        every_volume_is_zero = false;
      }
      if (abs_v3(launch.volume[index]) > kValueFloorV3 && index > 0U) {
        const double return_value = div_rn_v3(
            abs_v3(sub_rn_v3(launch.close[index], launch.close[index - 1U])),
            max_v3(launch.close[index - 1U], kValueFloorV3));
        sum = add_rn_v3(
            sum, div_rn_v3(return_value, abs_v3(launch.volume[index])));
        ++count;
      }
    }
    if (every_volume_is_zero) {
      set_invalid_v3(launch, 29, row, kZeroDenominatorV3);
    } else {
      set_valid_v3(launch, 29, row,
                   count > 0U ? div_rn_v3(sum, static_cast<double>(count))
                              : 0.0);
    }
  }

  for (std::size_t row = 21U; row < rows; ++row) {
    double covariance_sum = 0.0;
    std::size_t count = 0U;
    for (std::size_t index = row - 19U; index <= row; ++index) {
      if (index >= 2U) {
        const double current =
            sub_rn_v3(launch.close[index], launch.close[index - 1U]);
        const double previous = sub_rn_v3(launch.close[index - 1U],
                                          launch.close[index - 2U]);
        covariance_sum =
            add_rn_v3(covariance_sum, mul_rn_v3(current, previous));
        ++count;
      }
    }
    const double covariance =
        div_rn_v3(covariance_sum, static_cast<double>(count));
    set_valid_v3(launch, 30, row,
                 covariance < 0.0
                     ? mul_rn_v3(2.0, sqrt_rn_v3(-covariance))
                     : 0.0);
  }

  double consecutive_up = 0.0;
  double consecutive_down = 0.0;
  for (std::size_t row = 1U; row < rows; ++row) {
    if (launch.close[row] > launch.close[row - 1U]) {
      consecutive_up = add_rn_v3(consecutive_up, 1.0);
      consecutive_down = 0.0;
    } else if (launch.close[row] < launch.close[row - 1U]) {
      consecutive_down = add_rn_v3(consecutive_down, 1.0);
      consecutive_up = 0.0;
    } else {
      consecutive_up = 0.0;
      consecutive_down = 0.0;
    }
    set_valid_v3(launch, 31, row, consecutive_up);
    set_valid_v3(launch, 32, row, consecutive_down);
  }

  for (std::size_t row = 1U; row < rows; ++row) {
    set_valid_v3(launch, 33, row,
                 launch.high[row] <= launch.high[row - 1U] &&
                         launch.low[row] >= launch.low[row - 1U]
                     ? 1.0
                     : 0.0);
    set_valid_v3(launch, 34, row,
                 launch.high[row] > launch.high[row - 1U] &&
                         launch.low[row] < launch.low[row - 1U]
                     ? 1.0
                     : 0.0);
  }

  for (std::size_t row = 0; row < rows; ++row) {
    const double range = sub_rn_v3(launch.high[row], launch.low[row]);
    if (range <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 35, row, kZeroDenominatorV3);
      set_invalid_v3(launch, 36, row, kZeroDenominatorV3);
      set_invalid_v3(launch, 37, row, kZeroDenominatorV3);
      continue;
    }
    if (range > kValueFloorV3) {
      const double body_top = max_v3(launch.close[row], launch.open[row]);
      const double body_bottom = min_v3(launch.close[row], launch.open[row]);
      set_valid_v3(
          launch, 35, row,
          div_rn_v3(abs_v3(sub_rn_v3(launch.close[row], launch.open[row])),
                    range));
      set_valid_v3(launch, 36, row,
                   div_rn_v3(sub_rn_v3(launch.high[row], body_top), range));
      set_valid_v3(launch, 37, row,
                   div_rn_v3(sub_rn_v3(body_bottom, launch.low[row]), range));
    } else {
      set_valid_v3(launch, 35, row, 0.0);
      set_valid_v3(launch, 36, row, 0.0);
      set_valid_v3(launch, 37, row, 0.0);
    }
  }

  for (std::size_t row = 20U; row < rows; ++row) {
    double range_sum = 0.0;
    double early_sum = 0.0;
    double recent_range = 0.0;
    for (std::size_t offset = 0; offset < 20U; ++offset) {
      const std::size_t index = row - 20U + offset;
      const double range = sub_rn_v3(launch.high[index], launch.low[index]);
      range_sum = add_rn_v3(range_sum, range);
      if (offset < 6U) early_sum = add_rn_v3(early_sum, range);
      if (offset == 19U) recent_range = range;
    }
    const double average = div_rn_v3(range_sum, 20.0);
    if (average <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 45, row, kZeroDenominatorV3);
      continue;
    }
    const double early = div_rn_v3(early_sum, div_rn_v3(20.0, 3.0));
    double phase = 0.0;
    if (early < mul_rn_v3(average, 0.6) &&
        recent_range > mul_rn_v3(average, 1.5)) {
      phase = launch.close[row] > launch.open[row] ? 1.0 : -1.0;
    } else if (early < mul_rn_v3(average, 0.7)) {
      phase = 0.3;
    }
    set_valid_v3(launch, 45, row, phase);
  }

  for (std::size_t row = 30U; row < rows; ++row) {
    double period_low = from_bits_v3(0x7ff0000000000000ULL);
    double period_high = -from_bits_v3(0x7ff0000000000000ULL);
    for (std::size_t index = row - 30U; index < row; ++index) {
      period_low = min_v3(period_low, launch.low[index]);
      period_high = max_v3(period_high, launch.high[index]);
    }
    double phase = 0.0;
    if (launch.low[row] < period_low && launch.close[row] > period_low) {
      phase = 1.0;
    }
    if (launch.high[row] > period_high && launch.close[row] < period_high) {
      phase = -1.0;
    }
    set_valid_v3(launch, 46, row, phase);
  }

  for (std::size_t row = 1U; row < rows; ++row) {
    const double previous_body =
        abs_v3(sub_rn_v3(launch.close[row - 1U], launch.open[row - 1U]));
    const double current_body =
        abs_v3(sub_rn_v3(launch.close[row], launch.open[row]));
    const bool volume_increase =
        launch.volume[row] > mul_rn_v3(launch.volume[row - 1U], 1.2);
    double engulfing = 0.0;
    if (launch.close[row - 1U] < launch.open[row - 1U] &&
        launch.close[row] > launch.open[row] &&
        launch.open[row] <= launch.close[row - 1U] &&
        launch.close[row] >= launch.open[row - 1U] &&
        current_body > previous_body && volume_increase) {
      engulfing = 1.0;
    }
    if (launch.close[row - 1U] > launch.open[row - 1U] &&
        launch.close[row] < launch.open[row] &&
        launch.open[row] >= launch.close[row - 1U] &&
        launch.close[row] <= launch.open[row - 1U] &&
        current_body > previous_body && volume_increase) {
      engulfing = -1.0;
    }
    set_valid_v3(launch, 47, row, engulfing);
  }

  const int zscore_windows[2] = {20, 50};
  for (int window_index = 0; window_index < 2; ++window_index) {
    const std::size_t window =
        static_cast<std::size_t>(zscore_windows[window_index]);
    const int slot = 55 + window_index;
    for (std::size_t row = window; row < rows; ++row) {
      double sum = 0.0;
      for (std::size_t index = row - window; index < row; ++index) {
        sum = add_rn_v3(sum, launch.close[index]);
      }
      const double mean = div_rn_v3(sum, static_cast<double>(window));
      double variance_sum = 0.0;
      for (std::size_t index = row - window; index < row; ++index) {
        const double deviation = sub_rn_v3(launch.close[index], mean);
        variance_sum =
            add_rn_v3(variance_sum, mul_rn_v3(deviation, deviation));
      }
      const double standard_deviation = sqrt_rn_v3(div_rn_v3(
          variance_sum, static_cast<double>(window - 1U)));
      if (standard_deviation <= kValidityEpsilonV3) {
        set_invalid_v3(launch, slot, row, kZeroDenominatorV3);
      } else {
        set_valid_v3(launch, slot, row,
                     standard_deviation > kValueFloorV3
                         ? div_rn_v3(sub_rn_v3(launch.close[row], mean),
                                     standard_deviation)
                         : 0.0);
      }
    }
  }

  const int relative_volume_windows[3] = {10, 20, 50};
  for (int window_index = 0; window_index < 3; ++window_index) {
    const std::size_t window =
        static_cast<std::size_t>(relative_volume_windows[window_index]);
    const int slot = 58 + window_index;
    for (std::size_t row = window; row < rows; ++row) {
      double sum = 0.0;
      for (std::size_t index = row - window; index < row; ++index) {
        sum = add_rn_v3(sum, launch.volume[index]);
      }
      const double average = div_rn_v3(sum, static_cast<double>(window));
      if (average <= kValidityEpsilonV3) {
        set_invalid_v3(launch, slot, row, kZeroDenominatorV3);
      } else {
        set_valid_v3(launch, slot, row,
                     average > kValueFloorV3
                         ? div_rn_v3(launch.volume[row], average)
                         : 1.0);
      }
    }
  }

  double cumulative_delta = 0.0;
  double cumulative_ring[50] = {0.0};
  // The preserved CPU f64 contract has two intentionally distinct schedules:
  // values keep the legacy 1e-10 range floor, while validity replays delta for
  // every range above the logical epsilon. Keep both to preserve v2 value
  // bits while semantic-v4 applies the corrected rolling validity dependency.
  double validity_cumulative_delta = 0.0;
  double validity_cumulative_ring[50] = {0.0};
  bool invalid_delta_ring[50] = {false};
  std::size_t invalid_delta_window_count = 0U;
  for (std::size_t row = 0; row < rows; ++row) {
    const double range = sub_rn_v3(launch.high[row], launch.low[row]);
    double delta = 0.0;
    double validity_delta = 0.0;
    const bool invalid_delta = range <= kValidityEpsilonV3;
    const std::size_t delta_ring_slot = row % 50U;
    if (row >= 50U && invalid_delta_ring[delta_ring_slot]) {
      --invalid_delta_window_count;
    }
    invalid_delta_ring[delta_ring_slot] = invalid_delta;
    if (invalid_delta) ++invalid_delta_window_count;
    if (range > kValidityEpsilonV3) {
      const double validity_buy_fraction =
          div_rn_v3(sub_rn_v3(launch.close[row], launch.low[row]), range);
      validity_delta = mul_rn_v3(
          launch.volume[row],
          sub_rn_v3(mul_rn_v3(2.0, validity_buy_fraction), 1.0));
      validity_cumulative_delta =
          add_rn_v3(validity_cumulative_delta, validity_delta);
    }
    if (range > kValueFloorV3) {
      const double buy_fraction =
          div_rn_v3(sub_rn_v3(launch.close[row], launch.low[row]), range);
      delta = mul_rn_v3(
          launch.volume[row], sub_rn_v3(mul_rn_v3(2.0, buy_fraction), 1.0));
      cumulative_delta = add_rn_v3(cumulative_delta, delta);
    }
    if (range <= kValidityEpsilonV3) {
      set_invalid_v3(launch, 61, row, kZeroDenominatorV3);
    } else {
      set_valid_v3(launch, 61, row, delta);
    }

    if (row >= 50U) {
      if (invalid_delta_window_count != 0U) {
        set_invalid_v3(launch, 62, row, kZeroDenominatorV3);
      } else {
        double validity_sum = 0.0;
        for (std::size_t index = row - 50U; index < row; ++index) {
          validity_sum = add_rn_v3(
              validity_sum, validity_cumulative_ring[index % 50U]);
        }
        const double validity_mean = div_rn_v3(validity_sum, 50.0);
        double validity_variance_sum = 0.0;
        for (std::size_t index = row - 50U; index < row; ++index) {
          const double validity_deviation = sub_rn_v3(
              validity_cumulative_ring[index % 50U], validity_mean);
          validity_variance_sum = add_rn_v3(
              validity_variance_sum,
              mul_rn_v3(validity_deviation, validity_deviation));
        }
        const double validity_standard_deviation =
            sqrt_rn_v3(div_rn_v3(validity_variance_sum, 49.0));
        if (validity_standard_deviation <= kValidityEpsilonV3) {
          set_invalid_v3(launch, 62, row, kZeroDenominatorV3);
        } else {
          double value_sum = 0.0;
          for (std::size_t index = row - 50U; index < row; ++index) {
            value_sum =
                add_rn_v3(value_sum, cumulative_ring[index % 50U]);
          }
          const double value_mean = div_rn_v3(value_sum, 50.0);
          double value_variance_sum = 0.0;
          for (std::size_t index = row - 50U; index < row; ++index) {
            const double value_deviation =
                sub_rn_v3(cumulative_ring[index % 50U], value_mean);
            value_variance_sum = add_rn_v3(
                value_variance_sum,
                mul_rn_v3(value_deviation, value_deviation));
          }
          const double value_standard_deviation =
              sqrt_rn_v3(div_rn_v3(value_variance_sum, 49.0));
          set_valid_v3(
              launch, 62, row,
              value_standard_deviation > kValueFloorV3
                  ? div_rn_v3(sub_rn_v3(cumulative_delta, value_mean),
                              value_standard_deviation)
                  : 0.0);
        }
      }
    }
    cumulative_ring[row % 50U] = cumulative_delta;
    validity_cumulative_ring[row % 50U] = validity_cumulative_delta;
  }
}

__device__ __forceinline__ std::int64_t floor_div_positive_v3(
    std::int64_t value, std::int64_t divisor) {
  std::int64_t quotient = value / divisor;
  const std::int64_t remainder = value % divisor;
  if (remainder < 0) --quotient;
  return quotient;
}

__device__ __forceinline__ std::int64_t euclidean_remainder_v3(
    std::int64_t value, std::int64_t divisor) {
  const std::int64_t remainder = value % divisor;
  return remainder < 0 ? remainder + divisor : remainder;
}

__device__ void compute_temporal_session_v3(
    const NeoResidentQuantLaunchV3& launch) {
  const std::size_t rows = static_cast<std::size_t>(launch.row_count);
  double completed_high[5] = {0.0};
  double completed_low[5] = {0.0};
  double completed_close[5] = {0.0};
  std::size_t completed_count = 0U;

  std::int64_t current_day_key =
      floor_div_positive_v3(launch.timestamps[0], kUtcDayMillisV3);
  double current_day_high = -from_bits_v3(0x7ff0000000000000ULL);
  double current_day_low = from_bits_v3(0x7ff0000000000000ULL);
  double current_day_close = launch.close[0];
  std::size_t orb_count = 0U;
  double orb_high[3] = {
      -from_bits_v3(0x7ff0000000000000ULL),
      -from_bits_v3(0x7ff0000000000000ULL),
      -from_bits_v3(0x7ff0000000000000ULL)};
  double orb_low[3] = {from_bits_v3(0x7ff0000000000000ULL),
                       from_bits_v3(0x7ff0000000000000ULL),
                       from_bits_v3(0x7ff0000000000000ULL)};
  const std::size_t orb_thresholds[3] = {4U, 8U, 12U};

  for (std::size_t row = 0; row < rows; ++row) {
    const std::int64_t day_key =
        floor_div_positive_v3(launch.timestamps[row], kUtcDayMillisV3);
    if (day_key != current_day_key) {
      if (completed_count < 5U) {
        completed_high[completed_count] = current_day_high;
        completed_low[completed_count] = current_day_low;
        completed_close[completed_count] = current_day_close;
        ++completed_count;
      } else {
        for (std::size_t index = 0; index < 4U; ++index) {
          completed_high[index] = completed_high[index + 1U];
          completed_low[index] = completed_low[index + 1U];
          completed_close[index] = completed_close[index + 1U];
        }
        completed_high[4] = current_day_high;
        completed_low[4] = current_day_low;
        completed_close[4] = current_day_close;
      }
      current_day_key = day_key;
      current_day_high = -from_bits_v3(0x7ff0000000000000ULL);
      current_day_low = from_bits_v3(0x7ff0000000000000ULL);
      current_day_close = launch.close[row];
      orb_count = 0U;
      for (int slot = 0; slot < 3; ++slot) {
        orb_high[slot] = -from_bits_v3(0x7ff0000000000000ULL);
        orb_low[slot] = from_bits_v3(0x7ff0000000000000ULL);
      }
    }

    if (completed_count > 0U) {
      const std::size_t previous_index = completed_count - 1U;
      const double previous_high = completed_high[previous_index];
      const double previous_low = completed_low[previous_index];
      const double previous_close = completed_close[previous_index];
      const double previous_range = sub_rn_v3(previous_high, previous_low);
      if (previous_range <= kValidityEpsilonV3) {
        set_invalid_v3(launch, 38, row, kZeroDenominatorV3);
        set_invalid_v3(launch, 39, row, kZeroDenominatorV3);
      } else {
        const double denominator = max_v3(previous_range, kValueFloorV3);
        set_valid_v3(
            launch, 38, row,
            div_rn_v3(sub_rn_v3(launch.close[row], previous_high),
                      denominator));
        set_valid_v3(
            launch, 39, row,
            div_rn_v3(sub_rn_v3(launch.close[row], previous_low),
                      denominator));
      }

      const double pivot = div_rn_v3(
          add_rn_v3(add_rn_v3(previous_high, previous_low), previous_close),
          3.0);
      const double r1 = sub_rn_v3(mul_rn_v3(2.0, pivot), previous_low);
      const double r2 = add_rn_v3(pivot, previous_range);
      const double s1 = sub_rn_v3(mul_rn_v3(2.0, pivot), previous_high);
      const double s2 = sub_rn_v3(pivot, previous_range);
      const double camarilla_step =
          div_rn_v3(mul_rn_v3(previous_range, 1.1), 4.0);
      const double camarilla_r3 = add_rn_v3(previous_close, camarilla_step);
      const double camarilla_s3 = sub_rn_v3(previous_close, camarilla_step);
      const double current_range =
          sub_rn_v3(launch.high[row], launch.low[row]);
      if (current_range <= kValidityEpsilonV3) {
        for (int slot = 48; slot <= 54; ++slot) {
          set_invalid_v3(launch, slot, row, kZeroDenominatorV3);
        }
      } else {
        const double denominator = max_v3(current_range, kValueFloorV3);
        const double levels[7] = {pivot, r1, r2, s1, s2, camarilla_r3,
                                  camarilla_s3};
        for (int index = 0; index < 7; ++index) {
          set_valid_v3(
              launch, 48 + index, row,
              div_rn_v3(sub_rn_v3(launch.close[row], levels[index]),
                        denominator));
        }
      }
    }

    if (completed_count == 5U) {
      double week_high = completed_high[0];
      double week_low = completed_low[0];
      for (std::size_t index = 1U; index < 5U; ++index) {
        week_high = max_v3(week_high, completed_high[index]);
        week_low = min_v3(week_low, completed_low[index]);
      }
      const double week_range = sub_rn_v3(week_high, week_low);
      if (week_range <= kValidityEpsilonV3) {
        set_invalid_v3(launch, 40, row, kZeroDenominatorV3);
        set_invalid_v3(launch, 41, row, kZeroDenominatorV3);
      } else {
        const double denominator = max_v3(week_range, kValueFloorV3);
        set_valid_v3(
            launch, 40, row,
            div_rn_v3(sub_rn_v3(launch.close[row], week_high), denominator));
        set_valid_v3(
            launch, 41, row,
            div_rn_v3(sub_rn_v3(launch.close[row], week_low), denominator));
      }
    }

    for (int orb_slot = 0; orb_slot < 3; ++orb_slot) {
      if (orb_count >= orb_thresholds[orb_slot]) {
        const double signal =
            launch.close[row] > orb_high[orb_slot]
                ? 1.0
                : (launch.close[row] < orb_low[orb_slot] ? -1.0 : 0.0);
        set_valid_v3(launch, 42 + orb_slot, row, signal);
      }
    }

    const std::int64_t millis_in_day =
        euclidean_remainder_v3(launch.timestamps[row], kUtcDayMillisV3);
    if (millis_in_day < kAsianSessionMillisV3) {
      for (int orb_slot = 0; orb_slot < 3; ++orb_slot) {
        if (orb_count < orb_thresholds[orb_slot]) {
          orb_high[orb_slot] = max_v3(orb_high[orb_slot], launch.high[row]);
          orb_low[orb_slot] = min_v3(orb_low[orb_slot], launch.low[row]);
        }
      }
      ++orb_count;
    }

    current_day_high = max_v3(current_day_high, launch.high[row]);
    current_day_low = min_v3(current_day_low, launch.low[row]);
    current_day_close = launch.close[row];
  }
}

// One lane deliberately owns the complete fixed-order schedule. Every rolling
// scan has a fixed maximum lookback of 500 bars, so the schedule is O(N) with
// a larger constant than the legacy host route. Temporal state consumes only
// the previous completed UTC day, the last five completed observed days, and
// the first N observed Asian-session bars. No feature value leaves the device.
__global__ void resident_quant_all_f64_v3(NeoResidentQuantLaunchV3 launch) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  compute_base_and_log_v3(launch);
  compute_log_affected_v3(launch);
  compute_bitwise_preserved_v3(launch);
  compute_temporal_session_v3(launch);
}

bool checked_product_v3(std::uint64_t left, std::uint64_t right,
                        std::uint64_t* output) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left)
    return false;
  *output = left * right;
  return true;
}

}  // namespace

extern "C" int neoethos_resident_quant_f64_v3(
    const NeoResidentQuantLaunchV3* launch, CUstream_st* stream_handle) {
  if (launch == nullptr || stream_handle == nullptr ||
      launch->abi_version != NEOETHOS_RESIDENT_QUANT_ABI_VERSION_V3 ||
      launch->semantic_version !=
          NEOETHOS_RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4 ||
      launch->feature_column_count !=
          NEOETHOS_RESIDENT_QUANT_FEATURE_COLUMNS_V3 ||
      launch->reserved != 0U || launch->row_count == 0U ||
      launch->timeframe_millis == 0U || launch->open == nullptr ||
      launch->high == nullptr || launch->low == nullptr ||
      launch->close == nullptr || launch->volume == nullptr ||
      launch->timestamps == nullptr || launch->feature_values == nullptr ||
      launch->feature_validity_u8 == nullptr ||
      launch->row_count >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      launch->bars_per_asian_session < 12U ||
      launch->bars_per_trading_week != launch->bars_per_utc_day * 5U ||
      launch->trading_sessions_per_year != kTradingSessionsPerYearV3 ||
      launch->timeframe_millis > static_cast<std::uint64_t>(kUtcDayMillisV3) ||
      static_cast<std::uint64_t>(kUtcDayMillisV3) % launch->timeframe_millis !=
          0U ||
      launch->bars_per_utc_day !=
          static_cast<std::uint64_t>(kUtcDayMillisV3) /
              launch->timeframe_millis ||
      static_cast<std::uint64_t>(kAsianSessionMillisV3) %
              launch->timeframe_millis !=
          0U ||
      launch->bars_per_asian_session !=
          static_cast<std::uint64_t>(kAsianSessionMillisV3) /
              launch->timeframe_millis) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  std::uint64_t expected_annualization = 0U;
  std::uint64_t feature_cells = 0U;
  if (!checked_product_v3(launch->trading_sessions_per_year,
                          launch->bars_per_utc_day,
                          &expected_annualization) ||
      launch->annualization_periods_per_year != expected_annualization ||
      !checked_product_v3(launch->row_count, kFeatureColumnCountV3,
                          &feature_cells) ||
      feature_cells >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const NeoResidentQuantLaunchV3 descriptor = *launch;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  resident_quant_all_f64_v3<<<1U, 1U, 0U, stream>>>(descriptor);
  return static_cast<int>(cudaGetLastError());
}
