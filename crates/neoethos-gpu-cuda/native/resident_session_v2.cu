#include "resident_session_v2_abi.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint64_t kFeatureColumnCountV2 = 23ULL;
constexpr std::uint64_t kRetainedBytesPerRowV2 = 207ULL;
constexpr std::int64_t kUtcDayMillisV2 = 86'400'000LL;
constexpr double kDenominatorEpsilonV2 = 1.0e-10;
constexpr unsigned char kValidV2 = 0U;
constexpr unsigned char kWarmupV2 = 1U;
constexpr unsigned char kZeroDenominatorV2 = 5U;

static_assert(kRetainedBytesPerRowV2 ==
              kFeatureColumnCountV2 *
                  (sizeof(double) + sizeof(unsigned char)));

enum SessionColumnV2 : std::size_t {
  kLondonOpenDistanceV2 = 0,
  kLondonHighDistanceV2,
  kLondonLowDistanceV2,
  kLondonRangeV2,
  kLondonVwapDistanceV2,
  kNewYorkOpenDistanceV2,
  kNewYorkHighDistanceV2,
  kNewYorkLowDistanceV2,
  kNewYorkRangeV2,
  kNewYorkVwapDistanceV2,
  kAsianOpenDistanceV2,
  kAsianCloseDistanceV2,
  kAsianRangeNormalizedV2,
  kLondonNewYorkOverlapV2,
  kSessionVolatilityRatioV2,
  kPreviousSessionCloseDistanceV2,
  kSessionOpenGapV2,
  kDailyRangePercentV2,
  kDailyBodyPercentV2,
  kDailyPositionV2,
  kDailyHighDistanceV2,
  kDailyLowDistanceV2,
  kDailyVwapDistanceV2,
};

struct SessionAccumV2 {
  double open;
  double high;
  double low;
  double close;
  double volume_sum;
  double vwap_numerator;
  double vwap_denominator;
  std::uint64_t bar_count;
  bool started;
};

__device__ __forceinline__ double abs_v2(double value) {
  return value < 0.0 ? -value : value;
}

__device__ __forceinline__ double max_v2(double left, double right) {
  return left > right ? left : right;
}

__device__ __forceinline__ double canonical_nan_v2() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

__device__ __forceinline__ bool finite_v2(double value) {
  return isfinite(value);
}

__device__ __forceinline__ void reset_accum_v2(SessionAccumV2* state,
                                                double open_price) {
  state->open = open_price;
  state->high = open_price;
  state->low = open_price;
  state->close = open_price;
  state->volume_sum = 0.0;
  state->vwap_numerator = 0.0;
  state->vwap_denominator = 0.0;
  state->bar_count = 0ULL;
  state->started = true;
}

__device__ __forceinline__ void update_accum_v2(SessionAccumV2* state,
                                                 double high, double low,
                                                 double close,
                                                 double volume) {
  if (high > state->high) {
    state->high = high;
  }
  if (low < state->low) {
    state->low = low;
  }
  state->close = close;
  state->volume_sum += volume;
  const double typical = (high + low + close) / 3.0;
  state->vwap_numerator += typical * volume;
  state->vwap_denominator += volume;
  state->bar_count += 1ULL;
}

__device__ __forceinline__ double range_v2(const SessionAccumV2& state) {
  return state.high - state.low;
}

__device__ __forceinline__ double body_v2(const SessionAccumV2& state) {
  return abs_v2(state.close - state.open);
}

__device__ __forceinline__ double vwap_v2(const SessionAccumV2& state) {
  return state.vwap_denominator > kDenominatorEpsilonV2
             ? state.vwap_numerator / state.vwap_denominator
             : state.close;
}

__device__ __forceinline__ void set_invalid_v2(
    double* values, unsigned char* validity, std::size_t rows,
    std::size_t column, std::size_t row, unsigned char reason) {
  const std::size_t index = column * rows + row;
  values[index] = canonical_nan_v2();
  validity[index] = reason;
}

__device__ __forceinline__ void set_valid_v2(
    double* values, unsigned char* validity, std::size_t rows,
    std::size_t column, std::size_t row, double value) {
  const std::size_t index = column * rows + row;
  values[index] = value;
  validity[index] = kValidV2;
}

__device__ __forceinline__ void set_atr_normalized_v2(
    double* values, unsigned char* validity, std::size_t rows,
    std::size_t column, std::size_t row, double numerator,
    double atr_denominator, unsigned char atr_validity) {
  if (atr_validity == kValidV2) {
    set_valid_v2(values, validity, rows, column, row,
                 numerator / atr_denominator);
  } else {
    set_invalid_v2(values, validity, rows, column, row, atr_validity);
  }
}

__global__ void resident_session_all_f64_v2(NeoResidentSessionLaunchV2 launch) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  const std::size_t rows = static_cast<std::size_t>(launch.row_count);
  SessionAccumV2 asian{};
  SessionAccumV2 london{};
  SessionAccumV2 new_york{};
  SessionAccumV2 daily{};
  double previous_session_close = canonical_nan_v2();
  double previous_asian_range = 0.0;
  bool has_previous_asian = false;
  double atr_sum = 0.0;
  std::uint64_t atr_count = 0ULL;

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < kFeatureColumnCountV2; ++column) {
      set_invalid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                     column, row, kWarmupV2);
    }

    const double open = launch.open[row];
    const double high = launch.high[row];
    const double low = launch.low[row];
    const double close = launch.close[row];
    const double volume = launch.volume[row];

    // Exact CPU cumulative ATR order: row zero is high-low; each later row
    // appends one true range before dividing by the cumulative observation
    // count. The value denominator keeps the legacy max(epsilon) behavior.
    if (row > 0U) {
      const double high_low = high - low;
      const double high_previous = abs_v2(high - launch.close[row - 1U]);
      const double low_previous = abs_v2(low - launch.close[row - 1U]);
      const double true_range =
          max_v2(max_v2(high_low, high_previous), low_previous);
      atr_sum += true_range;
      atr_count += 1ULL;
    }
    const double atr = atr_count > 0ULL
                           ? atr_sum / static_cast<double>(atr_count)
                           : high - low;
    const double atr_denominator = max_v2(atr, kDenominatorEpsilonV2);
    const unsigned char atr_validity =
        atr > kDenominatorEpsilonV2 ? kValidV2 : kZeroDenominatorV2;

    // The value clock consumes the admitted millisecond inference; the
    // validity clock consumes the original canonical millisecond timestamp.
    // Strict Session-v2 admission proves those clocks are the same bytes.
    const std::int64_t millis_in_day =
        launch.timestamps_ms[row] % kUtcDayMillisV2;
    const unsigned hour =
        static_cast<unsigned>(millis_in_day / 3'600'000LL);
    const unsigned minute =
        static_cast<unsigned>((millis_in_day % 3'600'000LL) / 60'000LL);
    const bool asian_open = hour == 0U && minute == 0U;
    const bool london_open = hour == 7U && minute == 0U;
    const bool new_york_open = hour == 12U && minute == 0U;

    if (asian_open) {
      if (asian.started) {
        previous_session_close = asian.close;
        previous_asian_range = range_v2(asian);
        has_previous_asian = true;
      }
      reset_accum_v2(&asian, open);
    }
    if (hour < 8U && asian.started) {
      update_accum_v2(&asian, high, low, close, volume);
    }

    if (london_open) {
      if (london.started) {
        previous_session_close = london.close;
      }
      reset_accum_v2(&london, open);
    }
    if (hour >= 7U && hour < 16U && london.started) {
      update_accum_v2(&london, high, low, close, volume);
    }

    if (new_york_open) {
      if (new_york.started) {
        previous_session_close = new_york.close;
      }
      reset_accum_v2(&new_york, open);
    }
    if (hour >= 12U && hour < 21U && new_york.started) {
      update_accum_v2(&new_york, high, low, close, volume);
    }

    if (asian_open) {
      reset_accum_v2(&daily, open);
    }
    update_accum_v2(&daily, high, low, close, volume);

    set_valid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                 kLondonNewYorkOverlapV2, row,
                 hour >= 12U && hour < 16U ? 1.0 : 0.0);

    if ((london_open || new_york_open) &&
        !finite_v2(previous_session_close)) {
      set_invalid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                     kSessionOpenGapV2, row, kWarmupV2);
    } else if ((london_open || new_york_open) &&
               atr_validity != kValidV2) {
      set_invalid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                     kSessionOpenGapV2, row, atr_validity);
    } else {
      const double open_gap =
          london_open || new_york_open
              ? (open - previous_session_close) / atr_denominator
              : 0.0;
      set_valid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                   kSessionOpenGapV2, row, open_gap);
    }

    if (london.started && london.bar_count > 0ULL) {
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kLondonOpenDistanceV2, row, close - london.open, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kLondonHighDistanceV2, row, close - london.high, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kLondonLowDistanceV2, row, close - london.low, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(launch.feature_values,
                            launch.feature_validity_u8, rows, kLondonRangeV2,
                            row, range_v2(london), atr_denominator,
                            atr_validity);
      const unsigned char vwap_validity =
          london.vwap_denominator <= kDenominatorEpsilonV2
              ? kZeroDenominatorV2
              : atr_validity;
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kLondonVwapDistanceV2, row, close - vwap_v2(london),
          atr_denominator, vwap_validity);
    }

    if (new_york.started && new_york.bar_count > 0ULL) {
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kNewYorkOpenDistanceV2, row, close - new_york.open, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kNewYorkHighDistanceV2, row, close - new_york.high, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kNewYorkLowDistanceV2, row, close - new_york.low, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(launch.feature_values,
                            launch.feature_validity_u8, rows, kNewYorkRangeV2,
                            row, range_v2(new_york), atr_denominator,
                            atr_validity);
      const unsigned char vwap_validity =
          new_york.vwap_denominator <= kDenominatorEpsilonV2
              ? kZeroDenominatorV2
              : atr_validity;
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kNewYorkVwapDistanceV2, row, close - vwap_v2(new_york),
          atr_denominator, vwap_validity);
    }

    if (asian.started && asian.bar_count > 0ULL) {
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kAsianOpenDistanceV2, row, close - asian.open, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kAsianCloseDistanceV2, row, close - asian.close, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kAsianRangeNormalizedV2, row, range_v2(asian), atr_denominator,
          atr_validity);
    }

    if (london.started && has_previous_asian) {
      if (previous_asian_range > kDenominatorEpsilonV2) {
        set_valid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                     kSessionVolatilityRatioV2, row,
                     range_v2(london) / previous_asian_range);
      } else {
        set_invalid_v2(launch.feature_values, launch.feature_validity_u8,
                       rows, kSessionVolatilityRatioV2, row,
                       kZeroDenominatorV2);
      }
    }

    if (finite_v2(previous_session_close)) {
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kPreviousSessionCloseDistanceV2, row,
          close - previous_session_close, atr_denominator, atr_validity);
    }

    if (daily.started && daily.bar_count > 0ULL) {
      const double daily_range = range_v2(daily);
      set_valid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                   kDailyRangePercentV2, row,
                   close > kDenominatorEpsilonV2 ? daily_range / close : 0.0);
      set_valid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                   kDailyBodyPercentV2, row,
                   close > kDenominatorEpsilonV2 ? body_v2(daily) / close
                                                 : 0.0);
      if (daily_range > kDenominatorEpsilonV2) {
        set_valid_v2(launch.feature_values, launch.feature_validity_u8, rows,
                     kDailyPositionV2, row,
                     (close - daily.low) / daily_range);
      } else {
        set_invalid_v2(launch.feature_values, launch.feature_validity_u8,
                       rows, kDailyPositionV2, row, kZeroDenominatorV2);
      }
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kDailyHighDistanceV2, row, close - daily.high, atr_denominator,
          atr_validity);
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kDailyLowDistanceV2, row, close - daily.low, atr_denominator,
          atr_validity);
      const unsigned char vwap_validity =
          daily.vwap_denominator <= kDenominatorEpsilonV2
              ? kZeroDenominatorV2
              : atr_validity;
      set_atr_normalized_v2(
          launch.feature_values, launch.feature_validity_u8, rows,
          kDailyVwapDistanceV2, row, close - vwap_v2(daily), atr_denominator,
          vwap_validity);
    }
  }
}

int launch_status_v2() {
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int neoethos_resident_session_f64_v2(
    const NeoResidentSessionLaunchV2* launch, cudaStream_t stream) {
  if (launch == nullptr || stream == nullptr ||
      launch->abi_version != NEOETHOS_RESIDENT_SESSION_ABI_VERSION_V2 ||
      launch->semantic_version !=
          NEOETHOS_RESIDENT_SESSION_SEMANTIC_VERSION_V2 ||
      launch->feature_column_count !=
          NEOETHOS_RESIDENT_SESSION_FEATURE_COLUMNS_V2 ||
      launch->reserved != 0U || launch->row_count == 0ULL ||
      launch->row_count >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      launch->row_count >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     kFeatureColumnCountV2) ||
      launch->open == nullptr || launch->high == nullptr ||
      launch->low == nullptr || launch->close == nullptr ||
      launch->volume == nullptr || launch->timestamps_ms == nullptr ||
      launch->feature_values == nullptr ||
      launch->feature_validity_u8 == nullptr) {
    return -1;
  }
  resident_session_all_f64_v2<<<1U, 1U, 0U, stream>>>(*launch);
  return launch_status_v2();
}
