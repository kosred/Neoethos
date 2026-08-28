#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kFootprintColumnsV2 = 7;
constexpr int kPrefixSeriesV2 = 8;
constexpr std::size_t kRollingWindowV2 = 96;
constexpr std::size_t kCorrelationWindowV2 = 48;
constexpr std::size_t kDeltaWindowV2 = 24;
constexpr double kEpsilonV2 = 1.0e-12;

enum FootprintColumnV2 : int {
  kVolumeZV2 = 0,
  kAbsorptionV2,
  kEffortResultDivergenceV2,
  kClimaxV2,
  kDeltaProxyV2,
  kVolumePriceCorrelationV2,
  kFixWindowV2,
};

enum FootprintValidityV2 : unsigned char {
  kValidV2 = 0,
  kWarmupV2 = 1,
  kMissingInputV2 = 2,
  kGapV2 = 3,
  kStaleV2 = 4,
  kZeroDenominatorV2 = 5,
  kDegenerateV2 = 6,
  kNonFiniteV2 = 7,
  kComputeFailureV2 = 8,
  kAlignmentMissingV2 = 9,
};

enum PrefixSeriesV2 : int {
  kVolumePrefixV2 = 0,
  kVolumeSquaredPrefixV2,
  kRangePrefixV2,
  kRangeSquaredPrefixV2,
  kAbsoluteReturnPrefixV2,
  kAbsoluteReturnSquaredPrefixV2,
  kSignedVolumePrefixV2,
  kVolumeAbsoluteReturnProductPrefixV2,
};

__device__ __forceinline__ double abs_v2(double value) {
  return value < 0.0 ? -value : value;
}

__device__ __forceinline__ double min_v2(double left, double right) {
  return left < right ? left : right;
}

__device__ __forceinline__ double max_v2(double left, double right) {
  return left > right ? left : right;
}

__device__ __forceinline__ double clamp_v2(double value, double low,
                                            double high) {
  return min_v2(max_v2(value, low), high);
}

__device__ __forceinline__ double canonical_nan_v2() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

__device__ __forceinline__ double* prefix_series_v2(double* scratch,
                                                     std::size_t stride,
                                                     PrefixSeriesV2 series) {
  return scratch + static_cast<std::size_t>(series) * stride;
}

__device__ __forceinline__ const double* prefix_series_v2(
    const double* scratch, std::size_t stride, PrefixSeriesV2 series) {
  return scratch + static_cast<std::size_t>(series) * stride;
}

__global__ void resident_footprint_prefix_v2(
    const double* open, const double* high, const double* low,
    const double* close, const double* volume, std::size_t rows,
    double* prefix_scratch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::size_t stride = rows + 1;
  double* volume_prefix =
      prefix_series_v2(prefix_scratch, stride, kVolumePrefixV2);
  double* volume_squared_prefix =
      prefix_series_v2(prefix_scratch, stride, kVolumeSquaredPrefixV2);
  double* range_prefix =
      prefix_series_v2(prefix_scratch, stride, kRangePrefixV2);
  double* range_squared_prefix =
      prefix_series_v2(prefix_scratch, stride, kRangeSquaredPrefixV2);
  double* absolute_return_prefix =
      prefix_series_v2(prefix_scratch, stride, kAbsoluteReturnPrefixV2);
  double* absolute_return_squared_prefix = prefix_series_v2(
      prefix_scratch, stride, kAbsoluteReturnSquaredPrefixV2);
  double* signed_volume_prefix =
      prefix_series_v2(prefix_scratch, stride, kSignedVolumePrefixV2);
  double* product_prefix = prefix_series_v2(
      prefix_scratch, stride, kVolumeAbsoluteReturnProductPrefixV2);

  volume_prefix[0] = 0.0;
  volume_squared_prefix[0] = 0.0;
  range_prefix[0] = 0.0;
  range_squared_prefix[0] = 0.0;
  absolute_return_prefix[0] = 0.0;
  absolute_return_squared_prefix[0] = 0.0;
  signed_volume_prefix[0] = 0.0;
  product_prefix[0] = 0.0;

  double volume_sum = 0.0;
  double volume_squared_sum = 0.0;
  double range_sum = 0.0;
  double range_squared_sum = 0.0;
  double absolute_return_sum = 0.0;
  double absolute_return_squared_sum = 0.0;
  double signed_volume_sum = 0.0;
  double product_sum = 0.0;
  for (std::size_t row = 0; row < rows; ++row) {
    const double row_volume = volume[row];
    const double row_range = abs_v2(high[row] - low[row]);
    const double absolute_return =
        row == 0 ? 0.0 : abs_v2(close[row] - close[row - 1]);
    const double bar_change = close[row] - open[row];
    const double direction =
        bar_change > 0.0 ? 1.0 : (bar_change < 0.0 ? -1.0 : 0.0);
    const double signed_volume = row_volume * direction;
    const double product = row_volume * absolute_return;

    volume_sum = volume_sum + row_volume;
    volume_squared_sum = volume_squared_sum + row_volume * row_volume;
    range_sum = range_sum + row_range;
    range_squared_sum = range_squared_sum + row_range * row_range;
    absolute_return_sum = absolute_return_sum + absolute_return;
    absolute_return_squared_sum =
        absolute_return_squared_sum + absolute_return * absolute_return;
    signed_volume_sum = signed_volume_sum + signed_volume;
    product_sum = product_sum + product;

    const std::size_t end = row + 1;
    volume_prefix[end] = volume_sum;
    volume_squared_prefix[end] = volume_squared_sum;
    range_prefix[end] = range_sum;
    range_squared_prefix[end] = range_squared_sum;
    absolute_return_prefix[end] = absolute_return_sum;
    absolute_return_squared_prefix[end] = absolute_return_squared_sum;
    signed_volume_prefix[end] = signed_volume_sum;
    product_prefix[end] = product_sum;
  }
}

struct MeanStdV2 {
  double mean;
  double standard_deviation;
};

__device__ __forceinline__ MeanStdV2 mean_std_v2(
    std::size_t row, std::size_t window, const double* prefix,
    const double* squared_prefix) {
  const std::size_t end = row + 1;
  const std::size_t start = end > window ? end - window : 0;
  const double count = static_cast<double>(end - start);
  const double sum = prefix[end] - prefix[start];
  const double squared_sum = squared_prefix[end] - squared_prefix[start];
  const double mean = sum / count;
  const double variance = max_v2(squared_sum / count - mean * mean, 0.0);
  return MeanStdV2{mean, sqrt(variance)};
}

__device__ __forceinline__ double z_score_v2(double value, double mean,
                                              double standard_deviation) {
  return standard_deviation > kEpsilonV2
             ? clamp_v2((value - mean) / standard_deviation, -6.0, 6.0)
             : 0.0;
}

__device__ __forceinline__ void write_footprint_cell_v2(
    std::size_t row, std::size_t rows, FootprintColumnV2 column, double value,
    FootprintValidityV2 validity, double* feature_values,
    unsigned char* feature_validity_u8) {
  const std::size_t cell = static_cast<std::size_t>(column) * rows + row;
  feature_validity_u8[cell] = static_cast<unsigned char>(validity);
  feature_values[cell] = validity == kValidV2 ? value : canonical_nan_v2();
}

__global__ void resident_footprint_features_v2(
    const double* open, const double* high, const double* low,
    const double* close, const double* volume,
    const std::int64_t* timestamps_ms, std::size_t rows,
    const double* prefix_scratch, double* feature_values,
    unsigned char* feature_validity_u8) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
  if (row >= rows) {
    return;
  }
  const std::size_t stride = rows + 1;
  const double* volume_prefix =
      prefix_series_v2(prefix_scratch, stride, kVolumePrefixV2);
  const double* volume_squared_prefix =
      prefix_series_v2(prefix_scratch, stride, kVolumeSquaredPrefixV2);
  const double* range_prefix =
      prefix_series_v2(prefix_scratch, stride, kRangePrefixV2);
  const double* range_squared_prefix =
      prefix_series_v2(prefix_scratch, stride, kRangeSquaredPrefixV2);
  const double* absolute_return_prefix =
      prefix_series_v2(prefix_scratch, stride, kAbsoluteReturnPrefixV2);
  const double* absolute_return_squared_prefix = prefix_series_v2(
      prefix_scratch, stride, kAbsoluteReturnSquaredPrefixV2);
  const double* signed_volume_prefix =
      prefix_series_v2(prefix_scratch, stride, kSignedVolumePrefixV2);
  const double* product_prefix = prefix_series_v2(
      prefix_scratch, stride, kVolumeAbsoluteReturnProductPrefixV2);

  const MeanStdV2 volume_stats = mean_std_v2(
      row, kRollingWindowV2, volume_prefix, volume_squared_prefix);
  const MeanStdV2 range_stats =
      mean_std_v2(row, kRollingWindowV2, range_prefix, range_squared_prefix);
  const MeanStdV2 absolute_return_stats = mean_std_v2(
      row, kRollingWindowV2, absolute_return_prefix,
      absolute_return_squared_prefix);
  const double row_range = abs_v2(high[row] - low[row]);
  const double absolute_return =
      row == 0 ? 0.0 : abs_v2(close[row] - close[row - 1]);
  const double volume_z = z_score_v2(
      volume[row], volume_stats.mean, volume_stats.standard_deviation);
  const double range_z = z_score_v2(
      row_range, range_stats.mean, range_stats.standard_deviation);
  const double absolute_return_z =
      z_score_v2(absolute_return, absolute_return_stats.mean,
                 absolute_return_stats.standard_deviation);

  const FootprintValidityV2 volume_z_validity =
      row == 0 ? kWarmupV2
               : (volume_stats.standard_deviation > kEpsilonV2
                      ? kValidV2
                      : kZeroDenominatorV2);
  const FootprintValidityV2 absorption_validity =
      row == 0
          ? kWarmupV2
          : (volume_stats.standard_deviation > kEpsilonV2 &&
                     range_stats.standard_deviation > kEpsilonV2
                 ? kValidV2
                 : kZeroDenominatorV2);
  const FootprintValidityV2 effort_validity =
      row == 0
          ? kWarmupV2
          : (volume_stats.standard_deviation > kEpsilonV2 &&
                     absolute_return_stats.standard_deviation > kEpsilonV2
                 ? kValidV2
                 : kZeroDenominatorV2);
  const double absorption =
      volume_z > 0.0 && range_z < 0.0 ? volume_z * (-range_z) : 0.0;
  const double effort_result = volume_z - absolute_return_z;
  // Rust f64::signum returns +/-1 for +/-0; copysign preserves that exact
  // semantic-v2 boundary while signed volume above keeps doji bars at zero.
  const double bar_sign = copysign(1.0, close[row] - open[row]);
  const double climax = volume_z > 0.0 && range_z > 0.0
                            ? volume_z * range_z * bar_sign
                            : 0.0;
  write_footprint_cell_v2(row, rows, kVolumeZV2, volume_z,
                          volume_z_validity, feature_values,
                          feature_validity_u8);
  write_footprint_cell_v2(row, rows, kAbsorptionV2, absorption,
                          absorption_validity, feature_values,
                          feature_validity_u8);
  write_footprint_cell_v2(row, rows, kEffortResultDivergenceV2,
                          effort_result, effort_validity, feature_values,
                          feature_validity_u8);
  write_footprint_cell_v2(row, rows, kClimaxV2, climax,
                          absorption_validity, feature_values,
                          feature_validity_u8);

  const std::size_t end = row + 1;
  const std::size_t delta_start =
      end > kDeltaWindowV2 ? end - kDeltaWindowV2 : 0;
  const double rolling_volume =
      volume_prefix[end] - volume_prefix[delta_start];
  const FootprintValidityV2 delta_validity =
      rolling_volume > kEpsilonV2 ? kValidV2 : kZeroDenominatorV2;
  const double delta =
      delta_validity == kValidV2
          ? clamp_v2((signed_volume_prefix[end] -
                      signed_volume_prefix[delta_start]) /
                         rolling_volume,
                     -1.0, 1.0)
          : 0.0;
  write_footprint_cell_v2(row, rows, kDeltaProxyV2, delta, delta_validity,
                          feature_values, feature_validity_u8);

  const std::size_t correlation_start =
      end > kCorrelationWindowV2 ? end - kCorrelationWindowV2 : 0;
  const double correlation_count =
      static_cast<double>(end - correlation_start);
  FootprintValidityV2 correlation_validity = kWarmupV2;
  double correlation = 0.0;
  if (correlation_count >= 8.0) {
    const double expected_product =
        (product_prefix[end] - product_prefix[correlation_start]) /
        correlation_count;
    const double expected_volume =
        (volume_prefix[end] - volume_prefix[correlation_start]) /
        correlation_count;
    const double expected_absolute_return =
        (absolute_return_prefix[end] -
         absolute_return_prefix[correlation_start]) /
        correlation_count;
    const double volume_variance =
        max_v2((volume_squared_prefix[end] -
                volume_squared_prefix[correlation_start]) /
                       correlation_count -
                   expected_volume * expected_volume,
               0.0);
    const double absolute_return_variance =
        max_v2((absolute_return_squared_prefix[end] -
                absolute_return_squared_prefix[correlation_start]) /
                       correlation_count -
                   expected_absolute_return * expected_absolute_return,
               0.0);
    const double denominator =
        sqrt(volume_variance * absolute_return_variance);
    if (denominator > kEpsilonV2) {
      correlation_validity = kValidV2;
      correlation =
          clamp_v2((expected_product -
                    expected_volume * expected_absolute_return) /
                       denominator,
                   -1.0, 1.0);
    } else {
      correlation_validity = kZeroDenominatorV2;
    }
  }
  write_footprint_cell_v2(row, rows, kVolumePriceCorrelationV2,
                          correlation, correlation_validity, feature_values,
                          feature_validity_u8);

  const std::int64_t minute_of_day =
      (timestamps_ms[row] / 60000LL) % 1440LL;
  const bool winter_fix = minute_of_day >= 945 && minute_of_day <= 975;
  const bool summer_fix = minute_of_day >= 885 && minute_of_day <= 915;
  write_footprint_cell_v2(row, rows, kFixWindowV2,
                          winter_fix || summer_fix ? 1.0 : 0.0, kValidV2,
                          feature_values, feature_validity_u8);
}

}  // namespace

extern "C" int neoethos_resident_footprint_f64_v2(
    const double* open, const double* high, const double* low,
    const double* close, const double* volume,
    const std::int64_t* timestamps_ms, std::size_t rows,
    double* feature_values, unsigned char* feature_validity_u8,
    double* prefix_scratch, cudaStream_t stream) {
  if (open == nullptr || high == nullptr || low == nullptr || close == nullptr ||
      volume == nullptr || timestamps_ms == nullptr || feature_values == nullptr ||
      feature_validity_u8 == nullptr || prefix_scratch == nullptr ||
      stream == nullptr || rows == 0 || rows >= (1ULL << 53U) ||
      rows > std::numeric_limits<std::size_t>::max() /
                 static_cast<std::size_t>(kFootprintColumnsV2) ||
      rows == std::numeric_limits<std::size_t>::max() ||
      rows + 1 > std::numeric_limits<std::size_t>::max() /
                     static_cast<std::size_t>(kPrefixSeriesV2)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  resident_footprint_prefix_v2<<<1, 1, 0, stream>>>(
      open, high, low, close, volume, rows, prefix_scratch);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  constexpr unsigned int kThreads = 256U;
  const std::size_t block_count =
      (rows + static_cast<std::size_t>(kThreads) - 1U) /
      static_cast<std::size_t>(kThreads);
  if (block_count > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidConfiguration);
  }
  resident_footprint_features_v2<<<static_cast<unsigned int>(block_count),
                                    kThreads, 0, stream>>>(
      open, high, low, close, volume, timestamps_ms, rows, prefix_scratch,
      feature_values, feature_validity_u8);
  return static_cast<int>(cudaGetLastError());
}
