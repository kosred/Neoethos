#include "resident_higher_timeframe_alignment_v3_abi.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint8_t kValidV3 = 0U;
constexpr std::uint8_t kStaleV3 = 4U;
constexpr std::uint8_t kNonFiniteV3 = 7U;
constexpr std::uint8_t kComputeFailureV3 = 8U;
constexpr std::uint8_t kAlignmentMissingV3 = 9U;
constexpr std::uint32_t kThreadsV3 = 256U;

__device__ __forceinline__ double canonical_nan_v3() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

__device__ __forceinline__ bool availability_at_v3(
    const NeoResidentHigherTimeframeParentSegmentV3& segment,
    std::uint64_t parent_row, std::int64_t* available_at_ms) {
  if (segment.availability_rule ==
      NEOETHOS_RESIDENT_HTF_AVAILABILITY_FIXED_V3) {
    const std::int64_t open_ms = segment.parent_open_ms[parent_row];
    if (open_ms > INT64_MAX - segment.fixed_period_ms) {
      return false;
    }
    *available_at_ms = open_ms + segment.fixed_period_ms;
    return true;
  }
  if (parent_row >= segment.parent_row_count - 1U) {
    return false;
  }
  *available_at_ms = segment.parent_open_ms[parent_row + 1U];
  return true;
}

__device__ __forceinline__ bool latest_available_parent_row_v3(
    const NeoResidentHigherTimeframeParentSegmentV3& segment,
    std::int64_t base_timestamp_ms, std::uint64_t* resolved_parent_row,
    std::int64_t* resolved_available_at_ms) {
  std::uint64_t low = 0U;
  std::uint64_t high = segment.parent_row_count;
  while (low < high) {
    const std::uint64_t parent_row = low + (high - low) / 2U;
    std::int64_t available_at_ms = 0;
    const bool evidenced = availability_at_v3(
        segment, parent_row, &available_at_ms);
    if (evidenced && available_at_ms <= base_timestamp_ms) {
      low = parent_row + 1U;
    } else {
      high = parent_row;
    }
  }
  if (low == 0U) {
    return false;
  }
  const std::uint64_t parent_row = low - 1U;
  std::int64_t available_at_ms = 0;
  if (!availability_at_v3(segment, parent_row, &available_at_ms) ||
      available_at_ms > base_timestamp_ms) {
    return false;
  }
  *resolved_parent_row = parent_row;
  *resolved_available_at_ms = available_at_ms;
  return true;
}

__global__ void resident_higher_timeframe_alignment_f64_v3(
    NeoResidentHigherTimeframeLaunchV3 launch,
    NeoResidentHigherTimeframeParentSegmentV3 segment) {
  const std::uint64_t base_row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (base_row >= launch.base_row_count) {
    return;
  }

  const std::int64_t base_timestamp_ms = launch.base_open_ms[base_row];
  std::uint64_t parent_row = 0U;
  std::int64_t available_at_ms = 0;
  const bool parent_resolved = latest_available_parent_row_v3(
      segment, base_timestamp_ms, &parent_row, &available_at_ms);
  std::uint8_t unavailable_validity = kAlignmentMissingV3;
  bool source_available = parent_resolved;
  if (parent_resolved && segment.availability_rule ==
                             NEOETHOS_RESIDENT_HTF_AVAILABILITY_FIXED_V3) {
    if (base_timestamp_ms < available_at_ms) {
      source_available = false;
      unavailable_validity = kComputeFailureV3;
    } else {
      const std::int64_t age_ms = base_timestamp_ms - available_at_ms;
      if (age_ms > segment.max_age_ms) {
        source_available = false;
        unavailable_validity = kStaleV3;
      }
    }
  }

  for (std::uint64_t local_column = 0U;
       local_column < segment.column_count; ++local_column) {
    const std::uint64_t column = segment.first_column + local_column;
    const std::uint64_t destination = column * launch.base_row_count + base_row;
    launch.feature_values[destination] = canonical_nan_v3();
    launch.feature_validity_u8[destination] = unavailable_validity;
    if (!source_available) {
      continue;
    }
    const std::uint64_t source_value_index =
        launch.source_value_offsets_device[column] + parent_row;
    const std::uint64_t source_validity_index =
        launch.source_validity_offsets_device[column] + parent_row;
    const std::uint8_t source_validity =
        launch.source_validity_buffers_device[column][source_validity_index];
    if (source_validity <= kAlignmentMissingV3) {
      launch.feature_validity_u8[destination] = source_validity;
      if (source_validity == kValidV3) {
        const double value =
            launch.source_value_buffers_device[column][source_value_index];
        if (isfinite(value)) {
          launch.feature_values[destination] = value;
        } else {
          launch.feature_validity_u8[destination] = kNonFiniteV3;
        }
      }
    } else {
      launch.feature_validity_u8[destination] = kComputeFailureV3;
    }
  }
}

}  // namespace

extern "C" std::int32_t neoethos_resident_higher_timeframe_alignment_f64_v3(
    const NeoResidentHigherTimeframeLaunchV3* launch, cudaStream_t stream) {
  if (launch == nullptr || stream == nullptr ||
      launch->abi_version != NEOETHOS_RESIDENT_HTF_ABI_VERSION_V3 ||
      launch->semantic_version !=
          NEOETHOS_RESIDENT_HTF_SEMANTIC_VERSION_V3 ||
      launch->base_row_count == 0U || launch->parent_segment_count == 0U ||
      launch->feature_column_count == 0U ||
      launch->feature_column_count > NEOETHOS_RESIDENT_HTF_MAX_BATCH_COLUMNS_V3 ||
      launch->parent_segment_count > launch->feature_column_count ||
      launch->base_open_ms == nullptr ||
      launch->source_value_buffers_device == nullptr ||
      launch->source_validity_buffers_device == nullptr ||
      launch->source_value_offsets_device == nullptr ||
      launch->source_validity_offsets_device == nullptr ||
      launch->feature_values == nullptr ||
      launch->feature_validity_u8 == nullptr ||
      launch->parent_segments_host == nullptr) {
    return -1;
  }
  if (launch->base_row_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) *
          kThreadsV3) {
    return -5;
  }

  const std::uint32_t blocks = static_cast<std::uint32_t>(
      (launch->base_row_count + kThreadsV3 - 1U) / kThreadsV3);
  std::uint32_t next_first_column = 0U;
  for (std::uint32_t segment_index = 0U;
       segment_index < launch->parent_segment_count; ++segment_index) {
    const NeoResidentHigherTimeframeParentSegmentV3 segment =
        launch->parent_segments_host[segment_index];
    if (segment.first_column != next_first_column ||
        segment.column_count == 0U || segment.parent_row_count == 0U ||
        segment.parent_open_ms == nullptr || segment.reserved != 0U ||
        segment.first_column > launch->feature_column_count ||
        segment.column_count >
            launch->feature_column_count - segment.first_column) {
      return -2;
    }
    if (segment.availability_rule ==
        NEOETHOS_RESIDENT_HTF_AVAILABILITY_FIXED_V3) {
      if (segment.fixed_period_ms <= 0) {
        return -3;
      }
      const std::int64_t expected_max_age =
          segment.fixed_period_ms >
                  std::numeric_limits<std::int64_t>::max() / 2
              ? std::numeric_limits<std::int64_t>::max()
              : segment.fixed_period_ms * 2;
      if (segment.max_age_ms != expected_max_age) {
        return -3;
      }
    } else if (segment.availability_rule ==
               NEOETHOS_RESIDENT_HTF_AVAILABILITY_NEXT_DIRECT_OPEN_V3) {
      if (segment.fixed_period_ms != 0 || segment.max_age_ms != -1) {
        return -4;
      }
    } else {
      return -4;
    }
    const dim3 grid(blocks, 1U, 1U);
    resident_higher_timeframe_alignment_f64_v3<<<grid, kThreadsV3, 0U,
                                                  stream>>>(*launch, segment);
    if (cudaGetLastError() != cudaSuccess) {
      return -6;
    }
    next_first_column += segment.column_count;
  }
  return next_first_column == launch->feature_column_count ? 0 : -2;
}
