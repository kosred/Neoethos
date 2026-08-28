#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#define NEOETHOS_RESIDENT_HTF_ABI_VERSION_V3 3u
#define NEOETHOS_RESIDENT_HTF_SEMANTIC_VERSION_V3 3u
#define NEOETHOS_RESIDENT_HTF_AVAILABILITY_FIXED_V3 1u
#define NEOETHOS_RESIDENT_HTF_AVAILABILITY_NEXT_DIRECT_OPEN_V3 2u
#define NEOETHOS_RESIDENT_HTF_MAX_BATCH_COLUMNS_V3 64u

/// One contiguous parent-clock segment inside a globally batched HTF pack.
/// The array of these descriptors remains host-resident: the synchronous ABI
/// wrapper validates it and passes each descriptor by value as CUDA kernel
/// parameters. It therefore adds no resident/H2D descriptor allocation.
struct NeoResidentHigherTimeframeParentSegmentV3 {
  std::uint32_t first_column;
  std::uint32_t column_count;
  std::uint32_t availability_rule;
  std::uint32_t reserved;
  std::uint64_t parent_row_count;
  std::int64_t fixed_period_ms;
  std::int64_t max_age_ms;
  const std::int64_t* parent_open_ms;
};

/// One variable-width global alignment batch. The four pointer/offset tables
/// are device arrays with `feature_column_count` entries. Source feature data
/// remains in direct-timeframe producer allocations; outputs are feature-major
/// `[feature_column_count][base_row_count]` f64/u8 matrices. A batch may cross
/// parent boundaries, represented by contiguous host descriptors above.
struct NeoResidentHigherTimeframeLaunchV3 {
  std::uint32_t abi_version;
  std::uint32_t semantic_version;
  std::uint32_t feature_column_count;
  std::uint32_t parent_segment_count;
  std::uint64_t base_row_count;
  const std::int64_t* base_open_ms;
  const double* const* source_value_buffers_device;
  const std::uint8_t* const* source_validity_buffers_device;
  const std::uint64_t* source_value_offsets_device;
  const std::uint64_t* source_validity_offsets_device;
  double* feature_values;
  std::uint8_t* feature_validity_u8;
  const NeoResidentHigherTimeframeParentSegmentV3* parent_segments_host;
};

static_assert(sizeof(void*) == 8,
              "resident HTF-v3 requires a 64-bit CUDA ABI");
static_assert(sizeof(NeoResidentHigherTimeframeParentSegmentV3) == 48,
              "resident HTF-v3 parent-segment ABI changed");
static_assert(sizeof(NeoResidentHigherTimeframeLaunchV3) == 88,
              "resident HTF-v3 launch ABI changed");

extern "C" std::int32_t neoethos_resident_higher_timeframe_alignment_f64_v3(
    const NeoResidentHigherTimeframeLaunchV3* launch, cudaStream_t stream);
