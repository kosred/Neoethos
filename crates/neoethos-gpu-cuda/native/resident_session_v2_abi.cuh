#pragma once

#include <cstddef>
#include <cstdint>

#define NEOETHOS_RESIDENT_SESSION_ABI_VERSION_V2 2u
#define NEOETHOS_RESIDENT_SESSION_SEMANTIC_VERSION_V2 2u
#define NEOETHOS_RESIDENT_SESSION_FEATURE_COLUMNS_V2 23u

struct CUstream_st;

/// Fixed-width launch descriptor for the atomic resident Session-v2 family.
/// Every pointer names an already-resident allocation in `stream`'s primary
/// context. Outputs are feature-major `[23][row_count]` f64/u8 matrices.
struct NeoResidentSessionLaunchV2 {
  std::uint32_t abi_version;
  std::uint32_t semantic_version;
  std::uint32_t feature_column_count;
  std::uint32_t reserved;
  std::uint64_t row_count;
  const double* open;
  const double* high;
  const double* low;
  const double* close;
  const double* volume;
  const std::int64_t* timestamps_ms;
  double* feature_values;
  std::uint8_t* feature_validity_u8;
};

static_assert(sizeof(NeoResidentSessionLaunchV2) == 88U);
static_assert(offsetof(NeoResidentSessionLaunchV2, row_count) == 16U);
static_assert(offsetof(NeoResidentSessionLaunchV2, open) == 24U);
static_assert(offsetof(NeoResidentSessionLaunchV2, timestamps_ms) == 64U);
static_assert(offsetof(NeoResidentSessionLaunchV2, feature_values) == 72U);
static_assert(offsetof(NeoResidentSessionLaunchV2, feature_validity_u8) == 80U);

extern "C" int neoethos_resident_session_f64_v2(
    const NeoResidentSessionLaunchV2* launch, CUstream_st* stream);
