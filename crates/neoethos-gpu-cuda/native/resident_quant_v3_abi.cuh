#pragma once

#include <cstddef>
#include <cstdint>

#define NEOETHOS_RESIDENT_QUANT_ABI_VERSION_V3 3u
#define NEOETHOS_RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4 4u
#define NEOETHOS_RESIDENT_QUANT_FEATURE_COLUMNS_V3 63u

struct CUstream_st;

/// Fixed-width V3 ABI descriptor for the complete Quant semantic-v4 family.
/// Every pointer names an already-resident allocation in `stream`'s primary
/// context. Outputs are feature-major `[63][row_count]` f64/u8 matrices.
struct NeoResidentQuantLaunchV3 {
  std::uint32_t abi_version;
  std::uint32_t semantic_version;
  std::uint32_t feature_column_count;
  std::uint32_t reserved;
  std::uint64_t row_count;
  std::uint64_t timeframe_millis;
  std::uint64_t bars_per_asian_session;
  std::uint64_t bars_per_utc_day;
  std::uint64_t bars_per_trading_week;
  std::uint64_t trading_sessions_per_year;
  std::uint64_t annualization_periods_per_year;
  const double* open;
  const double* high;
  const double* low;
  const double* close;
  const double* volume;
  const std::int64_t* timestamps;
  double* feature_values;
  std::uint8_t* feature_validity_u8;
};

static_assert(sizeof(NeoResidentQuantLaunchV3) == 136);
static_assert(offsetof(NeoResidentQuantLaunchV3, open) == 72);
static_assert(offsetof(NeoResidentQuantLaunchV3, feature_validity_u8) == 128);

extern "C" int neoethos_resident_quant_f64_v3(
    const NeoResidentQuantLaunchV3* launch, CUstream_st* stream);
