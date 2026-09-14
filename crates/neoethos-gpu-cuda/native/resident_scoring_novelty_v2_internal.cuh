#pragma once

#include "resident_scoring_novelty_v1_abi.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace neoethos::resident_archive_knn_v2 {
struct NeoResidentArchiveKnnBindV2;
}

namespace neoethos::resident_search_generation_v2 {
struct NeoResidentScoringPopulationSourceV2;
}

// Native-only composition seam. None of these declarations is part of the
// public C ABI or the Rust FFI surface.
namespace neoethos::resident_scoring_novelty_v2_internal {

enum class ResidentMetricStatusV2 : std::uint32_t {
  Finite = 0,
  EconomicMonthlyEquityReject = 1,
  Fault = 2,
};

#if defined(__CUDACC__)
// Valid only behind current sealed producer semantics. This is not a generic
// non-finite exception: NaN/+infinity and every other non-finite slot fault.
__device__ inline ResidentMetricStatusV2 classify_resident_metrics_v2(
    const double* values) {
  const double negative_infinity =
      -__longlong_as_double(static_cast<long long>(0x7ff0000000000000ULL));
  for (int metric = 0; metric < 11; ++metric) {
    if (!isfinite(values[metric]) &&
        !(metric == 1 && values[metric] == negative_infinity)) {
      return ResidentMetricStatusV2::Fault;
    }
  }
  if (values[1] == negative_infinity) {
    return values[3] >= 1.0
               ? ResidentMetricStatusV2::EconomicMonthlyEquityReject
               : ResidentMetricStatusV2::Fault;
  }
  return ResidentMetricStatusV2::Finite;
}
#endif

using resident_scoring_novelty_v1::NeoResidentScoringNoveltyDeviceSealV1;
using resident_scoring_novelty_v1::NeoResidentScoringNoveltyMetricRowV1;
using resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1;

constexpr std::uint64_t NEO_RESIDENT_SCORING_SLICE2_CONTROL_BYTES_V2 = 64;
constexpr std::uint64_t NEO_RESIDENT_SCORING_SLICE2_ARCHIVE_CONTROL_OFFSET_V2 = 64;

/// The only archive-visible view of the opaque scoring allocation. The
/// accessor validates the retained Slice2 binding before returning it.
struct ResidentScoringArenaAccessV2 {
  cudaStream_t admitted_run_stream;
  void* allocation_base;
  std::uint64_t allocation_bytes;
  std::uint64_t same_stream_enqueue_count;
};

/// Same-stream, device-resident output of checked canonical objective scoring
/// (finite fitness or authenticated economic rejection, never arithmetic fault).
/// It deliberately carries no completion event or host-readable receipt.
struct ResidentScoringFiniteObjectiveRowsV2 {
  NeoResidentScoringNoveltyRunV1* scoring_owner;
  cudaStream_t admitted_run_stream;
  const NeoResidentScoringNoveltyMetricRowV1* metric_rows_device;
  const std::uint64_t* expected_scenario_ids_device;
  double* fitness_scores_device;
  std::uint64_t* decision_keys_device;
  const NeoResidentScoringNoveltyDeviceSealV1* device_seal;
  std::uint64_t logical_population_count;
  std::uint64_t same_stream_enqueue_count;
  std::uint8_t metric_semantics_sha256[32];
  std::uint8_t scoring_semantics_sha256[32];
  std::uint8_t novelty_semantics_sha256[32];
  std::uint8_t scenario_order_semantics_sha256[32];
  std::uint8_t rank_semantics_sha256[32];
#if defined(__HIP_PLATFORM_AMD__)
  std::uint8_t hip_build_manifest_sha256[32];
#else
  std::uint8_t cuda_build_manifest_sha256[32];
#endif
#if defined(__HIP_PLATFORM_AMD__)
  std::uint8_t hip_math_flags_sha256[32];
#else
  std::uint8_t cuda_math_flags_sha256[32];
#endif
};

/// Allocation-free validation/query for the one Slice2 ScoringArchiveArena.
/// The caller supplies the already captured same-context free-memory snapshot;
/// this helper never probes CUDA and never allocates.
std::int32_t query_slice2_combined_scoring_archive_run_v2(
    const resident_scoring_novelty_v1::NeoResidentScoringAdmissionV2*
        admission,
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* plan,
    const resident_archive_knn_v2::NeoResidentArchiveKnnBindV2* binding,
    std::uint64_t same_context_free_bytes,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyAllocationReceiptV1*
        receipt);

/// Creates the one physical ScoringArchiveArena allocation. The complete
/// checked BindV2 is retained by the opaque scoring owner so a later archive
/// borrower cannot substitute a base, size, stream, or region layout.
std::int32_t create_slice2_combined_scoring_archive_run_v2(
    const resident_scoring_novelty_v1::NeoResidentScoringAdmissionV2*
        admission,
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* plan,
    const resident_archive_knn_v2::NeoResidentArchiveKnnBindV2* binding,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1** run);

std::int32_t borrow_resident_scoring_archive_arena_v2(
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* run,
    const resident_archive_knn_v2::NeoResidentArchiveKnnBindV2* binding,
    ResidentScoringArenaAccessV2* access);

/// Enqueues checked canonical objective scoring, ordered objective keys, and a
/// device seal behind the already-established same-stream parent dependency.
/// There is no wait, current-population novelty, host transfer, event
/// record/query, or synchronization in this helper.
std::int32_t enqueue_resident_scoring_finite_objective_v2(
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* run,
    const resident_search_generation_v2::NeoResidentScoringPopulationSourceV2*
        population,
    ResidentScoringFiniteObjectiveRowsV2* rows);

}  // namespace neoethos::resident_scoring_novelty_v2_internal
