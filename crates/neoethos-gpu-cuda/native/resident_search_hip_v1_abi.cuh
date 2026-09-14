#pragma once

#include "../hip/hip_runtime_owner_v1.h"
#include "resident_archive_knn_v2_abi.cuh"

namespace neoethos::resident_search_hip_v1 {

constexpr std::uint32_t kAbiV1 = 1;
constexpr std::uint32_t kBackendV1 = 2;

// Private typed Data handoff. These are initialized lease-scoped buffer keys,
// never raw addresses. The consumer resolves and pins every exact-sized key.
// The producer and consumer use the same owned nondefault stream, so recording
// the population ready event after binding orders it after all Data writes.
struct NeoHipResidentFeatureStoreV1 {
  std::uint32_t abi_version;
  std::uint32_t backend_kind;
  std::uint64_t lease_id;
  std::uint64_t row_count;
  std::uint32_t feature_count;
  std::uint32_t smc_slots;
  // close, high, low, bar-major features, packed-u4 validity, months, days,
  // timestamps, row-major eleven-slot SMC. No host feature/price arrays.
  std::uint64_t buffer_keys[9];
  std::uint64_t allocator_reserve_bytes;
  std::uint8_t admission_identity_sha256[32];
  std::uint8_t canonical_content_merkle[32];
  std::uint8_t run_stream_process_token[32];
};

struct NeoResidentSearchHipRuntimeFactsV1 {
  std::uint32_t abi_version;
  std::uint32_t backend_kind;
  std::uint64_t run_admission_ordinal;
  NeoHipRuntimeFactsV1 owner;
  std::uint64_t allocator_context_reserve_bytes;
  std::uint8_t run_stream_process_token[32];
};

struct NeoResidentSearchHipCombinedAdmissionV1 {
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint32_t free_memory_snapshot_count;
  std::uint32_t generation_allocation_count;
  std::uint32_t scoring_allocation_count;
  std::uint32_t terminal_host_allocation_count;
  std::uint64_t terminal_host_receipt_bytes;
  std::uint64_t same_lease_free_bytes;
  std::uint64_t same_lease_total_bytes;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint64_t generation_device_bytes;
  std::uint64_t scoring_device_bytes;
  std::uint64_t total_device_bytes;
  std::uint64_t pool_reserved_current_bytes;
  std::uint64_t pool_used_current_bytes;
  NeoResidentSearchHipRuntimeFactsV1 runtime;
  resident_generation_v1::NeoResidentGenerationAllocationReceiptV1 generation;
  resident_scoring_novelty_v1::NeoResidentScoringNoveltyAllocationReceiptV1 scoring;
  std::uint8_t receipt_identity_sha256[32];
};

static_assert(sizeof(NeoHipResidentFeatureStoreV1) == 208);
static_assert(sizeof(NeoResidentSearchHipRuntimeFactsV1) == 424);
static_assert(sizeof(NeoResidentSearchHipCombinedAdmissionV1) == 856);
static_assert(offsetof(NeoResidentSearchHipCombinedAdmissionV1, runtime) == 96);
static_assert(offsetof(NeoResidentSearchHipCombinedAdmissionV1, generation) == 520);
static_assert(offsetof(NeoResidentSearchHipCombinedAdmissionV1, receipt_identity_sha256) == 824);

#if defined(__HIP_PLATFORM_AMD__)
// Generated outside the kernel archive to avoid a self-hashing artifact. The
// digest binds the actual native archive, sources, compiler, target and flags.
extern "C" const unsigned char* neoethos_hip_native_build_manifest_sha256_v1(void);
extern "C" NeoCudaPopulationSession* neoethos_hip_population_bind_resident_feature_store_v1(
    const NeoHipResidentFeatureStoreV1* resident, std::int32_t* status);
extern "C" std::int32_t neoethos_hip_population_reserve_resident_search_runtime_v1(
    void* session, NeoResidentSearchHipRuntimeFactsV1* facts);
// A null archive binding selects the preliminary combined query; the final
// repeated Search query/create supplies the exact admitted archive binding.
extern "C" std::int32_t neoethos_hip_population_query_resident_search_v1(
    void* session,
    const resident_generation_v1::NeoResidentGenerationPlanV1* generation_plan,
    const resident_generation_v1::NeoResidentAdaptivePolicyV3* adaptive_policy,
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* scoring_plan,
    const NeoResidentSearchHipRuntimeFactsV1* runtime,
    const resident_archive_knn_v2::NeoResidentArchiveKnnBindV2* binding,
    NeoResidentSearchHipCombinedAdmissionV1* admission);
extern "C" std::int32_t neoethos_hip_population_create_resident_search_v1(
    void* session,
    const resident_generation_v1::NeoResidentGenerationPlanV1* generation_plan,
    const resident_generation_v1::NeoResidentAdaptivePolicyV3* adaptive_policy,
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* scoring_plan,
    const NeoResidentSearchHipCombinedAdmissionV1* admission,
    const resident_archive_knn_v2::NeoResidentArchiveKnnBindV2* binding,
    resident_generation_v1::NeoResidentGenerationRunV1** generation,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1** scoring);

// Native-only check reached through a validated generation lifecycle owner.
// It does not accept a raw stream registration or a caller-created lease.
bool validate_population_owner_v1(
    void* population_owner,
    const resident_archive_knn_v2::NeoResidentArchiveKnnBindV2& binding,
    cudaStream_t retained_stream);
#endif

}  // namespace neoethos::resident_search_hip_v1
