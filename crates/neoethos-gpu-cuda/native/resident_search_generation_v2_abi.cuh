#pragma once

#include "resident_generation_v2_abi.cuh"
#include "resident_scoring_novelty_v1_abi.cuh"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace neoethos::resident_search_generation_v2 {

constexpr std::uint32_t NEO_RESIDENT_SEARCH_GENERATION_ABI_V2 = 2;

/// Exact process-lifetime CUDA facts reserved before the Search plans are
/// sealed. Raw handles never leave gpu-cuda; the public owner exposes only a
/// validated summary and persistent evidence uses UUID plus run ordinal.
struct NeoResidentSearchRuntimeFactsV2 {
  std::uint32_t abi_version;
  std::uint32_t selected_cuda_ordinal;
  std::uint64_t run_admission_ordinal;
  std::uint8_t device_uuid[16];
  std::uint32_t compute_capability_major;
  std::uint32_t compute_capability_minor;
  std::uint64_t primary_context_id;
  std::uint64_t run_stream_id;
  CUcontext admitted_primary_context;
  cudaStream_t admitted_run_stream;
  cudaMemPool_t admitted_memory_pool;
  std::uint32_t pool_location_type;
  std::int32_t pool_location_id;
  std::uint32_t pool_allocation_type;
  std::uint32_t pool_handle_types;
  std::uint32_t active_pool_is_default;
  std::uint32_t reserved;
  std::uint64_t pool_reserved_current_bytes;
  std::uint64_t pool_used_current_bytes;
  std::uint64_t allocator_context_reserve_bytes;
  std::uint8_t run_stream_process_token[32];
};

/// Single-snapshot, pre-allocation authority for the bounded generation and
/// scoring stores plus the compact pinned terminal receipt.
struct NeoResidentSearchCombinedAdmissionV2 {
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint32_t free_memory_snapshot_count;
  std::uint32_t generation_allocation_count;
  std::uint32_t scoring_allocation_count;
  std::uint32_t terminal_host_allocation_count;
  std::uint64_t terminal_host_receipt_bytes;
  std::uint64_t same_context_free_bytes;
  std::uint64_t same_context_total_bytes;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint64_t generation_device_bytes;
  std::uint64_t scoring_device_bytes;
  std::uint64_t total_device_bytes;
  std::uint64_t pool_reserved_current_bytes;
  std::uint64_t pool_used_current_bytes;
  NeoResidentSearchRuntimeFactsV2 runtime;
  resident_generation_v1::NeoResidentGenerationAllocationReceiptV1 generation;
  resident_scoring_novelty_v1::NeoResidentScoringNoveltyAllocationReceiptV1 scoring;
  std::uint8_t receipt_identity_sha256[32];
};

static_assert(sizeof(NeoResidentSearchRuntimeFactsV2) == 160,
              "resident Search runtime facts V2 ABI changed");
static_assert(sizeof(NeoResidentSearchCombinedAdmissionV2) == 592,
              "resident Search combined admission V2 ABI changed");

/// Private one-shot source minted only by the exact boxed population receipt.
/// The session owner is retained separately so later native stages can prove
/// the same population lifetime without treating the receipt address as it.
struct NeoResidentScoringPopulationSourceV2 {
  std::uint32_t abi_version;
  std::uint32_t selected_cuda_ordinal;
  cudaStream_t admitted_run_stream;
  cudaEvent_t metrics_ready_event;
  cudaEvent_t scoring_ready_event;
  const void* receipt_token;
  void* population_lifetime_owner;
  const resident_scoring_novelty_v1::NeoResidentScoringNoveltyMetricRowV1*
      metric_rows_device;
  const std::uint64_t* expected_scenario_ids_device;
  std::uint64_t logical_population_count;
  std::uint64_t feature_count;
  std::uint32_t max_terms_per_gene;
  std::uint32_t reserved;
  std::uint64_t full_discovery_reserve_bytes;
};

static_assert(sizeof(NeoResidentScoringPopulationSourceV2) == 96,
              "resident Search scoring source V2 ABI changed");
static_assert(alignof(NeoResidentScoringPopulationSourceV2) == 8,
              "resident Search scoring source V2 alignment changed");
static_assert(
    offsetof(NeoResidentScoringPopulationSourceV2,
             population_lifetime_owner) == 40,
    "resident Search scoring source V2 lifetime owner offset changed");
static_assert(
    offsetof(NeoResidentScoringPopulationSourceV2,
             full_discovery_reserve_bytes) == 88,
    "resident Search scoring source V2 reserve offset changed");

extern "C" std::int32_t enqueue_full_population_scored_generation_advance_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* scoring,
    const NeoResidentScoringPopulationSourceV2* population,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1* dependency,
    resident_generation_v2::NeoResidentSearchAdvancePendingReceiptV2* pending);

extern "C" std::int32_t neoethos_gpu_cuda_population_reserve_resident_search_runtime_v2(
    void* session, NeoResidentSearchRuntimeFactsV2* facts);

extern "C" std::int32_t neoethos_gpu_cuda_population_query_resident_search_combined_v2(
    void* session,
    const resident_generation_v1::NeoResidentGenerationPlanV1* generation_plan,
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* scoring_plan,
    const NeoResidentSearchRuntimeFactsV2* expected_runtime,
    NeoResidentSearchCombinedAdmissionV2* admission);

extern "C" std::int32_t neoethos_gpu_cuda_population_create_resident_search_combined_v2(
    void* session,
    const resident_generation_v1::NeoResidentGenerationPlanV1* generation_plan,
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* scoring_plan,
    const NeoResidentSearchCombinedAdmissionV2* admission,
    resident_generation_v1::NeoResidentGenerationRunV1** generation,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1** scoring);

#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
struct NeoResidentGenerationAdvanceFixtureSnapshotV2 {
  std::uint32_t abi_version;
  std::uint32_t device_content_fault;
  std::uint32_t gene_hash_collision_fault;
  std::uint32_t control_fault_word;
  std::uint32_t stop_requested;
  std::uint32_t current_store_index;
  std::uint32_t max_terms_per_gene;
  std::uint32_t survivor_count;
  std::uint32_t selected_count;
  std::uint32_t dedup_run_count;
  std::uint32_t reserved;
  std::uint64_t logical_population_count;
  std::uint64_t generation_index;
  std::uint64_t store_epoch;
  std::uint64_t terminal_synchronization_count;
  std::uint64_t terminal_readback_count;
  std::uint64_t terminal_readback_bytes;
};
static_assert(sizeof(NeoResidentGenerationAdvanceFixtureSnapshotV2) == 96);

extern "C" std::int32_t fixture_set_resident_generation_gene_identity_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    std::uint64_t candidate, std::uint64_t gene_identity);
extern "C" std::int32_t fixture_set_duplicate_final_gene_content_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    std::uint64_t source_candidate, std::uint64_t destination_candidate);
extern "C" std::int32_t fixture_copy_resident_generation_advance_snapshot_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    std::uint64_t* ranked_population_ordinals_host,
    resident_generation_v1::NeoResidentGenerationGeneScalarV1* initial_genes_host,
    resident_generation_v1::NeoResidentGenerationGeneScalarV1* final_genes_host,
    std::uint64_t* initial_term_indices_host,
    double* initial_term_weights_host,
    std::uint64_t* final_term_indices_host,
    double* final_term_weights_host,
    std::uint64_t* parent_a_host, std::uint64_t* parent_b_host,
    std::uint64_t* selected_survivors_host,
    std::uint8_t* sorted_dedup_flags_host,
    std::uint8_t* candidate_valid_flags_host,
    std::uint64_t population_capacity, std::uint64_t term_capacity,
    std::uint64_t survivor_capacity,
    NeoResidentGenerationAdvanceFixtureSnapshotV2* snapshot);
#endif

}  // namespace neoethos::resident_search_generation_v2
