#pragma once

#include "neoethos_gpu_cuda.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace neoethos::resident_generation_v1 {

constexpr std::uint32_t NEO_RESIDENT_GENERATION_ABI_V1 = 1;
constexpr std::uint32_t NEO_RESIDENT_PARENT_RANK_WEIGHTED_V1 = 1;
constexpr std::uint32_t NEO_RESIDENT_SURVIVOR_RANK_WEIGHTED_V1 = 1;

constexpr std::int32_t NEO_RESIDENT_STATUS_OK_V1 = 0;
constexpr std::int32_t NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1 = -1;
constexpr std::int32_t NEO_RESIDENT_STATUS_ABI_MISMATCH_V1 = -2;
constexpr std::int32_t NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1 = -3;
constexpr std::int32_t NEO_RESIDENT_STATUS_UNSUPPORTED_SELECTION_V1 = -4;
constexpr std::int32_t NEO_RESIDENT_STATUS_ARITHMETIC_OVERFLOW_V1 = -5;
constexpr std::int32_t NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1 = -6;
constexpr std::int32_t NEO_RESIDENT_STATUS_CUDA_ERROR_V1 = -7;
constexpr std::int32_t NEO_RESIDENT_STATUS_CUB_ERROR_V1 = -8;
constexpr std::int32_t NEO_RESIDENT_STATUS_STATE_ERROR_V1 = -9;
constexpr std::int32_t NEO_RESIDENT_STATUS_RANGE_ERROR_V1 = -10;
constexpr std::int32_t NEO_RESIDENT_STATUS_CONTENT_COLLISION_V1 = -11;
constexpr std::int32_t NEO_RESIDENT_STATUS_DEVICE_FAULT_V1 = -12;
constexpr std::int32_t NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 = -13;
constexpr std::int32_t NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2 = -14;

enum class NeoResidentPhiloxOperatorV1 : std::uint32_t {
  InitializeTermCount = 1,
  InitializeIndicator = 2,
  InitializeWeightLevel = 3,
  InitializeWeightSign = 4,
  InitializeThreshold = 5,
  InitializeStopGeometry = 6,
  InitializeSmcFlag = 7,
  ParentA = 8,
  ParentB = 9,
  CrossoverScalar = 10,
  MutationKind = 11,
  MutationValue = 12,
  MutationSmc = 13,
  Survivor = 14,
};

/// Private bridge output. Only the population-session translation unit may
/// mint it; callers outside gpu-cuda never receive a CUDA stream or event.
struct NeoResidentGenerationPopulationSessionImportV1 {
  std::uint32_t abi_version;
  std::uint32_t selected_cuda_ordinal;
  cudaStream_t admitted_run_stream;
  cudaEvent_t resident_parent_ready_event;
  cudaEvent_t generation_ready_event;
  void* population_lifetime_owner;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t resident_input_content_sha256[32];
};

/// Fixed scalar portion of one normalized gene. Terms live in two fixed-stride
/// arrays indexed by `candidate * max_terms_per_gene + term`.
struct NeoResidentGenerationGeneScalarV1 {
  std::uint64_t gene_identity;
  std::uint64_t content_hash;
  std::uint32_t term_count;
  std::uint32_t smc_flags;
  double long_threshold;
  double short_threshold;
  double target_pips;
  double stop_pips;
  double stop_vol_multiplier;
  std::uint32_t generation;
  std::uint32_t reserved;
};

/// Exact metrics-only row emitted by the resident population reducer. V1 uses
/// it solely to bind candidate/scenario identity and final content; ranking is
/// driven by the separately sealed integer decision-key store below.
using NeoResidentGenerationMetricRowV1 = ::NeoPopulationMetricRow;

/// All f64 inputs are transmitted as raw bits in the sealed metadata plan.
/// Device code converts them with bit-casts, preventing locale or text parsing
/// from changing the search space.
struct NeoResidentGenerationPlanV1 {
  std::uint32_t abi_version;
  std::uint32_t parent_selection_policy;
  std::uint32_t survivor_selection_policy;
  std::uint32_t max_terms_per_gene;
  std::uint32_t minimum_terms_per_gene;
  std::uint32_t threshold_level_count;
  std::uint32_t smc_flag_count;
  std::uint32_t reserved;
  std::uint64_t logical_population_count;
  std::uint64_t retained_evaluation_capacity;
  std::uint64_t feature_count;
  std::uint64_t generation_count;
  std::uint64_t survivor_count;
  std::uint64_t immigrant_count;
  std::uint64_t search_seed;
  std::uint64_t mutation_intensity_q32;
  std::uint64_t threshold_ladder_bits[6];
  std::uint64_t stop_bounds_bits[6];
  std::uint64_t smc_probability_q32[11];
  std::uint8_t generation_semantics_sha256[32];
  std::uint8_t run_identity_sha256[32];
  std::uint8_t strategy_gene_schema_sha256[32];
  std::uint8_t rank_semantics_sha256[32];
  std::uint8_t metric_semantics_sha256[32];
  std::uint8_t scoring_semantics_sha256[32];
  std::uint8_t novelty_semantics_sha256[32];
  std::uint8_t scenario_order_semantics_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t rng_mapping_sha256[32];
  std::uint8_t plan_identity_sha256[32];
};

struct NeoResidentGenerationAllocationReceiptV1 {
  std::uint32_t abi_version;
  std::uint32_t generation_store_allocation_count;
  std::uint64_t logical_gene_scalar_bytes;
  std::uint64_t logical_gene_index_bytes;
  std::uint64_t logical_gene_weight_bytes;
  std::uint64_t offspring_bytes;
  std::uint64_t metric_row_bytes;
  std::uint64_t rank_key_bytes;
  std::uint64_t selection_bytes;
  std::uint64_t dedup_hash_bytes;
  std::uint64_t cub_scratch_bytes;
  std::uint64_t retained_evaluation_workspace_bytes;
  std::uint64_t terminal_device_receipt_bytes;
  std::uint64_t total_device_bytes;
  std::uint64_t same_context_free_bytes;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint64_t logical_population_count;
  std::uint64_t retained_evaluation_capacity;
  std::uint64_t generation_chunk_count;
  std::uint8_t allocation_plan_sha256[32];
};

/// Metrics already produced by the admitted population session. Every pointer
/// and event is native-private; Rust receives only the owning opaque wrapper.
struct NeoResidentGenerationMetricRowsImportV1 {
  std::uint32_t abi_version;
  std::uint32_t metric_value_count;
  const NeoResidentGenerationMetricRowV1* metric_rows_device;
  const std::uint64_t* resident_decision_keys_device;
  const std::uint64_t* expected_scenario_ids_device;
  std::uint64_t logical_offset;
  std::uint64_t active_scenarios;
  cudaEvent_t scoring_novelty_ready_event;
  std::uint8_t metric_semantics_sha256[32];
  std::uint8_t scoring_semantics_sha256[32];
  std::uint8_t novelty_semantics_sha256[32];
  std::uint8_t scenario_order_semantics_sha256[32];
  std::uint8_t rank_semantics_sha256[32];
};

struct NeoResidentGenerationReadyEventV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t event_id;
  std::uint64_t generation_index;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t intermediate_host_wait_count;
  std::uint64_t intermediate_readback_count;
};

/// Device-resident identities. The values are handles into the owning native
/// run, not detached pointers and not host copies of the content digests.
struct NeoResidentGenerationContentReceiptV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t gene_content_identity_handle;
  std::uint64_t metric_content_identity_handle;
  std::uint64_t generation_receipt_identity_handle;
  std::uint64_t ready_event_id;
  std::uint64_t final_compact_readback_count;
};

/// In-place ownership receipt for the A1 generation-to-post-GA bridge. It
/// carries no device pointer, stream, event or detached native-run handle.
/// Zero additional allocation is a hard V1 boundary: post-GA kernels remain
/// blocked until the full workspace plan charges their resident buffers.
struct NeoResidentGenerationPostGaInPlaceReceiptV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t ready_event_id;
  std::uint64_t current_generation_index;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t logical_population_count;
  std::uint64_t retained_evaluation_capacity;
  std::uint64_t generation_allocation_total_device_bytes;
  std::uint64_t additional_allocation_count;
  std::uint64_t additional_device_bytes;
  std::uint64_t gene_content_identity_handle;
  std::uint64_t metric_content_identity_handle;
  std::uint64_t generation_receipt_identity_handle;
};

static_assert(sizeof(void*) == 8, "resident generation V1 requires a 64-bit host ABI");
static_assert(sizeof(NeoResidentGenerationPopulationSessionImportV1) == 208,
              "population import ABI changed");
static_assert(sizeof(NeoResidentGenerationGeneScalarV1) == 72,
              "fixed gene scalar ABI changed");
static_assert(sizeof(NeoResidentGenerationMetricRowV1) == 104,
              "metric row ABI changed");
static_assert(sizeof(NeoResidentGenerationPlanV1) == 632,
              "generation plan ABI changed");
static_assert(sizeof(NeoResidentGenerationAllocationReceiptV1) == 176,
              "allocation receipt ABI changed");
static_assert(sizeof(NeoResidentGenerationMetricRowsImportV1) == 216,
              "metric import ABI changed");
static_assert(sizeof(NeoResidentGenerationReadyEventV1) == 48,
              "ready event ABI changed");
static_assert(sizeof(NeoResidentGenerationContentReceiptV1) == 48,
              "content receipt ABI changed");
static_assert(sizeof(NeoResidentGenerationPostGaInPlaceReceiptV1) == 96,
              "post-GA in-place receipt ABI changed");

struct NeoResidentGenerationRunV1;

extern "C" {

std::int32_t query_resident_generation_allocation_v1(
    const NeoResidentGenerationPopulationSessionImportV1* import,
    const NeoResidentGenerationPlanV1* plan,
    NeoResidentGenerationAllocationReceiptV1* receipt);

/// Computes the exact generation layout against the single free-memory
/// snapshot owned by the composite Search admission. It performs no allocation,
/// event creation, kernel launch or additional cudaMemGetInfo call.
std::int32_t calculate_resident_generation_allocation_v2(
    const NeoResidentGenerationPlanV1* plan,
    cudaStream_t admitted_run_stream,
    std::uint64_t same_context_free_bytes,
    std::uint64_t full_discovery_reserve_bytes,
    NeoResidentGenerationAllocationReceiptV1* receipt);

std::int32_t create_resident_generation_run_from_import_v1(
    const NeoResidentGenerationPopulationSessionImportV1* import,
    const NeoResidentGenerationPlanV1* plan,
    const NeoResidentGenerationAllocationReceiptV1* receipt,
    NeoResidentGenerationRunV1** run);

std::int32_t initialize_resident_generation_population_v1(
    NeoResidentGenerationRunV1* run,
    NeoResidentGenerationReadyEventV1* ready);

std::int32_t enqueue_exact_generation_chunk_v1(
    NeoResidentGenerationRunV1* run,
    const NeoResidentGenerationMetricRowsImportV1* metrics,
    NeoResidentGenerationReadyEventV1* ready);

std::int32_t enqueue_resident_rank_selection_offspring_v1(
    NeoResidentGenerationRunV1* run,
    std::uint64_t generation_index,
    NeoResidentGenerationReadyEventV1* ready);

std::int32_t seal_resident_generation_content_v1(
    NeoResidentGenerationRunV1* run,
    NeoResidentGenerationContentReceiptV1* receipt,
    NeoResidentGenerationReadyEventV1* ready);

std::int32_t begin_resident_post_ga_in_place_v1(
    NeoResidentGenerationRunV1* run,
    const NeoResidentGenerationReadyEventV1* dependency,
    std::uint64_t gene_content_identity_handle,
    std::uint64_t metric_content_identity_handle,
    std::uint64_t generation_receipt_identity_handle,
    NeoResidentGenerationPostGaInPlaceReceiptV1* receipt);

std::int32_t enqueue_resident_generation_release_v1(
    NeoResidentGenerationRunV1* run);
std::int32_t detach_resident_search_terminal_receipt_v2(
    NeoResidentGenerationRunV1* run,
    const void* expected_terminal_host_receipt);

}  // extern "C"

}  // namespace neoethos::resident_generation_v1
