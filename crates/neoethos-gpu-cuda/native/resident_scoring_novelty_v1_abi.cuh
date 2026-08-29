#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace neoethos::resident_scoring_novelty_v1 {

constexpr std::uint32_t NEO_RESIDENT_SCORING_NOVELTY_ABI_V1 = 1;
constexpr std::uint32_t NEO_RESIDENT_SCORING_VERSION_V1 = 5;
constexpr std::uint32_t NEO_RESIDENT_SCORING_PROPFIRM_V4 = 1;
constexpr std::uint32_t NEO_RESIDENT_SCORING_RISKY_GROWTH_V5 = 2;

constexpr std::int32_t NEO_SCORING_STATUS_OK_V1 = 0;
constexpr std::int32_t NEO_SCORING_STATUS_INVALID_ARGUMENT_V1 = -1;
constexpr std::int32_t NEO_SCORING_STATUS_ABI_MISMATCH_V1 = -2;
constexpr std::int32_t NEO_SCORING_STATUS_IDENTITY_MISMATCH_V1 = -3;
constexpr std::int32_t NEO_SCORING_STATUS_ARITHMETIC_OVERFLOW_V1 = -4;
constexpr std::int32_t NEO_SCORING_STATUS_OUT_OF_MEMORY_V1 = -5;
constexpr std::int32_t NEO_SCORING_STATUS_CUDA_ERROR_V1 = -6;
constexpr std::int32_t NEO_SCORING_STATUS_CUB_ERROR_V1 = -7;
constexpr std::int32_t NEO_SCORING_STATUS_STATE_ERROR_V1 = -8;

/// Exact metrics-only row from the resident population reducer. The values
/// remain borrowed device data; no host representation is minted here.
struct NeoResidentScoringNoveltyMetricRowV1 {
  std::uint64_t candidate_id;
  std::uint64_t scenario_id;
  double values[11];
};

/// Fixed scalar portion of the normalized Generation V1 gene ABI.
struct NeoResidentScoringNoveltyGeneScalarV1 {
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

/// Private one-shot import minted by the admitted population session. Both
/// events are pre-owned by that session and retained by `population_lifetime_owner`.
struct NeoResidentScoringNoveltyPopulationImportV1 {
  std::uint32_t abi_version;
  std::uint32_t selected_cuda_ordinal;
  cudaStream_t admitted_run_stream;
  cudaEvent_t metrics_ready_event;
  cudaEvent_t scoring_novelty_ready_event;
  void* population_lifetime_owner;
  const NeoResidentScoringNoveltyMetricRowV1* metric_rows_device;
  const NeoResidentScoringNoveltyGeneScalarV1* gene_scalars_device;
  const std::uint64_t* gene_indices_device;
  const std::uint64_t* expected_scenario_ids_device;
  std::uint64_t logical_population_count;
  std::uint64_t feature_count;
  std::uint32_t max_terms_per_gene;
  std::uint32_t reserved;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  std::uint8_t metric_semantics_sha256[32];
  std::uint8_t gene_schema_sha256[32];
  std::uint8_t scenario_order_semantics_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t cuda_math_flags_sha256[32];
  std::uint8_t resident_input_content_sha256[32];
  std::uint8_t gene_content_sha256[32];
  std::uint8_t metric_content_sha256[32];
  std::uint8_t scenario_order_content_sha256[32];
};

struct NeoResidentScoringNoveltyPlanV1 {
  std::uint32_t abi_version;
  std::uint32_t scoring_objective;
  std::uint32_t scoring_version;
  std::uint32_t reserved;
  std::uint64_t logical_population_count;
  std::uint64_t feature_count;
  std::uint32_t max_terms_per_gene;
  std::uint32_t reserved_extents;
  std::uint64_t novelty_weight_bits;
  std::uint8_t metric_semantics_sha256[32];
  std::uint8_t scoring_semantics_sha256[32];
  std::uint8_t novelty_semantics_sha256[32];
  std::uint8_t scenario_order_semantics_sha256[32];
  std::uint8_t gene_schema_sha256[32];
  std::uint8_t rank_semantics_sha256[32];
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t cuda_math_flags_sha256[32];
  std::uint8_t plan_identity_sha256[32];
};

struct NeoResidentScoringNoveltyAllocationReceiptV1 {
  std::uint32_t abi_version;
  std::uint32_t scoring_store_allocation_count;
  std::uint64_t set_bitmap_bytes;
  std::uint64_t fitness_score_bytes;
  std::uint64_t novelty_score_bytes;
  std::uint64_t decision_key_bytes;
  std::uint64_t cub_scratch_bytes;
  std::uint64_t device_control_bytes;
  std::uint64_t total_device_bytes;
  std::uint64_t same_context_free_bytes;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint64_t logical_population_count;
  std::uint64_t feature_word_count;
  std::uint8_t allocation_plan_sha256[32];
};

struct NeoResidentScoringNoveltyReadyEventV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t event_id;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t intermediate_host_wait_count;
  std::uint64_t intermediate_readback_count;
};

/// Device-written validity and content identity. The next same-stream stage
/// must inspect `valid` before it can use any decision key.
struct NeoResidentScoringNoveltyDeviceSealV1 {
  std::uint32_t abi_version;
  std::uint32_t valid;
  std::uint32_t device_fault_word;
  std::uint32_t reserved;
  std::uint64_t content_lanes[4];
};

/// Private opaque handoff metadata. Pointers never cross the gpu-cuda crate.
struct NeoResidentScoredDecisionRowsV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  const NeoResidentScoringNoveltyMetricRowV1* metric_rows_device;
  const std::uint64_t* resident_decision_keys_device;
  const std::uint64_t* expected_scenario_ids_device;
  const NeoResidentScoringNoveltyDeviceSealV1* device_seal;
  cudaEvent_t scoring_novelty_ready_event;
  std::uint64_t logical_population_count;
  std::uint64_t event_id;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t intermediate_host_wait_count;
  std::uint64_t intermediate_readback_count;
  std::uint8_t metric_semantics_sha256[32];
  std::uint8_t scoring_semantics_sha256[32];
  std::uint8_t novelty_semantics_sha256[32];
  std::uint8_t scenario_order_semantics_sha256[32];
  std::uint8_t rank_semantics_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t cuda_math_flags_sha256[32];
};

static_assert(sizeof(void*) == 8, "resident scoring/novelty V1 requires 64-bit ABI");
static_assert(sizeof(NeoResidentScoringNoveltyMetricRowV1) == 104,
              "metric row ABI changed");
static_assert(sizeof(NeoResidentScoringNoveltyGeneScalarV1) == 72,
              "gene scalar ABI changed");
static_assert(sizeof(NeoResidentScoringNoveltyPopulationImportV1) == 488,
               "population import ABI changed");
static_assert(sizeof(NeoResidentScoringNoveltyPlanV1) == 432,
               "scoring plan ABI changed");
static_assert(sizeof(NeoResidentScoringNoveltyAllocationReceiptV1) == 128,
              "allocation receipt ABI changed");
static_assert(sizeof(NeoResidentScoringNoveltyReadyEventV1) == 40,
              "ready event ABI changed");
static_assert(sizeof(NeoResidentScoringNoveltyDeviceSealV1) == 48,
              "device seal ABI changed");
static_assert(sizeof(NeoResidentScoredDecisionRowsV1) == 312,
              "scored decision-row ABI changed");

struct NeoResidentScoringNoveltyRunV1;

extern "C" std::int32_t query_resident_scoring_novelty_allocation_v1(
    const NeoResidentScoringNoveltyPopulationImportV1* import,
    const NeoResidentScoringNoveltyPlanV1* plan,
    NeoResidentScoringNoveltyAllocationReceiptV1* receipt);

extern "C" std::int32_t create_resident_scoring_novelty_run_v1(
    const NeoResidentScoringNoveltyPopulationImportV1* import,
    const NeoResidentScoringNoveltyPlanV1* plan,
    const NeoResidentScoringNoveltyAllocationReceiptV1* receipt,
    NeoResidentScoringNoveltyRunV1** run);

extern "C" std::int32_t enqueue_and_seal_resident_scoring_novelty_v1(
    NeoResidentScoringNoveltyRunV1* run,
    NeoResidentScoredDecisionRowsV1* output,
    NeoResidentScoringNoveltyReadyEventV1* ready);

extern "C" std::int32_t enqueue_resident_scoring_novelty_release_v1(
    NeoResidentScoringNoveltyRunV1* run);

}  // namespace neoethos::resident_scoring_novelty_v1
