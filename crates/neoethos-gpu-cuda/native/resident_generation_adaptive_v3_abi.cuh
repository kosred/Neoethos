#pragma once

#include "resident_generation_v1_abi.cuh"

namespace neoethos::resident_generation_v1 {

// Additive policy for the configurable resident algorithm. V1 remains the
// explicitly legacy fixture algorithm; this does not claim CPU RNG identity.
struct NeoResidentAdaptivePolicyV3 {
  std::uint32_t abi_version;
  std::uint32_t algorithm_version;
  std::uint32_t parent_policy;  // 1 rank, 2 uniform, 3 softmax, 4 tournament
  std::uint32_t survivor_policy;  // 1 rank, 2 elitist, 3 tournament, 4 generational
  std::uint32_t tournament_size;
  std::uint32_t min_structural_smc_flags;
  std::uint32_t adaptive_stops_enabled;
  std::uint32_t reserved;
  std::uint64_t seen_capacity;
  std::uint64_t seen_retry_attempts;
  std::uint64_t seen_initial_count;
  std::uint64_t seed_template_count;
  std::uint64_t template_count;
  std::uint64_t soft_stagnation_patience;
  double survivor_fraction;
  double immigrant_fraction;
  double selection_temperature;
  double minimum_improvement;
  double gate_start;
  double gate_end;
  double gate_curve;
  double gate_stagnation_step;
  double smc_force_ratio;
  std::uint8_t run_identity_sha256[32];
  std::uint8_t policy_identity_sha256[32];
};
static_assert(sizeof(NeoResidentAdaptivePolicyV3) == 216);
static_assert(alignof(NeoResidentAdaptivePolicyV3) == 8);
static_assert(offsetof(NeoResidentAdaptivePolicyV3, survivor_fraction) == 80);
static_assert(offsetof(NeoResidentAdaptivePolicyV3, policy_identity_sha256) == 184);

struct NeoResidentAdaptiveCheckpointV3 {
  std::uint32_t abi_version;
  std::uint32_t algorithm_version;
  std::uint64_t run_identity;
  std::uint64_t evaluated_generation;
  std::uint64_t evaluated_generations;
  std::uint64_t evaluation_slots;
  std::uint64_t evaluated_gate_bits;
  std::uint64_t stagnant_generations;
  std::uint64_t best_score_bits;
  std::uint64_t survivor_count;
  std::uint64_t immigrant_count;
  std::uint64_t rescue_count;
  std::uint32_t mutation_count;
  std::uint32_t reserved;
  double mutation_intensity;
  std::uint64_t control_copy_count;
  std::uint64_t control_copy_bytes;
  std::uint64_t initial_upload_count;
  std::uint64_t initial_upload_bytes;
};
static_assert(sizeof(NeoResidentAdaptiveCheckpointV3) == 136);

extern "C" std::int32_t copy_resident_adaptive_checkpoint_v3(
    NeoResidentGenerationRunV1* run, std::uint64_t expected_run_identity,
    std::uint64_t expected_completed_generations, NeoResidentAdaptiveCheckpointV3* checkpoint);

extern "C" std::int32_t calculate_resident_generation_allocation_v3(
    const NeoResidentGenerationPlanV1* plan,
    const NeoResidentAdaptivePolicyV3* policy,
    cudaStream_t admitted_run_stream, std::uint64_t same_context_free_bytes,
    std::uint64_t full_discovery_reserve_bytes,
    NeoResidentGenerationAllocationReceiptV1* receipt);

extern "C" std::int32_t create_resident_generation_run_from_import_v3(
    const NeoResidentGenerationPopulationSessionImportV1* import,
    const NeoResidentGenerationPlanV1* plan,
    const NeoResidentAdaptivePolicyV3* policy,
    const NeoResidentGenerationAllocationReceiptV1* receipt,
    NeoResidentGenerationRunV1** run);

// One-time control upload, after admitted allocation and before initialization.
// Counts and physical extents come exclusively from the admitted policy/plan.
extern "C" std::int32_t configure_resident_generation_adaptive_inputs_v3(
    NeoResidentGenerationRunV1* run,
    const std::uint8_t expected_policy_identity_sha256[32],
    const NeoResidentGenerationGeneScalarV1* template_scalars,
    const std::uint64_t* template_indices, const double* template_weights,
    const std::uint64_t* initial_seen_hashes);

}  // namespace neoethos::resident_generation_v1
