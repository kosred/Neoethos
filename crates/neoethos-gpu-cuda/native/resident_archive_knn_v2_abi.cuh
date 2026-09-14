#pragma once

#include "resident_generation_v2_abi.cuh"
#include "resident_scoring_novelty_v1_abi.cuh"
#include "resident_search_generation_v2_abi.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace neoethos::resident_archive_knn_v2 {

#if defined(__HIP_PLATFORM_AMD__)
// Distinct HIP wire protocol; NVIDIA retains its original ABI value.
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 = 0x00010002u;
#else
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 = 2;
#endif
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_ARENA_REGION_COUNT_V2 = 15;
constexpr std::uint64_t NEO_RESIDENT_ARCHIVE_KNN_MAX_POPULATION_COUNT_V2 =
    2'147'483'647;
constexpr std::uint64_t NEO_RESIDENT_ARCHIVE_KNN_MAX_CAPACITY_V2 = 65'535;
// Minimum signature stride also provides four disjoint population-sized CUB
// key/value arrays. Wider vocabularies retain every feature bit.
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 = 4;
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_K_V2 = 15;
// Archive wire/storage stride; the bound active term limit may be smaller.
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 = 16;
constexpr std::uint32_t NEO_RESIDENT_ARCHIVE_KNN_METRIC_COUNT_V2 = 11;
constexpr std::uint64_t NEO_RESIDENT_ARCHIVE_KNN_NOVELTY_WEIGHT_BITS_V2 =
    0x3fc9'9999'9999'999aull;

// Zero denotes an invalid or unrepresentable signature extent. Quotient and
// remainder avoid overflowing feature_count + 63 at the ABI boundary.
constexpr std::uint32_t signature_word_count_v2(std::uint64_t feature_count) {
  const std::uint64_t words = feature_count / 64ull +
                              (feature_count % 64ull != 0ull ? 1ull : 0ull);
  return feature_count == 0ull || words > 0xffff'ffffull
             ? 0u
             : (words < NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2
                    ? NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2
                    : static_cast<std::uint32_t>(words));
}

static_assert(signature_word_count_v2(0) == 0);
static_assert(signature_word_count_v2(1) == 4);
static_assert(signature_word_count_v2(256) == 4);
static_assert(signature_word_count_v2(257) == 5);
static_assert(signature_word_count_v2(1'924) == 31);
static_assert(signature_word_count_v2(0xffff'ffffull * 64ull) == 0xffff'ffffu);
static_assert(signature_word_count_v2(0xffff'ffffull * 64ull + 1ull) == 0);
static_assert(signature_word_count_v2(~std::uint64_t{0}) == 0);

// The packed-count capacity bound proves both doubling and the next power of
// two representable. Zero reports an invalid extent, never an empty hash mask.
constexpr std::uint64_t archive_hash_table_capacity_v3(std::uint64_t capacity) {
  if (capacity == 0 || capacity > NEO_RESIDENT_ARCHIVE_KNN_MAX_CAPACITY_V2) return 0;
  std::uint64_t slots = 1;
  while (slots < 2 * capacity) slots <<= 1;
  return slots;
}
static_assert(archive_hash_table_capacity_v3(0) == 0);
static_assert(archive_hash_table_capacity_v3(1) == 2);
static_assert(archive_hash_table_capacity_v3(2) == 4);
static_assert(archive_hash_table_capacity_v3(3) == 8);
static_assert(archive_hash_table_capacity_v3(32'768) == 65'536);
static_assert(archive_hash_table_capacity_v3(32'769) == 131'072);
static_assert(archive_hash_table_capacity_v3(65'535) == 131'072);
static_assert(archive_hash_table_capacity_v3(65'536) == 0);
static_assert(archive_hash_table_capacity_v3(~std::uint64_t{0}) == 0);

constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_OK_V2 = 0;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_NOT_READY_V2 = 1;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2 = -1;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_ABI_MISMATCH_V2 = -2;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2 = -3;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2 = -4;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_RANGE_ERROR_V2 = -5;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2 = -6;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2 = -7;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_CUB_ERROR_V2 = -8;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_DEVICE_FAULT_V2 = -9;
constexpr std::int32_t NEO_ARCHIVE_KNN_STATUS_CLEANUP_ALREADY_ACKED_V2 = -10;
constexpr std::uint32_t NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2 = 1;
constexpr std::uint32_t NEO_ARCHIVE_KNN_TERMINAL_FAULT_V2 = 2;

/// A checked byte range within the one borrowed ScoringArchiveArena allocation.
struct NeoResidentArchiveKnnArenaRegionV2 {
  std::uint64_t offset_bytes;
  std::uint64_t size_bytes;
};

/// Exact layout and authority retained by the validated Rust admission owner.
///
/// The allocation base and admitted stream are derived from the retained
/// scoring/generation owners; neither can be substituted through this DTO. It
/// does not authorize an allocation, a free or a memory query. The fifteen
/// named ranges must be contiguous, 256-byte aligned and end at
/// `total_device_bytes`; the CUB range is deliberately runtime-sized.
struct NeoResidentArchiveKnnBindV2 {
  std::uint32_t abi_version;
#if defined(__HIP_PLATFORM_AMD__)
  std::uint32_t backend_kind;
#else
  std::uint32_t reserved;
#endif

  NeoResidentArchiveKnnArenaRegionV2 fitness_scores;
  NeoResidentArchiveKnnArenaRegionV2 decision_keys;
  NeoResidentArchiveKnnArenaRegionV2 cub_scratch;
  NeoResidentArchiveKnnArenaRegionV2 archive_gene_scalars;
  NeoResidentArchiveKnnArenaRegionV2 archive_term_indices;
  NeoResidentArchiveKnnArenaRegionV2 archive_term_weights;
  NeoResidentArchiveKnnArenaRegionV2 archive_metric_rows;
  NeoResidentArchiveKnnArenaRegionV2 archive_signatures;
  // [2A hashes][2A sequences][H hash slots][A heap slots][A heap positions].
  // H is archive_hash_table_capacity_v3(A); only the first 4A words are banked.
  NeoResidentArchiveKnnArenaRegionV2 archive_hashes;
  NeoResidentArchiveKnnArenaRegionV2 current_population_signatures;
  NeoResidentArchiveKnnArenaRegionV2 novelty_scores;
  NeoResidentArchiveKnnArenaRegionV2 exact_top_k_keys;
  NeoResidentArchiveKnnArenaRegionV2 admission_flags;
  NeoResidentArchiveKnnArenaRegionV2 admission_offsets;
  NeoResidentArchiveKnnArenaRegionV2 archive_control_and_seal;
  std::uint64_t total_device_bytes;

  std::uint64_t population_count;
  std::uint64_t archive_capacity;
  std::uint32_t signature_word_count;
  std::uint32_t novelty_neighbor_count;
  // Actual generation stride/active limit; archive rows remain padded to 16.
  std::uint32_t max_terms_per_gene;
  std::uint32_t reserved_extents;

  std::uint8_t device_uuid[16];
#if defined(__HIP_PLATFORM_AMD__)
  std::uint64_t hip_lease_identity;
#else
  std::uint64_t primary_context_identity;
#endif
  std::uint64_t search_stream_identity;
  std::uint64_t active_pool_identity;
#if defined(__HIP_PLATFORM_AMD__)
  std::uint64_t hip_build_identity;
#else
  std::uint64_t cuda_build_identity;
#endif
  std::uint64_t kernel_semantics_identity;
  std::uint64_t binary64_math_identity;
  std::uint64_t plan_identity;
  std::uint64_t run_identity;
  std::uint64_t full_workspace_receipt_identity;
  std::uint64_t post_trim_receipt_identity;
};

static_assert(sizeof(void*) == 8,
              "resident archive/kNN V2 requires a 64-bit ABI");
static_assert(sizeof(NeoResidentArchiveKnnArenaRegionV2) == 16,
              "resident archive/kNN arena-region ABI changed");
static_assert(sizeof(NeoResidentArchiveKnnBindV2) == 384,
              "resident archive/kNN bind ABI changed");
static_assert(
    sizeof(resident_generation_v2::NeoResidentGenerationGeneViewV2) == 136,
    "resident generation V2 gene-view ABI changed");
static_assert(
    sizeof(resident_scoring_novelty_v1::NeoResidentScoredDecisionRowsV1) ==
        312,
    "resident scoring V1 decision-row ABI changed");

/// Exact-identity pending authority for the one terminal projection. The
/// target packed word is device-derived and is authoritative only after the
/// completion event; before that boundary the pending receipt binds its exact
/// device receipt address instead of fabricating a host-side target value.
struct NeoResidentArchiveKnnPendingV2 {
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint64_t source_packed_commit_word;
  std::uint64_t terminal_device_receipt_identity;
  std::uint64_t run_identity;
  std::uint64_t boxed_receipt_identity;
  std::uint64_t staged_dependency_identity;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t completion_event_identity;
  std::uint64_t terminal_host_receipt_identity;
};

/// The terminal control D2H payload. `packed_commit_word` is the only publication
/// authority; consumers decode store, generation, archive count and epoch from
/// it after successful event proof.
struct NeoResidentArchiveKnnTerminalV2 {
  std::uint32_t abi_version;
  std::uint32_t terminal_status;
  std::uint32_t device_fault_word;
  std::uint32_t validation_fault_word;
  std::uint64_t receipt_identity;
  std::uint64_t run_identity;
  std::uint64_t packed_commit_word;
  std::uint64_t collision_count;
  std::uint64_t compact_async_d2h_count;
  std::uint64_t compact_async_d2h_bytes;
  std::uint64_t completion_event_query_count;
  std::uint64_t completion_stream_synchronize_count;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t completion_event_identity;
  std::uint64_t validator_digest;
};

static_assert(sizeof(NeoResidentArchiveKnnPendingV2) == 72,
              "resident archive/kNN pending ABI changed");
static_assert(sizeof(NeoResidentArchiveKnnTerminalV2) == 104,
              "resident archive/kNN terminal ABI changed");

/// Separate accounting for final candidate output, after terminal event proof.
/// Counts cover only occupied archive entries, never uninitialized capacity or
/// the feature matrix. This is archive output, not the final-population union.
struct NeoResidentArchiveExportReceiptV3 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t run_identity;
  std::uint64_t packed_commit_word;
  std::uint64_t candidate_count;
  std::uint64_t term_count;
  std::uint64_t feature_count;
  std::uint64_t host_copy_count;
  std::uint64_t host_copy_bytes;
};

static_assert(sizeof(NeoResidentArchiveExportReceiptV3) == 64,
              "resident archive export receipt ABI changed");

/// Host-only lifecycle wrapper. It borrows both native run owners and the
/// ScoringArchiveArena; the outer combined Search owner remains their owner.
struct NeoResidentArchiveKnnOwnerV2;

// Immutable adaptive algorithm policy, configured while Bound. The run hash
// binds these source settings; this DTO grants no allocation or execution.
struct NeoResidentArchivePolicyV3 {
  std::uint32_t abi_version;
  std::uint32_t mode;  // 0 net, 1 active, 2 profit factor, 3 Sharpe
  double minimum_net;
  double minimum_profit_factor;
  double minimum_sharpe;
  double novelty_weight;
};
static_assert(sizeof(NeoResidentArchivePolicyV3) == 40);

extern "C" {
std::int32_t configure_resident_archive_policy_v3(
    NeoResidentArchiveKnnOwnerV2* owner, std::uint64_t expected_run_identity,
    const NeoResidentArchivePolicyV3* policy);
/// Immutable per-run blend policy, supplied before the first ranking enqueue.
/// Legacy direct V2 callers retain the original 0.2 default when not configured.
std::int32_t configure_resident_archive_novelty_v3(
    NeoResidentArchiveKnnOwnerV2* owner, std::uint64_t expected_run_identity,
    double novelty_weight);

std::int32_t bind_preallocated_resident_archive_knn_v2(
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* scoring,
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const resident_generation_v2::NeoResidentGenerationGeneViewV2* genes,
    const NeoResidentArchiveKnnBindV2* binding,
    NeoResidentArchiveKnnOwnerV2** owner);

std::int32_t enqueue_resident_archive_score_and_rank_v2(
    NeoResidentArchiveKnnOwnerV2* owner,
    const resident_search_generation_v2::NeoResidentScoringPopulationSourceV2*
        population,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1*
        dependency);

std::int32_t enqueue_resident_archive_stage_from_rank_v2(
    NeoResidentArchiveKnnOwnerV2* owner);

/// Invokes the generation split helper through the retained generation owner.
/// No caller-supplied gene pointer, store index or scalar commit proof is
/// accepted at this publication boundary.
std::int32_t enqueue_resident_archive_evolve_and_publish_v2(
    NeoResidentArchiveKnnOwnerV2* owner);

/// The only transition allowed to enqueue a D2H copy and record the terminal
/// event. The output is the exact 72-byte pending identity authority.
std::int32_t enqueue_resident_archive_terminal_seal_v2(
    NeoResidentArchiveKnnOwnerV2* owner,
    NeoResidentArchiveKnnPendingV2* pending);

std::int32_t try_complete_resident_archive_terminal_v2(
    NeoResidentArchiveKnnOwnerV2* owner,
    const NeoResidentArchiveKnnPendingV2* pending,
    resident_generation_v1::NeoResidentGenerationReadyEventV1* committed_ready,
    NeoResidentArchiveKnnTerminalV2* terminal_copy);

/// One-shot, blocking terminal export into caller-owned, exactly sized host
/// buffers. No kernel, allocation or per-generation readback is introduced.
/// Success means all five D2H copies have completed. The output receipt remains
/// zero on failure and the caller must discard any partially filled buffers.
std::int32_t copy_resident_archive_terminal_candidates_v4(
    NeoResidentArchiveKnnOwnerV2* owner,
    const NeoResidentArchiveKnnTerminalV2* expected_terminal,
    resident_generation_v1::NeoResidentGenerationGeneScalarV1* scalars,
    std::uint64_t* term_indices, double* term_weights,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyMetricRowV1* metrics,
    std::uint64_t* admission_sequences,
    std::uint64_t candidate_capacity, std::uint64_t term_capacity,
    NeoResidentArchiveExportReceiptV3* receipt);

/// Detaches the borrowed state and acknowledges the native wrapper only. The
/// outer combined owner releases the two arenas and 104-byte host receipt.
std::int32_t neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2(
    void* session, NeoResidentArchiveKnnOwnerV2* owner);

}  // extern "C"

}  // namespace neoethos::resident_archive_knn_v2
