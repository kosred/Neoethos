#include "resident_generation_v1_abi.cuh"
#include "resident_generation_v2_abi.cuh"
#include "resident_generation_v2_internal.cuh"
#include "resident_search_generation_v2_abi.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace neoethos::resident_generation_v1 {
namespace {

using resident_generation_v2::NeoResidentGenerationDeviceSealV2;
using resident_generation_v2::NeoResidentGenerationGeneViewV2;
using resident_generation_v2::NeoResidentSearchAdvancePendingReceiptV2;
using resident_generation_v2::NeoResidentSearchDeviceControlV2;
using resident_generation_v2::NeoResidentSearchTerminalReceiptV2;
using resident_generation_v2_internal::ResidentGenerationPreparedAdvanceV2;

constexpr std::uint32_t PHILOX_M0_V1 = 0xD2511F53u;
constexpr std::uint32_t PHILOX_M1_V1 = 0xCD9E8D57u;
constexpr std::uint32_t PHILOX_W0_V1 = 0x9E3779B9u;
constexpr std::uint32_t PHILOX_W1_V1 = 0xBB67AE85u;
constexpr std::uint32_t METRIC_VALUE_COUNT_V1 = 11;
constexpr std::size_t DEVICE_ALIGNMENT_V1 = 256;
constexpr std::uint64_t FNV_OFFSET_0_V1 = 14695981039346656037ull;
constexpr std::uint64_t FNV_OFFSET_1_V1 = 1099511628211ull ^ 0x9e3779b97f4a7c15ull;
constexpr std::uint64_t FNV_PRIME_V1 = 1099511628211ull;
constexpr std::uint64_t RESIDENT_CUB_FAULT_SENTINEL_KEY_V2 = 0ull;

struct Uint4V1 {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
  std::uint32_t w;
};

struct Uint2V1 {
  std::uint32_t x;
  std::uint32_t y;
};

struct GenerationPhysicalLayoutV1 {
  std::size_t logical_gene_scalar_bytes;
  std::size_t logical_gene_index_bytes;
  std::size_t logical_gene_weight_bytes;
  std::size_t offspring_bytes;
  std::size_t metric_row_bytes;
  std::size_t rank_key_bytes;
  std::size_t selection_bytes;
  std::size_t dedup_hash_bytes;
  std::size_t cub_scratch_bytes;
  std::size_t retained_evaluation_workspace_bytes;
  std::size_t terminal_device_receipt_bytes;
  std::size_t total_device_bytes;
  std::size_t generation_chunk_count;
};

struct DeviceCursorV1 {
  std::uint8_t* base;
  std::size_t offset;
  std::size_t capacity;
};

template <typename T>
bool checked_add_v1(T left, T right, T* output) {
  if (output == nullptr || right > std::numeric_limits<T>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

template <typename T>
bool checked_mul_v1(T left, T right, T* output) {
  if (output == nullptr || (left != 0 && right > std::numeric_limits<T>::max() / left)) {
    return false;
  }
  *output = left * right;
  return true;
}

bool align_device_bytes_v1(std::size_t bytes, std::size_t* aligned) {
  std::size_t value = 0;
  if (!checked_add_v1(bytes, DEVICE_ALIGNMENT_V1 - 1, &value)) {
    return false;
  }
  *aligned = value & ~(DEVICE_ALIGNMENT_V1 - 1);
  return true;
}

template <typename T>
T* take_device_array_v1(DeviceCursorV1* cursor, std::size_t count) {
  std::size_t bytes = 0;
  std::size_t end = 0;
  if (cursor == nullptr || !checked_mul_v1(count, sizeof(T), &bytes) ||
      !checked_add_v1(cursor->offset, bytes, &end) || end > cursor->capacity) {
    return nullptr;
  }
  auto* result = reinterpret_cast<T*>(cursor->base + cursor->offset);
  cursor->offset = end;
  return result;
}

bool all_identity_bytes_present_v1(const std::uint8_t identity[32]) {
  std::uint8_t aggregate = 0;
  for (std::size_t index = 0; index < 32; ++index) {
    aggregate |= identity[index];
  }
  return aggregate != 0;
}

bool identity_equal_v1(const std::uint8_t left[32], const std::uint8_t right[32]) {
  std::uint8_t difference = 0;
  for (std::size_t index = 0; index < 32; ++index) {
    difference |= left[index] ^ right[index];
  }
  return difference == 0;
}

void copy_identity_v1(std::uint8_t output[32], const std::uint8_t input[32]) {
  for (std::size_t index = 0; index < 32; ++index) {
    output[index] = input[index];
  }
}

std::int32_t cuda_status_v1(cudaError_t status) {
  return status == cudaSuccess ? NEO_RESIDENT_STATUS_OK_V1
                               : NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
}

bool validate_import_v1(const NeoResidentGenerationPopulationSessionImportV1* import) {
  return import != nullptr && import->abi_version == NEO_RESIDENT_GENERATION_ABI_V1 &&
         import->selected_cuda_ordinal != std::numeric_limits<std::uint32_t>::max() &&
         import->admitted_run_stream != nullptr && import->resident_parent_ready_event != nullptr &&
         import->generation_ready_event != nullptr &&
         import->generation_ready_event != import->resident_parent_ready_event &&
         import->population_lifetime_owner != nullptr &&
         all_identity_bytes_present_v1(import->cuda_device_identity_sha256) &&
         all_identity_bytes_present_v1(import->primary_context_identity_sha256) &&
         all_identity_bytes_present_v1(import->run_stream_identity_sha256) &&
         all_identity_bytes_present_v1(import->cuda_build_manifest_sha256) &&
         all_identity_bytes_present_v1(import->resident_input_content_sha256);
}

bool validate_plan_v1(const NeoResidentGenerationPlanV1* plan) {
  if (plan == nullptr || plan->abi_version != NEO_RESIDENT_GENERATION_ABI_V1 ||
      plan->parent_selection_policy != NEO_RESIDENT_PARENT_RANK_WEIGHTED_V1 ||
      plan->survivor_selection_policy != NEO_RESIDENT_SURVIVOR_RANK_WEIGHTED_V1 ||
      plan->logical_population_count == 0 || plan->retained_evaluation_capacity == 0 ||
      plan->retained_evaluation_capacity > plan->logical_population_count ||
      plan->feature_count == 0 || plan->generation_count == 0 ||
      plan->max_terms_per_gene == 0 || plan->minimum_terms_per_gene == 0 ||
      plan->minimum_terms_per_gene > plan->max_terms_per_gene ||
      plan->max_terms_per_gene > plan->feature_count ||
      plan->survivor_count > plan->logical_population_count ||
      plan->immigrant_count > plan->logical_population_count - plan->survivor_count ||
      plan->threshold_level_count != 6 || plan->smc_flag_count != 11 ||
      plan->logical_population_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      !all_identity_bytes_present_v1(plan->generation_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->run_identity_sha256) ||
      !all_identity_bytes_present_v1(plan->strategy_gene_schema_sha256) ||
      !all_identity_bytes_present_v1(plan->rank_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->metric_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->scoring_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->novelty_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->scenario_order_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->cuda_build_manifest_sha256) ||
      !all_identity_bytes_present_v1(plan->rng_mapping_sha256) ||
      !all_identity_bytes_present_v1(plan->plan_identity_sha256)) {
    return false;
  }
  for (std::size_t index = 0; index < 11; ++index) {
    if (plan->smc_probability_q32[index] > (1ull << 32)) {
      return false;
    }
  }
  return true;
}

std::int32_t query_cub_generation_scratch_bytes_v1(
    const NeoResidentGenerationPlanV1& plan,
    cudaStream_t stream,
    std::size_t* scratch_bytes) {
  if (scratch_bytes == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  const int count = static_cast<int>(plan.logical_population_count);
  std::size_t maximum = 0;
  std::size_t candidate = 0;
  auto* keys = static_cast<std::uint64_t*>(nullptr);
  auto* values = static_cast<std::uint64_t*>(nullptr);
  auto* flags = static_cast<std::uint8_t*>(nullptr);
  auto* run_lengths = static_cast<std::uint32_t*>(nullptr);
  auto* selected_count = static_cast<std::uint32_t*>(nullptr);

  cudaError_t status = cub::DeviceRadixSort::SortPairs(
      nullptr, candidate, keys, keys, values, values, count, 0, 64, stream);
  if (status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate;
  candidate = 0;
  status = cub::DeviceRadixSort::SortPairsDescending(
      nullptr, candidate, keys, keys, values, values, count, 0, 64, stream);
  if (status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate > maximum ? candidate : maximum;
  candidate = 0;
  status = cub::DeviceRadixSort::SortKeys(
      nullptr, candidate, keys, keys, count, 0, 64, stream);
  if (status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate > maximum ? candidate : maximum;
  candidate = 0;
  status = cub::DeviceSelect::Flagged(
      nullptr, candidate, values, flags, values, selected_count, count, stream);
  if (status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate > maximum ? candidate : maximum;
  candidate = 0;
  status = cub::DeviceRunLengthEncode::Encode(
      nullptr, candidate, keys, keys, run_lengths, selected_count, count, stream);
  if (status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate > maximum ? candidate : maximum;
  if (!align_device_bytes_v1(maximum, scratch_bytes)) {
    return NEO_RESIDENT_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  return NEO_RESIDENT_STATUS_OK_V1;
}

__host__ __device__ bool checked_gene_term_extent_v1(
    std::uint64_t candidate_count,
    std::uint32_t max_terms_per_gene,
    std::uint64_t* extent);

bool checked_physical_layout_v1(const NeoResidentGenerationPlanV1& plan,
                                cudaStream_t stream,
                                GenerationPhysicalLayoutV1* layout) {
  if (layout == nullptr) {
    return false;
  }
  const std::size_t population = static_cast<std::size_t>(plan.logical_population_count);
  std::uint64_t term_extent_u64 = 0;
  if (!checked_gene_term_extent_v1(plan.logical_population_count,
                                   plan.max_terms_per_gene,
                                   &term_extent_u64) ||
      term_extent_u64 > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const std::size_t term_extent = static_cast<std::size_t>(term_extent_u64);
  std::size_t bytes = 0;
  if (!checked_mul_v1(population, sizeof(NeoResidentGenerationGeneScalarV1), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->logical_gene_scalar_bytes) ||
      !checked_mul_v1(term_extent, sizeof(std::uint64_t), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->logical_gene_index_bytes) ||
      !checked_mul_v1(term_extent, sizeof(double), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->logical_gene_weight_bytes)) {
    return false;
  }
  std::size_t offspring_raw = 0;
  if (!checked_add_v1(layout->logical_gene_scalar_bytes, layout->logical_gene_index_bytes,
                      &offspring_raw) ||
      !checked_add_v1(offspring_raw, layout->logical_gene_weight_bytes, &offspring_raw) ||
      !align_device_bytes_v1(offspring_raw, &layout->offspring_bytes)) {
    return false;
  }
  if (!checked_mul_v1(population, sizeof(NeoResidentGenerationMetricRowV1), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->metric_row_bytes) ||
      !checked_mul_v1(population, std::size_t{5} * sizeof(std::uint64_t), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->rank_key_bytes)) {
    return false;
  }
  std::size_t selection_raw = 0;
  std::size_t selection_flags = 0;
  if (!checked_mul_v1(population, std::size_t{3} * sizeof(std::uint64_t), &selection_raw) ||
      !checked_mul_v1(population, sizeof(std::uint8_t), &selection_flags) ||
      !checked_add_v1(selection_raw, selection_flags, &selection_raw) ||
      !checked_add_v1(selection_raw, std::size_t{2} * sizeof(std::uint32_t), &selection_raw) ||
      !align_device_bytes_v1(selection_raw, &layout->selection_bytes)) {
    return false;
  }
  std::size_t dedup_raw = 0;
  std::size_t dedup_run_lengths = 0;
  std::size_t dedup_flags = 0;
  if (!checked_mul_v1(population, std::size_t{5} * sizeof(std::uint64_t), &dedup_raw) ||
      !checked_mul_v1(population, sizeof(std::uint32_t), &dedup_run_lengths) ||
      !checked_mul_v1(population, sizeof(std::uint8_t), &dedup_flags) ||
      !checked_add_v1(dedup_raw, dedup_run_lengths, &dedup_raw) ||
      !checked_add_v1(dedup_raw, dedup_flags, &dedup_raw) ||
      !checked_add_v1(dedup_raw, std::size_t{2} * sizeof(std::uint32_t), &dedup_raw) ||
      !checked_add_v1(dedup_raw, std::size_t{12} * sizeof(std::uint64_t), &dedup_raw) ||
      !checked_add_v1(dedup_raw, sizeof(NeoResidentGenerationDeviceSealV2), &dedup_raw) ||
      !checked_add_v1(dedup_raw, sizeof(NeoResidentSearchDeviceControlV2), &dedup_raw) ||
      !checked_add_v1(dedup_raw, std::size_t{11} * sizeof(double), &dedup_raw) ||
      !align_device_bytes_v1(dedup_raw, &layout->dedup_hash_bytes)) {
    return false;
  }
  if (query_cub_generation_scratch_bytes_v1(plan, stream, &layout->cub_scratch_bytes) !=
      NEO_RESIDENT_STATUS_OK_V1) {
    return false;
  }
  std::size_t coverage_bytes = 0;
  if (!checked_mul_v1(population, sizeof(std::uint8_t), &coverage_bytes) ||
      !align_device_bytes_v1(coverage_bytes,
                             &layout->retained_evaluation_workspace_bytes)) {
    return false;
  }
  if (!align_device_bytes_v1(sizeof(NeoResidentSearchTerminalReceiptV2),
                             &layout->terminal_device_receipt_bytes)) {
    return false;
  }
  std::size_t total = 0;
  const std::size_t charges[] = {
      layout->logical_gene_scalar_bytes,
      layout->logical_gene_index_bytes,
      layout->logical_gene_weight_bytes,
      layout->offspring_bytes,
      layout->metric_row_bytes,
      layout->rank_key_bytes,
      layout->selection_bytes,
      layout->dedup_hash_bytes,
      layout->cub_scratch_bytes,
      layout->retained_evaluation_workspace_bytes,
      layout->terminal_device_receipt_bytes,
  };
  for (const std::size_t charge : charges) {
    if (!checked_add_v1(total, charge, &total)) {
      return false;
    }
  }
  layout->total_device_bytes = total;
  std::size_t numerator = 0;
  if (!checked_add_v1(population,
                      static_cast<std::size_t>(plan.retained_evaluation_capacity) - 1,
                      &numerator)) {
    return false;
  }
  layout->generation_chunk_count =
      numerator / static_cast<std::size_t>(plan.retained_evaluation_capacity);
  return layout->generation_chunk_count != 0;
}

}  // namespace

struct NeoResidentGenerationRunV1 {
  NeoResidentGenerationPlanV1 plan;
  NeoResidentGenerationAllocationReceiptV1 allocation;
  // Exact run-scoped identities minted from the admitted CUDA UUID/context/
  // stream/pool facts. These are retained separately from plan semantics so a
  // later scoring bind cannot substitute run/plan hashes for runtime facts.
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  cudaStream_t admitted_run_stream;
  cudaEvent_t resident_parent_ready_event;
  cudaEvent_t ready_event;
  void* population_lifetime_owner;
  void* allocation_base;
  NeoResidentGenerationGeneScalarV1* gene_scalars_device;
  std::uint64_t* gene_indices_device;
  double* gene_weights_device;
  NeoResidentGenerationGeneScalarV1* offspring_gene_scalars_device;
  std::uint64_t* offspring_gene_indices_device;
  double* offspring_gene_weights_device;
  NeoResidentGenerationMetricRowV1* metric_rows_device;
  std::uint64_t* resident_decision_keys_device;
  std::uint64_t* rank_keys_a_device;
  std::uint64_t* rank_keys_b_device;
  std::uint64_t* rank_values_a_device;
  std::uint64_t* rank_values_b_device;
  std::uint64_t* parent_a_device;
  std::uint64_t* parent_b_device;
  std::uint64_t* selected_indices_device;
  std::uint8_t* selection_flags_device;
  std::uint32_t* selected_count_device;
  std::uint32_t* dedup_run_count_device;
  std::uint64_t* gene_hashes_a_device;
  std::uint64_t* gene_hashes_b_device;
  std::uint64_t* dedup_values_a_device;
  std::uint64_t* dedup_values_b_device;
  std::uint64_t* unique_gene_hashes_device;
  std::uint32_t* dedup_run_lengths_device;
  std::uint8_t* dedup_flags_device;
  std::uint32_t* gene_hash_collision_fault_device;
  std::uint32_t* device_content_fault_device;
  std::uint64_t* content_identities_device;
  NeoResidentGenerationDeviceSealV2* device_seal_v2;
  NeoResidentSearchDeviceControlV2* resident_control_device_v2;
  double* smc_weights_device_v2;
  void* cub_scratch_device;
  std::uint8_t* exact_chunk_coverage_device;
  NeoResidentSearchTerminalReceiptV2* terminal_device_receipt_v2;
  NeoResidentSearchTerminalReceiptV2* terminal_host_receipt_v2;
  std::size_t cub_scratch_bytes;
  std::uint64_t logical_population_count;
  std::uint64_t retained_evaluation_capacity;
  std::uint64_t next_expected_logical_offset;
  std::uint64_t current_generation_index;
  std::uint64_t store_epoch_v2;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t next_event_id;
  std::uint64_t run_token;
  std::uint64_t completion_event_query_count_v2;
  std::uint32_t current_store_index_v2;
  const NeoResidentGenerationReadyEventV1* ready_receipt_token_v2;
  const NeoResidentGenerationReadyEventV1* source_ready_receipt_token_v2;
  std::uint64_t source_event_id_v2;
  std::uint64_t source_same_stream_enqueue_count_v2;
  const NeoResidentSearchAdvancePendingReceiptV2* pending_receipt_token_v2;
  bool sealed;
  bool post_ga_in_place_bound;
  bool initialized_v2;
  bool evaluator_constants_configured_v2;
  bool smc_gate_disabled_v2;
  bool one_generation_advance_enqueued_v2;
  bool one_generation_advance_pending_v2;
  bool terminal_committed_v2;
  bool terminal_event_proven_v2;
  bool poisoned_v2;
  bool allocation_free_issued_v2;
  bool free_outcome_unknown_deliberate_leak_v2;
#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
  bool fixture_duplicate_final_content_v2;
  std::uint64_t fixture_duplicate_source_v2;
  std::uint64_t fixture_duplicate_destination_v2;
#endif
};

namespace {

static_assert(sizeof(NeoResidentGenerationGeneScalarV1) == 72,
              "fixed resident generation scalar ABI changed");
static_assert(sizeof(NeoResidentGenerationMetricRowV1) == 104,
              "resident generation metric row ABI changed");

void* retire_generation_allocation_identity_v2(
    NeoResidentGenerationRunV1* run) {
  if (run == nullptr || run->allocation_free_issued_v2) {
    return nullptr;
  }
  void* allocation_to_retire = run->allocation_base;
  run->allocation_base = nullptr;
  run->gene_scalars_device = nullptr;
  run->gene_indices_device = nullptr;
  run->gene_weights_device = nullptr;
  run->offspring_gene_scalars_device = nullptr;
  run->offspring_gene_indices_device = nullptr;
  run->offspring_gene_weights_device = nullptr;
  run->metric_rows_device = nullptr;
  run->resident_decision_keys_device = nullptr;
  run->rank_keys_a_device = nullptr;
  run->rank_keys_b_device = nullptr;
  run->rank_values_a_device = nullptr;
  run->rank_values_b_device = nullptr;
  run->parent_a_device = nullptr;
  run->parent_b_device = nullptr;
  run->selected_indices_device = nullptr;
  run->selection_flags_device = nullptr;
  run->selected_count_device = nullptr;
  run->dedup_run_count_device = nullptr;
  run->gene_hashes_a_device = nullptr;
  run->gene_hashes_b_device = nullptr;
  run->dedup_values_a_device = nullptr;
  run->dedup_values_b_device = nullptr;
  run->unique_gene_hashes_device = nullptr;
  run->dedup_run_lengths_device = nullptr;
  run->dedup_flags_device = nullptr;
  run->gene_hash_collision_fault_device = nullptr;
  run->device_content_fault_device = nullptr;
  run->content_identities_device = nullptr;
  run->device_seal_v2 = nullptr;
  run->resident_control_device_v2 = nullptr;
  run->smc_weights_device_v2 = nullptr;
  run->cub_scratch_device = nullptr;
  run->cub_scratch_bytes = 0;
  run->exact_chunk_coverage_device = nullptr;
  run->terminal_device_receipt_v2 = nullptr;
  run->allocation_free_issued_v2 = true;
  // CUDA's stream-ordered allocator contract says cudaFreeAsync may surface a
  // prior asynchronous error. The identity is therefore retired before
  // invocation and is never queried, accessed, or freed again, regardless of
  // the returned status:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
  return allocation_to_retire;
}

std::uint64_t generation_content_identity_handle_v1(std::uint64_t run_token,
                                                     std::uint64_t lane) {
  return (run_token << 2) | lane;
}

bool partition_generation_allocation_v1(NeoResidentGenerationRunV1* run) {
  DeviceCursorV1 cursor{static_cast<std::uint8_t*>(run->allocation_base), 0,
                        static_cast<std::size_t>(run->allocation.total_device_bytes)};
  const std::size_t population = static_cast<std::size_t>(run->plan.logical_population_count);
  std::uint64_t term_extent_u64 = 0;
  if (!checked_gene_term_extent_v1(run->plan.logical_population_count,
                                   run->plan.max_terms_per_gene,
                                   &term_extent_u64) ||
      term_extent_u64 > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const std::size_t term_extent = static_cast<std::size_t>(term_extent_u64);
  const std::size_t scalar_start = cursor.offset;
  run->gene_scalars_device =
      take_device_array_v1<NeoResidentGenerationGeneScalarV1>(&cursor, population);
  cursor.offset = scalar_start + static_cast<std::size_t>(run->allocation.logical_gene_scalar_bytes);
  const std::size_t index_start = cursor.offset;
  run->gene_indices_device = take_device_array_v1<std::uint64_t>(&cursor, term_extent);
  cursor.offset = index_start + static_cast<std::size_t>(run->allocation.logical_gene_index_bytes);
  const std::size_t weight_start = cursor.offset;
  run->gene_weights_device = take_device_array_v1<double>(&cursor, term_extent);
  cursor.offset = weight_start + static_cast<std::size_t>(run->allocation.logical_gene_weight_bytes);

  const std::size_t offspring_start = cursor.offset;
  run->offspring_gene_scalars_device =
      take_device_array_v1<NeoResidentGenerationGeneScalarV1>(&cursor, population);
  run->offspring_gene_indices_device = take_device_array_v1<std::uint64_t>(&cursor, term_extent);
  run->offspring_gene_weights_device = take_device_array_v1<double>(&cursor, term_extent);
  cursor.offset = offspring_start + static_cast<std::size_t>(run->allocation.offspring_bytes);

  const std::size_t metric_start = cursor.offset;
  run->metric_rows_device =
      take_device_array_v1<NeoResidentGenerationMetricRowV1>(&cursor, population);
  cursor.offset = metric_start + static_cast<std::size_t>(run->allocation.metric_row_bytes);

  const std::size_t rank_start = cursor.offset;
  run->resident_decision_keys_device =
      take_device_array_v1<std::uint64_t>(&cursor, population);
  run->rank_keys_a_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->rank_keys_b_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->rank_values_a_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->rank_values_b_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  cursor.offset = rank_start + static_cast<std::size_t>(run->allocation.rank_key_bytes);

  const std::size_t selection_start = cursor.offset;
  run->parent_a_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->parent_b_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->selected_indices_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->selected_count_device = take_device_array_v1<std::uint32_t>(&cursor, 1);
  run->dedup_run_count_device = take_device_array_v1<std::uint32_t>(&cursor, 1);
  run->selection_flags_device = take_device_array_v1<std::uint8_t>(&cursor, population);
  cursor.offset = selection_start + static_cast<std::size_t>(run->allocation.selection_bytes);

  const std::size_t dedup_start = cursor.offset;
  run->device_seal_v2 =
      take_device_array_v1<NeoResidentGenerationDeviceSealV2>(&cursor, 1);
  run->resident_control_device_v2 =
      take_device_array_v1<NeoResidentSearchDeviceControlV2>(&cursor, 1);
  run->smc_weights_device_v2 = take_device_array_v1<double>(&cursor, 11);
  run->gene_hashes_a_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->gene_hashes_b_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->dedup_values_a_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->dedup_values_b_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->unique_gene_hashes_device = take_device_array_v1<std::uint64_t>(&cursor, population);
  run->content_identities_device = take_device_array_v1<std::uint64_t>(&cursor, 12);
  run->dedup_run_lengths_device = take_device_array_v1<std::uint32_t>(&cursor, population);
  run->gene_hash_collision_fault_device = take_device_array_v1<std::uint32_t>(&cursor, 1);
  run->device_content_fault_device = take_device_array_v1<std::uint32_t>(&cursor, 1);
  run->dedup_flags_device = take_device_array_v1<std::uint8_t>(&cursor, population);
  cursor.offset = dedup_start + static_cast<std::size_t>(run->allocation.dedup_hash_bytes);

  const std::size_t scratch_start = cursor.offset;
  run->cub_scratch_device = cursor.base + cursor.offset;
  run->cub_scratch_bytes = static_cast<std::size_t>(run->allocation.cub_scratch_bytes);
  cursor.offset = scratch_start + run->cub_scratch_bytes;
  const std::size_t coverage_start = cursor.offset;
  run->exact_chunk_coverage_device =
      take_device_array_v1<std::uint8_t>(&cursor, population);
  cursor.offset = coverage_start +
                  static_cast<std::size_t>(run->allocation.retained_evaluation_workspace_bytes);
  const std::size_t terminal_start = cursor.offset;
  run->terminal_device_receipt_v2 =
      take_device_array_v1<NeoResidentSearchTerminalReceiptV2>(&cursor, 1);
  cursor.offset = terminal_start +
                  static_cast<std::size_t>(
                      run->allocation.terminal_device_receipt_bytes);

  return run->gene_scalars_device != nullptr && run->gene_indices_device != nullptr &&
         run->gene_weights_device != nullptr && run->offspring_gene_scalars_device != nullptr &&
         run->offspring_gene_indices_device != nullptr &&
         run->offspring_gene_weights_device != nullptr && run->metric_rows_device != nullptr &&
         run->resident_decision_keys_device != nullptr &&
         run->rank_keys_a_device != nullptr && run->rank_keys_b_device != nullptr &&
         run->rank_values_a_device != nullptr && run->rank_values_b_device != nullptr &&
         run->parent_a_device != nullptr && run->parent_b_device != nullptr &&
         run->selected_indices_device != nullptr && run->selection_flags_device != nullptr &&
         run->selected_count_device != nullptr && run->dedup_run_count_device != nullptr &&
         run->gene_hashes_a_device != nullptr && run->gene_hashes_b_device != nullptr &&
         run->dedup_values_a_device != nullptr && run->dedup_values_b_device != nullptr &&
         run->unique_gene_hashes_device != nullptr && run->dedup_run_lengths_device != nullptr &&
         run->dedup_flags_device != nullptr && run->gene_hash_collision_fault_device != nullptr &&
         run->device_content_fault_device != nullptr && run->content_identities_device != nullptr &&
         run->device_seal_v2 != nullptr && run->resident_control_device_v2 != nullptr &&
         run->smc_weights_device_v2 != nullptr &&
         run->cub_scratch_device != nullptr && run->exact_chunk_coverage_device != nullptr &&
         run->terminal_device_receipt_v2 != nullptr &&
         cursor.offset == cursor.capacity;
}

__host__ __device__ std::uint64_t f64_bits_v1(double value) {
  union {
    double floating;
    std::uint64_t bits;
  } representation{};
  representation.floating = value;
  return representation.bits;
}

__host__ __device__ double f64_from_bits_v1(std::uint64_t bits) {
  union {
    double floating;
    std::uint64_t bits;
  } representation{};
  representation.bits = bits;
  return representation.floating;
}

__device__ double clamp_f64_v1(double value, double minimum, double maximum) {
  return value < minimum ? minimum : (value > maximum ? maximum : value);
}

__device__ Uint4V1 philox4x32_10_v1(Uint4V1 counter, Uint2V1 key) {
  for (int round = 0; round < 10; ++round) {
    const std::uint64_t product_0 =
        static_cast<std::uint64_t>(PHILOX_M0_V1) * counter.x;
    const std::uint64_t product_1 =
        static_cast<std::uint64_t>(PHILOX_M1_V1) * counter.z;
    counter = Uint4V1{
        static_cast<std::uint32_t>(product_1 >> 32) ^ counter.y ^ key.x,
        static_cast<std::uint32_t>(product_1),
        static_cast<std::uint32_t>(product_0 >> 32) ^ counter.w ^ key.y,
        static_cast<std::uint32_t>(product_0),
    };
    key.x += PHILOX_W0_V1;
    key.y += PHILOX_W1_V1;
  }
  return counter;
}

__device__ Uint4V1 philox_draw_v1(const NeoResidentGenerationPlanV1& plan,
                                  std::uint64_t generation_index,
                                  std::uint64_t candidate_identity,
                                  NeoResidentPhiloxOperatorV1 genetic_operator_identity,
                                  std::uint64_t draw_index) {
  std::uint32_t run_word_0 = 0;
  std::uint32_t run_word_1 = 0;
  for (std::uint32_t byte = 0; byte < 4; ++byte) {
    run_word_0 |= static_cast<std::uint32_t>(plan.run_identity_sha256[byte]) << (8 * byte);
    run_word_1 |= static_cast<std::uint32_t>(plan.run_identity_sha256[byte + 4]) << (8 * byte);
  }
  const Uint4V1 counter{
      static_cast<std::uint32_t>(candidate_identity),
      static_cast<std::uint32_t>(candidate_identity >> 32),
      static_cast<std::uint32_t>(generation_index),
      static_cast<std::uint32_t>(draw_index),
  };
  const Uint2V1 key{
      static_cast<std::uint32_t>(plan.search_seed) ^ run_word_0 ^
          static_cast<std::uint32_t>(genetic_operator_identity),
      static_cast<std::uint32_t>(plan.search_seed >> 32) ^ run_word_1 ^
          static_cast<std::uint32_t>(draw_index >> 32),
  };
  return philox4x32_10_v1(counter, key);
}

__device__ std::uint64_t philox_uniform_below_without_modulo_bias_v1(
    const NeoResidentGenerationPlanV1& plan,
    std::uint64_t generation_index,
    std::uint64_t candidate_identity,
    NeoResidentPhiloxOperatorV1 genetic_operator_identity,
    std::uint64_t decision_slot,
    std::uint64_t exclusive_upper_bound) {
  if (exclusive_upper_bound <= 1) {
    return 0;
  }
  const std::uint32_t u32_max_v1 = ~std::uint32_t{0};
  const std::uint64_t u64_max_v1 = ~std::uint64_t{0};
  if (decision_slot > u32_max_v1) {
    asm("trap;");
    return 0;
  }
  const std::uint64_t rejection_limit =
      u64_max_v1 - (u64_max_v1 % exclusive_upper_bound);
  for (std::uint64_t attempt = 0; attempt <= u32_max_v1; ++attempt) {
    const std::uint64_t draw_index = (decision_slot << 32) | attempt;
    const Uint4V1 draw = philox_draw_v1(plan, generation_index, candidate_identity,
                                        genetic_operator_identity, draw_index);
    const std::uint64_t value =
        (static_cast<std::uint64_t>(draw.y) << 32) | draw.x;
    if (value < rejection_limit) {
      return value % exclusive_upper_bound;
    }
  }
  asm("trap;");
  return 0;
}

__host__ __device__ bool checked_gene_term_extent_v1(
    std::uint64_t candidate_count,
    std::uint32_t max_terms_per_gene,
    std::uint64_t* extent) {
  const std::uint64_t u64_max_v1 = ~std::uint64_t{0};
  if (extent == nullptr ||
      (candidate_count != 0 && max_terms_per_gene > u64_max_v1 / candidate_count)) {
    return false;
  }
  *extent = candidate_count * max_terms_per_gene;
  return true;
}

__device__ void deterministic_empty_gene_repair_v1(
    NeoResidentGenerationGeneScalarV1* scalar,
    std::uint64_t* indices,
    double* weights,
    const NeoResidentGenerationPlanV1& plan) {
  scalar->term_count = 1;
  indices[0] = scalar->gene_identity % plan.feature_count;
  weights[0] = (scalar->gene_identity & 1ull) == 0 ? 1.0 : -1.0;
}

__device__ void normalize_fixed_stride_gene_v1(
    NeoResidentGenerationGeneScalarV1* scalar,
    std::uint64_t* indices,
    double* weights,
    const NeoResidentGenerationPlanV1& plan) {
  const std::uint32_t term_count =
      scalar->term_count <= plan.max_terms_per_gene ? scalar->term_count
                                                    : plan.max_terms_per_gene;
  std::uint32_t write = 0;
  for (std::uint32_t term = 0; term < term_count; ++term) {
    const std::uint64_t indicator_index = indices[term];
    double weight = weights[term];
    weight = clamp_f64_v1(weight, -5.0, 5.0);
    if (indicator_index < plan.feature_count && isfinite(weight) && fabs(weight) > 1.0e-6) {
      indices[write] = indicator_index;
      weights[write] = weight;
      ++write;
    }
  }
  scalar->term_count = write;
  if (write == 0) {
    deterministic_empty_gene_repair_v1(scalar, indices, weights, plan);
    write = 1;
  }
  for (std::uint32_t term = write; term < plan.max_terms_per_gene; ++term) {
    indices[term] = 0;
    weights[term] = 0.0;
  }
  scalar->long_threshold = clamp_f64_v1(scalar->long_threshold, 0.0, 1.0);
  scalar->short_threshold = clamp_f64_v1(scalar->short_threshold, 0.0, 1.0);
  scalar->target_pips = clamp_f64_v1(scalar->target_pips, 0.0, 1000000.0);
  scalar->stop_pips = clamp_f64_v1(scalar->stop_pips, 0.0, 1000000.0);
  scalar->stop_vol_multiplier =
      clamp_f64_v1(scalar->stop_vol_multiplier, 0.0, 1000000.0);
  scalar->smc_flags &= (1u << plan.smc_flag_count) - 1u;
}

__device__ void initialize_one_gene_v1(
    std::uint64_t candidate,
    std::uint64_t generation_index,
    NeoResidentGenerationGeneScalarV1* scalars,
    std::uint64_t* indices,
    double* weights,
    const NeoResidentGenerationPlanV1& plan) {
  const std::uint64_t identity = (generation_index << 32) ^ candidate;
  const std::uint64_t term_span =
      static_cast<std::uint64_t>(plan.max_terms_per_gene - plan.minimum_terms_per_gene) + 1;
  auto& scalar = scalars[candidate];
  scalar.gene_identity = identity;
  scalar.content_hash = 0;
  scalar.term_count = static_cast<std::uint32_t>(plan.minimum_terms_per_gene +
      philox_uniform_below_without_modulo_bias_v1(
          plan, generation_index, identity,
          NeoResidentPhiloxOperatorV1::InitializeTermCount, 0, term_span));
  scalar.smc_flags = 0;
  scalar.generation = static_cast<std::uint32_t>(generation_index);
  scalar.reserved = 0;
  const std::uint64_t long_level = philox_uniform_below_without_modulo_bias_v1(
      plan, generation_index, identity,
      NeoResidentPhiloxOperatorV1::InitializeThreshold, 0,
      plan.threshold_level_count);
  const std::uint64_t short_level = philox_uniform_below_without_modulo_bias_v1(
      plan, generation_index, identity,
      NeoResidentPhiloxOperatorV1::InitializeThreshold, 1,
      plan.threshold_level_count);
  scalar.long_threshold =
      f64_from_bits_v1(plan.threshold_ladder_bits[long_level]);
  scalar.short_threshold =
      f64_from_bits_v1(plan.threshold_ladder_bits[short_level]);
  const std::uint64_t target_level = philox_uniform_below_without_modulo_bias_v1(
      plan, generation_index, identity,
      NeoResidentPhiloxOperatorV1::InitializeStopGeometry, 0, 6);
  const std::uint64_t stop_level = philox_uniform_below_without_modulo_bias_v1(
      plan, generation_index, identity,
      NeoResidentPhiloxOperatorV1::InitializeStopGeometry, 1, 6);
  const std::uint64_t volatility_level =
      philox_uniform_below_without_modulo_bias_v1(
          plan, generation_index, identity,
          NeoResidentPhiloxOperatorV1::InitializeStopGeometry, 2, 6);
  scalar.target_pips = f64_from_bits_v1(plan.stop_bounds_bits[target_level]);
  scalar.stop_pips = f64_from_bits_v1(plan.stop_bounds_bits[stop_level]);
  scalar.stop_vol_multiplier =
      f64_from_bits_v1(plan.stop_bounds_bits[volatility_level]);
  for (std::uint32_t flag = 0; flag < plan.smc_flag_count; ++flag) {
    const Uint4V1 smc_draw = philox_draw_v1(
        plan, generation_index, identity,
        NeoResidentPhiloxOperatorV1::InitializeSmcFlag, flag);
    if (static_cast<std::uint64_t>(smc_draw.x) < plan.smc_probability_q32[flag]) {
      scalar.smc_flags |= 1u << flag;
    }
  }
  const std::uint64_t base = candidate * plan.max_terms_per_gene;
  for (std::uint32_t term = 0; term < plan.max_terms_per_gene; ++term) {
    indices[base + term] = philox_uniform_below_without_modulo_bias_v1(
        plan, generation_index, identity,
        NeoResidentPhiloxOperatorV1::InitializeIndicator,
        term, plan.feature_count);
    const std::uint64_t weight_level = philox_uniform_below_without_modulo_bias_v1(
        plan, generation_index, identity,
        NeoResidentPhiloxOperatorV1::InitializeWeightLevel,
        term, 5);
    const Uint4V1 sign_draw = philox_draw_v1(
        plan, generation_index, identity,
        NeoResidentPhiloxOperatorV1::InitializeWeightSign, term);
    const double magnitude = 1.0 + static_cast<double>(weight_level);
    weights[base + term] = (sign_draw.x & 1u) == 0 ? magnitude : -magnitude;
  }
  normalize_fixed_stride_gene_v1(&scalar, indices + base, weights + base, plan);
}

__global__ void initialize_fixed_stride_population_kernel_v1(
    NeoResidentGenerationGeneScalarV1* scalars,
    std::uint64_t* indices,
    double* weights,
    NeoResidentGenerationPlanV1 plan) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate < plan.logical_population_count) {
    initialize_one_gene_v1(candidate, 0, scalars, indices, weights, plan);
  }
}

__device__ std::uint64_t hash_mix_u64_v1(std::uint64_t hash, std::uint64_t value) {
  for (std::uint32_t byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffull;
    hash *= FNV_PRIME_V1;
  }
  return hash;
}

__device__ std::uint64_t full_fixed_stride_gene_hash_v1(
    const NeoResidentGenerationGeneScalarV1& scalar,
    const std::uint64_t* indices,
    const double* weights,
    const NeoResidentGenerationPlanV1& plan) {
  std::uint64_t hash = FNV_OFFSET_0_V1;
  hash = hash_mix_u64_v1(hash, scalar.term_count);
  hash = hash_mix_u64_v1(hash, scalar.smc_flags);
  hash = hash_mix_u64_v1(hash, f64_bits_v1(scalar.long_threshold));
  hash = hash_mix_u64_v1(hash, f64_bits_v1(scalar.short_threshold));
  hash = hash_mix_u64_v1(hash, f64_bits_v1(scalar.target_pips));
  hash = hash_mix_u64_v1(hash, f64_bits_v1(scalar.stop_pips));
  hash = hash_mix_u64_v1(hash, f64_bits_v1(scalar.stop_vol_multiplier));
  for (std::uint32_t term = 0; term < plan.max_terms_per_gene; ++term) {
    hash = hash_mix_u64_v1(hash, indices[term]);
    hash = hash_mix_u64_v1(hash, f64_bits_v1(weights[term]));
  }
  return hash;
}

__device__ bool full_fixed_stride_gene_equal_v1(
    const NeoResidentGenerationGeneScalarV1* scalars,
    const std::uint64_t* indices,
    const double* weights,
    std::uint64_t left,
    std::uint64_t right,
    const NeoResidentGenerationPlanV1& plan) {
  const auto& a = scalars[left];
  const auto& b = scalars[right];
  if (a.term_count != b.term_count || a.smc_flags != b.smc_flags ||
      f64_bits_v1(a.long_threshold) != f64_bits_v1(b.long_threshold) ||
      f64_bits_v1(a.short_threshold) != f64_bits_v1(b.short_threshold) ||
      f64_bits_v1(a.target_pips) != f64_bits_v1(b.target_pips) ||
      f64_bits_v1(a.stop_pips) != f64_bits_v1(b.stop_pips) ||
      f64_bits_v1(a.stop_vol_multiplier) != f64_bits_v1(b.stop_vol_multiplier)) {
    return false;
  }
  const std::uint64_t left_base = left * plan.max_terms_per_gene;
  const std::uint64_t right_base = right * plan.max_terms_per_gene;
  for (std::uint32_t term = 0; term < plan.max_terms_per_gene; ++term) {
    if (indices[left_base + term] != indices[right_base + term] ||
        f64_bits_v1(weights[left_base + term]) !=
            f64_bits_v1(weights[right_base + term])) {
      return false;
    }
  }
  return true;
}

__global__ void gene_hash_kernel_v1(
    NeoResidentGenerationGeneScalarV1* scalars,
    const std::uint64_t* indices,
    const double* weights,
    std::uint64_t* hashes,
    std::uint64_t* values,
    NeoResidentGenerationPlanV1 plan,
    const std::uint32_t* device_content_fault) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= plan.logical_population_count) {
    return;
  }
  // CUB is enqueued without a host-side fault round trip. Its complete input
  // range therefore has to be defined even when an earlier device stage has
  // latched a fault. Later semantic kernels and commit remain fault-gated.
  values[candidate] = candidate;
  if (*device_content_fault != 0u) {
    hashes[candidate] = RESIDENT_CUB_FAULT_SENTINEL_KEY_V2;
    return;
  }
  const std::uint64_t base = candidate * plan.max_terms_per_gene;
  const std::uint64_t hash = full_fixed_stride_gene_hash_v1(
      scalars[candidate], indices + base, weights + base, plan);
  scalars[candidate].content_hash = hash;
  hashes[candidate] = hash;
}

__global__ void verify_sorted_gene_dedup_kernel_v1(
    const NeoResidentGenerationGeneScalarV1* scalars,
    const std::uint64_t* indices,
    const double* weights,
    const std::uint64_t* sorted_hashes,
    const std::uint64_t* sorted_values,
    std::uint8_t* sorted_unique_flags,
    std::uint8_t* candidate_valid_flags,
    std::uint32_t* gene_hash_collision_fault_device,
    std::uint32_t* device_content_fault_device,
    NeoResidentGenerationPlanV1 plan) {
  const std::uint64_t position =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (position >= plan.logical_population_count) {
    return;
  }
  // DeviceSelect consumes every flag. On a fault, write a deterministic empty
  // selection instead of leaving cudaMallocAsync storage undefined.
  if (*device_content_fault_device != 0u) {
    sorted_unique_flags[position] = 0;
    candidate_valid_flags[position] = 0;
    return;
  }
  const std::uint64_t candidate = sorted_values[position];
  bool unique = true;
  if (position != 0 && sorted_hashes[position] == sorted_hashes[position - 1]) {
    const std::uint64_t prior = sorted_values[position - 1];
    if (!full_fixed_stride_gene_equal_v1(scalars, indices, weights, prior, candidate, plan)) {
      atomicExch(gene_hash_collision_fault_device, 1u);
    } else {
      unique = false;
      atomicExch(device_content_fault_device, 1u);
    }
  }
  sorted_unique_flags[position] = unique ? 1 : 0;
  candidate_valid_flags[candidate] = unique ? 1 : 0;
}

__device__ std::uint64_t stable_gene_identity_tie_key_v1(
    const NeoResidentGenerationGeneScalarV1& scalar) {
  return scalar.gene_identity;
}

__global__ void build_gene_identity_rank_keys_kernel_v1(
    const NeoResidentGenerationGeneScalarV1* scalars,
    std::uint64_t* keys,
    std::uint64_t* values,
    std::uint64_t count,
    const std::uint32_t* device_content_fault) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  values[index] = index;
  keys[index] = *device_content_fault == 0u
                    ? stable_gene_identity_tie_key_v1(scalars[index])
                    : RESIDENT_CUB_FAULT_SENTINEL_KEY_V2;
}

__global__ void gather_resident_decision_rank_keys_kernel_v1(
    const std::uint64_t* resident_decision_keys,
    const std::uint8_t* candidate_valid_flags,
    const std::uint64_t* stable_order,
    std::uint64_t* keys,
    std::uint64_t* values,
    std::uint64_t count,
    const std::uint32_t* device_content_fault) {
  const std::uint64_t position =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (position >= count) {
    return;
  }
  if (*device_content_fault != 0u) {
    keys[position] = RESIDENT_CUB_FAULT_SENTINEL_KEY_V2;
    values[position] = position;
    return;
  }
  const std::uint64_t candidate = stable_order[position];
  keys[position] = candidate_valid_flags[candidate] != 0
                       ? resident_decision_keys[candidate]
                       : 0;
  values[position] = candidate;
}

__device__ bool checked_rank_weight_total_v1(std::uint64_t count,
                                              std::uint64_t* total) {
  const std::uint64_t u64_max_v1 = ~std::uint64_t{0};
  if (total == nullptr || count == u64_max_v1) {
    return false;
  }
  const std::uint64_t left = (count & 1ull) == 0 ? count / 2 : count;
  const std::uint64_t right = (count & 1ull) == 0 ? count + 1 : (count + 1) / 2;
  if (left != 0 && right > u64_max_v1 / left) {
    return false;
  }
  *total = left * right;
  return true;
}

__device__ std::uint64_t rank_weight_v1(std::uint64_t logical_population_count,
                                        std::uint64_t rank) {
  return logical_population_count - rank;
}

__device__ std::uint64_t rank_from_weighted_draw_v1(std::uint64_t draw,
                                                     std::uint64_t count) {
  std::uint64_t low = 0;
  std::uint64_t high = count;
  while (low < high) {
    const std::uint64_t middle = low + (high - low) / 2;
    const std::uint64_t items = middle + 1;
    const std::uint64_t prefix = items * (2 * count - middle) / 2;
    if (draw < prefix) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return low < count ? low : count - 1;
}

__global__ void select_rank_weighted_parents_kernel_v1(
    const std::uint64_t* ranked_candidates,
    std::uint64_t* parent_a,
    std::uint64_t* parent_b,
    NeoResidentGenerationPlanV1 plan,
    std::uint64_t generation_index,
    std::uint32_t* device_content_fault_device) {
  const std::uint64_t child =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (*device_content_fault_device != 0u ||
      child >= plan.logical_population_count) {
    return;
  }
  const std::uint64_t logical_population_count = plan.logical_population_count;
  std::uint64_t total = 0;
  if (!checked_rank_weight_total_v1(logical_population_count, &total)) {
    atomicExch(device_content_fault_device, 1u);
    return;
  }
  const std::uint64_t draw_a = philox_uniform_below_without_modulo_bias_v1(
      plan, generation_index, child, NeoResidentPhiloxOperatorV1::ParentA, 0, total);
  const std::uint64_t draw_b = philox_uniform_below_without_modulo_bias_v1(
      plan, generation_index, child, NeoResidentPhiloxOperatorV1::ParentB, 0, total);
  parent_a[child] = ranked_candidates[rank_from_weighted_draw_v1(draw_a, logical_population_count)];
  parent_b[child] = ranked_candidates[rank_from_weighted_draw_v1(draw_b, logical_population_count)];
}

__global__ void select_rank_weighted_survivors_kernel_v1(
    std::uint64_t* selected_rank_indices,
    std::uint8_t* rank_available,
    NeoResidentGenerationPlanV1 plan,
    std::uint64_t generation_index,
    std::uint32_t* device_content_fault_device) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  // SortKeys always consumes survivor_count entries. Identity sentinels make
  // the full range deterministic even if a prior or in-kernel fault aborts the
  // semantic survivor draw before overwriting every entry.
  for (std::uint64_t selected = 0; selected < plan.survivor_count; ++selected) {
    selected_rank_indices[selected] = selected;
  }
  for (std::uint64_t rank = 0; rank < plan.logical_population_count; ++rank) {
    rank_available[rank] = *device_content_fault_device == 0u ? 1 : 0;
  }
  if (*device_content_fault_device != 0u) {
    return;
  }
  const std::uint64_t u64_max_v1 = ~std::uint64_t{0};
  for (std::uint64_t selected = 0; selected < plan.survivor_count; ++selected) {
    std::uint64_t total = 0;
    for (std::uint64_t rank = 0; rank < plan.logical_population_count; ++rank) {
      if (rank_available[rank] != 0) {
        const std::uint64_t weight = rank_weight_v1(plan.logical_population_count, rank);
        if (weight > u64_max_v1 - total) {
          atomicExch(device_content_fault_device, 1u);
          return;
        }
        total += weight;
      }
    }
    if (total == 0) {
      atomicExch(device_content_fault_device, 1u);
      return;
    }
    const std::uint64_t draw = philox_uniform_below_without_modulo_bias_v1(
        plan, generation_index, selected,
        NeoResidentPhiloxOperatorV1::Survivor, 0, total);
    std::uint64_t cumulative = 0;
    std::uint64_t chosen = plan.logical_population_count;
    for (std::uint64_t rank = 0; rank < plan.logical_population_count; ++rank) {
      if (rank_available[rank] == 0) {
        continue;
      }
      cumulative += rank_weight_v1(plan.logical_population_count, rank);
      if (draw < cumulative) {
        chosen = rank;
        break;
      }
    }
    if (chosen == plan.logical_population_count) {
      atomicExch(device_content_fault_device, 1u);
      return;
    }
    rank_available[chosen] = 0;
    selected_rank_indices[selected] = chosen;
  }
}

__global__ void gather_rank_weighted_survivors_kernel_v1(
    const std::uint64_t* ranked_candidates,
    const std::uint64_t* sorted_selected_ranks,
    std::uint64_t* selected_candidates,
    std::uint64_t survivor_count,
    const std::uint32_t* device_content_fault) {
  const std::uint64_t selected =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (*device_content_fault == 0u && selected < survivor_count) {
    selected_candidates[selected] = ranked_candidates[sorted_selected_ranks[selected]];
  }
}

__device__ void copy_fixed_stride_gene_v1(
    std::uint64_t source,
    std::uint64_t destination,
    const NeoResidentGenerationGeneScalarV1* source_scalars,
    const std::uint64_t* source_indices,
    const double* source_weights,
    NeoResidentGenerationGeneScalarV1* destination_scalars,
    std::uint64_t* destination_indices,
    double* destination_weights,
    const NeoResidentGenerationPlanV1& plan) {
  destination_scalars[destination] = source_scalars[source];
  const std::uint64_t source_base = source * plan.max_terms_per_gene;
  const std::uint64_t destination_base = destination * plan.max_terms_per_gene;
  for (std::uint32_t term = 0; term < plan.max_terms_per_gene; ++term) {
    destination_indices[destination_base + term] = source_indices[source_base + term];
    destination_weights[destination_base + term] = source_weights[source_base + term];
  }
}

__global__ void crossover_resident_genes_kernel_v1(
    const NeoResidentGenerationGeneScalarV1* source_scalars,
    const std::uint64_t* source_indices,
    const double* source_weights,
    const std::uint64_t* ranked_survivors,
    const std::uint64_t* parent_a,
    const std::uint64_t* parent_b,
    NeoResidentGenerationGeneScalarV1* offspring_scalars,
    std::uint64_t* offspring_indices,
    double* offspring_weights,
    NeoResidentGenerationPlanV1 plan,
    std::uint64_t generation_index,
    const std::uint32_t* device_content_fault) {
  const std::uint64_t destination =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (*device_content_fault != 0u ||
      destination >= plan.logical_population_count) {
    return;
  }
  if (destination < plan.survivor_count) {
    copy_fixed_stride_gene_v1(ranked_survivors[destination], destination, source_scalars,
                              source_indices, source_weights, offspring_scalars,
                              offspring_indices, offspring_weights, plan);
    offspring_scalars[destination].generation = static_cast<std::uint32_t>(generation_index + 1);
    return;
  }
  const std::uint64_t immigrant_begin = plan.logical_population_count - plan.immigrant_count;
  if (destination >= immigrant_begin) {
    initialize_one_gene_v1(destination, generation_index + 1, offspring_scalars,
                           offspring_indices, offspring_weights, plan);
    return;
  }
  const std::uint64_t left = parent_a[destination];
  const std::uint64_t right = parent_b[destination];
  copy_fixed_stride_gene_v1(left, destination, source_scalars, source_indices, source_weights,
                            offspring_scalars, offspring_indices, offspring_weights, plan);
  auto& child = offspring_scalars[destination];
  child.gene_identity = ((generation_index + 1) << 32) ^ destination;
  child.generation = static_cast<std::uint32_t>(generation_index + 1);
  const Uint4V1 scalar_draw = philox_draw_v1(
      plan, generation_index + 1, child.gene_identity,
      NeoResidentPhiloxOperatorV1::CrossoverScalar, 0);
  const auto& right_scalar = source_scalars[right];
  if ((scalar_draw.x & 1u) != 0) {
    child.long_threshold = right_scalar.long_threshold;
  }
  if ((scalar_draw.y & 1u) != 0) {
    child.short_threshold = right_scalar.short_threshold;
  }
  if ((scalar_draw.z & 1u) != 0) {
    child.target_pips = right_scalar.target_pips;
  }
  if ((scalar_draw.w & 1u) != 0) {
    child.stop_pips = right_scalar.stop_pips;
  }
  const std::uint64_t destination_base = destination * plan.max_terms_per_gene;
  const std::uint64_t right_base = right * plan.max_terms_per_gene;
  for (std::uint32_t term = 0; term < plan.max_terms_per_gene; ++term) {
    const Uint4V1 term_draw = philox_draw_v1(
        plan, generation_index + 1, child.gene_identity,
        NeoResidentPhiloxOperatorV1::CrossoverScalar, term + 1);
    if ((term_draw.x & 1u) != 0) {
      offspring_indices[destination_base + term] = source_indices[right_base + term];
      offspring_weights[destination_base + term] = source_weights[right_base + term];
    }
  }
}

__global__ void mutate_resident_genes_kernel_v1(
    NeoResidentGenerationGeneScalarV1* scalars,
    std::uint64_t* indices,
    double* weights,
    NeoResidentGenerationPlanV1 plan,
    std::uint64_t generation_index,
    const std::uint32_t* device_content_fault) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (*device_content_fault != 0u ||
      candidate >= plan.logical_population_count ||
      candidate < plan.survivor_count) {
    return;
  }
  auto& scalar = scalars[candidate];
  const Uint4V1 mutation_draw = philox_draw_v1(
      plan, generation_index + 1, scalar.gene_identity,
      NeoResidentPhiloxOperatorV1::MutationKind, 0);
  if (static_cast<std::uint64_t>(mutation_draw.x) < plan.mutation_intensity_q32) {
    const std::uint32_t term = static_cast<std::uint32_t>(
        philox_uniform_below_without_modulo_bias_v1(
            plan, generation_index + 1, scalar.gene_identity,
            NeoResidentPhiloxOperatorV1::MutationValue, 0,
            plan.max_terms_per_gene));
    const std::uint64_t base = candidate * plan.max_terms_per_gene;
    indices[base + term] = philox_uniform_below_without_modulo_bias_v1(
            plan, generation_index + 1, scalar.gene_identity,
            NeoResidentPhiloxOperatorV1::MutationValue, 1, plan.feature_count);
    const std::uint64_t weight_level = philox_uniform_below_without_modulo_bias_v1(
        plan, generation_index + 1, scalar.gene_identity,
        NeoResidentPhiloxOperatorV1::MutationValue, 2, 5);
    const Uint4V1 sign_draw = philox_draw_v1(
        plan, generation_index + 1, scalar.gene_identity,
        NeoResidentPhiloxOperatorV1::MutationValue, 3ull << 32);
    const double magnitude = 1.0 + static_cast<double>(weight_level);
    weights[base + term] = (sign_draw.x & 1u) == 0 ? magnitude : -magnitude;
  }
  for (std::uint32_t flag = 0; flag < plan.smc_flag_count; ++flag) {
    const Uint4V1 flag_draw = philox_draw_v1(
        plan, generation_index + 1, scalar.gene_identity,
        NeoResidentPhiloxOperatorV1::MutationSmc, flag);
    if (static_cast<std::uint64_t>(flag_draw.x) < plan.smc_probability_q32[flag]) {
      scalar.smc_flags ^= 1u << flag;
    }
  }
  normalize_fixed_stride_gene_v1(&scalar, indices + candidate * plan.max_terms_per_gene,
                                  weights + candidate * plan.max_terms_per_gene, plan);
}

__global__ void validate_and_import_scored_rows_kernel_v1(
    const NeoResidentGenerationMetricRowV1* source_rows,
    const std::uint64_t* source_decision_keys,
    const std::uint64_t* expected_scenario_ids,
    const NeoResidentGenerationGeneScalarV1* scalars,
    NeoResidentGenerationMetricRowV1* destination_rows,
    std::uint64_t* destination_decision_keys,
    std::uint64_t logical_offset,
    std::uint64_t active_scenarios,
    std::uint8_t* exact_chunk_coverage_device,
    std::uint32_t* device_content_fault_device) {
  const std::uint64_t item =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= active_scenarios) {
    return;
  }
  if (*device_content_fault_device != 0u) {
    return;
  }
  const std::uint64_t logical_candidate = logical_offset + item;
  const NeoResidentGenerationMetricRowV1 row = source_rows[item];
  if (row.candidate_id != scalars[logical_candidate].gene_identity ||
      row.scenario_id != expected_scenario_ids[item]) {
    atomicExch(device_content_fault_device, 1u);
  }
  destination_rows[logical_candidate] = row;
  destination_decision_keys[logical_candidate] = source_decision_keys[item];
  exact_chunk_coverage_device[logical_candidate] = 1;
}

__global__ void clear_generation_metadata_kernel_v1(
    std::uint8_t* exact_chunk_coverage_device,
    std::uint8_t* candidate_valid_flags,
    std::uint32_t* selected_count_device,
    std::uint32_t* dedup_run_count_device,
    std::uint32_t* gene_hash_collision_fault_device,
    std::uint32_t* device_content_fault_device,
    std::uint64_t population) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < population) {
    exact_chunk_coverage_device[index] = 0;
    candidate_valid_flags[index] = 1;
  }
  if (index == 0) {
    *selected_count_device = 0;
    *dedup_run_count_device = 0;
    *gene_hash_collision_fault_device = 0;
    *device_content_fault_device = 0;
  }
}

__global__ void promote_scoring_device_seal_kernel_v2(
    const resident_scoring_novelty_v1::NeoResidentScoringNoveltyDeviceSealV1*
        scoring_device_seal,
    std::uint32_t* device_content_fault) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      device_content_fault == nullptr) {
    return;
  }
  if (scoring_device_seal == nullptr || scoring_device_seal->abi_version != 1u ||
      scoring_device_seal->valid == 0u ||
      scoring_device_seal->device_fault_word != 0u) {
    atomicExch(device_content_fault,
               scoring_device_seal == nullptr ||
                       scoring_device_seal->device_fault_word == 0u
                   ? 1u
                   : scoring_device_seal->device_fault_word);
  }
}

__global__ void publish_one_generation_commit_kernel_v2(
    ResidentGenerationPreparedAdvanceV2 prepared,
    NeoResidentSearchTerminalReceiptV2* terminal,
    std::uint64_t run_token,
    std::uint64_t completion_event_id,
    std::uint64_t terminal_same_stream_enqueue_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || terminal == nullptr) {
    return;
  }
  const auto publish = prepared.publish_device_v2(0u);
  *terminal = {};
  terminal->abi_version =
      resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2;
  terminal->scoring_device_fault = publish.scoring_fault;
  terminal->generation_device_fault = publish.generation_fault;
  terminal->run_token = run_token;
  terminal->compact_async_d2h_count = 1;
  terminal->compact_async_d2h_bytes = sizeof(*terminal);
  terminal->completion_stream_synchronize_count = 0;
  terminal->same_stream_enqueue_count = terminal_same_stream_enqueue_count;
  terminal->completion_event_id = completion_event_id;
  if (publish.combined_fault != 0u) {
    terminal->terminal_status =
        resident_generation_v2::NEO_RESIDENT_SEARCH_TERMINAL_FAULT_V2;
    terminal->control_fault_word = publish.combined_fault;
    terminal->stop_requested = 1;
    terminal->current_store_index = publish.current_store_index;
    terminal->generation_index = publish.generation_index;
    terminal->store_epoch = publish.store_epoch;
    return;
  }
  // The population ordinal is the initial stable value. A stable gene-id pass
  // followed by a stable descending score pass therefore leaves ordinal as the
  // tertiary key without a host tie-break.
  terminal->terminal_status =
      resident_generation_v2::NEO_RESIDENT_SEARCH_TERMINAL_COMMITTED_V2;
  terminal->current_store_index = publish.current_store_index;
  terminal->generation_index = publish.generation_index;
  terminal->store_epoch = publish.store_epoch;
}

#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
__global__ void fixture_set_resident_gene_identity_kernel_v2(
    NeoResidentGenerationGeneScalarV1* scalars, std::uint64_t candidate,
    std::uint64_t gene_identity) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    scalars[candidate].gene_identity = gene_identity;
  }
}

__global__ void fixture_duplicate_final_gene_content_kernel_v2(
    NeoResidentGenerationGeneScalarV1* scalars, std::uint64_t* indices,
    double* weights, std::uint64_t source_candidate,
    std::uint64_t destination_candidate, std::uint32_t max_terms_per_gene) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t destination_identity =
      scalars[destination_candidate].gene_identity;
  scalars[destination_candidate] = scalars[source_candidate];
  // Gene identity is the deterministic rank tie-break, not part of content.
  scalars[destination_candidate].gene_identity = destination_identity;
  const std::uint64_t source_base = source_candidate * max_terms_per_gene;
  const std::uint64_t destination_base =
      destination_candidate * max_terms_per_gene;
  for (std::uint32_t term = 0; term < max_terms_per_gene; ++term) {
    indices[destination_base + term] = indices[source_base + term];
    weights[destination_base + term] = weights[source_base + term];
  }
}
#endif

__global__ void publish_resident_generation_device_seal_v2(
    NeoResidentGenerationDeviceSealV2* seal,
    NeoResidentSearchDeviceControlV2* control,
    const std::uint32_t* device_content_fault,
    NeoResidentGenerationGeneScalarV1* current_scalars,
    std::uint64_t* current_indices,
    double* current_weights,
    NeoResidentGenerationGeneScalarV1* next_scalars,
    std::uint64_t* next_indices,
    double* next_weights,
    const double* smc_weights,
    NeoResidentGenerationPlanV1 plan,
    std::uint64_t run_token,
    std::uint64_t generation_index,
    std::uint64_t store_epoch,
    std::uint32_t current_store_index,
    bool smc_gate_disabled) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (seal == nullptr || control == nullptr || device_content_fault == nullptr ||
      current_store_index > 1u) {
    return;
  }
  const std::uint32_t other_store = current_store_index ^ 1u;
  seal->abi_version =
      resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2;
  seal->flags = resident_generation_v2::NEO_RESIDENT_GENERATION_SEAL_INITIALIZED_V2;
  if (smc_gate_disabled) {
    seal->flags |=
        resident_generation_v2::NEO_RESIDENT_GENERATION_SEAL_SMC_GATE_DISABLED_V2;
  }
  if (*device_content_fault != 0u) {
    atomicCAS(&seal->fault_code, 0u, *device_content_fault);
  }
  if (seal->fault_code != 0u) {
    seal->flags |= resident_generation_v2::NEO_RESIDENT_GENERATION_SEAL_POISONED_V2;
  }
  seal->current_store_index = current_store_index;
  seal->generation_index = generation_index;
  seal->store_epoch = store_epoch;
  seal->logical_population_count = plan.logical_population_count;
  seal->max_terms_per_gene = plan.max_terms_per_gene;
  seal->smc_flag_count = plan.smc_flag_count;
  seal->run_token = run_token;
  seal->feature_count = plan.feature_count;
  seal->scalar_store[current_store_index] = current_scalars;
  seal->scalar_store[other_store] = next_scalars;
  seal->term_index_store[current_store_index] = current_indices;
  seal->term_index_store[other_store] = next_indices;
  seal->term_weight_store[current_store_index] = current_weights;
  seal->term_weight_store[other_store] = next_weights;
  seal->smc_weights = smc_weights;
  seal->gate_threshold_bits = 0;
  for (std::size_t index = 0; index < 32; ++index) {
    seal->plan_identity_sha256[index] = plan.plan_identity_sha256[index];
  }
  control->abi_version =
      resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2;
  control->fault_word = *device_content_fault;
  control->generation_index = generation_index;
  control->executed_generations = generation_index;
  control->stagnant_generations = 0;
  control->best_score_order_key = 0;
  control->gate_threshold_bits = seal->gate_threshold_bits;
  control->archive_count = 0;
  control->stop_requested = 0;
  control->current_store_index = current_store_index;
  control->reserved = 0;
}

__global__ void clear_exact_chunk_coverage_kernel_v1(
    std::uint8_t* exact_chunk_coverage_device,
    std::uint64_t population) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < population) {
    exact_chunk_coverage_device[index] = 0;
  }
}

__global__ void verify_exact_chunk_coverage_kernel_v1(
    const std::uint8_t* exact_chunk_coverage_device,
    std::uint64_t population,
    std::uint32_t* device_content_fault_device) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (*device_content_fault_device != 0u) {
    return;
  }
  if (index < population && exact_chunk_coverage_device[index] != 1) {
    atomicExch(device_content_fault_device, 1u);
  }
}

__global__ void seal_generation_content_kernel_v1(
    const NeoResidentGenerationGeneScalarV1* scalars,
    const std::uint64_t* indices,
    const double* weights,
    const NeoResidentGenerationMetricRowV1* metrics,
    const std::uint64_t* resident_decision_keys,
    const std::uint32_t* collision_fault,
    const std::uint32_t* content_fault,
    std::uint64_t* identities,
    NeoResidentGenerationPlanV1 plan) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  std::uint64_t gene_lanes[4] = {FNV_OFFSET_0_V1, FNV_OFFSET_1_V1,
                                 FNV_OFFSET_0_V1 ^ 0xa0761d6478bd642full,
                                 FNV_OFFSET_1_V1 ^ 0xe7037ed1a0b428dbull};
  std::uint64_t metric_lanes[4] = {FNV_OFFSET_1_V1, FNV_OFFSET_0_V1,
                                   FNV_OFFSET_1_V1 ^ 0x8ebc6af09c88c6e3ull,
                                   FNV_OFFSET_0_V1 ^ 0x589965cc75374cc3ull};
  for (std::uint64_t candidate = 0; candidate < plan.logical_population_count; ++candidate) {
    const std::uint64_t base = candidate * plan.max_terms_per_gene;
    const std::uint64_t hash = full_fixed_stride_gene_hash_v1(
        scalars[candidate], indices + base, weights + base, plan);
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
      gene_lanes[lane] = hash_mix_u64_v1(gene_lanes[lane], hash ^ (candidate + lane));
    }
    const NeoResidentGenerationMetricRowV1& row = metrics[candidate];
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
      metric_lanes[lane] =
          hash_mix_u64_v1(metric_lanes[lane], row.candidate_id ^ lane);
      metric_lanes[lane] = hash_mix_u64_v1(
          metric_lanes[lane], row.scenario_id ^ (candidate + lane));
      metric_lanes[lane] = hash_mix_u64_v1(
          metric_lanes[lane], resident_decision_keys[candidate] ^ lane);
    }
    for (std::uint32_t metric = 0; metric < METRIC_VALUE_COUNT_V1; ++metric) {
      const std::uint64_t bits = f64_bits_v1(row.values[metric]);
      for (std::uint32_t lane = 0; lane < 4; ++lane) {
        metric_lanes[lane] =
            hash_mix_u64_v1(metric_lanes[lane], bits ^ (metric + lane));
      }
    }
  }
  for (std::uint32_t index = 0; index < 32; ++index) {
    const std::uint64_t semantic_byte =
        static_cast<std::uint64_t>(plan.metric_semantics_sha256[index]) |
        (static_cast<std::uint64_t>(plan.scoring_semantics_sha256[index]) << 8) |
        (static_cast<std::uint64_t>(plan.novelty_semantics_sha256[index]) << 16) |
        (static_cast<std::uint64_t>(plan.scenario_order_semantics_sha256[index]) << 24) |
        (static_cast<std::uint64_t>(plan.rank_semantics_sha256[index]) << 32) |
        (static_cast<std::uint64_t>(plan.cuda_build_manifest_sha256[index]) << 40);
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
      metric_lanes[lane] = hash_mix_u64_v1(metric_lanes[lane], semantic_byte ^ lane);
    }
  }
  for (std::uint32_t lane = 0; lane < 4; ++lane) {
    identities[lane] = gene_lanes[lane];
    identities[4 + lane] = metric_lanes[lane];
    identities[8 + lane] = hash_mix_u64_v1(
        gene_lanes[lane], metric_lanes[lane] ^ plan.run_identity_sha256[lane]);
  }
  if (*collision_fault != 0 || *content_fault != 0) {
    identities[8] = 0;
  }
}

std::int32_t launch_status_v1() {
  return cuda_status_v1(cudaPeekAtLastError());
}

std::uint32_t grid_for_v1(std::uint64_t count) {
  constexpr std::uint32_t threads = 256;
  return static_cast<std::uint32_t>((count + threads - 1) / threads);
}

std::int32_t record_ready_event_v1(NeoResidentGenerationRunV1* run,
                                   NeoResidentGenerationReadyEventV1* ready) {
  publish_resident_generation_device_seal_v2<<<1, 1, 0, run->admitted_run_stream>>>(
      run->device_seal_v2, run->resident_control_device_v2,
      run->device_content_fault_device,
      run->gene_scalars_device, run->gene_indices_device, run->gene_weights_device,
      run->offspring_gene_scalars_device, run->offspring_gene_indices_device,
      run->offspring_gene_weights_device, run->smc_weights_device_v2, run->plan,
      run->run_token, run->current_generation_index, run->store_epoch_v2,
      run->current_store_index_v2, run->smc_gate_disabled_v2);
  ++run->same_stream_enqueue_count;
  const std::int32_t launch_status = launch_status_v1();
  if (launch_status != NEO_RESIDENT_STATUS_OK_V1) {
    return launch_status;
  }
  const cudaError_t event_status = cudaEventRecord(run->ready_event, run->admitted_run_stream);
  if (event_status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  ++run->next_event_id;
  ready->abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  ready->reserved = 0;
  ready->event_id = run->next_event_id;
  ready->generation_index = run->current_generation_index;
  ready->same_stream_enqueue_count = run->same_stream_enqueue_count;
  ready->intermediate_host_wait_count = 0;
  ready->intermediate_readback_count = 0;
  run->ready_receipt_token_v2 = ready;
  return NEO_RESIDENT_STATUS_OK_V1;
}

std::int32_t consume_resident_generation_event_dependency_v1(
    NeoResidentGenerationRunV1* run) {
  if (run == nullptr || run->ready_event == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  return cuda_status_v1(
      cudaStreamWaitEvent(run->admitted_run_stream, run->ready_event, 0));
}

std::int32_t launch_device_gene_hash_v1(NeoResidentGenerationRunV1* run,
                                        NeoResidentGenerationGeneScalarV1* scalars,
                                        std::uint64_t* indices,
                                        double* weights) {
  constexpr std::uint32_t threads = 256;
  gene_hash_kernel_v1<<<grid_for_v1(run->logical_population_count), threads, 0,
                        run->admitted_run_stream>>>(
      scalars, indices, weights, run->gene_hashes_a_device,
      run->dedup_values_a_device, run->plan,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  std::int32_t status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  std::size_t scratch = run->cub_scratch_bytes;
  status = cuda_status_v1(cub::DeviceRadixSort::SortPairs(
      run->cub_scratch_device, scratch, run->gene_hashes_a_device,
      run->gene_hashes_b_device, run->dedup_values_a_device,
      run->dedup_values_b_device, static_cast<int>(run->logical_population_count),
      0, 64, run->admitted_run_stream));
  ++run->same_stream_enqueue_count;
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  scratch = run->cub_scratch_bytes;
  status = cuda_status_v1(cub::DeviceRunLengthEncode::Encode(
      run->cub_scratch_device, scratch, run->gene_hashes_b_device,
      run->unique_gene_hashes_device, run->dedup_run_lengths_device,
      run->dedup_run_count_device, static_cast<int>(run->logical_population_count),
      run->admitted_run_stream));
  ++run->same_stream_enqueue_count;
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  verify_sorted_gene_dedup_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                       threads, 0, run->admitted_run_stream>>>(
      scalars, indices, weights, run->gene_hashes_b_device,
      run->dedup_values_b_device, run->dedup_flags_device,
      run->selection_flags_device, run->gene_hash_collision_fault_device,
      run->device_content_fault_device, run->plan);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  scratch = run->cub_scratch_bytes;
  status = cuda_status_v1(cub::DeviceSelect::Flagged(
      run->cub_scratch_device, scratch, run->dedup_values_b_device,
      run->dedup_flags_device, run->selected_indices_device,
      run->selected_count_device, static_cast<int>(run->logical_population_count),
      run->admitted_run_stream));
  ++run->same_stream_enqueue_count;
  return status;
}

std::int32_t launch_device_parent_selection_v1(NeoResidentGenerationRunV1* run,
                                                std::uint64_t generation_index) {
  constexpr std::uint32_t threads = 256;
  build_gene_identity_rank_keys_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                            threads, 0, run->admitted_run_stream>>>(
      run->gene_scalars_device, run->rank_keys_a_device,
      run->rank_values_a_device, run->logical_population_count,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  std::int32_t status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  std::size_t scratch = run->cub_scratch_bytes;
  status = cuda_status_v1(cub::DeviceRadixSort::SortPairs(
      run->cub_scratch_device, scratch, run->rank_keys_a_device,
      run->rank_keys_b_device, run->rank_values_a_device,
      run->rank_values_b_device, static_cast<int>(run->logical_population_count),
      0, 64, run->admitted_run_stream));
  ++run->same_stream_enqueue_count;
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  gather_resident_decision_rank_keys_kernel_v1<<<
      grid_for_v1(run->logical_population_count), threads, 0,
      run->admitted_run_stream>>>(
      run->resident_decision_keys_device, run->selection_flags_device,
      run->rank_values_b_device, run->rank_keys_a_device,
      run->rank_values_a_device, run->logical_population_count,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  scratch = run->cub_scratch_bytes;
  status = cuda_status_v1(cub::DeviceRadixSort::SortPairsDescending(
      run->cub_scratch_device, scratch, run->rank_keys_a_device,
      run->rank_keys_b_device, run->rank_values_a_device,
      run->rank_values_b_device, static_cast<int>(run->logical_population_count),
      0, 64, run->admitted_run_stream));
  ++run->same_stream_enqueue_count;
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  select_rank_weighted_parents_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                           threads, 0, run->admitted_run_stream>>>(
      run->rank_values_b_device, run->parent_a_device, run->parent_b_device,
      run->plan, generation_index, run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  if (run->plan.survivor_count != 0) {
    select_rank_weighted_survivors_kernel_v1<<<1, 1, 0,
                                               run->admitted_run_stream>>>(
        run->selected_indices_device, run->dedup_flags_device, run->plan,
        generation_index, run->device_content_fault_device);
    ++run->same_stream_enqueue_count;
    status = launch_status_v1();
    if (status != NEO_RESIDENT_STATUS_OK_V1) {
      return status;
    }
    scratch = run->cub_scratch_bytes;
    status = cuda_status_v1(cub::DeviceRadixSort::SortKeys(
        run->cub_scratch_device, scratch, run->selected_indices_device,
        run->rank_keys_a_device, static_cast<int>(run->plan.survivor_count),
        0, 64, run->admitted_run_stream));
    ++run->same_stream_enqueue_count;
    if (status != NEO_RESIDENT_STATUS_OK_V1) {
      return status;
    }
    gather_rank_weighted_survivors_kernel_v1<<<
        grid_for_v1(run->plan.survivor_count), threads, 0,
        run->admitted_run_stream>>>(run->rank_values_b_device,
                                    run->rank_keys_a_device,
                                    run->selected_indices_device,
                                    run->plan.survivor_count,
                                    run->device_content_fault_device);
    ++run->same_stream_enqueue_count;
    status = launch_status_v1();
  }
  return status;
}

std::int32_t launch_device_crossover_v1(NeoResidentGenerationRunV1* run,
                                         std::uint64_t generation_index) {
  constexpr std::uint32_t threads = 256;
  crossover_resident_genes_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                       threads, 0, run->admitted_run_stream>>>(
      run->gene_scalars_device, run->gene_indices_device, run->gene_weights_device,
      run->selected_indices_device, run->parent_a_device, run->parent_b_device,
      run->offspring_gene_scalars_device, run->offspring_gene_indices_device,
      run->offspring_gene_weights_device, run->plan, generation_index,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  return launch_status_v1();
}

std::int32_t launch_device_mutation_v1(NeoResidentGenerationRunV1* run,
                                        std::uint64_t generation_index) {
  constexpr std::uint32_t threads = 256;
  mutate_resident_genes_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                    threads, 0, run->admitted_run_stream>>>(
      run->offspring_gene_scalars_device, run->offspring_gene_indices_device,
      run->offspring_gene_weights_device, run->plan, generation_index,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  return launch_status_v1();
}

void rotate_resident_generation_stores_v1(NeoResidentGenerationRunV1* run) {
  auto* scalar = run->gene_scalars_device;
  run->gene_scalars_device = run->offspring_gene_scalars_device;
  run->offspring_gene_scalars_device = scalar;
  auto* index = run->gene_indices_device;
  run->gene_indices_device = run->offspring_gene_indices_device;
  run->offspring_gene_indices_device = index;
  auto* weight = run->gene_weights_device;
  run->gene_weights_device = run->offspring_gene_weights_device;
  run->offspring_gene_weights_device = weight;
  run->current_store_index_v2 ^= 1u;
}

constexpr bool strict_generation_has_no_candidate_revival_v1 = true;
constexpr bool exact_candidate_identity_without_fillers_v1 = true;

}  // namespace

extern "C" std::int32_t query_resident_generation_allocation_v1(
    const NeoResidentGenerationPopulationSessionImportV1* import,
    const NeoResidentGenerationPlanV1* plan,
    NeoResidentGenerationAllocationReceiptV1* receipt) {
  if (!validate_import_v1(import) || receipt == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (plan == nullptr || plan->abi_version != NEO_RESIDENT_GENERATION_ABI_V1) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  if (plan->parent_selection_policy != NEO_RESIDENT_PARENT_RANK_WEIGHTED_V1 ||
      plan->survivor_selection_policy != NEO_RESIDENT_SURVIVOR_RANK_WEIGHTED_V1) {
    return NEO_RESIDENT_STATUS_UNSUPPORTED_SELECTION_V1;
  }
  if (!validate_plan_v1(plan)) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (!identity_equal_v1(import->cuda_build_manifest_sha256,
                         plan->cuda_build_manifest_sha256)) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0 ||
      static_cast<std::uint32_t>(current_device) != import->selected_cuda_ordinal) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  std::size_t same_context_free_bytes = 0;
  std::size_t same_context_total_bytes = 0;
  if (cudaMemGetInfo(&same_context_free_bytes, &same_context_total_bytes) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  (void)same_context_total_bytes;
  return calculate_resident_generation_allocation_v2(
      plan, import->admitted_run_stream, same_context_free_bytes,
      import->full_discovery_reserve_bytes, receipt);
}

extern "C" std::int32_t calculate_resident_generation_allocation_v2(
    const NeoResidentGenerationPlanV1* plan,
    cudaStream_t admitted_run_stream,
    std::uint64_t same_context_free_bytes,
    std::uint64_t full_discovery_reserve_bytes,
    NeoResidentGenerationAllocationReceiptV1* receipt) {
  if (plan == nullptr || receipt == nullptr || admitted_run_stream == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (plan->abi_version != NEO_RESIDENT_GENERATION_ABI_V1) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  if (plan->parent_selection_policy != NEO_RESIDENT_PARENT_RANK_WEIGHTED_V1 ||
      plan->survivor_selection_policy != NEO_RESIDENT_SURVIVOR_RANK_WEIGHTED_V1) {
    return NEO_RESIDENT_STATUS_UNSUPPORTED_SELECTION_V1;
  }
  if (!validate_plan_v1(plan)) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  GenerationPhysicalLayoutV1 layout{};
  if (!checked_physical_layout_v1(*plan, admitted_run_stream, &layout)) {
    return NEO_RESIDENT_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  if (full_discovery_reserve_bytes > same_context_free_bytes ||
      layout.total_device_bytes >
          same_context_free_bytes - full_discovery_reserve_bytes) {
    return NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1;
  }
  std::memset(receipt, 0, sizeof(*receipt));
  receipt->abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  receipt->generation_store_allocation_count = 1;
  receipt->logical_gene_scalar_bytes = layout.logical_gene_scalar_bytes;
  receipt->logical_gene_index_bytes = layout.logical_gene_index_bytes;
  receipt->logical_gene_weight_bytes = layout.logical_gene_weight_bytes;
  receipt->offspring_bytes = layout.offspring_bytes;
  receipt->metric_row_bytes = layout.metric_row_bytes;
  receipt->rank_key_bytes = layout.rank_key_bytes;
  receipt->selection_bytes = layout.selection_bytes;
  receipt->dedup_hash_bytes = layout.dedup_hash_bytes;
  receipt->cub_scratch_bytes = layout.cub_scratch_bytes;
  receipt->retained_evaluation_workspace_bytes =
      layout.retained_evaluation_workspace_bytes;
  receipt->terminal_device_receipt_bytes =
      layout.terminal_device_receipt_bytes;
  receipt->total_device_bytes = layout.total_device_bytes;
  receipt->same_context_free_bytes = same_context_free_bytes;
  receipt->full_discovery_reserve_bytes = full_discovery_reserve_bytes;
  receipt->logical_population_count = plan->logical_population_count;
  receipt->retained_evaluation_capacity = plan->retained_evaluation_capacity;
  receipt->generation_chunk_count = layout.generation_chunk_count;
  copy_identity_v1(receipt->allocation_plan_sha256, plan->plan_identity_sha256);
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t create_resident_generation_run_from_import_v1(
    const NeoResidentGenerationPopulationSessionImportV1* import,
    const NeoResidentGenerationPlanV1* plan,
    const NeoResidentGenerationAllocationReceiptV1* receipt,
    NeoResidentGenerationRunV1** run) {
  if (!validate_import_v1(import) || !validate_plan_v1(plan) || receipt == nullptr ||
      run == nullptr || *run != nullptr ||
      receipt->abi_version != NEO_RESIDENT_GENERATION_ABI_V1 ||
      receipt->generation_store_allocation_count != 1 ||
      receipt->logical_population_count != plan->logical_population_count ||
      receipt->retained_evaluation_capacity != plan->retained_evaluation_capacity ||
      receipt->full_discovery_reserve_bytes != import->full_discovery_reserve_bytes ||
      !identity_equal_v1(import->cuda_build_manifest_sha256,
                         plan->cuda_build_manifest_sha256) ||
      !identity_equal_v1(receipt->allocation_plan_sha256, plan->plan_identity_sha256)) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  auto* created = new (std::nothrow) NeoResidentGenerationRunV1{};
  if (created == nullptr) {
    return NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1;
  }
  created->plan = *plan;
  created->allocation = *receipt;
  std::memcpy(created->cuda_device_identity_sha256,
              import->cuda_device_identity_sha256,
              sizeof(created->cuda_device_identity_sha256));
  std::memcpy(created->primary_context_identity_sha256,
              import->primary_context_identity_sha256,
              sizeof(created->primary_context_identity_sha256));
  std::memcpy(created->run_stream_identity_sha256,
              import->run_stream_identity_sha256,
              sizeof(created->run_stream_identity_sha256));
  created->admitted_run_stream = import->admitted_run_stream;
  created->resident_parent_ready_event = import->resident_parent_ready_event;
  created->ready_event = import->generation_ready_event;
  created->population_lifetime_owner = import->population_lifetime_owner;
  created->logical_population_count = plan->logical_population_count;
  created->retained_evaluation_capacity = plan->retained_evaluation_capacity;
  created->next_expected_logical_offset = 0;
  created->current_generation_index = 0;
  created->store_epoch_v2 = 0;
  created->same_stream_enqueue_count = 0;
  created->next_event_id = 0;
  created->completion_event_query_count_v2 = 0;
  created->sealed = false;
  created->post_ga_in_place_bound = false;
  created->current_store_index_v2 = 0;
  created->ready_receipt_token_v2 = nullptr;
  created->source_ready_receipt_token_v2 = nullptr;
  created->source_event_id_v2 = 0;
  created->source_same_stream_enqueue_count_v2 = 0;
  created->pending_receipt_token_v2 = nullptr;
  created->terminal_host_receipt_v2 = nullptr;
  created->initialized_v2 = false;
  created->evaluator_constants_configured_v2 = false;
  created->smc_gate_disabled_v2 = false;
  created->one_generation_advance_enqueued_v2 = false;
  created->one_generation_advance_pending_v2 = false;
  created->terminal_committed_v2 = false;
  created->terminal_event_proven_v2 = false;
  created->poisoned_v2 = false;
  created->allocation_free_issued_v2 = false;
  created->free_outcome_unknown_deliberate_leak_v2 = false;
#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
  created->fixture_duplicate_final_content_v2 = false;
  created->fixture_duplicate_source_v2 = 0;
  created->fixture_duplicate_destination_v2 = 0;
#endif
  created->run_token = FNV_OFFSET_0_V1;
  for (std::size_t index = 0; index < 32; ++index) {
    created->run_token =
        (created->run_token ^ plan->run_identity_sha256[index]) * FNV_PRIME_V1;
  }
  if (created->run_token == 0) {
    created->run_token = 1;
  }
  void* attempted_allocation = nullptr;
  cudaError_t status = cudaMallocAsync(
      &attempted_allocation,
      static_cast<std::size_t>(receipt->total_device_bytes),
      created->admitted_run_stream);
  if (status != cudaSuccess) {
    // Runtime allocation APIs may surface an earlier asynchronous fault. An
    // attempted output identity is deliberately discarded without query/free.
    attempted_allocation = nullptr;
    delete created;
    return NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2;
  }
  created->allocation_base = attempted_allocation;
  if (!partition_generation_allocation_v1(created)) {
    void* allocation_to_retire =
        retire_generation_allocation_identity_v2(created);
    const cudaError_t release_status =
        cudaFreeAsync(allocation_to_retire, created->admitted_run_stream);
    if (release_status != cudaSuccess) {
      created->poisoned_v2 = true;
      created->free_outcome_unknown_deliberate_leak_v2 = true;
      delete created;
      return NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2;
    }
    delete created;
    return NEO_RESIDENT_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  status = cudaStreamWaitEvent(created->admitted_run_stream,
                               created->resident_parent_ready_event, 0);
  if (status != cudaSuccess) {
    void* allocation_to_retire =
        retire_generation_allocation_identity_v2(created);
    const cudaError_t release_status =
        cudaFreeAsync(allocation_to_retire, created->admitted_run_stream);
    if (release_status != cudaSuccess) {
      created->poisoned_v2 = true;
      created->free_outcome_unknown_deliberate_leak_v2 = true;
      delete created;
      return NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2;
    }
    delete created;
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  created->same_stream_enqueue_count = 1;
  *run = created;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t bind_resident_search_terminal_receipt_v2(
    NeoResidentGenerationRunV1* run,
    NeoResidentSearchTerminalReceiptV2* pinned_host_receipt) {
  if (run == nullptr || pinned_host_receipt == nullptr || run->initialized_v2 ||
      run->terminal_host_receipt_v2 != nullptr ||
      run->terminal_device_receipt_v2 == nullptr) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  std::memset(pinned_host_receipt, 0, sizeof(*pinned_host_receipt));
  run->terminal_host_receipt_v2 = pinned_host_receipt;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t try_complete_resident_generation_advance_v2(
    NeoResidentGenerationRunV1* run,
    const NeoResidentSearchAdvancePendingReceiptV2* pending,
    NeoResidentGenerationReadyEventV1* committed_ready,
    NeoResidentSearchTerminalReceiptV2* terminal_copy) {
  if (run == nullptr || pending == nullptr || committed_ready == nullptr ||
      terminal_copy == nullptr || run->poisoned_v2 ||
      !run->one_generation_advance_pending_v2 ||
      run->pending_receipt_token_v2 != pending ||
      pending->abi_version !=
          resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2 ||
      pending->reserved != 0u ||
      pending->dependency_receipt_token != run->ready_receipt_token_v2 ||
      pending->terminal_host_receipt_token != run->terminal_host_receipt_v2 ||
      pending->completion_event_id != run->next_event_id ||
      pending->target_generation_index != 1ull ||
      pending->target_store_epoch != 2ull ||
      pending->target_store_index != 1ull ||
      pending->run_token != run->run_token) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  const cudaError_t query = cudaEventQuery(run->ready_event);
  ++run->completion_event_query_count_v2;
  if (query == cudaErrorNotReady) {
    return resident_generation_v2::NEO_RESIDENT_SEARCH_NOT_READY_V2;
  }
  if (query != cudaSuccess) {
    run->poisoned_v2 = true;
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  // The event was recorded after the compact D2H. Only now may the host read
  // the pinned destination; NotReady never writes memory owned by that DMA.
  *terminal_copy = *run->terminal_host_receipt_v2;
  const std::uint64_t device_reported_query_count =
      terminal_copy->completion_event_query_count;
  terminal_copy->completion_event_query_count =
      run->completion_event_query_count_v2;
  const bool exact_terminal =
      terminal_copy->abi_version ==
          resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2 &&
      terminal_copy->reserved == 0u &&
      terminal_copy->run_token == run->run_token &&
      terminal_copy->completion_event_id == pending->completion_event_id &&
      terminal_copy->same_stream_enqueue_count ==
          pending->same_stream_enqueue_count &&
      terminal_copy->compact_async_d2h_count == 1ull &&
      terminal_copy->compact_async_d2h_bytes == sizeof(*terminal_copy) &&
      device_reported_query_count == 0ull &&
      terminal_copy->completion_stream_synchronize_count == 0ull;
  if (!exact_terminal) {
    run->poisoned_v2 = true;
    run->one_generation_advance_pending_v2 = false;
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  const bool has_device_fault = terminal_copy->scoring_device_fault != 0u ||
                                terminal_copy->generation_device_fault != 0u ||
                                terminal_copy->control_fault_word != 0u ||
                                terminal_copy->stop_requested != 0u;
  const bool exact_fault_terminal =
      terminal_copy->terminal_status ==
          resident_generation_v2::NEO_RESIDENT_SEARCH_TERMINAL_FAULT_V2 &&
      has_device_fault && terminal_copy->generation_index == 0ull &&
      terminal_copy->store_epoch == 1ull &&
      terminal_copy->current_store_index == 0u;
  const bool exact_committed_terminal =
      terminal_copy->terminal_status ==
          resident_generation_v2::NEO_RESIDENT_SEARCH_TERMINAL_COMMITTED_V2 &&
      !has_device_fault && terminal_copy->generation_index == 1ull &&
      terminal_copy->store_epoch == 2ull &&
      terminal_copy->current_store_index == 1u;
  if (!exact_fault_terminal && !exact_committed_terminal) {
    run->poisoned_v2 = true;
    run->one_generation_advance_pending_v2 = false;
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  // Cleanup authority requires both event completion and an exact compact
  // receipt, including its fault/commit generation-store tuple. Event readiness
  // alone cannot authorize release/reuse.
  run->terminal_event_proven_v2 = true;
  run->one_generation_advance_pending_v2 = false;
  run->one_generation_advance_enqueued_v2 = true;
  run->pending_receipt_token_v2 = nullptr;
  if (exact_fault_terminal) {
    run->poisoned_v2 = true;
    return NEO_RESIDENT_STATUS_DEVICE_FAULT_V1;
  }
  rotate_resident_generation_stores_v1(run);
  run->current_generation_index = terminal_copy->generation_index;
  run->store_epoch_v2 = terminal_copy->store_epoch;
  run->current_store_index_v2 = terminal_copy->current_store_index;
  run->terminal_committed_v2 = true;
  committed_ready->abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  committed_ready->reserved = 0;
  committed_ready->event_id = terminal_copy->completion_event_id;
  committed_ready->generation_index = terminal_copy->generation_index;
  committed_ready->same_stream_enqueue_count =
      terminal_copy->same_stream_enqueue_count;
  committed_ready->intermediate_host_wait_count = 0;
  committed_ready->intermediate_readback_count = 0;
  run->ready_receipt_token_v2 = committed_ready;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t initialize_resident_generation_population_v1(
    NeoResidentGenerationRunV1* run,
    NeoResidentGenerationReadyEventV1* ready) {
  if (run == nullptr || ready == nullptr || run->sealed || run->initialized_v2) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  if (cudaMemsetAsync(run->device_seal_v2, 0,
                      sizeof(NeoResidentGenerationDeviceSealV2),
                      run->admitted_run_stream) != cudaSuccess ||
      cudaMemsetAsync(run->resident_control_device_v2, 0,
                      sizeof(NeoResidentSearchDeviceControlV2),
                      run->admitted_run_stream) != cudaSuccess ||
      cudaMemsetAsync(run->smc_weights_device_v2, 0, 11 * sizeof(double),
                      run->admitted_run_stream) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  run->same_stream_enqueue_count += 3;
  constexpr std::uint32_t threads = 256;
  clear_generation_metadata_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                        threads, 0, run->admitted_run_stream>>>(
      run->exact_chunk_coverage_device, run->selection_flags_device,
      run->selected_count_device, run->dedup_run_count_device,
      run->gene_hash_collision_fault_device, run->device_content_fault_device,
      run->logical_population_count);
  ++run->same_stream_enqueue_count;
  std::int32_t status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  initialize_fixed_stride_population_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                                  threads, 0,
                                                  run->admitted_run_stream>>>(
      run->gene_scalars_device, run->gene_indices_device,
      run->gene_weights_device, run->plan);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_gene_hash_v1(run, run->gene_scalars_device,
                                      run->gene_indices_device,
                                      run->gene_weights_device);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  (void)strict_generation_has_no_candidate_revival_v1;
  (void)exact_candidate_identity_without_fillers_v1;
  run->initialized_v2 = true;
  run->store_epoch_v2 = 1;
  return record_ready_event_v1(run, ready);
}

extern "C" std::int32_t enqueue_exact_generation_chunk_v1(
    NeoResidentGenerationRunV1* run,
    const NeoResidentGenerationMetricRowsImportV1* metrics,
    NeoResidentGenerationReadyEventV1* ready) {
  const std::uint64_t logical_offset = metrics == nullptr ? 0 : metrics->logical_offset;
  const std::uint64_t active_scenarios = metrics == nullptr ? 0 : metrics->active_scenarios;
  if (run == nullptr || metrics == nullptr || ready == nullptr || run->sealed ||
      metrics->abi_version != NEO_RESIDENT_GENERATION_ABI_V1 ||
      metrics->metric_value_count != METRIC_VALUE_COUNT_V1 ||
      metrics->metric_rows_device == nullptr ||
      metrics->resident_decision_keys_device == nullptr ||
      metrics->expected_scenario_ids_device == nullptr ||
      metrics->scoring_novelty_ready_event == nullptr ||
      !identity_equal_v1(metrics->metric_semantics_sha256,
                         run->plan.metric_semantics_sha256) ||
      !identity_equal_v1(metrics->scoring_semantics_sha256,
                         run->plan.scoring_semantics_sha256) ||
      !identity_equal_v1(metrics->novelty_semantics_sha256,
                         run->plan.novelty_semantics_sha256) ||
      !identity_equal_v1(metrics->scenario_order_semantics_sha256,
                         run->plan.scenario_order_semantics_sha256) ||
      !identity_equal_v1(metrics->rank_semantics_sha256,
                         run->plan.rank_semantics_sha256) ||
      metrics->active_scenarios == 0 ||
      !(active_scenarios <= run->retained_evaluation_capacity) ||
      logical_offset != run->next_expected_logical_offset ||
      logical_offset > run->logical_population_count ||
      active_scenarios > run->logical_population_count - logical_offset ||
      !(logical_offset + active_scenarios <= run->logical_population_count)) {
    return NEO_RESIDENT_STATUS_RANGE_ERROR_V1;
  }
  cudaError_t cuda_status = cudaStreamWaitEvent(run->admitted_run_stream,
                                                metrics->scoring_novelty_ready_event, 0);
  if (cuda_status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  constexpr std::uint32_t threads = 256;
  validate_and_import_scored_rows_kernel_v1<<<
      grid_for_v1(active_scenarios), threads, 0, run->admitted_run_stream>>>(
      metrics->metric_rows_device, metrics->resident_decision_keys_device,
      metrics->expected_scenario_ids_device, run->gene_scalars_device,
      run->metric_rows_device, run->resident_decision_keys_device,
      logical_offset, active_scenarios, run->exact_chunk_coverage_device,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  const std::int32_t status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  run->next_expected_logical_offset += active_scenarios;
  return record_ready_event_v1(run, ready);
}

extern "C" std::int32_t enqueue_resident_rank_selection_offspring_v1(
    NeoResidentGenerationRunV1* run,
    std::uint64_t generation_index,
    NeoResidentGenerationReadyEventV1* ready) {
  if (run == nullptr || ready == nullptr || run->sealed ||
      generation_index != run->current_generation_index ||
      generation_index >= run->plan.generation_count ||
      run->next_expected_logical_offset != run->logical_population_count) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  std::int32_t status = consume_resident_generation_event_dependency_v1(run);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  ++run->same_stream_enqueue_count;
  constexpr std::uint32_t threads = 256;
  verify_exact_chunk_coverage_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                          threads, 0,
                                          run->admitted_run_stream>>>(
      run->exact_chunk_coverage_device, run->logical_population_count,
      run->device_content_fault_device);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_parent_selection_v1(run, generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_crossover_v1(run, generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_mutation_v1(run, generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_gene_hash_v1(run, run->offspring_gene_scalars_device,
                                      run->offspring_gene_indices_device,
                                      run->offspring_gene_weights_device);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  rotate_resident_generation_stores_v1(run);
  ++run->current_generation_index;
  ++run->store_epoch_v2;
  run->next_expected_logical_offset = 0;
  clear_exact_chunk_coverage_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                         threads, 0, run->admitted_run_stream>>>(
      run->exact_chunk_coverage_device, run->logical_population_count);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  return record_ready_event_v1(run, ready);
}

extern "C" std::int32_t seal_resident_generation_content_v1(
    NeoResidentGenerationRunV1* run,
    NeoResidentGenerationContentReceiptV1* receipt,
    NeoResidentGenerationReadyEventV1* ready) {
  if (run == nullptr || receipt == nullptr || ready == nullptr || run->sealed ||
      run->next_expected_logical_offset != run->logical_population_count) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  std::int32_t status = consume_resident_generation_event_dependency_v1(run);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  ++run->same_stream_enqueue_count;
  seal_generation_content_kernel_v1<<<1, 1, 0, run->admitted_run_stream>>>(
      run->gene_scalars_device, run->gene_indices_device,
      run->gene_weights_device, run->metric_rows_device,
      run->resident_decision_keys_device,
      run->gene_hash_collision_fault_device, run->device_content_fault_device,
      run->content_identities_device, run->plan);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  run->sealed = true;
  status = record_ready_event_v1(run, ready);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  receipt->abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  receipt->reserved = 0;
  receipt->gene_content_identity_handle =
      generation_content_identity_handle_v1(run->run_token, 1);
  receipt->metric_content_identity_handle =
      generation_content_identity_handle_v1(run->run_token, 2);
  receipt->generation_receipt_identity_handle =
      generation_content_identity_handle_v1(run->run_token, 3);
  receipt->ready_event_id = ready->event_id;
  receipt->final_compact_readback_count = 0;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t begin_resident_post_ga_in_place_v1(
    NeoResidentGenerationRunV1* run,
    const NeoResidentGenerationReadyEventV1* dependency,
    std::uint64_t gene_content_identity_handle,
    std::uint64_t metric_content_identity_handle,
    std::uint64_t generation_receipt_identity_handle,
    NeoResidentGenerationPostGaInPlaceReceiptV1* receipt) {
  const bool sealed_generation_ready_for_post_ga =
      run != nullptr && run->sealed && !run->post_ga_in_place_bound;
  if (run == nullptr || dependency == nullptr || receipt == nullptr ||
      dependency->abi_version != NEO_RESIDENT_GENERATION_ABI_V1 ||
      !sealed_generation_ready_for_post_ga || run->allocation_base == nullptr ||
      run->admitted_run_stream == nullptr || run->ready_event == nullptr ||
      run->gene_scalars_device == nullptr || run->gene_indices_device == nullptr ||
      run->gene_weights_device == nullptr || run->metric_rows_device == nullptr ||
      run->resident_decision_keys_device == nullptr) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  const bool exact_ready_event = dependency->event_id == run->next_event_id;
  const bool exact_generation =
      dependency->generation_index == run->current_generation_index;
  const bool exact_enqueue_count =
      dependency->same_stream_enqueue_count == run->same_stream_enqueue_count;
  const bool no_host_crossing = dependency->intermediate_host_wait_count == 0 &&
                                dependency->intermediate_readback_count == 0;
  const bool exact_content =
      gene_content_identity_handle ==
          generation_content_identity_handle_v1(run->run_token, 1) &&
      metric_content_identity_handle ==
          generation_content_identity_handle_v1(run->run_token, 2) &&
      generation_receipt_identity_handle ==
          generation_content_identity_handle_v1(run->run_token, 3);
  if (!exact_ready_event || !exact_generation || !exact_enqueue_count ||
      !no_host_crossing || !exact_content) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  const std::int32_t status =
      consume_resident_generation_event_dependency_v1(run);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  run->post_ga_in_place_bound = true;
  receipt->abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  receipt->reserved = 0;
  receipt->ready_event_id = dependency->event_id;
  receipt->current_generation_index = run->current_generation_index;
  receipt->same_stream_enqueue_count = run->same_stream_enqueue_count;
  receipt->logical_population_count = run->logical_population_count;
  receipt->retained_evaluation_capacity = run->retained_evaluation_capacity;
  receipt->generation_allocation_total_device_bytes =
      run->allocation.total_device_bytes;
  receipt->additional_allocation_count = 0;
  receipt->additional_device_bytes = 0;
  receipt->gene_content_identity_handle = gene_content_identity_handle;
  receipt->metric_content_identity_handle = metric_content_identity_handle;
  receipt->generation_receipt_identity_handle =
      generation_receipt_identity_handle;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t enqueue_resident_generation_release_v1(
    NeoResidentGenerationRunV1* run) {
  if (run == nullptr || run->allocation_base == nullptr ||
      run->admitted_run_stream == nullptr ||
      run->allocation_free_issued_v2) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (run->one_generation_advance_pending_v2 ||
      (run->poisoned_v2 && !run->terminal_event_proven_v2)) {
    // In-flight or fault-reachable storage is intentionally leaked rather than
    // being reused/freed before its admitted stream reaches a safe boundary.
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  void* allocation_to_retire =
      retire_generation_allocation_identity_v2(run);
  const cudaError_t release_status =
      cudaFreeAsync(allocation_to_retire, run->admitted_run_stream);
  if (release_status != cudaSuccess) {
    run->poisoned_v2 = true;
    run->free_outcome_unknown_deliberate_leak_v2 = true;
    return NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2;
  }
  run->ready_event = nullptr;
  delete run;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t detach_resident_search_terminal_receipt_v2(
    NeoResidentGenerationRunV1* run,
    const void* expected_terminal_host_receipt) {
  if (run == nullptr || expected_terminal_host_receipt == nullptr ||
      static_cast<const void*>(run->terminal_host_receipt_v2) !=
          expected_terminal_host_receipt ||
      run->one_generation_advance_pending_v2 ||
      (run->poisoned_v2 && !run->terminal_event_proven_v2)) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  // The population session remains the allocation owner until cudaFreeHost
  // succeeds. Detaching first keeps the native generation owner valid if that
  // host release fails and Rust must retain the handle fail-closed.
  run->terminal_host_receipt_v2 = nullptr;
  return NEO_RESIDENT_STATUS_OK_V1;
}

}  // namespace neoethos::resident_generation_v1

namespace neoethos::resident_generation_v2_internal {

std::int32_t enqueue_resident_generation_offspring_from_scored_rows_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const ResidentGenerationScoredRowsV2* scored_rows,
    ResidentGenerationPreparedAdvanceV2* prepared) {
  using namespace resident_generation_v1;
  using namespace resident_scoring_novelty_v1;

  if (generation == nullptr || scored_rows == nullptr || prepared == nullptr ||
      scored_rows->sealed_scoring_rows == nullptr ||
      scored_rows->ranked_decision_keys_device == nullptr ||
      scored_rows->retained_generation_view == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  const auto* scored = scored_rows->sealed_scoring_rows;
  const auto* retained_view = scored_rows->retained_generation_view;
  if (scored->abi_version != NEO_RESIDENT_SCORING_NOVELTY_ABI_V1 ||
      scored->reserved != 0u || scored->metric_rows_device == nullptr ||
      scored->expected_scenario_ids_device == nullptr ||
      scored->device_seal == nullptr ||
      scored->scoring_novelty_ready_event == nullptr ||
      scored->logical_population_count != generation->logical_population_count ||
      scored->intermediate_host_wait_count != 0ull ||
      scored->intermediate_readback_count != 0ull ||
      retained_view->abi_version !=
          resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  const bool exact_retained_generation_identity =
      retained_view->flags == 0u &&
      retained_view->seal_device == generation->device_seal_v2 &&
      retained_view->control_device == generation->resident_control_device_v2 &&
      retained_view->expected_run_token == generation->run_token &&
      retained_view->expected_generation_index ==
          generation->current_generation_index &&
      retained_view->expected_store_epoch == generation->store_epoch_v2 &&
      retained_view->logical_population_count ==
          generation->logical_population_count &&
      retained_view->feature_count == generation->plan.feature_count &&
      retained_view->max_terms_per_gene ==
          generation->plan.max_terms_per_gene &&
      retained_view->smc_flag_count == generation->plan.smc_flag_count &&
      std::memcmp(retained_view->plan_identity_sha256,
                  generation->plan.plan_identity_sha256, 32) == 0 &&
      std::memcmp(retained_view->generation_semantics_sha256,
                  generation->plan.generation_semantics_sha256, 32) == 0;
  const bool exact_scoring_semantics =
      std::memcmp(scored->metric_semantics_sha256,
                  generation->plan.metric_semantics_sha256, 32) == 0 &&
      std::memcmp(scored->scoring_semantics_sha256,
                  generation->plan.scoring_semantics_sha256, 32) == 0 &&
      std::memcmp(scored->novelty_semantics_sha256,
                  generation->plan.novelty_semantics_sha256, 32) == 0 &&
      std::memcmp(scored->scenario_order_semantics_sha256,
                  generation->plan.scenario_order_semantics_sha256, 32) == 0 &&
      std::memcmp(scored->rank_semantics_sha256,
                  generation->plan.rank_semantics_sha256, 32) == 0 &&
      std::memcmp(scored->cuda_build_manifest_sha256,
                  generation->plan.cuda_build_manifest_sha256, 32) == 0 &&
      std::memcmp(
          scored->cuda_math_flags_sha256,
          resident_scoring_novelty_v1::
              NEO_RESIDENT_CUDA_MATH_SEMANTICS_SHA256_V2,
          32) == 0;
  if (!exact_retained_generation_identity || !exact_scoring_semantics) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  if (!generation->initialized_v2 ||
      !generation->evaluator_constants_configured_v2 || generation->sealed ||
      generation->poisoned_v2 ||
      generation->one_generation_advance_enqueued_v2 ||
      generation->one_generation_advance_pending_v2 ||
      generation->terminal_committed_v2 ||
      generation->allocation_base == nullptr ||
      generation->admitted_run_stream == nullptr ||
      generation->device_seal_v2 == nullptr ||
      generation->resident_control_device_v2 == nullptr ||
      generation->device_content_fault_device == nullptr ||
      generation->gene_hash_collision_fault_device == nullptr ||
      generation->current_store_index_v2 > 1u ||
      generation->current_generation_index >= generation->plan.generation_count ||
      generation->store_epoch_v2 == ~std::uint64_t{0}) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }

  using GenerationGeneV1 = NeoResidentGenerationGeneScalarV1;
  using ScoringGeneV1 = NeoResidentScoringNoveltyGeneScalarV1;
  using GenerationMetricV1 = NeoResidentGenerationMetricRowV1;
  using ScoringMetricV1 = NeoResidentScoringNoveltyMetricRowV1;
  static_assert(sizeof(GenerationGeneV1) == sizeof(ScoringGeneV1));
  static_assert(alignof(GenerationGeneV1) == alignof(ScoringGeneV1));
  static_assert(sizeof(GenerationMetricV1) == sizeof(ScoringMetricV1));
  static_assert(alignof(GenerationMetricV1) == alignof(ScoringMetricV1));

  constexpr std::uint32_t threads = 256;
  std::int32_t status = NEO_RESIDENT_STATUS_OK_V1;
  if (generation->current_generation_index != 0ull) {
    clear_exact_chunk_coverage_kernel_v1<<<
        grid_for_v1(generation->logical_population_count), threads, 0,
        generation->admitted_run_stream>>>(
        generation->exact_chunk_coverage_device,
        generation->logical_population_count);
    ++generation->same_stream_enqueue_count;
    status = launch_status_v1();
    if (status != NEO_RESIDENT_STATUS_OK_V1) {
      return status;
    }
  }

  promote_scoring_device_seal_kernel_v2<<<
      1, 1, 0, generation->admitted_run_stream>>>(
      scored->device_seal, generation->device_content_fault_device);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  validate_and_import_scored_rows_kernel_v1<<<
      grid_for_v1(generation->logical_population_count), threads, 0,
      generation->admitted_run_stream>>>(
      scored->metric_rows_device,
      scored_rows->ranked_decision_keys_device,
      scored->expected_scenario_ids_device, generation->gene_scalars_device,
      generation->metric_rows_device,
      generation->resident_decision_keys_device, 0,
      generation->logical_population_count,
      generation->exact_chunk_coverage_device,
      generation->device_content_fault_device);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  verify_exact_chunk_coverage_kernel_v1<<<
      grid_for_v1(generation->logical_population_count), threads, 0,
      generation->admitted_run_stream>>>(
      generation->exact_chunk_coverage_device,
      generation->logical_population_count,
      generation->device_content_fault_device);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_parent_selection_v1(
      generation, generation->current_generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  if (generation->plan.survivor_count != 0 &&
      cudaMemcpyAsync(generation->rank_keys_a_device,
                      generation->selected_indices_device,
                      static_cast<std::size_t>(generation->plan.survivor_count) *
                          sizeof(std::uint64_t),
                      cudaMemcpyDeviceToDevice,
                      generation->admitted_run_stream) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  generation->same_stream_enqueue_count +=
      generation->plan.survivor_count == 0 ? 0ull : 1ull;
  status = launch_device_crossover_v1(
      generation, generation->current_generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_mutation_v1(
      generation, generation->current_generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
  if (generation->fixture_duplicate_final_content_v2) {
    fixture_duplicate_final_gene_content_kernel_v2<<<
        1, 1, 0, generation->admitted_run_stream>>>(
        generation->offspring_gene_scalars_device,
        generation->offspring_gene_indices_device,
        generation->offspring_gene_weights_device,
        generation->fixture_duplicate_source_v2,
        generation->fixture_duplicate_destination_v2,
        generation->plan.max_terms_per_gene);
    ++generation->same_stream_enqueue_count;
    status = launch_status_v1();
    if (status != NEO_RESIDENT_STATUS_OK_V1) {
      return status;
    }
  }
#endif
  status = launch_device_gene_hash_v1(
      generation, generation->offspring_gene_scalars_device,
      generation->offspring_gene_indices_device,
      generation->offspring_gene_weights_device);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }

  *prepared = {};
  prepared->generation_owner_ = generation;
  prepared->admitted_run_stream_ = generation->admitted_run_stream;
  prepared->device_seal_identity_ = generation->device_seal_v2;
  prepared->device_control_ = generation->resident_control_device_v2;
  prepared->scoring_device_seal_ = scored->device_seal;
  prepared->retained_generation_view_ =
      scored_rows->retained_generation_view;
  prepared->device_content_fault_ = generation->device_content_fault_device;
  prepared->gene_hash_collision_fault_ =
      generation->gene_hash_collision_fault_device;
  prepared->expected_old_generation_index_ =
      generation->current_generation_index;
  prepared->expected_next_generation_index_ =
      generation->current_generation_index + 1ull;
  prepared->expected_old_store_epoch_ = generation->store_epoch_v2;
  prepared->expected_next_store_epoch_ = generation->store_epoch_v2 + 1ull;
  prepared->run_token_ = generation->run_token;
  prepared->same_stream_enqueue_count_ =
      generation->same_stream_enqueue_count;
  prepared->expected_old_store_index_ = generation->current_store_index_v2;
  prepared->expected_next_store_index_ =
      generation->current_store_index_v2 ^ 1u;
  return NEO_RESIDENT_STATUS_OK_V1;
}

std::int32_t enqueue_resident_generation_offspring_from_finite_rows_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const resident_scoring_novelty_v2_internal::ResidentScoringFiniteObjectiveRowsV2*
        finite_rows,
    const std::uint64_t* ranked_decision_keys_device,
    resident_generation_v2::NeoResidentGenerationGeneViewV2*
        retained_generation_view,
    ResidentGenerationPreparedAdvanceV2* prepared) {
  using namespace resident_generation_v1;
  using namespace resident_scoring_novelty_v1;

  if (generation == nullptr || finite_rows == nullptr ||
      ranked_decision_keys_device == nullptr ||
      retained_generation_view == nullptr || prepared == nullptr ||
      finite_rows->scoring_owner == nullptr ||
      finite_rows->admitted_run_stream == nullptr ||
      finite_rows->metric_rows_device == nullptr ||
      finite_rows->expected_scenario_ids_device == nullptr ||
      finite_rows->fitness_scores_device == nullptr ||
      finite_rows->decision_keys_device == nullptr ||
      finite_rows->device_seal == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (finite_rows->admitted_run_stream != generation->admitted_run_stream ||
      finite_rows->logical_population_count !=
          generation->logical_population_count ||
      finite_rows->logical_population_count !=
          generation->retained_evaluation_capacity ||
      retained_generation_view->abi_version !=
          resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  const bool exact_retained_generation_identity =
      retained_generation_view->flags == 0u &&
      retained_generation_view->seal_device == generation->device_seal_v2 &&
      retained_generation_view->control_device ==
          generation->resident_control_device_v2 &&
      retained_generation_view->expected_run_token == generation->run_token &&
      retained_generation_view->expected_generation_index ==
          generation->current_generation_index &&
      retained_generation_view->expected_store_epoch ==
          generation->store_epoch_v2 &&
      retained_generation_view->logical_population_count ==
          generation->logical_population_count &&
      retained_generation_view->feature_count ==
          generation->plan.feature_count &&
      retained_generation_view->max_terms_per_gene ==
          generation->plan.max_terms_per_gene &&
      retained_generation_view->smc_flag_count ==
          generation->plan.smc_flag_count &&
      std::memcmp(retained_generation_view->plan_identity_sha256,
                  generation->plan.plan_identity_sha256, 32) == 0 &&
      std::memcmp(retained_generation_view->generation_semantics_sha256,
                  generation->plan.generation_semantics_sha256, 32) == 0;
  const bool exact_scoring_semantics =
      std::memcmp(finite_rows->metric_semantics_sha256,
                  generation->plan.metric_semantics_sha256, 32) == 0 &&
      std::memcmp(finite_rows->scoring_semantics_sha256,
                  generation->plan.scoring_semantics_sha256, 32) == 0 &&
      std::memcmp(finite_rows->novelty_semantics_sha256,
                  generation->plan.novelty_semantics_sha256, 32) == 0 &&
      std::memcmp(finite_rows->scenario_order_semantics_sha256,
                  generation->plan.scenario_order_semantics_sha256, 32) == 0 &&
      std::memcmp(finite_rows->rank_semantics_sha256,
                  generation->plan.rank_semantics_sha256, 32) == 0 &&
      std::memcmp(finite_rows->cuda_build_manifest_sha256,
                  generation->plan.cuda_build_manifest_sha256, 32) == 0 &&
      std::memcmp(
          finite_rows->cuda_math_flags_sha256,
          resident_scoring_novelty_v1::
              NEO_RESIDENT_CUDA_MATH_SEMANTICS_SHA256_V2,
          32) == 0;
  if (!exact_retained_generation_identity || !exact_scoring_semantics) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }
  if (!generation->initialized_v2 ||
      !generation->evaluator_constants_configured_v2 || generation->sealed ||
      generation->poisoned_v2 ||
      generation->one_generation_advance_enqueued_v2 ||
      generation->one_generation_advance_pending_v2 ||
      generation->terminal_committed_v2 || generation->allocation_base == nullptr ||
      generation->admitted_run_stream == nullptr ||
      generation->device_seal_v2 == nullptr ||
      generation->resident_control_device_v2 == nullptr ||
      generation->device_content_fault_device == nullptr ||
      generation->gene_hash_collision_fault_device == nullptr ||
      generation->current_store_index_v2 > 1u ||
      generation->current_generation_index >= generation->plan.generation_count ||
      generation->store_epoch_v2 == ~std::uint64_t{0}) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }

  using GenerationMetricV1 = NeoResidentGenerationMetricRowV1;
  using ScoringMetricV1 = NeoResidentScoringNoveltyMetricRowV1;
  static_assert(sizeof(GenerationMetricV1) == sizeof(ScoringMetricV1));
  static_assert(alignof(GenerationMetricV1) == alignof(ScoringMetricV1));

  constexpr std::uint32_t threads = 256;
  std::int32_t status = NEO_RESIDENT_STATUS_OK_V1;
  if (generation->current_generation_index != 0ull) {
    clear_exact_chunk_coverage_kernel_v1<<<
        grid_for_v1(generation->logical_population_count), threads, 0,
        generation->admitted_run_stream>>>(
        generation->exact_chunk_coverage_device,
        generation->logical_population_count);
    ++generation->same_stream_enqueue_count;
    status = launch_status_v1();
    if (status != NEO_RESIDENT_STATUS_OK_V1) {
      return status;
    }
  }
  promote_scoring_device_seal_kernel_v2<<<
      1, 1, 0, generation->admitted_run_stream>>>(
      finite_rows->device_seal, generation->device_content_fault_device);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  validate_and_import_scored_rows_kernel_v1<<<
      grid_for_v1(generation->logical_population_count), threads, 0,
      generation->admitted_run_stream>>>(
      finite_rows->metric_rows_device, ranked_decision_keys_device,
      finite_rows->expected_scenario_ids_device,
      generation->gene_scalars_device, generation->metric_rows_device,
      generation->resident_decision_keys_device, 0,
      generation->logical_population_count,
      generation->exact_chunk_coverage_device,
      generation->device_content_fault_device);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  verify_exact_chunk_coverage_kernel_v1<<<
      grid_for_v1(generation->logical_population_count), threads, 0,
      generation->admitted_run_stream>>>(
      generation->exact_chunk_coverage_device,
      generation->logical_population_count,
      generation->device_content_fault_device);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_parent_selection_v1(
      generation, generation->current_generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  if (generation->plan.survivor_count != 0 &&
      cudaMemcpyAsync(generation->rank_keys_a_device,
                      generation->selected_indices_device,
                      static_cast<std::size_t>(generation->plan.survivor_count) *
                          sizeof(std::uint64_t),
                      cudaMemcpyDeviceToDevice,
                      generation->admitted_run_stream) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  generation->same_stream_enqueue_count +=
      generation->plan.survivor_count == 0 ? 0ull : 1ull;
  status = launch_device_crossover_v1(
      generation, generation->current_generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = launch_device_mutation_v1(
      generation, generation->current_generation_index);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
  if (generation->fixture_duplicate_final_content_v2) {
    fixture_duplicate_final_gene_content_kernel_v2<<<
        1, 1, 0, generation->admitted_run_stream>>>(
        generation->offspring_gene_scalars_device,
        generation->offspring_gene_indices_device,
        generation->offspring_gene_weights_device,
        generation->fixture_duplicate_source_v2,
        generation->fixture_duplicate_destination_v2,
        generation->plan.max_terms_per_gene);
    ++generation->same_stream_enqueue_count;
    status = launch_status_v1();
    if (status != NEO_RESIDENT_STATUS_OK_V1) {
      return status;
    }
  }
#endif
  status = launch_device_gene_hash_v1(
      generation, generation->offspring_gene_scalars_device,
      generation->offspring_gene_indices_device,
      generation->offspring_gene_weights_device);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }

  *prepared = {};
  prepared->generation_owner_ = generation;
  prepared->admitted_run_stream_ = generation->admitted_run_stream;
  prepared->device_seal_identity_ = generation->device_seal_v2;
  prepared->device_control_ = generation->resident_control_device_v2;
  prepared->scoring_device_seal_ = finite_rows->device_seal;
  prepared->retained_generation_view_ = retained_generation_view;
  prepared->device_content_fault_ = generation->device_content_fault_device;
  prepared->gene_hash_collision_fault_ =
      generation->gene_hash_collision_fault_device;
  prepared->expected_old_generation_index_ =
      generation->current_generation_index;
  prepared->expected_next_generation_index_ =
      generation->current_generation_index + 1ull;
  prepared->expected_old_store_epoch_ = generation->store_epoch_v2;
  prepared->expected_next_store_epoch_ = generation->store_epoch_v2 + 1ull;
  prepared->run_token_ = generation->run_token;
  prepared->same_stream_enqueue_count_ =
      generation->same_stream_enqueue_count;
  prepared->expected_old_store_index_ = generation->current_store_index_v2;
  prepared->expected_next_store_index_ =
      generation->current_store_index_v2 ^ 1u;
  return NEO_RESIDENT_STATUS_OK_V1;
}

std::int32_t accept_resident_generation_combined_publish_v2(
    const ResidentGenerationPreparedAdvanceV2* prepared) {
  using namespace resident_generation_v1;
  if (prepared == nullptr || prepared->generation_owner_ == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  auto* generation = prepared->generation_owner_;
  const bool exact =
      !generation->one_generation_advance_pending_v2 &&
      !generation->one_generation_advance_enqueued_v2 &&
      !generation->terminal_committed_v2 && !generation->poisoned_v2 &&
      prepared->admitted_run_stream_ == generation->admitted_run_stream &&
      prepared->device_seal_identity_ == generation->device_seal_v2 &&
      prepared->device_control_ == generation->resident_control_device_v2 &&
      prepared->retained_generation_view_ != nullptr &&
      prepared->retained_generation_view_->expected_generation_index ==
          prepared->expected_old_generation_index_ &&
      prepared->retained_generation_view_->expected_store_epoch ==
          prepared->expected_old_store_epoch_ &&
      prepared->device_content_fault_ ==
          generation->device_content_fault_device &&
      prepared->gene_hash_collision_fault_ ==
          generation->gene_hash_collision_fault_device &&
      prepared->run_token_ == generation->run_token &&
      prepared->expected_old_generation_index_ ==
          generation->current_generation_index &&
      prepared->expected_old_store_epoch_ == generation->store_epoch_v2 &&
      prepared->expected_old_store_index_ ==
          generation->current_store_index_v2 &&
      prepared->expected_next_generation_index_ ==
          generation->current_generation_index + 1ull &&
      prepared->expected_next_store_epoch_ == generation->store_epoch_v2 + 1ull &&
      prepared->expected_next_store_index_ ==
          (generation->current_store_index_v2 ^ 1u) &&
      prepared->same_stream_enqueue_count_ ==
          generation->same_stream_enqueue_count;
  if (!exact) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }

  // The composite caller invokes this only after its single final publish
  // kernel was accepted by the same stream. This is planned host bookkeeping,
  // not a host claim that the device committed: a device fault leaves the seal
  // on the old tuple, so the next proof's exact device-identity check fails
  // closed. No host completion boundary is introduced here.
  ++generation->same_stream_enqueue_count;
  rotate_resident_generation_stores_v1(generation);
  generation->current_generation_index =
      prepared->expected_next_generation_index_;
  generation->store_epoch_v2 = prepared->expected_next_store_epoch_;
  generation->current_store_index_v2 =
      prepared->expected_next_store_index_;
  prepared->retained_generation_view_->expected_generation_index =
      prepared->expected_next_generation_index_;
  prepared->retained_generation_view_->expected_store_epoch =
      prepared->expected_next_store_epoch_;
  generation->next_expected_logical_offset = 0;
  generation->ready_receipt_token_v2 = nullptr;
  return NEO_RESIDENT_STATUS_OK_V1;
}

bool borrow_resident_generation_terminal_lifecycle_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    std::uint64_t expected_terminal_host_receipt_bytes,
    ResidentGenerationTerminalLifecycleV2* lifecycle) {
  using resident_generation_v2::NeoResidentSearchTerminalReceiptV2;
  if (run == nullptr || lifecycle == nullptr ||
      expected_terminal_host_receipt_bytes !=
          sizeof(NeoResidentSearchTerminalReceiptV2) ||
      run->admitted_run_stream == nullptr || run->ready_event == nullptr ||
      run->resident_parent_ready_event == nullptr ||
      run->terminal_host_receipt_v2 == nullptr || run->run_token == 0ull ||
      run->next_event_id == ~std::uint64_t{0} ||
      run->current_store_index_v2 > 1u || run->sealed || run->poisoned_v2 ||
      run->one_generation_advance_enqueued_v2 ||
      run->one_generation_advance_pending_v2) {
    return false;
  }

  if (run->source_ready_receipt_token_v2 == nullptr) {
    const auto* source = run->ready_receipt_token_v2;
    if (source == nullptr ||
        source->abi_version !=
            resident_generation_v1::NEO_RESIDENT_GENERATION_ABI_V1 ||
        source->reserved != 0u || source->event_id != run->next_event_id ||
        source->generation_index != run->current_generation_index ||
        source->same_stream_enqueue_count != run->same_stream_enqueue_count ||
        source->intermediate_host_wait_count != 0ull ||
        source->intermediate_readback_count != 0ull) {
      return false;
    }
    run->source_ready_receipt_token_v2 = source;
    run->source_event_id_v2 = source->event_id;
    run->source_same_stream_enqueue_count_v2 =
        source->same_stream_enqueue_count;
  }
  const auto* source = run->source_ready_receipt_token_v2;
  if (source == nullptr ||
      source->abi_version !=
          resident_generation_v1::NEO_RESIDENT_GENERATION_ABI_V1 ||
      source->reserved != 0u || source->event_id != run->source_event_id_v2 ||
      source->same_stream_enqueue_count !=
          run->source_same_stream_enqueue_count_v2 ||
      source->intermediate_host_wait_count != 0ull ||
      source->intermediate_readback_count != 0ull) {
    return false;
  }

  *lifecycle = {};
  lifecycle->generation_owner_ = run;
  lifecycle->population_lifetime_owner_ = run->population_lifetime_owner;
  lifecycle->admitted_run_stream_ = run->admitted_run_stream;
  lifecycle->completion_event_ = run->ready_event;
  lifecycle->terminal_host_receipt_ = run->terminal_host_receipt_v2;
  lifecycle->terminal_host_receipt_bytes_ =
      expected_terminal_host_receipt_bytes;
  lifecycle->completion_event_identity_ = run->next_event_id + 1ull;
  lifecycle->source_ready_receipt_ = run->source_ready_receipt_token_v2;
  lifecycle->resident_parent_ready_event_ = run->resident_parent_ready_event;
  lifecycle->source_event_id_ = run->source_event_id_v2;
  lifecycle->source_same_stream_enqueue_count_ =
      run->source_same_stream_enqueue_count_v2;
  lifecycle->run_token_ = run->run_token;
  lifecycle->generation_index_ = run->current_generation_index;
  lifecycle->store_epoch_ = run->store_epoch_v2;
  lifecycle->same_stream_enqueue_count_ = run->same_stream_enqueue_count;
  lifecycle->current_store_index_ = run->current_store_index_v2;
  lifecycle->reserved_ = 0u;
  return true;
}

bool accept_resident_generation_terminal_enqueue_v2(
    const ResidentGenerationTerminalLifecycleV2* lifecycle,
    std::uint64_t final_same_stream_enqueue_count) {
  using resident_generation_v2::NeoResidentSearchTerminalReceiptV2;
  if (lifecycle == nullptr || lifecycle->generation_owner_v2() == nullptr ||
      lifecycle->same_stream_enqueue_count_v2() >
          ~std::uint64_t{0} - 3ull ||
      final_same_stream_enqueue_count !=
          lifecycle->same_stream_enqueue_count_v2() + 3ull) {
    return false;
  }
  auto* generation = lifecycle->generation_owner_v2();
  const bool exact =
      lifecycle->population_lifetime_owner_v2() ==
          generation->population_lifetime_owner &&
      lifecycle->admitted_run_stream_v2() ==
          generation->admitted_run_stream &&
      lifecycle->completion_event_v2() == generation->ready_event &&
      lifecycle->terminal_host_receipt_v2() ==
          generation->terminal_host_receipt_v2 &&
      lifecycle->terminal_host_receipt_bytes_v2() ==
          sizeof(NeoResidentSearchTerminalReceiptV2) &&
      generation->next_event_id != ~std::uint64_t{0} &&
      lifecycle->completion_event_identity_v2() != 0ull &&
      lifecycle->completion_event_identity_v2() ==
          generation->next_event_id + 1ull &&
      lifecycle->source_ready_receipt_v2() ==
          generation->source_ready_receipt_token_v2 &&
      lifecycle->source_ready_receipt_v2() != nullptr &&
      lifecycle->source_ready_receipt_v2()->event_id ==
          lifecycle->source_event_id_v2() &&
      lifecycle->source_ready_receipt_v2()->same_stream_enqueue_count ==
          lifecycle->source_same_stream_enqueue_count_v2() &&
      lifecycle->resident_parent_ready_event_v2() ==
          generation->resident_parent_ready_event &&
      lifecycle->source_event_id_v2() == generation->source_event_id_v2 &&
      lifecycle->source_same_stream_enqueue_count_v2() ==
          generation->source_same_stream_enqueue_count_v2 &&
      lifecycle->run_token_v2() == generation->run_token &&
      lifecycle->generation_index_v2() ==
          generation->current_generation_index &&
      lifecycle->store_epoch_v2() == generation->store_epoch_v2 &&
      lifecycle->current_store_index_v2() ==
          generation->current_store_index_v2 &&
      lifecycle->same_stream_enqueue_count_v2() ==
          generation->same_stream_enqueue_count &&
      !generation->sealed && !generation->poisoned_v2 &&
      !generation->one_generation_advance_enqueued_v2 &&
      !generation->one_generation_advance_pending_v2;
  if (!exact) {
    return false;
  }
  generation->same_stream_enqueue_count = final_same_stream_enqueue_count;
  generation->next_event_id = lifecycle->completion_event_identity_v2();
  return true;
}

}  // namespace neoethos::resident_generation_v2_internal

namespace neoethos::resident_search_generation_v2 {

extern "C" std::int32_t enqueue_full_population_scored_generation_advance_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* scoring,
    const NeoResidentScoringPopulationSourceV2* population,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1* dependency,
    resident_generation_v2::NeoResidentSearchAdvancePendingReceiptV2* pending) {
  using namespace resident_generation_v1;
  using namespace resident_scoring_novelty_v1;
  if (generation == nullptr || scoring == nullptr || population == nullptr ||
      dependency == nullptr || pending == nullptr ||
      population->abi_version != NEO_RESIDENT_SEARCH_GENERATION_ABI_V2 ||
      population->reserved != 0u || population->receipt_token == nullptr ||
      population->admitted_run_stream != generation->admitted_run_stream ||
      population->metrics_ready_event == nullptr ||
      population->scoring_ready_event == nullptr ||
      population->metrics_ready_event == population->scoring_ready_event ||
      population->metric_rows_device == nullptr ||
      population->expected_scenario_ids_device == nullptr ||
      population->logical_population_count != generation->logical_population_count ||
      population->logical_population_count !=
          generation->retained_evaluation_capacity ||
      population->feature_count != generation->plan.feature_count ||
      population->max_terms_per_gene != generation->plan.max_terms_per_gene ||
      population->full_discovery_reserve_bytes == 0ull ||
      population->full_discovery_reserve_bytes !=
          generation->allocation.full_discovery_reserve_bytes ||
      !generation->initialized_v2 ||
      !generation->evaluator_constants_configured_v2 || generation->sealed ||
      generation->one_generation_advance_enqueued_v2 ||
      generation->one_generation_advance_pending_v2 ||
      generation->terminal_host_receipt_v2 == nullptr ||
      generation->terminal_device_receipt_v2 == nullptr ||
      generation->current_generation_index != 0 ||
      generation->ready_receipt_token_v2 != dependency ||
      dependency->event_id != generation->next_event_id ||
      dependency->generation_index != generation->current_generation_index ||
      dependency->same_stream_enqueue_count !=
          generation->same_stream_enqueue_count ||
      dependency->intermediate_host_wait_count != 0 ||
      dependency->intermediate_readback_count != 0) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }

  NeoResidentScoringNoveltyPopulationImportV1 import{};
  import.abi_version = NEO_RESIDENT_SCORING_NOVELTY_ABI_V1;
  import.selected_cuda_ordinal = population->selected_cuda_ordinal;
  import.admitted_run_stream = population->admitted_run_stream;
  import.metrics_ready_event = population->metrics_ready_event;
  import.scoring_novelty_ready_event = population->scoring_ready_event;
  import.population_lifetime_owner =
      const_cast<void*>(population->receipt_token);
  import.metric_rows_device = population->metric_rows_device;
  using GenerationGeneV1 = NeoResidentGenerationGeneScalarV1;
  using ScoringGeneV1 = NeoResidentScoringNoveltyGeneScalarV1;
  static_assert(sizeof(GenerationGeneV1) == sizeof(ScoringGeneV1));
  static_assert(alignof(GenerationGeneV1) == alignof(ScoringGeneV1));
  static_assert(offsetof(GenerationGeneV1, gene_identity) ==
                offsetof(ScoringGeneV1, gene_identity));
  static_assert(offsetof(GenerationGeneV1, content_hash) ==
                offsetof(ScoringGeneV1, content_hash));
  static_assert(offsetof(GenerationGeneV1, term_count) ==
                offsetof(ScoringGeneV1, term_count));
  static_assert(offsetof(GenerationGeneV1, smc_flags) ==
                offsetof(ScoringGeneV1, smc_flags));
  static_assert(offsetof(GenerationGeneV1, long_threshold) ==
                offsetof(ScoringGeneV1, long_threshold));
  static_assert(offsetof(GenerationGeneV1, short_threshold) ==
                offsetof(ScoringGeneV1, short_threshold));
  static_assert(offsetof(GenerationGeneV1, target_pips) ==
                offsetof(ScoringGeneV1, target_pips));
  static_assert(offsetof(GenerationGeneV1, stop_pips) ==
                offsetof(ScoringGeneV1, stop_pips));
  static_assert(offsetof(GenerationGeneV1, stop_vol_multiplier) ==
                offsetof(ScoringGeneV1, stop_vol_multiplier));
  static_assert(offsetof(GenerationGeneV1, generation) ==
                offsetof(ScoringGeneV1, generation));
  static_assert(offsetof(GenerationGeneV1, reserved) ==
                offsetof(ScoringGeneV1, reserved));
  // The existing generation and scoring ABIs deliberately duplicate this
  // fixed-width POD. Keep the single representation bridge here and ratchet
  // every field offset before passing the resident device allocation.
  import.gene_scalars_device =
      reinterpret_cast<const ScoringGeneV1*>(generation->gene_scalars_device);
  import.gene_indices_device = generation->gene_indices_device;
  import.expected_scenario_ids_device =
      population->expected_scenario_ids_device;
  import.logical_population_count = population->logical_population_count;
  import.feature_count = population->feature_count;
  import.max_terms_per_gene = population->max_terms_per_gene;
  import.full_discovery_reserve_bytes =
      population->full_discovery_reserve_bytes;
  std::memcpy(import.cuda_device_identity_sha256,
              generation->cuda_device_identity_sha256, 32);
  std::memcpy(import.primary_context_identity_sha256,
              generation->primary_context_identity_sha256, 32);
  std::memcpy(import.run_stream_identity_sha256,
              generation->run_stream_identity_sha256, 32);
  std::memcpy(import.metric_semantics_sha256,
              generation->plan.metric_semantics_sha256, 32);
  std::memcpy(import.gene_schema_sha256,
              generation->plan.strategy_gene_schema_sha256, 32);
  std::memcpy(import.scenario_order_semantics_sha256,
              generation->plan.scenario_order_semantics_sha256, 32);
  std::memcpy(import.cuda_build_manifest_sha256,
              generation->plan.cuda_build_manifest_sha256, 32);
  std::memcpy(import.cuda_math_flags_sha256,
              NEO_RESIDENT_CUDA_MATH_SEMANTICS_SHA256_V2, 32);
  std::memcpy(import.resident_input_content_sha256,
              generation->plan.plan_identity_sha256, 32);
  std::memcpy(import.gene_content_sha256,
              generation->plan.run_identity_sha256, 32);
  std::memcpy(import.metric_content_sha256,
              generation->plan.metric_semantics_sha256, 32);
  std::memcpy(import.scenario_order_content_sha256,
              generation->plan.scenario_order_semantics_sha256, 32);

  NeoResidentScoredDecisionRowsV1 scored{};
  NeoResidentScoringNoveltyReadyEventV1 scoring_ready{};
  std::int32_t status = bind_and_seal_resident_scoring_v2(
      scoring, &import, &scored, &scoring_ready);
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  if (cudaStreamWaitEvent(generation->admitted_run_stream,
                          scored.scoring_novelty_ready_event, 0) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  ++generation->same_stream_enqueue_count;
  resident_generation_v2::NeoResidentGenerationGeneViewV2 retained_view{};
  status = resident_generation_v2::export_current_resident_gene_view_v2(
      generation, dependency, &retained_view);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  const resident_generation_v2_internal::ResidentGenerationScoredRowsV2
      generation_rows{
          &scored, scored.resident_decision_keys_device, &retained_view};
  resident_generation_v2_internal::ResidentGenerationPreparedAdvanceV2
      prepared_generation{};
  status = resident_generation_v2_internal::
      enqueue_resident_generation_offspring_from_scored_rows_v2(
          generation, &generation_rows, &prepared_generation);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  const std::uint64_t completion_event_id = generation->next_event_id + 1ull;
  const std::uint64_t terminal_same_stream_enqueue_count =
      generation->same_stream_enqueue_count + 3ull;
  publish_one_generation_commit_kernel_v2<<<
      1, 1, 0, generation->admitted_run_stream>>>(
      prepared_generation, generation->terminal_device_receipt_v2,
      generation->run_token,
      completion_event_id, terminal_same_stream_enqueue_count);
  ++generation->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  if (cudaMemcpyAsync(generation->terminal_host_receipt_v2,
                      generation->terminal_device_receipt_v2,
                      sizeof(NeoResidentSearchTerminalReceiptV2),
                      cudaMemcpyDeviceToHost,
                      generation->admitted_run_stream) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  ++generation->same_stream_enqueue_count;
  if (cudaEventRecord(generation->ready_event,
                      generation->admitted_run_stream) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  ++generation->same_stream_enqueue_count;
  ++generation->next_event_id;
  *pending = {};
  pending->abi_version =
      resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2;
  pending->completion_event_id = generation->next_event_id;
  pending->target_generation_index = 1;
  pending->target_store_epoch = generation->store_epoch_v2 + 1;
  pending->target_store_index = 1;
  pending->run_token = generation->run_token;
  pending->same_stream_enqueue_count = generation->same_stream_enqueue_count;
  pending->dependency_receipt_token = dependency;
  pending->terminal_host_receipt_token = generation->terminal_host_receipt_v2;
  generation->pending_receipt_token_v2 = pending;
  generation->one_generation_advance_pending_v2 = true;
  generation->next_expected_logical_offset = 0;
  return NEO_RESIDENT_STATUS_OK_V1;
}

#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
extern "C" std::int32_t fixture_set_resident_generation_gene_identity_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    std::uint64_t candidate, std::uint64_t gene_identity) {
  using namespace resident_generation_v1;
  if (generation == nullptr || !generation->initialized_v2 ||
      generation->one_generation_advance_enqueued_v2 ||
      generation->one_generation_advance_pending_v2 ||
      candidate >= generation->logical_population_count) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  fixture_set_resident_gene_identity_kernel_v2<<<
      1, 1, 0, generation->admitted_run_stream>>>(
      generation->gene_scalars_device, candidate, gene_identity);
  return launch_status_v1();
}

extern "C" std::int32_t fixture_set_duplicate_final_gene_content_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    std::uint64_t source_candidate, std::uint64_t destination_candidate) {
  using namespace resident_generation_v1;
  if (generation == nullptr || !generation->initialized_v2 ||
      generation->one_generation_advance_enqueued_v2 ||
      generation->one_generation_advance_pending_v2 ||
      source_candidate >= generation->logical_population_count ||
      destination_candidate >= generation->logical_population_count ||
      source_candidate == destination_candidate) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  generation->fixture_duplicate_final_content_v2 = true;
  generation->fixture_duplicate_source_v2 = source_candidate;
  generation->fixture_duplicate_destination_v2 = destination_candidate;
  return NEO_RESIDENT_STATUS_OK_V1;
}

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
    NeoResidentGenerationAdvanceFixtureSnapshotV2* snapshot) {
  using namespace resident_generation_v1;
  if (generation == nullptr || ranked_population_ordinals_host == nullptr ||
      initial_genes_host == nullptr || final_genes_host == nullptr ||
      initial_term_indices_host == nullptr || initial_term_weights_host == nullptr ||
      final_term_indices_host == nullptr || final_term_weights_host == nullptr ||
      parent_a_host == nullptr || parent_b_host == nullptr ||
      selected_survivors_host == nullptr || sorted_dedup_flags_host == nullptr ||
      candidate_valid_flags_host == nullptr || snapshot == nullptr ||
      !generation->one_generation_advance_enqueued_v2 ||
      !generation->terminal_committed_v2 ||
      population_capacity != generation->logical_population_count ||
      term_capacity != generation->logical_population_count *
                           generation->plan.max_terms_per_gene ||
      survivor_capacity != generation->plan.survivor_count) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  const std::size_t count = static_cast<std::size_t>(population_capacity);
  const std::size_t term_count = static_cast<std::size_t>(term_capacity);
  const std::size_t survivor_count = static_cast<std::size_t>(survivor_capacity);
  const std::size_t rank_bytes = count * sizeof(std::uint64_t);
  const std::size_t gene_bytes =
      count * sizeof(NeoResidentGenerationGeneScalarV1);
  const std::size_t index_bytes = term_count * sizeof(std::uint64_t);
  const std::size_t weight_bytes = term_count * sizeof(double);
  const std::size_t survivor_bytes = survivor_count * sizeof(std::uint64_t);
  const std::size_t flag_bytes = count * sizeof(std::uint8_t);
  NeoResidentGenerationDeviceSealV2 seal{};
  NeoResidentSearchDeviceControlV2 control{};
  std::uint32_t device_content_fault = 0;
  std::uint32_t gene_hash_collision_fault = 0;
  std::uint32_t selected_count = 0;
  std::uint32_t dedup_run_count = 0;
  if (cudaStreamSynchronize(generation->admitted_run_stream) != cudaSuccess ||
      cudaMemcpy(ranked_population_ordinals_host,
                 generation->rank_values_b_device, rank_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(initial_genes_host,
                 generation->offspring_gene_scalars_device, gene_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(final_genes_host,
                 generation->gene_scalars_device, gene_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(initial_term_indices_host,
                 generation->offspring_gene_indices_device, index_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(initial_term_weights_host,
                 generation->offspring_gene_weights_device, weight_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(final_term_indices_host,
                 generation->gene_indices_device, index_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(final_term_weights_host,
                 generation->gene_weights_device, weight_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(parent_a_host, generation->parent_a_device, rank_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(parent_b_host, generation->parent_b_device, rank_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(selected_survivors_host, generation->rank_keys_a_device,
                 survivor_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(sorted_dedup_flags_host, generation->dedup_flags_device,
                 flag_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(candidate_valid_flags_host,
                 generation->selection_flags_device, flag_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&seal, generation->device_seal_v2, sizeof(seal),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&control, generation->resident_control_device_v2,
                 sizeof(control), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&device_content_fault,
                 generation->device_content_fault_device,
                 sizeof(device_content_fault),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&gene_hash_collision_fault,
                 generation->gene_hash_collision_fault_device,
                 sizeof(gene_hash_collision_fault),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&selected_count,
                 generation->selected_count_device,
                 sizeof(selected_count),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&dedup_run_count,
                 generation->dedup_run_count_device,
                 sizeof(dedup_run_count),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  *snapshot = {};
  snapshot->abi_version = NEO_RESIDENT_SEARCH_GENERATION_ABI_V2;
  snapshot->device_content_fault = device_content_fault;
  snapshot->gene_hash_collision_fault = gene_hash_collision_fault;
  snapshot->control_fault_word = control.fault_word;
  snapshot->stop_requested = control.stop_requested;
  snapshot->current_store_index = seal.current_store_index;
  snapshot->max_terms_per_gene = generation->plan.max_terms_per_gene;
  snapshot->survivor_count =
      static_cast<std::uint32_t>(generation->plan.survivor_count);
  snapshot->selected_count = selected_count;
  snapshot->dedup_run_count = dedup_run_count;
  snapshot->logical_population_count = population_capacity;
  snapshot->generation_index = seal.generation_index;
  snapshot->store_epoch = seal.store_epoch;
  snapshot->terminal_synchronization_count = 1;
  snapshot->terminal_readback_count = 18;
  snapshot->terminal_readback_bytes =
      3 * rank_bytes + 2 * gene_bytes + 2 * index_bytes +
      2 * weight_bytes + survivor_bytes + 2 * flag_bytes + sizeof(seal) +
      sizeof(control) + sizeof(device_content_fault) +
      sizeof(gene_hash_collision_fault) + 2 * sizeof(std::uint32_t);
  return NEO_RESIDENT_STATUS_OK_V1;
}
#endif

}  // namespace neoethos::resident_search_generation_v2

namespace neoethos::resident_generation_v2 {

extern "C" std::int32_t configure_resident_generation_evaluator_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1* dependency,
    const double* smc_weights,
    std::uint32_t smc_gate_disabled,
    resident_generation_v1::NeoResidentGenerationReadyEventV1* ready) {
  using namespace resident_generation_v1;
  if (run == nullptr || dependency == nullptr || smc_weights == nullptr || ready == nullptr ||
      dependency == ready || smc_gate_disabled > 1u) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (dependency->abi_version != NEO_RESIDENT_GENERATION_ABI_V1) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  if (!run->initialized_v2 || run->evaluator_constants_configured_v2 || run->sealed ||
      run->smc_weights_device_v2 == nullptr || run->ready_receipt_token_v2 != dependency ||
      dependency->event_id != run->next_event_id ||
      dependency->generation_index != run->current_generation_index ||
      dependency->same_stream_enqueue_count != run->same_stream_enqueue_count ||
      dependency->intermediate_host_wait_count != 0 ||
      dependency->intermediate_readback_count != 0) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  for (std::size_t slot = 0; slot < 11; ++slot) {
    if (!std::isfinite(smc_weights[slot]) || smc_weights[slot] < 0.0) {
      return NEO_RESIDENT_STATUS_RANGE_ERROR_V1;
    }
  }
  if (cudaMemcpyAsync(run->smc_weights_device_v2, smc_weights,
                      11 * sizeof(double), cudaMemcpyHostToDevice,
                      run->admitted_run_stream) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  run->smc_gate_disabled_v2 = smc_gate_disabled != 0u;
  run->evaluator_constants_configured_v2 = true;
  return resident_generation_v1::record_ready_event_v1(run, ready);
}

extern "C" std::int32_t export_current_resident_gene_view_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1* ready,
    NeoResidentGenerationGeneViewV2* view) {
  using namespace resident_generation_v1;
  if (run == nullptr || ready == nullptr || view == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (ready->abi_version != NEO_RESIDENT_GENERATION_ABI_V1) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  if (!run->initialized_v2 || !run->evaluator_constants_configured_v2 || run->sealed ||
      run->allocation_base == nullptr ||
      run->ready_event == nullptr || run->device_seal_v2 == nullptr ||
      run->resident_control_device_v2 == nullptr ||
      run->smc_weights_device_v2 == nullptr || run->current_store_index_v2 > 1u ||
      run->store_epoch_v2 == 0 || run->run_token == 0) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  const bool exact_receipt_pointer = run->ready_receipt_token_v2 == ready;
  const bool exact_receipt_values =
      ready->event_id == run->next_event_id &&
      ready->generation_index == run->current_generation_index &&
      ready->same_stream_enqueue_count == run->same_stream_enqueue_count &&
      ready->intermediate_host_wait_count == 0 &&
      ready->intermediate_readback_count == 0;
  if (!exact_receipt_pointer || !exact_receipt_values) {
    return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
  }

  std::memset(view, 0, sizeof(*view));
  view->abi_version = NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2;
  view->flags = 0;
  view->seal_device = run->device_seal_v2;
  view->control_device = run->resident_control_device_v2;
  view->expected_generation_index = run->current_generation_index;
  view->expected_store_epoch = run->store_epoch_v2;
  view->expected_run_token = run->run_token;
  view->logical_population_count = run->logical_population_count;
  view->feature_count = run->plan.feature_count;
  view->max_terms_per_gene = run->plan.max_terms_per_gene;
  view->smc_flag_count = run->plan.smc_flag_count;
  std::memcpy(view->plan_identity_sha256, run->plan.plan_identity_sha256, 32);
  std::memcpy(view->generation_semantics_sha256,
              run->plan.generation_semantics_sha256, 32);
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t validate_resident_gene_view_owner_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const NeoResidentGenerationGeneViewV2* view) {
  using namespace resident_generation_v1;
  if (run == nullptr || view == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  if (view->abi_version != NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2) {
    return NEO_RESIDENT_STATUS_ABI_MISMATCH_V1;
  }
  if (!run->initialized_v2 || !run->evaluator_constants_configured_v2 || run->sealed ||
      run->allocation_base == nullptr ||
      run->device_seal_v2 == nullptr || run->resident_control_device_v2 == nullptr ||
      run->run_token == 0) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
  const bool exact =
      view->flags == 0 && view->seal_device == run->device_seal_v2 &&
      view->control_device == run->resident_control_device_v2 &&
      view->expected_generation_index == run->current_generation_index &&
      view->expected_store_epoch == run->store_epoch_v2 &&
      view->expected_run_token == run->run_token &&
      view->logical_population_count == run->logical_population_count &&
      view->feature_count == run->plan.feature_count &&
      view->max_terms_per_gene == run->plan.max_terms_per_gene &&
      view->smc_flag_count == run->plan.smc_flag_count &&
      std::memcmp(view->plan_identity_sha256, run->plan.plan_identity_sha256, 32) == 0 &&
      std::memcmp(view->generation_semantics_sha256,
                  run->plan.generation_semantics_sha256, 32) == 0;
  return exact ? NEO_RESIDENT_STATUS_OK_V1
               : NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;
}
}  // namespace neoethos::resident_generation_v2
