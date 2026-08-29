#include "resident_generation_v1_abi.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace neoethos::resident_generation_v1 {
namespace {

constexpr std::uint32_t PHILOX_M0_V1 = 0xD2511F53u;
constexpr std::uint32_t PHILOX_M1_V1 = 0xCD9E8D57u;
constexpr std::uint32_t PHILOX_W0_V1 = 0x9E3779B9u;
constexpr std::uint32_t PHILOX_W1_V1 = 0xBB67AE85u;
constexpr std::uint32_t METRIC_VALUE_COUNT_V1 = 11;
constexpr std::size_t DEVICE_ALIGNMENT_V1 = 256;
constexpr std::uint64_t FNV_OFFSET_0_V1 = 14695981039346656037ull;
constexpr std::uint64_t FNV_OFFSET_1_V1 = 1099511628211ull ^ 0x9e3779b97f4a7c15ull;
constexpr std::uint64_t FNV_PRIME_V1 = 1099511628211ull;

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
  void* cub_scratch_device;
  std::uint8_t* exact_chunk_coverage_device;
  std::size_t cub_scratch_bytes;
  std::uint64_t logical_population_count;
  std::uint64_t retained_evaluation_capacity;
  std::uint64_t next_expected_logical_offset;
  std::uint64_t current_generation_index;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t next_event_id;
  std::uint64_t run_token;
  bool sealed;
  bool post_ga_in_place_bound;
};

namespace {

static_assert(sizeof(NeoResidentGenerationGeneScalarV1) == 72,
              "fixed resident generation scalar ABI changed");
static_assert(sizeof(NeoResidentGenerationMetricRowV1) == 104,
              "resident generation metric row ABI changed");

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
         run->cub_scratch_device != nullptr && run->exact_chunk_coverage_device != nullptr &&
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
    NeoResidentGenerationPlanV1 plan) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate < plan.logical_population_count) {
    const std::uint64_t base = candidate * plan.max_terms_per_gene;
    const std::uint64_t hash = full_fixed_stride_gene_hash_v1(
        scalars[candidate], indices + base, weights + base, plan);
    scalars[candidate].content_hash = hash;
    hashes[candidate] = hash;
    values[candidate] = candidate;
  }
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
    std::uint64_t count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    keys[index] = stable_gene_identity_tie_key_v1(scalars[index]);
    values[index] = index;
  }
}

__global__ void gather_resident_decision_rank_keys_kernel_v1(
    const std::uint64_t* resident_decision_keys,
    const std::uint8_t* candidate_valid_flags,
    const std::uint64_t* stable_order,
    std::uint64_t* keys,
    std::uint64_t* values,
    std::uint64_t count) {
  const std::uint64_t position =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (position < count) {
    const std::uint64_t candidate = stable_order[position];
    keys[position] = candidate_valid_flags[candidate] != 0
                         ? resident_decision_keys[candidate]
                         : 0;
    values[position] = candidate;
  }
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
  if (child >= plan.logical_population_count) {
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
  const std::uint64_t u64_max_v1 = ~std::uint64_t{0};
  for (std::uint64_t rank = 0; rank < plan.logical_population_count; ++rank) {
    rank_available[rank] = 1;
  }
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
    std::uint64_t survivor_count) {
  const std::uint64_t selected =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (selected < survivor_count) {
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
    std::uint64_t generation_index) {
  const std::uint64_t destination =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (destination >= plan.logical_population_count) {
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
    std::uint64_t generation_index) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= plan.logical_population_count || candidate < plan.survivor_count) {
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
  const cudaError_t status = cudaEventRecord(run->ready_event, run->admitted_run_stream);
  if (status != cudaSuccess) {
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
      run->dedup_values_a_device, run->plan);
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
      run->rank_values_a_device, run->logical_population_count);
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
      run->rank_values_a_device, run->logical_population_count);
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
                                    run->plan.survivor_count);
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
      run->offspring_gene_weights_device, run->plan, generation_index);
  ++run->same_stream_enqueue_count;
  return launch_status_v1();
}

std::int32_t launch_device_mutation_v1(NeoResidentGenerationRunV1* run,
                                        std::uint64_t generation_index) {
  constexpr std::uint32_t threads = 256;
  mutate_resident_genes_kernel_v1<<<grid_for_v1(run->logical_population_count),
                                    threads, 0, run->admitted_run_stream>>>(
      run->offspring_gene_scalars_device, run->offspring_gene_indices_device,
      run->offspring_gene_weights_device, run->plan, generation_index);
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
  GenerationPhysicalLayoutV1 layout{};
  if (!checked_physical_layout_v1(*plan, import->admitted_run_stream, &layout)) {
    return NEO_RESIDENT_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  std::size_t same_context_free_bytes = 0;
  std::size_t same_context_total_bytes = 0;
  if (cudaMemGetInfo(&same_context_free_bytes, &same_context_total_bytes) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  (void)same_context_total_bytes;
  if (import->full_discovery_reserve_bytes > same_context_free_bytes ||
      layout.total_device_bytes >
          same_context_free_bytes -
              static_cast<std::size_t>(import->full_discovery_reserve_bytes)) {
    return NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1;
  }
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
  receipt->total_device_bytes = layout.total_device_bytes;
  receipt->same_context_free_bytes = same_context_free_bytes;
  receipt->full_discovery_reserve_bytes = import->full_discovery_reserve_bytes;
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
  std::size_t current_free = 0;
  std::size_t current_total = 0;
  if (cudaMemGetInfo(&current_free, &current_total) != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  (void)current_total;
  if (receipt->full_discovery_reserve_bytes > current_free ||
      receipt->total_device_bytes > current_free - receipt->full_discovery_reserve_bytes) {
    return NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1;
  }
  auto* created = new (std::nothrow) NeoResidentGenerationRunV1{};
  if (created == nullptr) {
    return NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1;
  }
  created->plan = *plan;
  created->allocation = *receipt;
  created->admitted_run_stream = import->admitted_run_stream;
  created->resident_parent_ready_event = import->resident_parent_ready_event;
  created->ready_event = import->generation_ready_event;
  created->population_lifetime_owner = import->population_lifetime_owner;
  created->logical_population_count = plan->logical_population_count;
  created->retained_evaluation_capacity = plan->retained_evaluation_capacity;
  created->next_expected_logical_offset = 0;
  created->current_generation_index = 0;
  created->same_stream_enqueue_count = 0;
  created->next_event_id = 0;
  created->sealed = false;
  created->post_ga_in_place_bound = false;
  created->run_token = FNV_OFFSET_0_V1;
  for (std::size_t index = 0; index < 32; ++index) {
    created->run_token =
        (created->run_token ^ plan->run_identity_sha256[index]) * FNV_PRIME_V1;
  }
  if (created->run_token == 0) {
    created->run_token = 1;
  }
  cudaError_t status = cudaMallocAsync(&created->allocation_base,
                                       static_cast<std::size_t>(receipt->total_device_bytes),
                                       created->admitted_run_stream);
  if (status != cudaSuccess) {
    delete created;
    return NEO_RESIDENT_STATUS_OUT_OF_MEMORY_V1;
  }
  if (!partition_generation_allocation_v1(created)) {
    const cudaError_t release_status =
        cudaFreeAsync(created->allocation_base, created->admitted_run_stream);
    if (release_status == cudaSuccess) {
      delete created;
    }
    return NEO_RESIDENT_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  status = cudaStreamWaitEvent(created->admitted_run_stream,
                               created->resident_parent_ready_event, 0);
  if (status != cudaSuccess) {
    const cudaError_t release_status =
        cudaFreeAsync(created->allocation_base, created->admitted_run_stream);
    if (release_status == cudaSuccess) {
      delete created;
    }
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  created->same_stream_enqueue_count = 1;
  *run = created;
  return NEO_RESIDENT_STATUS_OK_V1;
}

extern "C" std::int32_t initialize_resident_generation_population_v1(
    NeoResidentGenerationRunV1* run,
    NeoResidentGenerationReadyEventV1* ready) {
  if (run == nullptr || ready == nullptr || run->sealed) {
    return NEO_RESIDENT_STATUS_STATE_ERROR_V1;
  }
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
  if (run == nullptr || run->allocation_base == nullptr || run->admitted_run_stream == nullptr) {
    return NEO_RESIDENT_STATUS_INVALID_ARGUMENT_V1;
  }
  const cudaError_t release_status =
      cudaFreeAsync(run->allocation_base, run->admitted_run_stream);
  if (release_status != cudaSuccess) {
    return NEO_RESIDENT_STATUS_CUDA_ERROR_V1;
  }
  run->allocation_base = nullptr;
  run->ready_event = nullptr;
  delete run;
  return NEO_RESIDENT_STATUS_OK_V1;
}

}  // namespace neoethos::resident_generation_v1
