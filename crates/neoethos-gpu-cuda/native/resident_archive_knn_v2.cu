#include "resident_archive_knn_v2_abi.cuh"
#include "resident_generation_v2_internal.cuh"
#include "resident_scoring_novelty_v2_internal.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace neoethos::resident_archive_knn_v2 {

namespace {

using GeneScalarV2 =
    resident_generation_v1::NeoResidentGenerationGeneScalarV1;
using GeneViewV2 = resident_generation_v2::NeoResidentGenerationGeneViewV2;
using GeneSealV2 = resident_generation_v2::NeoResidentGenerationDeviceSealV2;
using MetricRowV2 =
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyMetricRowV1;
using ScoringSealV2 =
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyDeviceSealV1;
using FiniteRowsV2 = resident_scoring_novelty_v2_internal::
    ResidentScoringFiniteObjectiveRowsV2;
using PreparedAdvanceV2 =
    resident_generation_v2_internal::ResidentGenerationPreparedAdvanceV2;
using TerminalLifecycleV2 =
    resident_generation_v2_internal::ResidentGenerationTerminalLifecycleV2;

constexpr std::uint64_t kAlignmentV2 = 256;
constexpr std::uint32_t kThreadsV2 = 256;
constexpr std::uint32_t kSourceCurrentV2 = 0;
constexpr std::uint32_t kSourceArchiveV2 = 1;
constexpr std::uint32_t kNetMetricSlotV2 = 0;
constexpr std::uint32_t kTradeCountMetricSlotV2 = 8;
constexpr std::uint64_t kGenerationMaskV2 = 0xffffull;
constexpr std::uint64_t kArchiveMaskV2 = 0xffffull;
constexpr std::uint64_t kEpochMaskV2 = 0x7fffffffull;
constexpr std::uint64_t kArchiveControlPrefixBytesV2 =
    resident_scoring_novelty_v2_internal::
        NEO_RESIDENT_SCORING_SLICE2_ARCHIVE_CONTROL_OFFSET_V2;

enum class HostPhaseV2 : std::uint32_t {
  Bound = 0,
  Ranked = 1,
  Staged = 2,
  Published = 3,
  TerminalPending = 4,
  TerminalComplete = 5,
};

enum DeviceFaultV2 : std::uint32_t {
  kNoFaultV2 = 0,
  kIdentityFaultV2 = 1,
  kNonFiniteMetricFaultV2 = 2,
  kGeneShapeFaultV2 = 3,
  kSignatureFaultV2 = 4,
  kNeighborBoundFaultV2 = 5,
  kArchiveBoundFaultV2 = 6,
  kPublicationFaultV2 = 7,
  kScoringSealFaultV2 = 8,
};

struct alignas(8) ExactNeighborKeyV2 {
  std::uint32_t numerator;
  std::uint32_t denominator;
  std::uint64_t gene_identity;
  std::uint32_t source_kind;
  std::uint32_t source_ordinal;
  std::uint64_t reserved;
};

static_assert(sizeof(ExactNeighborKeyV2) == 32,
              "exact kNN key layout changed");

// The scoring owner reserves bytes [0, 64). The remaining 192 bytes are one
// archive control followed by the one device terminal receipt.
struct alignas(8) ArchiveControlV2 {
  unsigned long long packed_commit_word;
  std::uint64_t ranked_source_commit_word;
  std::uint64_t staged_count;
  std::uint64_t staged_collision_count;
  std::uint64_t committed_collision_count;
  std::uint64_t run_identity;
  std::uint32_t device_fault_word;
  std::uint32_t validation_fault_word;
  std::uint32_t ranked_ready;
  std::uint32_t staged_ready;
  std::uint32_t publication_count;
  std::uint32_t terminal_status;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t validator_digest;
};

static_assert(sizeof(ArchiveControlV2) == 88,
              "archive control must leave exactly 104 terminal bytes");
static_assert(kArchiveControlPrefixBytesV2 + sizeof(ArchiveControlV2) +
                      sizeof(NeoResidentArchiveKnnTerminalV2) ==
                  256,
              "shared scoring/archive control partition changed");

struct DeviceGeneSourcesV2 {
  const GeneScalarV2* scalars;
  const std::uint64_t* term_indices;
  const double* term_weights;
};

bool checked_add_v2(std::uint64_t left, std::uint64_t right,
                    std::uint64_t* result) {
  if (result == nullptr ||
      right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *result = left + right;
  return true;
}

bool checked_mul_v2(std::uint64_t left, std::uint64_t right,
                    std::uint64_t* result) {
  if (result == nullptr ||
      (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool checked_aligned_region_size_v2(std::uint64_t item_count,
                                    std::uint64_t elements_per_item,
                                    std::uint64_t element_size,
                                    std::uint64_t* result) {
  std::uint64_t element_count = 0;
  std::uint64_t raw_bytes = 0;
  if (item_count == 0 || elements_per_item == 0 || element_size == 0 ||
      !checked_mul_v2(item_count, elements_per_item, &element_count) ||
      !checked_mul_v2(element_count, element_size, &raw_bytes)) {
    return false;
  }
  const std::uint64_t remainder = raw_bytes % kAlignmentV2;
  const std::uint64_t padding =
      remainder == 0 ? 0 : kAlignmentV2 - remainder;
  return checked_add_v2(raw_bytes, padding, result);
}

bool nonzero_uuid_v2(const std::uint8_t uuid[16]) {
  std::uint8_t aggregate = 0;
  for (std::size_t index = 0; index < 16; ++index) {
    aggregate |= uuid[index];
  }
  return aggregate != 0;
}

bool validate_region_v2(const NeoResidentArchiveKnnArenaRegionV2& region,
                        std::uint64_t expected_size,
                        std::uint64_t* cursor) {
  std::uint64_t end = 0;
  if (cursor == nullptr || region.offset_bytes != *cursor ||
      region.offset_bytes % kAlignmentV2 != 0 ||
      region.size_bytes != expected_size || region.size_bytes == 0 ||
      region.size_bytes % kAlignmentV2 != 0 ||
      !checked_add_v2(region.offset_bytes, region.size_bytes, &end)) {
    return false;
  }
  *cursor = end;
  return true;
}

bool validate_binding_layout_v2(const NeoResidentArchiveKnnBindV2& binding) {
  if (binding.abi_version != NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 ||
      binding.reserved != 0 || binding.reserved_extents != 0 ||
      binding.population_count == 0 ||
      binding.population_count >
          NEO_RESIDENT_ARCHIVE_KNN_MAX_POPULATION_COUNT_V2 ||
      binding.archive_capacity == 0 ||
      binding.archive_capacity > NEO_RESIDENT_ARCHIVE_KNN_MAX_CAPACITY_V2 ||
      binding.signature_word_count !=
          NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 ||
      binding.novelty_neighbor_count != NEO_RESIDENT_ARCHIVE_KNN_K_V2 ||
      binding.max_terms_per_gene != NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 ||
      !nonzero_uuid_v2(binding.device_uuid) ||
      binding.primary_context_identity == 0 ||
      binding.search_stream_identity == 0 ||
      binding.active_pool_identity == 0 || binding.cuda_build_identity == 0 ||
      binding.kernel_semantics_identity == 0 ||
      binding.binary64_math_identity == 0 || binding.plan_identity == 0 ||
      binding.run_identity == 0 ||
      binding.full_workspace_receipt_identity == 0 ||
      binding.post_trim_receipt_identity == 0) {
    return false;
  }

  std::uint64_t population_scalar_bytes = 0;
  std::uint64_t archive_gene_scalar_bytes = 0;
  std::uint64_t archive_term_index_bytes = 0;
  std::uint64_t archive_term_weight_bytes = 0;
  std::uint64_t archive_metric_row_bytes = 0;
  std::uint64_t archive_signature_bytes = 0;
  std::uint64_t archive_hash_bytes = 0;
  std::uint64_t population_signature_bytes = 0;
  std::uint64_t exact_top_k_bytes = 0;
  std::uint64_t admission_flag_bytes = 0;
  std::uint64_t admission_offset_bytes = 0;
  if (!checked_aligned_region_size_v2(binding.population_count, 1,
                                      sizeof(double),
                                      &population_scalar_bytes) ||
      !checked_aligned_region_size_v2(binding.archive_capacity, 1,
                                      sizeof(GeneScalarV2),
                                      &archive_gene_scalar_bytes) ||
      !checked_aligned_region_size_v2(
          binding.archive_capacity, binding.max_terms_per_gene,
          sizeof(std::uint64_t), &archive_term_index_bytes) ||
      !checked_aligned_region_size_v2(
          binding.archive_capacity, binding.max_terms_per_gene,
          sizeof(double), &archive_term_weight_bytes) ||
      !checked_aligned_region_size_v2(binding.archive_capacity, 1,
                                      sizeof(MetricRowV2),
                                      &archive_metric_row_bytes) ||
      !checked_aligned_region_size_v2(
          binding.archive_capacity, binding.signature_word_count,
          sizeof(std::uint64_t), &archive_signature_bytes) ||
      !checked_aligned_region_size_v2(binding.archive_capacity, 1,
                                      sizeof(std::uint64_t),
                                      &archive_hash_bytes) ||
      !checked_aligned_region_size_v2(
          binding.population_count, binding.signature_word_count,
          sizeof(std::uint64_t), &population_signature_bytes) ||
      !checked_aligned_region_size_v2(
          binding.population_count, binding.novelty_neighbor_count,
          sizeof(ExactNeighborKeyV2), &exact_top_k_bytes) ||
      !checked_aligned_region_size_v2(binding.population_count, 1,
                                      sizeof(std::uint32_t),
                                      &admission_flag_bytes) ||
      !checked_aligned_region_size_v2(binding.population_count, 1,
                                      sizeof(std::uint64_t),
                                      &admission_offset_bytes)) {
    return false;
  }

  std::uint64_t cursor = 0;
  if (!validate_region_v2(binding.fitness_scores, population_scalar_bytes,
                          &cursor) ||
      !validate_region_v2(binding.decision_keys, population_scalar_bytes,
                          &cursor) ||
      binding.cub_scratch.offset_bytes != cursor ||
      binding.cub_scratch.size_bytes == 0 ||
      binding.cub_scratch.size_bytes % kAlignmentV2 != 0 ||
      !checked_add_v2(binding.cub_scratch.offset_bytes,
                      binding.cub_scratch.size_bytes, &cursor) ||
      !validate_region_v2(binding.archive_gene_scalars,
                          archive_gene_scalar_bytes, &cursor) ||
      !validate_region_v2(binding.archive_term_indices,
                          archive_term_index_bytes, &cursor) ||
      !validate_region_v2(binding.archive_term_weights,
                          archive_term_weight_bytes, &cursor) ||
      !validate_region_v2(binding.archive_metric_rows,
                          archive_metric_row_bytes, &cursor) ||
      !validate_region_v2(binding.archive_signatures, archive_signature_bytes,
                          &cursor) ||
      !validate_region_v2(binding.archive_hashes, archive_hash_bytes,
                          &cursor) ||
      !validate_region_v2(binding.current_population_signatures,
                          population_signature_bytes, &cursor) ||
      !validate_region_v2(binding.novelty_scores, population_scalar_bytes,
                          &cursor) ||
      !validate_region_v2(binding.exact_top_k_keys, exact_top_k_bytes,
                          &cursor) ||
      !validate_region_v2(binding.admission_flags, admission_flag_bytes,
                          &cursor) ||
      !validate_region_v2(binding.admission_offsets, admission_offset_bytes,
                          &cursor) ||
      !validate_region_v2(binding.archive_control_and_seal, 256, &cursor)) {
    return false;
  }
  return cursor == binding.total_device_bytes;
}

template <typename T>
T* region_pointer_v2(void* allocation_base,
                     const NeoResidentArchiveKnnArenaRegionV2& region) {
  return reinterpret_cast<T*>(static_cast<std::uint8_t*>(allocation_base) +
                              region.offset_bytes);
}

std::uint32_t grid_for_v2(std::uint64_t count) {
  return static_cast<std::uint32_t>((count + kThreadsV2 - 1) / kThreadsV2);
}

std::int32_t launch_status_v2() {
  return cudaPeekAtLastError() == cudaSuccess
             ? NEO_ARCHIVE_KNN_STATUS_OK_V2
             : NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
}

__host__ __device__ std::uint64_t pack_commit_word_v2(
    std::uint32_t current_store, std::uint64_t generation,
    std::uint64_t archive_count, std::uint64_t commit_epoch) {
  return static_cast<std::uint64_t>(current_store & 1u) |
         ((generation & kGenerationMaskV2) << 1) |
         ((archive_count & kArchiveMaskV2) << 17) |
         ((commit_epoch & kEpochMaskV2) << 33);
}

__host__ __device__ std::uint32_t unpack_store_v2(std::uint64_t word) {
  return static_cast<std::uint32_t>(word & 1ull);
}

__host__ __device__ std::uint64_t unpack_generation_v2(std::uint64_t word) {
  return (word >> 1) & kGenerationMaskV2;
}

__host__ __device__ std::uint64_t unpack_archive_count_v2(
    std::uint64_t word) {
  return (word >> 17) & kArchiveMaskV2;
}

__host__ __device__ std::uint64_t unpack_epoch_v2(std::uint64_t word) {
  return (word >> 33) & kEpochMaskV2;
}

__device__ std::uint64_t atomic_read_commit_v2(
    const unsigned long long* word) {
  return atomicCAS(const_cast<unsigned long long*>(word), 0ull, 0ull);
}

__device__ void latch_device_fault_v2(ArchiveControlV2* control,
                                      std::uint32_t fault) {
  if (control != nullptr && fault != 0) {
    atomicCAS(&control->device_fault_word, 0u, fault);
  }
}

__device__ bool scoring_seal_valid_v2(const ScoringSealV2* seal) {
  return seal != nullptr &&
         seal->abi_version ==
             resident_scoring_novelty_v1::NEO_RESIDENT_SCORING_NOVELTY_ABI_V1 &&
         seal->valid == 1u && seal->device_fault_word == 0u;
}

__device__ bool load_current_gene_sources_v2(
    const GeneSealV2* seal, const GeneViewV2& expected,
    ArchiveControlV2* control, DeviceGeneSourcesV2* sources) {
  if (seal == nullptr || sources == nullptr ||
      seal->abi_version !=
          resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2 ||
      seal->fault_code != 0 || seal->current_store_index > 1u ||
      seal->generation_index != expected.expected_generation_index ||
      seal->store_epoch != expected.expected_store_epoch ||
      seal->run_token != expected.expected_run_token ||
      seal->logical_population_count != expected.logical_population_count ||
      seal->feature_count != expected.feature_count ||
      seal->max_terms_per_gene != expected.max_terms_per_gene ||
      seal->max_terms_per_gene != NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 ||
      seal->scalar_store[seal->current_store_index] == nullptr ||
      seal->term_index_store[seal->current_store_index] == nullptr ||
      seal->term_weight_store[seal->current_store_index] == nullptr) {
    latch_device_fault_v2(control, kIdentityFaultV2);
    return false;
  }
  sources->scalars = seal->scalar_store[seal->current_store_index];
  sources->term_indices = seal->term_index_store[seal->current_store_index];
  sources->term_weights = seal->term_weight_store[seal->current_store_index];
  return true;
}

__device__ std::uint64_t f64_bits(double value) {
  return static_cast<std::uint64_t>(__double_as_longlong(value));
}

__device__ bool full_fixed_stride_gene_equal_v2(
    const GeneScalarV2& left, const std::uint64_t* left_term_indices,
    const double* left_term_weights, std::uint64_t left_ordinal,
    const GeneScalarV2& right, const std::uint64_t* right_term_indices,
    const double* right_term_weights, std::uint64_t right_ordinal) {
  if (left.term_count != right.term_count ||
      left.smc_flags != right.smc_flags ||
      f64_bits(left.long_threshold) != f64_bits(right.long_threshold) ||
      f64_bits(left.short_threshold) != f64_bits(right.short_threshold) ||
      f64_bits(left.target_pips) != f64_bits(right.target_pips) ||
      f64_bits(left.stop_pips) != f64_bits(right.stop_pips) ||
      f64_bits(left.stop_vol_multiplier) !=
          f64_bits(right.stop_vol_multiplier)) {
    return false;
  }
  const std::uint64_t left_base =
      left_ordinal * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2;
  const std::uint64_t right_base =
      right_ordinal * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2;
  for (std::uint32_t term = 0;
       term < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++term) {
    if (left_term_indices[left_base + term] !=
            right_term_indices[right_base + term] ||
        f64_bits(left_term_weights[left_base + term]) !=
            f64_bits(right_term_weights[right_base + term])) {
      return false;
    }
  }
  return true;
}

__device__ bool neighbor_less_v2(const ExactNeighborKeyV2& left,
                                 const ExactNeighborKeyV2& right) {
  if (left.denominator == 0 || right.denominator == 0 ||
      left.denominator > 32 || right.denominator > 32 ||
      left.numerator > left.denominator ||
      right.numerator > right.denominator) {
    return false;
  }
  const std::uint64_t left_product =
      static_cast<std::uint64_t>(left.numerator) * right.denominator;
  const std::uint64_t right_product =
      static_cast<std::uint64_t>(right.numerator) * left.denominator;
  if (left_product != right_product) {
    return left_product < right_product;
  }
  if (left.gene_identity != right.gene_identity) {
    return left.gene_identity < right.gene_identity;
  }
  if (left.source_kind != right.source_kind) {
    return left.source_kind < right.source_kind;
  }
  return left.source_ordinal < right.source_ordinal;
}

__device__ void insert_neighbor_v2(const ExactNeighborKeyV2& candidate,
                                   ExactNeighborKeyV2* selected,
                                   std::uint32_t* selected_count) {
  std::uint32_t count = *selected_count;
  if (count == NEO_RESIDENT_ARCHIVE_KNN_K_V2 &&
      !neighbor_less_v2(candidate, selected[count - 1])) {
    return;
  }
  std::uint32_t position = count;
  if (position == NEO_RESIDENT_ARCHIVE_KNN_K_V2) {
    --position;
  } else {
    ++count;
  }
  while (position != 0 &&
         neighbor_less_v2(candidate, selected[position - 1])) {
    selected[position] = selected[position - 1];
    --position;
  }
  selected[position] = candidate;
  *selected_count = count;
}

__global__ void initialize_archive_control_v2(
    ArchiveControlV2* control, NeoResidentArchiveKnnTerminalV2* terminal,
    const GeneSealV2* seal, GeneViewV2 expected, std::uint64_t run_identity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *control = {};
  *terminal = {};
  control->run_identity = run_identity;
  if (seal == nullptr || seal->current_store_index > 1u ||
      seal->generation_index != expected.expected_generation_index ||
      seal->store_epoch != expected.expected_store_epoch ||
      seal->run_token != expected.expected_run_token ||
      seal->run_token != run_identity ||
      seal->generation_index > kGenerationMaskV2 ||
      seal->store_epoch > kEpochMaskV2) {
    control->device_fault_word = kIdentityFaultV2;
    return;
  }
  control->packed_commit_word =
      pack_commit_word_v2(seal->current_store_index, seal->generation_index,
                          0, seal->store_epoch);
}

__global__ void build_population_signatures_v2(
    const GeneSealV2* seal, GeneViewV2 expected,
    const MetricRowV2* metric_rows, const std::uint64_t* expected_scenarios,
    const ScoringSealV2* scoring_seal, std::uint64_t* signatures,
    std::uint32_t* admission_flags, ArchiveControlV2* control) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= expected.logical_population_count) {
    return;
  }
  DeviceGeneSourcesV2 genes{};
  if (!scoring_seal_valid_v2(scoring_seal)) {
    latch_device_fault_v2(control, kScoringSealFaultV2);
    admission_flags[candidate] = 0;
    return;
  }
  if (!load_current_gene_sources_v2(seal, expected, control, &genes)) {
    admission_flags[candidate] = 0;
    return;
  }
  const GeneScalarV2 scalar = genes.scalars[candidate];
  const MetricRowV2 row = metric_rows[candidate];
  if (scalar.term_count == 0 ||
      scalar.term_count > NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 ||
      row.candidate_id != scalar.gene_identity ||
      row.scenario_id != expected_scenarios[candidate]) {
    latch_device_fault_v2(control, kGeneShapeFaultV2);
    admission_flags[candidate] = 0;
    return;
  }
  for (std::uint32_t metric = 0;
       metric < NEO_RESIDENT_ARCHIVE_KNN_METRIC_COUNT_V2; ++metric) {
    if (!isfinite(row.values[metric])) {
      latch_device_fault_v2(control, kNonFiniteMetricFaultV2);
      admission_flags[candidate] = 0;
      return;
    }
  }

  std::uint64_t local[NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2] = {};
  const std::uint64_t base =
      candidate * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2;
  for (std::uint32_t term = 0;
       term < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++term) {
    const std::uint64_t feature = genes.term_indices[base + term];
    const double weight = genes.term_weights[base + term];
    if (term < scalar.term_count) {
      if (feature >= expected.feature_count || feature / 64ull >=
                                                   NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 ||
          !isfinite(weight)) {
        latch_device_fault_v2(control, kGeneShapeFaultV2);
        admission_flags[candidate] = 0;
        return;
      }
      local[feature / 64ull] |= 1ull << (feature % 64ull);
    } else if (feature != 0 || f64_bits(weight) != 0) {
      latch_device_fault_v2(control, kGeneShapeFaultV2);
      admission_flags[candidate] = 0;
      return;
    }
  }
  bool any_signature_bit = false;
  for (std::uint32_t word = 0;
       word < NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2; ++word) {
    signatures[candidate * NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 + word] =
        local[word];
    any_signature_bit = any_signature_bit || local[word] != 0;
  }
  if (!any_signature_bit) {
    latch_device_fault_v2(control, kSignatureFaultV2);
    admission_flags[candidate] = 0;
    return;
  }
  admission_flags[candidate] =
      row.values[kTradeCountMetricSlotV2] > 0.0 &&
              row.values[kNetMetricSlotV2] > 0.0
          ? 1u
          : 0u;
}

__global__ void exact_archive_population_knn_v2(
    const GeneSealV2* seal, GeneViewV2 expected,
    const std::uint64_t* current_signatures,
    const GeneScalarV2* archive_scalars,
    const std::uint64_t* archive_signatures, ExactNeighborKeyV2* top_k,
    double* novelty_scores, const ScoringSealV2* scoring_seal,
    ArchiveControlV2* control, std::uint64_t archive_capacity) {
  const std::uint64_t query =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= expected.logical_population_count) {
    return;
  }
  DeviceGeneSourcesV2 genes{};
  if (!scoring_seal_valid_v2(scoring_seal)) {
    latch_device_fault_v2(control, kScoringSealFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }
  if (!load_current_gene_sources_v2(seal, expected, control, &genes)) {
    novelty_scores[query] = 0.0;
    return;
  }
  const std::uint64_t source_commit =
      atomic_read_commit_v2(&control->packed_commit_word);
  const std::uint64_t archive_count = unpack_archive_count_v2(source_commit);
  if (archive_count > archive_capacity) {
    latch_device_fault_v2(control, kArchiveBoundFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }

  ExactNeighborKeyV2 selected[NEO_RESIDENT_ARCHIVE_KNN_K_V2] = {};
  std::uint32_t selected_count = 0;
  const std::uint64_t* query_signature =
      current_signatures +
      query * NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2;
  const std::uint64_t available =
      expected.logical_population_count - 1ull + archive_count;
  if (available == 0) {
    latch_device_fault_v2(control, kNeighborBoundFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }

  const std::uint64_t neighbor_extent =
      expected.logical_population_count + archive_count;
  for (std::uint64_t combined = 0; combined < neighbor_extent; ++combined) {
    const bool current = combined < expected.logical_population_count;
    const std::uint64_t ordinal =
        current ? combined : combined - expected.logical_population_count;
    if (current && ordinal == query) {
      continue;
    }
    const std::uint64_t* signature =
        current
            ? current_signatures +
                  ordinal * NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2
            : archive_signatures +
                  ordinal * NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2;
    std::uint32_t intersection = 0;
    std::uint32_t union_count = 0;
#pragma unroll
    for (std::uint32_t word = 0;
         word < NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2; ++word) {
      intersection += __popcll(query_signature[word] & signature[word]);
      union_count += __popcll(query_signature[word] | signature[word]);
    }
    if (union_count == 0 || union_count > 32 ||
        intersection > union_count || ordinal > 0xffffffffull) {
      latch_device_fault_v2(control, kNeighborBoundFaultV2);
      novelty_scores[query] = 0.0;
      return;
    }
    ExactNeighborKeyV2 neighbor{};
    neighbor.numerator = union_count - intersection;
    neighbor.denominator = union_count;
    neighbor.gene_identity =
        current ? genes.scalars[ordinal].gene_identity
                : archive_scalars[ordinal].gene_identity;
    neighbor.source_kind = current ? kSourceCurrentV2 : kSourceArchiveV2;
    neighbor.source_ordinal = static_cast<std::uint32_t>(ordinal);
    insert_neighbor_v2(neighbor, selected, &selected_count);
  }

  const std::uint32_t expected_count =
      static_cast<std::uint32_t>(
          available < NEO_RESIDENT_ARCHIVE_KNN_K_V2
              ? available
              : NEO_RESIDENT_ARCHIVE_KNN_K_V2);
  if (selected_count != expected_count || selected_count == 0) {
    latch_device_fault_v2(control, kNeighborBoundFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }
  double sum = 0.0;
  for (std::uint32_t neighbor = 0;
       neighbor < NEO_RESIDENT_ARCHIVE_KNN_K_V2; ++neighbor) {
    const std::uint64_t output =
        query * NEO_RESIDENT_ARCHIVE_KNN_K_V2 + neighbor;
    top_k[output] = neighbor < selected_count ? selected[neighbor]
                                             : ExactNeighborKeyV2{};
    if (neighbor < selected_count) {
      const double numerator = __uint2double_rn(selected[neighbor].numerator);
      const double denominator =
          __uint2double_rn(selected[neighbor].denominator);
      const double term = __ddiv_rn(numerator, denominator);
      sum = __dadd_rn(sum, term);
    }
  }
  const double novelty =
      __ddiv_rn(sum, __uint2double_rn(selected_count));
  if (!isfinite(novelty) || novelty < 0.0) {
    latch_device_fault_v2(control, kNeighborBoundFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }
  novelty_scores[query] = novelty;
}

__device__ std::uint64_t ordered_finite_f64_key_v2(double value) {
  if (!isfinite(value)) {
    return 0;
  }
  const double canonical = value == 0.0 ? 0.0 : value;
  const std::uint64_t bits = f64_bits(canonical);
  const std::uint64_t ordered =
      (bits >> 63) == 0 ? bits ^ (1ull << 63) : ~bits;
  return ordered == 0 ? 1 : ordered;
}

__global__ void build_blended_rank_inputs_v2(
    const GeneSealV2* seal, GeneViewV2 expected, const double* fitness_scores,
    const double* novelty_scores, std::uint64_t* decision_keys,
    std::uint64_t* ordinal_keys, std::uint64_t* ordinal_values,
    const ScoringSealV2* scoring_seal, ArchiveControlV2* control) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  control->ranked_ready = 0;
  control->staged_ready = 0;
  control->staged_count = 0;
  control->staged_collision_count = 0;
  DeviceGeneSourcesV2 genes{};
  if (!scoring_seal_valid_v2(scoring_seal)) {
    latch_device_fault_v2(control, kScoringSealFaultV2);
    return;
  }
  if (!load_current_gene_sources_v2(seal, expected, control, &genes)) {
    return;
  }
  double minimum_fitness = DBL_MAX;
  double maximum_fitness = -DBL_MAX;
  double maximum_novelty = 0.0;
  for (std::uint64_t candidate = 0;
       candidate < expected.logical_population_count; ++candidate) {
    if (!isfinite(fitness_scores[candidate]) ||
        !isfinite(novelty_scores[candidate]) || novelty_scores[candidate] < 0.0) {
      latch_device_fault_v2(control, kNonFiniteMetricFaultV2);
      return;
    }
    minimum_fitness = fitness_scores[candidate] < minimum_fitness
                          ? fitness_scores[candidate]
                          : minimum_fitness;
    maximum_fitness = fitness_scores[candidate] > maximum_fitness
                          ? fitness_scores[candidate]
                          : maximum_fitness;
    maximum_novelty = novelty_scores[candidate] > maximum_novelty
                          ? novelty_scores[candidate]
                          : maximum_novelty;
  }

  double fitness_range = __dsub_rn(maximum_fitness, minimum_fitness);
  fitness_range = fitness_range < 1.0e-9 ? 1.0e-9 : fitness_range;
  const double novelty_range =
      maximum_novelty < 1.0e-9 ? 1.0e-9 : maximum_novelty;
  const double novelty_weight =
      __longlong_as_double(
          static_cast<long long>(NEO_RESIDENT_ARCHIVE_KNN_NOVELTY_WEIGHT_BITS_V2));
  const double fitness_weight = __dsub_rn(1.0, novelty_weight);
  for (std::uint64_t candidate = 0;
       candidate < expected.logical_population_count; ++candidate) {
    const double normalized_fitness =
        __ddiv_rn(__dsub_rn(fitness_scores[candidate], minimum_fitness),
                  fitness_range);
    const double normalized_novelty =
        __ddiv_rn(novelty_scores[candidate], novelty_range);
    const double blended = __dadd_rn(
        __dmul_rn(fitness_weight, normalized_fitness),
        __dmul_rn(novelty_weight, normalized_novelty));
    decision_keys[candidate] = ordered_finite_f64_key_v2(blended);
    if (decision_keys[candidate] == 0) {
      latch_device_fault_v2(control, kNonFiniteMetricFaultV2);
      return;
    }
    ordinal_keys[candidate] = candidate;
    ordinal_values[candidate] = candidate;
  }
}

__global__ void gather_gene_identity_rank_keys_v2(
    const GeneSealV2* seal, GeneViewV2 expected,
    const std::uint64_t* ranked_ordinals,
    std::uint64_t* gene_identity_keys, ArchiveControlV2* control) {
  const std::uint64_t rank =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (rank >= expected.logical_population_count) {
    return;
  }
  DeviceGeneSourcesV2 genes{};
  if (!load_current_gene_sources_v2(seal, expected, control, &genes)) {
    return;
  }
  const std::uint64_t ordinal = ranked_ordinals[rank];
  if (ordinal >= expected.logical_population_count) {
    latch_device_fault_v2(control, kGeneShapeFaultV2);
    return;
  }
  gene_identity_keys[rank] = genes.scalars[ordinal].gene_identity;
}

__global__ void gather_blended_rank_keys_v2(
    const std::uint64_t* decision_keys,
    const std::uint64_t* ranked_ordinals, std::uint64_t* blended_keys,
    std::uint64_t population_count, ArchiveControlV2* control) {
  const std::uint64_t rank =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (rank >= population_count) {
    return;
  }
  const std::uint64_t ordinal = ranked_ordinals[rank];
  if (ordinal >= population_count) {
    latch_device_fault_v2(control, kGeneShapeFaultV2);
    return;
  }
  blended_keys[rank] = decision_keys[ordinal];
}

__global__ void copy_ranked_ordinals_v2(
    const std::uint64_t* ranked_ordinals,
    std::uint64_t* admission_offsets, std::uint64_t population_count,
    ArchiveControlV2* control) {
  const std::uint64_t rank =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (rank >= population_count) {
    return;
  }
  if (ranked_ordinals[rank] >= population_count) {
    latch_device_fault_v2(control, kGeneShapeFaultV2);
    return;
  }
  admission_offsets[rank] = ranked_ordinals[rank];
}

__global__ void seal_ranked_population_v2(
    const std::uint64_t* ranked_ordinals, std::uint64_t population_count,
    ArchiveControlV2* control) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (std::uint64_t rank = 0; rank < population_count; ++rank) {
    if (ranked_ordinals[rank] >= population_count) {
      latch_device_fault_v2(control, kGeneShapeFaultV2);
      return;
    }
  }
  control->ranked_source_commit_word =
      atomic_read_commit_v2(&control->packed_commit_word);
  control->ranked_ready = control->device_fault_word == 0 ? 1u : 0u;
}

__global__ void stage_ranked_archive_tail_v2(
    const GeneSealV2* seal, GeneViewV2 expected,
    const MetricRowV2* current_metrics,
    const std::uint64_t* current_signatures,
    const std::uint64_t* ranked_ordinals, std::uint32_t* admission_flags,
    std::uint64_t* admission_offsets, double* fitness_scores_scratch,
    GeneScalarV2* archive_scalars,
    std::uint64_t* archive_term_indices, double* archive_term_weights,
    MetricRowV2* archive_metrics, std::uint64_t* archive_signatures,
    std::uint64_t* archive_hashes, ArchiveControlV2* control,
    std::uint64_t archive_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  DeviceGeneSourcesV2 current{};
  if (!load_current_gene_sources_v2(seal, expected, control, &current)) {
    return;
  }
  const std::uint64_t source_commit =
      atomic_read_commit_v2(&control->packed_commit_word);
  if (control->ranked_ready != 1u ||
      source_commit != control->ranked_source_commit_word) {
    latch_device_fault_v2(control, kPublicationFaultV2);
    return;
  }
  const std::uint64_t committed = unpack_archive_count_v2(source_commit);
  if (committed > archive_capacity) {
    latch_device_fault_v2(control, kArchiveBoundFaultV2);
    return;
  }

  if (fitness_scores_scratch == nullptr) {
    latch_device_fault_v2(control, kGeneShapeFaultV2);
    return;
  }
  auto* staged_destinations =
      reinterpret_cast<std::uint64_t*>(fitness_scores_scratch);
  std::uint64_t staged = 0;
  std::uint64_t collisions = 0;
  for (std::uint64_t candidate = 0;
       candidate < expected.logical_population_count; ++candidate) {
    staged_destinations[candidate] = ~std::uint64_t{0};
  }
  for (std::uint64_t rank = 0;
       rank < expected.logical_population_count; ++rank) {
    const std::uint64_t candidate = ranked_ordinals[rank];
    if (candidate >= expected.logical_population_count ||
        admission_flags[candidate] == 0) {
      continue;
    }
    const GeneScalarV2 scalar = current.scalars[candidate];
    bool duplicate = false;
    for (std::uint64_t archived = 0; archived < committed + staged;
         ++archived) {
      if (archive_hashes[archived] != scalar.content_hash) {
        continue;
      }
      if (full_fixed_stride_gene_equal_v2(
              scalar, current.term_indices, current.term_weights, candidate,
              archive_scalars[archived], archive_term_indices,
              archive_term_weights, archived)) {
        duplicate = true;
        break;
      }
      ++collisions;
    }
    if (duplicate) {
      admission_flags[candidate] = 2u;
      continue;
    }
    if (committed + staged >= archive_capacity) {
      admission_flags[candidate] = 3u;
      continue;
    }

    const std::uint64_t destination = committed + staged;
    archive_scalars[destination] = scalar;
    archive_metrics[destination] = current_metrics[candidate];
    archive_hashes[destination] = scalar.content_hash;
#pragma unroll
    for (std::uint32_t term = 0;
         term < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++term) {
      archive_term_indices[
          destination * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + term] =
          current.term_indices[
              candidate * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + term];
      archive_term_weights[
          destination * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + term] =
          current.term_weights[
              candidate * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + term];
    }
#pragma unroll
    for (std::uint32_t word = 0;
         word < NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2; ++word) {
      archive_signatures[
          destination * NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 + word] =
          current_signatures[
              candidate * NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 + word];
    }
    staged_destinations[candidate] = destination;
    admission_flags[candidate] = 4u;
    ++staged;
  }
  for (std::uint64_t candidate = 0;
       candidate < expected.logical_population_count; ++candidate) {
    admission_offsets[candidate] = staged_destinations[candidate];
  }
  control->staged_count = staged;
  control->staged_collision_count = collisions;
  control->staged_ready = control->device_fault_word == 0 ? 1u : 0u;
}

__global__ void publish_generation_and_archive_v2(
    PreparedAdvanceV2 prepared, ArchiveControlV2* control,
    std::uint64_t archive_capacity, std::uint64_t run_identity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t source_commit =
      atomic_read_commit_v2(&control->packed_commit_word);
  const std::uint64_t committed_archive_count =
      unpack_archive_count_v2(source_commit);
  if (control->run_identity != run_identity ||
      control->ranked_source_commit_word != source_commit ||
      control->ranked_ready != 1u || control->staged_ready != 1u ||
      committed_archive_count > archive_capacity ||
      control->staged_count > archive_capacity - committed_archive_count) {
    latch_device_fault_v2(control, kPublicationFaultV2);
  }
  const auto generation_result =
      prepared.publish_device_v2(control->device_fault_word);
  if (generation_result.combined_fault != 0 ||
      generation_result.committed != 1u) {
    latch_device_fault_v2(
        control, generation_result.combined_fault == 0
                     ? kPublicationFaultV2
                     : generation_result.combined_fault);
    control->staged_count = 0;
    control->staged_ready = 0;
    return;
  }
  const std::uint64_t target_archive_count =
      committed_archive_count + control->staged_count;
  if (generation_result.current_store_index > 1u ||
      generation_result.generation_index > kGenerationMaskV2 ||
      target_archive_count > kArchiveMaskV2 ||
      generation_result.store_epoch > kEpochMaskV2) {
    latch_device_fault_v2(control, kPublicationFaultV2);
    control->staged_count = 0;
    control->staged_ready = 0;
    return;
  }
  const std::uint64_t target_commit = pack_commit_word_v2(
      generation_result.current_store_index,
      generation_result.generation_index, target_archive_count,
      generation_result.store_epoch);
  control->committed_collision_count += control->staged_collision_count;
  control->publication_count += 1u;
  control->ranked_ready = 0;
  control->staged_ready = 0;
  control->staged_count = 0;
  control->staged_collision_count = 0;
  __threadfence();
  atomicExch(&control->packed_commit_word,
             static_cast<unsigned long long>(target_commit));
}

__host__ __device__ std::uint64_t terminal_digest_v2(
    std::uint64_t commit, std::uint64_t collisions,
    std::uint64_t run_identity, std::uint32_t device_fault) {
  std::uint64_t digest = 1469598103934665603ull;
  const std::uint64_t lanes[4] = {commit, collisions, run_identity,
                                  device_fault};
#pragma unroll
  for (std::uint32_t lane = 0; lane < 4; ++lane) {
#pragma unroll
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
      digest ^= (lanes[lane] >> (byte * 8)) & 0xffull;
      digest *= 1099511628211ull;
    }
  }
  return digest;
}

__global__ void seal_archive_terminal_v2(
    ArchiveControlV2* control, NeoResidentArchiveKnnTerminalV2* terminal,
    std::uint64_t receipt_identity, std::uint64_t run_identity,
    std::uint64_t completion_event_identity,
    std::uint64_t final_same_stream_enqueue_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t packed_commit =
      atomic_read_commit_v2(&control->packed_commit_word);
  *terminal = {};
  terminal->abi_version = NEO_RESIDENT_ARCHIVE_KNN_ABI_V2;
  terminal->terminal_status =
      control->device_fault_word == 0 && control->validation_fault_word == 0
          ? NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2
          : NEO_ARCHIVE_KNN_TERMINAL_FAULT_V2;
  terminal->device_fault_word = control->device_fault_word;
  terminal->validation_fault_word = control->validation_fault_word;
  terminal->receipt_identity = receipt_identity;
  terminal->run_identity = run_identity;
  terminal->packed_commit_word = packed_commit;
  terminal->collision_count = control->committed_collision_count;
  terminal->compact_async_d2h_count = 1;
  terminal->compact_async_d2h_bytes =
      sizeof(NeoResidentArchiveKnnTerminalV2);
  terminal->completion_event_query_count = 0;
  terminal->completion_stream_synchronize_count = 0;
  terminal->same_stream_enqueue_count = final_same_stream_enqueue_count;
  terminal->completion_event_identity = completion_event_identity;
  terminal->validator_digest = terminal_digest_v2(
      packed_commit, control->committed_collision_count, run_identity,
      control->device_fault_word);
  control->terminal_status = terminal->terminal_status;
  control->same_stream_enqueue_count = final_same_stream_enqueue_count;
  control->validator_digest = terminal->validator_digest;
}

bool valid_finite_rows_v2(const FiniteRowsV2& rows,
                          const NeoResidentArchiveKnnOwnerV2& owner);

}  // namespace

struct NeoResidentArchiveKnnOwnerV2 {
  resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* scoring;
  resident_generation_v1::NeoResidentGenerationRunV1* generation;
  GeneViewV2 retained_gene_view;
  NeoResidentArchiveKnnBindV2 binding;
  resident_scoring_novelty_v2_internal::ResidentScoringArenaAccessV2
      arena_access;
  FiniteRowsV2 finite_rows;
  PreparedAdvanceV2 prepared_generation;
  TerminalLifecycleV2 terminal_lifecycle;

  double* fitness_scores;
  std::uint64_t* decision_keys;
  void* cub_scratch;
  GeneScalarV2* archive_gene_scalars;
  std::uint64_t* archive_term_indices;
  double* archive_term_weights;
  MetricRowV2* archive_metric_rows;
  std::uint64_t* archive_signatures;
  std::uint64_t* archive_hashes;
  std::uint64_t* current_population_signatures;
  double* novelty_scores;
  ExactNeighborKeyV2* exact_top_k_keys;
  std::uint32_t* admission_flags;
  std::uint64_t* admission_offsets;
  ArchiveControlV2* control;
  NeoResidentArchiveKnnTerminalV2* terminal_device;
  NeoResidentArchiveKnnTerminalV2* terminal_host;

  const NeoResidentArchiveKnnPendingV2* pending_identity;
  std::uint64_t initial_source_commit_word;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t completion_event_query_count;
  HostPhaseV2 phase;
  bool poisoned;
  bool terminal_event_proven;
};

namespace {

bool valid_finite_rows_v2(const FiniteRowsV2& rows,
                          const NeoResidentArchiveKnnOwnerV2& owner) {
  return rows.scoring_owner == owner.scoring &&
         rows.admitted_run_stream == owner.arena_access.admitted_run_stream &&
         rows.metric_rows_device != nullptr &&
         rows.expected_scenario_ids_device != nullptr &&
         rows.fitness_scores_device == owner.fitness_scores &&
         rows.decision_keys_device == owner.decision_keys &&
         rows.device_seal != nullptr &&
         rows.logical_population_count == owner.binding.population_count;
}

bool advance_global_enqueue_count_v2(NeoResidentArchiveKnnOwnerV2* owner,
                                     std::uint64_t delta) {
  if (owner == nullptr ||
      delta > std::numeric_limits<std::uint64_t>::max() -
                  owner->same_stream_enqueue_count) {
    return false;
  }
  owner->same_stream_enqueue_count += delta;
  return true;
}

void partition_owner_v2(NeoResidentArchiveKnnOwnerV2* owner) {
  void* base = owner->arena_access.allocation_base;
  owner->fitness_scores =
      region_pointer_v2<double>(base, owner->binding.fitness_scores);
  owner->decision_keys =
      region_pointer_v2<std::uint64_t>(base, owner->binding.decision_keys);
  owner->cub_scratch =
      region_pointer_v2<void>(base, owner->binding.cub_scratch);
  owner->archive_gene_scalars =
      region_pointer_v2<GeneScalarV2>(base, owner->binding.archive_gene_scalars);
  owner->archive_term_indices = region_pointer_v2<std::uint64_t>(
      base, owner->binding.archive_term_indices);
  owner->archive_term_weights =
      region_pointer_v2<double>(base, owner->binding.archive_term_weights);
  owner->archive_metric_rows =
      region_pointer_v2<MetricRowV2>(base, owner->binding.archive_metric_rows);
  owner->archive_signatures = region_pointer_v2<std::uint64_t>(
      base, owner->binding.archive_signatures);
  owner->archive_hashes =
      region_pointer_v2<std::uint64_t>(base, owner->binding.archive_hashes);
  owner->current_population_signatures = region_pointer_v2<std::uint64_t>(
      base, owner->binding.current_population_signatures);
  owner->novelty_scores =
      region_pointer_v2<double>(base, owner->binding.novelty_scores);
  owner->exact_top_k_keys = region_pointer_v2<ExactNeighborKeyV2>(
      base, owner->binding.exact_top_k_keys);
  owner->admission_flags =
      region_pointer_v2<std::uint32_t>(base, owner->binding.admission_flags);
  owner->admission_offsets = region_pointer_v2<std::uint64_t>(
      base, owner->binding.admission_offsets);
  auto* shared_control = region_pointer_v2<std::uint8_t>(
      base, owner->binding.archive_control_and_seal);
  owner->control = reinterpret_cast<ArchiveControlV2*>(
      shared_control + kArchiveControlPrefixBytesV2);
  owner->terminal_device =
      reinterpret_cast<NeoResidentArchiveKnnTerminalV2*>(
          reinterpret_cast<std::uint8_t*>(owner->control) +
          sizeof(ArchiveControlV2));
}

std::int32_t poison_owner_v2(NeoResidentArchiveKnnOwnerV2* owner,
                             std::int32_t status) {
  if (owner != nullptr) {
    owner->poisoned = true;
  }
  return status;
}

bool exact_pending_v2(const NeoResidentArchiveKnnOwnerV2& owner,
                      const NeoResidentArchiveKnnPendingV2& pending) {
  return pending.abi_version == NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 &&
         pending.flags == 0 &&
         pending.source_packed_commit_word ==
             owner.initial_source_commit_word &&
         pending.terminal_device_receipt_identity ==
             reinterpret_cast<std::uint64_t>(owner.terminal_device) &&
         pending.run_identity == owner.binding.run_identity &&
         pending.boxed_receipt_identity ==
             reinterpret_cast<std::uint64_t>(&pending) &&
         pending.staged_dependency_identity ==
             reinterpret_cast<std::uint64_t>(&owner.retained_gene_view) &&
         pending.same_stream_enqueue_count == owner.same_stream_enqueue_count &&
         pending.completion_event_identity ==
             owner.terminal_lifecycle.completion_event_identity_v2() &&
         pending.terminal_host_receipt_identity ==
             reinterpret_cast<std::uint64_t>(owner.terminal_host);
}

}  // namespace

extern "C" std::int32_t bind_preallocated_resident_archive_knn_v2(
    resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* scoring,
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const resident_generation_v2::NeoResidentGenerationGeneViewV2* genes,
    const NeoResidentArchiveKnnBindV2* binding,
    NeoResidentArchiveKnnOwnerV2** owner) {
  if (scoring == nullptr || generation == nullptr || genes == nullptr ||
      binding == nullptr || owner == nullptr || *owner != nullptr ||
      binding->reserved != 0 || binding->reserved_extents != 0) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (!validate_binding_layout_v2(*binding)) {
    return NEO_ARCHIVE_KNN_STATUS_ABI_MISMATCH_V2;
  }
  if (genes->abi_version !=
          resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2 ||
      genes->flags != 0 || genes->seal_device == nullptr ||
      genes->control_device == nullptr ||
      genes->expected_run_token != binding->run_identity ||
      genes->logical_population_count != binding->population_count ||
      genes->max_terms_per_gene != binding->max_terms_per_gene ||
      genes->feature_count >
          binding->signature_word_count * std::uint64_t{64}) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  std::int32_t status =
      resident_generation_v2::validate_resident_gene_view_owner_v2(generation,
                                                                    genes);
  if (status != 0) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }

  resident_scoring_novelty_v2_internal::ResidentScoringArenaAccessV2 access{};
  status = resident_scoring_novelty_v2_internal::
      borrow_resident_scoring_archive_arena_v2(scoring, binding, &access);
  if (status != 0 || access.admitted_run_stream == nullptr ||
      access.allocation_base == nullptr ||
      access.allocation_bytes != binding->total_device_bytes ||
      reinterpret_cast<std::uintptr_t>(access.allocation_base) %
              kAlignmentV2 !=
          0) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }

  TerminalLifecycleV2 lifecycle{};
  const bool lifecycle_borrowed = resident_generation_v2_internal::
      borrow_resident_generation_terminal_lifecycle_v2(
          generation, sizeof(NeoResidentArchiveKnnTerminalV2), &lifecycle);
  if (!lifecycle_borrowed || lifecycle.generation_owner_v2() != generation ||
      lifecycle.admitted_run_stream_v2() != access.admitted_run_stream ||
      lifecycle.completion_event_v2() == nullptr ||
      lifecycle.terminal_host_receipt_v2() == nullptr ||
      lifecycle.terminal_host_receipt_bytes_v2() !=
          sizeof(NeoResidentArchiveKnnTerminalV2) ||
      lifecycle.completion_event_identity_v2() == 0 ||
      lifecycle.source_ready_receipt_v2() == nullptr ||
      lifecycle.resident_parent_ready_event_v2() == nullptr ||
      lifecycle.source_event_id_v2() == 0 ||
      lifecycle.source_ready_receipt_v2()->abi_version !=
          resident_generation_v1::NEO_RESIDENT_GENERATION_ABI_V1 ||
      lifecycle.source_ready_receipt_v2()->reserved != 0u ||
      lifecycle.source_ready_receipt_v2()->event_id !=
          lifecycle.source_event_id_v2() ||
      lifecycle.source_ready_receipt_v2()->generation_index !=
          genes->expected_generation_index ||
      lifecycle.source_ready_receipt_v2()->same_stream_enqueue_count !=
          lifecycle.source_same_stream_enqueue_count_v2() ||
      lifecycle.source_ready_receipt_v2()->intermediate_host_wait_count != 0 ||
      lifecycle.source_ready_receipt_v2()->intermediate_readback_count != 0 ||
      lifecycle.source_same_stream_enqueue_count_v2() ==
          std::numeric_limits<std::uint64_t>::max() ||
      lifecycle.run_token_v2() != binding->run_identity ||
      lifecycle.generation_index_v2() != genes->expected_generation_index ||
      lifecycle.store_epoch_v2() != genes->expected_store_epoch ||
      lifecycle.current_store_index_v2() > 1u ||
      lifecycle.generation_index_v2() > kGenerationMaskV2 ||
      lifecycle.store_epoch_v2() > kEpochMaskV2) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }

  auto* created = new (std::nothrow) NeoResidentArchiveKnnOwnerV2{};
  if (created == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }
  created->scoring = scoring;
  created->generation = generation;
  created->retained_gene_view = *genes;
  created->binding = *binding;
  created->arena_access = access;
  created->terminal_lifecycle = lifecycle;
  created->terminal_host =
      static_cast<NeoResidentArchiveKnnTerminalV2*>(
          lifecycle.terminal_host_receipt_v2());
  created->phase = HostPhaseV2::Bound;
  created->same_stream_enqueue_count =
      lifecycle.source_same_stream_enqueue_count_v2();
  created->initial_source_commit_word = pack_commit_word_v2(
      lifecycle.current_store_index_v2(), lifecycle.generation_index_v2(), 0,
      lifecycle.store_epoch_v2());
  partition_owner_v2(created);

  initialize_archive_control_v2<<<1, 1, 0, access.admitted_run_stream>>>(
      created->control, created->terminal_device, genes->seal_device,
      created->retained_gene_view, binding->run_identity);
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    delete created;
    return status;
  }
  ++created->same_stream_enqueue_count;
  *owner = created;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t enqueue_resident_archive_score_and_rank_v2(
    NeoResidentArchiveKnnOwnerV2* owner,
    const resident_search_generation_v2::NeoResidentScoringPopulationSourceV2*
        population,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1*
        dependency) {
  if (owner == nullptr || population == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->poisoned || owner->pending_identity != nullptr ||
      (owner->phase != HostPhaseV2::Bound &&
       owner->phase != HostPhaseV2::Published)) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }
  const bool initial_generation = owner->phase == HostPhaseV2::Bound;
  const bool continued_generation = owner->phase == HostPhaseV2::Published;
  if ((initial_generation && dependency == nullptr) ||
      (continued_generation && dependency != nullptr)) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  if (initial_generation &&
      (dependency != owner->terminal_lifecycle.source_ready_receipt_v2() ||
       dependency->abi_version !=
           resident_generation_v1::NEO_RESIDENT_GENERATION_ABI_V1 ||
       dependency->reserved != 0 ||
       dependency->event_id !=
           owner->terminal_lifecycle.source_event_id_v2() ||
       dependency->generation_index !=
           owner->retained_gene_view.expected_generation_index ||
       dependency->same_stream_enqueue_count !=
           owner->terminal_lifecycle.source_same_stream_enqueue_count_v2() ||
       dependency->intermediate_host_wait_count != 0 ||
       dependency->intermediate_readback_count != 0)) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  if (population->logical_population_count != owner->binding.population_count ||
      population->max_terms_per_gene != owner->binding.max_terms_per_gene ||
      population->feature_count != owner->retained_gene_view.feature_count ||
      population->admitted_run_stream != owner->arena_access.admitted_run_stream ||
      population->metrics_ready_event !=
          owner->terminal_lifecycle.resident_parent_ready_event_v2() ||
      population->population_lifetime_owner !=
          owner->terminal_lifecycle.population_lifetime_owner_v2()) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }

  resident_scoring_novelty_v2_internal::ResidentScoringArenaAccessV2
      scoring_before{};
  std::int32_t status = resident_scoring_novelty_v2_internal::
      borrow_resident_scoring_archive_arena_v2(
          owner->scoring, &owner->binding, &scoring_before);
  if (status != 0 ||
      scoring_before.admitted_run_stream !=
          owner->arena_access.admitted_run_stream ||
      scoring_before.allocation_base != owner->arena_access.allocation_base ||
      scoring_before.allocation_bytes != owner->arena_access.allocation_bytes) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2);
  }

  FiniteRowsV2 finite_rows{};
  status = resident_scoring_novelty_v2_internal::
      enqueue_resident_scoring_finite_objective_v2(owner->scoring, population,
                                                    &finite_rows);
  if (status != 0 || !valid_finite_rows_v2(finite_rows, *owner) ||
      finite_rows.same_stream_enqueue_count <
          scoring_before.same_stream_enqueue_count ||
      !advance_global_enqueue_count_v2(
          owner, finite_rows.same_stream_enqueue_count -
                     scoring_before.same_stream_enqueue_count)) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_DEVICE_FAULT_V2);
  }
  owner->finite_rows = finite_rows;

  const cudaStream_t stream = owner->arena_access.admitted_run_stream;
  build_population_signatures_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      finite_rows.metric_rows_device, finite_rows.expected_scenario_ids_device,
      finite_rows.device_seal, owner->current_population_signatures,
      owner->admission_flags, owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  exact_archive_population_knn_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      owner->current_population_signatures, owner->archive_gene_scalars,
      owner->archive_signatures, owner->exact_top_k_keys,
      owner->novelty_scores, finite_rows.device_seal, owner->control,
      owner->binding.archive_capacity);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  auto* rank_keys_a = owner->current_population_signatures;
  auto* rank_keys_b = rank_keys_a + owner->binding.population_count;
  auto* rank_values_a = rank_keys_b + owner->binding.population_count;
  auto* rank_values_b = rank_values_a + owner->binding.population_count;

  build_blended_rank_inputs_v2<<<1, 1, 0, stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      owner->fitness_scores, owner->novelty_scores, owner->decision_keys,
      rank_keys_a, rank_values_a, finite_rows.device_seal, owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }

  std::size_t scratch_bytes = static_cast<std::size_t>(
      owner->binding.cub_scratch.size_bytes);
  cudaError_t cub_status = cub::DeviceRadixSort::SortPairs(
      owner->cub_scratch, scratch_bytes, rank_keys_a, rank_keys_b,
      rank_values_a, rank_values_b,
      static_cast<int>(owner->binding.population_count), 0, 64, stream);
  if (cub_status != cudaSuccess) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUB_ERROR_V2);
  }
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }

  gather_gene_identity_rank_keys_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      rank_values_b, rank_keys_a, owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }

  scratch_bytes = static_cast<std::size_t>(
      owner->binding.cub_scratch.size_bytes);
  cub_status = cub::DeviceRadixSort::SortPairs(
      owner->cub_scratch, scratch_bytes, rank_keys_a, rank_keys_b,
      rank_values_b, rank_values_a,
      static_cast<int>(owner->binding.population_count), 0, 64, stream);
  if (cub_status != cudaSuccess) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUB_ERROR_V2);
  }
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }

  gather_blended_rank_keys_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      owner->decision_keys, rank_values_a, rank_keys_a,
      owner->binding.population_count, owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }

  scratch_bytes = static_cast<std::size_t>(
      owner->binding.cub_scratch.size_bytes);
  cub_status = cub::DeviceRadixSort::SortPairsDescending(
      owner->cub_scratch, scratch_bytes, rank_keys_a, rank_keys_b,
      rank_values_a, rank_values_b,
      static_cast<int>(owner->binding.population_count), 0, 64, stream);
  if (cub_status != cudaSuccess) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUB_ERROR_V2);
  }
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }

  copy_ranked_ordinals_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      rank_values_b, owner->admission_offsets,
      owner->binding.population_count, owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }

  build_population_signatures_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      finite_rows.metric_rows_device, finite_rows.expected_scenario_ids_device,
      finite_rows.device_seal, owner->current_population_signatures,
      owner->admission_flags, owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }

  seal_ranked_population_v2<<<1, 1, 0, stream>>>(
      owner->admission_offsets, owner->binding.population_count,
      owner->control);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  owner->phase = HostPhaseV2::Ranked;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t enqueue_resident_archive_stage_from_rank_v2(
    NeoResidentArchiveKnnOwnerV2* owner) {
  if (owner == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->poisoned || owner->phase != HostPhaseV2::Ranked ||
      !valid_finite_rows_v2(owner->finite_rows, *owner)) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }
  stage_ranked_archive_tail_v2<<<
      1, 1, 0, owner->arena_access.admitted_run_stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      owner->finite_rows.metric_rows_device,
      owner->current_population_signatures, owner->admission_offsets,
      owner->admission_flags, owner->admission_offsets,
      owner->fitness_scores,
      owner->archive_gene_scalars, owner->archive_term_indices,
      owner->archive_term_weights, owner->archive_metric_rows,
      owner->archive_signatures, owner->archive_hashes, owner->control,
      owner->binding.archive_capacity);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  const std::int32_t status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  owner->phase = HostPhaseV2::Staged;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t enqueue_resident_archive_evolve_and_publish_v2(
    NeoResidentArchiveKnnOwnerV2* owner) {
  if (owner == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->poisoned || owner->phase != HostPhaseV2::Staged ||
      !valid_finite_rows_v2(owner->finite_rows, *owner)) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }
  TerminalLifecycleV2 generation_before{};
  if (!resident_generation_v2_internal::
           borrow_resident_generation_terminal_lifecycle_v2(
               owner->generation, sizeof(NeoResidentArchiveKnnTerminalV2),
               &generation_before) ||
      generation_before.admitted_run_stream_v2() !=
          owner->arena_access.admitted_run_stream ||
      generation_before.run_token_v2() != owner->binding.run_identity) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2);
  }
  PreparedAdvanceV2 prepared{};
  std::int32_t status = resident_generation_v2_internal::
      enqueue_resident_generation_offspring_from_finite_rows_v2(
          owner->generation, &owner->finite_rows, owner->decision_keys,
          &owner->retained_gene_view, &prepared);
  if (status != 0) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_DEVICE_FAULT_V2);
  }
  owner->prepared_generation = prepared;
  publish_generation_and_archive_v2<<<
      1, 1, 0, owner->arena_access.admitted_run_stream>>>(
      prepared, owner->control, owner->binding.archive_capacity,
      owner->binding.run_identity);
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  status = resident_generation_v2_internal::
      accept_resident_generation_combined_publish_v2(&prepared);
  if (status != 0) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2);
  }
  TerminalLifecycleV2 generation_after{};
  if (!resident_generation_v2_internal::
           borrow_resident_generation_terminal_lifecycle_v2(
               owner->generation, sizeof(NeoResidentArchiveKnnTerminalV2),
               &generation_after) ||
      generation_after.same_stream_enqueue_count_v2() <
          generation_before.same_stream_enqueue_count_v2()) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2);
  }
  const std::uint64_t generation_delta =
      generation_after.same_stream_enqueue_count_v2() -
      generation_before.same_stream_enqueue_count_v2();
  if (generation_delta > std::numeric_limits<std::uint64_t>::max() -
                             owner->same_stream_enqueue_count) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  owner->same_stream_enqueue_count += generation_delta;
  owner->phase = HostPhaseV2::Published;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t enqueue_resident_archive_terminal_seal_v2(
    NeoResidentArchiveKnnOwnerV2* owner,
    NeoResidentArchiveKnnPendingV2* pending) {
  if (owner == nullptr || pending == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->poisoned || owner->phase != HostPhaseV2::Published ||
      owner->pending_identity != nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }

  TerminalLifecycleV2 lifecycle{};
  const bool lifecycle_borrowed = resident_generation_v2_internal::
      borrow_resident_generation_terminal_lifecycle_v2(
          owner->generation, sizeof(NeoResidentArchiveKnnTerminalV2),
          &lifecycle);
  if (!lifecycle_borrowed ||
      lifecycle.generation_owner_v2() != owner->generation ||
      lifecycle.admitted_run_stream_v2() !=
          owner->arena_access.admitted_run_stream ||
      lifecycle.completion_event_v2() == nullptr ||
      lifecycle.terminal_host_receipt_v2() == nullptr ||
      lifecycle.terminal_host_receipt_bytes_v2() !=
          sizeof(NeoResidentArchiveKnnTerminalV2) ||
      lifecycle.run_token_v2() != owner->binding.run_identity ||
      lifecycle.generation_index_v2() !=
          owner->retained_gene_view.expected_generation_index ||
      lifecycle.store_epoch_v2() !=
          owner->retained_gene_view.expected_store_epoch) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2);
  }
  owner->terminal_lifecycle = lifecycle;
  owner->terminal_host = static_cast<NeoResidentArchiveKnnTerminalV2*>(
      lifecycle.terminal_host_receipt_v2());
  if (owner->same_stream_enqueue_count >
          std::numeric_limits<std::uint64_t>::max() - 3ull ||
      lifecycle.same_stream_enqueue_count_v2() >
          std::numeric_limits<std::uint64_t>::max() - 3ull) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  const std::uint64_t global_final_enqueue_count =
      owner->same_stream_enqueue_count + 3ull;
  const std::uint64_t generation_final_enqueue_count =
      lifecycle.same_stream_enqueue_count_v2() + 3ull;
  const std::uint64_t receipt_identity =
      reinterpret_cast<std::uint64_t>(owner->terminal_host);
  seal_archive_terminal_v2<<<
      1, 1, 0, owner->arena_access.admitted_run_stream>>>(
      owner->control, owner->terminal_device, receipt_identity,
      owner->binding.run_identity, lifecycle.completion_event_identity_v2(),
      global_final_enqueue_count);
  std::int32_t status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  if (cudaMemcpyAsync(owner->terminal_host, owner->terminal_device,
                      sizeof(NeoResidentArchiveKnnTerminalV2),
                      cudaMemcpyDeviceToHost,
                      owner->arena_access.admitted_run_stream) != cudaSuccess) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2);
  }
  if (cudaEventRecord(lifecycle.completion_event_v2(),
                      owner->arena_access.admitted_run_stream) != cudaSuccess) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2);
  }
  if (!resident_generation_v2_internal::
           accept_resident_generation_terminal_enqueue_v2(
               &lifecycle, generation_final_enqueue_count)) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2);
  }
  owner->same_stream_enqueue_count = global_final_enqueue_count;

  std::memset(pending, 0, sizeof(*pending));
  pending->abi_version = NEO_RESIDENT_ARCHIVE_KNN_ABI_V2;
  pending->source_packed_commit_word = owner->initial_source_commit_word;
  pending->terminal_device_receipt_identity =
      reinterpret_cast<std::uint64_t>(owner->terminal_device);
  pending->run_identity = owner->binding.run_identity;
  pending->boxed_receipt_identity = reinterpret_cast<std::uint64_t>(pending);
  pending->staged_dependency_identity =
      reinterpret_cast<std::uint64_t>(&owner->retained_gene_view);
  pending->same_stream_enqueue_count = global_final_enqueue_count;
  pending->completion_event_identity = lifecycle.completion_event_identity_v2();
  pending->terminal_host_receipt_identity = receipt_identity;
  owner->pending_identity = pending;
  owner->phase = HostPhaseV2::TerminalPending;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t try_complete_resident_archive_terminal_v2(
    NeoResidentArchiveKnnOwnerV2* owner,
    const NeoResidentArchiveKnnPendingV2* pending,
    resident_generation_v1::NeoResidentGenerationReadyEventV1* committed_ready,
    NeoResidentArchiveKnnTerminalV2* terminal_copy) {
  if (owner == nullptr || pending == nullptr || committed_ready == nullptr ||
      terminal_copy == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->phase != HostPhaseV2::TerminalPending ||
      owner->pending_identity != pending || owner->terminal_host == nullptr ||
      !exact_pending_v2(*owner, *pending)) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  const cudaError_t query =
      cudaEventQuery(owner->terminal_lifecycle.completion_event_v2());
  ++owner->completion_event_query_count;
  if (query == cudaErrorNotReady) {
    return NEO_ARCHIVE_KNN_STATUS_NOT_READY_V2;
  }
  if (query != cudaSuccess) {
    owner->poisoned = true;
    owner->terminal_event_proven = false;
    return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  }
  owner->terminal_event_proven = true;
  *terminal_copy = *owner->terminal_host;
  terminal_copy->completion_event_query_count =
      owner->completion_event_query_count;

  const std::uint64_t packed_commit = terminal_copy->packed_commit_word;
  const bool bounded_commit =
      unpack_store_v2(packed_commit) <= 1u &&
      unpack_generation_v2(packed_commit) <= kGenerationMaskV2 &&
      unpack_archive_count_v2(packed_commit) <=
          owner->binding.archive_capacity &&
      unpack_epoch_v2(packed_commit) <= kEpochMaskV2;
  const std::uint64_t expected_digest = terminal_digest_v2(
      packed_commit, terminal_copy->collision_count,
      owner->binding.run_identity, terminal_copy->device_fault_word);
  const bool exact_common =
      terminal_copy->abi_version == NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 &&
      terminal_copy->receipt_identity ==
          pending->terminal_host_receipt_identity &&
      terminal_copy->run_identity == owner->binding.run_identity &&
      terminal_copy->compact_async_d2h_count == 1 &&
      terminal_copy->compact_async_d2h_bytes ==
          sizeof(NeoResidentArchiveKnnTerminalV2) &&
      terminal_copy->completion_stream_synchronize_count == 0 &&
      terminal_copy->same_stream_enqueue_count ==
          pending->same_stream_enqueue_count &&
      terminal_copy->completion_event_identity ==
          pending->completion_event_identity &&
      terminal_copy->validator_digest == expected_digest && bounded_commit;
  const bool exact_commit =
      terminal_copy->terminal_status ==
          NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2 &&
      terminal_copy->device_fault_word == 0 &&
      terminal_copy->validation_fault_word == 0;
  const bool exact_fault =
      terminal_copy->terminal_status == NEO_ARCHIVE_KNN_TERMINAL_FAULT_V2 &&
      (terminal_copy->device_fault_word != 0 ||
       terminal_copy->validation_fault_word != 0);
  owner->pending_identity = nullptr;
  owner->phase = HostPhaseV2::TerminalComplete;
  if (!exact_common || (!exact_commit && !exact_fault)) {
    owner->poisoned = true;
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }

  std::memset(committed_ready, 0, sizeof(*committed_ready));
  committed_ready->abi_version =
      resident_generation_v1::NEO_RESIDENT_GENERATION_ABI_V1;
  committed_ready->event_id = pending->completion_event_identity;
  committed_ready->generation_index = unpack_generation_v2(packed_commit);
  committed_ready->same_stream_enqueue_count =
      pending->same_stream_enqueue_count;
  if (exact_fault) {
    owner->poisoned = true;
    return NEO_ARCHIVE_KNN_STATUS_DEVICE_FAULT_V2;
  }
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2(
    void* session, NeoResidentArchiveKnnOwnerV2* owner) {
  if (session == nullptr || owner == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->phase != HostPhaseV2::TerminalComplete ||
      !owner->terminal_event_proven) {
    return NEO_ARCHIVE_KNN_STATUS_NOT_READY_V2;
  }
  if (session !=
      owner->terminal_lifecycle.population_lifetime_owner_v2()) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  owner->scoring = nullptr;
  owner->generation = nullptr;
  owner->arena_access = {};
  owner->terminal_host = nullptr;
  owner->terminal_device = nullptr;
  delete owner;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

}  // namespace neoethos::resident_archive_knn_v2
