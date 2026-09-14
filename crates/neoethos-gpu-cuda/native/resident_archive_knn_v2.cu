#include "resident_archive_knn_v2_abi.cuh"
#include "resident_archive_layout_v3.hpp"
#include "resident_backend_identity_v3.cuh"
#if defined(__HIP_PLATFORM_AMD__)
#include "resident_search_hip_v1_abi.cuh"
#endif
#include "resident_generation_v2_internal.cuh"
#include "resident_scoring_novelty_v2_internal.cuh"

#include "resident_parallel_primitives_v1.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace backend_identity_v3 = ::neoethos::resident_backend_identity_v3;

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
bool nonzero_uuid_v2(const std::uint8_t uuid[16]) {
  std::uint8_t aggregate = 0;
  for (std::size_t index = 0; index < 16; ++index) {
    aggregate |= uuid[index];
  }
  return aggregate != 0;
}
bool validate_binding_layout_v2(const NeoResidentArchiveKnnBindV2& binding) {
  if (binding.abi_version != NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 ||
      !backend_identity_v3::archive_backend_valid(binding) || binding.reserved_extents != 0 ||
      binding.population_count == 0 ||
      binding.population_count >
          NEO_RESIDENT_ARCHIVE_KNN_MAX_POPULATION_COUNT_V2 ||
      binding.archive_capacity == 0 ||
      binding.archive_capacity > NEO_RESIDENT_ARCHIVE_KNN_MAX_CAPACITY_V2 ||
      binding.signature_word_count <
          NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 ||
      binding.novelty_neighbor_count == 0 ||
      binding.max_terms_per_gene == 0u ||
      binding.max_terms_per_gene > NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 ||
      !nonzero_uuid_v2(binding.device_uuid) ||
      backend_identity_v3::archive_owner_identity(binding) == 0 ||
      binding.search_stream_identity == 0 ||
      binding.active_pool_identity == 0 || backend_identity_v3::archive_build_identity(binding) == 0 ||
      binding.kernel_semantics_identity == 0 ||
      binding.binary64_math_identity == 0 || binding.plan_identity == 0 ||
      binding.run_identity == 0 ||
      binding.full_workspace_receipt_identity == 0 ||
      binding.post_trim_receipt_identity == 0) {
    return false;
  }

  return archive_layout_v3::validate_geometry_v3<
      sizeof(GeneScalarV2), sizeof(MetricRowV2), sizeof(ExactNeighborKeyV2)>(binding);
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
      seal->max_terms_per_gene == 0u ||
      seal->max_terms_per_gene > NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 ||
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
    std::uint32_t left_stride,
    const GeneScalarV2& right, const std::uint64_t* right_term_indices,
    const double* right_term_weights, std::uint64_t right_ordinal,
    std::uint32_t right_stride) {
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
  const std::uint64_t left_base = left_ordinal * left_stride;
  const std::uint64_t right_base = right_ordinal * right_stride;
  for (std::uint32_t term = 0;
       term < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++term) {
    const std::uint64_t left_index =
        term < left_stride ? left_term_indices[left_base + term] : 0ull;
    const std::uint64_t right_index =
        term < right_stride ? right_term_indices[right_base + term] : 0ull;
    const double left_weight =
        term < left_stride ? left_term_weights[left_base + term] : 0.0;
    const double right_weight =
        term < right_stride ? right_term_weights[right_base + term] : 0.0;
    if (left_index != right_index ||
        f64_bits(left_weight) != f64_bits(right_weight)) {
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
                                   std::uint32_t* selected_count, std::uint32_t capacity) {
  std::uint32_t count = *selected_count;
  if (count == capacity &&
      !neighbor_less_v2(candidate, selected[count - 1])) {
    return;
  }
  std::uint32_t position = count;
  if (position == capacity) {
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
    std::uint32_t* admission_flags, ArchiveControlV2* control,
    std::uint32_t signature_word_count, NeoResidentArchivePolicyV3 policy) {
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
      scalar.term_count > expected.max_terms_per_gene ||
      row.candidate_id != scalar.gene_identity ||
      row.scenario_id != expected_scenarios[candidate]) {
    latch_device_fault_v2(control, kGeneShapeFaultV2);
    admission_flags[candidate] = 0;
    return;
  }
  using resident_scoring_novelty_v2_internal::ResidentMetricStatusV2;
  const auto metric_status =
      resident_scoring_novelty_v2_internal::classify_resident_metrics_v2(row.values);
  if (metric_status == ResidentMetricStatusV2::Fault) {
    latch_device_fault_v2(control, kNonFiniteMetricFaultV2);
    admission_flags[candidate] = 0;
    return;
  }

  // This kernel also runs after CUB reused the first four population-sized
  // arrays. Clear the complete dynamic row, including its last partial word.
  std::uint64_t* signature = signatures + candidate * signature_word_count;
  for (std::uint32_t word = 0; word < signature_word_count; ++word) {
    signature[word] = 0ull;
  }
  const std::uint64_t base = candidate * expected.max_terms_per_gene;
  bool any_signature_bit = false;
  for (std::uint32_t term = 0; term < expected.max_terms_per_gene; ++term) {
    const std::uint64_t feature = genes.term_indices[base + term];
    const double weight = genes.term_weights[base + term];
    if (term < scalar.term_count) {
      if (feature >= expected.feature_count ||
          feature / 64ull >= signature_word_count ||
          !isfinite(weight)) {
        latch_device_fault_v2(control, kGeneShapeFaultV2);
        admission_flags[candidate] = 0;
        return;
      }
      signature[feature / 64ull] |= 1ull << (feature % 64ull);
      any_signature_bit = true;
    } else if (feature != 0 || f64_bits(weight) != 0) {
      latch_device_fault_v2(control, kGeneShapeFaultV2);
      admission_flags[candidate] = 0;
      return;
    }
  }
  if (!any_signature_bit) {
    latch_device_fault_v2(control, kSignatureFaultV2);
    admission_flags[candidate] = 0;
    return;
  }
  const bool mode_passed = policy.mode == 1u ||
      (policy.mode == 2u ? row.values[5] > policy.minimum_profit_factor :
       policy.mode == 3u ? row.values[1] > policy.minimum_sharpe :
                          row.values[kNetMetricSlotV2] > policy.minimum_net);
  admission_flags[candidate] = metric_status == ResidentMetricStatusV2::Finite &&
      row.values[kTradeCountMetricSlotV2] > 0.0 && mode_passed ? 1u : 0u;
}

__global__ void exact_archive_population_knn_v2(
    const GeneSealV2* seal, GeneViewV2 expected,
    const std::uint64_t* current_signatures,
    const GeneScalarV2* archive_scalars,
    const std::uint64_t* archive_signatures, ExactNeighborKeyV2* top_k,
    double* novelty_scores, const ScoringSealV2* scoring_seal,
    ArchiveControlV2* control, std::uint64_t archive_capacity,
    std::uint32_t signature_word_count, std::uint32_t neighbor_count,
    bool population_only) {
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

  // Each query owns its pre-admitted k-row segment. This supports the actual
  // configured k without fixed local-array truncation or a new allocation.
  ExactNeighborKeyV2* selected = top_k + query * neighbor_count;
  std::uint32_t selected_count = 0;
  const std::uint64_t* query_signature =
      current_signatures + query * signature_word_count;
  const std::uint64_t available =
      expected.logical_population_count - 1ull + (population_only ? 0ull : archive_count);
  if (available == 0) {
    if (population_only) {
      novelty_scores[query] = 0.0;
      for (std::uint32_t i = 0; i < neighbor_count; ++i) selected[i] = {};
      return;
    }
    latch_device_fault_v2(control, kNeighborBoundFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }

  const std::uint64_t neighbor_extent =
      expected.logical_population_count + (population_only ? 0ull : archive_count);
  for (std::uint64_t combined = 0; combined < neighbor_extent; ++combined) {
    const bool current = combined < expected.logical_population_count;
    const std::uint64_t ordinal =
        current ? combined : combined - expected.logical_population_count;
    if (current && ordinal == query) {
      continue;
    }
    const std::uint64_t* signature =
        current
            ? current_signatures + ordinal * signature_word_count
            : archive_signatures + ordinal * signature_word_count;
    std::uint64_t intersection = 0;
    std::uint64_t union_count = 0;
    for (std::uint32_t word = 0; word < signature_word_count; ++word) {
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
    // At most 16 active terms per gene still bounds a valid union to 32,
    // independently of vocabulary width; narrow only after that check.
    neighbor.numerator = static_cast<std::uint32_t>(union_count - intersection);
    neighbor.denominator = static_cast<std::uint32_t>(union_count);
    neighbor.gene_identity =
        current ? genes.scalars[ordinal].gene_identity
                : archive_scalars[ordinal].gene_identity;
    neighbor.source_kind = current ? kSourceCurrentV2 : kSourceArchiveV2;
    neighbor.source_ordinal = static_cast<std::uint32_t>(ordinal);
    insert_neighbor_v2(neighbor, selected, &selected_count, neighbor_count);
  }

  const std::uint32_t expected_count =
      static_cast<std::uint32_t>(
          available < neighbor_count
              ? available
              : neighbor_count);
  if (selected_count != expected_count || selected_count == 0) {
    latch_device_fault_v2(control, kNeighborBoundFaultV2);
    novelty_scores[query] = 0.0;
    return;
  }
  double sum = 0.0;
  for (std::uint32_t neighbor = 0;
       neighbor < neighbor_count; ++neighbor) {
    const std::uint64_t output =
        query * neighbor_count + neighbor;
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
    const ScoringSealV2* scoring_seal, ArchiveControlV2* control,
    double novelty_weight) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  control->ranked_ready = 0;
  control->staged_ready = 0;
  control->staged_count = 0;
  control->staged_collision_count = 0;
  if (!isfinite(novelty_weight) || novelty_weight < 0.0 || novelty_weight > 1.0) {
    latch_device_fault_v2(control, kNonFiniteMetricFaultV2);
    return;
  }
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
  bool any_finite_fitness = false;
  const double rejected_fitness =
      -__longlong_as_double(static_cast<long long>(0x7ff0000000000000ULL));
  for (std::uint64_t candidate = 0;
       candidate < expected.logical_population_count; ++candidate) {
    if ((!isfinite(fitness_scores[candidate]) &&
         fitness_scores[candidate] != rejected_fitness) ||
        !isfinite(novelty_scores[candidate]) || novelty_scores[candidate] < 0.0) {
      latch_device_fault_v2(control, kNonFiniteMetricFaultV2);
      return;
    }
    // The sealed scoring producer distinguishes economic -infinity from a
    // device/math fault. CPU novelty uses all genes, but fitness normalization
    // uses only finite scores; archive metric eligibility is separate.
    if (isfinite(fitness_scores[candidate])) {
      any_finite_fitness = true;
      minimum_fitness = fitness_scores[candidate] < minimum_fitness
                            ? fitness_scores[candidate]
                            : minimum_fitness;
      maximum_fitness = fitness_scores[candidate] > maximum_fitness
                            ? fitness_scores[candidate]
                            : maximum_fitness;
    }
    maximum_novelty = novelty_scores[candidate] > maximum_novelty
                          ? novelty_scores[candidate]
                          : maximum_novelty;
  }

  // An all-rejected population still has a deterministic rank. Never evaluate
  // an unused subtraction of the uninitialized finite extrema in that case.
  double fitness_range = any_finite_fitness
                             ? __dsub_rn(maximum_fitness, minimum_fitness)
                             : 1.0e-9;
  fitness_range = fitness_range < 1.0e-9 ? 1.0e-9 : fitness_range;
  const double novelty_range =
      maximum_novelty < 1.0e-9 ? 1.0e-9 : maximum_novelty;
  const double fitness_weight = __dsub_rn(1.0, novelty_weight);
  for (std::uint64_t candidate = 0;
       candidate < expected.logical_population_count; ++candidate) {
    ordinal_keys[candidate] = candidate;
    ordinal_values[candidate] = candidate;
    if (fitness_scores[candidate] == rejected_fitness) {
      // Zero remains invalid; one sorts below every finite blended key.
      decision_keys[candidate] = 1ull;
      continue;
    }
    // With novelty disabled, CPU selection uses the raw score. Normalizing it
    // would preserve order but change softmax temperature and improvement epsilon.
    if (novelty_weight == 0.0 || expected.logical_population_count <= 1) {
      decision_keys[candidate] = ordered_finite_f64_key_v2(fitness_scores[candidate]);
      continue;
    }
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
    std::uint64_t archive_capacity, std::uint32_t signature_word_count) {
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
              expected.max_terms_per_gene,
              archive_scalars[archived], archive_term_indices,
              archive_term_weights, archived,
              NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2)) {
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
    archive_hashes[2 * archive_capacity + destination] = destination;
#pragma unroll
    for (std::uint32_t term = 0;
         term < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++term) {
      archive_term_indices[
          destination * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + term] =
          term < expected.max_terms_per_gene
              ? current.term_indices[candidate * expected.max_terms_per_gene + term]
              : 0ull;
      archive_term_weights[
          destination * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + term] =
          term < expected.max_terms_per_gene
              ? current.term_weights[candidate * expected.max_terms_per_gene + term]
              : 0.0;
    }
    for (std::uint32_t word = 0; word < signature_word_count; ++word) {
      archive_signatures[destination * signature_word_count + word] =
          current_signatures[candidate * signature_word_count + word];
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

// Adaptive retention uses two admitted banks. The current packed store bit
// selects committed observations; no staged replacement can corrupt that bank.
__global__ void copy_committed_archive_bank_v3(
    GeneScalarV2* scalars, std::uint64_t* indices, double* weights,
    MetricRowV2* metrics, std::uint64_t* signatures, std::uint64_t* hashes,
    ArchiveControlV2* control, std::uint64_t capacity, std::uint32_t words) {
  const auto item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto commit = atomic_read_commit_v2(&control->packed_commit_word);
  const auto count = unpack_archive_count_v2(commit);
  if (control->device_fault_word || item >= count) return;
  if (count > capacity || !control->ranked_ready || commit != control->ranked_source_commit_word) {
    latch_device_fault_v2(control, kPublicationFaultV2); return;
  }
  const auto from = unpack_store_v2(commit) * capacity + item;
  const auto to = (1u - unpack_store_v2(commit)) * capacity + item;
  scalars[to] = scalars[from]; metrics[to] = metrics[from]; hashes[to] = hashes[from];
  hashes[2 * capacity + to] = hashes[2 * capacity + from];
  for (std::uint32_t t = 0; t < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++t) {
    indices[to * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + t] = indices[from * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + t];
    weights[to * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + t] = weights[from * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + t];
  }
  for (std::uint32_t w = 0; w < words; ++w) signatures[to * words + w] = signatures[from * words + w];
}

// Rebuilt from the staged bank each generation. This scratch never participates
// in publication: a failed offer leaves the committed bank and its order intact.
struct AdaptiveArchiveIndexV3 {
  std::uint64_t* table;
  std::uint64_t* heap;
  std::uint64_t* positions;
  const GeneScalarV2* scalars;
  const MetricRowV2* metrics;
  const std::uint64_t* sequence;
  ArchiveControlV2* control;
  std::uint64_t base;
  std::uint64_t capacity;
  std::uint64_t table_capacity;
};

__device__ std::uint64_t adaptive_archive_hash_home_v3(
    std::uint64_t hash, std::uint64_t table_capacity) {
  hash ^= hash >> 33; hash *= 0xff51afd7ed558ccdull;
  hash ^= hash >> 33; hash *= 0xc4ceb9fe1a85ec53ull;
  hash ^= hash >> 33;
  return hash & (table_capacity - 1);
}

__device__ bool adaptive_archive_index_fault_v3(const AdaptiveArchiveIndexV3& index) {
  latch_device_fault_v2(index.control, kArchiveBoundFaultV2);
  return false;
}

__device__ bool adaptive_archive_hash_insert_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t slot, std::uint64_t count) {
  if (slot >= count || count > index.capacity) return adaptive_archive_index_fault_v3(index);
  auto at = adaptive_archive_hash_home_v3(index.scalars[index.base + slot].content_hash,
                                         index.table_capacity);
  for (std::uint64_t probe = 0; probe < index.table_capacity; ++probe) {
    const auto entry = index.table[at];
    if (entry == 0) { index.table[at] = slot + 1; return true; }
    if (entry > count || entry == slot + 1) return adaptive_archive_index_fault_v3(index);
    at = (at + 1) & (index.table_capacity - 1);
  }
  return adaptive_archive_index_fault_v3(index);
}

__device__ bool adaptive_archive_hash_erase_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t slot, std::uint64_t count) {
  if (slot >= count || count > index.capacity) return adaptive_archive_index_fault_v3(index);
  const auto mask = index.table_capacity - 1;
  auto hole = adaptive_archive_hash_home_v3(index.scalars[index.base + slot].content_hash,
                                           index.table_capacity);
  bool found = false;
  for (std::uint64_t probe = 0; probe < index.table_capacity; ++probe) {
    const auto entry = index.table[hole];
    if (entry == 0 || entry > count) return adaptive_archive_index_fault_v3(index);
    if (entry == slot + 1) { found = true; break; }
    hole = (hole + 1) & mask;
  }
  if (!found) return adaptive_archive_index_fault_v3(index);
  // Backward shifting preserves the search chain without accumulating tombstones.
  // The evicted scalar must still contain its OLD key throughout this operation.
  auto scan = (hole + 1) & mask;
  for (std::uint64_t probe = 0; probe < index.table_capacity; ++probe) {
    const auto entry = index.table[scan];
    if (entry == 0) { index.table[hole] = 0; return true; }
    if (entry > count) return adaptive_archive_index_fault_v3(index);
    const auto home = adaptive_archive_hash_home_v3(
        index.scalars[index.base + entry - 1].content_hash, index.table_capacity);
    if (((scan - home) & mask) >= ((scan - hole) & mask)) {
      index.table[hole] = entry;
      hole = scan;
    }
    scan = (scan + 1) & mask;
  }
  return adaptive_archive_index_fault_v3(index);
}

__device__ bool adaptive_archive_hash_lookup_v3(
    const AdaptiveArchiveIndexV3& index, const DeviceGeneSourcesV2& current,
    const GeneScalarV2& scalar, std::uint64_t candidate, std::uint32_t active_terms,
    const std::uint64_t* archive_indices, const double* archive_weights,
    std::uint64_t count, std::uint64_t* result, std::uint64_t* collisions) {
  if (count > index.capacity) return adaptive_archive_index_fault_v3(index);
  const auto home = adaptive_archive_hash_home_v3(scalar.content_hash, index.table_capacity);
  auto at = home;
  *result = count;
  *collisions = 0;
  bool complete = false;
  for (std::uint64_t probe = 0; probe < index.table_capacity; ++probe) {
    const auto entry = index.table[at];
    if (entry == 0) { complete = true; break; }
    if (entry > count) return adaptive_archive_index_fault_v3(index);
    const auto slot = entry - 1;
    const auto& archived = index.scalars[index.base + slot];
    if (archived.content_hash == scalar.content_hash) {
      if (full_fixed_stride_gene_equal_v2(scalar, current.term_indices, current.term_weights,
          candidate, active_terms, archived, archive_indices, archive_weights,
          index.base + slot, NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2)) {
        if (slot < *result) *result = slot;
      } else ++*collisions;
    }
    at = (at + 1) & (index.table_capacity - 1);
  }
  if (!complete) return adaptive_archive_index_fault_v3(index);
  if (*result < count && *collisions != 0) {
    // Preserve the old ordinal-scan collision diagnostic as well as membership:
    // only unequal same-hash slots preceding the first duplicate were visited.
    *collisions = 0;
    at = home;
    for (std::uint64_t probe = 0; probe < index.table_capacity; ++probe) {
      const auto entry = index.table[at];
      if (entry == 0) return true;
      if (entry > count) return adaptive_archive_index_fault_v3(index);
      if (entry - 1 < *result &&
          index.scalars[index.base + entry - 1].content_hash == scalar.content_hash) ++*collisions;
      at = (at + 1) & (index.table_capacity - 1);
    }
    return adaptive_archive_index_fault_v3(index);
  }
  return true;
}

__device__ bool adaptive_archive_heap_worse_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t left, std::uint64_t right) {
  const auto left_net = index.metrics[index.base + left].values[kNetMetricSlotV2];
  const auto right_net = index.metrics[index.base + right].values[kNetMetricSlotV2];
  const auto left_sequence = index.sequence[index.base + left];
  const auto right_sequence = index.sequence[index.base + right];
  return left_net < right_net || (left_net == right_net &&
      (left_sequence > right_sequence || (left_sequence == right_sequence && left < right)));
}

__device__ bool adaptive_archive_heap_node_valid_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t node, std::uint64_t count) {
  return node < count && index.heap[node] < count &&
         index.positions[index.heap[node]] == node;
}

__device__ void adaptive_archive_heap_swap_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t left, std::uint64_t right) {
  const auto slot = index.heap[left];
  index.heap[left] = index.heap[right]; index.heap[right] = slot;
  index.positions[index.heap[left]] = left; index.positions[index.heap[right]] = right;
}

__device__ bool adaptive_archive_heap_down_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t node, std::uint64_t count) {
  if (count > index.capacity || !adaptive_archive_heap_node_valid_v3(index, node, count))
    return adaptive_archive_index_fault_v3(index);
  while (node < count / 2) {
    auto worst = node * 2 + 1;
    if (!adaptive_archive_heap_node_valid_v3(index, worst, count))
      return adaptive_archive_index_fault_v3(index);
    if (worst + 1 < count) {
      if (!adaptive_archive_heap_node_valid_v3(index, worst + 1, count))
        return adaptive_archive_index_fault_v3(index);
      if (adaptive_archive_heap_worse_v3(index, index.heap[worst + 1], index.heap[worst])) ++worst;
    }
    if (!adaptive_archive_heap_worse_v3(index, index.heap[worst], index.heap[node])) break;
    adaptive_archive_heap_swap_v3(index, node, worst);
    node = worst;
  }
  return true;
}

__device__ bool adaptive_archive_heap_up_v3(
    const AdaptiveArchiveIndexV3& index, std::uint64_t node, std::uint64_t count) {
  if (count > index.capacity || !adaptive_archive_heap_node_valid_v3(index, node, count))
    return adaptive_archive_index_fault_v3(index);
  while (node != 0) {
    const auto parent = (node - 1) / 2;
    if (!adaptive_archive_heap_node_valid_v3(index, parent, count))
      return adaptive_archive_index_fault_v3(index);
    if (!adaptive_archive_heap_worse_v3(index, index.heap[node], index.heap[parent])) break;
    adaptive_archive_heap_swap_v3(index, node, parent);
    node = parent;
  }
  return true;
}

__global__ void stage_adaptive_archive_v3(
    const GeneSealV2* seal, GeneViewV2 expected, const MetricRowV2* current_metrics,
    const std::uint64_t* current_signatures, const std::uint64_t* ranked_ordinals,
    const std::uint32_t* admission_flags, GeneScalarV2* archive_scalars,
    std::uint64_t* archive_indices, double* archive_weights, MetricRowV2* archive_metrics,
    std::uint64_t* archive_signatures, std::uint64_t* archive_hashes,
    ArchiveControlV2* control, std::uint64_t capacity, std::uint32_t words,
    std::uint64_t hash_capacity) {
  if (blockIdx.x || threadIdx.x) return;
  DeviceGeneSourcesV2 current{};
  if (!load_current_gene_sources_v2(seal, expected, control, &current)) return;
  const auto commit = atomic_read_commit_v2(&control->packed_commit_word);
  const auto committed = unpack_archive_count_v2(commit);
  if (capacity == 0 || hash_capacity < 2 * capacity ||
      (hash_capacity & (hash_capacity - 1)) != 0 ||
      committed > capacity || !control->ranked_ready || commit != control->ranked_source_commit_word) {
    latch_device_fault_v2(control, kPublicationFaultV2); return;
  }
  const auto base = (1u - unpack_store_v2(commit)) * capacity;
  auto* sequence = archive_hashes + 2 * capacity;
  auto* table = archive_hashes + 4 * capacity;
  auto* heap = table + hash_capacity;
  auto* positions = heap + capacity;
  const AdaptiveArchiveIndexV3 index{table, heap, positions, archive_scalars,
      archive_metrics, sequence, control, base, capacity, hash_capacity};
  std::uint64_t next_sequence = 0, count = committed, collisions = 0;
  for (std::uint64_t i = 0; i < count; ++i) {
    if (sequence[base + i] == ~std::uint64_t{0}) { latch_device_fault_v2(control, kArchiveBoundFaultV2); return; }
    if (next_sequence <= sequence[base + i]) next_sequence = sequence[base + i] + 1;
    heap[i] = i; positions[i] = i;
    if (!adaptive_archive_hash_insert_v3(index, i, count)) return;
  }
  // Floyd construction is linear; thereafter the root is the exact worst entry.
  for (std::uint64_t node = count / 2; node != 0; --node)
    if (!adaptive_archive_heap_down_v3(index, node - 1, count)) return;
  // CPU offers the descending selection rank in order. Exact behavior equality
  // excludes display identity/ancestry; hashes accelerate, never replace, it.
  // Expected hash lookup plus indexed heap repair removes the two archive-wide
  // scans per offer without replacing causal eviction/reintroduction by top-A.
  for (std::uint64_t rank = 0; rank < expected.logical_population_count; ++rank) {
    const auto candidate = ranked_ordinals[rank];
    if (candidate >= expected.logical_population_count) { latch_device_fault_v2(control, kGeneShapeFaultV2); return; }
    if (!admission_flags[candidate]) continue;
    const auto scalar = current.scalars[candidate];
    const auto net = current_metrics[candidate].values[kNetMetricSlotV2];
    std::uint64_t slot = count;
    std::uint64_t candidate_collisions = 0;
    if (!adaptive_archive_hash_lookup_v3(index, current, scalar, candidate,
        expected.max_terms_per_gene, archive_indices, archive_weights,
        count, &slot, &candidate_collisions)) return;
    if (candidate_collisions > ~std::uint64_t{0} - collisions) {
      latch_device_fault_v2(control, kArchiveBoundFaultV2); return;
    }
    collisions += candidate_collisions;
    const bool duplicate = slot < count;
    bool appended = false;
    if (duplicate) {
      if (net <= archive_metrics[base + slot].values[kNetMetricSlotV2]) continue;
      // Improving an existing behavior retains its original admission priority.
    } else {
      if (count == capacity) {
        if (!adaptive_archive_heap_node_valid_v3(index, 0, count)) {
          adaptive_archive_index_fault_v3(index); return;
        }
        slot = heap[0];
        if (net <= archive_metrics[base + slot].values[kNetMetricSlotV2]) continue;
        if (!adaptive_archive_hash_erase_v3(index, slot, count)) return;
      } else { slot = count++; appended = true; }
      if (next_sequence == ~std::uint64_t{0}) { latch_device_fault_v2(control, kArchiveBoundFaultV2); return; }
      sequence[base + slot] = next_sequence++;
    }
    const auto to = base + slot;
    archive_scalars[to] = scalar; archive_metrics[to] = current_metrics[candidate];
    archive_hashes[to] = scalar.content_hash;
    for (std::uint32_t t = 0; t < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2; ++t) {
      archive_indices[to * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + t] = t < expected.max_terms_per_gene
          ? current.term_indices[candidate * expected.max_terms_per_gene + t] : 0;
      archive_weights[to * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 + t] = t < expected.max_terms_per_gene
          ? current.term_weights[candidate * expected.max_terms_per_gene + t] : 0.0;
    }
    for (std::uint32_t w = 0; w < words; ++w)
      archive_signatures[to * words + w] = current_signatures[candidate * words + w];
    if (!duplicate && !adaptive_archive_hash_insert_v3(index, slot, count)) return;
    if (appended) {
      heap[count - 1] = slot; positions[slot] = count - 1;
      if (!adaptive_archive_heap_up_v3(index, count - 1, count)) return;
    } else {
      // Both duplicate improvement and replacement strictly increase net; no
      // upward repair is needed and duplicate admission sequence stays intact.
      if (!adaptive_archive_heap_down_v3(index, positions[slot], count)) return;
    }
  }
  control->staged_count = count - committed;
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
  bool candidates_exported;
  double novelty_weight;
  bool novelty_configured;
  NeoResidentArchivePolicyV3 policy;
  bool adaptive_policy_configured;
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
      !backend_identity_v3::archive_backend_valid(*binding) || binding->reserved_extents != 0) {
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
      signature_word_count_v2(genes->feature_count) !=
          binding->signature_word_count) {
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

#if defined(__HIP_PLATFORM_AMD__)
  if (!resident_search_hip_v1::validate_population_owner_v1(
          lifecycle.population_lifetime_owner_v2(), *binding,
          access.admitted_run_stream)) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
#endif
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
  const std::uint64_t legacy_novelty_bits = NEO_RESIDENT_ARCHIVE_KNN_NOVELTY_WEIGHT_BITS_V2;
  std::memcpy(&created->novelty_weight, &legacy_novelty_bits, sizeof(double));
  created->policy.abi_version = 3u;
  created->policy.novelty_weight = created->novelty_weight;
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

extern "C" std::int32_t configure_resident_archive_novelty_v3(
    NeoResidentArchiveKnnOwnerV2* owner, std::uint64_t expected_run_identity,
    double novelty_weight) {
  if (owner == nullptr || !std::isfinite(novelty_weight) ||
      novelty_weight < 0.0 || novelty_weight > 1.0) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->poisoned || owner->phase != HostPhaseV2::Bound || owner->novelty_configured) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }
  if (expected_run_identity == 0 || expected_run_identity != owner->binding.run_identity ||
      expected_run_identity != owner->retained_gene_view.expected_run_token) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  owner->novelty_weight = novelty_weight;
  owner->novelty_configured = true;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}

extern "C" std::int32_t configure_resident_archive_policy_v3(
    NeoResidentArchiveKnnOwnerV2* owner, std::uint64_t expected_run_identity,
    const NeoResidentArchivePolicyV3* policy) {
  if (policy == nullptr || policy->abi_version != 3u || policy->mode > 3u ||
      !std::isfinite(policy->minimum_net) || !std::isfinite(policy->minimum_profit_factor) ||
      !std::isfinite(policy->minimum_sharpe)) return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  const auto status = configure_resident_archive_novelty_v3(owner, expected_run_identity, policy->novelty_weight);
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) return status;
  owner->policy = *policy;
  owner->adaptive_policy_configured = true;
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
      owner->admission_flags, owner->control,
      owner->binding.signature_word_count, owner->policy);
  if (!advance_global_enqueue_count_v2(owner, 1)) {
    return poison_owner_v2(owner,
                           NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
  }
  status = launch_status_v2();
  if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) {
    return poison_owner_v2(owner, status);
  }
  if (owner->novelty_weight == 0.0) {
    const auto cleared = cudaMemsetAsync(owner->novelty_scores, 0,
        owner->binding.population_count * sizeof(double), stream);
    if (cleared != cudaSuccess) return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2);
  } else exact_archive_population_knn_v2<<<
      grid_for_v2(owner->binding.population_count), kThreadsV2, 0, stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      owner->current_population_signatures, owner->archive_gene_scalars,
      owner->archive_signatures, owner->exact_top_k_keys,
      owner->novelty_scores, finite_rows.device_seal, owner->control,
      owner->binding.archive_capacity, owner->binding.signature_word_count,
      owner->binding.novelty_neighbor_count, owner->adaptive_policy_configured);
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
      rank_keys_a, rank_values_a, finite_rows.device_seal, owner->control,
      owner->novelty_weight);
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
  cudaError_t cub_status = neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(
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
  cub_status = neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(
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
  cub_status = neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairsDescending(
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
      owner->admission_flags, owner->control,
      owner->binding.signature_word_count, owner->policy);
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
  if (owner->adaptive_policy_configured) {
    const auto stream = owner->arena_access.admitted_run_stream;
    const auto hash_capacity = archive_hash_table_capacity_v3(owner->binding.archive_capacity);
    // Only transient index slots are cleared. Both payload/sequence banks stay
    // intact until the existing checked staged-bank copy/publication protocol.
    if (cudaMemsetAsync(owner->archive_hashes + 4 * owner->binding.archive_capacity,
                        0, hash_capacity * sizeof(std::uint64_t), stream) != cudaSuccess)
      return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2);
    if (!advance_global_enqueue_count_v2(owner, 1))
      return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
    copy_committed_archive_bank_v3<<<grid_for_v2(owner->binding.archive_capacity), kThreadsV2, 0, stream>>>(
        owner->archive_gene_scalars, owner->archive_term_indices, owner->archive_term_weights,
        owner->archive_metric_rows, owner->archive_signatures, owner->archive_hashes,
        owner->control, owner->binding.archive_capacity, owner->binding.signature_word_count);
    if (!advance_global_enqueue_count_v2(owner, 1))
      return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2);
    const auto status = launch_status_v2();
    if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) return poison_owner_v2(owner, status);
    stage_adaptive_archive_v3<<<1, 1, 0, stream>>>(
        owner->retained_gene_view.seal_device, owner->retained_gene_view,
        owner->finite_rows.metric_rows_device, owner->current_population_signatures,
        owner->admission_offsets, owner->admission_flags, owner->archive_gene_scalars,
        owner->archive_term_indices, owner->archive_term_weights, owner->archive_metric_rows,
        owner->archive_signatures, owner->archive_hashes, owner->control,
        owner->binding.archive_capacity, owner->binding.signature_word_count, hash_capacity);
  } else stage_ranked_archive_tail_v2<<<
      1, 1, 0, owner->arena_access.admitted_run_stream>>>(
      owner->retained_gene_view.seal_device, owner->retained_gene_view,
      owner->finite_rows.metric_rows_device,
      owner->current_population_signatures, owner->admission_offsets,
      owner->admission_flags, owner->admission_offsets,
      owner->fitness_scores,
      owner->archive_gene_scalars, owner->archive_term_indices,
      owner->archive_term_weights, owner->archive_metric_rows,
      owner->archive_signatures, owner->archive_hashes, owner->control,
      owner->binding.archive_capacity, owner->binding.signature_word_count);
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
               &generation_before)) {
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
#if defined(__HIP_PLATFORM_AMD__)
  if (!resident_search_hip_v1::validate_population_owner_v1(
          owner->terminal_lifecycle.population_lifetime_owner_v2(),
          owner->binding, owner->arena_access.admitted_run_stream)) {
    owner->poisoned = true;
    owner->terminal_event_proven = false;
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
#endif
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

const TerminalLifecycleV2* borrow_completed_archive_terminal_lifecycle_v3(
    const NeoResidentArchiveKnnOwnerV2* owner,
    const resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const NeoResidentArchiveKnnTerminalV2* expected_terminal) {
  if (owner == nullptr || generation == nullptr || expected_terminal == nullptr ||
      owner->generation != generation || owner->poisoned ||
      owner->phase != HostPhaseV2::TerminalComplete ||
      !owner->terminal_event_proven || owner->terminal_host == nullptr) {
    return nullptr;
  }
  auto normalized = *expected_terminal;
  if (normalized.completion_event_query_count != owner->completion_event_query_count ||
      normalized.terminal_status != NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2 ||
      normalized.device_fault_word != 0 || normalized.validation_fault_word != 0) {
    return nullptr;
  }
  normalized.completion_event_query_count = 0;
  const auto& lifecycle = owner->terminal_lifecycle;
  if (std::memcmp(&normalized, owner->terminal_host, sizeof(normalized)) != 0 ||
      lifecycle.generation_owner_v2() != generation ||
      lifecycle.terminal_host_receipt_v2() != owner->terminal_host ||
      lifecycle.run_token_v2() != normalized.run_identity ||
      normalized.run_identity != owner->binding.run_identity ||
      lifecycle.generation_index_v2() != unpack_generation_v2(normalized.packed_commit_word) ||
      lifecycle.store_epoch_v2() != unpack_epoch_v2(normalized.packed_commit_word) ||
      lifecycle.current_store_index_v2() != unpack_store_v2(normalized.packed_commit_word) ||
      lifecycle.completion_event_identity_v2() != normalized.completion_event_identity ||
      lifecycle.admitted_run_stream_v2() != owner->arena_access.admitted_run_stream ||
      normalized.same_stream_enqueue_count != owner->same_stream_enqueue_count) {
    return nullptr;
  }
#if defined(__HIP_PLATFORM_AMD__)
  if (!resident_search_hip_v1::validate_population_owner_v1(
          lifecycle.population_lifetime_owner_v2(), owner->binding,
          lifecycle.admitted_run_stream_v2())) return nullptr;
#else
  CUcontext context = nullptr;
  unsigned long long context_id = 0;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr ||
      cuCtxGetId(context, &context_id) != CUDA_SUCCESS ||
      context_id != backend_identity_v3::archive_owner_identity(owner->binding)) {
    return nullptr;
  }
#endif
  return &lifecycle;
}

extern "C" std::int32_t copy_resident_archive_terminal_candidates_v4(
    NeoResidentArchiveKnnOwnerV2* owner,
    const NeoResidentArchiveKnnTerminalV2* expected_terminal,
    GeneScalarV2* scalars, std::uint64_t* term_indices, double* term_weights,
    MetricRowV2* metrics, std::uint64_t* admission_sequences, std::uint64_t candidate_capacity,
    std::uint64_t term_capacity, NeoResidentArchiveExportReceiptV3* receipt) {
  if (receipt == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  *receipt = {};
  if (owner == nullptr || expected_terminal == nullptr) {
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  }
  if (owner->poisoned || owner->phase != HostPhaseV2::TerminalComplete ||
      !owner->terminal_event_proven || owner->terminal_host == nullptr ||
      owner->candidates_exported) {
    return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;
  }
  // try_complete adds its host-side polling count to the returned copy, while
  // the device-produced control receipt retains zero in that field.
  auto normalized = *expected_terminal;
  if (normalized.completion_event_query_count !=
          owner->completion_event_query_count ||
      normalized.terminal_status != NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2 ||
      normalized.device_fault_word != 0 ||
      normalized.validation_fault_word != 0) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
  normalized.completion_event_query_count = 0;
  if (std::memcmp(&normalized, owner->terminal_host, sizeof(normalized)) != 0) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
#if defined(__HIP_PLATFORM_AMD__)
  if (!resident_search_hip_v1::validate_population_owner_v1(
          owner->terminal_lifecycle.population_lifetime_owner_v2(),
          owner->binding, owner->arena_access.admitted_run_stream)) {
    owner->poisoned = true;
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
#else
  CUcontext current_context = nullptr;
  unsigned long long current_context_id = 0;
  if (cuCtxGetCurrent(&current_context) != CUDA_SUCCESS ||
      current_context == nullptr ||
      cuCtxGetId(current_context, &current_context_id) != CUDA_SUCCESS ||
      current_context_id != backend_identity_v3::archive_owner_identity(owner->binding)) {
    return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;
  }
#endif
  const std::uint64_t count =
      unpack_archive_count_v2(normalized.packed_commit_word);
  std::uint64_t terms = 0;
  std::uint64_t scalar_bytes = 0;
  std::uint64_t index_bytes = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t metric_bytes = 0;
  std::uint64_t sequence_bytes = 0;
  std::uint64_t total_bytes = 0;
  if (count > owner->binding.archive_capacity ||
      !checked_mul_v2(count, NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2, &terms) ||
      !checked_mul_v2(count, sizeof(GeneScalarV2), &scalar_bytes) ||
      !checked_mul_v2(terms, sizeof(std::uint64_t), &index_bytes) ||
      !checked_mul_v2(terms, sizeof(double), &weight_bytes) ||
      !checked_mul_v2(count, sizeof(MetricRowV2), &metric_bytes) ||
      !checked_mul_v2(count, sizeof(std::uint64_t), &sequence_bytes) ||
      !checked_add_v2(scalar_bytes, index_bytes, &total_bytes) ||
      !checked_add_v2(total_bytes, weight_bytes, &total_bytes) ||
      !checked_add_v2(total_bytes, metric_bytes, &total_bytes) ||
      !checked_add_v2(total_bytes, sequence_bytes, &total_bytes)) {
    return NEO_ARCHIVE_KNN_STATUS_ARITHMETIC_OVERFLOW_V2;
  }
  if (candidate_capacity != count || term_capacity != terms ||
      (count != 0 && (scalars == nullptr || term_indices == nullptr ||
                      term_weights == nullptr || metrics == nullptr || admission_sequences == nullptr))) {
    return NEO_ARCHIVE_KNN_STATUS_RANGE_ERROR_V2;
  }
  // These copies occur only after every Search kernel has completed. CUDA's
  // synchronous D2H API returns after the host buffer is populated, including
  // pageable memory. Do not substitute Async here without retaining output
  // ownership through a separate completion event on every error path.
  const auto bank_offset = owner->adaptive_policy_configured
      ? unpack_store_v2(normalized.packed_commit_word) * owner->binding.archive_capacity : 0ull;
  const auto bank_terms = bank_offset * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2;
  if (count != 0 &&
      (cudaMemcpy(scalars, owner->archive_gene_scalars + bank_offset, scalar_bytes,
                  cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(term_indices, owner->archive_term_indices + bank_terms, index_bytes,
                  cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(term_weights, owner->archive_term_weights + bank_terms, weight_bytes,
                  cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(metrics, owner->archive_metric_rows + bank_offset, metric_bytes,
                  cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(admission_sequences, owner->archive_hashes + 2 * owner->binding.archive_capacity + bank_offset, sequence_bytes,
                  cudaMemcpyDeviceToHost) != cudaSuccess)) {
    return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2);
  }
  receipt->abi_version = 4;
  receipt->run_identity = normalized.run_identity;
  receipt->packed_commit_word = normalized.packed_commit_word;
  receipt->candidate_count = count;
  receipt->term_count = terms;
  receipt->feature_count = owner->retained_gene_view.feature_count;
  receipt->host_copy_count = count == 0 ? 0 : 5;
  receipt->host_copy_bytes = total_bytes;
  owner->candidates_exported = true;
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

#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
struct AdaptiveArchiveIndexFixtureV3 {
  ArchiveControlV2 control;
  std::uint64_t passed_checks;
};

// Primitive regression only, not a fabricated Search run/receipt. The exact
// production helpers execute on device with deliberately colliding test keys.
__global__ void fixture_check_adaptive_archive_index_kernel_v3(
    AdaptiveArchiveIndexFixtureV3* output) {
  if (blockIdx.x || threadIdx.x) return;
  *output = {};
  constexpr std::uint64_t capacity = 4, hash_capacity = 8;
  std::uint64_t table[hash_capacity]{}, heap[capacity]{}, positions[capacity]{};
  std::uint64_t sequence[capacity]{}, indices[capacity * 16]{};
  double weights[capacity * 16]{};
  GeneScalarV2 scalars[capacity]{};
  MetricRowV2 metrics[capacity]{};
  const AdaptiveArchiveIndexV3 index{table, heap, positions, scalars, metrics,
      sequence, &output->control, 0, capacity, hash_capacity};
  const DeviceGeneSourcesV2 current{scalars, indices, weights};
  std::uint64_t colliding_hash = 0;
  for (; colliding_hash < 1024; ++colliding_hash)
    if (adaptive_archive_hash_home_v3(colliding_hash, hash_capacity) == hash_capacity - 1) break;
  if (colliding_hash == 1024) return;
  for (std::uint64_t i = 0; i < 3; ++i) {
    scalars[i].content_hash = colliding_hash; scalars[i].term_count = 1;
    scalars[i].gene_identity = i; indices[i * 16] = i; weights[i * 16] = 1.0;
    heap[i] = i; positions[i] = i; sequence[i] = i + 1;
    if (!adaptive_archive_hash_insert_v3(index, i, 3)) return;
  }
  std::uint64_t found = 0, collisions = 0;
  if (!adaptive_archive_hash_lookup_v3(index, current, scalars[2], 2, 16,
      indices, weights, 3, &found, &collisions) || found != 2 || collisions != 2 ||
      table[7] != 1 || table[0] != 2 || table[1] != 3) return;
  output->passed_checks |= 1ull;
  if (!adaptive_archive_hash_erase_v3(index, 1, 3) ||
      !adaptive_archive_hash_lookup_v3(index, current, scalars[2], 2, 16,
          indices, weights, 3, &found, &collisions) || found != 2 || collisions != 1 ||
      !adaptive_archive_hash_lookup_v3(index, current, scalars[1], 1, 16,
          indices, weights, 3, &found, &collisions) || found != 3 || collisions != 2 ||
      table[7] != 1 || table[0] != 3 || table[1] != 0) return;
  output->passed_checks |= 2ull;
  indices[16] = 9;
  if (!adaptive_archive_hash_insert_v3(index, 1, 3) ||
      !adaptive_archive_hash_lookup_v3(index, current, scalars[1], 1, 16,
          indices, weights, 3, &found, &collisions) || found != 1 || collisions != 1) return;
  output->passed_checks |= 4ull;

  metrics[0].values[kNetMetricSlotV2] = 10;
  metrics[1].values[kNetMetricSlotV2] = 10;
  metrics[2].values[kNetMetricSlotV2] = 20;
  if (!adaptive_archive_heap_down_v3(index, 0, 3) || heap[0] != 1) return;
  output->passed_checks |= 8ull;
  metrics[1].values[kNetMetricSlotV2] = 30;
  if (!adaptive_archive_heap_down_v3(index, positions[1], 3) || heap[0] != 0 ||
      sequence[1] != 2) return;
  output->passed_checks |= 16ull;

  // Remove the old root from its wraparound chain BEFORE changing its hash.
  if (!adaptive_archive_hash_erase_v3(index, 0, 3)) return;
  scalars[0].content_hash = colliding_hash + 1; indices[0] = 42;
  metrics[0].values[kNetMetricSlotV2] = 25; sequence[0] = 4;
  if (!adaptive_archive_hash_insert_v3(index, 0, 3) ||
      !adaptive_archive_heap_down_v3(index, positions[0], 3) || heap[0] != 2) return;
  for (std::uint64_t slot = 0; slot < 3; ++slot) {
    if (!adaptive_archive_hash_lookup_v3(index, current, scalars[slot], slot, 16,
        indices, weights, 3, &found, &collisions) || found != slot ||
        !adaptive_archive_heap_node_valid_v3(index, positions[slot], 3)) return;
  }
  output->passed_checks |= 32ull;

  scalars[3].content_hash = colliding_hash + 2; scalars[3].term_count = 1;
  indices[48] = 43; weights[48] = 1.0; sequence[3] = 5;
  metrics[3].values[kNetMetricSlotV2] = -0.0;
  if (!adaptive_archive_hash_insert_v3(index, 3, 4)) return;
  heap[3] = 3; positions[3] = 3;
  if (!adaptive_archive_heap_up_v3(index, 3, 4) || heap[0] != 3) return;
  metrics[2].values[kNetMetricSlotV2] = 0.0;
  if (!adaptive_archive_heap_up_v3(index, positions[2], 4) || heap[0] != 3 ||
      !adaptive_archive_heap_worse_v3(index, 3, 2) ||
      adaptive_archive_heap_worse_v3(index, 2, 3) || output->control.device_fault_word) return;
  output->passed_checks |= 64ull;

  table[adaptive_archive_hash_home_v3(scalars[0].content_hash, hash_capacity)] = 5;
  if (adaptive_archive_hash_lookup_v3(index, current, scalars[0], 0, 16,
      indices, weights, 4, &found, &collisions) || !output->control.device_fault_word) return;
  output->passed_checks |= 128ull;
}

extern "C" std::int32_t fixture_check_adaptive_archive_index_v3(
    std::uint32_t device, std::uint64_t* passed_checks) {
  if (passed_checks == nullptr || device > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;
  *passed_checks = 0;
  auto* copied_checks = new (std::nothrow) std::uint64_t{};
  if (copied_checks == nullptr) return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  if (cudaSetDevice(static_cast<int>(device)) != cudaSuccess) {
    delete copied_checks;
    return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    delete copied_checks;
    return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  }
  AdaptiveArchiveIndexFixtureV3* state = nullptr;
  if (cudaMalloc(&state, sizeof(*state)) != cudaSuccess) {
    const auto cleanup = cudaStreamDestroy(stream);
    if (cleanup != cudaSuccess) {
      // No fixture work was submitted, but destruction is not acknowledged.
      // Report it without retrying the handle or replacing the first failure.
      std::fprintf(stderr, "archive fixture stream cleanup failed (%d); handle retained\n",
                   static_cast<int>(cleanup));
    }
    delete copied_checks;
    return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  }
  fixture_check_adaptive_archive_index_kernel_v3<<<1, 1, 0, stream>>>(state);
  // Ambiguous execution/copy errors retain this fixture's tiny native graph and
  // heap readback destination. No caller-owned pointer is an async destination.
  if (cudaPeekAtLastError() != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess ||
      cudaMemcpy(copied_checks,
          reinterpret_cast<const std::uint8_t*>(state) + offsetof(AdaptiveArchiveIndexFixtureV3, passed_checks),
          sizeof(*copied_checks), cudaMemcpyDeviceToHost) != cudaSuccess)
    return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  const auto checks = *copied_checks;
  delete copied_checks;
  if (cudaFree(state) != cudaSuccess || cudaStreamDestroy(stream) != cudaSuccess)
    return NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2;
  *passed_checks = checks;
  return NEO_ARCHIVE_KNN_STATUS_OK_V2;
}
#endif

}  // namespace neoethos::resident_archive_knn_v2
