#pragma once

#include "resident_archive_knn_v2_abi.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace neoethos::resident_archive_knn_v2::archive_layout_v3 {

constexpr std::uint64_t kAlignmentV3 = 256;

inline bool checked_add_v3(std::uint64_t left, std::uint64_t right,
                           std::uint64_t* result) {
  if (result == nullptr ||
      right > std::numeric_limits<std::uint64_t>::max() - left) return false;
  *result = left + right;
  return true;
}

inline bool checked_mul_v3(std::uint64_t left, std::uint64_t right,
                           std::uint64_t* result) {
  if (result == nullptr ||
      (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left))
    return false;
  *result = left * right;
  return true;
}

inline bool aligned_region_bytes_v3(std::uint64_t count,
                                    std::uint64_t elements,
                                    std::uint64_t element_bytes,
                                    std::uint64_t* result) {
  std::uint64_t raw = 0;
  if (count == 0 || elements == 0 || element_bytes == 0 ||
      !checked_mul_v3(count, elements, &raw) ||
      !checked_mul_v3(raw, element_bytes, &raw)) return false;
  const auto remainder = raw % kAlignmentV3;
  return checked_add_v3(raw, remainder == 0 ? 0 : kAlignmentV3 - remainder,
                        result);
}

inline bool validate_region_v3(const NeoResidentArchiveKnnArenaRegionV2& region,
                                std::uint64_t size, std::uint64_t* cursor) {
  return cursor != nullptr && region.offset_bytes == *cursor &&
         region.offset_bytes % kAlignmentV3 == 0 && region.size_bytes == size &&
         size != 0 && size % kAlignmentV3 == 0 &&
         checked_add_v3(region.offset_bytes, size, cursor);
}

// One geometry authority for scoring admission and the archive borrower.
// Identity/ABI validation remains with each owner; this helper grants neither.
// Sizes are tied to each caller's actual wire types, not caller-supplied bytes.
template <std::size_t GeneScalarBytes, std::size_t MetricRowBytes,
          std::size_t ExactNeighborKeyBytes>
inline bool validate_geometry_v3(const NeoResidentArchiveKnnBindV2& binding) {
  static_assert(GeneScalarBytes == 72 && MetricRowBytes == 104 &&
                ExactNeighborKeyBytes == 32, "archive wire geometry changed");
  if (binding.population_count == 0 ||
      binding.population_count > NEO_RESIDENT_ARCHIVE_KNN_MAX_POPULATION_COUNT_V2 ||
      binding.archive_capacity == 0 ||
      binding.archive_capacity > NEO_RESIDENT_ARCHIVE_KNN_MAX_CAPACITY_V2 ||
      binding.signature_word_count < NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2 ||
      binding.novelty_neighbor_count == 0 || binding.max_terms_per_gene == 0 ||
      binding.max_terms_per_gene > NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 ||
      binding.total_device_bytes == 0 ||
      binding.total_device_bytes > std::numeric_limits<std::size_t>::max())
    return false;

  std::uint64_t population_scalar = 0, archive_scalars = 0;
  std::uint64_t archive_indices = 0, archive_weights = 0, archive_metrics = 0;
  std::uint64_t archive_signatures = 0, archive_hashes = 0, index_words = 0;
  std::uint64_t population_signatures = 0, top_k = 0, flags = 0, offsets = 0;
  const auto hash_slots = archive_hash_table_capacity_v3(binding.archive_capacity);
  if (hash_slots == 0 ||
      !checked_mul_v3(binding.archive_capacity, 6, &index_words) ||
      !checked_add_v3(index_words, hash_slots, &index_words) ||
      !aligned_region_bytes_v3(binding.population_count, 1, sizeof(double),
                               &population_scalar) ||
      !aligned_region_bytes_v3(binding.archive_capacity, 2, GeneScalarBytes,
                               &archive_scalars) ||
      !aligned_region_bytes_v3(binding.archive_capacity,
                               2ull * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2,
                               sizeof(std::uint64_t), &archive_indices) ||
      !aligned_region_bytes_v3(binding.archive_capacity,
                               2ull * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2,
                               sizeof(double), &archive_weights) ||
      !aligned_region_bytes_v3(binding.archive_capacity, 2, MetricRowBytes,
                               &archive_metrics) ||
      !aligned_region_bytes_v3(binding.archive_capacity,
                               2ull * binding.signature_word_count,
                               sizeof(std::uint64_t), &archive_signatures) ||
      !aligned_region_bytes_v3(index_words, 1, sizeof(std::uint64_t),
                               &archive_hashes) ||
      !aligned_region_bytes_v3(binding.population_count, binding.signature_word_count,
                               sizeof(std::uint64_t), &population_signatures) ||
      !aligned_region_bytes_v3(binding.population_count, binding.novelty_neighbor_count,
                               ExactNeighborKeyBytes, &top_k) ||
      !aligned_region_bytes_v3(binding.population_count, 1, sizeof(std::uint32_t),
                               &flags) ||
      !aligned_region_bytes_v3(binding.population_count, 1, sizeof(std::uint64_t),
                               &offsets)) return false;

  std::uint64_t cursor = 0;
  return validate_region_v3(binding.fitness_scores, population_scalar, &cursor) &&
         validate_region_v3(binding.decision_keys, population_scalar, &cursor) &&
         validate_region_v3(binding.cub_scratch, binding.cub_scratch.size_bytes, &cursor) &&
         validate_region_v3(binding.archive_gene_scalars, archive_scalars, &cursor) &&
         validate_region_v3(binding.archive_term_indices, archive_indices, &cursor) &&
         validate_region_v3(binding.archive_term_weights, archive_weights, &cursor) &&
         validate_region_v3(binding.archive_metric_rows, archive_metrics, &cursor) &&
         validate_region_v3(binding.archive_signatures, archive_signatures, &cursor) &&
         validate_region_v3(binding.archive_hashes, archive_hashes, &cursor) &&
         validate_region_v3(binding.current_population_signatures, population_signatures, &cursor) &&
         validate_region_v3(binding.novelty_scores, population_scalar, &cursor) &&
         validate_region_v3(binding.exact_top_k_keys, top_k, &cursor) &&
         validate_region_v3(binding.admission_flags, flags, &cursor) &&
         validate_region_v3(binding.admission_offsets, offsets, &cursor) &&
         validate_region_v3(binding.archive_control_and_seal, 256, &cursor) &&
         cursor == binding.total_device_bytes;
}

}  // namespace neoethos::resident_archive_knn_v2::archive_layout_v3
