#include "../native/resident_archive_layout_v3.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

namespace archive = neoethos::resident_archive_knn_v2;
using Binding = archive::NeoResidentArchiveKnnBindV2;
using Region = archive::NeoResidentArchiveKnnArenaRegionV2;
using GenerationScalar = neoethos::resident_generation_v1::NeoResidentGenerationGeneScalarV1;
using ScoringScalar = neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyGeneScalarV1;
using Metric = neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyMetricRowV1;

constexpr std::array<Region Binding::*, 15> regions = {
    &Binding::fitness_scores, &Binding::decision_keys, &Binding::cub_scratch,
    &Binding::archive_gene_scalars, &Binding::archive_term_indices,
    &Binding::archive_term_weights, &Binding::archive_metric_rows,
    &Binding::archive_signatures, &Binding::archive_hashes,
    &Binding::current_population_signatures, &Binding::novelty_scores,
    &Binding::exact_top_k_keys, &Binding::admission_flags,
    &Binding::admission_offsets, &Binding::archive_control_and_seal};

// Place supplied literal sizes contiguously; this does not calculate geometry.
void position_regions(Binding& binding) {
  std::uint64_t offset = 0;
  for (auto member : regions) {
    auto& region = binding.*member;
    region.offset_bytes = offset;
    offset += region.size_bytes;
  }
  binding.total_device_bytes = offset;
}

Binding fixture(bool eight_archive_members = false) {
  Binding binding{};
  binding.population_count = 4;
  binding.archive_capacity = eight_archive_members ? 8 : 4;
  binding.signature_word_count = 4;
  binding.max_terms_per_gene = 12;  // Active K is not the padded archive stride.
  binding.novelty_neighbor_count = 7;
  // Independently calculated P=4, W=4, k=7, scratch=256 layouts.
  // A=4: H=8, (6A+H)*8=256. A=8: H=16, (6A+H)*8=512.
  const std::array<std::uint64_t, 15> sizes = eight_archive_members
      ? std::array<std::uint64_t, 15>{256, 256, 256, 1280, 2048, 2048,
                                    1792, 512, 512, 256, 256, 1024, 256, 256, 256}
      : std::array<std::uint64_t, 15>{256, 256, 256, 768, 1024, 1024,
                                    1024, 256, 256, 256, 256, 1024, 256, 256, 256};
  for (std::size_t index = 0; index < regions.size(); ++index)
    (binding.*regions[index]).size_bytes = sizes[index];
  position_regions(binding);
  return binding;
}

void expect_geometry(const Binding& binding, bool expected, const char* name) {
  const bool archive_result = archive::archive_layout_v3::validate_geometry_v3<
      sizeof(GenerationScalar), sizeof(Metric), 32>(binding);
  const bool scoring_result = archive::archive_layout_v3::validate_geometry_v3<
      sizeof(ScoringScalar), sizeof(Metric), 32>(binding);
  if (archive_result != expected || scoring_result != expected) {
    std::fprintf(stderr, "FAIL %s: archive=%d scorer=%d expected=%d\n",
                 name, archive_result, scoring_result, expected);
    std::exit(1);
  }
  std::printf("PASS %s\n", name);
}

int main() {
  auto binding = fixture();
  if (binding.total_device_bytes != 7424) return 2;
  expect_geometry(binding, true, "configured k7 two banks");
  binding.novelty_neighbor_count = 15;
  binding.exact_top_k_keys.size_bytes = 2048;
  position_regions(binding);
  if (binding.total_device_bytes != 8448) return 2;
  expect_geometry(binding, true, "legacy k15 correct banks");
  expect_geometry(fixture(true), true, "A8 indexed hash extent");

  // Reflow each mutation so rejection proves its extent, not an accidental gap.
  const std::array<std::pair<Region Binding::*, std::uint64_t>, 6> old_extents = {{
      {&Binding::archive_gene_scalars, 768},
      {&Binding::archive_term_indices, 1024},
      {&Binding::archive_term_weights, 1024},
      {&Binding::archive_metric_rows, 1024},
      {&Binding::archive_signatures, 256},
      {&Binding::archive_hashes, 256}}};
  const char* labels[] = {"reject one scalar bank", "reject one index bank",
                         "reject one weight bank", "reject one metric bank",
                         "reject one signature bank", "reject old hash extent"};
  for (std::size_t index = 0; index < old_extents.size(); ++index) {
    binding = fixture(true);
    (binding.*old_extents[index].first).size_bytes = old_extents[index].second;
    position_regions(binding);
    expect_geometry(binding, false, labels[index]);
  }
  binding = fixture(); binding.novelty_neighbor_count = 0;
  expect_geometry(binding, false, "reject zero neighbors");
  binding = fixture(); binding.novelty_neighbor_count = 15;
  expect_geometry(binding, false, "reject undersized configured k15");
  binding = fixture(); binding.population_count = 0;
  expect_geometry(binding, false, "reject empty population");
  binding = fixture(); binding.archive_capacity = 65536;
  expect_geometry(binding, false, "reject unrepresentable archive count");
  binding = fixture(); binding.signature_word_count = 3;
  expect_geometry(binding, false, "reject undersized signature stride");
  binding = fixture(); binding.max_terms_per_gene = 17;
  expect_geometry(binding, false, "reject excessive active K");
  binding = fixture(); binding.cub_scratch.size_bytes = 0; position_regions(binding);
  expect_geometry(binding, false, "reject empty scratch");
  binding = fixture(); binding.cub_scratch.size_bytes = 255; position_regions(binding);
  expect_geometry(binding, false, "reject unaligned scratch");
  binding = fixture(); binding.archive_hashes.offset_bytes -= 256;
  expect_geometry(binding, false, "reject overlapping region");
  binding = fixture(); binding.archive_hashes.offset_bytes += 256;
  expect_geometry(binding, false, "reject region gap");
  binding = fixture(); binding.total_device_bytes += 256;
  expect_geometry(binding, false, "reject trailing unaccounted bytes");
  binding = fixture();
  binding.population_count = archive::NEO_RESIDENT_ARCHIVE_KNN_MAX_POPULATION_COUNT_V2;
  binding.novelty_neighbor_count = std::numeric_limits<std::uint32_t>::max();
  expect_geometry(binding, false, "reject P times k overflow");
  binding = fixture(); binding.cub_scratch.size_bytes = ~std::uint64_t{255};
  expect_geometry(binding, false, "reject scratch end overflow");
  std::puts("22 host geometry checks passed; no device execution");
}
