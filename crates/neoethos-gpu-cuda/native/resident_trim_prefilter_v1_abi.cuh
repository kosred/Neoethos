#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace neoethos::resident_trim_prefilter_v1 {

constexpr std::uint32_t NEO_RESIDENT_TRIM_PREFILTER_ABI_V1 = 1U;

constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_OK_V1 = 0;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_INVALID_ARGUMENT_V1 = -1;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_ABI_MISMATCH_V1 = -2;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_IDENTITY_MISMATCH_V1 = -3;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_ARITHMETIC_OVERFLOW_V1 = -4;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_OUT_OF_MEMORY_V1 = -5;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_CUDA_ERROR_V1 = -6;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_CUB_ERROR_V1 = -7;
constexpr std::int32_t NEO_TRIM_PREFILTER_STATUS_STATE_ERROR_V1 = -8;

constexpr std::uint32_t NEO_TRIM_PREFILTER_FAULT_NONE_V1 = 0U;
constexpr std::uint32_t NEO_TRIM_PREFILTER_FAULT_INSUFFICIENT_LABELS_V1 = 1U;
constexpr std::uint32_t NEO_TRIM_PREFILTER_FAULT_NONFINITE_DECISION_V1 = 2U;
constexpr std::uint32_t NEO_TRIM_PREFILTER_FAULT_INVALID_VALIDITY_V1 = 3U;
constexpr std::uint32_t NEO_TRIM_PREFILTER_FAULT_CPCV_GEOMETRY_V1 = 4U;
constexpr std::uint32_t NEO_TRIM_PREFILTER_FAULT_SELECTED_MAP_OVERFLOW_V1 = 5U;

constexpr std::uint8_t NEO_COLUMN_CLASS_STATE_V1 = 1U << 0U;
constexpr std::uint8_t NEO_COLUMN_CLASS_TEMPLATE_V1 = 1U << 1U;

/// Crate-private one-shot import derived from the already-admitted resident
/// parent and its sealed schema-classification owner. Every pointer remains
/// borrowed from those lifetime owners.
struct NeoResidentTrimPrefilterImportV1 {
  std::uint32_t abi_version;
  std::uint32_t selected_cuda_ordinal;
  std::uint64_t parent_row_count;
  std::uint64_t parent_column_count;
  std::uint64_t packed_validity_bytes;
  std::uint64_t schema_metadata_bytes;
  std::uint64_t timeframe_group_count;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint64_t trim_prefilter_reserved_bytes;
  cudaStream_t admitted_run_stream;
  cudaEvent_t parent_ready_event;
  cudaEvent_t schema_ready_event;
  cudaEvent_t trim_prefilter_ready_event;
  void* parent_lifetime_owner;
  void* schema_lifetime_owner;
  const double* indicators_bar_major;
  const unsigned char* indicators_validity_u4;
  const double* close;
  const double* high;
  const double* low;
  const unsigned char* column_class_flags_device;
  const std::uint32_t* timeframe_group_ids_device;
  const unsigned char* template_force_keep_flags_device;
  std::uint8_t canonical_search_input_receipt_sha256[32];
  std::uint8_t canonical_content_merkle_sha256[32];
  std::uint8_t normalization_fit_sha256[32];
  std::uint8_t feature_plan_sha256[32];
  std::uint8_t source_provenance_sha256[32];
  std::uint8_t ordered_feature_schema_sha256[32];
  std::uint8_t column_classification_content_sha256[32];
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t cuda_math_flags_sha256[32];
};

struct NeoResidentTrimPrefilterPlanV1 {
  std::uint32_t abi_version;
  std::uint32_t atr_period;
  std::uint64_t parent_row_count;
  std::uint64_t parent_column_count;
  std::uint64_t global_row_cap;
  std::uint64_t timeframe_row_cap;
  std::uint64_t outer_split_at;
  std::uint64_t selection_row_start;
  std::uint64_t selection_row_end;
  std::uint64_t holdout_row_start;
  std::uint64_t holdout_row_end;
  std::uint64_t configured_top_k;
  std::uint64_t resolved_top_k;
  std::uint64_t minimum_per_timeframe;
  std::uint64_t max_hold_bars;
  std::uint64_t minimum_pairwise_samples;
  std::uint64_t minimum_decided_labels;
  std::uint64_t maximum_refit_folds;
  std::uint64_t cpcv_split_count;
  std::uint64_t cpcv_test_group_count;
  std::uint64_t cpcv_max_rows;
  double insample_fraction;
  double stop_atr_multiplier;
  double reward_risk_ratio;
  double round_trip_cost_price;
  double cpcv_embargo_fraction;
  double cpcv_purge_fraction;
  std::uint64_t charged_peak_device_bytes;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint8_t semantics_sha256[32];
  std::uint8_t state_family_semantics_sha256[32];
  std::uint8_t timeframe_group_semantics_sha256[32];
  std::uint8_t template_force_keep_semantics_sha256[32];
  std::uint8_t score_order_semantics_sha256[32];
  std::uint8_t plan_identity_sha256[32];
  std::uint8_t allocation_plan_sha256[32];
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
  std::uint8_t cuda_math_flags_sha256[32];
};

struct NeoResidentTrimPrefilterAllocationReceiptV1 {
  std::uint32_t abi_version;
  std::uint32_t allocation_count;
  std::uint64_t long_labels_bytes;
  std::uint64_t short_labels_bytes;
  std::uint64_t label_census_bytes;
  std::uint64_t fold_descriptor_bytes;
  std::uint64_t column_score_bytes;
  std::uint64_t column_instability_bytes;
  std::uint64_t column_rankability_bytes;
  std::uint64_t state_template_timeframe_metadata_bytes;
  std::uint64_t radix_key_ping_pong_bytes;
  std::uint64_t radix_index_ping_pong_bytes;
  std::uint64_t timeframe_group_counter_bytes;
  std::uint64_t selected_column_map_bytes;
  std::uint64_t selected_column_count_bytes;
  std::uint64_t cub_select_scratch_bytes;
  std::uint64_t cub_radix_sort_scratch_bytes;
  std::uint64_t device_seal_bytes;
  std::uint64_t retained_device_bytes;
  std::uint64_t peak_device_bytes;
  std::uint64_t same_context_free_bytes;
  std::uint64_t full_discovery_reserve_bytes;
  std::uint8_t allocation_plan_sha256[32];
};

struct NeoResidentTrimPrefilterFoldDescriptorV1 {
  std::uint64_t sampled_combination;
  std::uint64_t available_combinations;
  std::uint64_t capped_rows;
  std::uint64_t fit_tail_offset;
  std::uint64_t split_count;
  std::uint64_t test_group_count;
  std::uint64_t purge_rows;
  std::uint64_t embargo_rows;
  std::uint64_t prefix_exclusive_end;
  std::uint32_t use_cpcv;
  std::uint32_t valid;
};

struct NeoResidentTrimPrefilterLabelCensusV1 {
  unsigned long long label_up;
  unsigned long long label_down;
  unsigned long long label_vertical;
  unsigned long long label_ambiguous;
  unsigned long long label_short_win;
  unsigned long long label_short_loss;
  unsigned long long label_vertical_short;
  unsigned long long label_ambiguous_short;
  unsigned long long label_undefined;
  unsigned long long reserved[3];
};

/// Device-written authority. `selected_count` and the content digest never
/// cross to host at this stage.
struct NeoResidentTrimPrefilterDeviceSealV1 {
  std::uint32_t abi_version;
  std::uint32_t valid;
  std::uint32_t device_fault_word;
  std::uint32_t reserved;
  std::uint64_t selected_count;
  std::uint64_t selection_row_start;
  std::uint64_t selection_row_end;
  std::uint64_t holdout_row_start;
  std::uint64_t holdout_row_end;
  std::uint8_t selected_map_sha256[32];
};

struct NeoResidentTrimPrefilterReadyEventV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t intermediate_host_wait_count;
  std::uint64_t intermediate_readback_count;
  std::uint64_t host_to_device_transfer_count;
  std::uint64_t device_to_host_transfer_count;
  std::uint64_t explicit_synchronization_count;
};

/// Private same-run handoff. The next gpu-cuda stage consumes this descriptor;
/// Search sees only an owning opaque Rust type.
struct NeoResidentTrimPrefilterViewsV1 {
  std::uint32_t abi_version;
  std::uint32_t same_selected_column_map_for_holdout;
  const std::uint32_t* selected_compact_to_parent_columns_device;
  const std::uint64_t* selected_column_count_device;
  const NeoResidentTrimPrefilterDeviceSealV1* device_seal;
  cudaEvent_t trim_prefilter_ready_event;
  std::uint64_t parent_row_count;
  std::uint64_t parent_column_count;
  std::uint64_t selection_row_start;
  std::uint64_t selection_row_end;
  std::uint64_t holdout_row_start;
  std::uint64_t holdout_row_end;
  std::uint8_t plan_identity_sha256[32];
  std::uint8_t view_semantics_sha256[32];
  std::uint8_t canonical_content_merkle_sha256[32];
  std::uint8_t ordered_feature_schema_sha256[32];
  std::uint8_t cuda_device_identity_sha256[32];
  std::uint8_t primary_context_identity_sha256[32];
  std::uint8_t run_stream_identity_sha256[32];
  std::uint8_t cuda_build_manifest_sha256[32];
};

static_assert(sizeof(void*) == 8,
              "resident trim/prefilter V1 requires a 64-bit ABI");
static_assert(sizeof(NeoResidentTrimPrefilterImportV1) == 560,
              "resident trim/prefilter import ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterPlanV1) == 608,
              "resident trim/prefilter plan ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterAllocationReceiptV1) == 200,
              "resident trim/prefilter allocation receipt ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterFoldDescriptorV1) == 80,
              "fold descriptor ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterLabelCensusV1) == 96,
              "label census ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterDeviceSealV1) == 88,
              "device seal ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterReadyEventV1) == 56,
              "ready-event ABI changed");
static_assert(sizeof(NeoResidentTrimPrefilterViewsV1) == 344,
              "resident trim/prefilter views ABI changed");

struct NeoResidentTrimPrefilterRunV1;

extern "C" std::int32_t query_resident_trim_prefilter_scratch_v1(
    cudaStream_t admitted_run_stream, std::uint32_t selected_cuda_ordinal,
    std::uint64_t parent_column_count, std::uint32_t prefilter_active,
    std::uint64_t* cub_select_scratch_bytes,
    std::uint64_t* cub_radix_sort_scratch_bytes);

extern "C" std::int32_t query_resident_trim_prefilter_allocation_v1(
    const NeoResidentTrimPrefilterImportV1* import,
    const NeoResidentTrimPrefilterPlanV1* plan,
    NeoResidentTrimPrefilterAllocationReceiptV1* receipt);

extern "C" std::int32_t create_resident_trim_prefilter_run_v1(
    const NeoResidentTrimPrefilterImportV1* import,
    const NeoResidentTrimPrefilterPlanV1* plan,
    const NeoResidentTrimPrefilterAllocationReceiptV1* receipt,
    NeoResidentTrimPrefilterRunV1** run);

extern "C" std::int32_t enqueue_resident_trim_prefilter_stage_v1(
    NeoResidentTrimPrefilterRunV1* run, std::uint32_t stage);

extern "C" std::int32_t seal_resident_trim_prefilter_views_v1(
    NeoResidentTrimPrefilterRunV1* run,
    NeoResidentTrimPrefilterViewsV1* views,
    NeoResidentTrimPrefilterReadyEventV1* ready);

extern "C" std::int32_t enqueue_resident_trim_prefilter_release_v1(
    NeoResidentTrimPrefilterRunV1* run);

}  // namespace neoethos::resident_trim_prefilter_v1
