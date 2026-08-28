#include "resident_trim_prefilter_v1_abi.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>

namespace neoethos::resident_trim_prefilter_v1 {
namespace {

static_assert(sizeof(double) == 8, "resident prefilter requires IEEE f64 storage");

constexpr std::uint64_t MINIMUM_PAIRWISE_SAMPLES_V1 = 30U;
constexpr std::uint64_t MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1 = 100U;
constexpr std::uint64_t MAXIMUM_REFIT_FOLDS_V1 = 8U;
constexpr std::uint64_t MAX_EXACT_F64_INTEGER_V1 = 9007199254740992U;
constexpr std::uint64_t U64_MAX_V1 = ~std::uint64_t{0};
constexpr std::uint32_t U32_MAX_V1 = ~std::uint32_t{0};
constexpr double F64_INFINITY_V1 = std::numeric_limits<double>::infinity();
constexpr std::uint64_t MAX_GRID_X_V1 = 0x7fffffffU;
constexpr std::uint64_t LAUNCH_THREADS_V1 = 256U;

constexpr std::uint32_t STAGE_LABELS_V1 = 1U;
constexpr std::uint32_t STAGE_LABEL_GUARD_V1 = 2U;
constexpr std::uint32_t STAGE_FOLDS_V1 = 3U;
constexpr std::uint32_t STAGE_CORRELATIONS_V1 = 4U;
constexpr std::uint32_t STAGE_RANK_V1 = 5U;
constexpr std::uint32_t STAGE_QUOTAS_V1 = 6U;
constexpr std::uint32_t STAGE_ASCENDING_MAP_V1 = 7U;
constexpr std::uint32_t STAGE_DEVICE_SEAL_V1 = 8U;

__host__ __device__ constexpr std::uint64_t max_u64_v1(
    std::uint64_t left, std::uint64_t right) {
  return left > right ? left : right;
}

__host__ __device__ constexpr std::uint64_t gcd_u64_v1(
    std::uint64_t left, std::uint64_t right) {
  while (right != 0U) {
    const std::uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

bool checked_add_v1(std::uint64_t left, std::uint64_t right,
                    std::uint64_t* output) {
  if (output == nullptr || left > U64_MAX_V1 - right) {
    return false;
  }
  *output = left + right;
  return true;
}

bool checked_mul_v1(std::uint64_t left, std::uint64_t right,
                    std::uint64_t* output) {
  if (output == nullptr || (right != 0U && left > U64_MAX_V1 / right)) {
    return false;
  }
  *output = left * right;
  return true;
}

bool checked_accumulate_v1(std::uint64_t value, std::uint64_t* total) {
  std::uint64_t next = 0U;
  if (total == nullptr || !checked_add_v1(*total, value, &next)) {
    return false;
  }
  *total = next;
  return true;
}

std::uint64_t min_nonzero_v1(std::uint64_t left, std::uint64_t right) {
  if (left == 0U) {
    return right;
  }
  if (right == 0U) {
    return left;
  }
  return left < right ? left : right;
}

bool hash_is_nonzero_v1(const std::uint8_t hash[32]) {
  if (hash == nullptr) {
    return false;
  }
  std::uint8_t folded = 0U;
  for (std::size_t i = 0U; i < 32U; ++i) {
    folded = static_cast<std::uint8_t>(folded | hash[i]);
  }
  return folded != 0U;
}

bool hashes_equal_v1(const std::uint8_t left[32],
                     const std::uint8_t right[32]) {
  return left != nullptr && right != nullptr && std::memcmp(left, right, 32U) == 0;
}

bool finite_positive_v1(double value) { return std::isfinite(value) && value > 0.0; }

bool finite_fraction_v1(double value) {
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

struct AbsoluteViewRangesV1 {
  std::uint64_t outer_split_at;
  std::uint64_t selection_row_start;
  std::uint64_t selection_row_end;
  std::uint64_t holdout_row_start;
  std::uint64_t holdout_row_end;
};

bool resolve_absolute_view_ranges_v1(std::uint64_t parent_row_count,
                                     std::uint64_t global_row_cap,
                                     std::uint64_t timeframe_row_cap,
                                     AbsoluteViewRangesV1* output) {
  if (output == nullptr || parent_row_count == 0U ||
      parent_row_count > MAX_EXACT_F64_INTEGER_V1) {
    return false;
  }
  const std::uint64_t outer_split_at = static_cast<std::uint64_t>(
      floor(0.8 * static_cast<double>(parent_row_count)));
  if (outer_split_at < 64U || outer_split_at >= parent_row_count) {
    return false;
  }
  const std::uint64_t row_cap =
      min_nonzero_v1(global_row_cap, timeframe_row_cap);
  const std::uint64_t retained_selection_rows =
      row_cap > 0U && row_cap < outer_split_at ? row_cap : outer_split_at;
  const std::uint64_t selection_row_start =
      outer_split_at - retained_selection_rows;
  const std::uint64_t selection_row_end = outer_split_at;
  const std::uint64_t holdout_row_start = outer_split_at;
  const std::uint64_t holdout_row_end = parent_row_count;
  output->outer_split_at = outer_split_at;
  output->selection_row_start = selection_row_start;
  output->selection_row_end = selection_row_end;
  output->holdout_row_start = holdout_row_start;
  output->holdout_row_end = holdout_row_end;
  return true;
}

bool validate_import_and_plan_v1(const NeoResidentTrimPrefilterImportV1* import,
                                 const NeoResidentTrimPrefilterPlanV1* plan) {
  if (import == nullptr || plan == nullptr ||
      import->abi_version != NEO_RESIDENT_TRIM_PREFILTER_ABI_V1 ||
      plan->abi_version != NEO_RESIDENT_TRIM_PREFILTER_ABI_V1 ||
      import->admitted_run_stream == nullptr ||
      import->parent_ready_event == nullptr ||
      import->schema_ready_event == nullptr ||
      import->trim_prefilter_ready_event == nullptr ||
      import->parent_lifetime_owner == nullptr ||
      import->schema_lifetime_owner == nullptr ||
      import->indicators_bar_major == nullptr ||
      import->indicators_validity_u4 == nullptr || import->close == nullptr ||
      import->high == nullptr || import->low == nullptr ||
      import->column_class_flags_device == nullptr ||
      import->timeframe_group_ids_device == nullptr ||
      import->template_force_keep_flags_device == nullptr ||
      import->parent_row_count == 0U || import->parent_column_count == 0U ||
      import->parent_column_count > MAX_GRID_X_V1 ||
      import->parent_row_count != plan->parent_row_count ||
      import->parent_column_count != plan->parent_column_count ||
      import->selected_cuda_ordinal > static_cast<std::uint32_t>(0x7fffffff) ||
      import->full_discovery_reserve_bytes == 0U ||
      import->trim_prefilter_reserved_bytes == 0U ||
      import->timeframe_group_count == 0U || plan->atr_period != 14U ||
      plan->max_hold_bars == 0U ||
      plan->minimum_pairwise_samples != MINIMUM_PAIRWISE_SAMPLES_V1 ||
      plan->minimum_decided_labels !=
          MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1 ||
      plan->maximum_refit_folds != MAXIMUM_REFIT_FOLDS_V1 ||
      !finite_positive_v1(plan->insample_fraction) ||
      plan->insample_fraction >= 1.0 ||
      !finite_positive_v1(plan->stop_atr_multiplier) ||
      !finite_positive_v1(plan->reward_risk_ratio) ||
      !std::isfinite(plan->round_trip_cost_price) ||
      plan->round_trip_cost_price < 0.0 ||
      !finite_fraction_v1(plan->cpcv_embargo_fraction) ||
      !finite_fraction_v1(plan->cpcv_purge_fraction)) {
    return false;
  }
  const std::uint64_t cells =
      import->parent_row_count * import->parent_column_count;
  if (import->parent_column_count != 0U &&
      cells / import->parent_column_count != import->parent_row_count) {
    return false;
  }
  const std::uint64_t packed = cells / 2U + cells % 2U;
  if (import->packed_validity_bytes < packed ||
      import->schema_metadata_bytes == 0U ||
      import->trim_prefilter_reserved_bytes < plan->charged_peak_device_bytes ||
      import->full_discovery_reserve_bytes !=
          plan->full_discovery_reserve_bytes ||
      plan->charged_peak_device_bytes > plan->full_discovery_reserve_bytes) {
    return false;
  }
  AbsoluteViewRangesV1 ranges{};
  if (!resolve_absolute_view_ranges_v1(
          import->parent_row_count, plan->global_row_cap,
          plan->timeframe_row_cap, &ranges) ||
      ranges.outer_split_at != plan->outer_split_at ||
      ranges.selection_row_start != plan->selection_row_start ||
      ranges.selection_row_end != plan->selection_row_end ||
      ranges.holdout_row_start != plan->holdout_row_start ||
      ranges.holdout_row_end != plan->holdout_row_end) {
    return false;
  }
  const std::uint64_t selection_rows =
      ranges.selection_row_end - ranges.selection_row_start;
  if (selection_rows > MAX_GRID_X_V1 * LAUNCH_THREADS_V1) {
    return false;
  }
  for (const std::uint8_t* hash : {
           import->canonical_search_input_receipt_sha256,
           import->canonical_content_merkle_sha256,
           import->normalization_fit_sha256,
           import->feature_plan_sha256,
           import->source_provenance_sha256,
           import->ordered_feature_schema_sha256,
           import->column_classification_content_sha256,
           import->cuda_device_identity_sha256,
           import->primary_context_identity_sha256,
           import->run_stream_identity_sha256,
           import->cuda_build_manifest_sha256,
           import->cuda_math_flags_sha256,
           plan->semantics_sha256,
           plan->state_family_semantics_sha256,
           plan->timeframe_group_semantics_sha256,
           plan->template_force_keep_semantics_sha256,
           plan->score_order_semantics_sha256,
           plan->plan_identity_sha256,
           plan->allocation_plan_sha256,
       }) {
    if (!hash_is_nonzero_v1(hash)) {
      return false;
    }
  }
  return hashes_equal_v1(import->cuda_device_identity_sha256,
                         plan->cuda_device_identity_sha256) &&
         hashes_equal_v1(import->primary_context_identity_sha256,
                         plan->primary_context_identity_sha256) &&
         hashes_equal_v1(import->run_stream_identity_sha256,
                         plan->run_stream_identity_sha256) &&
         hashes_equal_v1(import->cuda_build_manifest_sha256,
                         plan->cuda_build_manifest_sha256) &&
         hashes_equal_v1(import->cuda_math_flags_sha256,
                         plan->cuda_math_flags_sha256);
}

__device__ unsigned char validity_code_v1(const unsigned char* packed,
                                          std::uint64_t cell) {
  const unsigned char value = packed[cell / 2U];
  return (cell & 1U) == 0U ? static_cast<unsigned char>(value & 0x0fU)
                           : static_cast<unsigned char>(value >> 4U);
}

__device__ double rolling_atr_simple_finite_mean_v1(
    const double* close, const double* high, const double* low,
    std::uint64_t selection_row_start, std::uint64_t local_row,
    std::uint32_t period) {
  const std::uint64_t local_start =
      local_row + 1U > period ? local_row + 1U - period : 0U;
  double sum = 0.0;
  std::uint64_t count = 0U;
  for (std::uint64_t local = local_start; local <= local_row; ++local) {
    const std::uint64_t absolute = selection_row_start + local;
    const double hi = high[absolute];
    const double lo = low[absolute];
    const double previous_close =
        local > 0U ? close[absolute - 1U] : close[absolute];
    if (!isfinite(hi) || !isfinite(lo) || !isfinite(previous_close)) {
      continue;
    }
    const double true_range =
        fmax(hi - lo, fmax(fabs(hi - previous_close),
                           fabs(lo - previous_close)));
    sum += true_range;
    ++count;
  }
  return count == 0U ? nan("") : sum / static_cast<double>(count);
}

__global__ void first_passage_labels_kernel_v1(
    const double* close, const double* high, const double* low,
    std::uint64_t selection_row_start, std::uint64_t selection_rows,
    std::uint32_t atr_period, std::uint64_t max_hold_bars,
    double stop_atr_multiplier, double reward_risk_ratio,
    double round_trip_cost_price, double* long_labels,
    double* short_labels, NeoResidentTrimPrefilterLabelCensusV1* census) {
  const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                            static_cast<std::uint64_t>(threadIdx.x);
  if (row >= selection_rows) {
    return;
  }
  const std::uint64_t absolute_row = selection_row_start + row;
  const double entry = close[absolute_row];
  const double atr = rolling_atr_simple_finite_mean_v1(
      close, high, low, selection_row_start, row, atr_period);
  if (!isfinite(entry) || !isfinite(atr) || atr <= 0.0 ||
      row + 1U >= selection_rows) {
    long_labels[row] = nan("");
    short_labels[row] = nan("");
    atomicAdd(&census->label_undefined, 1ULL);
    return;
  }
  const double stop_distance = stop_atr_multiplier * atr;
  const double take_distance = reward_risk_ratio * stop_distance;
  const double long_take = entry + take_distance + round_trip_cost_price;
  const double long_stop = entry - stop_distance + round_trip_cost_price;
  const double short_take = entry - take_distance - round_trip_cost_price;
  const double short_stop = entry + stop_distance - round_trip_cost_price;
  const std::uint64_t remaining_horizon = selection_rows - 1U - row;
  const std::uint64_t horizon_step =
      max_hold_bars < remaining_horizon ? max_hold_bars : remaining_horizon;
  const std::uint64_t horizon_end = row + horizon_step;
  double long_label = 0.0;
  double short_label = 0.0;
  bool long_decided = false;
  bool short_decided = false;
  for (std::uint64_t future = row + 1U; future <= horizon_end; ++future) {
    const std::uint64_t absolute_future = selection_row_start + future;
    const double hi = high[absolute_future];
    const double lo = low[absolute_future];
    const bool high_ok = isfinite(hi);
    const bool low_ok = isfinite(lo);
    if (!long_decided) {
      const bool long_take_hit = high_ok && hi >= long_take;
      const bool long_stop_hit = low_ok && lo <= long_stop;
      if (long_take_hit && long_stop_hit) {
        atomicAdd(&census->label_ambiguous, 1ULL);
        long_decided = true;
      } else if (long_take_hit) {
        long_label = 1.0;
        atomicAdd(&census->label_up, 1ULL);
        long_decided = true;
      } else if (long_stop_hit) {
        long_label = -1.0;
        atomicAdd(&census->label_down, 1ULL);
        long_decided = true;
      }
    }
    if (!short_decided) {
      const bool short_take_hit = low_ok && lo <= short_take;
      const bool short_stop_hit = high_ok && hi >= short_stop;
      if (short_take_hit && short_stop_hit) {
        atomicAdd(&census->label_ambiguous_short, 1ULL);
        short_decided = true;
      } else if (short_take_hit) {
        short_label = 1.0;
        atomicAdd(&census->label_short_win, 1ULL);
        short_decided = true;
      } else if (short_stop_hit) {
        short_label = -1.0;
        atomicAdd(&census->label_short_loss, 1ULL);
        short_decided = true;
      }
    }
    if (long_decided && short_decided) {
      break;
    }
  }
  if (!long_decided) {
    atomicAdd(&census->label_vertical, 1ULL);
  }
  if (!short_decided) {
    atomicAdd(&census->label_vertical_short, 1ULL);
  }
  long_labels[row] = long_label;
  short_labels[row] = short_label;
}

__global__ void invalidate_insufficient_labels_kernel_v1(
    const NeoResidentTrimPrefilterLabelCensusV1* census,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  const std::uint64_t decided_long = census->label_up + census->label_down;
  const std::uint64_t decided_short =
      census->label_short_win + census->label_short_loss;
  if (max_u64_v1(decided_long, decided_short) <
      MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1) {
    device_seal->valid = 0U;
    device_seal->device_fault_word =
        NEO_TRIM_PREFILTER_FAULT_INSUFFICIENT_LABELS_V1;
  }
}

__host__ __device__ bool combination_count_checked_v1(
    std::uint64_t n, std::uint64_t k, std::uint64_t* output) {
  if (output == nullptr || k > n) {
    return false;
  }
  if (k > n - k) {
    k = n - k;
  }
  std::uint64_t value = 1U;
  for (std::uint64_t i = 1U; i <= k; ++i) {
    std::uint64_t factor = n - k + i;
    std::uint64_t divisor = i;
    const std::uint64_t value_divisor = gcd_u64_v1(value, divisor);
    value /= value_divisor;
    divisor /= value_divisor;
    const std::uint64_t factor_divisor = gcd_u64_v1(factor, divisor);
    factor /= factor_divisor;
    const std::uint64_t remaining_divisor = divisor / factor_divisor;
    if ((remaining_divisor > 1U && factor % remaining_divisor != 0U) ||
        remaining_divisor == 0U) {
      return false;
    }
    const std::uint64_t reduced_factor = factor / remaining_divisor;
    if (reduced_factor != 0U && value > U64_MAX_V1 / reduced_factor) {
      return false;
    }
    value *= reduced_factor;
  }
  *output = value;
  return true;
}

__device__ bool lexicographic_test_group_combination_v1(
    std::uint64_t split_count, std::uint64_t test_group_count,
    std::uint64_t rank, std::uint64_t query_group) {
  std::uint64_t next = 0U;
  for (std::uint64_t position = 0U; position < test_group_count; ++position) {
    const std::uint64_t remaining = test_group_count - position - 1U;
    const std::uint64_t last_candidate = split_count - (remaining + 1U);
    bool chosen = false;
    for (std::uint64_t candidate = next; candidate <= last_candidate; ++candidate) {
      std::uint64_t block = 0U;
      if (!combination_count_checked_v1(split_count - candidate - 1U,
                                        remaining, &block)) {
        return false;
      }
      if (rank < block) {
        if (candidate == query_group) {
          return true;
        }
        next = candidate + 1U;
        chosen = true;
        break;
      }
      rank -= block;
    }
    if (!chosen) {
      return false;
    }
  }
  return false;
}

__device__ bool cpcv_training_group_range_v1(
    std::uint64_t capped_rows, std::uint64_t split_count,
    std::uint64_t test_group_count, std::uint64_t sampled_combination,
    std::uint64_t query_group, std::uint64_t purge_rows,
    std::uint64_t embargo_rows, std::uint64_t* valid_start_output,
    std::uint64_t* valid_end_output) {
  if (valid_start_output == nullptr || valid_end_output == nullptr ||
      split_count < 2U || query_group >= split_count) {
    return false;
  }
  const std::uint64_t group_size = capped_rows / split_count;
  if (group_size == 0U || lexicographic_test_group_combination_v1(
                              split_count, test_group_count,
                              sampled_combination, query_group)) {
    return false;
  }
  const std::uint64_t group_start = query_group * group_size;
  const std::uint64_t group_end =
      query_group + 1U == split_count ? capped_rows
                                      : (query_group + 1U) * group_size;
  std::uint64_t valid_start = group_start;
  std::uint64_t valid_end = group_end;
  for (std::uint64_t test_group = 0U; test_group < split_count;
       ++test_group) {
    if (!lexicographic_test_group_combination_v1(
            split_count, test_group_count, sampled_combination, test_group)) {
      continue;
    }
    const std::uint64_t test_start = test_group * group_size;
    const std::uint64_t test_end =
        test_group + 1U == split_count ? capped_rows
                                       : (test_group + 1U) * group_size;
    if (group_end <= test_start) {
      const std::uint64_t potential_end =
          test_start > purge_rows ? test_start - purge_rows : 0U;
      if (potential_end < valid_end && potential_end >= group_start) {
        valid_end = potential_end;
      }
    }
    if (group_start >= test_end) {
      const std::uint64_t potential_start = test_end + embargo_rows;
      if (potential_start > valid_start && potential_start <= group_end) {
        valid_start = potential_start;
      }
    }
  }
  *valid_start_output = valid_start;
  *valid_end_output = valid_end;
  return valid_start < valid_end;
}

__device__ bool cpcv_combination_has_training_rows_v1(
    std::uint64_t capped_rows, std::uint64_t split_count,
    std::uint64_t test_group_count, std::uint64_t sampled_combination,
    std::uint64_t purge_rows, std::uint64_t embargo_rows) {
  for (std::uint64_t group = 0U; group < split_count; ++group) {
    std::uint64_t valid_start = 0U;
    std::uint64_t valid_end = 0U;
    if (cpcv_training_group_range_v1(
            capped_rows, split_count, test_group_count, sampled_combination,
            group, purge_rows, embargo_rows, &valid_start, &valid_end)) {
      return true;
    }
  }
  return false;
}

__host__ __device__ std::uint64_t divide_round_up_v1(std::uint64_t value,
                                                      std::uint64_t divisor) {
  return divisor == 0U ? 0U : value / divisor + (value % divisor != 0U ? 1U : 0U);
}

__device__ std::uint64_t ceil_fraction_rows_v1(std::uint64_t rows,
                                               double fraction) {
  return static_cast<std::uint64_t>(ceil(static_cast<double>(rows) * fraction));
}

__global__ void exact_fold_descriptors_kernel_v1(
    std::uint64_t selection_rows, double insample_fraction,
    std::uint64_t split_count, std::uint64_t test_group_count,
    double embargo_fraction, double purge_fraction,
    std::uint64_t cpcv_max_rows,
    NeoResidentTrimPrefilterFoldDescriptorV1* descriptors,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  for (std::uint64_t fold = 0U; fold < MAXIMUM_REFIT_FOLDS_V1; ++fold) {
    descriptors[fold] = {};
  }
  bool emitted_cpcv = false;
  if (split_count >= 2U && test_group_count > 0U &&
      test_group_count < split_count) {
    const std::uint64_t capped_rows =
        cpcv_max_rows > 0U && cpcv_max_rows < selection_rows
            ? cpcv_max_rows
            : selection_rows;
    const std::uint64_t fit_tail_offset = selection_rows - capped_rows;
    const std::uint64_t group_size = capped_rows / split_count;
    std::uint64_t raw_available_combinations = 0U;
    if (group_size > 0U &&
         combination_count_checked_v1(split_count, test_group_count,
                                      &raw_available_combinations) &&
        raw_available_combinations > 0U) {
      const std::uint64_t purge_rows =
          ceil_fraction_rows_v1(capped_rows, purge_fraction);
      const std::uint64_t embargo_rows =
          ceil_fraction_rows_v1(capped_rows, embargo_fraction);
      std::uint64_t valid_available_combinations = 0U;
      for (std::uint64_t raw_combination = 0U;
           raw_combination < raw_available_combinations; ++raw_combination) {
        if (cpcv_combination_has_training_rows_v1(
                capped_rows, split_count, test_group_count, raw_combination,
                purge_rows, embargo_rows)) {
          if (valid_available_combinations == U64_MAX_V1) {
            device_seal->valid = 0U;
            device_seal->device_fault_word =
                NEO_TRIM_PREFILTER_FAULT_CPCV_GEOMETRY_V1;
            return;
          }
          ++valid_available_combinations;
        }
      }
      if (valid_available_combinations > 0U) {
        const std::uint64_t step = divide_round_up_v1(
            valid_available_combinations, MAXIMUM_REFIT_FOLDS_V1);
        std::uint64_t sampled_valid_combination_rank = 0U;
        std::uint64_t next_fold = 0U;
        for (std::uint64_t raw_combination = 0U;
             raw_combination < raw_available_combinations &&
             next_fold < MAXIMUM_REFIT_FOLDS_V1;
             ++raw_combination) {
          if (!cpcv_combination_has_training_rows_v1(
                  capped_rows, split_count, test_group_count, raw_combination,
                  purge_rows, embargo_rows)) {
            continue;
          }
          const std::uint64_t target_valid_rank = next_fold * step;
          if (sampled_valid_combination_rank == target_valid_rank) {
            auto& descriptor = descriptors[next_fold];
            descriptor.sampled_combination = raw_combination;
            descriptor.available_combinations = valid_available_combinations;
            descriptor.capped_rows = capped_rows;
            descriptor.fit_tail_offset = fit_tail_offset;
            descriptor.split_count = split_count;
            descriptor.test_group_count = test_group_count;
            descriptor.purge_rows = purge_rows;
            descriptor.embargo_rows = embargo_rows;
            descriptor.use_cpcv = 1U;
            descriptor.valid = 1U;
            emitted_cpcv = true;
            ++next_fold;
          }
          ++sampled_valid_combination_rank;
        }
      }
    }
  }
  if (!emitted_cpcv) {
    std::uint64_t prefix_train_end = static_cast<std::uint64_t>(
        floor(insample_fraction * static_cast<double>(selection_rows)));
    if (prefix_train_end < 2U) {
      prefix_train_end = 2U;
    }
    if (prefix_train_end > selection_rows - 1U) {
      prefix_train_end = selection_rows - 1U;
    }
    const std::uint64_t prefix_exclusive_end = prefix_train_end - 1U;
    descriptors[0].capped_rows = selection_rows;
    descriptors[0].prefix_exclusive_end = prefix_exclusive_end;
    descriptors[0].use_cpcv = 0U;
    descriptors[0].valid = 1U;
  }
  if (descriptors[0].valid == 0U) {
    device_seal->valid = 0U;
    device_seal->device_fault_word = NEO_TRIM_PREFILTER_FAULT_CPCV_GEOMETRY_V1;
  }
}

__device__ bool row_in_fold_train_v1(
    std::uint64_t row,
    const NeoResidentTrimPrefilterFoldDescriptorV1& descriptor) {
  if (descriptor.valid == 0U) {
    return false;
  }
  if (descriptor.use_cpcv == 0U) {
    return row < descriptor.prefix_exclusive_end;
  }
  if (row < descriptor.fit_tail_offset ||
      row >= descriptor.fit_tail_offset + descriptor.capped_rows ||
      descriptor.split_count < 2U) {
    return false;
  }
  const std::uint64_t capped_row = row - descriptor.fit_tail_offset;
  const std::uint64_t split_count = descriptor.split_count;
  const std::uint64_t group_size = descriptor.capped_rows / split_count;
  if (group_size == 0U) {
    return false;
  }
  std::uint64_t group = capped_row / group_size;
  if (group >= split_count) {
    group = split_count - 1U;
  }
  if (lexicographic_test_group_combination_v1(
          split_count, descriptor.test_group_count,
          descriptor.sampled_combination, group)) {
    return false;
  }
  std::uint64_t valid_start = 0U;
  std::uint64_t valid_end = 0U;
  if (!cpcv_training_group_range_v1(
          descriptor.capped_rows, split_count, descriptor.test_group_count,
          descriptor.sampled_combination, group, descriptor.purge_rows,
          descriptor.embargo_rows, &valid_start, &valid_end)) {
    return false;
  }
  return capped_row >= valid_start && capped_row < valid_end;
}

struct CorrelationOutcomeV1 {
  double absolute_correlation;
  std::uint64_t used;
  std::uint64_t skipped;
  bool rankable;
};

__device__ CorrelationOutcomeV1 pairwise_two_pass_one_direction_v1(
    const double* indicators_bar_major,
    const unsigned char* indicators_validity_u4,
    const double* labels, std::uint64_t parent_column_count,
    std::uint64_t selection_row_start, std::uint64_t selection_rows,
    std::uint64_t column,
    const NeoResidentTrimPrefilterFoldDescriptorV1& descriptor,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  std::uint64_t used = 0U;
  std::uint64_t skipped = 0U;
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (std::uint64_t row = 0U; row < selection_rows; ++row) {
    if (!row_in_fold_train_v1(row, descriptor)) {
      continue;
    }
    const std::uint64_t parent_row = selection_row_start + row;
    const std::uint64_t cell = parent_row * parent_column_count + column;
    const unsigned char validity =
        validity_code_v1(indicators_validity_u4, cell);
    if (validity > 9U) {
      atomicCAS(&device_seal->device_fault_word, NEO_TRIM_PREFILTER_FAULT_NONE_V1,
                NEO_TRIM_PREFILTER_FAULT_INVALID_VALIDITY_V1);
      device_seal->valid = 0U;
      ++skipped;
      continue;
    }
    const double x = indicators_bar_major[cell];
    const double y = labels[row];
    if (validity_code_v1(indicators_validity_u4, cell) == 0U && isfinite(x) &&
        isfinite(y)) {
      ++used;
      sum_x += x;
      sum_y += y;
    } else {
      ++skipped;
    }
  }
  if (used < MINIMUM_PAIRWISE_SAMPLES_V1) {
    return {0.0, used, skipped, false};
  }
  const double mean_x = sum_x / static_cast<double>(used);
  const double mean_y = sum_y / static_cast<double>(used);
  double sxx = 0.0;
  double syy = 0.0;
  double sxy = 0.0;
  for (std::uint64_t row = 0U; row < selection_rows; ++row) {
    if (!row_in_fold_train_v1(row, descriptor)) {
      continue;
    }
    const std::uint64_t parent_row = selection_row_start + row;
    const std::uint64_t cell = parent_row * parent_column_count + column;
    const double x = indicators_bar_major[cell];
    const double y = labels[row];
    if (validity_code_v1(indicators_validity_u4, cell) != 0U || !isfinite(x) ||
        !isfinite(y)) {
      continue;
    }
    const double dx = x - mean_x;
    const double dy = y - mean_y;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const double denominator = sqrt(sxx * syy);
  if (!isfinite(denominator) || denominator <= 0.0) {
    return {0.0, used, skipped, false};
  }
  double correlation = sxy / denominator;
  if (!isfinite(correlation)) {
    atomicCAS(&device_seal->device_fault_word, NEO_TRIM_PREFILTER_FAULT_NONE_V1,
              NEO_TRIM_PREFILTER_FAULT_NONFINITE_DECISION_V1);
    device_seal->valid = 0U;
    return {0.0, used, skipped, false};
  }
  correlation = fmax(-1.0, fmin(1.0, correlation));
  return {fabs(correlation), used, skipped, true};
}

__global__ void pairwise_two_pass_correlation_kernel_v1(
    const double* indicators_bar_major,
    const unsigned char* indicators_validity_u4,
    const unsigned char* column_class_flags_device, const double* long_labels,
    const double* short_labels,
    const NeoResidentTrimPrefilterFoldDescriptorV1* descriptors,
    std::uint64_t parent_column_count, std::uint64_t selection_row_start,
    std::uint64_t selection_rows, double* column_scores,
    double* column_instability, unsigned char* column_rankability,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  const std::uint64_t column = blockIdx.x;
  if (column >= parent_column_count) {
    return;
  }
  if (threadIdx.x != 0U) {
    return;
  }
  if ((column_class_flags_device[column] & NEO_COLUMN_CLASS_STATE_V1) != 0U) {
    column_scores[column] = F64_INFINITY_V1;
    column_instability[column] = 0.0;
    column_rankability[column] = 1U;
    return;
  }
  double worst = F64_INFINITY_V1;
  double best = 0.0;
  bool rankable_in_all = true;
  bool saw_fold = false;
  for (std::uint64_t fold = 0U; fold < MAXIMUM_REFIT_FOLDS_V1; ++fold) {
    if (descriptors[fold].valid == 0U) {
      continue;
    }
    saw_fold = true;
    const CorrelationOutcomeV1 long_outcome =
        pairwise_two_pass_one_direction_v1(
            indicators_bar_major, indicators_validity_u4, long_labels,
            parent_column_count, selection_row_start, selection_rows, column,
            descriptors[fold], device_seal);
    const CorrelationOutcomeV1 short_outcome =
        pairwise_two_pass_one_direction_v1(
            indicators_bar_major, indicators_validity_u4, short_labels,
            parent_column_count, selection_row_start, selection_rows, column,
            descriptors[fold], device_seal);
    if (!long_outcome.rankable && !short_outcome.rankable) {
      rankable_in_all = false;
      break;
    }
    double direction_score = long_outcome.rankable
                                 ? long_outcome.absolute_correlation
                                 : short_outcome.absolute_correlation;
    if (short_outcome.rankable) {
      direction_score = fmax(direction_score, short_outcome.absolute_correlation);
    }
    if (!isfinite(direction_score)) {
      atomicCAS(&device_seal->device_fault_word,
                NEO_TRIM_PREFILTER_FAULT_NONE_V1,
                NEO_TRIM_PREFILTER_FAULT_NONFINITE_DECISION_V1);
      device_seal->valid = 0U;
      rankable_in_all = false;
      break;
    }
    worst = fmin(worst, direction_score);
    best = fmax(best, direction_score);
  }
  if (!saw_fold || !rankable_in_all || !isfinite(worst)) {
    column_scores[column] = -F64_INFINITY_V1;
    column_instability[column] = 0.0;
    column_rankability[column] = 0U;
    return;
  }
  column_scores[column] = worst;
  column_instability[column] = best - worst;
  column_rankability[column] = 1U;
}

__device__ std::uint64_t monotone_nonnegative_f64_key_v1(double value) {
  if (value < 0.0 || isnan(value)) {
    return 0U;
  }
  const std::uint64_t bits = __double_as_longlong(value);
  return bits == U64_MAX_V1 ? U64_MAX_V1 : bits + 1U;
}

__global__ void initialize_score_rank_inputs_kernel_v1(
    const double* column_scores, const unsigned char* column_rankability,
    std::uint64_t parent_column_count, std::uint64_t* keys,
    std::uint32_t* parent_indices,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  const std::uint64_t column = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               static_cast<std::uint64_t>(threadIdx.x);
  if (column >= parent_column_count) {
    return;
  }
  if (column > U32_MAX_V1) {
    device_seal->valid = 0U;
    atomicCAS(&device_seal->device_fault_word, NEO_TRIM_PREFILTER_FAULT_NONE_V1,
              NEO_TRIM_PREFILTER_FAULT_SELECTED_MAP_OVERFLOW_V1);
    return;
  }
  parent_indices[column] = static_cast<std::uint32_t>(column);
  keys[column] = column_rankability[column] != 0U
                     ? monotone_nonnegative_f64_key_v1(column_scores[column])
                     : 0U;
}

// `input_parent_indices_are_ascending` is established by the initialization
// kernel. CUB radix sort is stable, so
// `stable_equal_keys_preserve_parent_index_order` for the score tie-break.

__global__ void finalize_state_template_timeframe_quota_kernel_v1(
    const std::uint64_t* sorted_keys,
    const std::uint32_t* sorted_parent_indices,
    const unsigned char* column_class_flags_device,
    const std::uint32_t* timeframe_group_ids_device,
    const unsigned char* template_force_keep_flags_device,
    std::uint64_t parent_column_count, std::uint64_t resolved_top_k,
    std::uint64_t minimum_per_timeframe, std::uint64_t timeframe_group_count,
    unsigned char* column_rankability_and_keep_flags,
    std::uint32_t* timeframe_group_counts,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  for (std::uint64_t group = 0U; group < timeframe_group_count; ++group) {
    timeframe_group_counts[group] = 0U;
  }
  for (std::uint64_t column = 0U; column < parent_column_count; ++column) {
    column_rankability_and_keep_flags[column] = 0U;
  }
  std::uint64_t state_count = 0U;
  for (std::uint64_t column = 0U; column < parent_column_count; ++column) {
    if ((column_class_flags_device[column] & NEO_COLUMN_CLASS_STATE_V1) != 0U) {
      ++state_count;
    }
  }
  std::uint64_t actual_top_k = resolved_top_k + state_count;
  if (actual_top_k < resolved_top_k || actual_top_k > parent_column_count) {
    actual_top_k = parent_column_count;
  }
  std::uint64_t admitted_by_rank = 0U;
  for (std::uint64_t rank = 0U; rank < parent_column_count; ++rank) {
    if (sorted_keys[rank] == 0U || admitted_by_rank >= actual_top_k) {
      break;
    }
    const std::uint32_t parent = sorted_parent_indices[rank];
    column_rankability_and_keep_flags[parent] = 1U;
    ++admitted_by_rank;
    const std::uint32_t group = timeframe_group_ids_device[parent];
    if (group > 0U && group <= timeframe_group_count) {
      ++timeframe_group_counts[group - 1U];
    }
  }
  if (minimum_per_timeframe > 0U) {
    for (std::uint64_t rank = 0U; rank < parent_column_count; ++rank) {
      if (sorted_keys[rank] == 0U) {
        break;
      }
      const std::uint32_t parent = sorted_parent_indices[rank];
      const std::uint32_t group = timeframe_group_ids_device[parent];
      if (group == 0U || group > timeframe_group_count ||
          timeframe_group_counts[group - 1U] >= minimum_per_timeframe) {
        continue;
      }
      if (column_rankability_and_keep_flags[parent] == 0U) {
        column_rankability_and_keep_flags[parent] = 1U;
        ++timeframe_group_counts[group - 1U];
      }
    }
  }
  for (std::uint64_t parent = 0U; parent < parent_column_count; ++parent) {
    if (template_force_keep_flags_device[parent] != 0U ||
        (column_class_flags_device[parent] & NEO_COLUMN_CLASS_TEMPLATE_V1) != 0U) {
      column_rankability_and_keep_flags[parent] = 1U;
    }
  }
  if (device_seal->device_fault_word != NEO_TRIM_PREFILTER_FAULT_NONE_V1) {
    device_seal->valid = 0U;
  }
}

__global__ void select_ascending_parent_map_kernel_v1(
    std::uint64_t parent_column_count, std::uint32_t* parent_indices) {
  const std::uint64_t parent = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               static_cast<std::uint64_t>(threadIdx.x);
  if (parent < parent_column_count) {
    parent_indices[parent] = static_cast<std::uint32_t>(parent);
  }
}

struct Sha256StateV1 {
  std::uint32_t state[8];
  std::uint8_t block[64];
  std::uint64_t total_bytes;
  std::uint32_t block_len;
};

__device__ __forceinline__ std::uint32_t rotate_right_v1(std::uint32_t value,
                                                         std::uint32_t amount) {
  return (value >> amount) | (value << (32U - amount));
}

__device__ void sha256_transform_v1(Sha256StateV1* state) {
  constexpr std::uint32_t k[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::uint32_t words[64];
  for (std::uint32_t i = 0U; i < 16U; ++i) {
    const std::uint32_t offset = i * 4U;
    words[i] = (static_cast<std::uint32_t>(state->block[offset]) << 24U) |
               (static_cast<std::uint32_t>(state->block[offset + 1U]) << 16U) |
               (static_cast<std::uint32_t>(state->block[offset + 2U]) << 8U) |
               static_cast<std::uint32_t>(state->block[offset + 3U]);
  }
  for (std::uint32_t i = 16U; i < 64U; ++i) {
    const std::uint32_t s0 = rotate_right_v1(words[i - 15U], 7U) ^
                             rotate_right_v1(words[i - 15U], 18U) ^
                             (words[i - 15U] >> 3U);
    const std::uint32_t s1 = rotate_right_v1(words[i - 2U], 17U) ^
                             rotate_right_v1(words[i - 2U], 19U) ^
                             (words[i - 2U] >> 10U);
    words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
  }
  std::uint32_t a = state->state[0];
  std::uint32_t b = state->state[1];
  std::uint32_t c = state->state[2];
  std::uint32_t d = state->state[3];
  std::uint32_t e = state->state[4];
  std::uint32_t f = state->state[5];
  std::uint32_t g = state->state[6];
  std::uint32_t h = state->state[7];
  for (std::uint32_t i = 0U; i < 64U; ++i) {
    const std::uint32_t s1 = rotate_right_v1(e, 6U) ^ rotate_right_v1(e, 11U) ^
                             rotate_right_v1(e, 25U);
    const std::uint32_t choose = (e & f) ^ (~e & g);
    const std::uint32_t temp1 = h + s1 + choose + k[i] + words[i];
    const std::uint32_t s0 = rotate_right_v1(a, 2U) ^ rotate_right_v1(a, 13U) ^
                             rotate_right_v1(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state->state[0] += a;
  state->state[1] += b;
  state->state[2] += c;
  state->state[3] += d;
  state->state[4] += e;
  state->state[5] += f;
  state->state[6] += g;
  state->state[7] += h;
}

__device__ void sha256_init_v1(Sha256StateV1* state) {
  const std::uint32_t initial[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                    0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                    0x1f83d9abU, 0x5be0cd19U};
  for (std::uint32_t i = 0U; i < 8U; ++i) {
    state->state[i] = initial[i];
  }
  state->total_bytes = 0U;
  state->block_len = 0U;
}

__device__ void sha256_update_byte_v1(Sha256StateV1* state, std::uint8_t byte) {
  state->block[state->block_len++] = byte;
  ++state->total_bytes;
  if (state->block_len == 64U) {
    sha256_transform_v1(state);
    state->block_len = 0U;
  }
}

__device__ void sha256_update_u64_le_v1(Sha256StateV1* state,
                                        std::uint64_t value) {
  for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
    sha256_update_byte_v1(state, static_cast<std::uint8_t>(value >> shift));
  }
}

__device__ void sha256_update_u32_le_v1(Sha256StateV1* state,
                                        std::uint32_t value) {
  for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
    sha256_update_byte_v1(state, static_cast<std::uint8_t>(value >> shift));
  }
}

__device__ void sha256_finalize_v1(Sha256StateV1* state,
                                   std::uint8_t output[32]) {
  const std::uint64_t bit_length = state->total_bytes * 8U;
  sha256_update_byte_v1(state, 0x80U);
  while (state->block_len != 56U) {
    sha256_update_byte_v1(state, 0U);
  }
  for (std::int32_t shift = 56; shift >= 0; shift -= 8) {
    sha256_update_byte_v1(state,
                          static_cast<std::uint8_t>(bit_length >> shift));
  }
  for (std::uint32_t i = 0U; i < 8U; ++i) {
    output[i * 4U] = static_cast<std::uint8_t>(state->state[i] >> 24U);
    output[i * 4U + 1U] = static_cast<std::uint8_t>(state->state[i] >> 16U);
    output[i * 4U + 2U] = static_cast<std::uint8_t>(state->state[i] >> 8U);
    output[i * 4U + 3U] = static_cast<std::uint8_t>(state->state[i]);
  }
}

__global__ void seal_selected_map_kernel_v1(
    const std::uint32_t* selected_compact_to_parent_columns_device,
    const std::uint64_t* selected_column_count_device,
    NeoResidentTrimPrefilterPlanV1 plan,
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  if (device_seal->device_fault_word != NEO_TRIM_PREFILTER_FAULT_NONE_V1) {
    device_seal->valid = 0U;
    return;
  }
  const std::uint64_t selected = *selected_column_count_device;
  if (selected > plan.parent_column_count) {
    device_seal->valid = 0U;
    device_seal->device_fault_word =
        NEO_TRIM_PREFILTER_FAULT_SELECTED_MAP_OVERFLOW_V1;
    return;
  }
  Sha256StateV1 hash{};
  sha256_init_v1(&hash);
  constexpr char domain[] =
      "neoethos.resident-trim-prefilter-selected-map.v1\0";
  for (std::uint32_t i = 0U; i < sizeof(domain) - 1U; ++i) {
    sha256_update_byte_v1(&hash, static_cast<std::uint8_t>(domain[i]));
  }
  for (std::uint32_t i = 0U; i < 32U; ++i) {
    sha256_update_byte_v1(&hash, plan.plan_identity_sha256[i]);
  }
  sha256_update_u64_le_v1(&hash, selected);
  sha256_update_u64_le_v1(&hash, plan.selection_row_start);
  sha256_update_u64_le_v1(&hash, plan.selection_row_end);
  sha256_update_u64_le_v1(&hash, plan.holdout_row_start);
  sha256_update_u64_le_v1(&hash, plan.holdout_row_end);
  std::uint32_t previous = 0U;
  for (std::uint64_t compact = 0U; compact < selected; ++compact) {
    const std::uint32_t parent =
        selected_compact_to_parent_columns_device[compact];
    if (parent >= plan.parent_column_count ||
        (compact > 0U && parent <= previous)) {
      device_seal->valid = 0U;
      device_seal->device_fault_word =
          NEO_TRIM_PREFILTER_FAULT_SELECTED_MAP_OVERFLOW_V1;
      return;
    }
    previous = parent;
    sha256_update_u32_le_v1(&hash, parent);
  }
  device_seal->selected_count = selected;
  device_seal->selection_row_start = plan.selection_row_start;
  device_seal->selection_row_end = plan.selection_row_end;
  device_seal->holdout_row_start = plan.holdout_row_start;
  device_seal->holdout_row_end = plan.holdout_row_end;
  sha256_finalize_v1(&hash, device_seal->selected_map_sha256);
  device_seal->valid = 1U;
}

__global__ void initialize_device_seal_kernel_v1(
    NeoResidentTrimPrefilterDeviceSealV1* device_seal) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *device_seal = {};
    device_seal->abi_version = NEO_RESIDENT_TRIM_PREFILTER_ABI_V1;
    device_seal->valid = 0U;
    device_seal->device_fault_word = NEO_TRIM_PREFILTER_FAULT_NONE_V1;
  }
}

__global__ void fill_identity_selected_map_kernel_v1(
    std::uint64_t parent_column_count,
    std::uint32_t* selected_compact_to_parent_columns_device,
    std::uint64_t* selected_column_count_device) {
  const std::uint64_t column = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               static_cast<std::uint64_t>(threadIdx.x);
  if (column < parent_column_count) {
    selected_compact_to_parent_columns_device[column] =
        static_cast<std::uint32_t>(column);
  }
  if (column == 0U) {
    *selected_column_count_device = parent_column_count;
  }
}

template <typename T>
cudaError_t allocate_async_v1(T** pointer, std::uint64_t count,
                              cudaStream_t stream) {
  if (pointer == nullptr || count == 0U ||
      count > U64_MAX_V1 / static_cast<std::uint64_t>(sizeof(T))) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t bytes = count * static_cast<std::uint64_t>(sizeof(T));
  if (bytes > static_cast<std::uint64_t>(SIZE_MAX)) {
    return cudaErrorInvalidValue;
  }
  return cudaMallocAsync(reinterpret_cast<void**>(pointer),
                         static_cast<std::size_t>(bytes), stream);
}

cudaError_t allocate_bytes_async_v1(void** pointer, std::uint64_t bytes,
                                    cudaStream_t stream) {
  if (pointer == nullptr || bytes == 0U ||
      bytes > static_cast<std::uint64_t>(SIZE_MAX)) {
    return cudaErrorInvalidValue;
  }
  return cudaMallocAsync(pointer, static_cast<std::size_t>(bytes), stream);
}

template <typename T>
cudaError_t release_one_async_v1(T** pointer, cudaStream_t stream) {
  if (pointer != nullptr && *pointer != nullptr) {
    const cudaError_t status = cudaFreeAsync(*pointer, stream);
    if (status == cudaSuccess) {
      *pointer = nullptr;
    }
    return status;
  }
  return cudaSuccess;
}

}  // namespace

struct NeoResidentTrimPrefilterRunV1 {
  NeoResidentTrimPrefilterImportV1 import;
  NeoResidentTrimPrefilterPlanV1 plan;
  NeoResidentTrimPrefilterAllocationReceiptV1 receipt;
  cudaStream_t admitted_run_stream;
  cudaEvent_t parent_ready_event;
  cudaEvent_t schema_ready_event;
  cudaEvent_t trim_prefilter_ready_event;
  std::uint32_t next_stage;
  bool prefilter_active;
  double* long_labels;
  double* short_labels;
  NeoResidentTrimPrefilterLabelCensusV1* label_census;
  NeoResidentTrimPrefilterFoldDescriptorV1* fold_descriptors;
  double* column_scores;
  double* column_instability;
  unsigned char* column_rankability_and_keep_flags;
  std::uint64_t* radix_keys_input;
  std::uint64_t* radix_keys_output;
  std::uint32_t* radix_indices_input;
  std::uint32_t* radix_indices_output;
  std::uint32_t* timeframe_group_counts;
  std::uint32_t* selected_compact_to_parent_columns_device;
  std::uint64_t* selected_column_count_device;
  NeoResidentTrimPrefilterDeviceSealV1* device_seal;
  void* cub_select_scratch;
  void* cub_radix_sort_scratch;
  std::uint64_t same_stream_enqueue_count;
};

namespace {

cudaError_t release_intermediate_buffers_async_v1(
    NeoResidentTrimPrefilterRunV1* run) {
  if (run == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t first_error = cudaSuccess;
#define NEO_RELEASE_INTERMEDIATE_V1(pointer)                                  \
  do {                                                                        \
    const cudaError_t release_status =                                        \
        release_one_async_v1(&(pointer), run->admitted_run_stream);           \
    if (first_error == cudaSuccess && release_status != cudaSuccess) {         \
      first_error = release_status;                                           \
    }                                                                         \
  } while (false)
  NEO_RELEASE_INTERMEDIATE_V1(run->long_labels);
  NEO_RELEASE_INTERMEDIATE_V1(run->short_labels);
  NEO_RELEASE_INTERMEDIATE_V1(run->label_census);
  NEO_RELEASE_INTERMEDIATE_V1(run->fold_descriptors);
  NEO_RELEASE_INTERMEDIATE_V1(run->column_scores);
  NEO_RELEASE_INTERMEDIATE_V1(run->column_instability);
  NEO_RELEASE_INTERMEDIATE_V1(run->column_rankability_and_keep_flags);
  NEO_RELEASE_INTERMEDIATE_V1(run->radix_keys_input);
  NEO_RELEASE_INTERMEDIATE_V1(run->radix_keys_output);
  NEO_RELEASE_INTERMEDIATE_V1(run->radix_indices_input);
  NEO_RELEASE_INTERMEDIATE_V1(run->radix_indices_output);
  NEO_RELEASE_INTERMEDIATE_V1(run->timeframe_group_counts);
#undef NEO_RELEASE_INTERMEDIATE_V1
  if (run->cub_select_scratch != nullptr) {
    const cudaError_t release_status =
        cudaFreeAsync(run->cub_select_scratch, run->admitted_run_stream);
    if (release_status == cudaSuccess) {
      run->cub_select_scratch = nullptr;
    } else if (first_error == cudaSuccess) {
      first_error = release_status;
    }
  }
  if (run->cub_radix_sort_scratch != nullptr) {
    const cudaError_t release_status =
        cudaFreeAsync(run->cub_radix_sort_scratch, run->admitted_run_stream);
    if (release_status == cudaSuccess) {
      run->cub_radix_sort_scratch = nullptr;
    } else if (first_error == cudaSuccess) {
      first_error = release_status;
    }
  }
  return first_error;
}

cudaError_t release_all_buffers_async_v1(NeoResidentTrimPrefilterRunV1* run) {
  if (run == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t first_error = release_intermediate_buffers_async_v1(run);
  const cudaError_t map_status = release_one_async_v1(
      &run->selected_compact_to_parent_columns_device,
      run->admitted_run_stream);
  if (first_error == cudaSuccess && map_status != cudaSuccess) {
    first_error = map_status;
  }
  const cudaError_t count_status = release_one_async_v1(
      &run->selected_column_count_device, run->admitted_run_stream);
  if (first_error == cudaSuccess && count_status != cudaSuccess) {
    first_error = count_status;
  }
  const cudaError_t seal_status =
      release_one_async_v1(&run->device_seal, run->admitted_run_stream);
  if (first_error == cudaSuccess && seal_status != cudaSuccess) {
    first_error = seal_status;
  }
  return first_error;
}

bool query_cub_scratch_v1(std::uint64_t parent_column_count,
                          cudaStream_t stream, std::uint64_t* select_bytes,
                          std::uint64_t* radix_sort_bytes) {
  if (select_bytes == nullptr || radix_sort_bytes == nullptr ||
      parent_column_count == 0U || parent_column_count > 0x7fffffffU) {
    return false;
  }
  std::size_t selected = 0U;
  const cudaError_t select_status = cub::DeviceSelect::Flagged(
      nullptr, selected, static_cast<const std::uint32_t*>(nullptr),
      static_cast<const unsigned char*>(nullptr),
      static_cast<std::uint32_t*>(nullptr),
      static_cast<std::uint64_t*>(nullptr),
      static_cast<int>(parent_column_count), stream);
  std::size_t sorted = 0U;
  const cudaError_t sort_status = cub::DeviceRadixSort::SortPairsDescending(
      nullptr, sorted, static_cast<const std::uint64_t*>(nullptr),
      static_cast<std::uint64_t*>(nullptr),
      static_cast<const std::uint32_t*>(nullptr),
      static_cast<std::uint32_t*>(nullptr),
      static_cast<int>(parent_column_count), 0, 64, stream);
  if (select_status != cudaSuccess || sort_status != cudaSuccess) {
    return false;
  }
  *select_bytes = static_cast<std::uint64_t>(selected);
  *radix_sort_bytes = static_cast<std::uint64_t>(sorted);
  return true;
}

bool fill_allocation_receipt_v1(
    const NeoResidentTrimPrefilterImportV1* import,
    const NeoResidentTrimPrefilterPlanV1* plan,
    NeoResidentTrimPrefilterAllocationReceiptV1* receipt) {
  if (!validate_import_and_plan_v1(import, plan) || receipt == nullptr) {
    return false;
  }
  const bool prefilter_active =
      plan->resolved_top_k > 0U &&
      plan->parent_column_count > plan->resolved_top_k;
  const std::uint64_t selection_rows =
      plan->selection_row_end - plan->selection_row_start;
  std::uint64_t long_labels_bytes = 0U;
  std::uint64_t short_labels_bytes = 0U;
  std::uint64_t label_census_bytes = 0U;
  std::uint64_t fold_descriptor_bytes = 0U;
  std::uint64_t column_score_bytes = 0U;
  std::uint64_t column_instability_bytes = 0U;
  std::uint64_t column_rankability_bytes = 0U;
  std::uint64_t radix_key_ping_pong_bytes = 0U;
  std::uint64_t radix_index_ping_pong_bytes = 0U;
  std::uint64_t timeframe_group_counter_bytes = 0U;
  std::uint64_t cub_select_scratch_bytes = 0U;
  std::uint64_t cub_radix_sort_scratch_bytes = 0U;
  if (prefilter_active) {
    std::uint64_t one_key_array = 0U;
    std::uint64_t one_index_array = 0U;
    if (!checked_mul_v1(selection_rows, sizeof(double), &long_labels_bytes) ||
        !checked_mul_v1(selection_rows, sizeof(double), &short_labels_bytes) ||
        !checked_mul_v1(1U, sizeof(NeoResidentTrimPrefilterLabelCensusV1),
                        &label_census_bytes) ||
        !checked_mul_v1(MAXIMUM_REFIT_FOLDS_V1,
                        sizeof(NeoResidentTrimPrefilterFoldDescriptorV1),
                        &fold_descriptor_bytes) ||
        !checked_mul_v1(plan->parent_column_count, sizeof(double),
                        &column_score_bytes) ||
        !checked_mul_v1(plan->parent_column_count, sizeof(double),
                        &column_instability_bytes) ||
        !checked_mul_v1(plan->parent_column_count, sizeof(unsigned char),
                        &column_rankability_bytes) ||
        !checked_mul_v1(plan->parent_column_count, sizeof(std::uint64_t),
                        &one_key_array) ||
        !checked_mul_v1(one_key_array, 2U, &radix_key_ping_pong_bytes) ||
        !checked_mul_v1(plan->parent_column_count, sizeof(std::uint32_t),
                        &one_index_array) ||
        !checked_mul_v1(one_index_array, 2U,
                        &radix_index_ping_pong_bytes) ||
        !checked_mul_v1(import->timeframe_group_count, sizeof(std::uint32_t),
                        &timeframe_group_counter_bytes) ||
        !query_cub_scratch_v1(plan->parent_column_count,
                              import->admitted_run_stream,
                              &cub_select_scratch_bytes,
                              &cub_radix_sort_scratch_bytes)) {
      return false;
    }
  }
  std::uint64_t selected_column_map_bytes = 0U;
  if (!checked_mul_v1(plan->parent_column_count, sizeof(std::uint32_t),
                      &selected_column_map_bytes)) {
    return false;
  }
  const std::uint64_t selected_column_count_bytes = sizeof(std::uint64_t);
  const std::uint64_t device_seal_bytes =
      sizeof(NeoResidentTrimPrefilterDeviceSealV1);
  std::uint64_t retained_device_bytes = 0U;
  for (std::uint64_t bytes : {selected_column_map_bytes,
                              selected_column_count_bytes,
                              device_seal_bytes}) {
    if (!checked_accumulate_v1(bytes, &retained_device_bytes)) {
      return false;
    }
  }
  std::uint64_t peak_device_bytes = retained_device_bytes;
  for (std::uint64_t bytes : {
           long_labels_bytes,
           short_labels_bytes,
           label_census_bytes,
           fold_descriptor_bytes,
           column_score_bytes,
           column_instability_bytes,
           column_rankability_bytes,
           radix_key_ping_pong_bytes,
           radix_index_ping_pong_bytes,
           timeframe_group_counter_bytes,
           cub_select_scratch_bytes,
           cub_radix_sort_scratch_bytes,
       }) {
    if (!checked_accumulate_v1(bytes, &peak_device_bytes)) {
      return false;
    }
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      peak_device_bytes > import->trim_prefilter_reserved_bytes ||
      peak_device_bytes != plan->charged_peak_device_bytes ||
      peak_device_bytes > static_cast<std::uint64_t>(free_bytes)) {
    return false;
  }
  *receipt = {};
  receipt->abi_version = NEO_RESIDENT_TRIM_PREFILTER_ABI_V1;
  receipt->allocation_count = prefilter_active ? 17U : 3U;
  receipt->long_labels_bytes = long_labels_bytes;
  receipt->short_labels_bytes = short_labels_bytes;
  receipt->label_census_bytes = label_census_bytes;
  receipt->fold_descriptor_bytes = fold_descriptor_bytes;
  receipt->column_score_bytes = column_score_bytes;
  receipt->column_instability_bytes = column_instability_bytes;
  receipt->column_rankability_bytes = column_rankability_bytes;
  receipt->state_template_timeframe_metadata_bytes =
      import->schema_metadata_bytes;
  receipt->radix_key_ping_pong_bytes = radix_key_ping_pong_bytes;
  receipt->radix_index_ping_pong_bytes = radix_index_ping_pong_bytes;
  receipt->timeframe_group_counter_bytes = timeframe_group_counter_bytes;
  receipt->selected_column_map_bytes = selected_column_map_bytes;
  receipt->selected_column_count_bytes = selected_column_count_bytes;
  receipt->cub_select_scratch_bytes = cub_select_scratch_bytes;
  receipt->cub_radix_sort_scratch_bytes = cub_radix_sort_scratch_bytes;
  receipt->device_seal_bytes = device_seal_bytes;
  receipt->retained_device_bytes = retained_device_bytes;
  receipt->peak_device_bytes = peak_device_bytes;
  receipt->same_context_free_bytes = static_cast<std::uint64_t>(free_bytes);
  receipt->full_discovery_reserve_bytes =
      import->full_discovery_reserve_bytes;
  std::memcpy(receipt->allocation_plan_sha256,
              plan->allocation_plan_sha256, 32U);
  return true;
}

bool allocation_shape_equal_v1(
    const NeoResidentTrimPrefilterAllocationReceiptV1& left,
    const NeoResidentTrimPrefilterAllocationReceiptV1& right) {
  return left.abi_version == right.abi_version &&
         left.allocation_count == right.allocation_count &&
         left.long_labels_bytes == right.long_labels_bytes &&
         left.short_labels_bytes == right.short_labels_bytes &&
         left.label_census_bytes == right.label_census_bytes &&
         left.fold_descriptor_bytes == right.fold_descriptor_bytes &&
         left.column_score_bytes == right.column_score_bytes &&
         left.column_instability_bytes == right.column_instability_bytes &&
         left.column_rankability_bytes == right.column_rankability_bytes &&
         left.state_template_timeframe_metadata_bytes ==
             right.state_template_timeframe_metadata_bytes &&
         left.radix_key_ping_pong_bytes == right.radix_key_ping_pong_bytes &&
         left.radix_index_ping_pong_bytes ==
             right.radix_index_ping_pong_bytes &&
         left.timeframe_group_counter_bytes ==
             right.timeframe_group_counter_bytes &&
         left.selected_column_map_bytes == right.selected_column_map_bytes &&
         left.selected_column_count_bytes ==
             right.selected_column_count_bytes &&
         left.cub_select_scratch_bytes == right.cub_select_scratch_bytes &&
         left.cub_radix_sort_scratch_bytes ==
             right.cub_radix_sort_scratch_bytes &&
         left.device_seal_bytes == right.device_seal_bytes &&
         left.retained_device_bytes == right.retained_device_bytes &&
         left.peak_device_bytes == right.peak_device_bytes &&
         left.full_discovery_reserve_bytes ==
             right.full_discovery_reserve_bytes &&
         hashes_equal_v1(left.allocation_plan_sha256,
                         right.allocation_plan_sha256);
}

cudaError_t allocate_run_buffers_v1(NeoResidentTrimPrefilterRunV1* run) {
  const std::uint64_t columns = run->plan.parent_column_count;
  const std::uint64_t selection_rows =
      run->plan.selection_row_end - run->plan.selection_row_start;
  cudaError_t status = allocate_async_v1(
      &run->selected_compact_to_parent_columns_device, columns,
      run->admitted_run_stream);
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->selected_column_count_device, 1U,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->device_seal, 1U,
                               run->admitted_run_stream);
  }
  if (!run->prefilter_active || status != cudaSuccess) {
    return status;
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->long_labels, selection_rows,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->short_labels, selection_rows,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->label_census, 1U,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->fold_descriptors,
                               MAXIMUM_REFIT_FOLDS_V1,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->column_scores, columns,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->column_instability, columns,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->column_rankability_and_keep_flags,
                               columns, run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->radix_keys_input, columns,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->radix_keys_output, columns,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->radix_indices_input, columns,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->radix_indices_output, columns,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_async_v1(&run->timeframe_group_counts,
                               run->import.timeframe_group_count,
                               run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_bytes_async_v1(&run->cub_select_scratch,
                                     run->receipt.cub_select_scratch_bytes,
                                     run->admitted_run_stream);
  }
  if (status == cudaSuccess) {
    status = allocate_bytes_async_v1(
        &run->cub_radix_sort_scratch,
        run->receipt.cub_radix_sort_scratch_bytes, run->admitted_run_stream);
  }
  return status;
}

cudaError_t initialize_run_buffers_v1(NeoResidentTrimPrefilterRunV1* run) {
  cudaError_t status = cudaMemsetAsync(
      run->selected_column_count_device, 0, sizeof(std::uint64_t),
      run->admitted_run_stream);
  if (status == cudaSuccess) {
    initialize_device_seal_kernel_v1<<<1, 1, 0, run->admitted_run_stream>>>(
        run->device_seal);
    status = cudaPeekAtLastError();
  }
  if (run->prefilter_active && status == cudaSuccess) {
    status = cudaMemsetAsync(run->label_census, 0,
                             sizeof(NeoResidentTrimPrefilterLabelCensusV1),
                             run->admitted_run_stream);
  }
  if (run->prefilter_active && status == cudaSuccess) {
    status = cudaMemsetAsync(
        run->fold_descriptors, 0,
        sizeof(NeoResidentTrimPrefilterFoldDescriptorV1) *
            MAXIMUM_REFIT_FOLDS_V1,
        run->admitted_run_stream);
  }
  if (run->prefilter_active && status == cudaSuccess) {
    status = cudaMemsetAsync(run->column_rankability_and_keep_flags, 0,
                             run->plan.parent_column_count,
                             run->admitted_run_stream);
  }
  return status;
}

}  // namespace

extern "C" std::int32_t query_resident_trim_prefilter_scratch_v1(
    cudaStream_t admitted_run_stream, std::uint32_t selected_cuda_ordinal,
    std::uint64_t parent_column_count, std::uint32_t prefilter_active,
    std::uint64_t* cub_select_scratch_bytes,
    std::uint64_t* cub_radix_sort_scratch_bytes) {
  if (admitted_run_stream == nullptr || cub_select_scratch_bytes == nullptr ||
      cub_radix_sort_scratch_bytes == nullptr || parent_column_count == 0U) {
    return NEO_TRIM_PREFILTER_STATUS_INVALID_ARGUMENT_V1;
  }
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0 ||
      static_cast<std::uint32_t>(current_device) != selected_cuda_ordinal) {
    return NEO_TRIM_PREFILTER_STATUS_IDENTITY_MISMATCH_V1;
  }
  if (prefilter_active == 0U) {
    *cub_select_scratch_bytes = 0U;
    *cub_radix_sort_scratch_bytes = 0U;
    return NEO_TRIM_PREFILTER_STATUS_OK_V1;
  }
  return query_cub_scratch_v1(parent_column_count, admitted_run_stream,
                              cub_select_scratch_bytes,
                              cub_radix_sort_scratch_bytes)
             ? NEO_TRIM_PREFILTER_STATUS_OK_V1
             : NEO_TRIM_PREFILTER_STATUS_CUB_ERROR_V1;
}

extern "C" std::int32_t query_resident_trim_prefilter_allocation_v1(
    const NeoResidentTrimPrefilterImportV1* import,
    const NeoResidentTrimPrefilterPlanV1* plan,
    NeoResidentTrimPrefilterAllocationReceiptV1* receipt) {
  if (receipt == nullptr) {
    return NEO_TRIM_PREFILTER_STATUS_INVALID_ARGUMENT_V1;
  }
  return fill_allocation_receipt_v1(import, plan, receipt)
             ? NEO_TRIM_PREFILTER_STATUS_OK_V1
             : NEO_TRIM_PREFILTER_STATUS_INVALID_ARGUMENT_V1;
}

extern "C" std::int32_t create_resident_trim_prefilter_run_v1(
    const NeoResidentTrimPrefilterImportV1* import,
    const NeoResidentTrimPrefilterPlanV1* plan,
    const NeoResidentTrimPrefilterAllocationReceiptV1* receipt,
    NeoResidentTrimPrefilterRunV1** output) {
  if (receipt == nullptr || output == nullptr) {
    return NEO_TRIM_PREFILTER_STATUS_INVALID_ARGUMENT_V1;
  }
  *output = nullptr;
  NeoResidentTrimPrefilterAllocationReceiptV1 expected{};
  if (!fill_allocation_receipt_v1(import, plan, &expected) ||
      !allocation_shape_equal_v1(*receipt, expected)) {
    return NEO_TRIM_PREFILTER_STATUS_IDENTITY_MISMATCH_V1;
  }
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0 ||
      static_cast<std::uint32_t>(current_device) !=
          import->selected_cuda_ordinal) {
    return NEO_TRIM_PREFILTER_STATUS_IDENTITY_MISMATCH_V1;
  }
  auto* run = new (std::nothrow) NeoResidentTrimPrefilterRunV1{};
  if (run == nullptr) {
    return NEO_TRIM_PREFILTER_STATUS_OUT_OF_MEMORY_V1;
  }
  run->import = *import;
  run->plan = *plan;
  run->receipt = *receipt;
  run->admitted_run_stream = import->admitted_run_stream;
  run->parent_ready_event = import->parent_ready_event;
  run->schema_ready_event = import->schema_ready_event;
  run->trim_prefilter_ready_event = import->trim_prefilter_ready_event;
  run->next_stage = STAGE_LABELS_V1;
  run->prefilter_active =
      plan->resolved_top_k > 0U &&
      plan->parent_column_count > plan->resolved_top_k;
  cudaError_t status = allocate_run_buffers_v1(run);
  if (status == cudaSuccess) {
    status = initialize_run_buffers_v1(run);
  }
  if (status != cudaSuccess) {
    if (release_all_buffers_async_v1(run) == cudaSuccess) {
      delete run;
    }
    return NEO_TRIM_PREFILTER_STATUS_OUT_OF_MEMORY_V1;
  }
  *output = run;
  return NEO_TRIM_PREFILTER_STATUS_OK_V1;
}

extern "C" std::int32_t enqueue_resident_trim_prefilter_stage_v1(
    NeoResidentTrimPrefilterRunV1* run, std::uint32_t stage) {
  if (run == nullptr || stage != run->next_stage ||
      stage < STAGE_LABELS_V1 || stage > STAGE_DEVICE_SEAL_V1) {
    return NEO_TRIM_PREFILTER_STATUS_STATE_ERROR_V1;
  }
  cudaError_t status = cudaSuccess;
  const std::uint64_t selection_rows =
      run->plan.selection_row_end - run->plan.selection_row_start;
  const std::uint64_t columns = run->plan.parent_column_count;
  constexpr std::uint32_t threads = 256U;
  const std::uint32_t row_blocks = static_cast<std::uint32_t>(
      divide_round_up_v1(selection_rows, threads));
  const std::uint32_t column_blocks = static_cast<std::uint32_t>(
      divide_round_up_v1(columns, threads));

  if (stage == STAGE_LABELS_V1) {
    status = cudaStreamWaitEvent(run->admitted_run_stream,
                                 run->parent_ready_event, 0U);
    if (status == cudaSuccess) {
      status = cudaStreamWaitEvent(run->admitted_run_stream,
                                   run->schema_ready_event, 0U);
    }
    if (status == cudaSuccess && run->prefilter_active) {
      first_passage_labels_kernel_v1<<<row_blocks, threads, 0,
                                       run->admitted_run_stream>>>(
          run->import.close, run->import.high, run->import.low,
          run->plan.selection_row_start, selection_rows,
          run->plan.atr_period, run->plan.max_hold_bars,
          run->plan.stop_atr_multiplier, run->plan.reward_risk_ratio,
          run->plan.round_trip_cost_price, run->long_labels,
          run->short_labels, run->label_census);
      status = cudaPeekAtLastError();
    }
  } else if (stage == STAGE_LABEL_GUARD_V1 && run->prefilter_active) {
    invalidate_insufficient_labels_kernel_v1<<<1, 1, 0,
                                               run->admitted_run_stream>>>(
        run->label_census, run->device_seal);
    status = cudaPeekAtLastError();
  } else if (stage == STAGE_FOLDS_V1 && run->prefilter_active) {
    exact_fold_descriptors_kernel_v1<<<1, 1, 0, run->admitted_run_stream>>>(
        selection_rows, run->plan.insample_fraction,
        run->plan.cpcv_split_count, run->plan.cpcv_test_group_count,
        run->plan.cpcv_embargo_fraction, run->plan.cpcv_purge_fraction,
        run->plan.cpcv_max_rows, run->fold_descriptors, run->device_seal);
    status = cudaPeekAtLastError();
  } else if (stage == STAGE_CORRELATIONS_V1 && run->prefilter_active) {
    pairwise_two_pass_correlation_kernel_v1<<<
        static_cast<std::uint32_t>(columns), 32, 0,
        run->admitted_run_stream>>>(
        run->import.indicators_bar_major,
        run->import.indicators_validity_u4,
        run->import.column_class_flags_device, run->long_labels,
        run->short_labels, run->fold_descriptors, columns,
        run->plan.selection_row_start, selection_rows, run->column_scores,
        run->column_instability, run->column_rankability_and_keep_flags,
        run->device_seal);
    status = cudaPeekAtLastError();
  } else if (stage == STAGE_RANK_V1 && run->prefilter_active) {
    initialize_score_rank_inputs_kernel_v1<<<
        column_blocks, threads, 0, run->admitted_run_stream>>>(
        run->column_scores, run->column_rankability_and_keep_flags, columns,
        run->radix_keys_input, run->radix_indices_input, run->device_seal);
    status = cudaPeekAtLastError();
    if (status == cudaSuccess) {
      std::size_t scratch =
          static_cast<std::size_t>(run->receipt.cub_radix_sort_scratch_bytes);
      status = cub::DeviceRadixSort::SortPairsDescending(
          run->cub_radix_sort_scratch, scratch, run->radix_keys_input,
          run->radix_keys_output, run->radix_indices_input,
          run->radix_indices_output, static_cast<int>(columns), 0, 64,
          run->admitted_run_stream);
    }
  } else if (stage == STAGE_QUOTAS_V1 && run->prefilter_active) {
    finalize_state_template_timeframe_quota_kernel_v1<<<
        1, 1, 0, run->admitted_run_stream>>>(
        run->radix_keys_output, run->radix_indices_output,
        run->import.column_class_flags_device,
        run->import.timeframe_group_ids_device,
        run->import.template_force_keep_flags_device, columns,
        run->plan.resolved_top_k, run->plan.minimum_per_timeframe,
        run->import.timeframe_group_count,
        run->column_rankability_and_keep_flags,
        run->timeframe_group_counts, run->device_seal);
    status = cudaPeekAtLastError();
  } else if (stage == STAGE_ASCENDING_MAP_V1) {
    if (!run->prefilter_active) {
      fill_identity_selected_map_kernel_v1<<<
          column_blocks, threads, 0, run->admitted_run_stream>>>(
          columns, run->selected_compact_to_parent_columns_device,
          run->selected_column_count_device);
      status = cudaPeekAtLastError();
    } else {
      select_ascending_parent_map_kernel_v1<<<
          column_blocks, threads, 0, run->admitted_run_stream>>>(
          columns, run->radix_indices_input);
      status = cudaPeekAtLastError();
      if (status == cudaSuccess) {
        std::size_t scratch =
            static_cast<std::size_t>(run->receipt.cub_select_scratch_bytes);
        status = cub::DeviceSelect::Flagged(
            run->cub_select_scratch, scratch, run->radix_indices_input,
            run->column_rankability_and_keep_flags,
            run->selected_compact_to_parent_columns_device,
            run->selected_column_count_device, static_cast<int>(columns),
            run->admitted_run_stream);
      }
    }
  } else if (stage == STAGE_DEVICE_SEAL_V1) {
    seal_selected_map_kernel_v1<<<1, 1, 0, run->admitted_run_stream>>>(
        run->selected_compact_to_parent_columns_device,
        run->selected_column_count_device, run->plan, run->device_seal);
    status = cudaPeekAtLastError();
    if (status == cudaSuccess) {
      status = release_intermediate_buffers_async_v1(run);
    }
  }
  if (status != cudaSuccess) {
    return stage == STAGE_RANK_V1 || stage == STAGE_ASCENDING_MAP_V1
               ? NEO_TRIM_PREFILTER_STATUS_CUB_ERROR_V1
               : NEO_TRIM_PREFILTER_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  ++run->next_stage;
  return NEO_TRIM_PREFILTER_STATUS_OK_V1;
}

extern "C" std::int32_t seal_resident_trim_prefilter_views_v1(
    NeoResidentTrimPrefilterRunV1* run,
    NeoResidentTrimPrefilterViewsV1* views,
    NeoResidentTrimPrefilterReadyEventV1* ready) {
  if (run == nullptr || views == nullptr || ready == nullptr ||
      run->next_stage != STAGE_DEVICE_SEAL_V1 + 1U ||
      run->selected_compact_to_parent_columns_device == nullptr ||
      run->selected_column_count_device == nullptr || run->device_seal == nullptr) {
    return NEO_TRIM_PREFILTER_STATUS_STATE_ERROR_V1;
  }
  const cudaError_t status = cudaEventRecord(run->trim_prefilter_ready_event,
                                             run->admitted_run_stream);
  if (status != cudaSuccess) {
    return NEO_TRIM_PREFILTER_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  *views = {};
  views->abi_version = NEO_RESIDENT_TRIM_PREFILTER_ABI_V1;
  views->same_selected_column_map_for_holdout = 1U;
  views->selected_compact_to_parent_columns_device =
      run->selected_compact_to_parent_columns_device;
  views->selected_column_count_device = run->selected_column_count_device;
  views->device_seal = run->device_seal;
  views->trim_prefilter_ready_event = run->trim_prefilter_ready_event;
  views->parent_row_count = run->plan.parent_row_count;
  views->parent_column_count = run->plan.parent_column_count;
  views->selection_row_start = run->plan.selection_row_start;
  views->selection_row_end = run->plan.selection_row_end;
  views->holdout_row_start = run->plan.holdout_row_start;
  views->holdout_row_end = run->plan.holdout_row_end;
  std::memcpy(views->plan_identity_sha256, run->plan.plan_identity_sha256,
              32U);
  std::memcpy(views->view_semantics_sha256, run->plan.semantics_sha256, 32U);
  std::memcpy(views->canonical_content_merkle_sha256,
              run->import.canonical_content_merkle_sha256, 32U);
  std::memcpy(views->ordered_feature_schema_sha256,
              run->import.ordered_feature_schema_sha256, 32U);
  std::memcpy(views->cuda_device_identity_sha256,
              run->import.cuda_device_identity_sha256, 32U);
  std::memcpy(views->primary_context_identity_sha256,
              run->import.primary_context_identity_sha256, 32U);
  std::memcpy(views->run_stream_identity_sha256,
              run->import.run_stream_identity_sha256, 32U);
  std::memcpy(views->cuda_build_manifest_sha256,
              run->import.cuda_build_manifest_sha256, 32U);
  *ready = {};
  ready->abi_version = NEO_RESIDENT_TRIM_PREFILTER_ABI_V1;
  ready->same_stream_enqueue_count = run->same_stream_enqueue_count;
  ready->intermediate_host_wait_count = 0U;
  ready->intermediate_readback_count = 0U;
  ready->host_to_device_transfer_count = 0U;
  ready->device_to_host_transfer_count = 0U;
  ready->explicit_synchronization_count = 0U;
  return NEO_TRIM_PREFILTER_STATUS_OK_V1;
}

extern "C" std::int32_t enqueue_resident_trim_prefilter_release_v1(
    NeoResidentTrimPrefilterRunV1* run) {
  if (run == nullptr || run->admitted_run_stream == nullptr) {
    return NEO_TRIM_PREFILTER_STATUS_INVALID_ARGUMENT_V1;
  }
  if (release_all_buffers_async_v1(run) != cudaSuccess) {
    return NEO_TRIM_PREFILTER_STATUS_CUDA_ERROR_V1;
  }
  delete run;
  return NEO_TRIM_PREFILTER_STATUS_OK_V1;
}

}  // namespace neoethos::resident_trim_prefilter_v1
