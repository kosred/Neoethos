#include "resident_scoring_novelty_v1_abi.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace neoethos::resident_scoring_novelty_v1 {
namespace {

constexpr std::size_t DEVICE_ALIGNMENT_V1 = 256;
constexpr std::size_t CONTROL_BYTES_RAW_V1 = 80;
constexpr std::uint64_t FNV_OFFSET_V1 = 14695981039346656037ull;
constexpr std::uint64_t FNV_PRIME_V1 = 1099511628211ull;

struct PhysicalLayoutV1 {
  std::size_t feature_word_count;
  std::size_t set_bitmap_bytes;
  std::size_t fitness_score_bytes;
  std::size_t novelty_score_bytes;
  std::size_t decision_key_bytes;
  std::size_t cub_scratch_bytes;
  std::size_t device_control_bytes;
  std::size_t total_device_bytes;
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
  std::size_t expanded = 0;
  if (aligned == nullptr || !checked_add_v1(bytes, DEVICE_ALIGNMENT_V1 - 1, &expanded)) {
    return false;
  }
  *aligned = expanded & ~(DEVICE_ALIGNMENT_V1 - 1);
  return true;
}

std::uint8_t* take_device_region_v1(DeviceCursorV1* cursor, std::size_t bytes) {
  std::size_t end = 0;
  if (cursor == nullptr || !checked_add_v1(cursor->offset, bytes, &end) ||
      end > cursor->capacity) {
    return nullptr;
  }
  std::uint8_t* result = cursor->base + cursor->offset;
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

__host__ __device__ double f64_from_bits_v1(std::uint64_t bits) {
  union {
    std::uint64_t bits;
    double value;
  } raw{};
  raw.bits = bits;
  return raw.value;
}

__device__ std::uint64_t f64_bits_v1(double value) {
  return static_cast<std::uint64_t>(__double_as_longlong(value));
}

bool checked_feature_word_count_v1(std::uint64_t feature_count,
                                   std::size_t* feature_word_count) {
  std::uint64_t expanded = 0;
  if (feature_word_count == nullptr || feature_count == 0 ||
      !checked_add_v1(feature_count, std::uint64_t{63}, &expanded) ||
      expanded / 64ull > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  *feature_word_count = static_cast<std::size_t>(expanded / 64ull);
  return *feature_word_count != 0;
}

bool validate_import_v1(const NeoResidentScoringNoveltyPopulationImportV1* import) {
  return import != nullptr &&
         import->abi_version == NEO_RESIDENT_SCORING_NOVELTY_ABI_V1 &&
         import->selected_cuda_ordinal != std::numeric_limits<std::uint32_t>::max() &&
         import->admitted_run_stream != nullptr && import->metrics_ready_event != nullptr &&
         import->scoring_novelty_ready_event != nullptr &&
         import->metrics_ready_event != import->scoring_novelty_ready_event &&
         import->population_lifetime_owner != nullptr && import->metric_rows_device != nullptr &&
         import->gene_scalars_device != nullptr && import->gene_indices_device != nullptr &&
         import->expected_scenario_ids_device != nullptr &&
         import->logical_population_count != 0 && import->feature_count != 0 &&
         import->max_terms_per_gene != 0 &&
         import->max_terms_per_gene <= import->feature_count &&
         import->full_discovery_reserve_bytes != 0 &&
         all_identity_bytes_present_v1(import->cuda_device_identity_sha256) &&
         all_identity_bytes_present_v1(import->primary_context_identity_sha256) &&
         all_identity_bytes_present_v1(import->run_stream_identity_sha256) &&
         all_identity_bytes_present_v1(import->metric_semantics_sha256) &&
         all_identity_bytes_present_v1(import->gene_schema_sha256) &&
         all_identity_bytes_present_v1(import->scenario_order_semantics_sha256) &&
         all_identity_bytes_present_v1(import->cuda_build_manifest_sha256) &&
         all_identity_bytes_present_v1(import->cuda_math_flags_sha256) &&
         all_identity_bytes_present_v1(import->resident_input_content_sha256) &&
         all_identity_bytes_present_v1(import->gene_content_sha256) &&
         all_identity_bytes_present_v1(import->metric_content_sha256) &&
         all_identity_bytes_present_v1(import->scenario_order_content_sha256);
}

bool validate_plan_v1(const NeoResidentScoringNoveltyPlanV1* plan) {
  if (plan == nullptr || plan->abi_version != NEO_RESIDENT_SCORING_NOVELTY_ABI_V1 ||
      plan->scoring_version != NEO_RESIDENT_SCORING_VERSION_V1 ||
      (plan->scoring_objective != NEO_RESIDENT_SCORING_PROPFIRM_V4 &&
       plan->scoring_objective != NEO_RESIDENT_SCORING_RISKY_GROWTH_V5) ||
      plan->logical_population_count == 0 ||
      plan->logical_population_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      plan->feature_count == 0 || plan->max_terms_per_gene == 0 ||
      plan->max_terms_per_gene > plan->feature_count ||
      !all_identity_bytes_present_v1(plan->metric_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->scoring_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->novelty_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->scenario_order_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->gene_schema_sha256) ||
      !all_identity_bytes_present_v1(plan->rank_semantics_sha256) ||
      !all_identity_bytes_present_v1(plan->cuda_device_identity_sha256) ||
      !all_identity_bytes_present_v1(plan->primary_context_identity_sha256) ||
      !all_identity_bytes_present_v1(plan->run_stream_identity_sha256) ||
      !all_identity_bytes_present_v1(plan->cuda_build_manifest_sha256) ||
      !all_identity_bytes_present_v1(plan->cuda_math_flags_sha256) ||
      !all_identity_bytes_present_v1(plan->plan_identity_sha256)) {
    return false;
  }
  const double novelty_weight = f64_from_bits_v1(plan->novelty_weight_bits);
  return std::isfinite(novelty_weight) && novelty_weight >= 0.0 && novelty_weight <= 1.0;
}

std::int32_t cuda_status_v1(cudaError_t status) {
  return status == cudaSuccess ? NEO_SCORING_STATUS_OK_V1
                               : NEO_SCORING_STATUS_CUDA_ERROR_V1;
}

std::int32_t query_cub_reduce_scratch_bytes_v1(
    const NeoResidentScoringNoveltyPlanV1& plan,
    cudaStream_t stream,
    std::size_t* scratch_bytes) {
  if (scratch_bytes == nullptr) {
    return NEO_SCORING_STATUS_INVALID_ARGUMENT_V1;
  }
  const int count = static_cast<int>(plan.logical_population_count);
  auto* input = static_cast<const double*>(nullptr);
  auto* output = static_cast<double*>(nullptr);
  std::size_t candidate = 0;
  std::size_t maximum = 0;
  cudaError_t status =
      cub::DeviceReduce::Min(nullptr, candidate, input, output, count, stream);
  if (status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate;
  candidate = 0;
  status = cub::DeviceReduce::Max(nullptr, candidate, input, output, count, stream);
  if (status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUB_ERROR_V1;
  }
  maximum = candidate > maximum ? candidate : maximum;
  if (!align_device_bytes_v1(maximum, scratch_bytes)) {
    return NEO_SCORING_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  return NEO_SCORING_STATUS_OK_V1;
}

bool checked_physical_layout_v1(const NeoResidentScoringNoveltyPlanV1& plan,
                                cudaStream_t stream,
                                PhysicalLayoutV1* layout) {
  if (layout == nullptr ||
      !checked_feature_word_count_v1(plan.feature_count, &layout->feature_word_count)) {
    return false;
  }
  const std::size_t population = static_cast<std::size_t>(plan.logical_population_count);
  std::size_t bitmap_words = 0;
  std::size_t bytes = 0;
  if (!checked_mul_v1(population, layout->feature_word_count, &bitmap_words) ||
      !checked_mul_v1(bitmap_words, sizeof(std::uint64_t), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->set_bitmap_bytes) ||
      !checked_mul_v1(population, sizeof(double), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->fitness_score_bytes)) {
    return false;
  }
  layout->novelty_score_bytes = layout->fitness_score_bytes;
  if (!checked_mul_v1(population, sizeof(std::uint64_t), &bytes) ||
      !align_device_bytes_v1(bytes, &layout->decision_key_bytes) ||
      query_cub_reduce_scratch_bytes_v1(plan, stream, &layout->cub_scratch_bytes) !=
          NEO_SCORING_STATUS_OK_V1 ||
      !align_device_bytes_v1(CONTROL_BYTES_RAW_V1, &layout->device_control_bytes)) {
    return false;
  }
  std::size_t total = 0;
  const std::size_t charges[] = {
      layout->set_bitmap_bytes,   layout->fitness_score_bytes,
      layout->novelty_score_bytes, layout->decision_key_bytes,
      layout->cub_scratch_bytes,  layout->device_control_bytes,
  };
  for (const std::size_t charge : charges) {
    if (!checked_add_v1(total, charge, &total)) {
      return false;
    }
  }
  layout->total_device_bytes = total;
  return total != 0;
}

__device__ double clamp_f64_v1(double value, double lower, double upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

__device__ double trades_confidence_v1(double trades) {
  const double confidence = sqrt(trades) / 10.0;
  return confidence < 1.0 ? confidence : 1.0;
}

__device__ double sharpe_component_v1(double sharpe, double confidence) {
  return clamp_f64_v1(sharpe, -2.0, 4.0) * confidence;
}

__device__ double consistency_component_v1(double consistency) {
  return clamp_f64_v1(consistency, 0.0, 1.0);
}

__device__ double drawdown_penalty_v1(double max_drawdown) {
  const double nonnegative = max_drawdown > 0.0 ? max_drawdown : 0.0;
  const double penalty = nonnegative * 15.0;
  return penalty < 5.0 ? penalty : 5.0;
}

__device__ double ga_pf_component_v1(double profit_factor) {
  if (profit_factor >= 1.0) {
    const double positive = (profit_factor - 1.0) * 0.5;
    return positive < 1.5 ? positive : 1.5;
  }
  const double denominator = profit_factor > 0.1 ? profit_factor : 0.1;
  return -(1.0 / denominator);
}

__device__ double win_rate_component_v1(double win_rate) {
  const double normalized = clamp_f64_v1(win_rate, 0.0, 1.0);
  return clamp_f64_v1((normalized - 0.45) * 2.0, 0.0, 0.5);
}

__device__ double score_prop_firm_ga_fitness_v4(const double metrics[11]) {
  const double net = metrics[0];
  const double sharpe = metrics[1];
  const double max_drawdown = metrics[3];
  const double win_rate = metrics[4];
  const double profit_factor = metrics[5];
  const double monthly_hit = metrics[7];
  const double trades = metrics[8];
  const double consistency = metrics[9];
  const double max_daily_drawdown = metrics[10];
  if (trades < 1.0) {
    return -100.0;
  }
  const double activity = clamp_f64_v1(trades / 30.0, 0.0, 1.0);
  const double activity_multiplier = 0.3 + 0.7 * activity;
  const double confidence = trades_confidence_v1(trades);
  const double hit = clamp_f64_v1(monthly_hit, 0.0, 1.0) * 0.45;
  const double net_return = clamp_f64_v1(net / 20000.0, -2.0, 2.0) * 0.15;
  const double sharpe_score = sharpe_component_v1(sharpe, confidence) * 0.10;
  const double consistency_score = consistency_component_v1(consistency) * 0.10;
  const double profit_factor_score =
      ga_pf_component_v1(profit_factor) * (profit_factor >= 1.0 ? 0.15 : 0.25);
  const double win_rate_score = win_rate_component_v1(win_rate) * 0.10;
  const double drawdown = drawdown_penalty_v1(max_drawdown);
  const double daily_drawdown = clamp_f64_v1(max_daily_drawdown, 0.0, 1.0) * 10.0;
  return (hit + net_return + sharpe_score + consistency_score + profit_factor_score +
          win_rate_score) *
             activity_multiplier -
         drawdown - daily_drawdown;
}

__device__ double score_risky_ga_fitness_growth_v5(const double metrics[11]) {
  const double net = metrics[0];
  const double sharpe = metrics[1];
  const double win_rate = metrics[4];
  const double profit_factor = metrics[5];
  const double trades = metrics[8];
  (void)sharpe;
  if (trades < 1.0) {
    return -100.0;
  }
  const double p = clamp_f64_v1(win_rate, 0.0, 0.99);
  const double pf = clamp_f64_v1(profit_factor, 0.0, 10.0);
  const double f_star = pf > 1.0 && p > 0.0 ? p * (pf - 1.0) / pf : 0.0;
  const double f = clamp_f64_v1(f_star * 0.5, 0.0, 0.25);
  const double rr = p > 0.0 ? pf * (1.0 - p) / p : 0.0;
  const double growth_per_trade =
      f > 0.0 && rr > 0.0
          ? p * log(1.0 + rr * f) + (1.0 - p) * log(1.0 - f)
          : 0.0;
  const double growth = growth_per_trade * trades;
  const double edge_gradient = clamp_f64_v1(pf - 1.0, -1.0, 0.0) * 0.05 +
                               clamp_f64_v1(p - 0.5, -0.5, 0.0) * 0.05 +
                               clamp_f64_v1(net / 20000.0, -2.0, 0.0) * 0.01;
  return growth * 10.0 + edge_gradient;
}

__device__ bool all_metric_values_finite_v1(
    const NeoResidentScoringNoveltyMetricRowV1& row) {
  for (std::uint32_t metric = 0; metric < 11; ++metric) {
    if (!isfinite(row.values[metric])) {
      return false;
    }
  }
  return true;
}

__global__ void build_checked_gene_set_bitmap_kernel_v1(
    const NeoResidentScoringNoveltyMetricRowV1* metric_rows,
    const NeoResidentScoringNoveltyGeneScalarV1* gene_scalars,
    const std::uint64_t* gene_indices,
    const std::uint64_t* expected_scenario_ids,
    std::uint64_t* set_words,
    std::uint32_t* device_fault_word,
    NeoResidentScoringNoveltyPlanV1 plan,
    std::uint64_t feature_word_count) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= plan.logical_population_count) {
    return;
  }
  const auto& scalar = gene_scalars[candidate];
  const auto& row = metric_rows[candidate];
  if (scalar.term_count == 0 || scalar.term_count > plan.max_terms_per_gene ||
      row.candidate_id != scalar.gene_identity ||
      row.scenario_id != expected_scenario_ids[candidate]) {
    atomicExch(device_fault_word, 1u);
    return;
  }
  const std::uint64_t base = candidate * plan.max_terms_per_gene;
  for (std::uint32_t term = 0; term < scalar.term_count; ++term) {
    const std::uint64_t feature_index = gene_indices[base + term];
    if (feature_index >= plan.feature_count) {
      atomicExch(device_fault_word, 1u);
      continue;
    }
    const std::uint64_t word = feature_index / 64ull;
    const std::uint64_t bit = 1ull << (feature_index % 64ull);
    if (word >= feature_word_count) {
      atomicExch(device_fault_word, 1u);
      continue;
    }
    set_words[candidate * feature_word_count + word] |= bit;
  }
}

__global__ void score_canonical_metrics_kernel_v1(
    const NeoResidentScoringNoveltyMetricRowV1* metric_rows,
    double* fitness_scores,
    std::uint32_t* device_fault_word,
    NeoResidentScoringNoveltyPlanV1 plan) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= plan.logical_population_count) {
    return;
  }
  const auto& row = metric_rows[candidate];
  if (!all_metric_values_finite_v1(row)) {
    atomicExch(device_fault_word, 1u);
    fitness_scores[candidate] = -DBL_MAX;
    return;
  }
  const double score = plan.scoring_objective == NEO_RESIDENT_SCORING_PROPFIRM_V4
                           ? score_prop_firm_ga_fitness_v4(row.values)
                           : score_risky_ga_fitness_growth_v5(row.values);
  if (!isfinite(score)) {
    atomicExch(device_fault_word, 1u);
    fitness_scores[candidate] = -DBL_MAX;
    return;
  }
  fitness_scores[candidate] = score;
}

__global__ void candidate_ordered_mean_jaccard_kernel_v1(
    const std::uint64_t* set_words,
    double* novelty_scores,
    std::uint32_t* device_fault_word,
    NeoResidentScoringNoveltyPlanV1 plan,
    std::uint64_t feature_word_count) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= plan.logical_population_count) {
    return;
  }
  const double novelty_weight = f64_from_bits_v1(plan.novelty_weight_bits);
  if (novelty_weight <= 0.0 || plan.logical_population_count <= 1) {
    novelty_scores[candidate] = 0.0;
    return;
  }
  double dist_sum = 0.0;
  for (std::uint64_t other = 0; other < plan.logical_population_count; ++other) {
    if (candidate == other) {
      continue;
    }
    std::uint64_t intersection = 0;
    std::uint64_t union_count = 0;
    for (std::uint64_t word = 0; word < feature_word_count; ++word) {
      const std::uint64_t left = set_words[candidate * feature_word_count + word];
      const std::uint64_t right = set_words[other * feature_word_count + word];
      intersection += static_cast<std::uint64_t>(__popcll(left & right));
      union_count += static_cast<std::uint64_t>(__popcll(left | right));
    }
    if (union_count == 0) {
      atomicExch(device_fault_word, 1u);
      continue;
    }
    dist_sum += 1.0 - static_cast<double>(intersection) /
                          static_cast<double>(union_count);
  }
  const double novelty =
      dist_sum / static_cast<double>(plan.logical_population_count - 1);
  if (!isfinite(novelty)) {
    atomicExch(device_fault_word, 1u);
    novelty_scores[candidate] = 0.0;
    return;
  }
  novelty_scores[candidate] = novelty;
}

__device__ std::uint64_t ordered_f64_decision_key_v1(double value) {
  const double canonical = value == 0.0 ? 0.0 : value;
  const std::uint64_t bits = f64_bits_v1(canonical);
  const std::uint64_t key = (bits >> 63) == 0 ? bits ^ (1ull << 63) : ~bits;
  return key == 0 ? 1 : key;
}

__global__ void blend_and_encode_decision_keys_kernel_v1(
    const NeoResidentScoringNoveltyMetricRowV1* metric_rows,
    const double* fitness_scores,
    const double* novelty_scores,
    const double* min_fitness_device,
    const double* max_fitness_device,
    const double* max_novelty_device,
    std::uint64_t* decision_keys,
    std::uint32_t* device_fault_word,
    NeoResidentScoringNoveltyPlanV1 plan) {
  const std::uint64_t candidate =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= plan.logical_population_count) {
    return;
  }
  if (*device_fault_word != 0) {
    decision_keys[candidate] = 0;
    return;
  }
  double score = fitness_scores[candidate];
  const double novelty_weight = f64_from_bits_v1(plan.novelty_weight_bits);
  if (novelty_weight > 0.0 && plan.logical_population_count > 1) {
    const double min_fitness = *min_fitness_device;
    const double max_fitness = *max_fitness_device;
    const double max_novelty = *max_novelty_device;
    double fit_range = max_fitness - min_fitness;
    fit_range = fit_range < 1.0e-9 ? 1.0e-9 : fit_range;
    const double novelty_denominator = max_novelty < 1.0e-9 ? 1.0e-9 : max_novelty;
    const double normalized_fitness = (score - min_fitness) / fit_range;
    const double normalized_novelty = novelty_scores[candidate] / novelty_denominator;
    score = (1.0 - novelty_weight) * normalized_fitness +
            novelty_weight * normalized_novelty;
  }
  if (!isfinite(score)) {
    atomicExch(device_fault_word, 1u);
    decision_keys[candidate] = 0;
    return;
  }
  (void)metric_rows[candidate].candidate_id;
  decision_keys[candidate] = ordered_f64_decision_key_v1(score);
}

__device__ std::uint64_t hash_mix_v1(std::uint64_t hash, std::uint64_t value) {
  for (std::uint32_t byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffull;
    hash *= FNV_PRIME_V1;
  }
  return hash;
}

__global__ void seal_scoring_novelty_content_kernel_v1(
    const NeoResidentScoringNoveltyMetricRowV1* metric_rows,
    const std::uint64_t* decision_keys,
    const std::uint32_t* device_fault_word,
    NeoResidentScoringNoveltyDeviceSealV1* seal,
    NeoResidentScoringNoveltyPlanV1 plan) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  seal->abi_version = NEO_RESIDENT_SCORING_NOVELTY_ABI_V1;
  seal->valid = *device_fault_word == 0 ? 1u : 0u;
  seal->device_fault_word = *device_fault_word;
  seal->reserved = 0;
  std::uint64_t lanes[4] = {FNV_OFFSET_V1, FNV_OFFSET_V1 ^ 0x9e3779b97f4a7c15ull,
                            FNV_OFFSET_V1 ^ 0xa0761d6478bd642full,
                            FNV_OFFSET_V1 ^ 0xe7037ed1a0b428dbull};
  for (std::uint64_t candidate = 0; candidate < plan.logical_population_count;
       ++candidate) {
    const auto& row = metric_rows[candidate];
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
      lanes[lane] = hash_mix_v1(lanes[lane], row.candidate_id ^ candidate ^ lane);
      lanes[lane] = hash_mix_v1(lanes[lane], row.scenario_id ^ lane);
      lanes[lane] = hash_mix_v1(lanes[lane], decision_keys[candidate] ^ lane);
    }
    for (std::uint32_t metric = 0; metric < 11; ++metric) {
      for (std::uint32_t lane = 0; lane < 4; ++lane) {
        lanes[lane] = hash_mix_v1(lanes[lane], f64_bits_v1(row.values[metric]) ^ lane);
      }
    }
  }
  for (std::uint32_t identity_index = 0; identity_index < 32; ++identity_index) {
    const std::uint64_t bound =
        static_cast<std::uint64_t>(plan.metric_semantics_sha256[identity_index]) |
        (static_cast<std::uint64_t>(plan.scoring_semantics_sha256[identity_index]) << 8) |
        (static_cast<std::uint64_t>(plan.novelty_semantics_sha256[identity_index]) << 16) |
        (static_cast<std::uint64_t>(plan.rank_semantics_sha256[identity_index]) << 24) |
        (static_cast<std::uint64_t>(plan.cuda_build_manifest_sha256[identity_index]) << 32) |
        (static_cast<std::uint64_t>(plan.cuda_math_flags_sha256[identity_index]) << 40);
    const std::uint64_t execution_and_schema =
        static_cast<std::uint64_t>(plan.cuda_device_identity_sha256[identity_index]) |
        (static_cast<std::uint64_t>(plan.primary_context_identity_sha256[identity_index]) << 8) |
        (static_cast<std::uint64_t>(plan.run_stream_identity_sha256[identity_index]) << 16) |
        (static_cast<std::uint64_t>(plan.scenario_order_semantics_sha256[identity_index]) << 24) |
        (static_cast<std::uint64_t>(plan.gene_schema_sha256[identity_index]) << 32);
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
      lanes[lane] = hash_mix_v1(lanes[lane], bound ^ lane);
      lanes[lane] = hash_mix_v1(lanes[lane], execution_and_schema ^ lane);
    }
  }
  for (std::uint32_t lane = 0; lane < 4; ++lane) {
    seal->content_lanes[lane] = lanes[lane];
  }
  // An opaque device seal must be checked by the same-stream consumer.
  if (seal->valid == 0) {
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
      seal->content_lanes[lane] = 0;
    }
  }
}

std::uint32_t grid_for_v1(std::uint64_t count) {
  constexpr std::uint64_t threads = 256;
  return static_cast<std::uint32_t>((count + threads - 1) / threads);
}

std::int32_t launch_status_v1() {
  return cuda_status_v1(cudaPeekAtLastError());
}

}  // namespace

struct NeoResidentScoringNoveltyRunV1 {
  NeoResidentScoringNoveltyPlanV1 plan;
  NeoResidentScoringNoveltyAllocationReceiptV1 allocation;
  cudaStream_t admitted_run_stream;
  cudaEvent_t metrics_ready_event;
  cudaEvent_t scoring_novelty_ready_event;
  void* population_lifetime_owner;
  const NeoResidentScoringNoveltyMetricRowV1* metric_rows_device;
  const NeoResidentScoringNoveltyGeneScalarV1* gene_scalars_device;
  const std::uint64_t* gene_indices_device;
  const std::uint64_t* expected_scenario_ids_device;
  void* allocation_base;
  std::uint64_t* set_words_device;
  double* fitness_scores_device;
  double* novelty_scores_device;
  std::uint64_t* decision_keys_device;
  void* cub_scratch_device;
  std::uint32_t* device_fault_word;
  double* min_fitness_device;
  double* max_fitness_device;
  double* max_novelty_device;
  NeoResidentScoringNoveltyDeviceSealV1* device_seal;
  std::uint64_t feature_word_count;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t next_event_id;
  bool sealed;
};

namespace {

bool partition_allocation_v1(NeoResidentScoringNoveltyRunV1* run) {
  if (run == nullptr || run->allocation_base == nullptr) {
    return false;
  }
  DeviceCursorV1 cursor{static_cast<std::uint8_t*>(run->allocation_base), 0,
                        static_cast<std::size_t>(run->allocation.total_device_bytes)};
  run->set_words_device = reinterpret_cast<std::uint64_t*>(
      take_device_region_v1(&cursor, run->allocation.set_bitmap_bytes));
  run->fitness_scores_device = reinterpret_cast<double*>(
      take_device_region_v1(&cursor, run->allocation.fitness_score_bytes));
  run->novelty_scores_device = reinterpret_cast<double*>(
      take_device_region_v1(&cursor, run->allocation.novelty_score_bytes));
  run->decision_keys_device = reinterpret_cast<std::uint64_t*>(
      take_device_region_v1(&cursor, run->allocation.decision_key_bytes));
  run->cub_scratch_device =
      take_device_region_v1(&cursor, run->allocation.cub_scratch_bytes);
  std::uint8_t* control =
      take_device_region_v1(&cursor, run->allocation.device_control_bytes);
  if (control == nullptr) {
    return false;
  }
  run->device_fault_word = reinterpret_cast<std::uint32_t*>(control);
  run->min_fitness_device = reinterpret_cast<double*>(control + 8);
  run->max_fitness_device = reinterpret_cast<double*>(control + 16);
  run->max_novelty_device = reinterpret_cast<double*>(control + 24);
  run->device_seal =
      reinterpret_cast<NeoResidentScoringNoveltyDeviceSealV1*>(control + 32);
  return cursor.offset == cursor.capacity && run->set_words_device != nullptr &&
         run->fitness_scores_device != nullptr && run->novelty_scores_device != nullptr &&
         run->decision_keys_device != nullptr && run->cub_scratch_device != nullptr &&
         run->device_fault_word != nullptr && run->device_seal != nullptr;
}

std::int32_t record_ready_event_v1(NeoResidentScoringNoveltyRunV1* run,
                                   NeoResidentScoringNoveltyReadyEventV1* ready) {
  if (cudaEventRecord(run->scoring_novelty_ready_event, run->admitted_run_stream) !=
      cudaSuccess) {
    return NEO_SCORING_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  ++run->next_event_id;
  ready->abi_version = NEO_RESIDENT_SCORING_NOVELTY_ABI_V1;
  ready->reserved = 0;
  ready->event_id = run->next_event_id;
  ready->same_stream_enqueue_count = run->same_stream_enqueue_count;
  ready->intermediate_host_wait_count = 0;
  ready->intermediate_readback_count = 0;
  return NEO_SCORING_STATUS_OK_V1;
}

}  // namespace

extern "C" std::int32_t query_resident_scoring_novelty_allocation_v1(
    const NeoResidentScoringNoveltyPopulationImportV1* import,
    const NeoResidentScoringNoveltyPlanV1* plan,
    NeoResidentScoringNoveltyAllocationReceiptV1* receipt) {
  if (!validate_import_v1(import) || receipt == nullptr) {
    return NEO_SCORING_STATUS_INVALID_ARGUMENT_V1;
  }
  if (plan == nullptr || plan->abi_version != NEO_RESIDENT_SCORING_NOVELTY_ABI_V1) {
    return NEO_SCORING_STATUS_ABI_MISMATCH_V1;
  }
  if (!validate_plan_v1(plan) ||
      import->logical_population_count != plan->logical_population_count ||
      import->feature_count != plan->feature_count ||
      import->max_terms_per_gene != plan->max_terms_per_gene ||
      !identity_equal_v1(import->cuda_device_identity_sha256,
                         plan->cuda_device_identity_sha256) ||
      !identity_equal_v1(import->primary_context_identity_sha256,
                         plan->primary_context_identity_sha256) ||
      !identity_equal_v1(import->run_stream_identity_sha256,
                         plan->run_stream_identity_sha256) ||
      !identity_equal_v1(import->metric_semantics_sha256,
                         plan->metric_semantics_sha256) ||
      !identity_equal_v1(import->gene_schema_sha256,
                         plan->gene_schema_sha256) ||
      !identity_equal_v1(import->scenario_order_semantics_sha256,
                         plan->scenario_order_semantics_sha256) ||
      !identity_equal_v1(import->cuda_build_manifest_sha256,
                         plan->cuda_build_manifest_sha256) ||
      !identity_equal_v1(import->cuda_math_flags_sha256,
                         plan->cuda_math_flags_sha256)) {
    return NEO_SCORING_STATUS_IDENTITY_MISMATCH_V1;
  }
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0 ||
      static_cast<std::uint32_t>(current_device) != import->selected_cuda_ordinal) {
    return NEO_SCORING_STATUS_IDENTITY_MISMATCH_V1;
  }
  PhysicalLayoutV1 layout{};
  if (!checked_physical_layout_v1(*plan, import->admitted_run_stream, &layout)) {
    return NEO_SCORING_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  std::size_t same_context_free_bytes = 0;
  std::size_t same_context_total_bytes = 0;
  if (cudaMemGetInfo(&same_context_free_bytes, &same_context_total_bytes) != cudaSuccess) {
    return NEO_SCORING_STATUS_CUDA_ERROR_V1;
  }
  (void)same_context_total_bytes;
  if (import->full_discovery_reserve_bytes > same_context_free_bytes ||
      layout.total_device_bytes >
          same_context_free_bytes -
              static_cast<std::size_t>(import->full_discovery_reserve_bytes)) {
    return NEO_SCORING_STATUS_OUT_OF_MEMORY_V1;
  }
  receipt->abi_version = NEO_RESIDENT_SCORING_NOVELTY_ABI_V1;
  receipt->scoring_store_allocation_count = 1;
  receipt->set_bitmap_bytes = layout.set_bitmap_bytes;
  receipt->fitness_score_bytes = layout.fitness_score_bytes;
  receipt->novelty_score_bytes = layout.novelty_score_bytes;
  receipt->decision_key_bytes = layout.decision_key_bytes;
  receipt->cub_scratch_bytes = layout.cub_scratch_bytes;
  receipt->device_control_bytes = layout.device_control_bytes;
  receipt->total_device_bytes = layout.total_device_bytes;
  receipt->same_context_free_bytes = same_context_free_bytes;
  receipt->full_discovery_reserve_bytes = import->full_discovery_reserve_bytes;
  receipt->logical_population_count = plan->logical_population_count;
  receipt->feature_word_count = layout.feature_word_count;
  copy_identity_v1(receipt->allocation_plan_sha256, plan->plan_identity_sha256);
  return NEO_SCORING_STATUS_OK_V1;
}

extern "C" std::int32_t create_resident_scoring_novelty_run_v1(
    const NeoResidentScoringNoveltyPopulationImportV1* import,
    const NeoResidentScoringNoveltyPlanV1* plan,
    const NeoResidentScoringNoveltyAllocationReceiptV1* receipt,
    NeoResidentScoringNoveltyRunV1** run) {
  if (!validate_import_v1(import) || !validate_plan_v1(plan) || receipt == nullptr ||
      run == nullptr || *run != nullptr ||
      receipt->abi_version != NEO_RESIDENT_SCORING_NOVELTY_ABI_V1 ||
      receipt->scoring_store_allocation_count != 1 ||
      receipt->logical_population_count != plan->logical_population_count ||
      receipt->full_discovery_reserve_bytes != import->full_discovery_reserve_bytes ||
      !identity_equal_v1(import->cuda_device_identity_sha256,
                         plan->cuda_device_identity_sha256) ||
      !identity_equal_v1(import->primary_context_identity_sha256,
                         plan->primary_context_identity_sha256) ||
      !identity_equal_v1(import->run_stream_identity_sha256,
                         plan->run_stream_identity_sha256) ||
      !identity_equal_v1(import->metric_semantics_sha256,
                         plan->metric_semantics_sha256) ||
      !identity_equal_v1(import->gene_schema_sha256,
                         plan->gene_schema_sha256) ||
      !identity_equal_v1(import->scenario_order_semantics_sha256,
                         plan->scenario_order_semantics_sha256) ||
      !identity_equal_v1(import->cuda_build_manifest_sha256,
                         plan->cuda_build_manifest_sha256) ||
      !identity_equal_v1(import->cuda_math_flags_sha256,
                         plan->cuda_math_flags_sha256) ||
      !identity_equal_v1(receipt->allocation_plan_sha256, plan->plan_identity_sha256)) {
    return NEO_SCORING_STATUS_IDENTITY_MISMATCH_V1;
  }
  std::size_t current_free = 0;
  std::size_t current_total = 0;
  if (cudaMemGetInfo(&current_free, &current_total) != cudaSuccess) {
    return NEO_SCORING_STATUS_CUDA_ERROR_V1;
  }
  (void)current_total;
  if (receipt->full_discovery_reserve_bytes > current_free ||
      receipt->total_device_bytes > current_free - receipt->full_discovery_reserve_bytes) {
    return NEO_SCORING_STATUS_OUT_OF_MEMORY_V1;
  }
  auto* created = new (std::nothrow) NeoResidentScoringNoveltyRunV1{};
  if (created == nullptr) {
    return NEO_SCORING_STATUS_OUT_OF_MEMORY_V1;
  }
  created->plan = *plan;
  created->allocation = *receipt;
  created->admitted_run_stream = import->admitted_run_stream;
  created->metrics_ready_event = import->metrics_ready_event;
  created->scoring_novelty_ready_event = import->scoring_novelty_ready_event;
  created->population_lifetime_owner = import->population_lifetime_owner;
  created->metric_rows_device = import->metric_rows_device;
  created->gene_scalars_device = import->gene_scalars_device;
  created->gene_indices_device = import->gene_indices_device;
  created->expected_scenario_ids_device = import->expected_scenario_ids_device;
  created->feature_word_count = receipt->feature_word_count;
  created->same_stream_enqueue_count = 0;
  created->next_event_id = 0;
  created->sealed = false;
  cudaError_t status = cudaMallocAsync(&created->allocation_base,
                                       static_cast<std::size_t>(receipt->total_device_bytes),
                                       created->admitted_run_stream);
  if (status != cudaSuccess) {
    delete created;
    return NEO_SCORING_STATUS_OUT_OF_MEMORY_V1;
  }
  if (!partition_allocation_v1(created)) {
    const cudaError_t release_status =
        cudaFreeAsync(created->allocation_base, created->admitted_run_stream);
    if (release_status == cudaSuccess) {
      delete created;
    }
    return NEO_SCORING_STATUS_ARITHMETIC_OVERFLOW_V1;
  }
  status = cudaStreamWaitEvent(created->admitted_run_stream,
                               created->metrics_ready_event, 0);
  if (status != cudaSuccess) {
    const cudaError_t release_status =
        cudaFreeAsync(created->allocation_base, created->admitted_run_stream);
    if (release_status == cudaSuccess) {
      delete created;
    }
    return NEO_SCORING_STATUS_CUDA_ERROR_V1;
  }
  created->same_stream_enqueue_count = 1;
  *run = created;
  return NEO_SCORING_STATUS_OK_V1;
}

extern "C" std::int32_t enqueue_and_seal_resident_scoring_novelty_v1(
    NeoResidentScoringNoveltyRunV1* run,
    NeoResidentScoredDecisionRowsV1* output,
    NeoResidentScoringNoveltyReadyEventV1* ready) {
  if (run == nullptr || output == nullptr || ready == nullptr || run->sealed) {
    return NEO_SCORING_STATUS_STATE_ERROR_V1;
  }
  cudaError_t cuda_status = cudaMemsetAsync(
      run->allocation_base, 0, static_cast<std::size_t>(run->allocation.total_device_bytes),
      run->admitted_run_stream);
  if (cuda_status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUDA_ERROR_V1;
  }
  ++run->same_stream_enqueue_count;
  constexpr std::uint32_t threads = 256;
  const std::uint32_t grid = grid_for_v1(run->plan.logical_population_count);
  build_checked_gene_set_bitmap_kernel_v1<<<grid, threads, 0, run->admitted_run_stream>>>(
      run->metric_rows_device, run->gene_scalars_device, run->gene_indices_device,
      run->expected_scenario_ids_device, run->set_words_device,
      run->device_fault_word, run->plan, run->feature_word_count);
  ++run->same_stream_enqueue_count;
  std::int32_t status = launch_status_v1();
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  score_canonical_metrics_kernel_v1<<<grid, threads, 0, run->admitted_run_stream>>>(
      run->metric_rows_device, run->fitness_scores_device,
      run->device_fault_word, run->plan);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  candidate_ordered_mean_jaccard_kernel_v1<<<grid, threads, 0,
                                              run->admitted_run_stream>>>(
      run->set_words_device, run->novelty_scores_device,
      run->device_fault_word, run->plan, run->feature_word_count);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  const int count = static_cast<int>(run->plan.logical_population_count);
  std::size_t scratch = static_cast<std::size_t>(run->allocation.cub_scratch_bytes);
  cuda_status = cub::DeviceReduce::Min(
      run->cub_scratch_device, scratch, run->fitness_scores_device,
      run->min_fitness_device, count, run->admitted_run_stream);
  ++run->same_stream_enqueue_count;
  if (cuda_status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUB_ERROR_V1;
  }
  scratch = static_cast<std::size_t>(run->allocation.cub_scratch_bytes);
  cuda_status = cub::DeviceReduce::Max(
      run->cub_scratch_device, scratch, run->fitness_scores_device,
      run->max_fitness_device, count, run->admitted_run_stream);
  ++run->same_stream_enqueue_count;
  if (cuda_status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUB_ERROR_V1;
  }
  scratch = static_cast<std::size_t>(run->allocation.cub_scratch_bytes);
  cuda_status = cub::DeviceReduce::Max(
      run->cub_scratch_device, scratch, run->novelty_scores_device,
      run->max_novelty_device, count, run->admitted_run_stream);
  ++run->same_stream_enqueue_count;
  if (cuda_status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUB_ERROR_V1;
  }
  blend_and_encode_decision_keys_kernel_v1<<<grid, threads, 0,
                                              run->admitted_run_stream>>>(
      run->metric_rows_device, run->fitness_scores_device,
      run->novelty_scores_device, run->min_fitness_device,
      run->max_fitness_device, run->max_novelty_device,
      run->decision_keys_device, run->device_fault_word, run->plan);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  seal_scoring_novelty_content_kernel_v1<<<1, 1, 0, run->admitted_run_stream>>>(
      run->metric_rows_device, run->decision_keys_device,
      run->device_fault_word, run->device_seal, run->plan);
  ++run->same_stream_enqueue_count;
  status = launch_status_v1();
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  run->sealed = true;
  status = record_ready_event_v1(run, ready);
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  output->abi_version = NEO_RESIDENT_SCORING_NOVELTY_ABI_V1;
  output->reserved = 0;
  output->metric_rows_device = run->metric_rows_device;
  output->resident_decision_keys_device = run->decision_keys_device;
  output->expected_scenario_ids_device = run->expected_scenario_ids_device;
  output->device_seal = run->device_seal;
  output->scoring_novelty_ready_event = run->scoring_novelty_ready_event;
  output->logical_population_count = run->plan.logical_population_count;
  output->event_id = ready->event_id;
  output->same_stream_enqueue_count = ready->same_stream_enqueue_count;
  output->intermediate_host_wait_count = 0;
  output->intermediate_readback_count = 0;
  copy_identity_v1(output->metric_semantics_sha256,
                   run->plan.metric_semantics_sha256);
  copy_identity_v1(output->scoring_semantics_sha256,
                   run->plan.scoring_semantics_sha256);
  copy_identity_v1(output->novelty_semantics_sha256,
                   run->plan.novelty_semantics_sha256);
  copy_identity_v1(output->scenario_order_semantics_sha256,
                   run->plan.scenario_order_semantics_sha256);
  copy_identity_v1(output->rank_semantics_sha256,
                   run->plan.rank_semantics_sha256);
  copy_identity_v1(output->cuda_build_manifest_sha256,
                   run->plan.cuda_build_manifest_sha256);
  copy_identity_v1(output->cuda_math_flags_sha256,
                   run->plan.cuda_math_flags_sha256);
  return NEO_SCORING_STATUS_OK_V1;
}

extern "C" std::int32_t enqueue_resident_scoring_novelty_release_v1(
    NeoResidentScoringNoveltyRunV1* run) {
  if (run == nullptr || run->allocation_base == nullptr ||
      run->admitted_run_stream == nullptr) {
    return NEO_SCORING_STATUS_INVALID_ARGUMENT_V1;
  }
  const cudaError_t status =
      cudaFreeAsync(run->allocation_base, run->admitted_run_stream);
  if (status != cudaSuccess) {
    return NEO_SCORING_STATUS_CUDA_ERROR_V1;
  }
  run->allocation_base = nullptr;
  run->scoring_novelty_ready_event = nullptr;
  delete run;
  return NEO_SCORING_STATUS_OK_V1;
}

}  // namespace neoethos::resident_scoring_novelty_v1
