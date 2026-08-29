#pragma once

#include "resident_generation_v2_abi.cuh"
#include "resident_scoring_novelty_v1_abi.cuh"
#include "resident_scoring_novelty_v2_internal.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

// Native-only composition seam. This header is consumed by CUDA translation
// units inside neoethos-gpu-cuda; none of these types or functions is part of
// the Rust FFI surface.
namespace neoethos::resident_generation_v2_internal {

/// Same-stream scored rows retained by the composite Search owner. The sealed
/// scoring output supplies metrics, scenarios, device-fault identity and
/// semantics; the ranked decision keys may be replaced by Slice2's blended
/// score without copying either payload through the host.
struct ResidentGenerationScoredRowsV2 {
  const resident_scoring_novelty_v1::NeoResidentScoredDecisionRowsV1*
      sealed_scoring_rows;
  const std::uint64_t* ranked_decision_keys_device;
  resident_generation_v2::NeoResidentGenerationGeneViewV2*
      retained_generation_view;
};

struct ResidentGenerationDevicePublishResultV2 {
  std::uint32_t scoring_fault;
  std::uint32_t generation_fault;
  std::uint32_t upstream_fault;
  std::uint32_t combined_fault;
  std::uint32_t committed;
  std::uint32_t current_store_index;
  std::uint64_t generation_index;
  std::uint64_t store_epoch;
};

class ResidentGenerationPreparedAdvanceV2;
class ResidentGenerationTerminalLifecycleV2;

std::int32_t enqueue_resident_generation_offspring_from_scored_rows_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const ResidentGenerationScoredRowsV2* scored_rows,
    ResidentGenerationPreparedAdvanceV2* prepared);

std::int32_t enqueue_resident_generation_offspring_from_finite_rows_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* generation,
    const resident_scoring_novelty_v2_internal::ResidentScoringFiniteObjectiveRowsV2*
        finite_rows,
    const std::uint64_t* ranked_decision_keys_device,
    resident_generation_v2::NeoResidentGenerationGeneViewV2*
        retained_generation_view,
    ResidentGenerationPreparedAdvanceV2* prepared);

/// Acknowledges the caller's immediately preceding combined publish-kernel
/// enqueue and advances only host-side pointer/tuple bookkeeping. It performs
/// no transfer, event operation or synchronization.
std::int32_t accept_resident_generation_combined_publish_v2(
    const ResidentGenerationPreparedAdvanceV2* prepared);

bool borrow_resident_generation_terminal_lifecycle_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    std::uint64_t expected_terminal_host_receipt_bytes,
    ResidentGenerationTerminalLifecycleV2* lifecycle);

bool accept_resident_generation_terminal_enqueue_v2(
    const ResidentGenerationTerminalLifecycleV2* lifecycle,
    std::uint64_t final_same_stream_enqueue_count);

class ResidentGenerationTerminalLifecycleV2 {
 public:
  ResidentGenerationTerminalLifecycleV2() = default;

  resident_generation_v1::NeoResidentGenerationRunV1* generation_owner_v2()
      const {
    return generation_owner_;
  }
  void* population_lifetime_owner_v2() const {
    return population_lifetime_owner_;
  }
  cudaStream_t admitted_run_stream_v2() const { return admitted_run_stream_; }
  cudaEvent_t completion_event_v2() const { return completion_event_; }
  void* terminal_host_receipt_v2() const { return terminal_host_receipt_; }
  std::uint64_t terminal_host_receipt_bytes_v2() const {
    return terminal_host_receipt_bytes_;
  }
  std::uint64_t completion_event_identity_v2() const {
    return completion_event_identity_;
  }
  const resident_generation_v1::NeoResidentGenerationReadyEventV1*
  source_ready_receipt_v2() const {
    return source_ready_receipt_;
  }
  cudaEvent_t resident_parent_ready_event_v2() const {
    return resident_parent_ready_event_;
  }
  std::uint64_t source_event_id_v2() const { return source_event_id_; }
  std::uint64_t source_same_stream_enqueue_count_v2() const {
    return source_same_stream_enqueue_count_;
  }
  std::uint64_t run_token_v2() const { return run_token_; }
  std::uint64_t generation_index_v2() const { return generation_index_; }
  std::uint64_t store_epoch_v2() const { return store_epoch_; }
  std::uint32_t current_store_index_v2() const {
    return current_store_index_;
  }
  std::uint64_t same_stream_enqueue_count_v2() const {
    return same_stream_enqueue_count_;
  }

 private:
  resident_generation_v1::NeoResidentGenerationRunV1* generation_owner_;
  void* population_lifetime_owner_;
  cudaStream_t admitted_run_stream_;
  cudaEvent_t completion_event_;
  void* terminal_host_receipt_;
  std::uint64_t terminal_host_receipt_bytes_;
  std::uint64_t completion_event_identity_;
  const resident_generation_v1::NeoResidentGenerationReadyEventV1*
      source_ready_receipt_;
  cudaEvent_t resident_parent_ready_event_;
  std::uint64_t source_event_id_;
  std::uint64_t source_same_stream_enqueue_count_;
  std::uint64_t run_token_;
  std::uint64_t generation_index_;
  std::uint64_t store_epoch_;
  std::uint64_t same_stream_enqueue_count_;
  std::uint32_t current_store_index_;
  std::uint32_t reserved_;

  friend bool borrow_resident_generation_terminal_lifecycle_v2(
      resident_generation_v1::NeoResidentGenerationRunV1*, std::uint64_t,
      ResidentGenerationTerminalLifecycleV2*);
};

/// Non-mintable outside the native generation implementation. The composite
/// archive owner may copy this value directly into its one-thread final commit
/// kernel and invoke publish_device_v2 before publishing its packed commit
/// word. It cannot detach any of the proof's pointer or tuple fields.
class ResidentGenerationPreparedAdvanceV2 {
 public:
  ResidentGenerationPreparedAdvanceV2() = default;

  __device__ ResidentGenerationDevicePublishResultV2 publish_device_v2(
      std::uint32_t upstream_fault) const {
    ResidentGenerationDevicePublishResultV2 result{};
    result.upstream_fault = upstream_fault;

    auto* seal = device_seal_identity_;
    auto* control = device_control_;
    const bool exact_device_identity =
        seal != nullptr && control != nullptr &&
        device_content_fault_ != nullptr &&
        gene_hash_collision_fault_ != nullptr &&
        seal->abi_version ==
            resident_generation_v2::NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2 &&
        seal->run_token == run_token_ &&
        seal->generation_index == expected_old_generation_index_ &&
        seal->store_epoch == expected_old_store_epoch_ &&
        seal->current_store_index == expected_old_store_index_;
    const bool packed_commit_bounds_v2 =
        expected_next_generation_index_ <= 0xffffull &&
        expected_next_store_epoch_ <= 0x7fffffffull;

    result.scoring_fault =
        scoring_device_seal_ == nullptr ||
                scoring_device_seal_->abi_version != 1u ||
                scoring_device_seal_->valid == 0u
            ? 1u
            : scoring_device_seal_->device_fault_word;
    result.generation_fault =
        !exact_device_identity || !packed_commit_bounds_v2
            ? 1u
            : (*device_content_fault_ != 0u
                   ? *device_content_fault_
                   : *gene_hash_collision_fault_);
    result.combined_fault =
        result.upstream_fault != 0u
            ? result.upstream_fault
            : (result.scoring_fault != 0u ? result.scoring_fault
                                          : result.generation_fault);

    if (!packed_commit_bounds_v2) {
      return result;
    }
    if (seal == nullptr || control == nullptr) {
      return result;
    }
    result.current_store_index = seal->current_store_index;
    result.generation_index = seal->generation_index;
    result.store_epoch = seal->store_epoch;
    if (result.combined_fault != 0u) {
      seal->fault_code = result.combined_fault;
      seal->flags |=
          resident_generation_v2::NEO_RESIDENT_GENERATION_SEAL_POISONED_V2;
      control->fault_word = result.combined_fault;
      control->stop_requested = 1;
      return result;
    }

    seal->current_store_index = expected_next_store_index_;
    seal->generation_index = expected_next_generation_index_;
    seal->store_epoch = expected_next_store_epoch_;
    control->fault_word = 0;
    control->generation_index = expected_next_generation_index_;
    control->executed_generations = expected_next_generation_index_;
    control->current_store_index = expected_next_store_index_;
    result.committed = 1u;
    result.current_store_index = expected_next_store_index_;
    result.generation_index = expected_next_generation_index_;
    result.store_epoch = expected_next_store_epoch_;
    return result;
  }

 private:
  resident_generation_v1::NeoResidentGenerationRunV1* generation_owner_;
  cudaStream_t admitted_run_stream_;
  resident_generation_v2::NeoResidentGenerationDeviceSealV2*
      device_seal_identity_;
  resident_generation_v2::NeoResidentSearchDeviceControlV2* device_control_;
  const resident_scoring_novelty_v1::NeoResidentScoringNoveltyDeviceSealV1*
      scoring_device_seal_;
  resident_generation_v2::NeoResidentGenerationGeneViewV2*
      retained_generation_view_;
  const std::uint32_t* device_content_fault_;
  const std::uint32_t* gene_hash_collision_fault_;
  std::uint64_t expected_old_generation_index_;
  std::uint64_t expected_next_generation_index_;
  std::uint64_t expected_old_store_epoch_;
  std::uint64_t expected_next_store_epoch_;
  std::uint64_t run_token_;
  std::uint64_t same_stream_enqueue_count_;
  std::uint32_t expected_old_store_index_;
  std::uint32_t expected_next_store_index_;

  friend std::int32_t
  enqueue_resident_generation_offspring_from_scored_rows_v2(
      resident_generation_v1::NeoResidentGenerationRunV1*,
      const ResidentGenerationScoredRowsV2*,
      ResidentGenerationPreparedAdvanceV2*);
  friend std::int32_t
  enqueue_resident_generation_offspring_from_finite_rows_v2(
      resident_generation_v1::NeoResidentGenerationRunV1*,
      const resident_scoring_novelty_v2_internal::ResidentScoringFiniteObjectiveRowsV2*,
      const std::uint64_t*,
      resident_generation_v2::NeoResidentGenerationGeneViewV2*,
      ResidentGenerationPreparedAdvanceV2*);
  friend std::int32_t accept_resident_generation_combined_publish_v2(
      const ResidentGenerationPreparedAdvanceV2*);
};

static_assert(std::is_trivially_copyable_v<ResidentGenerationPreparedAdvanceV2>,
              "prepared generation proof must remain kernel-copyable");
static_assert(std::is_standard_layout_v<ResidentGenerationPreparedAdvanceV2>,
              "prepared generation proof must remain standard-layout");
static_assert(
    std::is_trivially_copyable_v<ResidentGenerationTerminalLifecycleV2>,
    "terminal lifecycle proof must remain trivially copyable");
static_assert(std::is_standard_layout_v<ResidentGenerationTerminalLifecycleV2>,
              "terminal lifecycle proof must remain standard-layout");

}  // namespace neoethos::resident_generation_v2_internal
