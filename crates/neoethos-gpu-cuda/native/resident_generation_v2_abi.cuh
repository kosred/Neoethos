#pragma once

#include "resident_generation_v1_abi.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace neoethos::resident_generation_v2 {

constexpr std::uint32_t NEO_RESIDENT_GENERATION_GENE_VIEW_ABI_V2 = 2;

constexpr std::uint32_t NEO_RESIDENT_GENERATION_SEAL_INITIALIZED_V2 = 1u << 0;
constexpr std::uint32_t NEO_RESIDENT_GENERATION_SEAL_SMC_GATE_DISABLED_V2 = 1u << 1;
constexpr std::uint32_t NEO_RESIDENT_GENERATION_SEAL_POISONED_V2 = 1u << 2;
constexpr std::int32_t NEO_RESIDENT_SEARCH_NOT_READY_V2 = 1;
constexpr std::uint32_t NEO_RESIDENT_SEARCH_TERMINAL_COMMITTED_V2 = 1;
constexpr std::uint32_t NEO_RESIDENT_SEARCH_TERMINAL_FAULT_V2 = 2;

/// Device-resident control plane owned by the generation run allocation.
/// Rust carries only the enclosing gene view and cannot substitute this
/// pointer independently of the run that minted it.
struct NeoResidentSearchDeviceControlV2 {
  std::uint32_t abi_version;
  std::uint32_t fault_word;
  std::uint64_t generation_index;
  std::uint64_t executed_generations;
  std::uint64_t stagnant_generations;
  std::uint64_t best_score_order_key;
  std::uint64_t gate_threshold_bits;
  std::uint64_t archive_count;
  std::uint64_t stop_requested;
  std::uint64_t current_store_index;
  std::uint64_t reserved;
};

/// Device-only authority paired with every exported gene view. The evaluator
/// resolves the active data pointers from this seal after its event wait; Rust
/// treats the type as opaque and cannot detach a pointer triplet from the swap.
struct alignas(16) NeoResidentGenerationDeviceSealV2 {
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint32_t fault_code;
  std::uint32_t current_store_index;
  std::uint64_t generation_index;
  std::uint64_t store_epoch;
  std::uint64_t logical_population_count;
  std::uint32_t max_terms_per_gene;
  std::uint32_t smc_flag_count;
  std::uint64_t run_token;
  std::uint64_t feature_count;
  resident_generation_v1::NeoResidentGenerationGeneScalarV1* scalar_store[2];
  std::uint64_t* term_index_store[2];
  double* term_weight_store[2];
  const double* smc_weights;
  std::uint64_t gate_threshold_bits;
  std::uint8_t plan_identity_sha256[32];
};

/// Borrowed host description of a device-only seal. No gene pointer is repeated
/// here: doing so would let a host-side view race a device-side store swap.
struct NeoResidentGenerationGeneViewV2 {
  std::uint32_t abi_version;
  std::uint32_t flags;
  const NeoResidentGenerationDeviceSealV2* seal_device;
  const NeoResidentSearchDeviceControlV2* control_device;
  std::uint64_t expected_generation_index;
  std::uint64_t expected_store_epoch;
  std::uint64_t expected_run_token;
  std::uint64_t logical_population_count;
  std::uint64_t feature_count;
  std::uint32_t max_terms_per_gene;
  std::uint32_t smc_flag_count;
  std::uint8_t plan_identity_sha256[32];
  std::uint8_t generation_semantics_sha256[32];
};

/// Compact device-authoritative terminal record. The admitted stream copies
/// exactly this bounded record to run-owned pinned host storage before it
/// records the completion event; no metric or gene storage crosses the host.
struct NeoResidentSearchTerminalReceiptV2 {
  std::uint32_t abi_version;
  std::uint32_t terminal_status;
  std::uint32_t scoring_device_fault;
  std::uint32_t generation_device_fault;
  std::uint32_t control_fault_word;
  std::uint32_t stop_requested;
  std::uint32_t current_store_index;
  std::uint32_t reserved;
  std::uint64_t generation_index;
  std::uint64_t store_epoch;
  std::uint64_t run_token;
  std::uint64_t compact_async_d2h_count;
  std::uint64_t compact_async_d2h_bytes;
  std::uint64_t completion_event_query_count;
  std::uint64_t completion_stream_synchronize_count;
  std::uint64_t same_stream_enqueue_count;
  std::uint64_t completion_event_id;
};

/// Move-only host receipt for an enqueued but not yet committed generation.
/// Both pointers are exact-address authority retained by the Rust owner.
struct NeoResidentSearchAdvancePendingReceiptV2 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t completion_event_id;
  std::uint64_t target_generation_index;
  std::uint64_t target_store_epoch;
  std::uint64_t target_store_index;
  std::uint64_t run_token;
  std::uint64_t same_stream_enqueue_count;
  const resident_generation_v1::NeoResidentGenerationReadyEventV1*
      dependency_receipt_token;
  const NeoResidentSearchTerminalReceiptV2* terminal_host_receipt_token;
};

static_assert(sizeof(NeoResidentGenerationDeviceSealV2) == 160,
              "resident generation V2 device seal ABI changed");
static_assert(alignof(NeoResidentGenerationDeviceSealV2) == 16,
              "resident generation V2 device seal alignment changed");
static_assert(sizeof(NeoResidentSearchDeviceControlV2) == 80,
              "resident Search V2 device control ABI changed");
static_assert(sizeof(NeoResidentGenerationGeneViewV2) == 136,
              "resident generation V2 gene view ABI changed");
static_assert(sizeof(NeoResidentSearchTerminalReceiptV2) == 104,
              "resident Search V2 terminal receipt ABI changed");
static_assert(sizeof(NeoResidentSearchAdvancePendingReceiptV2) == 72,
              "resident Search V2 pending receipt ABI changed");

extern "C" std::int32_t export_current_resident_gene_view_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1* ready,
    NeoResidentGenerationGeneViewV2* view);

extern "C" std::int32_t configure_resident_generation_evaluator_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const resident_generation_v1::NeoResidentGenerationReadyEventV1* dependency,
    const double* smc_weights,
    std::uint32_t smc_gate_disabled,
    resident_generation_v1::NeoResidentGenerationReadyEventV1* ready);

extern "C" std::int32_t validate_resident_gene_view_owner_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const NeoResidentGenerationGeneViewV2* view);

extern "C" std::int32_t bind_resident_search_terminal_receipt_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    NeoResidentSearchTerminalReceiptV2* pinned_host_receipt);

extern "C" std::int32_t try_complete_resident_generation_advance_v2(
    resident_generation_v1::NeoResidentGenerationRunV1* run,
    const NeoResidentSearchAdvancePendingReceiptV2* pending,
    resident_generation_v1::NeoResidentGenerationReadyEventV1* committed_ready,
    NeoResidentSearchTerminalReceiptV2* terminal_copy);

}  // namespace neoethos::resident_generation_v2
