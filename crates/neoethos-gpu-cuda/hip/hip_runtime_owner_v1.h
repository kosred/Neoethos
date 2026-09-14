#pragma once

#include <stddef.h>
#include <stdint.h>

// This additive HIP namespace is not the CUDA V2 context identity protocol.
// Handles are nonreused native registry keys, never caller-registered pointers.
enum NeoHipRuntimeStatusV1 {
  NEO_HIP_RUNTIME_OK_V1 = 0,
  NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 = -1,
  NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 = -2,
  NEO_HIP_RUNTIME_BACKEND_ERROR_V1 = -3,
  NEO_HIP_RUNTIME_QUARANTINED_V1 = -4,
  NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1 = -5,
  NEO_HIP_RUNTIME_CAPACITY_V1 = -6,
  NEO_HIP_RUNTIME_BUSY_V1 = -7,
  NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1 = -8
};

enum NeoHipRuntimeOperationV1 {
  NEO_HIP_OP_NONE_V1 = 0,
  NEO_HIP_OP_HOST_ALLOCATION_V1 = 1,
  NEO_HIP_OP_SELECT_DEVICE_V1 = 2,
  // 3, 5 and 21 belonged to removed, deprecated HIP context operations.
  NEO_HIP_OP_CREATE_STREAM_V1 = 4,
  NEO_HIP_OP_CURRENT_DEVICE_V1 = 6,
  NEO_HIP_OP_STREAM_DEVICE_V1 = 7,
  NEO_HIP_OP_STREAM_ID_V1 = 8,
  NEO_HIP_OP_DEVICE_UUID_V1 = 9,
  NEO_HIP_OP_DEVICE_PROPERTIES_V1 = 10,
  NEO_HIP_OP_RUNTIME_VERSION_V1 = 11,
  NEO_HIP_OP_DRIVER_VERSION_V1 = 12,
  NEO_HIP_OP_MEMORY_INFO_V1 = 13,
  NEO_HIP_OP_CURRENT_POOL_V1 = 14,
  NEO_HIP_OP_DEFAULT_POOL_V1 = 15,
  NEO_HIP_OP_POOL_RESERVED_V1 = 16,
  NEO_HIP_OP_POOL_USED_V1 = 17,
  NEO_HIP_OP_IDENTITY_V1 = 18,
  NEO_HIP_OP_SYNCHRONIZE_V1 = 19,
  NEO_HIP_OP_DESTROY_STREAM_V1 = 20,
  NEO_HIP_OP_DEVICE_ALLOCATE_V1 = 22,
  NEO_HIP_OP_DEVICE_FREE_V1 = 23,
  NEO_HIP_OP_HOST_PIN_V1 = 24,
  NEO_HIP_OP_HOST_UNPIN_V1 = 25,
  NEO_HIP_OP_UPLOAD_V1 = 26,
  NEO_HIP_OP_DOWNLOAD_V1 = 27,
  NEO_HIP_OP_SESSION_LAUNCH_V1 = 28,
  NEO_HIP_OP_SMC_LAUNCH_V3 = 29,
  NEO_HIP_OP_SMC_SEMANTIC_V3 = 30,
  NEO_HIP_OP_FEATURE_PACK_V3 = 31,
  NEO_HIP_OP_FEATURE_SEMANTIC_V3 = 32,
  NEO_HIP_OP_FEATURE_MERKLE_V3 = 33,
  NEO_HIP_OP_FEATURE_NORMALIZE_V3 = 34,
  NEO_HIP_OP_NORMALIZATION_SEMANTIC_V3 = 35
};

typedef struct NeoHipRuntimeErrorV1 {
  uint32_t abi_version;
  uint32_t operation;
  int32_t backend_status;
  uint32_t reserved;
} NeoHipRuntimeErrorV1;

typedef struct NeoHipRuntimeFactsV1 {
  uint32_t abi_version;
  uint32_t backend_kind; // 2 = HIP; never interpreted as a CUDA build/context.
  uint64_t lease_id;
  int32_t device_ordinal;
  int32_t runtime_version;
  int32_t driver_version;
  uint32_t warp_size;
  uint8_t uuid[16];
  uint64_t stream_handle;  // Private borrowed representation; no ownership transfer.
  uint64_t stream_id;      // Queried from HIP, distinct from native lease_id.
  uint64_t free_memory_bytes;
  uint64_t total_memory_bytes;
  uint64_t current_pool_handle;
  uint64_t default_pool_handle;
  // Separately queried telemetry, not an atomic pair or a memory reservation.
  // Concurrent pool activity may make observed used exceed earlier reserved.
  uint64_t pool_reserved_bytes;
  uint64_t pool_used_bytes;
  uint8_t architecture[256]; // Exact zero-terminated gcnArchName, no truncation.
} NeoHipRuntimeFactsV1;

typedef struct NeoHipFeatureColumnV1 {
  uint64_t values_key;
  uint64_t values_element_offset;
  uint64_t validity_key;
  uint64_t validity_byte_offset;
} NeoHipFeatureColumnV1;

typedef struct NeoHipFeatureNormalizationV3 {
  uint64_t training_row_start;
  uint64_t training_row_end;
  const uint8_t* column_modes; // Robust=0, Binary=1, SignedState=2, SignedContinuous=3.
  uint64_t column_mode_count;
} NeoHipFeatureNormalizationV3;

typedef struct NeoHipFeatureStoreReceiptV4 {
  uint32_t abi_version;
  uint32_t backend_kind;
  uint64_t rows;
  uint64_t columns;
  uint64_t value_bytes;
  uint64_t validity_bytes;
  uint64_t transient_device_bytes;
  uint64_t metadata_upload_bytes;
  uint32_t control;
  uint32_t readback_count;
  uint64_t readback_bytes;
  uint8_t root[32];
  uint64_t normalization_training_start;
  uint64_t normalization_training_end;
  uint64_t normalization_fit_word_count;
  uint8_t fit_metadata_digest[32];
} NeoHipFeatureStoreReceiptV4;

#ifdef __cplusplus
static_assert(sizeof(NeoHipRuntimeErrorV1) == 16);
static_assert(sizeof(NeoHipRuntimeFactsV1) == 368);
static_assert(sizeof(NeoHipFeatureColumnV1) == 32);
static_assert(sizeof(NeoHipFeatureNormalizationV3) == 32);
static_assert(sizeof(NeoHipFeatureStoreReceiptV4) == 160);
static_assert(alignof(NeoHipRuntimeFactsV1) == 8);
static_assert(offsetof(NeoHipRuntimeFactsV1, stream_handle) == 48);
static_assert(offsetof(NeoHipRuntimeFactsV1, architecture) == 112);
extern "C" {
#endif

// No reset or caller-owned raw stream registration is exposed. Callers must
// prevent external reset/stream destruction throughout
// the lease. Modern HIP device/stream APIs do not enforce that external lifetime.
// All output/error pointers are required. Create zeroes outputs and publishes
// only on complete success; uncertain partial resources are quarantined inside
// the registry. Query/synchronize/close serialize per lease, not per device or
// registry. Each query selects its owned device before checking its facts. A failed live
// identity, synchronization or destruction quarantines the lease permanently:
// no later call retries an uncertain resource handle. Close succeeds only after
// live-identity revalidation and stream synchronization, then retires the ID.
int32_t neoethos_hip_runtime_lease_create_v1(
    int32_t ordinal, uint64_t* lease, NeoHipRuntimeFactsV1* facts,
    NeoHipRuntimeErrorV1* error);
int32_t neoethos_hip_runtime_lease_query_v1(
    uint64_t lease, NeoHipRuntimeFactsV1* facts, NeoHipRuntimeErrorV1* error);
int32_t neoethos_hip_runtime_lease_synchronize_v1(
    uint64_t lease, NeoHipRuntimeErrorV1* error);
int32_t neoethos_hip_runtime_lease_close_v1(
    uint64_t lease, NeoHipRuntimeErrorV1* error);

// Buffer keys are scoped to this lease. No caller-supplied device address is
// accepted. Upload copies host input into native-owned pinned staging before
// returning; staging survives until stream completion, including error paths.
int32_t neoethos_hip_runtime_buffer_create_v1(
    uint64_t lease, uint64_t bytes, const uint8_t* optional_upload,
    uint64_t* buffer, NeoHipRuntimeErrorV1* error);
int32_t neoethos_hip_runtime_buffer_free_v1(
    uint64_t lease, uint64_t buffer, NeoHipRuntimeErrorV1* error);
int32_t neoethos_hip_runtime_buffer_read_v1(
    uint64_t lease, uint64_t buffer, uint8_t* output, uint64_t bytes,
    NeoHipRuntimeErrorV1* error);
// Explicit numerical producer, not an arbitrary raw-kernel launch. The first
// six keys are OHLCV f64 then timestamps i64; final two are f64/u8 outputs.
int32_t neoethos_hip_runtime_session_f64_v2(
    uint64_t lease, uint64_t rows, const uint64_t* eight_buffers,
    NeoHipRuntimeErrorV1* error);

// Same-lease, exact-sized, distinct buffers: inputs O/H/L/C f64 and timestamp
// i64; outputs column-major f64 values (368N), u8 validity (46N), i64 months
// (8N), i64 days (8N), row-major i8 SMC slots (11N), hashes (96), error (4).
// The shared native SMC producer is unchanged. Success proves its error word
// was copied through retained pinned storage and synchronized to zero, then
// its 96-byte hashes were copied and synchronized. Only then are all outputs
// initialized and host_hashes published. -8 denotes a completed semantic
// rejection (error.backend_status is the SMC code), NOT a HIP API failure.
// No borrowed or logically released output can be overwritten.
int32_t neoethos_hip_runtime_smc_parent_f64_v3(
    uint64_t lease, uint64_t rows, const uint64_t* five_input_buffers,
    const uint64_t* seven_output_buffers, uint8_t* host_hashes_96,
    NeoHipRuntimeErrorV1* error);

// One complete, ordered selected-column set. Inputs are initialized same-lease
// buffer ranges, not caller device pointers. Names are unique nonempty UTF-8
// spans with exact F+1 cumulative offsets. Timestamps are exact 8N bytes: this
// content/layout receipt does NOT assert chronology or canonical Data admission.
// Outputs are fresh, exact-sized, unborrowed buffers. Success follows 4B control
// and 32B root readbacks, stream completion and checked transient retirement.
// It permanently write-seals timestamps and outputs until their normal free.
// Fresh available-memory checks include the supplied reserve; not a reservation.
// Optional policy3 normalization runs between completed pack control and Merkle.
// Its actual six u64 fit words per column and separate32B digest are published
// only with the successful final receipt. Enabled:5 readbacks,72+48F bytes;
// disabled:2 readbacks,36 bytes, null fit destination and zero extra fields.
int32_t neoethos_hip_runtime_pack_feature_store_v4(
    uint64_t lease, uint64_t rows, uint64_t columns,
    const NeoHipFeatureColumnV1* descriptors, uint64_t timestamps_key,
    const uint64_t* name_offsets, const uint8_t* name_bytes, uint64_t name_bytes_len,
    uint64_t allocator_reserve_bytes, const NeoHipFeatureNormalizationV3* normalization,
    uint64_t output_values_key, uint64_t output_validity_key,
    uint64_t* host_fit_words, uint64_t host_fit_word_capacity,
    NeoHipFeatureStoreReceiptV4* receipt,
    NeoHipRuntimeErrorV1* error);

// Private native consumer ownership. Borrow takes one fresh memory snapshot;
// query revalidates device/stream/pool identity without querying dynamic memory
// counters. Its free/reserved/used fields are zero, NOT capacity authority.
// Close returns BUSY without touching resources while a borrower is live.
int32_t neoethos_hip_runtime_borrow_v1(
    uint64_t lease, uint64_t* borrower, NeoHipRuntimeFactsV1* facts,
    NeoHipRuntimeErrorV1* error);
int32_t neoethos_hip_runtime_borrow_query_v1(
    uint64_t lease, uint64_t borrower, NeoHipRuntimeFactsV1* identity,
    NeoHipRuntimeErrorV1* error);
// Resolves an initialized exact-sized key owned by this lease, and pins it for
// this borrower. No new lookup accepts a logically released buffer. Existing
// pins delay stream-ordered free until their last borrower releases ownership.
int32_t neoethos_hip_runtime_borrow_buffer_v1(
    uint64_t lease, uint64_t borrower, uint64_t buffer, uint64_t required_bytes,
    uint64_t* address, NeoHipRuntimeErrorV1* error);
// Consumer must stop submitting work before release; all consumer work and
// delayed frees use the borrowed stream. No device-wide wait/reset is implied.
int32_t neoethos_hip_runtime_borrow_release_v1(
    uint64_t lease, uint64_t borrower, NeoHipRuntimeErrorV1* error);

#ifdef __cplusplus
}
#endif
