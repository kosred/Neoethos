#include "hip_runtime_lifecycle_v1.hpp"

#if !defined(__HIP_PLATFORM_AMD__)
#error "The explicit HIP runtime owner requires the AMD HIP backend."
#endif
#include <hip/hip_runtime_api.h>
#if defined(NEOETHOS_HIP_SESSION_KERNELS_V1)
#include "resident_session_v2_abi.cuh"
#endif

#include <cstdint>
#include <cstring>
#include <exception>
#include <new>

#if defined(NEOETHOS_HIP_NATIVE_KERNELS_V1)
// The original, shared producer is compiled by official HIPIFY. This host
// declaration introduces no second numerical implementation or detached stream.
extern "C" int neoethos_resident_smc_parent_features_f64_v3(
    const double*, const double*, const double*, const double*, const int64_t*, size_t,
    double*, unsigned char*, int64_t*, int64_t*, signed char*, unsigned char*,
    unsigned int*, hipStream_t);
extern "C" int neoethos_resident_initialize_validity_u4_v3(
    unsigned char*, size_t, size_t, unsigned int*, hipStream_t);
extern "C" int neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(
    const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,
    size_t, size_t, size_t, size_t, double*, unsigned char*, unsigned int*, hipStream_t);
extern "C" int neoethos_resident_canonical_merkle_sha256_v3(
    const int64_t*, size_t, size_t, const uint64_t*, const unsigned char*,
    const double*, const unsigned char*, unsigned char*, unsigned char*,
    size_t, unsigned char*, hipStream_t);
extern "C" int neoethos_resident_robust_normalize_bar_major_f64_u4_v3(
    double*, unsigned char*, size_t, size_t, size_t, size_t, size_t, size_t,
    uint64_t*, size_t, uint64_t*, size_t, const unsigned char*, size_t,
    unsigned int*, hipStream_t);
#endif

namespace {
using namespace neoethos::hip_runtime_v1;

static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
static_assert(sizeof(size_t) == sizeof(uint64_t));
static_assert(sizeof(hipUUID) == 16);

template<class Handle>
uint64_t handle_bits(Handle value) noexcept {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
}

int32_t checked(hipError_t status, uint32_t operation,
                NeoHipRuntimeErrorV1& error) noexcept {
  if (status != hipSuccess)
    error = {1u, operation, static_cast<int32_t>(status), 0u};
  return static_cast<int32_t>(status);
}

int32_t identity_error(NeoHipRuntimeErrorV1& error) noexcept {
  error = {1u, NEO_HIP_OP_IDENTITY_V1, 0, 0u};
  return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
}

// Only real runtime calls implement this policy seam. There is no exported
// backend injection or registration of caller-owned stream pointers.
struct HipOperationsV1 {
#if defined(NEOETHOS_HIP_NATIVE_KERNELS_V1)
  int32_t normalize_feature_store(uint64_t stream, uint64_t rows, uint64_t columns,
      uint64_t values, uint64_t validity, uint64_t validity_bytes,
      const NeoHipFeatureNormalizationV3& normalization, uint64_t padded_rows,
      uint64_t sort, uint64_t sort_slots, uint64_t fits, uint64_t fit_words,
      uint64_t control, NeoHipRuntimeErrorV1& error) noexcept {
    const auto status = neoethos_resident_robust_normalize_bar_major_f64_u4_v3(
        reinterpret_cast<double*>(static_cast<uintptr_t>(values)),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(validity)),
        static_cast<size_t>(validity_bytes), static_cast<size_t>(rows), static_cast<size_t>(columns),
        static_cast<size_t>(normalization.training_row_start), static_cast<size_t>(normalization.training_row_end),
        static_cast<size_t>(padded_rows), reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(sort)),
        static_cast<size_t>(sort_slots), reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(fits)),
        static_cast<size_t>(fit_words), normalization.column_modes,
        static_cast<size_t>(normalization.column_mode_count),
        reinterpret_cast<unsigned int*>(static_cast<uintptr_t>(control)),
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream)));
    return checked(static_cast<hipError_t>(status), NEO_HIP_OP_FEATURE_NORMALIZE_V3, error);
  }

  int32_t pack_feature_store(uint64_t stream, uint64_t rows, uint64_t columns,
      uint64_t metadata, uint64_t values, uint64_t validity, uint64_t validity_bytes,
      uint64_t control, NeoHipRuntimeErrorV1& error) noexcept {
    const auto work_stream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream));
    auto* output_codes = reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(validity));
    auto* output_control = reinterpret_cast<unsigned int*>(static_cast<uintptr_t>(control));
    const uint64_t cells = rows * columns; // Checked by the resource owner.
    auto status = neoethos_resident_initialize_validity_u4_v3(output_codes,
        static_cast<size_t>(cells / 2 + (cells % 2 != 0)),
        static_cast<size_t>(validity_bytes), output_control, work_stream);
    if (status != 0)
      return checked(static_cast<hipError_t>(status), NEO_HIP_OP_FEATURE_PACK_V3, error);
    const auto* addresses = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(metadata));
    status = neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(
        addresses, addresses + columns, addresses + columns * 2, addresses + columns * 3,
        static_cast<size_t>(rows), static_cast<size_t>(columns), static_cast<size_t>(columns), 0,
        reinterpret_cast<double*>(static_cast<uintptr_t>(values)), output_codes,
        output_control, work_stream);
    return checked(static_cast<hipError_t>(status), NEO_HIP_OP_FEATURE_PACK_V3, error);
  }

  int32_t feature_store_merkle(uint64_t stream, uint64_t rows, uint64_t columns,
      uint64_t timestamps, uint64_t metadata, uint64_t values, uint64_t validity,
      uint64_t scratch_a, uint64_t scratch_b, uint64_t leaves, uint64_t root,
      NeoHipRuntimeErrorV1& error) noexcept {
    const auto* offsets = reinterpret_cast<const uint64_t*>(
        static_cast<uintptr_t>(metadata + columns * 32));
    const auto status = neoethos_resident_canonical_merkle_sha256_v3(
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(timestamps)),
        static_cast<size_t>(rows), static_cast<size_t>(columns), offsets,
        reinterpret_cast<const unsigned char*>(offsets + columns + 1),
        reinterpret_cast<const double*>(static_cast<uintptr_t>(values)),
        reinterpret_cast<const unsigned char*>(static_cast<uintptr_t>(validity)),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(scratch_a)),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(scratch_b)),
        static_cast<size_t>(leaves), reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(root)),
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream)));
    return checked(static_cast<hipError_t>(status), NEO_HIP_OP_FEATURE_MERKLE_V3, error);
  }

  int32_t smc_parent(uint64_t stream, uint64_t rows, const uint64_t* pointers,
                     NeoHipRuntimeErrorV1& error) noexcept {
    const auto status = neoethos_resident_smc_parent_features_f64_v3(
        reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[0])),
        reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[1])),
        reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[2])),
        reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[3])),
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(pointers[4])),
        static_cast<size_t>(rows),
        reinterpret_cast<double*>(static_cast<uintptr_t>(pointers[5])),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(pointers[6])),
        reinterpret_cast<int64_t*>(static_cast<uintptr_t>(pointers[7])),
        reinterpret_cast<int64_t*>(static_cast<uintptr_t>(pointers[8])),
        reinterpret_cast<signed char*>(static_cast<uintptr_t>(pointers[9])),
        reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(pointers[10])),
        reinterpret_cast<unsigned int*>(static_cast<uintptr_t>(pointers[11])),
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream)));
    return checked(static_cast<hipError_t>(status), NEO_HIP_OP_SMC_LAUNCH_V3, error);
  }
#endif
  int32_t available_memory(uint64_t& available, NeoHipRuntimeErrorV1& error) noexcept {
    size_t free_bytes = 0, total_bytes = 0;
    const auto status = checked(hipMemGetInfo(&free_bytes, &total_bytes), NEO_HIP_OP_MEMORY_INFO_V1, error);
    if (status != 0) return status;
    if (!total_bytes || free_bytes > total_bytes) return identity_error(error);
    available = static_cast<uint64_t>(free_bytes);
    return 0;
  }
  int32_t allocate(uint64_t stream, uint64_t bytes, uint64_t& result,
                    NeoHipRuntimeErrorV1& error) noexcept {
    void* pointer = nullptr;
    const auto status = hipMallocAsync(&pointer, static_cast<size_t>(bytes),
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream)));
    result = handle_bits(pointer);
    checked(status, NEO_HIP_OP_DEVICE_ALLOCATE_V1, error);
    return status == hipErrorOutOfMemory && !pointer
        ? NEO_HIP_RUNTIME_CAPACITY_V1 : static_cast<int32_t>(status);
  }
  int32_t free(uint64_t stream, uint64_t pointer, NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipFreeAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)),
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream))),
        NEO_HIP_OP_DEVICE_FREE_V1, error);
  }
  int32_t pin(uint64_t bytes, uint64_t& result, NeoHipRuntimeErrorV1& error) noexcept {
    void* pointer = nullptr;
    const auto status = hipHostMalloc(&pointer, static_cast<size_t>(bytes), hipHostMallocDefault);
    result = handle_bits(pointer);
    checked(status, NEO_HIP_OP_HOST_PIN_V1, error);
    return status == hipErrorOutOfMemory && !pointer
        ? NEO_HIP_RUNTIME_CAPACITY_V1 : static_cast<int32_t>(status);
  }
  int32_t unpin(uint64_t pointer, NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipHostFree(reinterpret_cast<void*>(static_cast<uintptr_t>(pointer))),
                   NEO_HIP_OP_HOST_UNPIN_V1, error);
  }
  int32_t upload(uint64_t stream, uint64_t device, uint64_t host, uint64_t bytes,
                 NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipMemcpyAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(device)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(host)), static_cast<size_t>(bytes),
        hipMemcpyHostToDevice, reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream))),
        NEO_HIP_OP_UPLOAD_V1, error);
  }
  int32_t download(uint64_t stream, uint64_t host, uint64_t device, uint64_t bytes,
                   NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipMemcpyAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(host)),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(device)), static_cast<size_t>(bytes),
        hipMemcpyDeviceToHost, reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream))),
        NEO_HIP_OP_DOWNLOAD_V1, error);
  }
  int32_t select_device(int32_t ordinal, NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipSetDevice(ordinal), NEO_HIP_OP_SELECT_DEVICE_V1, error);
  }

  int32_t create_stream(uint64_t& stream, NeoHipRuntimeErrorV1& error) noexcept {
    hipStream_t created = nullptr;
    const auto status = hipStreamCreateWithFlags(&created, hipStreamNonBlocking);
    stream = handle_bits(created);
    return checked(status, NEO_HIP_OP_CREATE_STREAM_V1, error);
  }

  int32_t inspect(int32_t ordinal, uint64_t owned_stream,
                  NeoHipRuntimeFactsV1& facts, NeoHipRuntimeErrorV1& error) noexcept {
    return inspect_common(ordinal, owned_stream, facts, error, true);
  }

  int32_t inspect_identity(int32_t ordinal, uint64_t owned_stream,
                           NeoHipRuntimeFactsV1& facts, NeoHipRuntimeErrorV1& error) noexcept {
    return inspect_common(ordinal, owned_stream, facts, error, false);
  }

  int32_t inspect_common(int32_t ordinal, uint64_t owned_stream,
                         NeoHipRuntimeFactsV1& facts, NeoHipRuntimeErrorV1& error,
                         bool memory) noexcept {
    // The registry supplies immutable total capacity for identity-only checks.
    // Dynamic capacity/counters are deliberately absent on the Search hot path.
    const uint64_t initial_total = facts.total_memory_bytes;
    facts = {};
    if (owned_stream <= 2) return identity_error(error);
    // Current-device selection is thread-local. Another live owner may have
    // selected a different device since the previous call; restore this owner's
    // device before querying, without adopting any foreign stream. HIP's
    // deprecated context APIs add no reset-generation authority here.
    auto status = select_device(ordinal, error);
    if (status != 0) return status;
    int current_device = -1;
    status = checked(hipGetDevice(&current_device), NEO_HIP_OP_CURRENT_DEVICE_V1, error);
    if (status != 0) return status;
    if (current_device != ordinal) return identity_error(error);

    const auto stream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(owned_stream));
    hipDevice_t stream_device = -1;
    status = checked(hipStreamGetDevice(stream, &stream_device), NEO_HIP_OP_STREAM_DEVICE_V1, error);
    if (status != 0) return status;
    unsigned long long stream_id = 0;
    status = checked(hipStreamGetId(stream, &stream_id), NEO_HIP_OP_STREAM_ID_V1, error);
    if (status != 0) return status;
    if (stream_device != ordinal || stream_id == 0) return identity_error(error);

    hipUUID uuid{};
    status = checked(hipDeviceGetUuid(&uuid, ordinal), NEO_HIP_OP_DEVICE_UUID_V1, error);
    if (status != 0) return status;
    hipDeviceProp_t properties{};
    status = checked(hipGetDeviceProperties(&properties, ordinal), NEO_HIP_OP_DEVICE_PROPERTIES_V1, error);
    if (status != 0) return status;
    static_assert(sizeof(properties.gcnArchName) == sizeof(facts.architecture));
    const auto* architecture_end = static_cast<const char*>(
        std::memchr(properties.gcnArchName, 0, sizeof(properties.gcnArchName)));
    if (properties.warpSize <= 0 || architecture_end == nullptr ||
        architecture_end == properties.gcnArchName) return identity_error(error);
    int runtime_version = 0;
    status = checked(hipRuntimeGetVersion(&runtime_version), NEO_HIP_OP_RUNTIME_VERSION_V1, error);
    if (status != 0) return status;
    int driver_version = 0;
    status = checked(hipDriverGetVersion(&driver_version), NEO_HIP_OP_DRIVER_VERSION_V1, error);
    if (status != 0) return status;
    size_t free_bytes = 0, total_bytes = static_cast<size_t>(initial_total);
    if (memory) {
      status = checked(hipMemGetInfo(&free_bytes, &total_bytes), NEO_HIP_OP_MEMORY_INFO_V1, error);
      if (status != 0) return status;
    }
    hipMemPool_t current_pool = nullptr, default_pool = nullptr;
    status = checked(hipDeviceGetMemPool(&current_pool, ordinal), NEO_HIP_OP_CURRENT_POOL_V1, error);
    if (status != 0) return status;
    status = checked(hipDeviceGetDefaultMemPool(&default_pool, ordinal), NEO_HIP_OP_DEFAULT_POOL_V1, error);
    if (status != 0) return status;
    if (current_pool == nullptr || current_pool != default_pool) return identity_error(error);
    uint64_t reserved = 0, used = 0;
    if (memory) {
      status = checked(hipMemPoolGetAttribute(current_pool, hipMemPoolAttrReservedMemCurrent, &reserved),
                       NEO_HIP_OP_POOL_RESERVED_V1, error);
      if (status != 0) return status;
      status = checked(hipMemPoolGetAttribute(current_pool, hipMemPoolAttrUsedMemCurrent, &used),
                       NEO_HIP_OP_POOL_USED_V1, error);
      if (status != 0) return status;
    }

    facts.device_ordinal = ordinal;
    facts.runtime_version = runtime_version;
    facts.driver_version = driver_version;
    facts.warp_size = static_cast<uint32_t>(properties.warpSize);
    std::memcpy(facts.uuid, &uuid, sizeof(uuid));
    facts.stream_handle = owned_stream;
    facts.stream_id = static_cast<uint64_t>(stream_id);
    facts.free_memory_bytes = static_cast<uint64_t>(free_bytes);
    facts.total_memory_bytes = static_cast<uint64_t>(total_bytes);
    facts.current_pool_handle = handle_bits(current_pool);
    facts.default_pool_handle = handle_bits(default_pool);
    facts.pool_reserved_bytes = reserved;
    facts.pool_used_bytes = used;
    std::memcpy(facts.architecture, properties.gcnArchName,
                static_cast<size_t>(architecture_end - properties.gcnArchName));
    return 0;
  }

  int32_t synchronize(uint64_t stream, NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipStreamSynchronize(reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream))),
                   NEO_HIP_OP_SYNCHRONIZE_V1, error);
  }
  int32_t destroy_stream(uint64_t stream, NeoHipRuntimeErrorV1& error) noexcept {
    return checked(hipStreamDestroy(reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream))),
                   NEO_HIP_OP_DESTROY_STREAM_V1, error);
  }
};

struct RuntimeV1 {
  HipOperationsV1 operations;
  LeaseRegistryV1<HipOperationsV1> registry{operations};
};

RuntimeV1& runtime() {
  // Deliberately no process-teardown HIP calls: quarantined resource records
  // remain owned until process exit, and a failed driver handle is never retried.
  static RuntimeV1* instance = new RuntimeV1;
  return *instance;
}

template<class Function>
int32_t ffi_boundary(NeoHipRuntimeErrorV1* error, Function&& function) noexcept {
  try { return function(); }
  catch (const std::bad_alloc&) {
    if (error) *error = {1u, NEO_HIP_OP_HOST_ALLOCATION_V1, 0, 0u};
    return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }
  catch (const std::exception&) {
    // Actual HIP operations are noexcept; this can only be a host lock/runtime
    // failure. Never attempt driver cleanup from an exception handler.
    if (error) *error = {1u, NEO_HIP_OP_IDENTITY_V1, 0, 0u};
    return NEO_HIP_RUNTIME_QUARANTINED_V1;
  }
  catch (...) {
    if (error) *error = {1u, NEO_HIP_OP_IDENTITY_V1, 0, 0u};
    return NEO_HIP_RUNTIME_QUARANTINED_V1;
  }
}
} // namespace

extern "C" int32_t neoethos_hip_runtime_lease_create_v1(
    int32_t ordinal, uint64_t* lease, NeoHipRuntimeFactsV1* facts,
    NeoHipRuntimeErrorV1* error) {
  if (lease) *lease = 0;
  if (facts) *facts = {};
  if (error) clear_error_v1(*error);
  if (!lease || !facts || !error || ordinal < 0) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
  return ffi_boundary(error, [&] { return runtime().registry.create(ordinal, lease, facts, error); });
}
extern "C" int32_t neoethos_hip_runtime_lease_query_v1(
    uint64_t lease, NeoHipRuntimeFactsV1* facts, NeoHipRuntimeErrorV1* error) {
  if (facts) *facts = {};
  if (error) clear_error_v1(*error);
  if (!facts || !error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
  return ffi_boundary(error, [&] { return runtime().registry.query(lease, facts, error); });
}
extern "C" int32_t neoethos_hip_runtime_lease_synchronize_v1(
    uint64_t lease, NeoHipRuntimeErrorV1* error) {
  if (!error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
  clear_error_v1(*error);
  return ffi_boundary(error, [&] { return runtime().registry.synchronize(lease, error); });
}
extern "C" int32_t neoethos_hip_runtime_lease_close_v1(
    uint64_t lease, NeoHipRuntimeErrorV1* error) {
  if (!error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
  clear_error_v1(*error);
  return ffi_boundary(error, [&] { return runtime().registry.close(lease, error); });
}

extern "C" int32_t neoethos_hip_runtime_borrow_v1(
    uint64_t lease, uint64_t* borrower, NeoHipRuntimeFactsV1* facts,
    NeoHipRuntimeErrorV1* error) {
  if (borrower) *borrower = 0;
  if (facts) *facts = {};
  return ffi_boundary(error, [&] { return runtime().registry.borrow(lease, borrower, facts, error); });
}
extern "C" int32_t neoethos_hip_runtime_borrow_query_v1(
    uint64_t lease, uint64_t borrower, NeoHipRuntimeFactsV1* identity,
    NeoHipRuntimeErrorV1* error) {
  if (identity) *identity = {};
  return ffi_boundary(error, [&] {
    return runtime().registry.borrow_query(lease, borrower, identity, error);
  });
}
extern "C" int32_t neoethos_hip_runtime_borrow_buffer_v1(
    uint64_t lease, uint64_t borrower, uint64_t buffer, uint64_t required_bytes,
    uint64_t* address, NeoHipRuntimeErrorV1* error) {
  if (address) *address = 0;
  return ffi_boundary(error, [&] {
    return runtime().registry.borrow_buffer(lease, borrower, buffer, required_bytes, address, error);
  });
}
extern "C" int32_t neoethos_hip_runtime_borrow_release_v1(
    uint64_t lease, uint64_t borrower, NeoHipRuntimeErrorV1* error) {
  return ffi_boundary(error, [&] { return runtime().registry.borrow_release(lease, borrower, error); });
}

extern "C" int32_t neoethos_hip_runtime_buffer_create_v1(
    uint64_t lease, uint64_t bytes, const uint8_t* upload, uint64_t* buffer,
    NeoHipRuntimeErrorV1* error) {
  if (buffer) *buffer = 0;
  if (!buffer || !error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
  return ffi_boundary(error, [&] {
    auto& rt = runtime();
    return rt.registry.with_resources(lease, error,
      [&](RuntimeBuffersV1& resources, uint64_t stream, const NeoHipRuntimeFactsV1&) {
        return resources.create(rt.operations, stream, bytes, upload, buffer, *error);
      });
  });
}

extern "C" int32_t neoethos_hip_runtime_buffer_free_v1(
    uint64_t lease, uint64_t buffer, NeoHipRuntimeErrorV1* error) {
  return ffi_boundary(error, [&] {
    auto& rt = runtime();
    return rt.registry.with_resources(lease, error,
      [&](RuntimeBuffersV1& resources, uint64_t stream, const NeoHipRuntimeFactsV1&) {
        return resources.release(rt.operations, stream, buffer, *error);
      });
  });
}

extern "C" int32_t neoethos_hip_runtime_buffer_read_v1(
    uint64_t lease, uint64_t buffer, uint8_t* output, uint64_t bytes,
    NeoHipRuntimeErrorV1* error) {
  return ffi_boundary(error, [&] {
    auto& rt = runtime();
    return rt.registry.with_resources(lease, error,
      [&](RuntimeBuffersV1& resources, uint64_t stream, const NeoHipRuntimeFactsV1&) {
        return resources.read(rt.operations, stream, buffer, output, bytes, *error);
      });
  });
}

extern "C" int32_t neoethos_hip_runtime_session_f64_v2(
    uint64_t lease, uint64_t rows, const uint64_t* keys, NeoHipRuntimeErrorV1* error) {
#if defined(NEOETHOS_HIP_SESSION_KERNELS_V1)
  return ffi_boundary(error, [&] {
    auto& rt = runtime();
    return rt.registry.with_resources(lease, error,
      [&](RuntimeBuffersV1& resources, uint64_t stream, const NeoHipRuntimeFactsV1& facts) {
        uint64_t pointers[8]{};
        if (!resources.session_inputs(rows, keys, pointers))
          return static_cast<int32_t>(NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1);
        // Built for one explicit exact target. No closest-architecture choice
        // or CPU substitute, including when a different AMD card is present.
        const char* arch = reinterpret_cast<const char*>(facts.architecture);
        const size_t base_length = std::strcspn(arch, ":");
        if (base_length != std::strlen(NEOETHOS_HIP_SESSION_TARGET_V1) ||
            std::strncmp(arch, NEOETHOS_HIP_SESSION_TARGET_V1, base_length) != 0)
          return identity_error(*error);
        NeoResidentSessionLaunchV2 launch{};
        launch.abi_version = NEOETHOS_RESIDENT_SESSION_ABI_VERSION_V2;
        launch.semantic_version = NEOETHOS_RESIDENT_SESSION_SEMANTIC_VERSION_V2;
        launch.feature_column_count = NEOETHOS_RESIDENT_SESSION_FEATURE_COLUMNS_V2;
        launch.row_count = rows;
        launch.open = reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[0]));
        launch.high = reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[1]));
        launch.low = reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[2]));
        launch.close = reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[3]));
        launch.volume = reinterpret_cast<const double*>(static_cast<uintptr_t>(pointers[4]));
        launch.timestamps_ms = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(pointers[5]));
        launch.feature_values = reinterpret_cast<double*>(static_cast<uintptr_t>(pointers[6]));
        launch.feature_validity_u8 = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(pointers[7]));
        const auto status = neoethos_hip_resident_session_f64_v2(
            &launch, reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream)));
        if (status != 0) {
          *error = {1u, NEO_HIP_OP_SESSION_LAUNCH_V1, status, 0u};
          return static_cast<int32_t>(NEO_HIP_RUNTIME_BACKEND_ERROR_V1);
        }
        resources.buffers.at(keys[6]).initialized = true;
        resources.buffers.at(keys[7]).initialized = true;
        return int32_t{0};
      });
  });
#else
  (void)lease; (void)rows; (void)keys;
  if (error) *error = {1u, NEO_HIP_OP_SESSION_LAUNCH_V1, 0, 0u};
  return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
#endif
}

extern "C" int32_t neoethos_hip_runtime_smc_parent_f64_v3(
    uint64_t lease, uint64_t rows, const uint64_t* inputs,
    const uint64_t* outputs, uint8_t* host_hashes, NeoHipRuntimeErrorV1* error) {
  if (!error || !inputs || !outputs || !host_hashes)
    return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
#if defined(NEOETHOS_HIP_NATIVE_KERNELS_V1)
  return ffi_boundary(error, [&] {
    auto& rt = runtime();
    return rt.registry.with_resources(lease, error,
      [&](RuntimeBuffersV1& resources, uint64_t stream, const NeoHipRuntimeFactsV1& facts) {
        const char* arch = reinterpret_cast<const char*>(facts.architecture);
        const size_t base_length = std::strcspn(arch, ":");
        if (base_length != std::strlen(NEOETHOS_HIP_NATIVE_TARGET_V1) ||
            std::strncmp(arch, NEOETHOS_HIP_NATIVE_TARGET_V1, base_length) != 0)
          return identity_error(*error);
        return resources.smc_parent(rt.operations, stream, rows, inputs, outputs,
                                    host_hashes, *error);
      });
  });
#else
  (void)lease; (void)rows;
  *error = {1u, NEO_HIP_OP_SMC_LAUNCH_V3, 0, 0u};
  return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
#endif
}

extern "C" int32_t neoethos_hip_runtime_pack_feature_store_v4(
    uint64_t lease, uint64_t rows, uint64_t columns,
    const NeoHipFeatureColumnV1* descriptors, uint64_t timestamps_key,
    const uint64_t* name_offsets, const uint8_t* names, uint64_t names_len,
    uint64_t reserve, const NeoHipFeatureNormalizationV3* normalization,
    uint64_t values_key, uint64_t validity_key, uint64_t* host_fit_words,
    uint64_t host_fit_word_capacity, NeoHipFeatureStoreReceiptV4* receipt,
    NeoHipRuntimeErrorV1* error) {
  if (receipt) *receipt = {};
  if (error) clear_error_v1(*error);
  if (!error || !receipt || !descriptors || !name_offsets || !names)
    return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
#if defined(NEOETHOS_HIP_NATIVE_KERNELS_V1)
  return ffi_boundary(error, [&] {
    auto& rt = runtime();
    return rt.registry.with_resources(lease, error,
      [&](RuntimeBuffersV1& resources, uint64_t stream, const NeoHipRuntimeFactsV1& facts) {
        const char* arch = reinterpret_cast<const char*>(facts.architecture);
        const size_t base_length = std::strcspn(arch, ":");
        if (base_length != std::strlen(NEOETHOS_HIP_NATIVE_TARGET_V1) ||
            std::strncmp(arch, NEOETHOS_HIP_NATIVE_TARGET_V1, base_length) != 0)
          return identity_error(*error);
        return resources.pack_feature_store(rt.operations, stream, rows, columns, descriptors,
            timestamps_key, name_offsets, names, names_len, reserve, normalization,
            values_key, validity_key, host_fit_words, host_fit_word_capacity, *receipt, *error);
      });
  });
#else
  (void)lease; (void)rows; (void)columns; (void)timestamps_key; (void)names_len;
  (void)reserve; (void)normalization; (void)values_key; (void)validity_key;
  (void)host_fit_words; (void)host_fit_word_capacity;
  *error = {1u, NEO_HIP_OP_FEATURE_PACK_V3, 0, 0u};
  return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
#endif
}
