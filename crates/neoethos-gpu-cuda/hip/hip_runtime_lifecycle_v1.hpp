#pragma once

#include "hip_runtime_owner_v1.h"
#include "hip_runtime_buffers_v1.hpp"

#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>

namespace neoethos::hip_runtime_v1 {

inline void clear_error_v1(NeoHipRuntimeErrorV1& error) {
  error = {1u, NEO_HIP_OP_NONE_V1, 0, 0u};
}

inline bool valid_facts_v1(const NeoHipRuntimeFactsV1& facts) {
  uint8_t uuid = 0;
  for (const auto byte : facts.uuid) uuid |= byte;
  return facts.abi_version == 1u && facts.backend_kind == 2u &&
      facts.lease_id != 0 && facts.device_ordinal >= 0 &&
      facts.runtime_version > 0 && facts.driver_version > 0 &&
      facts.warp_size != 0 && uuid != 0 &&
      facts.stream_handle > 2 && facts.stream_id != 0 &&
      facts.total_memory_bytes != 0 &&
      facts.free_memory_bytes <= facts.total_memory_bytes &&
      facts.current_pool_handle != 0 &&
      facts.current_pool_handle == facts.default_pool_handle &&
      facts.architecture[0] != 0 &&
      std::memchr(facts.architecture, 0, sizeof(facts.architecture)) != nullptr;
}

inline bool same_identity_v1(const NeoHipRuntimeFactsV1& a,
                             const NeoHipRuntimeFactsV1& b) {
  return a.abi_version == b.abi_version && a.backend_kind == b.backend_kind &&
      a.lease_id == b.lease_id && a.device_ordinal == b.device_ordinal &&
      a.runtime_version == b.runtime_version && a.driver_version == b.driver_version &&
      a.warp_size == b.warp_size && std::memcmp(a.uuid, b.uuid, sizeof(a.uuid)) == 0 &&
      a.stream_handle == b.stream_handle &&
      a.stream_id == b.stream_id && a.total_memory_bytes == b.total_memory_bytes &&
      a.current_pool_handle == b.current_pool_handle &&
      a.default_pool_handle == b.default_pool_handle &&
      std::memcmp(a.architecture, b.architecture, sizeof(a.architecture)) == 0;
}

// Production instantiates this once with real HIP operations. The type seam
// permits host tests of this exact lifecycle; it is not an exported override.
// A short registry lock pins entries. Per-entry locks protect API use/close;
// unrelated leases never wait on another lease's HIP synchronization. External
// HIP reset remains forbidden, including from code outside this registry.
template<class Ops>
class LeaseRegistryV1 {
  struct Entry {
    std::mutex mutex;
    int32_t ordinal = -1;
    uint64_t stream = 0;
    NeoHipRuntimeFactsV1 identity{};
    RuntimeBuffersV1 resources;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> borrowers;
    uint64_t next_borrower = 1;
    bool quarantined = false;
    bool retired = false;
  };
  Ops& ops_;
  std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<Entry>> entries_;
  uint64_t next_id_ = 1;

  static int32_t quarantine(Entry& entry, int32_t status) {
    entry.quarantined = true;
    return status == NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1
        ? status : NEO_HIP_RUNTIME_QUARANTINED_V1;
  }

  int32_t inspect(uint64_t id, Entry& entry, NeoHipRuntimeFactsV1& facts,
                  NeoHipRuntimeErrorV1& error, bool initial, bool memory = true) {
    if (!memory) facts.total_memory_bytes = entry.identity.total_memory_bytes;
    const auto status = memory
        ? ops_.inspect(entry.ordinal, entry.stream, facts, error)
        : ops_.inspect_identity(entry.ordinal, entry.stream, facts, error);
    if (status != 0) return quarantine(entry, status);
    facts.abi_version = 1;
    facts.backend_kind = 2;
    facts.lease_id = id;
    if (!valid_facts_v1(facts) || facts.device_ordinal != entry.ordinal ||
        facts.stream_handle != entry.stream ||
        (!initial && !same_identity_v1(entry.identity, facts))) {
      error = {1u, NEO_HIP_OP_IDENTITY_V1, 0, 0u};
      return quarantine(entry, NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1);
    }
    return NEO_HIP_RUNTIME_OK_V1;
  }

  std::shared_ptr<Entry> pin(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(id);
    return id == 0 || found == entries_.end() ? nullptr : found->second;
  }

  static int32_t live(const Entry& entry) {
    if (entry.retired) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    return entry.quarantined ? NEO_HIP_RUNTIME_QUARANTINED_V1 : 0;
  }

  void retire(uint64_t id, Entry& entry) {
    entry.retired = true; // Already-pinned waiters must not touch closed handles.
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(id);
  }

public:
  explicit LeaseRegistryV1(Ops& ops) : ops_(ops) {}
  LeaseRegistryV1(const LeaseRegistryV1&) = delete;
  LeaseRegistryV1& operator=(const LeaseRegistryV1&) = delete;

  // Private native operations receive only resources from this validated
  // owner. No registry lock is held across runtime work, allocation or waits.
  template<class Function>
  int32_t with_resources(uint64_t id, NeoHipRuntimeErrorV1* error, Function&& function) {
    if (!error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    clear_error_v1(*error);
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    auto status = live(*entry);
    if (status != 0) return status;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false, false);
    if (status != 0) return status;
    try { status = function(entry->resources, entry->stream, observed); }
    catch (...) {
      // A host allocation/exception after partial runtime work must not make
      // an uncertain owner reusable. Native records keep every acquired handle.
      error->operation = NEO_HIP_OP_HOST_ALLOCATION_V1;
      return quarantine(*entry, NEO_HIP_RUNTIME_BACKEND_ERROR_V1);
    }
    if (status == 0 || status == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 ||
        status == NEO_HIP_RUNTIME_CAPACITY_V1 ||
        status == NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1) return status;
    return quarantine(*entry, status);
  }

  int32_t create(int32_t ordinal, uint64_t* lease, NeoHipRuntimeFactsV1* facts,
                 NeoHipRuntimeErrorV1* error) {
    if (lease) *lease = 0;
    if (facts) *facts = {};
    if (error) clear_error_v1(*error);
    if (!lease || !facts || !error || ordinal < 0) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    std::shared_ptr<Entry> allocation;
    try { allocation = std::make_shared<Entry>(); }
    catch (const std::bad_alloc&) {
      error->operation = NEO_HIP_OP_HOST_ALLOCATION_V1;
      return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
    }
    allocation->ordinal = ordinal;
    std::lock_guard<std::mutex> entry_lock(allocation->mutex);
    uint64_t id = 0;
    try {
      std::lock_guard<std::mutex> map_lock(mutex_);
      if (next_id_ == std::numeric_limits<uint64_t>::max()) {
        error->operation = NEO_HIP_OP_HOST_ALLOCATION_V1;
        return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
      }
      id = next_id_++; // Never rolled back or reused, including failed creation.
      entries_.emplace(id, allocation);
    }
    catch (const std::bad_alloc&) {
      error->operation = NEO_HIP_OP_HOST_ALLOCATION_V1;
      return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
    }
    auto& entry = *allocation;
    auto status = ops_.select_device(ordinal, *error);
    if (status != 0) { retire(id, entry); return NEO_HIP_RUNTIME_BACKEND_ERROR_V1; }
    status = ops_.create_stream(entry.stream, *error);
    if (status != 0) return quarantine(entry, status);
    if (entry.stream <= 2) {
      error->operation = NEO_HIP_OP_IDENTITY_V1;
      return quarantine(entry, NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1);
    }
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, entry, observed, *error, true);
    if (status != 0) return status;
    entry.identity = observed;
    *facts = observed;
    *lease = id;
    return NEO_HIP_RUNTIME_OK_V1;
  }

  int32_t query(uint64_t id, NeoHipRuntimeFactsV1* facts, NeoHipRuntimeErrorV1* error) {
    if (facts) *facts = {};
    if (error) clear_error_v1(*error);
    if (!facts || !error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    int32_t status = live(*entry);
    if (status != 0) return status;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false);
    if (status == 0) *facts = observed;
    return status;
  }

  int32_t borrow(uint64_t id, uint64_t* borrower, NeoHipRuntimeFactsV1* facts,
                 NeoHipRuntimeErrorV1* error) {
    if (borrower) *borrower = 0;
    if (facts) *facts = {};
    if (error) clear_error_v1(*error);
    if (!borrower || !facts || !error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    auto status = live(*entry);
    if (status != 0) return status;
    if (entry->next_borrower == std::numeric_limits<uint64_t>::max())
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false);
    if (status != 0) return status;
    const auto token = entry->next_borrower++;
    // No GPU resource has been acquired if this host insertion fails.
    entry->borrowers.emplace(token, std::unordered_set<uint64_t>{});
    *facts = observed;
    *borrower = token;
    return 0;
  }

  int32_t borrow_query(uint64_t id, uint64_t borrower, NeoHipRuntimeFactsV1* facts,
                       NeoHipRuntimeErrorV1* error) {
    if (facts) *facts = {};
    if (error) clear_error_v1(*error);
    if (!facts || !error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    auto status = live(*entry);
    if (status != 0) return status;
    if (!entry->borrowers.count(borrower)) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false, false);
    if (status == 0) *facts = observed;
    return status;
  }

  int32_t borrow_buffer(uint64_t id, uint64_t borrower, uint64_t key, uint64_t bytes,
                        uint64_t* address, NeoHipRuntimeErrorV1* error) {
    if (address) *address = 0;
    if (error) clear_error_v1(*error);
    if (!address || !error || !bytes) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    auto status = live(*entry);
    if (status != 0) return status;
    const auto consumer = entry->borrowers.find(borrower);
    const auto found = entry->resources.buffers.find(key);
    if (consumer == entry->borrowers.end() || found == entry->resources.buffers.end())
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto& value = found->second;
    if (!value.device || value.release_requested || !value.initialized || value.bytes != bytes ||
        value.borrowers == std::numeric_limits<uint64_t>::max())
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false, false);
    if (status != 0) return status;
    // Insert before increasing the count: allocation failure cannot orphan a pin.
    if (consumer->second.insert(key).second) ++value.borrowers;
    *address = value.device;
    return 0;
  }

  int32_t borrow_release(uint64_t id, uint64_t borrower, NeoHipRuntimeErrorV1* error) {
    if (!error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    clear_error_v1(*error);
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    auto status = live(*entry);
    if (status != 0) return status;
    const auto consumer = entry->borrowers.find(borrower);
    if (consumer == entry->borrowers.end()) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false, false);
    if (status != 0) return status;
    for (const auto key : consumer->second) {
      const auto found = entry->resources.buffers.find(key);
      if (found == entry->resources.buffers.end() || !found->second.borrowers)
        return quarantine(*entry, NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1);
      auto& value = found->second;
      if (--value.borrowers == 0 && value.release_requested) {
        value.release_requested = false;
        status = entry->resources.release(ops_, entry->stream, key, *error);
        if (status != 0) return quarantine(*entry, status);
      }
    }
    entry->borrowers.erase(consumer);
    return 0;
  }

  int32_t synchronize(uint64_t id, NeoHipRuntimeErrorV1* error) {
    if (!error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    clear_error_v1(*error);
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    int32_t status = live(*entry);
    if (status != 0) return status;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false);
    if (status != 0) return status;
    status = ops_.synchronize(entry->stream, *error);
    if (status == 0) status = entry->resources.completed(ops_, *error);
    return status == 0 ? 0 : quarantine(*entry, status);
  }

  int32_t close(uint64_t id, NeoHipRuntimeErrorV1* error) {
    if (!error) return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    clear_error_v1(*error);
    auto entry = pin(id);
    if (!entry) return NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    int32_t status = live(*entry);
    if (status != 0) return status;
    if (!entry->borrowers.empty()) return NEO_HIP_RUNTIME_BUSY_V1;
    NeoHipRuntimeFactsV1 observed{};
    status = inspect(id, *entry, observed, *error, false);
    if (status != 0) return status;
    status = ops_.synchronize(entry->stream, *error);
    if (status != 0) return quarantine(*entry, status);
    status = entry->resources.close(ops_, entry->stream, *error);
    if (status != 0) return quarantine(*entry, status);
    status = ops_.destroy_stream(entry->stream, *error);
    if (status != 0) return quarantine(*entry, status);
    entry->stream = 0; // A successful destruction must never be retried.
    retire(id, *entry);
    return NEO_HIP_RUNTIME_OK_V1;
  }
};

} // namespace neoethos::hip_runtime_v1
