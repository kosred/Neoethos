#pragma once

#include "hip_runtime_owner_v1.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neoethos::hip_runtime_v1 {

// All methods run under their owning lease's lock. Raw resources deliberately
// have no destructors: a quarantined lease retains uncertain handles/staging.
struct RuntimeBuffersV1 {
  struct Buffer {
    uint64_t device = 0;
    uint64_t bytes = 0;
    uint64_t upload = 0;
    uint64_t readback = 0;
    uint64_t borrowers = 0;
    bool initialized = false;
    bool release_requested = false;
    // Content-sealed timestamps/outputs may be read or freed, never rewritten
    // by another typed producer merely because its requested extent matches.
    bool write_sealed = false;
  };
  std::unordered_map<uint64_t, Buffer> buffers;
  uint64_t next_id = 1;

  template<class Ops>
  int32_t completed(Ops& ops, NeoHipRuntimeErrorV1& error) {
    for (auto it = buffers.begin(); it != buffers.end();) {
      auto& value = it->second;
      if (value.upload) {
        const auto status = ops.unpin(value.upload, error);
        if (status != 0) return status;
        value.upload = 0;
      }
      // A readback is released by read() after copying completed bytes to the
      // caller. On failed synchronization it remains retained, never exposed.
      if (!value.device && !value.readback) it = buffers.erase(it);
      else ++it;
    }
    return 0;
  }

  template<class Ops>
  int32_t create(Ops& ops, uint64_t stream, uint64_t bytes, const uint8_t* source,
                 uint64_t* key, NeoHipRuntimeErrorV1& error) {
    *key = 0;
    if (!bytes || bytes > std::numeric_limits<size_t>::max() ||
        next_id == std::numeric_limits<uint64_t>::max())
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    const uint64_t id = next_id++;
    // Allocate registry storage BEFORE any runtime resource can exist.
    auto& value = buffers.emplace(id, Buffer{}).first->second;
    value.bytes = bytes;
    auto status = ops.allocate(stream, bytes, value.device, error);
    if (status == NEO_HIP_RUNTIME_CAPACITY_V1 && !value.device) {
      buffers.erase(id);
      return status;
    }
    if (status != 0) return status;
    if (!value.device) return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
    if (source) {
      status = ops.pin(bytes, value.upload, error);
      if (status == NEO_HIP_RUNTIME_CAPACITY_V1 && !value.upload) {
        const auto cleanup = ops.free(stream, value.device, error);
        if (cleanup != 0) return cleanup;
        value.device = 0;
        buffers.erase(id);
        return status;
      }
      if (status != 0) return status;
      if (!value.upload) return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
      std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(value.upload)),
                  source, static_cast<size_t>(bytes));
      status = ops.upload(stream, value.device, value.upload, bytes, error);
      if (status != 0) return status;
      value.initialized = true; // Completion remains ordered on this stream.
    }
    *key = id;
    return 0;
  }

  template<class Ops>
  int32_t release(Ops& ops, uint64_t stream, uint64_t id,
                  NeoHipRuntimeErrorV1& error) {
    const auto found = buffers.find(id);
    if (found == buffers.end() || !found->second.device || found->second.release_requested)
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto& value = found->second;
    if (value.borrowers) {
      value.release_requested = true;
      return 0; // A native consumer still owns a pin; do not invalidate its pointer.
    }
    const auto status = ops.free(stream, value.device, error);
    if (status != 0) return status;
    value.device = 0; // Already queued; never enqueue the same free twice.
    if (!value.upload && !value.readback) buffers.erase(found);
    return 0;
  }

  template<class Ops>
  int32_t read(Ops& ops, uint64_t stream, uint64_t id, uint8_t* destination,
               uint64_t bytes, NeoHipRuntimeErrorV1& error) {
    const auto found = buffers.find(id);
    if (found == buffers.end() || found->second.bytes != bytes)
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    return read_prefix(ops, stream, id, destination, bytes, error);
  }

  // Internal bounded metadata reads only. The public buffer-read ABI above
  // remains exact-sized; no arbitrary feature/scratch view is exported.
  template<class Ops>
  int32_t read_prefix(Ops& ops, uint64_t stream, uint64_t id, uint8_t* destination,
                      uint64_t bytes, NeoHipRuntimeErrorV1& error) {
    const auto found = buffers.find(id);
    if (!destination || found == buffers.end() || !found->second.device ||
        found->second.release_requested || !found->second.initialized || !bytes || found->second.bytes < bytes)
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto& value = found->second;
    auto status = ops.pin(bytes, value.readback, error);
    if (status != 0) return status;
    if (!value.readback) return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
    status = ops.download(stream, value.readback, value.device, bytes, error);
    if (status != 0) return status;
    status = ops.synchronize(stream, error);
    if (status != 0) return status;
    std::memcpy(destination,
                reinterpret_cast<const void*>(static_cast<uintptr_t>(value.readback)),
                static_cast<size_t>(bytes));
    status = ops.unpin(value.readback, error);
    if (status != 0) return status;
    value.readback = 0;
    return completed(ops, error);
  }

  template<class Ops>
  int32_t close(Ops& ops, uint64_t stream, NeoHipRuntimeErrorV1& error) {
    // Caller already synchronized, so releasing upload staging is safe.
    auto status = completed(ops, error);
    if (status != 0) return status;
    bool freed = false;
    for (auto& pair : buffers) {
      auto& value = pair.second;
      if (value.device) {
        status = ops.free(stream, value.device, error);
        if (status != 0) return status;
        value.device = 0;
        freed = true;
      }
      if (value.readback) {
        status = ops.unpin(value.readback, error);
        if (status != 0) return status;
        value.readback = 0;
      }
    }
    if (freed) {
      status = ops.synchronize(stream, error);
      if (status != 0) return status;
    }
    buffers.clear();
    return 0;
  }

  bool session_inputs(uint64_t rows, const uint64_t* keys, uint64_t* pointers) {
    if (!rows || !keys || rows > std::numeric_limits<uint64_t>::max() / 184)
      return false;
    for (size_t index = 0; index < 8; ++index) {
      const auto found = buffers.find(keys[index]);
      const uint64_t bytes = rows * (index < 6 ? 8 : index == 6 ? 184 : 23);
      if (found == buffers.end() || !found->second.device || found->second.release_requested ||
          found->second.bytes != bytes || (index < 6 && !found->second.initialized) ||
          (index >= 6 && (found->second.borrowers != 0 || found->second.write_sealed)))
        return false;
      pointers[index] = found->second.device;
    }
    return true;
  }

  static bool smc_parent_sizes(uint64_t rows, uint64_t* sizes) {
    // The original producer stores each row ordinal in int. The last row,
    // rather than the row count, must be representable; no silent truncation.
    if (!rows || !sizes || rows - 1 > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        rows > (std::numeric_limits<size_t>::max() - 100) / 441)
      return false;
    for (size_t i = 0; i < 5; ++i) sizes[i] = rows * 8;
    sizes[5] = rows * 368; sizes[6] = rows * 46;
    sizes[7] = rows * 8; sizes[8] = rows * 8; sizes[9] = rows * 11;
    sizes[10] = 96; sizes[11] = 4;
    return true;
  }

  bool smc_parent_inputs(uint64_t rows, const uint64_t* inputs,
                         const uint64_t* outputs, uint64_t* pointers) {
    uint64_t sizes[12]{};
    if (!inputs || !outputs || !pointers || !smc_parent_sizes(rows, sizes)) return false;
    for (size_t i = 0; i < 12; ++i) {
      const auto key = i < 5 ? inputs[i] : outputs[i - 5];
      for (size_t earlier = 0; earlier < i; ++earlier)
        if (key == (earlier < 5 ? inputs[earlier] : outputs[earlier - 5])) return false;
      const auto found = buffers.find(key);
      if (found == buffers.end()) return false;
      const auto& value = found->second;
      if (!value.device || value.release_requested || value.bytes != sizes[i] ||
          (i < 5 && !value.initialized) ||
          (i >= 5 && (value.borrowers != 0 || value.write_sealed))) return false;
      pointers[i] = value.device;
    }
    return true;
  }

  template<class Ops>
  int32_t smc_parent(Ops& ops, uint64_t stream, uint64_t rows,
                     const uint64_t* inputs, const uint64_t* outputs,
                     uint8_t* host_hashes, NeoHipRuntimeErrorV1& error) {
    uint64_t pointers[12]{};
    if (!host_hashes || !smc_parent_inputs(rows, inputs, outputs, pointers))
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    auto& error_buffer = buffers.at(outputs[6]);
    auto& hash_buffer = buffers.at(outputs[5]);
    // Buffer-owned slots survive ambiguous runtime outcomes. No DMA targets
    // caller memory or a temporary stack allocation, including failure paths.
    auto status = ops.pin(4, error_buffer.readback, error);
    if (status != 0) return status;
    if (!error_buffer.readback) return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
    for (size_t i = 0; i < 7; ++i) buffers.at(outputs[i]).initialized = false;
    status = ops.smc_parent(stream, rows, pointers, error);
    if (status != 0) return status;
    status = ops.download(stream, error_buffer.readback, pointers[11], 4, error);
    if (status != 0) return status;
    status = ops.synchronize(stream, error);
    if (status != 0) return status;
    uint32_t semantic = 0;
    std::memcpy(&semantic, reinterpret_cast<const void*>(
        static_cast<uintptr_t>(error_buffer.readback)), sizeof(semantic));
    status = ops.unpin(error_buffer.readback, error);
    if (status != 0) return status;
    error_buffer.readback = 0;
    status = completed(ops, error);
    if (status != 0) return status;
    if (semantic != 0) {
      if (semantic > 5) {
        error = {1u, NEO_HIP_OP_SMC_SEMANTIC_V3, 0, 0u};
        return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
      }
      error = {1u, NEO_HIP_OP_SMC_SEMANTIC_V3, static_cast<int32_t>(semantic), 0u};
      return NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1;
    }
    // A clean second pin OOM leaves the lease reusable, but none of the
    // kernel-written outputs published. Retrying reruns the same producer.
    status = ops.pin(96, hash_buffer.readback, error);
    if (status != 0) return status;
    if (!hash_buffer.readback) return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
    status = ops.download(stream, hash_buffer.readback, pointers[10], 96, error);
    if (status != 0) return status;
    status = ops.synchronize(stream, error);
    if (status != 0) return status;
    uint8_t hashes[96];
    std::memcpy(hashes, reinterpret_cast<const void*>(
        static_cast<uintptr_t>(hash_buffer.readback)), sizeof(hashes));
    status = ops.unpin(hash_buffer.readback, error);
    if (status != 0) return status;
    hash_buffer.readback = 0;
    status = completed(ops, error);
    if (status != 0) return status;
    for (size_t i = 0; i < 7; ++i) buffers.at(outputs[i]).initialized = true;
    std::memcpy(host_hashes, hashes, sizeof(hashes));
    return 0;
  }

  struct FeatureStoreSizesV3 {
    uint64_t values = 0, validity = 0, timestamps = 0, leaves = 0;
    uint64_t metadata = 0, scratch = 0, transient = 0;
  };

  static bool feature_store_sizes(uint64_t rows, uint64_t columns,
                                   uint64_t names, FeatureStoreSizesV3& out) {
    out = {};
    const uint64_t max = std::numeric_limits<size_t>::max();
    if (!rows || !columns || !names || columns == max ||
        rows > max / 8 || columns > max / (rows * 8) ||
        columns > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 32)
      return false;
    const uint64_t cells = rows * columns;
    const uint64_t chunks = rows / 4096 + (rows % 4096 != 0);
    if (columns + 1 > max / chunks || columns > max / 32 ||
        columns + 1 > max / 8) return false;
    const uint64_t leaves = chunks * (columns + 1);
    const uint64_t table = columns * 32, offsets = (columns + 1) * 8;
    if (leaves > max / 64 || table > max - offsets ||
        names > max - table - offsets) return false;
    const uint64_t metadata = table + offsets + names;
    if (metadata > max - 36 || leaves * 64 > max - 36 - metadata) return false;
    out = {cells * 8, (cells / 8 + (cells % 8 != 0)) * 4,
           rows * 8, leaves, metadata, leaves * 32, metadata + leaves * 64 + 36};
    return true;
  }

  static bool feature_name_utf8(const uint8_t* bytes, uint64_t count) {
    for (uint64_t i = 0; i < count;) {
      const uint32_t first = bytes[i++];
      if (first < 0x80) continue;
      uint32_t value = 0, minimum = 0; uint64_t remaining = 0;
      if (first >= 0xc2 && first <= 0xdf) { value = first & 0x1f; minimum = 0x80; remaining = 1; }
      else if (first >= 0xe0 && first <= 0xef) { value = first & 0x0f; minimum = 0x800; remaining = 2; }
      else if (first >= 0xf0 && first <= 0xf4) { value = first & 7; minimum = 0x10000; remaining = 3; }
      else return false;
      if (remaining > count - i) return false;
      while (remaining--) {
        const uint32_t next = bytes[i++];
        if ((next & 0xc0) != 0x80) return false;
        value = (value << 6) | (next & 0x3f);
      }
      if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    }
    return true;
  }

  struct FeatureNormalizationSizesV3 {
    uint64_t padded_rows = 0, sort_bytes = 0, fit_bytes = 0;
  };

  static bool feature_normalization_sizes(uint64_t rows, uint64_t columns,
      const NeoHipFeatureNormalizationV3& request, FeatureNormalizationSizesV3& out) {
    out = {};
    const uint64_t max = std::numeric_limits<size_t>::max();
    if (!rows || rows > max / 8 || !columns || columns > max / 48 ||
        !request.column_modes || request.column_mode_count != columns) return false;
    // The owning Data producer supplies this exact canonical interval. Recheck
    // the current native contract, rather than accepting another caller window.
    const uint64_t canonical_end = static_cast<uint64_t>(std::floor(static_cast<double>(rows) * (1.0 - 0.2)));
    if (request.training_row_start != 0 || request.training_row_end != canonical_end ||
        canonical_end < 64 || canonical_end >= rows) return false;
    for (uint64_t i = 0; i < columns; ++i)
      if (request.column_modes[i] > 3) return false;
    uint64_t padded = 1;
    while (padded < canonical_end) {
      if (padded > max / 2) return false;
      padded *= 2;
    }
    const uint64_t batch = columns < 64 ? columns : 64;
    if (padded > max / 8 / batch) return false;
    out = {padded, padded * batch * 8, columns * 48};
    return true;
  }

  template<class Ops>
  int32_t pack_feature_store(Ops& ops, uint64_t stream, uint64_t rows, uint64_t columns,
      const NeoHipFeatureColumnV1* descriptors, uint64_t timestamps_key,
      const uint64_t* name_offsets, const uint8_t* names, uint64_t names_len,
      uint64_t reserve, const NeoHipFeatureNormalizationV3* normalization,
      uint64_t values_key, uint64_t validity_key, uint64_t* host_fit_words,
      uint64_t host_fit_word_capacity, NeoHipFeatureStoreReceiptV4& receipt,
      NeoHipRuntimeErrorV1& error) {
    receipt = {};
    FeatureStoreSizesV3 sizes;
    FeatureNormalizationSizesV3 norm_sizes;
    const size_t temporary_count = normalization ? 7 : 5;
    if (!descriptors || !name_offsets || !names ||
        !feature_store_sizes(rows, columns, names_len, sizes) ||
        next_id > std::numeric_limits<uint64_t>::max() - temporary_count ||
        values_key == validity_key || values_key == timestamps_key || validity_key == timestamps_key ||
        name_offsets[0] != 0 || name_offsets[columns] != names_len)
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    if (normalization) {
      if (!feature_normalization_sizes(rows, columns, *normalization, norm_sizes) ||
          !host_fit_words || host_fit_word_capacity != norm_sizes.fit_bytes / 8 ||
          norm_sizes.sort_bytes > std::numeric_limits<uint64_t>::max() - sizes.transient ||
          norm_sizes.fit_bytes > std::numeric_limits<uint64_t>::max() - sizes.transient - norm_sizes.sort_bytes)
        return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
      sizes.transient += norm_sizes.sort_bytes + norm_sizes.fit_bytes;
    } else if (host_fit_words || host_fit_word_capacity != 0) {
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    }
    if (reserve > std::numeric_limits<uint64_t>::max() - sizes.transient)
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    // These are bounded fit/control results, never feature arrays. They are
    // populated only from completed native readbacks and published at commit.
    std::vector<uint64_t> fit_words(static_cast<size_t>(norm_sizes.fit_bytes / 8));
    uint8_t fit_digest[32]{};
    const auto live = [&](uint64_t key, bool initialized) -> const Buffer* {
      const auto found = buffers.find(key);
      if (found == buffers.end() || !found->second.device || found->second.release_requested ||
          (initialized && !found->second.initialized)) return nullptr;
      return &found->second;
    };
    const auto* timestamps = live(timestamps_key, true);
    const auto* values = live(values_key, false);
    const auto* validity = live(validity_key, false);
    if (!timestamps || !values || !validity || timestamps->bytes != sizes.timestamps ||
        values->bytes != sizes.values || validity->bytes != sizes.validity ||
        values->initialized || validity->initialized || values->borrowers || validity->borrowers ||
        values->write_sealed || validity->write_sealed)
      return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
    const uint64_t timestamp_address = timestamps->device;
    const uint64_t values_address = values->device, validity_address = validity->device;
    std::unordered_set<std::string_view> unique_names;
    std::vector<uint8_t> metadata(static_cast<size_t>(sizes.metadata));
    const auto word = [&](uint64_t index, uint64_t value) {
      std::memcpy(metadata.data() + index * 8, &value, 8);
    };
    for (uint64_t i = 0; i < columns; ++i) {
      if (name_offsets[i] >= name_offsets[i + 1] || name_offsets[i + 1] > names_len ||
          !feature_name_utf8(names + name_offsets[i], name_offsets[i + 1] - name_offsets[i]) ||
          !unique_names.emplace(reinterpret_cast<const char*>(names + name_offsets[i]),
                                static_cast<size_t>(name_offsets[i + 1] - name_offsets[i])).second)
        return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
      const auto& column = descriptors[i];
      if (column.values_key == values_key || column.values_key == validity_key ||
          column.validity_key == values_key || column.validity_key == validity_key)
        return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
      const auto* source = live(column.values_key, true);
      const auto* codes = live(column.validity_key, true);
      if (!source || !codes || source->bytes % 8 != 0 ||
          column.values_element_offset > source->bytes / 8 ||
          rows > source->bytes / 8 - column.values_element_offset ||
          column.validity_byte_offset > codes->bytes || rows > codes->bytes - column.validity_byte_offset)
        return NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1;
      word(i, source->device); word(columns + i, column.values_element_offset);
      word(columns * 2 + i, codes->device); word(columns * 3 + i, column.validity_byte_offset);
    }
    std::memcpy(metadata.data() + columns * 32, name_offsets, static_cast<size_t>((columns + 1) * 8));
    std::memcpy(metadata.data() + columns * 32 + (columns + 1) * 8, names, static_cast<size_t>(names_len));
    uint64_t available = 0;
    auto status = ops.available_memory(available, error);
    if (status != 0) return status;
    if (sizes.transient + reserve > available) return NEO_HIP_RUNTIME_CAPACITY_V1;

    // All transient resources and their pinned staging are registered
    // before use. Unknown completion retains them in this quarantined lease.
    uint64_t temporary[7]{};
    const uint64_t extents[7]{sizes.metadata, sizes.scratch, sizes.scratch, 4, 32,
                              norm_sizes.sort_bytes, norm_sizes.fit_bytes};
    const auto cleanup = [&]() -> int32_t {
      for (const auto key : temporary) {
        if (!key) continue;
        const auto result = release(ops, stream, key, error);
        if (result != 0) return result;
      }
      auto result = ops.synchronize(stream, error);
      if (result != 0) return result;
      return completed(ops, error);
    };
    const auto clean_rejection = [&](int32_t reason) -> int32_t {
      const auto original_error = error;
      const auto result = cleanup();
      if (result != 0) return result;
      error = original_error;
      return reason;
    };
    for (size_t i = 0; i < temporary_count; ++i) {
      status = create(ops, stream, extents[i], i == 0 ? metadata.data() : nullptr, &temporary[i], error);
      if (status == NEO_HIP_RUNTIME_CAPACITY_V1) return clean_rejection(status);
      if (status != 0) return status;
    }
    uint64_t device[7]{};
    for (size_t i = 0; i < temporary_count; ++i) device[i] = buffers.at(temporary[i]).device;
    status = ops.pack_feature_store(stream, rows, columns, device[0], values_address,
                                    validity_address, sizes.validity, device[3], error);
    if (status != 0) return status;
    buffers.at(temporary[3]).initialized = true;
    uint32_t control = 0;
    status = read(ops, stream, temporary[3], reinterpret_cast<uint8_t*>(&control), 4, error);
    if (status == NEO_HIP_RUNTIME_CAPACITY_V1) return clean_rejection(status);
    if (status != 0) return status;
    if (control != 0) {
      if (control != 1) {
        error = {1u, NEO_HIP_OP_FEATURE_SEMANTIC_V3, 0, 0u};
        return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
      }
      error = {1u, NEO_HIP_OP_FEATURE_SEMANTIC_V3, 1, 0u};
      return clean_rejection(NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1);
    }
    if (normalization) {
      // Pack's control word is already proven zero. The normalizer ORs its own
      // 2|4|8 semantic bits into that same retained word on the same stream.
      status = ops.normalize_feature_store(stream, rows, columns, values_address,
          validity_address, sizes.validity, *normalization, norm_sizes.padded_rows,
          device[5], norm_sizes.sort_bytes / 8, device[6], norm_sizes.fit_bytes / 8,
          device[3], error);
      if (status != 0) return status;
      status = read(ops, stream, temporary[3], reinterpret_cast<uint8_t*>(&control), 4, error);
      if (status == NEO_HIP_RUNTIME_CAPACITY_V1) return clean_rejection(status);
      if (status != 0) return status;
      if (control != 0) {
        if ((control & ~uint32_t{14}) != 0) {
          error = {1u, NEO_HIP_OP_NORMALIZATION_SEMANTIC_V3, 0, 0u};
          return NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1;
        }
        error = {1u, NEO_HIP_OP_NORMALIZATION_SEMANTIC_V3, static_cast<int32_t>(control), 0u};
        return clean_rejection(NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1);
      }
      buffers.at(temporary[6]).initialized = true;
      status = read(ops, stream, temporary[6], reinterpret_cast<uint8_t*>(fit_words.data()),
                    norm_sizes.fit_bytes, error);
      if (status == NEO_HIP_RUNTIME_CAPACITY_V1) return clean_rejection(status);
      if (status != 0) return status;
      buffers.at(temporary[5]).initialized = true;
      status = read_prefix(ops, stream, temporary[5], fit_digest, sizeof(fit_digest), error);
      if (status == NEO_HIP_RUNTIME_CAPACITY_V1) return clean_rejection(status);
      if (status != 0) return status;
    }
    // The final content root now sees the actual transformed values/validity.
    status = ops.feature_store_merkle(stream, rows, columns, timestamp_address, device[0],
        values_address, validity_address, device[1], device[2], sizes.leaves, device[4], error);
    if (status != 0) return status;
    buffers.at(temporary[4]).initialized = true;
    uint8_t root[32]{};
    status = read(ops, stream, temporary[4], root, sizeof(root), error);
    if (status == NEO_HIP_RUNTIME_CAPACITY_V1) return clean_rejection(status);
    if (status != 0) return status;
    status = cleanup();
    if (status != 0) return status;
    buffers.at(values_key).initialized = true;
    buffers.at(validity_key).initialized = true;
    buffers.at(values_key).write_sealed = true;
    buffers.at(validity_key).write_sealed = true;
    buffers.at(timestamps_key).write_sealed = true;
    receipt = {4u, 2u, rows, columns, sizes.values, sizes.validity, sizes.transient,
               sizes.metadata, 0u, normalization ? 5u : 2u,
               normalization ? 72u + norm_sizes.fit_bytes : 36u, {}, 0u, 0u, 0u, {}};
    std::memcpy(receipt.root, root, sizeof(root));
    if (normalization) {
      receipt.normalization_training_start = normalization->training_row_start;
      receipt.normalization_training_end = normalization->training_row_end;
      receipt.normalization_fit_word_count = norm_sizes.fit_bytes / 8;
      std::memcpy(receipt.fit_metadata_digest, fit_digest, sizeof(fit_digest));
      std::memcpy(host_fit_words, fit_words.data(), static_cast<size_t>(norm_sizes.fit_bytes));
    }
    return 0;
  }
};
} // namespace neoethos::hip_runtime_v1
