// Host ownership tests of the EXACT production RuntimeBuffersV1 and
// LeaseRegistryV1::with_resources. No HIP library, device, kernel, or numerical
// GPU simulation is involved. Injected Ops own ordinary host test allocations
// and a deferred copy/free queue solely to detect premature host cleanup.
// Compile independently with C++17, pthread, ASan/UBSan and strict warnings.
#include "../hip/hip_runtime_lifecycle_v1.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using neoethos::hip_runtime_v1::LeaseRegistryV1;
using neoethos::hip_runtime_v1::RuntimeBuffersV1;
using Error = NeoHipRuntimeErrorV1;
using Facts = NeoHipRuntimeFactsV1;

void require(bool condition, const char* message) {
  if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}

enum class Step { Inspect, IdentityInspect, Allocate, Pin, Upload, Download, Free, Synchronize, Unpin, Destroy,
                  SmcLaunch, MemoryInfo, PackLaunch, MerkleLaunch, NormalizeLaunch };
enum class ActionKind { Upload, Download, Free, SmcControl, PackControl, MerkleRoot, NormalizeControl };
struct Action { ActionKind kind; uint64_t stream, device, host, bytes; };

struct HostOps {
  struct Device { uint64_t stream; std::vector<uint8_t> bytes; bool freeing = false; };
  struct Pinned { uint64_t bytes; std::unique_ptr<uint8_t[]> data; };
  std::unordered_map<uint64_t, Device> devices;
  std::unordered_map<uint64_t, Pinned> pinned;
  std::unordered_map<uint64_t, int32_t> streams;
  std::vector<Action> pending;
  std::vector<Step> trace;
  uint64_t next_stream = 100, next_device = 1000;
  int32_t selected = -1;
  bool fail = false;
  Step failure = Step::Allocate;
  bool capacity_failure = false;
  unsigned failure_occurrence = 1, matching_failures = 0;
  uint32_t smc_semantic = 0;
  uint32_t pack_semantic = 0;
  uint32_t normalization_semantic = 0;
  uint64_t available_bytes = 4096;
  std::vector<uint8_t> expected_metadata;
  unsigned frees = 0, unpins = 0, destroys = 0;
  unsigned memory_queries = 0, identity_queries = 0;

  int32_t status(Step step, uint32_t operation, Error& error) {
    trace.push_back(step);
    if (!fail || failure != step) return 0;
    if (++matching_failures != failure_occurrence) return 0;
    error = {1u, operation, 719, 0u};
    return capacity_failure ? NEO_HIP_RUNTIME_CAPACITY_V1 : NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }
  int32_t select_device(int32_t ordinal, Error&) { selected = ordinal; return 0; }
  int32_t create_stream(uint64_t& result, Error&) {
    result = next_stream++;
    streams.emplace(result, selected);
    return 0;
  }
  void fill_identity(int32_t ordinal, uint64_t stream, Facts& facts) {
    require(streams.at(stream) == ordinal, "inspect uses the lease's own device/stream");
    facts = {};
    facts.device_ordinal = ordinal;
    facts.runtime_version = 70200000;
    facts.driver_version = 70200000;
    facts.warp_size = 64;
    facts.uuid[0] = static_cast<uint8_t>(ordinal + 1);
    facts.stream_handle = stream;
    facts.stream_id = stream + 1000;
    facts.total_memory_bytes = 8192;
    facts.current_pool_handle = 20;
    facts.default_pool_handle = 20;
    constexpr char arch[] = "gfx942";
    std::memcpy(facts.architecture, arch, sizeof(arch));
  }
  int32_t inspect(int32_t ordinal, uint64_t stream, Facts& facts, Error& error) {
    ++memory_queries;
    fill_identity(ordinal, stream, facts);
    facts.free_memory_bytes = 4096;
    facts.pool_reserved_bytes = 1024;
    facts.pool_used_bytes = 512;
    return status(Step::Inspect, NEO_HIP_OP_IDENTITY_V1, error);
  }
  int32_t inspect_identity(int32_t ordinal, uint64_t stream, Facts& facts, Error& error) {
    ++identity_queries;
    const auto retained_total = facts.total_memory_bytes;
    fill_identity(ordinal, stream, facts);
    facts.total_memory_bytes = retained_total;
    // No scripted capacity observation occurs in the identity-only operation.
    require(facts.free_memory_bytes == 0 && facts.pool_reserved_bytes == 0 && facts.pool_used_bytes == 0,
            "identity-only facts cannot masquerade as a fresh capacity snapshot");
    return status(Step::IdentityInspect, NEO_HIP_OP_IDENTITY_V1, error);
  }
  int32_t allocate(uint64_t stream, uint64_t bytes, uint64_t& result, Error& error) {
    result = 0;
    const auto code = status(Step::Allocate, NEO_HIP_OP_DEVICE_ALLOCATE_V1, error);
    if (code == NEO_HIP_RUNTIME_CAPACITY_V1) return code;
    require(streams.count(stream) != 0 && bytes <= 4096, "bounded allocation on live stream");
    result = next_device++;
    devices.emplace(result, Device{stream, std::vector<uint8_t>(static_cast<size_t>(bytes), 0xcd), false});
    // Non-capacity errors may have acquired a handle, testing retention.
    return code;
  }
  int32_t pin(uint64_t bytes, uint64_t& result, Error& error) {
    result = 0;
    const auto code = status(Step::Pin, NEO_HIP_OP_HOST_PIN_V1, error);
    if (code == NEO_HIP_RUNTIME_CAPACITY_V1) return code;
    auto allocation = std::make_unique<uint8_t[]>(static_cast<size_t>(bytes));
    std::fill_n(allocation.get(), static_cast<size_t>(bytes), uint8_t{0xcc});
    result = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(allocation.get()));
    pinned.emplace(result, Pinned{bytes, std::move(allocation)});
    return code;
  }
  int32_t upload(uint64_t stream, uint64_t device, uint64_t host, uint64_t bytes, Error& error) {
    require(devices.at(device).stream == stream && pinned.at(host).bytes == bytes,
            "upload references same-stream device and retained native host staging");
    pending.push_back({ActionKind::Upload, stream, device, host, bytes});
    // A failure may follow accepted submission; never release its staging early.
    return status(Step::Upload, NEO_HIP_OP_UPLOAD_V1, error);
  }
  int32_t download(uint64_t stream, uint64_t host, uint64_t device, uint64_t bytes, Error& error) {
    require(devices.at(device).stream == stream && pinned.at(host).bytes == bytes,
            "download uses native-owned staging, not the borrowed caller destination");
    pending.push_back({ActionKind::Download, stream, device, host, bytes});
    return status(Step::Download, NEO_HIP_OP_DOWNLOAD_V1, error);
  }
  int32_t free(uint64_t stream, uint64_t device, Error& error) {
    auto& allocation = devices.at(device);
    require(allocation.stream == stream && !allocation.freeing, "free is queued exactly once on its owner");
    ++frees;
    const auto code = status(Step::Free, NEO_HIP_OP_DEVICE_FREE_V1, error);
    if (code != 0) return code;
    allocation.freeing = true;
    pending.push_back({ActionKind::Free, stream, device, 0, 0});
    return 0;
  }
  int32_t smc_parent(uint64_t stream, uint64_t rows, const uint64_t* pointers, Error& error) {
    require(rows == 2, "only bounded SMC control fixtures are dispatched by host tests");
    for (size_t i = 0; i < 12; ++i)
      require(devices.at(pointers[i]).stream == stream, "SMC uses only its lease's stream/storage");
    // No SMC math is simulated. This deferred fixture supplies only the
    // producer's error/hash control protocol to test actual ownership policy.
    pending.push_back({ActionKind::SmcControl, stream, pointers[11], pointers[10], 0});
    return status(Step::SmcLaunch, NEO_HIP_OP_SMC_LAUNCH_V3, error);
  }
  int32_t available_memory(uint64_t& bytes, Error& error) {
    bytes = available_bytes;
    return status(Step::MemoryInfo, NEO_HIP_OP_MEMORY_INFO_V1, error);
  }
  int32_t pack_feature_store(uint64_t stream, uint64_t rows, uint64_t columns,
      uint64_t metadata, uint64_t values, uint64_t validity, uint64_t validity_bytes,
      uint64_t control, Error& error) {
    require((rows == 2 || rows == 100) && columns == 2, "host pack control fixture is bounded, not a numerical implementation");
    for (auto address : {metadata, values, validity, control})
      require(devices.at(address).stream == stream, "pack owns every address on the same stream");
    require(devices.at(values).bytes.size() == rows * 16 && devices.at(validity).bytes.size() == validity_bytes &&
                validity_bytes == ((rows * 2 + 7) / 8) * 4 && devices.at(control).bytes.size() == 4,
            "pack passes exact native extents");
    pending.push_back({ActionKind::PackControl, stream, control, metadata, 0});
    return status(Step::PackLaunch, NEO_HIP_OP_FEATURE_PACK_V3, error);
  }
  int32_t feature_store_merkle(uint64_t stream, uint64_t rows, uint64_t columns,
      uint64_t timestamps, uint64_t metadata, uint64_t values, uint64_t validity,
      uint64_t scratch_a, uint64_t scratch_b, uint64_t leaves, uint64_t root, Error& error) {
    require((rows == 2 || rows == 100) && columns == 2 && leaves == 3 && pending.empty() &&
                pack_semantic == 0 && normalization_semantic == 0,
            "Merkle only follows completed zero pack control");
    for (auto address : {timestamps, metadata, values, validity, scratch_a, scratch_b, root})
      require(devices.at(address).stream == stream, "Merkle retains same-stream actual resources");
    require(devices.at(scratch_a).bytes.size() == 96 && devices.at(scratch_b).bytes.size() == 96 &&
                devices.at(root).bytes.size() == 32, "both exact leaf scratch extents are present");
    pending.push_back({ActionKind::MerkleRoot, stream, root, 0, 0});
    return status(Step::MerkleLaunch, NEO_HIP_OP_FEATURE_MERKLE_V3, error);
  }
  int32_t normalize_feature_store(uint64_t stream, uint64_t rows, uint64_t columns,
      uint64_t values, uint64_t validity, uint64_t validity_bytes,
      const NeoHipFeatureNormalizationV3& normalization, uint64_t padded_rows,
      uint64_t sort, uint64_t sort_slots, uint64_t fits, uint64_t fit_words,
      uint64_t control, Error& error) {
    require(rows == 100 && columns == 2 && pending.empty() && pack_semantic == 0,
            "normalizer requires completed pack proof before its control fixture");
    for (auto address : {values, validity, sort, fits, control})
      require(devices.at(address).stream == stream, "normalizer receives owned same-stream resources");
    require(validity_bytes == 100 && padded_rows == 128 && sort_slots == 256 && fit_words == 12 &&
                devices.at(sort).bytes.size() == 2048 && devices.at(fits).bytes.size() == 96 &&
                normalization.training_row_start == 0 && normalization.training_row_end == 80 &&
                normalization.column_mode_count == 2 && normalization.column_modes[0] == 0 &&
                normalization.column_modes[1] == 1, "actual canonical interval/modes/extents reach the producer");
    // Control/fit metadata only: no host normalization or arithmetic oracle.
    pending.push_back({ActionKind::NormalizeControl, stream, control, sort, fits});
    return status(Step::NormalizeLaunch, NEO_HIP_OP_FEATURE_NORMALIZE_V3, error);
  }
  int32_t synchronize(uint64_t stream, Error& error) {
    const auto code = status(Step::Synchronize, NEO_HIP_OP_SYNCHRONIZE_V1, error);
    if (code != 0) return code;
    for (auto item = pending.begin(); item != pending.end();) {
      if (item->stream != stream) { ++item; continue; }
      if (item->kind == ActionKind::Free) {
        require(devices.erase(item->device) == 1, "deferred free executes once");
      } else if (item->kind == ActionKind::SmcControl) {
        auto& error_bytes = devices.at(item->device).bytes;
        auto& hashes = devices.at(item->host).bytes;
        require(error_bytes.size() == 4 && hashes.size() == 96, "SMC control extents are exact");
        std::memcpy(error_bytes.data(), &smc_semantic, 4);
        if (smc_semantic == 0)
          for (size_t i = 0; i < 96; ++i) hashes[i] = static_cast<uint8_t>(i + 1);
      } else if (item->kind == ActionKind::PackControl) {
        require(devices.at(item->host).bytes == expected_metadata,
                "actual completed metadata upload preserves four ordered address/offset arrays and names");
        std::memcpy(devices.at(item->device).bytes.data(), &pack_semantic, 4);
      } else if (item->kind == ActionKind::MerkleRoot) {
        auto& root = devices.at(item->device).bytes;
        for (size_t i = 0; i < root.size(); ++i) root[i] = static_cast<uint8_t>(i + 1);
      } else if (item->kind == ActionKind::NormalizeControl) {
        std::memcpy(devices.at(item->device).bytes.data(), &normalization_semantic, 4);
        if (normalization_semantic == 0) {
          const uint64_t words[12]{0, 80, 0, 0x3ff0000000000000ULL, 80, 0,
                                  0, 80, 0, 0x3ff0000000000000ULL, 80, 0};
          std::memcpy(devices.at(item->bytes).bytes.data(), words, sizeof(words));
          auto& scratch = devices.at(item->host).bytes;
          for (size_t i = 0; i < 32; ++i) scratch[i] = static_cast<uint8_t>(128 + i);
        }
      } else {
        auto& device = devices.at(item->device).bytes;
        auto& host = pinned.at(item->host);
        require(host.bytes == item->bytes && device.size() >= item->bytes &&
                    (item->kind != ActionKind::Upload || device.size() == item->bytes),
                "queued upload is exact and metadata-prefix read is bounded");
        if (item->kind == ActionKind::Upload)
          std::memcpy(device.data(), host.data.get(), static_cast<size_t>(item->bytes));
        else std::memcpy(host.data.get(), device.data(), static_cast<size_t>(item->bytes));
      }
      item = pending.erase(item);
    }
    return 0;
  }
  int32_t unpin(uint64_t host, Error& error) {
    require(std::none_of(pending.begin(), pending.end(), [=](const Action& item) {
      return item.host == host;
    }), "host staging cannot be unpinned while a copy still references it");
    ++unpins;
    const auto code = status(Step::Unpin, NEO_HIP_OP_HOST_UNPIN_V1, error);
    if (code != 0) return code;
    require(pinned.erase(host) == 1, "host staging unpinned exactly once");
    return 0;
  }
  int32_t destroy_stream(uint64_t stream, Error& error) {
    require(std::none_of(pending.begin(), pending.end(), [=](const Action& item) {
      return item.stream == stream;
    }), "stream destruction follows completion of queued copies/frees");
    require(std::none_of(devices.begin(), devices.end(), [=](const auto& item) {
      return item.second.stream == stream;
    }), "forgotten buffers are freed before stream destruction");
    ++destroys;
    require(streams.erase(stream) == 1, "destroy stream exactly once");
    return status(Step::Destroy, NEO_HIP_OP_DESTROY_STREAM_V1, error);
  }
  // Only the fixture owns these ordinary host allocations. Their final RAII
  // cleanup makes sanitizers useful without claiming that a quarantined native
  // HIP owner can free uncertain resources. No pending queue is run at teardown.
};

struct Fixture {
  HostOps ops;
  LeaseRegistryV1<HostOps> registry{ops};
  Error error{};
  uint64_t create_lease(int32_t ordinal = 0) {
    uint64_t lease = 0;
    Facts facts{};
    require(registry.create(ordinal, &lease, &facts, &error) == 0 && lease != 0,
            "real production registry issues the test lease key");
    return lease;
  }
  int32_t create_buffer(uint64_t lease, uint64_t bytes, const uint8_t* input, uint64_t& key) {
    return registry.with_resources(lease, &error, [&](auto& buffers, auto stream, const auto&) {
      return buffers.create(ops, stream, bytes, input, &key, error);
    });
  }
  int32_t release(uint64_t lease, uint64_t key) {
    return registry.with_resources(lease, &error, [&](auto& buffers, auto stream, const auto&) {
      return buffers.release(ops, stream, key, error);
    });
  }
  int32_t read(uint64_t lease, uint64_t key, uint8_t* output, uint64_t bytes) {
    return registry.with_resources(lease, &error, [&](auto& buffers, auto stream, const auto&) {
      return buffers.read(ops, stream, key, output, bytes, error);
    });
  }
  void quarantine_does_not_retry(uint64_t lease) {
    ops.fail = false;
    const auto calls = ops.trace.size();
    bool called = false;
    require(registry.with_resources(lease, &error, [&](auto&, auto, const auto&) {
      called = true; return int32_t{0};
    }) == NEO_HIP_RUNTIME_QUARANTINED_V1 && !called, "quarantined resources never enter callback");
    require(registry.synchronize(lease, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
                registry.close(lease, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
                ops.trace.size() == calls, "no backend cleanup/retry after quarantine");
  }
};

void upload_staging_lives_until_completion() {
  Fixture f;
  const auto lease = f.create_lease();
  std::array<uint8_t, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
  const auto expected = input;
  uint64_t key = 0;
  require(f.create_buffer(lease, input.size(), input.data(), key) == 0 && key != 0,
          "uploaded buffer publishes a key only after copy submission");
  require(f.ops.pinned.size() == 1 && f.ops.unpins == 0 && f.ops.pending.size() == 1,
          "successful submission retains staging until actual completion");
  input.fill(42);
  require(f.registry.synchronize(lease, &f.error) == 0 && f.ops.pinned.empty() && f.ops.unpins == 1,
          "completion releases staging exactly once");
  std::array<uint8_t, 8> output{};
  require(f.read(lease, key, output.data(), output.size()) == 0 && output == expected,
          "staged input is independent of later caller mutation");
  require(f.registry.close(lease, &f.error) == 0 && f.ops.devices.empty() && f.ops.pinned.empty(),
          "successful read and forgotten device buffer close without leaks");
}

void pure_capacity_refusal_is_reusable(bool pinned_oom) {
  Fixture f;
  const auto lease = f.create_lease();
  f.ops.fail = true;
  f.ops.failure = pinned_oom ? Step::Pin : Step::Allocate;
  f.ops.capacity_failure = true;
  uint64_t key = 999;
  std::array<uint8_t, 8> input{};
  require(f.create_buffer(lease, input.size(), input.data(), key) == NEO_HIP_RUNTIME_CAPACITY_V1 && key == 0,
          "unambiguous OOM has no published buffer key and does not quarantine");
  require(f.ops.pinned.empty(), "capacity refusal leaves no host staging");
  require(f.ops.frees == (pinned_oom ? 1u : 0u), "pin OOM queues device cleanup exactly once");
  f.ops.fail = false;
  require(f.create_buffer(lease, input.size(), input.data(), key) == 0 && key > 1,
          "healthy retry uses a fresh key without reviving the rejected allocation");
  require(f.registry.close(lease, &f.error) == 0 && f.ops.devices.empty() && f.ops.pinned.empty(),
          "capacity failure and retry leave no forgotten resources");
}

void queued_free_keeps_pending_upload_and_is_not_repeated() {
  Fixture f;
  const auto lease = f.create_lease();
  const std::array<uint8_t, 8> input{};
  uint64_t key = 0;
  require(f.create_buffer(lease, input.size(), input.data(), key) == 0, "upload submits");
  require(f.release(lease, key) == 0 && f.ops.pinned.size() == 1 && f.ops.frees == 1,
          "queued free does not unpin in-flight upload staging");
  require(f.release(lease, key) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 && f.ops.frees == 1,
          "second free cannot enqueue duplicate work");
  require(f.registry.synchronize(lease, &f.error) == 0 && f.ops.devices.empty() && f.ops.pinned.empty(),
          "stream completes upload then free before staging disappears");
  require(f.registry.close(lease, &f.error) == 0 && f.ops.frees == 1, "terminal close does not repeat queued free");
}

void failed_transfer_retains_owned_staging(Step step) {
  Fixture f;
  const auto lease = f.create_lease();
  const std::array<uint8_t, 8> input{8, 7, 6, 5, 4, 3, 2, 1};
  uint64_t key = 0;
  if (step == Step::Upload) {
    f.ops.fail = true; f.ops.failure = step;
    require(f.create_buffer(lease, input.size(), input.data(), key) == NEO_HIP_RUNTIME_QUARANTINED_V1 && key == 0,
            "accepted-before-error upload publishes no buffer and quarantines");
  } else {
    require(f.create_buffer(lease, input.size(), input.data(), key) == 0 &&
                f.registry.synchronize(lease, &f.error) == 0, "test starts with completed input upload");
    std::array<uint8_t, 8> destination{};
    destination.fill(91);
    f.ops.fail = true; f.ops.failure = step;
    require(f.read(lease, key, destination.data(), destination.size()) == NEO_HIP_RUNTIME_QUARANTINED_V1,
            "read submission or synchronization failure quarantines");
    require(std::all_of(destination.begin(), destination.end(), [](uint8_t byte) { return byte == 91; }),
            "failed read never exposes partial device bytes to borrowed caller storage");
    require(f.ops.pending.back().host != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination.data())),
            "deferred read references retained native staging, not expired stack storage");
  }
  require(f.ops.pinned.size() == 1 && !f.ops.pending.empty() && f.ops.devices.size() == 1,
          "uncertain accepted copy retains device and pinned payload");
  f.quarantine_does_not_retry(lease);
}

void free_or_cleanup_failure_retains_owner(Step failed) {
  Fixture f;
  const auto lease = f.create_lease();
  std::array<uint8_t, 8> input{};
  uint64_t key = 0;
  require(f.create_buffer(lease, input.size(), input.data(), key) == 0, "upload submitted before failure");
  f.ops.fail = true; f.ops.failure = failed;
  const auto result = failed == Step::Free ? f.release(lease, key)
                                         : f.registry.synchronize(lease, &f.error);
  require(result == NEO_HIP_RUNTIME_QUARANTINED_V1, "uncertain free/unpin quarantines lease");
  require(f.ops.pinned.size() == 1 && f.ops.devices.size() == 1 && f.ops.destroys == 0,
          "cleanup failure does not discard retained resources or destroy stream");
  f.quarantine_does_not_retry(lease);
}

void uninitialized_and_wrong_shape_reads_have_no_copy() {
  Fixture f;
  const auto lease = f.create_lease();
  uint64_t key = 0;
  require(f.create_buffer(lease, 8, nullptr, key) == 0, "uninitialized device buffer may be allocated");
  std::array<uint8_t, 8> output{};
  require(f.read(lease, key, output.data(), 8) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.ops.pinned.empty() && f.ops.pending.empty(),
          "uninitialized read enqueues nothing");
  // Use initialized storage for the independent extent/null checks, so the
  // uninitialized guard cannot accidentally make those negative controls pass.
  const std::array<uint8_t, 8> input{};
  uint64_t initialized = 0;
  require(f.create_buffer(lease, 8, input.data(), initialized) == 0 &&
              f.registry.synchronize(lease, &f.error) == 0, "initialized input completes");
  require(f.read(lease, initialized, output.data(), 7) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.read(lease, initialized, nullptr, 8) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.ops.pinned.empty() && f.ops.pending.empty(),
          "wrong-size and null-destination reads independently enqueue nothing");
  require(f.registry.close(lease, &f.error) == 0, "pure input refusal leaves healthy owner closable");
}

void partial_allocation_error_and_readback_capacity_have_distinct_ownership() {
  Fixture f;
  const auto lease = f.create_lease();
  f.ops.fail = true; f.ops.failure = Step::Allocate;
  uint64_t key = 999;
  require(f.create_buffer(lease, 8, nullptr, key) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              key == 0 && f.ops.devices.size() == 1 && f.ops.frees == 0,
          "non-capacity allocation failure with partial handle is retained, not treated as clean OOM");
  f.quarantine_does_not_retry(lease);

  Fixture g;
  const auto second = g.create_lease();
  const std::array<uint8_t, 8> input{9};
  std::array<uint8_t, 8> output{};
  require(g.create_buffer(second, 8, input.data(), key) == 0 &&
              g.registry.synchronize(second, &g.error) == 0, "readback test starts from initialized input");
  g.ops.fail = true; g.ops.failure = Step::Pin; g.ops.capacity_failure = true;
  require(g.read(second, key, output.data(), 8) == NEO_HIP_RUNTIME_CAPACITY_V1 &&
              g.ops.pinned.empty() && g.ops.pending.empty(), "clean readback staging OOM submits no download");
  g.ops.fail = false;
  require(g.read(second, key, output.data(), 8) == 0 && output == input &&
              g.registry.close(second, &g.error) == 0, "clean readback OOM permits healthy retry and close");
}

void zero_extents_and_exhausted_ids_refuse_before_runtime() {
  Fixture f;
  const auto lease = f.create_lease();
  uint64_t key = 999;
  const auto before = f.ops.trace.size();
  require(f.create_buffer(lease, 0, nullptr, key) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 && key == 0 &&
              f.ops.trace.size() == before + 1 && f.ops.trace.back() == Step::IdentityInspect,
          "zero extent revalidates owner but invokes no allocator");
  require(f.registry.with_resources(lease, &f.error, [&](auto& resources, auto stream, const auto&) {
    resources.next_id = UINT64_MAX;
    require(resources.create(f.ops, stream, 8, nullptr, &key, f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
                key == 0 && resources.buffers.empty(), "exhausted key space refuses without wrap or allocation");
    return int32_t{0};
  }) == 0 && f.ops.devices.empty(), "exhausted resource identifiers do not fabricate a native handle");
  require(f.registry.close(lease, &f.error) == 0, "pure bounds refusal leaves owner closable");
}

void buffer_keys_are_scoped_to_the_validated_lease() {
  Fixture f;
  const auto first = f.create_lease(0), second = f.create_lease(1);
  const std::array<uint8_t, 8> a{1}, b{2};
  uint64_t first_key = 0, second_key = 0, absent_key = 0;
  require(f.create_buffer(first, 8, a.data(), first_key) == 0 &&
              f.create_buffer(first, 8, a.data(), absent_key) == 0 &&
              f.create_buffer(second, 8, b.data(), second_key) == 0,
          "two independent leases allocate their own buffers");
  require(first_key == second_key && absent_key != second_key,
          "buffer key is a local key, not fabricated globally unique pointer identity");
  std::array<uint8_t, 8> output{};
  require(f.read(second, absent_key, output.data(), 8) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1,
          "another owner's unmatched key cannot access foreign storage");
  require(f.read(second, second_key, output.data(), 8) == 0 && output == b,
          "same numeric key resolves exclusively to the supplied validated owner's storage");
  require(f.registry.close(first, &f.error) == 0 && f.registry.close(second, &f.error) == 0,
          "independent scoped resources close on their own streams");
}

void resource_callback_checks_identity_and_catches_partial_exceptions() {
  Fixture f;
  const auto lease = f.create_lease();
  bool called = false;
  require(f.registry.with_resources(0, &f.error, [&](auto&, auto, const auto&) {
    called = true; return int32_t{0};
  }) == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 && !called, "unknown owner never invokes resource callback");
  require(f.registry.with_resources(lease, nullptr, [&](auto&, auto, const auto&) {
    called = true; return int32_t{0};
  }) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 && !called, "missing error destination refuses before callback");
  f.ops.fail = true; f.ops.failure = Step::IdentityInspect;
  require(f.registry.with_resources(lease, &f.error, [&](auto&, auto, const auto&) {
    called = true; return int32_t{0};
  }) == NEO_HIP_RUNTIME_QUARANTINED_V1 && !called, "live identity failure prevents resource use");
  f.quarantine_does_not_retry(lease);

  Fixture g;
  const auto second = g.create_lease();
  uint64_t key = 0;
  require(g.registry.with_resources(second, &g.error, [&](auto& resources, auto stream, const auto&) -> int32_t {
    const auto code = resources.create(g.ops, stream, 8, nullptr, &key, g.error);
    require(code == 0 && key != 0, "partial callback acquires a genuine core resource");
    throw std::runtime_error("injected host failure after acquisition");
  }) == NEO_HIP_RUNTIME_QUARANTINED_V1 && g.error.operation == NEO_HIP_OP_HOST_ALLOCATION_V1,
          "partial host exception quarantines instead of exposing reusable state");
  require(g.ops.devices.size() == 1 && g.ops.frees == 0, "partial exception retains acquired allocation");
  g.quarantine_does_not_retry(second);
}

void session_requires_exact_eight_owned_extents() {
  Fixture f;
  const auto lease = f.create_lease();
  constexpr uint64_t rows = 2;
  std::array<uint64_t, 8> keys{};
  const std::array<uint8_t, 16> lane{};
  for (size_t index = 0; index < keys.size(); ++index) {
    const uint64_t bytes = rows * (index < 6 ? 8 : index == 6 ? 184 : 23);
    require(f.create_buffer(lease, bytes, index < 6 ? lane.data() : nullptr, keys[index]) == 0,
            "eight exact Session allocations are created through the real core");
  }
  require(f.registry.with_resources(lease, &f.error, [&](auto& resources, auto, const auto&) {
    uint64_t pointers[8]{};
    require(resources.session_inputs(rows, keys.data(), pointers), "exact Session inputs validate");
    for (size_t i = 0; i < keys.size(); ++i)
      require(pointers[i] == resources.buffers.at(keys[i]).device, "Session resolves its own stored pointer");
    require(!resources.session_inputs(0, keys.data(), pointers) &&
                !resources.session_inputs(UINT64_MAX / 184 + 1, keys.data(), pointers) &&
                !resources.session_inputs(rows, nullptr, pointers), "Session row/pointer/overflow guards refuse");
    for (size_t i = 0; i < keys.size(); ++i) {
      auto& buffer = resources.buffers.at(keys[i]);
      const auto saved_bytes = buffer.bytes;
      --buffer.bytes;
      require(!resources.session_inputs(rows, keys.data(), pointers), "every one of eight sizes is checked");
      buffer.bytes = saved_bytes;
      const auto saved_device = buffer.device;
      buffer.device = 0;
      require(!resources.session_inputs(rows, keys.data(), pointers), "every one of eight handles must be live");
      buffer.device = saved_device;
      buffer.release_requested = true;
      require(!resources.session_inputs(rows, keys.data(), pointers), "Session cannot use logically released storage");
      buffer.release_requested = false;
      if (i < 6) {
        buffer.initialized = false;
        require(!resources.session_inputs(rows, keys.data(), pointers), "each input lane must be initialized");
        buffer.initialized = true;
      } else {
        buffer.borrowers = 1;
        require(!resources.session_inputs(rows, keys.data(), pointers),
                "each Session output is immutable while a native borrower pins it");
        buffer.borrowers = 0;
        buffer.write_sealed = true;
        require(!resources.session_inputs(rows, keys.data(), pointers),
                "each Session output rejects permanent content seals even without borrowers");
        buffer.write_sealed = false;
      }
      auto changed = keys; changed[i] = UINT64_MAX;
      require(!resources.session_inputs(rows, changed.data(), pointers), "every key belongs to current resources");
    }
    // This test intentionally does not mark outputs initialized or invoke a
    // fake Session kernel. Only the real native launch may publish that state.
    return int32_t{0};
  }) == 0, "Session extent checks run under genuine owner validation");
  require(f.registry.close(lease, &f.error) == 0 && f.ops.frees == 8 && f.ops.pinned.empty() && f.ops.devices.empty(),
          "terminal cleanup owns all eight forgotten buffers and six upload stages");
}

uint64_t borrow(Fixture& f, uint64_t lease) {
  uint64_t token = 0;
  Facts facts{};
  require(f.registry.borrow(lease, &token, &facts, &f.error) == 0 && token != 0 && facts.lease_id == lease,
          "native core issues a borrower token bound to its real lease");
  return token;
}

void borrower_identity_is_not_a_memory_snapshot_and_busy_close_has_no_ops() {
  Fixture f;
  const auto lease = f.create_lease();
  const auto first = borrow(f, lease);
  const auto calls = f.ops.trace.size();
  require(f.registry.close(lease, &f.error) == NEO_HIP_RUNTIME_BUSY_V1 && f.ops.trace.size() == calls &&
              f.ops.destroys == 0, "live borrower refuses close before any backend observation or cleanup");
  const auto memory_queries = f.ops.memory_queries;
  Facts facts{};
  require(f.registry.borrow_query(lease, first, &facts, &f.error) == 0 && facts.lease_id == lease &&
              facts.total_memory_bytes == 8192 && facts.free_memory_bytes == 0 &&
              facts.pool_reserved_bytes == 0 && facts.pool_used_bytes == 0 &&
              f.ops.memory_queries == memory_queries && f.ops.identity_queries == 1,
          "borrow query validates identity without another memory snapshot");
  require(f.registry.borrow_release(lease, first, &f.error) == 0 &&
              f.ops.memory_queries == memory_queries, "empty borrower release is also identity-only");
  const auto retired_calls = f.ops.trace.size();
  facts.lease_id = 999;
  require(f.registry.borrow_query(lease, first, &facts, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              facts.lease_id == 0 && f.registry.borrow_release(lease, first, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.ops.trace.size() == retired_calls, "retired token never reenters the backend");
  const auto second = borrow(f, lease);
  require(second > first, "retired borrower IDs are not reused within their owning lease");
  require(f.registry.borrow_release(lease, second, &f.error) == 0 && f.registry.close(lease, &f.error) == 0,
          "lease becomes closable only after every borrower has left");
}

void two_borrowers_pin_once_and_defer_one_free() {
  Fixture f;
  const auto lease = f.create_lease();
  const std::array<uint8_t, 8> input{7};
  uint64_t key = 0;
  require(f.create_buffer(lease, 8, input.data(), key) == 0, "pending upload is retained during borrowing");
  const auto first = borrow(f, lease), second = borrow(f, lease);
  const auto memory_queries = f.ops.memory_queries;
  uint64_t address_a = 0, address_b = 0;
  require(f.registry.borrow_buffer(lease, first, key, 8, &address_a, &f.error) == 0 && address_a != 0 &&
              f.registry.borrow_buffer(lease, first, key, 8, &address_b, &f.error) == 0 && address_a == address_b &&
              f.registry.borrow_buffer(lease, second, key, 8, &address_b, &f.error) == 0 && address_a == address_b &&
              f.ops.memory_queries == memory_queries, "repeated same-borrower pin is idempotent and takes no capacity snapshot");
  require(f.registry.with_resources(lease, &f.error, [&](auto& resources, auto, const auto&) {
    require(resources.buffers.at(key).borrowers == 2, "two distinct borrower pins, not three lookups");
    return int32_t{0};
  }) == 0, "resource pin count is inspectable only through owned test seam");
  require(f.release(lease, key) == 0 && f.ops.frees == 0 && f.ops.pinned.size() == 1,
          "logical release keeps borrowed device pointer and in-flight host upload alive");
  address_b = 999;
  std::array<uint8_t, 8> output{};
  require(f.registry.borrow_buffer(lease, first, key, 8, &address_b, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              address_b == 0 && f.read(lease, key, output.data(), 8) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.release(lease, key) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 && f.ops.frees == 0,
          "late lookup/read/second release cannot reopen logically released storage");
  require(f.registry.borrow_release(lease, first, &f.error) == 0 && f.ops.frees == 0,
          "first borrower leaving cannot free another borrower's pointer");
  require(f.registry.borrow_release(lease, second, &f.error) == 0 && f.ops.frees == 1 && f.ops.pinned.size() == 1,
          "last borrower queues exactly one free without premature unpin");
  require(f.registry.close(lease, &f.error) == 0 && f.ops.frees == 1 && f.ops.devices.empty() && f.ops.pinned.empty(),
          "terminal stream completion orders original upload before deferred free and staging release");
}

void borrower_and_buffer_tuples_require_matching_lease_and_extent() {
  Fixture f;
  const auto first = f.create_lease(0), second = f.create_lease(1);
  const std::array<uint8_t, 8> input{};
  uint64_t key = 0, first_only_key = 0, uninitialized = 0, second_initialized = 0;
  require(f.create_buffer(first, 8, input.data(), key) == 0 &&
              f.create_buffer(first, 8, input.data(), first_only_key) == 0 &&
              f.create_buffer(first, 8, nullptr, uninitialized) == 0 &&
              f.create_buffer(second, 8, input.data(), second_initialized) == 0, "two leases own distinct resource maps");
  const auto a = borrow(f, first), only_first = borrow(f, first), b = borrow(f, second);
  require(a == b && only_first != b, "borrower IDs are nonreused local tokens, scoped by exact lease");
  uint64_t address = 999;
  require(f.registry.borrow_buffer(second, b, first_only_key, 8, &address, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              address == 0, "foreign unmatched buffer key exposes no address");
  require(f.registry.borrow_buffer(second, only_first, second_initialized, 8, &address, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.registry.borrow_buffer(first, a, uninitialized, 8, &address, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.registry.borrow_buffer(first, a, key, 7, &address, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.registry.borrow_buffer(first, a, key, 0, &address, &f.error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1,
          "wrong borrower, uninitialized input and wrong extents each refuse before pointer publication");
  require(f.registry.borrow_buffer(first, a, key, 8, &address, &f.error) == 0 && address != 0,
          "initialized matching lease/key/extent receives its own pointer");
  require(f.registry.borrow_release(first, a, &f.error) == 0 &&
              f.registry.borrow_release(first, only_first, &f.error) == 0 &&
              f.registry.borrow_release(second, b, &f.error) == 0 &&
              f.registry.close(first, &f.error) == 0 && f.registry.close(second, &f.error) == 0,
          "all valid ownership can be released after pure argument refusals");
}

void borrower_release_failure_quarantines_without_retry(Step failure) {
  Fixture f;
  const auto lease = f.create_lease();
  const std::array<uint8_t, 8> input{};
  uint64_t key = 0, address = 0;
  require(f.create_buffer(lease, 8, input.data(), key) == 0, "borrow release fault fixture uploads");
  const auto token = borrow(f, lease);
  require(f.registry.borrow_buffer(lease, token, key, 8, &address, &f.error) == 0 && f.release(lease, key) == 0,
          "last borrower owns deferred-release buffer");
  f.ops.fail = true; f.ops.failure = failure;
  require(f.registry.borrow_release(lease, token, &f.error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              f.ops.devices.size() == 1 && f.ops.pinned.size() == 1 && f.ops.destroys == 0,
          "failed last-pin release retains resources and quarantines");
  f.ops.fail = false;
  const auto calls = f.ops.trace.size();
  Facts facts{};
  require(f.registry.borrow_query(lease, token, &facts, &f.error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              f.registry.borrow_buffer(lease, token, key, 8, &address, &f.error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              f.registry.borrow_release(lease, token, &f.error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              f.ops.trace.size() == calls, "quarantined borrower calls never retry the backend");
  f.quarantine_does_not_retry(lease);
}
struct SmcFixture : Fixture {
  uint64_t lease = create_lease();
  std::array<uint64_t, 5> inputs{};
  std::array<uint64_t, 7> outputs{};
  std::array<uint8_t, 96> hashes{};
  SmcFixture() {
    hashes.fill(0xab);
    const std::array<uint8_t, 16> input{};
    uint64_t sizes[12]{};
    require(RuntimeBuffersV1::smc_parent_sizes(2, sizes), "bounded SMC shape is checked");
    for (size_t i = 0; i < 5; ++i)
      require(create_buffer(lease, sizes[i], input.data(), inputs[i]) == 0, "SMC initialized input created");
    for (size_t i = 0; i < 7; ++i)
      require(create_buffer(lease, sizes[i + 5], nullptr, outputs[i]) == 0, "SMC output created uninitialized");
    require(registry.synchronize(lease, &error) == 0, "input uploads completed before scripted faults");
    ops.trace.clear();
  }
  int32_t run() {
    return registry.with_resources(lease, &error, [&](auto& resources, auto stream, const auto&) {
      return resources.smc_parent(ops, stream, 2, inputs.data(), outputs.data(), hashes.data(), error);
    });
  }
  void assert_unpublished() {
    require(std::all_of(hashes.begin(), hashes.end(), [](auto byte) { return byte == 0xab; }),
            "failed SMC never publishes hashes to caller storage");
  }
  void assert_output_state(bool initialized) {
    require(registry.with_resources(lease, &error, [&](auto& resources, auto, const auto&) {
      for (auto key : outputs)
        require(resources.buffers.at(key).initialized == initialized,
                "SMC output initialization is all-or-none after the control proof");
      return int32_t{0};
    }) == 0, "healthy SMC output state inspected through actual core");
  }
};

void smc_shapes_aliases_and_pins_are_checked_before_dispatch() {
  SmcFixture f;
  uint64_t sizes[12]{};
  require(!RuntimeBuffersV1::smc_parent_sizes(0, sizes) &&
              RuntimeBuffersV1::smc_parent_sizes(uint64_t{INT32_MAX} + 1, sizes) &&
              !RuntimeBuffersV1::smc_parent_sizes(uint64_t{INT32_MAX} + 2, sizes) &&
              !RuntimeBuffersV1::smc_parent_sizes(UINT64_MAX, sizes),
          "SMC last-row int boundary accepts INT_MAX+1 rows and rejects the next row count");
  require(RuntimeBuffersV1::smc_parent_sizes(2, sizes) &&
              sizes[5] == 736 && sizes[6] == 92 && sizes[7] == 16 &&
              sizes[8] == 16 && sizes[9] == 22 && sizes[10] == 96 && sizes[11] == 4,
          "actual SMC output shape totals 441N+100 independently checked literal extents");
  require(f.registry.with_resources(f.lease, &f.error, [&](auto& resources, auto, const auto&) {
    uint64_t pointers[12]{};
    require(resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers),
            "all twelve genuine keys resolve with their exact sizes");
    require(!resources.smc_parent_inputs(2, nullptr, f.outputs.data(), pointers) &&
                !resources.smc_parent_inputs(2, f.inputs.data(), nullptr, pointers) &&
                !resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), nullptr),
            "null shape inputs are rejected before dispatch");
    for (size_t i = 0; i < 12; ++i) {
      auto& buffer = resources.buffers.at(i < 5 ? f.inputs[i] : f.outputs[i - 5]);
      --buffer.bytes;
      require(!resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers), "each exact SMC extent checked");
      ++buffer.bytes;
      const auto saved = buffer.device; buffer.device = 0;
      require(!resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers), "each SMC handle must be live");
      buffer.device = saved;
      buffer.release_requested = true;
      require(!resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers), "no logically released SMC resource used");
      buffer.release_requested = false;
      if (i < 5) {
        buffer.initialized = false;
        require(!resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers), "all SMC inputs initialized");
        buffer.initialized = true;
      } else {
        buffer.borrowers = 1;
        require(!resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers), "all seven outputs reject borrowed writes");
        buffer.borrowers = 0;
        buffer.write_sealed = true;
        require(!resources.smc_parent_inputs(2, f.inputs.data(), f.outputs.data(), pointers),
                "all seven SMC outputs reject permanent content seals without borrowers");
        buffer.write_sealed = false;
      }
    }
    auto aliased = f.inputs; aliased[1] = aliased[0];
    require(!resources.smc_parent_inputs(2, aliased.data(), f.outputs.data(), pointers), "same-sized input alias rejected");
    auto aliased_outputs = f.outputs; aliased_outputs[3] = aliased_outputs[2];
    require(!resources.smc_parent_inputs(2, f.inputs.data(), aliased_outputs.data(), pointers), "same-sized output alias rejected");
    aliased_outputs = f.outputs; aliased_outputs[2] = f.inputs[0];
    require(!resources.smc_parent_inputs(2, f.inputs.data(), aliased_outputs.data(), pointers), "input/output alias rejected");
    aliased = f.inputs; aliased[0] = UINT64_MAX;
    require(!resources.smc_parent_inputs(2, aliased.data(), f.outputs.data(), pointers), "unknown key rejected");
    return int32_t{0};
  }) == 0, "pure SMC guards execute under the real lease policy");
  require(std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::SmcLaunch) == 0,
          "shape-only tests enqueue no numerical producer");
  require(f.registry.close(f.lease, &f.error) == 0, "shape refusals retain healthy owner");
}

void smc_success_requires_ordered_error_then_hash_completion() {
  SmcFixture f;
  require(f.run() == 0, "SMC control success completes");
  const std::vector<Step> expected{Step::IdentityInspect, Step::Pin, Step::SmcLaunch,
      Step::Download, Step::Synchronize, Step::Unpin,
      Step::Pin, Step::Download, Step::Synchronize, Step::Unpin};
  require(f.ops.trace == expected, "exact error-zero proof precedes separate hash copy/completion");
  for (size_t i = 0; i < f.hashes.size(); ++i)
    require(f.hashes[i] == static_cast<uint8_t>(i + 1), "completed native control bytes published unchanged");
  f.assert_output_state(true);
  const auto borrower = borrow(f, f.lease);
  uint64_t address = 0;
  require(f.registry.borrow_buffer(f.lease, borrower, f.outputs[2], 16, &address, &f.error) == 0,
          "completed SMC output can be genuinely borrowed");
  const auto launches = std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::SmcLaunch);
  require(f.run() == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::SmcLaunch) == launches,
          "a second SMC call cannot overwrite genuinely borrowed output");
  require(f.registry.borrow_release(f.lease, borrower, &f.error) == 0 &&
              f.registry.close(f.lease, &f.error) == 0 && f.ops.pinned.empty(), "SMC success releases cleanly");
}

void smc_semantic_refusal_publishes_nothing_and_keeps_owner(uint32_t code) {
  SmcFixture f;
  require(f.run() == 0, "semantic rerun starts with previously initialized output");
  f.assert_output_state(true);
  f.hashes.fill(0xab);
  f.ops.trace.clear();
  f.ops.smc_semantic = code;
  require(f.run() == NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1 &&
              f.error.operation == NEO_HIP_OP_SMC_SEMANTIC_V3 &&
              f.error.backend_status == static_cast<int32_t>(code), "semantic result is not a HIP API error");
  f.assert_unpublished(); f.assert_output_state(false);
  require(std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Download) == 1 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Synchronize) == 1 && f.ops.pinned.empty(),
          "semantic failure reads only four bytes, never the potentially unwritten hashes");
  f.ops.smc_semantic = 0;
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0,
          "verified semantic rejection leaves the owner reusable without publishing failed output");
}

void smc_runtime_failure_retains_native_staging(Step step, unsigned occurrence) {
  SmcFixture f;
  f.ops.fail = true; f.ops.failure = step; f.ops.failure_occurrence = occurrence;
  require(f.run() == NEO_HIP_RUNTIME_QUARANTINED_V1, "SMC runtime uncertainty quarantines");
  f.assert_unpublished();
  require(!f.ops.pinned.empty() && f.ops.devices.size() == 12 && f.ops.destroys == 0,
          "failed control transfer/launch/cleanup retains buffer-owned staging and all device resources");
  f.quarantine_does_not_retry(f.lease);
}

void smc_pinned_capacity_refusal_can_retry(unsigned occurrence) {
  SmcFixture f;
  f.ops.fail = true; f.ops.failure = Step::Pin;
  f.ops.failure_occurrence = occurrence; f.ops.capacity_failure = true;
  require(f.run() == NEO_HIP_RUNTIME_CAPACITY_V1, "unambiguous control pin OOM remains distinct");
  f.assert_unpublished(); f.assert_output_state(false);
  require(f.ops.pinned.empty(), "clean pin OOM leaves no orphan staging");
  f.ops.fail = false;
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0, "pin OOM safely retries without stale initialized flags");
}

void smc_unknown_device_error_is_not_a_valid_semantic_receipt() {
  SmcFixture f;
  f.ops.smc_semantic = UINT32_MAX;
  require(f.run() == NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1, "unknown control code fails closed without narrowing");
  f.assert_unpublished(); f.quarantine_does_not_retry(f.lease);
}

struct PackFixture : Fixture {
  uint64_t rows;
  uint64_t lease = create_lease(), timestamps = 0, values = 0, validity = 0;
  uint64_t source = 0, codes = 0;
  std::array<NeoHipFeatureColumnV1, 2> columns{};
  std::array<uint64_t, 3> offsets{0, 1, 2};
  std::array<uint8_t, 2> names{'a', 'b'};
  NeoHipFeatureStoreReceiptV4 receipt{};
  explicit PackFixture(uint64_t row_count = 2) : rows(row_count) {
    const std::vector<uint8_t> inputs(static_cast<size_t>((rows * 2 + 1) * 8));
    require(create_buffer(lease, (rows * 2 + 1) * 8, inputs.data(), source) == 0 &&
                create_buffer(lease, rows * 2 + 1, inputs.data(), codes) == 0 &&
                create_buffer(lease, rows * 8, inputs.data(), timestamps) == 0 &&
                create_buffer(lease, rows * 16, nullptr, values) == 0 &&
                create_buffer(lease, ((rows * 2 + 7) / 8) * 4, nullptr, validity) == 0,
            "pack fixture uses real exact output buffers and initialized producer subranges");
    columns = {{{source, 1, codes, 1}, {source, rows + 1, codes, rows + 1}}};
    require(registry.synchronize(lease, &error) == 0, "pack source uploads complete");
    require(registry.with_resources(lease, &error, [&](auto& resources, auto, const auto&) {
      const uint64_t table[11]{resources.buffers.at(source).device, resources.buffers.at(source).device,
          1, rows + 1, resources.buffers.at(codes).device, resources.buffers.at(codes).device,
          1, rows + 1, 0, 1, 2};
      ops.expected_metadata.resize(90);
      std::memcpy(ops.expected_metadata.data(), table, sizeof(table));
      std::memcpy(ops.expected_metadata.data() + sizeof(table), names.data(), names.size());
      return int32_t{0};
    }) == 0, "independent exact metadata expectation binds actual resource addresses");
    ops.trace.clear();
  }
  int32_t run(uint64_t reserve = 0, const NeoHipFeatureNormalizationV3* normalization = nullptr,
               uint64_t* host_fits = nullptr, uint64_t fit_capacity = 0) {
    return registry.with_resources(lease, &error, [&](auto& resources, auto stream, const auto&) {
      return resources.pack_feature_store(ops, stream, rows, 2, columns.data(), timestamps,
          offsets.data(), names.data(), names.size(), reserve, normalization,
          values, validity, host_fits, fit_capacity, receipt, error);
    });
  }
  void unpublished() {
    const NeoHipFeatureStoreReceiptV4 empty{};
    require(std::memcmp(&receipt, &empty, sizeof(empty)) == 0,
            "failed logical pack publishes no receipt or root, even after completed readback");
  }
  void state(bool sealed) {
    require(registry.with_resources(lease, &error, [&](auto& resources, auto, const auto&) {
      require(resources.buffers.size() == 5, "all transient registry records are retired before reusable return");
      for (auto key : {values, validity})
        require(resources.buffers.at(key).initialized == sealed &&
                    resources.buffers.at(key).write_sealed == sealed, "final outputs publish all-or-none");
      require(resources.buffers.at(timestamps).write_sealed == sealed,
              "timestamp content seal is committed with the final content root");
      return int32_t{0};
    }) == 0, "healthy pack state remains inspectable through actual owner");
  }
};

void pack_shapes_and_utf8_have_independent_expected_values() {
  RuntimeBuffersV1::FeatureStoreSizesV3 sizes;
  require(RuntimeBuffersV1::feature_store_sizes(2, 2, 2, sizes) && sizes.values == 32 &&
              sizes.validity == 4 && sizes.timestamps == 16 && sizes.leaves == 3 &&
              sizes.metadata == 90 && sizes.scratch == 96 && sizes.transient == 318,
          "small plan matches independent literal exact byte census");
  require(RuntimeBuffersV1::feature_store_sizes(3, 3, 3, sizes) && sizes.values == 72 &&
              sizes.validity == 8 && sizes.metadata == 131 && sizes.scratch == 128 && sizes.transient == 423,
          "odd cell count charges whole aligned u4 words and both Merkle banks");
  require(RuntimeBuffersV1::feature_store_sizes(4097, 1, 1, sizes) && sizes.leaves == 4,
          "partial timestamp chunk has a separate leaf per feature and timestamp lane");
  for (auto bad : {uint64_t{0}, UINT64_MAX})
    require(!RuntimeBuffersV1::feature_store_sizes(bad, 2, 2, sizes) &&
                !RuntimeBuffersV1::feature_store_sizes(2, bad, 2, sizes) &&
                !RuntimeBuffersV1::feature_store_sizes(2, 2, bad, sizes), "zero/overflow dimensions reject");
  const std::vector<std::vector<uint8_t>> invalid{{0x80}, {0xc0, 0x80}, {0xe0, 0x80, 0x80},
      {0xed, 0xa0, 0x80}, {0xf4, 0x90, 0x80, 0x80}, {0xf0, 0x80, 0x80, 0x80}, {0xc2}, {0xff}};
  for (const auto& bytes : invalid)
    require(!RuntimeBuffersV1::feature_name_utf8(bytes.data(), bytes.size()), "invalid UTF-8 cannot enter content metadata");
  const uint8_t valid[]{0x61, 0xce, 0xb1, 0xe2, 0x82, 0xac, 0xf4, 0x8f, 0xbf, 0xbf};
  require(RuntimeBuffersV1::feature_name_utf8(valid, sizeof(valid)), "valid UTF-8 through U+10FFFF preserves bytes");
}

void pack_success_retires_scratch_and_seals_content() {
  PackFixture f;
  require(f.run() == 0, "logical pack succeeds only through actual production resource policy");
  require(f.receipt.abi_version == 4 && f.receipt.backend_kind == 2 && f.receipt.rows == 2 &&
              f.receipt.columns == 2 && f.receipt.value_bytes == 32 && f.receipt.validity_bytes == 4 &&
              f.receipt.metadata_upload_bytes == 90 && f.receipt.transient_device_bytes == 318 &&
              f.receipt.control == 0 && f.receipt.readback_count == 2 && f.receipt.readback_bytes == 36 &&
              f.receipt.normalization_training_start == 0 && f.receipt.normalization_training_end == 0 &&
              f.receipt.normalization_fit_word_count == 0 &&
              std::all_of(std::begin(f.receipt.fit_metadata_digest), std::end(f.receipt.fit_metadata_digest),
                          [](uint8_t byte) { return byte == 0; }),
          "receipt matches completed control protocol and independently counted extents");
  for (size_t i = 0; i < 32; ++i)
    require(f.receipt.root[i] == i + 1, "completed scripted control bytes copied without fabricating kernel parity");
  require(std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Allocate) == 5 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Download) == 2 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Synchronize) == 3 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Free) == 5 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::MemoryInfo) == 1 &&
              f.ops.pending.empty() && f.ops.pinned.empty() && f.ops.devices.size() == 5,
          "one upload and five transient allocations fully retire before success; no full feature readback");
  f.state(true);
  require(f.run() == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1, "one-shot output cannot be overwritten by another pack");
  require(f.registry.close(f.lease, &f.error) == 0, "content-sealed buffers remain normally closable");
}

void pack_invalid_metadata_ranges_and_output_authority_have_no_dispatch() {
  PackFixture f;
  const auto rejected = [&] {
    f.ops.trace.clear();
    require(f.run() == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
                f.ops.trace == std::vector<Step>{Step::IdentityInspect},
            "invalid pack checks precede capacity query, allocation, transfers, and kernels");
    f.unpublished();
  };
  f.offsets[0] = 1; rejected(); f.offsets[0] = 0;
  f.offsets[1] = 0; rejected(); f.offsets[1] = 1;
  f.offsets[2] = 3; rejected(); f.offsets[2] = 2;
  f.names[1] = 'a'; rejected(); f.names[1] = 0xff; rejected(); f.names[1] = 'b';
  f.columns[1].values_element_offset = 4; rejected(); f.columns[1].values_element_offset = UINT64_MAX;
  rejected(); f.columns[1].values_element_offset = 3;
  f.columns[1].validity_byte_offset = 4; rejected(); f.columns[1].validity_byte_offset = UINT64_MAX;
  rejected(); f.columns[1].validity_byte_offset = 3;
  f.columns[0].values_key = f.values; rejected(); f.columns[0].values_key = f.source;
  f.columns[0].validity_key = f.validity; rejected(); f.columns[0].validity_key = f.codes;
  for (auto key : {f.values, f.validity, f.timestamps, f.source, f.codes}) {
    for (unsigned mode = 0; mode < 5; ++mode) {
      // Change exactly one real stored field, then run the actual public-policy body.
      RuntimeBuffersV1::Buffer original;
      require(f.registry.with_resources(f.lease, &f.error, [&](auto& resources, auto, const auto&) {
        auto& buffer = resources.buffers.at(key); original = buffer;
        if (mode == 0) buffer.device = 0;
        if (mode == 1) buffer.release_requested = true;
        if (mode == 2) buffer.bytes = 1;
        if (mode == 3) buffer.initialized = !buffer.initialized;
        if (mode == 4 && (key == f.values || key == f.validity)) buffer.write_sealed = true;
        return int32_t{0};
      }) == 0, "single field mutation installed by test seam");
      if (mode != 4 || key == f.values || key == f.validity) rejected();
      require(f.registry.with_resources(f.lease, &f.error, [&](auto& resources, auto, const auto&) {
        resources.buffers.at(key) = original; return int32_t{0};
      }) == 0, "mutation restored without invoking a backend operation");
    }
  }
  for (auto key : {f.values, f.validity}) {
    require(f.registry.with_resources(f.lease, &f.error, [&](auto& resources, auto, const auto&) {
      resources.buffers.at(key).borrowers = 1; return int32_t{0};
    }) == 0, "output borrower installed");
    rejected();
    require(f.registry.with_resources(f.lease, &f.error, [&](auto& resources, auto, const auto&) {
      resources.buffers.at(key).borrowers = 0; return int32_t{0};
    }) == 0, "output borrower removed");
  }
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0,
          "all negative controls preserve a positive end-to-end ownership path");
}

void pack_capacity_and_semantic_refusals_retire_only_owned_temporaries() {
  PackFixture f;
  require(f.run(UINT64_MAX) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1, "reserve overflow rejects");
  f.ops.available_bytes = 317;
  require(f.run() == NEO_HIP_RUNTIME_CAPACITY_V1, "fresh native free rejects insufficient scratch");
  f.ops.available_bytes = 418;
  require(f.run(101) == NEO_HIP_RUNTIME_CAPACITY_V1, "caller reserve remains included after final output allocation");
  f.unpublished(); f.state(false);
  f.ops.available_bytes = 4096; f.ops.pack_semantic = 1; f.ops.trace.clear();
  require(f.run() == NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1 && f.error.operation == NEO_HIP_OP_FEATURE_SEMANTIC_V3 &&
              f.error.backend_status == 1, "completed invalid validity code is semantic, not HIP failure");
  require(std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Download) == 1 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::MerkleLaunch) == 0 &&
              f.ops.devices.size() == 5 && f.ops.pinned.empty(), "bad validity never launches Merkle or reads a root");
  f.unpublished(); f.state(false); f.ops.pack_semantic = 0;
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0, "clean rejection can retry with fresh output authority");
}

void pack_clean_oom_releases_transient_subset(Step step, unsigned occurrence) {
  PackFixture f;
  f.ops.fail = true; f.ops.failure = step; f.ops.failure_occurrence = occurrence; f.ops.capacity_failure = true;
  require(f.run() == NEO_HIP_RUNTIME_CAPACITY_V1, "clean allocation/pin OOM is reusable");
  f.unpublished(); f.state(false);
  require(f.ops.devices.size() == 5 && f.ops.pinned.empty() && f.ops.pending.empty(),
          "partial clean OOM retires all operation-owned memory and staging before return");
  f.ops.fail = false;
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0, "clean OOM retry retains caller buffers");
}

void pack_runtime_failure_quarantines_without_receipt(Step step, unsigned occurrence) {
  PackFixture f;
  f.ops.fail = true; f.ops.failure = step; f.ops.failure_occurrence = occurrence;
  require(f.run() == NEO_HIP_RUNTIME_QUARANTINED_V1, "pack runtime uncertainty quarantines exactly once");
  f.unpublished(); f.quarantine_does_not_retry(f.lease);
}

void pack_unknown_control_quarantines_without_merkle() {
  PackFixture f;
  f.ops.pack_semantic = UINT32_MAX;
  require(f.run() == NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::MerkleLaunch) == 0,
          "unrecognized control bytes cannot be relabeled a clean semantic outcome");
  f.unpublished(); f.quarantine_does_not_retry(f.lease);
}

struct NormalizedPackFixture : PackFixture {
  std::array<uint8_t, 2> modes{0, 1};
  NeoHipFeatureNormalizationV3 normalization{0, 80, modes.data(), 2};
  std::array<uint64_t, 12> fits;
  NormalizedPackFixture() : PackFixture(100) { fits.fill(UINT64_MAX); }
  int32_t run() { return PackFixture::run(0, &normalization, fits.data(), fits.size()); }
  void unpublished() {
    PackFixture::unpublished();
    require(std::all_of(fits.begin(), fits.end(), [](uint64_t word) { return word == UINT64_MAX; }),
            "failed normalized pack never publishes completed-but-uncommitted fit words");
  }
};

void normalized_pack_success_counts_every_readback_and_retired_extent() {
  NormalizedPackFixture f;
  require(f.run() == 0, "normalization control joins the existing complete pack operation");
  require(f.receipt.transient_device_bytes == 2462 && f.receipt.metadata_upload_bytes == 90 &&
              f.receipt.readback_count == 5 && f.receipt.readback_bytes == 168 &&
              f.receipt.normalization_training_start == 0 && f.receipt.normalization_training_end == 80 &&
              f.receipt.normalization_fit_word_count == 12,
          "318 base plus2048 sort plus96 fits; two4B controls plus96 fits plus two32B digests");
  const uint64_t expected[]{0, 80, 0, 0x3ff0000000000000ULL, 80, 0,
                            0, 80, 0, 0x3ff0000000000000ULL, 80, 0};
  require(std::equal(f.fits.begin(), f.fits.end(), expected), "actual completed fit words copied in original column order");
  for (size_t i = 0; i < 32; ++i)
    require(f.receipt.fit_metadata_digest[i] == 128 + i && f.receipt.root[i] == i + 1,
            "normalization digest and final content root are separate exact control outputs");
  require(std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Allocate) == 7 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Download) == 5 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Synchronize) == 6 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Free) == 7 &&
              f.ops.devices.size() == 5 && f.ops.pinned.empty() && f.ops.pending.empty(),
          "all seven transient buffers retire; only32B prefix of2048B sort allocation is read");
  f.state(true);
  require(f.registry.close(f.lease, &f.error) == 0, "normalization success retains ordinary sealed-owner cleanup");
}

void normalization_request_and_fit_capacity_are_checked_before_gpu_work() {
  const std::array<uint8_t, 4> modes{0, 1, 2, 3};
  NeoHipFeatureNormalizationV3 request{0, 64, modes.data(), 4};
  RuntimeBuffersV1::FeatureNormalizationSizesV3 sizes;
  require(RuntimeBuffersV1::feature_normalization_sizes(80, 4, request, sizes) &&
              sizes.padded_rows == 64 && sizes.sort_bytes == 2048 && sizes.fit_bytes == 192,
          "all four policy modes and first64-row fit have exact independent extents");
  request.training_row_end = 63;
  require(!RuntimeBuffersV1::feature_normalization_sizes(79, 4, request, sizes), "minimum64 training rows cannot be bypassed");
  NormalizedPackFixture f;
  const auto rejected = [&] {
    f.ops.trace.clear();
    require(f.run() == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
                f.ops.trace == std::vector<Step>{Step::IdentityInspect},
            "invalid normalization window/schema invokes no capacity query, allocation or producer");
    f.unpublished();
  };
  f.normalization.training_row_start = 1; rejected(); f.normalization.training_row_start = 0;
  for (auto end : {uint64_t{0}, uint64_t{79}, uint64_t{81}, uint64_t{100}, UINT64_MAX}) {
    f.normalization.training_row_end = end; rejected();
  }
  f.normalization.training_row_end = 80;
  f.normalization.column_mode_count = 1; rejected(); f.normalization.column_mode_count = 2;
  f.normalization.column_modes = nullptr; rejected(); f.normalization.column_modes = f.modes.data();
  f.modes[1] = 4; rejected(); f.modes[1] = 1;
  for (auto capacity : {uint64_t{0}, uint64_t{11}, uint64_t{13}, UINT64_MAX})
    require(f.PackFixture::run(0, &f.normalization, f.fits.data(), capacity) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1,
            "fit destination must have exactly6F words, never silently truncated");
  require(f.PackFixture::run(0, &f.normalization, nullptr, 12) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.PackFixture::run(0, nullptr, f.fits.data(), 12) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              f.PackFixture::run(0, nullptr, nullptr, 12) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1,
          "enabled and disabled fit destination states are unambiguous");
  f.unpublished(); f.state(false);
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0, "valid policy retains positive path after all negative controls");
}

void normalized_pack_semantic_error_cannot_publish_fit_or_root(uint32_t code) {
  NormalizedPackFixture f;
  f.ops.normalization_semantic = code;
  require(f.run() == NEO_HIP_RUNTIME_SEMANTIC_ERROR_V1 &&
              f.error.operation == NEO_HIP_OP_NORMALIZATION_SEMANTIC_V3 &&
              f.error.backend_status == static_cast<int32_t>(code),
          "all known2|4|8 combinations are completed semantic refusals");
  require(std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::Download) == 2 &&
              std::count(f.ops.trace.begin(), f.ops.trace.end(), Step::MerkleLaunch) == 0 &&
              f.ops.devices.size() == 5 && f.ops.pinned.empty(),
          "normalization error reads only both4B controls then retires all seven temporaries");
  f.unpublished(); f.state(false);
  f.ops.normalization_semantic = 0;
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0, "completed rejection safely reruns original pack before normalization");
}

void normalized_pack_unknown_control_does_not_authorize_a_clean_retry() {
  NormalizedPackFixture f;
  f.ops.normalization_semantic = 1; // Pack bit cannot recur after the earlier zero proof.
  require(f.run() == NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1, "unknown normalization bit quarantines");
  f.unpublished(); f.quarantine_does_not_retry(f.lease);
}

void normalized_pack_clean_oom_retains_unpublished_caller_buffers(Step step, unsigned occurrence) {
  NormalizedPackFixture f;
  f.ops.fail = true; f.ops.failure = step; f.ops.failure_occurrence = occurrence; f.ops.capacity_failure = true;
  require(f.run() == NEO_HIP_RUNTIME_CAPACITY_V1, "clean extended allocation/readback pin OOM stays reusable");
  f.unpublished(); f.state(false);
  require(f.ops.devices.size() == 5 && f.ops.pinned.empty() && f.ops.pending.empty(),
          "failed extended plan retains no orphan transient memory/staging");
  f.ops.fail = false;
  require(f.run() == 0 && f.registry.close(f.lease, &f.error) == 0, "clean extended OOM retry preserves original source contents");
}

void normalized_pack_runtime_error_keeps_fit_output_unpublished(Step step, unsigned occurrence) {
  NormalizedPackFixture f;
  f.ops.fail = true; f.ops.failure = step; f.ops.failure_occurrence = occurrence;
  require(f.run() == NEO_HIP_RUNTIME_QUARANTINED_V1, "normalization/readback/retirement uncertainty quarantines");
  f.unpublished(); f.quarantine_does_not_retry(f.lease);
}
} // namespace

int main() {
  upload_staging_lives_until_completion();
  pure_capacity_refusal_is_reusable(false);
  pure_capacity_refusal_is_reusable(true);
  queued_free_keeps_pending_upload_and_is_not_repeated();
  failed_transfer_retains_owned_staging(Step::Upload);
  failed_transfer_retains_owned_staging(Step::Download);
  failed_transfer_retains_owned_staging(Step::Synchronize);
  free_or_cleanup_failure_retains_owner(Step::Free);
  free_or_cleanup_failure_retains_owner(Step::Unpin);
  uninitialized_and_wrong_shape_reads_have_no_copy();
  partial_allocation_error_and_readback_capacity_have_distinct_ownership();
  zero_extents_and_exhausted_ids_refuse_before_runtime();
  buffer_keys_are_scoped_to_the_validated_lease();
  resource_callback_checks_identity_and_catches_partial_exceptions();
  session_requires_exact_eight_owned_extents();
  borrower_identity_is_not_a_memory_snapshot_and_busy_close_has_no_ops();
  two_borrowers_pin_once_and_defer_one_free();
  borrower_and_buffer_tuples_require_matching_lease_and_extent();
  borrower_release_failure_quarantines_without_retry(Step::Free);
  borrower_release_failure_quarantines_without_retry(Step::IdentityInspect);
  smc_shapes_aliases_and_pins_are_checked_before_dispatch();
  smc_success_requires_ordered_error_then_hash_completion();
  for (uint32_t code = 1; code <= 5; ++code)
    smc_semantic_refusal_publishes_nothing_and_keeps_owner(code);
  smc_runtime_failure_retains_native_staging(Step::SmcLaunch, 1);
  for (auto step : {Step::Pin, Step::Download, Step::Synchronize, Step::Unpin})
    for (unsigned occurrence : {1u, 2u}) smc_runtime_failure_retains_native_staging(step, occurrence);
  smc_pinned_capacity_refusal_can_retry(1);
  smc_pinned_capacity_refusal_can_retry(2);
  smc_unknown_device_error_is_not_a_valid_semantic_receipt();
  pack_shapes_and_utf8_have_independent_expected_values();
  pack_success_retires_scratch_and_seals_content();
  pack_invalid_metadata_ranges_and_output_authority_have_no_dispatch();
  pack_capacity_and_semantic_refusals_retire_only_owned_temporaries();
  pack_unknown_control_quarantines_without_merkle();
  for (unsigned occurrence = 1; occurrence <= 5; ++occurrence)
    pack_clean_oom_releases_transient_subset(Step::Allocate, occurrence);
  for (unsigned occurrence = 1; occurrence <= 3; ++occurrence)
    pack_clean_oom_releases_transient_subset(Step::Pin, occurrence);
  for (auto step : {Step::MemoryInfo, Step::Upload, Step::PackLaunch, Step::MerkleLaunch})
    pack_runtime_failure_quarantines_without_receipt(step, 1);
  for (unsigned occurrence = 1; occurrence <= 5; ++occurrence) {
    pack_runtime_failure_quarantines_without_receipt(Step::Allocate, occurrence);
    pack_runtime_failure_quarantines_without_receipt(Step::Free, occurrence);
  }
  for (auto step : {Step::Pin, Step::Synchronize, Step::Unpin})
    for (unsigned occurrence = 1; occurrence <= 3; ++occurrence)
      pack_runtime_failure_quarantines_without_receipt(step, occurrence);
  for (unsigned occurrence = 1; occurrence <= 2; ++occurrence)
    pack_runtime_failure_quarantines_without_receipt(Step::Download, occurrence);
  normalized_pack_success_counts_every_readback_and_retired_extent();
  normalization_request_and_fit_capacity_are_checked_before_gpu_work();
  for (uint32_t code = 2; code <= 14; code += 2)
    normalized_pack_semantic_error_cannot_publish_fit_or_root(code);
  normalized_pack_unknown_control_does_not_authorize_a_clean_retry();
  for (unsigned occurrence : {6u, 7u}) {
    normalized_pack_clean_oom_retains_unpublished_caller_buffers(Step::Allocate, occurrence);
    normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::Allocate, occurrence);
    normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::Free, occurrence);
  }
  for (unsigned occurrence = 3; occurrence <= 6; ++occurrence) {
    normalized_pack_clean_oom_retains_unpublished_caller_buffers(Step::Pin, occurrence);
    normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::Pin, occurrence);
    normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::Unpin, occurrence);
  }
  normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::NormalizeLaunch, 1);
  for (unsigned occurrence = 2; occurrence <= 5; ++occurrence)
    normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::Download, occurrence);
  for (unsigned occurrence = 2; occurrence <= 6; ++occurrence)
    normalized_pack_runtime_error_keeps_fit_output_unpublished(Step::Synchronize, occurrence);
  std::puts("PASS: 115 host HIP buffer/borrower/SMC/pack/normalization-control policy cases; no GPU execution or numerical parity");
  return 0;
}
