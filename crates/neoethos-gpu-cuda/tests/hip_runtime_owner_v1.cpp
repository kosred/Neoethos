// Host policy tests of the EXACT production lifecycle core, not a HIP runtime
// simulation or device-readiness claim. Scripted Ops inject API observations
// and faults; no GPU library, allocation, stream or context is created here.
// Build separately with C++17, pthread, ASan/UBSan and strict host warnings.
#include "../hip/hip_runtime_lifecycle_v1.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <thread>
#include <vector>

namespace {
using neoethos::hip_runtime_v1::LeaseRegistryV1;
using Facts = NeoHipRuntimeFactsV1;
using Error = NeoHipRuntimeErrorV1;

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::abort();
  }
}

template<class Predicate>
void wait_until(Predicate predicate, const char* message) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!predicate()) {
    require(std::chrono::steady_clock::now() < deadline, message);
    std::this_thread::yield();
  }
}

bool zero_facts(const Facts& facts) {
  const Facts zero{};
  return std::memcmp(&facts, &zero, sizeof(facts)) == 0;
}

enum class Step { Select, Create, Inspect, IdentityInspect, Synchronize, Destroy };

Facts observed_facts(int32_t ordinal, uint64_t stream, uint64_t stream_id) {
  Facts facts{};
  facts.device_ordinal = ordinal;
  facts.runtime_version = 70200000;
  facts.driver_version = 70200000;
  facts.warp_size = 64;
  for (unsigned i = 0; i < sizeof(facts.uuid); ++i) facts.uuid[i] = uint8_t(i + 1);
  facts.stream_handle = stream;
  facts.stream_id = stream_id;
  facts.total_memory_bytes = uint64_t{16} << 30;
  facts.free_memory_bytes = uint64_t{12} << 30;
  facts.current_pool_handle = 0x3000;
  facts.default_pool_handle = 0x3000;
  facts.pool_reserved_bytes = 4096;
  facts.pool_used_bytes = 1024;
  constexpr char architecture[] = "gfx942:sramecc+:xnack-";
  std::memcpy(facts.architecture, architecture, sizeof(architecture));
  return facts;
}

struct ScriptedOps {
  std::vector<Step> trace;
  std::atomic<unsigned> active{0};
  bool failing = false;
  Step failure = Step::Select;
  uint64_t stream = 0x2000;
  uint64_t stream_id = 73;
  bool selected = false;
  bool stream_live = false;
  unsigned successful_destroys = 0;
  std::function<void(Facts&)> change_facts;
  std::function<void()> during_inspect;

  struct Call {
    ScriptedOps& ops;
    Call(ScriptedOps& owner, Step step) : ops(owner) {
      require(ops.active.fetch_add(1) == 0,
              "registry must serialize backend observations and destruction");
      ops.trace.push_back(step);
    }
    ~Call() { ops.active.fetch_sub(1); }
  };

  int32_t status(Step step, uint32_t operation, Error& error) const {
    if (!failing || failure != step) return 0;
    error = {1u, operation, 719, 0u};
    return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }

  int32_t select_device(int32_t ordinal, Error& error) {
    Call call(*this, Step::Select);
    require(ordinal == 3, "requested ordinal reaches backend unchanged");
    const auto result = status(Step::Select, NEO_HIP_OP_SELECT_DEVICE_V1, error);
    selected = result == 0;
    return result;
  }
  int32_t create_stream(uint64_t& result, Error& error) {
    Call call(*this, Step::Create);
    require(selected, "stream creation follows device selection");
    // A partial handle on failure is not proof that cleanup is safe.
    result = stream;
    stream_live = stream > 2;
    return status(Step::Create, NEO_HIP_OP_CREATE_STREAM_V1, error);
  }
  int32_t inspect(int32_t ordinal, uint64_t expected_stream, Facts& facts, Error& error) {
    Call call(*this, Step::Inspect);
    require(ordinal == 3 && expected_stream == stream,
            "inspection receives owned handles, never caller-supplied replacements");
    require(stream_live, "never inspect already destroyed resources");
    if (during_inspect) during_inspect();
    facts = observed_facts(ordinal, stream, stream_id);
    if (change_facts) change_facts(facts);
    return status(Step::Inspect, NEO_HIP_OP_STREAM_ID_V1, error);
  }
  int32_t inspect_identity(int32_t ordinal, uint64_t expected_stream, Facts& facts, Error& error) {
    Call call(*this, Step::IdentityInspect);
    require(ordinal == 3 && expected_stream == stream && stream_live,
            "identity-only observation uses the owned live stream");
    const auto retained_total = facts.total_memory_bytes;
    facts = observed_facts(ordinal, stream, stream_id);
    facts.total_memory_bytes = retained_total;
    facts.free_memory_bytes = facts.pool_reserved_bytes = facts.pool_used_bytes = 0;
    if (change_facts) change_facts(facts);
    return status(Step::IdentityInspect, NEO_HIP_OP_STREAM_ID_V1, error);
  }
  int32_t synchronize(uint64_t expected_stream, Error& error) {
    Call call(*this, Step::Synchronize);
    require(expected_stream == stream && stream_live, "synchronize only owned live stream");
    return status(Step::Synchronize, NEO_HIP_OP_SYNCHRONIZE_V1, error);
  }
  int32_t destroy_stream(uint64_t expected_stream, Error& error) {
    Call call(*this, Step::Destroy);
    require(expected_stream == stream && stream_live, "destroy only owned live stream once");
    const auto result = status(Step::Destroy, NEO_HIP_OP_DESTROY_STREAM_V1, error);
    if (result == 0) { stream_live = false; ++successful_destroys; }
    return result;
  }
  int32_t free(uint64_t, uint64_t, Error&) {
    require(false, "zero-resource lease tests must never free a device buffer");
    return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }
  int32_t unpin(uint64_t, Error&) {
    require(false, "zero-resource lease tests must never release staging");
    return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }
};

using Registry = LeaseRegistryV1<ScriptedOps>;

uint64_t create(Registry& registry, Facts* result = nullptr) {
  uint64_t lease = 0;
  Facts facts{};
  Error error{};
  require(registry.create(3, &lease, &facts, &error) == NEO_HIP_RUNTIME_OK_V1,
          "healthy creation succeeds");
  require(lease != 0 && facts.lease_id == lease && facts.abi_version == 1 &&
              facts.backend_kind == 2 && facts.device_ordinal == 3 &&
              facts.stream_id == 73 && error.operation == NEO_HIP_OP_NONE_V1,
          "only complete creation publishes native-issued HIP identity");
  if (result) *result = facts;
  return lease;
}

void quarantined_has_no_backend_retry(Registry& registry, ScriptedOps& ops, uint64_t lease) {
  const auto calls = ops.trace.size();
  Facts facts{};
  Error error{};
  require(registry.query(lease, &facts, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              zero_facts(facts), "quarantined query clears facts and refuses");
  require(registry.synchronize(lease, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1,
          "quarantined synchronization refuses");
  require(registry.close(lease, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1,
          "quarantined close cannot retry uncertain resources");
  require(ops.trace.size() == calls, "quarantine never invokes backend again");
}

void successful_lifecycle_and_nonreused_ids() {
  ScriptedOps ops;
  Registry registry(ops);
  Facts initial{};
  const auto lease = create(registry, &initial);
  require(ops.trace == std::vector<Step>{Step::Select, Step::Create, Step::Inspect},
          "creation order is select device, own stream, inspect");
  Facts current{};
  Error error{};
  require(registry.query(lease, &current, &error) == 0 &&
              std::memcmp(&current, &initial, sizeof(current)) == 0,
          "query returns exact captured identity");
  require(registry.synchronize(lease, &error) == 0, "explicit sync succeeds");
  ops.trace.clear();
  require(registry.close(lease, &error) == 0, "checked terminal close succeeds");
  require(ops.trace == std::vector<Step>{Step::Inspect, Step::Synchronize, Step::Destroy},
          "close validates and waits before destroying the owned stream");
  require(!ops.stream_live && ops.successful_destroys == 1,
          "successful cleanup happens exactly once");
  ops.trace.clear();
  require(registry.query(lease, &current, &error) == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 &&
              registry.close(lease, &error) == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 && ops.trace.empty(),
          "retired ID refuses before backend access");
  const auto next = create(registry);
  require(next != lease, "closed lease IDs are never reused");
  require(registry.close(next, &error) == 0, "second independently owned lease closes");
}

void invalid_arguments_and_foreign_ids() {
  ScriptedOps ops;
  Registry registry(ops);
  uint64_t lease = 99;
  Facts facts{};
  facts.lease_id = 99;
  Error error{};
  require(registry.create(-1, &lease, &facts, &error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              lease == 0 && zero_facts(facts), "invalid creation cannot leave stale outputs");
  require(registry.create(3, nullptr, &facts, &error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              registry.create(3, &lease, nullptr, &error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              registry.create(3, &lease, &facts, nullptr) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1,
          "required creation outputs cannot be omitted");
  for (const auto foreign : {uint64_t{0}, uint64_t{1}, UINT64_MAX}) {
    require(registry.query(foreign, &facts, &error) == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 &&
                zero_facts(facts) &&
                registry.synchronize(foreign, &error) == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 &&
                registry.close(foreign, &error) == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1,
            "unknown keys never authorize a foreign stream");
  }
  require(registry.query(1, nullptr, &error) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              registry.query(1, &facts, nullptr) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              registry.synchronize(1, nullptr) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1 &&
              registry.close(1, nullptr) == NEO_HIP_RUNTIME_INVALID_ARGUMENT_V1,
          "all API error/fact outputs remain mandatory");
  require(ops.trace.empty(), "invalid arguments and unknown keys make no backend calls");
}

void creation_fault(Step failed, std::size_t expected_calls, uint32_t operation) {
  ScriptedOps ops;
  ops.failing = true;
  ops.failure = failed;
  uint64_t lease = 99;
  Facts facts{};
  facts.lease_id = 99;
  Error error{};
  {
    Registry registry(ops);
    const auto result = registry.create(3, &lease, &facts, &error);
    require(result == (failed == Step::Select ? NEO_HIP_RUNTIME_BACKEND_ERROR_V1
                                             : NEO_HIP_RUNTIME_QUARANTINED_V1),
            "creation reports actual failure, not partial success");
    require(lease == 0 && zero_facts(facts) && error.operation == operation &&
                error.backend_status == 719 && ops.trace.size() == expected_calls,
            "creation preserves failing operation/status and stops immediately");
  }
  require(ops.trace.size() == expected_calls && ops.successful_destroys == 0,
          "registry destruction cannot silently retry quarantined partial handles");
}

void invalid_created_stream(uint64_t handle) {
  ScriptedOps ops;
  ops.stream = handle;
  Registry registry(ops);
  uint64_t lease = 99;
  Facts facts{};
  Error error{};
  require(registry.create(3, &lease, &facts, &error) == NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1 &&
              lease == 0 && zero_facts(facts) && error.operation == NEO_HIP_OP_IDENTITY_V1,
          "successful API status cannot bless a null or special stream");
  require(ops.trace.size() == 2u,
          "invalid created handle is rejected before inspect/cleanup");
}

void changed_identity_or_malformed_facts(const std::function<void(Facts&)>& mutation) {
  ScriptedOps ops;
  Registry registry(ops);
  const auto lease = create(registry);
  ops.change_facts = mutation;
  Facts facts{};
  facts.lease_id = lease;
  Error error{};
  require(registry.query(lease, &facts, &error) == NEO_HIP_RUNTIME_IDENTITY_MISMATCH_V1 &&
              zero_facts(facts) && error.operation == NEO_HIP_OP_IDENTITY_V1,
          "changed or malformed live identity fails before it can escape");
  ops.change_facts = nullptr;
  quarantined_has_no_backend_retry(registry, ops, lease);
  require(ops.stream_live && ops.successful_destroys == 0,
          "stale stream is retained, not passed to unsafe destruction");
}

void mutable_memory_observations_are_not_identity() {
  ScriptedOps ops;
  Registry registry(ops);
  const auto lease = create(registry);
  ops.change_facts = [](Facts& f) {
    f.free_memory_bytes = 1024;
    f.pool_reserved_bytes = 8192;
    // HIP samples reserved and used through separate API calls. Another
    // owner can allocate between them; their ordering is not an identity fault.
    // These observations are neither an atomic snapshot nor a reservation.
    f.pool_used_bytes = 16384;
  };
  Facts facts{};
  Error error{};
  require(registry.query(lease, &facts, &error) == 0 && facts.free_memory_bytes == 1024 &&
              facts.pool_reserved_bytes == 8192 && facts.pool_used_bytes == 16384,
          "honest live memory changes remain observable without invented stale identity");
  require(registry.close(lease, &error) == 0, "changed memory snapshot does not prevent safe close");
}

void close_fault(Step failed, uint32_t operation, const std::vector<Step>& expected) {
  ScriptedOps ops;
  Registry registry(ops);
  const auto lease = create(registry);
  ops.trace.clear();
  ops.failing = true;
  ops.failure = failed;
  Error error{};
  require(registry.close(lease, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              error.operation == operation && error.backend_status == 719 && ops.trace == expected,
          "close stops at the exact failing operation and preserves its diagnostic");
  require(ops.stream_live && ops.successful_destroys == 0,
          "cleanup failure retains the stream and never destroys too early or twice");
  ops.failing = false;
  quarantined_has_no_backend_retry(registry, ops, lease);
}

void explicit_sync_fault_is_permanent() {
  ScriptedOps ops;
  Registry registry(ops);
  const auto lease = create(registry);
  ops.failing = true;
  ops.failure = Step::Synchronize;
  Error error{};
  require(registry.synchronize(lease, &error) == NEO_HIP_RUNTIME_QUARANTINED_V1 &&
              error.operation == NEO_HIP_OP_SYNCHRONIZE_V1,
          "uncertain explicit synchronization poisons subsequent use");
  ops.failing = false;
  quarantined_has_no_backend_retry(registry, ops, lease);
  require(ops.stream_live, "sync failure releases no resource");
}

void query_and_close_are_serialized() {
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    ScriptedOps ops;
    Registry registry(ops);
    const auto lease = create(registry);
    std::atomic<bool> inspecting{false}, close_started{false}, proceed{false};
    ops.during_inspect = [&] {
      if (!inspecting.exchange(true)) {
        wait_until([&] { return proceed.load(std::memory_order_acquire); },
                   "same-owner query gate must be released promptly");
      }
    };
    int32_t query_result = -99, close_result = -99;
    Facts facts{};
    std::thread query([&] { Error e{}; query_result = registry.query(lease, &facts, &e); });
    wait_until([&] { return inspecting.load(std::memory_order_acquire); },
               "query must reach the scripted backend");
    std::thread close([&] {
      Error e{};
      close_started.store(true, std::memory_order_release);
      close_result = registry.close(lease, &e);
    });
    wait_until([&] { return close_started.load(std::memory_order_acquire); },
               "close thread must start promptly");
    proceed.store(true, std::memory_order_release);
    query.join();
    close.join();
    require(query_result == 0 && close_result == 0 && facts.lease_id == lease &&
                ops.successful_destroys == 1,
            "pinned query completes before close; no external-reset claim");
  }
}

void close_before_waiting_query_refuses_retired_owner() {
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    ScriptedOps ops;
    Registry registry(ops);
    const auto lease = create(registry);
    std::atomic<bool> closing{false}, query_started{false}, proceed{false};
    ops.during_inspect = [&] {
      closing.store(true, std::memory_order_release);
      wait_until([&] { return proceed.load(std::memory_order_acquire); },
                 "close gate must be released promptly");
    };
    int32_t close_result = -99, query_result = -99;
    Facts facts{};
    facts.lease_id = lease;
    std::thread close([&] { Error e{}; close_result = registry.close(lease, &e); });
    wait_until([&] { return closing.load(std::memory_order_acquire); },
               "close must hold the owner before query starts");
    std::thread query([&] {
      Error e{};
      query_started.store(true, std::memory_order_release);
      query_result = registry.query(lease, &facts, &e);
    });
    wait_until([&] { return query_started.load(std::memory_order_acquire); },
               "waiting query thread must start promptly");
    proceed.store(true, std::memory_order_release);
    close.join();
    query.join();
    require(close_result == 0 && query_result == NEO_HIP_RUNTIME_UNKNOWN_LEASE_V1 &&
                zero_facts(facts) && ops.successful_destroys == 1,
            "close-first query never observes already destroyed handles");
  }
}

// Two explicit scripted streams let this test distinguish a per-owner lock
// from a registry-wide lock held across a blocking runtime synchronization.
struct IndependentStreamOps {
  uint64_t next_stream = 0x2000;
  uint64_t blocked_stream = 0;
  std::atomic<bool> entered{false}, proceed{false};
  int32_t select_device(int32_t, Error&) { return 0; }
  int32_t create_stream(uint64_t& stream, Error&) { stream = next_stream++; return 0; }
  int32_t inspect(int32_t ordinal, uint64_t stream, Facts& facts, Error&) {
    facts = observed_facts(ordinal, stream, stream + 1);
    return 0;
  }
  int32_t inspect_identity(int32_t ordinal, uint64_t stream, Facts& facts, Error&) {
    const auto retained_total = facts.total_memory_bytes;
    facts = observed_facts(ordinal, stream, stream + 1);
    facts.total_memory_bytes = retained_total;
    facts.free_memory_bytes = facts.pool_reserved_bytes = facts.pool_used_bytes = 0;
    return 0;
  }
  int32_t synchronize(uint64_t stream, Error&) {
    if (stream == blocked_stream) {
      entered.store(true, std::memory_order_release);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
      while (!proceed.load(std::memory_order_acquire)) {
        require(std::chrono::steady_clock::now() < deadline,
                "independent-owner gate must be released even after a failed progress check");
        std::this_thread::yield();
      }
    }
    return 0;
  }
  int32_t destroy_stream(uint64_t, Error&) { return 0; }
  int32_t free(uint64_t, uint64_t, Error&) {
    require(false, "independent zero-resource lease must not free buffers");
    return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }
  int32_t unpin(uint64_t, Error&) {
    require(false, "independent zero-resource lease must not release staging");
    return NEO_HIP_RUNTIME_BACKEND_ERROR_V1;
  }
};

void blocked_owner_does_not_block_independent_owner() {
  IndependentStreamOps ops;
  LeaseRegistryV1<IndependentStreamOps> registry(ops);
  uint64_t first = 0, second = 0;
  Facts a{}, b{};
  Error error{};
  require(registry.create(3, &first, &a, &error) == 0 &&
              registry.create(3, &second, &b, &error) == 0 && first != second &&
              a.stream_handle != b.stream_handle,
          "independent leases own distinct streams");
  ops.blocked_stream = a.stream_handle;
  int32_t sync_result = -99, query_result = -99;
  std::atomic<bool> query_finished{false};
  std::thread waiting([&] { Error e{}; sync_result = registry.synchronize(first, &e); });
  wait_until([&] { return ops.entered.load(std::memory_order_acquire); },
             "first owner must reach the blocking backend operation");
  std::thread query([&] {
    Error e{};
    query_result = registry.query(second, &b, &e);
    query_finished.store(true, std::memory_order_release);
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!query_finished.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  const bool independent_progress = query_finished.load(std::memory_order_acquire);
  // Always release/join even under the old global-lock negative control.
  ops.proceed.store(true, std::memory_order_release);
  waiting.join();
  query.join();
  require(independent_progress && query_result == 0 && sync_result == 0 && b.lease_id == second,
          "one blocked synchronization must not stall another lease's identity query");
  require(registry.close(first, &error) == 0 && registry.close(second, &error) == 0,
          "both independent leases remain closable");
}
} // namespace

int main() {
  successful_lifecycle_and_nonreused_ids();
  invalid_arguments_and_foreign_ids();
  creation_fault(Step::Select, 1, NEO_HIP_OP_SELECT_DEVICE_V1);
  creation_fault(Step::Create, 2, NEO_HIP_OP_CREATE_STREAM_V1);
  creation_fault(Step::Inspect, 3, NEO_HIP_OP_STREAM_ID_V1);
  for (uint64_t special : {uint64_t{0}, uint64_t{1}, uint64_t{2}})
    invalid_created_stream(special);

  // Single-field changes include pointer ABA (unchanged stream handle, new ID),
  // same ordinal with a foreign UUID/pool, malformed observations and
  // exact architecture suffix/termination. No alternative identity is minted.
  const std::vector<std::function<void(Facts&)>> mutations{
    [](Facts& f) { ++f.device_ordinal; },
    [](Facts& f) { ++f.runtime_version; },
    [](Facts& f) { ++f.driver_version; },
    [](Facts& f) { f.warp_size = 32; },
    [](Facts& f) { f.uuid[15] ^= 1; },
    [](Facts& f) { std::memset(f.uuid, 0, sizeof(f.uuid)); },
    [](Facts& f) { ++f.stream_handle; },
    [](Facts& f) { ++f.stream_id; },
    [](Facts& f) { f.stream_id = 0; },
    [](Facts& f) { ++f.total_memory_bytes; },
    [](Facts& f) { f.total_memory_bytes = 0; },
    [](Facts& f) { f.free_memory_bytes = f.total_memory_bytes + 1; },
    [](Facts& f) { ++f.current_pool_handle; },
    [](Facts& f) { ++f.default_pool_handle; },
    [](Facts& f) { ++f.current_pool_handle; ++f.default_pool_handle; },
    [](Facts& f) { f.architecture[5] = '0'; },
    [](Facts& f) { f.architecture[0] = 0; },
    [](Facts& f) { std::memset(f.architecture, 'x', sizeof(f.architecture)); }
  };
  for (const auto& mutation : mutations) changed_identity_or_malformed_facts(mutation);
  mutable_memory_observations_are_not_identity();
  close_fault(Step::Inspect, NEO_HIP_OP_STREAM_ID_V1, {Step::Inspect});
  close_fault(Step::Synchronize, NEO_HIP_OP_SYNCHRONIZE_V1, {Step::Inspect, Step::Synchronize});
  close_fault(Step::Destroy, NEO_HIP_OP_DESTROY_STREAM_V1,
              {Step::Inspect, Step::Synchronize, Step::Destroy});
  explicit_sync_fault_is_permanent();
  query_and_close_are_serialized();
  close_before_waiting_query_refuses_retired_owner();
  blocked_owner_does_not_block_independent_owner();
  std::puts("PASS: 34 host HIP lease-policy cases including 128 serialized query/close interleavings and independent-owner progress; no GPU execution");
  return 0;
}
