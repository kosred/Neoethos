// Exercises the exact production host retirement algorithm with controlled
// submission/completion faults. No CUDA/HIP runtime or device is simulated as
// validation evidence. Example (from this crate):
// Compile this file with C++17, pthread, -fsanitize=address,undefined and
// -fno-omit-frame-pointer; then run the resulting host executable.
#include "../native/resident_host_staging_v1.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <thread>

namespace {
using neoethos_host_staging_v1::CallbackTicketV1;
using neoethos_host_staging_v1::RetirementResultV1;
using neoethos_host_staging_v1::retire_after_stream_v1;

void require(bool condition, const char* detail) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", detail);
    std::abort();
  }
}

struct Payload {
  std::atomic<unsigned>& deletions;
  ~Payload() noexcept { deletions.fetch_add(1u); }
};

void queued_callback_after_submitter_returns() {
  std::atomic<unsigned> deletions{0};
  void* pending = nullptr;
  unsigned waits = 0;
  const auto result = retire_after_stream_v1(
      new Payload{deletions},
      [&](void* ticket) { pending = ticket; return true; },
      [&] { ++waits; return true; });
  require(result == RetirementResultV1::Queued, "accepted callback result");
  require(waits == 0 && deletions == 0, "success never waits or frees early");
  CallbackTicketV1<Payload>::complete(pending);
  require(deletions == 1, "late accepted callback releases exactly once");
}

void callback_before_submission_returns(bool accepted, bool wait_success) {
  std::atomic<unsigned> deletions{0};
  unsigned waits = 0;
  const auto result = retire_after_stream_v1(
      new Payload{deletions},
      [&](void* ticket) {
        CallbackTicketV1<Payload>::complete(ticket);
        return accepted;
      },
      [&] { ++waits; return wait_success; });
  require(result == (accepted ? RetirementResultV1::Queued
                             : RetirementResultV1::SubmissionFailed),
          "callback before return preserves actual submission result");
  require(waits == (accepted ? 0u : 1u) && deletions == 1,
          "submitter survives inline callback and cannot double-delete");
}

void callback_after_error_and_wait(bool wait_success) {
  std::atomic<unsigned> deletions{0};
  void* pending = nullptr;
  unsigned waits = 0;
  const auto result = retire_after_stream_v1(
      new Payload{deletions},
      [&](void* ticket) { pending = ticket; return false; },
      [&] { ++waits; return wait_success; });
  require(result == RetirementResultV1::SubmissionFailed && waits == 1,
          "accepted-before-error remains failure");
  require(deletions == (wait_success ? 1u : 0u),
          "ambiguous wait cannot release payload");
  CallbackTicketV1<Payload>::complete(pending);
  require(deletions == 1, "late callback remains safe after error/wait");
}

void callback_during_wait(bool wait_success) {
  std::atomic<unsigned> deletions{0};
  void* pending = nullptr;
  const auto result = retire_after_stream_v1(
      new Payload{deletions},
      [&](void* ticket) { pending = ticket; return false; },
      [&] {
        CallbackTicketV1<Payload>::complete(pending);
        return wait_success;
      });
  require(result == RetirementResultV1::SubmissionFailed && deletions == 1,
          "callback during fallback wait cannot double-delete");
}

void allocation_failure(bool wait_success) {
  std::atomic<unsigned> deletions{0};
  unsigned submits = 0;
  unsigned waits = 0;
  auto* payload = new Payload{deletions};
  const auto result = retire_after_stream_v1(
      payload,
      [&](void*) { ++submits; return true; },
      [&] { ++waits; return wait_success; },
      [](Payload*) noexcept -> CallbackTicketV1<Payload>* { return nullptr; });
  require(result == RetirementResultV1::TicketAllocationFailed,
          "ticket OOM stays explicit");
  require(submits == 0 && waits == 1,
          "ticket OOM must not submit raw staging as a callback ticket");
  require(deletions == (wait_success ? 1u : 0u),
          "ticket OOM plus failed wait retains in-flight source");
  // The fixture has no DMA. Only the test can prove this retained raw allocation
  // unused and reclaim it; production deliberately has no such assumption.
  if (!wait_success) delete payload;
  require(deletions == 1, "allocation failure test fixture cleanup");
}

void concurrent_callback_and_failed_submitter(bool wait_success) {
  // Exercise both reference-release orders and concurrent payload exchanges.
  // This bounds thread creation; it is a host lifetime stress, not a benchmark.
  for (unsigned iteration = 0; iteration < 128; ++iteration) {
    std::atomic<unsigned> deletions{0};
    std::atomic<bool> proceed{false};
    std::thread callback;
    const auto result = retire_after_stream_v1(
        new Payload{deletions},
        [&](void* ticket) {
          callback = std::thread([&, ticket] {
            while (!proceed.load(std::memory_order_acquire)) std::this_thread::yield();
            CallbackTicketV1<Payload>::complete(ticket);
          });
          return false;
        },
        [&] {
          proceed.store(true, std::memory_order_release);
          return wait_success;
        });
    callback.join();
    require(result == RetirementResultV1::SubmissionFailed && deletions == 1,
            "concurrent callback/error retirement must release exactly once");
  }
}

void rejected_without_callback(bool wait_success) {
  std::atomic<unsigned> deletions{0};
  void* retained = nullptr;
  const auto result = retire_after_stream_v1(
      new Payload{deletions},
      [&](void* ticket) { retained = ticket; return false; },
      [&] { return wait_success; });
  require(result == RetirementResultV1::SubmissionFailed,
          "unaccepted submission remains failure");
  require(deletions == (wait_success ? 1u : 0u),
          "unaccepted ambiguous failure preserves staging");
  // The injected submit function retained the address and never enqueued work.
  // Discharge its potential-callback reference solely to clean up this fixture.
  // Production cannot know this and retains the ticket if no callback arrives.
  CallbackTicketV1<Payload>::complete(retained);
  require(deletions == 1, "controlled nonacceptance fixture cleanup");
}
}  // namespace

int main() {
  queued_callback_after_submitter_returns();
  callback_before_submission_returns(true, true);
  for (const bool wait_success : {false, true}) {
    callback_before_submission_returns(false, wait_success);
    callback_after_error_and_wait(wait_success);
    callback_during_wait(wait_success);
    allocation_failure(wait_success);
    concurrent_callback_and_failed_submitter(wait_success);
    rejected_without_callback(wait_success);
  }
  std::puts("PASS: 14 host lifetime cases, including 256 concurrent interleavings; no GPU execution");
  return 0;
}
