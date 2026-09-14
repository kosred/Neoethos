#pragma once

#include <atomic>
#include <new>

// Host-only ownership; callbacks must not invoke CUDA/HIP APIs. A failed
// submission is not proof that a callback was rejected: ROCm 7.2.3 can enqueue
// its callback marker before failing to allocate the following blocking marker.
namespace neoethos_host_staging_v1 {

template <class T>
class CallbackTicketV1 final {
 public:
  static CallbackTicketV1* create(T* payload) noexcept {
    return new (std::nothrow) CallbackTicketV1(payload);
  }

  // Only after confirmed stream retirement or inside the ordered callback.
  // An atomic exchange arbitrates those two potentially concurrent paths.
  void release_payload_once() noexcept {
    delete payload_.exchange(nullptr, std::memory_order_acq_rel);
  }

  void release_submitter() noexcept { release_reference(); }

  static void complete(void* opaque) noexcept {
    auto* ticket = static_cast<CallbackTicketV1*>(opaque);
    ticket->release_payload_once();
    ticket->release_reference();
  }

  CallbackTicketV1(const CallbackTicketV1&) = delete;
  CallbackTicketV1& operator=(const CallbackTicketV1&) = delete;

 private:
  explicit CallbackTicketV1(T* payload) noexcept : payload_(payload) {}
  ~CallbackTicketV1() = default;

  void release_reference() noexcept {
    if (references_.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
      delete this;
    }
  }

  // The submitter keeps the ticket alive even if the callback runs inline.
  // Only the callback may consume the other reference. In particular, an
  // error return must not consume it on behalf of a possibly delayed callback.
  std::atomic<unsigned> references_{2u};
  std::atomic<T*> payload_;
};

enum class RetirementResultV1 { Queued, SubmissionFailed, TicketAllocationFailed };

// Takes ownership of nonnull staging, which earlier asynchronous copies may
// already reference. The backend adapters must not throw. submit(ticket) and
// synchronize() return true only on their actual API success status.
template <class T, class Submit, class Synchronize,
          class CreateTicket = CallbackTicketV1<T>* (*)(T*) noexcept>
RetirementResultV1 retire_after_stream_v1(
    T* staging, Submit&& submit, Synchronize&& synchronize,
    CreateTicket create_ticket = &CallbackTicketV1<T>::create) noexcept {
  auto* ticket = create_ticket(staging);
  if (ticket == nullptr) {
    // No callback was submitted. Only a successful wait permits raw cleanup.
    if (synchronize()) delete staging;
    return RetirementResultV1::TicketAllocationFailed;
  }
  if (submit(static_cast<void*>(ticket))) {
    ticket->release_submitter();
    return RetirementResultV1::Queued;
  }
  if (synchronize()) ticket->release_payload_once();
  ticket->release_submitter();
  // If no callback was accepted, its reference is deliberately retained. With
  // a proven wait only the small ticket remains; without one the payload also
  // remains. No backend API proves callback nonacceptance on this error path.
  // Never replace this with premature reclamation, nor report resident success.
  return RetirementResultV1::SubmissionFailed;
}

}  // namespace neoethos_host_staging_v1
