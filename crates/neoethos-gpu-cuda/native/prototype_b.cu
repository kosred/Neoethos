#include "neoethos_gpu_cuda.h"
#include <climits>
#include <cuda_runtime.h>

namespace {

// One logical subgroup per event, including on AMD hardware with 64-lane waves.
// Keep the launch, bar partition and shuffle width on this same 32-lane group.
constexpr unsigned FIRST_HIT_LOGICAL_WIDTH_V1 = 32u;

__device__ std::int32_t first_hit_reason(double high,
                                         double low,
                                         const NeoFirstHitEvent& event) {
  bool stop_hit = false;
  bool target_hit = false;
  if (event.direction > 0) {
    stop_hit = low <= event.stop_price;
    target_hit = high >= event.target_price;
  } else {
    stop_hit = high >= event.stop_price;
    target_hit = low <= event.target_price;
  }
  if (stop_hit && target_hit) {
    return event.precedence == 0 ? 1 : 2;
  }
  if (stop_hit) return 1;
  if (target_hit) return 2;
  return 0;
}

__global__ void warp_first_hit_kernel(const double* highs,
                                      const double* lows,
                                      std::size_t rows,
                                      const NeoFirstHitEvent* events,
                                      NeoFirstHitResult* results,
                                      std::size_t event_count) {
  const std::size_t event_index = static_cast<std::size_t>(blockIdx.x);
  const unsigned lane = threadIdx.x & (FIRST_HIT_LOGICAL_WIDTH_V1 - 1u);
  if (event_index >= event_count) return;

  const NeoFirstHitEvent event = events[event_index];
  int best_bar = INT_MAX;
  int best_reason = 0;
  const std::uint32_t first_bar = event.entry_bar + 1u;
  for (std::uint32_t bar = first_bar + lane;
       bar <= event.last_bar && static_cast<std::size_t>(bar) < rows;
       bar += FIRST_HIT_LOGICAL_WIDTH_V1) {
    const int reason = first_hit_reason(highs[bar], lows[bar], event);
    if (reason != 0 && static_cast<int>(bar) < best_bar) {
      best_bar = static_cast<int>(bar);
      best_reason = reason;
    }
  }

  // HIP requires a 64-bit mask even for a 32-lane logical subgroup. Only this
  // block's low 32 lanes participate; CUDA receives the identical low 32 bits.
  constexpr unsigned long long mask = 0xffffffffull;
  for (int offset = FIRST_HIT_LOGICAL_WIDTH_V1 / 2; offset > 0; offset >>= 1) {
    const int other_bar =
        __shfl_down_sync(mask, best_bar, offset, FIRST_HIT_LOGICAL_WIDTH_V1);
    const int other_reason =
        __shfl_down_sync(mask, best_reason, offset, FIRST_HIT_LOGICAL_WIDTH_V1);
    if (other_bar < best_bar ||
        (other_bar == best_bar && other_bar != INT_MAX && other_reason < best_reason)) {
      best_bar = other_bar;
      best_reason = other_reason;
    }
  }

  if (lane == 0u) {
    if (best_bar == INT_MAX) {
      results[event_index] = NeoFirstHitResult{-1, 0};
    } else {
      results[event_index] = NeoFirstHitResult{best_bar, best_reason};
    }
  }
}

}  // namespace

extern "C" std::int32_t neoethos_gpu_cuda_warp_first_hit(
    const double* highs,
    const double* lows,
    std::size_t rows,
    const NeoFirstHitEvent* events,
    NeoFirstHitResult* results,
    std::size_t event_count) {
  if (event_count == 0) return 0;
  if (highs == nullptr || lows == nullptr || events == nullptr || results == nullptr || rows == 0) {
    return -20;
  }

  double* device_highs = nullptr;
  double* device_lows = nullptr;
  NeoFirstHitEvent* device_events = nullptr;
  NeoFirstHitResult* device_results = nullptr;
  const std::size_t price_bytes = rows * sizeof(double);
  const std::size_t event_bytes = event_count * sizeof(NeoFirstHitEvent);
  const std::size_t result_bytes = event_count * sizeof(NeoFirstHitResult);

  auto cleanup = [&](std::int32_t operation_status) -> std::int32_t {
    bool cleanup_failed = false;
    // Attempt every release, preserving the original operation error. A
    // cleanup-only failure must not certify successfully completed ownership.
    if (device_results != nullptr && cudaFree(device_results) != cudaSuccess) cleanup_failed = true;
    if (device_events != nullptr && cudaFree(device_events) != cudaSuccess) cleanup_failed = true;
    if (device_lows != nullptr && cudaFree(device_lows) != cudaSuccess) cleanup_failed = true;
    if (device_highs != nullptr && cudaFree(device_highs) != cudaSuccess) cleanup_failed = true;
    return operation_status != 0 ? operation_status : (cleanup_failed ? -28 : 0);
  };

  if (cudaMalloc(reinterpret_cast<void**>(&device_highs), price_bytes) != cudaSuccess) {
    return cleanup(-21);
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_lows), price_bytes) != cudaSuccess) {
    return cleanup(-22);
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_events), event_bytes) != cudaSuccess) {
    return cleanup(-23);
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_results), result_bytes) != cudaSuccess) {
    return cleanup(-24);
  }

  if (cudaMemcpy(device_highs, highs, price_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_lows, lows, price_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_events, events, event_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    return cleanup(-25);
  }

  warp_first_hit_kernel<<<static_cast<unsigned>(event_count), FIRST_HIT_LOGICAL_WIDTH_V1>>>(
      device_highs, device_lows, rows, device_events, device_results, event_count);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    return cleanup(-26);
  }
  if (cudaMemcpy(results, device_results, result_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    return cleanup(-27);
  }

  return cleanup(0);
}
