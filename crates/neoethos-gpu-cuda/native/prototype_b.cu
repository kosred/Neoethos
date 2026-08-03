#include "neoethos_gpu_cuda.h"
#include <climits>
#include <cuda_runtime.h>

namespace {

__device__ std::int32_t first_hit_reason(float high,
                                         float low,
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

__global__ void warp_first_hit_kernel(const float* highs,
                                      const float* lows,
                                      std::size_t rows,
                                      const NeoFirstHitEvent* events,
                                      NeoFirstHitResult* results,
                                      std::size_t event_count) {
  const std::size_t event_index = static_cast<std::size_t>(blockIdx.x);
  const unsigned lane = threadIdx.x & 31u;
  if (event_index >= event_count) return;

  const NeoFirstHitEvent event = events[event_index];
  int best_bar = INT_MAX;
  int best_reason = 0;
  const std::uint32_t first_bar = event.entry_bar + 1u;
  for (std::uint32_t bar = first_bar + lane;
       bar <= event.last_bar && static_cast<std::size_t>(bar) < rows;
       bar += 32u) {
    const int reason = first_hit_reason(highs[bar], lows[bar], event);
    if (reason != 0 && static_cast<int>(bar) < best_bar) {
      best_bar = static_cast<int>(bar);
      best_reason = reason;
    }
  }

  constexpr unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const int other_bar = __shfl_down_sync(mask, best_bar, offset);
    const int other_reason = __shfl_down_sync(mask, best_reason, offset);
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
    const float* highs,
    const float* lows,
    std::size_t rows,
    const NeoFirstHitEvent* events,
    NeoFirstHitResult* results,
    std::size_t event_count) {
  if (event_count == 0) return 0;
  if (highs == nullptr || lows == nullptr || events == nullptr || results == nullptr || rows == 0) {
    return -20;
  }

  float* device_highs = nullptr;
  float* device_lows = nullptr;
  NeoFirstHitEvent* device_events = nullptr;
  NeoFirstHitResult* device_results = nullptr;
  const std::size_t price_bytes = rows * sizeof(float);
  const std::size_t event_bytes = event_count * sizeof(NeoFirstHitEvent);
  const std::size_t result_bytes = event_count * sizeof(NeoFirstHitResult);

  auto cleanup = [&]() {
    if (device_results != nullptr) cudaFree(device_results);
    if (device_events != nullptr) cudaFree(device_events);
    if (device_lows != nullptr) cudaFree(device_lows);
    if (device_highs != nullptr) cudaFree(device_highs);
  };

  if (cudaMalloc(reinterpret_cast<void**>(&device_highs), price_bytes) != cudaSuccess) {
    cleanup();
    return -21;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_lows), price_bytes) != cudaSuccess) {
    cleanup();
    return -22;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_events), event_bytes) != cudaSuccess) {
    cleanup();
    return -23;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_results), result_bytes) != cudaSuccess) {
    cleanup();
    return -24;
  }

  if (cudaMemcpy(device_highs, highs, price_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_lows, lows, price_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_events, events, event_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cleanup();
    return -25;
  }

  warp_first_hit_kernel<<<static_cast<unsigned>(event_count), 32>>>(
      device_highs, device_lows, rows, device_events, device_results, event_count);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return -26;
  }
  if (cudaMemcpy(results, device_results, result_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cleanup();
    return -27;
  }

  cleanup();
  return 0;
}
