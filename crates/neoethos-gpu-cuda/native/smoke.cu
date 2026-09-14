#include "neoethos_gpu_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>

namespace {
/// Report the concrete CUDA error behind a numeric status.
///
/// A bare status code is unactionable on rented hardware, where every minute
/// of guesswork is billed. Failures are rare, so this always prints.
void report(const char* stage, cudaError_t error) {
#if defined(__HIP_PLATFORM_AMD__)
  std::fprintf(stderr, "[neoethos-hip] %s failed: %s\n", stage, cudaGetErrorString(error));
#else
  std::fprintf(stderr, "[neoethos-cuda] %s failed: %s\n", stage, cudaGetErrorString(error));
#endif
}

// Preserve the first failure, but also report every subsequent cleanup failure.
// In particular, a correct kernel result cannot make failed cleanup a success.
void record_failure(const char* stage, cudaError_t error, std::int32_t code,
                    std::int32_t& status) {
  if (error != cudaSuccess) {
    report(stage, error);
    if (status == 0) status = code;
  }
}
}  // namespace

namespace {
__global__ void add_one_kernel(const std::uint32_t* input,
                               std::uint32_t* output,
                               std::size_t len) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < len) {
    output[index] = input[index] + 1u;
  }
}
}

extern "C" std::int32_t neoethos_gpu_cuda_runtime_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" std::int32_t neoethos_gpu_cuda_probe_device_count_v1(
    std::uint32_t* out_count) {
  if (out_count == nullptr) {
    return NEO_CUDA_DEVICE_PROBE_INVALID_OUTPUT;
  }
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    return static_cast<std::int32_t>(status);
  }
  if (count < 0) {
    return NEO_CUDA_DEVICE_PROBE_INVALID_OUTPUT;
  }
  *out_count = static_cast<std::uint32_t>(count);
  return NEO_CUDA_DEVICE_PROBE_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t* input,
                                                  std::uint32_t* output,
                                                  std::size_t len) {
  if (len == 0) {
    return 0;
  }
  constexpr unsigned threads = 256;
  if (input == nullptr || output == nullptr ||
      len > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      (len - 1) / threads + 1 > std::numeric_limits<unsigned>::max()) {
    return -2;
  }
  std::uint32_t* device_input = nullptr;
  std::uint32_t* device_output = nullptr;
  const std::size_t bytes = len * sizeof(std::uint32_t);
  std::int32_t status = 0;
  record_failure("smoke input allocation", cudaMalloc(&device_input, bytes), -3, status);
  if (status != 0) return status;
  record_failure("smoke output allocation", cudaMalloc(&device_output, bytes), -4, status);
  if (status != 0) {
    record_failure("smoke input cleanup", cudaFree(device_input), -8, status);
    return status;
  }
  record_failure("smoke host-to-device copy",
                 cudaMemcpy(device_input, input, bytes, cudaMemcpyHostToDevice), -5, status);
  if (status == 0) {
    const unsigned blocks = static_cast<unsigned>((len - 1) / threads + 1);
    add_one_kernel<<<blocks, threads>>>(device_input, device_output, len);
    const cudaError_t launch = cudaGetLastError();
    const cudaError_t sync = cudaDeviceSynchronize();
    record_failure("smoke kernel launch", launch, -6, status);
    record_failure("smoke kernel completion", sync, -6, status);
    if (status == 0) {
      const cudaError_t copy =
          cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost);
      if (copy != cudaSuccess) {
        report("smoke device-to-host copy", copy);
        status = -7;
      }
    }
  }
  record_failure("smoke output cleanup", cudaFree(device_output), -8, status);
  record_failure("smoke input cleanup", cudaFree(device_input), -8, status);
  return status;
}

extern "C" std::int32_t neoethos_gpu_cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return static_cast<std::int32_t>(count);
}

// Free device memory, so a session can be sized from the hardware rather than
// from what the caller asked for. Returns 0 when it cannot be determined, which
// callers must treat as "unknown" and refuse to guess around.
extern "C" std::uint64_t neoethos_gpu_cuda_device_free_memory(std::int32_t device) {
  int previous = 0;
  if (cudaGetDevice(&previous) != cudaSuccess) {
    return 0ull;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    return 0ull;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  const cudaError_t restore = cudaSetDevice(previous);
  if (restore != cudaSuccess) {
    report("device-memory probe restore", restore);
    return 0ull;
  }
  if (status != cudaSuccess) {
    report("device-memory probe", status);
    return 0ull;
  }
  return static_cast<std::uint64_t>(free_bytes);
}
