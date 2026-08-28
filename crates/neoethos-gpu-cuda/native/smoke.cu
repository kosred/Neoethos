#include "neoethos_gpu_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>

namespace {
/// Report the concrete CUDA error behind a numeric status.
///
/// A bare status code is unactionable on rented hardware, where every minute
/// of guesswork is billed. Failures are rare, so this always prints.
void report(const char* stage, cudaError_t error) {
  std::fprintf(stderr, "[neoethos-cuda] %s failed: %s\n", stage, cudaGetErrorString(error));
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
  if (input == nullptr || output == nullptr) {
    return -2;
  }
  std::uint32_t* device_input = nullptr;
  std::uint32_t* device_output = nullptr;
  const std::size_t bytes = len * sizeof(std::uint32_t);
  if (cudaMalloc(&device_input, bytes) != cudaSuccess) return -3;
  if (cudaMalloc(&device_output, bytes) != cudaSuccess) {
    cudaFree(device_input);
    return -4;
  }
  std::int32_t status = 0;
  if (cudaMemcpy(device_input, input, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    status = -5;
  } else {
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((len + threads - 1) / threads);
    add_one_kernel<<<blocks, threads>>>(device_input, device_output, len);
    const cudaError_t launch = cudaGetLastError();
    const cudaError_t sync = cudaDeviceSynchronize();
    if (launch != cudaSuccess || sync != cudaSuccess) {
      report("smoke kernel launch", launch != cudaSuccess ? launch : sync);
      status = -6;
    } else {
      const cudaError_t copy =
          cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost);
      if (copy != cudaSuccess) {
        report("smoke device-to-host copy", copy);
        status = -7;
      }
    }
  }
  cudaFree(device_output);
  cudaFree(device_input);
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
  cudaSetDevice(previous);
  if (status != cudaSuccess) {
    return 0ull;
  }
  return static_cast<std::uint64_t>(free_bytes);
}
