#include "neoethos_gpu_cuda.h"
#include <cuda_runtime.h>

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
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      status = -6;
    } else if (cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
      status = -7;
    }
  }
  cudaFree(device_output);
  cudaFree(device_input);
  return status;
}
