#include "neoethos_gpu_cuda.h"

extern "C" std::int32_t neoethos_gpu_cuda_runtime_available() { return 0; }
extern "C" std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t*,
                                                  std::uint32_t*,
                                                  std::size_t) {
  return -1;
}
