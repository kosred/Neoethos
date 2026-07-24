#include "neoethos_gpu_cuda.h"
#include <cstddef>

static_assert(sizeof(NeoBufferRef) == 16);
static_assert(alignof(NeoBufferRef) == 8);
static_assert(sizeof(NeoHandleToken) == 32);
static_assert(offsetof(NeoHandleToken, generation) == 16);
static_assert(sizeof(NeoDatasetHeader) == 152);
static_assert(alignof(NeoDatasetHeader) == 8);
static_assert(offsetof(NeoDatasetHeader, timestamps) == 24);
static_assert(offsetof(NeoDatasetHeader, features) == 104);
static_assert(offsetof(NeoDatasetHeader, days) == 136);
static_assert(sizeof(NeoGeneDescriptor) == 56);
static_assert(offsetof(NeoGeneDescriptor, stop_ticks) == 24);
static_assert(offsetof(NeoGeneDescriptor, reserved) == 48);
static_assert(sizeof(NeoScenarioDescriptor) == 72);
static_assert(offsetof(NeoScenarioDescriptor, commission_micros) == 48);
static_assert(offsetof(NeoScenarioDescriptor, reserved) == 64);
static_assert(sizeof(NeoTradeOutcome) == 56);
static_assert(sizeof(NeoMetrics) == 80);
static_assert(sizeof(NeoPropFirmState) == 56);

extern "C" std::uint32_t neoethos_gpu_cuda_abi_version() {
  return NEOETHOS_GPU_ABI_VERSION;
}
