#include "neoethos_gpu_cuda.h"

// Honest no-CUDA build. Every entry point reports that the runtime is absent;
// none of them fabricates a successful-looking result.

extern "C" std::int32_t neoethos_gpu_cuda_runtime_available() { return 0; }
extern "C" std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t*,
                                                  std::uint32_t*,
                                                  std::size_t) {
  return -1;
}
extern "C" std::int32_t neoethos_gpu_cuda_warp_first_hit(
    const float*,
    const float*,
    std::size_t,
    const NeoFirstHitEvent*,
    NeoFirstHitResult*,
    std::size_t) {
  return -1;
}

extern "C" NeoCudaPopulationSession* neoethos_gpu_cuda_population_create(
    std::uint32_t abi_version,
    std::int32_t,
    std::size_t,
    std::int32_t* status) {
  if (status != nullptr) {
    *status = abi_version == NEOETHOS_GPU_ABI_VERSION
                  ? NEO_POPULATION_STATUS_UNSUPPORTED
                  : NEO_POPULATION_STATUS_ABI_MISMATCH;
  }
  return nullptr;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_dataset(
    NeoCudaPopulationSession*,
    const NeoPopulationDatasetView*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_genes(
    NeoCudaPopulationSession*,
    const NeoPopulationGeneView*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_scenarios(
    NeoCudaPopulationSession*,
    const NeoPopulationScenarioView*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_b_evaluate(
    NeoCudaPopulationSession*,
    const NeoPopulationSettings*,
    std::uint64_t*,
    NeoPopulationCounters*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_wait(NeoCudaPopulationSession*,
                                                          std::uint64_t) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_metrics(
    NeoCudaPopulationSession*,
    NeoPopulationReadback*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_diagnostics(
    NeoCudaPopulationSession*,
    NeoPopulationDiagnosticReadback*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" void neoethos_gpu_cuda_population_destroy(NeoCudaPopulationSession*) {}
