#include "neoethos_gpu_cuda.h"

// Honest no-CUDA build. Every entry point reports that the runtime is absent;
// none of them fabricates a successful-looking result.

extern "C" std::int32_t neoethos_gpu_cuda_runtime_available() { return 0; }
extern "C" std::int32_t neoethos_gpu_cuda_probe_device_count_v1(std::uint32_t*) {
  return NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE;
}
extern "C" std::int32_t neoethos_gpu_cuda_device_count() { return 0; }
extern "C" std::uint64_t neoethos_gpu_cuda_device_free_memory(std::int32_t) { return 0ull; }
extern "C" std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t*,
                                                  std::uint32_t*,
                                                  std::size_t) {
  return -1;
}
extern "C" std::int32_t neoethos_gpu_cuda_warp_first_hit(
    const double*,
    const double*,
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

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_parent_v1(
    NeoCudaPopulationSession*,
    const NeoPopulationParentDatasetV1*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" NeoCudaPopulationSession*
neoethos_gpu_cuda_population_bind_resident_feature_store_v3(
    const NeoPopulationResidentFeatureStoreV3*,
    std::int32_t* status) {
  if (status != nullptr) {
    *status = NEO_POPULATION_STATUS_UNSUPPORTED;
  }
  return nullptr;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_bind_view_v1(
    NeoCudaPopulationSession*,
    const NeoPopulationEvaluationViewV1*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_residency_counters_v1(
    NeoCudaPopulationSession*,
    NeoPopulationResidencyCountersV1*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_device_identity_v1(
    NeoCudaPopulationSession*,
    NeoPopulationDeviceIdentityV1*) {
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

extern "C" std::int32_t neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(
    NeoCudaPopulationSession*,
    const NeoPopulationSettings*,
    NeoPopulationResidentMetricsHandleV1*,
    NeoPopulationCounters*) {
  return NEO_POPULATION_STATUS_UNSUPPORTED;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(
    NeoCudaPopulationSession*,
    const NeoPopulationResidentMetricsHandleV1*,
    NeoPopulationTerminalCompactResultV1*) {
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
