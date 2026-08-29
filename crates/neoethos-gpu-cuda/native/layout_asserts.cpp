#include "neoethos_gpu_cuda.h"
#include <cstddef>
#include <type_traits>

static_assert(sizeof(NeoBufferRef) == 16);
static_assert(alignof(NeoBufferRef) == 8);
static_assert(sizeof(NeoHandleToken) == 32);
static_assert(offsetof(NeoHandleToken, generation) == 16);
static_assert(sizeof(NeoDatasetHeader) == 152);
static_assert(alignof(NeoDatasetHeader) == 8);
static_assert(offsetof(NeoDatasetHeader, timestamps) == 24);
static_assert(offsetof(NeoDatasetHeader, features) == 104);
static_assert(offsetof(NeoDatasetHeader, days) == 136);
static_assert(sizeof(NeoGeneDescriptor) == 72);
static_assert(alignof(NeoGeneDescriptor) == 8);
static_assert(offsetof(NeoGeneDescriptor, long_threshold) == 16);
static_assert(offsetof(NeoGeneDescriptor, short_threshold) == 24);
static_assert(offsetof(NeoGeneDescriptor, stop_ticks) == 32);
static_assert(offsetof(NeoGeneDescriptor, target_ticks) == 40);
static_assert(offsetof(NeoGeneDescriptor, stop_vol_multiplier) == 48);
static_assert(offsetof(NeoGeneDescriptor, flags) == 56);
static_assert(offsetof(NeoGeneDescriptor, reserved) == 64);
static_assert(sizeof(NeoScenarioDescriptor) == 72);
static_assert(offsetof(NeoScenarioDescriptor, commission_micros) == 48);
static_assert(offsetof(NeoScenarioDescriptor, reserved) == 64);
static_assert(sizeof(NeoTradeOutcome) == 56);
static_assert(sizeof(NeoMetrics) == 80);
static_assert(sizeof(NeoPropFirmState) == 56);
static_assert(sizeof(NeoFirstHitEvent) == 32);
static_assert(alignof(NeoFirstHitEvent) == 8);
static_assert(offsetof(NeoFirstHitEvent, stop_price) == 16);
static_assert(offsetof(NeoFirstHitEvent, target_price) == 24);
static_assert(sizeof(NeoFirstHitResult) == 8);

static_assert(sizeof(NeoPopulationSettings) == 184);
static_assert(offsetof(NeoPopulationSettings, spread_pips_asian) == 160);
static_assert(offsetof(NeoPopulationSettings, spread_pips_late_ny) == 176);
static_assert(offsetof(NeoPopulationSettings, trailing_enabled) == 128);
static_assert(offsetof(NeoPopulationSettings, trailing_min_lock_pips) == 152);
static_assert(alignof(NeoPopulationSettings) == 8);
static_assert(offsetof(NeoPopulationSettings, gap_threshold_ms) == 24);
static_assert(offsetof(NeoPopulationSettings, initial_equity) == 32);
static_assert(offsetof(NeoPopulationSettings, adaptive_rr) == 120);
static_assert(sizeof(NeoPopulationEvent) == 56);
static_assert(alignof(NeoPopulationEvent) == 8);
static_assert(offsetof(NeoPopulationEvent, direction) == 24);
static_assert(offsetof(NeoPopulationEvent, stop_price) == 32);
static_assert(offsetof(NeoPopulationEvent, entry_price) == 48);
static_assert(sizeof(NeoPopulationOutcome) == 72);
static_assert(offsetof(NeoPopulationOutcome, exit_price) == 48);
static_assert(alignof(NeoPopulationOutcome) == 8);
static_assert(offsetof(NeoPopulationOutcome, exit_bar) == 16);
static_assert(offsetof(NeoPopulationOutcome, entry_bar) == 24);
static_assert(offsetof(NeoPopulationOutcome, mfe) == 32);
static_assert(offsetof(NeoPopulationOutcome, mae) == 40);
static_assert(offsetof(NeoPopulationOutcome, pnl) == 56);
static_assert(offsetof(NeoPopulationOutcome, r_multiple) == 64);
static_assert(sizeof(NeoPopulationMetricRow) == 104);
static_assert(alignof(NeoPopulationMetricRow) == 8);
static_assert(offsetof(NeoPopulationMetricRow, values) == 16);
static_assert(sizeof(NeoPopulationCounters) == 96);
static_assert(alignof(NeoPopulationCounters) == 8);
static_assert(offsetof(NeoPopulationCounters, reserved) == 72);

static_assert(sizeof(NeoPopulationDatasetView) == 232);
static_assert(alignof(NeoPopulationDatasetView) == 8);
static_assert(offsetof(NeoPopulationDatasetView, close) == 152);
static_assert(offsetof(NeoPopulationDatasetView, indicators) == 176);
static_assert(offsetof(NeoPopulationDatasetView, smc_rows) == 208);
static_assert(offsetof(NeoPopulationDatasetView, adaptive_base_pips_len) == 224);
static_assert(std::is_same_v<decltype(NeoPopulationDatasetView::indicators), const double*>);
static_assert(sizeof(NeoPopulationParentDatasetV1) == 216);
static_assert(alignof(NeoPopulationParentDatasetV1) == 8);
static_assert(offsetof(NeoPopulationParentDatasetV1, close) == 152);
static_assert(offsetof(NeoPopulationParentDatasetV1, indicators_feature_major) == 176);
static_assert(offsetof(NeoPopulationParentDatasetV1, smc_rows) == 208);
static_assert(sizeof(NeoPopulationResidentFeatureStoreV3) == 216);
static_assert(alignof(NeoPopulationResidentFeatureStoreV3) == 8);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, row_count) == 8);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, compute_capability_major) == 24);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, packed_validity_bytes) == 32);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, close) == 40);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, indicators_bar_major) == 64);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, indicators_validity_u4) == 72);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, admitted_primary_context) == 112);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, device_uuid) == 136);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, admission_identity_sha256) == 152);
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3, canonical_content_merkle) == 184);
static_assert(std::is_same_v<decltype(NeoPopulationResidentFeatureStoreV3::months),
                             const std::int64_t*>);
static_assert(std::is_same_v<decltype(NeoPopulationResidentFeatureStoreV3::days),
                             const std::int64_t*>);
static_assert(std::is_same_v<decltype(NeoPopulationResidentFeatureStoreV3::timestamps),
                             const std::int64_t*>);
static_assert(sizeof(NeoPopulationEvaluationViewV1) == 72);
static_assert(alignof(NeoPopulationEvaluationViewV1) == 8);
static_assert(offsetof(NeoPopulationEvaluationViewV1, parent_row_count) == 8);
static_assert(offsetof(NeoPopulationEvaluationViewV1, ordered_indices) == 32);
static_assert(offsetof(NeoPopulationEvaluationViewV1, timestamp_mode) == 48);
static_assert(offsetof(NeoPopulationEvaluationViewV1, adaptive_base_pips) == 56);
static_assert(sizeof(NeoPopulationResidencyCountersV1) == 144);
static_assert(alignof(NeoPopulationResidencyCountersV1) == 8);
static_assert(offsetof(NeoPopulationResidencyCountersV1, metric_rows_readback_count) == 80);
static_assert(offsetof(NeoPopulationResidencyCountersV1, metric_rows_readback_rows) == 88);
static_assert(offsetof(NeoPopulationResidencyCountersV1, metric_rows_readback_bytes) == 96);
static_assert(offsetof(NeoPopulationResidencyCountersV1, diagnostic_readback_count) == 104);
static_assert(offsetof(NeoPopulationResidencyCountersV1, diagnostic_readback_rows) == 112);
static_assert(offsetof(NeoPopulationResidencyCountersV1, diagnostic_readback_bytes) == 120);
static_assert(offsetof(NeoPopulationResidencyCountersV1,
                       accepted_trade_total_readback_count) == 128);
static_assert(offsetof(NeoPopulationResidencyCountersV1,
                       accepted_trade_total_readback_bytes) == 136);
static_assert(sizeof(NeoPopulationResidentMetricsHandleV1) == 88);
static_assert(alignof(NeoPopulationResidentMetricsHandleV1) == 8);
static_assert(offsetof(NeoPopulationResidentMetricsHandleV1, event_id) == 8);
static_assert(offsetof(NeoPopulationResidentMetricsHandleV1, total_device_bytes) == 64);
static_assert(sizeof(NeoPopulationTerminalCompactResultV1) == 160);
static_assert(alignof(NeoPopulationTerminalCompactResultV1) == 8);
static_assert(offsetof(NeoPopulationTerminalCompactResultV1, event_id) == 8);
static_assert(offsetof(NeoPopulationTerminalCompactResultV1, metric_row) == 24);
static_assert(offsetof(NeoPopulationTerminalCompactResultV1,
                       terminal_readback_bytes) == 152);
static_assert(sizeof(NeoPopulationDeviceIdentityV1) == 312);
static_assert(alignof(NeoPopulationDeviceIdentityV1) == 8);
static_assert(offsetof(NeoPopulationDeviceIdentityV1, total_global_memory_bytes) == 16);
static_assert(offsetof(NeoPopulationDeviceIdentityV1, uuid) == 36);
static_assert(offsetof(NeoPopulationDeviceIdentityV1, name) == 52);
static_assert(sizeof(NeoPopulationGeneView) == 104);
static_assert(alignof(NeoPopulationGeneView) == 8);
static_assert(offsetof(NeoPopulationGeneView, count) == 8);
static_assert(offsetof(NeoPopulationGeneView, weights) == 32);
static_assert(offsetof(NeoPopulationGeneView, stop_pips) == 48);
static_assert(offsetof(NeoPopulationGeneView, smc_flags) == 72);
static_assert(offsetof(NeoPopulationGeneView, gate_threshold) == 88);
static_assert(offsetof(NeoPopulationGeneView, smc_gate_disabled) == 96);
static_assert(std::is_same_v<decltype(NeoPopulationGeneView::weights), const double*>);
static_assert(std::is_same_v<decltype(NeoPopulationGeneView::smc_weights), const double*>);
static_assert(std::is_same_v<decltype(NeoPopulationGeneView::gate_threshold), double>);
static_assert(sizeof(NeoPopulationScenarioView) == 16);
static_assert(alignof(NeoPopulationScenarioView) == 8);
static_assert(offsetof(NeoPopulationScenarioView, count) == 8);
static_assert(sizeof(NeoPopulationReadback) == 24);
static_assert(alignof(NeoPopulationReadback) == 8);
static_assert(offsetof(NeoPopulationReadback, written) == 16);
static_assert(sizeof(NeoPopulationDiagnosticReadback) == 32);
static_assert(alignof(NeoPopulationDiagnosticReadback) == 8);
static_assert(offsetof(NeoPopulationDiagnosticReadback, outcomes) == 8);
static_assert(offsetof(NeoPopulationDiagnosticReadback, written) == 24);

using NeoAbiVersionFn = std::uint32_t (*)();
using NeoRuntimeAvailableFn = std::int32_t (*)();
using NeoProbeDeviceCountV1Fn = std::int32_t (*)(std::uint32_t*);
using NeoDeviceCountFn = std::int32_t (*)();
using NeoDeviceFreeMemoryFn = std::uint64_t (*)(std::int32_t);
using NeoSmokeFn = std::int32_t (*)(const std::uint32_t*, std::uint32_t*, std::size_t);
using NeoWarpFirstHitFn = std::int32_t (*)(const double*, const double*, std::size_t,
                                           const NeoFirstHitEvent*, NeoFirstHitResult*,
                                           std::size_t);
using NeoPopulationCreateFn = NeoCudaPopulationSession* (*)(std::uint32_t, std::int32_t,
                                                             std::size_t, std::int32_t*);
using NeoPopulationUploadDatasetFn = std::int32_t (*)(NeoCudaPopulationSession*,
                                                       const NeoPopulationDatasetView*);
using NeoPopulationUploadParentV1Fn = std::int32_t (*)(
    NeoCudaPopulationSession*, const NeoPopulationParentDatasetV1*);
using NeoPopulationBindResidentFeatureStoreV3Fn = NeoCudaPopulationSession* (*)(
    const NeoPopulationResidentFeatureStoreV3*, std::int32_t*);
using NeoPopulationBindViewV1Fn = std::int32_t (*)(
    NeoCudaPopulationSession*, const NeoPopulationEvaluationViewV1*);
using NeoPopulationReadResidencyCountersV1Fn = std::int32_t (*)(
    NeoCudaPopulationSession*, NeoPopulationResidencyCountersV1*);
using NeoPopulationReadDeviceIdentityV1Fn = std::int32_t (*)(
    NeoCudaPopulationSession*, NeoPopulationDeviceIdentityV1*);
using NeoPopulationUploadGenesFn = std::int32_t (*)(NeoCudaPopulationSession*,
                                                     const NeoPopulationGeneView*);
using NeoPopulationUploadScenariosFn = std::int32_t (*)(NeoCudaPopulationSession*,
                                                         const NeoPopulationScenarioView*);
using NeoPopulationEvaluateFn = std::int32_t (*)(NeoCudaPopulationSession*,
                                                  const NeoPopulationSettings*, std::uint64_t*,
                                                  NeoPopulationCounters*);
using NeoPopulationEnqueueMetricsOnlyV1Fn = std::int32_t (*)(
    NeoCudaPopulationSession*, const NeoPopulationSettings*,
    NeoPopulationResidentMetricsHandleV1*, NeoPopulationCounters*);
using NeoPopulationConsumeTerminalCompactResultV1Fn = std::int32_t (*)(
    NeoCudaPopulationSession*, const NeoPopulationResidentMetricsHandleV1*,
    NeoPopulationTerminalCompactResultV1*);
using NeoPopulationWaitFn = std::int32_t (*)(NeoCudaPopulationSession*, std::uint64_t);
using NeoPopulationReadMetricsFn = std::int32_t (*)(NeoCudaPopulationSession*,
                                                     NeoPopulationReadback*);
using NeoPopulationReadDiagnosticsFn = std::int32_t (*)(NeoCudaPopulationSession*,
                                                         NeoPopulationDiagnosticReadback*);
using NeoPopulationDestroyFn = void (*)(NeoCudaPopulationSession*);

static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_abi_version), NeoAbiVersionFn>);
static_assert(
    std::is_same_v<decltype(&neoethos_gpu_cuda_runtime_available), NeoRuntimeAvailableFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_probe_device_count_v1),
                             NeoProbeDeviceCountV1Fn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_device_count), NeoDeviceCountFn>);
static_assert(
    std::is_same_v<decltype(&neoethos_gpu_cuda_device_free_memory), NeoDeviceFreeMemoryFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_smoke), NeoSmokeFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_warp_first_hit), NeoWarpFirstHitFn>);
static_assert(
    std::is_same_v<decltype(&neoethos_gpu_cuda_population_create), NeoPopulationCreateFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_upload_dataset),
                             NeoPopulationUploadDatasetFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_upload_parent_v1),
                             NeoPopulationUploadParentV1Fn>);
static_assert(
    std::is_same_v<decltype(&neoethos_gpu_cuda_population_bind_resident_feature_store_v3),
                   NeoPopulationBindResidentFeatureStoreV3Fn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_bind_view_v1),
                             NeoPopulationBindViewV1Fn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_read_residency_counters_v1),
                             NeoPopulationReadResidencyCountersV1Fn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_read_device_identity_v1),
                             NeoPopulationReadDeviceIdentityV1Fn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_upload_genes),
                             NeoPopulationUploadGenesFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_upload_scenarios),
                             NeoPopulationUploadScenariosFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_b_evaluate),
                             NeoPopulationEvaluateFn>);
static_assert(
    std::is_same_v<decltype(&neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1),
                   NeoPopulationEnqueueMetricsOnlyV1Fn>);
static_assert(std::is_same_v<
              decltype(&neoethos_gpu_cuda_population_consume_terminal_compact_result_v1),
              NeoPopulationConsumeTerminalCompactResultV1Fn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_wait), NeoPopulationWaitFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_read_metrics),
                             NeoPopulationReadMetricsFn>);
static_assert(std::is_same_v<decltype(&neoethos_gpu_cuda_population_read_diagnostics),
                             NeoPopulationReadDiagnosticsFn>);
static_assert(
    std::is_same_v<decltype(&neoethos_gpu_cuda_population_destroy), NeoPopulationDestroyFn>);

extern "C" std::uint32_t neoethos_gpu_cuda_abi_version() {
  return NEOETHOS_GPU_ABI_VERSION;
}
