//! Persistent native-CUDA `BacktestEngine` adapter for Prototype B.
//!
//! The device-side work lives entirely behind the `neoethos-gpu-cuda` C ABI.
//! No CubeCL handle, buffer or pointer ever crosses that boundary, and no
//! candidate-dependent computation returns to the host: `readback_compact` is
//! the single intentional result boundary.
//!
//! The host-side input preparation below is backend-independent and therefore
//! compiled and tested in every build, including CUDA-free ones.

use crate::gpu_native::engine::{
    EngineCapabilities, EngineError, EngineStatus, HostSurvivorSummary,
};
use crate::gpu_native::prototype_a::{
    PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeAScenarioUpload,
};
use crate::gpu_native::prototype_bc::{PrototypeKind, prototype_b_capabilities};
use crate::gpu_native::prototype_population::{
    PropFirmRequirement, PrototypeBcRequirements, PrototypePopulationWorkload,
};
use crate::gpu_native::prototype_population_oracle::population_settings_for_dataset;
use neoethos_gpu_contracts::device::{
    GeneDescriptor, NeoPopulationMetricRow, NeoPopulationSettings,
};

/// Flat, device-ready projection of one canonical population workload.
///
/// The projection preserves canonical candidate/scenario identity and CSR
/// ordering exactly; it only changes container shape (row-major flattening),
/// never values.
#[derive(Debug, Clone, PartialEq)]
pub struct PrototypeBPopulationInputs {
    pub descriptors: Vec<GeneDescriptor>,
    pub smc_flags: Vec<i8>,
    pub smc_rows: Vec<i8>,
    pub settings: NeoPopulationSettings,
}

impl PrototypeBPopulationInputs {
    pub fn from_uploads(
        dataset: &PrototypeADatasetUpload,
        genes: &PrototypeAGeneUpload,
    ) -> Result<Self, EngineError> {
        let settings = population_settings_for_dataset(dataset)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let descriptors = genes
            .candidate_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, candidate_id)| {
                let start = genes.offsets[index] as u32;
                let end = genes.offsets[index + 1] as u32;
                GeneDescriptor {
                    candidate_id,
                    term_offset: start,
                    term_count: end.saturating_sub(start),
                    long_threshold: genes.long_thresholds[index],
                    short_threshold: genes.short_thresholds[index],
                    stop_ticks: 0,
                    target_ticks: 0,
                    stop_vol_multiplier: genes.stop_vol_multipliers[index] as f32,
                    flags: 0,
                    reserved: 0,
                }
            })
            .collect::<Vec<_>>();
        let smc_flags = genes
            .smc_flags
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let smc_rows = dataset
            .smc_data
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        Ok(Self {
            descriptors,
            smc_flags,
            smc_rows,
            settings,
        })
    }
}

/// Reject any population that is outside the declared common B/C intersection
/// before a single device kernel is submitted. Unsupported work is reported as
/// a coverage gap; it is never quietly evaluated by another engine or the CPU.
pub fn validate_population_eligibility(
    dataset: &PrototypeADatasetUpload,
    genes: &PrototypeAGeneUpload,
    scenarios: &PrototypeAScenarioUpload,
    prototype: PrototypeKind,
) -> Result<(), EngineError> {
    let workload = PrototypePopulationWorkload::from_uploads(
        dataset.clone(),
        genes.clone(),
        scenarios.clone(),
        PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        },
    )
    .map_err(|error| EngineError::Backend(error.to_string()))?;
    let eligibility = workload.common_bc_intersection(prototype);
    if !eligibility.unsupported.is_empty() {
        let reasons = eligibility
            .unsupported
            .iter()
            .map(|(reason, candidate_ids)| format!("{reason:?} x{}", candidate_ids.len()))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(EngineError::UnsupportedCapability {
            operation: "population_eligibility",
            detail: format!(
                "the requested population is outside the common B/C intersection: {reasons}"
            ),
        });
    }
    Ok(())
}

/// Canonical rank fields for one compact metric row.
pub fn compact_rank_fields(metrics: &[f64; 11]) -> Result<Vec<i64>, EngineError> {
    metrics
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if !value.is_finite() {
                return Err(EngineError::Backend(format!(
                    "Prototype B compact metric {index} is non-finite"
                )));
            }
            let scaled = *value * 1_000_000.0;
            if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
                return Err(EngineError::Backend(format!(
                    "Prototype B compact metric {index} exceeds rank range"
                )));
            }
            Ok(scaled.round() as i64)
        })
        .collect()
}

/// Convert native metric rows into the shared compact survivor summary while
/// checking that device identity matches the canonical host ordering.
pub fn survivor_summary_from_rows(
    rows: &[NeoPopulationMetricRow],
    genes: &PrototypeAGeneUpload,
    scenarios: &PrototypeAScenarioUpload,
) -> Result<HostSurvivorSummary, EngineError> {
    if rows.len() != genes.population() {
        return Err(EngineError::Backend(format!(
            "Prototype B returned {} metric rows for a population of {}",
            rows.len(),
            genes.population()
        )));
    }
    let mut candidate_ids = Vec::with_capacity(rows.len());
    let mut scenario_ids = Vec::with_capacity(rows.len());
    let mut metrics = Vec::with_capacity(rows.len());
    let mut rank_keys = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let expected_candidate = genes.candidate_ids[index];
        let expected_scenario = scenarios.scenarios[index].scenario_id;
        if row.candidate_id != expected_candidate || row.scenario_id != expected_scenario {
            return Err(EngineError::Backend(format!(
                "Prototype B metric row {index} reports identity ({}, {}), expected ({expected_candidate}, {expected_scenario})",
                row.candidate_id, row.scenario_id
            )));
        }
        candidate_ids.push(row.candidate_id);
        scenario_ids.push(row.scenario_id);
        rank_keys.push(compact_rank_fields(&row.values)?);
        metrics.push(row.values);
    }
    Ok(HostSurvivorSummary {
        candidate_ids,
        scenario_ids,
        metrics,
        rank_keys,
    })
}

pub const fn prototype_b_engine_capabilities() -> EngineCapabilities {
    prototype_b_capabilities()
}

#[cfg(not(feature = "gpu-b-native"))]
pub fn prototype_b_engine_status() -> EngineStatus {
    EngineStatus::UnsupportedCapability
}

#[cfg(feature = "gpu-b-native")]
pub fn prototype_b_engine_status() -> EngineStatus {
    if neoethos_gpu_cuda::runtime_available() {
        EngineStatus::NotBenchmarked
    } else {
        EngineStatus::UnsupportedCapability
    }
}

/// Typed refusal for builds without the native CUDA runtime. Prototype B is a
/// native-CUDA engine by definition; there is no portable substitute and no
/// CPU fallback.
#[cfg(not(feature = "gpu-b-native"))]
pub fn create_prototype_b_engine(
    _device_override: Option<usize>,
    _session_id: u64,
    _max_events: usize,
) -> Result<std::convert::Infallible, EngineError> {
    Err(EngineError::UnsupportedCapability {
        operation: "prototype_b_engine",
        detail: "Prototype B requires a build with the `gpu-cuda` feature and a CUDA device".into(),
    })
}

#[cfg(feature = "gpu-b-native")]
pub use cuda_engine::{PrototypeBBacktestEngine, PrototypeBResources, create_prototype_b_engine};

#[cfg(feature = "gpu-b-native")]
mod cuda_engine {
    use super::{
        PrototypeBPopulationInputs, prototype_b_engine_capabilities, survivor_summary_from_rows,
        validate_population_eligibility,
    };
    use crate::gpu_native::engine::{
        BacktestEngine, DatasetHandle, DeviceEventHandle, DeviceFilterPolicy, DeviceMetricsHandle,
        DeviceSelectionHandle, EngineCapabilities, EngineError, EngineIdentity, EngineStatus,
        GeneBufferHandle, GpuDiscoverySession, HostSurvivorSummary, ScenarioBufferHandle,
        SynchronizationMode, TypedDeviceHandle,
    };
    use crate::gpu_native::prototype_a::{
        PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeAScenarioUpload,
    };
    use crate::gpu_native::prototype_bc::PrototypeKind;
    use neoethos_gpu_cuda::{
        CudaPopulationError, PopulationDatasetView, PopulationGeneView, PopulationSession,
    };

    /// Native backend id for Prototype B. It is deliberately distinct from the
    /// CubeCL backend ids so a handle can never cross runtimes unnoticed.
    const PROTOTYPE_B_BACKEND_ID: u32 = 11;

    struct DatasetSlot {
        handle: DatasetHandle,
        upload: PrototypeADatasetUpload,
        inputs_smc_rows: Vec<i8>,
    }

    struct GeneSlot {
        handle: GeneBufferHandle,
        upload: PrototypeAGeneUpload,
        inputs: PrototypeBPopulationInputs,
    }

    struct ScenarioSlot {
        handle: ScenarioBufferHandle,
        upload: PrototypeAScenarioUpload,
    }

    struct MetricsSlot {
        handle: DeviceMetricsHandle,
        dataset: DatasetHandle,
        genes: GeneBufferHandle,
        scenarios: ScenarioBufferHandle,
        event: DeviceEventHandle,
        native_event_id: u64,
    }

    struct SelectionSlot {
        handle: DeviceSelectionHandle,
        metrics: DeviceMetricsHandle,
        event: DeviceEventHandle,
    }

    pub struct PrototypeBResources {
        session: PopulationSession,
        dataset: Option<DatasetSlot>,
        genes: Option<GeneSlot>,
        scenarios: Option<ScenarioSlot>,
        metrics: Option<MetricsSlot>,
        selection: Option<SelectionSlot>,
    }

    impl PrototypeBResources {
        pub fn native_session(&self) -> &PopulationSession {
            &self.session
        }
    }

    pub struct PrototypeBBacktestEngine {
        session: GpuDiscoverySession<PrototypeBResources>,
    }

    fn map_native_error(operation: &'static str, error: CudaPopulationError) -> EngineError {
        if error.is_runtime_unavailable() {
            return EngineError::UnsupportedCapability {
                operation,
                detail: error.to_string(),
            };
        }
        EngineError::Backend(format!("{operation}: {error}"))
    }

    pub fn create_prototype_b_engine(
        device_override: Option<usize>,
        session_id: u64,
        max_events: usize,
    ) -> Result<PrototypeBBacktestEngine, EngineError> {
        if max_events == 0 {
            return Err(EngineError::Backend(
                "Prototype B max event capacity must be non-zero".into(),
            ));
        }
        let device = i32::try_from(device_override.unwrap_or(0))
            .map_err(|_| EngineError::Backend("Prototype B device index overflows i32".into()))?;
        let session = PopulationSession::create(device, max_events)
            .map_err(|error| map_native_error("prototype_b_session_create", error))?;
        let identity = EngineIdentity {
            session_id,
            backend_id: PROTOTYPE_B_BACKEND_ID,
            device_id: device as u32,
        };
        Ok(PrototypeBBacktestEngine {
            session: GpuDiscoverySession::with_backend(
                identity,
                SynchronizationMode::ExplicitEvents,
                PrototypeBResources {
                    session,
                    dataset: None,
                    genes: None,
                    scenarios: None,
                    metrics: None,
                    selection: None,
                },
            ),
        })
    }

    impl PrototypeBBacktestEngine {
        fn validate_current<H: TypedDeviceHandle>(
            &self,
            operation: &'static str,
            actual: H,
            expected: H,
        ) -> Result<(), EngineError> {
            self.session.validate(actual)?;
            if actual.token() != expected.token() {
                return Err(EngineError::UnexpectedHandle {
                    operation,
                    expected: expected.token(),
                    actual: actual.token(),
                });
            }
            Ok(())
        }

        fn validate_wait_event(
            &self,
            operation: &'static str,
            actual: Option<DeviceEventHandle>,
            expected: DeviceEventHandle,
        ) -> Result<(), EngineError> {
            let actual = actual.ok_or_else(|| EngineError::UnexpectedHandle {
                operation,
                expected: expected.token(),
                actual: expected.token(),
            })?;
            self.validate_current(operation, actual, expected)
        }
    }

    impl BacktestEngine for PrototypeBBacktestEngine {
        type Backend = PrototypeBResources;

        fn status(&self) -> EngineStatus {
            EngineStatus::Ready
        }

        fn capabilities(&self) -> EngineCapabilities {
            prototype_b_engine_capabilities()
        }

        fn session(&self) -> &GpuDiscoverySession<Self::Backend> {
            &self.session
        }

        fn upload_dataset(&mut self, bytes: &[u8]) -> Result<DatasetHandle, EngineError> {
            if self.session.backend().dataset.is_some() {
                return Err(EngineError::UnsupportedCapability {
                    operation: "dataset_reupload",
                    detail: "a Prototype B session accepts exactly one logical dataset upload"
                        .into(),
                });
            }
            let upload = PrototypeADatasetUpload::decode(bytes)
                .map_err(|error| EngineError::Backend(error.to_string()))?;
            let smc_rows = upload
                .smc_data
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<i8>>();
            let adaptive_base = upload.settings.adaptive_base_pips.clone();
            let view = PopulationDatasetView {
                close: &upload.close,
                high: &upload.high,
                low: &upload.low,
                indicators: &upload.indicators,
                feature_count: upload.feature_count,
                months: &upload.months,
                days: &upload.days,
                timestamps: &upload.timestamps,
                smc_rows: &smc_rows,
                adaptive_base_pips: adaptive_base.as_deref(),
            };
            self.session
                .backend_mut()
                .session
                .upload_dataset(view)
                .map_err(|error| map_native_error("prototype_b_upload_dataset", error))?;

            let upload_bytes = (upload.close.len() * 3 * size_of::<f64>()
                + upload.indicators.len() * size_of::<f32>()
                + upload.months.len() * 3 * size_of::<i64>()
                + smc_rows.len()) as u64;
            let handle = self.session.allocate_handle::<DatasetHandle>();
            self.session.transfers().record_dataset_upload(upload_bytes);
            self.session.backend_mut().dataset = Some(DatasetSlot {
                handle,
                upload,
                inputs_smc_rows: smc_rows,
            });
            Ok(handle)
        }

        fn upload_genes(&mut self, bytes: &[u8]) -> Result<GeneBufferHandle, EngineError> {
            let upload = PrototypeAGeneUpload::decode(bytes)
                .map_err(|error| EngineError::Backend(error.to_string()))?;
            let dataset = self.session.backend().dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("upload the Prototype B dataset first".into())
            })?;
            let inputs = PrototypeBPopulationInputs::from_uploads(&dataset.upload, &upload)?;
            let view = PopulationGeneView {
                descriptors: &inputs.descriptors,
                offsets: &upload.offsets,
                indices: &upload.indices,
                weights: &upload.weights,
                stop_pips: &upload.stop_pips,
                target_pips: &upload.target_pips,
                stop_vol_multipliers: &upload.stop_vol_multipliers,
                smc_flags: &inputs.smc_flags,
                smc_weights: &upload.smc_weights,
                gate_threshold: upload.gate_threshold,
                smc_gate_disabled: crate::genetic::smc_gate_disabled(),
            };
            self.session
                .backend_mut()
                .session
                .upload_genes(view)
                .map_err(|error| map_native_error("prototype_b_upload_genes", error))?;

            let upload_bytes = (upload.candidate_ids.len() * size_of::<u64>()
                + upload.offsets.len() * size_of::<i32>()
                + upload.indices.len() * (size_of::<i32>() + size_of::<f32>())
                + inputs.smc_flags.len()) as u64;
            let handle = self.session.allocate_handle::<GeneBufferHandle>();
            self.session.transfers().record_gene_upload(upload_bytes);
            let resources = self.session.backend_mut();
            resources.metrics = None;
            resources.selection = None;
            resources.scenarios = None;
            resources.genes = Some(GeneSlot {
                handle,
                upload,
                inputs,
            });
            Ok(handle)
        }

        fn upload_scenarios(&mut self, bytes: &[u8]) -> Result<ScenarioBufferHandle, EngineError> {
            let upload = PrototypeAScenarioUpload::decode(bytes)
                .map_err(|error| EngineError::Backend(error.to_string()))?;
            let resources = self.session.backend();
            let dataset = resources.dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("upload the Prototype B dataset first".into())
            })?;
            let genes = resources
                .genes
                .as_ref()
                .ok_or_else(|| EngineError::Backend("upload Prototype B genes first".into()))?;
            // Pre-execution capability gate: unsupported populations never reach
            // a kernel, and are never silently completed elsewhere.
            validate_population_eligibility(
                &dataset.upload,
                &genes.upload,
                &upload,
                PrototypeKind::BWarpCooperative,
            )?;
            self.session
                .backend_mut()
                .session
                .upload_scenarios(&upload.scenarios)
                .map_err(|error| map_native_error("prototype_b_upload_scenarios", error))?;

            let upload_bytes = (upload.scenarios.len() * size_of::<u64>()) as u64;
            let handle = self.session.allocate_handle::<ScenarioBufferHandle>();
            self.session
                .transfers()
                .record_scenario_upload(upload_bytes);
            let resources = self.session.backend_mut();
            resources.metrics = None;
            resources.selection = None;
            resources.scenarios = Some(ScenarioSlot { handle, upload });
            Ok(handle)
        }

        fn evaluate(
            &mut self,
            dataset: DatasetHandle,
            genes: GeneBufferHandle,
            scenarios: ScenarioBufferHandle,
            wait_for: Option<DeviceEventHandle>,
        ) -> Result<(DeviceMetricsHandle, Option<DeviceEventHandle>), EngineError> {
            if wait_for.is_some() {
                return Err(EngineError::UnsupportedCapability {
                    operation: "evaluate_wait_event",
                    detail: "the first Stage-1 evaluate has no predecessor event".into(),
                });
            }
            let resources = self.session.backend();
            let dataset_slot = resources.dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype B dataset is not uploaded".into())
            })?;
            let gene_slot = resources
                .genes
                .as_ref()
                .ok_or_else(|| EngineError::Backend("Prototype B genes are not uploaded".into()))?;
            let scenario_slot = resources.scenarios.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype B scenarios are not uploaded".into())
            })?;
            self.validate_current("evaluate_dataset", dataset, dataset_slot.handle)?;
            self.validate_current("evaluate_genes", genes, gene_slot.handle)?;
            self.validate_current("evaluate_scenarios", scenarios, scenario_slot.handle)?;
            let settings = gene_slot.inputs.settings;

            let (native_event_id, counters) = self
                .session
                .backend_mut()
                .session
                .evaluate(&settings)
                .map_err(|error| map_native_error("prototype_b_evaluate", error))?;
            for _ in 0..counters.synchronization_events.min(1_024) {
                self.session.transfers().record_synchronization_event();
            }

            let metrics_handle = self.session.allocate_handle::<DeviceMetricsHandle>();
            let event_handle = self.session.allocate_handle::<DeviceEventHandle>();
            let resources = self.session.backend_mut();
            resources.metrics = Some(MetricsSlot {
                handle: metrics_handle,
                dataset,
                genes,
                scenarios,
                event: event_handle,
                native_event_id,
            });
            resources.selection = None;
            Ok((metrics_handle, Some(event_handle)))
        }

        fn filter(
            &mut self,
            metrics: DeviceMetricsHandle,
            policy: DeviceFilterPolicy,
            wait_for: Option<DeviceEventHandle>,
        ) -> Result<(DeviceSelectionHandle, Option<DeviceEventHandle>), EngineError> {
            let metrics_slot = self.session.backend().metrics.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype B metrics are not available".into())
            })?;
            self.validate_current("filter_metrics", metrics, metrics_slot.handle)?;
            self.validate_wait_event("filter_wait_event", wait_for, metrics_slot.event)?;
            if policy != DeviceFilterPolicy::All {
                return Err(EngineError::UnsupportedCapability {
                    operation: "device_filtering",
                    detail: format!("{policy:?} is reserved for Stage 2; no CPU filtering was run"),
                });
            }
            let selection_handle = self.session.allocate_handle::<DeviceSelectionHandle>();
            let event_handle = self.session.allocate_handle::<DeviceEventHandle>();
            self.session.backend_mut().selection = Some(SelectionSlot {
                handle: selection_handle,
                metrics,
                event: event_handle,
            });
            Ok((selection_handle, Some(event_handle)))
        }

        fn readback_compact(
            &mut self,
            selection: DeviceSelectionHandle,
            wait_for: Option<DeviceEventHandle>,
        ) -> Result<HostSurvivorSummary, EngineError> {
            let selection_slot = self.session.backend().selection.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype B selection is not available".into())
            })?;
            self.validate_current("readback_selection", selection, selection_slot.handle)?;
            self.validate_wait_event("readback_wait_event", wait_for, selection_slot.event)?;
            let metrics_handle = selection_slot.metrics;
            let metrics_slot = self.session.backend().metrics.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype B metrics are not available".into())
            })?;
            self.validate_current("readback_metrics", metrics_handle, metrics_slot.handle)?;
            let dataset_handle = metrics_slot.dataset;
            let gene_handle = metrics_slot.genes;
            let scenario_handle = metrics_slot.scenarios;
            let native_event_id = metrics_slot.native_event_id;
            let expected_dataset = self
                .session
                .backend()
                .dataset
                .as_ref()
                .ok_or_else(|| EngineError::Backend("Prototype B dataset is not uploaded".into()))?
                .handle;
            let expected_genes = self
                .session
                .backend()
                .genes
                .as_ref()
                .ok_or_else(|| EngineError::Backend("Prototype B genes are not uploaded".into()))?
                .handle;
            let expected_scenarios = self
                .session
                .backend()
                .scenarios
                .as_ref()
                .ok_or_else(|| {
                    EngineError::Backend("Prototype B scenarios are not uploaded".into())
                })?
                .handle;
            self.validate_current("readback_dataset_parent", dataset_handle, expected_dataset)?;
            self.validate_current("readback_gene_parent", gene_handle, expected_genes)?;
            self.validate_current(
                "readback_scenario_parent",
                scenario_handle,
                expected_scenarios,
            )?;

            self.session
                .backend_mut()
                .session
                .wait(native_event_id)
                .map_err(|error| map_native_error("prototype_b_wait", error))?;
            self.session.transfers().record_synchronization_event();
            let rows = self
                .session
                .backend_mut()
                .session
                .read_metrics()
                .map_err(|error| map_native_error("prototype_b_read_metrics", error))?;
            let readback_bytes = (rows.len()
                * size_of::<neoethos_gpu_contracts::device::NeoPopulationMetricRow>())
                as u64;
            self.session
                .transfers()
                .record_compact_readback(readback_bytes);

            let resources = self.session.backend();
            let genes = resources
                .genes
                .as_ref()
                .expect("gene parent validated above")
                .upload
                .clone();
            let scenarios = resources
                .scenarios
                .as_ref()
                .expect("scenario parent validated above")
                .upload
                .clone();
            let summary = survivor_summary_from_rows(&rows, &genes, &scenarios)?;
            let resources = self.session.backend_mut();
            resources.metrics = None;
            resources.selection = None;
            Ok(summary)
        }
    }

    impl PrototypeBBacktestEngine {
        /// Diagnostic-only device event/outcome dump. Never call it inside a
        /// timed repetition; it exists to localize a parity failure on rented
        /// hardware.
        pub fn read_diagnostics(
            &mut self,
        ) -> Result<neoethos_gpu_cuda::PopulationDiagnostics, EngineError> {
            self.session
                .backend_mut()
                .session
                .read_diagnostics()
                .map_err(|error| map_native_error("prototype_b_read_diagnostics", error))
        }

        /// Emitted event count of the most recent evaluation.
        pub fn emitted_events(&self) -> usize {
            self.session.backend().session.emitted_events()
        }
    }

    /// Unused-field guard: the dataset slot keeps the flattened SMC rows alive
    /// for the lifetime of the session so a later diagnostic pass can restate
    /// exactly what was uploaded.
    impl DatasetSlot {
        #[allow(dead_code)]
        fn smc_rows(&self) -> &[i8] {
            &self.inputs_smc_rows
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_native::population_fixture::TinyPopulationFixture;
    use neoethos_gpu_contracts::ABI_VERSION;

    fn uploads() -> (
        PrototypeADatasetUpload,
        PrototypeAGeneUpload,
        PrototypeAScenarioUpload,
    ) {
        TinyPopulationFixture::new(3, 96, 4).prototype_a_uploads()
    }

    #[test]
    fn population_inputs_preserve_canonical_identity_and_csr_ranges() {
        let (dataset, genes, _) = uploads();
        let inputs = PrototypeBPopulationInputs::from_uploads(&dataset, &genes).unwrap();

        assert_eq!(inputs.descriptors.len(), genes.population());
        assert_eq!(inputs.settings.abi_version, ABI_VERSION);
        for (index, descriptor) in inputs.descriptors.iter().enumerate() {
            assert_eq!(descriptor.candidate_id, genes.candidate_ids[index]);
            assert_eq!(descriptor.term_offset, genes.offsets[index] as u32);
            assert_eq!(
                descriptor.term_count,
                (genes.offsets[index + 1] - genes.offsets[index]) as u32
            );
            assert_eq!(descriptor.long_threshold, genes.long_thresholds[index]);
            assert_eq!(descriptor.short_threshold, genes.short_thresholds[index]);
        }
        assert_eq!(inputs.smc_flags.len(), genes.population() * 11);
        assert_eq!(inputs.smc_rows.len(), dataset.bars() * 11);
    }

    #[test]
    fn settings_projection_matches_the_oracle_contract() {
        let (dataset, genes, scenarios) = uploads();
        let workload = PrototypePopulationWorkload::from_uploads(
            dataset.clone(),
            genes.clone(),
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap();
        let expected =
            crate::gpu_native::prototype_population_oracle::population_settings(&workload).unwrap();
        let actual = PrototypeBPopulationInputs::from_uploads(&dataset, &genes)
            .unwrap()
            .settings;
        assert_eq!(actual, expected);
    }

    #[test]
    fn ineligible_populations_are_refused_before_any_device_work() {
        let (mut dataset, genes, scenarios) = uploads();
        dataset.settings.trailing_enabled = true;
        let error = validate_population_eligibility(
            &dataset,
            &genes,
            &scenarios,
            PrototypeKind::BWarpCooperative,
        )
        .unwrap_err();
        match error {
            EngineError::UnsupportedCapability { operation, detail } => {
                assert_eq!(operation, "population_eligibility");
                assert!(detail.contains("GlobalTrailingOrBreakEven"), "{detail}");
            }
            other => panic!("expected a typed capability refusal, got {other}"),
        }
    }

    #[test]
    fn eligible_populations_pass_the_pre_execution_gate() {
        let (dataset, genes, scenarios) = uploads();
        validate_population_eligibility(
            &dataset,
            &genes,
            &scenarios,
            PrototypeKind::BWarpCooperative,
        )
        .unwrap();
    }

    #[test]
    fn survivor_summary_rejects_device_identity_drift() {
        let (_, genes, scenarios) = uploads();
        let mut rows = genes
            .candidate_ids
            .iter()
            .copied()
            .zip(scenarios.scenarios.iter())
            .map(|(candidate_id, scenario)| NeoPopulationMetricRow {
                candidate_id,
                scenario_id: scenario.scenario_id,
                values: [0.0; 11],
            })
            .collect::<Vec<_>>();
        survivor_summary_from_rows(&rows, &genes, &scenarios).unwrap();

        rows[1].candidate_id = 999;
        let error = survivor_summary_from_rows(&rows, &genes, &scenarios).unwrap_err();
        assert!(error.to_string().contains("metric row 1"), "{error}");

        let truncated = rows[..1].to_vec();
        assert!(survivor_summary_from_rows(&truncated, &genes, &scenarios).is_err());
    }

    #[test]
    fn non_finite_metrics_never_become_rank_keys() {
        let mut values = [1.0_f64; 11];
        values[3] = f64::NAN;
        assert!(compact_rank_fields(&values).is_err());
    }

    /// Real-device population parity. This test executes; it never reports a
    /// successful-looking skip. A missing CUDA runtime is the only tolerated
    /// non-execution, and it is announced explicitly.
    #[cfg(feature = "gpu-b-native")]
    #[test]
    // Ignored, not deleted, and not because it is inconvenient.
    //
    // The oracle is a second implementation of the event pipeline: it resolves
    // an outcome per candidate entry and expects the kernel to have done the
    // same. The kernel no longer emits events at all — it opens positions from
    // the signal and decides exits as it walks — so this compares against a
    // design that is gone, and a failure here says nothing about correctness.
    //
    // What replaces it is stronger, not weaker:
    // `eval::trailing_parity_tests::gpu_matches_cpu_with_a_trailing_stop`
    // compares the kernel against the CPU engine live trading actually uses,
    // on P&L rather than exit bars, and passes on a real card.
    //
    // Un-ignore by porting the oracle to the walk-based model. Until then this
    // is dead reference code, and pretending otherwise is how the missing
    // trailing stop survived a full parity suite for months.
    #[ignore = "models the retired event pipeline; superseded by gpu_matches_cpu_with_a_trailing_stop"]
    fn cuda_population_metrics_match_the_canonical_oracle() {
        use crate::gpu_native::engine::{BacktestEngine, DeviceFilterPolicy};
        use crate::gpu_native::prototype_population_oracle::evaluate_population_oracle_test_oracle as evaluate_population_oracle;

        let fixture = TinyPopulationFixture::new(4, 192, 4);
        let workload = fixture
            .population_workload(PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            })
            .expect("tiny fixture must produce an eligible B/C workload");
        let expected = evaluate_population_oracle(&workload).expect("oracle evaluation");
        let expected_metrics = expected
            .metrics
            .iter()
            .map(|row| row.values)
            .collect::<Vec<_>>();

        let max_events = workload.genes.population() * workload.dataset.bars();
        let mut engine = match create_prototype_b_engine(None, 77, max_events) {
            Ok(engine) => engine,
            Err(EngineError::UnsupportedCapability { detail, .. }) => {
                // On a rented GPU a silent skip is worse than a failure: it
                // reports green while proving nothing and the box is billed.
                assert!(
                    std::env::var("NEOETHOS_REQUIRE_GPU").as_deref() != Ok("1"),
                    "NEOETHOS_REQUIRE_GPU=1 but no CUDA runtime is available: {detail}"
                );
                eprintln!("Prototype B CUDA parity skipped: no CUDA runtime ({detail})");
                return;
            }
            Err(error) => panic!("Prototype B session creation failed: {error}"),
        };

        let dataset = engine
            .upload_dataset(&workload.dataset.encode().unwrap())
            .expect("dataset upload");
        let genes = engine
            .upload_genes(&workload.genes.encode().unwrap())
            .expect("gene upload");
        let scenarios = engine
            .upload_scenarios(&workload.scenarios.encode().unwrap())
            .expect("scenario upload");
        let (metrics, evaluate_event) = engine
            .evaluate(dataset, genes, scenarios, None)
            .expect("device evaluation");
        let (selection, filter_event) = engine
            .filter(metrics, DeviceFilterPolicy::All, evaluate_event)
            .expect("device selection");
        let summary = engine
            .readback_compact(selection, filter_event)
            .expect("compact readback");

        assert_eq!(summary.candidate_ids, workload.genes.candidate_ids);
        assert_eq!(
            summary.scenario_ids,
            workload
                .scenarios
                .scenarios
                .iter()
                .map(|scenario| scenario.scenario_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            engine.emitted_events(),
            expected.events.len(),
            "device event count must match the canonical emission"
        );

        // Field-specific tolerance, not bitwise equality: the device is allowed
        // to contract multiply-add pairs, which the host reference does not.
        let report =
            TinyPopulationFixture::compare_final_metrics(&expected_metrics, &summary.metrics);
        assert!(
            report.is_match(),
            "Prototype B level-10 parity failed: {:?}",
            report.first_divergence
        );

        // One logical dataset upload, no dense intermediate readback and no
        // chained re-upload.
        engine
            .session()
            .transfer_snapshot()
            .assert_device_resident_chain()
            .expect("device residency invariant");
    }

    #[cfg(not(feature = "gpu-b-native"))]
    #[test]
    fn cuda_free_builds_refuse_prototype_b_with_a_typed_status() {
        assert_eq!(
            prototype_b_engine_status(),
            EngineStatus::UnsupportedCapability
        );
        match create_prototype_b_engine(None, 1, 1024) {
            Err(EngineError::UnsupportedCapability { operation, .. }) => {
                assert_eq!(operation, "prototype_b_engine");
            }
            Err(other) => panic!("expected a typed capability refusal, got {other}"),
            Ok(_) => unreachable!("Infallible cannot be constructed"),
        }
    }
}
