//! Real device-resident `BacktestEngine` adapter for Prototype A.

use crate::cubecl_eval::{
    PrototypeADatasetResident, PrototypeAGenesResident, PrototypeAScenariosResident,
    create_gpu_client, evaluate_prototype_a_resident, read_prototype_a_resident_metrics,
    upload_prototype_a_dataset, upload_prototype_a_genes, upload_prototype_a_scenarios,
};
use crate::gpu_native::engine::{
    BacktestEngine, DatasetHandle, DeviceEventHandle, DeviceFilterPolicy, DeviceMetricsHandle,
    DeviceSelectionHandle, EngineCapabilities, EngineError, EngineIdentity, EngineStatus,
    GeneBufferHandle, GpuDiscoverySession, HostSurvivorSummary, ScenarioBufferHandle,
    SynchronizationMode, TypedDeviceHandle,
};
use crate::gpu_native::prototype_a::{
    PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeARebatchPlan, PrototypeAScenarioUpload,
    prototype_a_capabilities,
};
use cubecl::prelude::{ComputeClient, Runtime};

#[cfg(feature = "gpu-cuda")]
pub type PrototypeAActiveRuntime = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
pub type PrototypeAActiveRuntime = cubecl::wgpu::WgpuRuntime;

struct DatasetSlot {
    handle: DatasetHandle,
    resident: PrototypeADatasetResident,
}

struct GeneSlot {
    handle: GeneBufferHandle,
    upload: PrototypeAGeneUpload,
    resident: PrototypeAGenesResident,
}

struct ScenarioSlot {
    handle: ScenarioBufferHandle,
    upload: PrototypeAScenarioUpload,
    resident: PrototypeAScenariosResident,
}

struct MetricsSlot<R: Runtime> {
    handle: DeviceMetricsHandle,
    dataset: DatasetHandle,
    genes: GeneBufferHandle,
    scenarios: ScenarioBufferHandle,
    event: DeviceEventHandle,
    outputs: Vec<crate::cubecl_eval::FusedResidentMetrics<R>>,
}

struct SelectionSlot {
    handle: DeviceSelectionHandle,
    metrics: DeviceMetricsHandle,
    event: DeviceEventHandle,
}

pub struct PrototypeAResources<R: Runtime> {
    client: ComputeClient<R>,
    dataset: Option<DatasetSlot>,
    genes: Option<GeneSlot>,
    scenarios: Option<ScenarioSlot>,
    metrics: Option<MetricsSlot<R>>,
    selection: Option<SelectionSlot>,
}

pub struct PrototypeABacktestEngine<R: Runtime> {
    session: GpuDiscoverySession<PrototypeAResources<R>>,
    max_gene_batch: usize,
}

pub fn create_prototype_a_engine(
    device_override: Option<usize>,
    session_id: u64,
    max_gene_batch: usize,
) -> Result<PrototypeABacktestEngine<PrototypeAActiveRuntime>, EngineError> {
    if max_gene_batch == 0 {
        return Err(EngineError::Backend(
            "Prototype A max gene batch must be non-zero".into(),
        ));
    }
    let client = create_gpu_client(device_override)
        .map_err(|error| EngineError::Backend(error.to_string()))?;
    let identity = EngineIdentity {
        session_id,
        backend_id: active_backend_id(),
        device_id: device_override.unwrap_or(0) as u32,
    };
    Ok(PrototypeABacktestEngine {
        session: GpuDiscoverySession::with_backend(
            identity,
            SynchronizationMode::ExplicitEvents,
            PrototypeAResources {
                client,
                dataset: None,
                genes: None,
                scenarios: None,
                metrics: None,
                selection: None,
            },
        ),
        max_gene_batch,
    })
}

#[cfg(feature = "gpu-cuda")]
const fn active_backend_id() -> u32 {
    1
}

#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
const fn active_backend_id() -> u32 {
    2
}

impl<R: Runtime> PrototypeABacktestEngine<R> {
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

    fn validate_supported_scenarios(&self) -> Result<(), EngineError> {
        let resources = self.session.backend();
        let dataset = resources
            .dataset
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype A dataset is not uploaded".into()))?;
        let scenarios = resources
            .scenarios
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype A scenarios are not uploaded".into()))?;
        for (index, scenario) in scenarios.upload.scenarios.iter().enumerate() {
            let full_window = scenario.window_offset == 0
                && scenario.window_len as usize == dataset.resident.n_samples;
            let base_scenario = scenario.scenario_type == 0
                && scenario.spread_ticks == 0
                && scenario.slippage_ticks == 0
                && scenario.commission_micros == 0
                && scenario.perturbation_count == 0;
            if !full_window || !base_scenario {
                return Err(EngineError::UnsupportedCapability {
                    operation: "scenario_evaluation",
                    detail: format!(
                        "scenario {index} requests a window, cost override, or perturbation \
                         that the Stage-1 fused engine does not implement"
                    ),
                });
            }
        }
        Ok(())
    }
}

impl<R: Runtime> BacktestEngine for PrototypeABacktestEngine<R> {
    type Backend = PrototypeAResources<R>;

    fn status(&self) -> EngineStatus {
        EngineStatus::Ready
    }

    fn capabilities(&self) -> EngineCapabilities {
        prototype_a_capabilities()
    }

    fn session(&self) -> &GpuDiscoverySession<Self::Backend> {
        &self.session
    }

    fn upload_dataset(&mut self, bytes: &[u8]) -> Result<DatasetHandle, EngineError> {
        if self.session.backend().dataset.is_some() {
            return Err(EngineError::UnsupportedCapability {
                operation: "dataset_reupload",
                detail: "a Prototype A session accepts exactly one logical dataset upload".into(),
            });
        }
        let upload = PrototypeADatasetUpload::decode(bytes)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let resident = upload_prototype_a_dataset(
            &self.session.backend().client,
            &upload,
            self.max_gene_batch,
        )
        .map_err(|error| EngineError::Backend(error.to_string()))?;
        let upload_bytes = resident.upload_bytes as u64;
        let handle = self.session.allocate_handle::<DatasetHandle>();
        self.session.transfers().record_dataset_upload(upload_bytes);
        self.session
            .transfers()
            .record_workspace_allocations(resident.workspace_allocations);
        self.session.backend_mut().dataset = Some(DatasetSlot { handle, resident });
        Ok(handle)
    }

    fn upload_genes(&mut self, bytes: &[u8]) -> Result<GeneBufferHandle, EngineError> {
        let upload = PrototypeAGeneUpload::decode(bytes)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let dataset =
            self.session.backend().dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("upload the Prototype A dataset first".into())
            })?;
        if let Some((index, value)) = upload
            .indices
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value < 0 || *value as usize >= dataset.resident.n_indicators)
        {
            return Err(EngineError::Backend(format!(
                "Prototype A gene index {index} has feature {value}, but dataset has {} features",
                dataset.resident.n_indicators
            )));
        }
        let resident = upload_prototype_a_genes(
            &self.session.backend().client,
            &upload,
            dataset.resident.month_capacity,
        )
        .map_err(|error| EngineError::Backend(error.to_string()))?;
        let upload_bytes = resident.upload_bytes as u64;
        let handle = self.session.allocate_handle::<GeneBufferHandle>();
        self.session.transfers().record_gene_upload(upload_bytes);
        self.session
            .transfers()
            .record_workspace_allocations(resident.workspace_allocations);
        let resources = self.session.backend_mut();
        resources.metrics = None;
        resources.selection = None;
        resources.genes = Some(GeneSlot {
            handle,
            upload,
            resident,
        });
        Ok(handle)
    }

    fn upload_scenarios(&mut self, bytes: &[u8]) -> Result<ScenarioBufferHandle, EngineError> {
        let upload = PrototypeAScenarioUpload::decode(bytes)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let genes = self
            .session
            .backend()
            .genes
            .as_ref()
            .ok_or_else(|| EngineError::Backend("upload Prototype A genes first".into()))?;
        PrototypeARebatchPlan::new(&genes.upload, &upload, self.max_gene_batch)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let resident = upload_prototype_a_scenarios(&self.session.backend().client, &upload)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let upload_bytes = resident.upload_bytes as u64;
        let handle = self.session.allocate_handle::<ScenarioBufferHandle>();
        self.session
            .transfers()
            .record_scenario_upload(upload_bytes);
        let resources = self.session.backend_mut();
        resources.metrics = None;
        resources.selection = None;
        resources.scenarios = Some(ScenarioSlot {
            handle,
            upload,
            resident,
        });
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
        let dataset_slot = resources
            .dataset
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype A dataset is not uploaded".into()))?;
        let gene_slot = resources
            .genes
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype A genes are not uploaded".into()))?;
        let scenario_slot = resources
            .scenarios
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype A scenarios are not uploaded".into()))?;
        self.validate_current("evaluate_dataset", dataset, dataset_slot.handle)?;
        self.validate_current("evaluate_genes", genes, gene_slot.handle)?;
        self.validate_current("evaluate_scenarios", scenarios, scenario_slot.handle)?;
        self.validate_supported_scenarios()?;

        let (outputs, synchronization_events) = evaluate_prototype_a_resident(
            &resources.client,
            &dataset_slot.resident,
            &gene_slot.resident,
            self.max_gene_batch,
        )
        .map_err(|error| EngineError::Backend(error.to_string()))?;
        for _ in 0..synchronization_events {
            self.session.transfers().record_synchronization_event();
        }
        let metrics_handle = self.session.allocate_handle::<DeviceMetricsHandle>();
        let event_handle = self.session.allocate_handle::<DeviceEventHandle>();
        self.session.backend_mut().metrics = Some(MetricsSlot {
            handle: metrics_handle,
            dataset,
            genes,
            scenarios,
            event: event_handle,
            outputs,
        });
        self.session.backend_mut().selection = None;
        Ok((metrics_handle, Some(event_handle)))
    }

    fn filter(
        &mut self,
        metrics: DeviceMetricsHandle,
        policy: DeviceFilterPolicy,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<(DeviceSelectionHandle, Option<DeviceEventHandle>), EngineError> {
        let metrics_slot =
            self.session.backend().metrics.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype A metrics are not available".into())
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
        let selection_slot =
            self.session.backend().selection.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype A selection is not available".into())
            })?;
        self.validate_current("readback_selection", selection, selection_slot.handle)?;
        self.validate_wait_event("readback_wait_event", wait_for, selection_slot.event)?;
        let metrics_handle = selection_slot.metrics;
        let metrics_slot =
            self.session.backend().metrics.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype A metrics are not available".into())
            })?;
        self.validate_current("readback_metrics", metrics_handle, metrics_slot.handle)?;
        self.validate_current(
            "readback_dataset_parent",
            metrics_slot.dataset,
            self.session.backend().dataset.as_ref().unwrap().handle,
        )?;
        self.validate_current(
            "readback_gene_parent",
            metrics_slot.genes,
            self.session.backend().genes.as_ref().unwrap().handle,
        )?;
        self.validate_current(
            "readback_scenario_parent",
            metrics_slot.scenarios,
            self.session.backend().scenarios.as_ref().unwrap().handle,
        )?;

        let outputs = self
            .session
            .backend_mut()
            .metrics
            .take()
            .expect("metrics slot validated above")
            .outputs;
        let (metrics, readback_bytes) = read_prototype_a_resident_metrics(outputs)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        self.session
            .transfers()
            .record_compact_readback(readback_bytes as u64);
        let resources = self.session.backend();
        let candidate_ids = resources
            .genes
            .as_ref()
            .expect("gene parent validated above")
            .resident
            .canonical_candidate_ids
            .clone();
        let scenario_ids = resources
            .scenarios
            .as_ref()
            .expect("scenario parent validated above")
            .resident
            .canonical_descriptors
            .iter()
            .map(|scenario| scenario.scenario_id)
            .collect::<Vec<_>>();
        let rank_keys = metrics
            .iter()
            .map(|row| compact_rank_fields(row))
            .collect::<Result<Vec<_>, _>>()?;
        self.session.backend_mut().selection = None;
        Ok(HostSurvivorSummary {
            candidate_ids,
            scenario_ids,
            metrics,
            rank_keys,
        })
    }
}

fn compact_rank_fields(metrics: &[f64; 11]) -> Result<Vec<i64>, EngineError> {
    metrics
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if !value.is_finite() {
                return Err(EngineError::Backend(format!(
                    "Prototype A compact metric {index} is non-finite"
                )));
            }
            let scaled = *value * 1_000_000.0;
            if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
                return Err(EngineError::Backend(format!(
                    "Prototype A compact metric {index} exceeds rank range"
                )));
            }
            Ok(scaled.round() as i64)
        })
        .collect()
}
