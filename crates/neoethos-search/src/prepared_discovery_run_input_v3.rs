//! Exclusive CPU-or-native input preparation for one canonical Discovery run.
//!
//! The dispatcher performs the physical inventory/CUDA admission exactly once
//! before either factory can materialize data. A physical-GPU-free machine may
//! build the owned host input. A selected CUDA device instead consumes the
//! admitted full-workspace carrier into Data's sealed resident store.

use std::fmt;

use crate::data_selection::{
    CanonicalGpuResidentSearchArtifactScopeV3, CanonicalGpuResidentSearchInputReceiptV3,
    CanonicalSearchInput, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use crate::gpu_resident_current_config_plan_v1::{
    CurrentConfigResidentSearchAdmissionFactsV1, SealedCurrentConfigResidentSearchPlanV1,
    seal_current_config_resident_search_plan_v1,
};
use crate::gpu_resident_trim_prefilter_view_v1::{
    begin_gpu_resident_trim_prefilter_view_v1, execute_gpu_resident_trim_prefilter_view_v1,
    resolve_current_config_resident_trim_prefilter_plan_v1,
    seal_gpu_resident_trim_prefilter_view_v1,
};
use crate::prefilter_schema_v1::seal_prefilter_column_classification_v1;
use crate::resident_population_auto_sizing_receipt_v2::{
    RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2, ResidentPopulationAutoSizingReceiptV2,
    evaluation_config_from_canonical_trendbar_contract_v2,
    seal_resident_population_auto_for_canonical_trendbar_research_v2,
    seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_v2,
};
use crate::strict_discovery_device_route_v1::SealedStrictDiscoveryDeviceAdmissionV1;
use crate::strict_resident_feature_store_v3::{
    StrictResidentPopulationExecutionRunV3, bind_strict_resident_feature_store_v3_run_input,
    consume_strict_resident_population_execution_run_v3,
    record_resident_feature_store_consumer_completion_v3,
    validate_strict_resident_feature_store_v3,
};
use crate::{DiscoveryConfig, DiscoveryProgress, DiscoveryResult, PropFirmRiskRules};
use anyhow::{Context, Result, bail, ensure};
use neoethos_data::{PreparedGpuOnlyFeatureMaterializationV3, SealedGpuResidentFeatureStoreV3};
use neoethos_gpu_cuda::full_discovery_workspace_plan_v1::AdmittedNativeCudaFullDiscoveryRunV1;
use neoethos_gpu_cuda::resident_feature_store_v3::ResidentTrimPrefilterSchemaUploadV1;
use neoethos_gpu_cuda::resident_trim_prefilter_v1::ResidentTrimmedPopulationSessionV1;
use neoethos_gpu_cuda::run_device_admission_v1::SealedCpuNoPhysicalGpuRunDeviceAdmissionV1;
use neoethos_gpu_cuda::{
    AdmittedFullDiscoveryGpuRunV1, AdmittedNativeCudaDataPopulationRunV1,
    SealedDataPopulationGpuWorkspacePlanV1, SealedDiscoveryRunDeviceAdmissionV1,
    SealedFullDiscoveryGpuWorkspacePlanV1, SealedNativeCudaDataPopulationPreflightFactsV1,
    acquire_discovery_run_device_admission_v1, bind_data_population_gpu_workspace_plan_v1,
    bind_full_discovery_workspace_plan_v1, native_cuda_data_population_preflight_facts_v1,
};

#[derive(Debug)]
pub struct PreparedCpuCanonicalDiscoveryRunInputV3 {
    input: CanonicalSearchInput,
    receipt: CanonicalSearchInputReceiptV2,
    feature_names: Vec<String>,
    no_physical_gpu_admission: SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
}

#[derive(Debug)]
pub struct PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
    receipt: CanonicalGpuResidentSearchInputReceiptV3,
    feature_names: Vec<String>,
    sealed_store: SealedGpuResidentFeatureStoreV3,
}

#[derive(Debug)]
pub enum PreparedCanonicalDiscoveryRunInputV3 {
    Cpu(PreparedCpuCanonicalDiscoveryRunInputV3),
    NativeCuda(PreparedNativeCudaCanonicalDiscoveryRunInputV3),
}

#[derive(Debug)]
pub struct PreparedNativeCudaCanonicalDiscoveryRunInputV5 {
    receipt: CanonicalGpuResidentSearchInputReceiptV3,
    feature_names: Vec<String>,
    sealed_store: SealedGpuResidentFeatureStoreV3,
    population_sizing_receipt: ResidentPopulationAutoSizingReceiptV2,
    financial_contract:
        crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    evaluation_config: crate::genetic::EvaluationConfig,
    runtime_snapshot: crate::genetic::search_engine::ResidentGenerationZeroRuntimeSnapshotV1,
}

/// Staged canonical input whose native arm owns the exact admission-bound
/// population sizing receipt beside the resident Data store. V4 remains
/// source-compatible; V5 is the first carrier that cannot detach the resolved
/// population from the one pre-materialization device snapshot.
#[derive(Debug)]
pub struct PreparedCanonicalDiscoveryRunInputV5 {
    native: PreparedNativeCudaCanonicalDiscoveryRunInputV5,
}

/// Native-CUDA Generation-0 launch evidence only. This value deliberately is
/// not a Discovery result and carries no trim/prefilter, quality, validation,
/// funnel, portfolio, promotion or replay-identity claim.
#[derive(Debug)]
pub struct ResidentGenerationZeroMilestoneV1 {
    selected_device_ordinal: u32,
    engine: &'static str,
    native_input_receipt_identity_sha256: String,
    population_sizing_receipt_identity_sha256: String,
    resolved_population: usize,
    term_cap: usize,
    stage1_row_start: usize,
    stage1_row_end: usize,
    metrics_receipt_identities_sha256: Vec<[u8; 32]>,
    adaptive_token_identity_sha256: Option<[u8; 32]>,
    residency_counters: neoethos_gpu_cuda::PopulationResidencyCountersV1,
    search_result: crate::genetic::SearchResult,
    replay_identity_sealed: bool,
    consumer_completion_confirmed: bool,
}

/// Crate-private stage boundary for the V5 executor. Evaluation failures and
/// failures to record or await the resident consumer completion event have
/// different safety consequences and must not be flattened before the
/// production executor maps them into its own typed error.
#[derive(Debug)]
pub(crate) enum ResidentGenerationZeroStageErrorV1 {
    PreLaunchGate(anyhow::Error),
    GenerationZeroEvaluation(anyhow::Error),
    ConsumerCompletion(anyhow::Error),
}

impl fmt::Display for ResidentGenerationZeroStageErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreLaunchGate(error) => write!(
                output,
                "resident Generation-0 pre-launch gate rejected: {error}"
            ),
            Self::GenerationZeroEvaluation(error) => {
                write!(output, "resident Generation-0 evaluation failed: {error}")
            }
            Self::ConsumerCompletion(error) => write!(
                output,
                "resident Generation-0 consumer completion failed: {error}"
            ),
        }
    }
}

impl std::error::Error for ResidentGenerationZeroStageErrorV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PreLaunchGate(error)
            | Self::GenerationZeroEvaluation(error)
            | Self::ConsumerCompletion(error) => Some(error.as_ref()),
        }
    }
}

/// CPU-only combined research/training handoff. The exact input used by
/// Discovery stays owned and is moved into training after the research result
/// is sealed; it is never reconstructed from a path or current-generation
/// pointer.
pub struct PreparedCpuCanonicalTrendbarResearchRunV3 {
    research: crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3,
    training_input: CanonicalSearchInput,
}

impl PreparedCpuCanonicalTrendbarResearchRunV3 {
    pub const fn research_result(
        &self,
    ) -> &crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3 {
        &self.research
    }

    pub fn into_parts(
        self,
    ) -> (
        crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3,
        CanonicalSearchInput,
    ) {
        (self.research, self.training_input)
    }
}

impl PreparedCanonicalDiscoveryRunInputV3 {
    pub const fn cpu_receipt_v2(&self) -> Option<&CanonicalSearchInputReceiptV2> {
        match self {
            Self::Cpu(prepared) => Some(&prepared.receipt),
            Self::NativeCuda(_) => None,
        }
    }

    pub const fn native_receipt_v3(&self) -> Option<&CanonicalGpuResidentSearchInputReceiptV3> {
        match self {
            Self::Cpu(_) => None,
            Self::NativeCuda(prepared) => Some(&prepared.receipt),
        }
    }

    pub fn shape(&self) -> Result<(usize, usize)> {
        match self {
            Self::Cpu(prepared) => Ok((
                prepared.input.features().n_samples(),
                prepared.input.features().n_features(),
            )),
            Self::NativeCuda(prepared) => Ok((
                usize::try_from(prepared.sealed_store.contract().layout().row_count())
                    .context("resident V3 row count does not fit this process")?,
                usize::try_from(prepared.sealed_store.contract().layout().column_count())
                    .context("resident V3 column count does not fit this process")?,
            )),
        }
    }

    /// Metadata-only feature ordering used by the streaming survivor remap.
    /// The native branch copies names, never feature values, from the sealed
    /// resident-store contract.
    pub fn feature_names(&self) -> &[String] {
        match self {
            Self::Cpu(prepared) => &prepared.feature_names,
            Self::NativeCuda(prepared) => &prepared.feature_names,
        }
    }
}

impl PreparedCanonicalDiscoveryRunInputV5 {
    pub const fn native_receipt_v3(&self) -> &CanonicalGpuResidentSearchInputReceiptV3 {
        &self.native.receipt
    }

    pub const fn population_sizing_receipt_v2(&self) -> &ResidentPopulationAutoSizingReceiptV2 {
        &self.native.population_sizing_receipt
    }

    pub const fn financial_contract_v3(
        &self,
    ) -> &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3 {
        &self.native.financial_contract
    }

    pub const fn exact_evaluation_config_v2(&self) -> &crate::genetic::EvaluationConfig {
        &self.native.evaluation_config
    }

    pub fn shape(&self) -> Result<(usize, usize)> {
        Ok((
            usize::try_from(self.native.sealed_store.contract().layout().row_count())
                .context("resident V5 row count does not fit this process")?,
            usize::try_from(self.native.sealed_store.contract().layout().column_count())
                .context("resident V5 column count does not fit this process")?,
        ))
    }

    pub fn feature_names(&self) -> &[String] {
        &self.native.feature_names
    }
}

impl ResidentGenerationZeroMilestoneV1 {
    #[cfg(all(test, feature = "gpu-cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn test_fixture_v1(
        selected_device_ordinal: u32,
        native_input_receipt_identity_sha256: String,
        population_sizing_receipt_identity_sha256: String,
        resolved_population: usize,
        term_cap: usize,
        stage1_row_start: usize,
        stage1_row_end: usize,
        metrics_receipt_identities_sha256: Vec<[u8; 32]>,
        adaptive_token_identity_sha256: Option<[u8; 32]>,
        residency_counters: neoethos_gpu_cuda::PopulationResidencyCountersV1,
        search_result: crate::genetic::SearchResult,
    ) -> Self {
        Self {
            selected_device_ordinal,
            engine: "CudaNativeF64",
            native_input_receipt_identity_sha256,
            population_sizing_receipt_identity_sha256,
            resolved_population,
            term_cap,
            stage1_row_start,
            stage1_row_end,
            metrics_receipt_identities_sha256,
            adaptive_token_identity_sha256,
            residency_counters,
            search_result,
            replay_identity_sealed: false,
            consumer_completion_confirmed: true,
        }
    }

    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }

    pub const fn engine(&self) -> &'static str {
        self.engine
    }

    pub fn native_input_receipt_identity_sha256(&self) -> &str {
        &self.native_input_receipt_identity_sha256
    }

    pub fn population_sizing_receipt_identity_sha256(&self) -> &str {
        &self.population_sizing_receipt_identity_sha256
    }

    pub const fn resolved_population(&self) -> usize {
        self.resolved_population
    }

    pub const fn term_cap(&self) -> usize {
        self.term_cap
    }

    pub const fn stage1_row_start(&self) -> usize {
        self.stage1_row_start
    }

    pub const fn stage1_row_end(&self) -> usize {
        self.stage1_row_end
    }

    pub fn metrics_receipt_identities_sha256(&self) -> &[[u8; 32]] {
        &self.metrics_receipt_identities_sha256
    }

    pub const fn adaptive_token_identity_sha256(&self) -> Option<[u8; 32]> {
        self.adaptive_token_identity_sha256
    }

    pub const fn residency_counters(&self) -> neoethos_gpu_cuda::PopulationResidencyCountersV1 {
        self.residency_counters
    }

    pub fn search_result(&self) -> &crate::genetic::SearchResult {
        &self.search_result
    }

    /// Always false for this bounded launch milestone: RNG and seen-file
    /// contents are not yet persisted as a replay authority.
    pub const fn replay_identity_sealed(&self) -> bool {
        self.replay_identity_sealed
    }

    pub const fn consumer_completion_confirmed(&self) -> bool {
        self.consumer_completion_confirmed
    }
}

/// The single cross-crate admission dispatcher. Callers may pin immutable
/// generation leases before entering this function, but only the selected arm
/// receives authority to materialize values. The CPU arm must return the same
/// opaque zero-physical-GPU admission; the native arm receives the admitted
/// stage-scoped Data+population run by value. This authority deliberately
/// does not claim admission for later validation phases.
pub fn dispatch_staged_canonical_discovery_data_preparation_v4<
    Output,
    CpuPayload,
    NativePreflightPayload,
    NativeReadyPayload,
    CpuFactory,
    FinishCpu,
    NativePreflightFactory,
    NativePlanFactory,
    NativeFactory,
>(
    cpu_factory: CpuFactory,
    finish_cpu: FinishCpu,
    native_preflight_factory: NativePreflightFactory,
    native_workspace_plan_factory: NativePlanFactory,
    native_factory: NativeFactory,
) -> Result<Output>
where
    CpuFactory: FnOnce(
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<(CpuPayload, SealedCpuNoPhysicalGpuRunDeviceAdmissionV1)>,
    FinishCpu: FnOnce(CpuPayload, SealedCpuNoPhysicalGpuRunDeviceAdmissionV1) -> Result<Output>,
    NativePreflightFactory:
        FnOnce(&SealedNativeCudaDataPopulationPreflightFactsV1) -> Result<NativePreflightPayload>,
    NativePlanFactory:
        FnOnce(
            NativePreflightPayload,
            &SealedNativeCudaDataPopulationPreflightFactsV1,
        ) -> Result<(NativeReadyPayload, SealedDataPopulationGpuWorkspacePlanV1)>,
    NativeFactory:
        FnOnce(NativeReadyPayload, AdmittedNativeCudaDataPopulationRunV1) -> Result<Output>,
{
    let admission = acquire_discovery_run_device_admission_v1()
        .context("acquire one physical-inventory/CUDA Discovery run admission")?;
    let expected_admission_identity = admission.admission_identity_sha256();
    let counters = admission.probe_counters();
    let physical_inventory_probe_count = counters.physical_inventory_probe_count();
    let cuda_enumeration_count = counters.cuda_enumeration_count();
    let primary_context_acquisition_count = counters.primary_context_acquisition_count();
    let run_stream_creation_count = counters.run_stream_creation_count();

    match admission {
        SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission) => {
            ensure!(
                physical_inventory_probe_count == 1
                    && cuda_enumeration_count == 1
                    && primary_context_acquisition_count == 0
                    && run_stream_creation_count == 0,
                "CPU Discovery preparation lacks one complete physical inventory/CUDA enumeration"
            );
            let (payload, cpu_admission) = cpu_factory(cpu_admission)
                .context("materialize CPU Discovery data behind sealed physical-GPU absence")?;
            let returned = SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission);
            ensure!(
                returned.admission_identity_sha256() == expected_admission_identity,
                "CPU input factory returned a different physical-GPU absence authority"
            );
            let SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission) = returned
            else {
                unreachable!("the CPU factory result was wrapped as the CPU variant")
            };
            finish_cpu(payload, cpu_admission)
        }
        SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native_admission) => {
            ensure!(
                physical_inventory_probe_count == 1
                    && cuda_enumeration_count == 1
                    && primary_context_acquisition_count == 1
                    && run_stream_creation_count == 1,
                "native Discovery preparation lacks one context and one admitted run stream"
            );
            let native_admission_facts =
                native_cuda_data_population_preflight_facts_v1(&native_admission);
            let native_payload = native_preflight_factory(&native_admission_facts)
                .context("seal native Discovery Data preflight before workspace planning")?;
            let (native_ready_payload, workspace_plan) =
                native_workspace_plan_factory(native_payload, &native_admission_facts)
                    .context("seal native Data+population stage workspace plan")?;
            let admitted_native = bind_data_population_gpu_workspace_plan_v1(
                SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native_admission),
                workspace_plan,
            )
            .context("bind Data+population stage workspace to the one admitted CUDA run")?;
            native_factory(native_ready_payload, admitted_native)
                .context("materialize native Discovery data on the admitted CUDA run")
        }
    }
}

pub fn dispatch_canonical_discovery_data_preparation_v3<
    Output,
    CpuPayload,
    CpuFactory,
    FinishCpu,
    NativePlanFactory,
    NativeFactory,
>(
    cpu_factory: CpuFactory,
    finish_cpu: FinishCpu,
    native_workspace_plan_factory: NativePlanFactory,
    native_factory: NativeFactory,
) -> Result<Output>
where
    CpuFactory: FnOnce(
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<(CpuPayload, SealedCpuNoPhysicalGpuRunDeviceAdmissionV1)>,
    FinishCpu: FnOnce(CpuPayload, SealedCpuNoPhysicalGpuRunDeviceAdmissionV1) -> Result<Output>,
    NativePlanFactory: FnOnce() -> Result<SealedFullDiscoveryGpuWorkspacePlanV1>,
    NativeFactory: FnOnce(AdmittedNativeCudaFullDiscoveryRunV1) -> Result<Output>,
{
    let admission = acquire_discovery_run_device_admission_v1()
        .context("acquire one physical-inventory/CUDA Discovery run admission")?;
    let expected_admission_identity = admission.admission_identity_sha256();
    let counters = admission.probe_counters();
    let physical_inventory_probe_count = counters.physical_inventory_probe_count();
    let cuda_enumeration_count = counters.cuda_enumeration_count();
    let primary_context_acquisition_count = counters.primary_context_acquisition_count();
    let run_stream_creation_count = counters.run_stream_creation_count();

    match admission {
        SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission) => {
            ensure!(
                physical_inventory_probe_count == 1
                    && cuda_enumeration_count == 1
                    && primary_context_acquisition_count == 0
                    && run_stream_creation_count == 0,
                "CPU Discovery preparation lacks one complete physical inventory/CUDA enumeration"
            );
            let (payload, cpu_admission) = cpu_factory(cpu_admission)
                .context("materialize CPU Discovery data behind sealed physical-GPU absence")?;
            let returned = SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission);
            ensure!(
                returned.admission_identity_sha256() == expected_admission_identity,
                "CPU input factory returned a different physical-GPU absence authority"
            );
            let SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission) = returned
            else {
                unreachable!("the CPU factory result was wrapped as the CPU variant")
            };
            finish_cpu(payload, cpu_admission)
        }
        SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native_admission) => {
            ensure!(
                physical_inventory_probe_count == 1
                    && cuda_enumeration_count == 1
                    && primary_context_acquisition_count == 1
                    && run_stream_creation_count == 1,
                "native Discovery preparation lacks one context and one admitted run stream"
            );
            let workspace_plan = native_workspace_plan_factory()
                .context("seal complete native Discovery workspace plan")?;
            let admitted = bind_full_discovery_workspace_plan_v1(
                SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native_admission),
                workspace_plan,
            )
            .context("bind full Discovery workspace to the one admitted CUDA run")?;
            let admitted_native = match admitted {
                AdmittedFullDiscoveryGpuRunV1::NativeCuda(admitted_native) => admitted_native,
            };
            native_factory(admitted_native)
                .context("materialize native Discovery data on the admitted CUDA run")
        }
    }
}

pub fn prepare_canonical_discovery_run_input_v3<CpuFactory, NativePlanFactory, NativeFactory>(
    cpu_factory: CpuFactory,
    native_workspace_plan_factory: NativePlanFactory,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV3>
where
    CpuFactory: FnOnce(
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<(
        CanonicalSearchInput,
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    )>,
    NativePlanFactory: FnOnce() -> Result<SealedFullDiscoveryGpuWorkspacePlanV1>,
    NativeFactory: FnOnce(
        AdmittedNativeCudaFullDiscoveryRunV1,
    ) -> Result<(
        CanonicalGpuResidentSearchInputReceiptV3,
        SealedGpuResidentFeatureStoreV3,
    )>,
{
    dispatch_canonical_discovery_data_preparation_v3(
        cpu_factory,
        |input, cpu_admission| {
            let receipt = input
                .receipt()
                .context("seal prepared CPU canonical Search receipt")?;
            let feature_names = input.features().names.clone();
            Ok(PreparedCanonicalDiscoveryRunInputV3::Cpu(
                PreparedCpuCanonicalDiscoveryRunInputV3 {
                    input,
                    receipt,
                    feature_names,
                    no_physical_gpu_admission: cpu_admission,
                },
            ))
        },
        native_workspace_plan_factory,
        |admitted_native| {
            let (receipt, sealed_store) = native_factory(admitted_native)?;
            let anchor = receipt
                .validate()
                .context("validate native canonical Search receipt")?;
            receipt
                .validate_against_store(&anchor, &sealed_store)
                .context("bind native canonical Search receipt to its sealed Data store")?;
            let feature_names = sealed_store
                .ordered_feature_names()
                .map(str::to_owned)
                .collect();
            Ok(PreparedCanonicalDiscoveryRunInputV3::NativeCuda(
                PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
                    receipt,
                    feature_names,
                    sealed_store,
                },
            ))
        },
    )
}

/// Staged production preparation for the application/UI backend. The native
/// Data recipe is resolved exactly once before the Data+population stage plan
/// is sealed, then the move-only ready payload is consumed beside the admitted
/// CUDA run. This breaks the former plan/materializer cycle without exposing
/// raw pointers, host feature values or caller-supplied byte counts.
pub fn prepare_staged_canonical_discovery_run_input_v4<
    CpuFactory,
    NativePreflightPayload,
    NativeReadyPayload,
    NativePreflightFactory,
    NativePlanFactory,
    NativeFactory,
>(
    cpu_factory: CpuFactory,
    native_preflight_factory: NativePreflightFactory,
    native_workspace_plan_factory: NativePlanFactory,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV3>
where
    CpuFactory: FnOnce(
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<(
        CanonicalSearchInput,
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    )>,
    NativePreflightFactory:
        FnOnce(&SealedNativeCudaDataPopulationPreflightFactsV1) -> Result<NativePreflightPayload>,
    NativePlanFactory:
        FnOnce(
            NativePreflightPayload,
            &SealedNativeCudaDataPopulationPreflightFactsV1,
        ) -> Result<(NativeReadyPayload, SealedDataPopulationGpuWorkspacePlanV1)>,
    NativeFactory: FnOnce(
        NativeReadyPayload,
        AdmittedNativeCudaDataPopulationRunV1,
    ) -> Result<(
        CanonicalGpuResidentSearchInputReceiptV3,
        SealedGpuResidentFeatureStoreV3,
    )>,
{
    dispatch_staged_canonical_discovery_data_preparation_v4(
        cpu_factory,
        |input, cpu_admission| {
            let receipt = input
                .receipt()
                .context("seal prepared CPU canonical Search receipt")?;
            let feature_names = input.features().names.clone();
            Ok(PreparedCanonicalDiscoveryRunInputV3::Cpu(
                PreparedCpuCanonicalDiscoveryRunInputV3 {
                    input,
                    receipt,
                    feature_names,
                    no_physical_gpu_admission: cpu_admission,
                },
            ))
        },
        native_preflight_factory,
        native_workspace_plan_factory,
        |native_payload, admitted_native| {
            let (receipt, sealed_store) = native_factory(native_payload, admitted_native)?;
            let anchor = receipt
                .validate()
                .context("validate native GPU-resident Search receipt")?;
            receipt
                .validate_against_store(&anchor, &sealed_store)
                .context("bind native GPU-resident Search receipt to its sealed Data store")?;
            let feature_names = sealed_store
                .ordered_feature_names()
                .map(str::to_owned)
                .collect();
            Ok(PreparedCanonicalDiscoveryRunInputV3::NativeCuda(
                PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
                    receipt,
                    feature_names,
                    sealed_store,
                },
            ))
        },
    )
}

/// Source-compatible successor to V4 for the native resident population
/// route. Search, not the application, derives Stage1/month/runtime facts and
/// resolves population-auto from the same admitted pre-materialization
/// snapshot that is consumed by Data. The prepared recipe and receipt move
/// together into the native materializer; neither can be rebuilt after Data
/// allocation.
pub fn prepare_staged_canonical_trendbar_research_run_input_v5<
    NativePreflightFactory,
    NativeFactory,
>(
    config: &DiscoveryConfig,
    financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    native_preflight_factory: NativePreflightFactory,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV5>
where
    NativePreflightFactory: FnOnce(
        &SealedNativeCudaDataPopulationPreflightFactsV1,
    ) -> Result<PreparedGpuOnlyFeatureMaterializationV3>,
    NativeFactory: FnOnce(
        PreparedGpuOnlyFeatureMaterializationV3,
        AdmittedNativeCudaDataPopulationRunV1,
    ) -> Result<(
        CanonicalGpuResidentSearchInputReceiptV3,
        SealedGpuResidentFeatureStoreV3,
    )>,
{
    prepare_staged_canonical_trendbar_research_run_input_with_hard_cap_v5(
        config,
        financial_contract,
        RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2,
        None,
        native_preflight_factory,
        native_factory,
    )
}

pub(crate) fn prepare_prepared_canonical_trendbar_research_run_input_capped_v5<NativeFactory>(
    config: &DiscoveryConfig,
    financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    prepared: PreparedGpuOnlyFeatureMaterializationV3,
    max_resolved_population: usize,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV5>
where
    NativeFactory: FnOnce(
        PreparedGpuOnlyFeatureMaterializationV3,
        AdmittedNativeCudaDataPopulationRunV1,
    ) -> Result<(
        CanonicalGpuResidentSearchInputReceiptV3,
        SealedGpuResidentFeatureStoreV3,
    )>,
{
    let external_max_resolved_population = max_resolved_population;
    let hard_growth_cap =
        checked_v5_max_resolved_population_v1(config.population, max_resolved_population)?;
    prepare_staged_canonical_trendbar_research_run_input_with_hard_cap_v5(
        config,
        financial_contract,
        hard_growth_cap,
        Some(external_max_resolved_population),
        move |_native_facts| Ok(prepared),
        native_factory,
    )
}

fn checked_v5_max_resolved_population_v1(
    configured_population: usize,
    max_resolved_population: usize,
) -> Result<usize> {
    ensure!(
        max_resolved_population > 0,
        "V5 maximum resolved population must be non-zero"
    );
    let effective_cap = max_resolved_population.min(RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2);
    ensure!(
        configured_population <= max_resolved_population,
        "configured V5 population {configured_population} exceeds its external hard cap {max_resolved_population}"
    );
    Ok(effective_cap)
}

fn prepare_staged_canonical_trendbar_research_run_input_with_hard_cap_v5<
    NativePreflightFactory,
    NativeFactory,
>(
    config: &DiscoveryConfig,
    financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    hard_growth_cap: usize,
    external_resolved_population_cap: Option<usize>,
    native_preflight_factory: NativePreflightFactory,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV5>
where
    NativePreflightFactory: FnOnce(
        &SealedNativeCudaDataPopulationPreflightFactsV1,
    ) -> Result<PreparedGpuOnlyFeatureMaterializationV3>,
    NativeFactory: FnOnce(
        PreparedGpuOnlyFeatureMaterializationV3,
        AdmittedNativeCudaDataPopulationRunV1,
    ) -> Result<(
        CanonicalGpuResidentSearchInputReceiptV3,
        SealedGpuResidentFeatureStoreV3,
    )>,
{
    let exact_evaluation_config =
        evaluation_config_from_canonical_trendbar_contract_v2(config, financial_contract)
            .map_err(anyhow::Error::new)
            .context("resolve exact V5 evaluation settings from explicit financial contract")?;
    ensure!(
        !config.adaptive_thresholds,
        "V5 Generation-0 does not yet have a resident adaptive-threshold reduction; disable adaptive thresholds for this bounded native milestone"
    );
    ensure!(
        !crate::genetic::current_gene_stop_bounds_overrides().atr_scaled,
        "V5 Generation-0 does not yet have a resident median-ATR gene-band reduction; disable ATR-scaled gene bounds for this bounded native milestone"
    );
    // Match the canonical disabled-policy branch and prevent one prior host
    // Discovery run from leaking dataset-derived geometry into this V5 run.
    crate::genetic::clear_adaptive_threshold_ladder();
    crate::genetic::clear_gene_stop_atr_scale();
    ensure!(
        config.runtime_overrides.min_history_years == 0,
        "V5 Generation-0 cannot yet prove the canonical minimum-history preflight from resident timestamps; set min_history_years=0 for this bounded native milestone"
    );
    ensure!(
        !config.discovery_ledger_enabled,
        "V5 Generation-0 does not yet carry the canonical discovery-ledger seed authority; disable the discovery ledger for this bounded native milestone"
    );
    let payoff_inputs = crate::run_identity::payoff_inputs_for_config(
        config,
        exact_evaluation_config.pip_value_per_lot,
    );
    crate::run_identity::assert_payoff_floor_reachable(
        config.target_profile.min_payoff_ratio,
        &payoff_inputs,
    )
    .context(
        "V5 Generation-0 payoff floor is unreachable under fixed resident milestone geometry",
    )?;
    let carried_financial_contract = financial_contract.clone();
    let runtime_snapshot =
        crate::genetic::search_engine::ResidentGenerationZeroRuntimeSnapshotV1::capture();
    dispatch_staged_canonical_discovery_data_preparation_v4(
        |cpu_admission| Ok(((), cpu_admission)),
        |(), _cpu_admission| {
            bail!(
                "V5 requires one admitted native CUDA run; use the explicit-contract V3 CPU route when no physical GPU exists"
            )
        },
        native_preflight_factory,
        |prepared, native_facts| {
            let extent = prepared.workspace_extent();
            let resident_rows = usize::try_from(extent.row_count())
                .context("V5 resident parent row count does not fit this process")?;
            let resident_columns = usize::try_from(extent.column_count())
                .context("V5 resident feature count does not fit this process")?;
            let timeframe_row_cap = config
                .max_rows_by_timeframe
                .get(&config.timeframe_label)
                .copied()
                .unwrap_or(0);
            let row_cap = match (config.max_rows, timeframe_row_cap) {
                (0, 0) => 0,
                (0, timeframe) => timeframe,
                (global, 0) => global,
                (global, timeframe) => global.min(timeframe),
            };
            ensure!(
                row_cap == 0 || row_cap >= resident_rows,
                "V5 Generation-0 requires resident trim/remap before sizing: configured row cap {row_cap} would trim resident parent {resident_rows}"
            );
            let effective_prefilter = crate::discovery::resolve_prefilter_top_k(
                config.runtime_overrides.prefilter_top_k,
                resident_columns,
                config.population,
                config.max_indicators,
            );
            ensure!(
                effective_prefilter == 0 || effective_prefilter >= resident_columns,
                "V5 Generation-0 requires resident feature prefilter/remap before sizing: effective prefilter {effective_prefilter} would reduce {resident_columns} resident features"
            );
            let sizing =
                if let Some(external_resolved_population_cap) = external_resolved_population_cap {
                    seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_v2(
                        &prepared,
                        native_facts,
                        config,
                        financial_contract,
                        external_resolved_population_cap,
                    )
                } else {
                    seal_resident_population_auto_for_canonical_trendbar_research_v2(
                        &prepared,
                        native_facts,
                        config,
                        financial_contract,
                    )
                };
            let (population_sizing_receipt, workspace_plan) = sizing
                .map_err(anyhow::Error::new)
                .context("seal admission-bound resident population sizing receipt")?;
            if let Some(external_resolved_population_cap) = external_resolved_population_cap {
                ensure!(
                    population_sizing_receipt.hard_growth_cap() == hard_growth_cap,
                    "V5 population sizing receipt hard cap drifted from its external bound"
                );
                ensure!(
                    population_sizing_receipt.resolved_population()
                        <= external_resolved_population_cap,
                    "V5 resolved population exceeded its external hard cap"
                );
            }
            Ok(((prepared, population_sizing_receipt), workspace_plan))
        },
        |(prepared, population_sizing_receipt), admitted_native| {
            let (receipt, sealed_store) = native_factory(prepared, admitted_native)?;
            let anchor = receipt
                .validate()
                .context("validate V5 native GPU-resident Search receipt")?;
            receipt
                .validate_against_store(&anchor, &sealed_store)
                .context("bind V5 native Search receipt to its sealed Data store")?;
            population_sizing_receipt
                .validate_financial_authority_against_pinned_source_projection_v2(
                    &carried_financial_contract,
                    sealed_store.pinned_source_projection_v1(),
                )
                .map_err(anyhow::Error::new)
                .context("bind V5 financial value authority to native source rows")?;
            let feature_names = sealed_store
                .ordered_feature_names()
                .map(str::to_owned)
                .collect();
            Ok(PreparedCanonicalDiscoveryRunInputV5 {
                native: PreparedNativeCudaCanonicalDiscoveryRunInputV5 {
                    receipt,
                    feature_names,
                    sealed_store,
                    population_sizing_receipt,
                    financial_contract: carried_financial_contract,
                    evaluation_config: exact_evaluation_config,
                    runtime_snapshot,
                },
            })
        },
    )
}

fn retain_resident_completion_until_ready_v1(
    lease: neoethos_gpu_cuda::resident_feature_store_v3::ResidentFeatureStoreConsumerLeaseV3,
) -> Result<neoethos_gpu_cuda::resident_feature_store_v3::ResidentFeatureStoreConsumerLeaseV3> {
    const MAX_COMPLETION_POLLS_V1: usize = 1_000_000;
    for _ in 0..MAX_COMPLETION_POLLS_V1 {
        if lease
            .completion_is_ready()
            .map_err(anyhow::Error::new)
            .context("query resident consumer completion after Generation-0")?
        {
            return Ok(lease);
        }
        std::thread::yield_now();
    }
    let shape = (lease.rows(), lease.columns());
    // Dropping an unfinished lease intentionally leaks its CUDA owner to avoid
    // use-after-free. Make that exceptional retention explicit and fail loud.
    std::mem::forget(lease);
    bail!(
        "resident Generation-0 completion event remained pending after {MAX_COMPLETION_POLLS_V1} polls for shape {}x{}; CUDA ownership was retained for safety",
        shape.0,
        shape.1
    )
}

/// Execute exactly the first host-evolution population on the admitted
/// resident CUDA V3 evaluator and return a bounded launch milestone. This is
/// not a complete Discovery run and cannot enter trim/prefilter, quality,
/// validation, funnel, portfolio or promotion paths.
pub fn run_prepared_canonical_trendbar_research_generation_zero_v5<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV5,
    progress_fn: F,
) -> Result<ResidentGenerationZeroMilestoneV1>
where
    F: FnMut(DiscoveryProgress),
{
    run_prepared_canonical_trendbar_research_generation_zero_typed_v5(prepared, progress_fn)
        .map_err(anyhow::Error::new)
}

fn run_generation_zero_pre_launch_gate_v1<PreLaunchGate>(
    pre_launch_gate: PreLaunchGate,
) -> std::result::Result<(), ResidentGenerationZeroStageErrorV1>
where
    PreLaunchGate: FnOnce() -> anyhow::Result<()>,
{
    pre_launch_gate().map_err(ResidentGenerationZeroStageErrorV1::PreLaunchGate)
}

pub(crate) fn run_prepared_canonical_trendbar_research_generation_zero_typed_v5<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV5,
    progress_fn: F,
) -> std::result::Result<ResidentGenerationZeroMilestoneV1, ResidentGenerationZeroStageErrorV1>
where
    F: FnMut(DiscoveryProgress),
{
    run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5(
        prepared,
        progress_fn,
        || Ok(()),
    )
}

pub(crate) fn run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5<
    F,
    PreLaunchGate,
>(
    prepared: PreparedCanonicalDiscoveryRunInputV5,
    mut progress_fn: F,
    pre_launch_gate: PreLaunchGate,
) -> std::result::Result<ResidentGenerationZeroMilestoneV1, ResidentGenerationZeroStageErrorV1>
where
    F: FnMut(DiscoveryProgress),
    PreLaunchGate: FnOnce() -> anyhow::Result<()>,
{
    let PreparedCanonicalDiscoveryRunInputV5 { native } = prepared;
    let PreparedNativeCudaCanonicalDiscoveryRunInputV5 {
        receipt,
        feature_names,
        sealed_store,
        population_sizing_receipt,
        financial_contract,
        evaluation_config,
        runtime_snapshot,
    } = native;

    financial_contract
        .validate()
        .context("validate carried V5 financial contract before Generation-0")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    population_sizing_receipt
        .validate_financial_authority_against_pinned_source_projection_v2(
            &financial_contract,
            sealed_store.pinned_source_projection_v1(),
        )
        .map_err(anyhow::Error::new)
        .context("revalidate V5 financial authority against the sealed resident source")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    runtime_snapshot
        .validate_current("before binding resident V5 run")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    runtime_snapshot
        .validate_against_receipt_v2(&population_sizing_receipt)
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    if !(evaluation_config.symbol == financial_contract.symbol()
        && evaluation_config.account_currency == financial_contract.account_currency()
        && evaluation_config.pip_value.to_bits() == financial_contract.pip_size().to_bits()
        && evaluation_config.pip_value_per_lot.to_bits()
            == financial_contract.pip_value_per_lot().to_bits()
        && evaluation_config.spread_pips.to_bits()
            == financial_contract
                .screening_spread_and_slippage_round_trip_pips()
                .to_bits()
        && evaluation_config.commission_per_trade.to_bits()
            == financial_contract
                .round_trip_commission_account_per_lot()
                .to_bits()
        && evaluation_config.swap_long_pips_per_day.to_bits()
            == financial_contract.swap_long_pips_per_day().to_bits()
        && evaluation_config.swap_short_pips_per_day.to_bits()
            == financial_contract.swap_short_pips_per_day().to_bits()
        && evaluation_config.pnl_conversion_fee_rate.to_bits()
            == financial_contract.pnl_conversion_fee_rate().to_bits())
    {
        return Err(
            ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation(anyhow::anyhow!(
                "carried V5 evaluation settings drifted from their explicit financial contract"
            )),
        );
    }
    let native_input_receipt_identity_sha256 = receipt
        .identity_sha256()
        .context("hash native V5 input receipt before Generation-0")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    let population_sizing_receipt_identity_sha256 =
        population_sizing_receipt.identity_sha256().to_owned();
    let selected_device_ordinal = population_sizing_receipt.selected_device_ordinal();
    let resolved_population = population_sizing_receipt.resolved_population();
    let term_cap = population_sizing_receipt.term_cap();
    let stage1_row_start = population_sizing_receipt.stage1_row_start();
    let stage1_row_end = population_sizing_receipt.stage1_row_end();
    let replay_identity_sealed = runtime_snapshot.replay_identity_sealed();

    progress_fn(DiscoveryProgress::SearchStarted {
        population: resolved_population,
        generations: 0,
        max_indicators: population_sizing_receipt.requested_max_indicators(),
    });
    let _financial_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            &financial_contract,
        )
        .context("install the carried financial contract for resident Generation-0")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    let scope = CanonicalGpuResidentSearchArtifactScopeV3::for_entire_receipt(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt,
    )
    .context("bind resident Generation-0 parent to the canonical V5 receipt")
    .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    let run = bind_strict_resident_feature_store_v3_run_input(sealed_store, &scope)
        .context("bind strict resident V5 store for Generation-0")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    let resident_selected_device_ordinal = run
        .selected_device_ordinal()
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    if resident_selected_device_ordinal != selected_device_ordinal {
        return Err(
            ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation(anyhow::anyhow!(
                "resident V5 selected ordinal drifted before Generation-0"
            )),
        );
    }

    if let Err(gate_error) = run_generation_zero_pre_launch_gate_v1(pre_launch_gate) {
        let completion_lease = record_resident_feature_store_consumer_completion_v3(run)
            .context("record resident consumer completion after pre-launch gate rejection")
            .map_err(ResidentGenerationZeroStageErrorV1::ConsumerCompletion)?;
        let completion_lease = retain_resident_completion_until_ready_v1(completion_lease)
            .map_err(ResidentGenerationZeroStageErrorV1::ConsumerCompletion)?;
        drop(completion_lease);
        return Err(gate_error);
    }

    let (outcome, completion_lease) = consume_strict_resident_population_execution_run_v3(
        run,
        |run: &mut StrictResidentPopulationExecutionRunV3| {
            run.with_resident_population_session_v3(|session| {
                crate::genetic::search_engine::evolve_resident_generation_zero_v3(
                    session,
                    &feature_names,
                    &population_sizing_receipt,
                    &runtime_snapshot,
                    evaluation_config,
                    |_generation,
                     _total_generations,
                     _best_fitness,
                     _stagnant_generations,
                     _archived_profitable| {},
                )
            })
        },
    )
    .context("record resident Generation-0 consumer completion")
    .map_err(ResidentGenerationZeroStageErrorV1::ConsumerCompletion)?;
    let completion_lease = retain_resident_completion_until_ready_v1(completion_lease)
        .map_err(ResidentGenerationZeroStageErrorV1::ConsumerCompletion)?;
    let (search_result, evidence) = match outcome {
        Ok(outcome) => outcome,
        Err(error) => {
            drop(completion_lease);
            return Err(
                ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation(
                    error.context("resident Generation-0 CUDA evaluation failed"),
                ),
            );
        }
    };
    drop(completion_lease);
    runtime_snapshot
        .validate_current("after completing resident V5 run")
        .map_err(ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation)?;
    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "resident_cuda_generation_zero",
        detail: format!(
            "CudaNativeF64 Generation-0 evaluated {resolved_population} candidates in {} metrics-only launch(es); stopping before Discovery funnel/finalize",
            evidence.metrics_receipt_identities_sha256.len()
        ),
    });
    Ok(ResidentGenerationZeroMilestoneV1 {
        selected_device_ordinal,
        engine: "CudaNativeF64",
        native_input_receipt_identity_sha256,
        population_sizing_receipt_identity_sha256,
        resolved_population,
        term_cap,
        stage1_row_start,
        stage1_row_end,
        metrics_receipt_identities_sha256: evidence.metrics_receipt_identities_sha256,
        adaptive_token_identity_sha256: evidence.adaptive_token_identity_sha256,
        residency_counters: evidence.residency_counters,
        search_result,
        replay_identity_sealed,
        consumer_completion_confirmed: true,
    })
}

pub fn run_prepared_canonical_discovery_with_holdout_and_progress_v3<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    match prepared {
        PreparedCanonicalDiscoveryRunInputV3::Cpu(cpu) => {
            run_cpu_prepared_discovery_v3(cpu, config, prop_firm_rules, progress_fn)
        }
        PreparedCanonicalDiscoveryRunInputV3::NativeCuda(native) => {
            let PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
                receipt,
                feature_names,
                sealed_store,
            } = native;
            let scope = CanonicalGpuResidentSearchArtifactScopeV3::for_entire_receipt(
                CanonicalSearchWindowRoleV1::DiscoveryInput,
                receipt,
            )
            .context("bind native resident parent to the canonical Search receipt")?;
            validate_strict_resident_feature_store_v3(&sealed_store, &scope)
                .context("validate the resident parent before current-config preflight")?;
            let parent_rows = usize::try_from(sealed_store.contract().layout().row_count())
                .context("resident parent row count does not fit this process")?;
            let parent_columns =
                usize::try_from(sealed_store.contract().layout().column_count())
                    .context("resident parent column count does not fit this process")?;

            // This admission is deliberately resolved before the compact
            // schema upload or any trim allocation. Slice 1 has no authority
            // to invent archive-kNN throughput or memory-pool identity facts.
            let admission =
                require_current_config_resident_search_admission_facts_v1(&scope, &sealed_store)?;
            let runtime = crate::genetic::current_genetic_search_runtime_overrides();
            let current_config_plan = seal_current_config_resident_search_plan_v1(
                config,
                &runtime,
                parent_rows,
                parent_columns,
                admission,
            )
            .context("seal current-config resident Search plan before trim allocation")?;
            let trimmed_population = consume_native_store_into_trimmed_population_v1(
                sealed_store,
                &scope,
                feature_names,
                config,
                &current_config_plan,
            )?;
            run_native_cuda_prepared_discovery_v3(
                trimmed_population,
                current_config_plan,
                config,
                prop_firm_rules,
                &mut progress_fn,
            )
        }
    }
}

pub fn run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3>
where
    F: FnMut(DiscoveryProgress),
{
    contract.validate()?;
    let prepared_receipt = prepared
        .cpu_receipt_v2()
        .context("canonical-trendbar research V3 contract has no GPU-native receipt schema yet")?;
    ensure!(
        contract.input_receipt() == prepared_receipt,
        "canonical-trendbar research contract does not match the prepared CPU/native input"
    );
    let mut research_config = config.clone();
    crate::discovery::apply_research_contract_to_discovery_config(&mut research_config, contract);
    let _research_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            contract,
        )?;
    let result = run_prepared_canonical_discovery_with_holdout_and_progress_v3(
        prepared,
        &research_config,
        prop_firm_rules,
        progress_fn,
    )?;
    crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3::new(
        contract.clone(),
        result,
    )
}

/// Run the combined canonical research/training CPU path while carrying the
/// exact owned Search input forward. A native input is refused before Search:
/// the native Discovery store cannot be converted back into a host feature
/// frame, and the GPU-resident training handoff is not integrated yet.
pub fn run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<PreparedCpuCanonicalTrendbarResearchRunV3>
where
    F: FnMut(DiscoveryProgress),
{
    contract.validate()?;
    let prepared_receipt = prepared
        .cpu_receipt_v2()
        .context("canonical combined discovery+training has no GPU-native receipt schema yet")?;
    ensure!(
        contract.input_receipt() == prepared_receipt,
        "canonical-trendbar research contract does not match the prepared CPU/native input"
    );
    let PreparedCanonicalDiscoveryRunInputV3::Cpu(cpu) = prepared else {
        bail!(
            "canonical combined discovery+training requires a GPU-resident training handoff; refusing to reconstruct a host FeatureFrame from the native resident Discovery store"
        );
    };
    let mut research_config = config.clone();
    crate::discovery::apply_research_contract_to_discovery_config(&mut research_config, contract);
    let _research_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            contract,
        )?;
    let (result, training_input) = run_cpu_prepared_discovery_v3_with_input(
        cpu,
        &research_config,
        prop_firm_rules,
        progress_fn,
    )?;
    let research =
        crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3::new(
            contract.clone(),
            result,
        )?;
    Ok(PreparedCpuCanonicalTrendbarResearchRunV3 {
        research,
        training_input,
    })
}

fn run_cpu_prepared_discovery_v3<F>(
    prepared: PreparedCpuCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    run_cpu_prepared_discovery_v3_with_input(prepared, config, prop_firm_rules, progress_fn)
        .map(|(result, _input)| result)
}

fn run_cpu_prepared_discovery_v3_with_input<F>(
    prepared: PreparedCpuCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<(DiscoveryResult, CanonicalSearchInput)>
where
    F: FnMut(DiscoveryProgress),
{
    let strict_admission =
        SealedStrictDiscoveryDeviceAdmissionV1::from_no_physical_gpu_admission_v1(
            prepared.no_physical_gpu_admission,
        )
        .context("consume sealed physical-GPU absence into the CPU Discovery run")?;
    let input = prepared.input.as_run_input().map_err(anyhow::Error::new)?;
    let result = crate::discovery::run_discovery_cycle_with_prepared_cpu_admission_v3(
        &input,
        config,
        prop_firm_rules,
        strict_admission,
        progress_fn,
    )?;
    Ok((result, prepared.input))
}

fn require_current_config_resident_search_admission_facts_v1(
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
    sealed_store: &SealedGpuResidentFeatureStoreV3,
) -> Result<CurrentConfigResidentSearchAdmissionFactsV1> {
    scope
        .validate()
        .context("validate current-config Search scope before admission")?;
    ensure!(
        sealed_store.admission_identity_sha256() != [0; 32]
            && sealed_store
                .device_identity()
                .primary_context_process_token()
                != [0; 32],
        "resident store lacks its sealed CUDA admission identity"
    );
    bail!(
        "current-config resident Search requires an actual run memory-pool identity and a measured exact archive-kNN popcount calibration receipt before any trim allocation; Slice 1 refuses to fabricate either admission fact"
    )
}

fn decode_lower_hex_sha256_v1(value: &str, field: &'static str) -> Result<[u8; 32]> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} is not a canonical lowercase SHA-256"
    );
    let mut decoded = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte: u8| -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => unreachable!("canonical lowercase SHA-256 was validated"),
            }
        };
        decoded[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(decoded)
}

fn consume_native_store_into_trimmed_population_v1(
    sealed_store: SealedGpuResidentFeatureStoreV3,
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
    feature_names: Vec<String>,
    config: &DiscoveryConfig,
    current_config_plan: &SealedCurrentConfigResidentSearchPlanV1,
) -> Result<ResidentTrimmedPopulationSessionV1> {
    validate_strict_resident_feature_store_v3(&sealed_store, scope)
        .context("revalidate resident parent before moving it into trim")?;
    let resident_feature_names = sealed_store
        .ordered_feature_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        feature_names == resident_feature_names,
        "prepared feature order drifted from the sealed resident store"
    );
    let classification = seal_prefilter_column_classification_v1(&feature_names)
        .context("seal the shared CPU/resident prefilter schema classification")?;
    let canonical_receipt_identity = decode_lower_hex_sha256_v1(
        &scope
            .receipt()
            .identity_sha256()
            .context("seal canonical resident input receipt identity")?,
        "canonical resident input receipt identity",
    )?;
    let schema_upload = ResidentTrimPrefilterSchemaUploadV1::new(
        canonical_receipt_identity,
        sealed_store
            .contract()
            .canonical_feature_content_merkle_sha256(),
        sealed_store.normalization_fit_sha256(),
        sealed_store.final_feature_plan_v3_sha256(),
        sealed_store.source_provenance_sha256(),
        classification.ordered_feature_schema_sha256(),
        classification.column_classification_content_sha256(),
        classification.column_class_flags().to_vec(),
        classification.timeframe_group_ids().to_vec(),
        classification.template_force_keep_flags().to_vec(),
        classification.timeframe_group_count(),
    )
    .context("seal the compact resident trim schema upload")?;
    let resident_import = sealed_store
        .into_resident_feature_store_import_v3()
        .context("move the sealed resident store into its admitted-stream import")?;
    let trim_inputs = resident_import
        .consume_into_resident_trim_prefilter_v1(schema_upload)
        .context("move the resident feature store into the trim owner")?;
    let import_identity = *trim_inputs.identity();
    let resolved_plan = resolve_current_config_resident_trim_prefilter_plan_v1(
        config,
        current_config_plan,
        &import_identity,
        &classification,
        None,
    )
    .context("bind the trim plan to the exact current-config Search admission")?;
    let (parent, schema, admission) = trim_inputs.into_parts();
    let trim_run =
        begin_gpu_resident_trim_prefilter_view_v1(parent, schema, admission, resolved_plan)
            .context("begin the admitted resident trim/prefilter run")?;
    let trim_run = execute_gpu_resident_trim_prefilter_view_v1(trim_run)
        .context("enqueue the resident trim/prefilter stages")?;
    let sealed_views = seal_gpu_resident_trim_prefilter_view_v1(trim_run)
        .context("seal the resident trim/prefilter device views")?;
    sealed_views
        .consume_into_population_session_v3()
        .context("move sealed trim views into the resident population owner")
}

fn run_native_cuda_prepared_discovery_v3<F>(
    trimmed_population: ResidentTrimmedPopulationSessionV1,
    current_config_plan: SealedCurrentConfigResidentSearchPlanV1,
    _config: &DiscoveryConfig,
    _prop_firm_rules: PropFirmRiskRules,
    progress_fn: &mut F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    ensure!(
        current_config_plan.plan_identity_sha256() != [0; 32]
            && trimmed_population.population_rows() == current_config_plan.parent_row_range().end
            && trimmed_population.parent_columns() == current_config_plan.parent_column_count()
            && trimmed_population.selected_compact_to_parent_columns_device()
            && trimmed_population.selected_column_count_device()
            && trimmed_population.same_selected_column_map_for_holdout()
            && trimmed_population.has_zero_trim_host_boundary(),
        "native resident trim/population carrier drifted before Search execution"
    );
    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "gpu_native_trim_prefilter",
        detail: "sealed resident input was move-consumed through the real GPU-native trim/prefilter owner into the population carrier".to_owned(),
    });
    bail!(
        "resident multi-generation archive-kNN Search has no consumer for the sealed trim/population carrier yet; refusing host materialization, CPU fallback, or an armed-carrier readiness claim"
    )
}

#[cfg(all(test, feature = "gpu-cuda"))]
#[path = "prepared_discovery_run_input_v3/typed_stage_seam_tests.rs"]
mod typed_stage_seam_tests;
